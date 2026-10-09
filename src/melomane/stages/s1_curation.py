"""S1 - canonical vocabulary (paper-1 stages 1-5 + categorisation seed + pairwise-similarity prep)."""
from __future__ import annotations
import hashlib
import numpy as np
import pandas as pd
import scipy.sparse as sp

from ..annotation import category_from_tags, playlist_name_tags
from ..curation_pipeline import CurationConfig, curate, audit_merges
from ..llm_factory import LLMUnavailable
from ..pipeline import Ctx, Stage, StageResult, register, PARTIAL
from ..prompts import load_prompt
from ..runtime import get_logger, write_json
from .common import corpus_of

log = get_logger("melomane.s1")


def get_llm(ctx: Ctx):
    """Injected (tests/demo) or real client; None when unavailable (degraded mode)."""
    if ctx.opt("llm") is not None:
        return ctx.opt("llm")
    try:
        return ctx.llm()
    except LLMUnavailable:
        return None


def credit_gate(ctx: Ctx, llm, res, stage: str) -> None:
    """If the OpenRouter key ran out of credit (HTTP 402) during this stage, say so as a question for the owner, with the exact
    resume command.  LLM work stopped at the first 402; cached answers were still used, and what was left unlabelled is not faked."""
    if llm is not None and getattr(llm, "credits_exhausted", False):
        res.status = PARTIAL
        res.questions.append(
            f"{stage}: the OpenRouter key is out of credit (HTTP 402), so the LLM part was skipped (unlabelled items stay unlabelled, "
            f"results are marked degraded). Top up the key's credit limit (configs/llm.yaml caps spend at ${llm.budget:.0f}), then resume with "
            f"`python -m melomane run --stage {stage} --corpus {ctx.corpus} --profile {ctx.profile_name} --force` (cached answers make this cheap).")


def load_prompts(ws, names: dict[str, str]) -> dict:
    return {k: load_prompt(ws.root, v) for k, v in names.items()}


def dev_split(artist_id: np.ndarray, frac: float = 0.2) -> np.ndarray:
    h = np.array([int(hashlib.md5(str(int(a)).encode()).hexdigest()[:8], 16) % 1000 for a in artist_id])
    return np.where(h < frac * 1000, "dev", "test")


def merge_matrix(cmap: pd.DataFrame, n_raw: int, V: int) -> sp.csr_matrix:
    ok = cmap["cid"].to_numpy() >= 0
    return sp.csr_matrix((np.ones(ok.sum(), np.float32), (cmap["raw_id"].to_numpy()[ok], cmap["cid"].to_numpy()[ok])), shape=(n_raw, V))


@register
class Curation(Stage):
    name, title, deps = "S1", "Canonical vocabulary (curation)", ("S0-ingest",)

    def params(self, ctx):
        rc = ctx.ws.review / ctx.corpus / "corrections.parquet"
        h = hashlib.md5(rc.read_bytes()).hexdigest() if rc.exists() else None
        return {"v": 3, "vocab": ctx.profile["vocab"], "llm": ctx.profile["llm"], "corr": h, "have_llm": bool(ctx.opt("llm") is not None or __import__("os").environ.get("OPENROUTER_API_KEY")),
                "limit": ctx.profile.get("limit_tracks")}

    def outputs(self, ctx):
        d = ctx.stage_dir("S1")
        return [d / "vocab.parquet", d / "consolidation_map.parquet"]

    def run(self, ctx: Ctx) -> StageResult:
        ws, prof = ctx.ws, ctx.profile
        corp = corpus_of(ctx, load_families=False)
        tr, X = corp.tracks, corp.sessions.tocsc()
        n_sessions = np.diff(X.indptr).astype(np.int64)
        has_names = (tr["title"].astype(str).str.len() > 0).mean() > 0.5 and not corp.meta.get("blocker_a", False)
        cfg = CurationConfig(min_sessions=prof["vocab"]["min_sessions"], max_vocab=prof["vocab"]["max_vocab"], batch_items=40,
                             workers=prof["llm"]["workers"], audit_n=prof["llm"]["audit_n"], review_frac=prof["llm"]["review_frac"], seed=ctx.seed("S1"))
        llm = get_llm(ctx)
        prompts = load_prompts(ws, {"refine": "refine_v1", "ultra": "ultra_v1", "audit": "audit_v1"}) if llm is not None else None
        rc = ws.review / ctx.corpus / "corrections.parquet"
        corr = pd.read_parquet(rc) if rc.exists() else None
        r = curate(tr, n_sessions, cfg, llm, prompts, has_names=has_names, review_corrections=corr)
        d = ctx.stage_dir("S1")
        cmap, vocab = r.cmap, r.vocab
        V = len(vocab)
        # ---- per-vocab session counts (after merging) ----
        M = merge_matrix(cmap, len(tr), V)
        Sc = (corp.sessions @ M).tocsr(); Sc.data[:] = 1.0
        vocab["n_sessions"] = np.diff(Sc.tocsc().indptr).astype(np.int64)
        vocab["dev_split"] = dev_split(vocab["artist_id"].to_numpy())
        for col in ("year", "popularity", "album", "tags"):
            if col not in vocab:
                vocab[col] = np.nan if col in ("year", "popularity") else ""
        tags_note = None
        # tags: keep adapter tags; otherwise derive pseudo-tags from playlist names (Playlist2vec)
        if (vocab["tags"].astype(str).str.len() == 0).mean() > 0.9 and corp.sessions_meta is not None and "name" in corp.sessions_meta:
            tg = playlist_name_tags(Sc, corp.sessions_meta["name"].astype(str).tolist())
            vocab["tags"] = tg
            tags_note = "tags derived from playlist names (no Last.fm-style tags in this corpus)"
        vocab["category"] = [category_from_tags(t) for t in vocab["tags"].astype(str)]
        vocab["region_tags"] = ""
        vocab = vocab[["cid", "rep_raw", "artist_id", "artist", "title", "album", "year", "category", "n_sessions", "popularity", "dev_split",
                       "region_tags", "tags", "n_members"] + [c for c in vocab.columns if c in ("explicit", "duration_ms", "language", "has_lyrics", "n_artists")]]
        vocab.to_parquet(d / "vocab.parquet"); cmap.to_parquet(d / "consolidation_map.parquet")
        # ---- audit (second prompt variant) + review queue ----
        audit = {}
        if llm is not None and prompts:
            try:
                audit = audit_merges(cmap, tr, llm, prompts, n=prof["llm"]["audit_n"], seed=cfg.seed, workers=cfg.workers)
            except Exception as e:  # noqa: BLE001
                audit = {"error": str(e)[:200]}
        from ..release.review import sample_queue
        try:
            q = sample_queue(vocab, cmap, frac=cfg.review_frac, seed=cfg.seed, raw=tr[["artist", "title"]])
            qd = ws.d("review", ctx.corpus); q.to_parquet(qd / "queue.parquet")
        except Exception as e:  # noqa: BLE001
            q = pd.DataFrame(); log.warning("review queue not built: %s", e)
        # ---- acceptance checks ----
        counts = r.counts
        n_seq = [c["n"] for c in counts]
        monotone = all(n_seq[i] >= n_seq[i + 1] for i in range(len(n_seq) - 1) if not counts[i + 1]["stage"].startswith(("vocab", "5 human")))
        ok = cmap[cmap.cid >= 0]
        multi_artist = int((ok.assign(a=tr["artist_id"].to_numpy()[ok["raw_id"].to_numpy()]).groupby("cid")["a"].nunique() > 1).sum())
        stats = {"counts": counts, "consolidation_ratio": float(counts[1]["n"] / max(V, 1)), "V": V, "multi_artist_cids": multi_artist,
                 "audit": audit, "degraded": r.degraded, "monotone_counts": bool(monotone), "review_queue": int(len(q)),
                 "n_removed": int(((cmap.cid < 0) & (cmap.decision != "none")).sum()), "llm_calls": r.llm_stats.get("calls", 0),
                 "same_artist_share": float(1 - multi_artist / max(V, 1))}
        ctx.save_result("S1_curation", stats)
        lines = ["# S1 - canonical vocabulary", "", "| stage | tracks |", "|---|---|", *[f"| {c['stage']} | {c['n']:,} |" for c in counts], "",
                 f"Consolidation ratio (vocabulary candidates / canonical): {stats['consolidation_ratio']:.2f}x", f"Audit: {audit}", "",
                 *(["Degraded: " + "; ".join(r.degraded)] if r.degraded else [])]
        (ctx.reports_dir / "s1_curation.md").write_text("\n".join(lines), encoding="utf-8")
        res = StageResult(metrics={"V": V, "consolidation_ratio": stats["consolidation_ratio"], "removed": stats["n_removed"],
                                   "audit_agreement_merge": audit.get("agreement_merge")}, notes=r.degraded[:4] + ([tags_note] if tags_note else []))
        if r.degraded:
            res.status = PARTIAL
        credit_gate(ctx, llm, res, "S1")
        return res
