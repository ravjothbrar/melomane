"""S2 - LLM annotation (5-level dimensions, categories, regions) + reliability + LLM-vs-measured validation (H5)."""
from __future__ import annotations
import numpy as np
import pandas as pd

from ..annotation import DIMS, annotate, binary_pass, llm_vs_measured, reliability, category_from_tags
from ..pipeline import Ctx, Stage, StageResult, register, PARTIAL
from ..runtime import get_logger
from .common import corpus_of, vocab_of
from .s1_curation import credit_gate, get_llm, load_prompts

log = get_logger("melomane.s2")


def _empty_labels(vocab: pd.DataFrame) -> pd.DataFrame:
    lab = pd.DataFrame(index=np.arange(len(vocab)))
    for d in DIMS + ["era"]:
        lab[d] = np.nan
    lab["category"] = vocab["category"].to_numpy(); lab["regions"] = ""; lab["confidence"] = ""
    for f in ("has_vocals", "is_instrumental", "explicit_lyrics", "is_live"):
        lab[f] = False
    lab["labelled"] = False; lab["prompt_version"] = ""
    return lab


@register
class Annotation(Stage):
    name, title, deps = "S2", "LLM annotation & measured ground truth", ("S1",)
    requires_llm = True

    def params(self, ctx):
        return {"v": 3, "llm": ctx.profile["llm"], "have_llm": bool(ctx.opt("llm") is not None or __import__("os").environ.get("OPENROUTER_API_KEY"))}

    def outputs(self, ctx):
        return [ctx.stage_dir("S2") / "labels.parquet"]

    def run(self, ctx: Ctx) -> StageResult:
        ws, prof = ctx.ws, ctx.profile
        vocab, corp = vocab_of(ctx), corpus_of(ctx, load_families=False)
        d = ctx.stage_dir("S2")
        llm = get_llm(ctx)
        res = StageResult()
        if llm is None:
            lab = _empty_labels(vocab)
            lab.to_parquet(d / "labels.parquet")
            res.status = PARTIAL
            res.notes.append("no LLM: labels limited to rule-based categories; LLM-label probes (stratum c) and supervised poles disabled")
            ctx.save_result("S2_annotation", {"coverage": 0.0, "degraded": "no LLM"})
            return res
        P = load_prompts(ws, {"a1": "annotate_v1", "a2": "annotate_v2", "bin": "binary_v1"})
        V, cfgl = len(vocab), prof["llm"]
        n_max = min(V, cfgl["annotate_max_tracks"])
        # Which tracks get labels when the budget cannot cover all of them: a seeded random sample over the whole vocabulary, not the first
        # rows (the most-played 9,000 tracks come from only ~1,200 artists, which would starve artist-grouped evaluation). The pilot is the
        # head of the same permutation, so its paid answers are cache hits in the real run.
        order = np.random.default_rng(ctx.seed("S2sample")).permutation(V)
        spent0 = llm.spent() + getattr(llm, 'wasted_usd', 0.0)
        # ---- cost pilot: label 200, extrapolate, shrink the labelled set if it would blow the budget ----
        pilot_n = min(200, n_max)
        sub_p = vocab.iloc[order[:pilot_n]].reset_index(drop=True)
        _, _ = annotate(sub_p, llm, P["a1"][1], P["a1"][0], batch=cfgl["annotate_batch"], workers=cfgl["workers"])
        per_item = (llm.spent() + getattr(llm, 'wasted_usd', 0.0) - spent0) / max(pilot_n, 1)       # wasted (paid, uncached) calls count too
        remaining = getattr(llm, "budget", 1e9) - llm.spent()
        proj = per_item * n_max * 1.25 + 0.15 * per_item * n_max      # + reliability/binary passes
        res.metrics["pilot_usd_per_1k_tracks"] = per_item * 1000
        if per_item > 0 and proj > remaining:
            n_fit = int(max(pilot_n, remaining / (per_item * 1.4)))
            res.deviations.append(f"projected LLM spend ${proj:.0f} > remaining ${remaining:.0f}: annotating a random sample of {n_fit:,} tracks only")
            n_max = min(n_max, n_fit); res.status = PARTIAL
        rows = order[:n_max]
        sub = vocab.iloc[rows].reset_index(drop=True)
        lab_sub, st = annotate(sub, llm, P["a1"][1], P["a1"][0], batch=cfgl["annotate_batch"], workers=cfgl["workers"])
        lab = _empty_labels(vocab)
        for c in lab_sub.columns:
            lab.loc[rows, c] = lab_sub[c].to_numpy()
        # rule category for unlabelled rows stays; labelled rows get the LLM category
        lab.loc[~lab["labelled"].astype(bool), "category"] = vocab.loc[~lab["labelled"].astype(bool), "category"]
        # ---- reliability (second prompt variant + shuffled order) ----
        rng = np.random.default_rng(ctx.seed("S2rel"))
        lab_ok = np.flatnonzero(lab["labelled"].to_numpy())
        rel = pd.DataFrame()
        excluded: list[str] = []
        if len(lab_ok) >= 60:
            k = max(30, int(cfgl["reliability_frac"] * len(lab_ok)))
            idx = np.sort(rng.choice(lab_ok, min(k, len(lab_ok)), replace=False))
            sub2 = vocab.iloc[idx].reset_index(drop=True)
            lab2, _ = annotate(sub2, llm, P["a2"][1], P["a2"][0], batch=cfgl["annotate_batch"], workers=cfgl["workers"], order_seed=ctx.seed("S2rel2") % 1000)
            lab2.index = idx
            rel = reliability(lab, lab2.reindex(lab.index), idx)
            rel.to_parquet(d / "reliability.parquet")
            excluded = [r["dim"] for _, r in rel.iterrows() if r["dim"] in DIMS + ["era"] and not (r["spearman"] >= 0.6)]
            ctx.save_result("S2_reliability", {"table": rel.to_dict("records"), "excluded_dims": excluded, "threshold": 0.6})
            if excluded:
                res.notes.append(f"dimensions below reliability 0.6 (excluded from headline probes): {excluded}")
        # ---- binary yes/no pass on extreme levels ----
        try:
            bin_df = binary_pass(sub, lab_sub, llm, P["bin"][1], cfgl["binary_dims"], workers=cfgl["workers"])
            bfull = pd.DataFrame(index=lab.index, columns=bin_df.columns, dtype=float)
            bfull.iloc[rows] = bin_df.to_numpy()
            lab = pd.concat([lab, bfull], axis=1)
        except Exception as e:  # noqa: BLE001
            res.notes.append(f"binary pass skipped: {type(e).__name__}")
        lab["excluded_dim_flags"] = ",".join(excluded)
        lab.to_parquet(d / "labels.parquet")
        # ---- LLM vs measured (H5) ----
        vm = llm_vs_measured(lab[lab["labelled"].astype(bool)].reset_index(drop=True), corp.measured,
                             vocab.loc[lab["labelled"].astype(bool), "rep_raw"].to_numpy(),
                             extra={"year": vocab.loc[lab["labelled"].astype(bool), "year"].to_numpy(float)} if vocab["year"].notna().any() else None,
                             seed=ctx.seed("S2vm"), n_boot=prof["eval"]["knn_boot"] * 2)
        h5 = {"n_dims_with_counterpart": int(len(vm)), "share_positive_ci": float(((vm["rho"] > 0) & vm["ci_excludes_0"]).mean()) if len(vm) else None,
              "table": vm.to_dict("records")}
        ctx.save_result("S2_llm_vs_measured", h5)
        stats = {**st, "coverage_of_vocab": st["n_labelled"] / V, "n_annotated": int(n_max), "spent_usd": llm.spent() + getattr(llm, 'wasted_usd', 0.0) - spent0, "wasted_usd": getattr(llm, 'wasted_usd', 0.0), "prompt_version": P["a1"][0],
                 "excluded_dims": excluded, "category_counts": lab["category"].value_counts().head(25).to_dict()}
        ctx.save_result("S2_annotation", stats)
        (ctx.reports_dir / "s2_annotation.md").write_text(
            "# S2 - annotation\n\n" + f"Labelled {st['n_labelled']:,}/{V:,} (coverage of the vocabulary {stats['coverage_of_vocab']:.1%}; within the annotated sample {st['coverage']:.1%}); spent ${stats['spent_usd']:.2f}.\n\n"
            + "## Reliability (second prompt variant)\n\n" + (rel.to_markdown(index=False) if len(rel) else "n/a") + "\n\n## LLM vs measured\n\n"
            + (vm.to_markdown(index=False) if len(vm) else "no measured counterparts available") + "\n", encoding="utf-8")
        res.metrics.update({"coverage_of_vocab": stats["coverage_of_vocab"], "spent_usd": stats["spent_usd"], "h5_share": h5["share_positive_ci"]})
        if stats["coverage_of_vocab"] < 0.999:
            res.status = PARTIAL; res.notes.append(f"labels cover {stats['coverage_of_vocab']:.1%} of the vocabulary (<100%): unlabelled rows stay NaN, never fabricated")
        credit_gate(ctx, llm, res, "S2")
        return res
