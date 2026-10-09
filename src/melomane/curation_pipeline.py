"""Paper-1 curation pipeline analogue (stages 1-5 of Epicure's 8): rule simplification -> LLM refinement -> LLM
ultra-refinement -> human-in-the-loop QC queue -> consolidation map.  Pure logic + injected LLM (testable with a fake).

Counts are recorded at every stage (monotone non-increasing) so the paper can print the per-stage table and the consolidation
ratio.  Without an LLM the rule layer alone runs (exact strict-key merges + keyword non-music removal); the result is flagged
`degraded` because near-duplicate consolidation needs the LLM.
"""
from __future__ import annotations
import json, math, re
from dataclasses import dataclass, field
from collections import defaultdict

import numpy as np
import pandas as pd

from .curation import canonical_key, candidate_groups
from .llm import BudgetExceeded
from .runtime import get_logger

log = get_logger("melomane.curation")

NONMUSIC = re.compile(
    r"\b(white noise|brown noise|pink noise|rain sounds?|rain noise|nature sounds?|ocean waves?|thunderstorm|sleep (?:music|sounds?|aid)|"
    r"asmr|guided meditation|meditation (?:timer|music for)|audiobook|podcast|ringtone|karaoke|8d audio|binaural beats?|"
    r"lullaby versions? of|for babies|white-noise|fan noise|hair dryer|vacuum cleaner sound)\b", re.I)

REFINE_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["remove", "groups"],
                 "properties": {"remove": {"type": "array", "items": {"type": "integer"}},
                                "groups": {"type": "array", "items": {"type": "array", "items": {"type": "integer"}}}}}
ULTRA_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["verdicts", "restore"],
                "properties": {"verdicts": {"type": "array", "items": {"type": "object", "additionalProperties": False,
                                            "required": ["group", "verdict", "partition"],
                                            "properties": {"group": {"type": "integer"},
                                                           "verdict": {"type": "string", "enum": ["confirm", "split", "reject"]},
                                                           "partition": {"type": "array", "items": {"type": "array", "items": {"type": "integer"}}}}}},
                               "restore": {"type": "array", "items": {"type": "integer"}}}}
AUDIT_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["answers"],
                "properties": {"answers": {"type": "array", "items": {"type": "object", "additionalProperties": False,
                                           "required": ["pair", "same"],
                                           "properties": {"pair": {"type": "integer"}, "same": {"type": "string", "enum": ["yes", "no"]}}}}}}


@dataclass
class CurationConfig:
    min_sessions: int = 20
    max_vocab: int = 150_000
    level: str = "strict"
    jaccard: float = 0.6
    batch_items: int = 40
    ultra_batch_groups: int = 20
    workers: int = 8
    audit_n: int = 300
    review_frac: float = 0.05
    seed: int = 0


@dataclass
class CurationResult:
    cmap: pd.DataFrame                    # raw_id, cid, decision, group_id   (covers every raw track)
    vocab: pd.DataFrame
    counts: list[dict]
    audit: dict
    degraded: list[str] = field(default_factory=list)
    llm_stats: dict = field(default_factory=dict)


def _tags_str(v) -> str:
    return "" if v is None or (isinstance(v, float) and math.isnan(v)) else str(v)[:120]


def _items(df: pd.DataFrame, idx: np.ndarray) -> list[dict]:
    out = []
    for k, i in enumerate(idx):
        r = df.iloc[i]
        out.append({"id": k, "artist": str(r["artist"])[:80], "title": str(r["title"])[:120],
                    "album": str(r.get("album", "") or "")[:80], "tags": _tags_str(r.get("tags"))})
    return out


def _reconcile(n: int, remove: list[int], groups: list[list[int]]) -> tuple[set[int], list[list[int]]]:
    """Enforce 'every id exactly once': unknown ids dropped, duplicates keep first occurrence, missing ids become singletons."""
    seen: set[int] = set(); rm = set()
    for i in remove:
        if isinstance(i, int) and 0 <= i < n and i not in seen: rm.add(i); seen.add(i)
    out = []
    for g in groups:
        gg = []
        for i in g:
            if isinstance(i, int) and 0 <= i < n and i not in seen: gg.append(i); seen.add(i)
        if gg: out.append(gg)
    out += [[i] for i in range(n) if i not in seen]
    return rm, out


def _pack(units: list[list[int]], cap: int) -> list[list[list[int]]]:
    """Pack units (candidate groups / singletons) into requests of <= cap items."""
    batches, cur, size = [], [], 0
    for u in units:
        if size + len(u) > cap and cur:
            batches.append(cur); cur, size = [], 0
        cur.append(u); size += len(u)
    if cur: batches.append(cur)
    return batches


def curate(tracks: pd.DataFrame, n_sessions: np.ndarray, cfg: CurationConfig, llm=None, prompts: dict | None = None,
           has_names: bool = True, review_corrections: pd.DataFrame | None = None) -> CurationResult:
    N = len(tracks)
    counts: list[dict] = [{"stage": "0 raw catalogue", "n": int(N)}]
    degraded: list[str] = []
    cand = np.flatnonzero(n_sessions >= cfg.min_sessions)
    cand = cand[np.argsort(-n_sessions[cand], kind="stable")][: cfg.max_vocab]
    counts.append({"stage": f"vocabulary (>= {cfg.min_sessions} sessions, top {cfg.max_vocab})", "n": int(len(cand))})
    cmap = pd.DataFrame({"raw_id": np.arange(N), "cid": -1, "decision": "none", "group_id": -1})
    if len(cand) == 0:
        raise ValueError("no tracks pass the min_sessions filter - lower min_sessions in the profile")
    sub = tracks.iloc[cand].reset_index(drop=True)
    ns = n_sessions[cand]

    # ---------------- stage 1: simplification (rules) ----------------
    if has_names:
        keys = [canonical_key(a, t, cfg.level) for a, t in zip(sub["artist"].astype(str), sub["title"].astype(str))]
    else:
        keys = [(str(i), str(i)) for i in range(len(sub))]
        degraded.append("no artist/title available (Blocker-A): curation stages skipped, each track is its own canonical entry")
    by_key: dict = defaultdict(list)
    for i, k in enumerate(keys): by_key[k].append(i)
    key_order = list(by_key)
    rep_local = np.array([max(by_key[k], key=lambda i: ns[i]) for k in key_order])         # representative per exact group
    n_exact = len(key_order)
    counts.append({"stage": "1a exact-key simplification (strict)", "n": n_exact})
    rep_keys = [key_order[j] for j in range(n_exact)]

    removed_rep: set[int] = set()                     # indices j into key_order
    group_of = {j: j for j in range(n_exact)}         # j -> merged-group leader
    decision = {j: "rule" for j in range(n_exact)}
    llm_calls = 0

    # keyword non-music removal (also the whole non-music layer when there is no LLM)
    if has_names:
        for j, i in enumerate(rep_local):
            r = sub.iloc[i]
            if NONMUSIC.search(f"{r['artist']} {r['title']} {r.get('album', '')}"):
                removed_rep.add(j)
    counts.append({"stage": "1b keyword non-music removal", "n": n_exact - len(removed_rep)})

    cand_groups = candidate_groups(rep_keys, cfg.jaccard, 40) if has_names else []
    merged_llm: list[list[int]] = []
    if llm is not None and has_names and prompts:
        alive = [j for j in range(n_exact) if j not in removed_rep]
        in_group = {j for g in cand_groups for j in g}
        units = [[j for j in g if j not in removed_rep] for g in cand_groups] + [[j] for j in alive if j not in in_group]
        units = [u for u in units if u]
        batches = _pack(units, cfg.batch_items)
        jobs, meta = [], []
        v_ref, sys_ref = prompts["refine"]
        for b in batches:
            flat = [j for u in b for j in u]
            local = {j: k for k, j in enumerate(flat)}
            its = _items(sub, rep_local[flat])
            grp = [[local[j] for j in u] for u in b if len(u) > 1]
            jobs.append(dict(system=sys_ref, user=json.dumps({"items": its, "candidate_groups": grp}, ensure_ascii=False),
                             schema=REFINE_SCHEMA, name="refine"))
            meta.append(flat)
        try:
            res = llm.complete_json_many(jobs, workers=cfg.workers)
        except BudgetExceeded:
            res = [RuntimeError("budget")] * len(jobs); degraded.append("LLM budget reached during refinement")
        n_fail = 0
        for flat, r in zip(meta, res):
            llm_calls += 1
            if isinstance(r, Exception):
                n_fail += 1; continue
            rm, groups = _reconcile(len(flat), r.get("remove", []), r.get("groups", []))
            for k in rm: removed_rep.add(flat[k])
            for g in groups:
                if len(g) > 1:
                    merged_llm.append([flat[k] for k in g])
        if n_fail: degraded.append(f"{n_fail}/{len(jobs)} refinement requests failed; those items kept unmerged")
        counts.append({"stage": "2 LLM refinement (keep/remove/consolidate)",
                       "n": n_exact - len(removed_rep) - sum(len(g) - 1 for g in merged_llm)})
        # ---- stage 3: ultra-refinement ----
        v_ul, sys_ul = prompts["ultra"]
        jobs, meta = [], []
        ug = [g for g in merged_llm]
        removed_list = sorted(removed_rep - {j for j in range(n_exact) if NONMUSIC.search(f"{sub.iloc[rep_local[j]]['artist']} {sub.iloc[rep_local[j]]['title']}")})
        chunks_g = [ug[i:i + cfg.ultra_batch_groups] for i in range(0, len(ug), cfg.ultra_batch_groups)] or [[]]
        chunks_r = [removed_list[i:i + 60] for i in range(0, len(removed_list), 60)]
        for ci in range(max(len(chunks_g), len(chunks_r))):
            gs = chunks_g[ci] if ci < len(chunks_g) else []
            rs = chunks_r[ci] if ci < len(chunks_r) else []
            if not gs and not rs: continue
            groups_payload = [{"id": gi, "members": [{"id": j, "artist": str(sub.iloc[rep_local[j]]["artist"])[:80],
                                                       "title": str(sub.iloc[rep_local[j]]["title"])[:120],
                                                       "album": str(sub.iloc[rep_local[j]].get("album", "") or "")[:80]} for j in g]}
                              for gi, g in enumerate(gs)]
            removed_payload = [{"id": j, "artist": str(sub.iloc[rep_local[j]]["artist"])[:80], "title": str(sub.iloc[rep_local[j]]["title"])[:120],
                                "album": str(sub.iloc[rep_local[j]].get("album", "") or "")[:80]} for j in rs]
            jobs.append(dict(system=sys_ul, user=json.dumps({"groups": groups_payload, "removed": removed_payload}, ensure_ascii=False),
                             schema=ULTRA_SCHEMA, name="ultra"))
            meta.append(gs)
        final_groups: list[list[int]] = []
        if jobs:
            try:
                res = llm.complete_json_many(jobs, workers=cfg.workers)
            except BudgetExceeded:
                res = [RuntimeError("budget")] * len(jobs); degraded.append("LLM budget reached during ultra-refinement")
            for gs, r in zip(meta, res):
                llm_calls += 1
                if isinstance(r, Exception):
                    continue                      # unverified merges are dropped (conservative)
                for v in r.get("verdicts", []):
                    gi = v.get("group")
                    if not isinstance(gi, int) or not (0 <= gi < len(gs)): continue
                    members = gs[gi]
                    if v["verdict"] == "confirm":
                        final_groups.append(members)
                    elif v["verdict"] == "split":
                        ok = set(members); seen = set()
                        for part in v.get("partition", []):
                            part = [j for j in part if j in ok and j not in seen]
                            seen.update(part)
                            if len(part) > 1: final_groups.append(part)
                for j in r.get("restore", []):
                    if j in removed_rep: removed_rep.discard(j)
        merged_llm_final = final_groups
        counts.append({"stage": "3 LLM ultra-refinement (rule-checked merges, restores)",
                       "n": n_exact - len(removed_rep) - sum(len(g) - 1 for g in merged_llm_final)})
    else:
        merged_llm_final = []
        if has_names:
            degraded.append("no LLM available: near-duplicate consolidation and semantic non-music removal skipped (rule layer only)")
        counts.append({"stage": "2-3 LLM refinement / ultra-refinement", "n": n_exact - len(removed_rep)})

    for g in merged_llm_final:
        g = [j for j in g if j not in removed_rep]
        if len(g) < 2: continue
        lead = max(g, key=lambda j: ns[rep_local[j]])
        for j in g: group_of[j] = lead; decision[j] = "llm"

    # ---------------- assemble ----------------
    lead_ids = sorted({group_of[j] for j in range(n_exact) if j not in removed_rep}, key=lambda j: (-1 * sum(ns[i] for i in by_key[key_order[j]]), j))
    cid_of_lead = {j: c for c, j in enumerate(lead_ids)}
    for j in range(n_exact):
        members_local = by_key[key_order[j]]
        raw_ids = cand[members_local]
        if j in removed_rep:
            cmap.loc[raw_ids, "decision"] = "rule" if NONMUSIC.search(f"{sub.iloc[rep_local[j]]['artist']} {sub.iloc[rep_local[j]]['title']}") else "llm"
            continue
        c = cid_of_lead[group_of[j]]
        cmap.loc[raw_ids, "cid"] = c
        cmap.loc[raw_ids, "group_id"] = group_of[j]
        cmap.loc[raw_ids, "decision"] = "llm" if decision[j] == "llm" else ("rule" if len(members_local) > 1 else "none")
    # ---------------- human corrections ----------------
    if review_corrections is not None and len(review_corrections):
        from .release.review import apply_corrections
        cmap = apply_corrections(cmap, review_corrections)
        counts.append({"stage": f"5 human corrections applied ({len(review_corrections)})", "n": int(cmap[cmap.cid >= 0].cid.nunique())})
    # densify cids
    ok = cmap["cid"] >= 0
    remap = {c: i for i, c in enumerate(sorted(cmap.loc[ok, "cid"].unique()))}
    cmap.loc[ok, "cid"] = cmap.loc[ok, "cid"].map(remap)
    V = len(remap)
    counts.append({"stage": "5 consolidation -> canonical tracks", "n": V})
    # ---------------- vocab ----------------
    ns_all = np.zeros(N, dtype=np.int64); ns_all[cand] = ns
    m = cmap[ok].copy(); m["ns"] = ns_all[m["raw_id"].to_numpy()]
    rep = m.sort_values(["cid", "ns"], ascending=[True, False]).drop_duplicates("cid")
    vocab = tracks.iloc[rep["raw_id"].to_numpy()].copy().reset_index(drop=True)
    vocab.insert(0, "rep_raw", rep["raw_id"].to_numpy()); vocab.insert(0, "cid", rep["cid"].to_numpy())
    vocab["n_sessions_raw_max"] = rep["ns"].to_numpy()
    vocab["n_members"] = m.groupby("cid").size().reindex(vocab["cid"]).to_numpy()
    audit = {}
    return CurationResult(cmap, vocab, counts, audit, degraded, {"calls": llm_calls})


def audit_merges(cmap: pd.DataFrame, tracks: pd.DataFrame, llm, prompts: dict, n: int = 300, seed: int = 0,
                 workers: int = 8) -> dict:
    """Second-prompt-variant audit: sample merged pairs (same cid) and separate near-duplicate pairs; report agreement with the
    final decisions (target >= 90% on merges)."""
    rng = np.random.default_rng(seed)
    ok = cmap[cmap.cid >= 0]
    multi = ok.groupby("cid").filter(lambda g: len(g) > 1)
    pairs, truth = [], []
    if len(multi):
        grp = [g["raw_id"].to_numpy() for _, g in multi.groupby("cid")]
        for _ in range(min(n // 2, len(grp) * 3)):
            g = grp[rng.integers(len(grp))]
            a, b = rng.choice(g, 2, replace=False); pairs.append((int(a), int(b))); truth.append("yes")
    # separate pairs: neighbours by key within artist that were kept apart
    sub = ok.drop_duplicates("cid")
    keys = [canonical_key(str(tracks.at[r, "artist"]), str(tracks.at[r, "title"])) for r in sub["raw_id"]]
    for g in candidate_groups(keys, 0.6, 40)[: n]:
        if len(g) >= 2 and len(pairs) < n:
            a, b = rng.choice(g, 2, replace=False)
            pairs.append((int(sub["raw_id"].iloc[a]), int(sub["raw_id"].iloc[b]))); truth.append("no")
    if not pairs or llm is None:
        return {"n": 0, "agreement_merge": None, "agreement_split": None, "note": "nothing to audit or no LLM"}
    v, sysm = prompts["audit"]
    jobs, chunks = [], [list(range(i, min(i + 40, len(pairs)))) for i in range(0, len(pairs), 40)]
    for ch in chunks:
        payload = [{"pair": k, "a": {"artist": str(tracks.at[pairs[k][0], "artist"])[:80], "title": str(tracks.at[pairs[k][0], "title"])[:120]},
                    "b": {"artist": str(tracks.at[pairs[k][1], "artist"])[:80], "title": str(tracks.at[pairs[k][1], "title"])[:120]}} for k in ch]
        jobs.append(dict(system=sysm, user=json.dumps({"pairs": payload}, ensure_ascii=False), schema=AUDIT_SCHEMA, name="audit"))
    res = llm.complete_json_many(jobs, workers=workers)
    ans = {}
    for ch, r in zip(chunks, res):
        if isinstance(r, Exception): continue
        for a in r.get("answers", []):
            if a.get("pair") in ch: ans[a["pair"]] = a["same"]
    ym = [ans[k] == "yes" for k in ans if truth[k] == "yes"]
    ys = [ans[k] == "no" for k in ans if truth[k] == "no"]
    return {"n": len(ans), "prompt_version": v, "agreement_merge": float(np.mean(ym)) if ym else None, "n_merge": len(ym),
            "agreement_split": float(np.mean(ys)) if ys else None, "n_split": len(ys)}
