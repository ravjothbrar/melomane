"""S9 - extensions E1-E8 (+H4, H6, H8).  Each extension is independent: one that cannot be evaluated on this corpus is recorded as
`not evaluable` with the reason, one that crashes is recorded as `error` (stage becomes PARTIAL) - the others still run.

E1 compression frontier (H6)      E2 acoustic/lyric attribution (Shapley)   E3 partial-hub regime (propagation)
E4 equal-token control            E5 external-embedding agreement           E6 temporal stability
E7 curation ablation (H4)         E8 sensitivity grid ("multiverse")        H8 standard-vs-paper conclusions (subset)
"""
from __future__ import annotations

import traceback

import numpy as np
import pandas as pd
import scipy.sparse as sp

from ..analysis.agreement import agreement_report, knn_indices
from ..analysis.attribution import shapley_attribution
from ..analysis.compression import retention_curve, smallest_config
from ..analysis.multiverse import run_grid, summarise
from ..analysis.stats import paired_cluster_bootstrap
from ..analysis.temporal import mode_persistence, procrustes_disparity
from ..curation import exact_groups
from ..evaluation import item_matrix, paired_strata
from ..geometry import consolidate_embeddings, participation_ratio
from ..ext_subset import SubsetProblem, train_models
from ..ica_modes import factor_modes, residualise, stable_ica
from ..cooc import edges_to_csr, npmi_edges
from ..modes import project_sources
from ..pipeline import Ctx, PARTIAL, Stage, StageResult, register
from ..runtime import derive_seed, get_logger, jsonable
from ..training import model_specs, partial_hub_mask, train_one
from .common import available_models, corpus_of, load_embeddings
from .s1_curation import merge_matrix
from .s4_compounds import blocks_to_cid, family_kind
from .s5_training import load_graph_inputs, partial_pcts_of
from .s7_eval import EvalSet, build_eval

log = get_logger("melomane.s9")
MAIN = ["cooc", "composition", "core"]


class NotEvaluable(Exception):
    """The extension cannot be evaluated on this corpus / with this run's models (recorded with the reason, not a failure)."""


def _cv(ctx: Ctx) -> tuple[int, int]:
    return ctx.profile["cv"]["n_splits"], max(1, int(ctx.profile["ext"].get("cv_repeats", 1)))


def _need(models: list[str], *names: str) -> None:
    miss = [n for n in names if n not in models]
    if miss:
        raise NotEvaluable(f"needs trained model(s) {miss} (not present in this run)")


def _rho(ctx: Ctx, E, T, groups, seed) -> float:
    ns, nr = _cv(ctx)
    a = item_matrix(E, T, groups, ns, nr, seed)[0]
    return float(np.nanmean(a)) if np.isfinite(a).any() else float("nan")


def _paired(a, b, clusters, ctx, seed) -> dict | None:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 50:
        return None
    return paired_cluster_bootstrap(a, b, clusters, n_boot=min(ctx.profile["eval"]["boot_clusters"], 1000), seed=seed)


def _strata_items(ctx: Ctx, es: EvalSet, E: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    """Per-row mean item score for each probe stratum + the pooled H1 family ('primary')."""
    ns, nr = _cv(ctx)
    out = {}
    for st in ("a_baked", "b_heldout", "c_external"):
        T = es.targets(st)
        if T:
            out[st] = item_matrix(E, T, es.groups, ns, nr, seed)[0]
    P = es.primary()
    if P:
        out["primary"] = item_matrix(E, P, es.groups, ns, nr, seed)[0]
    return out


def _mean(a: np.ndarray) -> float:
    return float(np.nanmean(a)) if np.isfinite(a).any() else float("nan")


# ======================================================================================================================
# E1 - compression frontier / H6
# ======================================================================================================================
def e1_compression(ctx: Ctx, es: EvalSet, models: list[str]) -> dict:
    ext = ctx.profile["ext"]
    T = es.primary()
    if not T:
        raise NotEvaluable("no held-out / external probe targets")
    ns, nr = _cv(ctx)
    todo = [m for m in ("composition", "core", "cooc") if m in models]
    if not todo:
        raise NotEvaluable("no trained models")
    frames, per_model = [], {}
    for m in todo:
        E = load_embeddings(ctx, m)[es.idx]
        seed0 = ctx.seed("S9E1", m) % 10_000

        def probe(V, seed0=seed0):
            a = item_matrix(V, T, es.groups, ns, nr, seed0)[0]
            return float(np.nanmean(a)) if np.isfinite(a).any() else float("nan")
        df = retention_curve(E, probe, ext["dims"], ext["dtypes"], ext["methods"], seed=seed0, neighbour_k=10, n_queries=min(1000, len(E)), groups=es.groups)
        df.insert(0, "model", m)
        frames.append(df)
        per_model[m] = {"full_score": float(df.iloc[0]["score"]), "small": {str(t): smallest_config(df, t) for t in ext["small_thresholds"]}}
    allf = pd.concat(frames, ignore_index=True)
    ctx.stage_dir("S9").mkdir(parents=True, exist_ok=True)
    allf.to_parquet(ctx.stage_dir("S9") / "E1_compression.parquet")
    out = {"models": per_model, "n_rows": int(len(allf)), "thresholds": ext["small_thresholds"]}
    ship = {}
    for m, d in per_model.items():                                     # the config shipped as `<model>-small` (S12): 95 % retention, else 90 %
        pick = next((d["small"][str(t)] for t in sorted(ext["small_thresholds"], reverse=True) if d["small"][str(t)]), None)
        ship[m] = pick
    out["ship_small"] = ship
    H6 = {"claim": "Composition keeps >= 90% of full-precision held-out rho at <= 64 dims (fp16/int8)"}
    if "composition" in todo and per_model["composition"]["full_score"] > 0:
        d = allf[(allf["model"] == "composition") & (allf["dims"] <= 64) & (allf["dtype"].isin(["fp16", "int8"]))]
        if len(d):
            best = d.sort_values("retention", ascending=False).iloc[0]
            worst_of_best_method = d.groupby("method")["retention"].max().to_dict()
            H6.update({"status": "supported" if best["retention"] >= 0.90 else "not supported",
                       "evidence": {"best_retention": float(best["retention"]), "best_config": {k: (v.item() if hasattr(v, "item") else v) for k, v in best.items() if k in ("method", "dims", "dtype", "size_mb", "nn_recall")},
                                    "best_retention_by_method": worst_of_best_method}})
        else:
            H6.update({"status": "not evaluable", "evidence": "no config with <= 64 dims in the grid"})
    else:
        H6.update({"status": "not evaluable", "evidence": "composition not trained or full-precision score <= 0"})
    out["H6"] = H6
    return out


# ======================================================================================================================
# E2 - attribution
# ======================================================================================================================
def e2_attribution(ctx: Ctx, es: EvalSet, models: list[str]) -> dict:
    _need(models, "cooc", "composition_acoustic", "composition_lyric", "composition")
    items = {m: _strata_items(ctx, es, load_embeddings(ctx, m)[es.idx], ctx.seed("S9E2")) for m in ("cooc", "composition_acoustic", "composition_lyric", "composition")}
    out = {}
    for st in ("a_baked", "b_heldout", "c_external", "primary"):
        if all(st in items[m] for m in items):
            out[st] = shapley_attribution(items["cooc"][st], items["composition_acoustic"][st], items["composition_lyric"][st], items["composition"][st],
                                          clusters=es.groups, n_boot=min(ctx.profile["eval"]["boot_clusters"], 1000), seed=ctx.seed("S9E2", st))
    if not out:
        raise NotEvaluable("no probe stratum could be scored for all four systems")
    return {"by_stratum": out, "note": "Shapley split of the gain (Composition - Cooc) between acoustic and lyric families; 'synergy' = combined - acoustic - lyric + cooc"}


# ======================================================================================================================
# E3 - partial-hub regime
# ======================================================================================================================
def e3_partial_hub(ctx: Ctx, es: EvalSet, models: list[str]) -> dict:
    _need(models, "cooc")
    gi = load_graph_inputs(ctx)
    if gi.comp_edges is None:
        raise NotEvaluable("corpus has no compound graph")
    key = f"{ctx.corpus}/{ctx.profile_name}"
    specs = model_specs(ctx.profile["train"]["ii_repeat"], partial_pcts_of(ctx.profile))
    P = es.primary()
    if not P:
        raise NotEvaluable("no held-out / external targets")
    ns, nr = _cv(ctx)
    seed = ctx.seed("S9E3")
    cooc = item_matrix(load_embeddings(ctx, "cooc")[es.idx], P, es.groups, ns, nr, seed)[0]
    rows, systems = [], {}
    partial = sorted((n for n in models if n.startswith("core_partial")), key=lambda n: int(n[len("core_partial"):]))
    order = [(n, int(n[len("core_partial"):])) for n in partial] + ([("core", 100)] if "core" in models else [])
    natural = np.diff(gi.comp_edges.tocsr().indptr) > 0
    if not order:
        raise NotEvaluable("no core / core_partial* models were trained")
    for name, pct in order:
        E = load_embeddings(ctx, name)[es.idx]
        s = item_matrix(E, P, es.groups, ns, nr, seed)[0]
        hub_all = (partial_hub_mask(gi, specs[name], derive_seed(key, name, "partial")) if name in specs else natural)[es.idx]
        rec = {"model": name, "pct_hub": pct, "share_hub_in_eval": float(hub_all.mean())}
        for label, msk in (("all", np.ones(len(es.idx), bool)), ("hub", hub_all), ("nonhub", ~hub_all)):
            if msk.sum() >= 50:
                rec[f"rho_{label}"] = _mean(s[msk])
                rec[f"rho_cooc_{label}"] = _mean(cooc[msk])
                pb = _paired(s[msk], cooc[msk], es.groups[msk], ctx, ctx.seed("S9E3", name, label))
                rec[f"delta_vs_cooc_{label}"] = pb
        rows.append(rec)
        systems[name] = s
    out = {"systems": rows}
    # natural partial-hub regime (corpus-level): tracks with / without compound edges, Cooc vs Core vs Composition
    nat = natural[es.idx]
    if 50 <= nat.sum() and 50 <= (~nat).sum():
        nat_out = {"share_hub": float(nat.mean())}
        for m in [x for x in MAIN if x in models]:
            s = item_matrix(load_embeddings(ctx, m)[es.idx], P, es.groups, ns, nr, seed)[0]
            nat_out[m] = {"rho_hub": _mean(s[nat]), "rho_nonhub": _mean(s[~nat])}
            if m != "cooc":
                nat_out[m]["delta_vs_cooc_nonhub"] = _paired(s[~nat], cooc[~nat], es.groups[~nat], ctx, ctx.seed("S9E3nat", m))
        out["natural_hub_split"] = nat_out
    prop = [r for r in rows if r.get("delta_vs_cooc_nonhub") and r["pct_hub"] < 100]
    out["propagation"] = {"claim": "chemistry propagates through track-track hops: partial-hub Core beats Cooc on tracks WITHOUT compound edges",
                          "supported_at_pct": [r["pct_hub"] for r in prop if r["delta_vs_cooc_nonhub"]["ci_low"] > 0],
                          "not_supported_at_pct": [r["pct_hub"] for r in prop if r["delta_vs_cooc_nonhub"]["ci_high"] < 0],
                          "inconclusive_at_pct": [r["pct_hub"] for r in prop if r["delta_vs_cooc_nonhub"]["ci_low"] <= 0 <= r["delta_vs_cooc_nonhub"]["ci_high"]]}
    return out


# ======================================================================================================================
# E4 - equal-token control
# ======================================================================================================================
def e4_equal_token(ctx: Ctx, es: EvalSet, models: list[str]) -> dict:
    trio = ["cooc_eqtok", "composition_eqtok", "core_eqtok"]
    have = [m for m in trio if m in models]
    if len(have) < 2:
        raise NotEvaluable("needs at least two of cooc_eqtok / composition_eqtok / core_eqtok")
    ns, nr = _cv(ctx)
    strata = {k: v for k, v in {"a_baked": es.strata.get("a_baked", []), "b_heldout": es.strata.get("b_heldout", []), "c_external": es.strata.get("c_external", [])}.items() if v}
    Ts = {k: v for k, v in es.T.items() if v["kind"] != "binary"}
    out = {"means": {}, "paired": {}}
    for m in have:
        E = load_embeddings(ctx, m)[es.idx]
        out["means"][m] = {st: _mean(v) for st, v in _strata_items(ctx, es, E, ctx.seed("S9E4")).items()}
        out["means"][m]["participation_ratio"] = float(participation_ratio(E[np.random.default_rng(0).choice(len(E), min(len(E), 20000), replace=False)]))
    for a, b in (("composition_eqtok", "cooc_eqtok"), ("core_eqtok", "cooc_eqtok"), ("composition_eqtok", "core_eqtok")):
        if a in have and b in have and strata:
            out["paired"][f"{a}_minus_{b}"] = paired_strata(load_embeddings(ctx, a)[es.idx], load_embeddings(ctx, b)[es.idx], Ts, es.groups, strata, ns, nr,
                                                           n_boot=min(ctx.profile["eval"]["boot_clusters"], 1000), seed=ctx.seed("S9E4pair"))
    # does the headline (E4-free) H1 direction survive equalising the token budget?
    ph = out["paired"].get("composition_eqtok_minus_cooc_eqtok", {}).get("strata", {})
    ok = [ph[s] for s in ("b_heldout", "c_external") if s in ph]
    out["H1_equal_token"] = ("supported" if ok and all(e["ci_low"] > 0 for e in ok) else "not supported" if any(e["ci_high"] < 0 for e in ok) else "inconclusive") if ok else "not evaluable"
    return out


# ======================================================================================================================
# E5 - external-embedding agreement
# ======================================================================================================================
def e5_agreement(ctx: Ctx, es: EvalSet, models: list[str]) -> dict:
    corp = corpus_of(ctx, load_families=True)
    ext_blocks = dict(corp.external)
    if not ext_blocks and not corp.families:
        raise NotEvaluable("corpus has no external embeddings and no feature families")
    cmap = pd.read_parquet(ctx.stage_dir("S1") / "consolidation_map.parquet")
    K = ctx.profile["eval"]["knn_k"]
    ex = ctx.profile["ext"]
    targets = {}
    if ext_blocks:
        for n, X in blocks_to_cid(corp, es.vocab, cmap, ext_blocks).items():
            targets[f"external:{n}"] = ("external", X)
    for n, X in blocks_to_cid(corp, es.vocab, cmap).items():
        targets[f"family:{n}"] = ("baked_in_family", X)
    rows = []
    for m in [x for x in MAIN + ["composition_acoustic", "composition_lyric"] if x in models]:
        E = load_embeddings(ctx, m)[es.idx]
        for name, (kind, X) in targets.items():
            Xi = X[es.idx]
            ok = np.isfinite(Xi).all(axis=1)
            if ok.sum() < 300:
                continue
            rep = agreement_report(E[ok], Xi[ok], K, es.groups[ok], min(ex["agree_queries"], int(ok.sum())), min(ex["agree_sub"], int(ok.sum())), ctx.seed("S9E5", m, name) % 10_000)
            rows.append({"model": m, "target": name, "kind": kind, "n": int(ok.sum()), **rep})
    if not rows:
        raise NotEvaluable("no external/feature block covers >= 300 eval tracks")
    df = pd.DataFrame(rows)
    df.to_parquet(ctx.stage_dir("S9") / "E5_agreement.parquet")
    return {"rows": rows, "note": "external = learned embeddings that were NOT compound inputs; baked_in_family = the compound-graph inputs themselves (agreement is expected there)"}


# ======================================================================================================================
# E6 - temporal stability
# ======================================================================================================================
def _quick_modes(E: np.ndarray, cat: np.ndarray, ic: dict, seed: int) -> list[np.ndarray]:
    Er = residualise(E, cat)
    r = stable_ica(Er, ic["n_components"], max(3, ic["n_seeds"] // 2), ic["split_half_threshold"], 5, seed=seed % 10_000)
    if r.directions.shape[0] == 0:
        return []
    src = project_sources(Er, r.directions)
    out = []
    for f in range(src.shape[1]):
        for md in factor_modes(E, src[:, f], f, ic["top_quantile"], range(ic["gmm_k_range"][0], ic["gmm_k_range"][-1] + 1), ic["min_members"], seed=(seed + f) % 10_000):
            out.append(np.asarray(md.members, dtype=np.int64))
    return out


def e6_temporal(ctx: Ctx, es: EvalSet, models: list[str]) -> dict:
    corp = corpus_of(ctx, load_families=False)
    sm = corp.sessions_meta
    col = next((c for c in ("ts_start", "ts", "timestamp") if sm is not None and c in sm.columns), None)
    if col is None:
        raise NotEvaluable("sessions carry no timestamps (playlists / count-only logs): temporal stability needs dated sessions")
    ts = pd.to_numeric(sm[col], errors="coerce").to_numpy(float)
    ok = np.isfinite(ts)
    span_years = (np.nanmax(ts) - np.nanmin(ts)) / (365.25 * 86400) if ok.any() else 0.0
    need = float(ctx.profile["ext"]["temporal_min_years"])
    if span_years < need:
        raise NotEvaluable(f"session timestamps span {span_years:.2f} years < required {need}")
    cmap = pd.read_parquet(ctx.stage_dir("S1") / "consolidation_map.parquet")
    V = len(es.vocab)
    S = (corp.sessions @ merge_matrix(cmap, corp.n_tracks, V)).tocsr()
    S.data[:] = 1.0
    keep = np.flatnonzero((np.diff(S.indptr) >= 2) & ok)
    S, ts = S[keep], ts[keep]
    cut = np.median(ts)
    halves = {"early": S[np.flatnonzero(ts <= cut)], "late": S[np.flatnonzero(ts > cut)]}
    cfg, tc = ctx.profile["cooc"], dict(ctx.profile["train"])
    key = f"{ctx.corpus}/{ctx.profile_name}/E6"
    embs, masks = {}, {}
    spec = model_specs(tc["ii_repeat"])["cooc"]
    from ..training import GraphInputs
    for h, X in halves.items():
        r, c, v = npmi_edges(X, max(3, cfg["min_count"] // 2), max(2, cfg["min_pair_count"] - 1), cfg["topk"], cfg["chunk"], True)
        gi = GraphInputs(V, edges_to_csr(r, c, v, V))
        embs[h], _ = train_one(spec, gi, tc, 0, ctx.stage_dir("S9") / "E6" / h, key, device=ctx.opt("device"))
        masks[h] = np.bincount(r, minlength=V) > 0
    shared = np.flatnonzero(masks["early"] & masks["late"])
    if len(shared) < 300:
        raise NotEvaluable(f"only {len(shared)} tracks have edges in both time halves")
    pdisp = procrustes_disparity(embs["early"][shared], embs["late"][shared], n_null=10, seed=ctx.seed("S9E6"))
    cat = es.vocab["category"].astype(str).to_numpy()
    modes = {h: _quick_modes(embs[h][shared], cat[shared], ctx.profile["ica"], ctx.seed("S9E6", h)) for h in halves}
    mp = mode_persistence(modes["early"], modes["late"], embs["early"][shared], embs["late"][shared], n_null=20, seed=ctx.seed("S9E6mp")) if modes["early"] and modes["late"] else None
    return {"span_years": float(span_years), "n_sessions_early": int(halves["early"].shape[0]), "n_sessions_late": int(halves["late"].shape[0]), "n_shared_tracks": int(len(shared)),
            "procrustes": pdisp, "mode_persistence": mp, "n_modes": {h: len(m) for h, m in modes.items()}}


# ======================================================================================================================
# E7 - curation ablation (H4)
# ======================================================================================================================
def _purity_items(E: np.ndarray, labels: np.ndarray, groups: np.ndarray, k: int) -> np.ndarray:
    nn = knn_indices(E, k, None, groups)
    return (labels[nn] == labels[:, None]).mean(1)


def _baseline(lab: np.ndarray) -> float:
    return float((np.unique(lab, return_counts=True)[1] ** 2).sum() / len(lab) ** 2)


def _cluster_mean_ci(x: np.ndarray, g: np.ndarray, n_boot: int, seed: int) -> tuple[float, float, float, np.ndarray]:
    _, inv = np.unique(g, return_inverse=True)
    G = inv.max() + 1
    sums, cnts = np.bincount(inv, weights=x, minlength=G), np.bincount(inv, minlength=G).astype(float)
    W = np.random.default_rng(seed).multinomial(G, np.full(G, 1.0 / G), size=n_boot).astype(float)
    boots = (W @ sums) / np.maximum(W @ cnts, 1e-9)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(x.mean()), float(lo), float(hi), boots


def _lift_arm(E, lab, art, k, m, n_boot, seed) -> dict:
    """Artist-disjoint kNN purity lift on a size-matched random sub-sample of m rows (pool size affects purity, so both arms use the same m)."""
    rng = np.random.default_rng(seed)
    sel = np.sort(rng.choice(len(E), m, replace=False)) if len(E) > m else np.arange(len(E))
    x = _purity_items(E[sel], lab[sel], art[sel], k)
    base = _baseline(lab[sel])
    mean, lo, hi, boots = _cluster_mean_ci(x, art[sel], n_boot, seed)
    return {"m": int(len(sel)), "purity": mean, "baseline": base, "lift": mean / base, "lift_ci": (lo / base, hi / base), "boots": boots / base}


def e7_curation(ctx: Ctx, es: EvalSet, models: list[str]) -> dict:
    _need(models, "cooc", "cooc_raw")
    corp = corpus_of(ctx, load_families=False)
    cmap = pd.read_parquet(ctx.stage_dir("S1") / "consolidation_map.parquet")
    from .common import model_dir
    raw_ids = np.load(model_dir(ctx, "cooc_raw") / "raw_ids.npy")
    Eraw = load_embeddings(ctx, "cooc_raw")
    cid_of_raw = cmap.set_index("raw_id")["cid"].reindex(raw_ids).to_numpy()
    in_eval = np.isin(cid_of_raw, es.idx)                              # the same track pool as the curated arm, but with the duplicates left in
    K, nb = ctx.profile["eval"]["knn_k"], ctx.profile["eval"]["knn_boot"]
    cats = es.vocab["category"].astype(str).to_numpy()
    first_region = np.array([(s.split("|")[0] if s else "") for s in es.vocab["region_tags"].astype(str).to_numpy()])
    label_sets = {"category": cats, "region": first_region}
    art_raw = corp.tracks["artist_id"].to_numpy()[raw_ids]
    Ecur = load_embeddings(ctx, "cooc")
    out = {"label_sets": {}}
    verdicts = []
    for li, (lname, lab_v) in enumerate(label_sets.items()):
        bad = {"", "Other"}
        cur_rows = np.array([i for i in es.idx if lab_v[i] not in bad], dtype=np.int64)
        raw_rows = np.flatnonzero(in_eval & np.array([lab_v[int(c)] not in bad if c >= 0 else False for c in cid_of_raw]))
        if len(cur_rows) < 300 or len(raw_rows) < 300 or len(np.unique(lab_v[cur_rows])) < 2:
            continue
        m = int(min(len(cur_rows), len(raw_rows), 30000))
        cur = _lift_arm(Ecur[cur_rows], lab_v[cur_rows], es.vocab["artist_id"].to_numpy()[cur_rows], K, m, max(nb, 200), ctx.seed("S9E7", lname, "cur") % 10_000)
        raw = _lift_arm(Eraw[raw_rows], lab_v[cid_of_raw[raw_rows].astype(int)], art_raw[raw_rows], K, m, max(nb, 200), ctx.seed("S9E7", lname, "raw") % 10_000)
        diff = cur["boots"] - raw["boots"]                              # arms resampled independently (different artists/rows)
        lo, hi = np.percentile(diff, [2.5, 97.5])
        p = float(min(1.0, 2 * (min((diff <= 0).mean(), (diff >= 0).mean()) + 1 / len(diff))))
        v = "supported" if lo > 0 else ("not supported" if hi < 0 else "inconclusive")
        verdicts.append(v)
        out["label_sets"][lname] = {"curated": {k: v_ for k, v_ in cur.items() if k != "boots"}, "raw": {k: v_ for k, v_ in raw.items() if k != "boots"},
                                    "lift_diff": float(cur["lift"] - raw["lift"]), "diff_ci": (float(lo), float(hi)), "p_boot": p, "verdict": v,
                                    "n_raw_entries": int(len(raw_rows)), "n_curated": int(len(cur_rows))}
    if not verdicts:
        raise NotEvaluable("too few labelled tracks for a size-matched purity comparison")
    H4 = {"claim": "curated vocabulary > size-matched raw on artist-disjoint kNN purity lift (lift difference > 0)",
          "status": "supported" if all(v == "supported" for v in verdicts) else ("not supported" if any(v == "not supported" for v in verdicts) and not any(v == "supported" for v in verdicts) else "mixed / inconclusive"),
          "verdicts": dict(zip(out["label_sets"], verdicts))}
    out["H4"] = H4
    # ---- (b) rule level: how far merging goes (strict vs song-level), rules only, on the raw-trained embedding ----
    lab = cats[cid_of_raw[in_eval].astype(int)]
    Ein, artin = Eraw[in_eval], art_raw[in_eval]
    recs = list(zip(corp.tracks["artist"].to_numpy()[raw_ids][in_eval].tolist(), corp.tracks["title"].to_numpy()[raw_ids][in_eval].tolist()))
    rule = {"raw (no merging)": {"n": int(len(Ein)), **_rule_arm(Ein, lab, artin, K, ctx)}}
    for level in ("strict", "song"):
        groups = exact_groups(recs, level)
        gid = np.empty(len(recs), dtype=np.int64)
        for g, (_, members) in enumerate(groups.items()):
            gid[members] = g
        Eg, uid = consolidate_embeddings(Ein, gid)
        first = pd.Series(np.arange(len(gid))).groupby(gid).first().reindex(uid).to_numpy()
        rule[f"{level}-level merge"] = {"n": int(len(uid)), **_rule_arm(Eg, lab[first], artin[first], K, ctx)}
    out["rule_level"] = rule
    return out


def _rule_arm(E, lab, art, K, ctx) -> dict:
    keep = np.flatnonzero(~np.isin(lab, ["", "Other"]))
    if len(keep) < 300 or len(np.unique(lab[keep])) < 2:
        return {"lift": float("nan")}
    r = _lift_arm(E[keep], lab[keep], art[keep], K, min(len(keep), 20000), 200, ctx.seed("S9E7rule") % 10_000)
    return {"lift": r["lift"], "lift_ci": r["lift_ci"], "purity": r["purity"], "m": r["m"]}


# ======================================================================================================================
# E8 / H8 - subset re-training
# ======================================================================================================================
def _problem(ctx: Ctx, es: EvalSet, n_tracks: int, tag: str):
    """SubsetProblem on `n_tracks` eval tracks (hub tracks only when a compound graph exists) + aligned targets/groups."""
    corp = corpus_of(ctx, load_families=True)
    cmap = pd.read_parquet(ctx.stage_dir("S1") / "consolidation_map.parquet")
    feats = blocks_to_cid(corp, es.vocab, cmap)
    fk = {n: family_kind(n) for n in feats}
    S = sp.load_npz(ctx.stage_dir("S3") / "sessions_cid.npz").tocsr()
    pool = np.arange(len(es.idx))
    if feats:
        has = np.zeros(len(es.idx), bool)
        for X in feats.values():
            has |= np.isfinite(X[es.idx]).all(axis=1)
        pool = pool[has]
    if len(pool) < 300:
        raise NotEvaluable(f"only {len(pool)} eval tracks with compound features (need >= 300)")
    rng = np.random.default_rng(ctx.seed("S9sub", tag))
    pos = np.sort(rng.choice(pool, min(n_tracks, len(pool)), replace=False))
    sub = es.idx[pos]
    T = {k: {**v, "y": np.asarray(v["y"])[pos]} for k, v in es.primary().items()}
    prob = SubsetProblem(sub, S, feats, fk, ctx.profile["cooc"], ctx.profile["compounds"], len(es.vocab), seed=ctx.seed("S9sub", tag) % 10_000)
    return prob, T, es.groups[pos]


def e8_multiverse(ctx: Ctx, es: EvalSet, models: list[str]) -> dict:
    ext = ctx.profile["ext"]
    prob, T, groups = _problem(ctx, es, ext["multiverse_tracks"], "E8")
    if not T:
        raise NotEvaluable("no held-out / external probe targets")
    tc = {**ctx.profile["train"], **ext["multiverse_train"]}
    key, ns, nr = f"{ctx.corpus}/{ctx.profile_name}", *_cv(ctx)
    d = ctx.stage_dir("S9") / "E8"
    cache: dict = {}

    def rho(model, topk, k_scale, top_m):
        kk = (model, topk if model == "cooc" else None, None if model == "cooc" else k_scale, None if model == "cooc" else top_m)
        if kk not in cache:
            tag = "_".join(str(x) for x in kk if x is not None)
            E = train_models(prob, [model], tc, d, tag, key, topk, k_scale, top_m, device=ctx.opt("device"))[model]
            cache[kk] = (_rho(ctx, E, T, groups, ctx.seed("S9E8")), float(participation_ratio(E)))
        return cache[kk]

    def cell(k_scale, top_m, npmi_topk, seed):
        rc, prc = rho("cooc", npmi_topk, k_scale, top_m)
        rp, prp = rho("composition", npmi_topk, k_scale, top_m)
        return {"rho_cooc": rc, "rho_composition": rp, "delta": rp - rc, "pr_cooc": prc, "pr_composition": prp}
    grid = ext["multiverse"]
    df = run_grid(cell, grid, seed=ctx.seed("S9E8grid"))
    d.mkdir(parents=True, exist_ok=True)
    df.to_parquet(d / "grid.parquet")
    summ = summarise(df, "delta", criterion=0.0, higher_is_better=True, params=list(grid))
    return {"n_tracks": int(len(prob.sub)), "grid": {k: list(v) for k, v in grid.items()}, "summary_delta_composition_minus_cooc": summ,
            "cells": df.drop(columns=["error"], errors="ignore").to_dict("records"),
            "note": "whole grid reported (never the best cell). k_scale multiplies the cluster counts; the session-gap axis needs the raw event log and is fixed by the adapter (documented limitation)."}


def h8_paper_vs_standard(ctx: Ctx, es: EvalSet, models: list[str]) -> dict:
    if ctx.profile_name == "paper":
        raise NotEvaluable("run profile is already `paper`")
    ext = ctx.profile["ext"]
    paper_over = ext.get("h8_paper_train") or (ctx.ws.profiles().get("paper") or {}).get("train", {})
    if not paper_over:
        raise NotEvaluable("no `paper` profile train overrides to compare against")
    prob, T, groups = _problem(ctx, es, ext["h8_tracks"], "H8")
    if not T:
        raise NotEvaluable("no held-out / external probe targets")
    base_tc = dict(ctx.profile["train"])
    paper_tc = {**base_tc, **paper_over}
    key, d = f"{ctx.corpus}/{ctx.profile_name}", ctx.stage_dir("S9") / "H8"
    topk = ctx.profile["cooc"]["topk"]
    res = {}
    for tag, tc in (("standard", base_tc), ("paper", paper_tc)):
        Es = train_models(prob, MAIN, tc, d, tag, key, topk, 1.0, ctx.profile["compounds"]["top_m"], device=ctx.opt("device"))
        res[tag] = {m: {"rho": _rho(ctx, E, T, groups, ctx.seed("S9H8")), "PR": float(participation_ratio(E))} for m, E in Es.items()}
    if not all(all(m in r for m in MAIN) for r in res.values()):
        raise NotEvaluable("Composition/Core could not be trained on the subset")
    verdict = {}
    for tag, r in res.items():
        verdict[tag] = {"composition_minus_cooc": r["composition"]["rho"] - r["cooc"]["rho"], "rho_order": sorted(MAIN, key=lambda m: -r[m]["rho"]), "lowest_PR": min(MAIN, key=lambda m: r[m]["PR"])}
    same_sign = np.sign(verdict["standard"]["composition_minus_cooc"]) == np.sign(verdict["paper"]["composition_minus_cooc"])
    same_order = verdict["standard"]["rho_order"] == verdict["paper"]["rho_order"]
    same_pr = verdict["standard"]["lowest_PR"] == verdict["paper"]["lowest_PR"]
    status = "supported" if (same_sign and same_pr) else "not supported"
    return {"claim": "conclusions on the standard training budget match those on the paper budget (subset)", "status": status, "n_tracks": int(len(prob.sub)),
            "same_H1_direction": bool(same_sign), "same_rho_ordering": bool(same_order), "same_lowest_PR": bool(same_pr), "per_profile": res, "verdict": verdict,
            "paper_train_overrides": paper_over}


# ======================================================================================================================
@register
class Extensions(Stage):
    name, title, deps = "S9", "Extensions (E1-E8, H4, H6, H8)", ("S5", "S7")

    def params(self, ctx):
        return {"v": 1, "ext": ctx.profile["ext"], "eval": ctx.profile["eval"], "models": available_models(ctx)}

    def outputs(self, ctx):
        return [ctx.results_dir / "S9_extensions.json"]

    def run(self, ctx: Ctx) -> StageResult:
        es = build_eval(ctx)
        models = [m for m in available_models(ctx)]
        res, out, errors = StageResult(), {}, {}
        ctx.stage_dir("S9").mkdir(parents=True, exist_ok=True)
        plan = [("E1", e1_compression), ("E2", e2_attribution), ("E3", e3_partial_hub), ("E4", e4_equal_token), ("E5", e5_agreement),
                ("E6", e6_temporal), ("E7", e7_curation), ("E8", e8_multiverse), ("H8", h8_paper_vs_standard)]
        for name, fn in plan:
            try:
                log.info("[S9] %s", name)
                out[name] = {"status": "done", **fn(ctx, es, models)}
            except NotEvaluable as e:
                out[name] = {"status": "not evaluable", "reason": str(e)}
                res.notes.append(f"{name}: not evaluable - {e}")
            except Exception as e:  # noqa: BLE001 - one broken extension must not sink the rest
                errors[name] = traceback.format_exc()
                out[name] = {"status": "error", "reason": f"{type(e).__name__}: {e}"}
                log.error("[S9] %s failed: %s\n%s", name, e, errors[name])
                res.status = PARTIAL
                res.notes.append(f"{name} failed: {type(e).__name__}: {e}")
        hyp = {}
        for h, path in (("4", ("E7", "H4")), ("6", ("E1", "H6"))):
            v = out.get(path[0], {}).get(path[1])
            if v:
                hyp[h] = v
        if out.get("H8", {}).get("status") in ("supported", "not supported"):
            hyp["8"] = {k: out["H8"][k] for k in ("claim", "status", "same_H1_direction", "same_rho_ordering", "same_lowest_PR")}
        ctx.save_result("S9_extensions", jsonable(out))
        ctx.save_result("S9_hypotheses", jsonable(hyp))
        (ctx.reports_dir / "s9_extensions.md").write_text(report(out), encoding="utf-8")
        res.metrics = {k: v.get("status") for k, v in out.items()} | {f"H{k}": v["status"] for k, v in hyp.items()}
        return res


def report(out: dict) -> str:
    L = ["# S9 - extensions", ""]
    for k, v in out.items():
        L.append(f"- **{k}**: {v['status']}" + (f" - {v['reason']}" if v.get("reason") else ""))
    e1 = out.get("E1", {})
    if e1.get("status") == "done":
        L += ["", "## E1 smallest configs", ""]
        for m, d in e1["models"].items():
            for thr, sc in d["small"].items():
                L.append(f"- {m} >= {float(thr):.0%} retention: " + (f"{sc['method']} {sc['dims']}d {sc['dtype']} = {sc['size_mb']:.2f} MB (retention {sc['retention']:.3f})" if sc else "no config reaches it"))
        L.append(f"\nH6: **{e1['H6']['status']}**")
    if out.get("E7", {}).get("status") == "done":
        L.append(f"\nH4: **{out['E7']['H4']['status']}** {out['E7']['H4']['verdicts']}")
    if out.get("E8", {}).get("status") == "done":
        L += ["", "## E8 (whole grid)", "", out["E8"]["summary_delta_composition_minus_cooc"].get("statement", "")]
    if out.get("H8", {}).get("status") in ("supported", "not supported"):
        L.append(f"\nH8: **{out['H8']['status']}** (H1 direction same: {out['H8']['same_H1_direction']}, lowest-PR same: {out['H8']['same_lowest_PR']})")
    return "\n".join(L) + "\n"
