"""S7 evaluation battery: artist-grouped probes (ordinal / binary / continuous), stratum means, exact geometry summary,
and size-matched-bootstrap kNN-purity lift (PLAN §6 S7, §7.1-7.2).

Every probe takes `groups` (= artist id): folds keep whole artists together (`probes.repeated_splits`) and kNN retrieval is
artist-disjoint. Passing groups=None is allowed but every output row is then flagged `leaky=True`.

probe_battery output (tidy, one row per target x reading): target, stratum, kind, reading, metric ('rho' | 'd'), value,
rho, d, ci_low, ci_high, sd, n (items after masking), n_folds, leaky.
  readings: 'centroid' (ordinal/binary: train-fold centroid axis, paper 1), 'tercile_centroid' (continuous: centroid(top
  train-tercile) - centroid(bottom train-tercile), evaluated by Spearman against the CONTINUOUS y), 'ridge' (second reading:
  fold-trained ridge direction; ordinal and continuous targets).
  value = fold-mean held-out rho (or Cohen's d) exactly as `probes.*_cv`; ci_low/ci_high = percentile bootstrap (artists
  resampled) of the POOLED out-of-fold statistic (per-fold z-scored projections averaged over repeats), so the CI describes
  the pooled estimate, not the fold-mean."""
from __future__ import annotations

import warnings
from typing import Mapping

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from .. import probes
from ..geometry import mean_pairwise_cosine, participation_ratio, unit
from ._retrieval import cluster_index, resample_items
from .stats import bootstrap_ci


def _oof(E, fit_y, eval_y, groups, n_splits, n_repeats, seed, axis_fn, binary=False):
    """Pooled out-of-fold projections (z-scored per fold, averaged over repeats) and fold-wise scores."""
    n = len(E)
    acc, cnt = np.zeros(n), np.zeros(n)
    fold_scores = []
    for tr, te in probes.repeated_splits(n, n_splits, n_repeats, seed, groups):
        if binary:
            yt, ye = fit_y[tr], fit_y[te]
            if yt.sum() < 2 or (~yt).sum() < 2 or ye.sum() < 2 or (~ye).sum() < 2:
                continue
        elif len(np.unique(fit_y[tr])) < 2:
            continue
        a = axis_fn(E[tr], fit_y[tr])
        if a is None:
            continue
        p = E[te] @ unit(a)
        if binary:
            fold_scores.append(probes.cohens_d(p[fit_y[te]], p[~fit_y[te]]))
        else:
            r = _spearman(p, eval_y[te])
            if np.isfinite(r):
                fold_scores.append(r)
        sd = p.std()
        acc[te] += (p - p.mean()) / sd if sd > 0 else 0.0
        cnt[te] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        oof = np.where(cnt > 0, acc / cnt, np.nan)
    return oof, np.asarray(fold_scores)


def _spearman(x, y) -> float:
    rx, ry = rankdata(x) - (len(x) + 1) / 2, rankdata(y) - (len(y) + 1) / 2
    den = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / den) if den > 0 else float("nan")


def _d(p: np.ndarray, yes: np.ndarray) -> float:
    a, b = p[yes], p[~yes]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    sp_ = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return float((a.mean() - b.mean()) / sp_) if sp_ > 0 else float("nan")


def _boot_ci(stat, arrays, groups, n_boot, seed, alpha=0.05):
    if n_boot <= 0:
        return float("nan"), float("nan")
    _, order, starts = cluster_index(groups, len(arrays[0]))
    rng = np.random.default_rng(seed)
    vals = np.array([stat(*(a[ix] for a in arrays)) for ix in (resample_items(order, starts, rng) for _ in range(n_boot))])
    if not np.isfinite(vals).any():
        return float("nan"), float("nan")
    return tuple(float(x) for x in np.nanpercentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)]))


def _tercile_axis(Etr, ytr):
    lo, hi = np.quantile(ytr, [1 / 3, 2 / 3])
    a, b = ytr <= lo, ytr >= hi
    if a.sum() < 2 or b.sum() < 2 or lo == hi:
        return None
    return Etr[b].mean(0) - Etr[a].mean(0)


def _ridge_axis(Etr, ytr):
    return probes._axis(Etr, ytr, "ridge")


def _centroid_axis(Etr, ytr):
    return probes._axis(Etr, ytr, "centroid")


def probe_battery(E: np.ndarray, targets: Mapping[str, Mapping], groups=None, n_splits: int = 5, n_repeats: int = 3,
                  seed: int = 0, ci_boot: int = 200, ridge: bool = True) -> pd.DataFrame:
    """Run the grouped probe battery. targets[name] = dict(kind='ordinal'|'binary'|'continuous', y=array (V,),
    mask=optional bool (V,), stratum=optional str). See module docstring for the output columns."""
    E = np.asarray(E, dtype=np.float64)
    G = None if groups is None else np.asarray(groups)
    rows = []
    for name, spec in targets.items():
        kind = spec["kind"]
        if kind not in ("ordinal", "binary", "continuous"):
            raise ValueError(f"{name}: unknown kind {kind!r}")
        y = np.asarray(spec["y"])
        mask = spec.get("mask")
        m = np.ones(len(E), bool) if mask is None else np.asarray(mask, bool)
        m &= np.isfinite(y.astype(float)) if kind != "binary" else True
        Em, ym = E[m], y[m]
        Gm = None if G is None else G[m]
        common = {"target": name, "stratum": spec.get("stratum"), "kind": kind, "n": int(m.sum()), "leaky": G is None}
        readings = []
        if kind == "ordinal":
            yo = ym.astype(float)
            readings.append(("centroid", "rho", yo, _centroid_axis, lambda yy=yo: probes.ordinal_projection_cv(
                Em, yy, n_splits, n_repeats, "centroid", seed, None, Gm)))
            if ridge:
                readings.append(("ridge", "rho", yo, _ridge_axis, lambda yy=yo: probes.ordinal_projection_cv(
                    Em, yy, n_splits, n_repeats, "ridge", seed, None, Gm)))
        elif kind == "continuous":
            yc = ym.astype(float)
            readings.append(("tercile_centroid", "rho", yc, _tercile_axis, None))
            if ridge:
                readings.append(("ridge", "rho", yc, _ridge_axis, lambda yy=yc: probes.ordinal_projection_cv(
                    Em, yy, n_splits, n_repeats, "ridge", seed, None, Gm)))
        else:
            yb = ym.astype(bool)
            readings.append(("centroid", "d", yb, None, lambda yy=yb: probes.binary_separation_cv(
                Em, yy, n_splits, n_repeats, seed, Gm)))
        for reading, metric, yv, axis_fn, run in readings:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if metric == "d":
                    if yv.sum() < 4 or (~yv).sum() < 4:
                        continue
                    res = run()
                    oof, folds = _oof(Em, yv, None, Gm, n_splits, n_repeats, seed,
                                      lambda Et, yt: unit(Et[yt].mean(0) - Et[~yt].mean(0)), binary=True)
                    value, sd, nf = res["d"], res["d_sd"], res["n"]
                    ok = np.isfinite(oof)
                    lo, hi = _boot_ci(lambda p, yy: _d(p, yy), (oof[ok], yv[ok]), None if Gm is None else Gm[ok],
                                      ci_boot, seed)
                else:
                    if len(np.unique(yv)) < 2:
                        continue
                    oof, folds = _oof(Em, yv, yv, Gm, n_splits, n_repeats, seed, axis_fn)
                    if run is not None:
                        res = run()
                        value, sd, nf = res["rho"], res["rho_sd"], res["n"]
                    else:
                        value, sd, nf = (float(folds.mean()) if len(folds) else float("nan"),
                                         float(folds.std()) if len(folds) else float("nan"), int(len(folds)))
                    ok = np.isfinite(oof)
                    lo, hi = _boot_ci(_spearman, (oof[ok], yv[ok]), None if Gm is None else Gm[ok], ci_boot, seed)
            rows.append({**common, "reading": reading, "metric": metric, "value": float(value),
                         "rho": float(value) if metric == "rho" else float("nan"),
                         "d": float(value) if metric == "d" else float("nan"),
                         "ci_low": lo, "ci_high": hi, "sd": float(sd), "n_folds": int(nf)})
    cols = ["target", "stratum", "kind", "reading", "metric", "value", "rho", "d", "ci_low", "ci_high", "sd", "n",
            "n_folds", "leaky"]
    return pd.DataFrame(rows, columns=cols)


def stratum_means(df: pd.DataFrame, strata_map: Mapping | None = None, n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Mean probe score per (stratum, reading, metric) with a bootstrap CI over targets.
    strata_map: {target: stratum} or {stratum: [targets]}; default = the df's own `stratum` column. Targets missing from
    the map are dropped. Columns: stratum, reading, metric, mean, sd, ci_low, ci_high, n_targets."""
    d = df.copy()
    if strata_map is not None:
        if all(isinstance(v, (list, tuple, set, np.ndarray)) for v in strata_map.values()):
            strata_map = {t: s for s, ts in strata_map.items() for t in ts}
        d["stratum"] = d["target"].map(strata_map)
    d = d[d["stratum"].notna()]
    rows = []
    for (s, rd, mt), g in d.groupby(["stratum", "reading", "metric"], sort=True):
        v = g["value"].to_numpy(float)
        v = v[np.isfinite(v)]
        if not len(v):
            continue
        ci = bootstrap_ci(v, n_boot=n_boot, seed=seed) if len(v) > 1 else {"ci_low": float("nan"), "ci_high": float("nan")}
        rows.append({"stratum": s, "reading": rd, "metric": mt, "mean": float(v.mean()),
                     "sd": float(v.std(ddof=1)) if len(v) > 1 else float("nan"), "ci_low": ci["ci_low"],
                     "ci_high": ci["ci_high"], "n_targets": int(len(v))})
    return pd.DataFrame(rows, columns=["stratum", "reading", "metric", "mean", "sd", "ci_low", "ci_high", "n_targets"])


def geometry_summary(E: np.ndarray) -> dict:
    """Exact participation ratio (centred covariance), exact mean pairwise cosine (via ||sum u_i||^2), plus size/norm."""
    E = np.asarray(E, dtype=np.float64)
    return {"n": int(len(E)), "dim": int(E.shape[1]), "participation_ratio": participation_ratio(E),
            "mean_pairwise_cosine": mean_pairwise_cosine(E), "mean_norm": float(np.linalg.norm(E, axis=1).mean())}


def _baseline(lab: np.ndarray) -> float:
    return float((np.unique(lab, return_counts=True)[1] ** 2).sum() / len(lab) ** 2)


def knn_battery(E: np.ndarray, label_sets: Mapping, groups=None, k: int = 10, n_boot: int = 200, seed: int = 0,
                subsample: int | None = None, max_subsample: int = 5000, n_null: int = 5) -> pd.DataFrame:
    """Artist-disjoint kNN-purity lift per label set with a SIZE-MATCHED bootstrap: every bootstrap draw takes `subsample`
    tagged tracks without replacement (default: the smallest tagged count over all label sets, capped at `max_subsample`),
    so lifts are comparable across label sets. lift = purity / random-label baseline (sum of squared class shares).
    label_sets[name] = labels array or dict(labels=..., tagged=optional bool mask). Also reports `null_lift` (labels shuffled
    within the subsample; ~1 if there is no leakage) and purity/baseline means. Columns: label_set, n_tagged, n_classes,
    m, purity, baseline, lift, lift_sd, ci_low, ci_high, null_lift."""
    E = np.asarray(E)
    G = None if groups is None else np.asarray(groups)
    prepared = {}
    for name, ls in label_sets.items():
        if isinstance(ls, Mapping):
            lab, tag = np.asarray(ls["labels"]), ls.get("tagged")
        else:
            lab, tag = np.asarray(ls), None
        idx = np.arange(len(E)) if tag is None else np.flatnonzero(np.asarray(tag, bool))
        prepared[name] = (lab, idx)
    m = subsample or min(min(len(i) for _, i in prepared.values()), max_subsample)
    rows = []
    for j, (name, (lab, idx)) in enumerate(prepared.items()):
        rng = np.random.default_rng(np.random.SeedSequence([seed, j]))
        if len(idx) < m:
            raise ValueError(f"{name}: only {len(idx)} tagged tracks, need {m}")
        pur, base, lifts, nulls = [], [], [], []
        for b in range(n_boot):
            s = rng.choice(idx, size=m, replace=False)
            g = None if G is None else G[s]
            p = probes.knn_purity(E[s], lab[s], k, groups=g)
            bl = _baseline(lab[s])
            pur.append(p)
            base.append(bl)
            lifts.append(p / bl)
            if b < n_null:
                pn = probes.knn_purity(E[s], rng.permutation(lab[s]), k, groups=g)
                nulls.append(pn / bl)
        lifts = np.asarray(lifts)
        rows.append({"label_set": name, "n_tagged": int(len(idx)), "n_classes": int(len(np.unique(lab[idx]))), "m": int(m),
                     "purity": float(np.mean(pur)), "baseline": float(np.mean(base)), "lift": float(lifts.mean()),
                     "lift_sd": float(lifts.std()), "ci_low": float(np.percentile(lifts, 2.5)),
                     "ci_high": float(np.percentile(lifts, 97.5)),
                     "null_lift": float(np.mean(nulls)) if nulls else float("nan")})
    return pd.DataFrame(rows)
