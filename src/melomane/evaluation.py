"""Paired, artist-clustered model comparison on probe strata (PLAN §7.4).

Spearman's rho of pooled out-of-fold projections equals mean(z(rank p) * z(rank y)), i.e. it decomposes into per-item scores.  So
the difference between two models is a mean of per-item score differences and can be bootstrapped by resampling ARTISTS (cluster
bootstrap) - jointly across all targets of a stratum, so stratum-mean differences get a proper CI.
"""
from __future__ import annotations
from typing import Mapping
import numpy as np
from scipy.stats import rankdata

from .analysis import battery as _b


def _z(x):
    s = x.std()
    return (x - x.mean()) / s if s > 0 else np.zeros_like(x)


def item_scores(E: np.ndarray, spec: Mapping, groups: np.ndarray, n_splits: int, n_repeats: int, seed: int) -> tuple[np.ndarray, np.ndarray] | None:
    """(index into E, per-item score) for one ordinal/continuous target; None if the target cannot be evaluated."""
    y = np.asarray(spec["y"], dtype=float)
    m = np.isfinite(y)
    if spec.get("mask") is not None:
        m &= np.asarray(spec["mask"], bool)
    if m.sum() < 100 or np.nanstd(y[m]) == 0:
        return None
    idx = np.flatnonzero(m)
    Em, ym, Gm = E[idx].astype(np.float64), y[idx], groups[idx]
    axis = _b._tercile_axis if spec["kind"] == "continuous" else _b._centroid_axis
    oof, _ = _b._oof(Em, ym, ym, Gm, n_splits, n_repeats, seed, axis)
    ok = np.isfinite(oof)
    if ok.sum() < 100:
        return None
    zp, zy = _z(rankdata(oof[ok])), _z(rankdata(ym[ok]))
    return idx[ok], zp * zy


def paired_strata(E_a: np.ndarray, E_b: np.ndarray, targets: Mapping[str, Mapping], groups: np.ndarray, strata: Mapping[str, list[str]],
                  n_splits: int = 5, n_repeats: int = 3, n_boot: int = 1000, seed: int = 0) -> dict:
    """Delta rho (a - b) per stratum with artist-cluster bootstrap CI (targets resampled jointly), plus per-target deltas."""
    art_ids, inv = np.unique(groups, return_inverse=True)
    G = len(art_ids)
    rows_t, sums, cnts, names = {}, [], [], []
    for name, spec in targets.items():
        if spec["kind"] == "binary":
            continue
        ra, rb = item_scores(E_a, spec, groups, n_splits, n_repeats, seed), item_scores(E_b, spec, groups, n_splits, n_repeats, seed)
        if ra is None or rb is None:
            continue
        common, ia, ib = np.intersect1d(ra[0], rb[0], return_indices=True)
        if len(common) < 100:
            continue
        d = ra[1][ia] - rb[1][ib]
        a_id = inv[common]
        sums.append(np.bincount(a_id, weights=d, minlength=G)); cnts.append(np.bincount(a_id, minlength=G).astype(float))
        names.append(name)
        rows_t[name] = {"delta_rho": float(d.mean()), "rho_a": float(ra[1].mean()), "rho_b": float(rb[1].mean()), "n": int(len(common))}
    if not names:
        return {"targets": {}, "strata": {}}
    S, C = np.stack(sums, 1), np.stack(cnts, 1)                     # (G x T)
    rng = np.random.default_rng(seed)
    W = rng.multinomial(G, np.full(G, 1.0 / G), size=n_boot).astype(np.float64)          # cluster resampling weights (B x G)
    boot_t = (W @ S) / np.maximum(W @ C, 1e-9)                       # (B x T) delta rho per target per resample
    point_t = S.sum(0) / np.maximum(C.sum(0), 1e-9)
    out = {"targets": {}, "strata": {}}
    for j, n in enumerate(names):
        lo, hi = np.percentile(boot_t[:, j], [2.5, 97.5])
        out["targets"][n] = {**rows_t[n], "ci_low": float(lo), "ci_high": float(hi)}
    for sname, members in strata.items():
        cols = [names.index(t) for t in members if t in names]
        if not cols:
            continue
        pt, bt = point_t[cols].mean(), boot_t[:, cols].mean(1)
        lo, hi = np.percentile(bt, [2.5, 97.5])
        p = float(2 * min((bt <= 0).mean(), (bt >= 0).mean()))
        out["strata"][sname] = {"delta_rho": float(pt), "ci_low": float(lo), "ci_high": float(hi), "p_boot": max(p, 1.0 / n_boot), "n_targets": len(cols),
                                "share_targets_positive": float(np.mean([out["targets"][names[c]]["delta_rho"] > 0 for c in cols]))}
    return out


def item_matrix(E: np.ndarray, targets: Mapping[str, Mapping], groups: np.ndarray, n_splits: int, n_repeats: int, seed: int) -> tuple[np.ndarray, dict]:
    """Per-row mean (over the non-binary targets) of the per-item rho contributions, NaN where no target covers the row, and the
    per-target mean.  mean(nanmean(array)) is the family's mean rho; the array feeds paired artist-clustered bootstraps directly."""
    n = len(E)
    tot, cnt, per = np.zeros(n), np.zeros(n), {}
    for name, spec in targets.items():
        if spec["kind"] == "binary":
            continue
        r = item_scores(E, spec, groups, n_splits, n_repeats, seed)
        if r is None:
            continue
        i, sc = r
        tot[i] += sc
        cnt[i] += 1
        per[name] = float(sc.mean())
    out = np.full(n, np.nan)
    m = cnt > 0
    out[m] = tot[m] / cnt[m]
    return out, per
