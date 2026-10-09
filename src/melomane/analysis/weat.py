"""WEAT (Caliskan, Bryson & Narayanan 2017) on embedding rows, plus the Melomane region x energy design (PLAN §7.2).

s(w, A, B) = mean_a cos(w, a) - mean_b cos(w, b).   Test statistic = mean_{x in X} s(x) - mean_{y in Y} s(y)
(the paper's sum-difference divided by the set size, identical for |X| = |Y|).   Effect size
d = (mean_X s - mean_Y s) / sd_{w in X u Y} s(w)  (sample sd, ddof = 1).   The permutation p-value re-partitions X u Y into
sets of the original sizes: exact enumeration when the number of partitions <= n_perm, else `n_perm` random partitions
(add-one corrected)."""
from __future__ import annotations

from itertools import combinations
from math import comb

import numpy as np
import pandas as pd

from ..geometry import unit
from .stats import holm


def _assoc(U: np.ndarray, X, Y, A, B) -> tuple[np.ndarray, np.ndarray]:
    sx = U[X] @ U[A].T
    sy = U[Y] @ U[A].T
    s_x = sx.mean(1) - (U[X] @ U[B].T).mean(1)
    s_y = sy.mean(1) - (U[Y] @ U[B].T).mean(1)
    return s_x, s_y


def weat(E: np.ndarray, X, Y, A, B, n_perm: int = 10000, seed: int = 0) -> dict:
    """WEAT effect size d and permutation p-values (one- and two-sided) for target sets X, Y and attribute sets A, B,
    each an integer index array into the rows of E. Sets must be non-empty and X/Y, A/B should not overlap."""
    U = unit(np.asarray(E, dtype=np.float64))
    X, Y, A, B = (np.asarray(v, dtype=np.int64) for v in (X, Y, A, B))
    if min(len(X), len(Y), len(A), len(B)) < 1 or len(X) + len(Y) < 3:
        raise ValueError("WEAT needs non-empty sets and at least 3 target items")
    s_x, s_y = _assoc(U, X, Y, A, B)
    s_all = np.concatenate([s_x, s_y])
    nx, N = len(X), len(s_all)
    stat = float(s_x.mean() - s_y.mean())
    sd = float(s_all.std(ddof=1))
    d = stat / sd if sd > 0 else float("nan")
    total = s_all.sum()
    n_part = comb(N, nx)
    if n_part <= n_perm:
        sums = np.array([s_all[list(c)].sum() for c in combinations(range(N), nx)])
        null = sums / nx - (total - sums) / (N - nx)
        exact = True
        p1 = float((null >= stat - 1e-12).mean())
        p2 = float((np.abs(null) >= abs(stat) - 1e-12).mean())
    else:
        rng = np.random.default_rng(seed)
        null = np.empty(n_perm)
        block = max(1, min(n_perm, 2_000_000 // N))
        for s in range(0, n_perm, block):
            m = min(block, n_perm - s)
            sel = np.argsort(rng.random((m, N)), axis=1)[:, :nx]
            sums = s_all[sel].sum(1)
            null[s:s + m] = sums / nx - (total - sums) / (N - nx)
        exact = False
        p1 = float((1 + (null >= stat - 1e-12).sum()) / (n_perm + 1))
        p2 = float((1 + (np.abs(null) >= abs(stat) - 1e-12).sum()) / (n_perm + 1))
    return {"d": d, "statistic": stat, "p_one_sided": p1, "p_two_sided": p2, "exact": exact,
            "n_perm": int(len(null)), "n_x": len(X), "n_y": len(Y), "n_a": len(A), "n_b": len(B),
            "s_x_mean": float(s_x.mean()), "s_y_mean": float(s_y.mean())}


def weat_region_energy(E: np.ndarray, region, energy, high=None, low=None, regions=None, groups=None,
                       max_targets: int = 200, max_attrs: int = 200, min_size: int = 10, n_perm: int = 10000,
                       seed: int = 0, attr_from: str = "other_regions") -> pd.DataFrame:
    """Region-groups x high-vs-low-energy WEAT design (one row per ordered-once region pair).

    Targets X, Y = tracks of region r1 / r2 (subsampled to equal size <= max_targets). Attributes A, B = high- / low-energy
    tracks (energy >= `high`, <= `low`; defaults: max / min level present) NOT in r1 or r2 (attr_from='other_regions') or
    merely not sampled as targets (attr_from='any'); tracks by an artist (`groups`) that also supplies a target are dropped
    from the attribute sets so that artist identity cannot drive the association. `region` may hold -1 / NaN / None for
    untagged tracks (never used as targets). Effect size d > 0 means r1 is closer to high energy than r2.
    Returns a tidy DataFrame with Holm-adjusted two-sided p over all pairs (`p_holm`)."""
    E = np.asarray(E)
    region = np.asarray(region)
    energy = np.asarray(energy, dtype=float)
    if len(region) != len(E) or len(energy) != len(E):
        raise ValueError("region and energy must have one entry per embedding row")
    tagged = np.array([r is not None and not (isinstance(r, float) and np.isnan(r)) and r != -1 for r in region.tolist()])
    levels = np.unique(energy[np.isfinite(energy)])
    hi = levels.max() if high is None else high
    lo = levels.min() if low is None else low
    is_high, is_low = energy >= hi, energy <= lo
    regs = sorted(set(region[tagged].tolist())) if regions is None else list(regions)
    g = None if groups is None else np.asarray(groups)
    rng = np.random.default_rng(seed)
    rows = []
    for i, r1 in enumerate(regs):
        for r2 in regs[i + 1:]:
            m1 = np.flatnonzero(tagged & (region == r1))
            m2 = np.flatnonzero(tagged & (region == r2))
            n_t = min(len(m1), len(m2), max_targets)
            if n_t < min_size:
                continue
            X = np.sort(rng.choice(m1, n_t, replace=False))
            Y = np.sort(rng.choice(m2, n_t, replace=False))
            avail = np.ones(len(E), bool)
            avail[X] = False
            avail[Y] = False
            if attr_from == "other_regions":
                avail &= ~(tagged & ((region == r1) | (region == r2)))
            elif attr_from != "any":
                raise ValueError(attr_from)
            if g is not None:
                avail &= ~np.isin(g, np.unique(np.concatenate([g[X], g[Y]])))
            ha, la = np.flatnonzero(avail & is_high), np.flatnonzero(avail & is_low)
            n_a = min(len(ha), len(la), max_attrs)
            if n_a < min_size:
                continue
            A = rng.choice(ha, n_a, replace=False)
            B = rng.choice(la, n_a, replace=False)
            res = weat(E, X, Y, A, B, n_perm=n_perm, seed=int(rng.integers(2 ** 31)))
            rows.append({"region_x": r1, "region_y": r2, "n_x": res["n_x"], "n_y": res["n_y"], "n_a": n_a, "n_b": n_a,
                         "d": res["d"], "statistic": res["statistic"], "p_one_sided": res["p_one_sided"],
                         "p_two_sided": res["p_two_sided"]})
    df = pd.DataFrame(rows, columns=["region_x", "region_y", "n_x", "n_y", "n_a", "n_b", "d", "statistic",
                                     "p_one_sided", "p_two_sided"])
    df["p_holm"] = holm(df["p_two_sided"].to_numpy())["p_adj"] if len(df) else np.array([])
    return df
