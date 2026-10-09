"""Statistics used across the evaluation (PLAN §7.4): clustered paired bootstrap, Holm, permutation tests, kappa.

Cluster = artist. Resampling whole artists respects the dependence between tracks of one artist (artist leakage is the
#1 methodological trap in this project). All functions are seeded and return plain dicts of floats/arrays."""
from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.stats import rankdata, spearmanr

from ._retrieval import cluster_index, resample_items


def _pct_ci(x: np.ndarray, alpha: float) -> tuple[float, float]:
    lo, hi = np.nanpercentile(x, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def paired_cluster_bootstrap(scores_a, scores_b, clusters=None, n_boot: int = 2000, seed: int = 0,
                             alpha: float = 0.05) -> dict:
    """Paired difference (a - b) of per-item scores with artists (clusters) resampled with replacement.

    Returns mean_diff, ci_low/ci_high (percentile, 1-alpha), effect_size (Cohen's d_z = mean diff / sd of item diffs),
    p_boot (two-sided bootstrap p for diff = 0, add-one corrected), n, n_clusters, mean_a, mean_b. Pairs with NaN dropped."""
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("scores_a and scores_b must be 1-D and of equal length")
    cl = None if clusters is None else np.asarray(clusters)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    cl = None if cl is None else cl[ok]
    d = a - b
    n = len(d)
    if n < 2:
        raise ValueError("need at least 2 paired observations")
    inv, _, _ = cluster_index(cl, n)
    G = int(inv.max()) + 1
    sums = np.bincount(inv, weights=d, minlength=G)
    cnts = np.bincount(inv, minlength=G).astype(float)
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    block = max(1, min(n_boot, 4_000_000 // max(G, 1)))
    for s in range(0, n_boot, block):
        m = min(block, n_boot - s)
        idx = rng.integers(0, G, size=(m, G))
        boots[s:s + m] = sums[idx].sum(1) / cnts[idx].sum(1)
    md = float(d.mean())
    sd = float(d.std(ddof=1))
    k_le, k_ge = int((boots <= 0).sum()), int((boots >= 0).sum())
    p = min(1.0, 2 * (min(k_le, k_ge) + 1) / (n_boot + 1))
    lo, hi = _pct_ci(boots, alpha)
    return {"mean_diff": md, "ci_low": lo, "ci_high": hi, "effect_size": md / sd if sd > 0 else float("nan"),
            "p_boot": float(p), "n": int(n), "n_clusters": G, "n_boot": int(n_boot),
            "mean_a": float(a.mean()), "mean_b": float(b.mean())}


def holm(pvalues, alpha: float = 0.05) -> dict:
    """Holm-Bonferroni step-down. NaN p-values are ignored (kept NaN). Returns p_adj (monotone, capped at 1) and reject."""
    p = np.asarray(pvalues, dtype=float)
    adj = np.full(p.shape, np.nan)
    ok = np.flatnonzero(np.isfinite(p))
    m = len(ok)
    if m:
        order = ok[np.argsort(p[ok], kind="stable")]
        run = 0.0
        for rank, i in enumerate(order):
            run = max(run, (m - rank) * p[i])
            adj[i] = min(1.0, run)
    return {"p_adj": adj, "reject": np.where(np.isfinite(adj), adj <= alpha, False), "alpha": alpha, "m": int(m)}


def permutation_test(stat_fn: Callable[[np.ndarray], float], labels, n: int = 10000, seed: int = 0,
                     alternative: str = "two-sided") -> dict:
    """Label-permutation test: p = (1 + #{null >= obs}) / (n + 1) (add-one, never exactly 0).

    stat_fn(labels) -> float is evaluated on the observed labels and on `n` random permutations of them.
    alternative: 'two-sided' (|stat|), 'greater', 'less'."""
    if alternative not in ("two-sided", "greater", "less"):
        raise ValueError(alternative)
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    obs = float(stat_fn(labels))
    null = np.empty(n)
    for i in range(n):
        null[i] = stat_fn(rng.permutation(labels))
    if alternative == "two-sided":
        k = int((np.abs(null) >= abs(obs) - 1e-12).sum())
    elif alternative == "greater":
        k = int((null >= obs - 1e-12).sum())
    else:
        k = int((null <= obs + 1e-12).sum())
    return {"observed": obs, "p": (1 + k) / (n + 1), "null_mean": float(null.mean()), "null_sd": float(null.std()),
            "n_perm": int(n), "alternative": alternative}


def bootstrap_ci(x, clusters=None, stat: str | Callable = "mean", n_boot: int = 2000, alpha: float = 0.05,
                 seed: int = 0) -> dict:
    """Percentile bootstrap CI of a statistic of 1-D `x`, resampling clusters (artists) if given, else items.
    stat: 'mean' | 'median' or a callable(array) -> float."""
    x = np.asarray(x, dtype=float)
    cl = None if clusters is None else np.asarray(clusters)
    ok = np.isfinite(x)
    x = x[ok]
    cl = None if cl is None else cl[ok]
    n = len(x)
    if n == 0:
        raise ValueError("no finite values")
    inv, order, starts = cluster_index(cl, n)
    rng = np.random.default_rng(seed)
    G = len(starts) - 1
    if stat == "mean":
        est = float(x.mean())
        sums = np.bincount(inv, weights=x, minlength=G)
        cnts = np.bincount(inv, minlength=G).astype(float)
        boots = np.empty(n_boot)
        block = max(1, min(n_boot, 4_000_000 // max(G, 1)))
        for s in range(0, n_boot, block):
            m = min(block, n_boot - s)
            idx = rng.integers(0, G, size=(m, G))
            boots[s:s + m] = sums[idx].sum(1) / cnts[idx].sum(1)
    else:
        fn = {"median": np.median}.get(stat, stat) if isinstance(stat, str) else stat
        if not callable(fn):
            raise ValueError(f"unknown stat {stat!r}")
        est = float(fn(x))
        boots = np.array([fn(x[resample_items(order, starts, rng)]) for _ in range(n_boot)], dtype=float)
    lo, hi = _pct_ci(boots, alpha)
    return {"estimate": est, "ci_low": lo, "ci_high": hi, "se": float(np.nanstd(boots)), "n": int(n), "n_clusters": G}


def _rank_corr(x: np.ndarray, y: np.ndarray) -> float:
    rx, ry = rankdata(x), rankdata(y)
    rx -= rx.mean()
    ry -= ry.mean()
    den = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / den) if den > 0 else float("nan")


def spearman_ci(x, y, clusters=None, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> dict:
    """Spearman rho with a (cluster-)bootstrap percentile CI and the asymptotic p-value from scipy."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    cl = None if clusters is None else np.asarray(clusters)[ok]
    n = len(x)
    if n < 3:
        raise ValueError("need at least 3 observations")
    res = spearmanr(x, y)
    _, order, starts = cluster_index(cl, n)
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        ix = resample_items(order, starts, rng)
        boots[b] = _rank_corr(x[ix], y[ix])
    lo, hi = _pct_ci(boots, alpha)
    return {"rho": float(res.statistic), "ci_low": lo, "ci_high": hi, "p": float(res.pvalue), "n": int(n),
            "n_clusters": len(starts) - 1}


def _kappa_from_confusion(O: np.ndarray, W: np.ndarray) -> tuple[float, float, float]:
    n = O.sum()
    O = O / n
    E = np.outer(O.sum(1), O.sum(0))
    num, den = (W * O).sum(), (W * E).sum()
    kappa = float(1 - num / den) if den > 0 else float("nan")
    return kappa, float(1 - num), float(1 - den)      # kappa, observed agreement (weighted), chance agreement (weighted)


def _agreement(a, b, labels, weights, n_boot, clusters, seed, alpha) -> dict:
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("ratings must be 1-D and of equal length")
    labs = np.unique(np.concatenate([a, b])) if labels is None else np.asarray(labels)
    L = len(labs)
    pos = {v: i for i, v in enumerate(labs.tolist())}
    ia = np.array([pos[v] for v in a.tolist()])
    ib = np.array([pos[v] for v in b.tolist()])
    ii, jj = np.meshgrid(np.arange(L), np.arange(L), indexing="ij")
    if weights is None:
        W = (ii != jj).astype(float)
    elif weights == "linear":
        W = np.abs(ii - jj) / max(L - 1, 1)
    elif weights == "quadratic":
        W = (ii - jj) ** 2 / max(L - 1, 1) ** 2
    else:
        raise ValueError(weights)

    def conf(ix_a, ix_b):
        O = np.zeros((L, L))
        np.add.at(O, (ix_a, ix_b), 1)
        return O

    kappa, po, pe = _kappa_from_confusion(conf(ia, ib), W)
    out = {"kappa": kappa, "po": po, "pe": pe, "n": int(len(a)), "n_labels": int(L)}
    if n_boot > 0:
        _, order, starts = cluster_index(clusters, len(a))
        rng = np.random.default_rng(seed)
        boots = np.array([_kappa_from_confusion(conf(ia[ix], ib[ix]), W)[0]
                          for ix in (resample_items(order, starts, rng) for _ in range(n_boot))])
        out["ci_low"], out["ci_high"] = _pct_ci(boots, alpha)
    return out


def cohens_kappa(a, b, labels=None, n_boot: int = 0, clusters=None, seed: int = 0, alpha: float = 0.05) -> dict:
    """Cohen's kappa between two raters (nominal). Returns kappa, po, pe, n (+ bootstrap CI if n_boot > 0).
    kappa is NaN when chance agreement is 1 (both raters constant)."""
    return _agreement(a, b, labels, None, n_boot, clusters, seed, alpha)


def weighted_kappa(a, b, weights: str = "quadratic", labels=None, n_boot: int = 0, clusters=None, seed: int = 0,
                   alpha: float = 0.05) -> dict:
    """Weighted kappa for ordinal ratings ('linear' or 'quadratic'); `labels` gives the order (default: sorted unique)."""
    if weights not in ("linear", "quadratic"):
        raise ValueError("weights must be 'linear' or 'quadratic'")
    return _agreement(a, b, labels, weights, n_boot, clusters, seed, alpha)
