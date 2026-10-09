"""E2 attribution: exact 2-player Shapley split of the gain of Composition over Cooc between the acoustic and the lyric
families (PLAN §6 S9).

Value function v(S) = score(system built from families S) - score(Cooc), v({}) = 0 (Cooc is the baseline), with
v({A}) = acoustic-only, v({L}) = lyric-only, v({A, L}) = combined Composition. Shapley:
  phi_A = 1/2 [v(A) - v({})] + 1/2 [v(AL) - v(L)],   phi_L = 1/2 [v(L) - v({})] + 1/2 [v(AL) - v(A)],
  phi_A + phi_L = v(AL) (efficiency);  synergy = v(AL) - v(A) - v(L)  (positive = the families help each other)."""
from __future__ import annotations

import numpy as np

from ._retrieval import cluster_index, resample_items


def shapley_2player(v_a: float, v_l: float, v_al: float, v_empty: float = 0.0) -> dict:
    """Exact Shapley values for the two-player game with v({}) = v_empty, v({A}) = v_a, v({L}) = v_l, v({A,L}) = v_al."""
    phi_a = 0.5 * (v_a - v_empty) + 0.5 * (v_al - v_l)
    phi_l = 0.5 * (v_l - v_empty) + 0.5 * (v_al - v_a)
    return {"phi_acoustic": float(phi_a), "phi_lyric": float(phi_l), "gain": float(v_al - v_empty),
            "synergy": float(v_al - v_a - v_l + v_empty)}


def _from_means(cooc: float, acoustic: float, lyric: float, combined: float) -> dict:
    return shapley_2player(acoustic - cooc, lyric - cooc, combined - cooc, 0.0)


def shapley_attribution(cooc, acoustic, lyric, combined, clusters=None, n_boot: int = 2000, seed: int = 0,
                        alpha: float = 0.05) -> dict:
    """Shapley split of (combined - cooc) between acoustic and lyric families from score ARRAYS of the four systems
    (aligned units: probes, folds or tracks). Bootstrap over units (or `clusters`, e.g. artists) jointly across the four
    systems gives percentile CIs. Returns phi_acoustic, phi_lyric, gain, synergy, share_acoustic, share_lyric (of the gain;
    NaN when |gain| is ~0), the means of the four systems, and ci = {name: (lo, hi)}."""
    arrs = [np.asarray(x, dtype=float) for x in (cooc, acoustic, lyric, combined)]
    if len({a.shape for a in arrs}) != 1 or arrs[0].ndim != 1:
        raise ValueError("the four score arrays must be 1-D and of equal length")
    ok = np.all(np.isfinite(np.vstack(arrs)), axis=0)
    arrs = [a[ok] for a in arrs]
    cl = None if clusters is None else np.asarray(clusters)[ok]
    n = len(arrs[0])
    if n < 2:
        raise ValueError("need at least 2 aligned scores")
    means = [a.mean() for a in arrs]
    est = _from_means(*means)
    rng = np.random.default_rng(seed)
    _, order, starts = cluster_index(cl, n)
    names = ("phi_acoustic", "phi_lyric", "gain", "synergy")
    boots = np.empty((n_boot, 4))
    for b in range(n_boot):
        ix = resample_items(order, starts, rng)
        r = _from_means(*(a[ix].mean() for a in arrs))
        boots[b] = [r[k] for k in names]
    ci = {k: (float(np.percentile(boots[:, j], 100 * alpha / 2)), float(np.percentile(boots[:, j], 100 * (1 - alpha / 2))))
          for j, k in enumerate(names)}
    gain = est["gain"]
    share_ok = abs(gain) > 1e-12
    est.update({"share_acoustic": est["phi_acoustic"] / gain if share_ok else float("nan"),
                "share_lyric": est["phi_lyric"] / gain if share_ok else float("nan"),
                "means": {"cooc": float(means[0]), "acoustic": float(means[1]), "lyric": float(means[2]),
                          "combined": float(means[3])},
                "ci": ci, "n": int(n), "n_boot": int(n_boot)})
    return est
