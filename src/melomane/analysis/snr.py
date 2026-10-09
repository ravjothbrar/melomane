"""Orthogonal-residual SNR of poles (Epicure supplement `direction_orthogonal`; our operationalisation, recorded here).

For a pole p and a seed s (unit vectors): q = slerp(s, p, theta); N = the top-k cosine neighbours of q (seed excluded, and
its artist group if given). Remove the seed component from every neighbour, r_i = n_i - (n_i . s) s, and split r_i into
  t_i   = r_i . p_perp                     with p_perp = unit(p - (p . s) s)      (movement ALONG the pole),
  e_i   = r_i - t_i p_perp                 (everything orthogonal to the plane (s, p)).
  signal   = var_i(t_i)                    (sample variance over the k neighbours)
  noise    = sum_d var_i(e_id) / (D - 2)   (mean per-dimension residual variance orthogonal to (s, p))
  snr      = signal / noise                (`snr_total` divides by the summed residual variance instead).
High SNR: the neighbours spread out along the pole far more than along any typical orthogonal direction, i.e. the pole is
a clean, one-dimensional control (neighbours trade seed for pole inside the SLERP plane rather than scattering
sideways). Neighbours are unit vectors, so this is a cosine-geometry quantity. Vectorised over many seeds and poles with chunked matmuls."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..geometry import unit
from ._retrieval import GroupIndex, batch_slerp, topk_cosine, unit32


def _snr_from_neighbours(N: np.ndarray, seed_u: np.ndarray, pole_u: np.ndarray) -> dict:
    """N (n x k x D) neighbour unit vectors, seed_u / pole_u (n x D) float64 unit. Returns per-pair arrays."""
    n, k, D = N.shape
    N = N.astype(np.float64)
    s = seed_u
    pp = pole_u - (pole_u * s).sum(1, keepdims=True) * s
    pp = unit(pp)
    R = N - np.einsum("nkd,nd->nk", N, s)[:, :, None] * s[:, None, :]
    t = np.einsum("nkd,nd->nk", R, pp)
    E = R - t[:, :, None] * pp[:, None, :]
    sig = t.var(axis=1, ddof=1)
    tot = E.var(axis=1, ddof=1).sum(1)
    noise = tot / max(D - 2, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return {"snr": sig / noise, "snr_total": sig / tot, "signal_var": sig, "noise_var": noise,
                "signal_mean": t.mean(1)}


def orthogonal_residual_snr(E: np.ndarray, seeds, poles, theta: float = 60.0, k: int = 10, exclude_groups=None,
                            paired: bool = False, chunk: int = 256) -> dict:
    """SNR for every (pole, seed) combination (paired=False: `poles` (P x D), `seeds` (S,) -> arrays of shape P x S) or for
    aligned pairs (paired=True: poles (n x D), seeds (n,) -> arrays of shape (n,)).

    Returns dict(snr, snr_total, signal_var, noise_var, signal_mean, snr_median (per pole, or overall if paired), theta, k)."""
    U = unit32(E)
    seeds = np.asarray(seeds, dtype=np.int64)
    poles = np.asarray(poles, dtype=np.float64)
    if paired:
        if len(poles) != len(seeds):
            raise ValueError("paired=True needs one pole per seed")
        pi, si = np.arange(len(seeds)), np.arange(len(seeds))
        shape = (len(seeds),)
    else:
        P, S = len(poles), len(seeds)
        pi, si = np.repeat(np.arange(P), S), np.tile(np.arange(S), P)
        shape = (P, S)
    gi = None if exclude_groups is None else GroupIndex(exclude_groups)
    keys = ("snr", "snr_total", "signal_var", "noise_var", "signal_mean")
    out = {key: np.empty(len(pi)) for key in keys}
    step = max(chunk, 1)
    for a in range(0, len(pi), step):
        sl = slice(a, a + step)
        sd = seeds[si[sl]]
        Uv = U[sd].astype(np.float64)
        Pv = unit(poles[pi[sl]])
        Q = batch_slerp(Uv, Pv, theta)
        idx, _ = topk_cosine(U, Q.astype(np.float32), k, exclude_items=sd, group_index=gi, chunk=step)
        r = _snr_from_neighbours(U[idx], Uv, Pv)
        for key in keys:
            out[key][sl] = r[key]
    res = {key: v.reshape(shape) for key, v in out.items()}
    res["snr_median"] = float(np.nanmedian(res["snr"])) if paired else np.nanmedian(res["snr"], axis=1)
    res["theta"], res["k"] = float(theta), int(k)
    return res


def snr_ranking(result: dict, pole_names=None) -> pd.DataFrame:
    """Rank poles (rows of a non-paired result) by median SNR across seeds (descending)."""
    snr = result["snr"]
    if snr.ndim != 2:
        raise ValueError("ranking needs a non-paired result (poles x seeds)")
    names = np.arange(snr.shape[0]) if pole_names is None else np.asarray(pole_names)
    df = pd.DataFrame({"pole": names, "snr_median": np.nanmedian(snr, axis=1), "snr_mean": np.nanmean(snr, axis=1),
                       "snr_total_median": np.nanmedian(result["snr_total"], axis=1),
                       "n_seeds": np.isfinite(snr).sum(1)})
    df = df.sort_values("snr_median", ascending=False, kind="stable").reset_index(drop=True)
    df.insert(0, "rank", np.arange(1, len(df) + 1))
    return df
