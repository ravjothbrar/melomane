"""SLERP direction arithmetic (Epicure `epicure.py` semantics): rotate a unit seed toward a unit direction by theta
inside the plane spanned by (seed, direction):  q = cos(theta) v + sin(theta) d_perp,  d_perp = unit(d - (d.v) v).
At theta=0 q = seed; cos(q, seed) = cos(theta) exactly (theta=60deg -> 0.5)."""
from __future__ import annotations
import numpy as np
from .geometry import unit


def slerp(seed: np.ndarray, direction: np.ndarray, theta_deg: float) -> np.ndarray:
    v = unit(seed); d = unit(direction)
    d_perp = d - (d @ v) * v
    n = np.linalg.norm(d_perp)
    if n < 1e-9:
        return v
    th = np.deg2rad(theta_deg)
    return unit(np.cos(th) * v + np.sin(th) * (d_perp / n))


def compose_poles(poles: list[np.ndarray], weights: list[float] | None = None) -> np.ndarray:
    """Multi-constraint steering: weighted unit-sum of poles, returned as a unit direction."""
    w = np.ones(len(poles)) if weights is None else np.asarray(weights, float)
    return unit(sum(wi * unit(p) for wi, p in zip(w, poles)))


def nearest(E_unit: np.ndarray, q: np.ndarray, k: int = 10, exclude: list[int] | None = None):
    sims = E_unit @ unit(q)
    if exclude:
        sims[np.asarray(exclude)] = -np.inf
    idx = np.argpartition(-sims, k)[:k]
    idx = idx[np.argsort(-sims[idx])]
    return idx, sims[idx]


def theta_sweep(E_unit: np.ndarray, seed_idx: int, pole: np.ndarray, thetas=(0, 15, 30, 45, 60, 75, 90), k=10):
    """For each theta: mean cosine of the top-k retrievals to the pole and to the seed (the paper's qualitative
    'target-dominance' pattern, made quantitative)."""
    out = []
    for th in thetas:
        q = slerp(E_unit[seed_idx], pole, th)
        idx, _ = nearest(E_unit, q, k, exclude=[seed_idx])
        out.append((th, float((E_unit[idx] @ unit(pole)).mean()), float((E_unit[idx] @ E_unit[seed_idx]).mean())))
    return out
