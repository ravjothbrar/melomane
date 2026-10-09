"""Orthogonal Procrustes alignment between two embedding matrices on a shared vocabulary (PLAN §7.2, Epicure supplement).

Given row-aligned A (n x da) and B (n x db): centre (optional), scale each to unit Frobenius norm (optional) and find the
orthogonal R minimising ||A R - B||_F (SVD of A^T B). We report
  disparity   = 1 - (sum of singular values of A^T B)^2  (scipy.spatial.procrustes convention, 0 = identical up to rotation),
  row_cosine  = mean cosine between aligned rows,
  axis_cosine = per-coordinate cosine between columns of A R and B (meaningful when the axes of B are meaningful),
  pole_cosine = cosine between supplied directions of A (mapped by R) and their counterparts in B (optional)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..geometry import unit


def shared_rows(ids_a, ids_b) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Row positions of the ids present in both id arrays (unique ids): (shared_ids, pos_in_a, pos_in_b), sorted by id."""
    ids_a, ids_b = np.asarray(ids_a), np.asarray(ids_b)
    shared, ia, ib = np.intersect1d(ids_a, ids_b, assume_unique=False, return_indices=True)
    return shared, ia, ib


def _colcos(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    num = (P * Q).sum(0)
    den = np.linalg.norm(P, axis=0) * np.linalg.norm(Q, axis=0)
    return np.where(den > 0, num / np.maximum(den, 1e-300), 0.0)


def _fit(Xa: np.ndarray, Xb: np.ndarray) -> tuple[np.ndarray, float]:
    U, S, Vt = np.linalg.svd(Xa.T @ Xb, full_matrices=False)
    return U @ Vt, float(S.sum())


def procrustes_align(E_a: np.ndarray, E_b: np.ndarray, ids_a=None, ids_b=None, center: bool = True,
                     normalise: bool = True, n_null: int = 0, seed: int = 0, poles_a=None, poles_b=None,
                     return_aligned: bool = False) -> dict:
    """Align E_a to E_b by an orthogonal map on their shared rows.

    Rows are assumed already aligned unless `ids_a`/`ids_b` (row identifiers, e.g. cids) are given, in which case the
    intersection is used. Embedding dimensions may differ (R is then a partial isometry). n_null > 0 adds the disparity
    obtained after randomly permuting B's rows (chance level). poles_a/poles_b (k x da / k x db, matched rows) give
    `pole_cosine`. Returns dict(R, disparity, rss, row_cosine, axis_cosine, axis_cosine_mean, n_shared, ...)."""
    A, B = np.asarray(E_a, dtype=np.float64), np.asarray(E_b, dtype=np.float64)
    if (ids_a is None) != (ids_b is None):
        raise ValueError("give both ids_a and ids_b, or neither")
    if ids_a is not None:
        _, ia, ib = shared_rows(ids_a, ids_b)
        A, B = A[ia], B[ib]
    elif len(A) != len(B):
        raise ValueError("row counts differ; pass ids_a / ids_b to align on the shared vocabulary")
    n = len(A)
    if n < 2:
        raise ValueError("need at least 2 shared rows")
    if center:
        A, B = A - A.mean(0), B - B.mean(0)
    if normalise:
        A, B = A / np.linalg.norm(A), B / np.linalg.norm(B)
    R, ssum = _fit(A, B)
    al = A @ R
    den = np.linalg.norm(A) * np.linalg.norm(B)
    rss = float(((al - B) ** 2).sum())
    disparity = float(1 - (ssum / den) ** 2)
    axis = _colcos(al, B)
    out = {"R": R, "disparity": max(disparity, 0.0), "rss": rss, "n_shared": int(n),
           "row_cosine": float((unit(al) * unit(B)).sum(1).mean()), "axis_cosine": axis, "axis_cosine_mean": float(np.mean(np.abs(axis)))}
    if n_null > 0:
        rng = np.random.default_rng(seed)
        nulls = []
        for _ in range(n_null):
            Bp = B[rng.permutation(n)]
            _, s = _fit(A, Bp)
            nulls.append(1 - (s / den) ** 2)
        out["null_disparity_mean"] = float(np.mean(nulls))
        out["null_disparity_min"] = float(np.min(nulls))
    if poles_a is not None:
        if poles_b is None:
            raise ValueError("poles_b required with poles_a")
        pa, pb = np.atleast_2d(np.asarray(poles_a, float)), np.atleast_2d(np.asarray(poles_b, float))
        pc = (unit(pa @ R) * unit(pb)).sum(1)
        out["pole_cosine"] = pc
        out["pole_cosine_mean"] = float(pc.mean())
    if return_aligned:
        out["aligned"] = al
        out["target"] = B
    return out


def procrustes_matrix(models: dict[str, np.ndarray], ids: dict[str, np.ndarray] | None = None, center: bool = True,
                      normalise: bool = True, n_null: int = 0, seed: int = 0) -> dict:
    """All model pairs. `ids[name]` (row identifiers) is needed only when vocabularies differ; otherwise rows must align.
    Returns dict(names, disparity (M x M, diag 0), row_cosine (diag 1), axis_cosine (mean |cos|, diag 1), n_shared, table)
    where `table` is a tidy DataFrame of the upper-triangle pairs."""
    names = list(models)
    M = len(names)
    disp = np.zeros((M, M))
    rowc = np.eye(M)
    axc = np.eye(M)
    nsh = np.zeros((M, M), dtype=int)
    rows = []
    for i in range(M):
        nsh[i, i] = len(models[names[i]])
        for j in range(i + 1, M):
            a, b = names[i], names[j]
            r = procrustes_align(models[a], models[b], None if ids is None else ids[a], None if ids is None else ids[b],
                                 center=center, normalise=normalise, n_null=n_null, seed=seed)
            disp[i, j] = disp[j, i] = r["disparity"]
            rowc[i, j] = rowc[j, i] = r["row_cosine"]
            axc[i, j] = axc[j, i] = r["axis_cosine_mean"]
            nsh[i, j] = nsh[j, i] = r["n_shared"]
            row = {"model_a": a, "model_b": b, "disparity": r["disparity"], "row_cosine": r["row_cosine"],
                   "axis_cosine_mean": r["axis_cosine_mean"], "n_shared": r["n_shared"]}
            if n_null > 0:
                row["null_disparity_mean"] = r["null_disparity_mean"]
            rows.append(row)
    return {"names": names, "disparity": disp, "row_cosine": rowc, "axis_cosine": axc, "n_shared": nsh,
            "table": pd.DataFrame(rows)}
