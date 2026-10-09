"""E1 compression frontier (PLAN §6 S9, hypothesis H6): dimensionality reduction (PCA / random projection) x storage dtype
(fp32 / fp16 / per-row symmetric int8) versus probe retention and probe-independent neighbour recall.

Geometry convention: with center=True (default) the embedding is mean-centred before projection (the mean vector is stored
and counted); neighbour recall is then measured against the mean-centred full-precision embedding, so that a lossless
config scores 1. With center=False the raw cosine geometry is used on both sides. `size_bytes` is exact: payload
(V x dims x bytes-per-value) + int8 per-row fp32 scales (4 V) + projection matrix (dims x D fp32, not needed for method
'none') + mean vector (D fp32, only if centred). Sizes are reported in MB = 1e6 bytes."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from .agreement import knn_indices, neighbour_recall

_BYTES = {"fp32": 4, "fp16": 2, "int8": 1}


@dataclass
class CompressedEmbedding:
    """A compressed copy of an embedding. `view` is the dequantised float32 (V x dims) matrix probes should be run on."""
    method: str
    dims: int
    dtype: str
    view: np.ndarray
    size_bytes: int
    center: bool
    projection: np.ndarray | None = field(default=None, repr=False)   # (dims x D) float32, applied as (E - mean) @ projection.T
    mean: np.ndarray | None = field(default=None, repr=False)
    codes: np.ndarray | None = field(default=None, repr=False)        # stored payload (float32 / float16 / int8)
    scales: np.ndarray | None = field(default=None, repr=False)       # int8 per-row scales (V,) float32

    @property
    def size_mb(self) -> float:
        return self.size_bytes / 1e6

    def encode(self, X: np.ndarray) -> np.ndarray:
        """Project new full-dimensional vectors into this compressed space (float32, before quantisation)."""
        X = np.asarray(X, dtype=np.float32)
        if self.mean is not None:
            X = X - self.mean
        return X if self.projection is None else X @ self.projection.T


def quantise_int8(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-row symmetric int8: scale_i = max|x_i| / 127, code = round(x / scale) in [-127, 127]. Returns (codes, scales)."""
    X = np.asarray(X, dtype=np.float32)
    scales = np.abs(X).max(1) / 127.0
    scales = np.where(scales > 0, scales, 1.0).astype(np.float32)
    codes = np.clip(np.rint(X / scales[:, None]), -127, 127).astype(np.int8)
    return codes, scales


def dequantise_int8(codes: np.ndarray, scales: np.ndarray) -> np.ndarray:
    return codes.astype(np.float32) * scales[:, None]


def _projection(Xc: np.ndarray, method: str, dims: int, seed: int) -> np.ndarray | None:
    D = Xc.shape[1]
    if method == "none":
        return None
    if not 1 <= dims <= D:
        raise ValueError(f"dims must be in [1, {D}]")
    if method == "pca":
        cov = (Xc.T.astype(np.float64) @ Xc.astype(np.float64)) / len(Xc)
        w, V = np.linalg.eigh(cov)
        top = np.argsort(w)[::-1][:dims]
        return V[:, top].T.astype(np.float32)
    if method == "rp":
        rng = np.random.default_rng(seed)
        return (rng.standard_normal((dims, D)) / np.sqrt(dims)).astype(np.float32)
    raise ValueError(f"unknown method {method!r} (use 'pca', 'rp' or 'none')")


def _reduce(E: np.ndarray, method: str, dims: int, seed: int, center: bool):
    X = np.asarray(E, dtype=np.float32)
    mean = X.mean(0, dtype=np.float64).astype(np.float32) if center else None
    Xc = X - mean if center else X
    P = _projection(Xc, method, dims, seed)
    Y = Xc if P is None else Xc @ P.T
    return Y.astype(np.float32), P, mean


def _store(Y: np.ndarray, dtype: str):
    if dtype == "fp32":
        return Y, Y, None
    if dtype == "fp16":
        c = Y.astype(np.float16)
        return c.astype(np.float32), c, None
    if dtype == "int8":
        c, s = quantise_int8(Y)
        return dequantise_int8(c, s), c, s
    raise ValueError(f"unknown dtype {dtype!r} (use fp32, fp16, int8)")


def storage_size_bytes(n: int, D: int, dims: int, method: str, dtype: str, center: bool) -> int:
    """Exact size of a compressed embedding (see module docstring)."""
    size = n * dims * _BYTES[dtype]
    if dtype == "int8":
        size += 4 * n
    if method != "none":
        size += 4 * dims * D
    if center:
        size += 4 * D
    return int(size)


def compress(E: np.ndarray, dims: int, method: str = "pca", dtype: str = "fp32", seed: int = 0,
             center: bool = True) -> CompressedEmbedding:
    """Compress E (V x D). method: 'pca' | 'rp' (Gaussian random projection, seeded) | 'none' (dims ignored = D);
    dtype: 'fp32' | 'fp16' | 'int8'. Returns a CompressedEmbedding whose `view` is the dequantised float32 matrix."""
    E = np.asarray(E)
    if method == "none":
        dims = E.shape[1]
    Y, P, mean = _reduce(E, method, dims, seed, center)
    view, codes, scales = _store(Y, dtype)
    size = storage_size_bytes(len(E), E.shape[1], Y.shape[1], method, dtype, center)
    return CompressedEmbedding(method, int(Y.shape[1]), dtype, view, size, center, P, mean, codes, scales)


def retention_curve(E: np.ndarray, probe_fn: Callable[[np.ndarray], float], dims: Sequence[int] = (300, 150, 100, 64, 32, 16),
                    dtypes: Sequence[str] = ("fp32", "fp16", "int8"), methods: Sequence[str] = ("pca", "rp"), seed: int = 0,
                    center: bool = True, neighbour_k: int | None = 10, n_queries: int | None = 1000,
                    groups=None) -> pd.DataFrame:
    """Full-factorial compression frontier as a tidy DataFrame.

    probe_fn(view) -> float (higher = better, e.g. mean held-out rho) is run on the full-precision embedding once and on
    every dequantised config. Columns: method, dims, dtype, size_bytes, size_mb, score, retention (= score / full score,
    NaN if the full score <= 0), nn_recall (kNN overlap with the full-precision embedding, probe-independent; needs
    neighbour_k) and nn_recall_chance. The first row is the uncompressed baseline (method 'full'). Configs with dims > D
    are skipped; every remaining cell is reported."""
    E = np.asarray(E, dtype=np.float32)
    V, D = E.shape
    base = float(probe_fn(E))
    ref = queries = ref_idx = None
    if neighbour_k:
        ref = E - E.mean(0) if center else E
        queries = (np.arange(V) if n_queries is None or n_queries >= V
                   else np.sort(np.random.default_rng(seed).choice(V, n_queries, replace=False)))
        ref_idx = knn_indices(ref, neighbour_k, queries, groups)
    rows = [{"method": "full", "dims": D, "dtype": "fp32", "size_bytes": V * D * 4, "size_mb": V * D * 4 / 1e6,
             "score": base, "retention": 1.0 if base > 0 else float("nan"),
             "nn_recall": 1.0 if neighbour_k else float("nan"), "nn_recall_chance": float("nan")}]
    for method in methods:
        for d in dims:
            if d > D:
                continue
            Y, P, mean = _reduce(E, method, d, seed, center)
            for dt in dtypes:
                view, _, _ = _store(Y, dt)
                sc = float(probe_fn(view))
                row = {"method": method, "dims": int(d), "dtype": dt,
                       "size_bytes": storage_size_bytes(V, D, Y.shape[1], method, dt, center)}
                row["size_mb"] = row["size_bytes"] / 1e6
                row["score"] = sc
                row["retention"] = sc / base if base > 0 else float("nan")
                if neighbour_k:
                    r = neighbour_recall(ref, view, neighbour_k, groups, queries=queries, ref_idx=ref_idx)
                    row["nn_recall"], row["nn_recall_chance"] = r["recall"], r["chance"]
                else:
                    row["nn_recall"] = row["nn_recall_chance"] = float("nan")
                rows.append(row)
    return pd.DataFrame(rows)


def smallest_config(df: pd.DataFrame, threshold: float = 0.95, nn_threshold: float | None = None) -> dict | None:
    """Smallest (size_bytes, then dims) config whose probe retention >= threshold (and, if given, nn_recall >=
    nn_threshold). Returns the row as a dict, or None if no config qualifies."""
    ok = df["retention"] >= threshold
    if nn_threshold is not None:
        ok &= df["nn_recall"] >= nn_threshold
    cand = df[ok]
    if cand.empty:
        return None
    best = cand.sort_values(["size_bytes", "dims"], kind="stable").iloc[0]
    return {k: (v.item() if hasattr(v, "item") else v) for k, v in best.to_dict().items()}
