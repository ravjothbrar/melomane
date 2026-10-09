from __future__ import annotations
import numpy as np


def unit(x, axis=-1, eps=1e-9):
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(n, eps)


def participation_ratio(E: np.ndarray, center: bool = True) -> float:
    """PR = (sum lambda)^2 / sum lambda^2 over eigenvalues of the embedding covariance (max = dim)."""
    X = E - E.mean(0, keepdims=True) if center else E
    s = np.linalg.svd(X, compute_uv=False) ** 2
    return float(s.sum() ** 2 / (s ** 2).sum())


def mean_pairwise_cosine(E: np.ndarray) -> float:
    """Exact mean cosine over all i != j pairs via ||sum u_i||^2 (O(N d))."""
    U = unit(E); n = len(U)
    return float((np.linalg.norm(U.sum(0)) ** 2 - n) / (n * (n - 1)))


def consolidate_embeddings(E: np.ndarray, canonical_ids: np.ndarray, normalise_members: bool = False):
    """Paper 1 stage 6: e_consolidated = (1/|S|) sum_{i in S} e_i  (centre of mass of all raw entries mapped to one canonical item).
    Returns (E_canonical, unique_canonical_ids)."""
    canonical_ids = np.asarray(canonical_ids); uid, inv = np.unique(canonical_ids, return_inverse=True)
    X = unit(E) if normalise_members else E
    out = np.zeros((len(uid), E.shape[1]), dtype=np.float64); np.add.at(out, inv, X)
    out /= np.bincount(inv, minlength=len(uid))[:, None]
    return out.astype(E.dtype), uid
