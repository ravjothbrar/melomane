"""E5 external-embedding agreement (PLAN §6 S9): artist-disjoint kNN overlap, neighbour recall and CKA between two
embeddings of the SAME tracks (rows aligned; dimensions may differ). Chance levels are reported next to every overlap."""
from __future__ import annotations

import numpy as np

from ._retrieval import GroupIndex, topk_cosine, unit32


def _queries(V: int, n_queries: int | None, seed: int) -> np.ndarray:
    if n_queries is None or n_queries >= V:
        return np.arange(V)
    return np.sort(np.random.default_rng(seed).choice(V, n_queries, replace=False))


def knn_indices(E: np.ndarray, k: int, queries: np.ndarray | None = None, groups=None, chunk: int = 512) -> np.ndarray:
    """(n_queries x k) cosine neighbours of each query row among all rows; self (and, with `groups`, the query's whole
    artist group) excluded."""
    U = unit32(E)
    q = np.arange(len(U)) if queries is None else np.asarray(queries, dtype=np.int64)
    gi = None if groups is None else GroupIndex(groups)
    return topk_cosine(U, U[q], k, exclude_items=q, group_index=gi, chunk=chunk)[0]


def _overlap(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """|A_i n B_i| per row for index arrays with distinct entries per row."""
    both = np.concatenate([a, b], axis=1)
    s = np.sort(both, axis=1)
    return (s[:, 1:] == s[:, :-1]).sum(1)


def _pool_sizes(V: int, q: np.ndarray, groups) -> np.ndarray:
    if groups is None:
        return np.full(len(q), V - 1.0)
    return V - GroupIndex(groups).sizes_of(q).astype(float)


def knn_jaccard(E1: np.ndarray, E2: np.ndarray, k: int = 10, groups=None, n_queries: int | None = None, seed: int = 0,
                chunk: int = 512) -> dict:
    """Mean Jaccard overlap of the k-nearest-neighbour SETS of the same track in two embeddings (artist-disjoint when
    `groups` = artist ids: neither set may contain the query's artist). `chance` = expected Jaccard of two independent
    uniformly random k-sets from the candidate pool. Returns jaccard, jaccard_sd, overlap (mean |A n B| / k), chance."""
    E1, E2 = np.asarray(E1), np.asarray(E2)
    if len(E1) != len(E2):
        raise ValueError("embeddings must have the same number of rows")
    q = _queries(len(E1), n_queries, seed)
    a = knn_indices(E1, k, q, groups, chunk)
    b = knn_indices(E2, k, q, groups, chunk)
    inter = _overlap(a, b).astype(float)
    jac = inter / (2 * k - inter)
    M = _pool_sizes(len(E1), q, groups)
    e_int = k * k / M
    chance = float(np.mean(e_int / (2 * k - e_int)))
    return {"jaccard": float(jac.mean()), "jaccard_sd": float(jac.std()), "overlap": float(inter.mean() / k),
            "chance": chance, "k": int(k), "n_queries": int(len(q)), "per_item": jac}


def neighbour_recall(E_ref: np.ndarray, E_test: np.ndarray, k: int = 10, groups=None, n_queries: int | None = None,
                     seed: int = 0, ref_idx: np.ndarray | None = None, queries: np.ndarray | None = None,
                     chunk: int = 512) -> dict:
    """recall@k = mean |kNN_ref n kNN_test| / k for the same query rows (E_test is e.g. a compressed / re-trained copy).
    `ref_idx` (with the `queries` it was computed for) lets callers reuse the reference neighbours across many tests."""
    E_ref, E_test = np.asarray(E_ref), np.asarray(E_test)
    if len(E_ref) != len(E_test):
        raise ValueError("embeddings must have the same number of rows")
    q = _queries(len(E_ref), n_queries, seed) if queries is None else np.asarray(queries, dtype=np.int64)
    a = knn_indices(E_ref, k, q, groups, chunk) if ref_idx is None else np.asarray(ref_idx)
    b = knn_indices(E_test, k, q, groups, chunk)
    rec = _overlap(a, b) / k
    M = _pool_sizes(len(E_ref), q, groups)
    return {"recall": float(rec.mean()), "recall_sd": float(rec.std()), "chance": float(np.mean(k / M)), "k": int(k),
            "n_queries": int(len(q))}


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear CKA (Kornblith et al. 2019) between row-aligned representations, computed in feature space
    ||Y_c^T X_c||_F^2 / (||X_c^T X_c||_F ||Y_c^T Y_c||_F). 1 = identical up to orthogonal transform and isotropic scale."""
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if len(X) != len(Y):
        raise ValueError("X and Y must have the same number of rows")
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    num = np.linalg.norm(Y.T @ X) ** 2
    den = np.linalg.norm(X.T @ X) * np.linalg.norm(Y.T @ Y)
    return float(num / den) if den > 0 else float("nan")


def _rbf_gram(X: np.ndarray, sigma_factor: float) -> np.ndarray:
    sq = (X ** 2).sum(1)
    D2 = np.maximum(sq[:, None] + sq[None, :] - 2 * X @ X.T, 0.0)
    med = np.median(D2[np.triu_indices_from(D2, k=1)])
    return np.exp(-D2 / (2 * sigma_factor ** 2 * max(med, 1e-12)))


def rbf_cka(X: np.ndarray, Y: np.ndarray, n_sub: int = 2000, sigma_factor: float = 1.0, seed: int = 0,
            n_repeats: int = 1) -> float:
    """RBF-kernel CKA on random row subsamples of at most `n_sub` items (O(n_sub^2) memory), bandwidth
    sigma = sigma_factor x median pairwise distance (per representation); mean over `n_repeats` subsamples."""
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if len(X) != len(Y):
        raise ValueError("X and Y must have the same number of rows")
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_repeats):
        ix = np.arange(len(X)) if len(X) <= n_sub else np.sort(rng.choice(len(X), n_sub, replace=False))
        K, L = _rbf_gram(X[ix], sigma_factor), _rbf_gram(Y[ix], sigma_factor)
        n = len(ix)
        H = np.eye(n) - 1.0 / n
        Kc, Lc = H @ K @ H, H @ L @ H
        den = np.linalg.norm(Kc) * np.linalg.norm(Lc)
        vals.append(float((Kc * Lc).sum() / den) if den > 0 else float("nan"))
    return float(np.mean(vals))


def agreement_report(E1: np.ndarray, E2: np.ndarray, k: int = 10, groups=None, n_queries: int | None = 2000,
                     n_sub: int = 2000, seed: int = 0) -> dict:
    """One-call E5 summary: artist-disjoint kNN Jaccard + chance, neighbour recall, linear and RBF CKA."""
    j = knn_jaccard(E1, E2, k, groups, n_queries, seed)
    r = neighbour_recall(E1, E2, k, groups, n_queries, seed)
    return {"knn_jaccard": j["jaccard"], "knn_jaccard_chance": j["chance"], "neighbour_recall": r["recall"],
            "neighbour_recall_chance": r["chance"], "linear_cka": linear_cka(E1, E2),
            "rbf_cka": rbf_cka(E1, E2, n_sub=n_sub, seed=seed), "k": k, "n_queries": j["n_queries"]}
