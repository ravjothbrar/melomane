"""Behavioural co-occurrence graph: positive-NPMI edges from a (users x tracks) incidence matrix.

Mirrors Epicure paper 2: items appearing in fewer than `min_count` "documents" are dropped before NPMI;
only positive-NPMI pairs are kept.  Two additions needed at music scale (not in the paper):
  * `min_pair_count` - suppress pairs seen together in only 1-2 users (NPMI is ~1 for such pairs at low counts);
  * `topk` - cap edges per node (chunked sparse products keep memory bounded for 100k+ tracks).
"""
from __future__ import annotations
import numpy as np
import scipy.sparse as sp


def npmi_edges(X: sp.spmatrix, min_count: int = 20, min_pair_count: int = 3,
               topk: int | None = 200, chunk: int = 2048, positive_only: bool = True):
    """Return (rows, cols, npmi) of a *symmetric* edge list.

    X: binary (n_docs x n_items) incidence (users x tracks, or playlists x tracks).
    """
    X = sp.csr_matrix(X, dtype=np.float32)
    X.data[:] = 1.0
    n_docs, n_items = X.shape
    counts = np.asarray(X.sum(axis=0)).ravel()
    keep = counts >= min_count
    XT = X.T.tocsr()
    logn = np.log(n_docs)
    all_r, all_c, all_v = [], [], []
    for start in range(0, n_items, chunk):
        stop = min(start + chunk, n_items)
        block_rows = np.arange(start, stop)
        block_rows = block_rows[keep[block_rows]]
        if block_rows.size == 0:
            continue
        co = (XT[block_rows] @ X).tocoo()            # co-counts of block items with all items
        r = block_rows[co.row]; c = co.col; n_ij = co.data
        m = (r != c) & keep[c] & (n_ij >= min_pair_count)
        r, c, n_ij = r[m], c[m], n_ij[m]
        # NPMI = log(p_ij / (p_i p_j)) / -log(p_ij)
        log_pij = np.log(n_ij) - logn
        pmi = log_pij - (np.log(counts[r]) - logn) - (np.log(counts[c]) - logn)
        npmi = pmi / (-log_pij)
        if positive_only:
            m = npmi > 0
            r, c, npmi = r[m], c[m], npmi[m]
        if topk is not None and r.size:
            order = np.lexsort((-npmi, r))            # by row, then descending npmi
            r, c, npmi = r[order], c[order], npmi[order]
            first = np.r_[0, np.flatnonzero(np.diff(r)) + 1]
            rank = np.arange(r.size) - np.repeat(first, np.diff(np.r_[first, r.size]))
            m = rank < topk
            r, c, npmi = r[m], c[m], npmi[m]
        all_r.append(r); all_c.append(c); all_v.append(npmi)
    if not all_r:
        return np.array([], int), np.array([], int), np.array([], float)
    r = np.concatenate(all_r); c = np.concatenate(all_c); v = np.concatenate(all_v)
    # symmetrise: union of directed edges, keeping the max weight
    a = np.minimum(r, c); b = np.maximum(r, c)
    key = a.astype(np.int64) * n_items + b
    order = np.argsort(key, kind="stable")
    key, v = key[order], v[order]
    uniq, idx = np.unique(key, return_index=True)
    vmax = np.maximum.reduceat(v, idx)
    a = (uniq // n_items).astype(np.int64); b = (uniq % n_items).astype(np.int64)
    return (np.concatenate([a, b]), np.concatenate([b, a]), np.concatenate([vmax, vmax]))


def edges_to_csr(rows, cols, vals, n_items: int) -> sp.csr_matrix:
    return sp.csr_matrix((vals, (rows, cols)), shape=(n_items, n_items))
