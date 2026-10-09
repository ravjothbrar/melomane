"""Private shared helpers: chunked cosine retrieval with self/group exclusion, batched SLERP, cluster resampling.

All functions are pure numpy, deterministic given their inputs, and Windows-safe (no multiprocessing)."""
from __future__ import annotations

import numpy as np

from ..geometry import unit


def unit32(E: np.ndarray) -> np.ndarray:
    """Row-normalise to float32 (retrieval precision is ample; halves memory of the V x chunk similarity block)."""
    return unit(np.asarray(E, dtype=np.float32))


class GroupIndex:
    """Fast 'all members of the group of item i' lookup (used for artist-disjoint retrieval)."""

    def __init__(self, groups: np.ndarray):
        g = np.asarray(groups)
        if g.ndim != 1:
            raise ValueError("groups must be 1-D")
        _, self.inv = np.unique(g, return_inverse=True)
        self.order = np.argsort(self.inv, kind="stable")
        self.starts = np.concatenate([[0], np.cumsum(np.bincount(self.inv))])

    def sizes_of(self, items: np.ndarray) -> np.ndarray:
        gs = self.inv[np.asarray(items)]
        return self.starts[gs + 1] - self.starts[gs]

    def members_of_items(self, items: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(rows, cols): for each position r in `items`, every item sharing its group -> pairs (r, member)."""
        gs = self.inv[np.asarray(items)]
        cnt = self.starts[gs + 1] - self.starts[gs]
        rows = np.repeat(np.arange(len(gs)), cnt)
        offs = np.arange(int(cnt.sum())) - np.repeat(np.cumsum(cnt) - cnt, cnt)
        cols = self.order[np.repeat(self.starts[gs], cnt) + offs]
        return rows, cols


def topk_cosine(U: np.ndarray, Q: np.ndarray, k: int, exclude_items: np.ndarray | None = None,
                group_index: GroupIndex | None = None, chunk: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """Top-k rows of unit matrix U (V x D) for each unit query in Q (n x D), chunked over queries.

    exclude_items[i] (or -1 for none) is the item that query i was built from: it is excluded, and if `group_index`
    is given so is every item in its group (artist-disjoint retrieval). Returns (idx, sims) sorted by similarity desc."""
    n, V = len(Q), len(U)
    if not 1 <= k < V:
        raise ValueError("need 1 <= k < number of items")
    idx = np.empty((n, k), dtype=np.int64)
    sims = np.empty((n, k), dtype=np.float32)
    ex_all = None if exclude_items is None else np.asarray(exclude_items, dtype=np.int64)
    for s in range(0, n, chunk):
        S = Q[s:s + chunk] @ U.T
        if ex_all is not None:
            ex = ex_all[s:s + chunk]
            valid = np.flatnonzero(ex >= 0)
            if len(valid):
                if group_index is not None:
                    rows, cols = group_index.members_of_items(ex[valid])
                    S[valid[rows], cols] = -np.inf
                else:
                    S[valid, ex[valid]] = -np.inf
        part = np.argpartition(-S, k - 1, axis=1)[:, :k]
        ps = np.take_along_axis(S, part, axis=1)
        o = np.argsort(-ps, axis=1, kind="stable")
        idx[s:s + chunk] = np.take_along_axis(part, o, axis=1)
        sims[s:s + chunk] = np.take_along_axis(ps, o, axis=1)
    return idx, sims


def batch_slerp(seeds: np.ndarray, directions: np.ndarray, theta_deg: float) -> np.ndarray:
    """Row-wise `melomane.slerp.slerp`: q = cos(th) v + sin(th) unit(d - (d.v) v); degenerate d (parallel to v) returns v.
    float64 in, float64 unit rows out."""
    v = unit(np.asarray(seeds, dtype=np.float64))
    d = unit(np.asarray(directions, dtype=np.float64))
    dp = d - (d * v).sum(1, keepdims=True) * v
    nrm = np.linalg.norm(dp, axis=1, keepdims=True)
    th = np.deg2rad(theta_deg)
    q = np.cos(th) * v + np.sin(th) * dp / np.maximum(nrm, 1e-12)
    q = np.where(nrm > 1e-9, q, v)
    return unit(q)


def cluster_index(clusters: np.ndarray | None, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(inv, order, starts) for cluster-resampling; clusters=None => every item is its own cluster."""
    if clusters is None:
        inv = np.arange(n)
    else:
        clusters = np.asarray(clusters)
        if len(clusters) != n:
            raise ValueError("clusters must have the same length as the data")
        _, inv = np.unique(clusters, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    starts = np.concatenate([[0], np.cumsum(np.bincount(inv))])
    return inv, order, starts


def resample_items(order: np.ndarray, starts: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Item indices of one cluster bootstrap draw (clusters drawn with replacement, all their items kept)."""
    G = len(starts) - 1
    ch = rng.integers(0, G, G)
    cnt = starts[ch + 1] - starts[ch]
    offs = np.arange(int(cnt.sum())) - np.repeat(np.cumsum(cnt) - cnt, cnt)
    return order[np.repeat(starts[ch], cnt) + offs]
