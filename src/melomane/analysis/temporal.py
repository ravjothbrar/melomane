"""E6 temporal stability (PLAN §6 S9): a Cooc model trained on early sessions vs one trained on late sessions.

* `procrustes_disparity`: how far the late space is from the early space after the best orthogonal alignment on the shared
  vocabulary (0 = identical up to rotation), next to a row-permutation null (chance).
* `mode_persistence`: does each mode (a set of cids) of the early model have a near-identical counterpart in the late model?
  Best Jaccard match of member sets on the shared vocabulary (plus a one-to-one Hungarian variant), a same-size random-set
  null, and - when embeddings are given - the cosine between the Procrustes-mapped early pole and the matched late pole."""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linear_sum_assignment

from ..geometry import unit
from .procrustes import procrustes_align


def procrustes_disparity(E_a: np.ndarray, E_b: np.ndarray, ids_a=None, ids_b=None, n_null: int = 10, seed: int = 0,
                         center: bool = True, normalise: bool = True) -> dict:
    """Procrustes residual disparity between two embeddings on their shared rows (row-aligned, or matched by ids).
    Returns disparity, null_disparity_mean (row-permuted B; chance level), disparity_over_null, row_cosine,
    axis_cosine_mean, n_shared."""
    r = procrustes_align(E_a, E_b, ids_a, ids_b, center=center, normalise=normalise, n_null=n_null, seed=seed)
    null = r.get("null_disparity_mean", float("nan"))
    return {"disparity": r["disparity"], "null_disparity_mean": null,
            "disparity_over_null": r["disparity"] / null if null and null > 0 else float("nan"),
            "row_cosine": r["row_cosine"], "axis_cosine_mean": r["axis_cosine_mean"], "n_shared": r["n_shared"]}


def _members(modes) -> list[np.ndarray]:
    out = []
    for m in modes:
        mem = m["members"] if isinstance(m, dict) else m
        out.append(np.unique(np.asarray(mem, dtype=np.int64)))
    return out


def _incidence(sets: list[np.ndarray], universe: np.ndarray) -> sp.csr_matrix:
    """modes x universe 0/1 matrix (members outside the universe dropped). `universe` sorted unique ids."""
    rows, cols = [], []
    for i, s in enumerate(sets):
        pos = np.searchsorted(universe, s)
        pos = pos[(pos < len(universe)) & (universe[np.minimum(pos, len(universe) - 1)] == s)] if len(universe) else pos[:0]
        rows.append(np.full(len(pos), i))
        cols.append(pos)
    r, c = (np.concatenate(rows), np.concatenate(cols)) if rows else (np.array([], int), np.array([], int))
    return sp.csr_matrix((np.ones(len(r)), (r, c)), shape=(len(sets), len(universe)))


def _jaccard_matrix(A: sp.csr_matrix, B: sp.csr_matrix) -> np.ndarray:
    inter = (A @ B.T).toarray()
    sa = np.asarray(A.sum(1)).ravel()[:, None]
    sb = np.asarray(B.sum(1)).ravel()[None, :]
    union = sa + sb - inter
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(union > 0, inter / union, 0.0)


def mode_persistence(modes_a, modes_b, E_a: np.ndarray | None = None, E_b: np.ndarray | None = None, ids_a=None,
                     ids_b=None, min_members: int = 3, persist_threshold: float = 0.5, n_null: int = 20,
                     seed: int = 0) -> dict:
    """How well modes of model A persist in model B.

    modes_*: lists of member-id arrays or `modes.json` dicts with a "members" list (ids = cids = row indices of E_*, or
    values of ids_* when the two models have different vocabularies). Member sets are restricted to the shared vocabulary
    (ids present in both models; without E/ids no restriction). Modes with < min_members shared members are dropped.
    Returns best_jaccard / best_match per A-mode (over B-modes), mean/median, share_persisting (>= persist_threshold),
    matched_jaccard (one-to-one Hungarian), null_best_jaccard_mean (random same-size sets), and, if E_a/E_b given,
    pole_cosine per A-mode (Procrustes-mapped A pole vs best-matched B pole) and pole_cosine_other_mean (vs unmatched)."""
    sets_a, sets_b = _members(modes_a), _members(modes_b)
    if E_a is not None and E_b is not None:
        ia = np.arange(len(E_a)) if ids_a is None else np.asarray(ids_a)
        ib = np.arange(len(E_b)) if ids_b is None else np.asarray(ids_b)
        universe = np.intersect1d(ia, ib)
    elif ids_a is not None and ids_b is not None:
        ia, ib = np.asarray(ids_a), np.asarray(ids_b)
        universe = np.intersect1d(ia, ib)
    else:
        ia = ib = None
        universe = np.unique(np.concatenate(sets_a + sets_b)) if (sets_a or sets_b) else np.array([], dtype=np.int64)
    A, B = _incidence(sets_a, universe), _incidence(sets_b, universe)
    keep_a = np.flatnonzero(np.asarray(A.sum(1)).ravel() >= min_members)
    keep_b = np.flatnonzero(np.asarray(B.sum(1)).ravel() >= min_members)
    if len(keep_a) == 0 or len(keep_b) == 0:
        raise ValueError("no modes with enough members on the shared vocabulary")
    A, B = A[keep_a], B[keep_b]
    J = _jaccard_matrix(A, B)
    best = J.max(1)
    match = J.argmax(1)
    r, c = linear_sum_assignment(-J)
    matched = np.zeros(len(keep_a))
    matched[r] = J[r, c]
    rng = np.random.default_rng(seed)
    sizes = np.asarray(A.sum(1)).ravel().astype(int)
    null = []
    for _ in range(n_null):
        rs = [np.sort(rng.choice(len(universe), s, replace=False)) for s in sizes]
        Rm = sp.csr_matrix((np.ones(int(sizes.sum())), (np.repeat(np.arange(len(sizes)), sizes), np.concatenate(rs))),
                           shape=(len(sizes), len(universe)))
        null.append(_jaccard_matrix(Rm, B).max(1).mean())
    out = {"best_jaccard": best, "best_match": keep_b[match], "mode_index_a": keep_a, "mean_best_jaccard": float(best.mean()),
           "median_best_jaccard": float(np.median(best)), "share_persisting": float((best >= persist_threshold).mean()),
           "matched_jaccard": matched, "mean_matched_jaccard": float(matched.mean()),
           "null_best_jaccard_mean": float(np.mean(null)) if null else float("nan"),
           "n_modes_a": int(len(keep_a)), "n_modes_b": int(len(keep_b)), "n_shared": int(len(universe))}
    if E_a is not None and E_b is not None:
        E_a, E_b = np.asarray(E_a, dtype=np.float64), np.asarray(E_b, dtype=np.float64)
        al = procrustes_align(E_a, E_b, ids_a, ids_b, center=True, normalise=True)
        mean_a = E_a[np.isin(ia, universe)].mean(0) if ids_a is not None else E_a[universe].mean(0)
        mean_b = E_b[np.isin(ib, universe)].mean(0) if ids_b is not None else E_b[universe].mean(0)
        pos_a = {int(v): i for i, v in enumerate(ia)}
        pos_b = {int(v): i for i, v in enumerate(ib)}

        def pole(E, mean, pos, members):
            rows = [pos[int(m)] for m in members if int(m) in pos]
            return unit(unit(E[rows] - mean).mean(0))

        pa = np.array([pole(E_a, mean_a, pos_a, sets_a[i]) for i in keep_a])
        pb = np.array([pole(E_b, mean_b, pos_b, sets_b[j]) for j in keep_b])
        C = unit(pa @ al["R"]) @ unit(pb).T
        out["pole_cosine"] = C[np.arange(len(keep_a)), match]
        off = C.copy()
        off[np.arange(len(keep_a)), match] = np.nan
        out["pole_cosine_mean"] = float(out["pole_cosine"].mean())
        out["pole_cosine_other_mean"] = float(np.nanmean(off)) if C.shape[1] > 1 else float("nan")
    return out
