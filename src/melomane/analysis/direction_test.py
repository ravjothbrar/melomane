"""S8 quantitative direction test (PLAN §6 S8, hypothesis H7): does SLERP steering toward a pole retrieve pole-tagged tracks?

For every (seed, pole) pair and every theta the query is q = slerp(seed, pole, theta); retrieval is cosine top-k over the
whole vocabulary, ARTIST-DISJOINT (the seed and every track of the seed's artist are excluded when `exclude_groups`
is given, which is the headline setting). Per (pair, theta) we record
  q_cos_seed  cos(q, seed)  (= cos(theta) exactly; 0.5 at 60 degrees),
  cos_pole    mean cosine of the retrieved top-k to the pole,
  cos_seed    mean cosine of the retrieved top-k to the seed (seed retention),
  precision   fraction of retrieved tracks carrying the pole's tag (needs `pole_members`),
  prevalence  overall share of tracks carrying the tag (chance level).
Retrieval is chunked matmuls, so 500 pairs x 7 thetas x V = 100k x D = 300 runs in seconds-to-minutes on CPU.
Choose seeds that do NOT carry the pole's tag (see `sample_seed_pole_pairs`) or precision is trivially high at theta = 0."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from ..geometry import unit
from ._retrieval import GroupIndex, batch_slerp, topk_cosine, unit32
from .stats import bootstrap_ci


def supervised_pole(E: np.ndarray, members, contrast=None) -> np.ndarray:
    """Unit pole = centroid(unit rows of members) - centroid(unit rows of contrast); contrast defaults to all other rows."""
    U = unit(np.asarray(E, dtype=np.float64))
    m = _as_mask(members, len(U))
    c = ~m if contrast is None else _as_mask(contrast, len(U))
    return unit(U[m].mean(0) - U[c].mean(0))


def _as_mask(obj, V: int) -> np.ndarray:
    a = np.asarray(obj)
    if a.dtype == bool:
        if a.shape != (V,):
            raise ValueError("boolean member mask must have one entry per track")
        return a
    m = np.zeros(V, bool)
    m[a.astype(np.int64)] = True
    return m


class _Members:
    """Uniform access to pole membership: 2-D bool (P x V), list of index/bool arrays, 1-D bool (one shared tag) or callable."""

    def __init__(self, members, V: int):
        self.V = V
        self._cache: dict[int, np.ndarray] = {}
        if members is None:
            self.get = lambda p: None
        elif callable(members):
            self.get = lambda p: self._cached(p, lambda: _as_mask(members(p), V))
        elif isinstance(members, np.ndarray) and members.dtype == bool and members.ndim == 1:
            self.get = lambda p: members
        elif isinstance(members, np.ndarray) and members.ndim == 2 and members.dtype == bool:
            self.get = lambda p: members[p]
        else:
            seq = list(members)
            self.get = lambda p: self._cached(p, lambda: _as_mask(seq[p], V))

    def _cached(self, p, fn):
        if p not in self._cache:
            self._cache[p] = fn()
        return self._cache[p]


def sample_seed_pole_pairs(pole_members, n_pairs: int, seed: int = 0, V: int | None = None,
                           min_members: int = 10) -> dict:
    """Sample (seed track, pole) pairs: poles uniform over those with >= min_members tagged tracks, seeds uniform over
    tracks NOT carrying that pole's tag (so precision at theta = 0 is not trivially high).
    pole_members: 2-D bool (P x V) or list of index arrays (then give V). Returns dict(seeds, pole_index)."""
    if isinstance(pole_members, np.ndarray) and pole_members.ndim == 2:
        V = pole_members.shape[1]
        P = pole_members.shape[0]
    else:
        if V is None:
            raise ValueError("V (number of tracks) required when pole_members is a list")
        P = len(pole_members)
    mem = _Members(pole_members, V)
    ok = [p for p in range(P) if mem.get(p).sum() >= min_members and (~mem.get(p)).sum() > 0]
    if not ok:
        raise ValueError("no pole has enough members")
    rng = np.random.default_rng(seed)
    poles = rng.choice(ok, n_pairs, replace=True)
    seeds = np.empty(n_pairs, dtype=np.int64)
    for i, p in enumerate(poles):
        seeds[i] = rng.choice(np.flatnonzero(~mem.get(int(p))))
    return {"seeds": seeds, "pole_index": poles.astype(np.int64)}


def _pair_arrays(seeds, poles, pole_index):
    seeds = np.asarray(seeds, dtype=np.int64)
    poles = np.asarray(poles, dtype=np.float64)
    if pole_index is None:
        if poles.shape[0] != len(seeds):
            raise ValueError("without pole_index, poles must have one row per seed")
        return seeds, poles, np.arange(len(seeds))
    pole_index = np.asarray(pole_index, dtype=np.int64)
    if len(pole_index) != len(seeds):
        raise ValueError("pole_index must align with seeds")
    return seeds, poles[pole_index], pole_index


def _gather_mean_cos(U: np.ndarray, idx: np.ndarray, vec: np.ndarray, step: int = 512) -> np.ndarray:
    """mean over the k retrieved items of cos(item, vec_i) for each row i (chunked to bound memory)."""
    out = np.empty(len(idx))
    for s in range(0, len(idx), step):
        out[s:s + step] = np.einsum("nkd,nd->nk", U[idx[s:s + step]], vec[s:s + step].astype(np.float32)).mean(1)
    return out


def theta_sweep_metrics(E: np.ndarray, seeds, poles, pole_members=None, thetas=range(0, 91, 15), k: int = 10,
                        exclude_groups=None, pole_index=None, chunk: int = 256) -> pd.DataFrame:
    """Tidy per-(pair, theta) metrics of the SLERP direction test. See module docstring.

    seeds: (n,) track indices.  poles: (n, D) one pole per seed, or (P, D) together with `pole_index` (n,).
    pole_members: which tracks carry each pole's tag: 2-D bool (P x V), 1-D bool (one tag), list of index/bool arrays or
    callable(pole_id) -> mask|indices (pole_id = pole_index value, or the pair number if no pole_index).
    exclude_groups: (V,) artist ids => artist-disjoint retrieval. Columns: pair_id, seed, pole, theta, q_cos_seed,
    cos_pole, cos_seed, precision, prevalence."""
    U = unit32(E)
    V = len(U)
    seeds, Pv, pids = _pair_arrays(seeds, poles, pole_index)
    n = len(seeds)
    gi = None if exclude_groups is None else GroupIndex(exclude_groups)
    mem = _Members(pole_members, V)
    Uv = U[seeds].astype(np.float64)
    Pu = unit(Pv)
    prevalence = np.full(n, np.nan)
    upids = np.unique(pids)
    if pole_members is not None:
        for p in upids:
            prevalence[pids == p] = mem.get(int(p)).mean()
    frames = []
    for th in thetas:
        Q = batch_slerp(Uv, Pv, float(th))
        idx, _ = topk_cosine(U, Q.astype(np.float32), k, exclude_items=seeds, group_index=gi, chunk=chunk)
        prec = np.full(n, np.nan)
        if pole_members is not None:
            for p in upids:
                sel = np.flatnonzero(pids == p)
                prec[sel] = mem.get(int(p))[idx[sel]].mean(1)
        frames.append(pd.DataFrame({
            "pair_id": np.arange(n), "seed": seeds, "pole": pids, "theta": float(th),
            "q_cos_seed": (Q * Uv).sum(1), "cos_pole": _gather_mean_cos(U, idx, Pu),
            "cos_seed": _gather_mean_cos(U, idx, Uv), "precision": prec, "prevalence": prevalence}))
    df = pd.concat(frames, ignore_index=True)
    return df.sort_values(["pair_id", "theta"], kind="stable").reset_index(drop=True)


def _spearman_rows(M: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Row-wise Spearman between each row of M and x (vectorised); NaN for constant rows."""
    R = rankdata(M, axis=1)
    rx = rankdata(x)
    R = R - R.mean(1, keepdims=True)
    rx = rx - rx.mean()
    den = np.sqrt((R ** 2).sum(1) * (rx ** 2).sum())
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (R @ rx) / den
    out[den == 0] = np.nan
    return out


def _curves(df: pd.DataFrame, metric: str, by: str, up_to: float | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    d = df if up_to is None else df[df["theta"] <= up_to + 1e-9]
    piv = d.pivot_table(index=by, columns="theta", values=metric, aggfunc="mean")
    return piv.index.to_numpy(), piv.columns.to_numpy(dtype=float), piv.to_numpy(dtype=float)


def monotonicity(df: pd.DataFrame, metric: str = "precision", by: str = "pair_id", up_to: float | None = None) -> dict:
    """Spearman correlation across theta of `metric`, per pair (by='pair_id') or per pole on the seed-averaged curve
    (by='pole'). Returns per-group table, mean/median rho, share_positive/negative and n_undefined (constant curves)."""
    ids, th, M = _curves(df, metric, by, up_to)
    rho = _spearman_rows(M, th)
    ok = np.isfinite(rho)
    tab = pd.DataFrame({by: ids, "rho": rho})
    return {"per_group": tab, "mean_rho": float(np.nanmean(rho)) if ok.any() else float("nan"),
            "median_rho": float(np.nanmedian(rho)) if ok.any() else float("nan"),
            "share_positive": float((rho[ok] > 0).mean()) if ok.any() else float("nan"),
            "share_negative": float((rho[ok] < 0).mean()) if ok.any() else float("nan"),
            "n_groups": int(len(ids)), "n_undefined": int((~ok).sum()), "metric": metric, "by": by}


def increasing_share(df: pd.DataFrame, up_to: float = 60.0, metric: str = "precision", by: str = "pole",
                     criterion: str = "trend", tol: float = 0.0, n_boot: int = 2000, seed: int = 0) -> dict:
    """Share of poles (H7) whose seed-averaged `metric` curve increases with theta up to `up_to` degrees.

    criterion: 'trend'   Spearman rho over theta <= up_to is > 0 and the curve ends above where it started (default);
               'strict'  every step is >= -tol and the end is above the start;
               'endpoint' end > start only.
    The CI is a bootstrap over poles. Returns share, ci_low, ci_high, n, per_group table."""
    ids, th, M = _curves(df, metric, by, up_to)
    if M.shape[1] < 2:
        raise ValueError("need at least two theta values up to `up_to`")
    start, end = M[:, 0], M[:, -1]
    if criterion == "trend":
        inc = (_spearman_rows(M, th) > 0) & (end > start)
    elif criterion == "strict":
        inc = (np.diff(M, axis=1) >= -tol).all(1) & (end > start)
    elif criterion == "endpoint":
        inc = end > start
    else:
        raise ValueError(criterion)
    ok = np.isfinite(start) & np.isfinite(end)
    inc = inc & ok
    ci = bootstrap_ci(inc[ok].astype(float), n_boot=n_boot, seed=seed) if ok.any() else \
        {"estimate": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    tab = pd.DataFrame({by: ids, "start": start, "end": end, "increasing": inc})
    return {"share": ci["estimate"], "ci_low": ci["ci_low"], "ci_high": ci["ci_high"], "n": int(ok.sum()),
            "up_to": up_to, "criterion": criterion, "per_group": tab}


def multi_pole_composition(E: np.ndarray, seeds, poles: np.ndarray, combos, pole_members, thetas=(30, 60, 90),
                           k: int = 10, exclude_groups=None, weights=None, chunk: int = 256) -> pd.DataFrame:
    """Multi-constraint steering test: q = slerp(seed, compose_poles([p_a, p_b, ...]), theta) should retrieve tracks that
    carry ALL the poles' tags more often than steering with any single pole does.

    combos: (n, m) pole ids (rows of `poles`), one combination per seed. pole_members as in `theta_sweep_metrics`
    (indexed by pole id). Columns: pair_id, seed, combo ('a+b'), theta, q_cos_seed, joint_precision (all tags),
    mean_tag_precision, best_single_joint (max over single-pole queries of the joint precision), gain_over_single,
    prevalence_joint (chance)."""
    U = unit32(E)
    V = len(U)
    seeds = np.asarray(seeds, dtype=np.int64)
    combos = np.atleast_2d(np.asarray(combos, dtype=np.int64))
    poles = unit(np.asarray(poles, dtype=np.float64))
    n, m = combos.shape
    if len(seeds) != n:
        raise ValueError("combos must have one row per seed")
    w = np.ones(m) if weights is None else np.asarray(weights, dtype=float)
    gi = None if exclude_groups is None else GroupIndex(exclude_groups)
    mem = _Members(pole_members, V)
    Uv = U[seeds].astype(np.float64)
    comp = unit(np.einsum("nm,nmd->nd", np.broadcast_to(w, (n, m)), poles[combos]))
    joint_mask = [np.logical_and.reduce([mem.get(int(p)) for p in row]) for row in combos]
    prevalence = np.array([jm.mean() for jm in joint_mask])
    label = ["+".join(str(int(p)) for p in row) for row in combos]

    def metrics(direction: np.ndarray, th: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        Q = batch_slerp(Uv, direction, th)
        idx, _ = topk_cosine(U, Q.astype(np.float32), k, exclude_items=seeds, group_index=gi, chunk=chunk)
        joint = np.array([joint_mask[i][idx[i]].mean() for i in range(n)])
        each = np.array([np.mean([mem.get(int(p))[idx[i]].mean() for p in combos[i]]) for i in range(n)])
        return (Q * Uv).sum(1), joint, each, idx

    frames = []
    for th in thetas:
        qcs, joint, each, _ = metrics(comp, float(th))
        best = np.zeros(n)
        for j in range(m):
            _, jj, _, _ = metrics(poles[combos[:, j]], float(th))
            best = np.maximum(best, jj)
        frames.append(pd.DataFrame({"pair_id": np.arange(n), "seed": seeds, "combo": label, "theta": float(th),
                                    "q_cos_seed": qcs, "joint_precision": joint, "mean_tag_precision": each,
                                    "best_single_joint": best, "gain_over_single": joint - best,
                                    "prevalence_joint": prevalence}))
    return pd.concat(frames, ignore_index=True).sort_values(["pair_id", "theta"], kind="stable").reset_index(drop=True)
