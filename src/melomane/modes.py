"""Modes, poles and driver attribution helpers for S6/S8 (pure functions; tested on planted structure)."""
from __future__ import annotations
import numpy as np
import pandas as pd
import scipy.sparse as sp

from .geometry import unit


def zscore_cols(S: np.ndarray) -> np.ndarray:
    sd = S.std(axis=0); sd[sd < 1e-12] = 1.0
    return (S - S.mean(axis=0)) / sd


def project_sources(E_res: np.ndarray, directions: np.ndarray) -> np.ndarray:
    """ICA sources for arbitrary rows: for FastICA (whiten='unit-variance') components_ @ x is exactly the source up to scale, and
    `directions` are the unit-normalised components_, so a z-scored projection reproduces the sources (sign already oriented)."""
    return zscore_cols((E_res - E_res.mean(axis=0)) @ directions.T)


def _mean_pair_cos_sparse(A: sp.csr_matrix, ix: np.ndarray) -> float:
    """Mean pairwise cosine among rows `ix` of sparse A via ||sum u||^2 (rows without entries ignored)."""
    sub = A[ix]
    nrm = np.sqrt(np.asarray(sub.multiply(sub).sum(axis=1)).ravel())
    ok = nrm > 0
    if ok.sum() < 3:
        return float("nan")
    U = sp.diags(1.0 / nrm[ok]) @ sub[np.flatnonzero(ok)]
    s = np.asarray(U.sum(axis=0)).ravel()
    n = U.shape[0]
    return float((s @ s - n) / (n * (n - 1)))


def _density(tt: sp.csr_matrix, ix: np.ndarray) -> float:
    sub = tt[ix][:, ix]
    n = len(ix)
    return float((sub > 0).sum() / (n * (n - 1)))


def mode_drivers(members: list[np.ndarray], comp_edges: sp.csr_matrix | None, comp_kind_cols: dict[str, np.ndarray] | None,
                 tt: sp.csr_matrix, n_random: int = 30, seed: int = 0, z_threshold: float = 2.0) -> list[dict]:
    """For each mode: cohesion of its members under (a) acoustic compound membership, (b) lyric compound membership, (c) listening
    co-occurrence, each as a z-score against random same-size sets.  driver = largest z if > z_threshold else 'mixed'."""
    rng = np.random.default_rng(seed)
    V = tt.shape[0]
    out = []
    for mem in members:
        mem = np.asarray(mem)
        rec = {}
        sources = {"behaviour": lambda ix: _density(tt, ix)}
        if comp_edges is not None and comp_kind_cols:
            for kind, cols in comp_kind_cols.items():
                if len(cols):
                    A = comp_edges[:, cols].tocsr()
                    sources[kind] = (lambda ix, A=A: _mean_pair_cos_sparse(A, ix))
        for k, f in sources.items():
            obs = f(mem)
            rnd = np.array([f(rng.choice(V, len(mem), replace=False)) for _ in range(n_random)], dtype=float)
            rnd = rnd[np.isfinite(rnd)]
            sd = rnd.std() if len(rnd) > 2 else np.nan
            rec[k] = {"obs": obs, "random_mean": float(rnd.mean()) if len(rnd) else np.nan,
                      "z": float((obs - rnd.mean()) / (sd + 1e-9)) if np.isfinite(obs) and len(rnd) > 2 else np.nan}
        zs = {k: v["z"] for k, v in rec.items() if np.isfinite(v["z"])}
        best = max(zs, key=zs.get) if zs else None
        driver = best if best is not None and zs[best] > z_threshold else "mixed"
        out.append({"driver": driver, "z": {k: v["z"] for k, v in rec.items()}})
    return out


def decade(year: float) -> str | None:
    return None if year != year else f"{int(year) // 10 * 10}s"


def supervised_poles(E: np.ndarray, vocab: pd.DataFrame, labels: pd.DataFrame | None, mask: np.ndarray, min_members: int = 30,
                     reliable_dims: list[str] | None = None) -> tuple[list[dict], np.ndarray]:
    """Category / region / era poles = unit-mean of member unit vectors; dimension poles = unit(centroid(high) - centroid(low)) for
    reliable LLM dimensions.  Only rows in `mask` participate.  Returns (meta list, (n_poles x D) unit matrix)."""
    U = unit(E)
    poles, meta = [], []

    def add(name, kind, pos, neg=None, direction=None):
        if pos.sum() < min_members:
            return
        v = direction if direction is not None else U[pos].mean(0)
        if np.linalg.norm(v) < 1e-9:
            return
        poles.append(unit(v)); meta.append({"name": name, "kind": kind, "n_pos": int(pos.sum()), "n_neg": int(neg.sum()) if neg is not None else 0})

    cat = vocab["category"].astype(str).to_numpy()
    for c in sorted(set(cat)):
        if c and c not in ("Other", "nan"):
            add(f"category:{c}", "category", (cat == c) & mask)
    if "region_tags" in vocab:
        reg = vocab["region_tags"].astype(str).to_numpy()
        for r in sorted({x for s in reg for x in s.split("|") if x}):
            m = np.array([r in s.split("|") for s in reg]) & mask
            add(f"region:{r}", "region", m)
    if "year" in vocab and vocab["year"].notna().sum() > 100:
        dec = np.array([decade(y) or "" for y in vocab["year"].to_numpy(float)])
        for d in sorted(set(dec) - {""}):
            add(f"era:{d}", "era", (dec == d) & mask)
    elif labels is not None and labels["era"].notna().sum() > 100:
        dec = np.array([f"{int(y)}s" if y == y else "" for y in labels["era"].to_numpy(float)])
        for d in sorted(set(dec) - {""}):
            add(f"era:{d}", "era", (dec == d) & mask)
    if labels is not None and labels["labelled"].any():
        from .annotation import DIMS
        for dname in (reliable_dims if reliable_dims is not None else DIMS):
            if dname not in labels:
                continue
            y = labels[dname].to_numpy(float)
            hi = (y >= 3) & mask & np.isfinite(y); lo = (y <= 1) & mask & np.isfinite(y)
            if hi.sum() >= min_members and lo.sum() >= min_members:
                add(f"dim:{dname}+", "dimension", hi, lo, direction=U[hi].mean(0) - U[lo].mean(0))
    return meta, (np.vstack(poles).astype(np.float32) if poles else np.zeros((0, E.shape[1]), np.float32))


def pole_member_mask(name: str, vocab: pd.DataFrame, labels: pd.DataFrame | None) -> np.ndarray | None:
    """Boolean mask of tracks that 'carry the tag' of a supervised pole (used for SLERP precision@k)."""
    kind, _, val = name.partition(":")
    V = len(vocab)
    if kind == "category":
        return (vocab["category"].astype(str) == val).to_numpy()
    if kind == "region":
        return np.array([val in str(s).split("|") for s in vocab["region_tags"]])
    if kind == "era":
        if "year" in vocab and vocab["year"].notna().sum() > 100:
            return np.array([(decade(y) == val) for y in vocab["year"].to_numpy(float)])
        if labels is not None:
            return np.array([(f"{int(y)}s" == val) if y == y else False for y in labels["era"].to_numpy(float)])
    if kind == "dim" and labels is not None:
        d = val.rstrip("+")
        return (labels[d].to_numpy(float) >= 3)
    return None


def choose_modes_members_for_labelling(E: np.ndarray, members: np.ndarray, pole: np.ndarray, n: int = 25) -> np.ndarray:
    """The `n` members closest to the mode pole (most prototypical) - what the LLM sees when naming the mode."""
    U = unit(E[members])
    order = np.argsort(-(U @ unit(pole)))
    return members[order[:n]]
