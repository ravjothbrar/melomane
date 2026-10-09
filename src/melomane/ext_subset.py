"""Small-scale re-training on a track subset - the machinery behind the sensitivity grid (E8) and the paper-vs-standard check (H8).

Everything here is deterministic given (seed, arguments).  Graph pieces are cached per (parameter) key so a Cartesian grid
only pays for what actually changes: Composition (tt = 0) does not depend on the NPMI top-k at all, Cooc does not depend on the
compound parameters, so a 3 x 3 x 3 grid needs 3 + 9 trainings, not 27 + 27.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import scipy.sparse as sp

from .compounds import fit_compound_graph
from .cooc import edges_to_csr, npmi_edges
from .runtime import get_logger
from .training import GraphInputs, ModelSpec, model_specs, train_one

log = get_logger("melomane.ext_subset")


@dataclass
class SubsetProblem:
    """A sub-sample of the canonical vocabulary with everything needed to rebuild graphs on it."""
    sub: np.ndarray                                   # cids (rows of the full vocabulary) in the subset, sorted
    sessions: sp.csr_matrix                           # (n_sessions x n_full_vocab) binary, cid space
    features: dict[str, np.ndarray]                   # family -> (n_full_vocab x d) with NaN rows where missing
    family_kind: dict[str, str]
    cooc_cfg: dict
    comp_cfg: dict
    n_full: int
    seed: int = 0
    _tt: dict = field(default_factory=dict, repr=False)
    _comp: dict = field(default_factory=dict, repr=False)

    # ---- graph pieces ----------------------------------------------------------------------------------------------
    def tt(self, topk: int) -> sp.csr_matrix:
        if topk not in self._tt:
            S = self.sessions[:, self.sub].tocsr()
            S = S[np.flatnonzero(np.diff(S.indptr) >= 2)]
            frac = len(self.sub) / max(self.n_full, 1)
            min_count = max(3, int(round(self.cooc_cfg["min_count"] * min(1.0, frac * 2))))
            r, c, v = npmi_edges(S, min_count, 2, topk, self.cooc_cfg["chunk"], True)
            self._tt[topk] = edges_to_csr(r, c, v, len(self.sub))
        return self._tt[topk]

    def comp(self, k_scale: float, top_m: int):
        key = (float(k_scale), int(top_m))
        if key not in self._comp:
            frac = len(self.sub) / max(self.n_full, 1)
            feats = {n: X[self.sub] for n, X in self.features.items()}
            k = {}
            for n in feats:
                base = self.comp_cfg["k_lyric"] if self.family_kind[n] == "lyric" else self.comp_cfg["k_acoustic"]
                k[n] = max(4, int(round(base * k_scale * frac)))
            cg = fit_compound_graph(feats, k=k, top_m=top_m, seed=self.seed, pca_dim=self.comp_cfg.get("pca_dim"))
            self._comp[key] = cg
        return self._comp[key]

    def inputs(self, topk: int, k_scale: float, top_m: int, need_comp: bool = True) -> GraphInputs:
        gi = GraphInputs(len(self.sub), self.tt(topk))
        if need_comp and self.features:
            cg = self.comp(k_scale, top_m)
            gi.comp_edges = sp.csr_matrix(cg.edges)
            gi.comp_family = np.asarray(cg.comp_family, dtype=np.int64)
            gi.family_names = list(cg.family_names)
            gi.family_kind = dict(self.family_kind)
        return gi


def train_models(problem: SubsetProblem, models: list[str], train_cfg: dict, out_dir: Path, tag: str, key: str, topk: int, k_scale: float,
                 top_m: int, ii_repeat: int | None = None, device: str | None = None, seed: int = 0) -> dict[str, np.ndarray]:
    """Train each named model spec once on the subset; resumable through `train_one`'s own seed files under out_dir/tag/<model>."""
    specs = model_specs(ii_repeat if ii_repeat is not None else train_cfg["ii_repeat"])
    out: dict[str, np.ndarray] = {}
    for name in models:
        spec: ModelSpec = specs[name]
        need_comp = spec.compound != "none"
        if need_comp and not problem.features:
            continue
        gi = problem.inputs(topk, k_scale, top_m, need_comp)
        E, _ = train_one(spec, gi, train_cfg, seed, Path(out_dir) / tag / name, f"{key}/{tag}", device=device)
        out[name] = E
    return out


def mean_score(E: np.ndarray, T: dict, groups: np.ndarray, item_fn: Callable, n_splits: int, n_repeats: int, seed: int) -> float:
    """Mean probe score (rho) of an embedding over a target family; NaN if nothing could be evaluated."""
    arr = item_fn(E, T, groups, n_splits, n_repeats, seed)[0]
    return float(np.nanmean(arr)) if np.isfinite(arr).any() else float("nan")
