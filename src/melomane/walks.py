"""Vectorised typed random walks (metapath2vec-style) over the track-track and track-compound graphs.

Node id space: tracks 0..N-1, compounds N..N+M-1.  Templates follow Epicure paper 2:
  TT            pure track-track walk                       (Epicure I-I template)
  TCT[f]        t - c_f - t - c_f - t ...                   (Epicure H-C[x]-H, "shared compound of family x")
  TCT_cross     alternating families f,g per walk           (Epicure C[x]-H-N-H-C[y] cross-type bridge)
Cooc  = TT only.  Chem/Composition = all compound templates, ii_repeat=0.  Core = compound templates + ii_repeat x TT.
"""
from __future__ import annotations
import numpy as np
import scipy.sparse as sp


class CSRSampler:
    """Weighted neighbour sampling for many nodes at once."""
    def __init__(self, A: sp.csr_matrix):
        A = sp.csr_matrix(A)
        self.indptr, self.indices = A.indptr, A.indices
        data = A.data.astype(np.float64)
        self.cum = np.cumsum(data)
        before = np.r_[0.0, self.cum][self.indptr[:-1]]
        after = np.r_[0.0, self.cum][self.indptr[1:]]
        self.before, self.total = before, after - before
        self.deg = np.diff(self.indptr)

    def sample(self, nodes: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        start, end = self.indptr[nodes], self.indptr[nodes + 1]
        r = self.before[nodes] + rng.random(nodes.size) * self.total[nodes]
        pos = np.searchsorted(self.cum, r, side="right")
        pos = np.clip(pos, start, np.maximum(end - 1, start))
        out = np.where(self.deg[nodes] > 0, self.indices[np.minimum(pos, max(self.indices.size - 1, 0))], -1)
        return out


class TypedGraph:
    def __init__(self, tt: sp.csr_matrix | None, tc: sp.csr_matrix | None, comp_family: np.ndarray | None):
        self.n_tracks = (tt.shape[0] if tt is not None else tc.shape[0])
        self.tt = CSRSampler(tt) if tt is not None else None
        self.n_comp = tc.shape[1] if tc is not None else 0
        self.families = [] if comp_family is None else sorted(set(comp_family.tolist()))
        self.tc, self.ct = {}, {}
        if tc is not None:
            tc = sp.csr_matrix(tc)
            for f in self.families:
                cols = np.flatnonzero(comp_family == f)
                sub = tc[:, cols]                                  # tracks x comps of family f
                self.tc[f] = (CSRSampler(sub), cols)
                self.ct[f] = (CSRSampler(sub.T.tocsr()), cols)

    # ---- templates -------------------------------------------------------------------------------------
    def walk_tt(self, starts, length, rng):
        W = np.full((starts.size, length), -1, dtype=np.int32); W[:, 0] = starts
        cur = starts.copy()
        for i in range(1, length):
            nxt = self.tt.sample(cur, rng); alive = nxt >= 0
            W[alive, i] = nxt[alive]; cur = np.where(alive, nxt, cur)
            if not alive.any(): break
        return W

    def walk_tct(self, starts, length, fam_seq, rng):
        """t - c(f0) - t - c(f1) - ... ; fam_seq: (n_walks, n_hops) family id per compound hop."""
        W = np.full((starts.size, length), -1, dtype=np.int32); W[:, 0] = starts
        cur = starts.copy(); alive = np.ones(starts.size, bool)
        for i in range(1, length):
            if i % 2 == 1:      # track -> compound
                nxt = np.full(starts.size, -1, dtype=np.int64)
                fam_here = fam_seq[:, (i - 1) // 2 % fam_seq.shape[1]]
                for f in np.unique(fam_here[alive]):
                    sel = np.flatnonzero(alive & (fam_here == f))
                    smp, cols = self.tc[f]
                    loc = smp.sample(cur[sel], self.rng_for(rng))
                    ok = loc >= 0
                    nxt[sel[ok]] = self.n_tracks + cols[loc[ok]]
                    alive[sel[~ok]] = False
                cur_c = nxt
                W[alive, i] = cur_c[alive]; cur = np.where(alive, cur_c, cur)
            else:               # compound -> track (same family as the one we came through)
                fam_prev = fam_seq[:, (i - 2) // 2 % fam_seq.shape[1]]
                nxt = np.full(starts.size, -1, dtype=np.int64)
                for f in np.unique(fam_prev[alive]):
                    sel = np.flatnonzero(alive & (fam_prev == f))
                    smp, cols = self.ct[f]
                    loc_c = np.searchsorted(cols, cur[sel] - self.n_tracks)
                    t = smp.sample(loc_c, self.rng_for(rng))
                    ok = t >= 0
                    nxt[sel[ok]] = t[ok]; alive[sel[~ok]] = False
                W[alive, i] = nxt[alive]; cur = np.where(alive, nxt, cur)
            if not alive.any(): break
        return W

    @staticmethod
    def rng_for(rng): return rng


def generate_walks(graph: TypedGraph, rounds: int, walk_length: int, tt_per_round: int,
                   use_compound: bool, seed: int = 0, cross_families: bool = True):
    """Yield int32 arrays (n_walks, walk_length), padded with -1 for dead ends.

    One *round* = every track starts one walk per template:
      * `tt_per_round` pure track-track templates (Cooc: 1, Composition: 0, Core: 10  <- Epicure `ii_repeat`)
      * if `use_compound`: one TCT[f] template per compound family f, plus one cross-family bridge template.
    """
    rng = np.random.default_rng(seed)
    starts = np.arange(graph.n_tracks)
    fams = np.array(graph.families)
    n_hops = walk_length // 2 + 1
    for _ in range(rounds):
        for _ in range(tt_per_round):
            yield graph.walk_tt(starts, walk_length, rng)
        if use_compound:
            for f in fams:
                yield graph.walk_tct(starts, walk_length, np.full((starts.size, n_hops), f), rng)
            if cross_families and len(fams) > 1:
                pick = np.stack([rng.permutation(fams)[:2] for _ in range(starts.size)])
                seq = np.tile(pick, (1, n_hops // 2 + 1))[:, :n_hops]
                yield graph.walk_tct(starts, walk_length, seq, rng)
