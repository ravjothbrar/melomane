"""Typed 'sonic / lyrical compound' graph: the music analogue of FlavorDB's typed compound nodes.

Each *family* (e.g. timbre, harmony, rhythm, dynamics, lyric-theme) has its own feature block. Tracks are
clustered per family into K discrete compound types (k-means); every track links to its `top_m` nearest
compound clusters of each family (soft weights).  Tracks with no data for a family (NaN rows, e.g.
instrumentals for lyrics) simply get no edges in that family.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import scipy.sparse as sp
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA


@dataclass
class CompoundGraph:
    family_names: list[str]
    comp_family: np.ndarray            # (n_comp,) family index of each compound node
    edges: sp.csr_matrix               # (n_tracks x n_comp) weights
    centroids: dict[str, np.ndarray]
    scalers: dict[str, tuple[np.ndarray, np.ndarray]]

    @property
    def n_comp(self) -> int:
        return self.edges.shape[1]

    def family_slice(self, f: int) -> np.ndarray:
        return np.flatnonzero(self.comp_family == f)


def _standardise(X):
    mu = np.nanmean(X, axis=0); sd = np.nanstd(X, axis=0); sd[sd < 1e-9] = 1.0
    return (X - mu) / sd, (mu, sd)


def fit_compound_graph(features: dict[str, np.ndarray], k: dict[str, int] | int = 400, top_m: int = 3,
                       seed: int = 0, temperature: float | None = None, pca_dim: int | None = 32) -> CompoundGraph:
    names = list(features)
    n_tracks = next(iter(features.values())).shape[0]
    rows, cols, vals, fam = [], [], [], []
    centroids, scalers = {}, {}
    offset = 0
    for fi, name in enumerate(names):
        X = np.asarray(features[name], dtype=np.float64)
        Z, sc = _standardise(X); scalers[name] = sc
        ok = ~np.isnan(Z).any(axis=1)
        if pca_dim is not None and Z.shape[1] > pca_dim:      # wide blocks (MFCC stats, BERT, ...) -> whitened PCA before k-means
            pca = PCA(n_components=pca_dim, whiten=True, random_state=seed).fit(Z[ok])
            Zp = np.full((Z.shape[0], pca_dim), np.nan); Zp[ok] = pca.transform(Z[ok]); Z = Zp
            scalers[name] = (*sc, pca)
        kk = k if isinstance(k, int) else k[name]
        kk = min(kk, max(2, ok.sum() // 5))
        km = MiniBatchKMeans(n_clusters=kk, random_state=seed, n_init=3, batch_size=4096).fit(Z[ok])
        centroids[name] = km.cluster_centers_
        d2 = ((Z[ok][:, None, :] - km.cluster_centers_[None]) ** 2).sum(-1) if ok.sum() * kk < 5e7 else None
        if d2 is None:   # chunked for scale
            d2 = np.vstack([((Z[ok][i:i + 2048][:, None, :] - km.cluster_centers_[None]) ** 2).sum(-1)
                            for i in range(0, ok.sum(), 2048)])
        m = min(top_m, kk)
        nn = np.argpartition(d2, m - 1, axis=1)[:, :m]
        nd = np.take_along_axis(d2, nn, axis=1)
        tau = temperature if temperature is not None else max(np.median(nd), 1e-6)
        w = np.exp(-(nd - nd.min(axis=1, keepdims=True)) / tau)
        w /= w.sum(axis=1, keepdims=True)
        tr = np.flatnonzero(ok)
        rows.append(np.repeat(tr, m)); cols.append((nn + offset).ravel()); vals.append(w.ravel())
        fam.append(np.full(kk, fi)); offset += kk
    E = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(n_tracks, offset))
    return CompoundGraph(names, np.concatenate(fam), E, centroids, scalers)
