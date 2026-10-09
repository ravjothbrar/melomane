"""Evaluation metrics replicating Epicure papers 1 & 2 (CV protocols: paper 1 = 10-fold x 20 repeats, paper 2 = 5-fold repeated).

Direction for a probe is always learned on TRAINING folds only (no leakage):
  ordinal   : axis = centroid(highest level) - centroid(lowest level)      (paper 1)
  binary    : axis = centroid(No) -> centroid(Yes)                          (paper 1)
  ridge     : ridge-regression direction                                    (alt. 'fold-trained linear direction')
"""
from __future__ import annotations
import numpy as np
from scipy.stats import spearmanr, mannwhitneyu
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.linear_model import Ridge
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
from .geometry import unit


def repeated_splits(n, n_splits, n_repeats, seed=0, groups=None):
    """Repeated K-fold index splits. With `groups` (e.g. artist id) whole groups stay in one fold (GroupKFold-style,
    reshuffled each repeat) - REQUIRED for music, where same-artist tracks share labels and leak across random folds."""
    rng = np.random.default_rng(seed)
    if groups is None:
        for r in range(n_repeats):
            p = rng.permutation(n); folds = np.array_split(p, n_splits)
            for k in range(n_splits):
                yield np.concatenate([folds[j] for j in range(n_splits) if j != k]), folds[k]
        return
    groups = np.asarray(groups); ug, inv = np.unique(groups, return_inverse=True)
    for r in range(n_repeats):
        order = rng.permutation(len(ug)); fold_of_group = np.empty(len(ug), int)
        # greedy balance by group size: largest groups first into the currently smallest fold
        sizes = np.bincount(inv, minlength=len(ug)); load = np.zeros(n_splits)
        for g in order[np.argsort(-sizes[order], kind="stable")]:
            k = int(np.argmin(load)); fold_of_group[g] = k; load[k] += sizes[g]
        f = fold_of_group[inv]
        for k in range(n_splits):
            yield np.flatnonzero(f != k), np.flatnonzero(f == k)


def _axis(Etr, ytr, method):
    if method == "centroid":
        lo, hi = ytr.min(), ytr.max()
        return Etr[ytr == hi].mean(0) - Etr[ytr == lo].mean(0)
    if method == "ridge":
        return Ridge(alpha=1.0).fit(Etr, ytr).coef_
    raise ValueError(method)


def ordinal_projection_cv(E, y, n_splits=10, n_repeats=20, method="centroid", seed=0, mask=None, groups=None):
    """Mean fold-wise Spearman rho between held-out projection on a train-fold axis and ground truth y."""
    E = np.asarray(E); y = np.asarray(y, float)
    if mask is not None:
        E, y = E[mask], y[mask]
        groups = None if groups is None else np.asarray(groups)[mask]
    rhos = []
    for tr, te in repeated_splits(len(E), n_splits, n_repeats, seed, groups):
        if len(np.unique(y[tr])) < 2: continue
        a = unit(_axis(E[tr], y[tr], method))
        r = spearmanr(E[te] @ a, y[te]).statistic
        if np.isfinite(r): rhos.append(r)
    rhos = np.array(rhos)
    return {"rho": float(rhos.mean()), "rho_sd": float(rhos.std()), "n": int(rhos.size)}


def full_sample_rho(E, y, method="centroid"):
    """Non-CV (in-sample) rho, reported alongside CV as in paper 1 Table 3 (rho vs rho_CV)."""
    a = unit(_axis(E, np.asarray(y, float), method))
    return float(spearmanr(E @ a, y).statistic)


def cohens_d(a, b):
    na, nb = len(a), len(b)
    sp_ = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return float((a.mean() - b.mean()) / sp_)


def binary_separation_cv(E, yes, n_splits=10, n_repeats=20, seed=0, groups=None):
    """One-vs-rest separation: axis(out->in) from train folds, Cohen's d on held-out projections; + Mann-Whitney p."""
    E = np.asarray(E); yes = np.asarray(yes, bool)
    ds, ps = [], []
    for tr, te in repeated_splits(len(E), n_splits, n_repeats, seed, groups):
        if yes[tr].sum() < 2 or (~yes[tr]).sum() < 2 or yes[te].sum() < 2 or (~yes[te]).sum() < 2: continue
        a = unit(E[tr][yes[tr]].mean(0) - E[tr][~yes[tr]].mean(0))
        p = E[te] @ a
        ds.append(cohens_d(p[yes[te]], p[~yes[te]]))
        ps.append(mannwhitneyu(p[yes[te]], p[~yes[te]]).pvalue)
    return {"d": float(np.mean(ds)), "d_sd": float(np.std(ds)), "p_median": float(np.median(ps)), "n": len(ds)}


def knn_purity(E, labels, k=10, tagged=None, chunk=4096, groups=None):
    """Fraction of each tagged item's k nearest neighbours (cosine, among tagged items) that share its tag."""
    U = unit(np.asarray(E, np.float32)); labels = np.asarray(labels)
    g = None if groups is None else np.asarray(groups)
    if tagged is not None:
        U, labels = U[tagged], labels[tagged]
        g = None if g is None else g[tagged]
    pur = []
    for i in range(0, len(U), chunk):
        S = U[i:i + chunk] @ U.T
        if g is None:
            S[np.arange(S.shape[0]), np.arange(i, i + S.shape[0])] = -np.inf
        else:                                   # artist-disjoint: neighbours never come from the item's own group
            S[g[i:i + chunk, None] == g[None, :]] = -np.inf
        nn = np.argpartition(-S, k, axis=1)[:, :k]
        pur.append((labels[nn] == labels[i:i + chunk, None]).mean(1))
    return float(np.concatenate(pur).mean())


def knn_purity_lift(E, labels, k=10, tagged=None, n_boot=200, subsample=None, seed=0, groups=None):
    """purity / random-baseline purity, bootstrapped over size-matched subsamples (paper 1: 200 iterations)."""
    rng = np.random.default_rng(seed)
    E = np.asarray(E); labels = np.asarray(labels)
    idx = np.flatnonzero(tagged) if tagged is not None else np.arange(len(E))
    m = subsample or len(idx)
    lifts = []
    for _ in range(n_boot):
        s = rng.choice(idx, size=m, replace=False)
        p = knn_purity(E[s], labels[s], k, groups=None if groups is None else np.asarray(groups)[s])
        base = float((np.unique(labels[s], return_counts=True)[1] ** 2).sum() / len(s) ** 2)
        lifts.append(p / base)
    return {"lift": float(np.mean(lifts)), "lift_sd": float(np.std(lifts))}


def nmi_kmeans(E, labels, n_clusters=None, seed=0, tagged=None):
    E = np.asarray(E); labels = np.asarray(labels)
    if tagged is not None: E, labels = E[tagged], labels[tagged]
    k = n_clusters or len(np.unique(labels))
    cl = KMeans(n_clusters=k, n_init=5, random_state=seed).fit_predict(unit(E))
    return float(normalized_mutual_info_score(labels, cl))


def soft_nmi(E, tag_matrix, n_draws=20, seed=0):
    """Multi-label NMI: Monte-Carlo average of NMI over random single-label draws from each item's tag set.
    (Definition in paper 2's supplement was not accessible - our operationalisation; document in the write-up.)"""
    rng = np.random.default_rng(seed)
    T = np.asarray(tag_matrix, bool); has = T.any(1)
    vals = []
    for _ in range(n_draws):
        lab = np.array([rng.choice(np.flatnonzero(r)) for r in T[has]])
        vals.append(nmi_kmeans(np.asarray(E)[has], lab, n_clusters=T.shape[1], seed=int(rng.integers(1e9))))
    return float(np.mean(vals))


def mode_coherence(E, member_lists, n_random=50, seed=0):
    """Mean within-mode pairwise cosine vs random same-size sets (paper 2: 0.611 vs 0.097 for Cooc)."""
    rng = np.random.default_rng(seed); U = unit(np.asarray(E))
    def mpc(ix):
        S = U[ix] @ U[ix].T; n = len(ix); return (S.sum() - n) / (n * (n - 1))
    obs = [mpc(np.asarray(m)) for m in member_lists if len(m) > 2]
    rnd = [mpc(rng.choice(len(U), size=len(m), replace=False)) for m in member_lists if len(m) > 2 for _ in range(n_random // 10 + 1)]
    return {"coherence": float(np.mean(obs)), "random": float(np.mean(rnd)), "ratio": float(np.mean(obs) / max(np.mean(rnd), 1e-9))}
