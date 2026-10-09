"""Residualisation, multi-seed stable FastICA (Hungarian-matched), and GMM 'mode' discovery (paper 2, Sec. 3)."""
from __future__ import annotations
import warnings
from dataclasses import dataclass
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import FastICA, PCA
from sklearn.mixture import GaussianMixture
from .geometry import unit


def residualise(E: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Remove group-mean structure (OLS on one-hot group indicators) so recovered axes are orthogonal to the
    dominant categorical variance (paper 2: food-group residualisation; here: genre-family)."""
    groups = np.asarray(groups); levels = np.unique(groups)
    G = (groups[:, None] == levels[None]).astype(np.float64)
    coef, *_ = np.linalg.lstsq(G, E, rcond=None)
    return E - G @ coef


def _fit_ica(X, n_components, seed, max_iter=1000):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ica = FastICA(n_components=n_components, random_state=seed, max_iter=max_iter, whiten="unit-variance")
        S = ica.fit_transform(X)
    return unit(ica.components_), S      # (k x d) directions in embedding space, (n x k) sources


def _match(A, B):
    """Hungarian matching on |cosine| between rows of A and B. Returns (perm_of_B_for_each_A, matched |cos|)."""
    C = np.abs(A @ B.T)
    r, c = linear_sum_assignment(-C)
    return c[np.argsort(r)], C[r, c][np.argsort(r)]


@dataclass
class IcaResult:
    directions: np.ndarray      # (n_stable x d)
    sources: np.ndarray         # (n x n_stable) sign-normalised (positive skew)
    stability: np.ndarray       # split-half |cos| per retained factor
    seed_stability: np.ndarray  # mean matched |cos| across seeds per retained factor
    all_split_half: np.ndarray  # for all n_components (sorted desc)


def stable_ica(E_res, n_components=20, n_seeds=10, threshold=0.6, n_splits=5, seed=0) -> IcaResult:
    E_res = E_res - E_res.mean(0)
    runs = [_fit_ica(E_res, n_components, seed + s) for s in range(n_seeds)]
    dirs = [r[0] for r in runs]
    # choose the seed with highest mean matched cosine to the others (paper 2)
    score = []
    for i in range(n_seeds):
        score.append(np.mean([_match(dirs[i], dirs[j])[1].mean() for j in range(n_seeds) if j != i]))
    ref_i = int(np.argmax(score)); ref_dirs, ref_S = dirs[ref_i], runs[ref_i][1]
    seed_stab = np.mean([_match(ref_dirs, dirs[j])[1] for j in range(n_seeds) if j != ref_i], axis=0)
    # split-half stability of each reference factor
    rng = np.random.default_rng(seed); n = len(E_res); halves = []
    for _ in range(n_splits):
        p = rng.permutation(n); a, b = p[: n // 2], p[n // 2:]
        Da, _ = _fit_ica(E_res[a], n_components, seed + 1000); Db, _ = _fit_ica(E_res[b], n_components, seed + 2000)
        pa = _match(ref_dirs, Da)[1]; pb = _match(ref_dirs, Db)[1]
        halves.append(np.minimum(pa, pb))
    split = np.mean(halves, axis=0)
    order = np.argsort(-split)
    keep = [i for i in order if split[i] > threshold]
    S = ref_S[:, keep].copy()
    sk = ((S - S.mean(0)) ** 3).mean(0)
    sign = np.where(sk < 0, -1.0, 1.0)             # orient each factor so its high tail is the heavy tail
    S *= sign; D = ref_dirs[keep] * sign[:, None]
    return IcaResult(D, S, split[keep], seed_stab[keep], split[order])


@dataclass
class Mode:
    factor: int
    members: np.ndarray
    pole: np.ndarray
    bic_k: int


def factor_modes(E, source, factor_idx, top_quantile=0.75, k_range=range(3, 8), min_members=6,
                 pca_dim=10, seed=0) -> list[Mode]:
    """Top-quartile items of a factor -> PCA -> GMM (BIC over K) -> modes; each mode = unit-mean pole in full space."""
    thr = np.quantile(source, top_quantile)
    top = np.flatnonzero(source >= thr)
    if len(top) < 3 * min_members: return []
    Z = PCA(n_components=min(pca_dim, len(top) - 1), random_state=seed).fit_transform(E[top])
    best, best_bic = None, np.inf
    for K in k_range:
        if K * min_members > len(top): break
        gm = GaussianMixture(n_components=K, covariance_type="diag", random_state=seed, n_init=2, reg_covar=1e-4).fit(Z)
        bic = gm.bic(Z)
        if bic < best_bic: best, best_bic = gm, bic
    if best is None: return []
    lab = best.predict(Z); modes = []
    for m in np.unique(lab):
        mem = top[lab == m]
        if len(mem) >= min_members:
            modes.append(Mode(factor_idx, mem, unit(unit(E[mem]).mean(0)), best.n_components))
    return modes
