"""Track C analysis (PLAN S11): does the *sonic compound vocabulary* generalise to a corpus with NO behaviour data?

Given feature blocks and tag matrices (MTG-Jamendo genre / instrument / mood-theme tags, or FMA genres) we
  1. fit the same compound graph (k-means clusters per family, top-m soft membership) as S4,
  2. measure how much tag information the compounds carry: cluster-tag NMI, and artist-grouped tag prediction
     (ROC-AUC, logistic regression) from (a) the raw standardised features and (b) the compound membership vector alone.
Retention = AUC(compounds) / AUC(raw) says how much of the tag-relevant acoustic information survives the discretisation.
Everything is seeded; folds are grouped by artist (artist leakage would inflate AUC just as in the main evaluation)."""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import normalized_mutual_info_score, roc_auc_score
from sklearn.model_selection import GroupKFold

from .compounds import fit_compound_graph


def _auc_cv(X, y: np.ndarray, groups: np.ndarray, n_splits: int, seed: int) -> float:
    """Out-of-fold ROC-AUC of an L2 logistic regression, folds grouped by artist."""
    n_splits = min(n_splits, len(np.unique(groups)))
    if n_splits < 2 or y.sum() < n_splits or (~y).sum() < n_splits:
        return float("nan")
    oof = np.full(len(y), np.nan)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(np.unique(groups)))
    gid = perm[np.unique(groups, return_inverse=True)[1]]                 # shuffled group ids -> reproducible, non-alphabetical folds
    for tr, te in GroupKFold(n_splits).split(X, y, gid):
        if y[tr].min() == y[tr].max():
            continue
        clf = LogisticRegression(max_iter=300, C=1.0).fit(X[tr], y[tr])
        oof[te] = clf.decision_function(X[te])
    ok = np.isfinite(oof)
    return float(roc_auc_score(y[ok], oof[ok])) if ok.sum() > 20 and y[ok].min() != y[ok].max() else float("nan")


def sonic_transfer(features: dict[str, np.ndarray], tags: dict[str, tuple[sp.spmatrix, list[str]]], groups: np.ndarray, *, k: int = 200,
                   top_m: int = 3, pca_dim: int | None = 32, min_pos: int = 100, max_tags: int = 30, n_splits: int = 5, seed: int = 0) -> dict:
    """features: family -> (n x d) with NaN rows where missing;  tags: category -> (binary matrix (n x T), tag names)."""
    n = next(iter(features.values())).shape[0]
    have = np.zeros(n, bool)
    for X in features.values():
        have |= np.isfinite(X).all(axis=1)
    idx = np.flatnonzero(have)
    if len(idx) < 500:
        raise ValueError(f"only {len(idx)} tracks have features (need >= 500)")
    kk = {f: max(4, min(k, len(idx) // 20)) for f in features}
    cg = fit_compound_graph({f: X for f, X in features.items()}, k=kk, top_m=top_m, seed=seed, pca_dim=pca_dim)
    C = sp.csr_matrix(cg.edges)[idx]
    # raw baseline: concatenated standardised family features -> PCA(64) (a dense, information-rich reference of the same inputs)
    Z = []
    for X in features.values():
        Xi = X[idx].astype(np.float64)
        mu, sd = Xi.mean(0), Xi.std(0)
        Z.append((Xi - mu) / np.where(sd > 1e-9, sd, 1.0))
    Z = np.hstack(Z)
    Zr = PCA(min(64, Z.shape[1], len(idx) - 1), random_state=seed).fit_transform(Z)
    top = np.asarray(C.argmax(axis=1)).ravel()
    g = np.asarray(groups)[idx]
    out = {"n_tracks": int(len(idx)), "n_compounds": int(cg.n_comp), "k": kk, "categories": {}}
    for cat, (Y, names) in tags.items():
        Yi = sp.csr_matrix(Y)[idx].tocsc()
        pos = np.asarray(Yi.sum(0)).ravel()
        cols = [j for j in np.argsort(-pos) if pos[j] >= min_pos][:max_tags]
        rows = []
        for j in cols:
            y = Yi[:, j].toarray().ravel() > 0
            a_raw = _auc_cv(Zr, y, g, n_splits, seed)
            a_cmp = _auc_cv(C, y, g, n_splits, seed)
            rows.append({"tag": names[j], "n_pos": int(pos[j]), "auc_raw": a_raw, "auc_compounds": a_cmp,
                         "retention": (a_cmp - 0.5) / (a_raw - 0.5) if a_raw == a_raw and a_raw > 0.55 else float("nan"),
                         "nmi_cluster": float(normalized_mutual_info_score(y.astype(int), top))})
        if rows:
            ar = np.array([r["auc_raw"] for r in rows]); ac = np.array([r["auc_compounds"] for r in rows])
            ok = np.isfinite(ar) & np.isfinite(ac)
            out["categories"][cat] = {"n_tags": len(rows), "mean_auc_raw": float(ar[ok].mean()) if ok.any() else float("nan"),
                                      "mean_auc_compounds": float(ac[ok].mean()) if ok.any() else float("nan"),
                                      "retention_of_auc_above_chance": float((ac[ok].mean() - 0.5) / (ar[ok].mean() - 0.5)) if ok.any() and ar[ok].mean() > 0.55 else float("nan"),
                                      "mean_nmi": float(np.mean([r["nmi_cluster"] for r in rows])), "tags": rows}
    return out
