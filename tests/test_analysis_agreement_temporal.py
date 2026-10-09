import numpy as np
import pytest
from scipy.stats import ortho_group

from melomane.analysis import (knn_indices, knn_jaccard, neighbour_recall, linear_cka, rbf_cka, agreement_report,
                               procrustes_disparity, mode_persistence)


def _planted(n=1200, D=24, K=30, seed=0):
    rng = np.random.default_rng(seed)
    C = rng.normal(size=(K, D)) * 2.0; lab = rng.integers(0, K, n)
    return C[lab] + rng.normal(size=(n, D)), lab


def test_knn_jaccard_identical_independent_and_noisy_copy():
    E, _ = _planted()
    rng = np.random.default_rng(1)
    same = knn_jaccard(E, E, k=10)
    assert same["jaccard"] == 1.0 and same["overlap"] == 1.0
    rot = knn_jaccard(E, E @ ortho_group.rvs(24, random_state=0), k=10)              # cosine kNN is rotation invariant
    assert rot["jaccard"] > 0.99
    indep = knn_jaccard(E, rng.normal(size=E.shape), k=10)
    assert indep["chance"] < 0.02 and indep["jaccard"] < 3 * indep["chance"] + 0.01
    js = [knn_jaccard(E, E + s * rng.normal(size=E.shape), k=10, n_queries=600, seed=1)["jaccard"] for s in (0.5, 1.5, 4.0)]
    assert js[0] < 1.0 and js[0] > js[1] > js[2] > indep["chance"] and js[0] > 0.2
    assert abs(same["chance"] - 10 * 10 / 1199 / (20 - 100 / 1199)) < 1e-9


def test_artist_disjoint_neighbours_and_queries():
    rng = np.random.default_rng(0); n_art, per = 100, 6
    art = np.repeat(np.arange(n_art), per)
    E = np.repeat(rng.normal(size=(n_art, 16)), per, axis=0) + 0.05 * rng.normal(size=(n_art * per, 16))
    idx = knn_indices(E, 5, groups=art)
    assert not np.any(art[idx] == art[:, None])                                          # never an own-artist neighbour
    plain = knn_indices(E, 5)
    assert np.mean(art[plain] == art[:, None]) > 0.9                                      # ... which unfiltered kNN returns
    j_leaky = knn_jaccard(E, E + 0.02 * rng.normal(size=E.shape), k=5)
    j_disj = knn_jaccard(E, E + 0.02 * rng.normal(size=E.shape), k=5, groups=art)
    assert j_leaky["jaccard"] > j_disj["jaccard"] + 0.1                                    # artist twins inflate raw agreement
    assert j_disj["chance"] > j_leaky["chance"] * 0.99 and j_disj["n_queries"] == n_art * per
    sub = knn_jaccard(E, E, k=5, n_queries=50, seed=3)
    assert sub["n_queries"] == 50 and len(sub["per_item"]) == 50


def test_neighbour_recall_and_cached_reference():
    E, _ = _planted(seed=2)
    rng = np.random.default_rng(0)
    r = neighbour_recall(E, E, k=10)
    assert r["recall"] == 1.0
    Et = E + 0.5 * rng.normal(size=E.shape)
    q = np.arange(0, 1200, 3)
    a = neighbour_recall(E, Et, 10, queries=q)
    b = neighbour_recall(E, Et, 10, queries=q, ref_idx=knn_indices(E, 10, q))
    assert a == b and 0.2 < a["recall"] < 1.0
    assert neighbour_recall(E, rng.normal(size=E.shape), 10)["recall"] < 0.05
    with pytest.raises(ValueError):
        neighbour_recall(E, E[:-1], 10)


def test_cka_identical_rotated_independent_noisy():
    E, _ = _planted(n=1500, D=20, seed=4)
    rng = np.random.default_rng(0)
    Q = ortho_group.rvs(20, random_state=1)
    assert abs(linear_cka(E, E) - 1) < 1e-9 and abs(linear_cka(E, 3.7 * E @ Q) - 1) < 1e-9
    assert abs(rbf_cka(E, E, n_sub=400) - 1) < 1e-9 and abs(rbf_cka(E, 2.0 * E @ Q, n_sub=400) - 1) < 1e-6
    ind = rng.normal(size=E.shape)
    assert linear_cka(E, ind) < 0.1 and rbf_cka(E, ind, n_sub=500, seed=1) < 0.15
    lin = [linear_cka(E, E + s * rng.normal(size=E.shape)) for s in (1.0, 3.0, 8.0)]
    rbf = [rbf_cka(E, E + s * rng.normal(size=E.shape), n_sub=500) for s in (1.0, 3.0, 8.0)]
    assert 1 > lin[0] > lin[1] > lin[2] > linear_cka(E, ind) and 1 > rbf[0] > rbf[1] > rbf[2]
    assert linear_cka(E, rng.normal(size=(1500, 5))) < 0.1                               # different widths are fine


def test_rbf_cka_subsampling_is_seeded_and_stable():
    E, _ = _planted(n=3000, D=16, seed=5)
    Y = E + np.random.default_rng(0).normal(size=E.shape) * 2
    a = rbf_cka(E, Y, n_sub=500, seed=1); b = rbf_cka(E, Y, n_sub=500, seed=1); c = rbf_cka(E, Y, n_sub=500, seed=2)
    assert a == b and abs(a - c) < 0.08
    assert abs(rbf_cka(E, Y, n_sub=500, seed=1, n_repeats=3) - a) < 0.08


def test_agreement_report_keys():
    E, _ = _planted(n=600, D=16, seed=6)
    rep = agreement_report(E, E + np.random.default_rng(0).normal(size=E.shape), k=5, groups=np.arange(600) // 3,
                           n_queries=200, n_sub=300)
    assert {"knn_jaccard", "knn_jaccard_chance", "neighbour_recall", "linear_cka", "rbf_cka"} <= set(rep)
    assert rep["knn_jaccard"] > rep["knn_jaccard_chance"] and 0 < rep["linear_cka"] < 1


# --------------------------------------------------------------------------------------------------- temporal (E6)
def _drifting_world(frac, n=1500, K=10, D=24, seed=0):
    """Early space with K planted clusters; the late space is a rotated copy in which a fraction `frac` of the tracks migrated
    to a different cluster (+ fresh noise). Modes = the clusters in each period."""
    rng = np.random.default_rng(seed)
    C = rng.normal(size=(K, D)) * 3.0
    lab0 = rng.integers(0, K, n)
    E0 = C[lab0] + 0.6 * rng.normal(size=(n, D))
    lab1 = lab0.copy(); moved = rng.random(n) < frac
    lab1[moved] = (lab0[moved] + rng.integers(1, K, moved.sum())) % K
    E1 = (C[lab1] + 0.6 * rng.normal(size=(n, D))) @ ortho_group.rvs(D, random_state=seed + 1)
    modes = lambda lab: [np.flatnonzero(lab == k) for k in range(K)]
    return E0, E1, modes(lab0), modes(lab1)


def test_procrustes_disparity_and_mode_persistence_track_planted_drift():
    out = []
    for frac in (0.0, 0.15, 0.4, 0.8):
        E0, E1, m0, m1 = _drifting_world(frac)
        d = procrustes_disparity(E0, E1, n_null=3)
        p = mode_persistence(m0, m1, E0, E1, n_null=5)
        out.append((d["disparity"], p["mean_best_jaccard"], p["share_persisting"], p["mean_matched_jaccard"], p["pole_cosine_mean"]))
    disp, jac, share, matched, pc = map(np.array, zip(*out))
    assert disp[0] < 0.1 and np.all(np.diff(disp) > 0) and disp[-1] > 0.4                 # drift raises disparity monotonically
    assert jac[0] > 0.999 and np.all(np.diff(jac) < 0) and jac[-1] < 0.25                 # ... and lowers mode persistence
    assert share[0] == 1.0 and share[1] == 1.0 and share[-1] < 0.5
    assert np.all(np.diff(pc) < 0) and pc[0] > 0.95 and np.all(matched <= jac + 1e-12)
    d = procrustes_disparity(E0, E1, n_null=3)
    assert d["disparity"] < d["null_disparity_mean"] and d["disparity_over_null"] < 1


def test_mode_persistence_null_ids_and_dict_modes():
    E0, E1, m0, m1 = _drifting_world(0.0, seed=1)
    r = mode_persistence(m0, m1, n_null=10)                                               # no embeddings: sets only
    assert r["mean_best_jaccard"] == 1.0 and r["null_best_jaccard_mean"] < 0.25 and "pole_cosine" not in r
    modes_json = [{"mode_id": i, "members": list(map(int, m))} for i, m in enumerate(m1)]
    assert mode_persistence(m0, modes_json)["mean_best_jaccard"] == 1.0
    # different vocabularies: the late model lost 200 tracks and is stored in shuffled row order under new ids
    rng = np.random.default_rng(0); keep = np.sort(rng.choice(len(E0), 1300, replace=False)); perm = rng.permutation(1300)
    ids_b = keep[perm]; Eb = E1[keep][perm]
    r2 = mode_persistence(m0, m1, E0, Eb, ids_a=np.arange(len(E0)), ids_b=ids_b, n_null=3)
    assert r2["n_shared"] == 1300 and r2["mean_best_jaccard"] > 0.999 and r2["pole_cosine_mean"] > 0.95
    assert r2["pole_cosine_mean"] > r2["pole_cosine_other_mean"] + 0.5
    with pytest.raises(ValueError):
        mode_persistence([[1, 2]], m1, min_members=3)
