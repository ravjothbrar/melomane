from itertools import combinations

import numpy as np
import pytest
from scipy.spatial import procrustes as scipy_procrustes
from scipy.stats import ortho_group

from melomane.analysis import weat, weat_region_energy, procrustes_align, procrustes_matrix, procrustes_disparity
from melomane.geometry import unit


def _planted_weat(shift=1.5, seed=0, n=40, D=30):
    rng = np.random.default_rng(seed)
    u = unit(rng.normal(size=D))
    E = rng.normal(size=(4 * n, D)) * 0.5
    A = np.arange(0, n); B = np.arange(n, 2 * n); X = np.arange(2 * n, 3 * n); Y = np.arange(3 * n, 4 * n)
    E[A] += 2 * u; E[B] -= 2 * u
    E[X] += shift * u; E[Y] -= shift * u           # X leans to A, Y leans to B
    return E, X, Y, A, B


def test_weat_planted_effect_and_null():
    E, X, Y, A, B = _planted_weat()
    r = weat(E, X, Y, A, B, n_perm=2000, seed=0)
    assert r["d"] > 1.5 and r["p_one_sided"] < 0.01 and r["p_two_sided"] < 0.01
    En, Xn, Yn, An, Bn = _planted_weat(shift=0.0, seed=5)
    n = weat(En, Xn, Yn, An, Bn, n_perm=2000, seed=0)
    assert abs(n["d"]) < 0.7 and n["p_two_sided"] > 0.05
    rev = weat(E, Y, X, A, B, n_perm=2000, seed=0)                       # swap targets: d flips sign, one-sided p flips
    assert abs(rev["d"] + r["d"]) < 1e-9 and rev["p_one_sided"] > 0.95 and rev["p_two_sided"] < 0.01


def test_weat_exact_enumeration_matches_brute_force_and_formula():
    rng = np.random.default_rng(3)
    E = rng.normal(size=(20, 8)); X, Y, A, B = np.arange(4), np.arange(4, 8), np.arange(8, 14), np.arange(14, 20)
    r = weat(E, X, Y, A, B, n_perm=10000)
    assert r["exact"] and r["n_perm"] == 70
    U = unit(E)
    s = lambda w: (U[w] @ U[A].T).mean() - (U[w] @ U[B].T).mean()
    sv = np.array([s(w) for w in range(8)])
    stat = sv[:4].mean() - sv[4:].mean()
    assert abs(r["statistic"] - stat) < 1e-12 and abs(r["d"] - stat / sv.std(ddof=1)) < 1e-12
    null = np.array([sv[list(c)].mean() - np.delete(sv, list(c)).mean() for c in combinations(range(8), 4)])
    assert abs(r["p_one_sided"] - (null >= stat - 1e-12).mean()) < 1e-12
    assert abs(r["p_two_sided"] - (np.abs(null) >= abs(stat) - 1e-12).mean()) < 1e-12


def test_weat_region_energy_design_recovers_signs_and_excludes_artists():
    rng = np.random.default_rng(0); n, D = 1500, 24
    region = rng.integers(0, 3, n); energy = rng.integers(0, 5, n); art = rng.integers(0, 300, n)
    e = unit(rng.normal(size=D))
    shift = np.array([1.0, -1.0, 0.0])[region]                           # region 0 leans high-energy, region 1 low-energy
    E = rng.normal(size=(n, D)) * 0.6 + ((energy - 2) / 2.0 * 2.0 + shift * 1.2)[:, None] * e
    df = weat_region_energy(E, region, energy, groups=art, n_perm=1000, seed=0, max_targets=120, max_attrs=100)
    assert len(df) == 3 and {"d", "p_holm", "p_one_sided", "p_two_sided"} <= set(df.columns)
    row = lambda a, b: df[(df.region_x == a) & (df.region_y == b)].iloc[0]
    assert row(0, 1).d > 1.0 and row(0, 1).p_holm < 0.01
    assert row(0, 2).d > 0.3 and row(1, 2).d < -0.3
    df2 = weat_region_energy(E, region, energy, groups=art, n_perm=1000, seed=0, max_targets=120, max_attrs=100)
    assert np.allclose(df.d, df2.d)                                        # seeded
    # untagged regions (-1) are never targets
    reg2 = region.copy(); reg2[region == 2] = -1
    assert len(weat_region_energy(E, reg2, energy, n_perm=200, seed=0)) == 1


def test_procrustes_recovers_rotation_and_matches_scipy():
    rng = np.random.default_rng(0); n, D = 800, 20
    A = rng.normal(size=(n, D)) @ np.diag(np.linspace(2, 0.5, D)) + 3.0
    Q = ortho_group.rvs(D, random_state=1)
    B = (A - A.mean(0)) @ Q + 0.01 * rng.normal(size=(n, D))
    r = procrustes_align(A, B, n_null=3, seed=0)
    assert r["disparity"] < 1e-3 and r["row_cosine"] > 0.999 and r["axis_cosine_mean"] > 0.99
    assert np.abs(r["R"] - Q).max() < 0.02 and np.allclose(r["R"].T @ r["R"], np.eye(D), atol=1e-9)
    assert r["null_disparity_mean"] > 0.5
    C = rng.normal(size=(n, D))
    d_ref = scipy_procrustes(A, C)[2]
    assert abs(procrustes_align(A, C)["disparity"] - d_ref) < 1e-9
    assert procrustes_align(A, C)["disparity"] > 0.9


def test_procrustes_shared_vocabulary_poles_and_dims():
    rng = np.random.default_rng(1); n, D = 500, 16
    A = rng.normal(size=(n, D)); Q = ortho_group.rvs(D, random_state=2); B = A @ Q
    ids_a = np.arange(n); perm = rng.permutation(n)[:450]
    ids_b = ids_a[perm]; B2 = B[perm]                                     # B has a shuffled subset of A's vocabulary
    r = procrustes_align(A, B2, ids_a, ids_b)
    assert r["n_shared"] == 450 and r["disparity"] < 1e-9
    poles = rng.normal(size=(3, D))
    r = procrustes_align(A, B2, ids_a, ids_b, poles_a=poles, poles_b=poles @ Q)
    assert np.allclose(r["pole_cosine"], 1, atol=1e-6)
    with pytest.raises(ValueError):
        procrustes_align(A, B2)                                            # row counts differ and no ids
    r2 = procrustes_align(A, np.hstack([B, rng.normal(size=(n, 4))]))      # different widths are allowed
    assert r2["R"].shape == (D, D + 4) and abs(r2["disparity"] - 0.2) < 0.05     # 4 of 20 columns are pure noise


def test_procrustes_matrix_all_pairs_and_temporal_wrapper():
    rng = np.random.default_rng(2); n, D = 400, 12
    base = rng.normal(size=(n, D))
    models = {"cooc": base, "core": base @ ortho_group.rvs(D, random_state=3) + 0.3 * rng.normal(size=(n, D)),
              "comp": rng.normal(size=(n, D))}
    m = procrustes_matrix(models, n_null=2)
    assert m["names"] == ["cooc", "core", "comp"] and m["disparity"].shape == (3, 3)
    assert np.allclose(m["disparity"], m["disparity"].T) and np.allclose(np.diag(m["disparity"]), 0)
    assert m["disparity"][0, 1] < 0.3 < m["disparity"][0, 2]              # related pair << unrelated pair
    assert len(m["table"]) == 3 and set(m["table"].columns) >= {"model_a", "model_b", "disparity", "row_cosine"}
    t = procrustes_disparity(models["cooc"], models["comp"], n_null=3)
    assert t["disparity_over_null"] > 0.9 and t["n_shared"] == n
