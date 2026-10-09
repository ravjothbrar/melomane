import numpy as np
import pytest
from sklearn.metrics import cohen_kappa_score
from scipy.stats import spearmanr

from melomane.analysis import (paired_cluster_bootstrap, holm, permutation_test, bootstrap_ci, spearman_ci,
                               cohens_kappa, weighted_kappa)


def _artist_data(n_art=80, per=10, seed=0, effect=0.0, art_sd=1.0):
    rng = np.random.default_rng(seed)
    art = np.repeat(np.arange(n_art), per)
    shared = np.repeat(rng.normal(size=n_art) * art_sd, per)          # artist-level effect common to both models
    b = shared + rng.normal(size=n_art * per)
    a = shared + effect + rng.normal(size=n_art * per)
    return a, b, art


def test_paired_bootstrap_detects_planted_difference_and_null():
    a, b, art = _artist_data(effect=0.3)
    r = paired_cluster_bootstrap(a, b, art, n_boot=1000, seed=1)
    assert r["ci_low"] > 0 and r["p_boot"] < 0.05
    assert r["ci_low"] < r["mean_diff"] < r["ci_high"]
    assert abs(r["effect_size"] - (a - b).mean() / (a - b).std(ddof=1)) < 1e-12
    assert r["n_clusters"] == 80 and r["n"] == 800
    a0, b0, art0 = _artist_data(effect=0.0, seed=3)
    n0 = paired_cluster_bootstrap(a0, b0, art0, n_boot=1000, seed=1)
    assert n0["ci_low"] < 0 < n0["ci_high"] and n0["p_boot"] > 0.05


def test_paired_bootstrap_seeded_and_cluster_ci_is_wider_when_dependent():
    rng = np.random.default_rng(0)
    art = np.repeat(np.arange(60), 10)
    d_art = np.repeat(rng.normal(size=60), 10)                        # the DIFFERENCE itself is artist-level
    a = d_art + 0.2 * rng.normal(size=600); b = np.zeros(600)
    c = paired_cluster_bootstrap(a, b, art, n_boot=500, seed=2)
    i = paired_cluster_bootstrap(a, b, None, n_boot=500, seed=2)
    assert (c["ci_high"] - c["ci_low"]) > 1.8 * (i["ci_high"] - i["ci_low"])      # ignoring artists is over-confident
    assert paired_cluster_bootstrap(a, b, art, n_boot=200, seed=5) == paired_cluster_bootstrap(a, b, art, n_boot=200, seed=5)


def test_holm_matches_hand_computation_and_nan():
    p = np.array([0.01, 0.04, 0.03, 0.005])
    r = holm(p)
    assert np.allclose(r["p_adj"], [0.03, 0.06, 0.06, 0.02])
    assert list(r["reject"]) == [True, False, False, True]
    r2 = holm([0.02, np.nan, 0.5])
    assert np.isnan(r2["p_adj"][1]) and np.allclose(r2["p_adj"][[0, 2]], [0.04, 0.5]) and r2["m"] == 2
    assert np.all(holm(np.linspace(0.001, 0.9, 30))["p_adj"] >= np.linspace(0.001, 0.9, 30))


def test_permutation_test_planted_and_null():
    rng = np.random.default_rng(0)
    lab = np.repeat([0, 1], 60); x = rng.normal(size=120) + 0.9 * lab
    stat = lambda l: x[l == 1].mean() - x[l == 0].mean()
    r = permutation_test(stat, lab, n=2000, seed=0)
    assert r["p"] < 0.01 and r["observed"] > 0.4 and abs(r["null_mean"]) < 0.05
    xn = rng.normal(size=120)
    p_null = permutation_test(lambda l: xn[l == 1].mean() - xn[l == 0].mean(), lab, n=2000, seed=0)["p"]
    assert p_null > 0.05
    assert permutation_test(stat, lab, n=500, seed=1, alternative="less")["p"] > 0.9
    assert permutation_test(stat, lab, n=500, seed=1) == permutation_test(stat, lab, n=500, seed=1)


def test_bootstrap_ci_mean_median_cluster():
    rng = np.random.default_rng(0)
    x = rng.normal(2.0, 1.0, 400)
    r = bootstrap_ci(x, n_boot=1000, seed=0)
    assert r["ci_low"] < 2.0 < r["ci_high"] and abs(r["estimate"] - x.mean()) < 1e-12
    assert abs(r["se"] - 1 / np.sqrt(400)) < 0.02
    m = bootstrap_ci(x, stat="median", n_boot=300, seed=0)
    assert m["ci_low"] < np.median(x) < m["ci_high"]
    art = np.repeat(np.arange(40), 10); xa = np.repeat(rng.normal(size=40), 10) + 0.1 * rng.normal(size=400)
    assert bootstrap_ci(xa, art, n_boot=500)["se"] > 2 * bootstrap_ci(xa, None, n_boot=500)["se"]
    assert bootstrap_ci(np.array([1.0, np.nan, 3.0]), n_boot=50)["n"] == 2


def test_spearman_ci():
    rng = np.random.default_rng(1)
    x = rng.normal(size=300); y = x + rng.normal(size=300)
    r = spearman_ci(x, y, n_boot=400, seed=0)
    assert abs(r["rho"] - spearmanr(x, y).statistic) < 1e-12
    assert r["ci_low"] < r["rho"] < r["ci_high"] and r["p"] < 1e-10
    z = spearman_ci(x, np.random.default_rng(11).normal(size=300), n_boot=400, seed=0)
    assert z["ci_low"] < 0 < z["ci_high"]


def test_kappas_match_sklearn_and_ordering():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 5, 500); b = np.clip(a + rng.integers(-1, 2, 500), 0, 4)
    k = cohens_kappa(a, b)
    assert abs(k["kappa"] - cohen_kappa_score(a, b)) < 1e-12
    for w in ("linear", "quadratic"):
        wk = weighted_kappa(a, b, w)
        assert abs(wk["kappa"] - cohen_kappa_score(a, b, weights=w)) < 1e-12
    assert weighted_kappa(a, b, "quadratic")["kappa"] > k["kappa"]         # near-misses are forgiven
    assert cohens_kappa(a, a)["kappa"] == 1.0
    assert abs(cohens_kappa(a, rng.permutation(a))["kappa"]) < 0.1
    ci = cohens_kappa(a, b, n_boot=100, seed=0)
    assert ci["ci_low"] < k["kappa"] < ci["ci_high"]
    assert np.isnan(cohens_kappa([1, 1, 1], [1, 1, 1])["kappa"])
    with pytest.raises(ValueError):
        weighted_kappa(a, b, "cubic")
