import time

import numpy as np
import pytest

from melomane.analysis import (theta_sweep_metrics, monotonicity, increasing_share, multi_pole_composition,
                               supervised_pole, sample_seed_pole_pairs, orthogonal_residual_snr, snr_ranking)
from melomane.geometry import unit
from melomane.slerp import slerp, compose_poles, nearest


def _planted(V=3000, D=32, K=6, n_art=600, p_tag=0.15, seed=0, art_sd=1.0):
    """Tracks carry K independent tags, each planted as an offset along its own orthonormal direction; artists have offsets
    (so unfiltered retrieval would return the seed's own artist)."""
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.normal(size=(D, K))); U = Q.T
    tags = rng.random((K, V)) < p_tag
    art = rng.integers(0, n_art, V)
    E = 0.5 * rng.normal(size=(V, D)) + art_sd * rng.normal(size=(n_art, D))[art] / np.sqrt(D) * 3
    E = E + 2.0 * (tags.T.astype(float) @ U)
    return E, tags, art


@pytest.fixture(scope="module")
def planted():
    E, tags, art = _planted()
    poles = np.stack([supervised_pole(E, tags[k]) for k in range(len(tags))])
    pairs = sample_seed_pole_pairs(tags, 120, seed=1)
    return E, tags, art, poles, pairs


def test_precision_increases_with_theta_and_q_cos_seed_is_cos_theta(planted):
    E, tags, art, poles, pairs = planted
    df = theta_sweep_metrics(E, pairs["seeds"], poles, tags, pole_index=pairs["pole_index"], exclude_groups=art, k=10)
    assert len(df) == 120 * 7 and set(df.theta) == set(range(0, 91, 15))
    assert np.allclose(df.q_cos_seed, np.cos(np.deg2rad(df.theta)), atol=1e-9)
    assert np.allclose(df[df.theta == 60].q_cos_seed, 0.5, atol=1e-9)                          # paper property
    curve = df.groupby("theta")[["precision", "cos_pole", "cos_seed"]].mean()
    assert curve.precision.iloc[3] > curve.precision.iloc[0] + 0.2           # 45 deg beats the seed's own neighbourhood
    assert curve.precision.loc[60] > curve.precision.loc[0] + 0.3 and curve.precision.loc[90] > 0.7
    assert (np.diff(curve.cos_pole.to_numpy()) > 0).all()                    # target dominance grows
    assert curve.cos_seed.loc[0] > curve.cos_seed.loc[90]                     # seed retention decays
    assert np.allclose(df.prevalence, tags.mean(1)[df.pole])
    mono = monotonicity(df, "precision", by="pair_id")
    assert mono["mean_rho"] > 0.6 and mono["share_positive"] > 0.85
    inc = increasing_share(df, up_to=60, by="pole", n_boot=200)
    assert inc["share"] == 1.0 and inc["n"] == 6 and inc["ci_low"] <= 1.0
    assert increasing_share(df, up_to=60, criterion="endpoint", n_boot=50)["share"] == 1.0
    assert monotonicity(df, "cos_seed", by="pole")["share_negative"] == 1.0


def test_artist_disjoint_retrieval_and_no_signal_is_not_increasing(planted):
    E, tags, art, poles, pairs = planted
    seeds = pairs["seeds"][:40]
    Uu = unit(E)
    df = theta_sweep_metrics(E, seeds, poles, tags, pole_index=pairs["pole_index"][:40], thetas=(0, 30), k=10,
                             exclude_groups=art)
    # re-run retrieval by hand for theta=0: neighbours of the seed must never include its own artist
    for s in seeds[:10]:
        idx, _ = nearest(Uu, Uu[s], 10, exclude=list(np.flatnonzero(art == art[s])))
        assert not np.any(art[idx] == art[s])
    leaky = theta_sweep_metrics(E, seeds, poles, tags, pole_index=pairs["pole_index"][:40], thetas=(0,), k=10)
    assert leaky.cos_seed.mean() > df[df.theta == 0].cos_seed.mean()          # same-artist neighbours inflate seed similarity
    # tags unrelated to geometry => no upward trend
    rng = np.random.default_rng(0)
    junk = rng.random(tags.shape) < 0.15
    dfj = theta_sweep_metrics(E, seeds, poles, junk, pole_index=pairs["pole_index"][:40], exclude_groups=art)
    cj = dfj.groupby("theta").precision.mean()
    assert abs(cj.loc[60] - cj.loc[0]) < 0.06 and abs(cj.mean() - 0.15) < 0.05


def test_metrics_match_reference_slerp_and_are_chunk_invariant(planted):
    E, tags, art, poles, pairs = planted
    seeds, pi = pairs["seeds"][:25], pairs["pole_index"][:25]
    a = theta_sweep_metrics(E, seeds, poles, tags, pole_index=pi, exclude_groups=art, thetas=(0, 60, 90), chunk=256)
    b = theta_sweep_metrics(E, seeds, poles, tags, pole_index=pi, exclude_groups=art, thetas=(0, 60, 90), chunk=7)
    assert np.allclose(a.drop(columns="prevalence").to_numpy(float), b.drop(columns="prevalence").to_numpy(float), equal_nan=True)
    Uu = unit(E.astype(np.float64))
    for i in (0, 7, 19):                                                   # per-pair check against slerp.py + nearest
        s, p = seeds[i], poles[pi[i]]
        q = slerp(Uu[s], p, 60)
        idx, _ = nearest(Uu, q, 10, exclude=list(np.flatnonzero(art == art[s])))
        row = a[(a.pair_id == i) & (a.theta == 60)].iloc[0]
        assert abs(row.precision - tags[pi[i]][idx].mean()) < 1e-9
        assert abs(row.cos_pole - (Uu[idx] @ unit(p)).mean()) < 1e-5
        assert abs(row.cos_seed - (Uu[idx] @ Uu[s]).mean()) < 1e-5


def test_multi_pole_composition_beats_single_pole(planted):
    E, tags, art, poles, _ = planted
    rng = np.random.default_rng(3); n = 100
    combos = np.array([rng.choice(6, 2, replace=False) for _ in range(n)])
    seeds = np.array([rng.choice(np.flatnonzero(~tags[a] & ~tags[b])) for a, b in combos])
    df = multi_pole_composition(E, seeds, poles, combos, tags, thetas=(0, 60, 90), exclude_groups=art)
    assert len(df) == n * 3
    m = df.groupby("theta")[["joint_precision", "best_single_joint", "gain_over_single"]].mean()
    assert m.joint_precision.loc[0] < 0.1                                    # a seed with neither tag retrieves neither
    assert m.joint_precision.loc[60] > 0.25 and m.gain_over_single.loc[60] > 0.1
    assert m.joint_precision.loc[90] > m.best_single_joint.loc[90]
    assert df[df.theta == 60].joint_precision.mean() > 5 * df.prevalence_joint.mean()
    q = compose_poles([poles[combos[0, 0]], poles[combos[0, 1]]])
    assert abs(np.linalg.norm(q) - 1) < 1e-9


def test_scale_and_speed_of_vectorised_sweep():
    rng = np.random.default_rng(0); V, D = 20000, 64
    E = rng.normal(size=(V, D)).astype(np.float32); art = rng.integers(0, 4000, V)
    seeds = rng.integers(0, V, 200); poles = rng.normal(size=(200, D))
    t = time.time()
    df = theta_sweep_metrics(E, seeds, poles, exclude_groups=art)
    assert len(df) == 200 * 7 and df.precision.isna().all() and time.time() - t < 25
    assert np.allclose(df[df.theta == 60].q_cos_seed, 0.5)


def _tiny_snr_case():
    e = np.eye(5)
    rows = np.array([[1, 0, 0, 0, 0],                                          # 0: seed
                     [0.2, 1.0, 0.3, 0.0, 0.0], [0.0, 0.8, -0.2, 0.1, 0.0], [0.1, 0.9, 0.0, -0.3, 0.0],
                     [-0.1, 0.7, 0.2, 0.2, 0.1],                               # 1-4: neighbours of q = e1
                     [0.0, 0.0, 1, 0, 0], [0.0, 0.0, 0, 1, 0], [0.0, 0.0, 0, 0, 1]], float)
    return rows, e


def test_snr_matches_hand_computed_formula_and_paired_equals_grid():
    rows, e = _tiny_snr_case()
    r = orthogonal_residual_snr(rows, [0], e[[1]], theta=90, k=4)
    N = unit(rows[1:5]); R = N - np.outer(N[:, 0], e[0])                      # remove the seed component
    t = R[:, 1]; resid = R[:, [2, 3, 4]]                                      # orthogonal to (seed, pole) = dims 2,3,4
    sig = t.var(ddof=1); tot = resid.var(axis=0, ddof=1).sum()
    assert r["snr"].shape == (1, 1)
    assert abs(r["signal_var"][0, 0] - sig) < 1e-6 and abs(r["snr_total"][0, 0] - sig / tot) < 1e-6
    assert abs(r["snr"][0, 0] - sig / (tot / 3)) < 1e-6 and abs(r["signal_mean"][0, 0] - t.mean()) < 1e-6
    rng = np.random.default_rng(0); E = rng.normal(size=(500, 20)); seeds = np.array([3, 40, 77]); P = rng.normal(size=(4, 20))
    grid = orthogonal_residual_snr(E, seeds, P, theta=60, k=8, chunk=5)
    assert grid["snr"].shape == (4, 3)
    paired = orthogonal_residual_snr(E, np.repeat(seeds, 4), np.tile(P, (3, 1)), theta=60, k=8, paired=True)
    assert np.allclose(paired["snr"].reshape(3, 4).T, grid["snr"])


def test_snr_ranks_in_plane_pole_above_scattered_pole():
    """Operationalisation check on planted neighbourhoods. Pole e1: its neighbours are a continuum inside the (seed, pole)
    plane (they differ only in how much seed vs pole they carry) -> a clean, one-dimensional control -> high SNR.
    Pole e2: its neighbours all sit at the same place along the pole but scatter in random orthogonal directions -> low SNR."""
    rng = np.random.default_rng(0); D = 40
    seed = np.eye(D)[0]; e1, e2 = np.eye(D)[1], np.eye(D)[2]
    phi = np.deg2rad(np.linspace(50, 80, 30))
    ladder = np.cos(phi)[:, None] * seed + np.sin(phi)[:, None] * e1 + 0.002 * rng.normal(size=(30, D))
    r = unit(np.hstack([np.zeros((30, 3)), rng.normal(size=(30, D - 3))]))
    scatter = 0.9 * e2 + 0.4 * r + 0.02 * rng.normal(size=(30, D)) * np.array([1.0 if j == 2 else 0.05 for j in range(D)])
    background = unit(rng.normal(size=(300, D)))
    E = np.vstack([seed, ladder, scatter, background])
    res = orthogonal_residual_snr(E, [0], np.stack([e1, e2]), theta=90, k=10)
    assert res["snr"][0, 0] > 20 * res["snr"][1, 0]
    assert res["snr_total"][0, 0] > 20 * res["snr_total"][1, 0]
    rank = snr_ranking(res, ["in_plane", "scattered"])
    assert list(rank.pole) == ["in_plane", "scattered"] and list(rank["rank"]) == [1, 2]
    with pytest.raises(ValueError):
        snr_ranking(orthogonal_residual_snr(E, [0, 1], np.stack([e1, e2]), paired=True))
