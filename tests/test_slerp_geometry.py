import numpy as np
from melomane.slerp import slerp, compose_poles, nearest, theta_sweep
from melomane.geometry import unit, participation_ratio, mean_pairwise_cosine


def test_slerp_paper_property():
    rng = np.random.default_rng(0); v = rng.normal(size=300); d = rng.normal(size=300)
    for th in (0, 15, 30, 60, 90, 120):
        q = slerp(v, d, th)
        assert abs(np.linalg.norm(q) - 1) < 1e-9
        assert abs(unit(v) @ q - np.cos(np.deg2rad(th))) < 1e-9       # "at 60 deg cosine to seed = 0.5"
    assert np.allclose(slerp(v, d, 0), unit(v))
    assert abs(slerp(v, d, 90) @ unit(v)) < 1e-9


def test_slerp_degenerate_direction_returns_seed():
    v = np.array([1., 0, 0]); assert np.allclose(slerp(v, 5 * v, 45), v)


def test_compose_and_sweep_is_monotone_toward_pole():
    rng = np.random.default_rng(1)
    centers = unit(rng.normal(size=(2, 50)))
    E = unit(np.vstack([centers[0] + 0.5 * rng.normal(size=(200, 50)) / np.sqrt(50),
                        centers[1] + 0.5 * rng.normal(size=(200, 50)) / np.sqrt(50)]))
    pole = centers[1]; res = theta_sweep(E, seed_idx=0, pole=pole, thetas=(0, 30, 60, 90), k=10)
    to_pole = [r[1] for r in res]; to_seed = [r[2] for r in res]
    assert to_pole == sorted(to_pole) and to_seed == sorted(to_seed, reverse=True)
    assert np.allclose(np.linalg.norm(compose_poles([centers[0], centers[1]])), 1)


def test_nearest_excludes_seed():
    E = unit(np.random.default_rng(0).normal(size=(50, 8))); idx, _ = nearest(E, E[3], 5, exclude=[3])
    assert 3 not in idx and len(idx) == 5


def test_geometry_metrics():
    rng = np.random.default_rng(0)
    assert 90 < participation_ratio(rng.normal(size=(5000, 100))) <= 100
    a = rng.normal(size=(3000, 1)) @ rng.normal(size=(1, 40)) + 1e-4 * rng.normal(size=(3000, 40))
    assert participation_ratio(a) < 1.05
    E = rng.normal(size=(60, 10)); U = unit(E); S = U @ U.T
    brute = (S.sum() - 60) / (60 * 59)
    assert abs(mean_pairwise_cosine(E) - brute) < 1e-9
