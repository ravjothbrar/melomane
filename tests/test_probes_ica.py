import numpy as np
from melomane import probes
from melomane.ica_modes import residualise, stable_ica, factor_modes
from melomane.geometry import unit


def test_ordinal_probe_recovers_and_shuffle_is_null():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 5, 600); axis = unit(rng.normal(size=40))
    E = np.outer(y, axis) * 0.8 + rng.normal(size=(600, 40))
    r = probes.ordinal_projection_cv(E, y, n_splits=5, n_repeats=3)
    null = probes.ordinal_projection_cv(E, rng.permutation(y), n_splits=5, n_repeats=3)
    assert r["rho"] > 0.5 and abs(null["rho"]) < 0.1                   # leakage-free: shuffled labels give ~0


def test_binary_and_knn_and_nmi():
    rng = np.random.default_rng(1)
    lab = rng.integers(0, 3, 450); C = rng.normal(size=(3, 30)) * 2
    E = C[lab] + rng.normal(size=(450, 30))
    b = probes.binary_separation_cv(E, lab == 0, n_splits=5, n_repeats=3)
    assert b["d"] > 1.5
    assert probes.knn_purity(E, lab, k=10) > 0.9
    assert probes.knn_purity_lift(E, lab, k=10, n_boot=5, subsample=200)["lift"] > 2
    assert probes.nmi_kmeans(E, lab) > 0.8
    T = np.zeros((450, 3), bool); T[np.arange(450), lab] = True; T[::5, (lab[::5] + 1) % 3] = True
    sn = probes.soft_nmi(E, T, n_draws=3)
    assert 0.45 < sn < probes.nmi_kmeans(E, lab) + 1e-9      # noisy multi-label draws: above chance, below clean-label NMI


def test_mode_coherence_ratio():
    rng = np.random.default_rng(2); C = rng.normal(size=(4, 30)) * 3
    lab = rng.integers(0, 4, 400); E = C[lab] + rng.normal(size=(400, 30))
    r = probes.mode_coherence(E, [np.flatnonzero(lab == k) for k in range(4)])
    assert r["ratio"] > 3


def test_residualise_removes_group_means():
    rng = np.random.default_rng(0); g = rng.integers(0, 4, 500)
    E = rng.normal(size=(500, 20)) + np.eye(4)[g] @ rng.normal(size=(4, 20)) * 3
    R = residualise(E, g)
    assert all(np.abs(R[g == k].mean(0)).max() < 1e-8 for k in range(4))


def test_stable_ica_recovers_planted_sources_and_modes():
    rng = np.random.default_rng(0); n, d = 3000, 30
    S = np.column_stack([rng.laplace(size=n), rng.exponential(size=n) - 1, rng.uniform(-1, 1, n) * 1.7])
    # give source 0 a 3-cluster structure in its high tail
    tail = S[:, 0] > np.quantile(S[:, 0], 0.75); W = unit(rng.normal(size=(3, d)))
    Q, _ = np.linalg.qr(rng.normal(size=(d, 3))); W = Q.T                              # orthonormal planted axes
    C = rng.normal(size=(3, d)) * 1.0; sub = rng.integers(0, 3, n)
    E = S @ W * 3.0 + rng.normal(size=(n, d)) * 0.3 + tail[:, None] * C[sub]
    res = stable_ica(E, n_components=3, n_seeds=4, n_splits=2, threshold=0.5)
    assert res.directions.shape[0] >= 2
    best = np.abs(res.directions @ W.T).max(1)                                          # each retained axis ~ a planted one
    assert (best > 0.9).sum() >= 3
    ix = int(np.argmax(np.abs(res.directions @ W[0])))
    modes = factor_modes(E, res.sources[:, ix], ix, min_members=6, seed=0)
    assert len(modes) >= 1 and all(len(m.members) >= 6 for m in modes)
    assert all(abs(np.linalg.norm(m.pole) - 1) < 1e-6 for m in modes)


def test_grouped_cv_removes_artist_leakage():
    """Label is constant within an 'artist' and each artist has its own random offset: random folds leak, grouped folds do not."""
    from melomane.probes import repeated_splits
    rng = np.random.default_rng(0); n_art, per = 120, 8
    art = np.repeat(np.arange(n_art), per)
    y = np.repeat(rng.integers(0, 5, n_art), per)                          # artist-level label
    off = np.repeat(rng.normal(size=(n_art, 40)), per, axis=0) * 2.0        # artist identity is what the embedding encodes
    E = off + rng.normal(size=(n_art * per, 40)) * 0.3
    leaky = probes.ordinal_projection_cv(E, y, 5, 3)["rho"]
    honest = probes.ordinal_projection_cv(E, y, 5, 3, groups=art)["rho"]
    assert leaky > honest + 0.15 and abs(honest) < 0.35
    for tr, te in repeated_splits(len(E), 5, 2, groups=art):
        assert not (set(art[tr]) & set(art[te]))                            # no artist straddles the split


def test_artist_disjoint_knn_purity_drops():
    rng = np.random.default_rng(0); n_art, per = 60, 10
    art = np.repeat(np.arange(n_art), per); lab = np.repeat(rng.integers(0, 3, n_art), per)
    E = np.repeat(rng.normal(size=(n_art, 20)), per, axis=0) + 0.1 * rng.normal(size=(n_art * per, 20))
    assert probes.knn_purity(E, lab, k=5) > probes.knn_purity(E, lab, k=5, groups=art) + 0.3


def test_consolidate_embeddings_is_centre_of_mass():
    from melomane.geometry import consolidate_embeddings
    E = np.array([[1., 0], [3., 0], [0., 2.], [0., 4.]]); ids = np.array([7, 7, 3, 3])
    C, uid = consolidate_embeddings(E, ids)
    assert list(uid) == [3, 7] and np.allclose(C, [[0, 3], [2, 0]])
