import numpy as np, scipy.sparse as sp
from melomane.walks import CSRSampler, TypedGraph, generate_walks
from melomane.compounds import fit_compound_graph


def test_sampler_matches_weights():
    A = sp.csr_matrix(np.array([[0, 1, 3, 0], [2, 0, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1]], float))
    s = CSRSampler(A); rng = np.random.default_rng(0)
    out = s.sample(np.zeros(40000, int), rng)
    assert set(np.unique(out)) == {1, 2}
    assert abs((out == 2).mean() - 0.75) < 0.01
    assert (s.sample(np.full(50, 2), rng) == -1).all()          # dead end
    assert (s.sample(np.full(50, 1), rng) == 0).all()


def _toy_graph(n=60):
    rng = np.random.default_rng(0)
    tt = sp.csr_matrix(np.triu((rng.random((n, n)) < 0.1) * rng.random((n, n)), 1)); tt = tt + tt.T
    feats = {"a": rng.normal(size=(n, 4)), "b": rng.normal(size=(n, 3))}
    cg = fit_compound_graph(feats, k=6, top_m=2, seed=0)
    return TypedGraph(tt, cg.edges, cg.comp_family), cg, tt


def test_walk_structure_tt_and_tct():
    g, cg, tt = _toy_graph()
    walks = list(generate_walks(g, rounds=1, walk_length=9, tt_per_round=1, use_compound=True, seed=0))
    N = g.n_tracks
    tt_walk = walks[0]
    for w in tt_walk:                                            # every hop is an existing edge
        w = w[w >= 0]
        assert (w < N).all() and all(tt[a, b] > 0 for a, b in zip(w[:-1], w[1:]))
    for W in walks[1:]:                                          # alternates track/compound; compound family constant per hop
        for w in W:
            w = w[w >= 0]
            assert (w[0::2] < N).all() and (w[1::2] >= N).all()
            for a, c in zip(w[0::2], w[1::2]):
                assert cg.edges[a, c - N] > 0
    assert len(walks) == 1 + 2 + 1                               # 1 TT + 2 family templates + 1 cross bridge


def test_walks_reference_only_valid_nodes_and_pad():
    g, _, _ = _toy_graph()
    for W in generate_walks(g, 2, 12, 2, True, seed=1):
        assert W.max() < g.n_tracks + g.n_comp and W.min() >= -1


def test_compound_graph_shares_clusters_and_handles_nan():
    rng = np.random.default_rng(0)
    lat = rng.integers(0, 4, 400)
    centers = rng.normal(size=(4, 5)) * 4
    X = centers[lat] + rng.normal(size=(400, 5))
    L = X.copy(); L[:50] = np.nan
    cg = fit_compound_graph({"acoustic": X, "lyric": L}, k={"acoustic": 8, "lyric": 8}, top_m=3, seed=0)
    E = cg.edges.tocsr()
    assert (np.diff(E.indptr)[50:] == 6).all()                   # 3 per family
    assert (np.diff(E.indptr)[:50] == 3).all()                   # NaN rows: no lyric edges
    # tracks with the same latent share their top compound far more than different latents
    top = np.array([E[i, :8].toarray().argmax() for i in range(400)])
    same = np.mean([top[i] == top[j] for i in range(200) for j in range(i) if lat[i] == lat[j]])
    diff = np.mean([top[i] == top[j] for i in range(200) for j in range(i) if lat[i] != lat[j]])
    assert same > 3 * diff
    assert set(cg.comp_family.tolist()) == {0, 1}


def test_compound_graph_pca_for_wide_blocks():
    rng = np.random.default_rng(0)
    lat = rng.integers(0, 3, 300); X = np.eye(3)[lat] @ rng.normal(size=(3, 200)) * 3 + rng.normal(size=(300, 200))
    cg = fit_compound_graph({"wide": X}, k=6, top_m=2, seed=0, pca_dim=8)
    assert cg.centroids["wide"].shape[1] == 8 and cg.edges.shape == (300, 6)
