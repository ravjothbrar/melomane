import numpy as np, pytest
torch = pytest.importorskip("torch")
from melomane.synth import make_synth
from melomane.cooc import npmi_edges, edges_to_csr
from melomane.compounds import fit_compound_graph
from melomane.walks import TypedGraph, generate_walks
from melomane.torch_sgns import train_torch_sgns, _pairs
from melomane import probes


def test_pairs_window_and_padding():
    W = torch.tensor([[0, 1, 2, -1], [3, 4, -1, -1]])
    c, x = _pairs(W, window=2)
    got = set(zip(c.tolist(), x.tolist()))
    assert got == {(0, 1), (1, 0), (1, 2), (2, 1), (0, 2), (2, 0), (3, 4), (4, 3)}


def test_torch_sgns_recovers_planted_structure_like_gensim():
    N = 1000
    S = make_synth(n_tracks=N, n_users=2000, seed=0)
    r, c, v = npmi_edges(S.users_x_tracks, min_count=20, min_pair_count=3, topk=40)
    cg = fit_compound_graph(S.features, k=30, top_m=3)
    g = TypedGraph(edges_to_csr(r, c, v, N), cg.edges, cg.comp_family)
    res = {}
    for name, ttn, comp in [("cooc", 1, False), ("comp", 0, True)]:
        mk = lambda ep, ttn=ttn, comp=comp: generate_walks(g, rounds=3, walk_length=20, tt_per_round=ttn, use_compound=comp, seed=ep)
        E = train_torch_sgns(mk, N + cg.n_comp, dim=32, window=4, epochs=4, lr=0.01, pair_batch=2048, device="cpu", log=lambda *_: None)[:N]
        res[name] = (np.mean([probes.binary_separation_cv(E, S.region == k, 5, 2)["d"] for k in range(3)]),
                     probes.ordinal_projection_cv(E, S.energy, 5, 2)["rho"])
    print(res)
    assert res["cooc"][0] > 1.0 and res["comp"][1] > 0.5
    assert res["cooc"][0] > res["comp"][0]
