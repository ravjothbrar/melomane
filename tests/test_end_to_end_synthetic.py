"""Slow-ish integration test: the full Cooc / Composition / Core pipeline on a corpus with planted structure.
Checks the *qualitative* behaviours the Epicure papers report, so a bug anywhere in graph/walk/embedding/probe code
shows up as a broken expectation rather than silent nonsense."""
import numpy as np, pytest
from melomane.synth import make_synth
from melomane.cooc import npmi_edges, edges_to_csr
from melomane.compounds import fit_compound_graph
from melomane.walks import TypedGraph, generate_walks
from melomane.embed import train_skipgram
from melomane import probes


@pytest.fixture(scope="module")
def models():
    S = make_synth(n_tracks=1500, n_users=3000, seed=0)
    r, c, v = npmi_edges(S.users_x_tracks, min_count=20, min_pair_count=3, topk=40)
    tt = edges_to_csr(r, c, v, 1500)
    cg = fit_compound_graph(S.features, k=40, top_m=3, seed=0)
    g = TypedGraph(tt, cg.edges, cg.comp_family)
    out = {}
    for name, ttn, comp in [("cooc", 1, False), ("comp", 0, True), ("core", 10, True)]:
        mk = lambda ttn=ttn, comp=comp: generate_walks(g, rounds=6, walk_length=20, tt_per_round=ttn, use_compound=comp, seed=1)
        out[name] = train_skipgram(mk, 1500, dim=48, window=5, epochs=3, workers=2, seed=1)
    return S, out


def test_planted_structure_is_recovered(models):
    S, M = models
    res = {}
    for name, E in M.items():
        res[name] = {
            "region_d": np.mean([probes.binary_separation_cv(E, S.region == k, 5, 2)["d"] for k in range(3)]),
            "energy_rho": probes.ordinal_projection_cv(E, S.energy, 5, 2)["rho"],
            "era_rho": probes.ordinal_projection_cv(E, S.era, 5, 2)["rho"],
        }
    print(res)
    assert res["cooc"]["region_d"] > 1.0                       # behaviour encodes the culturally-linked latent
    assert res["comp"]["energy_rho"] > 0.5                     # acoustic composition encodes energy
    assert res["comp"]["energy_rho"] > res["cooc"]["energy_rho"]          # ... more cleanly than behaviour does
    assert res["comp"]["era_rho"] > res["cooc"]["era_rho"] + 0.3    # era is planted in the acoustics, not the listening data
    assert res["cooc"]["region_d"] > res["comp"]["region_d"]         # region is planted in the listening data more than the audio
    assert min(res["core"]["region_d"], res["core"]["energy_rho"] * 3) > 0.5   # Core blends both
