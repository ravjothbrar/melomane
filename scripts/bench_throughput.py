"""Measure walk-generation and SGNS throughput on this machine to size real runs (tokens/sec)."""
import time, sys, numpy as np
from melomane.synth import make_synth
from melomane.cooc import npmi_edges, edges_to_csr
from melomane.compounds import fit_compound_graph
from melomane.walks import TypedGraph, generate_walks
from melomane.embed import train_skipgram

N = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
S = make_synth(n_tracks=N, n_users=2 * N, seed=0)
t = time.time(); r, c, v = npmi_edges(S.users_x_tracks, min_count=20, min_pair_count=3, topk=100)
print(f"npmi_edges N={N}: {time.time()-t:.1f}s  edges={len(r)}")
t = time.time(); cg = fit_compound_graph(S.features, k=100, top_m=3); print(f"compound graph: {time.time()-t:.1f}s comps={cg.n_comp}")
g = TypedGraph(edges_to_csr(r, c, v, N), cg.edges, cg.comp_family)
t = time.time(); toks = sum(int((W >= 0).sum()) for W in generate_walks(g, 2, 50, 10, True, seed=0))
dt = time.time() - t; print(f"walks (Core template mix, 2 rounds, L=50): {toks/1e6:.1f}M tokens in {dt:.1f}s -> {toks/dt/1e6:.2f}M tokens/s")
mk = lambda: generate_walks(g, 1, 50, 10, True, seed=0)
toks1 = sum(int((W >= 0).sum()) for W in mk())
t = time.time(); train_skipgram(mk, N, dim=300, window=7, epochs=1, workers=2)
dt = time.time() - t; print(f"gensim SGNS dim=300 w=7 neg=5: {toks1/1e6:.1f}M tokens/epoch in {dt:.1f}s -> {toks1/dt/1e6:.3f}M tokens/s (2 workers, incl. python str conversion)")
