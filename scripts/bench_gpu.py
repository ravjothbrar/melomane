"""Measure PyTorch SGNS throughput (positive pairs/s) on THIS machine, then extrapolate wall-clock for real profiles.
Run in Stage 0:  python scripts/bench_gpu.py [--n 20000] [--device auto|cuda|cpu]"""
import argparse, time, numpy as np, torch
from melomane.synth import make_synth
from melomane.cooc import npmi_edges, edges_to_csr
from melomane.compounds import fit_compound_graph
from melomane.walks import TypedGraph, generate_walks
from melomane.torch_sgns import train_torch_sgns, pick_device

ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=3000); ap.add_argument("--device", default=None)
ap.add_argument("--dim", type=int, default=300); ap.add_argument("--pair_batch", type=int, default=1 << 17)
a = ap.parse_args()
dev = pick_device(None if a.device in (None, "auto") else a.device)
S = make_synth(n_tracks=a.n, n_users=2 * a.n, seed=0)
r, c, v = npmi_edges(S.users_x_tracks, min_count=20, min_pair_count=3, topk=100)
cg = fit_compound_graph(S.features, k=100, top_m=3); g = TypedGraph(edges_to_csr(r, c, v, a.n), cg.edges, cg.comp_family)
L, W, EP = 50, 7, 2
mk = lambda ep: generate_walks(g, 1, L, 10, True, seed=ep)
tok = sum(int((w >= 0).sum()) for w in mk(0)); pairs = tok * 2 * W
if dev.type == "cuda": torch.cuda.synchronize()
t = time.time(); train_torch_sgns(mk, a.n + cg.n_comp, dim=a.dim, window=W, epochs=EP, pair_batch=a.pair_batch, device=str(dev), log=lambda *_: None)
if dev.type == "cuda": torch.cuda.synchronize()
dt = time.time() - t; pps = pairs * EP / dt
print(f"device={dev} tokens/epoch={tok/1e6:.1f}M pairs/s={pps/1e6:.2f}M  ({tok*EP/dt/1e6:.2f}M tokens/s)")
for name, n_tracks, rounds, epochs in [("standard 109k tracks", 109_269, 8, 10), ("standard 500k tracks", 500_000, 8, 10), ("paper-profile 109k", 109_269, 100, 20)]:
    toks = n_tracks * 50 * 18 * rounds * epochs      # Core template mix: 10 TT + ~7 families + 1 cross = 18 walks per round
    print(f"  {name:22s} Core model: {toks/1e9:6.1f}B token-passes -> {toks*2*W/pps/3600:7.1f} h")
