"""GPU-ready skip-gram with negative sampling over typed walks (PyTorch).

Mirrors Epicure paper 2's training recipe: SGNS, dim 300, window 7, 5 negatives, lr 0.0025 (SparseAdam, sparse
embeddings), walks regenerated each epoch.  Device-agnostic: cuda / mps / cpu.  On an RTX 50-series GPU install a CUDA>=12.8
PyTorch wheel (cu128 or newer) and check `torch.cuda.get_device_capability()`; fall back to `melomane.embed` (gensim, CPU)
if the GPU build is unavailable.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def pick_device(prefer: str | None = None) -> torch.device:
    if prefer: return torch.device(prefer)
    if torch.cuda.is_available(): return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


class SGNS(nn.Module):
    def __init__(self, n_nodes: int, dim: int, noise: torch.Tensor):
        super().__init__()
        self.inp = nn.Embedding(n_nodes, dim, sparse=True); self.out = nn.Embedding(n_nodes, dim, sparse=True)
        nn.init.uniform_(self.inp.weight, -0.5 / dim, 0.5 / dim); nn.init.zeros_(self.out.weight)
        self.register_buffer("noise", noise)

    def loss(self, centers, contexts, n_neg: int):
        v = self.inp(centers); u = self.out(contexts)
        pos = F.logsigmoid((v * u).sum(-1))
        neg_ids = torch.multinomial(self.noise, centers.numel() * n_neg, replacement=True).view(-1, n_neg)
        un = self.out(neg_ids)
        neg = F.logsigmoid(-(un @ v.unsqueeze(-1)).squeeze(-1)).sum(-1)
        return -(pos + neg).mean()


def _pairs(W: torch.Tensor, window: int):
    """All (center, context) pairs within `window` of a (B x L) walk block; pads (-1) dropped."""
    cs, xs = [], []
    for o in range(1, window + 1):
        a, b = W[:, :-o], W[:, o:]
        m = (a >= 0) & (b >= 0)
        cs += [a[m], b[m]]; xs += [b[m], a[m]]
    return torch.cat(cs), torch.cat(xs)


def train_torch_sgns(make_walks, n_nodes: int, dim: int = 300, window: int = 7, negative: int = 5, epochs: int = 20,
                     lr: float = 0.0025, walk_block: int = 4096, pair_batch: int = 1 << 18, device=None,
                     seed: int = 0, noise_power: float = 0.75, log=print, on_epoch=None, resume: dict | None = None) -> np.ndarray:
    """make_walks(epoch) -> iterator of int32 numpy walk arrays (fresh random walks each epoch, as in PyG Node2Vec).
    n_nodes = n_tracks + n_compounds (walk token id space). Returns the full (n_nodes x dim) input-embedding matrix; slice [:n_tracks]."""
    torch.manual_seed(seed); dev = pick_device(device)
    counts = np.zeros(n_nodes)
    for W in make_walks(0):
        u, c = np.unique(W[W >= 0], return_counts=True); counts[u] += c
    noise = torch.tensor(counts ** noise_power, dtype=torch.float32); noise /= noise.sum()
    model = SGNS(n_nodes, dim, noise).to(dev)
    opt = torch.optim.SparseAdam(list(model.parameters()), lr=lr)
    start = 0
    if resume:
        model.load_state_dict(resume["state"]); opt.load_state_dict(resume["opt"]); start = int(resume["epoch"])
    for ep in range(start, epochs):
        tot, nb = 0.0, 0
        for W in make_walks(ep):
            W = torch.from_numpy(W.astype(np.int64)).to(dev)
            for i in range(0, W.shape[0], walk_block):
                c, x = _pairs(W[i:i + walk_block], window)
                perm = torch.randperm(c.numel(), device=dev); c, x = c[perm], x[perm]
                for j in range(0, c.numel(), pair_batch):
                    opt.zero_grad(set_to_none=True)
                    l = model.loss(c[j:j + pair_batch], x[j:j + pair_batch], negative)
                    l.backward(); opt.step(); tot += l.item(); nb += 1
        log(f"epoch {ep + 1}/{epochs} loss {tot / max(nb, 1):.4f}")
        if on_epoch is not None:
            on_epoch(ep + 1, model, opt, tot / max(nb, 1))
    return model.inp.weight.detach().cpu().numpy()
