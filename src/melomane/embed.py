"""Skip-gram with negative sampling over walks (gensim).  Epicure paper 2 used PyTorch/PyG (lr 0.0025, batch 32768);
gensim SGNS optimises the same objective with per-sample SGD - hyper-parameters below follow the paper where they
transfer (dim 300, window 7, 5 negatives) and are otherwise recorded in the run manifest."""
from __future__ import annotations
import numpy as np
from gensim.models import Word2Vec


def _sentences(walk_batches):
    for W in walk_batches:
        for row in W:
            row = row[row >= 0]
            if row.size > 1:
                yield [str(x) for x in row.tolist()]


class _Corpus:
    def __init__(self, batches_fn): self.batches_fn = batches_fn
    def __iter__(self): return _sentences(self.batches_fn())


def train_skipgram(batches_fn, n_nodes: int, dim: int = 300, window: int = 7, negative: int = 5,
                   epochs: int = 20, alpha: float = 0.025, min_alpha: float = 0.0001, workers: int = 4,
                   seed: int = 0) -> np.ndarray:
    """batches_fn: zero-arg callable returning a fresh iterator of walk arrays (called once per epoch)."""
    model = Word2Vec(vector_size=dim, window=window, sg=1, hs=0, negative=negative, min_count=1, sample=0,
                     workers=workers, seed=seed, alpha=alpha, min_alpha=min_alpha, epochs=epochs)
    corpus = _Corpus(batches_fn)
    model.build_vocab(corpus)
    model.train(corpus, total_examples=model.corpus_count, epochs=epochs)
    E = np.zeros((n_nodes, dim), dtype=np.float32)
    for tok, idx in model.wv.key_to_index.items():
        nid = int(tok)
        if nid < n_nodes:
            E[nid] = model.wv.vectors[idx]
    return E


def write_walks_file(batches, path: str) -> int:
    """Write walks as space-separated integer tokens, one walk per line (gensim `corpus_file` format).
    corpus_file training bypasses the Python GIL and scales ~linearly with cores - use it for real runs."""
    n_tok = 0
    with open(path, "w") as f:
        for W in batches:
            for row in W:
                row = row[row >= 0]
                if row.size > 1:
                    f.write(" ".join(map(str, row.tolist()))); f.write("\n"); n_tok += row.size
    return n_tok


def train_skipgram_file(path: str, n_nodes: int, dim: int = 300, window: int = 7, negative: int = 5,
                        epochs: int = 20, alpha: float = 0.025, min_alpha: float = 0.0001, workers: int = 8,
                        seed: int = 0) -> np.ndarray:
    model = Word2Vec(corpus_file=path, vector_size=dim, window=window, sg=1, hs=0, negative=negative, min_count=1,
                     sample=0, workers=workers, seed=seed, alpha=alpha, min_alpha=min_alpha, epochs=epochs)
    E = np.zeros((n_nodes, dim), dtype=np.float32)
    for tok, idx in model.wv.key_to_index.items():
        nid = int(tok)
        if nid < n_nodes:
            E[nid] = model.wv.vectors[idx]
    return E
