import numpy as np, scipy.sparse as sp
from melomane.cooc import npmi_edges


def dense_npmi(X, min_count):
    X = (X > 0).astype(float); n = X.shape[0]; c = X.sum(0)
    co = X.T @ X; out = {}
    for i in range(X.shape[1]):
        for j in range(i + 1, X.shape[1]):
            if c[i] < min_count or c[j] < min_count or co[i, j] == 0: continue
            pij = co[i, j] / n; pmi = np.log(pij / ((c[i] / n) * (c[j] / n)))
            v = pmi / -np.log(pij)
            if v > 0: out[(i, j)] = v
    return out


def test_matches_bruteforce():
    rng = np.random.default_rng(0)
    X = sp.csr_matrix((rng.random((300, 25)) < 0.25).astype(float))
    ref = dense_npmi(X.toarray(), min_count=5)
    r, c, v = npmi_edges(X, min_count=5, min_pair_count=1, topk=None, chunk=7)
    got = {(a, b): x for a, b, x in zip(r, c, v) if a < b}
    assert set(got) == set(ref)
    assert np.allclose([got[k] for k in ref], [ref[k] for k in ref], atol=1e-5)


def test_symmetric_and_topk_and_mincount():
    rng = np.random.default_rng(1)
    X = sp.csr_matrix((rng.random((500, 60)) < 0.3).astype(float))
    r, c, v = npmi_edges(X, min_count=20, min_pair_count=2, topk=5, chunk=16)
    d = {(a, b): x for a, b, x in zip(r, c, v)}
    assert all((b, a) in d and abs(d[(b, a)] - x) < 1e-9 for (a, b), x in d.items())
    assert (v > 0).all()
    cnt = np.asarray((X > 0).sum(0)).ravel()
    assert cnt[np.unique(r)].min() >= 20


def test_recovers_block_structure():
    rng = np.random.default_rng(2)
    n_items, n_blocks, n_users = 90, 3, 900
    blk = np.repeat(np.arange(n_blocks), n_items // n_blocks)
    ub = rng.integers(0, n_blocks, n_users)
    P = np.where(ub[:, None] == blk[None], 0.35, 0.05)
    X = sp.csr_matrix((rng.random((n_users, n_items)) < P).astype(float))
    r, c, v = npmi_edges(X, min_count=20, min_pair_count=3, topk=20)
    same = blk[r] == blk[c]
    assert same.mean() > 0.95
