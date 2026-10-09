import numpy as np
import pytest

from melomane import probes
from melomane.analysis import (compress, retention_curve, smallest_config, quantise_int8, dequantise_int8,
                               storage_size_bytes, neighbour_recall, shapley_2player, shapley_attribution)


@pytest.fixture(scope="module")
def planted():
    """Low-rank embedding (24 latent dims with a decaying spectrum, lifted to D = 128) + small noise, an artist-grouped
    planted ordinal probe living in the top latent dimensions."""
    rng = np.random.default_rng(0); n, D, r = 1500, 128, 24
    spec = np.linspace(3.0, 0.7, r)
    Z = rng.normal(size=(n, r)) * spec
    W, _ = np.linalg.qr(rng.normal(size=(D, r))); E = Z @ W.T + 0.03 * rng.normal(size=(n, D)) + 0.5 * rng.normal(size=D)
    lat = Z[:, 0] + 0.3 * rng.normal(size=n)
    y = np.digitize(lat, np.quantile(lat, [0.2, 0.4, 0.6, 0.8]))
    art = rng.integers(0, 300, n)
    probe = lambda X: probes.ordinal_projection_cv(X, y, n_splits=5, n_repeats=2, groups=art)["rho"]
    return E.astype(np.float32), y, art, probe


def test_int8_row_quantisation_properties():
    rng = np.random.default_rng(0); X = rng.normal(size=(200, 30)).astype(np.float32) * rng.uniform(0.1, 5, (200, 1)).astype(np.float32)
    X[3] = 0
    codes, scales = quantise_int8(X)
    assert codes.dtype == np.int8 and scales.dtype == np.float32 and codes.min() >= -127 and codes.max() <= 127
    assert np.all(np.abs(codes).max(1)[np.arange(200) != 3] == 127)                      # each row uses the full range
    back = dequantise_int8(codes, scales)
    assert np.all(np.abs(back - X) <= scales[:, None] / 2 + 1e-6)                        # error <= half a step per row
    assert np.all(back[3] == 0)
    for i in (0, 10, 50):                                                                 # cosine barely moves
        assert X[i] @ back[i] / (np.linalg.norm(X[i]) * np.linalg.norm(back[i])) > 0.999


@pytest.mark.parametrize("method,dtype,center", [("pca", "fp32", True), ("pca", "fp16", True), ("pca", "int8", True),
                                                 ("rp", "int8", True), ("rp", "fp16", False), ("none", "fp32", False)])
def test_size_bytes_is_exact(planted, method, dtype, center):
    E = planted[0]
    c = compress(E, 64, method, dtype, seed=0, center=center)
    stored = c.codes.nbytes + (0 if c.scales is None else c.scales.nbytes) + \
        (0 if c.projection is None else c.projection.nbytes) + (0 if c.mean is None else c.mean.nbytes)
    assert c.size_bytes == stored == storage_size_bytes(len(E), 128, c.dims, method, dtype, center)
    if method == "none":
        assert c.dims == 128
    assert c.view.dtype == np.float32 and c.view.shape == (len(E), c.dims)
    assert c.size_mb == c.size_bytes / 1e6
    n_, D_, d_ = len(E), 128, c.dims                                                       # hand-computed formula
    expect = n_ * d_ * {"fp32": 4, "fp16": 2, "int8": 1}[dtype] + (4 * n_ if dtype == "int8" else 0) + \
        (0 if method == "none" else 4 * d_ * D_) + (4 * D_ if center else 0)
    assert c.size_bytes == expect


def test_pca_64_int8_keeps_probe_and_neighbours(planted):
    E, y, art, probe = planted
    full = probe(E)
    c = compress(E, 64, "pca", "int8", seed=0)
    assert full > 0.6
    assert probe(c.view) / full >= 0.9                                                     # H6: >= 90 % retention at 64-d int8
    ref = E - E.mean(0)
    rec = neighbour_recall(ref, c.view, k=10, n_queries=500, seed=0)
    assert rec["recall"] >= 0.9 and rec["chance"] < 0.02
    assert c.size_bytes < 0.2 * E.nbytes                                                   # ~5x smaller than fp32 full
    # no-op sanity: lossless PCA rotation at full width keeps neighbours exactly
    assert neighbour_recall(ref, compress(E, 128, "pca", "fp32").view, 10, n_queries=300)["recall"] > 0.99


def test_retention_curve_frontier_and_smallest_config(planted):
    E, y, art, probe = planted
    df = retention_curve(E, probe, dims=(128, 64, 32, 8), dtypes=("fp32", "fp16", "int8"), methods=("pca", "rp"), seed=0,
                         n_queries=400)
    assert {"method", "dims", "dtype", "size_mb", "score", "retention", "nn_recall"} <= set(df.columns)
    assert len(df) == 1 + 2 * 4 * 3 and df.iloc[0].method == "full" and df.iloc[0].retention == 1.0
    assert (df[df.method != "full"].groupby(["method", "dims"]).size() == 3).all()
    q = lambda **k: df[(df.method == k["m"]) & (df.dims == k["d"]) & (df.dtype == k["t"])].iloc[0]
    assert q(m="pca", d=64, t="int8").retention >= 0.9 and q(m="pca", d=64, t="int8").nn_recall >= 0.9
    assert q(m="pca", d=8, t="fp32").nn_recall < 0.6 < q(m="pca", d=64, t="fp32").nn_recall  # 8 < 24 latent dims hurts neighbours
    assert q(m="pca", d=8, t="fp32").retention > 0.9                                          # ... but not a top-PC-driven probe
    assert q(m="rp", d=8, t="fp32").nn_recall < q(m="pca", d=8, t="fp32").nn_recall + 1e-9
    assert q(m="pca", d=64, t="int8").size_mb < q(m="pca", d=64, t="fp16").size_mb < q(m="pca", d=64, t="fp32").size_mb
    assert abs(q(m="pca", d=64, t="fp16").retention - q(m="pca", d=64, t="fp32").retention) < 0.03
    best = smallest_config(df, 0.95)
    assert best is not None and best["retention"] >= 0.95
    assert best["size_bytes"] == df[df.retention >= 0.95].size_bytes.min()
    assert best["dims"] < 128 and best["dtype"] in ("int8", "fp16")
    assert smallest_config(df, 0.95, nn_threshold=0.9)["nn_recall"] >= 0.9
    assert smallest_config(df, 1.5) is None
    df2 = retention_curve(E, probe, dims=(64,), dtypes=("int8",), methods=("pca",), neighbour_k=None)
    assert df2.nn_recall.iloc[1:].isna().all() and len(df2) == 2
    assert len(retention_curve(E, probe, dims=(300, 64), dtypes=("int8",), methods=("pca",), neighbour_k=None)) == 2   # dims > D skipped


def test_compress_is_seeded_and_encodes_new_vectors(planted):
    E = planted[0]
    a, b = compress(E, 32, "rp", "fp32", seed=3), compress(E, 32, "rp", "fp32", seed=3)
    assert np.array_equal(a.view, b.view) and not np.array_equal(a.view, compress(E, 32, "rp", "fp32", seed=4).view)
    p = compress(E, 32, "pca", "fp32")
    assert np.allclose(p.encode(E[:5]), p.view[:5], atol=1e-4)
    with pytest.raises(ValueError):
        compress(E, 32, "svd")
    with pytest.raises(ValueError):
        compress(E, 200, "pca")


# ----------------------------------------------------------------------------------------- E2 attribution (Shapley)
def test_shapley_exact_values_efficiency_and_symmetry():
    r = shapley_2player(v_a=0.10, v_l=0.04, v_al=0.20)
    assert abs(r["phi_acoustic"] - (0.5 * 0.10 + 0.5 * (0.20 - 0.04))) < 1e-12
    assert abs(r["phi_lyric"] - (0.5 * 0.04 + 0.5 * (0.20 - 0.10))) < 1e-12
    assert abs(r["phi_acoustic"] + r["phi_lyric"] - 0.20) < 1e-12 and abs(r["synergy"] - 0.06) < 1e-12
    add = shapley_2player(0.3, 0.1, 0.4)                                                   # additive game: phi = own effect
    assert abs(add["phi_acoustic"] - 0.3) < 1e-12 and abs(add["phi_lyric"] - 0.1) < 1e-12 and abs(add["synergy"]) < 1e-12
    sym = shapley_2player(0.2, 0.2, 0.5)
    assert sym["phi_acoustic"] == sym["phi_lyric"] == 0.25
    assert shapley_2player(0.0, 0.0, 0.0)["gain"] == 0.0


def test_shapley_attribution_recovers_planted_split_with_ci():
    rng = np.random.default_rng(0); n = 60                                                 # e.g. 60 probes
    base = rng.normal(0.3, 0.1, n)                                                         # per-probe Cooc scores
    a_eff, l_eff, syn = 0.15, 0.05, 0.06
    cooc = base
    acoustic = base + a_eff + rng.normal(0, 0.02, n)
    lyric = base + l_eff + rng.normal(0, 0.02, n)
    combined = base + a_eff + l_eff + syn + rng.normal(0, 0.02, n)
    r = shapley_attribution(cooc, acoustic, lyric, combined, n_boot=800, seed=1)
    assert abs(r["phi_acoustic"] - (a_eff + syn / 2)) < 0.02 and abs(r["phi_lyric"] - (l_eff + syn / 2)) < 0.02
    assert abs(r["phi_acoustic"] + r["phi_lyric"] - r["gain"]) < 1e-12
    assert abs(r["share_acoustic"] + r["share_lyric"] - 1) < 1e-12 and r["share_acoustic"] > 0.6
    lo, hi = r["ci"]["phi_acoustic"]; assert lo < r["phi_acoustic"] < hi and lo > r["ci"]["phi_lyric"][1]
    assert r["ci"]["synergy"][0] > 0 and abs(r["synergy"] - syn) < 0.03
    assert set(r["means"]) == {"cooc", "acoustic", "lyric", "combined"} and r["n"] == n
    assert r == shapley_attribution(cooc, acoustic, lyric, combined, n_boot=800, seed=1)
    nul = shapley_attribution(cooc, cooc + rng.normal(0, .02, n), cooc + rng.normal(0, .02, n), cooc + rng.normal(0, .02, n),
                              n_boot=500)
    assert nul["ci"]["gain"][0] < 0 < nul["ci"]["gain"][1]
    cl = shapley_attribution(cooc, acoustic, lyric, combined, clusters=np.arange(n) // 6, n_boot=300)
    assert cl["ci"]["gain"][1] - cl["ci"]["gain"][0] > 0
    with pytest.raises(ValueError):
        shapley_attribution(cooc, acoustic[:-1], lyric, combined)
