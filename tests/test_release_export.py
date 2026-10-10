"""Release bundles: contents, quantisation / PCA round trips, config + model card, retention."""
import json

import numpy as np
import pytest
from safetensors.numpy import load_file

from fixtures_run import fake_run, fake_bundles  # noqa: F401
from melomane.api import Melomane
from melomane.geometry import unit
from melomane.release.export import (CAVEATS, PROVENANCE, dequantise_int8, export_all, export_bundle, fit_projection,
                                     list_models, model_card, neighbour_retention, quantise_int8)

CONTRACT_FILES = {"embeddings.safetensors", "vocab.json", "modes.json", "factor_poles.npy", "supervised_poles.json",
                  "supervised_poles.npy", "config.json", "README.md"}


@pytest.fixture(scope="module")
def full_dir(fake_run, tmp_path_factory):
    return export_bundle(fake_run, "core", tmp_path_factory.mktemp("full") / "core")


@pytest.fixture(scope="module")
def ref(full_dir):
    return Melomane.from_pretrained(full_dir)


def test_bundle_has_contract_files_and_fp32_tensor(full_dir, fake_run):
    assert {p.name for p in full_dir.iterdir()} == CONTRACT_FILES
    t = load_file(str(full_dir / "embeddings.safetensors"))
    assert set(t) == {"embeddings"} and t["embeddings"].dtype == np.float32 and t["embeddings"].shape == (400, 32)
    np.testing.assert_array_equal(t["embeddings"], np.load(fake_run / "S5" / "models" / "core" / "embeddings.npy"))


def test_vocab_json_contains_only_whitelisted_fields(full_dir):
    vocab = json.loads((full_dir / "vocab.json").read_text(encoding="utf-8"))
    assert len(vocab) == 400 and [v["cid"] for v in vocab] == list(range(400))
    assert all(set(v) == {"cid", "artist", "title", "album", "year", "category", "dev_split", "region_tags"} for v in vocab)
    assert all(isinstance(v["region_tags"], list) for v in vocab)
    assert any(v["year"] is None for v in vocab) and any(v["region_tags"] for v in vocab)
    raw = (full_dir / "vocab.json").read_text(encoding="utf-8")
    for banned in ("artist_id", "n_sessions", "popularity", "rep_raw", "tags\":\"", "NaN"):
        assert banned not in raw


def test_config_json_provenance_and_caveats(full_dir):
    c = json.loads((full_dir / "config.json").read_text(encoding="utf-8"))
    assert c["model"] == "core" and c["dim"] == 32 and c["n_items"] == 400 and c["dtype"] == "float32"
    assert c["quantisation"] is None and c["pca"] is None and c["seeds"] == [0, 1] and c["profile"] == "smoke"
    assert c["corpus"] == "fakecorpus" and "git_commit" in c and c["created_utc"].endswith("Z")
    assert c["license"] == "CC BY 4.0" and c["contains"] == {"audio": False, "lyrics": False, "per_user_data": False}
    assert "Music4All-Onion" in c["attribution"] and "CC BY 4.0" in c["attribution"] and "Zenodo" in c["attribution"]
    assert c["caveats"] == list(CAVEATS) and c["n_modes"] > 0 and c["n_supervised_poles"] > 0
    assert set(c["files"]) >= {"embeddings.safetensors", "vocab.json", "modes.json"} and "config.json" not in c["files"]


def test_playlist2vec_provenance_is_flagged_unofficial(fake_run, tmp_path):
    out = export_bundle(fake_run, "cooc", tmp_path / "b", sources=["playlist2vec", "music4all-onion"])
    c = json.loads((out / "config.json").read_text(encoding="utf-8"))
    text = c["attribution"]
    assert "Playlist2vec" in text and "unofficial" in text and "Zenodo" in text and "Music4All-Onion" in text
    assert [p["source"] for p in c["data_provenance"]] == ["Playlist2vec", "Music4All-Onion"]
    with pytest.raises(ValueError):
        export_bundle(fake_run, "cooc", tmp_path / "x", sources=["spotify-mpd"])


@pytest.mark.parametrize("quantise,dtype", [(None, np.float32), ("fp16", np.float16), ("int8", np.int8)])
def test_quantisation_keys_and_dtypes(fake_run, tmp_path, quantise, dtype):
    out = export_bundle(fake_run, "core", tmp_path / "q", quantise=quantise)
    t = load_file(str(out / "embeddings.safetensors"))
    assert t["embeddings"].dtype == dtype
    assert ("scale" in t) == (quantise == "int8")
    if quantise == "int8":
        assert t["scale"].shape == (400,) and t["scale"].dtype == np.float32
    assert Melomane.from_pretrained(out).E.dtype == np.float32


def test_int8_roundtrip_error_bounded_by_half_step():
    X = np.random.default_rng(0).normal(size=(50, 16)).astype(np.float32)
    X[3] = 0
    q, s = quantise_int8(X)
    assert q.dtype == np.int8 and np.abs(q).max() <= 127 and s[3] == 1.0
    err = np.abs(dequantise_int8(q, s) - X)
    assert (err <= 0.5 * s[:, None] + 1e-7).all()


def test_fit_projection_is_orthonormal_and_validates():
    E = np.random.default_rng(0).normal(size=(200, 20)).astype(np.float32) * np.linspace(3, 0.1, 20, dtype=np.float32)
    C, mean = fit_projection(E, 5)
    assert mean is None and C.shape == (5, 20)
    np.testing.assert_allclose(C @ C.T, np.eye(5), atol=1e-5)
    C2, mean2 = fit_projection(E, 5, center=True)
    assert mean2.shape == (20,)
    for bad in (0, 20, 25):
        with pytest.raises(ValueError):
            fit_projection(E, bad)


def test_pca_bundle_stores_projection_and_projected_poles(fake_run, tmp_path, ref):
    out = export_bundle(fake_run, "core", tmp_path / "p", quantise="int8", dims=16)
    t = load_file(str(out / "embeddings.safetensors"))
    assert t["embeddings"].shape == (400, 16) and t["pca_components"].shape == (16, 32) and "pca_mean" not in t
    c = json.loads((out / "config.json").read_text(encoding="utf-8"))
    assert c["dim"] == 16 and c["source_dim"] == 32 and c["pca"]["dims"] == 16 and c["pca"]["centered"] is False
    assert np.load(out / "factor_poles.npy").shape[1] == 16 and np.load(out / "supervised_poles.npy").shape[1] == 16
    m = Melomane.from_pretrained(out)
    np.testing.assert_allclose(np.linalg.norm(m.supervised_poles, axis=1), 1.0, atol=1e-5)
    # the stored projection maps full-space points onto the stored (dequantised) rows up to int8 error
    proj = m.project(ref.E, is_direction=False)
    assert np.abs(unit(proj) - m.U).max() < 0.05
    centred = export_bundle(fake_run, "core", tmp_path / "pc", dims=16, pca_center=True)
    assert "pca_mean" in load_file(str(centred / "embeddings.safetensors"))
    mc = Melomane.from_pretrained(centred)
    assert np.abs(unit(mc.project(ref.E, is_direction=False)) - mc.U).max() < 1e-4


def test_retention_fp16_int8_and_pca(fake_run, tmp_path, ref):
    res = {}
    for name, kw in {"fp16": dict(quantise="fp16"), "int8": dict(quantise="int8"), "pca24": dict(quantise="int8", dims=24)}.items():
        out = export_bundle(fake_run, "core", tmp_path / name, **kw)
        res[name] = neighbour_retention(ref, Melomane.from_pretrained(out), k=10, n_queries=400, seed=1)
    assert res["fp16"]["knn_overlap"] >= 0.99 and res["fp16"]["top1_agreement"] >= 0.99
    assert res["int8"]["knn_overlap"] >= 0.97 and res["int8"]["top1_agreement"] >= 0.95
    assert res["pca24"]["knn_overlap"] >= 0.70                         # chance level would be 10/400 = 0.025
    assert res["fp16"]["knn_overlap"] >= res["int8"]["knn_overlap"] >= res["pca24"]["knn_overlap"]
    same = neighbour_retention(ref, ref, k=10, n_queries=50)
    assert same["knn_overlap"] == 1.0 and same["top1_agreement"] == 1.0
    with pytest.raises(ValueError):
        neighbour_retention(ref, Melomane(ref.E[:10], ref.vocab[:10]))


def test_retention_detects_a_broken_variant(ref):
    rng = np.random.default_rng(0)
    junk = Melomane(rng.normal(size=ref.E.shape).astype(np.float32), ref.vocab)
    assert neighbour_retention(ref, junk, k=10, n_queries=100)["knn_overlap"] < 0.2


def test_include_selects_optional_components(fake_run, tmp_path):
    out = export_bundle(fake_run, "core", tmp_path / "min", include=())
    assert {p.name for p in out.iterdir()} == {"embeddings.safetensors", "vocab.json", "config.json"}
    m = Melomane.from_pretrained(out)
    assert m.list_modes() == [] and m.list_supervised_poles() == [] and m.closest_mode(0) is None
    assert m.neighbors(0, k=3) and m.describe(0)["mode"] is None
    part = export_bundle(fake_run, "core", tmp_path / "part", include=("modes", "readme"))
    assert (part / "modes.json").exists() and (part / "README.md").exists() and not (part / "factor_poles.npy").exists()
    with pytest.raises(ValueError):
        export_bundle(fake_run, "core", tmp_path / "bad", include=("lyrics",))
    with pytest.raises(ValueError):
        export_bundle(fake_run, "core", tmp_path / "bad", quantise="int4")
    with pytest.raises(FileNotFoundError):
        export_bundle(fake_run, "nonexistent", tmp_path / "bad")


def test_nonfinite_embeddings_rejected(fake_run, tmp_path):
    import shutil
    run2 = tmp_path / "run"
    shutil.copytree(fake_run, run2)
    p = run2 / "S5" / "models" / "core" / "embeddings.npy"
    E = np.load(p); E[0, 0] = np.nan; np.save(p, E)
    with pytest.raises(ValueError):
        export_bundle(run2, "core", tmp_path / "o")


def test_export_all_produces_full_and_small_bundles(fake_run, fake_bundles):
    assert list_models(fake_run) == ["cooc", "composition", "core"]
    assert set(fake_bundles) == {"cooc", "cooc-small", "composition", "composition-small", "core", "core-small"}
    for name in ("cooc", "composition", "core"):
        full, small = fake_bundles[name], fake_bundles[f"{name}-small"]
        cf = json.loads((full / "config.json").read_text(encoding="utf-8"))
        cs = json.loads((small / "config.json").read_text(encoding="utf-8"))
        assert cf["dim"] == 32 and cs["dim"] == 24 and cs["quantisation"] == "int8" and cs["variant"] == f"{name}-small"
        assert "retention" not in cf and cs["retention"]["reference"] == name and cs["retention"]["knn_overlap"] > 0.6
        assert (small / "embeddings.safetensors").stat().st_size < (full / "embeddings.safetensors").stat().st_size / 2
        readme = (small / "README.md").read_text(encoding="utf-8")
        assert "Compression retention" in readme and f"{cs['retention']['knn_overlap']:.4g}" in readme
        assert cs["files"]["embeddings.safetensors"]                      # hashes refreshed after the retention rewrite


def test_export_all_small_dims_ge_dim_falls_back_to_int8_only(fake_run, tmp_path):
    out = export_all(fake_run, tmp_path, small_dims=64, models=["core"], retention_queries=50)
    c = json.loads((out["core-small"] / "config.json").read_text(encoding="utf-8"))
    assert c["dim"] == 32 and c["pca"] is None and c["quantisation"] == "int8" and c["retention"]["knn_overlap"] >= 0.95
    with pytest.raises(FileNotFoundError):
        export_all(fake_run / "S6", tmp_path / "none")


def test_model_card_content(full_dir):
    cfg = json.loads((full_dir / "config.json").read_text(encoding="utf-8"))
    card = model_card(cfg, {"probes": {"rho": 0.41234567, "n": 3}, "purity": 0.5})
    assert card.startswith("---\nlicense: cc-by-4.0")
    assert "## Metrics" in card and "| probes.rho | 0.4123 |" in card and "| probes.n | 3 |" in card and "| purity | 0.5 |" in card
    assert "Music4All-Onion" in card and "## Caveats" in card and all(c in card for c in CAVEATS)
    assert "Melomane.from_pretrained" in card and "no audio, no lyrics, no per-user data" in card.lower().replace("**", "")
    assert "## Metrics" not in model_card(cfg) and "## Compression retention" not in model_card(cfg)
    assert full_dir.joinpath("README.md").read_text(encoding="utf-8") == model_card(cfg)


def test_metrics_are_rendered_into_bundle_readme(fake_run, tmp_path):
    out = export_bundle(fake_run, "cooc", tmp_path / "m", metrics={"knn_purity_artist_disjoint": 0.25})
    assert "knn_purity_artist_disjoint" in (out / "README.md").read_text(encoding="utf-8")


def test_provenance_constants():
    assert "CC BY 4.0" in PROVENANCE["music4all-onion"]["licence"] and "unofficial" in PROVENANCE["playlist2vec"]["attribution"]


def test_list_models_skips_intermediate_models_and_s12_picks_the_documented_explorer_models(tmp_path):
    """Regression: S5's `cooc_raw` (raw-vocabulary Cooc, only used to average onto canonical tracks) has a different row count than the
    vocabulary, which made bundle export fail; and the explorer must show Cooc/Composition/Core + acoustic-only/lyric-only, not every control."""
    import numpy as np
    from melomane.release.export import list_models
    from melomane.stages.s12_products import EXPLORER_MODELS, explorer_models
    root = tmp_path / "S5" / "models"
    for n in ("core", "cooc", "composition", "cooc_raw", "core_partial30", "composition_lyric", "composition_acoustic", "cooc_eqtok"):
        (root / n).mkdir(parents=True)
        np.save(root / n / "embeddings.npy", np.zeros((2, 2), np.float32))
    names = list_models(tmp_path)
    assert "cooc_raw" not in names and names[:3] == ["cooc", "composition", "core"]
    assert "core_partial30" in names and "cooc_eqtok" in names                      # controls are still bundled
    assert explorer_models(names) == ["cooc", "composition", "core", "composition_acoustic", "composition_lyric"]
    assert explorer_models(["cooc"]) == ["cooc"] and EXPLORER_MODELS[:3] == ("cooc", "composition", "core")
