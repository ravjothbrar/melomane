"""Explorer: payload builder, payload integrity, browser-maths equivalence (numpy reference and node), server."""
import hashlib
import json
import re
import shutil
import subprocess
import urllib.request
from pathlib import Path

import numpy as np
import pytest

from fixtures_run import fake_run, fake_bundles  # noqa: F401
from melomane.api import Melomane
from melomane.release.explorer import (APP_FILES, UMAP_PARAMS, build_explorer, default_app_dir, load_payload, select_tracks,
                                       serve, umap_layout)
from melomane.release.export import load_vocab

TESTS = Path(__file__).resolve().parent
NODE = shutil.which("node")


@pytest.fixture(scope="module")
def built(fake_run, tmp_path_factory):
    out = tmp_path_factory.mktemp("explorer")
    man = build_explorer(fake_run, out, dims=16, seed=0)
    return out, man


def test_app_files_exist_and_are_dependency_free():
    app = default_app_dir()
    for f in APP_FILES:
        assert (app / f).exists()
    for f in (f for f in APP_FILES if not f.endswith(".png")):                       # text files only (og.png is the social-card image)
        text = (app / f).read_text(encoding="utf-8").replace('xmlns="http://www.w3.org/2000/svg"', "").replace('xmlns="http://www.w3.org/2000/svg"', "")   # an XML namespace id, not a request
        for bad in ("cdn.", "cdnjs", "unpkg", "jsdelivr", "googleapis", "http://", "eval(", "document.write"):
            assert bad not in text, (f, bad)
        assert not re.search(r"\.(inner|outer)HTML\s*[+]?=", text), f            # payload strings only ever go in via textContent
    js = (app / "explorer.js").read_text(encoding="utf-8")
    assert "fetch" in js and "itunes.apple.com" not in js                           # search links only, no iTunes calls
    assert "document" not in (app / "math.js").read_text(encoding="utf-8").replace("no DOM", "")
    html = (app / "index.html").read_text(encoding="utf-8")
    assert 'property="og:image"' in html and 'content="https://ravjothbrar.com/melomane/og.png"' in html and (app / "og.png").stat().st_size > 10_000
    assert all(f'{x}' in html for x in ("explorer.js", "style.css", "theta", "color-by", "pole-select", "mode-list", "model-toggle"))


def test_manifest_and_payload_integrity(built):
    out, man = built
    for f in APP_FILES:
        assert (out / f).exists()
    assert man["n"] == 400 and [m["name"] for m in man["models"]] == ["cooc", "composition", "core"]
    assert man["umap"]["metric"] == "cosine" and man["umap"]["n_neighbors"] == 32 and man["umap"]["min_dist"] == 0.004
    assert man["umap"]["note"].startswith("visualisation only")
    assert man["preview"]["kind"] == "search_links" and "20 requests/min" in man["preview"]["note"]
    assert man["license"] == "CC BY 4.0" and "Music4All-Onion" in man["attribution"][0]
    on_disk = json.loads((out / "data" / "manifest.json").read_text(encoding="utf-8"))
    assert on_disk == man
    total = sum(p.stat().st_size for p in (out / "data").rglob("*") if p.is_file())
    assert man["total_bytes"] == total - (out / "data" / "manifest.json").stat().st_size
    assert total < 50 * 1024 * 1024
    for m in man["models"]:
        assert m["n"] == 400 and m["dim"] == 16 and m["dtype"] == "int8"
        for k, e in m["files"].items():
            raw = (out / e["path"]).read_bytes()
            assert len(raw) == e["bytes"] and hashlib.sha256(raw).hexdigest() == e["sha256"], k
        f = {k: out / v["path"] for k, v in m["files"].items()}
        assert f["emb"].stat().st_size == 400 * 16 and f["scale"].stat().st_size == 400 * 4
        assert f["umap"].stat().st_size == 400 * 2 * 4 and f["mode_idx"].stat().st_size == 400 * 2
        assert f["poles"].stat().st_size == m["n_poles"] * 16 * 4


def test_load_payload_decodes_shapes_dtypes_and_geometry(built, fake_bundles):
    out, man = built
    p = load_payload(out)
    t = p.tracks
    assert t["n"] == 400 and len(t["title"]) == 400 and len(t["artist_idx"]) == 400 and len(t["year"]) == 400
    assert t["cid"] == list(range(400)) and max(t["artist_idx"]) < len(t["artist"]) and set(t["category_idx"]) <= set(range(len(t["categories"])))
    assert any(y is None for y in t["year"]) and all(y is None or 1960 <= y <= 2015 for y in t["year"])
    for name, m in p.models.items():
        assert m.U.shape == (400, 16) and m.U.dtype == np.float32 and np.allclose(np.linalg.norm(m.U, axis=1), 1, atol=1e-5)
        assert m.xy.shape == (400, 2) and np.abs(m.xy).max() <= 1.0 + 1e-6 and np.abs(m.xy).max() > 0.99
        assert np.isfinite(m.xy).all() and m.xy.std(0).min() > 0.05             # a real layout, not a collapse
        assert m.poles.shape == (len(m.pole_meta), 16) and np.allclose(np.linalg.norm(m.poles, axis=1), 1, atol=1e-5)
        groups = [x["group"] for x in m.pole_meta]
        assert groups == sorted(groups, key={"supervised": 0, "factor": 1, "mode": 2}.get) and set(groups) == {"supervised", "factor", "mode"}
        assert m.mode_idx.shape == (400,) and m.mode_idx.max() < len(m.modes) and (m.mode_idx >= 0).any()
        for md in m.modes:
            assert md["members"] and max(md["members"]) < 400 and md["n_in_payload"] == len(md["members"])
    # payload vectors are the same small vectors an int8 PCA bundle would carry: neighbours agree with the 24-d bundle loosely
    small = Melomane.from_pretrained(fake_bundles["core-small"])
    idx, _ = p.query("core", 7, exclude_artist=True, k=10)
    assert len(set(idx) & {r["cid"] for r in small.neighbors(7, k=10)}) >= 3


def test_python_reference_query_matches_melomane_api_on_the_payload_vectors(built):
    """Payload.query (the reference for math.js) must agree with the library's own slerp/neighbour code."""
    out, _ = built
    p = load_payload(out)
    mm = p.models["core"]
    api = Melomane(mm.U, [{"cid": i, "artist": p.tracks["artist"][p.tracks["artist_idx"][i]], "title": p.tracks["title"][i]}
                          for i in range(400)])
    for seed in (3, 50, 222):
        idx, sc = p.query("core", seed, [], 0, 10)
        assert idx.tolist() == [r["cid"] for r in api.neighbors(seed, k=10)]
        pole = p.pole("core", "category:jazz")
        idx, sc = p.query("core", seed, ["category:jazz"], 55, 10)
        assert idx.tolist() == [r["cid"] for r in api.slerp(seed, pole, 55, k=10)]
        assert np.allclose(sc, [r["score"] for r in api.slerp(seed, pole, 55, k=10)], atol=1e-3)


def _cases(p, n=24, seed=0):
    rng = np.random.default_rng(seed)
    cases = []
    for model in p.models:
        names = [x["name"] for x in p.models[model].pole_meta]
        for j in range(n // len(p.models)):
            poles = list(rng.choice(names, size=int(rng.integers(0, 4)), replace=False))
            cases.append({"model": model, "seed": int(rng.integers(0, 400)), "poles": poles,
                          "weights": [float(w) for w in rng.uniform(0.2, 2.0, len(poles))] if poles and j % 2 else None,
                          "theta": float(rng.choice([0, 15, 30, 45, 60, 75, 90])), "k": int(rng.choice([5, 10, 20])),
                          "exclude_artist": bool(j % 3)})
    return cases


def _reference(p, c):
    return p.query(c["model"], c["seed"], c["poles"], c["theta"], c["k"], c["weights"], c["exclude_artist"])


@pytest.mark.skipif(NODE is None, reason="node not installed")
@pytest.mark.parametrize("dtype", ["int8", "fp16"])
def test_browser_math_equals_numpy_reference_under_node(fake_run, tmp_path, dtype):
    out = tmp_path / "ex"
    build_explorer(fake_run, out, models=["cooc", "core"], dims=16, dtype=dtype)
    p = load_payload(out)
    cases = _cases(p)
    (tmp_path / "cases.json").write_text(json.dumps(cases))
    r = subprocess.run([NODE, str(TESTS / "node_explorer_check.mjs"), str(out), str(tmp_path / "cases.json")],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    res = json.loads(r.stdout)
    assert all(res["checks"].values()), {k: v for k, v in res["checks"].items() if not v}
    n_match = 0
    for c, got in zip(cases, res["results"]):
        idx, sc = _reference(p, c)
        assert got["idx"] == idx.tolist(), (c, got["idx"], idx.tolist())
        assert np.allclose(got["score"], sc, atol=2e-4)
        if c["theta"] == 0 or not c["poles"]:
            assert got["cosQ"] == pytest.approx(1.0, abs=1e-5)
        else:
            assert got["cosQ"] == pytest.approx(np.cos(np.deg2rad(c["theta"])), abs=2e-4)     # cos(q, seed) = cos(theta)
            n_match += 1
    assert n_match >= 8


def test_fp16_payload_is_smaller_or_equal_and_decodes(fake_run, tmp_path):
    out = tmp_path / "h"
    man = build_explorer(fake_run, out, models=["core"], dims=16, dtype="fp16")
    f = man["models"][0]["files"]
    assert man["models"][0]["dtype"] == "fp16" and "scale" not in f and (out / f["emb"]["path"]).stat().st_size == 400 * 16 * 2
    p = load_payload(out)
    assert p.models["core"].U.shape == (400, 16)


def test_max_tracks_subset_keeps_most_listened_and_remaps_modes(fake_run, tmp_path):
    out = tmp_path / "sub"
    man = build_explorer(fake_run, out, models=["core"], max_tracks=150, dims=16)
    p = load_payload(out)
    vocab = load_vocab(fake_run)
    keep = select_tracks(vocab, 150)
    assert man["n"] == 150 and man["n_vocab"] == 400 and p.tracks["cid"] == keep.tolist()
    assert vocab["n_sessions"].to_numpy()[keep].min() >= np.sort(vocab["n_sessions"].to_numpy())[-150] - 0
    m = p.models["core"]
    assert m.U.shape == (150, 16) and m.xy.shape == (150, 2)
    full_modes = json.loads((fake_run / "S6" / "core" / "modes.json").read_text(encoding="utf-8"))["modes"]
    for pm, fm in zip(m.modes, full_modes):
        assert pm["size"] == fm["size"] and pm["n_in_payload"] == len([c for c in fm["members"] if c in set(p.tracks["cid"])])
        assert [p.tracks["cid"][i] for i in pm["members"]] == [c for c in fm["members"] if c in set(p.tracks["cid"])]


def test_payload_size_cap_is_enforced(fake_run, tmp_path):
    with pytest.raises(ValueError, match="payload"):
        build_explorer(fake_run, tmp_path / "big", models=["core"], dims=16, max_payload_mb=0.001)
    with pytest.raises(ValueError):
        build_explorer(fake_run, tmp_path / "bad", dtype="fp32")
    with pytest.raises(FileNotFoundError):
        build_explorer(tmp_path, tmp_path / "none")                       # a run directory without any model


def test_no_dims_keeps_full_dim_and_bad_hash_is_detected(fake_run, tmp_path):
    out = tmp_path / "nod"
    man = build_explorer(fake_run, out, models=["cooc"], dims=None)
    assert man["models"][0]["dim"] == 32
    assert load_payload(out).models["cooc"].U.shape == (400, 32)
    emb = out / man["models"][0]["files"]["emb"]["path"]
    b = bytearray(emb.read_bytes()); b[0] ^= 1; emb.write_bytes(bytes(b))
    with pytest.raises(ValueError, match="hash"):
        load_payload(out)


def test_umap_layout_is_deterministic_and_separates_planted_clusters():
    rng = np.random.default_rng(0)
    C = rng.normal(size=(3, 12)) * 4
    lab = rng.integers(0, 3, 240)
    U = C[lab] + rng.normal(size=(240, 12))
    U /= np.linalg.norm(U, axis=1, keepdims=True)
    a, b = umap_layout(U, seed=1), umap_layout(U, seed=1)
    np.testing.assert_array_equal(a, b)
    assert a.shape == (240, 2) and np.abs(a).max() == pytest.approx(1.0, abs=1e-6)
    cent = np.array([a[lab == k].mean(0) for k in range(3)])
    within = np.mean([np.linalg.norm(a[lab == k] - cent[k], axis=1).mean() for k in range(3)])
    between = np.mean([np.linalg.norm(cent[i] - cent[j]) for i in range(3) for j in range(i)])
    assert between > 2 * within
    assert UMAP_PARAMS == {"metric": "cosine", "n_neighbors": 32, "min_dist": 0.004, "n_components": 2}


def test_serve_serves_payload_with_module_mime_types(built):
    out, _ = built
    httpd = serve(out, port=0, block=False)
    try:
        port = httpd.server_address[1]
        base = f"http://127.0.0.1:{port}/"
        with urllib.request.urlopen(base + "explorer.js", timeout=10) as r:
            assert r.status == 200 and "javascript" in r.headers["Content-Type"]
            assert b"export async function start" in r.read()
        with urllib.request.urlopen(base + "math.js", timeout=10) as r:
            assert "javascript" in r.headers["Content-Type"]
        with urllib.request.urlopen(base + "data/manifest.json", timeout=10) as r:
            assert json.loads(r.read())["n"] == 400 and "json" in r.headers["Content-Type"]
        with urllib.request.urlopen(base + "data/core/emb.bin", timeout=10) as r:
            assert len(r.read()) == 400 * 16
        with pytest.raises(urllib.error.HTTPError):
            urllib.request.urlopen(base + "nope.txt", timeout=10)
    finally:
        httpd.shutdown(); httpd.server_close()


@pytest.mark.skipif(NODE is None, reason="node not installed")
def test_explorer_ui_smoke_under_node_with_dom_and_webgl_stubs(built):
    out, _ = built
    r = subprocess.run([NODE, str(TESTS / "node_explorer_ui_smoke.mjs"), str(out)], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr + r.stdout
    o = json.loads(r.stdout)
    p = load_payload(out)
    # failure paths
    assert o["fail_returns_null"] and o["fail_visible"] and "served over http" in o["fail_message"]
    assert o["nogl_returns_null"] and "WebGL2" in o["nogl_message"]
    # phones are gated before any data is fetched; iPad-sized touch screens are not
    assert o["gate_returns_null"] and o["gate_fetches"] == 0 and o["gate_visible"] and o["gate_body_class"] and o["ipad_starts"]
    # boot
    assert o["started"] and o["error_hidden"] and o["toggle_buttons"] == 3 and o["status"].startswith("Cooc: 400 tracks, 16-d int8")
    assert "Music4All-Onion" in o["attribution"] and "CC BY 4.0" in o["attribution"]
    assert o["pole_groups"] == ["Supervised", "Emergent: modes", "Emergent: factors"] and o["mode_list_items"] > 5
    assert o["legend_items_category"] == len(p.tracks["categories"])
    # neighbours for seed 5 on the first model (cooc), identical to the reference
    ref_idx, _ = p.query("cooc", 5, [], 0, 10)
    assert o["results_rendered"] == 10 and o["state_results_idx"] == ref_idx.tolist()
    assert o["results_head"] == "Nearest neighbours of the seed" and "1. " in o["first_result"] and "cos " in o["first_result"]
    assert len(o["links"]) == 3 and all(u.startswith("https://") for u in o["links"])
    art, title = p.tracks["artist"][p.tracks["artist_idx"][5]], p.tracks["title"][5]
    assert art in o["seed_card"] and title in o["seed_card"]
    # steering
    ref_idx, _ = p.query("cooc", 5, [o["pole_name"]], 60, 10)
    assert o["steered_idx"] == ref_idx.tolist() and o["steered_theta"] == 60 and o["chips"] == 1
    assert o["steered_head"].startswith("Steered 60") and "0.50" in o["theta_cos"]
    assert "m=cooc" in o["hash"] and f"s={p.tracks['cid'][5]}" in o["hash"] and "t=60" in o["hash"]
    assert o["k5"] == 5
    a = np.asarray(p.tracks["artist_idx"])
    assert any(a[i] == a[5] for i in o["leaky_idx"]) or len(o["leaky_idx"]) == 5      # leaky mode may return the seed's artist
    assert o["legend_year"] == 3 and o["legend_mode"] == 1 and o["legend_category"] == len(p.tracks["categories"])
    # model switch keeps the seed; mode atlas; search; pointer
    assert o["model_after_switch"] == "composition" and o["seed_after_switch"] == 5 and o["results_after_switch"] == 5
    assert o["modes_visible"] and o["mode_detail_children"] >= 3 and o["mode_focus"] > 0 and o["mode_filter_none"] == 0
    assert 0 < o["suggestions"] <= 8
    assert o["hover"] == 42 and o["tooltip_visible"] and p.tracks["title"][42] in o["tooltip_text"]
    assert o["clicked_seed"] == 42 and o["drag_moved_view"] and o["wheel_zoomed"] and o["reset"] and o["fit_zoom_sane"]
    assert o["draw_calls"]["arrays"] > 0 and o["draw_calls"]["elements"] > 0


@pytest.mark.skipif(NODE is None, reason="node not installed")
def test_label_placement_and_palette_under_node():
    """Pure helpers behind the readable labels: placeLabels() never overlaps / leaves the viewport / drops the seed, and the
    categorical palette is bright and distinct enough for the dark stage."""
    r = subprocess.run([NODE, str(TESTS / "node_labels_check.mjs")], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    checks = json.loads(r.stdout)
    assert all(checks.values()), {k: v for k, v in checks.items() if not v}


def test_social_preview_image_is_1200x630_and_deterministic(built, tmp_path):
    """The og:image is drawn from the explorer's own payload (no mock-up): right size, real pixels, reproducible."""
    import matplotlib.image as mpimg
    from melomane.release.preview import NOTE_W, render_preview, word_layout
    out, _ = built
    a = render_preview(out, tmp_path / "a.png")
    b = render_preview(out, tmp_path / "b.png")
    img = mpimg.imread(a)
    assert img.shape[0] == 630 and img.shape[1] == 1200
    assert float(img[..., :3].std()) > 0.03                                          # not a blank card
    assert np.array_equal(mpimg.imread(b), img)                                      # deterministic
    cells, cols, note_col = word_layout()
    assert cols == 7 * 5 + NOTE_W + 7 and note_col == 2 * 6 and len(cells) > 100       # 7 letters, a 3-column slot for the crotchet at the L, 7 gaps


def test_hero_modes_are_compact_separated_named_and_deterministic():
    """The README hero picks its highlighted modes from the payload alone: compact, well-separated, distinctly named, never "unclear"."""
    from melomane.release.preview import pick_hero_modes
    rng = np.random.default_rng(0)
    xy = rng.normal(size=(3000, 2)).astype(np.float32)
    centres = [(-3, 0), (3, 0), (0, 3), (0, -3)]
    modes = []
    for j, (cx, cy) in enumerate(centres):                                            # four tight clumps, appended onto the base cloud
        idx = np.arange(100 * j, 100 * j + 100)
        xy[idx] = np.array([cx, cy]) + rng.normal(scale=0.1, size=(100, 2))
        modes.append({"id": j, "label": f"Mode {j}", "coherence": 0.1 + 0.1 * j, "members": idx.tolist()})
    modes.append({"id": 9, "label": "unclear", "coherence": 0.99, "members": list(range(100, 200))})
    modes.append({"id": 10, "label": "Mode 3", "coherence": 0.05, "members": list(range(300, 400))})          # duplicate name, lower coherence
    modes.append({"id": 11, "label": "Loose", "coherence": 0.9, "members": list(range(1000, 2500))})          # too diffuse
    a = pick_hero_modes(xy, modes, k=4, min_members=50)
    assert [m["label"] for m in a] == ["Mode 3", "Mode 2", "Mode 1", "Mode 0"]                                # by coherence; no "unclear", "Loose" or repeat
    assert [m["id"] for m in pick_hero_modes(xy, modes, k=4, min_members=50)] == [m["id"] for m in a]


def test_hero_card_is_2x_wide_real_and_drawn_from_the_payload(built, tmp_path):
    import matplotlib.image as mpimg
    from melomane.release.preview import render_hero
    out, _ = built
    png = render_hero(out, tmp_path / "hero.png", [("1", "tracks"), ("2", "things")])
    img = mpimg.imread(png)
    assert img.shape[0] == 1800 and img.shape[1] == 3200 and float(img[..., :3].std()) > 0.03


def test_artist_landmarks_rank_by_catalogue_size_and_sit_at_the_median_of_their_tracks():
    """The floating labels: biggest artists first, median position (robust to a stray track), first track as the click target, blanks skipped."""
    from melomane.release.explorer import artist_landmarks
    names = ["Big", "", "Mid", "Tiny"]
    idx = [0, 2, 0, 1, 0, 2, 0, 3, 0, 1]                      # Big x5, "" x2, Mid x2, Tiny x1
    xy = np.zeros((len(idx), 2), np.float32)
    for i, (x, y) in enumerate([(0, 0), (5, 5), (1, 0), (9, 9), (2, 0), (5, 7), (100, 0), (3, 3), (3, 0), (9, 9)]):
        xy[i] = (x, y)
    out = artist_landmarks(idx, names, xy, max_n=10, min_tracks=2)
    assert [o["name"] for o in out] == ["Big", "Mid"]                              # blank skipped, 1-track artist below min_tracks
    big = out[0]
    assert big["x"] == 2.0 and big["y"] == 0.0 and big["top_track_idx"] == 0 and big["n"] == 5     # median ignores the stray at x=100
    assert out[1]["top_track_idx"] == 1 and out[1]["n"] == 2
    assert len(artist_landmarks(idx, names, xy, max_n=1, min_tracks=1)) == 1


def test_built_payload_ships_artist_landmarks_listed_in_the_manifest(built):
    import hashlib
    out, man = built
    for m in man["models"]:
        ent = m["files"]["artists"]
        p = out / ent["path"]
        arts = json.loads(p.read_text(encoding="utf-8"))
        assert arts and set(arts[0]) == {"name", "x", "y", "top_track_idx", "n"}
        assert hashlib.sha256(p.read_bytes()).hexdigest() == ent["sha256"]           # integrity-checked like every other file
        assert all(-1.5 <= a["x"] <= 1.5 and 0 <= a["top_track_idx"] < man["n"] for a in arts)
