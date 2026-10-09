"""Melomane loader API over the fake run's release bundles (full fp32 and 24-d int8 'small')."""
import json

import numpy as np
import pytest

from fixtures_run import fake_run, fake_bundles  # noqa: F401  (session fixtures)
from melomane.api import Melomane, TrackNotFound, mode_poles_from_members, primary_mode_assignment
from melomane.curation import normalize_artist
from melomane.geometry import unit


@pytest.fixture(scope="module")
def m(fake_bundles):
    return Melomane.from_pretrained(fake_bundles["core"])


@pytest.fixture(scope="module")
def small(fake_bundles):
    return Melomane.from_pretrained(fake_bundles["core-small"])


def test_load_and_basic_shapes(m, small, fake_run):
    assert len(m) == 400 and m.dim == 32 and small.dim == 24
    E = np.load(fake_run / "S5" / "models" / "core" / "embeddings.npy")
    np.testing.assert_allclose(m.E, E, rtol=0, atol=0)                       # fp32 bundle is lossless
    assert m.vec(3).shape == (32,) and m.vec("cid:3").tolist() == m.vec(3).tolist()
    assert m.model_name == "core" and small.model_name == "core-small"
    assert len(m.list_modes()) > 5 and len(m.list_supervised_poles()) > 10


def test_from_pretrained_accepts_file_and_rejects_missing(fake_bundles, tmp_path):
    assert len(Melomane.from_pretrained(fake_bundles["core"] / "embeddings.safetensors")) == 400
    with pytest.raises(FileNotFoundError):
        Melomane.from_pretrained(tmp_path / "nope")
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError):
        Melomane.from_pretrained(tmp_path / "empty")


def test_results_are_json_able_and_metadata_only(m):
    outs = [m.neighbors(0), m.slerp(0, "category:jazz", 45), m.list_modes(), m.mode_members(0), m.list_supervised_poles(),
            m.describe(0), m.find("velvet", 3), m.list_poles()]
    json.dumps(outs)                                                          # raises on numpy leakage
    banned = {"lyrics", "audio", "preview", "user", "tags", "artist_id"}
    for o in outs:
        for row in (o if isinstance(o, list) else [o]):
            assert not banned & set(row)
    d = m.describe(0)
    assert {"artist", "title", "category", "mode", "year", "album", "region_tags"} <= set(d)


def test_vocab_fields_are_whitelisted_on_load():
    rows = [{"cid": i, "artist": f"A{i}", "title": f"T{i}", "lyrics": "secret", "user_id": 7, "category": "x"} for i in range(5)]
    E = np.random.default_rng(0).normal(size=(5, 4)).astype(np.float32)
    mm = Melomane(E, rows)
    assert "lyrics" not in mm.describe(0) and set(mm.vocab[0]) == {"cid", "artist", "title", "album", "year", "category", "dev_split", "region_tags"}
    with pytest.raises(ValueError):
        Melomane(E, [dict(r, cid=(i + 1) % 5) for i, r in enumerate(rows)])   # wrong cid order
    with pytest.raises(ValueError):
        Melomane(E, rows[:3])                                                 # length mismatch


def test_neighbors_artist_disjoint_by_default(m):
    same_artist_seen = False
    for seed in range(0, 400, 7):
        a = normalize_artist(m.vocab[seed]["artist"])
        res = m.neighbors(seed, k=10)
        assert len(res) == 10 and seed not in [r["cid"] for r in res]
        assert all(normalize_artist(r["artist"]) != a for r in res)
        assert [r["score"] for r in res] == sorted((r["score"] for r in res), reverse=True)
        leaky = m.neighbors(seed, k=10, exclude_artist=False)
        same_artist_seen |= any(normalize_artist(r["artist"]) == a for r in leaky)
    assert same_artist_seen                       # the planted artist structure really is retrievable when not excluded


def test_neighbors_exclude_self_flag(m):
    with_self = m.neighbors(5, k=3, exclude_self=False, exclude_artist=False)
    assert with_self[0]["cid"] == 5 and with_self[0]["score"] == pytest.approx(1.0, abs=1e-4)
    keep_seed_drop_artist = m.neighbors(5, k=3, exclude_self=False, exclude_artist=True)
    assert keep_seed_drop_artist[0]["cid"] == 5
    assert all(normalize_artist(r["artist"]) != normalize_artist(m.vocab[5]["artist"]) for r in keep_seed_drop_artist[1:])


def test_slerp_60_degrees_gives_cos_half(m):
    for pole in ("category:jazz", "mode:0", "factor:1", "track:cid:77"):
        q = m.slerp_vector(10, pole, 60)
        assert float(q @ m.U[10]) == pytest.approx(0.5, abs=1e-5)
        assert np.linalg.norm(q) == pytest.approx(1.0, abs=1e-5)
    assert m.slerp_vector(10, "category:jazz", 0) == pytest.approx(m.U[10], abs=1e-6)
    d = np.random.default_rng(1).normal(size=32)
    assert float(m.slerp_vector(10, d, 60) @ m.U[10]) == pytest.approx(0.5, abs=1e-5)


def test_slerp_theta_zero_equals_neighbors_and_moves_toward_pole(m, small):
    for mm in (m, small):
        base = mm.neighbors(20, k=10)
        s0 = mm.slerp(20, "category:jazz", 0, k=10)
        assert [r["cid"] for r in s0] == [r["cid"] for r in base]
        jazz = lambda rows: np.mean([r["category"] == "jazz" for r in rows])
        seeds = [i for i in range(0, 400, 9) if mm.vocab[i]["category"] != "jazz"]
        lo = np.mean([jazz(mm.slerp(i, "category:jazz", 0)) for i in seeds])
        hi = np.mean([jazz(mm.slerp(i, "category:jazz", 70)) for i in seeds])
        assert hi > lo + 0.3, (mm.model_name, lo, hi)


def test_slerp_excludes_seed_and_artist_and_reports_cosines(m):
    res = m.slerp(33, "category:folk", 30, k=8)
    a = normalize_artist(m.vocab[33]["artist"])
    assert 33 not in [r["cid"] for r in res] and all(normalize_artist(r["artist"]) != a for r in res)
    for r in res:
        assert r["cos_seed"] == pytest.approx(float(m.U[r["cid"]] @ m.U[33]), abs=1e-3)
    leaky = m.slerp(33, "category:folk", 30, k=8, exclude_artist=False)
    assert 33 not in [r["cid"] for r in leaky]


def test_slerp_multi_pole_weights_and_aliases(m):
    both = m.slerp_vector(4, ["category:jazz", "era:1990s"], 45)
    d_j, d_e = m.pole("category:jazz"), m.pole("era:1990s")
    tilt = lambda w: float(m.slerp_vector(4, ["category:jazz", "era:1990s"], 45, weights=w) @ d_j)
    assert tilt([1, 0.01]) > tilt([1, 1]) > tilt([0.01, 1])
    assert m.pole("jazz").tolist() == d_j.tolist()                            # bare label alias
    assert both.shape == (32,)
    single = m.slerp(4, ["category:jazz"], 45)
    assert [r["cid"] for r in single] == [r["cid"] for r in m.slerp(4, "category:jazz", 45)]
    with pytest.raises(ValueError):
        m.slerp(4, ["category:jazz", "era:1990s"], 45, weights=[1.0])
    with pytest.raises(KeyError):
        m.slerp(4, "category:jaz", 45)
    with pytest.raises(KeyError):
        m.pole("mode:9999")
    with pytest.raises(ValueError):
        m.slerp(4, "category:jazz", 200)
    with pytest.raises(ValueError):
        m.slerp(4, np.ones(7), 30)


def test_small_bundle_projects_full_space_directions(m, small):
    d = m.pole("category:jazz")                                               # 32-d direction
    p = small.project(d)
    assert p.shape == (24,)
    np.testing.assert_allclose(unit(p), small.pole("category:jazz"), atol=1e-4)     # stored poles are projected likewise
    assert small.slerp(4, d, 40, k=5)                                         # full-space vector accepted, projected on the fly
    x = small.project(m.E[:3], is_direction=False)
    assert x.shape == (3, 24)


def test_find_fuzzy_and_normalised(m):
    t = m.vocab[10]
    exact = m.find(f"{t['artist']} - {t['title']}", 3)
    assert exact[0]["cid"] == 10 and exact[0]["score"] == pytest.approx(1.0, abs=1e-6)
    typo = m.find(t["artist"].replace("e", "3", 1).lower() + " " + t["title"].lower(), 5)
    assert 10 in [r["cid"] for r in typo]
    the = next(v for v in m.vocab if v["artist"].startswith("The "))
    got = m.find(the["artist"][4:] + " " + the["title"], 3)
    assert the["cid"] in [r["cid"] for r in got]                              # 'The' is normalised away
    rem = next(v for v in m.vocab if "Remastered" in v["title"])
    got = m.find(rem["artist"] + " - " + rem["title"].split(" - ")[0], 3)
    assert got[0]["cid"] == rem["cid"]                                        # remaster noise is normalised away
    assert m.find("", 5) == [] and m.find("zzzzqqqq", 5) is not None
    assert len(m.find("velvet", 4)) == 4


def test_resolve_errors_and_ambiguity(m):
    with pytest.raises(TrackNotFound):
        m.resolve(10_000)
    with pytest.raises(TrackNotFound) as ei:
        m.resolve("Dancing Moonlight")                                        # a title shared by several artists
    assert ei.value.suggestions and "closest" in str(ei.value)
    assert m.resolve(np.int64(7)) == 7 and m.resolve({"cid": 9}) == 9
    t = m.vocab[15]
    assert m.resolve(f"{t['artist']} - {t['title']}") == 15
    assert m.resolve(f"{t['artist'].upper()} - {t['title'].lower()}") == 15


def test_modes_api(m):
    modes = m.list_modes()
    assert {"mode_id", "label", "size", "coherence", "factor"} <= set(modes[0]) and "members" not in modes[0]
    mid = modes[0]["mode_id"]
    mem = m.mode_members(mid)
    assert len(mem) == modes[0]["size"]
    assert [r["score"] for r in mem] == sorted((r["score"] for r in mem), reverse=True)
    with pytest.raises(KeyError):
        m.mode_members(10_000)
    cm = m.closest_mode(mem[0]["cid"])
    assert cm["member"] is True or cm["mode_id"] != mid
    # closest_mode is the argmax over all mode poles
    i = 123
    c = m.mode_poles @ m.U[i]
    assert m.closest_mode(i)["mode_id"] == m.modes[int(np.argmax(c))]["mode_id"]
    # member of a mode -> describe lists it and the mode label is set
    member = mem[0]["cid"]
    d = m.describe(member)
    assert d["mode"] is not None and mid in [x["mode_id"] for x in d["modes"]]


def test_mode_helpers_match_definition(m):
    U = m.U
    mem = [m.modes[0]["members"], m.modes[1]["members"]]
    poles = mode_poles_from_members(U, mem)
    np.testing.assert_allclose(poles[0], unit(U[mem[0]].mean(0)), atol=1e-6)
    prim = primary_mode_assignment(U, mem, poles)
    assert prim.shape == (400,) and set(np.unique(prim)) <= {-1, 0, 1}
    assert (prim[mem[0]] >= 0).all()


def test_list_supervised_poles_and_poles(m):
    sp = m.list_supervised_poles()
    assert {"name", "kind", "n_pos", "n_neg"} <= set(sp[0])
    assert {p["kind"] for p in sp} >= {"category", "region", "era", "dimension"}
    groups = {p["group"] for p in m.list_poles()}
    assert groups == {"supervised", "mode", "factor"}
    np.testing.assert_allclose(np.linalg.norm(m.pole("factor:0")), 1.0, atol=1e-5)
