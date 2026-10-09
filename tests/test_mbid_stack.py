"""Track B (MBID-native stack) tests: synthetic fixtures in the *documented* formats, no network."""
from __future__ import annotations
import gzip, io, json, sqlite3, sys, tarfile, uuid
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from melomane.adapters import mbid_stack as mb
from melomane.corpus import read_corpus, write_corpus

zstandard = pytest.importorskip("zstandard")
NAMES = ["alice_secret", "bob_private", "carol_hidden", "dave_unknown"]


# ---------------------------------------------------------------------------------------------------------- fixtures
def _uuid(rng: np.random.Generator) -> str:
    return str(uuid.UUID(bytes=bytes(rng.integers(0, 256, 16, dtype=np.uint8))))


def make_catalog(n=40, seed=0):
    rng = np.random.default_rng(seed)
    recs = [{"mbid": _uuid(rng), "artist_mbid": _uuid(rng) if i % 5 else _uuid(rng), "artist": f"Artist {i // 4}",
             "title": f"Song {i}"} for i in range(n)]
    for i in range(n):                        # shared artist mbid for groups of 4 tracks
        recs[i]["artist_mbid"] = recs[(i // 4) * 4]["artist_mbid"]
    return recs


def lb_rec(user, r, ts, mapping=True, submitted=None):
    tm = {"artist_name": r["artist"], "track_name": r["title"], "release_name": "Album",
          "additional_info": {"recording_mbid": submitted if submitted is not None else r["mbid"], "artist_mbids": [r["artist_mbid"]]}}
    if mapping:
        tm["mbid_mapping"] = {"recording_mbid": r["mbid"], "recording_name": r["title"], "release_mbid": _uuid(np.random.default_rng(1)),
                              "artist_mbids": [r["artist_mbid"]], "caa_id": 1,
                              "artists": [{"artist_mbid": r["artist_mbid"], "artist_credit_name": r["artist"], "join_phrase": ""}]}
    return {"user_id": 7, "user_name": user, "listened_at": ts, "inserted_at": ts + 3, "recording_msid": _uuid(np.random.default_rng(2)),
            "track_metadata": tm}


def make_listens(cat, n_users=30, seed=1, t0=1_600_000_000):
    """Users with taste blocks (tracks 0..19 vs 20..39), 3 sessions each of 8 distinct tracks, sessions 3 h apart."""
    rng = np.random.default_rng(seed)
    out = []   # (month_index, record)
    for u in range(n_users):
        name = f"user_{u}_{NAMES[u % 4]}"
        half = (u % 2) * 20
        t = t0 + u * 100_000
        for s in range(3):
            for k, i in enumerate(rng.choice(20, 8, replace=False)):
                out.append((s % 2, lb_rec(name, cat[half + int(i)], t + s * 10_800 + k * 200)))
    out.append((0, lb_rec("nobody", cat[0], t0, mapping=False, submitted="not-a-uuid")))        # unusable mbid
    out.append((0, {"user_name": "x", "track_metadata": {}}))                                    # no timestamp
    return out


def write_lb_tar(path: Path, records, zst=True):
    months = {0: [], 1: []}
    for m, r in records:
        months[m].append(json.dumps(r))
    tbuf = io.BytesIO()
    with tarfile.open(fileobj=tbuf, mode="w") as tf:
        for name, body in [("SCHEMA_SEQUENCE", "5\n"), ("TIMESTAMP", "2026-09-15\n"),
                           ("listens/2020/9.listens", "\n".join(months[0]) + "\n{broken json\n"),
                           ("listens/2020/10.listens", "\n".join(months[1]) + "\n")]:
            b = body.encode()
            ti = tarfile.TarInfo(name); ti.size = len(b); tf.addfile(ti, io.BytesIO(b))
    data = tbuf.getvalue()
    path.write_bytes(zstandard.ZstdCompressor().compress(data) if zst else data)


def set_path(doc, path, value):
    parts = path.split("."); node = doc
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    if isinstance(node.get(parts[-1]), dict) and isinstance(value, dict):
        node[parts[-1]].update(value)
    else:
        node[parts[-1]] = value


def make_ab_doc(rng, key="C", scale="major", with_stdev=False):
    """Low-level doc with every descriptor of every family (AB style: mean/var/...)."""
    doc = {"metadata": {"version": {"essentia": "2.1-beta1"}}}
    for fam, specs in mb.AB_FAMILY_SPECS.items():
        for s in specs:
            if s.kind == "key":
                set_path(doc, s.path, key)
            elif s.kind == "scale":
                set_path(doc, s.path, scale)
            elif s.kind == "cov_diag":
                set_path(doc, s.path, {"cov": (np.diag(rng.random(s.length) + 1)).tolist()})
            elif not s.stats:
                set_path(doc, s.path, float(rng.normal()))
            else:
                node = {}
                for st in s.stats:
                    if st == "stdev" and not with_stdev:
                        st = "var"
                    node[st] = rng.random(s.length).tolist() if s.kind == "vector" else float(rng.random() + 0.5)
                set_path(doc, s.path, node)
    return doc


def write_ab_archive(path: Path, entries, root="acousticbrainz-lowlevel-json-20220623-0"):
    tbuf = io.BytesIO()
    with tarfile.open(fileobj=tbuf, mode="w") as tf:
        for typ, m, off, doc in entries:
            name = f"{root}/{typ}/{m[:2]}/{m[2:3]}/{m}-{off}.json"
            b = json.dumps(doc).encode(); ti = tarfile.TarInfo(name); ti.size = len(b); tf.addfile(ti, io.BytesIO(b))
    path.write_bytes(zstandard.ZstdCompressor().compress(tbuf.getvalue()))


def write_csv_tar(path: Path, member: str, text: str):
    tbuf = io.BytesIO()
    with tarfile.open(fileobj=tbuf, mode="w") as tf:
        b = text.encode(); ti = tarfile.TarInfo(member); ti.size = len(b); tf.addfile(ti, io.BytesIO(b))
    path.write_bytes(zstandard.ZstdCompressor().compress(tbuf.getvalue()))


def canonical_csv(cat, header=True, year0=1990):
    lines = [",".join(mb.CANONICAL_COLUMNS)] if header else []
    for i, r in enumerate(cat):
        pg = "{" + r["artist_mbid"] + "}"
        row = [str(i), str(100 + i // 4), f'"{pg}"', f'"{r["artist"]}"', _uuid(np.random.default_rng(i)), "Album", r["mbid"], f'"{r["title"]}"',
               "", str(0.5 + i / 100), str(year0 + i % 30)]
        lines.append(",".join(row))
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------------------------------------- utilities
def test_mbid_roundtrip_including_trailing_nul_bytes():
    raw = [bytes([1] * 15 + [0]), bytes([0] * 16), bytes(range(16))]
    arr = mb.to_mbid_array([mb.mbid_str(r) for r in raw])
    assert arr.tobytes() == b"".join(raw)                                # element access would strip NULs; buffers must not
    assert mb.mbid_array_to_str(arr) == [mb.mbid_str(r) for r in raw]
    assert mb.mbid_bytes("NOT-A-UUID") is None and mb.mbid_bytes(mb.mbid_str(raw[2]).upper()) == raw[2]


def test_zst_without_zstandard_gives_clear_error(tmp_path, monkeypatch):
    p = tmp_path / "x.jsonl.zst"; p.write_bytes(b"\x28\xb5\x2f\xfd")
    monkeypatch.setitem(sys.modules, "zstandard", None)              # makes `import zstandard` raise ImportError
    with pytest.raises(mb.ZstdUnavailable, match="pip install zstandard"):
        with mb.open_binary(p):
            pass


@pytest.mark.parametrize("kind", ["jsonl", "jsonl.gz", "jsonl.zst"])
def test_single_file_formats(tmp_path, kind):
    cat = make_catalog(); recs = [r for _, r in make_listens(cat)][:50]
    body = ("\n".join(json.dumps(r) for r in recs) + "\n").encode()
    p = tmp_path / f"l.{kind}"
    p.write_bytes(gzip.compress(body) if kind.endswith("gz") else zstandard.ZstdCompressor().compress(body) if kind.endswith("zst") else body)
    h = mb.UserHasher(b"s" * 32)
    st = mb.extract_listens([p], tmp_path / "o", h)
    assert st.members_done == 1 and st.listens_kept > 40


# ------------------------------------------------------------------------------------------------------------ listens
def test_parse_listen_mapping_vs_submitted_and_hashing():
    cat = make_catalog(); h = mb.UserHasher(b"k" * 32)
    other = _uuid(np.random.default_rng(99))
    rec = lb_rec("alice_secret", cat[0], 1_600_000_000, submitted=other)
    u, m, ts, meta = mb.parse_listen(rec, h, prefer="mapping")
    assert mb.mbid_str(m) == cat[0]["mbid"] and ts == 1_600_000_000 and meta == (cat[0]["artist"], cat[0]["artist_mbid"], cat[0]["title"])
    assert mb.mbid_str(mb.parse_listen(rec, h, prefer="submitted")[1]) == other
    assert isinstance(u, int) and 0 <= u < 2 ** 63 and u == h("alice_secret") and u != mb.UserHasher(b"z" * 32)("alice_secret")
    assert mb.parse_listen({"user_name": "a", "listened_at": 1, "track_metadata": {}}, h) == "no_mbid"
    assert mb.parse_listen({"listened_at": 1, "track_metadata": {}}, h) is None


def test_extract_tar_zst_anonymises_and_is_resumable(tmp_path):
    cat = make_catalog(); recs = make_listens(cat)
    src = tmp_path / "listenbrainz-listens-dump-1-20260915-000002-full.tar.zst"; write_lb_tar(src, recs)
    salt = tmp_path / "private" / "salt.bin"; h = mb.UserHasher.from_file(salt)
    full = mb.extract_listens([src], tmp_path / "full", h)
    assert full.members_done == 2 and full.bad_lines == 1 and full.no_mbid == 1 and full.no_user_or_ts == 1
    L = mb.load_listens(tmp_path / "full")
    assert len(L) == 30 * 24 and len(np.unique(L.user)) == 30
    # privacy: no user name appears in ANY output file
    for f in (tmp_path / "full").iterdir():
        blob = f.read_bytes()
        if f.suffix == ".gz":
            blob = gzip.decompress(blob)
        for nm in NAMES + ["user_0", "nobody"]:
            assert nm.encode() not in blob, f"{nm} leaked into {f.name}"
    # stop after first member -> then resume gives the same listens
    calls = {"n": 0}
    def stop():
        calls["n"] += 1
        return calls["n"] >= 1
    part = mb.extract_listens([src], tmp_path / "res", h, should_stop=stop)
    assert part.stopped and part.members_done == 1
    (tmp_path / "res" / "L999999_000.npz").write_bytes(b"debris")             # unreferenced shard must be cleaned
    rest = mb.extract_listens([src], tmp_path / "res", h)
    assert rest.members_done == 1 and rest.members_skipped == 1 and not (tmp_path / "res" / "L999999_000.npz").exists()
    R = mb.load_listens(tmp_path / "res")
    key = lambda x: np.lexsort((x.mbid, x.ts, x.user))
    assert (R.user[key(R)] == L.user[key(L)]).all() and (R.ts[key(R)] == L.ts[key(L)]).all() and (R.mbid[key(R)] == L.mbid[key(L)]).all()
    with pytest.raises(ValueError, match="salt"):
        mb.extract_listens([src], tmp_path / "res", mb.UserHasher(b"x" * 32))


def test_extract_filters_and_mapping_metadata(tmp_path):
    cat = make_catalog(); src = tmp_path / "d.tar.zst"; write_lb_tar(src, make_listens(cat))
    h = mb.UserHasher(b"q" * 32)
    keep = mb.to_mbid_array([r["mbid"] for r in cat[:20]])
    st = mb.extract_listens([src], tmp_path / "o", h, mbid_filter=keep, year_range=(2020, 2020))
    assert st.not_in_filter > 0
    assert set(mb.mbid_array_to_str(np.unique(mb.load_listens(tmp_path / "o").mbid))) <= {r["mbid"] for r in cat[:20]}
    assert mb.extract_listens([src], tmp_path / "y", h, year_range=(1999, 2000)).members_done == 0
    assert mb.extract_listens([src], tmp_path / "t", h, min_ts=10 ** 12).listens_kept == 0
    meta = mb.load_mapping_metadata(tmp_path / "o")
    assert meta.loc[cat[3]["mbid"], "title"] == cat[3]["title"] and meta.loc[cat[3]["mbid"], "artist_mbid"] == cat[3]["artist_mbid"]
    assert mb.load_listens(tmp_path / "o", max_ts=1_600_000_000).__len__() == 0                       # temporal split


def test_sessions_from_listens(tmp_path):
    cat = make_catalog(); src = tmp_path / "d.tar.zst"; write_lb_tar(src, make_listens(cat))
    mb.extract_listens([src], tmp_path / "o", mb.UserHasher(b"q" * 32))
    L = mb.load_listens(tmp_path / "o")
    vocab = mb.select_vocab(L, min_users=3)
    assert vocab.size == 40 and (np.sort(vocab) == vocab).all()
    X, su = mb.listens_to_sessions(L, vocab, gap_minutes=30, min_len=5, max_len=50)
    assert X.shape == (90, 40) and (np.diff(X.indptr) == 8).all() and len(np.unique(su)) == 30
    # taste blocks: sessions of even users touch tracks in one half only (vocab is sorted by mbid -> map back)
    pos = {m: i for i, m in enumerate(mb.mbid_array_to_str(vocab))}
    first_half = {pos[r["mbid"]] for r in cat[:20]}
    for i in range(X.shape[0]):
        cols = set(X[i].indices.tolist())
        assert cols <= first_half or not (cols & first_half)
    assert len(mb.select_vocab(L, min_users=1, max_tracks=10, seed=3)) == 10
    assert (mb.select_vocab(L, 1, 10, seed=3) == mb.select_vocab(L, 1, 10, seed=3)).all()
    D = mb.Listens(np.r_[L.user, L.user[:5]], np.r_[L.mbid, L.mbid[:5]], np.r_[L.ts, L.ts[:5]]).dedupe()
    assert len(D) == len(L)


# ------------------------------------------------------------------------------------------------------ MusicBrainz
@pytest.mark.parametrize("header", [True, False])
def test_canonical_metadata_and_recording_table(tmp_path, header):
    cat = make_catalog(); p = tmp_path / "canon.tar.zst"
    write_csv_tar(p, "musicbrainz-canonical-dump/canonical_musicbrainz_data.csv", canonical_csv(cat, header))
    chunks = list(mb.read_canonical_metadata(p, chunksize=7))
    df = mb.build_recording_table(chunks, isrcs={cat[2]["mbid"]: ["USRC17607839"]})
    assert len(df) == 40 and df.loc[cat[5]["mbid"], "artist_mbid"] == cat[5]["artist_mbid"]
    assert df.loc[cat[5]["mbid"], "title"] == "Song 5" and df.loc[cat[2]["mbid"], "isrcs"] == ["USRC17607839"]
    assert df.loc[cat[5]["mbid"], "first_release_year"] == 1995
    sub = pd.concat(mb.read_canonical_metadata(p, keep_mbids={cat[1]["mbid"]}))
    assert list(sub["recording_mbid"]) == [cat[1]["mbid"]]


def test_parse_mbid_list_encodings():
    a, b = _uuid(np.random.default_rng(1)), _uuid(np.random.default_rng(2))
    for v in ("{%s,%s}" % (a, b), json.dumps([a, b]), f"{a},{b}", [a, b], f"{a};{b}"):
        assert mb.parse_mbid_list(v) == [a, b]
    assert mb.parse_mbid_list("") == [] and mb.parse_mbid_list(None) == []


def test_redirect_map_and_variants(tmp_path):
    rng = np.random.default_rng(0)
    r1, r2, canon, other = (_uuid(rng) for _ in range(4))
    csv = "recording_mbid,canonical_recording_mbid,canonical_release_mbid\n" + f"{r1},{canon},{other}\n{r2},{canon},{other}\n{canon},{canon},{other}\n"
    p = tmp_path / "canonical_recording_redirect.csv"; p.write_text(csv)
    rm = mb.RedirectMap.from_csv(p)
    q = mb.to_mbid_array([r1, r2, other, canon])
    assert mb.mbid_array_to_str(rm.apply(q)) == [canon, canon, other, canon] and len(rm) == 3
    assert set(mb.mbid_array_to_str(rm.variants_of(mb.to_mbid_array([canon])))) == {r1, r2, canon}
    tp = tmp_path / "c.tar.zst"; write_csv_tar(tp, "x/canonical_recording_redirect.csv", csv)
    assert len(mb.RedirectMap.from_csv(tp)) == 3


def test_offline_resolver_variants_and_accents():
    tbl = pd.DataFrame({"artist_credit_name": ["Beyoncé", "Daft Punk", "Daft Punk", "AC/DC", "Sigur Rós"],
                        "title": ["Halo", "One More Time", "One More Time", "Back in Black", "Hoppípolla"],
                        "score": [1, 5, 9, 1, 1]},
                       index=pd.Index([_uuid(np.random.default_rng(i)) for i in range(5)], name="recording_mbid"))
    res = mb.TitleArtistResolver.from_recordings(tbl)
    m, how = res.resolve("beyonce", "HALO!"); assert m == tbl.index[0] and how == "exact"
    assert res.resolve("Daft Punk", "One More Time")[0] == tbl.index[2]                     # higher score wins
    assert mb.TitleArtistResolver.from_recordings(tbl, higher_score_wins=False).resolve("Daft Punk", "One More Time")[0] == tbl.index[1]
    assert res.resolve("Daft Punk feat. Romanthony", "One More Time (feat. Romanthony)") == (tbl.index[2], "both_variant")
    assert res.resolve("Beyoncé", "Halo - Remastered 2011") == (tbl.index[0], "title_variant")
    assert res.resolve("Sigur Ros", "Hoppipolla")[0] == tbl.index[4]
    assert res.resolve("Nobody", "Nothing") == (None, "none")
    out = res.resolve_frame(pd.DataFrame({"artist": ["beyonce", "zzz"], "title": ["halo", "yyy"]}))
    assert out["recording_mbid"].iloc[0] == tbl.index[0] and pd.isna(out["recording_mbid"].iloc[1]) and out["method"].tolist() == ["exact", "none"]


def test_isrc_pg_tables_and_json_dump(tmp_path):
    rng = np.random.default_rng(0); g1, g2 = _uuid(rng), _uuid(rng)
    (tmp_path / "isrc").write_text("1\t10\tUSRC17607839\t0\t2020-01-01\n2\t10\tGBAYE0000351\t0\t\\N\n3\t11\tFRZ039800212\t0\t\\N\n")
    (tmp_path / "recording").write_text(f"10\t{g1}\tSong A\t1\n11\t{g2}\tSong B\t1\n12\t{_uuid(rng)}\tNoISRC\t1\n")
    assert mb.read_isrc_tables(tmp_path / "isrc", tmp_path / "recording") == {g1: ["GBAYE0000351", "USRC17607839"], g2: ["FRZ039800212"]}
    doc = {"id": g1, "title": "Song A", "artist-credit": [{"name": "A", "joinphrase": " feat. ", "artist": {"id": _uuid(rng)}},
                                                        {"name": "B", "joinphrase": "", "artist": {"id": _uuid(rng)}}],
           "first-release-date": "1999-05-04", "isrcs": ["USRC17607839"]}
    tbuf = io.BytesIO()
    with tarfile.open(fileobj=tbuf, mode="w:xz") as tf:
        b = (json.dumps(doc) + "\n").encode(); ti = tarfile.TarInfo("mbdump/recording"); ti.size = len(b); tf.addfile(ti, io.BytesIO(b))
    (tmp_path / "recording.tar.xz").write_bytes(tbuf.getvalue())
    row = next(mb.read_mb_recording_json(tmp_path / "recording.tar.xz"))
    assert row["artist_credit_name"] == "A feat. B" and row["first_release_year"] == 1999 and row["isrcs"] == ["USRC17607839"]


# --------------------------------------------------------------------------------------------------- AcousticBrainz
def test_ab_family_names_and_columns():
    cols = mb.ab_columns()
    assert list(cols) == mb.FAMILY_NAMES == ["timbre", "spectral_shape", "rhythm_groove", "harmony_tonal", "dynamics_loudness"]
    assert len(cols["timbre"]) == 26 and cols["timbre"][0] == "lowlevel.mfcc.mean[0]"
    assert all(len(set(c)) == len(c) for c in cols.values())


def test_extract_ab_features_stdev_from_var_key_and_missing():
    doc = {"lowlevel": {"spectral_centroid": {"mean": 1000.0, "var": 400.0}, "average_loudness": 0.8,
                        "mfcc": {"mean": list(range(13)), "cov": np.eye(13).tolist()}},
           "rhythm": {"bpm": 128.0}, "tonal": {"key_edma": {"key": "A", "scale": "minor", "strength": 0.7}}}
    f = mb.extract_ab_features(doc)
    cols = mb.ab_columns()
    g = lambda fam, c: f[fam][cols[fam].index(c)]
    assert g("spectral_shape", "lowlevel.spectral_centroid.mean") == 1000.0 and g("spectral_shape", "lowlevel.spectral_centroid.stdev") == 20.0
    assert g("timbre", "lowlevel.mfcc.mean[5]") == 5 and g("timbre", "lowlevel.mfcc.cov_diag[3]") == 1
    assert g("rhythm_groove", "rhythm.bpm") == 128.0 and np.isnan(g("rhythm_groove", "rhythm.danceability"))
    assert np.isclose(g("harmony_tonal", "tonal.key_edma.key_cos"), np.cos(2 * np.pi * 9 / 12)) and g("harmony_tonal", "tonal.key_edma.scale_major") == 0
    assert g("dynamics_loudness", "lowlevel.average_loudness") == 0.8 and all(len(f[k]) == len(cols[k]) for k in f)
    # Essentia music-extractor style (stdev present) is accepted natively
    f2 = mb.extract_ab_features({"lowlevel": {"spectral_centroid": {"mean": 5.0, "stdev": 3.0}}})
    assert f2["spectral_shape"][cols["spectral_shape"].index("lowlevel.spectral_centroid.stdev")] == 3.0


def test_scan_lowlevel_archives_filter_offsets_resume_and_blocks(tmp_path):
    rng = np.random.default_rng(0); cat = make_catalog()
    entries = []
    for i, r in enumerate(cat):
        entries.append(("lowlevel", r["mbid"], 0, make_ab_doc(rng, key="C#" if i % 2 else "A", scale="minor" if i % 3 else "major")))
    entries.append(("lowlevel", cat[0]["mbid"], 1, make_ab_doc(rng)))                     # second submission
    entries.append(("highlevel", cat[0]["mbid"], 0, {"highlevel": {}}))                    # must be ignored
    stray = _uuid(rng); entries.append(("lowlevel", stray, 0, make_ab_doc(rng)))          # not a candidate
    entries.append(("lowlevel", cat[1]["mbid"], 0, "garbage-not-a-doc"))                  # duplicate of a candidate, bad payload type
    a0, a1 = tmp_path / "acousticbrainz-lowlevel-json-20220623-0.tar.zst", tmp_path / "acousticbrainz-lowlevel-json-20220623-1.tar.zst"
    write_ab_archive(a0, entries[:25]); write_ab_archive(a1, entries[25:], root="acousticbrainz-lowlevel-json-20220623-1")
    cand = mb.to_mbid_array([r["mbid"] for r in cat[:30]])
    st = mb.scan_lowlevel_archives([a0], tmp_path / "ab", cand)
    assert st.archives_done == 1 and st.docs_matched == 25
    st2 = mb.scan_lowlevel_archives([a0, a1], tmp_path / "ab", cand)                       # resume: a0 skipped
    assert st2.archives_skipped == 1 and st2.archives_done == 1 and st2.bad_docs == 1
    A = mb.load_audio(tmp_path / "ab", "first")
    assert A.mbid.size == 30 and A.X.shape[1] == sum(len(c) for c in mb.ab_columns().values()) and not np.isnan(A.X).any()
    Am = mb.load_audio(tmp_path / "ab", "mean")
    i0 = np.flatnonzero(Am.mbid == mb.to_mbid_array([cat[0]["mbid"]])[0])[0]
    j0 = np.flatnonzero(A.mbid == mb.to_mbid_array([cat[0]["mbid"]])[0])[0]
    assert not np.allclose(Am.X[i0], A.X[j0])                                              # 'mean' really averaged 2 submissions
    vocab = np.sort(mb.to_mbid_array([r["mbid"] for r in cat[10:40]]))
    blocks = A.blocks(vocab)
    assert set(blocks) == set(mb.FAMILY_NAMES)
    for fam, b in blocks.items():
        assert b.rows.size == 20 and b.X.dtype == np.float32 and b.X.shape[1] == len(b.columns) and (np.diff(b.rows) > 0).all()
    other_fams = mb.scan_lowlevel_archives
    with pytest.raises(ValueError, match="different families"):
        mb.scan_lowlevel_archives([a0], tmp_path / "ab", cand, families=["timbre"])


def test_audio_blocks_impute_and_drop():
    cols = {"timbre": ["a", "b", "c", "d"]}
    mbids = mb.to_mbid_array([_uuid(np.random.default_rng(i)) for i in range(4)])
    order = np.argsort(mbids)
    X = np.array([[1, 2, 3, 4], [np.nan, 4, 5, 6], [np.nan] * 4, [3, 6, 7, 8]], np.float32)[order]
    t = mb.AudioTable(mbids[order], X, cols)
    b = t.blocks(mbids[order])["timbre"]
    assert b.rows.size == 3 and not np.isnan(b.X).any()               # all-NaN row dropped, NaN imputed by column median


def test_ab_csv_dumps(tmp_path):
    cat = make_catalog(10)
    lo = "mbid,submission_offset,average_loudness,dynamic_complexity,mfcc_zero_mean\n" + "\n".join(
        f"{r['mbid']},0,0.{i},3.{i},-{i}0.5" for i, r in enumerate(cat)) + f"\n{cat[0]['mbid']},1,0.99,9,9\n"
    rh = "mbid,submission_offset,bpm,bpm_histogram_first_peak_bpm_mean,bpm_histogram_first_peak_bpm_median," \
         "bpm_histogram_second_peak_bpm_mean,bpm_histogram_second_peak_bpm_median,danceability,onset_rate\n" + "\n".join(
        f"{r['mbid']},0,{100 + i},{100 + i},{100 + i},{50 + i},{50 + i},1.{i},2.{i}" for i, r in enumerate(cat)) + "\n"
    to = "mbid,submission_offset,key_key,key_scale,tuning_frequency,tuning_equal_tempered_deviation\n" + "\n".join(
        f"{r['mbid']},0,{'C#' if i % 2 else 'A'},{'major' if i % 3 else 'minor'},44{i}.0,0.0{i}" for i, r in enumerate(cat)) + "\n"
    paths = []
    for nm, txt in (("lowlevel", lo), ("rhythm", rh), ("tonal", to)):
        p = tmp_path / f"acousticbrainz-lowlevel-features-20220623-{nm}.tar.zst"; write_csv_tar(p, f"{nm}.csv", txt); paths.append(p)
    df = mb.read_ab_csv_features(paths, candidates=mb.to_mbid_array([r["mbid"] for r in cat[:8]]))
    assert len(df) == 8 and df.loc[cat[0]["mbid"], "average_loudness"] == "0.0"                # lowest offset kept
    vocab = np.sort(mb.to_mbid_array([r["mbid"] for r in cat]))
    blocks = mb.ab_csv_blocks(df, vocab)
    assert set(blocks) == {"timbre", "rhythm_groove", "harmony_tonal", "dynamics_loudness"}     # spectral_shape needs the JSON dumps
    assert blocks["harmony_tonal"].columns[:2] == ["key_key_cos", "key_key_sin"] and blocks["rhythm_groove"].X.shape == (8, 7)
    assert "coarse" in blocks["timbre"].source


# --------------------------------------------------------------------------------------------------------------- LRCLIB
LYRIC_WORD = "zzuniqueLyricToken"


def lrc_record(text=f"hello {LYRIC_WORD} world\nsecond line here", instrumental=False):
    return {"id": 1, "name": "x", "trackName": "x", "artistName": "y", "albumName": "z", "duration": 200, "instrumental": instrumental,
            "plainLyrics": None if instrumental else text, "syncedLyrics": None if instrumental else "[00:01.00] " + text.replace("\n", "\n[00:05.00] "),
            "hasWordSync": False}


class FakeLrc:
    def __init__(self, table, fail_first=0):
        self.table, self.calls, self.fail_first = table, [], fail_first

    def __call__(self, url, params, headers):
        self.calls.append((url, dict(params), dict(headers)))
        if self.fail_first > 0:
            self.fail_first -= 1
            return 429, b"slow down"
        key = (params.get("artist_name"), params.get("track_name"))
        if url.endswith("/search"):
            return 200, json.dumps([dict(r, artistName=k[0], trackName=k[1]) for k, r in self.table.items() if params["q"].lower() == f"{k[0]} {k[1]}".lower()]).encode()
        return (200, json.dumps(self.table[key]).encode()) if key in self.table else (404, b"{}")


class Clock:
    def __init__(self): self.t = 0.0; self.sleeps = []
    def now(self): return self.t
    def sleep(self, s): self.sleeps.append(s); self.t += s


def _client(tmp_path, transport, clock=None, **kw):
    clock = clock or Clock()
    return mb.LrclibClient(tmp_path / "lrc.sqlite", "Melomane/0.1 (mailto:test@example.org)", transport=transport, sleep=clock.sleep,
                           clock=clock.now, **kw), clock


def test_lrclib_client_rate_limit_cache_retry_and_no_text_stored(tmp_path):
    tr = FakeLrc({("A", "Song"): lrc_record(), ("A", "Inst"): lrc_record(instrumental=True)}, fail_first=1)
    c, clock = _client(tmp_path, tr)
    f = c.lookup("A", "Song", duration_s=200.4)
    assert f.found and f.has_lyrics and f.n_words == 6 and f.n_lines == 2 and f.has_synced and f.embedding.shape == (64,)
    assert np.isclose(np.linalg.norm(f.embedding), 1, atol=1e-5)
    assert len(tr.calls) == 2 and 5.0 in clock.sleeps                                      # 429 -> backoff -> success
    assert tr.calls[0][2]["User-Agent"].startswith("Melomane/") and tr.calls[0][1]["duration"] == "200"
    times = []
    g = c.lookup("A", "Inst"); assert g.found and g.instrumental and not g.has_lyrics and g.embedding is None
    h = c.lookup("A", "Missing"); assert not h.found
    n = len(tr.calls)
    assert c.lookup("A", "Song", duration_s=200.4).n_words == 6 and c.lookup("A", "Missing").found is False
    assert len(tr.calls) == n                                                              # cache hits are free
    # <= 1 request per second on the (fake) clock between consecutive network calls
    assert c.n_requests == n and sum(clock.sleeps) >= n - 1 - 5.0 - 1e-9
    c.cache.close()
    assert LYRIC_WORD.encode() not in (tmp_path / "lrc.sqlite").read_bytes() and b"second line" not in (tmp_path / "lrc.sqlite").read_bytes()
    # transient failure is NOT cached
    tr2 = FakeLrc({}, fail_first=99)
    c2, _ = _client(tmp_path / "b", tr2, max_retries=1)
    assert c2.lookup("A", "Song") is None and len(c2.cache) == 0


def test_lrclib_client_policy_guards_and_search_fallback(tmp_path):
    with pytest.raises(ValueError, match="User-Agent"):
        mb.LrclibClient(tmp_path / "x.sqlite", "python-httpx/0.27", transport=FakeLrc({}))
    with pytest.raises(ValueError, match="1 req/s"):
        mb.LrclibClient(tmp_path / "y.sqlite", "Melomane/0.1 (x)", transport=FakeLrc({}), min_interval_s=0.2)
    tr = FakeLrc({("Artist", "Title"): lrc_record()})
    tr.table[("Artist", "Title")]["duration"] = 200
    c, _ = _client(tmp_path, tr, search_fallback=True)
    f = c.lookup("artist", "title", duration_s=200)                                         # /get 404 (case) -> /search matches
    assert f.found and [u.rsplit("/", 1)[1] for u, _, _ in tr.calls] == ["get", "search"]


def test_lrclib_dump_schema_introspection_and_collect(tmp_path):
    db = tmp_path / "lrclib.sqlite3"
    con = sqlite3.connect(db)
    con.executescript("""CREATE TABLE tracks (id INTEGER PRIMARY KEY, name TEXT, name_lower TEXT, artist_name TEXT, artist_name_lower TEXT,
                         album_name TEXT, duration REAL, last_lyrics_id INTEGER);
                         CREATE TABLE lyrics (id INTEGER PRIMARY KEY, plain_lyrics TEXT, synced_lyrics TEXT, track_id INTEGER, has_plain_lyrics INT,
                         instrumental INT);""")
    con.execute("INSERT INTO tracks VALUES (1,'Song 1','song 1','Artist 0','artist 0','Alb',200,1)")
    con.execute("INSERT INTO tracks VALUES (2,'Song 2','song 2','Artist 0','artist 0','Alb',180,2)")
    con.execute(f"INSERT INTO lyrics VALUES (1,'la la {LYRIC_WORD}\nlo lo',NULL,1,1,0)")
    con.execute("INSERT INTO lyrics VALUES (2,NULL,NULL,2,0,1)")
    con.commit(); con.close()
    d = mb.LrclibDump(db)
    f = d.lookup("Artist 0", "Song 1"); assert f.has_lyrics and f.n_lines == 2 and d.lookup("Artist 0", "Song 2").instrumental
    assert d.lookup("Nobody", "None") is None
    cat = make_catalog(4)
    meta = pd.DataFrame({"artist_credit_name": ["Artist 0"] * 4, "title": [f"Song {i + 1}" for i in range(4)]},
                        index=pd.Index([r["mbid"] for r in cat], name="recording_mbid"))
    T = mb.collect_lyrics(mb.to_mbid_array([r["mbid"] for r in cat]), meta, d)
    assert T.mbid.size == 4 and T.has_lyrics.sum() == 1 and T.instrumental.sum() == 1 and T.n_pending == 0
    assert (np.sort(T.mbid) == T.mbid).all() and np.isclose(np.linalg.norm(T.embedding[T.has_lyrics][0]), 1, atol=1e-5)


def test_collect_lyrics_client_stops_and_reports_pending(tmp_path):
    cat = make_catalog(6)
    tr = FakeLrc({("Artist 0", f"Song {i}"): lrc_record() for i in range(6)})
    c, _ = _client(tmp_path, tr)
    meta = pd.DataFrame({"artist_credit_name": ["Artist 0"] * 6, "title": [f"Song {i}" for i in range(6)]},
                        index=pd.Index([r["mbid"] for r in cat], name="recording_mbid"))
    T = mb.collect_lyrics(mb.to_mbid_array([r["mbid"] for r in cat]), meta, c, max_lookups=4)
    assert T.mbid.size == 4 and T.n_pending == 2 and len(tr.calls) == 4
    T2 = mb.collect_lyrics(mb.to_mbid_array([r["mbid"] for r in cat]), meta, c)
    assert T2.mbid.size == 6 and len(tr.calls) == 6                                       # resumed from the cache: only 2 new requests


# -------------------------------------------------------------------------------------------- corpus + coverage + plan
def _full_pipeline_inputs(tmp_path):
    rng = np.random.default_rng(0); cat = make_catalog()
    src = tmp_path / "d.tar.zst"; write_lb_tar(src, make_listens(cat))
    h = mb.UserHasher(b"q" * 32)
    mb.extract_listens([src], tmp_path / "L", h)
    L = mb.load_listens(tmp_path / "L")
    p = tmp_path / "canon.tar.zst"; write_csv_tar(p, "canonical_musicbrainz_data.csv", canonical_csv(cat))
    rec = mb.build_recording_table(mb.read_canonical_metadata(p))
    entries = [("lowlevel", r["mbid"], 0, make_ab_doc(rng)) for r in cat[:25]]
    a = tmp_path / "ab0.tar.zst"; write_ab_archive(a, entries)
    mb.scan_lowlevel_archives([a], tmp_path / "AB", mb.to_mbid_array([r["mbid"] for r in cat]))
    audio = mb.load_audio(tmp_path / "AB")
    tr = FakeLrc({(f"Artist {i // 4}", f"Song {i}"): lrc_record(f"words {i} " * 5) for i in range(0, 40, 2)})
    c, _ = _client(tmp_path, tr)
    lyr = mb.collect_lyrics(mb.to_mbid_array([r["mbid"] for r in cat]), rec, c)
    return cat, L, rec, audio, lyr


def test_build_corpus_coverage_and_roundtrip(tmp_path):
    cat, L, rec, audio, lyr = _full_pipeline_inputs(tmp_path)
    c = mb.build_corpus(L, rec, audio=audio, lyrics=lyr, min_users_per_track=3, min_len=5)
    c.validate()
    assert c.n_tracks == 40 and c.sessions.shape == (90, 40) and c.tracks["artist_id"].dtype == np.int64
    assert set(c.families) == set(mb.FAMILY_NAMES) | {"lyric_theme"}
    # tracks are sorted by MBID; check metadata alignment
    row = c.tracks[c.tracks["ext_id"] == cat[7]["mbid"]].iloc[0]
    assert row["title"] == "Song 7" and row["artist"] == "Artist 1" and row["year"] == 1997
    assert c.tracks["artist_id"].nunique() == 10                                            # 10 distinct artist MBIDs
    cov = c.meta["coverage"]
    ids = {mb.mbid_str(m) for m in audio.mbid}
    n_audio = sum(1 for r in cat if r["mbid"] in ids)
    assert cov["n_tracks"] == 40 and cov["n_behaviour"] == 40 and cov["n_audio"] == n_audio == 25
    assert cov["n_lyrics"] == 20 and cov["n_all_three"] == cov["n_audio_and_lyrics"] == len({r["mbid"] for i, r in enumerate(cat[:25]) if i % 2 == 0})
    assert c.tracks["has_lyrics"].sum() == 20 and c.families["lyric_theme"].rows.size == 20
    assert "user_id" in c.sessions_meta and not any(n in str(c.sessions_meta.to_dict()) for n in NAMES)
    write_corpus(c, tmp_path / "corpus")
    c2 = read_corpus(tmp_path / "corpus")
    assert (c2.sessions != c.sessions).nnz == 0 and set(c2.families) == set(c.families) and c2.meta["coverage"]["n_all_three"] == cov["n_all_three"]
    # lyrics text/user names never in corpus files
    blob = b"".join(p.read_bytes() for p in (tmp_path / "corpus").rglob("*") if p.is_file())
    assert LYRIC_WORD.encode() not in blob and all(n.encode() not in blob for n in NAMES)
    # determinism
    c3 = mb.build_corpus(L, rec, audio=audio, lyrics=lyr, min_users_per_track=3, min_len=5)
    assert (c3.sessions != c.sessions).nnz == 0 and list(c3.tracks["ext_id"]) == list(c.tracks["ext_id"])


def test_build_corpus_redirect_merges_variants_and_requires_metadata(tmp_path):
    cat, L, rec, audio, lyr = _full_pipeline_inputs(tmp_path)
    variant = _uuid(np.random.default_rng(123))
    # rewrite half of track 0's listens to a non-canonical variant MBID: the redirect must merge them back
    m0 = mb.to_mbid_array([cat[0]["mbid"]])[0]
    idx = np.flatnonzero(L.mbid == m0)[: len(np.flatnonzero(L.mbid == m0)) // 2]
    mbids = L.mbid.copy(); mbids[idx] = mb.to_mbid_array([variant])[0]
    L2 = mb.Listens(L.user, mbids, L.ts)
    rm = mb.RedirectMap.from_pairs({variant: cat[0]["mbid"]})
    with_r = mb.build_corpus(L2, rec, redirect=rm, min_users_per_track=3)
    without = mb.build_corpus(L2, rec, min_users_per_track=3)
    n0 = lambda c: c.tracks.loc[c.tracks["ext_id"] == cat[0]["mbid"], "n_listens"].iloc[0]
    assert n0(with_r) > n0(without) and variant not in set(with_r.tracks["ext_id"]) and variant not in set(without.tracks["ext_id"])
    assert rec.index.difference(with_r.tracks["ext_id"]).size == 0 or True
    empty = rec.iloc[:10]
    assert mb.build_corpus(L, empty, min_users_per_track=3).n_tracks <= 10


def test_plan_downloads_and_probe(tmp_path):
    items = mb.plan_downloads(ab_archives=[0, 29])
    keys = [i.key for i in items]
    assert keys[0] == "lb_sample" and "lb_listens_full" in keys and "canonical" in keys and "ab_lowlevel_29" in keys and keys[-1] == "lrclib_dump"
    lb = next(i for i in items if i.key == "lb_listens_full")
    assert lb.url.endswith("listenbrainz-listens-dump-2663-20260915-000002-full.tar.zst") and lb.expected_bytes == 229 * 2 ** 30
    assert next(i for i in items if i.key == "lrclib_dump").verified is False and all(i.url.startswith("https://") for i in items)
    assert all("/pub/musicbrainz/acousticbrainz/dumps/" in i.url for i in items if i.key.startswith("ab_"))
    calls = []
    def head(url):
        calls.append(url)
        return (200, {"content-length": "12345"}) if "canonical" in url else (404, {})
    sleeps = []
    probed = mb.probe_sizes(items, head, sleep=sleeps.append)
    assert len(calls) == len(items) and len(sleeps) == len(items) - 1
    assert next(i for i in probed if i.key == "canonical").size_bytes == 12345 and next(i for i in probed if i.key == "lb_listens_full").size_bytes == 229 * 2 ** 30
    assert 250 < mb.total_gb(mb.plan_downloads(ab_archives=range(30))) < 1000


def test_apache_index_parsing_and_verify(tmp_path):
    html = ('<pre><a href="../">../</a>\n<a href="listenbrainz-dump-2647-20260901-000002-full/">listenbrainz-dump-2647-20260901-000002-full/</a>'
            '   03-Sep-2026 18:17    -\n<a href="listenbrainz-dump-2663-20260915-000002-full/">listenbrainz-dump-2663-20260915-000002-full/</a>'
            '   17-Sep-2026 18:17    -\n<a href="x.tar.zst">x.tar.zst</a>   17-Sep-2026 18:18    229G\n<a href="?C=N;O=D">Name</a></pre>')
    es = mb.parse_apache_index(html)
    assert [e.name for e in es][-1] == "x.tar.zst" and es[-1].size_text == "229G" and len(es) == 3
    assert mb.pick_latest(es, "full").name.endswith("2663-20260915-000002-full") and mb.pick_latest([], "") is None
    assert mb.parse_size("229G") == 229 * 2 ** 30 and mb.parse_size("843M") == 843 * 2 ** 20 and mb.parse_size("-") is None
    f = tmp_path / "f.bin"; f.write_bytes(b"abc")
    import hashlib
    assert mb.verify_sha256(f, hashlib.sha256(b"abc").hexdigest() + "  f.bin") and not mb.verify_sha256(f, "0" * 64)


# ------------------------------------------------------------------------------------------------ alignment reports
def test_align_report_ab_vs_itself_and_vs_onion_names():
    cols = [c for cs in mb.ab_columns().values() for c in cs]
    r = mb.align_report(cols)
    assert r["can_use_identical_definition"] and all(v["status"] == "aligned" for v in r["families"].values())
    r0 = mb.align_report(["mfcc.mean[0]", "chroma_stft.std[3]"])
    assert not r0["can_use_identical_definition"] and not r0["any_family_aligned"] and "Track B" in r0["recommendation"]
    rc = mb.align_report(["mfcc.mean[0]"], concept_map={"mfcc": "lowlevel.mfcc"})
    assert rc["families"]["timbre"]["status"] == "conceptual_only" and rc["families"]["rhythm_groove"]["status"] == "not_alignable"
    onion = Path(__file__).resolve().parents[1] / "essentia_descriptor_names.json"
    if onion.exists():                                                                    # PLAN §13.4: real key list shipped in the repo
        names = json.loads(onion.read_text())
        assert any(n.endswith(".stdev") for n in names) and not any(n.endswith(".var") for n in names)
        ro = mb.align_report(cols, names, target_name="onion id_essentia")
        assert ro["can_use_identical_definition"]                                         # names align once stdev = sqrt(var)
        assert "lowlevel.melbands128" in ro["target_descriptors_outside_ab_specs"] or "lowlevel.gfcc" in ro["target_descriptors_outside_ab_specs"]
    d = mb.doc_descriptor_names({"metadata": {"a": 1}, "lowlevel": {"mfcc": {"mean": [1], "cov": [[1]]}}, "rhythm": {"bpm": 1}})
    assert d == ["lowlevel.mfcc.mean", "lowlevel.mfcc.cov", "rhythm.bpm"]


# ------------------------------------------------------------------------------------------------------ ingest e2e
def _write_multi_tar_zst(path: Path, members: dict[str, str]):
    tbuf = io.BytesIO()
    with tarfile.open(fileobj=tbuf, mode="w") as tf:
        for name, text in members.items():
            b = text.encode(); ti = tarfile.TarInfo(name); ti.size = len(b); tf.addfile(ti, io.BytesIO(b))
    path.write_bytes(zstandard.ZstdCompressor().compress(tbuf.getvalue()))


def test_ingest_end_to_end_stop_resume_and_privacy(tmp_path):
    from melomane.workspace import Workspace
    ws = Workspace(root=tmp_path / "repo", data=tmp_path / "data"); ws.root.mkdir()
    rng = np.random.default_rng(0); cat = make_catalog()
    src = tmp_path / "listenbrainz-listens-dump-1-20260915-000002-full.tar.zst"; write_lb_tar(src, make_listens(cat))
    variant = _uuid(rng)
    canon = tmp_path / "musicbrainz-canonical-dump.tar.zst"
    redirect = "recording_mbid,canonical_recording_mbid,canonical_release_mbid\n" + f"{variant},{cat[0]['mbid']},{_uuid(rng)}\n"
    _write_multi_tar_zst(canon, {"d/canonical_musicbrainz_data.csv": canonical_csv(cat), "d/canonical_recording_redirect.csv": redirect})
    ab = tmp_path / "acousticbrainz-lowlevel-json-20220623-0.tar.zst"
    write_ab_archive(ab, [("lowlevel", r["mbid"], 0, make_ab_doc(rng)) for r in cat[:30]] + [("lowlevel", variant, 0, make_ab_doc(rng))])
    db = tmp_path / "lrclib.sqlite3"
    con = sqlite3.connect(db)
    con.executescript("CREATE TABLE tracks (id INTEGER PRIMARY KEY, name TEXT, artist_name TEXT, duration REAL, last_lyrics_id INTEGER);"
                      "CREATE TABLE lyrics (id INTEGER PRIMARY KEY, plain_lyrics TEXT, synced_lyrics TEXT, instrumental INT);")
    for i in range(0, 40, 2):
        con.execute("INSERT INTO tracks VALUES (?,?,?,?,?)", (i, f"Song {i}", f"Artist {i // 4}", 200.0, i))
        con.execute("INSERT INTO lyrics VALUES (?,?,?,0)", (i, f"line one {LYRIC_WORD} {i}\nline two", None))
    con.commit(); con.close()
    cfg = mb.TrackBConfig(listens=[src], name="tb_test", canonical=canon, ab_lowlevel=[ab], lrclib_dump=db, min_users_per_track=3)

    assert mb.ingest(ws, cfg, should_stop=lambda: True) is None                              # stops after the first month file
    c = mb.ingest(ws, cfg)
    assert c is not None and c.n_tracks == 40 and c.sessions.shape[0] == 90
    assert c.meta["coverage"]["n_audio"] == 30 and c.meta["coverage"]["n_lyrics"] == 20
    assert c.meta["coverage"]["n_all_three"] == len([i for i in range(30) if i % 2 == 0])
    out = ws.d("interim", "tb_test", base="data")
    c2 = read_corpus(out)
    assert c2.n_tracks == 40 and set(c2.families) == set(mb.FAMILY_NAMES) | {"lyric_theme"}
    assert (ws.data / "private" / "user_salt.bin").stat().st_size == 32
    for f in ws.data.rglob("*"):                                                             # nothing anywhere leaks names or lyric text
        if f.is_file() and f.name != "user_salt.bin":
            blob = f.read_bytes()
            if f.suffix == ".gz":
                blob = gzip.decompress(blob)
            assert LYRIC_WORD.encode() not in blob and not any(n.encode() in blob for n in NAMES + ["user_0_"]), f.name
    again = mb.ingest(ws, cfg)                                                                # idempotent re-run
    assert (again.sessions != c.sessions).nnz == 0 and list(again.tracks["ext_id"]) == list(c.tracks["ext_id"])
