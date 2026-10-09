import numpy as np, pytest
from pathlib import Path
from melomane.adapters import mysqldump as md, playlist2vec as p2v
from melomane.corpus import read_corpus, write_corpus
from melomane.workspace import Workspace
from fakedump import make_dump


def test_parse_row_edge_cases():
    r = md.parse_row(r"(1,'It\'s (a) \\ test, ok',NULL,-3.5,'',1e-3)")
    assert r == [1, "It's (a) \\ test, ok", None, -3.5, "", 0.001]
    assert md.parse_row("('a\\nb')") == ["a\nb"]


def test_iter_rows_handles_parens_and_commas_in_strings():
    rows = list(md.iter_rows("(1,'a)(b'),(2,'c,d'),(3,NULL);"))
    assert rows == [[1, "a)(b"], [2, "c,d"], [3, None]]


def test_mojibake_repair():
    assert md.fix_mojibake("CafÃ©") == "Café"
    assert md.fix_mojibake("Café") == "Café"
    assert md.fix_mojibake("plain") == "plain"


def test_table_columns_from_schema():
    ddl = open(Path(__file__).with_name("playlist2vec_schema.sql")).read()
    cols = {}
    for kind, tbl, payload in md.iter_statements(Path(__file__).with_name("playlist2vec_schema.sql")):
        if kind == "create":
            cols[tbl] = md.table_columns(payload)
    assert cols["track"] == ["id", "name", "duration", "popularity", "explicit", "preview_url", "uri", "album_id"]
    assert cols["track_playlist1"] == ["track_id", "playlist_id"]


@pytest.mark.parametrize("gz", [False, True])
def test_parse_dump_roundtrip(tmp_path, gz):
    p = tmp_path / ("d.sql.gz" if gz else "d.sql")
    truth = make_dump(p, gz=gz)
    t = p2v.parse_dump(p)
    assert len(t.track) == len(truth["tracks"]) and len(t.playlist) == len(truth["playlists"])
    assert set(t.track["id"]) == {x[0] for x in truth["tracks"]}
    # strings with quotes/backslashes/unicode survive
    want = {x[0]: x[1] for x in truth["tracks"]}
    got = dict(zip(t.track["id"], t.track["name"]))
    assert got == want
    # dangling FK dropped, duplicates are unique-index violations in real MySQL but we tolerate them
    assert t.stats["dropped_tp"] == 1
    assert t.tp_track.size == len(truth["tp"]) - 1


def test_build_corpus_filters_and_primary_artist(tmp_path):
    p = tmp_path / "d.sql"; truth = make_dump(p)
    t = p2v.parse_dump(p)
    c = p2v.build_corpus(t, min_playlist_len=5, max_playlist_len=60)
    c.validate()
    lens = np.diff(c.sessions.indptr)
    assert lens.min() >= 5 and lens.max() <= 60
    assert (c.sessions.data == 1).all()                              # duplicates collapsed
    # primary artist = first link
    first = {}
    for tid, aid in truth["ta"]:
        first.setdefault(tid, aid)
    a_name = {a[0]: a[1] for a in truth["artists"]}
    row = c.tracks.iloc[7]
    assert row["artist"] == a_name[first[row["ext_id"]]]
    assert c.tracks["n_artists"].max() == 2                            # collaboration flag
    assert c.tracks["artist_id"].dtype == np.int64
    assert (c.tracks["n_playlists"].to_numpy() == np.asarray(c.sessions.sum(axis=0)).ravel()).all()
    assert c.sessions_meta is not None and len(c.sessions_meta) == c.sessions.shape[0]


def test_mojibake_titles_repaired(tmp_path):
    p = tmp_path / "d.sql"; make_dump(p, mojibake=True)
    c = p2v.build_corpus(p2v.parse_dump(p), 5, 60)
    assert any("Ünïcödé" in s for s in c.tracks["title"])


def test_corpus_roundtrip_on_disk(tmp_path):
    p = tmp_path / "d.sql"; make_dump(p)
    c = p2v.build_corpus(p2v.parse_dump(p), 5, 60)
    write_corpus(c, tmp_path / "c")
    c2 = read_corpus(tmp_path / "c")
    assert (c2.sessions != c.sessions).nnz == 0
    assert list(c2.tracks["ext_id"]) == list(c.tracks["ext_id"]) and c2.meta["has_audio"] is False


def test_find_dump_detects_in_flight_download_and_completion(tmp_path):
    # schema-only file is ignored; in-flight crdownload detected as incomplete; finished .sql detected as complete
    schema = tmp_path / "schema.sql"; schema.write_bytes(open(Path(__file__).with_name("playlist2vec_schema.sql"), "rb").read())
    st = p2v.find_dump([tmp_path], include_downloads=False); assert st.path is None
    big = tmp_path / "Unconfirmed 123.crdownload"
    make_dump(big, n_tracks=300, complete=False)
    big.write_bytes(big.read_bytes() + b"-- pad\n" * 8_000_000)          # > 50 MB so it passes the size floor
    st = p2v.find_dump([tmp_path], include_downloads=False); assert st.path == big and not st.complete
    done = tmp_path / "spotifydb.sql"; big.rename(done)
    done.write_bytes(done.read_bytes() + b"-- Dump completed on 2021-02-11 15:42:39\n")
    st = p2v.find_dump([tmp_path], include_downloads=False); assert st.path == done and st.complete


def test_wait_for_dump_polls_until_complete(tmp_path):
    f = tmp_path / "x.sql"
    make_dump(f, complete=False); f.write_bytes(f.read_bytes() + b"-- pad\n" * 8_000_000)
    calls = {"n": 0}
    def fake_sleep(_):
        calls["n"] += 1
        if calls["n"] == 2:
            f.write_bytes(f.read_bytes() + b"-- Dump completed on 2021\n")
    st = p2v.wait_for_dump([tmp_path], poll_s=0, sleep=fake_sleep, timeout_h=1, include_downloads=False)
    assert st.complete and calls["n"] == 2
