"""Music4All-Onion adapter tests on small synthetic fixtures in two layouts (real-file formats are UNVERIFIED, so the
readers must cope with: tsv.bz2 with header; csv with list-valued cells / headerless events with ms timestamps / parquet)."""
import bz2
import json

import numpy as np
import pandas as pd
import pytest

from melomane.adapters import onion
from melomane.corpus import read_corpus, write_corpus

N_FEAT, N_EV_TRACKS, N_USERS = 40, 36, 12          # tracks 36..39 have features but no events
T0 = 1_400_000_000


def tid(i: int) -> str:
    return f"{i:04d}aBcD9xYz0Qw"


def make_events(seed=0):
    """12 users x (3 bursts of 7 distinct tracks + 1 burst of 3), bursts 2 h apart, plus a duplicate play and unknown tracks."""
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(N_USERS):
        for k in range(4):
            n = 7 if k < 3 else 3
            start = (u * 5 + k * 11) % N_EV_TRACKS
            base = T0 + u * 100_000 + k * 7200
            tracks = [(start + j) % N_EV_TRACKS for j in range(n)]
            for j, t in enumerate(tracks):
                rows.append((f"u{u:03d}", tid(t), base + 60 * j))
            rows.append((f"u{u:03d}", tid(tracks[0]), base + 60 * n))          # repeat play: still 7 distinct tracks
            if k == 0:
                rows.append((f"u{u:03d}", f"UNKNOWN{u:02d}", base + 5))         # track without features: dropped
    ev = pd.DataFrame(rows, columns=["userid", "trackid", "timestamp"])
    return ev.sample(frac=1.0, random_state=seed).reset_index(drop=True)           # shuffled like a real dump


def feature_frame(ids, cols, seed):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.normal(size=(len(ids), len(cols))), columns=cols)
    df.insert(0, "id", ids)
    return df


def write_layout_a(d):
    """tsv.bz2 with header everywhere; metadata present."""
    ids = [tid(i) for i in range(N_FEAT)]
    ev = make_events()
    ev.to_csv(d / "userid_trackid_timestamp.tsv.bz2", sep="\t", index=False)
    feature_frame(ids, [f"mfcc_{i}_{s}" for i in (1, 2, 3) for s in ("mean", "std")], 1).to_csv(d / "id_mfcc_stats.tsv.bz2", sep="\t", index=False)
    feature_frame(ids, ["logmel_sma_amean", "F0final_sma_amean", "jitterLocal_sma_amean", "pcm_zcr_sma_amean"], 2) \
        .to_csv(d / "id_compare_mfcc_stats.tsv.bz2", sep="\t", index=False)
    feature_frame(ids, [f"pcm{i}" for i in range(4)], 3).to_csv(d / "id_compare_pcm_stats.tsv.bz2", sep="\t", index=False)
    ess = feature_frame(ids, ["rhythm.bpm", "rhythm.danceability", "rhythm.onset_rate", "lowlevel.average_loudness",
                              "lowlevel.dynamic_complexity", "lowlevel.spectral_centroid.mean", "lowlevel.spectral_flux.mean",
                              "lowlevel.zerocrossingrate.mean", "tonal.chords_strength.mean", "tonal.key_edma.strength",
                              "tonal.hpcp_entropy.mean", "lowlevel.loudness_ebu128.integrated"], 4)
    ess["rhythm.bpm"] = 60 + 100 * np.abs(ess["rhythm.bpm"])
    ess["tonal.key_edma.key"] = "C"
    rng = np.random.default_rng(5)
    ess["rhythm.beats_position"] = ["[" + ", ".join(f"{x:.2f}" for x in np.cumsum(rng.random(rng.integers(3, 9)))) + "]" for _ in ids]
    ess.to_csv(d / "id_essentia.tsv.bz2", sep="\t", index=False)
    feature_frame(ids, [f"lf{i}" for i in range(5)], 6).to_csv(d / "id_blf_logfluc.tsv.bz2", sep="\t", index=False)
    feature_frame(ids, [f"chroma{i}" for i in range(12)], 7).to_csv(d / "id_chroma_bow.tsv.bz2", sep="\t", index=False)
    feature_frame(ids, [f"sp{i}" for i in range(4)], 8).to_csv(d / "id_blf_spectral.tsv.bz2", sep="\t", index=False)
    # lyrics: tracks 0..29 have lyrics; 30..35 instrumental (all zero); 36..39 absent
    lyr_ids = ids[:36]
    tf = np.abs(np.random.default_rng(9).normal(size=(36, 30))) * (np.random.default_rng(10).random((36, 30)) < 0.3)
    tf[:, 0] += 0.1
    tf[30:] = 0
    tfd = pd.DataFrame(tf, columns=[f"w{i}" for i in range(30)]); tfd.insert(0, "id", lyr_ids)
    tfd.to_csv(d / "id_lyrics_tf-idf.tsv.bz2", sep="\t", index=False)
    b = feature_frame(lyr_ids, [f"bert{i}" for i in range(10)], 11); b.iloc[30:, 1:] = 0
    b.to_csv(d / "id_bert.tsv.bz2", sep="\t", index=False)
    w = feature_frame(lyr_ids, [f"w2v{i}" for i in range(6)], 12); w.iloc[30:, 1:] = 0
    w.to_csv(d / "id_lyrics_word2vec.tsv.bz2", sep="\t", index=False)
    # held-out + external
    feature_frame(ids, [f"gem{i}" for i in range(5)], 13).to_csv(d / "id_gems.tsv.bz2", sep="\t", index=False)
    feature_frame(ids, [f"sent_mean_{i}" for i in range(3)] + ["sent_std"], 14).to_csv(d / "id_lyrics_sentiment_functionals.tsv.bz2", sep="\t", index=False)
    vad = feature_frame(ids, [f"valence_{i}" for i in range(4)] + [f"arousal_{i}" for i in range(4)], 15)
    vad.iloc[:, 1:] = vad.iloc[:, 1:].abs()
    vad.to_csv(d / "id_vad_bow.tsv.bz2", sep="\t", index=False)
    feature_frame(ids, [f"j{i}" for i in range(8)], 16).to_csv(d / "id_jukebox.tsv.bz2", sep="\t", index=False)
    feature_frame(ids, [f"m{i}" for i in range(5)], 17).to_csv(d / "id_musicnn.tsv.bz2", sep="\t", index=False)
    # metadata + tags
    meta = pd.DataFrame({"id": ids, "artist": [f"Artist {i % 12}" for i in range(N_FEAT)], "song": [f"Song {i}" for i in range(N_FEAT)],
                         "album_name": [f"Album {i % 9}" for i in range(N_FEAT)],
                         "release": [f"{1970 + i}-05-01" for i in range(N_FEAT)], "popularity": np.arange(N_FEAT, dtype=float)})
    meta.to_csv(d / "id_information.tsv.bz2", sep="\t", index=False)
    tags = pd.DataFrame(np.eye(N_FEAT, 5) * 50 + 1, columns=["rock", "pop", "jazz", "metal", "electronic"]); tags.insert(0, "id", ids)
    tags.to_csv(d / "id_tags_dense.tsv.bz2", sep="\t", index=False)
    return {"ids": ids, "events": ev, "ess": ess, "meta": meta}


def _lists(df, cols, name):
    out = pd.DataFrame({"id": df["id"]})
    out[name] = ["[" + ", ".join(f"{x:.5f}" for x in row) + "]" for row in df[cols].to_numpy()]
    return out


def write_layout_b(d):
    """csv; list-valued cells; headerless events with millisecond timestamps; parquet Essentia; mixed-case hyphenated names."""
    ids = [tid(i) for i in range(N_FEAT)]
    ev = make_events()
    ev2 = ev.copy(); ev2["timestamp"] = ev2["timestamp"] * 1000 + 123
    ev2.to_csv(d / "USERID-TRACKID-TIMESTAMP.csv", header=False, index=False)
    mf = feature_frame(ids, [f"mfcc_{i}_{s}" for i in (1, 2, 3) for s in ("mean", "std")], 1)
    _lists(mf, mf.columns[1:], "mfcc_stats").to_csv(d / "ID_MFCC_STATS.csv", index=False)         # commas inside quoted lists
    ch = feature_frame(ids, [f"c{i}" for i in range(12)], 7)
    _lists(ch, ch.columns[1:], "chroma").to_csv(d / "id_chroma-bow.csv", index=False)
    ess = feature_frame(ids, ["rhythm.bpm", "rhythm.danceability", "tonal.chords_strength.mean", "lowlevel.average_loudness"], 4)
    ess["rhythm.bpm"] = 60 + 100 * np.abs(ess["rhythm.bpm"])
    ess.to_parquet(d / "id_essentia.parquet")
    feature_frame(ids, [f"gem{i}" for i in range(5)], 13).to_csv(d / "id_gems.csv", index=False)
    return {"ids": ids, "events": ev, "ess": ess}


def truth_sessions(ev, ids, gap=30, min_len=5, max_len=200):
    known = set(ids)
    e = ev[ev["trackid"].isin(known)].sort_values(["userid", "timestamp"])
    n = 0
    for _, g in e.groupby("userid"):
        t = g["timestamp"].to_numpy()
        sid = np.r_[0, np.cumsum(np.diff(t) > gap * 60)]
        for s in np.unique(sid):
            L = g["trackid"].to_numpy()[sid == s]
            n += min_len <= len(set(L)) <= max_len
    return n


# ----------------------------------------------------------------------------------------------------------------------
def test_detect_timestamp_unit():
    assert onion.detect_timestamp_unit(np.array([1_400_000_000, 1_500_000_000])) == "s"
    assert onion.detect_timestamp_unit(np.array([1_400_000_000_123, 1_500_000_000_999])) == "ms"
    assert onion.detect_timestamp_unit(np.array([1_400_000_000_123_456])) == "us"
    assert onion.detect_timestamp_unit(np.array([1_400_000_000_123_456_789])) == "ns"
    assert onion.detect_timestamp_unit(np.array([])) == "s"
    assert onion.detect_timestamp_unit(np.array([0, 0, 5, 10])) == "s"
    np.testing.assert_array_equal(onion.to_seconds(np.array([1_400_000_000_999]), "ms"), [1_400_000_000])


def test_find_file_is_case_separator_and_extension_insensitive(tmp_path):
    for n in ["ID_Lyrics_TF-IDF.TSV.BZ2", "id-mfcc-stats.csv", "id_essentia.parquet", "id_essentia.tsv", "README.md", "INVENTORY.json"]:
        (tmp_path / n).write_bytes(b"x")
    (tmp_path / "sub").mkdir(); (tmp_path / "sub" / "id_gems.tsv.gz").write_bytes(b"x")
    assert onion.find_file(tmp_path, "id_lyrics_tf-idf").name == "ID_Lyrics_TF-IDF.TSV.BZ2"
    assert onion.find_file(tmp_path, "id_mfcc_stats").name == "id-mfcc-stats.csv"
    assert onion.find_file(tmp_path, "id_essentia").name == "id_essentia.parquet"           # parquet preferred
    assert onion.find_file(tmp_path, "id_gems").parent.name == "sub"
    assert onion.find_file(tmp_path, "id_bert") is None and onion.find_file(tmp_path, "readme") is None
    assert [p.name.lower()[:9] for p in onion.FileIndex(tmp_path).find_files("id_*")] == ["id_essent", "id_gems.t", "id_lyrics", "id-mfcc-s"]


def test_read_table_variants(tmp_path):
    # tsv.bz2 with header; ids that look numeric must stay strings
    with bz2.open(tmp_path / "a.tsv.bz2", "wt") as f:
        f.write("id\tx\ty\n0001\t1.5\tfoo\n0002\t2.5\tbar\n")
    df = onion.read_table(tmp_path / "a.tsv.bz2")
    assert list(df["id"]) == ["0001", "0002"] and df["x"].dtype == np.float32 and onion._is_text(df["y"])
    # headerless tab table
    (tmp_path / "b.tsv").write_text("t1\t1\t2\nt2\t3\t4\n")
    df = onion.read_table(tmp_path / "b.tsv")
    assert list(df["id"]) == ["t1", "t2"] and df.shape == (2, 3)
    # numeric column labels
    (tmp_path / "c.csv").write_text(",0,1,2\nt1,1,2,3\nt2,4,5,6\n")
    df = onion.read_table(tmp_path / "c.csv")
    assert list(df.columns) == ["id", "0", "1", "2"] and df.shape == (2, 4)
    # csv with quoted list cells (+ NaN token) and a semicolon file
    (tmp_path / "d.csv").write_text('id,emb\nt1,"[1.0, 2.0, 3.0]"\nt2,"[4, 5, nan]"\nt3,\n')
    df = onion.read_table(tmp_path / "d.csv")
    assert df.shape == (3, 4) and df.loc[0, "emb[1]"] == 2.0 and np.isnan(df.loc[1, "emb[2]"]) and np.isnan(df.loc[2, "emb[0]"])
    (tmp_path / "e.txt").write_text("id;a;b\nx;1;2\ny;3;4\n")
    assert onion.read_table(tmp_path / "e.txt")["b"].tolist() == [2.0, 4.0]
    # variable-length lists are skipped, not exploded
    (tmp_path / "f.tsv").write_text("id\tpos\tv\nt1\t[1, 2]\t1\nt2\t[1, 2, 3, 4]\t2\nt3\t[5]\t3\n")
    df = onion.read_table(tmp_path / "f.tsv")
    assert list(df.columns) == ["id", "v"]
    # parquet + column subset
    pd.DataFrame({"id": ["a", "b"], "p": [1.0, 2.0], "q": [3.0, 4.0]}).to_parquet(tmp_path / "g.parquet")
    assert list(onion.read_table(tmp_path / "g.parquet", columns=["q"]).columns) == ["id", "q"]


@pytest.mark.parametrize("engine", ["pyarrow", "pandas"])
def test_load_events_string_id_mapping_and_units(tmp_path, engine):
    ev = pd.DataFrame({"userid": ["b", "a", "b", "a"], "trackid": ["t2", "t1", "t9", "t1"],
                       "timestamp": [1_500_000_000_000, 1_500_000_100_000, 1_500_000_200_000, 1_500_000_300_000]})
    p = tmp_path / "userid_trackid_timestamp.tsv.bz2"
    ev.to_csv(p, sep="\t", index=False)
    t = onion.load_events(p, track_index=["t1", "t2", "t3"], engine=engine, chunk_rows=2)
    assert t.unit == "ms" and t.n_users == 2 and t.user.dtype == np.int32 and t.ts.dtype == np.int64
    assert t.track.tolist() == [1, 0, 0] and t.ts.tolist() == [1_500_000_000, 1_500_000_100, 1_500_000_300]   # t9 dropped, sec
    assert t.user.tolist() == [0, 1, 1]                        # dense codes by first appearance ('b' -> 0)
    t2 = onion.load_events(p, track_index=None, engine=engine)
    assert sorted(t2.track_keys.tolist()) == ["t1", "t2", "t9"] and t2.track.size == 4
    # count file (no timestamps)
    pc = tmp_path / "userid_trackid_count.tsv"
    pc.write_text("u1\tt1\t5\nu1\tt2\t1\nu2\tt1\t2\n")
    t3 = onion.load_events(pc, track_index=["t1", "t2"], engine=engine)
    assert t3.ts is None and t3.count.tolist() == [5, 1, 2]
    # ISO date strings -> seconds
    pd_ = tmp_path / "userid_trackid_timestamp.csv"
    pd_.write_text("userid,trackid,timestamp\nu,t1,2015-01-01 00:00:00\nu,t2,2015-01-01 00:01:00\n")
    t4 = onion.load_events(pd_, track_index=["t1", "t2"], engine=engine)
    assert t4.unit == "s" and np.diff(t4.ts)[0] == 60 and 1.4e9 < t4.ts[0] < 1.5e9


def test_layout_a_full_build(tmp_path):
    d = tmp_path / "onion"; d.mkdir()
    truth = write_layout_a(d)
    c = onion.build_corpus(d, gap_minutes=30, min_len=5, max_len=200, seed=0)
    c.validate()
    ids = truth["ids"]
    # ---- string-id -> raw-id mapping: tracks with events AND features, sorted by string id ----------------------------------
    ev_tracks = sorted(set(truth["events"]["trackid"]) & set(ids))
    assert c.n_tracks == len(ev_tracks) == N_EV_TRACKS
    assert c.tracks["ext_id"].tolist() == ev_tracks and (c.tracks.index.to_numpy() == np.arange(c.n_tracks)).all()
    assert not set(c.tracks["ext_id"]) & {tid(i) for i in range(N_EV_TRACKS, N_FEAT)}         # feature-only tracks excluded
    assert not any(x.startswith("UNKNOWN") for x in c.tracks["ext_id"])
    # ---- sessions -----------------------------------------------------------------------------------------------------------
    n_true = truth_sessions(truth["events"], ids)
    assert c.sessions.shape == (n_true, c.n_tracks) == (3 * N_USERS, N_EV_TRACKS)
    assert (np.diff(c.sessions.indptr) == 7).all() and set(c.sessions.data) == {1.0}
    assert c.sessions_meta["user_id"].nunique() == N_USERS and (c.sessions_meta["n_tracks"] == 7).all()
    assert c.sessions_meta.groupby("user_id").size().eq(3).all()
    assert c.meta["timestamp_unit"] == "s" and c.meta["session_semantics"] == "session" and c.meta["n_users"] == N_USERS
    # a specific session equals its ground-truth track set
    first_user = truth["events"].sort_values(["userid", "timestamp"]).iloc[0]["userid"]
    g = truth["events"][(truth["events"]["userid"] == first_user) & truth["events"]["trackid"].isin(ids)].sort_values("timestamp")
    burst0 = set(g["trackid"].iloc[:8])
    ext = c.tracks["ext_id"].to_numpy()
    assert len(burst0) == 7 and any(set(ext[r.indices]) == burst0 for r in c.sessions)
    # ---- families -----------------------------------------------------------------------------------------------------------
    assert set(c.families) == {"timbre", "spectral_shape", "rhythm_groove", "harmony_tonal", "dynamics_loudness", "lyric_theme", "lyric_lexicon"}
    tim = c.families["timbre"]
    assert tim.rows.size == N_EV_TRACKS and tim.X.dtype == np.float32 and np.isfinite(tim.X).all()
    assert any(col.startswith("id_mfcc_stats:") for col in tim.columns) and any(col.startswith("id_compare_mfcc_stats:") for col in tim.columns)
    assert not any("f0" in col.lower() or "jitter" in col.lower() for col in tim.columns)           # voice columns stripped
    rg = c.families["rhythm_groove"]
    assert any("rhythm.bpm" in col for col in rg.columns) and not any("beats_position" in col for col in rg.columns)
    assert not any("tonal.key_edma.key" in col for col in c.families["harmony_tonal"].columns)       # string column ignored
    lex = c.families["lyric_lexicon"]
    assert lex.X.shape[1] <= 64 and lex.rows.size == 30                                             # 6 instrumentals have no lexicon row
    assert c.families["lyric_theme"].rows.size == 30
    # has_lyrics: only tracks 0..29
    assert c.tracks.set_index("ext_id")["has_lyrics"].to_dict() == {tid(i): i < 30 for i in range(N_EV_TRACKS)}
    # ---- leakage: held-out / learned never in families ---------------------------------------------------------------------------
    assert onion.check_family_leakage(c.families) == []
    for b in c.families.values():
        assert not any(k in b.source for k in ("gems", "sentiment", "vad", "jukebox", "musicnn"))
    assert set(c.external) >= {"jukebox", "musicnn", "bert", "lyrics_word2vec"} and c.external["jukebox"].X.shape == (N_EV_TRACKS, 8)
    assert c.heldout is not None and len(c.heldout) == c.n_tracks
    hcols = set(c.heldout.columns)
    assert any(x.startswith("gems_") for x in hcols) and any(x.startswith("sentiment_") for x in hcols)
    assert {"vad_valence_bow_mean", "vad_arousal_bow_mean"} <= hcols and {"voice_f0", "voice_jitter"} <= hcols
    # ---- measured -----------------------------------------------------------------------------------------------------------------
    m = c.measured
    assert m is not None and len(m) == c.n_tracks
    assert {"bpm", "danceability", "onset_rate", "loudness", "dynamic_complexity", "spectral_centroid", "spectral_flux",
            "zero_crossing_rate", "chords_strength", "key_strength", "hpcp_entropy", "mfcc1_mean", "mfcc2_mean"} <= set(m.columns)
    ess = truth["ess"].set_index("id")
    got = m["bpm"].to_numpy(); want = ess.loc[c.tracks["ext_id"], "rhythm.bpm"].to_numpy()
    np.testing.assert_allclose(got, want, rtol=1e-5)
    # ---- metadata / blocker A ----------------------------------------------------------------------------------------------------
    assert c.meta["blocker_a"] is False
    tr = c.tracks
    assert tr["artist"].str.startswith("Artist").all() and tr["title"].str.startswith("Song").all()
    assert tr.set_index("ext_id").loc[tid(3), "year"] == 1973 and tr.set_index("ext_id").loc[tid(3), "popularity"] == 3
    assert tr.groupby("artist")["artist_id"].nunique().eq(1).all() and tr["artist_id"].nunique() == 12
    assert tr.loc[0, "tags"].split("|")[0] in ("rock", "pop", "jazz", "metal", "electronic")
    # ---- round trip through the on-disk contract -----------------------------------------------------------------------------
    write_corpus(c, tmp_path / "out")
    c2 = read_corpus(tmp_path / "out")
    assert c2.n_tracks == c.n_tracks and set(c2.families) == set(c.families) and c2.meta["n_sessions"] == c.sessions.shape[0]


def test_layout_b_csv_lists_ms_timestamps_and_blocker_a(tmp_path):
    d = tmp_path / "onion_b"; d.mkdir()
    truth = write_layout_b(d)
    c = onion.build_corpus(d, seed=1, events_engine="pandas")
    c.validate()
    assert c.meta["timestamp_unit"] == "ms"
    assert c.sessions.shape == (3 * N_USERS, N_EV_TRACKS)
    assert set(c.families) == {"timbre", "harmony_tonal", "rhythm_groove", "dynamics_loudness"}        # list cells expanded; others omitted
    assert c.families["timbre"].X.shape[1] == 6 and c.families["harmony_tonal"].X.shape[1] >= 12
    assert c.meta["family_report"]["lyric_lexicon"]["status"] == "omitted"
    assert "lyric_theme" not in c.families and not c.tracks["has_lyrics"].any()
    # blocker A: no metadata files at all
    assert c.meta["blocker_a"] is True and "Blocker-A" in c.meta["blocker_a_note"]
    assert (c.tracks["artist"] == "").all() and (c.tracks["title"] == "").all()
    assert (c.tracks["artist_id"] < 0).all() and c.tracks["artist_id"].nunique() == c.n_tracks
    assert set(c.heldout.columns) >= {"gems_gem0"}
    # both engines agree
    c_pa = onion.build_corpus(d, seed=1, events_engine="pyarrow")
    assert (c_pa.sessions != c.sessions).nnz == 0 and c_pa.tracks["ext_id"].tolist() == c.tracks["ext_id"].tolist()


def test_missing_files_never_crash_and_shrink_families(tmp_path):
    d = tmp_path / "o"; d.mkdir()
    write_layout_a(d)
    for n in ["id_compare_mfcc_stats", "id_essentia", "id_lyrics_tf-idf", "id_bert", "id_lyrics_word2vec", "id_gems", "id_information"]:
        for p in list(d.glob(f"{n}.*")):
            p.unlink()
    c = onion.build_corpus(d)
    rep = c.meta["family_report"]
    assert "id_compare_mfcc_stats" in rep["timbre"]["missing"] and rep["timbre"]["status"] == "ok"           # shrunk
    assert all(col.startswith("id_mfcc_stats:") for col in c.families["timbre"].columns)
    assert rep["rhythm_groove"]["status"] == "ok" and all("essentia" not in col for col in c.families["rhythm_groove"].columns)
    assert "lyric_theme" not in c.families and "lyric_lexicon" not in c.families and rep["lyric_lexicon"]["status"] == "omitted"
    assert c.measured is None or "bpm" not in c.measured.columns
    assert not c.tracks["has_lyrics"].any() and c.meta["blocker_a"] is True
    assert "id_essentia" in c.meta["missing_expected_files"]
    # no events at all
    for p in d.glob("userid_trackid*"):
        p.unlink()
    with pytest.raises(FileNotFoundError, match="listening-event"):
        onion.build_corpus(d)
    c0 = onion.build_corpus(d, require_events=False)
    assert c0.sessions.shape[0] == 0 and c0.n_tracks == N_FEAT and c0.meta["session_semantics"] == "none"
    with pytest.raises(FileNotFoundError):
        onion.build_corpus(tmp_path / "nope")


def test_family_config_override_and_forbidden_sources_refused(tmp_path):
    d = tmp_path / "o"; d.mkdir()
    write_layout_a(d)
    c = onion.build_corpus(d, families={
        "timbre": {"sources": ["id_mfcc_stats", "id_gems", "id_jukebox"]},                        # held-out + learned: refused
        "chroma_only": ["id_chroma_bow"],
        "spectral_shape": None, "lyric_lexicon": None, "lyric_theme": None,
        "rhythm_groove": {"sources": [{"file": "id_essentia", "columns": ["rhythm.bpm", "rhythm.danceability"]}]}})
    assert set(c.families) == {"timbre", "chroma_only", "rhythm_groove", "harmony_tonal", "dynamics_loudness"}
    assert c.meta["family_report"]["timbre"]["refused"] == ["id_gems", "id_jukebox"]
    assert c.families["timbre"].X.shape[1] == 6
    assert c.families["rhythm_groove"].X.shape[1] == 2
    assert onion.check_family_leakage(c.families) == []
    bad = {"x": onion.FamilyBlock(np.array([0]), np.zeros((1, 1), np.float32), ["id_gems:gem0"], "id_gems")}
    assert onion.check_family_leakage(bad)
    # limit_tracks: seeded subset of eligible tracks
    c5 = onion.build_corpus(d, limit_tracks=20, seed=3)
    assert c5.n_tracks <= 20 and c5.n_tracks > 0


def test_sharded_sessionisation_matches_unsharded(tmp_path):
    d = tmp_path / "o"; d.mkdir()
    write_layout_a(d)
    a = onion.build_corpus(d, max_shard_events=10 ** 9)
    b = onion.build_corpus(d, max_shard_events=50)
    assert a.sessions.shape == b.sessions.shape
    sig = lambda c: sorted(tuple(r.indices.tolist()) for r in c.sessions)     # noqa: E731
    assert sig(a) == sig(b)


def test_count_only_logs_degrade_to_user_profiles(tmp_path):
    d = tmp_path / "o"; d.mkdir()
    write_layout_a(d)
    (d / "userid_trackid_timestamp.tsv.bz2").unlink()
    rng = np.random.default_rng(0)
    rows = [(f"u{u}", tid(t), int(rng.integers(1, 9))) for u in range(6) for t in rng.choice(N_EV_TRACKS, 10, replace=False)]
    pd.DataFrame(rows, columns=["userid", "trackid", "count"]).to_csv(d / "userid_trackid_count.tsv", sep="\t", index=False)
    c = onion.build_corpus(d, max_len=8)
    assert c.meta["session_semantics"] == "user_profile" and c.sessions.shape[0] == 6
    assert (np.diff(c.sessions.indptr) == 8).all()


def test_inventory_writes_json(tmp_path):
    d = tmp_path / "o"; d.mkdir()
    write_layout_a(d)
    (d / "broken.tsv").write_bytes(b"\x00\x01\x02")
    inv = onion.write_inventory(d)
    saved = json.loads((d / "INVENTORY.json").read_text(encoding="utf-8"))
    assert saved["n_files"] == inv["n_files"] >= 19
    f = saved["files"]["id_essentia.tsv.bz2"]
    assert f["rows"] == N_FEAT and f["n_columns"] == 15 and f["has_header"] is True and f["separator"] == "\t"
    assert f["dtype_summary"]["list"] == 1 and len(f["sample_rows"]) == 3
    ev = saved["files"]["userid_trackid_timestamp.tsv.bz2"]
    assert ev["kind"] == "events" and ev["timestamp"]["unit"] == "s" and ev["event_columns"]["user"] == "userid"
    assert saved["blocker_a"] is False and saved["metadata_fields_found"]["artist"].startswith("id_information")
    assert "id_bert" not in saved["expected_files_missing"] and "userid_trackid_count" in saved["expected_files_missing"]
    assert "essentia_descriptor_overlap" in f or not (onion._essentia_reference())
    assert "broken.tsv" in saved["files"]                      # unreadable files are reported, not fatal


def test_prepare_metadata_lifts_blocker_a_and_gives_real_artist_ids(tmp_path):
    """The Zenodo record has no artist/title; an external table is normalised by ``prepare_metadata`` and then flows through the
    ordinary adapter path, giving one artist_id per *normalised artist name* (not one per track: that made artist-grouped
    CV a no-op)."""
    d = tmp_path / "onion_b"; d.mkdir()
    truth = write_layout_b(d)
    ids = truth["ids"]
    src = pd.DataFrame({"track_id": ids, "tag_list": [["x"]] * len(ids), "pseudo_caption": ["a made-up caption"] * len(ids),
                        "title": [f"Song {i}" for i in range(len(ids))], "artist_name": [("The " if i % 2 else "") + f"Band {i % 7}" for i in range(len(ids))],
                        "release": [f"Album {i % 5}" for i in range(len(ids))], "m4a_genres": ["rock"] * len(ids), "m4a_tags": ["rock"] * len(ids)})
    sp = tmp_path / "track_metadata.parquet"; src.to_parquet(sp)
    rec = onion.prepare_metadata(sp, d)
    assert rec["rows"] == len(ids) and rec["with_artist"] == len(ids) and rec["unique_artists"] == 7 and len(rec["source_sha256"]) == 64
    kept = pd.read_parquet(d / onion.METADATA_FILE)
    assert list(kept.columns) == ["id", "artist", "title", "album"]            # captions / tag lists are not carried over
    c = onion.build_corpus(d, seed=1, events_engine="pandas")
    assert c.meta["blocker_a"] is False
    assert (c.tracks["artist"] != "").all() and c.tracks["artist_id"].nunique() == 7 and (c.tracks["artist_id"] >= 0).all()
    same = c.tracks.groupby("artist_id")["artist"].apply(lambda s: s.map(lambda x: x.removeprefix("The ")).nunique())
    assert (same == 1).all()                                                    # "The Band 1" and "Band 1" are one artist


def test_prepare_metadata_rejects_a_table_without_artist(tmp_path):
    sp = tmp_path / "m.parquet"; pd.DataFrame({"track_id": ["a"], "title": ["t"]}).to_parquet(sp)
    with pytest.raises(ValueError, match="artist"):
        onion.prepare_metadata(sp, tmp_path)
