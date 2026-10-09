"""Track C (FMA, MTG-Jamendo) tests on synthetic fixtures that mimic the documented file layouts.  No network."""
from __future__ import annotations
import io, json, tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from melomane.adapters import fma, jamendo, mbid_stack as mb
from melomane.corpus import Corpus, read_corpus, write_corpus

STATS = ["kurtosis", "max", "mean", "median", "min", "skew", "std"]
LIBROSA = {"chroma_cens": 12, "chroma_cqt": 12, "chroma_stft": 12, "mfcc": 20, "rmse": 1, "spectral_bandwidth": 1, "spectral_centroid": 1,
           "spectral_contrast": 7, "spectral_rolloff": 1, "tonnetz": 6, "zcr": 1}


# ------------------------------------------------------------------------------------------------------------- FMA
def make_fma(tmp: Path, n=12, seed=0):
    rng = np.random.default_rng(seed)
    ids = [2, 3, 5, 10, 20, 134, 135, 136, 137, 138, 139, 140][:n]
    cols = [("album", "date_released"), ("album", "id"), ("album", "title"), ("artist", "id"), ("artist", "name"), ("set", "split"),
            ("set", "subset"), ("track", "date_recorded"), ("track", "duration"), ("track", "genre_top"), ("track", "genres"),
            ("track", "genres_all"), ("track", "language_code"), ("track", "license"), ("track", "listens"), ("track", "tags"), ("track", "title")]
    rows = []
    for k, t in enumerate(ids):
        sub = "small" if k < 4 else "medium" if k < 8 else "large"
        g = [21] if k % 2 == 0 else [12]
        allg = [21, 12] if k % 2 == 0 else [12]
        rows.append(["2009-01-05 00:00:00", 100 + k // 2, f"Album {k // 2}", (k // 2) if k != 11 else np.nan, f"Artist {k // 2}" if k != 11 else np.nan,
                     ["training", "validation", "test"][k % 3], sub, "2008-11-26 01:43:26" if k % 4 else np.nan, 168 + k, "Hip-Hop" if k % 2 == 0 else "Rock",
                     str(g), str(allg), "en", "CC BY 4.0", 1000 + k, "['tag a', 'tag b']" if k == 0 else "[]", f"Track {t}"])
    df = pd.DataFrame(rows, index=pd.Index(ids, name="track_id"), columns=pd.MultiIndex.from_tuples(cols))
    df.to_csv(tmp / "tracks.csv")                                                     # pandas writes the 3-line FMA header
    pd.DataFrame({"#tracks": [100, 50, 20], "parent": [0, 0, 12], "title": ["Hip-Hop", "Rock", "Indie-Rock"], "top_level": [21, 12, 12]},
                 index=pd.Index([21, 12, 25], name="genre_id")).to_csv(tmp / "genres.csv")
    fc = [(f, s, f"{i + 1:02d}") for f, n_ in LIBROSA.items() for s in STATS for i in range(n_)]
    feat = pd.DataFrame(rng.normal(size=(n, len(fc))), index=pd.Index(ids, name="track_id"),
                        columns=pd.MultiIndex.from_tuples(fc, names=["feature", "statistics", "number"]))
    feat.loc[ids[3], ("mfcc", "mean", "01")] = np.nan                                 # one incomplete row (raw id 3)
    feat.to_csv(tmp / "features.csv")
    ec = [("echonest", "audio_features", c) for c in ["acousticness", "danceability", "energy", "instrumentalness", "liveness", "speechiness",
                                                        "tempo", "valence"]] + [("echonest", "metadata", "artist_name")] \
         + [("echonest", "temporal_features", f"{i:03d}") for i in range(3)]
    ech_ids = ids[:6]
    e = pd.DataFrame([list(rng.random(8)) + ["nm"] + list(rng.random(3)) for _ in ech_ids], index=pd.Index(ech_ids, name="track_id"),
                     columns=pd.MultiIndex.from_tuples(ec))
    e.to_csv(tmp / "echonest.csv")
    return ids, fc


def test_fma_layouts_and_load_tracks(tmp_path):
    ids, fc = make_fma(tmp_path)
    head = (tmp_path / "tracks.csv").read_text().splitlines()[:3]
    assert head[0].startswith(",album") and head[2].startswith("track_id")           # 3 header lines as in the real file
    assert len(fc) == 518                                                             # the real features.csv width
    t = fma.load_tracks(tmp_path / "tracks.csv", tmp_path / "genres.csv")
    assert list(t.index) == list(range(12)) and t["ext_id"].tolist()[:3] == ["2", "3", "5"]
    assert t["artist_id"].dtype == np.int64 and t["artist_id"].iloc[11] < 0 and t["artist"].iloc[11] == ""
    assert t["title"].iloc[0] == "Track 2" and t["tags"].iloc[0] == "Hip-Hop|Rock" and t["tags"].iloc[1] == "Rock" and t["genres"].iloc[0] == "21"
    assert t["year"].iloc[1] == 2008 and t["year"].iloc[0] == 2009                     # date_recorded, else album release
    assert t["duration_ms"].iloc[0] == 168_000 and t["popularity"].iloc[0] == 1000 and t["track_tags"].iloc[0] == "tag a|tag b"
    assert set(t["subset"]) == {"small", "medium", "large"} and set(t["split"]) == {"training", "validation", "test"}
    assert len(fma.load_tracks(tmp_path / "tracks.csv", subset="small")) == 4 and len(fma.load_tracks(tmp_path / "tracks.csv", subset="medium")) == 8
    assert fma.load_tracks(tmp_path / "tracks.csv")["tags"].iloc[0] == "21|12"        # ids if genres.csv not supplied
    g = fma.load_genres(tmp_path / "genres.csv")
    assert list(g.columns) == ["title", "parent", "top_level", "n_tracks"] and g.loc[25, "title"] == "Indie-Rock"
    Y, names = fma.tag_matrix(t)
    assert names == ["Hip-Hop", "Rock"] and Y.shape == (12, 2) and Y[0].sum() == 2 and Y[1].sum() == 1
    assert t.attrs["has_behaviour"] is False and fma.empty_sessions(12).shape == (0, 12)


def test_fma_feature_blocks_and_corpus_contract(tmp_path):
    ids, _ = make_fma(tmp_path)
    t = fma.load_tracks(tmp_path / "tracks.csv", tmp_path / "genres.csv")
    blocks = fma.feature_blocks(t, tmp_path / "features.csv", tmp_path / "echonest.csv")
    assert set(blocks) == {"timbre", "spectral_shape", "harmony_tonal", "dynamics_loudness", "rhythm_groove", "heldout_echonest"}
    assert blocks["timbre"].X.shape == (11, 40) and blocks["timbre"].columns[0] == "mfcc.mean[01]"      # NaN row dropped; mean+std of 20 coefs
    assert blocks["spectral_shape"].X.shape[1] == 2 * (1 + 1 + 7 + 1 + 1) and blocks["harmony_tonal"].X.shape[1] == 2 * (36 + 6)
    assert blocks["dynamics_loudness"].X.shape[1] == 2 and blocks["rhythm_groove"].columns == ["tempo", "danceability"]
    assert blocks["rhythm_groove"].rows.tolist() == list(range(6)) and blocks["heldout_echonest"].X.shape == (6, 6)
    assert 3 not in blocks["timbre"].rows and 3 in blocks["dynamics_loudness"].rows and blocks["dynamics_loudness"].rows.size == 12
    assert blocks["timbre"].rows.tolist() == [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11]
    allstats = fma.feature_blocks(t, fma.load_features(tmp_path / "features.csv"), stats=fma.load_features(tmp_path / "features.csv").columns.levels[1],
                                  include_echonest=False)
    assert sum(b.X.shape[1] for b in allstats.values()) == 518                                             # every librosa column used exactly once
    assert set(fma.feature_blocks(t, tmp_path / "features.csv", families=["timbre"], include_echonest=False)) == {"timbre"}
    c = Corpus("fma", t, fma.empty_sessions(len(t)), blocks)
    c.validate()
    write_corpus(c, tmp_path / "corpus")
    c2 = read_corpus(tmp_path / "corpus")
    assert c2.sessions.shape == (0, 12) and set(c2.families) == set(blocks) and c2.tracks["ext_id"].tolist() == t["ext_id"].tolist()


def test_fma_align_report_says_not_alignable(tmp_path):
    make_fma(tmp_path)
    feat = fma.load_features(tmp_path / "features.csv")
    cols = fma.feature_columns(feat)
    assert cols[0] == "chroma_cens.kurtosis[1]" and len(cols) == 518
    r = fma.align_report(cols)
    assert r["can_use_identical_definition"] is False and r["any_family_aligned"] is False                      # PLAN S11: must be reported
    st = {k: v["status"] for k, v in r["families"].items()}
    assert st == {"timbre": "conceptual_only", "spectral_shape": "conceptual_only", "rhythm_groove": "not_alignable",
                  "harmony_tonal": "conceptual_only", "dynamics_loudness": "conceptual_only"}
    assert "Track B" in r["recommendation"] and "do NOT project" in r["recommendation"]
    r2 = fma.align_report(cols + ["tempo", "danceability"])
    assert r2["families"]["rhythm_groove"]["status"] == "conceptual_only"
    # against the Onion key list shipped in the repo (PLAN §13.4)
    onion = Path(__file__).resolve().parents[1] / "essentia_descriptor_names.json"
    if onion.exists():
        assert fma.align_report(cols, json.loads(onion.read_text()))["can_use_identical_definition"] is False


# --------------------------------------------------------------------------------------------------------- Jamendo
TSV_HEAD = "TRACK_ID\tARTIST_ID\tALBUM_ID\tPATH\tDURATION\tTAGS\n"


def make_jamendo(tmp: Path):
    rows = [("track_0000948", "artist_000087", "album_000149", "48/948.mp3", 212.7, ["mood/theme---background"]),
            ("track_0000950", "artist_000087", "album_000149", "50/950.mp3", 248.0, ["genre---rock", "instrument---guitar", "mood/theme---happy,calm"]),
            ("track_0002165", "artist_000326", "album_000347", "65/2165.mp3", 229.0, ["genre---pop", "genre---electronic"]),
            ("track_0002263", "artist_000320", "album_000366", "63/2263.mp3", 494.7, ["instrument---piano", "bogus---x", "no-hyphen"]),
            ("track_0003000", "artist_000320", "album_000366", "0/3000.mp3", 100.0, [])]
    (tmp / "autotagging.tsv").write_text(TSV_HEAD + "".join("\t".join(map(str, r[:5])) + "".join("\t" + t for t in r[5]) + "\n" for r in rows) + "\n",
                                         encoding="utf-8")
    (tmp / "split-train.tsv").write_text(TSV_HEAD + "track_0000948\ta\tb\tp\t1\tgenre---x\n", encoding="utf-8")
    (tmp / "split-test.tsv").write_text(TSV_HEAD + "track_0002165\ta\tb\tp\t1\tgenre---x\n", encoding="utf-8")
    return rows


def test_jamendo_tsv_variable_tag_columns_and_categories(tmp_path):
    rows = make_jamendo(tmp_path)
    raw = jamendo.read_tsv(tmp_path / "autotagging.tsv")
    assert len(raw) == 5 and raw["tags"].map(len).tolist() == [1, 3, 2, 3, 0] and raw["duration"].iloc[0] == 212.7
    assert jamendo.parse_tags(rows[1][5]) == {"genre": ["rock"], "instrument": ["guitar"], "mood/theme": ["happy", "calm"]}
    t = jamendo.load_tracks(tmp_path / "autotagging.tsv", split_files={"train": tmp_path / "split-train.tsv", "test": tmp_path / "split-test.tsv"})
    assert t["ext_id"].iloc[0] == "track_0000948" and t["artist_id"].tolist() == [87, 87, 326, 320, 320] and t["artist_id"].dtype == np.int64
    assert t["duration_ms"].iloc[0] == pytest.approx(212_700, rel=1e-6) and t["path"].iloc[2] == "65/2165.mp3"
    assert t["genre"].tolist() == ["", "rock", "pop|electronic", "", ""] and t["mood_theme"].iloc[1] == "happy|calm" and t["instrument"].iloc[3] == "piano"
    assert t["split"].tolist() == ["train", "", "test", "", ""]
    assert t["tags"].iloc[1] == "genre---rock|instrument---guitar|mood/theme---happy|mood/theme---calm"
    assert t.attrs["has_names"] is False and t["title"].iloc[0] == "track_0000948"                          # ids only in the TSVs
    Y, names = jamendo.tag_matrix(t)
    assert Y.shape == (5, len(names)) and "genre---rock" in names and Y[1].sum() == 4
    Yg, gn = jamendo.tag_matrix(t, "genre")
    assert gn == ["genre---electronic", "genre---pop", "genre---rock"] and Yg.sum() == 3
    (tmp_path / "names.csv").write_text("track_id,artist,title\ntrack_0000948,Some Band,Nice Song\n", encoding="utf-8")
    tn = jamendo.load_tracks(tmp_path / "autotagging.tsv", names_path=tmp_path / "names.csv")
    assert tn["title"].iloc[0] == "Nice Song" and tn["artist"].iloc[0] == "Some Band" and tn.attrs["has_names"] is True
    c = Corpus("jamendo", t, jamendo.empty_sessions(len(t)))
    c.validate()


def test_jamendo_headerless_and_empty(tmp_path):
    (tmp_path / "h.tsv").write_text("track_0000001\tartist_000001\talbum_000001\t1/1.mp3\t10.5\tgenre---jazz\n", encoding="utf-8")
    assert jamendo.read_tsv(tmp_path / "h.tsv")["tags"].iloc[0] == ["genre---jazz"]
    (tmp_path / "e.tsv").write_text("", encoding="utf-8")
    assert len(jamendo.read_tsv(tmp_path / "e.tsv")) == 0


def essentia_doc(rng, stat="stdev", skip=()):
    """Essentia music-extractor style document (statistics mean/stdev) covering the AcousticBrainz-based specs."""
    doc = {"metadata": {"version": {"essentia": "2.1-beta5"}}}

    def setp(path, value):
        parts = path.split("."); node = doc
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        if isinstance(node.get(parts[-1]), dict) and isinstance(value, dict):
            node[parts[-1]].update(value)
        else:
            node[parts[-1]] = value

    for fam, specs in mb.AB_FAMILY_SPECS.items():
        if fam in skip:
            continue
        for s in specs:
            if s.kind == "key":
                setp(s.path, "D")
            elif s.kind == "scale":
                setp(s.path, "minor")
            elif s.kind == "cov_diag":
                setp(s.path, {"cov": np.diag(rng.random(s.length) + 1).tolist()})
            elif not s.stats:
                setp(s.path, float(rng.normal()))
            else:
                setp(s.path, {("stdev" if st == "stdev" else st): (rng.random(s.length).tolist() if s.kind == "vector" else float(rng.random()))
                              for st in s.stats})
    return doc


def test_jamendo_feature_blocks_from_dir_and_tar_and_alignment(tmp_path):
    make_jamendo(tmp_path)
    t = jamendo.load_tracks(tmp_path / "autotagging.tsv")
    rng = np.random.default_rng(0)
    d = tmp_path / "feat"
    for tid in (948, 950, 2165, 2263):
        (d / str(tid % 100)).mkdir(parents=True, exist_ok=True)
        (d / str(tid % 100) / f"{tid}.json").write_text(json.dumps(essentia_doc(rng)), encoding="utf-8")
    (d / "0").mkdir(exist_ok=True); (d / "0" / "9999.json").write_text(json.dumps(essentia_doc(rng)))        # not in the tracks table
    blocks, names = jamendo.feature_blocks(t, d)
    assert set(blocks) == set(mb.FAMILY_NAMES) and blocks["timbre"].rows.tolist() == [0, 1, 2, 3]
    assert all(b.columns == mb.ab_columns()[k] and not np.isnan(b.X).any() and b.X.dtype == np.float32 for k, b in blocks.items())
    rep = jamendo.align_report(names)
    assert rep["can_use_identical_definition"] is True and all(v["status"] == "aligned" for v in rep["families"].values())
    # the same documents packed as a .tar.gz stream give identical blocks
    tg = tmp_path / "acousticbrainz-0.tar.gz"
    with tarfile.open(tg, "w:gz") as tf:
        for p in sorted(d.rglob("*.json")):
            tf.add(p, arcname=f"{p.parent.name}/{p.name}")
    b2, _ = jamendo.feature_blocks(t, tg)
    assert np.allclose(b2["rhythm_groove"].X, blocks["rhythm_groove"].X)
    # a catalog whose extractor lacks tonal + loudness descriptors is *not* fully alignable, and the report says which
    d2 = tmp_path / "feat2"; d2.mkdir()
    for tid in (948, 950):
        (d2 / f"{tid}.json").write_text(json.dumps(essentia_doc(rng, skip=("harmony_tonal", "dynamics_loudness"))))
    b3, n3 = jamendo.feature_blocks(t, d2)
    assert "harmony_tonal" not in b3 or b3["harmony_tonal"].rows.size == 0
    r3 = jamendo.align_report(n3)
    assert r3["can_use_identical_definition"] is False and r3["families"]["harmony_tonal"]["status"] == "not_alignable" \
        and r3["families"]["timbre"]["status"] == "aligned"
    empty, _ = jamendo.feature_blocks(t, iter([]))
    assert empty == {}
