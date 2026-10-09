import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from melomane.adapters import matching as mt
from melomane.corpus import Corpus, FamilyBlock


def _tracks(rows):
    return pd.DataFrame(rows, columns=["artist", "title", "popularity"]).rename_axis("raw_id")


TARGET = _tracks([
    ("The Beatles", "Yesterday - Remastered 2009", 80),          # 0  remaster suffix
    ("Drake feat. Rihanna", "Take Care (feat. Rihanna)", 70),    # 1  featuring-artist noise on both fields
    ("Björk", "Jóga", 60),                                        # 2  unicode
    ("Queen", "Bohemian Rhapsody", 90),                           # 3  exact
    ("Queen", "Bohemian Rhapsody (Live)", 40),                # 4  live: must NOT match strict
    ("Nirvana", "Smells Like Teen Spirit", 85),                   # 5  exact, duplicated on the source side
    ("Coldplay", "Yellow", 50),                                   # 6  fuzzy typo
    ("Coldplay", "Yellow Remix", 10),                             # 7  version word: no fuzzy
    ("Radiohead", "Paranoid Android", 75),                        # 8  no counterpart
    ("Muse", "Hysteria (Remastered)", 55),                        # 9
    ("Various Artists", "Party Mix 1", 5),                        # 10
    ("Daft Punk", "Around the World", 65),                        # 11 fuzzy 'Around The World' vs 'Around World' -> below thr
    ("Adele", "Rolling in the Deep", 88),                         # 12 fuzzy: 'Rolling in the Deep' vs 'Rolling In The Deepp'
])
SOURCE = _tracks([
    ("Beatles, The", "Yesterday", 10),                            # 0
    ("Drake", "Take Care", 20),                                   # 1
    ("Bjork", "Joga", 30),                                        # 2
    ("Queen", "Bohemian Rhapsody", 5),                            # 3
    ("Nirvana", "Smells Like Teen Spirit", 1),                    # 4
    ("Nirvana", "Smells Like Teen Spirit (Remastered)", 99),      # 5 -> popularity tie-break => 5
    ("Coldplay", "Yelow", 10),                                    # 6
    ("Muse", "Hysteria", 40),                                     # 7
    ("Daft Punk", "Harder Better Faster Stronger", 40),           # 8
    ("Adele", "Rolling in the Deepp", 50),                        # 9
    ("Queen", "Under Pressure", 50),                              # 10
])


def _m(**kw):
    return mt.match_tracks(TARGET, SOURCE, **kw).set_index("a_raw")


def test_exact_unicode_remaster_feat_and_tiebreak():
    m = _m(fuzzy=False)
    assert m.loc[0, "b_raw"] == 0 and m.loc[0, "method"] == "exact" and not m.loc[0, "ambiguous"]      # remaster + 'The'
    assert m.loc[2, "b_raw"] == 2                                                                        # Björk/Jóga
    assert m.loc[3, "b_raw"] == 3
    assert m.loc[9, "b_raw"] == 7                                                                        # (Remastered) stripped
    # duplicate canonical key on the source side -> best popularity wins and the pair is flagged
    assert m.loc[5, "b_raw"] == 5 and bool(m.loc[5, "ambiguous"])
    # 'Drake feat. Rihanna' needs the primary-artist pass (artist differs), title feat is stripped by canonical_key
    assert m.loc[1, "b_raw"] == 1 and m.loc[1, "method"] == "exact_primary_artist"
    assert 4 not in m.index and 8 not in m.index and 10 not in m.index                                   # live / absent / VA


def test_fuzzy_within_artist_and_version_guard():
    m = _m(fuzzy=True, threshold=0.85)
    assert m.loc[6, "b_raw"] == 6 and m.loc[6, "method"] == "fuzzy" and m.loc[6, "score"] >= 0.85       # Yellow ~ Yelow
    assert m.loc[12, "b_raw"] == 9 and m.loc[12, "method"] == "fuzzy"
    assert 7 not in m.index                                       # 'Yellow Remix' must not fuzz onto anything
    assert 11 not in m.index                                      # different Daft Punk song
    counts = mt.match_tracks(TARGET, SOURCE).attrs["counts"]
    assert counts["by_method"]["fuzzy"] == 2 and counts["by_method"]["exact"] >= 4


def test_fuzzy_never_crosses_artists():
    a = _tracks([("Artist One", "Sunshine Day", 1)])
    b = _tracks([("Artist Two", "Sunshine Day", 1), ("Artist Two", "Sunshine Dayy", 1)])
    assert mt.match_tracks(a, b).empty


def test_no_artist_strings_returns_empty_frame():
    a = _tracks([("", "", 0), ("", "", 0)])
    m = mt.match_tracks(a, SOURCE)
    assert m.empty and list(m.columns) == mt.MATCH_COLUMNS


def test_non_default_index_and_song_level():
    a = TARGET.copy(); a.index = np.arange(len(a)) + 1000
    m = mt.match_tracks(a, SOURCE, level="song", fuzzy=False).set_index("a_raw")
    assert m.loc[1004, "b_raw"] == 3                               # live version merges into the studio song at 'song' level
    assert m.index.min() >= 1000


def _corpus(tracks, sessions, families=None, measured=None, name="c"):
    tr = tracks.reset_index(drop=True).copy()
    tr["ext_id"] = [str(i) for i in range(len(tr))]
    tr["artist_id"] = np.arange(len(tr), dtype=np.int64)
    return Corpus(name, tr, sp.csr_matrix(sessions, dtype=np.float32), families or {}, measured)


def test_enrich_reindexes_blocks_and_coverage():
    rng = np.random.default_rng(0)
    src_fam = FamilyBlock(np.array([0, 1, 2, 3, 4, 5, 6, 7, 9]), rng.normal(size=(9, 4)).astype(np.float32), list("abcd"), "src")
    meas = pd.DataFrame({"bpm": np.arange(len(SOURCE), dtype=np.float32) + 100})
    src = _corpus(SOURCE, np.zeros((1, len(SOURCE))), {"timbre": src_fam}, meas, "onion")
    sess = np.zeros((4, len(TARGET)))
    sess[0, [0, 1, 2, 8, 10]] = 1; sess[1, [3, 4, 5, 8, 8 + 1]] = 1; sess[2, [0, 3, 5, 6, 12]] = 1; sess[3, [8, 10, 11, 7, 4]] = 1
    tgt = _corpus(TARGET, sess, name="p2v")
    matches = mt.match_tracks(tgt.tracks, src.tracks)
    e = mt.enrich(tgt, src, matches)
    e.validate()
    assert e.n_tracks == tgt.n_tracks and (e.sessions != tgt.sessions).nnz == 0
    b = e.families["timbre"]
    assert set(b.rows.tolist()) == set(matches.a_raw.tolist())
    # values moved to the right rows: target 3 <- source 3; target 9 <- source 7
    d = b.dense(tgt.n_tracks)
    np.testing.assert_allclose(d[3], src_fam.X[list(src_fam.rows).index(3)])
    np.testing.assert_allclose(d[9], src_fam.X[list(src_fam.rows).index(7)])
    assert np.isnan(d[8]).all()                                    # unmatched -> absent (partial hub)
    assert e.measured is not None and len(e.measured) == tgt.n_tracks
    assert e.measured.loc[3, "bpm"] == 103 and np.isnan(e.measured.loc[8, "bpm"])
    assert e.meta["enriched_from"] == "onion" and e.name == "p2v+onion"
    rep = mt.coverage_report(tgt, matches, min_sessions=2)
    assert rep["n_matched"] == matches["a_raw"].nunique()
    assert 0 < rep["track_share"] < 1 and 0 < rep["mass_share"] < 1
    assert rep["n_eligible"] >= rep["n_eligible_matched"] > 0
    assert 0 <= rep["share_sessions_half_covered"] <= 1


def test_scales_vectorised_smoke():
    """20k x 3k synthetic catalogues: exercises the vectorised paths (no per-row loops on the big side)."""
    rng = np.random.default_rng(1)
    words = np.array(["love", "night", "fire", "blue", "road", "heart", "rain", "gold", "wild", "dream", "star", "sun"])
    def title(n):
        return [" ".join(rng.choice(words, size=rng.integers(2, 4))) + f" {rng.integers(0, 50)}x" for _ in range(n)]
    art_b = [f"artist {i}" for i in range(300)]
    b = pd.DataFrame({"artist": rng.choice(art_b, 3000), "title": title(3000), "popularity": rng.integers(0, 100, 3000)})
    take = rng.choice(3000, 1000, replace=False)
    a_hit = b.iloc[take].copy(); a_hit["title"] = a_hit["title"] + " (Remastered 2011)"
    a_miss = pd.DataFrame({"artist": [f"other {i % 5000}" for i in range(19000)], "title": title(19000), "popularity": 1})
    a = pd.concat([a_hit, a_miss], ignore_index=True)
    m = mt.match_tracks(a, b, fuzzy=False)
    assert (m["a_raw"] < 1000).all() and m["a_raw"].nunique() == 1000
