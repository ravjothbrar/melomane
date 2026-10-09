import pytest
from melomane.curation import normalize_title, normalize_artist, canonical_key, exact_groups, candidate_groups


@pytest.mark.parametrize("a,b", [
    ("Yesterday - Remastered 2009", "Yesterday"),
    ("Yesterday (2009 Remaster)", "Yesterday"),
    ("Bohemian Rhapsody (Remastered 2011)", "Bohemian Rhapsody"),
    ("Hey Ya! - Radio Edit", "Hey Ya!"),
    ("Song Name (feat. Some Rapper)", "Song Name"),
    ("Song Name [Explicit]", "Song Name"),
    ("Song Name - Single Version", "Song Name"),
    ("Song Name (Deluxe Edition)", "Song Name"),
    ("Don't Stop Me Now (Mono)", "Don't Stop Me Now"),
    ("Café del Mar", "Cafe del Mar"),
    ("Rock & Roll", "Rock and Roll"),
])
def test_strict_merges(a, b):
    assert normalize_title(a) == normalize_title(b)


@pytest.mark.parametrize("a,b", [
    ("Hey Jude (Live)", "Hey Jude"),                 # live is a different recording under the strict policy
    ("Song (Acoustic Version)", "Song"),
    ("Song (Remix)", "Song"), ("Song (Instrumental)", "Song"), ("Song - Cover", "Song"), ("Song (Sped Up)", "Song"),
    ("Song (Live) (Remix)", "Song"),
])
def test_strict_keeps_distinct_recordings(a, b):
    assert normalize_title(a) != normalize_title(b)


def test_song_level_merges_live_and_acoustic_but_never_remix():
    assert normalize_title("Hey Jude (Live)", "song") == normalize_title("Hey Jude", "song")
    assert normalize_title("Hey Jude (Live at Wembley)", "song") == normalize_title("Hey Jude", "song")
    assert normalize_title("Song (Acoustic Version)", "song") == normalize_title("Song", "song")
    assert normalize_title("Song (Remix)", "song") != normalize_title("Song", "song")
    assert normalize_title("Song (Live) [Remastered]", "song") == normalize_title("Song", "song")


def test_artist_normalisation():
    assert normalize_artist("Beatles, The") == normalize_artist("The Beatles") == "beatles"
    assert normalize_artist("Simon & Garfunkel") == normalize_artist("Simon and Garfunkel")
    assert normalize_artist("Beyoncé") == "beyonce"


def test_exact_groups_and_candidates():
    recs = [("The Beatles", "Yesterday"), ("Beatles, The", "Yesterday - Remastered 2009"), ("The Beatles", "Yesterday (Live)"),
            ("Queen", "Bohemian Rhapsody"), ("Queen", "Bohemian Rhapsody (Live)"), ("Queen", "Under Pressure"),
            ("Solo", "Only One")]
    g = exact_groups(recs); assert sorted(map(len, g.values())) == [1, 1, 1, 1, 1, 2]
    keys = [canonical_key(*r) for r in recs]
    cand = candidate_groups(keys, threshold=0.6)
    assert any({0, 1, 2} <= set(c) for c in cand)               # Yesterday variants land in one LLM-review group
    assert any({3, 4} <= set(c) for c in cand)
    assert not any(6 in c for c in cand) and not any(5 in c and 3 in c for c in cand)
