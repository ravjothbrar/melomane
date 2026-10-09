import numpy as np
from melomane.sessions import sessionize, user_profiles


def test_sessionize_gaps_dedupe_and_length_filters():
    # user 0: two bursts separated by 2h; user 1: one burst (with a repeated track)
    user = np.array([0] * 12 + [1] * 6)
    ts = np.array(list(range(0, 600, 60)) + [10_000, 10_060] + list(range(0, 360, 60)))
    track = np.array(list(range(10)) + [3, 4] + [20, 21, 22, 22, 23, 24])
    X, su = sessionize(user, track, ts, gap_minutes=30, min_len=5, max_len=50, n_tracks=30)
    assert X.shape[0] == 2                                        # the 2-track burst of user 0 is dropped (< min_len)
    rows = [set(X[i].indices) for i in range(2)]
    assert set(range(10)) in rows and {20, 21, 22, 23, 24} in rows     # duplicates collapsed
    assert sorted(su.tolist()) == [0, 1]


def test_sessionize_unsorted_input_and_max_len():
    rng = np.random.default_rng(0)
    user = np.zeros(300, int); ts = rng.permutation(300) * 10; track = np.arange(300) % 250
    X, _ = sessionize(user, track, ts, gap_minutes=30, min_len=5, max_len=100, n_tracks=250)
    assert X.shape[0] == 0                                         # one 250-track session > max_len


def test_user_profiles_min_plays():
    user = np.array([0, 0, 0, 1, 1]); track = np.array([1, 1, 2, 3, 3])
    P = user_profiles(user, track, min_plays=2, n_tracks=5)
    assert P[0].indices.tolist() == [1] and P[1].indices.tolist() == [3]
