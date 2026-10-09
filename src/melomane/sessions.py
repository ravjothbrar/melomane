"""Listening logs -> co-occurrence 'documents'.

Epicure's documents are recipes (~10 ingredients).  The closest behavioural analogue of a recipe is a *listening
session* (tracks played together within a burst of activity), not a whole user profile (thousands of tracks).  Default:
sessions split at gaps > `gap_minutes`; sessions with < `min_len` or > `max_len` distinct tracks are dropped (too short
to carry co-occurrence / likely shuffle-of-library or bot runs).  A user-profile document is available for comparison.
"""
from __future__ import annotations
import numpy as np
import scipy.sparse as sp


def sessionize(user: np.ndarray, track: np.ndarray, ts: np.ndarray, gap_minutes: float = 30.0,
               min_len: int = 5, max_len: int = 200, n_tracks: int | None = None, return_start: bool = False):
    """Return (X, session_user): X = binary CSR (n_sessions x n_tracks); ts in seconds.  With `return_start=True` also returns the
    session start time (int64 seconds; used by the E6 temporal-stability extension) as a third element."""
    user = np.asarray(user); track = np.asarray(track); ts = np.asarray(ts, dtype=np.int64)
    order = np.lexsort((ts, user)); user, track, ts = user[order], track[order], ts[order]
    new = np.r_[True, (user[1:] != user[:-1]) | ((ts[1:] - ts[:-1]) > gap_minutes * 60)]
    sid = np.cumsum(new) - 1
    n_tracks = int(track.max()) + 1 if n_tracks is None else n_tracks
    key = np.unique(sid.astype(np.int64) * n_tracks + track)            # distinct (session, track)
    s, t = key // n_tracks, key % n_tracks
    length = np.bincount(s, minlength=sid.max() + 1)
    ok = (length >= min_len) & (length <= max_len)
    remap = np.cumsum(ok) - 1
    keep = ok[s]
    X = sp.csr_matrix((np.ones(keep.sum(), np.float32), (remap[s[keep]], t[keep])), shape=(int(ok.sum()), n_tracks))
    su = np.zeros(sid.max() + 1, dtype=user.dtype); su[sid] = user
    if return_start:
        t0 = np.zeros(sid.max() + 1, dtype=np.int64); t0[sid[::-1]] = ts[::-1]      # first event of each session
        return X, su[ok], t0[ok]
    return X, su[ok]


def user_profiles(user: np.ndarray, track: np.ndarray, min_plays: int = 2, n_tracks: int | None = None):
    """Whole-user documents (tracks a user played >= min_plays times)."""
    user = np.asarray(user); track = np.asarray(track)
    n_tracks = int(track.max()) + 1 if n_tracks is None else n_tracks
    uu, ui = np.unique(user, return_inverse=True)
    key, cnt = np.unique(ui.astype(np.int64) * n_tracks + track, return_counts=True)
    key = key[cnt >= min_plays]
    return sp.csr_matrix((np.ones(len(key), np.float32), (key // n_tracks, key % n_tracks)), shape=(len(uu), n_tracks))
