"""Playlist2vec (Zenodo 10.5281/zenodo.5002584, Papreja 2020) adapter.

The dataset is a MySQL dump (`spotifydb`): album, artist, playlist, track, track_artist1, track_playlist1.  Playlists are our
*documents* (the analogue of Epicure recipes / listening sessions).  There is no audio, no lyrics, no ISRC, no year and no
playlist order, so this corpus supports **Melomane-Cooc at scale**, and Composition/Core only for the tracks that can be
*matched* to a corpus with measured audio/lyrics (`matching.py`, partial-hub regime).

PROVENANCE (report in the paper's Limitations): an unofficial re-scrape via the Spotify developer API - not the official MPD.
"""
from __future__ import annotations
import re, time
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp

from ..corpus import Corpus, write_corpus
from ..runtime import get_logger
from . import mysqldump as md

log = get_logger("melomane.adapters.playlist2vec")

_PAIR = re.compile(r"\('([^']*)','([^']*)'\)")
HEADER = "-- MySQL dump"
TAIL_MARK = b"-- Dump completed on"


# ----------------------------------------------------------------------------------------------------------------------
# locating / waiting for the download
# ----------------------------------------------------------------------------------------------------------------------
@dataclass
class DumpStatus:
    path: Path | None
    complete: bool
    size: int = 0
    note: str = ""


def _sniff(p: Path) -> tuple[bool, bool]:
    """(is_mysqldump, looks_complete) by content, not by name (Chrome names in-flight files 'Unconfirmed 123.crdownload')."""
    try:
        sz = p.stat().st_size
        if sz < 1024:
            return False, False
        if p.suffix == ".gz":
            with open(p, "rb") as f:
                return f.read(2) == b"\x1f\x8b" and ("spotifydb" in p.name.lower() or "playlist" in p.name.lower()), True
        if p.suffix == ".zip":
            return "playlist" in p.name.lower() or "spotify" in p.name.lower(), True
        with open(p, "rb") as f:
            head = f.read(64)
            if not head.startswith(HEADER.encode()):
                return False, False
            f.seek(max(0, sz - 4096))
            tail = f.read()
        return True, TAIL_MARK in tail
    except OSError:
        return False, False


def find_dump(extra_dirs: list[Path] | None = None, env_path: str | None = None, include_downloads: bool = True) -> DumpStatus:
    """Search for the Playlist2vec dump (finished or still downloading).  Schema-only dumps (no INSERT data) are ignored
    because they are tiny (< 50 MB) - the real dump has 66M+ rows."""
    import os
    cands: list[Path] = []
    if env_path:
        cands.append(Path(env_path))
    dirs = list(extra_dirs or [])
    if include_downloads:
        home = Path.home()
        dirs += [home / "Downloads"]
    seen = set()
    for d in dirs:
        if not d.exists() or d in seen:
            continue
        seen.add(d)
        try:
            for p in d.iterdir():
                if p.is_file() and p.stat().st_size > 50_000_000 and p.suffix.lower() in (".sql", ".gz", ".zip", ".crdownload", ".part", ".tmp", ".download", ".partial", ""):
                    cands.append(p)
        except OSError:
            continue
    best: DumpStatus | None = None
    for p in cands:
        if not p.exists():
            continue
        ok, complete = _sniff(p)
        if not ok:
            continue
        st = DumpStatus(p, complete and p.suffix.lower() not in (".crdownload", ".part", ".tmp", ".download", ".partial"), p.stat().st_size,
                        "complete" if complete else "still downloading")
        if best is None or (st.complete, st.size) > (best.complete, best.size):
            best = st
    return best or DumpStatus(None, False, 0, "not found")


def wait_for_dump(extra_dirs, env_path=None, poll_s: float = 60.0, timeout_h: float = 24.0, sleep=time.sleep, include_downloads: bool = True) -> DumpStatus:
    """Block (politely, logging progress) until the dump is complete - the user starts the pipeline while the download runs."""
    t0 = time.time()
    last = -1
    while True:
        st = find_dump(extra_dirs, env_path, include_downloads=include_downloads)
        if st.path is None:
            if time.time() - t0 > 120:
                return st
        elif st.complete:
            return st
        else:
            if st.size != last:
                log.info("Playlist2vec download in progress: %.2f GB so far (%s) ...", st.size / 1e9, st.path.name)
            elif time.time() - t0 > 600 and st.size == last:
                log.warning("download size has not changed for a while - paused? (%s)", st.path)
            last = st.size
        if time.time() - t0 > timeout_h * 3600:
            return st
        sleep(poll_s)


# ----------------------------------------------------------------------------------------------------------------------
# parsing
# ----------------------------------------------------------------------------------------------------------------------
@dataclass
class RawTables:
    album: pd.DataFrame
    artist: pd.DataFrame
    playlist: pd.DataFrame
    track: pd.DataFrame
    ta_track: np.ndarray            # int32 index into track
    ta_artist: np.ndarray           # int32 index into artist
    tp_track: np.ndarray            # int32 index into track
    tp_playlist: np.ndarray         # int32 index into playlist
    stats: dict


def _df(rows, cols, names):
    if not rows:
        return pd.DataFrame(columns=names)
    cols = cols if cols and len(cols) == len(rows[0]) else names
    return pd.DataFrame(rows, columns=cols)


def parse_dump(path: Path, progress_every: int = 5_000_000) -> RawTables:
    """Single streaming pass.  Tables appear in alphabetical order in mysqldump output (album, artist, playlist, track,
    track_artist1, track_playlist1); we rely on that order only for the link tables, which need the id indices - and
    fall back to a second pass if a link table precedes its parents."""
    t0 = time.time()
    dims: dict[str, list] = {"album": [], "artist": [], "playlist": [], "track": []}
    pending_links: dict[str, list] = {"track_artist1": [], "track_playlist1": []}
    tindex = pindex = aindex = None
    ta_t, ta_a, tp_t, tp_p = [], [], [], []
    n_links = {"track_artist1": 0, "track_playlist1": 0}
    dropped = {"track_artist1": 0, "track_playlist1": 0}

    def build_indices():
        nonlocal tindex, pindex, aindex
        if tindex is None and dims["track"]:
            tindex = pd.Index([r[0] for r in dims["track"]])
        if pindex is None and dims["playlist"]:
            pindex = pd.Index([r[0] for r in dims["playlist"]])
        if aindex is None and dims["artist"]:
            aindex = pd.Index([r[0] for r in dims["artist"]])

    def flush_links(kind, values):
        pairs = _PAIR.findall(values)
        if not pairs:
            return
        a = np.array(pairs, dtype=object)
        n_links[kind] += len(pairs)
        if kind == "track_artist1":
            ti, ai = tindex.get_indexer(a[:, 0]), aindex.get_indexer(a[:, 1])
            ok = (ti >= 0) & (ai >= 0)
            dropped[kind] += int((~ok).sum()); ta_t.append(ti[ok].astype(np.int32)); ta_a.append(ai[ok].astype(np.int32))
        else:
            ti, pi = tindex.get_indexer(a[:, 0]), pindex.get_indexer(a[:, 1])
            ok = (ti >= 0) & (pi >= 0)
            dropped[kind] += int((~ok).sum()); tp_t.append(ti[ok].astype(np.int32)); tp_p.append(pi[ok].astype(np.int32))
        if n_links[kind] // progress_every != (n_links[kind] - len(pairs)) // progress_every:
            log.info("  %s: %.1fM rows (%.0fs)", kind, n_links[kind] / 1e6, time.time() - t0)

    deferred = []
    for kind, table, payload in md.iter_statements(path):
        if kind != "insert":
            continue
        if table in dims:
            dims[table].extend(md.iter_rows(payload))
            continue
        if table in pending_links:
            build_indices()
            need = {"track_artist1": (tindex, aindex), "track_playlist1": (tindex, pindex)}[table]
            if any(x is None for x in need):
                deferred.append((table, payload))         # parent tables not seen yet (unusual dump order)
                continue
            flush_links(table, payload)
    if deferred:
        build_indices()
        for table, payload in deferred:
            flush_links(table, payload)
    build_indices()
    log.info("parsed dimension tables in %.0fs: %d tracks, %d playlists, %d artists, %d albums",
             time.time() - t0, len(dims["track"]), len(dims["playlist"]), len(dims["artist"]), len(dims["album"]))
    cat = lambda xs: np.concatenate(xs) if xs else np.zeros(0, np.int32)
    return RawTables(
        _df(dims["album"], None, ["id", "name", "uri"]), _df(dims["artist"], None, ["id", "name", "uri"]),
        _df(dims["playlist"], None, ["id", "name", "followers", "uri", "total_tracks"]),
        _df(dims["track"], None, ["id", "name", "duration", "popularity", "explicit", "preview_url", "uri", "album_id"]),
        cat(ta_t), cat(ta_a), cat(tp_t), cat(tp_p),
        {"track_artist_rows": n_links["track_artist1"], "track_playlist_rows": n_links["track_playlist1"],
         "dropped_ta": dropped["track_artist1"], "dropped_tp": dropped["track_playlist1"], "parse_seconds": time.time() - t0})


# ----------------------------------------------------------------------------------------------------------------------
# corpus construction
# ----------------------------------------------------------------------------------------------------------------------
def build_corpus(t: RawTables, min_playlist_len: int = 5, max_playlist_len: int = 300, name: str = "playlist2vec",
                 source_path: str = "") -> Corpus:
    n_tr = len(t.track)
    fix = lambda s: md.fix_mojibake(s) if isinstance(s, str) else ""
    tracks = pd.DataFrame({"ext_id": t.track["id"].astype(str).values})
    tracks["title"] = [fix(x) for x in t.track["name"].tolist()]
    # primary artist = first link row per track (dump order is stable); all artists kept for collaboration flags
    order = np.argsort(t.ta_track, kind="stable")
    ta_t, ta_a = t.ta_track[order], t.ta_artist[order]
    first = np.full(n_tr, -1, dtype=np.int64)
    if ta_t.size:
        uniq, idx = np.unique(ta_t, return_index=True)
        first[uniq] = ta_a[idx]
    n_art = np.bincount(ta_t, minlength=n_tr).astype(np.int16) if ta_t.size else np.zeros(n_tr, np.int16)
    artist_names = np.array([fix(x) for x in t.artist["name"].tolist()] + [""], dtype=object)
    tracks["artist_id"] = np.where(first >= 0, first, -1).astype(np.int64)
    # tracks with no artist link get a unique negative id so they never collapse into one "artist" group
    miss = tracks["artist_id"] < 0
    tracks.loc[miss, "artist_id"] = -(np.arange(int(miss.sum())) + 1)
    tracks["artist"] = np.where(first >= 0, artist_names[np.where(first >= 0, first, -1)], "")
    tracks["n_artists"] = n_art
    album_name = dict(zip(t.album["id"].astype(str), t.album["name"].map(fix)))
    tracks["album"] = [album_name.get(a, "") for a in t.track["album_id"].astype(str).tolist()]
    tracks["album_id"] = t.track["album_id"].astype(str).values
    tracks["duration_ms"] = pd.to_numeric(t.track["duration"], errors="coerce").astype("float32").values
    tracks["popularity"] = pd.to_numeric(t.track["popularity"], errors="coerce").astype("float32").values
    tracks["explicit"] = t.track["explicit"].astype(str).str.lower().isin(["1", "true", "yes"]).values
    # sessions = playlists with tp links; length filter on *linked* distinct tracks
    n_pl = len(t.playlist)
    X = sp.csr_matrix((np.ones(t.tp_track.size, np.float32), (t.tp_playlist, t.tp_track)), shape=(n_pl, n_tr))
    X.sum_duplicates(); X.data[:] = 1.0
    lens = np.diff(X.indptr)
    keep = (lens >= min_playlist_len) & (lens <= max_playlist_len)
    X = X[np.flatnonzero(keep)].tocsr()
    pm = t.playlist.iloc[np.flatnonzero(keep)].reset_index(drop=True)
    sessions_meta = pd.DataFrame({"playlist_id": pm["id"].astype(str), "name": [fix(x) for x in pm["name"].tolist()],
                                  "followers": pd.to_numeric(pm["followers"], errors="coerce").astype("float32"),
                                  "n_tracks": np.diff(X.indptr).astype(np.int32)})
    tracks["n_playlists"] = np.asarray(X.sum(axis=0)).ravel().astype(np.int32)
    meta = {"source": "Playlist2vec (Zenodo 10.5281/zenodo.5002584; Papreja 2020; unofficial Spotify-API re-scrape)",
            "source_path": source_path, "document_type": "playlist (unordered set)", "n_playlists_raw": int(n_pl),
            "n_playlists_kept": int(X.shape[0]), "playlist_len_range": [min_playlist_len, max_playlist_len],
            "has_audio": False, "has_lyrics": False, "has_year": False, "has_order": False,
            "notes": ["No audio/lyrics/year/ISRC in this dump: compound (Composition/Core) models require matching to a "
                      "feature corpus (partial-hub regime)."], **t.stats}
    c = Corpus(name, tracks, X, {}, None, None, {}, sessions_meta, meta)
    c.validate()
    return c


def ingest(ws, dump: Path, min_playlist_len: int = 5, max_playlist_len: int = 300, name: str = "playlist2vec") -> Corpus:
    log.info("parsing %s (%.2f GB) ...", dump, dump.stat().st_size / 1e9)
    t = parse_dump(dump)
    c = build_corpus(t, min_playlist_len, max_playlist_len, name, str(dump))
    write_corpus(c, ws.d("interim", name, base="data"))
    log.info("corpus written: %d tracks, %d playlists (kept), %d links", c.n_tracks, c.sessions.shape[0], c.sessions.nnz)
    return c
