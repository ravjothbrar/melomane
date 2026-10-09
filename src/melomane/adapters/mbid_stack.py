"""Track B — MBID-native open stack (PLAN §4.2 / S10):  ListenBrainz + MusicBrainz + AcousticBrainz + LRCLIB.

Everything here is *streaming*, *resumable* (per-archive / per-month shards + JSON manifests) and testable offline
(all network access goes through injectable transports).  Windows-safe: pathlib only, no symlinks, no fork.

VERIFICATION STATUS (docs read 2026-09-29; details and URLs in docs/data_sources_addendum.md).  Every docstring below marks
what is **VERIFIED** (read on an official page/listing) versus **UNVERIFIED** (code is defensive; Stage-0 must check on the
real files):

* VERIFIED   ListenBrainz listen JSON: ``listened_at`` (unix seconds), ``track_metadata.additional_info.recording_mbid`` /
             ``artist_mbids``; server-side ``mbid_mapping`` (``recording_mbid``, ``recording_name``, ``artist_mbids``,
             ``artists[{artist_mbid, artist_credit_name, join_phrase}]``) is *response only* in the submission docs.
             Dump layout: ``listenbrainz-listens-dump-<n>-<date>-<time>-full.tar.zst`` holding ``listens/YYYY/M.listens``,
             one JSON document per line.
* UNVERIFIED whether the dump lines carry ``mbid_mapping`` and ``user_name`` under exactly these keys (docs excerpt shows no
             sample dump line) -> parser tries mapping first, then additional_info, and falls back to ``user_id``.
* VERIFIED   MusicBrainz canonical dumps: ``canonical_musicbrainz_data.csv`` (id, artist_credit_id, artist_mbids,
             artist_credit_name, release_mbid, release_name, recording_mbid, recording_name, combined_lookup, score, year),
             ``canonical_recording_redirect.csv`` (recording_mbid, canonical_recording_mbid, canonical_release_mbid), CC0.
* UNVERIFIED encoding of ``artist_mbids`` (Postgres ``{a,b}`` array vs JSON vs comma list), direction of ``score``, whether
             ``year`` is the first-release year, and the member names inside the .tar.zst.
* VERIFIED   AcousticBrainz 2022-06-23 dumps: ``type/mb/i/mbid-n.json`` inside 30 lowlevel archives; CSV feature dumps with
             columns mbid, submission_offset + {average_loudness, dynamic_complexity, mfcc_zero_mean} /
             {bpm, bpm_histogram_{first,second}_peak_bpm_{mean,median}, danceability, onset_rate} /
             {key_key, key_scale, tuning_frequency, tuning_equal_tempered_deviation}; low-level statistics are
             mean/median/min/max/var/dmean/dmean2/dvar/dvar2 (NOT ``stdev``).
* VERIFIED   LRCLIB ``GET /api/get`` and ``GET /api/search?q=`` return JSON with id, name, trackName, artistName,
             albumName, duration, instrumental, plainLyrics, syncedLyrics (+ hasWordSync, lyricsfile in 2026).
* UNVERIFIED LRCLIB rate limit / User-Agent policy (docs page is a JS app, unreadable) and the SQLite dump schema.

PRIVACY / LICENCE RULES enforced in code: user names are HMAC-hashed to 63-bit ints at parse time and never stored; lyrics
text is reduced to (presence, counts, aggregate embedding) inside ``LrclibClient``/``LrclibDump`` and never written.
"""
from __future__ import annotations

import bz2
import contextlib
import dataclasses
import gzip
import hashlib
import hmac
import io
import json
import math
import os
import re
import secrets
import sqlite3
import tarfile
import time
import unicodedata
import warnings
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO, Callable, Iterable, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp

from ..corpus import Corpus, FamilyBlock, write_corpus
from ..runtime import get_logger, read_json, sha256_file, write_json
from ..sessions import sessionize

if TYPE_CHECKING:                       # pragma: no cover
    from ..workspace import Workspace

log = get_logger("melomane.adapters.mbid_stack")

FAMILY_NAMES = ["timbre", "spectral_shape", "rhythm_groove", "harmony_tonal", "dynamics_loudness"]   # PLAN §4.4 acoustic
LYRIC_FAMILY = "lyric_theme"                                                                          # PLAN §4.4 (lyrics)
USER_AGENT_HINT = "Melomane/0.1 ( https://example.org/contact-or-mailto )"   # REPLACE with a real contact before running


# ======================================================================================================================
# 0. small utilities: MBIDs, streams, tar members, jsonl
# ======================================================================================================================
class ZstdUnavailable(ImportError):
    """Raised when a .zst file is opened but the ``zstandard`` package is not installed."""


_HEX32 = re.compile(r"^[0-9a-f]{32}$")


def mbid_bytes(s) -> bytes | None:
    """UUID string (hyphenated or bare hex, any case) -> 16 raw bytes; None if not a UUID."""
    if isinstance(s, (bytes, bytearray)):
        return bytes(s) if len(s) == 16 else None
    if not isinstance(s, str):
        return None
    t = s.strip().lower()
    if len(t) == 36 and t[8] == t[13] == t[18] == t[23] == "-":
        t = t.replace("-", "")
    return bytes.fromhex(t) if _HEX32.match(t) else None


def mbid_str(b: bytes) -> str:
    h = bytes(b).hex()
    return f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:]}"


def to_mbid_array(mbids: Iterable) -> np.ndarray:
    """Iterable of UUID strings/bytes -> ``S16`` array (invalid entries raise ValueError).

    NOTE numpy strips trailing NUL bytes when an *element* of an ``S16`` array is read; always go through
    :func:`mbid_array_to_str` / ``tobytes`` (sorting, unique, searchsorted and isin are unaffected)."""
    out = []
    for m in mbids:
        b = mbid_bytes(m)
        if b is None:
            raise ValueError(f"not an MBID: {m!r}")
        out.append(b)
    return np.frombuffer(b"".join(out), dtype="S16").copy() if out else np.zeros(0, dtype="S16")


def mbid_array_to_str(arr: np.ndarray) -> list[str]:
    raw = np.ascontiguousarray(arr, dtype="S16").tobytes()
    return [mbid_str(raw[i:i + 16]) for i in range(0, len(raw), 16)]


def _mbid_set(arr: np.ndarray) -> set[bytes]:
    raw = np.ascontiguousarray(arr, dtype="S16").tobytes()
    return {raw[i:i + 16] for i in range(0, len(raw), 16)}


def _lookup(sorted_keys: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Index of each query in ``sorted_keys`` or -1 (both S16, keys unique + sorted)."""
    if sorted_keys.size == 0 or query.size == 0:
        return np.full(query.shape, -1, dtype=np.int64)
    pos = np.minimum(np.searchsorted(sorted_keys, query), sorted_keys.size - 1)
    return np.where(sorted_keys[pos] == query, pos, -1).astype(np.int64)


def _compression(p: Path) -> tuple[str | None, bool]:
    n = p.name.lower()
    is_tar = ".tar" in n or n.endswith((".tgz", ".tbz2", ".txz"))
    for ext, comp in ((".zst", "zst"), (".zstd", "zst"), (".gz", "gz"), (".tgz", "gz"), (".bz2", "bz2"), (".tbz2", "bz2"),
                      (".xz", "xz"), (".txz", "xz")):
        if n.endswith(ext):
            return comp, is_tar
    return None, is_tar


@contextlib.contextmanager
def open_binary(path: str | Path) -> Iterator[BinaryIO]:
    """Open a (possibly zst/gz/bz2/xz-compressed) file as a *decompressed* binary stream.  A ``.zst`` file needs the
    optional ``zstandard`` package (clear :class:`ZstdUnavailable` otherwise)."""
    p = Path(path)
    comp, _ = _compression(p)
    if comp == "zst":
        try:
            import zstandard
        except ImportError as e:
            raise ZstdUnavailable(f"{p.name} is zstd-compressed: `pip install zstandard` (needed for ListenBrainz / "
                                  "AcousticBrainz / MusicBrainz canonical dumps)") from e
        raw = open(p, "rb")
        try:
            rd = zstandard.ZstdDecompressor(max_window_size=2 ** 31).stream_reader(raw, read_across_frames=True)
            yield io.BufferedReader(rd, 1 << 20)          # type: ignore[arg-type]
        finally:
            raw.close()
    elif comp == "gz":
        with gzip.open(p, "rb") as f:
            yield f                                        # type: ignore[misc]
    elif comp == "bz2":
        with bz2.open(p, "rb") as f:
            yield f                                        # type: ignore[misc]
    elif comp == "xz":
        import lzma
        with lzma.open(p, "rb") as f:
            yield f                                        # type: ignore[misc]
    else:
        with open(p, "rb") as f:
            yield f


def iter_members(path: str | Path, member_filter: Callable[[str], bool] | None = None) -> Iterator[tuple[str, BinaryIO]]:
    """Stream ``(member_name, binary_file)`` from a tar(.zst/.gz/.bz2/.xz) *without* extracting to disk, or yield the
    single decompressed file for a non-tar path.  The yielded file object is only valid until the next iteration
    (tar stream mode); members rejected by ``member_filter`` are skipped without being read."""
    p = Path(path)
    comp, is_tar = _compression(p)
    if not is_tar:
        with open_binary(p) as f:
            if member_filter is None or member_filter(p.name):
                yield p.name, f
        return
    with contextlib.ExitStack() as st:
        if comp == "zst":
            stream = st.enter_context(open_binary(p))
            tf = st.enter_context(tarfile.open(fileobj=stream, mode="r|"))
        else:
            tf = st.enter_context(tarfile.open(p, mode="r|*"))
        for m in tf:
            if not m.isfile():
                continue
            if member_filter is not None and not member_filter(m.name):
                continue
            f = tf.extractfile(m)
            if f is not None:
                yield m.name, f


def iter_json_lines(f: BinaryIO, stats: "ExtractStats | None" = None) -> Iterator[dict]:
    """JSONL reader tolerant of blank / broken lines (counted in ``stats.bad_lines``)."""
    for raw in f:
        raw = raw.strip()
        if not raw:
            continue
        try:
            obj = json.loads(raw)
        except ValueError:
            if stats is not None:
                stats.bad_lines += 1
            continue
        if isinstance(obj, dict):
            yield obj


# ======================================================================================================================
# 1. ListenBrainz listens -> (hashed user, recording MBID, timestamp)
# ======================================================================================================================
class UserHasher:
    """Keyed one-way hash of user names -> non-negative 63-bit ints (HMAC-SHA256, first 8 bytes).

    The secret salt lives in a private file *outside* every corpus artefact (``from_file``), so a leaked corpus cannot be
    re-identified by dictionary attack, while re-runs (resume) stay consistent.  User names are never stored anywhere:
    only the int survives ``parse_listen``."""

    def __init__(self, salt: bytes):
        if len(salt) < 16:
            raise ValueError("salt must be >= 16 bytes")
        self._salt = bytes(salt)
        self._cache: dict[str, int] = {}

    @classmethod
    def from_file(cls, path: str | Path) -> "UserHasher":
        p = Path(path)
        if not p.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(secrets.token_bytes(32))
            with contextlib.suppress(OSError):
                os.chmod(p, 0o600)
        return cls(p.read_bytes())

    def __call__(self, user) -> int:
        k = str(user)
        v = self._cache.get(k)
        if v is None:
            if len(self._cache) > 2_000_000:          # bounded memory; names are dropped with the cache
                self._cache.clear()
            d = hmac.new(self._salt, k.encode("utf-8"), hashlib.sha256).digest()
            v = self._cache[k] = int.from_bytes(d[:8], "big") & 0x7FFFFFFFFFFFFFFF
        return v

    @property
    def fingerprint(self) -> str:
        """Short id of the salt (for detecting a changed salt on resume); reveals nothing usable."""
        return hmac.new(self._salt, b"melomane-fingerprint", hashlib.sha256).hexdigest()[:12]


@dataclass
class ExtractStats:
    members_done: int = 0
    members_skipped: int = 0
    listens_seen: int = 0
    listens_kept: int = 0
    no_mbid: int = 0
    no_user_or_ts: int = 0
    outside_time: int = 0
    not_in_filter: int = 0
    bad_lines: int = 0
    stopped: bool = False

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


def parse_listen(rec: Mapping, hasher: UserHasher, prefer: str = "mapping"):
    """One ListenBrainz listen -> ``(user_int, recording_mbid_bytes, listened_at, meta)`` or None.

    ``meta`` = ``(artist_credit_name, artist_mbid_str, recording_name)`` from ``mbid_mapping`` when present (else None).
    ``prefer`` = "mapping" (server-resolved ``mbid_mapping.recording_mbid``, default: cleaner) | "submitted"
    (``additional_info.recording_mbid``).  UNVERIFIED: dump lines' exact keys - see module docstring."""
    ts = rec.get("listened_at")
    user = rec.get("user_name")
    if user is None:
        user = rec.get("user_id")
    if ts is None or user is None:
        return None
    tm = rec.get("track_metadata") or {}
    mp = tm.get("mbid_mapping") or {}
    info = tm.get("additional_info") or {}
    cands = [mp.get("recording_mbid"), info.get("recording_mbid")]
    if prefer == "submitted":
        cands.reverse()
    mb = None
    for c in cands:
        mb = mbid_bytes(c)
        if mb is not None:
            break
    if mb is None:
        return "no_mbid"
    meta = None
    if mp.get("recording_mbid") and mbid_bytes(mp.get("recording_mbid")) == mb:
        arts = mp.get("artists") or []
        name = "".join((a.get("artist_credit_name") or "") + (a.get("join_phrase") or "") for a in arts if isinstance(a, dict))
        a0 = None
        if arts and isinstance(arts[0], dict):
            a0 = arts[0].get("artist_mbid")
        elif mp.get("artist_mbids"):
            a0 = mp["artist_mbids"][0]
        meta = (name.strip(), a0 if (a0 and mbid_bytes(a0)) else "", mp.get("recording_name") or tm.get("track_name") or "")
    try:
        return hasher(user), mb, int(ts), meta
    except (TypeError, ValueError):
        return None


def _load_manifest(out_dir: Path, hasher: UserHasher) -> dict:
    p = out_dir / "extract_manifest.json"
    if p.exists():
        m = read_json(p)
        if m.get("salt_fingerprint") != hasher.fingerprint:
            raise ValueError("user-hash salt differs from the one used for the existing shards; refusing to mix ids "
                             f"(delete {out_dir} or use the original salt file)")
        return m
    return {"version": 1, "salt_fingerprint": hasher.fingerprint, "done": {}, "next_idx": 0}


def _is_listens_member(name: str) -> bool:
    n = name.lower()
    return n.endswith((".listens", ".jsonl", ".json", ".ndjson")) or n.endswith((".listens.gz", ".jsonl.gz"))


def extract_listens(sources: Sequence[str | Path], out_dir: str | Path, hasher: UserHasher, *,
                    mbid_filter: np.ndarray | None = None, min_ts: int | None = None, max_ts: int | None = None,
                    year_range: tuple[int, int] | None = None, prefer: str = "mapping", flush_rows: int = 2_000_000,
                    collect_meta: bool = True, meta_min_count: int = 2,
                    should_stop: Callable[[], bool] | None = None) -> ExtractStats:
    """Stream ListenBrainz listens (.tar.zst dump / .jsonl / .jsonl.gz / .zst) into resumable shards.

    Output in ``out_dir``: ``L<member>_<k>.npz`` (user int64, mbid S16, ts int64), ``meta_<member>.jsonl.gz`` (recording
    metadata harvested from ``mbid_mapping``, *no user data*) and ``extract_manifest.json``.  A member (= one month file)
    is recorded as done only after all of its shards are flushed; an interrupted member is discarded and redone, so the
    job can be stopped (``should_stop``) or killed at any moment.  Skipping already-done members still costs the
    decompression time of a tar stream (no index) - re-run cost is bounded by one pass over the archive.

    ``mbid_filter``: optional S16 array of MBIDs to keep (cuts a 229 GB dump to the candidate set).  ``year_range``
    filters ``listens/YYYY/M.listens`` members by name before reading."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    man = _load_manifest(out, hasher)
    referenced = {s for v in man["done"].values() for s in v["files"]}
    for f in list(out.glob("L*.npz")) + list(out.glob("meta_*.jsonl.gz")):      # debris of an interrupted member
        if f.name not in referenced:
            f.unlink()
    keep = _mbid_set(mbid_filter) if mbid_filter is not None else None
    st = ExtractStats()

    def stop() -> bool:
        return bool(should_stop and should_stop())

    for src in map(Path, sources):
        for name, f in iter_members(src, _is_listens_member if _compression(src)[1] else None):
            key = f"{src.name}::{name}"
            if key in man["done"]:
                st.members_skipped += 1
                continue
            m = re.search(r"(?:^|/)(\d{4})/(\d{1,2})\.listens", name)
            if year_range and m and not (year_range[0] <= int(m.group(1)) <= year_range[1]):
                st.members_skipped += 1
                continue
            idx = man["next_idx"]
            U: list[int] = []; M: list[bytes] = []; T: list[int] = []
            files: list[str] = []
            metas: dict[bytes, tuple] = {}
            mcount: Counter = Counter()

            def flush():
                if not U:
                    return
                fn = f"L{idx:06d}_{len(files):03d}.npz"
                np.savez(out / fn, user=np.asarray(U, dtype=np.int64), mbid=np.frombuffer(b"".join(M), dtype="S16"),
                         ts=np.asarray(T, dtype=np.int64))
                files.append(fn)
                U.clear(); M.clear(); T.clear()

            n_lines = 0
            aborted = False
            for rec in iter_json_lines(f, st):
                n_lines += 1
                if n_lines % 50_000 == 0 and stop():
                    aborted = True
                    break
                st.listens_seen += 1
                r = parse_listen(rec, hasher, prefer)
                if r is None:
                    st.no_user_or_ts += 1
                    continue
                if r == "no_mbid":
                    st.no_mbid += 1
                    continue
                u, mb, ts, meta = r
                if (min_ts is not None and ts < min_ts) or (max_ts is not None and ts >= max_ts):
                    st.outside_time += 1
                    continue
                if keep is not None and mb not in keep:
                    st.not_in_filter += 1
                    continue
                U.append(u); M.append(mb); T.append(ts)
                st.listens_kept += 1
                if collect_meta and meta is not None:
                    mcount[mb] += 1
                    metas.setdefault(mb, meta)
                if len(U) >= flush_rows:
                    flush()
            if aborted:
                st.stopped = True
                for fn in files:
                    (out / fn).unlink(missing_ok=True)
                log.warning("stopped inside %s; partial member discarded (resumable)", key)
                return st
            flush()
            if collect_meta and metas:
                mfn = f"meta_{idx:06d}.jsonl.gz"
                with gzip.open(out / mfn, "wt", encoding="utf-8") as g:
                    for mb, (a, am, t) in metas.items():
                        if mcount[mb] >= meta_min_count:
                            g.write(json.dumps({"recording_mbid": mbid_str(mb), "artist_credit_name": a,
                                                "artist_mbid": am, "title": t}, ensure_ascii=False) + "\n")
                files.append(mfn)
            man["done"][key] = {"idx": idx, "files": files, "lines": n_lines}
            man["next_idx"] = idx + 1
            write_json(out / "extract_manifest.json", man)
            st.members_done += 1
            log.info("listens: %s done (%d lines, %d kept so far)", key, n_lines, st.listens_kept)
            if stop():
                st.stopped = True
                return st
    return st


@dataclass
class Listens:
    """Anonymised listening events: ``user`` = hashed ints, ``mbid`` = S16, ``ts`` = unix seconds."""
    user: np.ndarray
    mbid: np.ndarray
    ts: np.ndarray

    def __len__(self) -> int:
        return int(self.user.size)

    def _view(self, idx: np.ndarray) -> "Listens":
        return Listens(self.user[idx], self.mbid[idx], self.ts[idx])

    def dedupe(self) -> "Listens":
        """Drop exact duplicates (full dump + incremental overlap)."""
        rec = np.empty(len(self), dtype=[("u", "i8"), ("t", "i8"), ("m", "S16")])
        rec["u"], rec["t"], rec["m"] = self.user, self.ts, self.mbid
        _, first = np.unique(rec, return_index=True)
        return self._view(np.sort(first))

    def time_slice(self, min_ts: int | None = None, max_ts: int | None = None) -> "Listens":
        ok = np.ones(len(self), bool)
        if min_ts is not None:
            ok &= self.ts >= min_ts
        if max_ts is not None:
            ok &= self.ts < max_ts
        return self._view(np.flatnonzero(ok))


def load_listens(out_dir: str | Path, min_ts: int | None = None, max_ts: int | None = None) -> Listens:
    """Concatenate all *completed* shards recorded in the manifest (temporal design: pass ``max_ts`` for train <= 2019,
    ``min_ts`` for eval >= 2020)."""
    out = Path(out_dir)
    man = read_json(out / "extract_manifest.json")
    U, M, T = [], [], []
    for v in man["done"].values():
        for fn in v["files"]:
            if fn.startswith("L") and fn.endswith(".npz"):
                with np.load(out / fn) as z:                 # context manager: no open handles (Windows file locks)
                    U.append(z["user"]); M.append(z["mbid"]); T.append(z["ts"])
    if not U:
        return Listens(np.zeros(0, np.int64), np.zeros(0, "S16"), np.zeros(0, np.int64))
    return Listens(np.concatenate(U), np.concatenate(M), np.concatenate(T)).time_slice(min_ts, max_ts)


def load_mapping_metadata(out_dir: str | Path) -> pd.DataFrame:
    """Recording metadata harvested from ``mbid_mapping`` during extraction (fallback when no MusicBrainz dump is used).
    Index = recording_mbid (str); columns artist_credit_name, artist_mbid, title."""
    rows = []
    for f in sorted(Path(out_dir).glob("meta_*.jsonl.gz")):
        with gzip.open(f, "rt", encoding="utf-8") as g:
            rows.extend(json.loads(line) for line in g if line.strip())
    if not rows:
        return pd.DataFrame(columns=["artist_credit_name", "artist_mbid", "title"]).rename_axis("recording_mbid")
    return pd.DataFrame(rows).drop_duplicates("recording_mbid").set_index("recording_mbid")


def select_vocab(listens: Listens, min_users: int = 5, max_tracks: int | None = None, seed: int = 0) -> np.ndarray:
    """Sorted S16 array of MBIDs listened by >= ``min_users`` distinct users (top-``max_tracks`` by users, ties broken
    with a seeded shuffle)."""
    if not len(listens):
        return np.zeros(0, "S16")
    rec = np.empty(len(listens), dtype=[("m", "S16"), ("u", "i8")])
    rec["m"], rec["u"] = listens.mbid, listens.user
    uniq = np.unique(rec)
    mb, cnt = np.unique(uniq["m"], return_counts=True)
    ok = cnt >= min_users
    mb, cnt = mb[ok], cnt[ok]
    if max_tracks is not None and mb.size > max_tracks:
        rng = np.random.default_rng(seed)
        order = np.lexsort((rng.random(mb.size), -cnt))[:max_tracks]
        mb = mb[np.sort(order)]
    return mb


def listens_to_sessions(listens: Listens, vocab: np.ndarray, gap_minutes: float = 30.0, min_len: int = 5,
                        max_len: int = 200) -> tuple[sp.csr_matrix, np.ndarray]:
    """Listens on ``vocab`` (sorted S16, raw id = position) -> (sessions CSR, hashed user id per session) via
    :func:`melomane.sessions.sessionize`.  Sessions are formed on the vocabulary-filtered stream (documented choice)."""
    ti = _lookup(vocab, listens.mbid)
    ok = ti >= 0
    if not ok.any():
        return sp.csr_matrix((0, vocab.size), dtype=np.float32), np.zeros(0, np.int64)
    return sessionize(listens.user[ok], ti[ok], listens.ts[ok], gap_minutes, min_len, max_len, n_tracks=vocab.size)


# ======================================================================================================================
# 2. MusicBrainz canonical data / recordings / ISRC / offline (artist, title) -> MBID
# ======================================================================================================================
CANONICAL_COLUMNS = ["id", "artist_credit_id", "artist_mbids", "artist_credit_name", "release_mbid", "release_name",
                     "recording_mbid", "recording_name", "combined_lookup", "score", "year"]
RECORDING_COLUMNS = ["artist_credit_name", "artist_mbid", "title", "release_name", "first_release_year", "isrcs", "score"]


def parse_mbid_list(v) -> list[str]:
    """artist_mbids cell -> list of lower-case MBIDs.  Handles Postgres ``{a,b}``, JSON ``["a","b"]``, ``a,b`` and
    python lists.  UNVERIFIED which encoding the dump uses."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return []
    if isinstance(v, (list, tuple, np.ndarray)):
        items = list(v)
    else:
        s = str(v).strip()
        if s.startswith("["):
            try:
                items = json.loads(s)
            except ValueError:
                items = re.findall(r"[0-9a-fA-F-]{32,36}", s)
        else:
            items = re.findall(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}", s) or \
                    [x for x in re.split(r"[,;{}\s]+", s) if x]
    out = []
    for x in items:
        b = mbid_bytes(x)
        if b is not None:
            out.append(mbid_str(b))
    return out


class _Prefixed(io.RawIOBase):
    """Raw stream = ``prefix`` bytes followed by ``f`` (used after peeking the CSV header line)."""

    def __init__(self, prefix: bytes, f: BinaryIO):
        self._p, self._f = prefix, f

    def readable(self) -> bool:
        return True

    def readinto(self, b) -> int:
        if self._p:
            n = min(len(b), len(self._p)); b[:n] = self._p[:n]; self._p = self._p[n:]
            return n
        d = self._f.read(len(b))
        b[:len(d)] = d
        return len(d)


def _csv_chunks(f: BinaryIO, default_columns: Sequence[str], chunksize: int, marker: str) -> Iterator[pd.DataFrame]:
    """CSV reader that copes with a missing header (``marker`` = a column name expected in a real header)."""
    first = f.readline()
    has_header = marker.encode() in first
    src = io.BufferedReader(_Prefixed(first, f), 1 << 20)
    kw = dict(dtype=str, keep_default_na=False, chunksize=chunksize, encoding="utf-8", encoding_errors="replace",
              on_bad_lines="skip")
    if has_header:
        yield from pd.read_csv(src, **kw)
    else:
        yield from pd.read_csv(src, header=None, names=list(default_columns), **kw)


def read_canonical_metadata(path: str | Path, keep_mbids: set[str] | None = None, chunksize: int = 1_000_000,
                            member: str = "canonical_musicbrainz_data.csv") -> Iterator[pd.DataFrame]:
    """Stream ``canonical_musicbrainz_data.csv`` (from the .tar.zst dump, a .csv, .csv.zst or .csv.gz) as normalised
    chunks with columns ``recording_mbid, artist_credit_name, artist_mbid, title, release_name, first_release_year,
    score, combined_lookup`` (~23 M rows in the real file per the canonical-data-example repo).

    ``first_release_year`` is the dump's ``year`` column: UNVERIFIED that it is the year of the *first* release (the
    canonical release is 'often the version on the first album', so it approximates it; reissues can bias it late).
    ``keep_mbids``: lower-case UUID strings to retain (filter early)."""
    def norm(df: pd.DataFrame) -> pd.DataFrame:
        if keep_mbids is not None:
            df = df[df["recording_mbid"].str.lower().isin(keep_mbids)]
        am = df["artist_mbids"].map(parse_mbid_list) if "artist_mbids" in df else pd.Series([[]] * len(df), index=df.index)
        yr = pd.to_numeric(df["year"], errors="coerce") if "year" in df else pd.Series(np.nan, index=df.index)
        yr = yr.where((yr >= 1000) & (yr <= 2100))
        return pd.DataFrame({
            "recording_mbid": df["recording_mbid"].str.lower(),
            "artist_credit_name": df["artist_credit_name"],
            "artist_mbid": am.map(lambda x: x[0] if x else ""),
            "title": df["recording_name"],
            "release_name": df["release_name"] if "release_name" in df else "",
            "first_release_year": yr.astype("float32"),
            "score": pd.to_numeric(df["score"], errors="coerce").astype("float32") if "score" in df else np.float32(0),
            "combined_lookup": df["combined_lookup"] if "combined_lookup" in df else "",
        })

    for name, f in iter_members(path, lambda n: Path(n).name == member if _compression(Path(path))[1] else True):
        for chunk in _csv_chunks(f, CANONICAL_COLUMNS, chunksize, "recording_mbid"):
            yield norm(chunk)


def build_recording_table(chunks: Iterable[pd.DataFrame], isrcs: Mapping[str, list[str]] | None = None) -> pd.DataFrame:
    """Chunks -> ``recording_mbid -> (artist_credit_name, artist_mbid, title, release_name, first_release_year, isrcs,
    score)`` (index = recording_mbid str; duplicates keep the highest score)."""
    parts = [c for c in chunks if len(c)]
    if not parts:
        return pd.DataFrame(columns=RECORDING_COLUMNS).rename_axis("recording_mbid")
    df = pd.concat(parts, ignore_index=True).sort_values("score", ascending=False, kind="stable")
    df = df.drop_duplicates("recording_mbid").set_index("recording_mbid")
    df["isrcs"] = [list((isrcs or {}).get(m, [])) for m in df.index]
    return df[RECORDING_COLUMNS]


def read_mb_recording_json(path: str | Path) -> Iterator[dict]:
    """MusicBrainz JSON dump ``recording.tar.xz`` (one WS/2-style JSON document per line) -> dict rows with
    ``recording_mbid, artist_credit_name, artist_mbid, title, first_release_year, isrcs``.  UNVERIFIED: the dump was
    listed (2026-09-26, recording.tar.xz 32 MB) but no line was inspected; key names follow the MusicBrainz web-service
    JSON (``id``, ``title``, ``artist-credit``, ``first-release-date``, ``isrcs``)."""
    for _, f in iter_members(path, lambda n: not n.endswith(("/", ".asc"))):
        for d in iter_json_lines(f):
            ac = d.get("artist-credit") or []
            name = "".join((c.get("name") or (c.get("artist") or {}).get("name") or "") + (c.get("joinphrase") or "")
                           for c in ac if isinstance(c, dict))
            a0 = (ac[0].get("artist") or {}).get("id") if ac and isinstance(ac[0], dict) else ""
            fr = str(d.get("first-release-date") or "")[:4]
            mb = mbid_bytes(d.get("id"))
            if mb is None:
                continue
            yield {"recording_mbid": mbid_str(mb), "artist_credit_name": name, "artist_mbid": a0 or "",
                   "title": d.get("title") or "", "first_release_year": int(fr) if fr.isdigit() else None,
                   "isrcs": list(d.get("isrcs") or [])}


def read_isrc_tables(isrc_table: str | Path, recording_table: str | Path, *, isrc_cols: tuple[int, int] = (1, 2),
                     recording_cols: tuple[int, int] = (0, 1)) -> dict[str, list[str]]:
    """PostgreSQL-dump route (``mbdump.tar.bz2`` members ``mbdump/isrc`` + ``mbdump/recording``, COPY text format: tab
    separated, ``\\N`` = NULL) -> ``recording_mbid -> [isrc, ...]``.

    VERIFIED ``isrc`` columns: id, recording (int FK), isrc, edits_pending, created  -> ``isrc_cols=(recording, isrc)``.
    UNVERIFIED ``recording`` columns (id, gid, ...): default ``recording_cols=(id, gid)`` is from memory of the schema.
    The recording table is streamed and only ids that own an ISRC are kept."""
    by_id: dict[int, list[str]] = {}
    for _, f in iter_members(isrc_table, lambda n: Path(n).name == "isrc" or not _compression(Path(isrc_table))[1]):
        for raw in f:
            p = raw.decode("utf-8", "replace").rstrip("\n").split("\t")
            try:
                by_id.setdefault(int(p[isrc_cols[0]]), []).append(p[isrc_cols[1]])
            except (ValueError, IndexError):
                continue
    out: dict[str, list[str]] = {}
    for _, f in iter_members(recording_table, lambda n: Path(n).name == "recording" or not _compression(Path(recording_table))[1]):
        for raw in f:
            p = raw.decode("utf-8", "replace").rstrip("\n").split("\t")
            try:
                rid = int(p[recording_cols[0]])
            except (ValueError, IndexError):
                continue
            if rid in by_id:
                b = mbid_bytes(p[recording_cols[1]])
                if b is not None:
                    out[mbid_str(b)] = sorted(set(by_id[rid]))
    return out


class RedirectMap:
    """``canonical_recording_redirect.csv`` as two sorted S16 arrays (memory-frugal for ~tens of millions of rows).
    ``apply`` maps recording MBIDs to their canonical recording MBID (unmapped stay as they are)."""

    def __init__(self, keys: np.ndarray, values: np.ndarray):
        order = np.argsort(keys, kind="stable")
        self.keys, self.values = keys[order], values[order]

    @classmethod
    def from_csv(cls, path: str | Path, member: str = "canonical_recording_redirect.csv", chunksize: int = 2_000_000) -> "RedirectMap":
        K, V = [], []
        for _, f in iter_members(path, lambda n: Path(n).name == member if _compression(Path(path))[1] else True):
            for ch in _csv_chunks(f, ["recording_mbid", "canonical_recording_mbid", "canonical_release_mbid"], chunksize,
                                  "recording_mbid"):
                k = [mbid_bytes(x) for x in ch["recording_mbid"]]
                v = [mbid_bytes(x) for x in ch["canonical_recording_mbid"]]
                ok = [i for i, (a, b) in enumerate(zip(k, v)) if a is not None and b is not None]
                K.append(b"".join(k[i] for i in ok)); V.append(b"".join(v[i] for i in ok))
        return cls(np.frombuffer(b"".join(K), "S16").copy(), np.frombuffer(b"".join(V), "S16").copy())

    @classmethod
    def from_pairs(cls, pairs: Mapping[str, str]) -> "RedirectMap":
        return cls(to_mbid_array(pairs.keys()), to_mbid_array(pairs.values()))

    def apply(self, mbids: np.ndarray) -> np.ndarray:
        idx = _lookup(self.keys, mbids)
        out = mbids.copy()
        hit = idx >= 0
        out[hit] = self.values[idx[hit]]
        return out

    def variants_of(self, canonical: np.ndarray) -> np.ndarray:
        """All non-canonical recording MBIDs whose canonical form is in ``canonical`` (to widen AcousticBrainz scans)."""
        return self.keys[np.isin(self.values, canonical)]

    def __len__(self) -> int:
        return int(self.keys.size)


# ---- offline (artist, title) -> MBID ---------------------------------------------------------------------------------
def norm_text(s: str) -> str:
    """Normalisation used for lookup keys: NFKD, drop combining marks, casefold, '&' -> 'and', drop every non-word
    character.  The canonical dump's ``combined_lookup`` uses a similar recipe (VERIFIED wording: punctuation and
    superfluous whitespace removed, non-ASCII converted to ASCII) but its exact form is UNVERIFIED, so keys are always
    recomputed here from ``artist_credit_name`` + ``recording_name`` rather than read from ``combined_lookup``."""
    s = unicodedata.normalize("NFKD", str(s or ""))
    s = "".join(c for c in s if not unicodedata.combining(c)).casefold().replace("&", " and ")
    return re.sub(r"[\W_]+", "", s)


_FEAT = re.compile(r"\s*[\(\[]\s*(?:feat\.?|featuring|ft\.?|with)\s[^\)\]]*[\)\]]", re.I)
_BRACKET = re.compile(r"\s*[\(\[][^\)\]]*[\)\]]")
_DASH_SUFFIX = re.compile(r"\s+-\s+(?:.*(?:remaster|version|edit|mix|live|mono|stereo|bonus|demo|single|radio)[^-]*)$", re.I)
_ARTIST_SPLIT = re.compile(r"\s*(?:,|;|&|\bfeat\.?\b|\bfeaturing\b|\bft\.?\b|\bwith\b|\bx\b|\band\b|/)\s*", re.I)


def title_variants(title: str) -> list[str]:
    """Progressively simplified titles: original, without '(feat. ..)', without ' - Remastered 2011'-style suffix,
    without any bracketed part."""
    t0 = str(title or "").strip()
    seq = [t0]
    t1 = _FEAT.sub("", t0).strip()
    t2 = _DASH_SUFFIX.sub("", t1).strip()
    t3 = _BRACKET.sub("", t2).strip()
    for t in (t1, t2, t3):
        if t and t not in seq:
            seq.append(t)
    return seq


def artist_variants(artist: str) -> list[str]:
    a0 = str(artist or "").strip()
    seq = [a0]
    first = _ARTIST_SPLIT.split(a0, maxsplit=1)[0].strip()
    if first and first not in seq:
        seq.append(first)
    return seq


def _h64(s: str) -> int:
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(), "big")


class TitleArtistResolver:
    """Offline ``(artist, title) -> recording MBID`` from the canonical metadata (built incrementally from chunks,
    stored as sorted uint64 key-hashes + S16 MBIDs + scores: ~30 bytes/row -> < 1 GB for 23 M rows).

    ``higher_score_wins``: which duplicate (same normalised artist+title) to keep.  The dump documents ``score`` as the
    'priority ordering' for search (VERIFIED) but not its direction (UNVERIFIED) - check on the real file.
    64-bit key hashes: collision probability ~ 1e-6 over 23 M keys; acceptable for a corpus-building heuristic."""

    def __init__(self, higher_score_wins: bool = True):
        self.higher_score_wins = higher_score_wins
        self._h: list[np.ndarray] = []; self._m: list[np.ndarray] = []; self._s: list[np.ndarray] = []
        self._keys: np.ndarray | None = None; self._mb: np.ndarray | None = None

    @staticmethod
    def key(artist: str, title: str) -> int:
        return _h64(norm_text(artist) + "|" + norm_text(title))

    def add(self, df: pd.DataFrame) -> None:
        """``df`` needs recording_mbid, artist_credit_name, title (+ score)."""
        if not len(df):
            return
        ok = [mbid_bytes(m) for m in df["recording_mbid"]]
        sel = [i for i, b in enumerate(ok) if b is not None]
        art = df["artist_credit_name"].to_numpy()[sel]; tit = df["title"].to_numpy()[sel]
        self._h.append(np.fromiter((self.key(a, t) for a, t in zip(art, tit)), dtype=np.uint64, count=len(sel)))
        self._m.append(np.frombuffer(b"".join(ok[i] for i in sel), dtype="S16").copy())
        sc = df["score"].to_numpy(dtype="float32")[sel] if "score" in df else np.zeros(len(sel), "float32")
        self._s.append(sc if self.higher_score_wins else -sc)
        self._keys = None

    def finalize(self) -> "TitleArtistResolver":
        if self._h:
            h = np.concatenate(self._h); m = np.concatenate(self._m); s = np.concatenate(self._s)
            order = np.lexsort((-s, h))
            h, m = h[order], m[order]
            first = np.r_[True, h[1:] != h[:-1]]
            self._keys, self._mb = h[first], m[first]
        else:
            self._keys, self._mb = np.zeros(0, np.uint64), np.zeros(0, "S16")
        self._h, self._m, self._s = [], [], []
        return self

    @classmethod
    def from_recordings(cls, table: pd.DataFrame, higher_score_wins: bool = True) -> "TitleArtistResolver":
        r = cls(higher_score_wins)
        r.add(table.reset_index().rename(columns={"index": "recording_mbid"}))
        return r.finalize()

    def _get(self, artist: str, title: str) -> bytes | None:
        if self._keys is None:
            self.finalize()
        k = np.uint64(self.key(artist, title))
        i = int(np.searchsorted(self._keys, k))
        if i < self._keys.size and self._keys[i] == k:
            return self._mb[i:i + 1].tobytes()
        return None

    def resolve(self, artist: str, title: str) -> tuple[str | None, str]:
        """-> (mbid or None, method) with method in exact | title_variant | artist_variant | both_variant | none."""
        av, tv = artist_variants(artist), title_variants(title)
        for ai, a in enumerate(av):
            for ti, t in enumerate(tv):
                if not norm_text(a) or not norm_text(t):
                    continue
                b = self._get(a, t)
                if b is not None:
                    method = "exact" if (ai == 0 and ti == 0) else ("title_variant" if ai == 0 else
                                                                    "artist_variant" if ti == 0 else "both_variant")
                    return mbid_str(b), method
        return None, "none"

    def resolve_frame(self, df: pd.DataFrame, artist_col: str = "artist", title_col: str = "title") -> pd.DataFrame:
        """Vector helper for Playlist2vec-style corpora: returns ``recording_mbid`` (NaN if unresolved) + ``method``."""
        res = [self.resolve(a, t) for a, t in zip(df[artist_col].tolist(), df[title_col].tolist())]
        return pd.DataFrame({"recording_mbid": [r[0] for r in res], "method": [r[1] for r in res]}, index=df.index)


# ======================================================================================================================
# 3. AcousticBrainz  (frozen 2022-06-23 dumps)  ->  FamilyBlocks
# ======================================================================================================================
@dataclass(frozen=True)
class FeatureSpec:
    """One AcousticBrainz low-level descriptor -> columns.  ``kind``: scalar | vector | cov_diag | key | scale."""
    path: str
    stats: tuple[str, ...] = ()
    kind: str = "scalar"
    length: int = 1

    def expand(self) -> list[tuple[str, str | None]]:
        """[(column_name, stat)]"""
        if self.kind == "key":
            return [(f"{self.path}_cos", None), (f"{self.path}_sin", None)]        # path already ends in '.key'
        if self.kind == "scale":
            return [(f"{self.path}_major", None)]                                # path already ends in '.scale' 
        if self.kind == "cov_diag":
            return [(f"{self.path}.cov_diag[{i}]", "cov") for i in range(self.length)]
        cols = []
        for s in (self.stats or (None,)):
            base = f"{self.path}.{s}" if s else self.path
            cols += [(f"{base}[{i}]", s) for i in range(self.length)] if self.kind == "vector" else [(base, s)]
        return cols

    def base_names(self) -> list[str]:
        """Essentia-style descriptor names (what Onion's ``id_essentia`` keys look like) this spec covers."""
        if self.kind == "cov_diag":
            return [f"{self.path}.cov"]
        if self.kind in ("key", "scale"):
            return [self.path]
        return [f"{self.path}.{s}" if s else self.path for s in (self.stats or (None,))]


_MS = ("mean", "stdev")
_S = FeatureSpec


def _scal(prefix: str, names: Sequence[str], stats: tuple[str, ...] = _MS) -> list[FeatureSpec]:
    return [_S(f"{prefix}.{n}", stats) for n in names]


AB_FAMILY_SPECS: dict[str, list[FeatureSpec]] = {
    # MFCC (Onion id_mfcc_stats / id_compare_mfcc_stats are librosa/openSMILE MFCCs - NOT Essentia; only the Essentia MFCC
    # is comparable across AcousticBrainz / Jamendo-Essentia).  cov diagonal = per-coefficient variance.
    "timbre": [_S("lowlevel.mfcc", ("mean",), "vector", 13), _S("lowlevel.mfcc", (), "cov_diag", 13)],
    "spectral_shape": (
        _scal("lowlevel", ["spectral_centroid", "spectral_spread", "spectral_skewness", "spectral_kurtosis",
                           "spectral_rolloff", "spectral_decrease", "spectral_flux", "spectral_entropy",
                           "spectral_complexity", "spectral_strongpeak", "hfc", "zerocrossingrate",
                           "barkbands_crest", "barkbands_flatness_db", "barkbands_kurtosis", "barkbands_skewness",
                           "barkbands_spread", "melbands_crest", "melbands_flatness_db", "melbands_kurtosis",
                           "melbands_skewness", "melbands_spread", "spectral_energyband_low",
                           "spectral_energyband_middle_low", "spectral_energyband_middle_high",
                           "spectral_energyband_high"])
        + [_S("lowlevel.spectral_contrast_coeffs", _MS, "vector", 6), _S("lowlevel.spectral_contrast_valleys", _MS, "vector", 6),
           _S("lowlevel.barkbands", ("mean",), "vector", 27)]),
    "rhythm_groove": (
        [_S("rhythm.bpm"), _S("rhythm.onset_rate"), _S("rhythm.danceability")]
        + [_S(f"rhythm.bpm_histogram_{p}", ("mean",)) for p in
           ("first_peak_bpm", "first_peak_weight", "first_peak_spread", "second_peak_bpm", "second_peak_weight",
            "second_peak_spread")]
        + [_S("rhythm.beats_loudness", _MS), _S("rhythm.beats_loudness_band_ratio", ("mean",), "vector", 6)]),
    "harmony_tonal": (
        [_S("tonal.hpcp", ("mean",), "vector", 36)]
        + _scal("tonal", ["hpcp_entropy", "hpcp_crest", "chords_strength"])
        + [_S(f"tonal.{n}") for n in ("chords_changes_rate", "chords_number_rate", "tuning_frequency",
                                       "tuning_equal_tempered_deviation", "tuning_diatonic_strength",
                                       "tuning_nontempered_energy_ratio")]
        + [_S(f"tonal.key_{k}.strength") for k in ("edma", "krumhansl", "temperley")]
        + [_S(f"tonal.key_{k}.key", kind="key") for k in ("edma", "krumhansl", "temperley")]
        + [_S(f"tonal.key_{k}.scale", kind="scale") for k in ("edma", "krumhansl", "temperley")]
        + _scal("lowlevel", ["dissonance", "pitch_salience"])),
    "dynamics_loudness": (
        [_S("lowlevel.average_loudness"), _S("lowlevel.dynamic_complexity")]
        + _scal("lowlevel", ["silence_rate_20dB", "silence_rate_30dB", "silence_rate_60dB", "spectral_rms", "spectral_energy"])
        # EBU R128 descriptors exist only in newer extractor versions (NaN for most AcousticBrainz submissions)
        + [_S("lowlevel.loudness_ebu128.integrated"), _S("lowlevel.loudness_ebu128.loudness_range"),
           _S("lowlevel.loudness_ebu128.short_term", _MS), _S("lowlevel.loudness_ebu128.momentary", _MS)]),
}

# Mapping to PLAN §4.4 / Onion (document, not code):
#   timbre            <- Onion id_mfcc_stats, id_compare_mfcc_stats            ~ Essentia lowlevel.mfcc.{mean,cov}
#   spectral_shape    <- Onion id_blf_spectral*, delta/vardelta                ~ Essentia lowlevel.spectral_*, *bands_* (BLF itself has no AB analogue)
#   rhythm_groove     <- Onion id_blf_logfluc + Essentia rhythm.*              ~ Essentia rhythm.* (logfluc has no AB analogue)
#   harmony_tonal     <- Onion id_chroma_bow, id_blf_correlation, Essentia tonal.* ~ Essentia tonal.* (chroma_bow/correlation: none)
#   dynamics_loudness <- Onion Essentia loudness/dynamic-complexity, id_compare_pcm_stats ~ Essentia lowlevel.{average_loudness,dynamic_complexity,silence_rate*}
# PLAN §13 item 4 flag: Onion's id_essentia keys use ``.stdev`` (see essentia_descriptor_names.json) whereas AcousticBrainz
# low-level JSON stores ``var`` (+median/min/max/dmean/...).  We derive stdev = sqrt(var), which is exact for the same
# frames, but (UNVERIFIED) the extractor *version/profile* differs (AB: mostly 2.1_beta1; Onion: newer, has melbands128,
# gfcc, loudness_ebu128), so values are not guaranteed numerically comparable across Track A and Track B.

_KEY_PC = {"C": 0, "C#": 1, "DB": 1, "D": 2, "D#": 3, "EB": 3, "E": 4, "F": 5, "F#": 6, "GB": 6, "G": 7, "G#": 8, "AB": 8,
           "A": 9, "A#": 10, "BB": 10, "B": 11}


def key_to_pc(k) -> int | None:
    """'C#', 'Db', 'a' -> pitch class 0..11 (None if unknown)."""
    return _KEY_PC.get(str(k).strip().upper().replace("♯", "#").replace("♭", "B")) if k is not None else None


def ab_columns(families: Sequence[str] | None = None) -> dict[str, list[str]]:
    """Fixed column names per family (order matters: shards and blocks are aligned on it)."""
    return {f: [c for s in AB_FAMILY_SPECS[f] for c, _ in s.expand()] for f in (families or FAMILY_NAMES)}


def _dig(doc: Mapping, path: str):
    node = doc
    for part in path.split("."):
        if isinstance(node, Mapping) and part in node:
            node = node[part]
        else:
            return None
    return node


def _stat(node, stat: str | None):
    """Pick a statistic from an AB descriptor node.  ``stdev`` falls back to sqrt(var) (AB stores var); a bare scalar
    has no spread so 'stdev' -> None."""
    if stat is None:
        return node
    if isinstance(node, Mapping):
        if stat in node:
            return node[stat]
        if stat == "stdev" and "var" in node:
            return np.sqrt(np.maximum(np.asarray(node["var"], dtype=float), 0.0))
        if stat == "var" and "stdev" in node:
            return np.square(np.asarray(node["stdev"], dtype=float))
        return None
    return node if stat == "mean" else None


def _num(x) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def extract_ab_features(doc: Mapping, families: Sequence[str] | None = None) -> dict[str, np.ndarray]:
    """One AcousticBrainz/Essentia low-level JSON document -> {family: float32 vector} (NaN where a descriptor is absent).
    Also accepts Essentia music-extractor output with ``stdev`` statistics (e.g. Jamendo's AcousticBrainz-type files)."""
    out: dict[str, np.ndarray] = {}
    for fam in (families or FAMILY_NAMES):
        vals: list[float] = []
        for spec in AB_FAMILY_SPECS[fam]:
            node = _dig(doc, spec.path)
            if spec.kind == "key":
                pc = key_to_pc(node)
                vals += [math.cos(2 * math.pi * pc / 12), math.sin(2 * math.pi * pc / 12)] if pc is not None else [float("nan")] * 2
            elif spec.kind == "scale":
                s = str(node).lower() if node is not None else ""
                vals.append(1.0 if s == "major" else 0.0 if s == "minor" else float("nan"))
            elif spec.kind == "cov_diag":
                cov = node.get("cov") if isinstance(node, Mapping) else None
                try:
                    d = np.diag(np.asarray(cov, dtype=float))
                    vals += [float(x) for x in d[:spec.length]] + [float("nan")] * max(0, spec.length - d.size)
                except (ValueError, TypeError):
                    vals += [float("nan")] * spec.length
            else:
                for s in (spec.stats or (None,)):
                    v = _stat(node, s)
                    if spec.kind == "vector":
                        try:
                            a = np.asarray(v, dtype=float).ravel()
                        except (TypeError, ValueError):
                            a = np.zeros(0)
                        a = a[:spec.length]
                        vals += [float(x) if math.isfinite(x) else float("nan") for x in a] + [float("nan")] * (spec.length - a.size)
                    else:
                        vals.append(_num(v))
        out[fam] = np.asarray(vals, dtype=np.float32)
    return out


AB_DOC_NAME = re.compile(r"(?:^|/)lowlevel/[^/]+/[^/]+/([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})-(\d+)\.json$")


@dataclass
class AudioScanStats:
    archives_done: int = 0
    archives_skipped: int = 0
    docs_seen: int = 0
    docs_matched: int = 0
    bad_docs: int = 0
    stopped: bool = False


def scan_lowlevel_archives(paths: Sequence[str | Path], out_dir: str | Path, candidates: np.ndarray | None,
                           families: Sequence[str] | None = None, should_stop: Callable[[], bool] | None = None) -> AudioScanStats:
    """Stream ``acousticbrainz-lowlevel-json-20220623-<k>.tar.zst`` archives (~20 GB each, 30 of them) and keep only
    documents whose MBID (from the member *name*, so non-candidates are skipped without JSON parsing) is in
    ``candidates`` (S16, None = keep everything - do not do that at full scale).  Layout VERIFIED from
    acousticbrainz-server ``db/dump.py``: ``<dir>/lowlevel/<mbid[0:2]>/<mbid[2:3]>/<mbid>-<offset>.json``.

    One shard ``ab_<archive>.npz`` (mbid S16, offset int32, X float32) per archive + ``ab_manifest.json``: interrupt at any
    time, re-run to continue with the next archive.  Uncompressed JSON is several times the archive size; expect
    minutes-to-tens-of-minutes per archive, dominated by zstd + tar streaming."""
    fams = list(families or FAMILY_NAMES)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    mp = out / "ab_manifest.json"
    man = read_json(mp) if mp.exists() else {"version": 1, "families": fams, "columns": ab_columns(fams), "done": {}}
    if man["families"] != fams:
        raise ValueError("existing audio shards were built with different families; use a fresh directory")
    keep = _mbid_set(candidates) if candidates is not None else None
    st = AudioScanStats()
    for arch in map(Path, paths):
        if arch.name in man["done"]:
            st.archives_skipped += 1
            continue
        M: list[bytes] = []; O: list[int] = []; X: list[np.ndarray] = []
        for name, f in iter_members(arch, lambda n: n.endswith(".json") and "lowlevel" in n):
            st.docs_seen += 1
            if st.docs_seen % 20000 == 0 and should_stop and should_stop():
                st.stopped = True
                return st
            m = AB_DOC_NAME.search(name)
            if m is None:
                continue
            mb = mbid_bytes(m.group(1))
            if keep is not None and mb not in keep:
                continue
            try:
                doc = json.loads(f.read())
                if not isinstance(doc, Mapping):
                    raise ValueError("low-level document is not a JSON object")
                feats = extract_ab_features(doc, fams)
            except (ValueError, AttributeError):
                st.bad_docs += 1
                continue
            M.append(mb); O.append(int(m.group(2))); X.append(np.concatenate([feats[k] for k in fams]))
            st.docs_matched += 1
        fn = f"ab_{arch.name.split('.')[0]}.npz"
        np.savez(out / fn, mbid=np.frombuffer(b"".join(M), "S16") if M else np.zeros(0, "S16"), offset=np.asarray(O, np.int32),
                 X=np.vstack(X).astype(np.float32) if X else np.zeros((0, sum(len(c) for c in man["columns"].values())), np.float32))
        man["done"][arch.name] = fn
        write_json(mp, man)
        st.archives_done += 1
        log.info("AcousticBrainz: %s -> %d matched docs (running total %d)", arch.name, len(M), st.docs_matched)
    return st


@dataclass
class AudioTable:
    """Per-recording low-level feature vectors (all families concatenated in ``ab_columns`` order)."""
    mbid: np.ndarray            # S16 unique, sorted
    X: np.ndarray               # (n, D) float32, NaN allowed
    columns: dict[str, list[str]]
    source: str = "acousticbrainz-lowlevel-2022-06-23"

    def blocks(self, vocab: np.ndarray, max_nan_frac: float = 0.3) -> dict[str, FamilyBlock]:
        """FamilyBlocks in the *corpus* index space (raw id = position in ``vocab``, sorted S16).  Column-median imputation;
        rows with more than ``max_nan_frac`` missing values in a family are dropped from that family."""
        ri = _lookup(vocab, self.mbid)
        keep = ri >= 0
        out: dict[str, FamilyBlock] = {}
        start = 0
        for fam, cols in self.columns.items():
            sl = slice(start, start + len(cols)); start += len(cols)
            Xf = self.X[keep][:, sl].astype(np.float32)
            rows = ri[keep]
            if not len(rows):
                continue
            ok = np.isnan(Xf).mean(axis=1) <= max_nan_frac
            Xf, rows = Xf[ok], rows[ok]
            if not len(rows):
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                med = np.nanmedian(Xf, axis=0)
            med = np.where(np.isfinite(med), med, 0.0).astype(np.float32)      # all-NaN column -> 0 (constant, z-scores to 0)
            r, c = np.where(np.isnan(Xf))
            Xf[r, c] = med[c]
            order = np.argsort(rows)
            out[fam] = FamilyBlock(rows[order].astype(np.int64), Xf[order], list(cols), self.source)
        return out

    def remap(self, redirect: RedirectMap | None) -> "AudioTable":
        """Canonicalise MBIDs and merge duplicates (mean of non-NaN)."""
        if redirect is None:
            return self
        return _merge_audio(redirect.apply(self.mbid), self.X, self.columns, self.source, "mean")


def _merge_audio(mbid: np.ndarray, X: np.ndarray, columns: dict, source: str, policy: str,
                 offset: np.ndarray | None = None) -> AudioTable:
    if not mbid.size:
        return AudioTable(mbid, X, columns, source)
    if policy == "first" and offset is not None:
        order = np.lexsort((offset, mbid))
        mbid, X = mbid[order], X[order]
        u, idx = np.unique(mbid, return_index=True)
        return AudioTable(u, X[idx], columns, source)
    u, inv = np.unique(mbid, return_inverse=True)
    if u.size == mbid.size:
        order = np.argsort(mbid)
        return AudioTable(mbid[order], X[order], columns, source)
    S = np.zeros((u.size, X.shape[1]), np.float64); C = np.zeros_like(S)
    fin = np.isfinite(X)
    np.add.at(S, inv, np.where(fin, X, 0.0)); np.add.at(C, inv, fin.astype(np.float64))
    with np.errstate(all="ignore"):
        M = np.where(C > 0, S / np.maximum(C, 1), np.nan)
    return AudioTable(u, M.astype(np.float32), columns, source)


def load_audio(out_dir: str | Path, policy: str = "first") -> AudioTable:
    """Combine completed AcousticBrainz shards.  ``policy``: 'first' (lowest submission offset) | 'mean' (nan-mean over
    submissions of the same recording)."""
    out = Path(out_dir)
    man = read_json(out / "ab_manifest.json")
    Ms, Os, Xs = [], [], []
    for fn in man["done"].values():
        with np.load(out / fn) as z:
            Ms.append(z["mbid"]); Os.append(z["offset"]); Xs.append(z["X"])
    D = sum(len(c) for c in man["columns"].values())
    if not Ms:
        return AudioTable(np.zeros(0, "S16"), np.zeros((0, D), np.float32), man["columns"])
    return _merge_audio(np.concatenate(Ms), np.vstack(Xs), man["columns"], "acousticbrainz-lowlevel-2022-06-23", policy,
                        np.concatenate(Os))


# ---- CSV feature dumps (small; coarse) --------------------------------------------------------------------------------
AB_CSV_FEATURES = {
    "lowlevel": ["average_loudness", "dynamic_complexity", "mfcc_zero_mean"],
    "rhythm": ["bpm", "bpm_histogram_first_peak_bpm_mean", "bpm_histogram_first_peak_bpm_median",
               "bpm_histogram_second_peak_bpm_mean", "bpm_histogram_second_peak_bpm_median", "danceability", "onset_rate"],
    "tonal": ["key_key", "key_scale", "tuning_frequency", "tuning_equal_tempered_deviation"],
}
AB_CSV_FAMILY_MAP = {                                    # coarse: only what the three CSVs contain (VERIFIED column lists)
    "timbre": ["mfcc_zero_mean"],
    "rhythm_groove": AB_CSV_FEATURES["rhythm"],
    "harmony_tonal": ["key_key", "key_scale", "tuning_frequency", "tuning_equal_tempered_deviation"],
    "dynamics_loudness": ["average_loudness", "dynamic_complexity"],
}                                                        # spectral_shape: NOT available from the CSVs (needs low-level JSON)


def read_ab_csv_features(paths: Sequence[str | Path], candidates: np.ndarray | None = None, chunksize: int = 500_000) -> pd.DataFrame:
    """The three ``acousticbrainz-lowlevel-features-20220623-{lowlevel,rhythm,tonal}.tar.zst`` CSV dumps (or plain CSVs)
    -> DataFrame indexed by recording MBID (str; lowest submission_offset kept) with the union of the feature columns.
    VERIFIED columns (acousticbrainz-server dump.py): ``mbid, submission_offset, <features>``.  UNVERIFIED: member names
    inside the .tar.zst (any ``.csv`` member is read)."""
    keep = {mbid_str(b) for b in _mbid_set(candidates)} if candidates is not None else None
    frames = []
    for p in paths:
        parts = []
        for _, f in iter_members(p, lambda n: n.lower().endswith(".csv") or not _compression(Path(p))[1]):
            for ch in pd.read_csv(io.BufferedReader(_Prefixed(b"", f), 1 << 20), dtype=str, keep_default_na=False, chunksize=chunksize, on_bad_lines="skip"):
                if "mbid" not in ch:
                    continue
                ch["mbid"] = ch["mbid"].str.lower()
                if keep is not None:
                    ch = ch[ch["mbid"].isin(keep)]
                if len(ch):
                    parts.append(ch)
        if parts:
            d = pd.concat(parts, ignore_index=True)
            d["submission_offset"] = pd.to_numeric(d.get("submission_offset", 0), errors="coerce").fillna(0)
            d = d.sort_values("submission_offset", kind="stable").drop_duplicates("mbid").drop(columns=["submission_offset"])
            frames.append(d.set_index("mbid"))
    if not frames:
        return pd.DataFrame().rename_axis("mbid")
    out = frames[0]
    for d in frames[1:]:
        out = out.join(d, how="outer", rsuffix="_dup")
    return out


def ab_csv_blocks(df: pd.DataFrame, vocab: np.ndarray) -> dict[str, FamilyBlock]:
    """Coarse family blocks from the CSV dumps (``source`` says so).  key_key -> (cos, sin) on the pitch-class circle,
    key_scale -> major=1.  Families without any CSV descriptor are omitted."""
    if df.empty:
        return {}
    df = df[[mbid_bytes(x) is not None for x in df.index]]
    mb = to_mbid_array(df.index)
    ri = _lookup(vocab, mb)                        # vocab: sorted S16, raw id = position (build_corpus contract)
    ok = ri >= 0
    out: dict[str, FamilyBlock] = {}
    for fam, feats in AB_CSV_FAMILY_MAP.items():
        cols, arrs = [], []
        for c in feats:
            if c not in df:
                continue
            s = df[c]
            if c == "key_key":
                pcs = s.map(key_to_pc)
                ang = pcs.map(lambda v: 2 * math.pi * v / 12 if v is not None and not pd.isna(v) else np.nan)
                cols += ["key_key_cos", "key_key_sin"]; arrs += [np.cos(ang.astype(float)), np.sin(ang.astype(float))]
            elif c == "key_scale":
                cols.append(c); arrs.append(s.str.lower().map({"major": 1.0, "minor": 0.0}).astype(float))
            else:
                cols.append(c); arrs.append(pd.to_numeric(s, errors="coerce"))
        if not cols:
            continue
        X = np.column_stack([np.asarray(a, dtype=float) for a in arrs])[ok]
        rows = ri[ok]
        good = ~np.isnan(X).any(axis=1)
        if good.any():
            order = np.argsort(rows[good])
            out[fam] = FamilyBlock(rows[good][order].astype(np.int64), X[good][order].astype(np.float32), cols,
                                   "acousticbrainz-csv-features-2022-06-23 (coarse)")
    return out


# ======================================================================================================================
# 4. LRCLIB lyrics: presence + aggregate embedding only (no lyric text leaves this section)
# ======================================================================================================================
LRCLIB_BASE = "https://lrclib.net/api"
_LRC_TS = re.compile(r"\[\d{1,3}:\d{2}(?:[.:]\d{1,3})?\]")


def hashing_embedding(text: str, dim: int = 64, seed: int = 0) -> np.ndarray:
    """Deterministic signed feature-hashing bag-of-words, sublinear tf, L2-normalised.  A dependency-free *placeholder*
    for the real lyric encoders (Onion uses BERT/word2vec/tf-idf; inject e.g. a sentence-transformer via ``embed_fn``)."""
    v = np.zeros(dim, np.float64)
    for tok in re.findall(r"\w+", text.lower()):
        h = int.from_bytes(hashlib.blake2b(f"{seed}:{tok}".encode(), digest_size=8).digest(), "big")
        v[h % dim] += 1.0 if (h >> 63) & 1 else -1.0
    v = np.sign(v) * np.log1p(np.abs(v))
    n = np.linalg.norm(v)
    return (v / n if n > 0 else v).astype(np.float32)


@dataclass
class LyricsFeatures:
    """Everything we keep about a track's lyrics: no text."""
    found: bool
    has_lyrics: bool
    instrumental: bool
    n_words: int = 0
    n_lines: int = 0
    has_synced: bool = False
    embedding: np.ndarray | None = None


def lyrics_features_from_record(rec: Mapping, embed_fn: Callable[[str], np.ndarray], dim: int) -> LyricsFeatures:
    """LRCLIB JSON record (VERIFIED field names) -> :class:`LyricsFeatures`; the text is dropped immediately."""
    plain = rec.get("plainLyrics") or rec.get("plain_lyrics") or ""
    synced = rec.get("syncedLyrics") or rec.get("synced_lyrics") or ""
    instr = bool(rec.get("instrumental"))
    text = plain if plain.strip() else _LRC_TS.sub("", synced)
    has = bool(text.strip()) and not instr
    emb = np.asarray(embed_fn(text), dtype=np.float32) if has else None
    if emb is not None and emb.shape != (dim,):
        raise ValueError(f"embed_fn returned {emb.shape}, expected ({dim},)")
    return LyricsFeatures(True, has, instr, len(text.split()) if has else 0, sum(1 for ln in text.splitlines() if ln.strip()) if has else 0,
                          bool(synced.strip()), emb)


class LrclibCache:
    """SQLite cache of *derived* features (presence, counts, embedding blob) keyed by normalised artist|title|duration."""

    def __init__(self, path: str | Path, dim: int):
        self.dim = dim
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path))
        self.db.execute("CREATE TABLE IF NOT EXISTS lrc (k TEXT PRIMARY KEY, found INT, has INT, instr INT, words INT, lines INT,"
                        " synced INT, emb BLOB, ts REAL)")
        self.db.commit()

    def get(self, k: str) -> LyricsFeatures | None:
        r = self.db.execute("SELECT found,has,instr,words,lines,synced,emb FROM lrc WHERE k=?", (k,)).fetchone()
        if r is None:
            return None
        return LyricsFeatures(bool(r[0]), bool(r[1]), bool(r[2]), r[3], r[4], bool(r[5]),
                              np.frombuffer(r[6], np.float32).copy() if r[6] else None)

    def put(self, k: str, f: LyricsFeatures) -> None:
        self.db.execute("INSERT OR REPLACE INTO lrc VALUES (?,?,?,?,?,?,?,?,?)",
                        (k, int(f.found), int(f.has_lyrics), int(f.instrumental), f.n_words, f.n_lines, int(f.has_synced),
                         f.embedding.astype(np.float32).tobytes() if f.embedding is not None else None, time.time()))
        self.db.commit()

    def __len__(self) -> int:
        return self.db.execute("SELECT COUNT(*) FROM lrc").fetchone()[0]

    def close(self) -> None:
        self.db.close()


Transport = Callable[[str, Mapping[str, str], Mapping[str, str]], "tuple[int, bytes]"]


def _httpx_transport(url: str, params: Mapping[str, str], headers: Mapping[str, str]) -> tuple[int, bytes]:
    import httpx
    r = httpx.get(url, params=dict(params), headers=dict(headers), timeout=30.0)
    return r.status_code, r.content


def _lrc_key(artist: str, title: str, duration_s: float | None) -> str:
    d = "" if duration_s is None else str(int(round(duration_s)))
    return hashlib.sha1(f"{norm_text(artist)}|{norm_text(title)}|{d}".encode()).hexdigest()


class LrclibClient:
    """LRCLIB API client: **<= 1 request/s** (``min_interval_s``; cache hits are free), a descriptive User-Agent (required
    - generic agents are rejected here), exponential back-off on 429/5xx, resumable through :class:`LrclibCache`.

    VERIFIED: ``GET {base}/get?artist_name&track_name[&album_name&duration]`` and ``GET {base}/search?q=`` return the JSON
    record(s) described in the module docstring.  UNVERIFIED: LRCLIB's published rate-limit/User-Agent policy (docs page not
    readable) - the 1 req/s ceiling is our own conservative choice; do not use this class for bulk (> ~50 k tracks): use
    the SQLite dump via :class:`LrclibDump`.  Lyrics *text is never stored* (only :class:`LyricsFeatures`).

    ``lookup`` returns None on a transient failure (not cached; retry later) and a :class:`LyricsFeatures` otherwise
    (``found=False`` for a definitive 404, cached)."""

    def __init__(self, cache_path: str | Path, user_agent: str, *, transport: Transport | None = None,
                 min_interval_s: float = 1.0, embed_fn: Callable[[str], np.ndarray] | None = None, embed_dim: int = 64,
                 max_retries: int = 3, search_fallback: bool = False, sleep: Callable[[float], None] = time.sleep,
                 clock: Callable[[], float] = time.monotonic, base_url: str = LRCLIB_BASE, seed: int = 0):
        if not user_agent or "/" not in user_agent or user_agent.lower().startswith(("python", "httpx", "curl")):
            raise ValueError("give a descriptive User-Agent like 'Melomane/0.1 (contact-url-or-email)'")
        if min_interval_s < 1.0:
            raise ValueError("min_interval_s < 1.0 violates the <= 1 req/s policy of this adapter")
        self.ua, self.transport, self.min_interval = user_agent, transport or _httpx_transport, min_interval_s
        self.dim = embed_dim
        self.embed_fn = embed_fn or (lambda t: hashing_embedding(t, embed_dim, seed))
        self.cache = LrclibCache(cache_path, embed_dim)
        self.max_retries, self.search_fallback, self.base = max_retries, search_fallback, base_url.rstrip("/")
        self._sleep, self._clock, self._last = sleep, clock, None
        self.n_requests = 0

    def _request(self, path: str, params: Mapping[str, str]) -> tuple[int, object]:
        delay = 5.0
        for attempt in range(self.max_retries + 1):
            if self._last is not None:
                wait = self.min_interval - (self._clock() - self._last)
                if wait > 0:
                    self._sleep(wait)
            self._last = self._clock()
            self.n_requests += 1
            try:
                status, body = self.transport(f"{self.base}/{path}", params, {"User-Agent": self.ua, "Accept": "application/json"})
            except Exception as e:                      # httpx errors are not OSError subclasses
                log.warning("LRCLIB transport error: %s", type(e).__name__)
                status, body = 599, b""
            if status == 200:
                try:
                    return 200, json.loads(body)
                except ValueError:
                    return 599, None
            if status == 404:
                return 404, None
            if status in (429, 500, 502, 503, 504, 599) and attempt < self.max_retries:
                self._sleep(delay); delay *= 2
                continue
            return status, None
        return 599, None

    def lookup(self, artist: str, title: str, album: str = "", duration_s: float | None = None) -> LyricsFeatures | None:
        k = _lrc_key(artist, title, duration_s)
        hit = self.cache.get(k)
        if hit is not None:
            return hit
        params = {"artist_name": artist, "track_name": title}
        if album:
            params["album_name"] = album
        if duration_s:
            params["duration"] = str(int(round(duration_s)))
        status, rec = self._request("get", params)
        if status == 404 and self.search_fallback:
            st2, res = self._request("search", {"q": f"{artist} {title}"})
            if st2 == 200 and isinstance(res, list):
                na, nt = norm_text(artist), norm_text(title)
                rec = next((c for c in res if isinstance(c, dict) and norm_text(c.get("artistName", "")) == na
                            and norm_text(c.get("trackName", "")) == nt
                            and (not duration_s or abs(float(c.get("duration") or 0) - duration_s) <= 3)), None)
                status = 200 if rec is not None else 404
            elif st2 != 404:
                status = st2
        if status == 200 and isinstance(rec, dict):
            f = lyrics_features_from_record(rec, self.embed_fn, self.dim)
        elif status == 404:
            f = LyricsFeatures(False, False, False)
        else:
            return None
        self.cache.put(k, f)
        return f


class LrclibDump:
    """Offline lookup in an LRCLIB SQLite dump (VERIFIED only that the dump is SQLite3, published at lrclib.net/db-dumps;
    size and schema UNVERIFIED).  The reader introspects the schema: it needs a tracks-like table (columns name|track_name +
    artist_name [+ duration]) and a lyrics-like table (plain_lyrics/plainLyrics [+ synced_lyrics, instrumental]) linked by
    ``lyrics.track_id`` or ``tracks.last_lyrics_id``.  Decompress a .gz/.zst dump first."""

    def __init__(self, path: str | Path, embed_fn: Callable[[str], np.ndarray] | None = None, embed_dim: int = 64, seed: int = 0):
        self.dim = embed_dim
        self.embed_fn = embed_fn or (lambda t: hashing_embedding(t, embed_dim, seed))
        uri = Path(path).resolve().as_uri() + "?mode=ro"
        self.db = sqlite3.connect(uri, uri=True)
        tabs = [r[0] for r in self.db.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        cols = {t: [r[1] for r in self.db.execute(f'PRAGMA table_info("{t}")')] for t in tabs}
        self.t_tracks = next((t for t, c in cols.items() if "artist_name" in c and ("name" in c or "track_name" in c)), None)
        self.t_lyrics = next((t for t, c in cols.items() if "plain_lyrics" in c or "plainLyrics" in c), None)
        if self.t_tracks is None or self.t_lyrics is None:
            raise ValueError(f"unrecognised LRCLIB dump schema; tables/columns seen: {cols}")
        tc, lc = cols[self.t_tracks], cols[self.t_lyrics]
        self.c_name = "name" if "name" in tc else "track_name"
        self.c_plain = "plain_lyrics" if "plain_lyrics" in lc else "plainLyrics"
        self.c_synced = next((c for c in ("synced_lyrics", "syncedLyrics") if c in lc), None)
        self.c_instr = "instrumental" if "instrumental" in lc else None
        self.c_dur = "duration" if "duration" in tc else None
        self.lower = "name_lower" in tc and "artist_name_lower" in tc
        if self.t_tracks == self.t_lyrics:
            self.join = ""
        elif "track_id" in lc:
            self.join = f'JOIN "{self.t_lyrics}" l ON l.track_id = t.id'
        elif "last_lyrics_id" in tc:
            self.join = f'JOIN "{self.t_lyrics}" l ON l.id = t.last_lyrics_id'
        else:
            raise ValueError("cannot link lyrics to tracks (no track_id / last_lyrics_id)")
        self.alias = "t" if not self.join else "l"

    def lookup(self, artist: str, title: str, duration_s: float | None = None) -> LyricsFeatures | None:
        """None = not in the dump."""
        a, n = (artist.strip().lower(), title.strip().lower())
        w = ("t.artist_name_lower=? AND t.name_lower=?" if self.lower
             else f't.artist_name=? COLLATE NOCASE AND t."{self.c_name}"=? COLLATE NOCASE')
        vals = (a, n) if self.lower else (artist.strip(), title.strip())
        sel = [f'{self.alias}."{self.c_plain}"', f'{self.alias}."{self.c_synced}"' if self.c_synced else "NULL",
               f'{self.alias}."{self.c_instr}"' if self.c_instr else "0", f't."{self.c_dur}"' if self.c_dur else "NULL"]
        rows = self.db.execute(f'SELECT {",".join(sel)} FROM "{self.t_tracks}" t {self.join} WHERE {w} LIMIT 20', vals).fetchall()
        if not rows:
            return None
        if duration_s:
            rows.sort(key=lambda r: abs((r[3] or 0) - duration_s))
        p, s, i, _ = rows[0]
        return lyrics_features_from_record({"plainLyrics": p, "syncedLyrics": s, "instrumental": bool(i)}, self.embed_fn, self.dim)

    def close(self) -> None:
        self.db.close()


@dataclass
class LyricsTable:
    """Per-recording lyric-derived features: presence + aggregate embedding (no text)."""
    mbid: np.ndarray                 # S16 sorted unique; only recordings that were *looked up*
    has_lyrics: np.ndarray           # bool
    instrumental: np.ndarray         # bool
    embedding: np.ndarray | None     # (n, d) float32, zero rows where has_lyrics is False
    n_pending: int = 0               # candidates not yet looked up (transient failures / stopped early)


def collect_lyrics(mbids: np.ndarray, meta: pd.DataFrame, source, *, duration_col: str | None = None,
                   should_stop: Callable[[], bool] | None = None, max_lookups: int | None = None) -> LyricsTable:
    """Look up every recording in ``mbids`` (S16; ``meta`` has artist_credit_name + title by recording_mbid str index)
    through ``source`` (:class:`LrclibClient` or :class:`LrclibDump`).  Resumable (the client's SQLite cache); stops
    cleanly on ``should_stop`` / ``max_lookups`` and reports ``n_pending``."""
    strs = mbid_array_to_str(mbids)
    order = np.argsort(mbids)
    keep_i: list[int] = []; feats: list[LyricsFeatures] = []; pending = 0; done = 0
    for j in order:
        m = strs[j]
        if m not in meta.index:
            continue
        if (should_stop and should_stop()) or (max_lookups is not None and done >= max_lookups):
            pending += 1
            continue
        r = meta.loc[m]
        dur = float(r[duration_col]) if duration_col and duration_col in meta and pd.notna(r[duration_col]) else None
        f = source.lookup(r["artist_credit_name"], r["title"], duration_s=dur)
        done += 1
        if f is None:
            if isinstance(source, LrclibDump):      # not in dump = definitive absence
                f = LyricsFeatures(False, False, False)
            else:
                pending += 1
                continue
        keep_i.append(int(j)); feats.append(f)
    dim = getattr(source, "dim", 64)
    sel = np.asarray(keep_i, dtype=np.int64)
    emb = np.zeros((len(feats), dim), np.float32)
    for i, f in enumerate(feats):
        if f.embedding is not None:
            emb[i] = f.embedding
    return LyricsTable(mbids[sel] if len(sel) else np.zeros(0, "S16"), np.array([f.has_lyrics for f in feats], bool),
                       np.array([f.instrumental for f in feats], bool), emb, pending)


# ======================================================================================================================
# 5. corpus assembly, coverage, downloads
# ======================================================================================================================
@dataclass
class CoverageReport:
    n_tracks: int
    n_behaviour: int
    n_metadata: int
    n_audio: int
    n_lyrics: int
    n_audio_and_lyrics: int
    n_all_three: int
    per_family: dict[str, int]
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["frac_all_three"] = self.n_all_three / self.n_tracks if self.n_tracks else 0.0
        return d


def coverage_report(c: Corpus) -> CoverageReport:
    """Coverage of a Track-B corpus: tracks with behaviour (>= 1 session) AND audio (any acoustic family) AND lyrics."""
    n = c.n_tracks
    beh = np.asarray(c.sessions.getnnz(axis=0) > 0).ravel() if c.sessions.shape[0] else np.zeros(n, bool)
    aud = np.zeros(n, bool)
    per = {}
    for k, b in c.families.items():
        per[k] = int(b.rows.size)
        if k in FAMILY_NAMES:
            aud[b.rows] = True
    lyr = c.tracks["has_lyrics"].to_numpy(bool) if "has_lyrics" in c.tracks else np.zeros(n, bool)
    meta = ((c.tracks["artist"].astype(str) != "") & (c.tracks["title"].astype(str) != "")).to_numpy()
    notes = []
    if lyr.sum() == 0:
        notes.append("no lyrics matched (or lyrics stage not run)")
    return CoverageReport(n, int(beh.sum()), int(meta.sum()), int(aud.sum()), int(lyr.sum()), int((aud & lyr).sum()),
                          int((beh & aud & lyr).sum()), per, notes)


def build_corpus(listens: Listens, recordings: pd.DataFrame, *, redirect: RedirectMap | None = None,
                 audio: AudioTable | None = None, lyrics: LyricsTable | None = None, ab_csv: pd.DataFrame | None = None,
                 name: str = "mbid_stack",
                 gap_minutes: float = 30.0, min_len: int = 5, max_len: int = 200, min_users_per_track: int = 5,
                 max_tracks: int | None = None, require_metadata: bool = True, seed: int = 0) -> Corpus:
    """Assemble the Track-B :class:`~melomane.corpus.Corpus`.

    ``recordings`` = :func:`build_recording_table` output (index recording_mbid str; artist_credit_name, artist_mbid,
    title, first_release_year, isrcs, ...).  Steps: canonicalise MBIDs (``redirect``) -> dedupe -> vocabulary = recordings
    with >= ``min_users_per_track`` distinct users (and metadata if ``require_metadata``) -> sessions
    (:func:`melomane.sessions.sessionize`) -> drop tracks left without any session -> ``tracks`` (artist_id = dense int per
    artist MBID, negative unique ids when unknown) -> acoustic FamilyBlocks (``audio``; ``ab_csv`` = coarse CSV-dump fallback for
    families the JSON scan did not cover) and ``lyric_theme`` block
    (``lyrics``) -> coverage in ``meta['coverage']``.  ``sessions_meta`` carries only the *hashed* user int."""
    L = listens
    if redirect is not None:
        L = Listens(L.user, redirect.apply(L.mbid), L.ts)
    L = L.dedupe()
    vocab = select_vocab(L, min_users_per_track, max_tracks, seed)
    strs = mbid_array_to_str(vocab)
    if require_metadata:
        ok = np.array([s in recordings.index and str(recordings.at[s, "title"]).strip() != "" for s in strs], bool)
        vocab = vocab[ok]
    X, su = listens_to_sessions(L, vocab, gap_minutes, min_len, max_len)
    used = np.asarray(X.getnnz(axis=0) > 0).ravel() if X.shape[0] else np.zeros(vocab.size, bool)
    vocab = vocab[used]
    X = X[:, np.flatnonzero(used)].tocsr() if X.shape[0] else sp.csr_matrix((0, vocab.size), dtype=np.float32)
    strs = mbid_array_to_str(vocab)
    n = len(strs)
    rec = recordings.reindex(strs)
    tracks = pd.DataFrame({"ext_id": strs})
    tracks["title"] = rec["title"].fillna("").astype(str).to_numpy()
    tracks["artist"] = rec["artist_credit_name"].fillna("").astype(str).to_numpy()
    amb = rec["artist_mbid"].fillna("").astype(str).to_numpy()
    aid = np.zeros(n, np.int64); dense: dict[str, int] = {}; neg = 0
    for i in range(n):
        key = amb[i] or (("name:" + norm_text(tracks.at[i, "artist"])) if tracks.at[i, "artist"] else "")
        if key:
            aid[i] = dense.setdefault(key, len(dense))
        else:
            neg -= 1; aid[i] = neg
    tracks["artist_id"] = aid
    tracks["album"] = rec["release_name"].fillna("").astype(str).to_numpy() if "release_name" in rec else ""
    tracks["year"] = pd.to_numeric(rec["first_release_year"], errors="coerce").astype("float32").to_numpy() \
        if "first_release_year" in rec else np.nan
    tracks["isrcs"] = ["|".join(x) if isinstance(x, (list, tuple)) else "" for x in (rec["isrcs"] if "isrcs" in rec else [[]] * n)]
    tracks["n_sessions"] = np.asarray(X.sum(axis=0)).ravel().astype(np.int32) if X.shape[0] else np.zeros(n, np.int32)
    ti = _lookup(vocab, L.mbid)
    sel = ti >= 0
    tracks["n_listens"] = np.bincount(ti[sel], minlength=n).astype(np.int64) if n else np.zeros(0, np.int64)
    fam: dict[str, FamilyBlock] = {}
    if audio is not None:
        fam.update(audio.remap(redirect).blocks(vocab))
    if ab_csv is not None and not ab_csv.empty:                # coarse CSV-dump features fill families the JSON scan did not cover
        cs = ab_csv[[mbid_bytes(x) is not None for x in ab_csv.index]]
        if redirect is not None:
            cs = cs.set_axis(mbid_array_to_str(redirect.apply(to_mbid_array(cs.index))), axis=0)
        for k, b in ab_csv_blocks(cs[~cs.index.duplicated()], vocab).items():
            fam.setdefault(k, b)
    tracks["has_lyrics"] = False
    tracks["is_instrumental"] = False
    if lyrics is not None and lyrics.mbid.size:
        li = _lookup(np.sort(lyrics.mbid), vocab)
        srt = np.argsort(lyrics.mbid)
        hit = li >= 0
        src = srt[li[hit]]
        tracks.loc[hit, "has_lyrics"] = lyrics.has_lyrics[src]
        tracks.loc[hit, "is_instrumental"] = lyrics.instrumental[src]
        if lyrics.embedding is not None:
            rows = np.flatnonzero(hit)[lyrics.has_lyrics[src]]
            if rows.size:
                fam[LYRIC_FAMILY] = FamilyBlock(rows.astype(np.int64), lyrics.embedding[src[lyrics.has_lyrics[src]]].astype(np.float32),
                                                [f"lyr{i}" for i in range(lyrics.embedding.shape[1])],
                                                "LRCLIB presence+aggregate embedding (text discarded)")
    sm = pd.DataFrame({"user_id": su.astype(np.int64), "n_tracks": np.diff(X.indptr).astype(np.int32)})
    meta = {"source": "MBID-native stack (ListenBrainz listens + MusicBrainz canonical + AcousticBrainz 2022 + LRCLIB)",
            "document_type": f"listening session (gap {gap_minutes:g} min, {min_len}-{max_len} distinct tracks)",
            "min_users_per_track": min_users_per_track, "has_audio": audio is not None, "has_lyrics": lyrics is not None,
            "has_year": True, "has_order": False, "user_ids": "HMAC-hashed 63-bit ints (salt kept outside artefacts)",
            "licences": {"listenbrainz_listens": "UNVERIFIED (read COPYING in the dump)", "musicbrainz_canonical": "CC0",
                         "acousticbrainz": "CC0 (per MetaBrainz 2022-02-16 blog)", "lrclib": "lyrics never stored/released"},
            "seed": seed, "notes": ["first_release_year = canonical dump 'year' (UNVERIFIED semantics)",
                                    "AcousticBrainz stdev derived as sqrt(var); Onion alignment UNVERIFIED (PLAN §13.4)"]}
    c = Corpus(name, tracks, X, fam, None, None, {}, sm, meta)
    c.validate()
    c.meta["coverage"] = coverage_report(c).to_dict()
    return c


# ---- resumable end-to-end orchestration ------------------------------------------------------------------------------------
@dataclass
class TrackBConfig:
    """Inputs of :func:`ingest`.  Every path is optional except ``listens``; missing stages are skipped and reported."""
    listens: list[Path]                                  # ListenBrainz dump(s): .tar.zst / .jsonl(.gz|.zst)
    name: str = "mbid_stack"
    canonical: Path | None = None                        # musicbrainz-canonical-dump-*.tar.zst (metadata + redirects)
    ab_lowlevel: list[Path] = field(default_factory=list)   # acousticbrainz-lowlevel-json-20220623-<k>.tar.zst
    ab_csv: list[Path] = field(default_factory=list)     # acousticbrainz-lowlevel-features-20220623-{lowlevel,rhythm,tonal}.tar.zst
    lrclib_dump: Path | None = None                      # decompressed LRCLIB SQLite dump (preferred for scale)
    lrclib_api: bool = False                             # else use the <= 1 req/s API client (cache under ws.cache)
    user_agent: str = USER_AGENT_HINT
    year_range: tuple[int, int] | None = None
    min_ts: int | None = None
    max_ts: int | None = None                            # temporal design: train <= 2019 -> max_ts = 2020-01-01
    prefer: str = "mapping"
    min_users_per_track: int = 5
    max_tracks: int | None = None
    gap_minutes: float = 30.0
    min_len: int = 5
    max_len: int = 200
    lyrics_max_lookups: int | None = None                # cap per run (API); rerun to continue
    seed: int = 0


def ingest(ws: "Workspace", cfg: TrackBConfig, should_stop: Callable[[], bool] | None = None) -> Corpus | None:
    """Run Track B end to end under ``ws.interim/<name>/track_b`` and write the corpus to ``ws.interim/<name>``.

    Stages (each idempotent; killing the process anywhere and re-running continues): 1 listens -> shards; 2 canonical +
    redirect; 3 AcousticBrainz JSON scan (candidates = vocabulary + non-canonical variants); 4 lyrics (dump or API cache);
    5 build_corpus + write.  Returns None if stopped early (``should_stop``) or the lyric API budget/queue is not finished -
    call again to resume.  The user-hash salt is kept in ``ws.data/private/user_salt.bin`` (never inside corpus artefacts)."""
    base = ws.d("interim", cfg.name, "track_b", base="data")
    hasher = UserHasher.from_file(ws.d("private", base="data") / "user_salt.bin")
    lstats = extract_listens(cfg.listens, base / "listens", hasher, min_ts=cfg.min_ts, max_ts=cfg.max_ts, year_range=cfg.year_range,
                             prefer=cfg.prefer, should_stop=should_stop)
    if lstats.stopped:
        log.warning("stopped during listens extraction; re-run to resume")
        return None
    L = load_listens(base / "listens")
    if not len(L):
        raise ValueError("no listens extracted (wrong files, filters too strict, or no recording MBIDs in the dump)")
    redirect = RedirectMap.from_csv(cfg.canonical) if cfg.canonical is not None else None
    if redirect is not None and not len(redirect):
        redirect = None
    Lc = Listens(L.user, redirect.apply(L.mbid), L.ts) if redirect is not None else L
    vocab0 = select_vocab(Lc.dedupe(), cfg.min_users_per_track, cfg.max_tracks, cfg.seed)
    log.info("candidate vocabulary: %d recordings (>= %d users)", vocab0.size, cfg.min_users_per_track)
    mapping_meta = load_mapping_metadata(base / "listens")
    if cfg.canonical is not None:
        canon = build_recording_table(read_canonical_metadata(cfg.canonical, keep_mbids=set(mbid_array_to_str(vocab0))))
        recs = pd.concat([canon, mapping_meta[~mapping_meta.index.isin(canon.index)]]) if len(mapping_meta) else canon
    else:
        recs = mapping_meta
    strs = mbid_array_to_str(vocab0)
    has_meta = np.array([m in recs.index and str(recs.at[m, "title"]).strip() != "" for m in strs], bool) if len(recs) else np.zeros(len(strs), bool)
    vocab = vocab0[has_meta]
    audio = None
    if cfg.ab_lowlevel:
        cand = vocab if redirect is None else np.unique(np.concatenate([vocab, redirect.variants_of(vocab)]))
        a = scan_lowlevel_archives(cfg.ab_lowlevel, base / "audio", cand, should_stop=should_stop)
        if a.stopped:
            log.warning("stopped during AcousticBrainz scan; re-run to resume")
            return None
        audio = load_audio(base / "audio")
    ab_csv = read_ab_csv_features(cfg.ab_csv, vocab if redirect is None else np.unique(np.concatenate([vocab, redirect.variants_of(vocab)]))) \
        if cfg.ab_csv else None
    lyrics = None
    if cfg.lrclib_dump is not None or cfg.lrclib_api:
        src = LrclibDump(cfg.lrclib_dump) if cfg.lrclib_dump is not None else \
            LrclibClient(ws.d("cache", "lrclib", base="data") / "lrclib_cache.sqlite", cfg.user_agent)
        try:
            lyrics = collect_lyrics(vocab, recs, src, should_stop=should_stop, max_lookups=cfg.lyrics_max_lookups)
        finally:
            (src.close() if isinstance(src, LrclibDump) else src.cache.close())
        if lyrics.n_pending:
            log.warning("lyrics: %d recordings still pending; re-run to continue (cached lookups are free)", lyrics.n_pending)
            return None
    c = build_corpus(L, recs, redirect=redirect, audio=audio, lyrics=lyrics, ab_csv=ab_csv, name=cfg.name,
                     gap_minutes=cfg.gap_minutes, min_len=cfg.min_len, max_len=cfg.max_len,
                     min_users_per_track=cfg.min_users_per_track, max_tracks=cfg.max_tracks, seed=cfg.seed)
    c.meta["extract_stats"] = lstats.as_dict()
    write_corpus(c, ws.d("interim", cfg.name, base="data"))
    write_json(base / "coverage.json", c.meta["coverage"])
    log.info("Track B corpus written: %d tracks, %d sessions; coverage %s", c.n_tracks, c.sessions.shape[0], c.meta["coverage"])
    return c


# ---- download planning --------------------------------------------------------------------------------------------------
_GB = 2 ** 30
_MB = 2 ** 20
LB_DUMP_DIR = "listenbrainz-dump-2663-20260915-000002-full"
CANON_DIR = "musicbrainz-canonical-dump-20260917-080002"
AB_ROOT = "https://data.metabrainz.org/pub/musicbrainz/acousticbrainz/dumps"
AB_SIZES_GB = [20, 20, 20, 20, 20, 20, 20, 20, 21, 20, 20, 20, 19, 19, 21, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 9]


@dataclass
class DownloadItem:
    key: str
    url: str
    expected_bytes: int | None
    licence: str
    purpose: str
    tier: str                    # core | recommended | optional
    verified: bool               # URL + size read from an official directory listing on 2026-09-29
    note: str = ""
    probed_bytes: int | None = None

    @property
    def size_bytes(self) -> int | None:
        return self.probed_bytes if self.probed_bytes is not None else self.expected_bytes


def plan_downloads(*, listenbrainz_dir: str = LB_DUMP_DIR, canonical_dir: str = CANON_DIR, ab_archives: Sequence[int] | None = None,
                   include_sample: bool = True, include_musicbrainz_json: bool = False) -> list[DownloadItem]:
    """Ordered download list for Track B (sizes = Apache index listing values, rounded; ``probe_sizes`` refines them).
    Dump directory names are date-stamped and rotate: use :func:`pick_latest` on the parent index to refresh them.
    ``ab_archives`` = which of the 30 low-level archives (default none: pull them after the candidate set is known)."""
    D = "https://data.metabrainz.org/pub/musicbrainz"
    ab_archives = [] if ab_archives is None else list(ab_archives)
    items: list[DownloadItem] = []
    if include_sample:
        items.append(DownloadItem("lb_sample", f"{D}/listenbrainz/sample/listenbrainz-sample-20250610-094311-full/listenbrainz-sample-dump-20250610-094311.tar.zst",
                                  235 * _MB, "UNVERIFIED", "schema check of real dump lines (Stage 0)", "core", True, "2025-06 sample"))
    items += [
        DownloadItem("canonical", f"{D}/canonical_data/{canonical_dir}/{canonical_dir}.tar.zst", 2 * _GB, "CC0",
                     "recording metadata, redirects, offline (artist,title)->MBID", "core", True),
        DownloadItem("ab_csv_lowlevel", f"{AB_ROOT}/acousticbrainz-lowlevel-features-20220623/acousticbrainz-lowlevel-features-20220623-lowlevel.tar.zst",
                     1 * _GB, "CC0 (blog)", "coarse loudness/MFCC0 features, all 29.5 M submissions", "core", True),
        DownloadItem("ab_csv_rhythm", f"{AB_ROOT}/acousticbrainz-lowlevel-features-20220623/acousticbrainz-lowlevel-features-20220623-rhythm.tar.zst",
                     1 * _GB, "CC0 (blog)", "bpm/danceability/onset rate", "core", True),
        DownloadItem("ab_csv_tonal", f"{AB_ROOT}/acousticbrainz-lowlevel-features-20220623/acousticbrainz-lowlevel-features-20220623-tonal.tar.zst",
                     843 * _MB, "CC0 (blog)", "key/scale/tuning", "core", True),
        DownloadItem("lb_listens_full", f"{D}/listenbrainz/fullexport/{listenbrainz_dir}/{listenbrainz_dir.replace('listenbrainz-dump-', 'listenbrainz-listens-dump-')}.tar.zst",
                     229 * _GB, "UNVERIFIED (see COPYING inside)", "behaviour: listens JSONL", "core", True,
                     "stream with year_range/min_ts + mbid_filter; do not extract to disk"),
    ]
    if include_musicbrainz_json:
        items.append(DownloadItem("mb_json_recording", f"{D}/data/json-dumps/20260926-001002/recording.tar.xz", 32 * _MB, "CC0 (core data)",
                                  "ISRC + first-release-date per recording", "optional", True, "size looks small for all recordings: inspect before relying"))
    for k in ab_archives:
        items.append(DownloadItem(f"ab_lowlevel_{k}", f"{AB_ROOT}/acousticbrainz-lowlevel-json-20220623/acousticbrainz-lowlevel-json-20220623-{k}.tar.zst",
                                  AB_SIZES_GB[k] * _GB, "CC0 (blog)", "full Essentia low-level JSON (stream-filtered)", "recommended", True))
    items.append(DownloadItem("lrclib_dump", "https://lrclib.net/db-dumps", None, "UNVERIFIED", "lyric presence + aggregate embedding",
                              "recommended", False, "page is a JS app: size, file name, compression unknown"))
    return items


@dataclass
class IndexEntry:
    name: str
    is_dir: bool
    date: str
    size_text: str


_IDX = re.compile(r'<a href="([^"?]+)">[^<]*</a>\s+(\d{2}-[A-Za-z]{3}-\d{4} \d{2}:\d{2})\s+(\S+)')


def parse_apache_index(html: str) -> list[IndexEntry]:
    """Parse an Apache/nginx autoindex page (as served by data.metabrainz.org) into entries."""
    out = []
    for m in _IDX.finditer(html):
        href = m.group(1)
        if href.startswith(("/", "..", "http")):
            continue
        out.append(IndexEntry(href.rstrip("/"), href.endswith("/"), m.group(2), m.group(3)))
    return out


def parse_size(txt: str) -> int | None:
    """'229G' -> bytes (powers of 1024, as autoindex rounds); '-' -> None."""
    m = re.fullmatch(r"([\d.]+)([KMGT]?)", txt.strip())
    if not m:
        return None
    return int(float(m.group(1)) * {"": 1, "K": 2 ** 10, "M": 2 ** 20, "G": 2 ** 30, "T": 2 ** 40}[m.group(2)])


_MON = {m: i for i, m in enumerate(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], 1)}


def _idx_date(d: str) -> tuple:
    """'17-Sep-2026 18:17' -> sortable tuple (locale independent)."""
    m = re.match(r"(\d{2})-([A-Za-z]{3})-(\d{4}) (\d{2}):(\d{2})", d)
    return (int(m.group(3)), _MON.get(m.group(2), 0), int(m.group(1)), int(m.group(4)), int(m.group(5))) if m else (0,)


def pick_latest(entries: Sequence[IndexEntry], contains: str = "") -> IndexEntry | None:
    """Most recent directory by its listing date (ties: name)."""
    ds = [e for e in entries if e.is_dir and contains in e.name]
    return max(ds, key=lambda e: (_idx_date(e.date), e.name), default=None)


ProbeTransport = Callable[[str], "tuple[int, Mapping[str, str]]"]


def _httpx_head(url: str) -> tuple[int, Mapping[str, str]]:
    import httpx
    r = httpx.head(url, follow_redirects=True, timeout=30.0, headers={"User-Agent": USER_AGENT_HINT})
    return r.status_code, {k.lower(): v for k, v in r.headers.items()}


def probe_sizes(items: Sequence[DownloadItem], transport: ProbeTransport | None = None, min_interval_s: float = 0.5,
                sleep: Callable[[float], None] = time.sleep) -> list[DownloadItem]:
    """HEAD every item and store ``Content-Length`` in ``probed_bytes`` (returns copies; failures keep the expected size).
    ``transport(url) -> (status, lower-cased headers)``; default uses httpx.  No downloads, one polite request per item."""
    tr = transport or _httpx_head
    out = []
    for i, it in enumerate(items):
        if i:
            sleep(min_interval_s)
        try:
            status, hd = tr(it.url)
        except Exception as e:
            log.warning("HEAD %s failed: %s", it.key, type(e).__name__)
            out.append(it); continue
        cl = hd.get("content-length")
        out.append(dataclasses.replace(it, probed_bytes=int(cl) if status == 200 and cl and cl.isdigit() else None))
    return out


def total_gb(items: Iterable[DownloadItem]) -> float:
    return sum(i.size_bytes or 0 for i in items) / 1e9


def verify_sha256(path: str | Path, expected: str) -> bool:
    """Compare against the ``.sha256`` file published next to each dump."""
    return sha256_file(Path(path)) == expected.strip().split()[0].lower()


# ======================================================================================================================
# 6. alignment report shared with Track C (PLAN S11 / §13.4)
# ======================================================================================================================
_STAT_SUFFIX = re.compile(r"\.(mean|stdev|std|var|median|min|max|skew|kurtosis|dmean|dmean2|dvar|dvar2|cov|icov)$")


def _stem(name: str) -> str:
    n = re.sub(r"\[\d+\]$", "", name.strip())
    n = re.sub(r"(\.key)_(?:cos|sin)$|(\.scale)_major$", lambda m: m.group(1) or m.group(2), n)   # expanded key/scale columns
    return _STAT_SUFFIX.sub("", n)


def doc_descriptor_names(doc: Mapping, skip: Sequence[str] = ("metadata",), max_depth: int = 5) -> list[str]:
    """Dotted leaf names of an Essentia/AcousticBrainz-style JSON document (``lowlevel.mfcc.mean`` ...), for feeding
    :func:`align_report` with the descriptors a catalog *actually* ships."""
    out: list[str] = []

    def walk(node, prefix: str, depth: int) -> None:
        if isinstance(node, Mapping) and depth < max_depth:
            for k, v in node.items():
                if not prefix and k in skip:
                    continue
                walk(v, f"{prefix}.{k}" if prefix else str(k), depth + 1)
        elif prefix:
            out.append(prefix)

    walk(doc, "", 0)
    return out


def target_stems(families: Sequence[str] | None = None) -> dict[str, set[str]]:
    """Essentia-descriptor stems each Melomane family needs (from the AB-based definition)."""
    return {f: {_stem(b) for s in AB_FAMILY_SPECS[f] for b in s.base_names()} for f in (families or FAMILY_NAMES)}


def align_report(source_columns: Sequence[str], target_columns: Sequence[str] | None = None, *,
                 concept_map: Mapping[str, str] | None = None, source_name: str = "source",
                 target_name: str = "acousticbrainz-lowlevel(Essentia)") -> dict:
    """Which Melomane families can be built from ``source_columns`` with the *same feature definition* as the target?

    ``target_columns``: descriptor names defining the target (default: the AcousticBrainz-based specs; pass the Onion
    ``id_essentia`` names - e.g. ``essentia_descriptor_names.json`` - to test Track A alignment).  Names are compared at
    descriptor level (statistic suffix and ``[i]`` stripped).  ``concept_map`` maps *source* names/stems to target stems
    that are the same *concept* but computed by a different extractor (e.g. librosa MFCC vs Essentia MFCC): those count
    as ``conceptual`` (usable only as an approximate proxy, NOT the same feature definition).

    Per family: ``status`` = 'aligned' (>= 80 % of the target descriptors present under identical names) | 'partial' (30-80 %)
    | 'conceptual_only' (no identical names but concept-mapped descriptors exist) | 'not_alignable' | 'not_in_target'.  ``can_use_identical_definition``
    is True only if every *present-in-target* acoustic family is 'aligned' or 'partial'.  PLAN S11: when False the pipeline
    must say so and run the cross-catalog transfer inside Track B instead."""
    base = target_stems()
    extra: list[str] = []
    if target_columns is None:
        tgt = base
    else:
        ts = {_stem(c) for c in target_columns}
        tgt = {f: st & ts for f, st in base.items()}                  # descriptors the target defines for the family
        known = set().union(*base.values())
        extra = sorted(x for x in ts if x.startswith(("lowlevel.", "rhythm.", "tonal.")) and x not in known)
    src = {_stem(c) for c in source_columns}
    cm = {_stem(k): _stem(v) for k, v in (concept_map or {}).items()}
    src_concept = {cm[s] for s in src if s in cm}
    fams: dict[str, dict] = {}
    for f in FAMILY_NAMES:
        need = tgt[f]
        if not need:
            fams[f] = {"target_descriptors": 0, "exact": 0, "conceptual": 0, "exact_fraction": 0.0, "status": "not_in_target",
                       "missing_examples": []}
            continue
        exact = need & src
        conc = (need & src_concept) - exact
        frac = len(exact) / len(need) if need else 0.0
        if frac >= 0.8:
            status = "aligned"
        elif frac >= 0.3:
            status = "partial"
        elif conc:
            status = "conceptual_only"
        else:
            status = "not_alignable"
        fams[f] = {"target_descriptors": len(need), "exact": len(exact), "conceptual": len(conc), "exact_fraction": round(frac, 3),
                   "status": status, "missing_examples": sorted(need - exact - conc)[:5]}
    ok = all(v["status"] in ("aligned", "partial") for v in fams.values() if v["status"] != "not_in_target")
    any_ok = any(v["status"] in ("aligned", "partial") for v in fams.values())
    if ok:
        rec = "Feature definitions align: the Track-A compound vocabulary can be projected with the same scalers/PCA/centroids."
    elif any_ok:
        rec = ("Only some families align identically (see per-family status): project those; report the others as not comparable "
               "(PLAN S11) or run the transfer inside Track B.")
    else:
        rec = ("No family shares the target's feature definition: do NOT project with Track-A scalers/PCA/centroids; refit inside this "
               "catalog or run the cross-catalog test inside Track B and document (PLAN S11).")
    return {"source": source_name, "target": target_name, "families": fams, "can_use_identical_definition": ok,
            "any_family_aligned": any_ok, "target_descriptors_outside_ab_specs": extra, "recommendation": rec}
