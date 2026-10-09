"""Music4All-Onion adapter (Zenodo 6609677 v1 / 15394646 v2, CC BY 4.0; 109,269 tracks, 119,140 users, 252,984,396 Last.fm
listening records, ~49 files).  Track A of PLAN: behaviour + measured audio/lyrics/tag features for the *same* tracks.

EVERYTHING ABOUT THE REAL FILES IS UNVERIFIED (the sandbox cannot reach Zenodo): file names follow the dataset description
(``userid_trackid_timestamp``, ``userid_trackid_count``, ``id_essentia``, ``id_mfcc_stats``, ``id_blf_*``, ``id_compare_*``,
``id_emobase_*``, ``id_gems``, ``id_vad_bow``, ``id_bert``, ``id_lyrics_*``, ``id_jukebox`` ..., ``id_tags_*``,
``id_genres_tf-idf``, possibly ``id_information`` / ``id_metadata``); their *formats* (separator, header, compression,
timestamp unit, id spelling, whether artist/title exist, Essentia column naming, list-valued cells) are guessed, so every
reader is tolerant and everything it decides is logged.  Run ``inventory`` / ``write_inventory`` on the downloaded files first
(PLAN S0) and adjust the ``families`` config if a guess was wrong.

Tolerance rules
  * file names are matched case-/underscore-/hyphen-insensitively and extension-agnostically (``find_file``);
  * delimited tables: separator (tab/comma/semicolon/pipe) and header presence are sniffed; bz2/gz/xz/zip by extension;
    first column = track id (kept as *string*), other columns numeric; cells like ``"[0.1, 0.2]"`` are expanded to columns
    (variable-length lists, e.g. beat positions, are skipped); parquet / json variants are read too;
  * a missing family source shrinks or omits the family (logged), it never crashes the build.

Leakage rules (PLAN 4.4): held-out measured signals (openSMILE/emobase voice columns, ``id_gems``,
``id_lyrics_sentiment_functionals``, ``id_vad_bow``) go to ``Corpus.heldout`` ONLY; learned embeddings (jukebox, maest,
musicnn, resnet, vgg19, ivec*, incp; also bert / lyrics word2vec) go to ``Corpus.external``; families are refused if they name a
forbidden file, voice-like columns are stripped from every family, and ``check_family_leakage`` re-verifies the result.
(``id_bert`` / ``id_lyrics_word2vec`` are *both* lyric_theme inputs, per PLAN, and external agreement targets.)

Blocker-A: if the open release carries no artist/title, ``tracks.artist/title`` are empty strings, ``artist_id`` are unique
negative ids and ``meta['blocker_a'] = True`` (+ ``meta['blocker_a_note']``); the pipeline keys off that flag.

Memory: all selected feature tables are cached in RAM until the build ends (Onion is ~110k rows, so a few GB); events are
streamed in chunks and only (user, track, ts) int arrays are retained (~16 B/event; 253M events ~ 4 GB before sharded
sessionisation, which caps the working set by processing user shards independently).
"""
from __future__ import annotations

import bz2
import contextlib
import csv
import fnmatch
import gzip
import lzma
import math
import re
import sys
import warnings
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Sequence

try:
    csv.field_size_limit(min(sys.maxsize, 2147483647))
except (OverflowError, ValueError):
    csv.field_size_limit(2147483647)

import numpy as np
import pandas as pd
import scipy.sparse as sp

from ..corpus import Corpus, FamilyBlock, write_corpus
from ..curation import normalize_artist
from ..runtime import get_logger, jsonable, read_json, write_json
from ..sessions import sessionize

log = get_logger("melomane.adapters.onion")

ZENODO_RECORDS = {"v1": "6609677", "v2": "15394646"}
LICENSE = "CC BY 4.0 (Music4All-Onion; Last.fm-derived listening data - keep attribution)"

_COMPRESSION = (".bz2", ".gz", ".xz", ".zip")
_TABLE_EXT = (".tsv", ".csv", ".txt", ".tab", ".parquet", ".json", ".jsonl", ".ndjson")
_EXT_RANK = {".parquet": 0, ".tsv": 1, ".csv": 1, ".tab": 1, ".txt": 1, ".json": 2, ".jsonl": 2, ".ndjson": 2}

# ----------------------------------------------------------------------------------------------------------------------
# configuration defaults (PLAN 4.4)
# ----------------------------------------------------------------------------------------------------------------------
DEFAULT_FAMILIES: dict[str, dict] = {
    "timbre": {"sources": ["id_mfcc_stats", "id_compare_mfcc_stats"]},
    "spectral_shape": {"sources": ["id_blf_spectral", "id_blf_spectralcontrast", "id_blf_deltaspectral", "id_blf_vardeltaspectral"]},
    "rhythm_groove": {"sources": ["id_blf_logfluc",
                                  {"file": "id_essentia", "columns": ["rhythm.*"], "exclude": ["*beats.position*"]}]},
    "harmony_tonal": {"sources": ["id_chroma_bow", "id_blf_correlation",
                                  {"file": "id_essentia", "columns": ["tonal.*"], "exclude": ["*beats.position*"]}]},
    "dynamics_loudness": {"sources": [{"file": "id_essentia", "columns": ["*loudness*", "*dynamic.complexity*"],
                                       "exclude": ["rhythm.*", "*.beats.*"]}, "id_compare_pcm_stats"]},
    "lyric_theme": {"sources": ["id_bert", "id_lyrics_word2vec"], "lyric": True},
    "lyric_lexicon": {"sources": ["id_lyrics_tf-idf"], "lyric": True, "reduce": {"method": "svd", "n_components": 64}},
}

# Essentia descriptor candidates per measured scalar (probe stratum (a)); first pattern that hits a numeric column wins.
DEFAULT_MEASURED: dict[str, list[str]] = {
    "bpm": ["rhythm.bpm"],
    "onset_rate": ["rhythm.onset_rate"],
    "danceability": ["rhythm.danceability"],
    "loudness": ["lowlevel.average_loudness"],
    "loudness_ebu128": ["lowlevel.loudness_ebu128.integrated"],
    "loudness_range": ["lowlevel.loudness_ebu128.loudness_range"],
    "dynamic_complexity": ["lowlevel.dynamic_complexity"],
    "spectral_centroid": ["lowlevel.spectral_centroid.mean"],
    "spectral_flux": ["lowlevel.spectral_flux.mean"],
    "zero_crossing_rate": ["lowlevel.zerocrossingrate.mean"],
    "beats_loudness": ["rhythm.beats_loudness.mean"],
    "chords_strength": ["tonal.chords_strength.mean"],
    "chords_change_rate": ["tonal.chords_changes_rate"],
    "key_strength": ["tonal.key_edma.strength", "tonal.key_krumhansl.strength", "tonal.key_temperley.strength"],
    "hpcp_entropy": ["tonal.hpcp_entropy.mean"],
}
MEASURED_SOURCES = ["id_essentia", "id_mfcc_stats"]
_MFCC_EXCLUDE = ("std", "stdev", "var", "delta", "cov", "min", "max", "median", "skew", "kurt", "range")

DEFAULT_EXTERNAL = ["id_jukebox", "id_maest", "id_musicnn", "id_bert", "id_lyrics_word2vec", "id_resnet", "id_vgg19",
                    "id_ivec*", "id_incp*"]
LYRIC_STEMS = ["id_lyrics_tf-idf", "id_lyrics_word2vec", "id_bert", "processed_lyrics"]
METADATA_STEMS = ["id_information", "id_metadata", "id_meta", "id_info", "metadata", "id_lang", "id_language"]
TAG_STEMS = ["id_genres", "id_genres_tf-idf", "id_tags_dict", "id_tags*"]
HELDOUT_STEMS = {"gems": "id_gems", "sentiment": "id_lyrics_sentiment_functionals", "vad": "id_vad_bow"}
VOICE_STEMS = ["id_compare_*", "id_emobase_*"]
FORBIDDEN_FAMILY_STEMS = ["id_gems", "id_lyrics_sentiment_functionals", "id_vad_bow", "id_emobase*", "id_jukebox", "id_maest",
                          "id_musicnn", "id_ivec*", "id_incp*", "id_resnet", "id_vgg19", "id_tags*", "id_genres*"]
_VOICE_GUARD = re.compile(r"(^|[^a-z])(f0|hnr|jitter|shimmer|voicing|voiceprob|voice|lsp)")
_VOICE_CATS = {"f0": ("f0", "pitch"), "hnr": ("hnr",), "jitter": ("jitter",), "shimmer": ("shimmer",),
               "voicing": ("voicing", "voiceprob", "voiced"), "lsp": ("lsp",)}
EXPECTED_STEMS = ["userid_trackid_timestamp", "userid_trackid_count", "id_essentia", "id_mfcc_stats", "id_blf_spectral",
                  "id_blf_spectralcontrast", "id_blf_deltaspectral", "id_blf_vardeltaspectral", "id_blf_logfluc",
                  "id_blf_correlation", "id_chroma_bow", "id_compare_mfcc_stats", "id_compare_pcm_stats", "id_gems",
                  "id_vad_bow", "id_bert", "id_lyrics_word2vec", "id_lyrics_tf-idf", "id_lyrics_sentiment_functionals",
                  "id_jukebox", "id_maest", "id_musicnn", "id_resnet", "id_vgg19", "id_genres_tf-idf"]

_META_ALIASES = {
    "artist": ("artist", "artistname", "artists", "performer", "artisttitle"),
    "title": ("song", "title", "track", "trackname", "songname", "songtitle", "name"),
    "album": ("album", "albumname", "albumtitle"),
    "year": ("year", "releaseyear", "release", "releasedate", "date"),
    "popularity": ("popularity", "pop"),
    "language": ("lang", "language"),
    "artist_id": ("artistid", "artistuid", "artistmbid"),
    "duration_ms": ("durationms", "duration", "length"),
    "tags": ("tags", "toptags", "genres", "genre"),
}


# ----------------------------------------------------------------------------------------------------------------------
# file discovery (case / underscore / hyphen / extension insensitive)
# ----------------------------------------------------------------------------------------------------------------------
def _strip_ext(name: str) -> str:
    n = name.lower()
    while True:
        for e in _COMPRESSION + _TABLE_EXT:
            if n.endswith(e) and len(n) > len(e):
                n = n[: -len(e)]
                break
        else:
            return n


def norm_stem(name: str) -> str:
    """'ID_Lyrics_TF-IDF.tsv.bz2' -> 'idlyricstfidf' (keeps ``*``/``?`` so patterns can be normalised too)."""
    return re.sub(r"[^a-z0-9*?]", "", _strip_ext(str(name)))


class FileIndex:
    """Listing of a raw directory; ``find_files('id_compare_*')`` supports fnmatch wildcards on normalised names."""

    def __init__(self, raw_dir: Path):
        self.root = Path(raw_dir)
        self.files: list[Path] = []
        for p in sorted(self.root.rglob("*")) if self.root.exists() else []:
            if not p.is_file() or p.suffix.lower() in (".part", ".tmp", ".crdownload", ".md", ".pdf", ".py", ".lock", ".tar") or p.name.endswith(".tar.gz"):
                continue
            ns = norm_stem(p.name)
            if ns.startswith(("inventory", "zenodo", "readme", "license", "licence", "checksum", "md5")):
                continue
            self.files.append(p)

    @staticmethod
    def _rank(p: Path) -> tuple:
        ext = p.suffix.lower() if p.suffix.lower() not in _COMPRESSION else Path(p.stem).suffix.lower()
        return (_EXT_RANK.get(ext, 3), p.suffix.lower() in _COMPRESSION, len(p.parts), str(p))

    def find_files(self, stem: str) -> list[Path]:
        pat = norm_stem(stem)
        best: dict[str, Path] = {}
        for p in self.files:
            ns = norm_stem(p.name)
            if fnmatch.fnmatchcase(ns, pat):
                if ns not in best or self._rank(p) < self._rank(best[ns]):
                    best[ns] = p
        return [best[k] for k in sorted(best)]

    def find_file(self, stem: str) -> Path | None:
        f = self.find_files(stem)
        return f[0] if f else None


def find_file(raw_dir: Path, stem: str) -> Path | None:
    """First file in ``raw_dir`` (recursive) whose normalised stem equals ``stem``'s, whatever extension/compression."""
    return FileIndex(raw_dir).find_file(stem)


# ----------------------------------------------------------------------------------------------------------------------
# tolerant table reading
# ----------------------------------------------------------------------------------------------------------------------
@contextlib.contextmanager
def _open_binary(path: Path):
    s = Path(path).suffix.lower()
    if s == ".bz2":
        f = bz2.open(path, "rb")
    elif s == ".gz":
        f = gzip.open(path, "rb")
    elif s == ".xz":
        f = lzma.open(path, "rb")
    elif s == ".zip":
        z = zipfile.ZipFile(path)
        try:
            names = [n for n in z.namelist() if not n.endswith("/")]
            with z.open(names[0]) as m:
                yield m
        finally:
            z.close()
        return
    else:
        f = open(path, "rb")
    try:
        yield f
    finally:
        f.close()


def _head_text(path: Path, max_lines: int = 16, max_bytes: int = 8 * 1024 * 1024) -> str:
    """Read the first several complete lines from a table (handles wide tables up to 8MB header)."""
    import io
    with _open_binary(path) as f:
        wrapper = io.TextIOWrapper(f, encoding="utf-8-sig", errors="replace")
        lines = []
        bytes_read = 0
        for _ in range(max_lines):
            line = wrapper.readline()
            if not line:
                break
            lines.append(line)
            bytes_read += len(line)
            if bytes_read >= max_bytes:
                break
    return "".join(lines)


def count_lines(path: Path, chunk: int = 1 << 24) -> int:
    """Number of physical lines (an upper bound on rows when quoted cells contain newlines)."""
    total, last = 0, b"\n"
    with _open_binary(path) as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            total += b.count(b"\n")
            last = b[-1:]
    return total + (0 if last == b"\n" else 1)


def sniff_separator(lines: Sequence[str]) -> str:
    """Tab / comma / semicolon / pipe by consistent field counts (csv-quote aware); tab wins ties."""
    lines = [ln for ln in lines if ln.strip()][:30]
    best, best_score = "\t", (-1, -1)
    for sep in ("\t", ",", ";", "|"):
        try:
            counts = [len(r) for r in csv.reader(lines, delimiter=sep)]
        except csv.Error:
            continue
        if not counts:
            continue
        mode = max(set(counts), key=counts.count)
        consistent = counts.count(mode) / len(counts)
        if mode >= 2 and consistent >= 0.9:
            score = (1, mode)
            if score > best_score:
                best, best_score = sep, score
                if sep == "\t":
                    break                       # tab first: list-valued cells may contain commas
    return best


def _is_numeric_like(x: str) -> bool:
    x = x.strip()
    if not x:
        return False
    if x[0] in "[(" and x[-1] in "])":
        inner = x[1:-1].strip()
        if not inner:
            return False
        first = inner.split(",")[0].strip()
        try:
            float(first)
            return True
        except ValueError:
            return False
    try:
        float(x)
        return True
    except ValueError:
        return False


_NAME_LIKE_ID = re.compile(r"(|index|id|uid|tid|[a-z_ ]*id|unnamed.*)")


def has_header(rows: Sequence[Sequence[str]]) -> bool:
    """Heuristic header detection on the first few parsed rows (see module docstring: UNVERIFIED on real files)."""
    if not rows:
        return False
    r0, rest = list(rows[0]), rows[1:]
    if len(r0) == 1:
        return r0[0].strip().lower() in {"id", "track_id", "trackid", "song_id", "user_id", "userid"}
    tail = r0[1:]
    flags = [_is_numeric_like(c) for c in tail]
    if not any(flags):
        if _NAME_LIKE_ID.fullmatch(r0[0].strip().lower()):
            return True
        return any(_is_numeric_like(c) for row in rest for c in list(row)[1:len(r0)])
    if all(re.fullmatch(r"\d+", c.strip()) for c in tail):               # numeric column labels 0..k-1 / 1..k
        if not _NAME_LIKE_ID.fullmatch(r0[0].strip().lower()):
            return False                                                # 't1,1,2' is data, ',1,2' / 'id,1,2' a header
        nums = [int(c) for c in tail]
        return nums in (list(range(len(nums))), list(range(1, len(nums) + 1))) and \
            bool(rest) and all(_is_numeric_like(c) for c in rest[0][1:len(r0)])
    return False


@dataclass
class TableLayout:
    kind: str                     # 'delimited' | 'parquet' | 'json'
    sep: str = "\t"
    header: bool = True
    names: list[str] = field(default_factory=list)


def _unique_names(names: Sequence[str]) -> list[str]:
    out, seen = [], {}
    for i, n in enumerate(names):
        n = str(n).strip() or ("id" if i == 0 else f"c{i}")
        if n in seen:
            seen[n] += 1
            n = f"{n}.{seen[n]}"
        else:
            seen[n] = 0
        out.append(n)
    return out


def sniff_layout(path: Path, default_names: Sequence[str] | None = None) -> TableLayout:
    p = Path(path)
    inner = p.suffix.lower() if p.suffix.lower() not in _COMPRESSION else Path(p.stem).suffix.lower()
    if inner == ".parquet":
        return TableLayout("parquet")
    if inner in (".json", ".jsonl", ".ndjson"):
        return TableLayout("json")
    lines = _head_text(p).splitlines()
    sep = sniff_separator(lines)
    rows = list(csv.reader(lines[:8], delimiter=sep))
    hdr = has_header(rows)
    ncols = len(rows[0]) if rows else 1
    if hdr:
        names = _unique_names(rows[0])
    elif default_names and len(default_names) == ncols:
        names = list(default_names)
    else:
        names = ["id"] + [f"c{i}" for i in range(1, ncols)]
    return TableLayout("delimited", sep, hdr, names)


_LIST_START = ("[", "(")


def _parse_list_column(s: pd.Series, name: str, max_len: int = 4096) -> pd.DataFrame | None:
    """Expand a column of stringified lists into float32 columns ``name[i]``; None if lengths are not (mostly) constant."""
    vals = s.to_numpy(object)
    parsed: list[np.ndarray | None] = []
    lens = []
    for v in vals:
        if isinstance(v, str) and v.strip():
            t = v.strip().strip("[]()").replace(",", " ").replace("None", "nan").replace("null", "nan")
            try:
                arr = np.array(t.split(), dtype=np.float64)
            except ValueError:
                arr = None
            parsed.append(arr)
            lens.append(-1 if arr is None else arr.size)
        else:
            parsed.append(None)
            lens.append(-1)
    lens_a = np.array(lens)
    good = lens_a[lens_a >= 0]
    if good.size == 0:
        return None
    L = int(np.bincount(good).argmax())
    if L == 0 or L > max_len or (good != L).mean() > 0.02:
        return None
    M = np.full((len(vals), L), np.nan, dtype=np.float32)
    for i, a in enumerate(parsed):
        if a is not None and a.size == L:
            M[i] = a
    return pd.DataFrame(M, columns=[f"{name}[{i}]" for i in range(L)], index=s.index)


def _finalize(df: pd.DataFrame, expand_lists: bool, float32: bool, source: str = "") -> pd.DataFrame:
    """id column -> stripped str; object columns -> numeric / expanded lists / kept as strings."""
    if df.empty or df.shape[1] == 0:
        return df
    df = df.copy()
    if df.shape[1] >= 2 and str(df.columns[0]).lower() in ("id", "unnamed: 0", "index", "") and str(df.columns[1]).lower() in ("id", "track_id", "trackid"):
        df = df.iloc[:, 1:]
    idc = df.columns[0]
    id_series = df.iloc[:, 0].astype(str).str.strip().rename("id")
    df = df.iloc[:, 1:]
    df.insert(0, "id", id_series)

    obj_cols = [c for c in df.columns[1:] if _is_text(df[c])]
    expanded_parts = []
    cols_to_drop = []
    if len(obj_cols) <= 2000:
        for c in obj_cols:
            col = df[c]
            nn = col.dropna()
            sample = nn.head(200).astype(str).str.strip()
            if len(sample) and sample.str.startswith(_LIST_START).mean() > 0.9:
                if expand_lists:
                    ex = _parse_list_column(col, str(c))
                    if ex is None:
                        log.warning("%s: list-valued column %r skipped (variable length / unparsable)", source, c)
                        cols_to_drop.append(c)
                    else:
                        expanded_parts.append(ex)
                        cols_to_drop.append(c)
                continue
            conv = pd.to_numeric(col, errors="coerce")
            if len(nn) and conv.notna().sum() >= 0.9 * len(nn):
                df[c] = conv

    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    if float32:
        f_cols = [c for c in df.columns[1:] if pd.api.types.is_float_dtype(df[c].dtype)]
        if f_cols:
            df[f_cols] = df[f_cols].astype(np.float32)
        b_cols = [c for c in df.columns[1:] if pd.api.types.is_bool_dtype(df[c].dtype)]
        if b_cols:
            df[b_cols] = df[b_cols].astype(np.float32)

    if expanded_parts:
        df = pd.concat([df] + expanded_parts, axis=1)

    df.attrs["id_name"] = str(idc)
    return df


def read_table(path: Path, columns: Sequence[str] | None = None, nrows: int | None = None,
               expand_lists: bool = True, float32: bool = True) -> pd.DataFrame:
    """Read one Onion table.  Returns a frame whose first column is ``id`` (str); the others numeric (float32) or, for
    text columns, object.  ``columns`` restricts the (non-id) columns parsed.  UNVERIFIED on real files - tolerant by design."""
    p = Path(path)
    lay = sniff_layout(p)
    if lay.kind == "parquet":
        if columns is None:
            df = pd.read_parquet(p)
        else:
            import pyarrow.parquet as pq
            names = pq.read_schema(p).names
            df = pd.read_parquet(p, columns=[names[0]] + [c for c in columns if c in names and c != names[0]])
        if not isinstance(df.index, pd.RangeIndex):
            df = df.reset_index()
        if nrows:
            df = df.head(nrows)
    elif lay.kind == "json":
        try:
            df = pd.read_json(p, lines=p.suffix.lower() in (".jsonl", ".ndjson"))
        except ValueError:
            df = pd.read_json(p, orient="index").reset_index()
        if columns is not None:
            df = df[[df.columns[0]] + [c for c in columns if c in df.columns]]
    else:
        names = lay.names
        usecols = None
        if columns is not None:
            want = set(columns)
            usecols = [names[0]] + [c for c in names[1:] if c in want]
        id_dtype = {names[0]: str} if lay.header and names else ({0: str} if not lay.header else None)
        df = pd.read_csv(p, sep=lay.sep, header=0 if lay.header else None, names=names if not lay.header else None,
                         dtype=id_dtype,
                         usecols=usecols, nrows=nrows, compression="infer", on_bad_lines="warn", low_memory=False)
    df = _finalize(df, expand_lists, float32, p.name)
    df = df.rename(columns={df.columns[0]: "id"})
    dup = int(df["id"].duplicated().sum())
    if dup:
        log.warning("%s: %d duplicate ids (keeping first)", p.name, dup)
        df = df.drop_duplicates("id", keep="first")
    return df.reset_index(drop=True)


def peek_columns(path: Path) -> list[str]:
    """Column names (without the id column) from the first rows only; list-valued cells are NOT expanded (cheap)."""
    lay = sniff_layout(path)
    if lay.kind == "delimited":
        return list(lay.names[1:])
    return [c for c in read_table(path, nrows=5).columns[1:]]


def _norm_col(c) -> str:
    return re.sub(r"[^a-z0-9*?]+", ".", str(c).lower()).strip(".")


def match_cols(cols: Sequence[str], patterns: Sequence[str], exclude: Sequence[str] = ()) -> list[str]:
    """Columns whose dot-normalised name matches any fnmatch pattern (also as a suffix, so 'rhythm.*' finds
    'essentia.rhythm.bpm'); order preserved."""
    pats = [_norm_col(p) for p in patterns]
    exc = [_norm_col(p) for p in exclude]

    def hit(n, ps):
        return any(fnmatch.fnmatchcase(n, p) or fnmatch.fnmatchcase(n, "*." + p) for p in ps)
    return [c for c in cols if hit(_norm_col(c), pats) and not (exc and hit(_norm_col(c), exc))]


def _natural_key(s: str):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def _is_text(s: pd.Series) -> bool:
    """object or (pandas >= 3) string dtype."""
    return s.dtype == object or pd.api.types.is_string_dtype(s.dtype)


def _numeric_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns[1:] if pd.api.types.is_numeric_dtype(df[c])]


# ----------------------------------------------------------------------------------------------------------------------
# events (listening records)
# ----------------------------------------------------------------------------------------------------------------------
_UNIT_DIV = {"s": 1, "ms": 10 ** 3, "us": 10 ** 6, "ns": 10 ** 9}


def detect_timestamp_unit(ts: np.ndarray) -> str:
    """'s' | 'ms' | 'us' | 'ns' from the magnitude of the median positive timestamp (Unix epoch assumed).
    < 1e11 -> s (year 5138), < 1e14 -> ms, < 1e17 -> us, else ns.  Empty input -> 's'."""
    a = np.asarray(ts, dtype=np.float64)
    a = a[np.isfinite(a) & (a > 0)]
    if a.size == 0:
        return "s"
    m = float(np.median(a))
    return "s" if m < 1e11 else "ms" if m < 1e14 else "us" if m < 1e17 else "ns"


def to_seconds(ts: np.ndarray, unit: str) -> np.ndarray:
    """Integer seconds from timestamps in ``unit`` (floor division)."""
    return np.asarray(ts, dtype=np.int64) // _UNIT_DIV[unit]


class _KeyMap:
    """Incremental string -> dense int32 code map (first-appearance order)."""

    def __init__(self) -> None:
        self.map: dict[str, int] = {}

    def encode(self, uniques: Sequence[str], inverse: np.ndarray) -> np.ndarray:
        m = self.map
        codes = np.fromiter((m.setdefault(u, len(m)) for u in uniques), dtype=np.int32, count=len(uniques))
        return codes[inverse]

    def keys(self) -> np.ndarray:
        return np.array(list(self.map), dtype=object)


@dataclass
class EventTable:
    user: np.ndarray                       # int32 dense user codes (first-appearance order)
    track: np.ndarray                      # int32 codes: positions in ``track_index`` (or first-appearance if None given)
    ts: np.ndarray | None                  # int64 seconds (None for count files)
    count: np.ndarray | None               # int64 play counts (count files only)
    n_users: int
    unit: str | None
    track_keys: np.ndarray | None = None   # original track ids when no ``track_index`` was given
    stats: dict = field(default_factory=dict)


def _event_layout(path: Path) -> tuple[TableLayout, dict[str, int | None]]:
    stem_tokens = [t for t in re.split(r"[_\-\s]+", _strip_ext(path.name)) if t]
    default = stem_tokens if 2 <= len(stem_tokens) <= 8 else None
    lay = sniff_layout(path, default_names=default)
    if lay.kind != "delimited":
        raise ValueError(f"{path.name}: events must be a delimited text table")
    names = [re.sub(r"[^a-z0-9]", "", n.lower()) for n in lay.names]
    idx: dict[str, int | None] = {"user": None, "track": None, "ts": None, "count": None}
    for i, n in enumerate(names):
        if idx["user"] is None and ("user" in n or n in ("uid", "u")):
            idx["user"] = i
        elif idx["track"] is None and any(k in n for k in ("track", "song", "item", "tid")):
            idx["track"] = i
        elif idx["ts"] is None and any(k in n for k in ("time", "date", "epoch", "ts")):
            idx["ts"] = i
        elif idx["count"] is None and any(k in n for k in ("count", "plays", "playcount", "listen", "freq")):
            idx["count"] = i
    if idx["user"] is None or idx["track"] is None:               # positional fallback: user, track, third
        idx["user"], idx["track"] = 0, 1
        third = 2 if len(names) > 2 else None
        if third is not None and idx["ts"] is None and idx["count"] is None:
            idx["count" if "count" in _strip_ext(path.name).lower() else "ts"] = third
    return lay, idx


def _ts_from_strings(vals: np.ndarray) -> tuple[np.ndarray, str | None]:
    """Parse a string column to int64 (epoch units) or, for date strings, to seconds (unit 's')."""
    s = pd.Series(vals)
    num = pd.to_numeric(s, errors="coerce")
    if num.notna().mean() > 0.9:
        return num.fillna(-1).to_numpy(np.int64), None
    dt = pd.to_datetime(s, errors="coerce", utc=True)
    sec = dt.fillna(pd.Timestamp(0, tz="UTC")).dt.as_unit("s").astype("int64")      # resolution differs across pandas versions
    return sec.where(dt.notna(), -1).to_numpy(np.int64), "s"


def _arrow_ok(path: Path) -> bool:
    return path.suffix.lower() in (".bz2", ".gz", ".tsv", ".csv", ".txt", ".tab")


def _iter_chunks_pyarrow(path: Path, lay: TableLayout, idx: dict, chunk_rows: int) -> Iterator[dict[str, np.ndarray]]:
    import pyarrow as pa
    import pyarrow.csv as pacsv
    import pyarrow.compute as pc
    chosen = [i for i in idx.values() if i is not None]
    cnames = [lay.names[i] for i in chosen]
    ro = pacsv.ReadOptions(block_size=int(min(1 << 29, max(1 << 22, chunk_rows * 40))), column_names=lay.names,
                           skip_rows=1 if lay.header else 0)
    po = pacsv.ParseOptions(delimiter=lay.sep)
    co = pacsv.ConvertOptions(column_types={n: pa.string() for n in cnames}, include_columns=cnames)
    stream = pa.input_stream(str(path), compression="detect")
    reader = pacsv.open_csv(stream, read_options=ro, parse_options=po, convert_options=co)
    try:
        for batch in reader:
            out = {"user": batch.column(lay.names[idx["user"]]), "track": batch.column(lay.names[idx["track"]])}
            for k in ("ts", "count"):
                if idx[k] is not None:
                    out[k] = batch.column(lay.names[idx[k]])
            yield out
    finally:
        with contextlib.suppress(Exception):
            reader.close()
        with contextlib.suppress(Exception):
            stream.close()


def _iter_chunks_pandas(path: Path, lay: TableLayout, idx: dict, chunk_rows: int) -> Iterator[dict[str, np.ndarray]]:
    chosen = {k: lay.names[i] for k, i in idx.items() if i is not None}
    with pd.read_csv(path, sep=lay.sep, header=0 if lay.header else None, names=lay.names, usecols=list(chosen.values()),
                     dtype=str, chunksize=chunk_rows, compression="infer", on_bad_lines="warn") as rd:
        for ch in rd:
            yield {k: ch[v].to_numpy(object) for k, v in chosen.items()}


def load_events(paths: Path | Sequence[Path], track_index: pd.Index | Sequence[str] | None = None,
                chunk_rows: int = 5_000_000, limit_rows: int | None = None, engine: str | None = None,
                unit: str | None = None) -> EventTable:
    """Stream listening events into int arrays.

    ``track_index``: known track ids (positions become the track codes; events of other tracks are dropped).  ``None`` builds
    the track vocabulary from the data (``EventTable.track_keys``).  ``engine``: 'pyarrow' (default when the file's
    compression is Arrow-readable) or 'pandas'.  ``unit`` overrides timestamp-unit detection (first chunk sample).
    Users are coded 0.. in order of first appearance; original user ids are not retained.  UNVERIFIED: the real column
    order/header of ``userid_trackid_timestamp``; layout is sniffed and column roles are chosen by name, then position.
    """
    plist = [Path(paths)] if isinstance(paths, (str, Path)) else [Path(x) for x in paths]
    tindex = None if track_index is None else pd.Index(np.asarray(track_index, dtype=object).astype(str))
    ukeys, tkeys = _KeyMap(), (_KeyMap() if tindex is None else None)
    U, T, TS, CN = [], [], [], []
    n_read = n_bad = 0
    ts_unit = unit
    date_parsed = False
    vocab = None
    for path in plist:
        lay, idx = _event_layout(path)
        eng = engine or ("pyarrow" if _arrow_ok(path) else "pandas")
        if eng == "pyarrow":
            try:
                import pyarrow  # noqa: F401
            except ImportError:
                eng = "pandas"
        log.info("events: %s (sep=%r header=%s cols=%s engine=%s)", path.name, lay.sep, lay.header,
                 {k: lay.names[i] for k, i in idx.items() if i is not None}, eng)
        it = (_iter_chunks_pyarrow if eng == "pyarrow" else _iter_chunks_pandas)(path, lay, idx, chunk_rows)
        for chunk in it:
            if eng == "pyarrow":
                import pyarrow as pa
                import pyarrow.compute as pc
                n = len(chunk["user"])
                valid = pc.is_valid(chunk["user"]).to_numpy(zero_copy_only=False)
                if tindex is not None:
                    if vocab is None:
                        vocab = pa.array(tindex.to_numpy(object), type=pa.string())
                    tc = pc.fill_null(pc.index_in(chunk["track"], value_set=vocab), -1).to_numpy(zero_copy_only=False).astype(np.int32)
                    keepmask = valid & (tc >= 0)
                    tcodes = tc[keepmask]
                else:
                    keepmask = valid & pc.is_valid(chunk["track"]).to_numpy(zero_copy_only=False)
                    tcol = chunk["track"] if keepmask.all() else chunk["track"].filter(pa.array(keepmask))
                    d = tcol.dictionary_encode()
                    tcodes = tkeys.encode(d.dictionary.to_pylist(), d.indices.to_numpy(zero_copy_only=False).astype(np.int64))
                mask = None if keepmask.all() else pa.array(keepmask)
                ucol = chunk["user"] if mask is None else chunk["user"].filter(mask)
                du = ucol.dictionary_encode()
                ucodes = ukeys.encode(du.dictionary.to_pylist(), du.indices.to_numpy(zero_copy_only=False).astype(np.int64))
                extra = {}
                for k in ("ts", "count"):
                    if k in chunk:
                        col = chunk[k] if mask is None else chunk[k].filter(mask)
                        extra[k] = col.to_numpy(zero_copy_only=False).astype(object)
            else:
                n = len(chunk["user"])
                tr = chunk["track"].astype(str)
                if tindex is not None:
                    tc = tindex.get_indexer(tr).astype(np.int32)
                    keep = np.flatnonzero(tc >= 0)
                    tcodes = tc[keep]
                else:
                    keep = np.arange(n)
                    inv, uq = pd.factorize(pd.Series(tr))
                    tcodes = tkeys.encode(list(uq), inv)
                us = chunk["user"][keep].astype(str)
                inv, uq = pd.factorize(pd.Series(us))
                ucodes = ukeys.encode(list(uq), inv)
                extra = {k: chunk[k][keep] for k in ("ts", "count") if k in chunk}
            n_read += n
            U.append(ucodes)
            T.append(tcodes)
            if "ts" in extra:
                ts, parsed_unit = _ts_from_strings(extra["ts"])
                if parsed_unit:
                    date_parsed, ts_unit = True, "s"
                TS.append(ts)
            if "count" in extra:
                CN.append(pd.to_numeric(pd.Series(extra["count"]), errors="coerce").fillna(1).to_numpy(np.int64))
            if limit_rows and n_read >= limit_rows:
                break
        if limit_rows and n_read >= limit_rows:
            break
    cat = lambda xs, dt: np.concatenate(xs) if xs else np.zeros(0, dt)      # noqa: E731
    user, track = cat(U, np.int32), cat(T, np.int32)
    ts = cat(TS, np.int64) if TS else None
    count = cat(CN, np.int64) if CN else None
    if limit_rows and user.size > limit_rows:
        user, track = user[:limit_rows], track[:limit_rows]
        ts = ts[:limit_rows] if ts is not None else None
        count = count[:limit_rows] if count is not None else None
    if ts is not None:
        bad = ts < 0
        n_bad = int(bad.sum())
        if n_bad:
            log.warning("events: %d rows with unparsable timestamps dropped", n_bad)
            ok = ~bad
            user, track, ts = user[ok], track[ok], ts[ok]
            count = count[ok] if count is not None else None
        if not date_parsed:
            ts_unit = unit or detect_timestamp_unit(ts[: min(ts.size, 2_000_000)])
            log.info("events: timestamp unit detected as %r (median %.3g)", ts_unit, float(np.median(ts)) if ts.size else 0.0)
            ts = to_seconds(ts, ts_unit)
    stats = {"rows_read": int(n_read), "rows_kept": int(user.size), "rows_bad_ts": n_bad}
    log.info("events: read %d rows, kept %d (%.1f%%), %d users", n_read, user.size, 100.0 * user.size / max(n_read, 1), len(ukeys.map))
    return EventTable(user, track, ts, count, len(ukeys.map), ts_unit if ts is not None else None,
                      None if tkeys is None else tkeys.keys(), stats)


def find_event_files(files: FileIndex) -> tuple[list[Path], str]:
    """(paths, kind) with kind 'timestamp' | 'count'; timestamped logs are preferred."""
    for stem in ("userid_trackid_timestamp*",):
        f = files.find_files(stem)
        if f:
            return f, "timestamp"
    cands = [p for p in files.files if "user" in norm_stem(p.name) and "track" in norm_stem(p.name)]
    ts = [p for p in cands if "time" in norm_stem(p.name)]
    if ts:
        return sorted(ts), "timestamp"
    f = files.find_files("userid_trackid_count*")
    if f:
        return f, "count"
    if cands:
        return sorted(cands), "count"
    return [], "none"


# ----------------------------------------------------------------------------------------------------------------------
# sessions
# ----------------------------------------------------------------------------------------------------------------------
def _sessionize_sharded(user, track, ts, n_tracks, gap_minutes, min_len, max_len, max_shard_events):
    """melomane.sessions.sessionize per user-shard (sessions never cross users, so this is exact) to bound memory."""
    n = user.size
    if n == 0:
        return sp.csr_matrix((0, n_tracks), dtype=np.float32), np.zeros(0, np.int32), np.zeros(0, np.int64)
    n_shards = max(1, math.ceil(n / max_shard_events))
    if n_shards == 1:
        return sessionize(user, track, ts, gap_minutes, min_len, max_len, n_tracks, return_start=True)
    log.info("sessionising %d events in %d user shards", n, n_shards)
    shard = user % n_shards
    Xs, Us, T0 = [], [], []
    for k in range(n_shards):
        sel = np.flatnonzero(shard == k)
        if sel.size == 0:
            continue
        X, su, t0 = sessionize(user[sel], track[sel], ts[sel], gap_minutes, min_len, max_len, n_tracks, return_start=True)
        Xs.append(X)
        Us.append(su)
        T0.append(t0)
    return sp.vstack(Xs, format="csr", dtype=np.float32), np.concatenate(Us), np.concatenate(T0)


def _profile_documents(user, track, count, n_tracks, min_len, max_len):
    """Fallback for count-only logs: one document per user = its ``max_len`` most-played distinct tracks."""
    key = user.astype(np.int64) * n_tracks + track
    uk, inv = np.unique(key, return_inverse=True)
    w = np.bincount(inv, weights=(count if count is not None else np.ones(user.size)))
    u, t = uk // n_tracks, uk % n_tracks
    order = np.lexsort((-w, u))
    u, t = u[order], t[order]
    starts = np.r_[0, np.flatnonzero(u[1:] != u[:-1]) + 1] if u.size else np.zeros(0, np.int64)
    sizes = np.diff(np.r_[starts, u.size])
    rank = np.arange(u.size) - np.repeat(starts, sizes)
    keep = rank < max_len
    u, t = u[keep], t[keep]
    uu, ui = np.unique(u, return_inverse=True)
    X = sp.csr_matrix((np.ones(u.size, np.float32), (ui, t)), shape=(uu.size, n_tracks))
    ok = np.diff(X.indptr) >= min_len
    return X[np.flatnonzero(ok)].tocsr(), uu[ok].astype(np.int32)


# ----------------------------------------------------------------------------------------------------------------------
# feature tables -> blocks
# ----------------------------------------------------------------------------------------------------------------------
class _Store:
    """Loads each table once (full, float32, ids stripped/deduped) and caches it until ``clear``."""

    def __init__(self, files: FileIndex):
        self.files = files
        self.cache: dict[Path, pd.DataFrame] = {}

    def get(self, path: Path) -> pd.DataFrame | None:
        if path not in self.cache:
            try:
                self.cache[path] = read_table(path)
                df = self.cache[path]
                log.info("loaded %s: %d rows x %d cols", path.name, len(df), df.shape[1] - 1)
            except Exception as e:                            # tolerant: a broken file must not sink the build
                log.error("could not read %s (%s: %s) - treated as missing", path.name, type(e).__name__, e)
                return None
        return self.cache[path]

    def evict(self, path: Path) -> None:
        self.cache.pop(path, None)


def _norm_entry(e) -> dict:
    return {"file": e} if isinstance(e, str) else dict(e)


def is_forbidden_stem(stem: str) -> bool:
    ns = norm_stem(stem)
    return any(fnmatch.fnmatchcase(ns, norm_stem(f)) for f in FORBIDDEN_FAMILY_STEMS)


def resolve_families(families: dict | None) -> dict[str, dict]:
    """Defaults overlaid with the caller's mapping (name -> {'sources': [...]} | [sources]; None removes a family)."""
    out = {k: dict(v) for k, v in DEFAULT_FAMILIES.items()}
    for k, v in (families or {}).items():
        if v is None:
            out.pop(k, None)
        else:
            out[k] = {"sources": list(v)} if isinstance(v, (list, tuple)) else dict(v)
    return out


def _source_paths(files: FileIndex, fam_cfg: dict) -> list[Path]:
    out: list[Path] = []
    for cfg in fam_cfg.values():
        for e in cfg.get("sources", []):
            stem = _norm_entry(e)["file"]
            if is_forbidden_stem(stem):
                continue
            for p in files.find_files(stem):
                if p not in out:
                    out.append(p)
    return out


def _reduce(X: np.ndarray, method: str, n_components: int, seed: int) -> tuple[np.ndarray, list[str]]:
    n, d = X.shape
    k = int(min(n_components, d - 1, n - 1))
    if k < 1:
        return X, [f"c{i}" for i in range(d)]
    method = method.lower()
    if method == "nmf":
        from sklearn.decomposition import NMF
        Z = NMF(n_components=k, init="nndsvda", random_state=seed, max_iter=300).fit_transform(sp.csr_matrix(np.clip(X, 0, None)))
    elif method == "pca":
        from sklearn.decomposition import PCA
        Z = PCA(n_components=k, svd_solver="randomized", random_state=seed).fit_transform(X)
    else:
        from sklearn.decomposition import TruncatedSVD
        Z = TruncatedSVD(n_components=k, algorithm="randomized", random_state=seed).fit_transform(sp.csr_matrix(X))
    return Z.astype(np.float32), [f"{method}{i}" for i in range(Z.shape[1])]


@dataclass
class _Part:
    pos: np.ndarray                 # positions in the final track index
    X: np.ndarray                   # float32 (len(pos), d)
    cols: list[str]
    stem: str
    lyric: bool = False


def _clean_matrix(M: np.ndarray) -> np.ndarray:
    M = np.asarray(M, dtype=np.float32)
    M[~np.isfinite(M)] = np.nan
    return M


def _select_part(df: pd.DataFrame, ent: dict, index: pd.Index, stem_label: str, lyric: bool, seed: int,
                 fam_reduce: dict | None) -> _Part | None:
    num = _numeric_cols(df)
    pats = ent.get("columns")
    excl = ent.get("exclude", ())
    if pats:
        cols = match_cols(num, pats, excl)
    else:
        bad = set(match_cols(num, excl)) if excl else set()
        cols = [c for c in num if c not in bad]
    guard = [c for c in cols if _VOICE_GUARD.search(_norm_col(c))]
    if guard:
        log.warning("%s: %d voice-like (held-out) columns stripped from family input, e.g. %s", stem_label, len(guard), guard[:3])
        cols = [c for c in cols if c not in set(guard)]
    if not cols:
        return None
    pos = index.get_indexer(df["id"])
    sel = np.flatnonzero(pos >= 0)
    if sel.size == 0:
        return None
    M = _clean_matrix(df[cols].to_numpy(np.float32, na_value=np.nan)[sel])
    p = pos[sel]
    if lyric:                                                # instrumentals / missing lyrics: all-NaN or all-zero rows
        alive = ~(np.isnan(M).all(axis=1) | (np.nan_to_num(np.abs(M)).sum(axis=1) == 0))
        M, p = M[alive], p[alive]
    else:
        alive = ~np.isnan(M).all(axis=1)
        M, p = M[alive], p[alive]
    if p.size == 0:
        return None
    names = [f"{stem_label}:{c}" for c in cols]
    red = ent.get("reduce") or None
    if red:
        Z, rn = _reduce(np.nan_to_num(M), red.get("method", "svd"), int(red.get("n_components", 64)), seed)
        M, names = Z, [f"{stem_label}:{r}" for r in rn]
    return _Part(p, M, names, stem_label, lyric)


def _combine_parts(parts: list[_Part], reduce_cfg: dict | None, seed: int, label: str) -> tuple[np.ndarray, np.ndarray, list[str]] | None:
    """Outer-join parts on track position; NaN cells -> column median; drop all-NaN / constant columns."""
    if not parts:
        return None
    union = np.unique(np.concatenate([p.pos for p in parts]))
    D = sum(p.X.shape[1] for p in parts)
    M = np.full((union.size, D), np.nan, dtype=np.float32)
    names: list[str] = []
    c0 = 0
    for p in parts:
        r = np.searchsorted(union, p.pos)
        M[r, c0:c0 + p.X.shape[1]] = p.X
        names += p.cols
        c0 += p.X.shape[1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(M, axis=0)
        sd = np.nanstd(M, axis=0)
    keep = np.isfinite(med) & (sd > 0)
    if not keep.all():
        log.info("%s: dropped %d all-NaN/constant columns", label, int((~keep).sum()))
    M, med, names = M[:, keep], med[keep], [n for n, k in zip(names, keep) if k]
    if M.shape[1] == 0:
        return None
    nan_share = float(np.isnan(M).mean())
    if nan_share:
        log.info("%s: imputing %.2f%% missing cells with column medians", label, 100 * nan_share)
        M = np.where(np.isnan(M), med[None, :], M)
    if reduce_cfg:
        M, rn = _reduce(M, reduce_cfg.get("method", "pca"), int(reduce_cfg.get("n_components", 64)), seed)
        names = [f"{label}:{r}" for r in rn]
    return union, np.ascontiguousarray(M, dtype=np.float32), names


def check_family_leakage(families: dict[str, FamilyBlock]) -> list[str]:
    """Violations of the never-in-compounds rule (forbidden source stems or voice-like columns); [] if clean."""
    bad = []
    for name, b in families.items():
        for stem in filter(None, (s.strip() for s in b.source.split(","))):
            if is_forbidden_stem(stem):
                bad.append(f"{name}: forbidden source {stem}")
        for c in b.columns:
            if _VOICE_GUARD.search(_norm_col(c.split(":", 1)[-1])):
                bad.append(f"{name}: voice-like column {c}")
    return bad


def _stem_label(path: Path) -> str:
    return re.sub(r"[^a-z0-9]+", "_", _strip_ext(path.name)).strip("_")


def _build_families(files: FileIndex, store: _Store, fam_cfg: dict, index: pd.Index, seed: int, lyric_ids: set,
                    report: dict) -> dict[str, FamilyBlock]:
    fams: dict[str, FamilyBlock] = {}
    for fam, cfg in fam_cfg.items():
        parts, used, missing, refused = [], [], [], []
        for raw_e in cfg.get("sources", []):
            ent = _norm_entry(raw_e)
            stem = ent["file"]
            if is_forbidden_stem(stem):
                log.error("family %s: source %r is held-out / learned-embedding data and is REFUSED", fam, stem)
                refused.append(stem)
                continue
            paths = files.find_files(stem)
            if not paths:
                missing.append(stem)
                continue
            for path in paths:
                df = store.get(path)
                is_lyric = bool(cfg.get("lyric")) or any(fnmatch.fnmatchcase(norm_stem(path.name), norm_stem(s)) for s in LYRIC_STEMS)
                part = None if df is None else _select_part(df, ent, index, _stem_label(path), is_lyric, seed, ent.get("reduce"))
                if part is None:
                    missing.append(stem)
                    continue
                parts.append(part)
                used.append({"file": path.name, "n_cols": len(part.cols), "n_rows": int(part.pos.size),
                             "columns_head": [c.split(":", 1)[1] for c in part.cols[:6]]})
                if part.lyric:
                    lyric_ids.update(part.pos.tolist())
        comb = _combine_parts(parts, cfg.get("reduce"), seed, fam)
        if comb is None:
            log.warning("family %s OMITTED (missing/empty sources: %s%s)", fam, missing, f"; refused: {refused}" if refused else "")
            report[fam] = {"status": "omitted", "missing": missing, "refused": refused}
            continue
        rows, X, cols = comb
        fams[fam] = FamilyBlock(rows.astype(np.int64), X, cols, ",".join(p.stem for p in parts))
        log.info("family %s: %d tracks x %d dims from %s%s", fam, rows.size, X.shape[1], [u["file"] for u in used],
                 f" (missing: {missing})" if missing else "")
        report[fam] = {"status": "ok", "n_rows": int(rows.size), "n_cols": int(X.shape[1]), "sources": used, "missing": missing,
                       "refused": refused}
    return fams


def _build_external(files: FileIndex, store: _Store, patterns: Sequence[str], index: pd.Index, lyric_ids: set,
                    report: dict) -> dict[str, FamilyBlock]:
    out: dict[str, FamilyBlock] = {}
    for pat in patterns:
        for path in files.find_files(pat):
            df = store.get(path)
            if df is None:
                continue
            cols = _numeric_cols(df)
            pos = index.get_indexer(df["id"])
            sel = np.flatnonzero(pos >= 0)
            if not cols or sel.size == 0:
                store.evict(path)
                continue
            M = _clean_matrix(df[cols].to_numpy(np.float32, na_value=np.nan)[sel])
            ok = np.isfinite(M).all(axis=1)
            if any(fnmatch.fnmatchcase(norm_stem(path.name), norm_stem(x)) for x in LYRIC_STEMS):
                alive = ok & (np.abs(M).sum(axis=1) > 0)
                lyric_ids.update(pos[sel][alive].tolist())
            store.evict(path)
            if not ok.any():
                continue
            r = pos[sel][ok]
            order = np.argsort(r, kind="stable")
            key = re.sub(r"^id_", "", _stem_label(path))
            out[key] = FamilyBlock(r[order].astype(np.int64), np.ascontiguousarray(M[ok][order]), [str(c) for c in cols], path.name)
            report[key] = {"file": path.name, "n_rows": int(ok.sum()), "n_cols": len(cols)}
            log.info("external %s: %d tracks x %d dims", key, int(ok.sum()), len(cols))
    return out


def _pick_col(df: pd.DataFrame, patterns: Sequence[str]) -> str | None:
    num = _numeric_cols(df)
    hits = match_cols(num, patterns)
    return sorted(hits, key=_natural_key)[0] if hits else None


def _build_measured(files: FileIndex, store: _Store, spec: dict[str, list[str]], index: pd.Index, report: dict) -> pd.DataFrame | None:
    frames = []
    for stem in MEASURED_SOURCES:
        for path in files.find_files(stem):
            df = store.get(path)
            if df is not None:
                frames.append((path, df))
    out: dict[str, np.ndarray] = {}
    used: dict[str, str] = {}

    def put(name, path, df, col):
        pos = index.get_indexer(df["id"])
        v = np.full(len(index), np.nan, dtype=np.float32)
        ok = pos >= 0
        v[pos[ok]] = df[col].to_numpy(np.float32, na_value=np.nan)[ok]
        if np.isfinite(v).any():
            out[name] = v
            used[name] = f"{path.name}:{col}"

    for name, pats in spec.items():
        for path, df in frames:
            c = _pick_col(df, pats)
            if c is not None:
                put(name, path, df, c)
                break
    for path, df in frames:                                   # MFCC-1/2 means (column naming differs between sources)
        cand = [c for c in _numeric_cols(df) if "mfcc" in _norm_col(c) and re.search(r"(^|\.)(mean|avg|average)", _norm_col(c))
                and not any(x in _norm_col(c) for x in _MFCC_EXCLUDE)]
        cand = sorted(cand, key=_natural_key)
        for i, nm in enumerate(("mfcc1_mean", "mfcc2_mean")):
            if nm not in out and len(cand) > i:
                put(nm, path, df, cand[i])
    missing = [k for k in list(spec) + ["mfcc1_mean", "mfcc2_mean"] if k not in out]
    report.update({"used": used, "missing": missing})
    log.info("measured scalars: %d found (%s); missing: %s", len(out), list(out), missing)
    return pd.DataFrame(out, index=range(len(index)), dtype=np.float32) if out else None


def _scalarize(df: pd.DataFrame, prefix: str, max_cols: int = 24) -> dict[str, np.ndarray | pd.Series]:
    """Keep up to ``max_cols`` numeric columns, preferring mean-like statistics (documented heuristic)."""
    num = _numeric_cols(df)
    pref = [c for c in num if re.search(r"(mean|avg|average|median)", _norm_col(c))]
    use = (pref or num)[:max_cols]
    return {f"{prefix}_{re.sub(r'[^A-Za-z0-9]+', '_', str(c)).strip('_')}": df[c] for c in use}


def _build_heldout(files: FileIndex, store: _Store, index: pd.Index, report: dict) -> pd.DataFrame | None:
    cols: dict[str, np.ndarray] = {}

    def put(name: str, ids: pd.Series, values: np.ndarray) -> None:
        pos = index.get_indexer(ids)
        ok = pos >= 0
        v = np.full(len(index), np.nan, dtype=np.float32)
        v[pos[ok]] = np.asarray(values, dtype=np.float32)[ok]
        if np.isfinite(v).any():
            cols[name] = v
    # 1. openSMILE / emobase voice descriptors, one aggregated scalar per category
    for stem in VOICE_STEMS:
        for path in files.find_files(stem):
            tag = "emobase" if "emobase" in norm_stem(path.name) else "voice"
            names = peek_columns(path) if sniff_layout(path).kind == "delimited" else []
            hits = {cat: [c for c in names if any(k in _norm_col(c) for k in keys)] for cat, keys in _VOICE_CATS.items()}
            need = sorted({c for v in hits.values() for c in v})
            if not need:
                continue
            df = store.get(path) if path in store.cache else None
            if df is None:
                try:
                    df = read_table(path, columns=need)
                except Exception as e:
                    log.error("held-out voice: could not read %s (%s)", path.name, e)
                    continue
            for cat, cs in hits.items():
                cs = [c for c in cs if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
                if not cs:
                    continue
                pref = [c for c in cs if re.search(r"mean", _norm_col(c))]
                use = pref or cs
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    v = np.nanmean(df[use].to_numpy(np.float32, na_value=np.nan), axis=1)
                put(f"{tag}_{cat}" if f"{tag}_{cat}" not in cols else f"{tag}_{cat}_{_stem_label(path)}", df["id"], v)
    # 2. GEMS, sentiment functionals, VAD bag-of-words
    for key, stem in HELDOUT_STEMS.items():
        path = files.find_file(stem)
        if path is None:
            continue
        df = store.get(path)
        if df is None:
            continue
        if key == "vad":
            done = False
            for dim in ("valence", "arousal", "dominance"):
                cs = [c for c in _numeric_cols(df) if dim in _norm_col(c)]
                if len(cs) >= 2:
                    W = np.nan_to_num(df[sorted(cs, key=_natural_key)].to_numpy(np.float32, na_value=np.nan))
                    tot = W.sum(axis=1)
                    v = np.where(tot > 0, (W * np.arange(W.shape[1])[None, :]).sum(axis=1) / np.maximum(tot, 1e-12), np.nan)
                    put(f"vad_{dim}_bow_mean", df["id"], v)
                    done = True
            if done:
                continue
        for name, ser in _scalarize(df, key).items():
            put(name, df["id"], ser.to_numpy(np.float32, na_value=np.nan))
        store.evict(path)
    report["columns"] = list(cols)
    log.info("held-out scalars: %s", list(cols))
    return pd.DataFrame(cols, index=range(len(index)), dtype=np.float32) if cols else None


# ----------------------------------------------------------------------------------------------------------------------
# metadata / tags
# ----------------------------------------------------------------------------------------------------------------------
def _find_alias(df: pd.DataFrame, aliases: Sequence[str], exclude: Sequence[str] = ()) -> str | None:
    norm = {c: re.sub(r"[^a-z0-9]", "", str(c).lower()) for c in df.columns[1:]}
    for a in aliases:
        for c, n in norm.items():
            if n == a and c not in exclude:
                return c
    return None


def _as_text(s: pd.Series) -> pd.Series:
    return s.where(s.notna(), "").astype(str).str.strip().replace({"nan": "", "None": "", "NaN": ""})


def _parse_year(s: pd.Series) -> np.ndarray:
    if pd.api.types.is_numeric_dtype(s):
        v = s.to_numpy(np.float64, na_value=np.nan)
        return np.where((v >= 1000) & (v <= 2100), v, np.nan).astype(np.float32)
    y = s.astype(str).str.extract(r"((?:1[0-9]|20)\d{2})", expand=False)
    return pd.to_numeric(y, errors="coerce").to_numpy(np.float32)


def _tags_from_frame(df: pd.DataFrame, k: int = 5) -> pd.Series | None:
    """'a|b|c' per row from a text column of tags/genres, or the top-k positive columns of a wide tag matrix."""
    tagcol = _find_alias(df, ("tags", "toptags", "genres", "genre", "tag", "tagweight"))
    if tagcol is None and df.shape[1] == 2 and any(term in str(df.columns[1]).lower() for term in ("tag", "genre")):
        tagcol = df.columns[1]
    if tagcol is not None and _is_text(df[tagcol]):
        col_s = _as_text(df[tagcol])
        sample_str = str(col_s.iloc[0]) if len(col_s) else ""
        if sample_str.startswith("{") and ":" in sample_str:
            import ast
            def _parse_dict(val):
                if not isinstance(val, str) or not val.strip() or val[0] != "{":
                    return ""
                try:
                    d = ast.literal_eval(val)
                    if isinstance(d, dict):
                        return "|".join(str(k) for k, _ in sorted(d.items(), key=lambda x: x[1], reverse=True)[:k])
                except Exception:
                    pass
                return ""
            parsed = col_s.apply(_parse_dict)
            return pd.Series(parsed.to_numpy(), index=df["id"].to_numpy())
        t = col_s.str.replace(r"[\[\]'\"]", "", regex=True).str.replace(r"\s*[;,|]\s*", "|", regex=True)
        return pd.Series(t.str.lower().str.strip("| ").to_numpy(), index=df["id"].to_numpy())
    num = _numeric_cols(df)
    if len(num) < 2 or all(re.fullmatch(r"(c?\d+|\w+\[\d+\])", str(c)) for c in num):
        return None                                        # unnamed dimensions carry no tag names
    if len(num) > 2000:
        return None                                        # Skip overly wide sparse matrices (e.g. 178k raw TF-IDF)
    V = np.nan_to_num(df[num].to_numpy(np.float32, na_value=np.nan))
    kk = min(k, V.shape[1])
    top = np.argpartition(-V, kk - 1, axis=1)[:, :kk]
    tv = np.take_along_axis(V, top, axis=1)
    order = np.argsort(-tv, axis=1)
    top, tv = np.take_along_axis(top, order, axis=1), np.take_along_axis(tv, order, axis=1)
    names = np.array([re.sub(r"^(tag|genre)s?[_.:-]?", "", str(c), flags=re.I) for c in num], dtype=object)
    lab = np.where(tv > 0, names[top], "")
    joined = ["|".join(x for x in row if x) for row in lab]
    return pd.Series(joined, index=df["id"].to_numpy())


def _build_tracks(files: FileIndex, store: _Store, ids: np.ndarray, report: dict) -> tuple[pd.DataFrame, bool]:
    idx = pd.Index(ids)
    N = len(ids)
    fields: dict[str, pd.Series] = {}
    src: dict[str, str] = {}

    def offer(field_: str, ser: pd.Series, path: Path) -> None:
        ser = ser[~ser.index.duplicated()]
        new = ser.reindex(idx)
        if field_ in fields:
            old = fields[field_]
            empty = old.isna() | (old.astype(str) == "")
            fields[field_] = old.where(~empty, new)
        else:
            fields[field_] = new
            src[field_] = path.name
    for stem in METADATA_STEMS:
        for path in files.find_files(stem):
            df = store.get(path)
            if df is None:
                continue
            for f_ in ("artist", "title", "album", "language"):
                c = _find_alias(df, _META_ALIASES[f_], exclude=[_find_alias(df, _META_ALIASES["artist_id"]) or ""])
                if c is not None and _is_text(df[c]):
                    offer(f_, pd.Series(_as_text(df[c]).to_numpy(), index=df["id"].to_numpy()), path)
            c = _find_alias(df, _META_ALIASES["artist_id"])
            if c is not None:
                offer("artist_id_raw", pd.Series(_as_text(df[c]).to_numpy(), index=df["id"].to_numpy()), path)
            c = _find_alias(df, _META_ALIASES["year"])
            if c is not None:
                offer("year", pd.Series(_parse_year(df[c]), index=df["id"].to_numpy()), path)
            for f_ in ("popularity", "duration_ms"):
                c = _find_alias(df, _META_ALIASES[f_])
                if c is not None:
                    offer(f_, pd.Series(pd.to_numeric(df[c], errors="coerce").to_numpy(np.float32), index=df["id"].to_numpy()), path)
            tg = _tags_from_frame(df) if _find_alias(df, ("tags", "toptags", "genres", "genre")) else None
            if tg is not None:
                offer("tags_meta", tg, path)
    tag_parts = []
    for stem in TAG_STEMS:
        for path in files.find_files(stem):
            ns = norm_stem(path.name)
            if "tag" in ns and "tfidf" in ns:
                continue                                            # 178k columns; id_tags_dict provides exact tags cleanly
            df = store.get(path)
            tg = None if df is None else _tags_from_frame(df, k=4 if "tag" in ns else 3)
            if tg is not None:
                tag_parts.append(tg[~tg.index.duplicated()].reindex(idx).fillna(""))
                src["tags"] = (src.get("tags", "") + "," + path.name).strip(",")
            store.evict(path)
    if "tags_meta" in fields:
        tag_parts.insert(0, fields.pop("tags_meta").fillna(""))
    if tag_parts:
        tags = tag_parts[0].astype(str)
        for t in tag_parts[1:]:
            tags = tags + "|" + t.astype(str)
        tags = tags.str.strip("|").str.replace(r"\|{2,}", "|", regex=True)
        tags = tags.map(lambda s: "|".join(list(dict.fromkeys(x for x in s.split("|") if x))[:8]))
        fields["tags"] = tags
    tr = pd.DataFrame({"ext_id": ids.astype(str)})
    artist = fields["artist"].fillna("").astype(str).to_numpy() if "artist" in fields else np.full(N, "", dtype=object)
    title = fields["title"].fillna("").astype(str).to_numpy() if "title" in fields else np.full(N, "", dtype=object)
    have_artist, have_title = bool((artist != "").any()), bool((title != "").any())
    # artist_id: explicit id column > normalised artist name > unique negative ids
    aid = np.full(N, -1, dtype=np.int64)
    if "artist_id_raw" in fields:
        raw = fields["artist_id_raw"].fillna("").astype(str).to_numpy()
        codes, _ = pd.factorize(pd.Series(raw))
        aid = np.where(raw != "", codes, -1).astype(np.int64)
        src["artist_id"] = src.get("artist_id_raw", "")
    elif have_artist:
        keys = pd.Series(artist).map(normalize_artist).to_numpy()
        codes, _ = pd.factorize(pd.Series(keys))
        aid = np.where(keys != "", codes, -1).astype(np.int64)
        src["artist_id"] = "normalised artist name"
    miss = aid < 0
    aid[miss] = -(np.arange(int(miss.sum())) + 1)
    tr["artist_id"] = aid
    tr["artist"] = artist
    tr["title"] = title
    for f_, dflt in (("album", ""), ("language", "")):
        tr[f_] = fields[f_].fillna("").astype(str).to_numpy() if f_ in fields else dflt
    if "year" in fields:
        tr["year"] = fields["year"].to_numpy(np.float32, na_value=np.nan)
    for f_ in ("popularity", "duration_ms"):
        if f_ in fields:
            tr[f_] = fields[f_].to_numpy(np.float32, na_value=np.nan)
    tr["tags"] = fields["tags"].to_numpy() if "tags" in fields else ""
    report["metadata_sources"] = src
    blocker = not (have_artist and have_title)
    return tr, blocker


# ----------------------------------------------------------------------------------------------------------------------
# public builders
# ----------------------------------------------------------------------------------------------------------------------
def build_corpus(raw_dir: Path, gap_minutes: float = 30.0, min_len: int = 5, max_len: int = 200,
                 limit_tracks: int | None = None, families: dict | None = None, measured: dict | None = None,
                 external: Sequence[str] | None = None, seed: int = 0, name: str = "onion", limit_events: int | None = None,
                 events_engine: str | None = None, chunk_rows: int = 5_000_000, max_shard_events: int = 60_000_000,
                 require_events: bool = True) -> Corpus:
    """Build the Onion ``Corpus`` from a directory of downloaded files (see module docstring for tolerance / leakage rules).

    Raw ids 0..N-1 are assigned (sorted by string id) over tracks that have BOTH >= 1 listening event and >= 1 family/measured
    feature row.  Sessions = ``melomane.sessions.sessionize`` (gap ``gap_minutes``, distinct-track length in
    [min_len, max_len]); ``sessions_meta`` holds a dense per-session ``user_id`` code (original ids are not kept) and
    ``n_tracks``.  Count-only logs (no timestamps) degrade to per-user profile documents (flagged in ``meta``).
    ``families`` overlays ``DEFAULT_FAMILIES`` ({name: {'sources': [...], 'reduce': {...}} | list | None}); ``measured`` overlays
    ``DEFAULT_MEASURED``; ``external`` replaces ``DEFAULT_EXTERNAL`` (file-stem patterns).  ``limit_tracks`` draws a seeded
    random subset of the eligible tracks (smoke runs); ``limit_events`` stops reading after that many rows.
    ``require_events=False`` returns a features-only corpus (0 sessions) if no event file exists.
    """
    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        raise FileNotFoundError(f"raw directory not found: {raw_dir}")
    files = FileIndex(raw_dir)
    log.info("Onion raw dir %s: %d data files", raw_dir, len(files.files))
    fam_cfg = resolve_families(families)
    meas_spec = {**DEFAULT_MEASURED, **{k: list(v) for k, v in (measured or {}).items()}}
    ext_pats = list(DEFAULT_EXTERNAL if external is None else external)
    store = _Store(files)
    rep: dict = {"families": {}, "external": {}, "measured": {}, "heldout": {}}

    # 1. eligible feature ids = union over family + measured source tables
    src_paths = _source_paths(files, fam_cfg)
    for stem in MEASURED_SOURCES:
        for p in files.find_files(stem):
            if p not in src_paths:
                src_paths.append(p)
    id_sets = []
    for p in src_paths:
        df = store.get(p)
        if df is not None:
            id_sets.append(np.asarray(pd.unique(df["id"]), dtype=object))
    feat_ids = np.unique(np.concatenate(id_sets)) if id_sets else np.array([], dtype=object)
    if feat_ids.size == 0:
        log.warning("no family/measured feature tables found in %s - corpus will be behaviour-only", raw_dir)
    else:
        log.info("%d tracks have at least one family/measured feature row (from %d files)", feat_ids.size, len(id_sets))
    if limit_tracks and feat_ids.size > limit_tracks:
        rng = np.random.default_rng(seed)
        feat_ids = feat_ids[np.sort(rng.choice(feat_ids.size, limit_tracks, replace=False))]

    # 2. events -> per-track presence
    ev_paths, ev_kind = find_event_files(files)
    if not ev_paths:
        if require_events:
            raise FileNotFoundError(f"no listening-event file (userid_trackid_timestamp / userid_trackid_count) in {raw_dir}")
        log.warning("no event file: building a features-only corpus with 0 sessions")
    ev = None
    if ev_paths:
        ev = load_events(ev_paths, track_index=feat_ids if feat_ids.size else None, chunk_rows=chunk_rows,
                         limit_rows=limit_events, engine=events_engine)
    if ev is not None and feat_ids.size:
        has = np.bincount(ev.track, minlength=feat_ids.size) > 0
        ids = feat_ids[has]
        cmap = (np.cumsum(has) - 1).astype(np.int32)
        track_codes = cmap[ev.track]
    elif ev is not None:                                    # behaviour-only
        ids = ev.track_keys.astype(object)
        order = np.argsort(ids.astype(str), kind="stable")
        inv = np.empty(order.size, np.int32)
        inv[order] = np.arange(order.size, dtype=np.int32)
        ids, track_codes = ids[order], inv[ev.track]
    else:
        ids, track_codes = feat_ids, None
    N = int(ids.size)
    log.info("raw ids: %d tracks with both events and features (features-only %d, so %d had no events)", N, feat_ids.size,
             int(feat_ids.size - N))
    index = pd.Index(ids)

    # 3. sessions
    doc_type, sem = "listening session (unordered set)", "session"
    if ev is None or N == 0:
        X, su, t0 = sp.csr_matrix((0, N), dtype=np.float32), np.zeros(0, np.int32), None
        doc_type, sem = "none", "none"
    elif ev.ts is not None:
        X, su, t0 = _sessionize_sharded(ev.user, track_codes, ev.ts, N, gap_minutes, min_len, max_len, max_shard_events)
    else:
        log.warning("event file has no timestamps (%s): DEGRADED to per-user profile documents (top-%d tracks)", ev_paths[0].name, max_len)
        X, su = _profile_documents(ev.user, track_codes, ev.count, N, min_len, max_len)
        t0 = None
        doc_type, sem = "user profile (top tracks by play count)", "user_profile"
    lens = np.diff(X.indptr)
    log.info("documents: %d kept (%s); mean length %.1f", X.shape[0], sem, float(lens.mean()) if lens.size else 0.0)
    sessions_meta = pd.DataFrame({"user_id": su.astype(np.int32), "n_tracks": lens.astype(np.int32)})
    if t0 is not None and len(t0) == len(sessions_meta):
        sessions_meta["ts_start"] = t0.astype(np.int64)                    # session start (unix s): enables E6 temporal stability

    # 4. blocks, held-out, measured, external, tracks
    lyric_pos: set = set()
    fams = _build_families(files, store, fam_cfg, index, seed, lyric_pos, rep["families"])
    bad = check_family_leakage(fams)
    if bad:
        raise ValueError("held-out / learned-embedding data leaked into families: " + "; ".join(bad))
    meas = _build_measured(files, store, meas_spec, index, rep["measured"])
    held = _build_heldout(files, store, index, rep["heldout"])
    ext = _build_external(files, store, ext_pats, index, lyric_pos, rep["external"])
    tracks, blocker = _build_tracks(files, store, ids, rep)
    for p in files.find_files("processed_lyrics") + files.find_files("id_lyrics"):           # text never leaves this function
        df = store.get(p)
        if df is not None:
            txt = [c for c in df.columns[1:] if _is_text(df[c])]
            if txt:
                nonempty = _as_text(df[txt[0]]).str.len().to_numpy() > 0
                pos = index.get_indexer(df["id"][nonempty])
                lyric_pos.update(pos[pos >= 0].tolist())
    tracks["has_lyrics"] = np.isin(np.arange(N), np.fromiter(lyric_pos, dtype=np.int64, count=len(lyric_pos)))
    if not lyric_pos:
        log.warning("no lyric source found: has_lyrics is False for all tracks")
    if track_codes is not None:
        tracks["n_events"] = np.bincount(track_codes, minlength=N).astype(np.int32)
    else:
        tracks["n_events"] = np.zeros(N, np.int32)
    tracks["n_sessions"] = np.asarray(X.sum(axis=0)).ravel().astype(np.int32) if X.shape[0] else np.zeros(N, np.int32)
    store.cache.clear()

    missing_expected = [s for s in EXPECTED_STEMS if not files.find_files(s)]
    meta = {
        "source": "Music4All-Onion (Zenodo %s / %s)" % (ZENODO_RECORDS["v1"], ZENODO_RECORDS["v2"]), "license": LICENSE,
        "raw_dir": str(raw_dir), "document_type": doc_type, "session_semantics": sem,
        "gap_minutes": gap_minutes, "session_len_range": [min_len, max_len], "timestamp_unit": ev.unit if ev is not None else None,
        "event_files": [p.name for p in ev_paths], "n_events_read": ev.stats["rows_read"] if ev else 0,
        "n_events_kept": ev.stats["rows_kept"] if ev else 0, "n_users": ev.n_users if ev else 0,
        "n_sessions": int(X.shape[0]), "n_tracks_with_features": int(feat_ids.size), "n_tracks_final": N,
        "family_report": rep["families"], "external_report": rep["external"], "measured_report": rep["measured"],
        "heldout_report": rep["heldout"], "metadata_sources": rep.get("metadata_sources", {}),
        "missing_expected_files": missing_expected, "has_audio": bool(fams), "has_lyrics": bool(lyric_pos),
        "has_year": "year" in tracks.columns, "has_order": False, "blocker_a": bool(blocker),
        "n_tracks_with_lyrics": int(tracks["has_lyrics"].sum()), "seed": seed,
        "notes": ["Onion file formats were inferred by tolerant readers (UNVERIFIED against the real release); see INVENTORY.json."],
    }
    if blocker:
        meta["blocker_a_note"] = ("Blocker-A: the open Onion release provided no artist/title (files searched: %s). tracks.artist/title "
                                  "are empty, artist_id are unique negative ids (artist-disjoint splits are impossible), "
                                  "cross-corpus matching is impossible; use tag/genre-only labelling and ask the Music4All authors."
                                  % ", ".join(METADATA_STEMS))
        log.warning(meta["blocker_a_note"])
    if sem == "user_profile":
        meta["notes"].append("DEGRADED: no timestamps; documents are per-user top-track profiles, not listening sessions.")
    c = Corpus(name, tracks, X, fams, meas, held, ext, sessions_meta, meta)
    c.validate()
    return c


def default_raw_dir(ws) -> Path:
    """``<data>/raw/onion`` (create + download target of ``download.fetch_zenodo``)."""
    return ws.d("raw", "onion", base="data")


def ingest(ws, raw_dir: Path | None = None, name: str = "onion", **kw) -> Corpus:
    """Build the corpus and write it to ``<data>/interim/<name>`` (idempotent; overwrites)."""
    raw = Path(raw_dir) if raw_dir is not None else default_raw_dir(ws)
    c = build_corpus(raw, name=name, **kw)
    write_corpus(c, ws.d("interim", name, base="data"))
    log.info("corpus written: %d tracks, %d sessions, families %s", c.n_tracks, c.sessions.shape[0], list(c.families))
    return c


# ----------------------------------------------------------------------------------------------------------------------
# track metadata (artist / title / album): the Zenodo v2 record ships none of it (Blocker-A)
# ----------------------------------------------------------------------------------------------------------------------
METADATA_FILE = "id_metadata.parquet"           # stem is in METADATA_STEMS, so ``build_corpus`` picks it up automatically
_METADATA_SRC_COLS = {"id": ("track_id", "id", "trackid"), "artist": ("artist_name", "artist"), "title": ("title", "track_name", "name"),
                      "album": ("release", "album", "album_name")}


def prepare_metadata(source: Path, raw_dir: Path) -> dict:
    """Normalise an externally obtained track-metadata table into ``<raw_dir>/id_metadata.parquet`` (columns ``id, artist,
    title, album``) so the adapter ingests artists/titles the ordinary way (-> real ``artist_id`` = normalised artist name;
    no more one-artist-per-track).  Only these four columns are kept: captions, tag lists and anything else in the source
    are dropped.  Returns a provenance record (source path, sha256, row counts) for the corpus manifest.

    The Zenodo record 15394646 itself contains no such table; whoever supplies one must confirm its provenance and licence
    before any name derived from it is published."""
    import hashlib

    source = Path(source)
    df = pd.read_parquet(source) if source.suffix.lower() == ".parquet" else pd.read_csv(source, sep=None, engine="python")
    low = {str(c).lower(): c for c in df.columns}
    pick = {}
    for field_, cands in _METADATA_SRC_COLS.items():
        hit = next((low[c] for c in cands if c in low), None)
        if hit is None and field_ in ("id", "artist", "title"):
            raise ValueError(f"{source.name}: no column for {field_!r} (looked for {cands}; have {list(df.columns)})")
        pick[field_] = hit
    out = pd.DataFrame({"id": df[pick["id"]].astype(str).str.strip()})
    for f_ in ("artist", "title", "album"):
        out[f_] = df[pick[f_]].fillna("").astype(str).str.strip() if pick[f_] is not None else ""
    out = out.drop_duplicates("id").reset_index(drop=True)
    dst = Path(raw_dir) / METADATA_FILE
    out.to_parquet(dst, index=False)
    h = hashlib.sha256()
    with open(source, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    rec = {"source_file": source.name, "source_sha256": h.hexdigest(), "rows": int(len(out)),
           "with_artist": int((out["artist"] != "").sum()), "with_title": int((out["title"] != "").sum()),
           "unique_artists": int(out.loc[out["artist"] != "", "artist"].map(normalize_artist).nunique()),
           "written_to": str(dst.name), "provenance": "UNVERIFIED - not part of Zenodo 15394646; confirm origin and licence before publishing names"}
    write_json(Path(raw_dir) / "METADATA_PROVENANCE.json", rec)
    return rec


# ----------------------------------------------------------------------------------------------------------------------
# inventory (PLAN S0)
# ----------------------------------------------------------------------------------------------------------------------
def _id_pattern(ids: Sequence[str]) -> str:
    s = [str(x) for x in ids if pd.notna(x) and str(x).strip()][:200]
    if not s:
        return "empty"
    if all(re.fullmatch(r"-?\d+", x) for x in s):
        return "integer"
    if all(re.fullmatch(r"[0-9a-fA-F]{8}-[0-9a-fA-F-]{27}", x) for x in s):
        return "uuid"
    ln = {len(x) for x in s}
    if all(re.fullmatch(r"[A-Za-z0-9]+", x) for x in s):
        return f"alnum{ln.pop()}" if len(ln) == 1 else "alnum-varlen"
    return "other"


def _essentia_reference() -> set[str]:
    for base in (Path(__file__).resolve().parents[3], Path.cwd()):
        p = base / "essentia_descriptor_names.json"
        if p.exists():
            try:
                return {_norm_col(x) for x in read_json(p)}
            except Exception:
                return set()
    return set()


def inventory(raw_dir: Path, sample_rows: int = 3, count_rows: bool | str = "auto", max_sample: int = 200,
              count_limit_bytes: int = 10 * 1024 ** 2) -> dict:
    """Describe every data file: rows (line count, exact-or-None), #columns, header/separator, dtype summary, ``sample_rows``
    sample rows, an id-semantics guess, plus dataset-level checks (events file, timestamp unit and range, Essentia
    resemblance to AcousticBrainz, metadata availability = Blocker-A probe, expected-but-missing files).
    ``count_rows``: True | False | 'auto' (count lines only for files <= ``count_limit_bytes`` compressed).
    All guesses are heuristics; nothing is verified against upstream documentation."""
    raw_dir = Path(raw_dir)
    files = FileIndex(raw_dir)
    out: dict = {"raw_dir": str(raw_dir), "n_files": len(files.files), "files": {}}
    patterns: dict[str, str] = {}
    ref = _essentia_reference()
    for p in files.files:
        rel = p.relative_to(raw_dir).as_posix()
        info: dict = {"bytes": p.stat().st_size, "kind": None, "error": None}
        try:
            lay = sniff_layout(p)
            info.update({"format": lay.kind, "separator": lay.sep if lay.kind == "delimited" else None, "has_header": lay.header})
            df = read_table(p, nrows=max_sample, expand_lists=False, float32=False)
            raw_cols = list(df.columns)
            info["n_columns"] = len(raw_cols)
            info["columns_head"] = [str(c) for c in raw_cols[:12]]
            kinds = {"numeric": 0, "list": 0, "string": 0}
            if len(raw_cols) <= 2000:
                for c in raw_cols[1:]:
                    if pd.api.types.is_numeric_dtype(df[c]):
                        kinds["numeric"] += 1
                    else:
                        s = df[c].dropna().astype(str).head(50).str.strip()
                        kinds["list" if len(s) and s.str.startswith(_LIST_START).mean() > 0.9 else "string"] += 1
            else:
                num_cnt = int(sum(bool(pd.api.types.is_numeric_dtype(dt)) for dt in df.dtypes[1:]))
                kinds = {"numeric": num_cnt, "list": 0, "string": int(len(raw_cols) - 1 - num_cnt)}
            info["dtype_summary"] = kinds
            info["sample_rows"] = [[str(v)[:40] for v in row[:8]] for row in df.head(sample_rows).itertuples(index=False, name=None)]
            pat = _id_pattern(df["id"].astype(str).tolist())
            ns = norm_stem(p.name)
            is_events = "user" in ns and "track" in ns
            info["kind"] = "events" if is_events else "features/metadata"
            info["id_semantics_guess"] = ("col0 = user id, col1 = track id" if is_events else "col0 = track id") + f" (pattern {pat})"
            info["id_pattern"] = pat
            patterns[rel] = pat
            do_count = count_rows is True or (count_rows == "auto" and info["bytes"] <= count_limit_bytes)
            if do_count and lay.kind == "delimited":
                info["rows"] = max(count_lines(p) - (1 if lay.header else 0), 0)
                info["rows_exact"] = "line count (upper bound if cells contain newlines)"
            elif lay.kind != "delimited":
                info["rows"] = len(df) if len(df) < max_sample else None
            else:
                info["rows"] = None
                info["rows_note"] = f"not counted (> {count_limit_bytes} bytes compressed); sampled {len(df)} rows"
            if is_events:
                _, idx = _event_layout(p)
                info["event_columns"] = {k: (lay.names[i] if i is not None else None) for k, i in idx.items()}
                if idx["ts"] is not None:
                    ts, unit = _ts_from_strings(df[lay.names[idx["ts"]]].astype(str).to_numpy())
                    if unit is None:
                        u = detect_timestamp_unit(ts[ts > 0])
                        sec = to_seconds(ts[ts > 0], u)
                        info["timestamp"] = {"unit": u, "sample_min_utc": pd.Timestamp(int(sec.min()), unit="s").isoformat() if sec.size else None,
                                             "sample_max_utc": pd.Timestamp(int(sec.max()), unit="s").isoformat() if sec.size else None}
            if "essentia" in ns and ref:
                cols = [_norm_col(c) for c in raw_cols[1:]]
                stripped = {re.sub(r"\[\d+\]$", "", c) for c in cols}
                ov = len(stripped & ref)
                info["essentia_descriptor_overlap"] = {"matching": ov, "columns_distinct": len(stripped),
                                                       "resembles_acousticbrainz": bool(ov >= 0.3 * max(len(stripped), 1))}
        except Exception as e:
            info["error"] = f"{type(e).__name__}: {e}"
            log.warning("inventory: %s failed: %s", rel, info["error"])
        out["files"][rel] = info
    # dataset-level
    ev, kind = find_event_files(files)
    out["events"] = {"files": [p.name for p in ev], "kind": kind}
    meta_found: dict[str, str] = {}
    for stem in METADATA_STEMS:
        for p in files.find_files(stem):
            try:
                names = peek_columns(p)
            except Exception:
                continue
            probe = pd.DataFrame(columns=["id"] + names)
            for f_, al in _META_ALIASES.items():
                c = _find_alias(probe, al)
                if c is not None:
                    meta_found.setdefault(f_, f"{p.name}:{c}")
    out["metadata_fields_found"] = meta_found
    out["blocker_a"] = not ("artist" in meta_found and "title" in meta_found)
    out["expected_files_missing"] = [s for s in EXPECTED_STEMS if not files.find_files(s)]
    out["id_patterns"] = {k: patterns[k] for k in sorted(patterns)}
    return out


def write_inventory(raw_dir: Path, out: Path | None = None, **kw) -> dict:
    """Run ``inventory`` and write ``<raw_dir>/INVENTORY.json`` (or ``out``); returns the dict."""
    inv = inventory(raw_dir, **kw)
    write_json(Path(out) if out is not None else Path(raw_dir) / "INVENTORY.json", jsonable(inv))
    log.info("inventory written for %d files", inv["n_files"])
    return inv
