"""Track C (no behaviour data) — MTG-Jamendo adapter (PLAN §4.2 / S11).

No listening data: used to test whether the sonic compound vocabulary generalises and to evaluate tag prediction from compound
membership (genre / instrument / mood-theme tags).  ``sessions`` is empty (:func:`empty_sessions`).

VERIFIED (github.com/MTG/mtg-jamendo-dataset README, ``scripts/commons.py``, ``data/autotagging_moodtheme.tsv``, project page;
read 2026-09-29):
  * TSV files (tab separated, header ``TRACK_ID ARTIST_ID ALBUM_ID PATH DURATION TAGS``); tags live in columns ``row[5:]`` (one
    tag per extra column, rows have a variable number of fields); a tag is ``<category>---<name>`` with category in
    {genre, instrument, mood/theme}; ids look like ``track_0000948`` / ``artist_000087`` / ``album_000149``; PATH like
    ``48/948.mp3``; DURATION in seconds.  Files: raw.tsv (56,639), raw_30s.tsv (55,701), raw_30s_cleantags.tsv,
    raw_30s_cleantags_50artists.tsv (55,609), autotagging.tsv (the baseline file; a repo alias of the 50-artists file -
    when fetched as raw content it showed only that file name), autotagging_{genre,instrument,moodtheme,top50tags}.tsv.
    195 tags total; 87 genre / 40 instrument / 56 mood-theme in the splits.
  * Licences: code Apache-2.0, **metadata CC BY-NC-SA 4.0** (non-commercial + share-alike: check before any release), audio
    individual CC licences, other use needs written permission of Jamendo S.A.
  * Download script offers ``audio`` (508 GB), ``audio-low`` (156 GB), ``melspecs`` (229 GB) and ``acousticbrainz``
    (Essentia statistics as JSON) for ``raw_30s``.
UNVERIFIED: (1) the TSVs contain **only ids** - no artist/track *names* (none of the verified columns hold names), so
``artist``/``title`` are filled with the ids (``attrs['has_names'] = False``) unless ``names_path`` is supplied; (2) the layout,
extractor version and statistic names (``var`` vs ``stdev``) of the ``acousticbrainz`` JSON files (the project page gives no
details); the reader accepts AcousticBrainz-style nested JSON either way; (3) whether comma-separated multi-tags inside one
column occur (commons.py splits on ',', we do too).

ALIGNMENT (PLAN S11): if the Jamendo JSON is produced by the same Essentia music extractor as AcousticBrainz, the five acoustic
families can be built with the same descriptor names (:func:`feature_blocks` reuses ``mbid_stack.extract_ab_features``), but
extractor *versions* may differ - :func:`align_report` computes the overlap from the descriptors actually present.  The
melspectrograms are learned-model inputs, not Essentia descriptors: not alignable.
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp

from ..corpus import FamilyBlock
from ..runtime import get_logger
from .mbid_stack import (FAMILY_NAMES, align_report as _align_report, doc_descriptor_names, extract_ab_features, ab_columns,
                         iter_members)

log = get_logger("melomane.adapters.jamendo")

CATEGORIES = ("genre", "instrument", "mood/theme")
TAG_HYPHEN = "---"


def empty_sessions(n_tracks: int) -> sp.csr_matrix:
    """Corpus.sessions for a corpus without behaviour data (0 x N)."""
    return sp.csr_matrix((0, n_tracks), dtype=np.float32)


def _digits(s: str) -> int:
    m = re.search(r"(\d+)", str(s))
    return int(m.group(1)) if m else -1


def read_tsv(path: str | Path) -> pd.DataFrame:
    """Jamendo metadata TSV -> DataFrame(track_id, artist_id, album_id, path, duration, tags:list[str]).  Streams with the csv
    module because rows have a variable number of tab-separated fields (tags in ``row[5:]``, as in the dataset's commons.py)."""
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        rd = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        header = next(rd, None)
        if header is None:
            return pd.DataFrame(columns=["track_id", "artist_id", "album_id", "path", "duration", "tags"])
        if header and header[0].strip().upper() != "TRACK_ID":                 # headerless variant: first row is data
            rows.append(header)
        for r in rd:
            if len(r) < 5 or not r[0].strip():
                continue
            rows.append(r)
    return pd.DataFrame({
        "track_id": [r[0].strip() for r in rows], "artist_id": [r[1].strip() for r in rows], "album_id": [r[2].strip() for r in rows],
        "path": [r[3].strip() for r in rows], "duration": pd.to_numeric([r[4] for r in rows], errors="coerce"),
        "tags": [[t.strip() for t in r[5:] if t.strip()] for r in rows]})


def parse_tags(tags: Iterable[str]) -> dict[str, list[str]]:
    """['genre---rock', 'mood/theme---happy,calm'] -> {'genre': ['rock'], 'instrument': [], 'mood/theme': ['happy', 'calm']}
    (unknown categories are dropped with the raw string kept out; comma-splitting mirrors commons.py)."""
    out = {c: [] for c in CATEGORIES}
    for t in tags:
        if TAG_HYPHEN not in t:
            continue
        cat, name = t.split(TAG_HYPHEN, 1)
        if cat in out:
            out[cat].extend(x.strip() for x in name.split(",") if x.strip())
    return out


def load_tracks(path: str | Path, *, split_files: Mapping[str, str | Path] | None = None,
                names_path: str | Path | None = None) -> pd.DataFrame:
    """Jamendo TSV -> Corpus-contract DataFrame (index = raw_id 0..N-1 in file order).

    Columns: ``ext_id`` (track_id string), ``artist_id`` (int64 from ``artist_000087``), ``artist``, ``title`` (ids unless
    ``names_path`` gives a CSV with columns track_id, artist, title), ``album_id``, ``path``, ``duration_ms``, ``genre``,
    ``instrument``, ``mood_theme`` ('|'-joined tag names), ``tags`` (full ``category---name`` '|'-joined), ``split`` (from
    ``split_files={'train': ..., 'validation': ..., 'test': ...}`` TSVs, else '')."""
    d = read_tsv(path)
    parsed = [parse_tags(t) for t in d["tags"]]
    out = pd.DataFrame({
        "ext_id": d["track_id"],
        "artist_id": np.array([_digits(a) for a in d["artist_id"]], dtype=np.int64),
        "artist": d["artist_id"], "title": d["track_id"], "album_id": d["album_id"], "path": d["path"],
        "duration_ms": (d["duration"] * 1000).astype("float32"),
        "genre": ["|".join(p["genre"]) for p in parsed], "instrument": ["|".join(p["instrument"]) for p in parsed],
        "mood_theme": ["|".join(p["mood/theme"]) for p in parsed],
        "tags": ["|".join(f"{c}{TAG_HYPHEN}{n}" for c in CATEGORIES for n in p[c]) for p in parsed],
        "split": "",
    })
    has_names = False
    if names_path is not None:
        nm = pd.read_csv(names_path, dtype=str).set_index("track_id")
        for col in ("artist", "title"):
            if col in nm:
                out[col] = out["ext_id"].map(nm[col]).fillna(out[col])
        has_names = True
    if split_files:
        sid = {}
        for name, p in split_files.items():
            for t in read_tsv(p)["track_id"]:
                sid[t] = name
        out["split"] = out["ext_id"].map(sid).fillna("")
    out.index = pd.RangeIndex(len(out))
    out.attrs.update({"source": "MTG-Jamendo (metadata CC BY-NC-SA 4.0)", "has_behaviour": False, "has_lyrics": False,
                      "has_names": has_names, "n_tracks": len(out)})
    return out


def tag_matrix(tracks: pd.DataFrame, category: str | None = None) -> tuple[sp.csr_matrix, list[str]]:
    """Binary (n_tracks x n_tags) matrix + tag names (``category---name``) for tag-prediction transfer (S11b).
    ``category`` in {'genre','instrument','mood/theme'} restricts the columns."""
    lists = [[t for t in str(s).split("|") if t and (category is None or t.startswith(category + TAG_HYPHEN))]
             for s in tracks["tags"].fillna("")]
    names = sorted({t for l in lists for t in l})
    ix = {t: i for i, t in enumerate(names)}
    rows = [i for i, l in enumerate(lists) for _ in set(l)]
    cols = [ix[t] for l in lists for t in set(l)]
    return sp.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(len(lists), len(names))), names


# ----------------------------------------------------------------------------------------------------------------------
# Essentia ('acousticbrainz'-type) feature files
# ----------------------------------------------------------------------------------------------------------------------
def iter_feature_docs(source: str | Path) -> Iterator[tuple[int, dict]]:
    """Yield ``(jamendo_numeric_id, json_document)`` from a directory of ``*.json`` files (any depth, e.g. ``48/948.json``) or
    from a tar / tar.gz / tar.zst of them (streamed).  The numeric id is the last digit run of the file *name*; UNVERIFIED that
    the official archives use this naming."""
    src = Path(source)
    if src.is_dir():
        for p in sorted(src.rglob("*.json")):
            try:
                yield _digits(p.stem), json.loads(p.read_text(encoding="utf-8"))
            except (ValueError, OSError):
                log.warning("unreadable feature file %s", p.name)
        return
    for name, f in iter_members(src, lambda n: n.lower().endswith(".json")):
        try:
            yield _digits(Path(name).stem), json.loads(f.read())
        except ValueError:
            log.warning("unreadable feature member %s", name)


def feature_blocks(tracks: pd.DataFrame, source: str | Path | Iterable[tuple[int, dict]], *,
                   families: Sequence[str] | None = None, max_nan_frac: float = 0.3) -> tuple[dict[str, FamilyBlock], list[str]]:
    """Essentia features of Jamendo tracks -> the five acoustic family blocks with the *same* descriptor definitions as
    AcousticBrainz (``mbid_stack.AB_FAMILY_SPECS``; ``stdev`` accepted natively or derived from ``var``).

    ``source``: directory / tar of JSON files, or an iterable of ``(numeric_id, doc)``.  Returns ``(blocks,
    descriptor_names_seen)``; feed the second item to :func:`align_report` to see which families are really comparable
    (extractor versions may differ - UNVERIFIED).  Rows with > ``max_nan_frac`` missing values in a family are dropped;
    remaining gaps are column-median imputed."""
    fams = list(families or FAMILY_NAMES)
    raw_of = {_digits(e): i for i, e in enumerate(tracks["ext_id"])}
    docs = iter_feature_docs(source) if isinstance(source, (str, Path)) else source
    rows: list[int] = []; X: dict[str, list[np.ndarray]] = {f: [] for f in fams}
    names: set[str] = set()
    for num, doc in docs:
        r = raw_of.get(num)
        if r is None:
            continue
        if len(names) < 4000:
            names.update(doc_descriptor_names(doc))
        feats = extract_ab_features(doc, fams)
        rows.append(r)
        for f in fams:
            X[f].append(feats[f])
    out: dict[str, FamilyBlock] = {}
    cols = ab_columns(fams)
    if not rows:
        return out, sorted(names)
    rr = np.asarray(rows, dtype=np.int64)
    for f in fams:
        M = np.vstack(X[f])
        ok = np.isnan(M).mean(axis=1) <= max_nan_frac
        if not ok.any():
            continue
        M, r = M[ok], rr[ok]
        med = np.nanmedian(np.where(np.isfinite(M), M, np.nan), axis=0) if M.size else np.zeros(M.shape[1])
        med = np.where(np.isfinite(med), med, 0.0)
        i, j = np.where(np.isnan(M))
        M[i, j] = med[j]
        order = np.argsort(r)
        out[f] = FamilyBlock(r[order], M[order].astype(np.float32), cols[f], "MTG-Jamendo Essentia features (UNVERIFIED extractor version)")
    return out, sorted(names)


def align_report(source_columns: Sequence[str], target_columns: Sequence[str] | None = None, **kw) -> dict:
    """PLAN S11 alignment report for Jamendo: pass the descriptor names actually seen in the Essentia JSON files (second
    return value of :func:`feature_blocks`).  Melspectrograms (``melspecs`` download) are learned-model inputs and are never
    alignable with the Essentia descriptor families.  ``target_columns`` = Onion ``id_essentia`` names to test Track-A alignment."""
    kw.setdefault("source_name", "MTG-Jamendo Essentia features")
    return _align_report(source_columns, target_columns, **kw)
