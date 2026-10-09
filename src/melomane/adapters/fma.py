"""Track C (no behaviour data) — Free Music Archive ``fma_metadata`` adapter (PLAN §4.2 / S11).

FMA has **no listening sessions**: it can only supply tracks + sonic features + genre tags, so it is used to test whether the
Track-A/B compound vocabulary generalises (cluster-usage entropy, tag prediction from compound membership).  The corpus
sessions matrix is therefore empty (``empty_sessions``) and Cooc models cannot be trained on it.

VERIFIED (github.com/mdeff/fma README + utils.py + features.py + FMA paper, read 2026-09-29):
  * ``fma_metadata.zip`` (342 MiB) holds ``tracks.csv`` (106,574 tracks), ``genres.csv`` (163 genres: name + parent),
    ``features.csv`` (librosa) and ``echonest.csv`` (Echonest/Spotify features for 13,129 tracks).
  * ``tracks.csv``: 2-row header (``pd.read_csv(index_col=0, header=[0, 1])``), top level in {track, album, artist, set};
    list-valued cells (tags, genres, genres_all) are Python literals; ``set.subset`` in {small, medium, large},
    ``set.split`` in {training, validation, test}.  Column names seen in the repo notebooks: track.{title, date_created,
    date_recorded, duration, genre_top, genres, genres_all, interest, language(_code), license, listens, tags, bit_rate},
    album.{id, title, date_released, ...}, artist.{id, name, ...}.
  * ``features.csv``: 3-level header (feature, statistics, number), 518 columns = 3 chroma sets x 84 + tonnetz 42 + mfcc 140 +
    spectral centroid/bandwidth/rolloff/rmse/zcr 7 each + spectral contrast 49; seven statistics (mean, std, skew, kurtosis,
    median, min, max); librosa 0.5.0, windows of 2048 / hop 512.
  * ``echonest.csv``: 3-level header, top level ``echonest``, second level in {audio_features, metadata, ranks,
    social_features, temporal_features}; audio_features include acousticness, danceability, energy, instrumentalness,
    liveness, speechiness, tempo, valence.
UNVERIFIED: exact ``genres.csv`` header spelling (code accepts ``genre_id`` index + ``#tracks``/``parent``/``title``/``top_level``),
exact ``tracks.csv`` column list (readers are tolerant; missing columns become NaN), and every count in echonest.csv beyond
the above.  Licences: metadata CC BY 4.0, code MIT, audio under per-artist CC licences.

ALIGNMENT (PLAN S11): FMA features come from librosa 0.5.0 (+ Echonest), NOT from the Essentia music extractor behind
AcousticBrainz / Onion ``id_essentia``.  No family is definitionally identical; see :func:`align_report` /
:data:`FMA_CONCEPT_MAP`.  Therefore feature_blocks() are *conceptual proxies* and the report says so.
"""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp

from ..corpus import FamilyBlock
from ..runtime import get_logger
from .mbid_stack import FAMILY_NAMES, align_report as _align_report

log = get_logger("melomane.adapters.fma")

SUBSET_ORDER = ("small", "medium", "large")

# Melomane family -> librosa feature names in features.csv
FMA_FAMILY_MAP: dict[str, list[str]] = {
    "timbre": ["mfcc"],
    "spectral_shape": ["spectral_bandwidth", "spectral_centroid", "spectral_contrast", "spectral_rolloff", "zcr"],
    "harmony_tonal": ["chroma_cens", "chroma_cqt", "chroma_stft", "tonnetz"],
    "dynamics_loudness": ["rmse", "rms"],           # 'rms' = newer librosa spelling
}                                                    # rhythm_groove: no librosa tempo feature -> Echonest tempo/danceability only
ECHONEST_FAMILY_MAP = {"rhythm_groove": ["tempo", "danceability"]}
ECHONEST_HELDOUT = ["acousticness", "energy", "instrumentalness", "liveness", "speechiness", "valence"]

# source stem -> Essentia (AcousticBrainz) descriptor stem of the same *concept*, different extractor/definition
FMA_CONCEPT_MAP: dict[str, str] = {
    "mfcc": "lowlevel.mfcc", "spectral_centroid": "lowlevel.spectral_centroid", "spectral_bandwidth": "lowlevel.spectral_spread",
    "spectral_rolloff": "lowlevel.spectral_rolloff", "spectral_contrast": "lowlevel.spectral_contrast_coeffs",
    "zcr": "lowlevel.zerocrossingrate", "rmse": "lowlevel.spectral_rms", "rms": "lowlevel.spectral_rms",
    "chroma_stft": "tonal.hpcp", "chroma_cqt": "tonal.hpcp", "chroma_cens": "tonal.hpcp",
    "tempo": "rhythm.bpm", "danceability": "rhythm.danceability",      # Echonest danceability is proprietary != Essentia's DFA measure
}


def align_report(source_columns: Sequence[str], target_columns: Sequence[str] | None = None, **kw) -> dict:
    """PLAN S11 alignment report for FMA columns (flattened as ``feature.stat[number]`` or ``feature``): identical-definition
    coverage vs the AcousticBrainz/Essentia target (or ``target_columns`` = Onion ``id_essentia`` names) plus the
    concept-level proxies in :data:`FMA_CONCEPT_MAP`.  Expected outcome on real FMA columns: every family
    ``conceptual_only`` / ``not_alignable`` and ``can_use_identical_definition == False`` -> run the transfer inside Track B."""
    kw.setdefault("concept_map", FMA_CONCEPT_MAP)
    kw.setdefault("source_name", "FMA (librosa 0.5.0 + Echonest)")
    return _align_report(source_columns, target_columns, **kw)


# ----------------------------------------------------------------------------------------------------------------------
# readers
# ----------------------------------------------------------------------------------------------------------------------
def _literal_list(x) -> list:
    if isinstance(x, (list, tuple)):
        return list(x)
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return []
    try:
        v = ast.literal_eval(str(x))
        return list(v) if isinstance(v, (list, tuple, set)) else []
    except (ValueError, SyntaxError):
        return []


def _clean_index(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the stray 'track_id'/NaN index row some pandas versions keep after a multi-row header, coerce index to int."""
    idx = pd.to_numeric(pd.Series(df.index, index=df.index), errors="coerce")
    df = df[idx.notna().to_numpy()].copy()
    df.index = idx[idx.notna()].astype("int64").to_numpy()
    df.index.name = "track_id"
    return df


def load_genres(path: str | Path) -> pd.DataFrame:
    """``genres.csv`` -> DataFrame indexed by genre_id with columns title, parent, top_level, n_tracks."""
    g = pd.read_csv(path, index_col=0)
    g = g.rename(columns={"#tracks": "n_tracks"})
    g.index = pd.to_numeric(g.index, errors="coerce")
    g = g[g.index.notna()]
    g.index = g.index.astype("int64")
    g.index.name = "genre_id"
    for c in ("title", "parent", "top_level", "n_tracks"):
        if c not in g:
            g[c] = np.nan
    return g[["title", "parent", "top_level", "n_tracks"]]


def _get(tr: pd.DataFrame, level0: str, level1: str, default=np.nan) -> pd.Series:
    col = (level0, level1)
    if col in tr.columns:
        return tr[col]
    return pd.Series(default, index=tr.index)


def load_tracks(path: str | Path, genres: str | Path | pd.DataFrame | None = None, *, subset: str | None = None,
                nrows: int | None = None) -> pd.DataFrame:
    """``tracks.csv`` -> Corpus-contract DataFrame (index = raw_id 0..N-1, file order).

    Columns: ``ext_id`` (FMA track_id as str), ``artist_id`` (int64; FMA artist id, unique negatives when missing),
    ``artist, title, album, year`` (recording year, else album release year), ``duration_ms``, ``popularity`` (= listens),
    ``language``, ``license``, ``genre_top``, ``genres`` (ids '|'-joined, own genres), ``tags`` (genre *titles* of genres_all,
    '|'-joined, ancestors included; ids if ``genres`` file not given), ``split``, ``subset``, ``artist_tags``.
    ``subset`` = 'small' | 'medium' | 'large' keeps tracks in that subset *or smaller* (FMA subsets are nested)."""
    tr = pd.read_csv(path, index_col=0, header=[0, 1], low_memory=False, nrows=nrows)
    if tr.columns.nlevels != 2:
        raise ValueError(f"{path}: expected the 2-level header of FMA tracks.csv, got {tr.columns.nlevels} level(s)")
    tr = _clean_index(tr)
    if subset is not None:
        if subset not in SUBSET_ORDER:
            raise ValueError(f"subset must be one of {SUBSET_ORDER}")
        rank = _get(tr, "set", "subset", "large").astype(str).map({s: i for i, s in enumerate(SUBSET_ORDER)})
        tr = tr[(rank <= SUBSET_ORDER.index(subset)).to_numpy()]
    n = len(tr)
    gdf = load_genres(genres) if isinstance(genres, (str, Path)) else genres
    gname = gdf["title"].to_dict() if gdf is not None else {}
    all_g = [_literal_list(v) for v in _get(tr, "track", "genres_all", "[]")]
    own_g = [_literal_list(v) for v in _get(tr, "track", "genres", "[]")]
    aid = pd.to_numeric(_get(tr, "artist", "id"), errors="coerce")
    miss = aid.isna().to_numpy()
    aid = aid.fillna(0).astype("int64").to_numpy().copy()
    aid[miss] = -(np.arange(int(miss.sum())) + 1)                       # never collapse unknown artists into one group
    year = pd.to_datetime(_get(tr, "track", "date_recorded"), errors="coerce").dt.year
    year = year.fillna(pd.to_datetime(_get(tr, "album", "date_released"), errors="coerce").dt.year)
    lang = _get(tr, "track", "language_code")
    if lang.isna().all():
        lang = _get(tr, "track", "language")
    out = pd.DataFrame({
        "ext_id": tr.index.astype(str),
        "artist_id": aid,
        "artist": _get(tr, "artist", "name", "").fillna("").astype(str).to_numpy(),
        "title": _get(tr, "track", "title", "").fillna("").astype(str).to_numpy(),
        "album": _get(tr, "album", "title", "").fillna("").astype(str).to_numpy(),
        "year": year.astype("float32").to_numpy(),
        "duration_ms": (pd.to_numeric(_get(tr, "track", "duration"), errors="coerce") * 1000).astype("float32").to_numpy(),
        "popularity": pd.to_numeric(_get(tr, "track", "listens"), errors="coerce").astype("float32").to_numpy(),
        "language": lang.astype(object).where(lang.notna(), "").astype(str).to_numpy(),
        "license": _get(tr, "track", "license", "").fillna("").astype(str).to_numpy(),
        "genre_top": _get(tr, "track", "genre_top", "").fillna("").astype(str).to_numpy(),
        "genres": ["|".join(map(str, g)) for g in own_g],
        "tags": ["|".join(str(gname.get(int(i), i)) for i in g) for g in all_g],
        "split": _get(tr, "set", "split", "").fillna("").astype(str).to_numpy(),
        "subset": _get(tr, "set", "subset", "").fillna("").astype(str).to_numpy(),
        "track_tags": ["|".join(map(str, _literal_list(v))) for v in _get(tr, "track", "tags", "[]")],
    })
    out.index = pd.RangeIndex(n)
    out.attrs.update({"source": "FMA fma_metadata (CC BY 4.0)", "has_behaviour": False, "has_lyrics": False, "n_tracks": n})
    return out


def empty_sessions(n_tracks: int) -> sp.csr_matrix:
    """Corpus.sessions for a corpus without behaviour data (0 x N)."""
    return sp.csr_matrix((0, n_tracks), dtype=np.float32)


def tag_matrix(tracks: pd.DataFrame, col: str = "tags") -> tuple[sp.csr_matrix, list[str]]:
    """Binary (n_tracks x n_tags) matrix from a '|'-joined tag column, plus the sorted tag names (tag-prediction targets)."""
    lists = [[t for t in str(s).split("|") if t] for s in tracks[col].fillna("")]
    names = sorted({t for l in lists for t in l})
    ix = {t: i for i, t in enumerate(names)}
    rows = [i for i, l in enumerate(lists) for _ in set(l)]
    cols = [ix[t] for l in lists for t in set(l)]
    return sp.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(len(lists), len(names))), names


def load_features(path: str | Path, nrows: int | None = None) -> pd.DataFrame:
    """``features.csv`` (3-level header: feature, statistics, number; index track_id) -> DataFrame, int track_id index."""
    f = pd.read_csv(path, index_col=0, header=[0, 1, 2], low_memory=False, nrows=nrows)
    if f.columns.nlevels != 3:
        raise ValueError(f"{path}: expected 3 header levels, got {f.columns.nlevels}")
    return _clean_index(f)


def load_echonest(path: str | Path, nrows: int | None = None) -> pd.DataFrame:
    """``echonest.csv`` -> DataFrame with 3-level columns (echonest, group, name).  Tolerates 1-level files (columns are
    then taken as audio_features names)."""
    e = pd.read_csv(path, index_col=0, header=[0, 1, 2], low_memory=False, nrows=nrows)
    if e.columns.nlevels == 3:
        return _clean_index(e)
    e = pd.read_csv(path, index_col=0, low_memory=False, nrows=nrows)
    e.columns = pd.MultiIndex.from_tuples([("echonest", "audio_features", str(c)) for c in e.columns])
    return _clean_index(e)


def feature_columns(features: pd.DataFrame) -> list[str]:
    """Flattened names ``feature.stat[number]`` (the form :func:`align_report` expects)."""
    return [f"{a}.{b}[{int(c) if str(c).isdigit() else c}]" for a, b, c in features.columns]


# ----------------------------------------------------------------------------------------------------------------------
# family blocks
# ----------------------------------------------------------------------------------------------------------------------
def _block(rows: np.ndarray, X: np.ndarray, cols: list[str], source: str) -> FamilyBlock | None:
    ok = ~np.isnan(X).any(axis=1)
    if not ok.any():
        return None
    order = np.argsort(rows[ok])
    return FamilyBlock(rows[ok][order].astype(np.int64), X[ok][order].astype(np.float32), cols, source)


def feature_blocks(tracks: pd.DataFrame, features: pd.DataFrame | str | Path | None = None,
                   echonest: pd.DataFrame | str | Path | None = None, *, stats: Sequence[str] = ("mean", "std"),
                   families: Iterable[str] | None = None, include_echonest: bool = True) -> dict[str, FamilyBlock]:
    """FamilyBlocks in the corpus index space (rows = position in ``tracks``, matched on ``ext_id`` = FMA track_id).

    * librosa families (``FMA_FAMILY_MAP``): timbre <- mfcc; spectral_shape <- spectral_{bandwidth,centroid,contrast,rolloff} +
      zcr; harmony_tonal <- chroma_{cens,cqt,stft} + tonnetz; dynamics_loudness <- rmse.  Only ``stats`` (default mean+std,
      the two statistics Essentia/Onion also carry) are used, all 518 columns if ``stats`` lists all seven.
    * ``rhythm_groove`` <- Echonest tempo + danceability (13,129 tracks only; there is no librosa tempo feature).
    * ``heldout_echonest`` (NOT a compound family; probe material): acousticness, energy, instrumentalness, liveness,
      speechiness, valence.
    All blocks are conceptual proxies of the Melomane families, not the Essentia definitions (see :func:`align_report`)."""
    want = set(families) if families is not None else set(FAMILY_NAMES) | {"heldout_echonest"}
    pos = pd.Series(np.arange(len(tracks)), index=pd.to_numeric(tracks["ext_id"], errors="coerce"))
    pos = pos[pos.index.notna()]
    out: dict[str, FamilyBlock] = {}
    if isinstance(features, (str, Path)):
        features = load_features(features)
    if features is not None:
        r = pos.reindex(features.index)
        have = r.notna().to_numpy()
        for fam, names in FMA_FAMILY_MAP.items():
            if fam not in want:
                continue
            sel = [c for c in features.columns if c[0] in names and c[1] in stats]
            if not sel or not have.any():
                continue
            X = features.loc[:, sel].to_numpy(dtype=np.float64)[have]
            cols = [f"{a}.{b}[{c}]" for a, b, c in sel]
            b = _block(r[have].to_numpy().astype(np.int64), X, cols, "FMA features.csv (librosa 0.5.0) - conceptual proxy")
            if b is not None:
                out[fam] = b
    if include_echonest and echonest is not None:
        if isinstance(echonest, (str, Path)):
            echonest = load_echonest(echonest)
        af = [c for c in echonest.columns if c[1] == "audio_features"]
        by_name = {c[2]: c for c in af}
        r = pos.reindex(echonest.index)
        have = r.notna().to_numpy()
        for fam, names in list(ECHONEST_FAMILY_MAP.items()) + [("heldout_echonest", ECHONEST_HELDOUT)]:
            if fam not in want:
                continue
            sel = [by_name[n] for n in names if n in by_name]
            if not sel or not have.any():
                continue
            X = echonest.loc[:, sel].to_numpy(dtype=np.float64)[have]
            b = _block(r[have].to_numpy().astype(np.int64), X, [c[2] for c in sel], "FMA echonest.csv (Echonest/Spotify) - conceptual proxy")
            if b is not None:
                out[fam] = b
    return out
