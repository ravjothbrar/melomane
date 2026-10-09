"""On-disk corpus contract shared by every adapter (PLAN §5.3).

A *corpus* is what an adapter (Onion, Playlist2vec, synthetic, ...) produces and what stages S1+ consume:

    tracks.parquet            one row per raw track, index = raw_id (0..N-1)
                              required: ext_id, artist_id, artist, title
                              optional: album, year, popularity, duration_ms, explicit, language, tags, has_lyrics
    sessions.npz              CSR (n_sessions x N) binary "documents" (listening sessions / playlists)
    sessions_meta.parquet     optional per-session columns (user_id, n_tracks, ...)
    families/<name>.npz       baked-in feature blocks (compound-graph inputs): rows (int64 raw ids), X (float32), columns
    measured.parquet          scalar measured summaries (probe stratum (a)), index = raw_id (NaN where unknown)
    heldout.parquet           held-out measured scalars (probe stratum (b))
    external/<name>.npz       learned external embeddings for agreement analyses (E5): rows, X
    corpus.json               name, source, version, notes, degraded flags, session semantics
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp

from .runtime import read_json, write_json, jsonable

REQUIRED_TRACK_COLS = ["ext_id", "artist_id", "artist", "title"]


@dataclass
class FamilyBlock:
    rows: np.ndarray                      # raw track ids covered by this block
    X: np.ndarray                         # (len(rows), d) float32
    columns: list[str] = field(default_factory=list)
    source: str = ""

    def dense(self, n_tracks: int) -> np.ndarray:
        """(n_tracks, d) with NaN rows for uncovered tracks."""
        out = np.full((n_tracks, self.X.shape[1]), np.nan, dtype=np.float32)
        out[self.rows] = self.X
        return out

    def subset(self, keep_raw: np.ndarray, mapping: np.ndarray) -> "FamilyBlock":
        """Restrict/re-index to a new index space: mapping[raw_id] = new id or -1."""
        new = mapping[self.rows]
        ok = new >= 0
        return FamilyBlock(new[ok].astype(np.int64), self.X[ok], self.columns, self.source)


@dataclass
class Corpus:
    name: str
    tracks: pd.DataFrame
    sessions: sp.csr_matrix
    families: dict[str, FamilyBlock] = field(default_factory=dict)
    measured: pd.DataFrame | None = None
    heldout: pd.DataFrame | None = None
    external: dict[str, FamilyBlock] = field(default_factory=dict)
    sessions_meta: pd.DataFrame | None = None
    meta: dict = field(default_factory=dict)

    @property
    def n_tracks(self) -> int:
        return len(self.tracks)

    def validate(self) -> None:
        miss = [c for c in REQUIRED_TRACK_COLS if c not in self.tracks.columns]
        if miss:
            raise ValueError(f"corpus {self.name}: tracks missing columns {miss}")
        if not (self.tracks.index.to_numpy() == np.arange(len(self.tracks))).all():
            raise ValueError("tracks index must be 0..N-1 (raw_id)")
        if self.sessions.shape[1] != len(self.tracks):
            raise ValueError(f"sessions has {self.sessions.shape[1]} columns, tracks has {len(self.tracks)}")
        for k, b in {**self.families, **self.external}.items():
            if b.rows.size and (b.rows.max() >= len(self.tracks) or b.rows.min() < 0):
                raise ValueError(f"block {k}: rows out of range")
            if b.X.shape[0] != b.rows.size:
                raise ValueError(f"block {k}: X/rows length mismatch")
        for nm, df in (("measured", self.measured), ("heldout", self.heldout)):
            if df is not None and len(df) != len(self.tracks):
                raise ValueError(f"{nm} must have one row per track")


def write_corpus(c: Corpus, path: Path) -> None:
    c.validate()
    path = Path(path); path.mkdir(parents=True, exist_ok=True)
    tr = c.tracks.copy()
    tr.index.name = "raw_id"
    tr.to_parquet(path / "tracks.parquet")
    sp.save_npz(path / "sessions.npz", sp.csr_matrix(c.sessions, dtype=np.float32))
    if c.sessions_meta is not None:
        c.sessions_meta.to_parquet(path / "sessions_meta.parquet")
    for sub, blocks in (("families", c.families), ("external", c.external)):
        for k, b in blocks.items():
            (path / sub).mkdir(exist_ok=True)
            np.savez(path / sub / f"{k}.npz", rows=b.rows.astype(np.int64), X=b.X.astype(np.float32),
                     columns=np.array(b.columns, dtype=object), source=np.array(b.source))
    for nm, df in (("measured", c.measured), ("heldout", c.heldout)):
        if df is not None:
            df.to_parquet(path / f"{nm}.parquet")
    write_json(path / "corpus.json", jsonable({"name": c.name, **c.meta}))


def read_corpus(path: Path, load_families: bool = True) -> Corpus:
    path = Path(path)
    tr = pd.read_parquet(path / "tracks.parquet").reset_index(drop=True)
    X = sp.load_npz(path / "sessions.npz").tocsr()
    meta = read_json(path / "corpus.json")
    name = meta.pop("name", path.name)

    def blocks(sub):
        out = {}
        d = path / sub
        if d.exists() and load_families:
            for f in sorted(d.glob("*.npz")):
                z = np.load(f, allow_pickle=True)
                out[f.stem] = FamilyBlock(z["rows"], z["X"], list(z["columns"]), str(z["source"]))
        return out

    def opt(nm):
        p = path / f"{nm}.parquet"
        return pd.read_parquet(p).reset_index(drop=True) if p.exists() else None

    sm = path / "sessions_meta.parquet"
    return Corpus(name, tr, X, blocks("families"), opt("measured"), opt("heldout"), blocks("external"),
                  pd.read_parquet(sm) if sm.exists() else None, meta)


def write_enrichment(c: Corpus, path: Path) -> None:
    """Write only the feature-side parts of `c` (families/external/measured/heldout + corpus.json) into an existing corpus
    directory, leaving tracks/sessions untouched.  Used when a behaviour corpus is enriched from a feature corpus."""
    import shutil
    path = Path(path)
    for sub in ("families", "external"):
        if (path / sub).exists():
            shutil.rmtree(path / sub)
    for sub, blocks in (("families", c.families), ("external", c.external)):
        for k, b in blocks.items():
            (path / sub).mkdir(exist_ok=True)
            np.savez(path / sub / f"{k}.npz", rows=b.rows.astype(np.int64), X=b.X.astype(np.float32),
                     columns=np.array(b.columns, dtype=object), source=np.array(b.source))
    for nm, df in (("measured", c.measured), ("heldout", c.heldout)):
        p = path / f"{nm}.parquet"
        if df is not None:
            df.to_parquet(p)
        elif p.exists():
            p.unlink()
    write_json(path / "corpus.json", jsonable({"name": c.name.split("+")[0], **c.meta}))
