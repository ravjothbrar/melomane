"""Shared loaders/helpers for stages (kept small; heavy logic lives in the dedicated modules)."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp

from ..corpus import Corpus, read_corpus
from ..pipeline import Ctx, StageSkip

_CACHE: dict = {}


def cached(key, fn):
    if key not in _CACHE:
        if len(_CACHE) > 6:
            _CACHE.pop(next(iter(_CACHE)))
        _CACHE[key] = fn()
    return _CACHE[key]


def clear_cache():
    _CACHE.clear()


def corpus_of(ctx: Ctx, load_families: bool = True) -> Corpus:
    d = ctx.corpus_dir
    if not (d / "corpus.json").exists():
        raise StageSkip(f"corpus '{ctx.corpus}' has not been ingested yet ({d})")
    mt = (d / "corpus.json").stat().st_mtime
    fam_mt = max([p.stat().st_mtime for p in (d / "families").glob("*.npz")], default=0) if (d / "families").exists() else 0
    return cached(("corpus", str(d), mt, fam_mt, load_families), lambda: read_corpus(d, load_families))


def vocab_of(ctx: Ctx) -> pd.DataFrame:
    """S1 vocabulary, with `category` / `region_tags` upgraded from S2 labels where the LLM labelled the track."""
    p = ctx.stage_dir("S1") / "vocab.parquet"
    lp = ctx.stage_dir("S2") / "labels.parquet"
    key = ("vocab", str(p), p.stat().st_mtime, lp.stat().st_mtime if lp.exists() else 0)

    def load():
        v = pd.read_parquet(p).reset_index(drop=True)
        if lp.exists():
            lab = pd.read_parquet(lp, columns=["category", "regions", "labelled"]).reset_index(drop=True)
            if len(lab) == len(v):
                m = lab["labelled"].to_numpy(bool)
                v.loc[m, "category"] = lab.loc[m, "category"].to_numpy()
                v.loc[m, "region_tags"] = lab.loc[m, "regions"].to_numpy()
        return v
    return cached(key, load)


def labels_of(ctx: Ctx) -> pd.DataFrame | None:
    p = ctx.stage_dir("S2") / "labels.parquet"
    if not p.exists():
        return None
    return cached(("labels", str(p), p.stat().st_mtime), lambda: pd.read_parquet(p).reset_index(drop=True))


def model_dir(ctx: Ctx, name: str) -> Path:
    return ctx.stage_dir("S5") / "models" / name


def load_embeddings(ctx: Ctx, name: str) -> np.ndarray:
    p = model_dir(ctx, name) / "embeddings.npy"
    return cached(("emb", str(p), p.stat().st_mtime), lambda: np.load(p))


def available_models(ctx: Ctx) -> list[str]:
    d = ctx.stage_dir("S5") / "models"
    return sorted(p.name for p in d.iterdir() if (p / "embeddings.npy").exists()) if d.exists() else []


def sessions_cid(ctx: Ctx) -> sp.csr_matrix:
    return sp.load_npz(ctx.stage_dir("S3") / "sessions_cid.npz").tocsr()


def stratum_targets(ctx: Ctx, vocab: pd.DataFrame, labels: pd.DataFrame | None, corp: Corpus) -> dict:
    """Probe registry (PLAN §7.1) restricted to what exists: measured (a), held-out (b), external LLM/other (c), cultural."""
    T: dict[str, dict] = {}
    raw = vocab["rep_raw"].to_numpy()
    for stratum, df in (("a_baked", corp.measured), ("b_heldout", corp.heldout)):
        if df is None:
            continue
        sub = df.iloc[raw].reset_index(drop=True)
        for c in sub.columns:
            y = sub[c].to_numpy(dtype=float)
            if np.isfinite(y).sum() >= 200 and np.nanstd(y) > 0:
                T[f"{stratum}:{c}"] = {"kind": "continuous", "y": y, "stratum": stratum}
    if labels is not None and labels["labelled"].any():
        from ..annotation import DIMS
        for d in DIMS + ["era"]:
            y = labels[d].to_numpy(float)
            if np.isfinite(y).sum() >= 200 and np.nanstd(y) > 0:
                T[f"c_llm:{d}"] = {"kind": "ordinal", "y": y, "stratum": "c_external"}
    if "year" in vocab and vocab["year"].notna().sum() > 200:
        T["c_ext:year"] = {"kind": "continuous", "y": vocab["year"].to_numpy(float), "stratum": "c_external"}
    if "popularity" in vocab and vocab["popularity"].notna().sum() > 200:
        T["c_ext:popularity_BEHAVIOUR_ADJACENT"] = {"kind": "continuous", "y": vocab["popularity"].to_numpy(float), "stratum": "c_external_excluded_H1"}
    return T
