"""Release bundles: turn a finished run (``docs/CONTRACTS.md`` layout) into an Epicure-style model directory.

Bundle files: ``embeddings.safetensors`` (key ``embeddings``; fp32 / fp16 / int8 with per-row ``scale``; optional PCA keys
``pca_components`` [+ ``pca_mean``]), ``vocab.json`` (cid / artist / title / album / year / category / dev_split /
region_tags only), ``modes.json``, ``factor_poles.npy``, ``supervised_poles.json`` + ``.npy``, ``config.json``, ``README.md``.

Reduced (``dims``) bundles are an orthogonal projection onto the top principal subspace of the *unit-normalised*
embeddings.  By default the projection is **uncentred** (truncated SVD) because the mean direction carries cosine
geometry (Core's mean pairwise cosine is ~0.35); ``pca_center=True`` gives classical centred PCA.  Poles in a reduced
bundle are stored already projected (as directions: no mean shift) so every file in the bundle shares one space; the
projection is kept so external full-space queries can be mapped in (``Melomane.project``).
No audio, no lyrics and no per-user data ever enter a bundle.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import logging
import subprocess
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
from safetensors.numpy import save_file

from ..geometry import unit

log = logging.getLogger(__name__)

FORMAT_VERSION = 1
VOCAB_FIELDS = ("cid", "artist", "title", "album", "year", "category", "dev_split", "region_tags")
OPTIONAL_COMPONENTS = ("modes", "factor_poles", "supervised_poles", "readme")
DEFAULT_INCLUDE = OPTIONAL_COMPONENTS
MODEL_ORDER = ("cooc", "composition", "core")

PROVENANCE: dict[str, dict[str, str]] = {
    "music4all-onion": {
        "source": "Music4All-Onion",
        "licence": "CC BY 4.0",
        "attribution": ("Derived from Music4All-Onion (Moscati et al., CIKM 2022, doi:10.1145/3511808.3557656; Zenodo), "
                        "licensed CC BY 4.0. Listening behaviour in that dataset is Last.fm-derived. This release contains "
                        "derived embeddings and metadata only - no audio, no lyrics, no per-user data. Artist, title and album strings are Music4All "
                        "metadata (Santana et al., IWSSIP 2020) as redistributed on Hugging Face (seungheondoh/enrich-music4all)."),
    },
    "mbid-stack": {
        "source": "MusicBrainz / ListenBrainz / AcousticBrainz / LRCLIB",
        "licence": "CC0 (MusicBrainz core data, AcousticBrainz) + ListenBrainz dump terms",
        "attribution": ("Derived from the MetaBrainz open data: MusicBrainz (CC0), ListenBrainz listens (hashed users, aggregate co-occurrence only), "
                        "AcousticBrainz low-level features (CC0) and LRCLIB lyric-derived aggregate features. This release contains derived "
                        "embeddings and metadata only - no audio, no lyrics, no per-user data."),
    },
    "playlist2vec": {
        "source": "Playlist2vec",
        "licence": "self-declared CC BY (unverified)",
        "attribution": ("Playlist2vec (Zenodo record 5002584) is an independent researcher's unofficial re-scrape of Spotify "
                        "API data; its provenance and Spotify terms-of-service status are unverified. Treat everything "
                        "derived from it as unofficial."),
    },
}
CAVEATS = (
    "Vectors are trained on listening co-occurrence (and, for Composition/Core, measured track features); they encode "
    "who-listens-to-what, not musical quality.",
    "The source data is skewed toward Last.fm-era, Western, English-language listening; cultural regions are imbalanced.",
    "Artist structure dominates raw nearest neighbours. Retrieval helpers are artist-disjoint by default; any evaluation "
    "you run must group by artist too, otherwise it is leaky.",
    "Category, mode and region labels are LLM-assisted and can be wrong; the embeddings themselves are LLM-free.",
    "Quantised (fp16 / int8) and PCA-reduced variants trade fidelity for size; see the retention figures if present.",
    "UMAP coordinates in the explorer are for visualisation only; every number is computed in the embedding space.",
    "No lyrics, audio or per-user data are included; do not attempt to re-identify listeners.",
)


# --------------------------------------------------------------------------------------------------------------------------
# numerics
# --------------------------------------------------------------------------------------------------------------------------
def fit_projection(E: np.ndarray, dims: int, center: bool = False) -> tuple[np.ndarray, np.ndarray | None]:
    """Top-``dims`` principal directions of the unit-normalised rows of ``E``.

    Returns ``(components (dims x D) float32, mean (D,) float32 | None)``.  Uses the DxD Gram matrix so it scales to
    ~10^5 rows.  Component signs are fixed deterministically (positive sum)."""
    E = np.asarray(E)
    d = E.shape[1]
    if not 1 <= dims < d:
        raise ValueError(f"dims must be in [1, {d - 1}] for a {d}-d embedding, got {dims}")
    X = unit(E.astype(np.float64))
    mean = X.mean(0) if center else None
    if center:
        X = X - mean
    w, v = np.linalg.eigh(X.T @ X)
    comps = v[:, ::-1][:, :dims].T.copy()
    comps *= np.where(comps.sum(1, keepdims=True) < 0, -1.0, 1.0)
    return comps.astype(np.float32), None if mean is None else mean.astype(np.float32)


def quantise_int8(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric per-row int8 quantisation: ``x ~ q * scale[:, None]`` with ``scale = max|x| / 127``."""
    X = np.asarray(X, dtype=np.float32)
    scale = np.abs(X).max(1) / 127.0
    scale = np.where(scale > 0, scale, 1.0).astype(np.float32)
    q = np.clip(np.rint(X / scale[:, None]), -127, 127).astype(np.int8)
    return q, scale


def dequantise_int8(q: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Inverse of :func:`quantise_int8` (float32)."""
    return q.astype(np.float32) * np.asarray(scale, dtype=np.float32)[:, None]


def project_directions(poles: np.ndarray, components: np.ndarray) -> np.ndarray:
    """Project directions (rows) into a reduced space and re-normalise (no mean shift: directions are differences)."""
    P = np.asarray(poles, dtype=np.float32) @ components.T
    return unit(P).astype(np.float32) if len(P) else P.astype(np.float32)


# --------------------------------------------------------------------------------------------------------------------------
# run access
# --------------------------------------------------------------------------------------------------------------------------
def list_models(run_dir: str | Path) -> list[str]:
    """Model names with an ``S5/models/<name>/embeddings.npy`` in the run (main models first, then ablations A-Z)."""
    root = Path(run_dir) / "S5" / "models"
    names = [p.name for p in sorted(root.iterdir()) if (p / "embeddings.npy").exists()] if root.is_dir() else []
    return sorted(names, key=lambda n: (MODEL_ORDER.index(n) if n in MODEL_ORDER else len(MODEL_ORDER), n))


def load_vocab(run_dir: str | Path) -> pd.DataFrame:
    """``S1/vocab.parquet`` validated to be in cid order 0..V-1."""
    df = pd.read_parquet(Path(run_dir) / "S1" / "vocab.parquet")
    if not (df["cid"].to_numpy() == np.arange(len(df))).all():
        raise ValueError("S1/vocab.parquet must be in cid order with cid == row index")
    return df


def vocab_records(df: pd.DataFrame) -> list[dict]:
    """Whitelisted, JSON-able track records (no ids other than cid, no tags, no popularity, no user data)."""
    out = []
    for r in df.reindex(columns=list(VOCAB_FIELDS)).itertuples(index=False):
        d = r._asdict()
        rec: dict[str, Any] = {"cid": int(d["cid"])}
        for k in ("artist", "title", "album", "category", "dev_split"):
            v = d[k]
            rec[k] = None if v is None or (isinstance(v, float) and v != v) else str(v)
        y = d["year"]
        rec["year"] = None if y is None or pd.isna(y) else int(y)
        rt = d["region_tags"]
        rec["region_tags"] = [t for t in str(rt).split("|") if t] if isinstance(rt, str) else []
        out.append(rec)
    return out


def git_commit(start: str | Path | None = None) -> str | None:
    """Current git commit of the repository containing ``start`` (default: this file), or None."""
    cwd = Path(start) if start else Path(__file__).resolve().parent
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(cwd), capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    sha = r.stdout.strip()
    return sha if r.returncode == 0 and len(sha) >= 7 else None


def _seeds(model_dir: Path) -> list[int]:
    man = model_dir / "manifest.json"
    if man.exists():
        try:
            d = json.loads(man.read_text(encoding="utf-8"))
            if isinstance(d.get("seeds"), list):
                return [int(s) for s in d["seeds"]]
            if d.get("seed") is not None:
                return [int(d["seed"])]
        except (ValueError, TypeError):
            log.warning("unreadable %s", man)
    return sorted(int(p.stem[4:]) for p in (model_dir / "seeds").glob("seed*.npy") if p.stem[4:].isdigit())


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sources_for(run_dir: Path, sources: Iterable[str] | None) -> list[str]:
    if sources is not None:
        keys = list(sources)
    else:
        keys = ["music4all-onion"]
        if "playlist2vec" in run_dir.parent.name.lower() and "music4all-onion" not in run_dir.parent.name.lower():
            keys = ["playlist2vec"]
    unknown = [k for k in keys if k not in PROVENANCE]
    if unknown:
        raise ValueError(f"unknown data source(s) {unknown}; known: {sorted(PROVENANCE)}")
    return keys


# --------------------------------------------------------------------------------------------------------------------------
# export
# --------------------------------------------------------------------------------------------------------------------------
def export_bundle(run_dir: str | Path, model: str, out_dir: str | Path, quantise: str | None = None,
                  dims: int | None = None, include: Iterable[str] = DEFAULT_INCLUDE, *, pca_center: bool = False,
                  variant: str | None = None, sources: Iterable[str] | None = None, metrics: Mapping | None = None,
                  extra_config: Mapping | None = None) -> Path:
    """Write the release bundle of ``model`` from ``run_dir`` into ``out_dir`` and return ``out_dir``.

    ``quantise``: None (fp32) | ``'fp16'`` | ``'int8'`` (per-row scale in key ``scale``).  ``dims``: PCA-reduce to that many
    dimensions (projection stored in the safetensors file).  ``include`` selects optional components from
    ``('modes', 'factor_poles', 'supervised_poles', 'readme')``.  ``metrics`` (from ``results/``) is rendered into the
    model card; numbers are never invented here."""
    if quantise not in (None, "fp16", "int8"):
        raise ValueError("quantise must be None, 'fp16' or 'int8'")
    include = set(include)
    bad = include - set(OPTIONAL_COMPONENTS)
    if bad:
        raise ValueError(f"unknown include component(s) {sorted(bad)}; choose from {OPTIONAL_COMPONENTS}")
    run_dir, out_dir = Path(run_dir), Path(out_dir)
    model_dir = run_dir / "S5" / "models" / model
    if not (model_dir / "embeddings.npy").exists():
        raise FileNotFoundError(f"no embeddings for model {model!r} under {model_dir}")
    vocab = load_vocab(run_dir)
    E = np.load(model_dir / "embeddings.npy").astype(np.float32)
    if len(E) != len(vocab):
        raise ValueError(f"embeddings have {len(E)} rows but vocab has {len(vocab)}")
    if not np.isfinite(E).all():
        raise ValueError("embeddings contain NaN/inf")
    out_dir.mkdir(parents=True, exist_ok=True)
    s6 = run_dir / "S6" / model

    comps = mean = None
    X = E
    if dims is not None:
        comps, mean = fit_projection(E, dims, center=pca_center)
        X = (E - mean if mean is not None else E) @ comps.T
    tensors: dict[str, np.ndarray] = {}
    if quantise == "int8":
        tensors["embeddings"], tensors["scale"] = quantise_int8(X)
    elif quantise == "fp16":
        tensors["embeddings"] = X.astype(np.float16)
    else:
        tensors["embeddings"] = np.ascontiguousarray(X, dtype=np.float32)
    if comps is not None:
        tensors["pca_components"] = comps
        if mean is not None:
            tensors["pca_mean"] = mean
    dtype = str(tensors["embeddings"].dtype)
    save_file(tensors, str(out_dir / "embeddings.safetensors"),
              metadata={"format": "melomane", "version": str(FORMAT_VERSION), "model": model, "dtype": dtype})

    records = vocab_records(vocab)
    (out_dir / "vocab.json").write_text(json.dumps(records, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

    def emit_poles(arr: np.ndarray) -> np.ndarray:
        return project_directions(arr, comps) if comps is not None else np.asarray(arr, dtype=np.float32)

    counts = {"n_modes": 0, "n_factors": 0, "n_supervised_poles": 0}
    if "modes" in include and (s6 / "modes.json").exists():
        modes = json.loads((s6 / "modes.json").read_text(encoding="utf-8"))
        (out_dir / "modes.json").write_text(json.dumps(modes, ensure_ascii=False), encoding="utf-8")
        counts["n_modes"] = len(modes.get("modes", []))
    if "factor_poles" in include and (s6 / "factor_poles.npy").exists():
        fp = emit_poles(np.load(s6 / "factor_poles.npy"))
        np.save(out_dir / "factor_poles.npy", fp)
        counts["n_factors"] = len(fp)
    if "supervised_poles" in include and (s6 / "supervised_poles.json").exists() and (s6 / "supervised_poles.npy").exists():
        sp = emit_poles(np.load(s6 / "supervised_poles.npy"))
        meta = json.loads((s6 / "supervised_poles.json").read_text(encoding="utf-8"))
        if len(meta.get("poles", [])) != len(sp):
            raise ValueError("supervised_poles.json and .npy disagree on pole count")
        (out_dir / "supervised_poles.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
        np.save(out_dir / "supervised_poles.npy", sp)
        counts["n_supervised_poles"] = len(sp)
    if not (s6 / "modes.json").exists():
        log.warning("run has no S6 outputs for %s: bundle will contain embeddings and vocabulary only", model)

    keys = _sources_for(run_dir, sources)
    config: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "model": model,
        "variant": variant or model,
        "n_items": len(E),
        "dim": int(X.shape[1]),
        "source_dim": int(E.shape[1]),
        "dtype": dtype,
        "quantisation": quantise,
        "int8_scale_key": "scale" if quantise == "int8" else None,
        "pca": None if comps is None else {"dims": int(dims), "centered": bool(mean is not None),
                                           "keys": ["pca_components"] + (["pca_mean"] if mean is not None else []),
                                           "fit_on": "unit-normalised embeddings"},
        "poles_space": "bundle",
        **counts,
        "seeds": _seeds(model_dir),
        "corpus": run_dir.parent.name,
        "profile": run_dir.name,
        "git_commit": git_commit(),
        "created_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "license": "CC BY 4.0",
        "data_provenance": [PROVENANCE[k] for k in keys],
        "attribution": " ".join(PROVENANCE[k]["attribution"] for k in keys),
        "caveats": list(CAVEATS),
        "contains": {"audio": False, "lyrics": False, "per_user_data": False},
    }
    if extra_config:
        config.update(extra_config)
    write_config_and_card(out_dir, config, metrics, readme="readme" in include)
    return out_dir


def write_config_and_card(out_dir: Path, config: dict, metrics: Mapping | None = None, readme: bool = True) -> None:
    """(Re)write ``config.json`` (with file hashes) and the model card."""
    files = {p.name: _sha256(p) for p in sorted(out_dir.iterdir())
             if p.is_file() and p.name not in ("config.json", "README.md")}
    config = {**config, "files": files}
    (out_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    if readme:
        (out_dir / "README.md").write_text(model_card(config, metrics), encoding="utf-8")


def export_all(run_dir: str | Path, out_dir: str | Path, small_dims: int = 64, models: Iterable[str] | None = None,
               metrics: Mapping[str, Mapping] | None = None, retention_k: int = 10, retention_queries: int = 500,
               seed: int = 0, **kw: Any) -> dict[str, Path]:
    """Full (fp32) and ``-small`` (``small_dims``-d int8) bundle for every model in the run (or ``models``).

    The small bundle's ``config.json`` and model card record the measured neighbour retention against the full bundle.
    Returns ``{bundle_name: directory}``."""
    from ..api import Melomane                                   # local import: api does not depend on release
    run_dir, out_dir = Path(run_dir), Path(out_dir)
    names = list(models) if models is not None else list_models(run_dir)
    if not names:
        raise FileNotFoundError(f"no models found under {run_dir / 'S5' / 'models'}")
    out: dict[str, Path] = {}
    for name in names:
        m_metrics = (metrics or {}).get(name)
        full = export_bundle(run_dir, name, out_dir / name, quantise=None, metrics=m_metrics, **kw)
        D = int(json.loads((full / "config.json").read_text(encoding="utf-8"))["dim"])
        dims = small_dims if small_dims < D else None
        if dims is None:
            log.info("%s: small_dims=%d >= dim %d, -small bundle is int8 only", name, small_dims, D)
        small = export_bundle(run_dir, name, out_dir / f"{name}-small", quantise="int8", dims=dims,
                              variant=f"{name}-small", metrics=m_metrics, **kw)
        ret = neighbour_retention(Melomane.from_pretrained(full), Melomane.from_pretrained(small), k=retention_k,
                                  n_queries=retention_queries, seed=seed)
        cfg = json.loads((small / "config.json").read_text(encoding="utf-8"))
        cfg["retention"] = {"reference": name, **ret}
        cfg.pop("files", None)
        write_config_and_card(small, cfg, m_metrics, readme=(small / "README.md").exists())
        out[name], out[f"{name}-small"] = full, small
    return out


def neighbour_retention(reference: Any, variant: Any, k: int = 10, n_queries: int = 200, seed: int = 0,
                        exclude_artist: bool = True) -> dict[str, float]:
    """How much of the reference top-``k`` neighbourhood a compressed variant preserves.

    ``reference`` / ``variant`` are :class:`~melomane.api.Melomane` objects or bundle paths over the same vocabulary.
    Returns ``knn_overlap`` (mean |A ∩ B| / k over random query tracks; artist-disjoint by default), ``top1_agreement`` and
    the query count."""
    from ..api import Melomane
    ref = reference if isinstance(reference, Melomane) else Melomane.from_pretrained(reference)
    var = variant if isinstance(variant, Melomane) else Melomane.from_pretrained(variant)
    if len(ref) != len(var):
        raise ValueError("reference and variant have different vocabularies")
    n = len(ref)
    rng = np.random.default_rng(seed)
    q = rng.choice(n, size=min(n_queries, n), replace=False)
    k = min(k, n - 1)
    tops = []
    for m in (ref, var):
        S = m.U[q] @ m.U.T
        S[np.arange(len(q)), q] = -np.inf
        if exclude_artist:
            S[ref._artist_code[q][:, None] == ref._artist_code[None, :]] = -np.inf   # same artist mask for both
        idx = np.argpartition(-S, k, axis=1)[:, :k]
        order = np.argsort(-np.take_along_axis(S, idx, 1), axis=1)
        tops.append(np.take_along_axis(idx, order, 1))
    a, b = tops
    overlap = np.mean([len(set(x) & set(y)) / k for x, y in zip(a, b)])
    return {"k": int(k), "n_queries": int(len(q)), "knn_overlap": round(float(overlap), 4),
            "top1_agreement": round(float((a[:, 0] == b[:, 0]).mean()), 4), "exclude_artist": bool(exclude_artist)}


# --------------------------------------------------------------------------------------------------------------------------
# model card
# --------------------------------------------------------------------------------------------------------------------------
def _flatten(d: Mapping, prefix: str = "") -> Iterable[tuple[str, Any]]:
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, Mapping):
            yield from _flatten(v, key + ".")
        else:
            yield key, v


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.4g}"
    if v is None:
        return "-"
    return str(v)


def model_card(config: Mapping, metrics: Mapping | None = None) -> str:
    """Markdown model card (Hugging Face front matter + summary, usage, files, provenance, caveats, metrics).

    ``metrics`` is an arbitrary (nested) mapping of numbers that the caller read from ``results/``; nothing is computed
    or hard-coded here."""
    variant = config.get("variant") or config.get("model", "model")
    pca = config.get("pca")
    lines = [
        "---", "license: cc-by-4.0", "library_name: melomane", "tags:", "  - music", "  - embeddings", "  - melomane",
        f"  - {config.get('model', 'model')}", "---", "", f"# Melomane - `{variant}`", "",
        "Track embeddings from the Melomane replication of the Epicure food-embedding papers "
        "(emergent geometry, ICA modes, SLERP steering) on music listening data. Artefacts only: "
        "**no audio, no lyrics, no per-user data**.", "",
        "## Summary", "", "| | |", "|---|---|",
        f"| model | `{config.get('model')}` |", f"| tracks | {config.get('n_items')} |",
        f"| dimensions | {config.get('dim')} (source {config.get('source_dim')}) |",
        f"| dtype | {config.get('dtype')} |", f"| quantisation | {config.get('quantisation') or 'none (fp32)'} |",
        f"| PCA projection | {'none' if not pca else str(pca['dims']) + ' dims, ' + ('centred' if pca.get('centered') else 'uncentred')} |",
        f"| modes / factors / supervised poles | {config.get('n_modes')} / {config.get('n_factors')} / {config.get('n_supervised_poles')} |",
        f"| training seeds | {', '.join(map(str, config.get('seeds') or [])) or '-'} |",
        f"| corpus / profile | {config.get('corpus')} / {config.get('profile')} |",
        f"| git commit | {config.get('git_commit') or '-'} |", f"| created (UTC) | {config.get('created_utc')} |", "",
        "## Usage", "", "```python", "from melomane.api import Melomane", "",
        f"m = Melomane.from_pretrained('path/to/{variant}')",
        "m.find('artist title')                          # fuzzy lookup",
        "m.neighbors('Artist - Title', k=10)             # artist-disjoint by default",
        "m.slerp('Artist - Title', 'category:jazz', 45)  # rotate 45 degrees toward a pole",
        "m.list_modes(); m.mode_members(0); m.describe('Artist - Title')", "```", "",
        "## Files", "", "| file | content |", "|---|---|",
        "| `embeddings.safetensors` | key `embeddings`"
        + (" (int8) + per-row `scale`" if config.get("quantisation") == "int8" else "")
        + (" + `pca_components`" + (" / `pca_mean`" if pca and pca.get("centered") else "") if pca else "") + " |",
        "| `vocab.json` | cid, artist, title, album, year, category, dev_split, region_tags |",
        "| `modes.json` | ICA modes (members, label, property, driver, coherence) |",
        "| `factor_poles.npy`, `supervised_poles.json/.npy` | emergent and supervised pole directions |",
        "| `config.json` | this card's machine-readable twin (with file hashes) |", "",
    ]
    ret = config.get("retention")
    if ret:
        lines += ["## Compression retention", "",
                  f"Measured against the full-precision `{ret.get('reference')}` bundle over {ret.get('n_queries')} random "
                  f"query tracks (top-{ret.get('k')}, {'artist-disjoint' if ret.get('exclude_artist') else 'leaky'}): "
                  f"neighbour overlap **{_fmt(ret.get('knn_overlap'))}**, top-1 agreement **{_fmt(ret.get('top1_agreement'))}**.", ""]
    if metrics:
        lines += ["## Metrics", "", "Generated from `results/`; not edited by hand.", "", "| metric | value |", "|---|---|"]
        lines += [f"| {k} | {_fmt(v)} |" for k, v in _flatten(metrics)]
        lines.append("")
    lines += ["## Data provenance and licence", "", f"Licence: **{config.get('license', 'CC BY 4.0')}**.", ""]
    lines += [f"- **{p['source']}** ({p['licence']}): {p['attribution']}" for p in config.get("data_provenance", [])]
    lines += ["", "## Caveats", ""] + [f"- {c}" for c in config.get("caveats", CAVEATS)] + [""]
    return "\n".join(lines)
