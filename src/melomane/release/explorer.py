"""Static explorer payload builder (``apps/explorer``) + a tiny local server.

For every requested model the builder writes, under ``<out>/data/<model>/``:
``emb.bin`` (int8 or fp16, N x d, row-major little-endian) [+ ``scale.bin`` float32 per-row scale for int8],
``umap.bin`` (float32 N x 2, normalised to [-1, 1]; **visualisation only**: cosine, n_neighbors 32, min_dist 0.004),
``poles.bin`` (float32 P x d unit poles: supervised, then ``factor:<i>``, then ``mode:<id>``) + ``poles.json``,
``modes.json`` (compact atlas, members as payload indices) and ``mode_idx.bin`` (int16 primary mode per track, -1 none).
Shared: ``data/tracks.json`` (columnar metadata) and ``data/manifest.json`` (files, shapes, dtypes, sha256, licence).
(A fixed UMAP seed makes the layout reproducible but single-threaded: allow minutes for 50k tracks.)
Vectors are the PCA-reduced (``dims``) small embeddings of :mod:`melomane.release.export`; every explorer number
(neighbours, SLERP) is computed in the browser from them, the 2-D layout only positions the points.
The whole payload must stay <= ``max_payload_mb`` (default 50 MB) or the build raises.

CLI: ``python -m melomane.release.explorer build RUN_DIR OUT_DIR [--models a b] [--max-tracks N]`` and
``python -m melomane.release.explorer serve OUT_DIR [--port 8000]``.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import functools
import hashlib
import http.server
import json
import logging
import os
import shutil
import threading
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..api import mode_poles_from_members, primary_mode_assignment
from ..curation import normalize_artist
from ..geometry import unit
from ..slerp import compose_poles, slerp as _slerp
from .export import (PROVENANCE, dequantise_int8, fit_projection, list_models, load_vocab, project_directions,
                     quantise_int8)

log = logging.getLogger(__name__)

PAYLOAD_VERSION = 1
UMAP_PARAMS = {"metric": "cosine", "n_neighbors": 32, "min_dist": 0.004, "n_components": 2}
APP_FILES = ("index.html", "explorer.js", "math.js", "style.css", "favicon.svg")
MODEL_LABELS = {"cooc": "Cooc", "composition": "Composition", "core": "Core", "composition_acoustic": "Composition (acoustic only)",
                "composition_lyric": "Composition (lyric only)", "core_acoustic": "Core (acoustic only)",
                "cooc_artistmasked": "Cooc (artist-masked)", "cooc_avg": "Cooc (averaged)"}


def default_app_dir() -> Path:
    """``apps/explorer`` of the repository this package lives in (or ``$MELOMANE_EXPLORER_APP``)."""
    env = os.environ.get("MELOMANE_EXPLORER_APP")
    cands = [Path(env)] if env else []
    cands += [p / "apps" / "explorer" for p in Path(__file__).resolve().parents]
    for c in cands:
        if (c / "index.html").exists():
            return c
    raise FileNotFoundError("apps/explorer not found; pass app_dir= or set MELOMANE_EXPLORER_APP")


def model_label(name: str) -> str:
    return MODEL_LABELS.get(name, name.replace("_", " "))


# --------------------------------------------------------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------------------------------------------------------
def select_tracks(vocab, max_tracks: int) -> np.ndarray:
    """Payload subset: the ``max_tracks`` most-listened tracks (ties by cid), returned as ascending cids."""
    n = len(vocab)
    if max_tracks >= n:
        return np.arange(n)
    order = np.lexsort((np.arange(n), -vocab["n_sessions"].to_numpy())) if "n_sessions" in vocab else np.arange(n)
    return np.sort(order[:max_tracks])


def umap_layout(U: np.ndarray, seed: int = 0, **overrides: Any) -> np.ndarray:
    """2-D UMAP of unit rows (cosine, n_neighbors 32, min_dist 0.004) scaled into [-1, 1]^2 (aspect preserved).
    Visualisation only: never use these coordinates for any measurement."""
    try:
        import sklearn.utils.validation as _sk_v, sklearn.utils as _sk_u
        _orig_check = _sk_v.check_array
        if not getattr(_orig_check, "_melomane_patched", False):
            def _compat_check(*args, **kwargs):
                if "ensure_all_finite" in kwargs and "force_all_finite" not in kwargs:
                    kwargs["force_all_finite"] = kwargs.pop("ensure_all_finite")
                return _orig_check(*args, **kwargs)
            _compat_check._melomane_patched = True
            _sk_v.check_array = _compat_check
            _sk_u.check_array = _compat_check
    except Exception:
        pass
    import umap                                                       # heavy import, kept local
    params = {**UMAP_PARAMS, **overrides}
    params["n_neighbors"] = int(min(params["n_neighbors"], max(2, len(U) - 1)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        xy = umap.UMAP(random_state=seed, **params).fit_transform(U.astype(np.float32))
    xy = np.asarray(xy, dtype=np.float64)
    xy -= (xy.min(0) + xy.max(0)) / 2
    xy /= max(np.abs(xy).max(), 1e-9)
    return xy.astype(np.float32)


def _write(path: Path, arr: np.ndarray, dtype: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(np.ascontiguousarray(arr, dtype=dtype).tobytes())


def _file_entry(path: Path, root: Path, **extra: Any) -> dict:
    return {"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), **extra}


def build_explorer(run_dir: str | Path, out_dir: str | Path, models: Sequence[str] | None = None, max_tracks: int = 50000,
                   dims: int = 64, dtype: str = "int8", seed: int = 0, max_payload_mb: float = 50.0,
                   pca_center: bool = False, umap_kwargs: dict | None = None, app_dir: str | Path | None = None,
                   sources: Sequence[str] | None = None) -> dict:
    """Build the static explorer into ``out_dir`` (app files + ``data/`` payload) and return the manifest dict.

    ``dtype``: ``'int8'`` (per-row scale) or ``'fp16'`` for the small vectors; ``dims``: PCA dimensionality of those
    vectors (skipped when >= the embedding dim).  Raises ``ValueError`` if the payload exceeds ``max_payload_mb``."""
    if dtype not in ("int8", "fp16"):
        raise ValueError("dtype must be 'int8' or 'fp16'")
    run_dir, out_dir = Path(run_dir), Path(out_dir)
    names = list(models) if models else list_models(run_dir)
    if not names:
        raise FileNotFoundError(f"no models under {run_dir / 'S5' / 'models'}")
    app = Path(app_dir) if app_dir else default_app_dir()
    vocab = load_vocab(run_dir)
    keep = select_tracks(vocab, max_tracks)
    N = len(keep)
    cid_to_idx = np.full(len(vocab), -1, dtype=np.int64)
    cid_to_idx[keep] = np.arange(N)
    data = out_dir / "data"
    if data.exists():
        shutil.rmtree(data)
    data.mkdir(parents=True)

    # ---- tracks (columnar, dictionary-encoded) --------------------------------------------------------------------
    sub = vocab.iloc[keep]
    art_codes: dict[str, int] = {}; art_names: list[str] = []; art_idx = []
    for a in sub["artist"].astype(str):
        code = art_codes.setdefault(normalize_artist(a), len(art_codes))
        if code == len(art_names):
            art_names.append(a)
        art_idx.append(code)
    cats = sorted({str(c) for c in sub["category"].dropna()})
    cat_of = {c: i for i, c in enumerate(cats)}
    tracks = {"n": int(N), "cid": [int(c) for c in keep], "artist": art_names, "artist_idx": art_idx,
              "title": [str(t) for t in sub["title"]], "categories": cats,
              "category_idx": [cat_of.get(str(c), -1) if c == c and c is not None else -1 for c in sub["category"]],
              "year": [None if y != y else int(y) for y in sub["year"].astype(float)]}
    (data / "tracks.json").write_text(json.dumps(tracks, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

    manifest_models = []
    for name in names:
        mdir = run_dir / "S5" / "models" / name
        E = np.load(mdir / "embeddings.npy").astype(np.float32)
        if len(E) != len(vocab):
            raise ValueError(f"{name}: embeddings rows {len(E)} != vocab {len(vocab)}")
        comps = mean = None
        X = E
        if dims is not None and dims < E.shape[1]:
            comps, mean = fit_projection(E, dims, center=pca_center)
            X = (E - mean if mean is not None else E) @ comps.T
        Xs = X[keep]
        d = Xs.shape[1]
        md = data / name
        if dtype == "int8":
            q, scale = quantise_int8(Xs)
            _write(md / "emb.bin", q, "i1"); _write(md / "scale.bin", scale, "<f4")
            U = unit(dequantise_int8(q, scale))
        else:
            h = Xs.astype(np.float16)
            _write(md / "emb.bin", h, "<f2")
            U = unit(h.astype(np.float32))
        xy = umap_layout(U, seed=seed, **(umap_kwargs or {}))
        _write(md / "umap.bin", xy, "<f4")

        # poles and modes (all expressed in the small space, re-normalised)
        s6 = run_dir / "S6" / name
        pole_meta: list[dict] = []; pole_vecs: list[np.ndarray] = []
        def add(arr: np.ndarray, metas: list[dict], in_small_space: bool = False) -> None:
            if len(arr):
                if in_small_space:
                    arr = unit(arr).astype(np.float32)
                else:
                    arr = project_directions(arr, comps) if comps is not None else unit(arr).astype(np.float32)
                pole_vecs.append(arr); pole_meta.extend(metas)
        if (s6 / "supervised_poles.json").exists():
            sm = json.loads((s6 / "supervised_poles.json").read_text(encoding="utf-8"))["poles"]
            add(np.load(s6 / "supervised_poles.npy"), [{"name": p["name"], "group": "supervised", "kind": p.get("kind"),
                                                          "n_pos": p.get("n_pos"), "n_neg": p.get("n_neg")} for p in sm])
        if (s6 / "factor_poles.npy").exists():
            fp = np.load(s6 / "factor_poles.npy")
            add(fp, [{"name": f"factor:{i}", "group": "factor", "kind": "emergent"} for i in range(len(fp))])
        modes_out: list[dict] = []
        mode_idx = np.full(N, -1, dtype="<i2")
        if (s6 / "modes.json").exists():
            raw_modes = json.loads((s6 / "modes.json").read_text(encoding="utf-8")).get("modes", [])
            members = [[int(cid_to_idx[c]) for c in m["members"] if cid_to_idx[c] >= 0] for m in raw_modes]
            mp = mode_poles_from_members(U, members)
            add(mp, [{"name": f"mode:{m['mode_id']}", "group": "mode", "kind": "emergent", "label": m.get("label")}
                     for m in raw_modes], in_small_space=True)
            mode_idx = primary_mode_assignment(U, members, mp).astype("<i2")
            modes_out = [{"id": int(m["mode_id"]), "label": m.get("label"), "property": m.get("property"),
                          "driver": m.get("driver"), "factor": m.get("factor"), "size": int(m.get("size", len(m["members"]))),
                          "n_in_payload": len(mem), "coherence": m.get("coherence"), "members": mem}
                         for m, mem in zip(raw_modes, members)]
        P = np.concatenate(pole_vecs) if pole_vecs else np.zeros((0, d), np.float32)
        _write(md / "poles.bin", P, "<f4")
        (md / "poles.json").write_text(json.dumps(pole_meta, ensure_ascii=False), encoding="utf-8")
        (md / "modes.json").write_text(json.dumps(modes_out, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        _write(md / "mode_idx.bin", mode_idx, "<i2")

        files = {"emb": "emb.bin", "umap": "umap.bin", "poles": "poles.bin", "poles_meta": "poles.json",
                 "modes": "modes.json", "mode_idx": "mode_idx.bin"}
        if dtype == "int8":
            files["scale"] = "scale.bin"
        manifest_models.append({
            "name": name, "label": model_label(name), "n": int(N), "dim": int(d), "dtype": dtype, "n_poles": int(len(P)),
            "n_modes": len(modes_out), "compression": {"pca_dims": int(d) if comps is not None else None,
                                                        "source_dim": int(E.shape[1])},
            "files": {k: _file_entry(md / v, out_dir) for k, v in files.items()}})
        log.info("explorer payload for %s: %d tracks, %d-d %s", name, N, d, dtype)

    for f in APP_FILES:
        shutil.copyfile(app / f, out_dir / f)
    src_keys = list(sources) if sources else ["music4all-onion"]
    manifest = {
        "version": PAYLOAD_VERSION, "n": int(N), "n_vocab": int(len(vocab)), "models": manifest_models,
        "tracks": _file_entry(data / "tracks.json", out_dir),
        "umap": {**UMAP_PARAMS, "seed": seed, "note": "visualisation only; all numbers are computed in the embedding space"},
        "preview": {"kind": "search_links", "note": "audio previews are 'search on...' links only; the iTunes Search API is "
                                                    "limited to ~20 requests/min per IP so nothing is prefetched"},
        "license": "CC BY 4.0", "attribution": [PROVENANCE[k]["attribution"] for k in src_keys],
        "created_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    total = sum(p.stat().st_size for p in data.rglob("*") if p.is_file())
    manifest["total_bytes"] = int(total)
    if total > max_payload_mb * 1024 * 1024:
        raise ValueError(f"explorer payload is {total / 1e6:.1f} MB > {max_payload_mb} MB: lower max_tracks/dims or use int8")
    (data / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")
    return manifest


# --------------------------------------------------------------------------------------------------------------------------
# reading a payload back (tests, reference implementation of the browser maths)
# --------------------------------------------------------------------------------------------------------------------------
@dataclass
class ModelPayload:
    name: str
    U: np.ndarray                      # (N x d) unit rows, decoded exactly as the browser does
    xy: np.ndarray                     # (N x 2)
    poles: np.ndarray                  # (P x d)
    pole_meta: list[dict]
    modes: list[dict]
    mode_idx: np.ndarray               # (N,) int16


@dataclass
class Payload:
    manifest: dict
    tracks: dict
    models: dict[str, ModelPayload] = field(default_factory=dict)
    root: Path = Path(".")

    def pole(self, model: str, name: str) -> np.ndarray:
        m = self.models[model]
        for i, p in enumerate(m.pole_meta):
            if p["name"] == name:
                return m.poles[i]
        raise KeyError(name)

    def query(self, model: str, seed_idx: int, poles: Sequence[str] = (), theta_deg: float = 0.0, k: int = 10,
              weights: Sequence[float] | None = None, exclude_artist: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """Reference implementation of the explorer's query in numpy (same algorithm as ``math.js``): SLERP the seed
        toward the (composed) poles, rank by cosine, drop the seed (and its artist).  Returns ``(idx, scores)``."""
        m = self.models[model]
        q = m.U[seed_idx]
        if len(poles) and theta_deg > 0:
            vecs = [self.pole(model, p) for p in poles]
            q = _slerp(q, vecs[0] if len(vecs) == 1 else compose_poles(vecs, None if weights is None else list(weights)), theta_deg)
        a = np.asarray(self.tracks["artist_idx"])
        mask = (a == a[seed_idx]) if exclude_artist else np.zeros(len(a), bool)
        mask[seed_idx] = True
        s = m.U.astype(np.float64) @ unit(q.astype(np.float64))
        s[mask] = -np.inf
        idx = np.argsort(-s, kind="stable")[:k]
        return idx, s[idx]


def _dtype_of(name: str) -> str:
    return {"int8": "i1", "fp16": "<f2", "fp32": "<f4"}[name]


def load_payload(directory: str | Path) -> Payload:
    """Read ``<directory>/data`` back (validating the manifest hashes and shapes) and decode it like the browser."""
    root = Path(directory)
    man = json.loads((root / "data" / "manifest.json").read_text(encoding="utf-8"))
    tracks = json.loads((root / man["tracks"]["path"]).read_text(encoding="utf-8"))
    out = Payload(man, tracks, {}, root)
    for m in man["models"]:
        n, d = m["n"], m["dim"]
        for key, e in m["files"].items():
            raw = (root / e["path"]).read_bytes()
            if len(raw) != e["bytes"] or hashlib.sha256(raw).hexdigest() != e["sha256"]:
                raise ValueError(f"{e['path']}: size/hash mismatch with manifest")
        f = {k: root / v["path"] for k, v in m["files"].items()}
        emb = np.fromfile(f["emb"], dtype=_dtype_of(m["dtype"]))
        if emb.size != n * d:
            raise ValueError(f"{f['emb']}: {emb.size} values, expected {n * d}")
        emb = emb.reshape(n, d).astype(np.float32)
        if m["dtype"] == "int8":
            emb = emb * np.fromfile(f["scale"], dtype="<f4")[:, None]
        xy = np.fromfile(f["umap"], dtype="<f4").reshape(n, 2)
        meta = json.loads(f["poles_meta"].read_text(encoding="utf-8"))
        poles = np.fromfile(f["poles"], dtype="<f4").reshape(len(meta), d)
        out.models[m["name"]] = ModelPayload(m["name"], unit(emb).astype(np.float32), xy, poles, meta,
                                             json.loads(f["modes"].read_text(encoding="utf-8")),
                                             np.fromfile(f["mode_idx"], dtype="<i2"))
    return out


# --------------------------------------------------------------------------------------------------------------------------
# serving
# --------------------------------------------------------------------------------------------------------------------------
class _Handler(http.server.SimpleHTTPRequestHandler):
    """Static handler with reliable MIME types (Windows registries often map .js to text/plain, which breaks ES modules)."""
    extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map, ".js": "text/javascript",
                      ".mjs": "text/javascript", ".json": "application/json", ".bin": "application/octet-stream",
                      ".css": "text/css", ".html": "text/html"}

    def log_message(self, format: str, *args: Any) -> None:            # noqa: A002 - stdlib signature
        log.debug("explorer http: " + format, *args)


def serve(directory: str | Path, port: int = 8000, host: str = "127.0.0.1", block: bool = True) -> http.server.ThreadingHTTPServer:
    """Serve ``directory`` over HTTP (the explorer needs http://, not file://, to fetch its payload).  With
    ``block=False`` the server runs on a daemon thread and is returned so the caller can ``shutdown()`` it."""
    handler = functools.partial(_Handler, directory=str(directory))
    httpd = http.server.ThreadingHTTPServer((host, port), handler)
    log.info("serving %s on http://%s:%d/", directory, host, httpd.server_address[1])
    if not block:
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        return httpd
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:                                          # pragma: no cover - interactive
        pass
    finally:
        httpd.server_close()
    return httpd


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m melomane.release.explorer", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build"); b.add_argument("run_dir"); b.add_argument("out_dir")
    b.add_argument("--models", nargs="*"); b.add_argument("--max-tracks", type=int, default=50000)
    b.add_argument("--dims", type=int, default=64); b.add_argument("--dtype", choices=["int8", "fp16"], default="int8")
    s = sub.add_parser("serve"); s.add_argument("directory"); s.add_argument("--port", type=int, default=8000)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if a.cmd == "build":
        man = build_explorer(a.run_dir, a.out_dir, a.models, a.max_tracks, a.dims, a.dtype)
        log.info("built %d tracks, %d models, %.1f MB payload", man["n"], len(man["models"]), man["total_bytes"] / 1e6)
    else:
        serve(a.directory, a.port)
    return 0


if __name__ == "__main__":                                              # pragma: no cover
    raise SystemExit(main())
