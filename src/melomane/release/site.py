"""Build the static GitHub Pages site (``site/``) from the S12 explorer payload.

The site is the explorer app (``apps/explorer``) plus its ``data/`` payload, copied verbatim, with Pages housekeeping
(``.nojekyll``, a ``404.html`` that falls back to the app, ``BUILD.json``).  Nothing is rewritten: all asset paths in the
app are relative, so the same files work at ``https://<user>.github.io/melomane/`` and at ``https://<custom domain>/melomane/``.

Guards (each one raises :class:`SiteBuildError`; a site is never half-built):

* **Names gate.**  The payload carries artist / title strings.  Their source table is not part of the Zenodo release;
  until ``configs/publish.yaml`` says ``names_cleared: true`` (the owner has confirmed where the table came from and that the
  names may be republished) the build refuses, unless ``allow_uncleared=True`` is passed for a *private* preview.
* **Integrity.**  Every file in ``manifest.json`` must exist with the recorded size and sha256.
* **Privacy.**  The track table may only contain whitelisted columns (no lyrics, audio, user data).
* **Size.**  The payload must stay within ``max_payload_mb`` (Pages soft limit: 1 GB; we keep to the explorer's 50 MB budget).
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import shutil
from pathlib import Path

from .explorer import APP_FILES, default_app_dir

ALLOWED_TRACK_COLUMNS = {"n", "cid", "artist", "artist_idx", "title", "categories", "category_idx", "year"}


class SiteBuildError(RuntimeError):
    pass


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def load_publish_config(path: Path) -> dict:
    import yaml
    return (yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}).get("site", {})


def verify_payload(src: Path, max_payload_mb: float = 50.0) -> dict:
    """Check manifest hashes/sizes and the track-table whitelist; returns the manifest."""
    mp = src / "data" / "manifest.json"
    if not mp.exists():
        raise SiteBuildError(f"{mp} not found: build the explorer payload first (stage S12 / `python -m melomane.release.explorer build`)")
    man = json.loads(mp.read_text(encoding="utf-8"))
    total = 0
    files = [man["tracks"]] if "tracks" in man else []
    for m in man.get("models", []):
        files += list(m["files"].values())
    for f in files:
        p = src / f["path"]
        if not p.exists():
            raise SiteBuildError(f"manifest lists {f['path']} but it is missing")
        if p.stat().st_size != f["bytes"] or _sha256(p) != f["sha256"]:
            raise SiteBuildError(f"{f['path']} does not match the manifest (size/sha256): rebuild the payload")
        total += p.stat().st_size
    if total > max_payload_mb * 1e6:
        raise SiteBuildError(f"payload is {total / 1e6:.1f} MB > budget {max_payload_mb} MB")
    tracks = json.loads((src / man["tracks"]["path"]).read_text(encoding="utf-8"))
    extra = set(tracks) - ALLOWED_TRACK_COLUMNS
    if extra:
        raise SiteBuildError(f"track table has non-whitelisted columns {sorted(extra)} (lyrics/audio/user data must never be published)")
    man["_total_bytes"] = total
    return man


def build_site(src: str | Path, out: str | Path, *, names_cleared: bool, allow_uncleared: bool = False,
               app_dir: str | Path | None = None, base_path: str = "/melomane/", max_payload_mb: float = 50.0,
               git_commit: str = "") -> dict:
    src, out = Path(src), Path(out)
    if not names_cleared and not allow_uncleared:
        raise SiteBuildError(
            "refusing to build a public site: artist/title strings come from a metadata table whose origin and licence are not confirmed "
            "(configs/publish.yaml: names_cleared is false). Confirm the source, set names_cleared: true, or pass allow_uncleared for a private preview.")
    man = verify_payload(src, max_payload_mb)
    app = Path(app_dir) if app_dir else default_app_dir()
    if out.exists():
        shutil.rmtree(out)
    (out / "data").parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src / "data", out / "data")
    for name in APP_FILES:
        shutil.copy2(app / name, out / name)
    shutil.copy2(out / "index.html", out / "404.html")                       # unknown paths fall back to the app
    (out / ".nojekyll").write_text("", encoding="utf-8")                       # serve files as-is (no Jekyll processing)
    info = {"built_at": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "git_commit": git_commit, "base_path": base_path,
            "n_tracks": man.get("n"), "models": [m["name"] for m in man.get("models", [])], "payload_bytes": man["_total_bytes"],
            "names_cleared": bool(names_cleared), "private_preview": bool(allow_uncleared and not names_cleared)}
    (out / "BUILD.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info
