"""Build the GitHub Pages site into ./site from the S12 explorer payload.

    python scripts/build_site.py                       # public build (refuses while configs/publish.yaml: names_cleared is false)
    python scripts/build_site.py --allow-uncleared     # private preview only; never publish its output
    python scripts/build_site.py --src release/onion/standard/explorer --out site
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from melomane.release.site import SiteBuildError, build_site, load_publish_config  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--src", default=str(ROOT / "release" / "onion" / "standard" / "explorer"))
    ap.add_argument("--out", default=str(ROOT / "site"))
    ap.add_argument("--allow-uncleared", action="store_true", help="private preview only")
    a = ap.parse_args()
    cfg = load_publish_config(ROOT / "configs" / "publish.yaml")
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    except OSError:
        commit = ""
    try:
        info = build_site(a.src, a.out, names_cleared=bool(cfg.get("names_cleared")), allow_uncleared=a.allow_uncleared,
                          app_dir=ROOT / "apps" / "explorer", base_path=cfg.get("base_path", "/melomane/"),
                          max_payload_mb=float(cfg.get("max_payload_mb", 50)), git_commit=commit)
    except SiteBuildError as e:
        print(f"BUILD REFUSED: {e}", file=sys.stderr)
        return 2
    print(f"built {a.out}: {info['n_tracks']} tracks, models {info['models']}, {info['payload_bytes'] / 1e6:.1f} MB"
          + ("  [PRIVATE PREVIEW - do not publish]" if info["private_preview"] else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
