"""Stage the *public* version of PROGRESS.md in the git index (current run only, local paths scrubbed) without touching the live file.

    python scripts/stage_public_progress.py [--since "2026-10-09 15:00"]
    git commit ...

The pipeline keeps appending the full log to PROGRESS.md on this machine; what is committed is the curated version.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from melomane.hygiene import public_progress  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--since", default="2026-10-09 15:00", help="first UTC timestamp of the current run (older entries are superseded)")
    a = ap.parse_args()
    text = (ROOT / "PROGRESS.md").read_text(encoding="utf-8")
    header = (ROOT / "docs" / "PROGRESS_HEADER.md").read_text(encoding="utf-8")
    clean = public_progress(text, since=a.since, corpora=("onion/standard",), root=ROOT, header=header)
    tmp = ROOT / ".progress_public.tmp"
    tmp.write_text(clean, encoding="utf-8", newline="\n")
    try:
        sha = subprocess.run(["git", "hash-object", "-w", str(tmp)], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
        subprocess.run(["git", "update-index", "--add", "--cacheinfo", f"100644,{sha},PROGRESS.md"], cwd=ROOT, check=True)
    finally:
        tmp.unlink(missing_ok=True)
    print(f"staged public PROGRESS.md ({len(clean.splitlines())} lines; live file has {len(text.splitlines())})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
