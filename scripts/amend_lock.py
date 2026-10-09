"""Append-only amendment to analysis_plan.lock.json.

The original lock (created before any training) is never rewritten.  When a pre-registered config changes afterwards, run

    python scripts/amend_lock.py --reason "why" [--before-results]

to record the new SHA-256 of every locked config file, the date and the reason.  ``--before-results`` states that no result produced
under the new config had been seen when the change was made (the honest default for a re-run from scratch).
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "analysis_plan.lock.json"


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--reason", required=True)
    ap.add_argument("--before-results", action="store_true", help="no result under the new config had been seen")
    a = ap.parse_args()
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    cfgs = sorted(lock["config_hashes_sha256"])
    entry = {
        "at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "reason": a.reason,
        "no_results_seen_under_new_config": bool(a.before_results),
        "config_hashes_sha256": {c: sha256(ROOT / c) for c in cfgs},
        "changed": [c for c in cfgs if sha256(ROOT / c) != lock["config_hashes_sha256"][c]],
    }
    lock.setdefault("amendments", []).append(entry)
    LOCK.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(entry, indent=2))


if __name__ == "__main__":
    main()
