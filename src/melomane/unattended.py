"""The unattended tail of a long run: wait for a stage to finish, then run the remaining stages, with the machine held awake.

Nothing here depends on an interactive session: ``python -m melomane.unattended ...`` can be started once (detached) and left alone.
The LLM budget for the tail is capped to ``extra_usd`` above what the cache already records as spent, so a late LLM stage (mode naming)
can never spend more than the owner left on the account.
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Sequence

from .runtime import get_logger, keep_awake

log = get_logger("melomane.unattended")


def llm_budget_cap(cache_path: str | Path, extra_usd: float) -> float:
    """Cap (USD) that allows exactly ``extra_usd`` of further spend: what the cache records as spent plus ``extra_usd``."""
    p = Path(cache_path)
    spent = 0.0
    if p.exists():
        with sqlite3.connect(str(p)) as db:
            spent = float(db.execute("SELECT COALESCE(SUM(cost),0) FROM cache").fetchone()[0])
    return round(spent + float(extra_usd), 4)


def wait_for_file(path: str | Path, timeout_s: float | None = None, poll_s: float = 60.0,
                  sleep: Callable[[float], None] = time.sleep, now: Callable[[], float] = time.time) -> bool:
    """Block until ``path`` exists.  True when it does, False on timeout (``timeout_s=None`` waits forever)."""
    p, t0 = Path(path), now()
    while not p.exists():
        if timeout_s is not None and now() - t0 >= timeout_s:
            return False
        sleep(poll_s)
    return True


def build_tail_command(corpus: str, profile: str, stages: Sequence[str], python: str = sys.executable) -> list[str]:
    return [python, "-m", "melomane", "run-all", "--corpus", corpus, "--profile", profile, "--stages", ",".join(stages), "--no-prompt", "--no-awake"]


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m melomane.unattended", description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", default=".", help="repository root")
    ap.add_argument("--corpus", default="onion")
    ap.add_argument("--profile", default="standard")
    ap.add_argument("--wait-for", required=True, help="file whose appearance means the previous stage finished (e.g. data/runs/onion/standard/manifests/S5.json)")
    ap.add_argument("--stages", default="S6,S7,S8,S9,S12,S13,S14")
    ap.add_argument("--llm-extra-usd", type=float, default=1.5, help="LLM spend allowed in the tail, above what is already spent")
    ap.add_argument("--timeout-h", type=float, default=36.0)
    a = ap.parse_args(argv)
    root = Path(a.root).resolve()
    with keep_awake(True) as awake:
        log.info("keep-awake %s; waiting for %s", "ON" if awake else "unavailable", a.wait_for)
        if not wait_for_file(root / a.wait_for, a.timeout_h * 3600):
            log.error("timed out waiting for %s", a.wait_for)
            return 2
        env = dict(os.environ)
        env["LLM_BUDGET_USD"] = str(llm_budget_cap(root / "data" / "cache" / "llm.sqlite", a.llm_extra_usd))
        env.pop("SSLKEYLOGFILE", None)
        cmd = build_tail_command(a.corpus, a.profile, a.stages.split(","))
        log.info("running %s (LLM_BUDGET_USD=%s)", " ".join(cmd), env["LLM_BUDGET_USD"])
        return subprocess.run(cmd, cwd=root, env=env).returncode


if __name__ == "__main__":                                              # pragma: no cover
    raise SystemExit(main())
