"""Keep what is committed to the public repository tidy: scrub machine-specific paths and curate the auto-written progress log.

``PROGRESS.md`` is appended by the pipeline on every stage run (including demo-corpus runs and superseded runs), which is useful on the
machine but noise in a public repository.  :func:`public_progress` produces the committed version: the header (with the "read this first"
note and open questions) plus only the entries of the current run, in order, with local paths replaced by ``<repo>``.  The working-tree file
is never modified.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence

_ENTRY = re.compile(r"^## (\d{4}-\d{2}-\d{2} \d{2}:\d{2}) UTC — .*? \[([^\]]+)\] — ", re.M)


def scrub_local_paths(text: str, root: str | Path) -> str:
    """Replace the absolute repository path (backslash, slash and JSON-escaped forms) with ``<repo>``."""
    r = str(root)
    forms = {r, r.replace("\\", "/"), r.replace("\\", "\\\\")}
    for f in sorted(forms, key=len, reverse=True):
        text = text.replace(f, "<repo>")
    return text


def public_progress(text: str, since: str, corpora: Sequence[str] = ("onion/standard",), root: str | Path = ".",
                    header: str | None = None) -> str:
    """Header + the entries whose corpus/profile tag is in ``corpora`` and whose UTC timestamp is >= ``since`` ("YYYY-MM-DD HH:MM").
    ``header`` replaces the log's own header (the committed one lives in docs/PROGRESS_HEADER.md); default: keep the log's header."""
    first = _ENTRY.search(text)
    if header is None:
        header = text if first is None else text[: first.start()]
    spans = [m for m in _ENTRY.finditer(text)]
    kept: list[str] = []
    for i, m in enumerate(spans):
        end = spans[i + 1].start() if i + 1 < len(spans) else len(text)
        stamp, tag = m.group(1), m.group(2)
        if tag in corpora and stamp >= since:
            kept.append(text[m.start():end].rstrip("\n") + "\n")
    out = header.rstrip("\n") + "\n\n" + "\n".join(kept)
    return scrub_local_paths(out, root)
