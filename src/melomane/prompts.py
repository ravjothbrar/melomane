"""Versioned prompt loader.  Prompts live in `prompts/*.md` with a first-line header `<!-- version: name_vN -->`.
The version string is stored with every LLM-derived output; changing a prompt changes the cache key of only its calls."""
from __future__ import annotations
import re
from pathlib import Path

_HDR = re.compile(r"^<!--\s*version:\s*([A-Za-z0-9_.\-]+)\s*-->\s*\n")


def load_prompt(root: Path, name: str) -> tuple[str, str]:
    """Returns (version, text without header)."""
    txt = (Path(root) / "prompts" / f"{name}.md").read_text(encoding="utf-8")
    m = _HDR.match(txt)
    if not m:
        raise ValueError(f"prompt {name} lacks a version header")
    return m.group(1), txt[m.end():].strip()
