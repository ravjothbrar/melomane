"""Render the README hero card (docs/assets/hero.png) from the built site payload and the run's vocabulary.

    python scripts/build_hero.py [--site site] [--corpus onion] [--profile standard] [--out docs/assets/hero.png]

Every number on the card is read from a file: track and mode counts from the payload manifest, the artist count from the run's vocabulary.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from melomane.release.preview import render_hero

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--site", default="site")
    ap.add_argument("--corpus", default="onion")
    ap.add_argument("--profile", default="standard")
    ap.add_argument("--out", default="docs/assets/hero.png")
    a = ap.parse_args()

    site = ROOT / a.site
    man = json.loads((site / "data" / "manifest.json").read_text(encoding="utf-8"))
    stats = [(f"{man['n_vocab']:,}", "tracks")]
    vocab = ROOT / "data" / "runs" / a.corpus / a.profile / "S1" / "vocab.parquet"
    if vocab.exists():
        import pandas as pd
        df = pd.read_parquet(vocab)
        named = df[df["artist_id"] >= 0] if "artist_id" in df else df.iloc[0:0]
        if len(named):
            stats.append((f"{int(named['artist_id'].nunique()):,}", "artists"))
    core = next(m for m in man["models"] if m["name"] == "core")
    stats += [("3", "models, one corpus"), (str(core["n_modes"]), "emergent modes per model")]
    out = render_hero(site, ROOT / a.out, stats)
    print(out)


if __name__ == "__main__":
    main()
