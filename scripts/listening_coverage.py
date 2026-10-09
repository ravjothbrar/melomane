"""How many of Music4All-Onion's tracks have listening data?  Streams userid_trackid_count and writes
results/onion/listening_coverage.json (the number quoted wherever the corpus size is explained)."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    raw = ROOT / "data" / "raw" / "onion"
    ids: set[str] = set()
    rows = 0
    for ch in pd.read_csv(raw / "userid_trackid_count.tsv.bz2", sep="\t", chunksize=5_000_000, dtype=str, usecols=["track_id"]):
        ids.update(ch["track_id"].unique())
        rows += len(ch)
    feat = pd.read_csv(raw / "id_mfcc_stats.tsv.bz2", sep="\t", usecols=[0], dtype=str).iloc[:, 0]
    out = {"listening_rows": rows, "tracks_with_listening_data": len(ids), "tracks_with_audio_features": int(feat.nunique()),
           "tracks_with_both": len(ids & set(feat)), "source": "userid_trackid_count.tsv.bz2, id_mfcc_stats.tsv.bz2 (Zenodo 15394646)"}
    dst = ROOT / "results" / "onion"
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "listening_coverage.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
