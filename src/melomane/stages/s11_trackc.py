"""S11 - Track C (no behaviour data): do the sonic compounds carry tag information on MTG-Jamendo / FMA?

Runs automatically when the raw files are present under data/raw/jamendo and/or data/raw/fma (no big downloads, no accounts needed to
*run* it - you place the files).  Results are flagged `release_ok: false` for Jamendo (metadata CC BY-NC-SA 4.0 conflicts with a
CC-BY release) so S12/S14 never bundle them."""
from __future__ import annotations
from pathlib import Path

import numpy as np

from ..pipeline import Ctx, Stage, StageResult, StageSkip, register, PARTIAL
from ..runtime import get_logger
from ..trackc import sonic_transfer

log = get_logger("melomane.s11")


def _first(d: Path, *pats: str):
    for p in pats:
        hit = sorted(d.glob(p)) if d.exists() else []
        if hit:
            return hit[0]
    return None


def _dense(blocks: dict, n: int) -> dict[str, np.ndarray]:
    return {k: b.dense(n) for k, b in blocks.items() if k != "heldout_echonest"}


def run_jamendo(ctx: Ctx, d: Path, cfg: dict) -> dict | None:
    from ..adapters import jamendo
    tsv = _first(d, "autotagging.tsv", "raw_30s_cleantags.tsv", "raw_30s.tsv", "*.tsv")
    feats = next((p for p in (d / "features", d / "acousticbrainz", *sorted(d.glob("*.tar*"))) if p.exists()), None)
    if tsv is None or feats is None:
        return None
    tr = jamendo.load_tracks(tsv)
    blocks, seen = jamendo.feature_blocks(tr, feats)
    if not blocks:
        return None
    tags = {c: jamendo.tag_matrix(tr, c) for c in jamendo.CATEGORIES}
    out = sonic_transfer(_dense(blocks, len(tr)), tags, tr["artist_id"].to_numpy(), k=cfg.get("k", 200), top_m=cfg.get("top_m", 3),
                         pca_dim=ctx.profile["compounds"]["pca_dim"], seed=ctx.seed("S11", "jamendo") % 10_000)
    out["alignment"] = jamendo.align_report(seen)
    out.update({"source": "MTG-Jamendo", "release_ok": False, "licence_note": "metadata CC BY-NC-SA 4.0 - non-commercial/share-alike; not bundled in any release"})
    return out


def run_fma(ctx: Ctx, d: Path, cfg: dict) -> dict | None:
    from ..adapters import fma
    tracks_csv, feat_csv = _first(d, "tracks.csv"), _first(d, "features.csv")
    if tracks_csv is None or feat_csv is None:
        return None
    tr = fma.load_tracks(tracks_csv, _first(d, "genres.csv"))
    blocks = fma.feature_blocks(tr, feat_csv, _first(d, "echonest.csv"))
    blocks = {k: v for k, v in blocks.items() if v.rows.size >= 500}
    if not blocks:
        return None
    tags = {"genre": fma.tag_matrix(tr, "tags")}
    out = sonic_transfer(_dense(blocks, len(tr)), tags, tr["artist_id"].to_numpy(), k=cfg.get("k", 200), top_m=cfg.get("top_m", 3),
                         pca_dim=ctx.profile["compounds"]["pca_dim"], seed=ctx.seed("S11", "fma") % 10_000)
    out.update({"source": "FMA (features are librosa/Echonest conceptual proxies, not Essentia)", "release_ok": True, "licence_note": "metadata CC BY 4.0"})
    return out


@register
class TrackC(Stage):
    name, title, scope = "S11", "Track C: sonic-compound transfer on Jamendo / FMA", "global"

    def params(self, ctx):
        return {"v": 1, "cfg": ctx.ws.config("data").get("trackc", {}), "compounds": ctx.profile["compounds"]}

    def outputs(self, ctx):
        return [ctx.ws.results / "S11_trackc.json"]

    def run(self, ctx: Ctx) -> StageResult:
        from ..runtime import jsonable, write_json
        cfg = ctx.ws.config("data").get("trackc", {})
        raw = ctx.ws.raw
        res, out = StageResult(), {}
        for name, fn, sub in (("jamendo", run_jamendo, cfg.get("jamendo_dir", "jamendo")), ("fma", run_fma, cfg.get("fma_dir", "fma"))):
            try:
                r = fn(ctx, raw / sub, cfg)
            except Exception as e:  # noqa: BLE001 - one source failing must not hide the other
                res.status = PARTIAL
                res.notes.append(f"{name} failed: {type(e).__name__}: {e}")
                continue
            if r is not None:
                out[name] = r
        if not out:
            raise StageSkip(f"no Track C raw files found under {raw}/{cfg.get('jamendo_dir', 'jamendo')} or /{cfg.get('fma_dir', 'fma')}")
        write_json(ctx.ws.results / "S11_trackc.json", jsonable(out))
        (ctx.ws.reports / "s11_trackc.md").write_text("# S11 - Track C\n\n" + "\n".join(
            f"- **{k}** ({v['source']}): {v['n_tracks']:,} tracks; " + "; ".join(f"{c}: AUC raw {d['mean_auc_raw']:.3f} vs compounds {d['mean_auc_compounds']:.3f} (retention {d['retention_of_auc_above_chance']:.2f})"
                                                                               for c, d in v["categories"].items()) for k, v in out.items()) + "\n", encoding="utf-8")
        res.metrics = {f"{k}_tracks": v["n_tracks"] for k, v in out.items()}
        return res
