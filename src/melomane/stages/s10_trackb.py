"""S10 - Track B (opt-in): MusicBrainz / ListenBrainz / AcousticBrainz / LRCLIB corpus -> Corpus contract (`mbid_stack`).

Off by default because the listens dump alone is ~230 GB.  `--with-track-b` runs it and adds the `mbid_stack` corpus to the run;
without the raw files it writes `reports/trackb_plan.md` (what to fetch, sizes, licences) and - only if `trackb.download: true` -
downloads the core files (resumable).  Killing the process anywhere and re-running continues."""
from __future__ import annotations
from pathlib import Path

from ..pipeline import Ctx, Stage, StageResult, StageSkip, register, PARTIAL
from ..runtime import disk_free_gb, get_logger

log = get_logger("melomane.s10")


def _find(raw: Path, *patterns: str) -> list[Path]:
    out: list[Path] = []
    for pat in patterns:
        out += sorted(raw.rglob(pat)) if raw.exists() else []
    return sorted(set(out))


def locate(raw: Path) -> dict:
    """Recognise the Track B raw files under `raw` by file name (dump names are date-stamped and rotate)."""
    return {"listens": [p for p in _find(raw, "*listens*.tar.zst", "*listens*.jsonl*") if "sample" not in p.name],
            "sample": _find(raw, "*sample*.tar.zst"),
            "canonical": (_find(raw, "musicbrainz-canonical-dump*.tar.zst") or [None])[0],
            "ab_csv": _find(raw, "acousticbrainz-lowlevel-features-*.tar.zst"),
            "ab_lowlevel": _find(raw, "acousticbrainz-lowlevel-json-*.tar.zst"),
            "lrclib_dump": (_find(raw, "lrclib*.sqlite3", "lrclib*.db") or [None])[0]}


@register
class TrackB(Stage):
    name, title, scope = "S10", "Track B corpus (MBID stack, opt-in)", "global"

    def params(self, ctx):
        return {"v": 1, "on": bool(ctx.opt("with_track_b")), "cfg": ctx.ws.config("data").get("trackb", {})}

    def outputs(self, ctx):
        if not ctx.opt("with_track_b"):
            return []
        return [ctx.ws.d("interim", ctx.ws.config("data").get("trackb", {}).get("name", "mbid_stack"), base="data") / "corpus.json"]

    def run(self, ctx: Ctx) -> StageResult:
        if not ctx.opt("with_track_b"):
            raise StageSkip("Track B is opt-in (~230 GB listens dump): run with --with-track-b")
        from ..adapters import mbid_stack as mb
        from .. import download
        ws, cfg = ctx.ws, ctx.ws.config("data").get("trackb", {})
        raw = ws.d("raw", cfg.get("raw_subdir", "trackb"), base="data")
        found = locate(raw)
        res = StageResult()
        if not found["listens"]:
            items = [i for i in mb.plan_downloads(ab_archives=cfg.get("ab_archives") or []) if i.tier in ("core", "recommended") and i.verified]
            try:
                mb.probe_sizes(items)
            except Exception as e:  # noqa: BLE001 - a size probe failing must not block planning
                log.warning("size probe failed: %s", e)
            total = mb.total_gb(items)
            lines = ["# Track B download plan", "", f"Total ~{total:.0f} GB; free on data drive {disk_free_gb(ws.data):.0f} GB. Files go to `{raw}`.", "",
                     "| key | tier | size GB | licence | purpose |", "|---|---|---|---|---|"]
            lines += [f"| {i.key} | {i.tier} | {((i.size_bytes or 0) / 2**30):.1f} | {i.licence} | {i.purpose} |" for i in items]
            (ws.reports / "trackb_plan.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
            if not cfg.get("download"):
                raise StageSkip(f"Track B raw files not found in {raw}. Plan (~{total:.0f} GB) written to reports/trackb_plan.md; set trackb.download: true "
                                "in configs/data.yaml to fetch them automatically, or place the files there yourself.")
            if disk_free_gb(ws.data) < total * 1.3:
                raise RuntimeError(f"Track B needs ~{total * 1.3:.0f} GB free; only {disk_free_gb(ws.data):.0f} GB")
            for it in items:
                if it.url.startswith("https://lrclib.net"):
                    continue                                     # JS page, no direct file: LRCLIB is fetched through the API client instead
                download.download(it.url, raw / Path(it.url).name, expected_size=it.size_bytes if it.verified and it.probed_bytes else None)
            found = locate(raw)
            res.decisions.append("Track B core files downloaded")
        tcfg = mb.TrackBConfig(listens=found["listens"], name=cfg.get("name", "mbid_stack"), canonical=found["canonical"], ab_lowlevel=found["ab_lowlevel"],
                               ab_csv=found["ab_csv"], lrclib_dump=found["lrclib_dump"], lrclib_api=bool(cfg.get("lrclib_api")) and found["lrclib_dump"] is None,
                               year_range=tuple(cfg["year_range"]) if cfg.get("year_range") else None, min_users_per_track=cfg.get("min_users_per_track", 5),
                               max_tracks=cfg.get("max_tracks"), seed=ctx.seed("S10") % 10_000)
        c = mb.ingest(ws, tcfg)
        if c is None:
            res.status = PARTIAL
            res.notes.append("Track B ingest is incomplete (lyric lookups pending or stopped): re-run to continue; cached work is kept")
            return res
        res.metrics = {"tracks": c.n_tracks, "sessions": int(c.sessions.shape[0]), "families": len(c.families)}
        res.notes.append("corpus 'mbid_stack' written; the per-corpus stages (S1-S9) run on it in this same run")
        return res
