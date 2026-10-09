"""S12 - products: release bundles (full + `-small`), the static explorer, MCP-server config, model cards.

Only corpora listed in `release.corpora` (CC-BY-compatible) are bundled.  Everything is written under `<repo>/release/<corpus>/<profile>/`;
nothing is uploaded anywhere - publishing is Ravjoth's decision."""
from __future__ import annotations
import json
import sys
from pathlib import Path

from ..pipeline import Ctx, Stage, StageResult, StageSkip, register, PARTIAL
from ..runtime import get_logger, jsonable, read_json, write_json

log = get_logger("melomane.s12")
SOURCE_KEY = {"onion": "music4all-onion", "mbid_stack": "mbid-stack", "playlist2vec": "playlist2vec"}


def release_dir(ctx: Ctx, corpus: str) -> Path:
    return ctx.ws.d("release", corpus, ctx.profile_name)


def flat_metrics(ctx: Ctx, corpus: str) -> dict:
    """Per-model numbers for the model card, straight from results/ (never typed)."""
    p = ctx.ws.d("results", corpus, ctx.profile_name) / "S7_summary.json"
    if not p.exists():
        return {}
    summ = read_json(p).get("summary_by_model", {})
    return {m: {k.replace("|", "_"): v["mean"] for k, v in d.items() if isinstance(v, dict) and "mean" in v} for m, d in summ.items()}


def small_dims(ctx: Ctx, corpus: str, default: int) -> int:
    """Shipped `-small` dimensionality: the smallest E1 config keeping >= 95% (else 90%) of held-out rho, if E1 ran; else the configured default."""
    p = ctx.ws.d("results", corpus, ctx.profile_name) / "S9_extensions.json"
    if p.exists():
        e1 = read_json(p).get("E1", {})
        pick = (e1.get("ship_small") or {}).get("composition") or next((v for v in (e1.get("ship_small") or {}).values() if v), None)
        if pick and pick.get("dims"):
            return int(pick["dims"])
    return default


@register
class Products(Stage):
    name, title, scope = "S12", "Products (bundles, explorer, MCP config)", "global"

    def params(self, ctx):
        return {"v": 1, "release": ctx.ws.config("data").get("release", {}), "corpora": ctx.opt("corpora", [])}

    def outputs(self, ctx):
        return [ctx.ws.d("release") / "RELEASE_MANIFEST.json"]

    def run(self, ctx: Ctx) -> StageResult:
        from ..release.export import export_all, list_models
        from ..release.explorer import build_explorer
        cfg = ctx.ws.config("data").get("release", {})
        allowed = set(cfg.get("corpora", []))
        res, manifest = StageResult(), {"corpora": {}}
        todo = [c for c in ctx.opt("corpora", []) if c in allowed]
        skipped = [c for c in ctx.opt("corpora", []) if c not in allowed]
        if skipped:
            res.notes.append(f"not bundled (licence policy): {skipped}")
        for c in todo:
            run_dir = ctx.ws.run_dir(c, ctx.profile_name)
            if not list_models(run_dir):
                res.notes.append(f"{c}: no trained models yet - nothing to bundle")
                continue
            out = release_dir(ctx, c)
            entry = {}
            try:
                dims = small_dims(ctx, c, cfg.get("small_dims", 64))
                bundles = export_all(run_dir, out / "bundles", small_dims=dims, metrics=flat_metrics(ctx, c),
                                     sources=[SOURCE_KEY[c]], extra_config={"attribution": cfg.get("attribution", {}).get(c, ""), "profile": ctx.profile_name})
                entry["bundles"] = {k: str(v.relative_to(ctx.ws.root)) for k, v in bundles.items()}
                entry["small_dims"] = dims
            except Exception as e:  # noqa: BLE001
                res.status = PARTIAL
                res.notes.append(f"{c}: bundle export failed: {type(e).__name__}: {e}")
                log.exception("bundle export failed for %s", c)
            try:
                man = build_explorer(run_dir, out / "explorer", max_tracks=cfg.get("explorer_max_tracks", 50000), seed=ctx.seed("S12", c) % 10_000, sources=[SOURCE_KEY[c]])
                entry["explorer"] = {"dir": str((out / "explorer").relative_to(ctx.ws.root)), "n_tracks": man.get("n")}
            except Exception as e:  # noqa: BLE001
                res.status = PARTIAL
                res.notes.append(f"{c}: explorer build failed: {type(e).__name__}: {e}")
                log.exception("explorer build failed for %s", c)
            if entry.get("bundles"):
                srv = Path(__file__).resolve().parents[3] / "apps" / "mcp_server" / "server.py"
                args = [str(srv)] + [x for k, p in entry["bundles"].items() for x in ("--bundle", f"{k}={ctx.ws.root / p}")]
                write_json(out / "mcp_config.json", {"mcpServers": {f"melomane-{c}": {"command": sys.executable, "args": args}}})
                entry["mcp_config"] = str((out / "mcp_config.json").relative_to(ctx.ws.root))
            manifest["corpora"][c] = entry
        if not manifest["corpora"]:
            raise StageSkip("nothing to bundle: no releasable corpus has trained models (see notes)" if not todo else "no trained models in releasable corpora")
        write_json(ctx.ws.d("release") / "RELEASE_MANIFEST.json", jsonable(manifest))
        (ctx.ws.d("release") / "README.md").write_text(
            "# Melomane release artefacts\n\nGenerated by S12 - **not published**. Each folder holds full and `-small` bundles (safetensors + vocab + modes + model card), "
            "a static explorer (`explorer/index.html`, serve with `python -m melomane.release.explorer serve <dir>`) and an `mcp_config.json` for a local MCP client.\n"
            "No lyrics or audio are included anywhere. Licence/attribution per corpus is in each bundle's `config.json`.\n", encoding="utf-8")
        res.metrics = {"corpora": len(manifest["corpora"])}
        return res
