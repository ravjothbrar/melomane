"""Top-level orchestration: which stages run for which corpus, in what order, with failure isolation and a final summary."""
from __future__ import annotations
import json, time
from pathlib import Path

from ..runtime import get_logger, keep_awake, setup_logging, write_json
from ..workspace import Workspace
from .framework import Ctx, REGISTRY, run_stages, GREEN, PARTIAL, FAILED, BLOCKED, SKIPPED, is_done, manifest_path
from .. import stages  # noqa: F401  (registers stages)

log = get_logger("melomane.runner")

CORPUS_STAGES = ["S0-ingest", "S0-enrich", "S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9"]
GLOBAL_STAGES_FIRST = ["S0-doctor", "S10", "S11"]
GLOBAL_STAGES_LAST = ["S12", "S13", "S14"]
DEFAULT_CORPORA = ["onion", "playlist2vec"]


def make_ctx(ws: Workspace, corpus: str, profile: str, options: dict | None = None) -> Ctx:
    opts = dict(options or {})
    if corpus == "synth" and opts.get("llm") is None:
        from ..demo import DemoLLM
        opts["llm"] = DemoLLM()
    return Ctx(ws, corpus, profile, ws.profile(profile), opts)


def plan(corpora: list[str], stages: list[str] | None = None, include_global: bool = True) -> list[tuple[str, str]]:
    """[(scope_corpus, stage)] in execution order.  scope_corpus '' = global."""
    sel = set(stages) if stages else None
    out: list[tuple[str, str]] = []
    if include_global:
        out += [("", s) for s in GLOBAL_STAGES_FIRST if (sel is None or s in sel)]
    for c in corpora:
        out += [(c, s) for s in CORPUS_STAGES if (sel is None or s in sel)]
    if include_global:
        out += [("", s) for s in GLOBAL_STAGES_LAST if (sel is None or s in sel)]
    return out


def run_all(ws: Workspace, corpora: list[str], profile: str, options: dict | None = None, stages: list[str] | None = None,
            force: bool = False, awake: bool = True) -> dict:
    setup_logging(ws.logs / f"run_{time.strftime('%Y%m%d_%H%M%S')}.log")
    t0 = time.time()
    results: list[dict] = []
    if options and options.get("with_track_b") and "mbid_stack" not in corpora:
        corpora = list(corpora) + ["mbid_stack"]
    with keep_awake(awake) as ka:
        log.info("run-all: corpora=%s profile=%s keep_awake=%s", corpora, profile, ka)
        first = ("", "S0-doctor") in plan(corpora, stages)
        for scope, stage in plan(corpora, stages):
            if scope == "":
                ctx = make_ctx(ws, corpora[0] if corpora else "synth", profile, options)
                ctx.options["corpora"] = corpora
            else:
                ctx = make_ctx(ws, scope, profile, options)
            if stage not in REGISTRY:
                continue
            # a corpus stage is BLOCKED when an earlier stage of the same corpus failed
            blocked = [r for r in results if r["corpus"] == scope and r["status"] in (FAILED, BLOCKED) and _dep_related(stage, r["stage"])]
            if blocked:
                results.append({"corpus": scope, "stage": stage, "status": BLOCKED}); continue
            r = run_stages(ctx, [stage], force=force)
            for x in r:
                x["corpus"] = scope
            results += r
    summary = {"seconds": time.time() - t0, "results": results,
               "failed": [r for r in results if r["status"] == FAILED], "partial": [r for r in results if r["status"] == PARTIAL]}
    write_json(ws.results / "last_run.json", {k: v for k, v in summary.items()})
    _print_summary(results, summary["seconds"])
    return summary


def _dep_related(stage: str, upstream: str) -> bool:
    """True if `upstream` is a (transitive) dependency of `stage`."""
    seen, todo = set(), [stage]
    while todo:
        s = todo.pop()
        for d in (REGISTRY[s].deps if s in REGISTRY else ()):
            if d == upstream:
                return True
            if d not in seen:
                seen.add(d); todo.append(d)
    return False


def _print_summary(results: list[dict], seconds: float) -> None:
    log.info("=" * 70)
    for r in results:
        log.info("%-14s %-12s %s", r.get("corpus") or "(global)", r["stage"], r["status"])
    log.info("total %.1f min", seconds / 60)


def status(ws: Workspace, corpora: list[str], profile: str) -> list[dict]:
    rows = []
    for c in corpora:
        ctx = make_ctx(ws, c, profile)
        for s in CORPUS_STAGES + GLOBAL_STAGES_LAST:
            st = REGISTRY.get(s)
            if st is None:
                continue
            mp = manifest_path(ctx, st)
            if mp.exists():
                m = json.loads(mp.read_text(encoding="utf-8"))
                rows.append({"corpus": c, "stage": s, "status": m["status"], "seconds": round(m.get("seconds", 0)), "finished": m.get("finished", "")[:19],
                             "up_to_date": is_done(ctx, st)})
            else:
                rows.append({"corpus": c, "stage": s, "status": "-", "seconds": 0, "finished": "", "up_to_date": False})
    return rows
