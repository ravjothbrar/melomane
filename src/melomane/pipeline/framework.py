"""Stage framework: registry, hash-based idempotency, manifests, PROGRESS.md logging, failure isolation.

Every stage is idempotent: it is skipped when a finished manifest exists whose `params_hash` and dependency hashes
match and all declared outputs still exist.  A failing stage never aborts the whole run: it is recorded as FAILED,
its dependants are BLOCKED, everything independent continues, and the exit code reports the failure at the end.
"""
from __future__ import annotations
import datetime as dt
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from ..runtime import Timer, get_logger, jsonable, read_json, stable_hash, write_json, disk_free_gb
from ..workspace import Workspace

log = get_logger("melomane.pipeline")

GREEN, PARTIAL, FAILED, BLOCKED, SKIPPED = "GREEN", "PARTIAL", "FAILED", "BLOCKED", "SKIPPED"


class StageSkip(Exception):
    """Raised by a stage that legitimately cannot run (missing optional input).  Recorded as SKIPPED with a reason."""


@dataclass
class StageResult:
    status: str = GREEN
    metrics: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    decisions: list[str] = field(default_factory=list)
    deviations: list[str] = field(default_factory=list)
    questions: list[str] = field(default_factory=list)


@dataclass
class Ctx:
    ws: Workspace
    corpus: str
    profile_name: str
    profile: dict
    options: dict = field(default_factory=dict)

    @property
    def run_dir(self) -> Path:
        return self.ws.run_dir(self.corpus, self.profile_name)

    @property
    def corpus_dir(self) -> Path:
        return self.ws.d("interim", self.corpus, base="data")

    def stage_dir(self, name: str) -> Path:
        return self.ws.d("runs", self.corpus, self.profile_name, name, base="data")

    @property
    def results_dir(self) -> Path:
        return self.ws.d("results", self.corpus, self.profile_name)

    @property
    def reports_dir(self) -> Path:
        return self.ws.d("reports", self.corpus, self.profile_name)

    def save_result(self, name: str, obj) -> Path:
        p = self.results_dir / f"{name}.json"
        write_json(p, jsonable(obj))
        return p

    def load_result(self, name: str):
        return read_json(self.results_dir / f"{name}.json")

    def has_result(self, name: str) -> bool:
        return (self.results_dir / f"{name}.json").exists()

    def opt(self, key: str, default=None):
        return self.options.get(key, default)

    def seed(self, *parts) -> int:
        from ..runtime import derive_seed
        return derive_seed(self.corpus, self.profile_name, *parts)

    def llm(self):
        from ..llm_factory import make_llm
        return make_llm(self.ws)


class Stage:
    name: str = ""
    title: str = ""
    deps: tuple[str, ...] = ()
    scope: str = "corpus"          # "corpus" (per corpus × profile) or "global"
    requires_llm: bool = False

    def params(self, ctx: Ctx) -> dict:
        return {}

    def outputs(self, ctx: Ctx) -> list[Path]:
        return []

    def run(self, ctx: Ctx) -> StageResult:  # pragma: no cover - interface
        raise NotImplementedError


REGISTRY: dict[str, Stage] = {}


def register(cls):
    inst = cls()
    if not inst.name:
        raise ValueError("stage needs a name")
    REGISTRY[inst.name] = inst
    return cls


def manifest_path(ctx: Ctx, stage: Stage) -> Path:
    if stage.scope == "global":
        return ctx.ws.d("runs", "_global", "manifests", base="data") / f"{stage.name}.json"
    return ctx.run_dir / "manifests" / f"{stage.name}.json"


def _dep_hash(ctx: Ctx, dep: str) -> str | None:
    st = REGISTRY.get(dep)
    if st is None:
        return None
    mp = manifest_path(ctx, st)
    if not mp.exists():
        return None
    m = read_json(mp)
    return m.get("hash") if m.get("status") in (GREEN, PARTIAL) else None


def is_done(ctx: Ctx, stage: Stage) -> bool:
    mp = manifest_path(ctx, stage)
    if not mp.exists():
        return False
    m = read_json(mp)
    if m.get("status") not in (GREEN, PARTIAL):
        return False
    if not all(Path(p).exists() for p in stage.outputs(ctx)):
        return False
    if m.get("params_hash") != stable_hash({"params": stage.params(ctx), "profile": ctx.profile}):
        return False
    for d in stage.deps:
        if REGISTRY.get(d) is not None and d in m.get("dep_hashes", {}):
            cur = _dep_hash(ctx, d)
            if cur is not None and cur != m["dep_hashes"][d]:
                return False
    return True


def _topo(names: list[str]) -> list[str]:
    seen, order = set(), []

    def visit(n):
        if n in seen or n not in REGISTRY:
            return
        seen.add(n)
        for d in REGISTRY[n].deps:
            visit(d)
        order.append(n)
    for n in names:
        visit(n)
    return order


def append_progress(ws: Workspace, ctx: Ctx, stage: Stage, res: StageResult, seconds: float, resume_hint: str) -> None:
    p = ws.root / "PROGRESS.md"
    if not p.exists():
        p.write_text("# PROGRESS\n\nAuto-written by the Melomane pipeline (newest last).\n", encoding="utf-8")
    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    key = ", ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in list(res.metrics.items())[:8])
    lines = [f"\n## {ts} — {stage.name} ({stage.title}) [{ctx.corpus}/{ctx.profile_name}] — {res.status}",
             f"Ran in {seconds:.0f}s. Key numbers: {key or '-'}"]
    for label, items in (("Decisions", res.decisions), ("Deviations", res.deviations), ("Notes", res.notes),
                         ("Questions for Ravjoth", res.questions)):
        if items:
            lines.append(f"{label}: " + " | ".join(items))
    lines.append(f"Resume: `{resume_hint}` · disk free {disk_free_gb(ws.data):.0f} GB")
    with open(p, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def run_stage(ctx: Ctx, name: str, force: bool = False) -> dict:
    stage = REGISTRY[name]
    mp = manifest_path(ctx, stage)
    if not force and is_done(ctx, stage):
        log.info("[%s] up to date - skipped", name)
        return {"stage": name, "status": SKIPPED, "reason": "up-to-date"}
    log.info("[%s] %s - starting (%s/%s)", name, stage.title, ctx.corpus, ctx.profile_name)
    tm = Timer()
    status, res, err = FAILED, StageResult(status=FAILED), None
    try:
        with tm:
            res = stage.run(ctx)
            status = res.status
    except StageSkip as e:
        log.warning("[%s] skipped: %s", name, e)
        return {"stage": name, "status": SKIPPED, "reason": str(e)}
    except Exception as e:  # noqa: BLE001 - we isolate failures by design
        err = traceback.format_exc()
        log.error("[%s] FAILED: %s\n%s", name, e, err)
        res = StageResult(status=FAILED, notes=[f"{type(e).__name__}: {e}"])
        status = FAILED
        tm.seconds = getattr(tm, "seconds", 0.0)
    m = {"stage": name, "status": status, "params_hash": stable_hash({"params": stage.params(ctx), "profile": ctx.profile}),
         "dep_hashes": {d: _dep_hash(ctx, d) for d in stage.deps}, "seconds": getattr(tm, "seconds", 0.0),
         "finished": dt.datetime.now(dt.timezone.utc).isoformat(), "metrics": res.metrics, "notes": res.notes,
         "decisions": res.decisions, "deviations": res.deviations, "questions": res.questions,
         "outputs": [str(p) for p in stage.outputs(ctx)], "error": err}
    m["hash"] = stable_hash({k: m[k] for k in ("params_hash", "dep_hashes", "finished")})
    write_json(mp, jsonable(m))
    try:
        append_progress(ctx.ws, ctx, stage, res, m["seconds"],
                        f"python -m melomane run --stage {name} --corpus {ctx.corpus} --profile {ctx.profile_name}")
    except Exception:  # pragma: no cover
        pass
    log.info("[%s] %s in %.1fs", name, status, m["seconds"])
    return {"stage": name, "status": status, "seconds": m["seconds"], "error": err}


def run_stages(ctx: Ctx, names: list[str], force: bool = False, stop_on_fail: bool = False) -> list[dict]:
    """Run `names` (plus dependencies that are not yet done) in topological order with failure isolation."""
    order = _topo(names)
    out, failed = [], set()
    for n in order:
        st = REGISTRY[n]
        if st.scope == "global" and ctx.opt("_global_done", set()) and n in ctx.opt("_global_done"):
            continue
        if any(d in failed for d in st.deps):
            log.warning("[%s] BLOCKED by failed dependency", n)
            out.append({"stage": n, "status": BLOCKED}); failed.add(n)
            continue
        r = run_stage(ctx, n, force=force and n in names)
        out.append(r)
        if r["status"] in (FAILED,):
            failed.add(n)
            if stop_on_fail:
                break
    return out
