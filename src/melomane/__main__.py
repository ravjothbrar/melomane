"""`python -m melomane ...` / `melomane ...` command-line interface."""
from __future__ import annotations
import argparse, os, sys
from pathlib import Path


def _ws(args):
    from .workspace import Workspace
    return Workspace.discover(getattr(args, "home", None))


def _corpora(args, default):
    c = getattr(args, "corpus", None)
    return [x.strip() for x in c.split(",") if x.strip()] if c else default


def cmd_run_all(args) -> int:
    from .pipeline import runner
    ws = _ws(args)
    ensure_key(ws, interactive=not args.no_prompt)
    corpora = _corpora(args, runner.DEFAULT_CORPORA)
    if args.with_track_b and "trackb" not in corpora:
        corpora.append("trackb")
    opts = {"skip_bench": args.skip_bench, "device": args.device}
    s = runner.run_all(ws, corpora, args.profile, opts, stages=args.stages.split(",") if args.stages else None, force=args.force, awake=not args.no_awake)
    return 1 if s["failed"] else 0


def cmd_demo(args) -> int:
    from .pipeline import runner
    ws = _ws(args)
    s = runner.run_all(ws, ["synth"], "smoke", {"demo_size": args.size, "skip_bench": False}, force=args.force, awake=False)
    return 1 if s["failed"] else 0


def cmd_run(args) -> int:
    from .pipeline import runner
    ws = _ws(args)
    from .runtime import setup_logging
    setup_logging(ws.logs / "run_stage.log")
    from .pipeline.framework import run_stages
    ctx = runner.make_ctx(ws, args.corpus, args.profile, {"device": args.device})
    out = run_stages(ctx, args.stage.split(","), force=args.force)
    return 1 if any(r["status"] == "FAILED" for r in out) else 0


def cmd_status(args) -> int:
    from .pipeline import runner
    ws = _ws(args)
    rows = runner.status(ws, _corpora(args, runner.DEFAULT_CORPORA + ["synth"]), args.profile)
    import pandas as pd
    print(pd.DataFrame(rows).to_string(index=False))
    return 0


def cmd_doctor(args) -> int:
    from .pipeline import runner
    ws = _ws(args)
    s = runner.run_all(ws, [], args.profile, {}, stages=["S0-doctor"], force=True, awake=False)
    print((ws.reports / "s0_doctor.md").read_text(encoding="utf-8"))
    return 0


def cmd_setup_key(args) -> int:
    ws = _ws(args)
    ensure_key(ws, interactive=True, force=True)
    return 0


def ensure_key(ws, interactive: bool = True, force: bool = False) -> None:
    """First-run convenience: ask once for the OpenRouter key and store it in .env (never printed, never logged)."""
    from .llm_factory import have_key
    from .workspace import set_env_key
    if have_key() and not force:
        return
    if not interactive or not sys.stdin or not sys.stdin.isatty():
        print("NOTE: OPENROUTER_API_KEY is not set - LLM stages will run in degraded (rule-only) mode. Add it to .env to enable them.")
        return
    import getpass
    print("\nMelomane uses OpenRouter (google/gemini-3.7-flash) for curation, annotation and mode naming.")
    print("Paste your OpenRouter API key (input hidden), or press Enter to continue without LLM stages:")
    try:
        k = getpass.getpass("OPENROUTER_API_KEY: ").strip()
    except (EOFError, KeyboardInterrupt):
        k = ""
    if k:
        set_env_key(ws.root / ".env", "OPENROUTER_API_KEY", k)
        os.environ["OPENROUTER_API_KEY"] = k
        print("Saved to .env (git-ignored).")


def cmd_serve(args) -> int:
    ws = _ws(args)
    from .release.explorer import serve
    d = Path(args.dir) if args.dir else ws.release / "explorer"
    serve(d, args.port)
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="melomane", description="Melomane: emergent geometry of music embeddings")
    p.add_argument("--home", help="repository root (default: auto-detect)")
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("run-all", help="run the whole pipeline (resumable)")
    a.add_argument("--profile", default="standard"); a.add_argument("--corpus", help="comma list; default onion,playlist2vec")
    a.add_argument("--stages", help="comma list of stage names to restrict to"); a.add_argument("--force", action="store_true")
    a.add_argument("--with-track-b", action="store_true"); a.add_argument("--skip-bench", action="store_true")
    a.add_argument("--no-prompt", action="store_true"); a.add_argument("--no-awake", action="store_true"); a.add_argument("--device")
    a.set_defaults(fn=cmd_run_all)
    d = sub.add_parser("demo", help="run everything on a small synthetic corpus (minutes) to verify the machine")
    d.add_argument("--size", type=int, default=2400); d.add_argument("--force", action="store_true"); d.set_defaults(fn=cmd_demo)
    r = sub.add_parser("run", help="run one stage"); r.add_argument("--stage", required=True); r.add_argument("--corpus", required=True)
    r.add_argument("--profile", default="standard"); r.add_argument("--force", action="store_true"); r.add_argument("--device"); r.set_defaults(fn=cmd_run)
    s = sub.add_parser("status", help="show stage status"); s.add_argument("--profile", default="standard"); s.add_argument("--corpus"); s.set_defaults(fn=cmd_status)
    dd = sub.add_parser("doctor", help="check the environment (GPU, packages, LLM key)"); dd.add_argument("--profile", default="smoke"); dd.set_defaults(fn=cmd_doctor)
    k = sub.add_parser("setup-key", help="store your OpenRouter key in .env"); k.set_defaults(fn=cmd_setup_key)
    e = sub.add_parser("serve", help="serve the explorer"); e.add_argument("--dir"); e.add_argument("--port", type=int, default=8765); e.set_defaults(fn=cmd_serve)
    args = p.parse_args(argv)
    return int(args.fn(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
