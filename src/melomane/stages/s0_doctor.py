"""S0-doctor: environment verification, GPU bench, OpenRouter smoke test, runtime estimate.  Never hard-fails on a warning."""
from __future__ import annotations
import importlib, os, platform, sys, time
from pathlib import Path
import numpy as np

from ..pipeline import Ctx, Stage, StageResult, register, GREEN, PARTIAL
from ..runtime import disk_free_gb, host_info, ram_gb, get_logger, write_json

log = get_logger("melomane.doctor")
REQUIRED = ["numpy", "scipy", "sklearn", "pandas", "pyarrow", "gensim", "torch", "yaml", "httpx", "jsonschema", "safetensors", "matplotlib"]
MIN_DISK_GB = {"onion": 60, "playlist2vec": 80, "default": 60}


def gpu_bench(n_nodes: int = 20000, dim: int = 300, seconds: float = 6.0) -> dict:
    """Measure SGNS pairs/s on the best device with a short synthetic run (what the runtime table is extrapolated from)."""
    import torch
    from ..torch_sgns import SGNS, _pairs, pick_device
    dev = pick_device()
    rng = np.random.default_rng(0)
    noise = torch.ones(n_nodes) / n_nodes
    m = SGNS(n_nodes, dim, noise).to(dev)
    opt = torch.optim.SparseAdam(list(m.parameters()), lr=0.0025)
    W = torch.from_numpy(rng.integers(0, n_nodes, (4096, 50)).astype(np.int64)).to(dev)
    batch = 1 << 18
    c, x = _pairs(W, 7)
    n_pairs, t0, steps = 0, time.time(), 0
    for _ in range(3):                                   # warm-up (cuda context, kernels)
        opt.zero_grad(); m.loss(c[:batch], x[:batch], 5).backward(); opt.step()
    if dev.type == "cuda": torch.cuda.synchronize()
    t0 = time.time()
    while time.time() - t0 < seconds:
        j = (steps * batch) % max(1, c.numel() - batch)
        opt.zero_grad(set_to_none=True)
        l = m.loss(c[j:j + batch], x[j:j + batch], 5); l.backward(); opt.step()
        n_pairs += min(batch, c.numel()); steps += 1
    if dev.type == "cuda": torch.cuda.synchronize()
    dt = time.time() - t0
    return {"device": str(dev), "pairs_per_s": n_pairs / dt, "steps": steps, "seconds": dt}


def cuda_selftest() -> tuple[bool, str]:
    try:
        import torch
        if not torch.cuda.is_available():
            return False, "torch.cuda.is_available() is False (CPU-only torch build, or driver too old)"
        x = torch.randn(512, 512, device="cuda")
        y = (x @ x).sum().item()
        cap = torch.cuda.get_device_capability(0)
        return True, f"{torch.cuda.get_device_name(0)} sm_{cap[0]}{cap[1]}; torch {torch.__version__} cuda {torch.version.cuda}"
    except Exception as e:
        return False, f"CUDA kernel launch failed: {type(e).__name__}: {e} (RTX 50-series needs a CUDA >= 12.8 PyTorch build: pip install torch --index-url https://download.pytorch.org/whl/cu128)"


def llm_smoke(ws) -> dict:
    from ..llm_factory import have_key, make_llm
    if not have_key():
        return {"status": "no_key"}
    try:
        llm = make_llm(ws)
        schema = {"type": "object", "additionalProperties": False, "required": ["answer"], "properties": {"answer": {"type": "integer"}}}
        t0 = time.time()
        r = llm.complete_json("Return JSON only.", 'What is 2+3? Reply as {"answer": <int>}.', schema, name="smoke")
        return {"status": "ok" if r.get("answer") == 5 else "wrong_answer", "seconds": time.time() - t0, "model": llm.model,
                "spent_usd": llm.spent(), "budget_usd": llm.budget}
    except Exception as e:
        return {"status": "error", "error": f"{type(e).__name__}: {str(e)[:300]}"}


@register
class Doctor(Stage):
    name, title, scope = "S0-doctor", "Environment, GPU, LLM smoke test", "global"

    def params(self, ctx):
        return {"v": 2, "have_key": bool(os.environ.get("OPENROUTER_API_KEY")), "skip_bench": ctx.opt("skip_bench", False)}

    def outputs(self, ctx):
        return [ctx.ws.results / "doctor.json"]

    def run(self, ctx: Ctx) -> StageResult:
        ws = ctx.ws
        res = StageResult()
        info = host_info()
        total, avail = ram_gb()
        info.update({"ram_total_gb": total, "ram_available_gb": avail, "disk_free_gb": disk_free_gb(ws.data), "data_dir": str(ws.data)})
        warns, checks = [], {}
        py = sys.version_info
        checks["python"] = f"{py.major}.{py.minor}.{py.micro}"
        if py < (3, 10):
            warns.append("Python >= 3.10 required")
        if py >= (3, 14):
            warns.append("Python 3.14 has few binary wheels yet; 3.12 is recommended")
        missing = []
        for m in REQUIRED:
            try:
                importlib.import_module(m)
            except Exception:
                missing.append(m)
        checks["missing_packages"] = missing
        if missing:
            warns.append(f"missing packages: {missing} (run run_melomane.bat again to reinstall)")
        ok, msg = cuda_selftest()
        checks["cuda"] = {"ok": ok, "detail": msg}
        if not ok:
            warns.append(f"GPU: {msg}. Falling back to CPU gensim training (much slower).")
        if info["disk_free_gb"] < MIN_DISK_GB["default"]:
            warns.append(f"only {info['disk_free_gb']:.0f} GB free on the data drive")
        if total == total and total < 24:
            warns.append(f"only {total:.0f} GB RAM: use the 'lite' profile / set max_vocab lower")
        checks["long_paths"] = {"root_len": len(str(ws.root)), "warn": len(str(ws.root)) > 120}
        if checks["long_paths"]["warn"]:
            warns.append("repository path is long; Windows path limits (260 chars) may bite - move the folder closer to C:\\")
        checks["llm"] = llm_smoke(ws)
        if checks["llm"]["status"] == "no_key":
            warns.append("OPENROUTER_API_KEY not set: LLM stages (curation, annotation, mode naming) will run in degraded rule-only mode")
        elif checks["llm"]["status"] != "ok":
            warns.append(f"LLM smoke test: {checks['llm']}")
        bench = None
        if not ctx.opt("skip_bench", False):
            try:
                bench = gpu_bench(seconds=3.0 if ctx.profile_name == "smoke" else 8.0)
                info["pairs_per_s"] = bench["pairs_per_s"]
            except Exception as e:
                warns.append(f"bench failed: {type(e).__name__}: {e}")
        est = {}
        if bench:
            pps = bench["pairs_per_s"]
            # standard profile @ 109k tracks, Core: 18 templates/round x rounds x epochs x V x L x window*2 pairs (approx)
            tr = ws.profile("standard")["train"]
            V = 109_000; L = tr["walk_length"]; pairs_per_walk = 2 * sum(max(L - o, 0) for o in range(1, tr["window"] + 1))
            for name, templ in (("cooc", 1), ("composition", 8), ("core", 18)):
                pairs = V * tr["walks_per_node"] * templ * tr["epochs"] * pairs_per_walk
                est[name + "_hours_standard_109k"] = pairs / pps / 3600
        write_json(ws.results / "doctor.json", {"host": info, "checks": checks, "bench": bench, "runtime_estimate": est, "warnings": warns})
        lines = ["# S0 doctor", "", f"Platform: {info['platform']}", f"Python {checks['python']} · torch {info.get('torch')} · GPU {info.get('gpu', 'none')}",
                 f"RAM {total:.0f} GB (avail {avail:.0f}) · disk free {info['disk_free_gb']:.0f} GB", ""]
        if bench:
            lines += [f"SGNS throughput: {bench['pairs_per_s'] / 1e6:.1f} M pairs/s on {bench['device']}", ""]
            lines += ["Estimated Core model time (standard profile, 109k tracks): " + ", ".join(f"{k}={v:.1f}h" for k, v in est.items()), ""]
        lines += ["## Warnings", *([f"- {w}" for w in warns] or ["- none"])]
        (ws.reports / "s0_doctor.md").write_text("\n".join(lines), encoding="utf-8")
        res.metrics = {"cuda": ok, "pairs_per_s": (bench or {}).get("pairs_per_s"), "disk_free_gb": info["disk_free_gb"], "llm": checks["llm"]["status"]}
        res.notes = warns[:6]
        res.status = PARTIAL if warns else GREEN
        return res
