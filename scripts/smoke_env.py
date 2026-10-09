"""Stage-0 environment smoke test. Prints a PASS/WARN/FAIL table; exits non-zero only on FAIL.
Checks: python, numpy/scipy/sklearn/gensim, torch + CUDA (RTX 50-series needs a CUDA>=12.8 build, compute capability 12.0),
essentia (Linux/WSL2 only), disk space, OpenRouter reachability + a 1-token structured-output call (skipped w/o key)."""
import importlib, os, shutil, sys, json

rows = []
def row(name, status, detail=""): rows.append((name, status, detail)); print(f"[{status:4}] {name}: {detail}")

row("python", "PASS" if sys.version_info >= (3, 10) else "FAIL", sys.version.split()[0])
for m in ["numpy", "scipy", "sklearn", "pandas", "pyarrow", "gensim", "httpx", "jsonschema", "yaml"]:
    try: row(m, "PASS", getattr(importlib.import_module(m), "__version__", "ok"))
    except Exception as e: row(m, "FAIL", repr(e))
try:
    import torch
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability(); name = torch.cuda.get_device_name()
        x = torch.randn(1024, 1024, device="cuda"); ok = bool(torch.isfinite((x @ x).sum()))
        row("torch+cuda", "PASS" if ok else "FAIL", f"{torch.__version__} {name} sm_{cap[0]}{cap[1]} matmul_ok={ok}")
        if cap >= (12, 0) and not (torch.version.cuda and tuple(map(int, torch.version.cuda.split(".")[:2])) >= (12, 8)):
            row("torch-cuda-version", "FAIL", f"RTX 50-series needs CUDA>=12.8 build, have {torch.version.cuda}")
    else: row("torch+cuda", "WARN", f"{torch.__version__} CPU only -> use gensim backend (melomane.embed)")
except Exception as e: row("torch", "WARN", repr(e))
try:
    import essentia; row("essentia", "PASS", essentia.__version__)
except Exception as e: row("essentia", "WARN", "not importable (Windows needs WSL2; only required for Track-B fallback) " + repr(e)[:80])
free = shutil.disk_usage(os.environ.get("MELOMANE_DATA_DIR", ".")).free / 1e9
row("disk_free_GB", "PASS" if free > 200 else "WARN", f"{free:.0f}")
key = os.environ.get("OPENROUTER_API_KEY")
if not key: row("openrouter", "WARN", "OPENROUTER_API_KEY not set (LLM stages will use cached answers only)")
else:
    try:
        from melomane.llm import LLMClient
        c = LLMClient(api_key=key, model=os.environ.get("LLM_MODEL", "google/gemini-3.7-flash"), cache_path="cache/smoke.sqlite", budget_usd=1.0)
        out = c.complete_json("Reply with JSON only.", 'Return {"ok": true}.', {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"], "additionalProperties": False})
        row("openrouter", "PASS", f"structured output ok={out} spent=${c.spent():.5f}")
    except Exception as e: row("openrouter", "FAIL", repr(e)[:200])
sys.exit(1 if any(s == "FAIL" for _, s, _ in rows) else 0)
