"""Cross-platform and Windows-native bootstrap for Melomane.

Sets up Python 3.12, installs PyTorch with CUDA 12.8 wheels for the RTX 5080 (sm_120),
verifies GPU acceleration, checks OpenRouter key, and orchestrates pipeline execution.
Usage:
    python scripts/bootstrap.py auto      # smoke test -> status -> overnight train
    python scripts/bootstrap.py setup     # create .venv and install deps
    python scripts/bootstrap.py smoke     # run quick synthetic demo
    python scripts/bootstrap.py train     # run overnight standard training on Onion
    python scripts/bootstrap.py doctor    # run environment diagnostics
"""
from __future__ import annotations
import argparse, os, platform, subprocess, sys, time
from pathlib import Path

os.environ["UV_SYSTEM_CERTS"] = "1"

ROOT = Path(__file__).resolve().parents[1]
VENV_DIR = ROOT / ".venv"
IS_WIN = platform.system() == "Windows"
VENV_PYTHON = VENV_DIR / ("Scripts/python.exe" if IS_WIN else "bin/python")
VENV_PIP = VENV_DIR / ("Scripts/pip.exe" if IS_WIN else "bin/pip")


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run_cmd(args: list[str], check: bool = True, cwd: Path = ROOT) -> int:
    cmd_str = " ".join(str(a) for a in args)
    log(f"Running: {cmd_str}")
    p = subprocess.run(args, cwd=str(cwd))
    if check and p.returncode != 0:
        raise RuntimeError(f"Command failed with code {p.returncode}: {cmd_str}")
    return p.returncode


def find_uv() -> str | None:
    import shutil
    u = shutil.which("uv")
    if u:
        return u
    win_uv = Path.home() / ".local" / "bin" / "uv.exe"
    if win_uv.exists():
        return str(win_uv)
    return None


def ensure_venv():
    if VENV_PYTHON.exists():
        log(f"Virtual environment exists at {VENV_DIR}")
        return

    uv_path = find_uv()
    if uv_path:
        log("Creating virtual environment using uv (Python 3.12)...")
        try:
            run_cmd([uv_path, "venv", str(VENV_DIR), "--python", "3.12"])
            return
        except Exception as e:
            log(f"uv venv with Python 3.12 warning: {e}. Trying default python...")
            run_cmd([uv_path, "venv", str(VENV_DIR)])
            return

    log("Creating virtual environment using system python...")
    run_cmd([sys.executable, "-m", "venv", str(VENV_DIR)])


def install_dependencies():
    log("Checking PyTorch with CUDA 12.8 for RTX 5080 (Blackwell)...")
    check_code = "import torch; print(torch.__version__, torch.cuda.is_available())"
    p = subprocess.run([str(VENV_PYTHON), "-c", check_code], capture_output=True, text=True)
    needs_torch = True
    if p.returncode == 0 and "True" in p.stdout:
        log(f"PyTorch with CUDA already operational: {p.stdout.strip()}")
        needs_torch = False

    uv_path = find_uv()
    if needs_torch:
        log("Installing PyTorch with CUDA 12.8 wheel (supports RTX 5080 sm_120)...")
        if uv_path:
            run_cmd([uv_path, "pip", "install", "--python", str(VENV_PYTHON), "torch", "--index-url", "https://download.pytorch.org/whl/cu128"])
        else:
            run_cmd([str(VENV_PIP), "install", "torch", "--index-url", "https://download.pytorch.org/whl/cu128"])

    log("Installing Melomane and required dependencies...")
    if uv_path:
        run_cmd([uv_path, "pip", "install", "--python", str(VENV_PYTHON), "-e", f"{ROOT}[dev]"])
    else:
        run_cmd([str(VENV_PIP), "install", "-e", f"{ROOT}[dev]"])


def verify_cuda():
    test_code = (
        "import torch; dev = 'cuda' if torch.cuda.is_available() else 'cpu'; "
        "print(f'Device: {torch.cuda.get_device_name(0) if dev==\"cuda\" else \"CPU\"}'); "
        "x = torch.randn(1024, 1024, device=dev); print(f'Matrix multiply OK: {(x @ x).shape}')"
    )
    run_cmd([str(VENV_PYTHON), "-c", test_code])


def ensure_env_file():
    env_p = ROOT / ".env"
    if not env_p.exists():
        example_p = ROOT / ".env.example"
        if example_p.exists():
            import shutil
            shutil.copy(example_p, env_p)
            log("Created .env from .env.example")

    # Prompt for OpenRouter key if not already set and stdin is interactive
    has_key = False
    if env_p.exists():
        txt = env_p.read_text(encoding="utf-8")
        for line in txt.splitlines():
            if line.startswith("OPENROUTER_API_KEY=") and len(line.split("=", 1)[1].strip()) > 5:
                has_key = True
                break

    if not has_key and os.environ.get("OPENROUTER_API_KEY"):
        has_key = True

    if not has_key and sys.stdin and sys.stdin.isatty():
        import getpass
        print("\n" + "=" * 60)
        print("Optional: Melomane uses OpenRouter (Gemini Flash) for LLM curation & mode naming.")
        print("Your budget is hard-capped at $30 USD in configs/llm.yaml.")
        print("Press Enter to skip (offline rule-based mode) or paste your OpenRouter key:")
        try:
            key = getpass.getpass("OPENROUTER_API_KEY (hidden): ").strip()
        except Exception:
            key = ""
        if key:
            lines = env_p.read_text(encoding="utf-8").splitlines() if env_p.exists() else []
            new_lines = []
            found = False
            for l in lines:
                if l.startswith("OPENROUTER_API_KEY="):
                    new_lines.append(f"OPENROUTER_API_KEY={key}")
                    found = True
                else:
                    new_lines.append(l)
            if not found:
                new_lines.append(f"OPENROUTER_API_KEY={key}")
            env_p.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
            os.environ["OPENROUTER_API_KEY"] = key
            log("Saved OPENROUTER_API_KEY to .env")
        else:
            log("No key entered: running in offline rule-based mode.")


def cmd_setup():
    ensure_venv()
    install_dependencies()
    verify_cuda()
    ensure_env_file()
    log("Environment setup complete.")


def cmd_smoke():
    log("Running smoke / demo verification on synthetic corpus...")
    cmd = [str(VENV_PYTHON), "-m", "melomane", "demo", "--size", "600", "--force"]
    run_cmd(cmd)
    log("Smoke test completed successfully!")


def cmd_status():
    log("Checking pipeline status...")
    cmd = [str(VENV_PYTHON), "-m", "melomane", "status", "--corpus", "onion", "--profile", "standard"]
    run_cmd(cmd, check=False)


def cmd_doctor():
    log("Running environment doctor diagnostics...")
    cmd = [str(VENV_PYTHON), "-m", "melomane", "doctor"]
    run_cmd(cmd, check=False)


def cmd_train(corpus="onion", profile="standard"):
    log(f"Launching full training on {corpus} (profile: {profile})...")
    cmd = [str(VENV_PYTHON), "-m", "melomane", "run-all", "--corpus", corpus, "--profile", profile]
    run_cmd(cmd)
    log("Training completed successfully!")


def cmd_auto(corpus="onion", profile="standard"):
    log("=" * 70)
    log("MELOMANE AUTO LAUNCHER: Setup -> Smoke Test -> Status -> Overnight Training")
    log("=" * 70)
    try:
        cmd_setup()
        log("Step 1: Running fast smoke / demo test...")
        cmd_smoke()
        log("Step 2: Checking system status...")
        cmd_status()
        log(f"Step 3: Launching full overnight standard training on {corpus} ({profile})...")
        cmd_train(corpus=corpus, profile=profile)
        log("ALL STAGES FINISHED SUCCESSFULLY!")
    except Exception as e:
        log(f"ERROR ENCOUNTERED: {e}")
        log("Running doctor diagnostics to report environment status...")
        cmd_doctor()
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Melomane Bootstrap & Launcher")
    parser.add_argument("command", choices=["setup", "smoke", "status", "doctor", "train", "auto"], default="auto", nargs="?")
    parser.add_argument("--corpus", default="onion")
    parser.add_argument("--profile", default="standard")
    args = parser.parse_args()

    if args.command == "setup":
        cmd_setup()
    elif args.command == "smoke":
        cmd_smoke()
    elif args.command == "status":
        cmd_status()
    elif args.command == "doctor":
        cmd_doctor()
    elif args.command == "train":
        cmd_train(corpus=args.corpus, profile=args.profile)
    elif args.command == "auto":
        cmd_auto(corpus=args.corpus, profile=args.profile)


if __name__ == "__main__":
    main()
