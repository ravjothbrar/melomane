"""Runtime helpers: logging, keep-awake (Windows), disk, timers, hashing, JSON I/O.  No third-party deps."""
from __future__ import annotations
import contextlib, hashlib, json, logging, os, platform, shutil, sys, time
from pathlib import Path

import numpy as np


def get_logger(name: str = "melomane") -> logging.Logger:
    return logging.getLogger(name)


def setup_logging(log_file: Path | None = None, level: int = logging.INFO) -> None:
    root = logging.getLogger("melomane")
    root.setLevel(level)
    for h in list(root.handlers):
        root.removeHandler(h)
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    try:                                     # Windows consoles: never die on a stray unicode character
        sys.stdout.reconfigure(errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass
    root.addHandler(sh)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s"))
        root.addHandler(fh)


@contextlib.contextmanager
def keep_awake(enabled: bool = True):
    """Prevent Windows from sleeping while a long run is in progress (replaces the manual powercfg step).
    No-op on other platforms.  Display may still turn off; the system stays awake."""
    if not enabled or platform.system() != "Windows":
        yield False
        return
    import ctypes
    ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
    ok = False
    try:
        ok = bool(ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED))  # type: ignore[attr-defined]
    except Exception:
        pass
    try:
        yield ok
    finally:
        with contextlib.suppress(Exception):
            ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)  # type: ignore[attr-defined]


def disk_free_gb(path: Path) -> float:
    p = Path(path)
    while not p.exists() and p != p.parent:
        p = p.parent
    return shutil.disk_usage(p).free / 1e9


def sha256_file(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def md5_file(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def stable_hash(obj) -> str:
    """Hash of any JSON-able object (sorted keys, numpy scalars handled)."""
    def default(o):
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        if isinstance(o, Path): return str(o)
        if isinstance(o, set): return sorted(o)
        raise TypeError(type(o))
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=default).encode()).hexdigest()[:16]


def derive_seed(*parts) -> int:
    """Seed derived from (run_id, stage, name, ...) — PLAN §5.4."""
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


def write_json(path: Path, obj, indent: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    def default(o):
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return None if np.isnan(o) else float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        if isinstance(o, Path): return str(o)
        raise TypeError(type(o))
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=indent, default=default, allow_nan=False, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def jsonable(obj):
    """Recursively replace NaN/inf by None so results files are strict JSON."""
    if isinstance(obj, dict): return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray): return jsonable(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        f = float(obj); return None if not np.isfinite(f) else f
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, (np.bool_,)): return bool(obj)
    return obj


class Timer:
    def __enter__(self): self.t0 = time.time(); return self
    def __exit__(self, *a): self.seconds = time.time() - self.t0


def host_info() -> dict:
    info = {"platform": platform.platform(), "python": sys.version.split()[0], "cpus": os.cpu_count()}
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["gpu_capability"] = ".".join(map(str, torch.cuda.get_device_capability(0)))
    except Exception as e:  # pragma: no cover
        info["torch"] = f"unavailable ({type(e).__name__})"
    return info


def ram_gb() -> tuple[float, float]:
    """(total, available) GB without psutil (Windows ctypes / Linux /proc); (nan, nan) if unknown."""
    try:
        if platform.system() == "Windows":
            import ctypes

            class MS(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong), ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong), ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong), ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]
            ms = MS(); ms.dwLength = ctypes.sizeof(MS)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms))  # type: ignore[attr-defined]
            return ms.ullTotalPhys / 1e9, ms.ullAvailPhys / 1e9
        info = {}
        for ln in open("/proc/meminfo"):
            k, v = ln.split(":"); info[k] = float(v.split()[0]) * 1024
        return info["MemTotal"] / 1e9, info.get("MemAvailable", info["MemFree"]) / 1e9
    except Exception:
        return float("nan"), float("nan")
