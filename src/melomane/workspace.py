"""Workspace layout, .env loading and config access.  Windows-safe (pathlib only, no shell-isms)."""
from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path
import yaml


def load_dotenv(path: Path, override: bool = False) -> dict[str, str]:
    """Minimal .env parser (KEY=VALUE, '#' comments, optional quotes).  Returns what it read; sets os.environ
    for keys not already set (unless override).  Never logs values."""
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
            v = v[1:-1]
        elif " #" in v:
            v = v.split(" #", 1)[0].strip()
        out[k] = v
        if override or k not in os.environ:
            os.environ[k] = v
    return out


def set_env_key(path: Path, key: str, value: str) -> None:
    """Insert/replace KEY=value in a .env file (used by the first-run key prompt)."""
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    done = False
    for i, ln in enumerate(lines):
        if ln.strip().startswith(key + "="):
            lines[i] = f"{key}={value}"; done = True
    if not done:
        lines.append(f"{key}={value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def deep_merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


@dataclass
class Workspace:
    root: Path            # repository root (contains configs/, src/)
    data: Path            # big-data directory (may be on another drive)

    # ---- discovery ------------------------------------------------------
    @classmethod
    def discover(cls, root: str | Path | None = None) -> "Workspace":
        if root is None:
            root = os.environ.get("MELOMANE_HOME")
        if root is None:
            p = Path.cwd().resolve()
            for cand in [p, *p.parents]:
                if (cand / "configs" / "profiles.yaml").exists():
                    root = cand
                    break
            else:
                root = Path(__file__).resolve().parents[2]
        root = Path(root).resolve()
        load_dotenv(root / ".env")
        data = Path(os.environ.get("MELOMANE_DATA_DIR") or (root / "data")).expanduser().resolve()
        return cls(root=root, data=data)

    # ---- directories ----------------------------------------------------
    def d(self, *parts: str, base: str = "root", create: bool = True) -> Path:
        p = (self.data if base == "data" else self.root).joinpath(*parts)
        if create:
            p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def raw(self) -> Path: return self.d("raw", base="data")
    @property
    def interim(self) -> Path: return self.d("interim", base="data")
    @property
    def processed(self) -> Path: return self.d("processed", base="data")
    @property
    def cache(self) -> Path: return self.d("cache", base="data")
    @property
    def runs(self) -> Path: return self.d("runs", base="data")
    @property
    def results(self) -> Path: return self.d("results")
    @property
    def reports(self) -> Path: return self.d("reports")
    @property
    def logs(self) -> Path: return self.d("logs")
    @property
    def review(self) -> Path: return self.d("review")
    @property
    def release(self) -> Path: return self.d("release")

    def run_dir(self, corpus: str, profile: str) -> Path:
        return self.d("runs", corpus, profile, base="data")

    # ---- config ---------------------------------------------------------
    def profiles(self) -> dict:
        return yaml.safe_load((self.root / "configs" / "profiles.yaml").read_text(encoding="utf-8"))

    def profile(self, name: str) -> dict:
        """`_base` deep-merged with the named profile (nested dicts merge key by key)."""
        allp = self.profiles()
        if name not in allp or name == "_base":
            raise KeyError(f"unknown profile {name!r}; have {[k for k in allp if k != '_base']}")
        return deep_merge(allp["_base"], allp[name] or {})

    def config(self, name: str) -> dict:
        p = self.root / "configs" / f"{name}.yaml"
        return yaml.safe_load(p.read_text(encoding="utf-8")) if p.exists() else {}
