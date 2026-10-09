import shutil
from pathlib import Path
from melomane.workspace import Workspace

REPO = Path(__file__).resolve().parents[1]


def make_ws(tmp_path: Path) -> Workspace:
    root = tmp_path / "repo"
    root.mkdir()
    shutil.copytree(REPO / "configs", root / "configs")
    shutil.copytree(REPO / "prompts", root / "prompts")
    for f in ("essentia_descriptor_names.json",):
        if (REPO / f).exists():
            shutil.copy(REPO / f, root / f)
    return Workspace(root=root, data=root / "data")
