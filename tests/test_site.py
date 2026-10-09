"""GitHub Pages site builder: integrity, privacy, and the names gate."""
import json
import shutil

import pytest

from fixtures_run import fake_run  # noqa: F401  (session fixture)
from melomane.release.explorer import APP_FILES, build_explorer, default_app_dir
from melomane.release.site import SiteBuildError, build_site, verify_payload


@pytest.fixture(scope="module")
def payload(fake_run, tmp_path_factory):
    out = tmp_path_factory.mktemp("payload")
    build_explorer(fake_run, out, dims=16, seed=0)
    return out


def test_refuses_a_public_build_while_names_are_not_cleared(payload, tmp_path):
    with pytest.raises(SiteBuildError, match="names_cleared"):
        build_site(payload, tmp_path / "site", names_cleared=False)
    assert not (tmp_path / "site").exists()                                    # nothing half-built


def test_builds_a_complete_self_contained_site(payload, tmp_path):
    info = build_site(payload, tmp_path / "site", names_cleared=True, git_commit="abc1234")
    site = tmp_path / "site"
    for name in APP_FILES:
        assert (site / name).exists()
    assert (site / ".nojekyll").exists() and (site / "404.html").read_text(encoding="utf-8") == (site / "index.html").read_text(encoding="utf-8")
    assert (site / "data" / "manifest.json").exists() and info["names_cleared"] and not info["private_preview"]
    assert json.loads((site / "BUILD.json").read_text())["git_commit"] == "abc1234"
    # every URL the app fetches is relative, so the same files work under /melomane/
    js = (site / "explorer.js").read_text(encoding="utf-8")
    import re
    assert "fetchFn(base + p)" in js                                            # payload URLs are base-relative ...
    assert not re.search(r"""(src|href)=["']/(?!/)""", (site / "index.html").read_text(encoding="utf-8"))   # ... and no root-absolute src/href


def test_private_preview_is_flagged_in_the_build_info(payload, tmp_path):
    info = build_site(payload, tmp_path / "site", names_cleared=False, allow_uncleared=True)
    assert info["private_preview"] is True


def test_detects_a_tampered_payload(payload, tmp_path):
    bad = tmp_path / "bad"
    shutil.copytree(payload, bad)
    man = json.loads((bad / "data" / "manifest.json").read_text())
    victim = bad / man["models"][0]["files"]["emb"]["path"]
    victim.write_bytes(victim.read_bytes()[:-1] + b"\x00")                      # same size, different content
    with pytest.raises(SiteBuildError, match="does not match the manifest"):
        verify_payload(bad)


def test_rejects_non_whitelisted_track_columns(payload, tmp_path):
    bad = tmp_path / "bad"
    shutil.copytree(payload, bad)
    man = json.loads((bad / "data" / "manifest.json").read_text())
    tp = bad / man["tracks"]["path"]
    t = json.loads(tp.read_text(encoding="utf-8")); t["lyrics"] = ["secret"] * t["n"]
    raw = json.dumps(t).encode("utf-8"); tp.write_bytes(raw)
    import hashlib
    man["tracks"]["bytes"], man["tracks"]["sha256"] = len(raw), hashlib.sha256(raw).hexdigest()
    (bad / "data" / "manifest.json").write_text(json.dumps(man))
    with pytest.raises(SiteBuildError, match="non-whitelisted"):
        verify_payload(bad)


def test_missing_payload_gives_a_helpful_error(tmp_path):
    with pytest.raises(SiteBuildError, match="build the explorer payload"):
        build_site(tmp_path, tmp_path / "site", names_cleared=True)
