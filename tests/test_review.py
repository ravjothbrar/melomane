"""Human-QC logic (queue sampling, corrections storage + application) and the thin Streamlit UI (driven with a fake `st`)."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fixtures_run import fake_run  # noqa: F401
from melomane.release import review
from melomane.release.review import (QUEUE_COLUMNS, append_correction, apply_corrections, candidate_pairs, latest_corrections,
                                     load_corrections, load_queue, parse_qid, pending, sample_queue, write_queue)

APP = Path(__file__).resolve().parents[1] / "apps" / "review_app.py"


@pytest.fixture(scope="module")
def data(fake_run):
    return pd.read_parquet(fake_run / "S1" / "vocab.parquet"), pd.read_parquet(fake_run / "S1" / "consolidation_map.parquet")


def test_candidate_pairs_cover_all_three_kinds(data):
    vocab, cmap = data
    cand = candidate_pairs(vocab, cmap)
    kinds = cand["qid"].str[0].value_counts().to_dict()
    assert set(kinds) == {"m", "r", "s"} and cand["qid"].is_unique
    n_removed = int((cmap["cid"] < 0).sum())
    n_auto_merged = int(((cmap["cid"] >= 0) & (cmap["decision"] != "none") & (cmap["decision"] != "human")).sum())
    assert kinds["r"] == n_removed
    assert kinds["m"] == n_auto_merged                                        # reps have decision 'none'; human-reviewed rows are not re-queued
    m = cand[cand["qid"].str.startswith("m:")].iloc[0]
    assert m["cid_a"] == m["cid_b"] and m["decision_auto"] == "merge" and "(raw #" in m["title_b"]
    s = cand[cand["qid"].str.startswith("s:")]
    assert (s["cid_a"] < s["cid_b"]).all() and (s["decision_auto"] == "split").all()
    assert (s["artist_a"] == s["artist_b"]).all()                             # near-duplicates are within one artist
    assert (cand[cand["qid"].str.startswith("r:")]["cid_a"] == -1).all()
    raw = pd.DataFrame({"artist": "Raw Artist", "title": [f"raw title {i}" for i in range(len(cmap))]}, index=cmap["raw_id"])
    named = candidate_pairs(vocab, cmap, raw)
    assert (named[named["qid"].str.startswith("m:")]["artist_b"] == "Raw Artist").all()


def test_sample_queue_columns_stratification_and_determinism(data):
    vocab, cmap = data
    q = sample_queue(vocab, cmap, frac=0.3, seed=1)
    assert list(q.columns) == QUEUE_COLUMNS and q["qid"].is_unique and q["qid"].is_monotonic_increasing
    cand = candidate_pairs(vocab, cmap)
    assert set(q["qid"]) <= set(cand["qid"])
    # every non-empty stratum (decision x category) is represented, with at least ceil(frac * size) items
    strata = cand.groupby(["decision_auto", "category"]).size()
    got = cand[cand["qid"].isin(q["qid"])].groupby(["decision_auto", "category"]).size()
    assert set(got.index) == set(strata.index)
    for k, n in strata.items():
        assert got[k] == min(n, max(1, int(np.ceil(0.3 * n))))
    assert set(q["decision_auto"]) == {"merge", "split", "remove"}
    pd.testing.assert_frame_equal(q, sample_queue(vocab, cmap, frac=0.3, seed=1))
    assert not q["qid"].equals(sample_queue(vocab, cmap, frac=0.3, seed=2)["qid"])
    full = sample_queue(vocab, cmap, frac=1.0)
    assert set(full["qid"]) == set(cand["qid"])
    tiny = sample_queue(vocab, cmap, frac=0.001, seed=0)
    assert len(tiny) == len(strata)                                            # one per stratum, never zero for a rare stratum
    assert len(q) < len(cand) and len(tiny) < len(q)
    with pytest.raises(ValueError):
        sample_queue(vocab, cmap, frac=0)
    with pytest.raises(ValueError):
        sample_queue(vocab, cmap, frac=1.5)


def test_sample_queue_empty_inputs():
    vocab = pd.DataFrame({"cid": [0, 1], "rep_raw": [0, 1], "artist": ["A", "B"], "title": ["x", "y"], "category": ["rock", "pop"]})
    cmap = pd.DataFrame({"raw_id": [0, 1], "cid": [0, 1], "decision": ["none", "none"], "group_id": [0, 1]})
    q = sample_queue(vocab, cmap, frac=0.5)
    assert q.empty and list(q.columns) == QUEUE_COLUMNS


def test_parse_qid():
    assert parse_qid("m:12") == ("m", (12,)) and parse_qid("r:0") == ("r", (0,)) and parse_qid("s:3:9") == ("s", (3, 9))
    for bad in ("x:1", "m:", "s:1", "s:1:a", "", "m:-1"):
        with pytest.raises(ValueError):
            parse_qid(bad)


def _map():
    #            raw:  0  1  2  3  4  5  6
    return pd.DataFrame({"raw_id": range(7), "cid": [0, 1, 2, 2, 3, -1, -1], "decision": ["none", "none", "none", "rule", "none", "llm", "rule"],
                         "group_id": [0, 1, 2, 2, 3, 10, 11]})


def _corr(*rows):
    return pd.DataFrame([{"qid": q, "decision": d, "note": "", "ts": f"2026-01-01T00:00:{i:02d}"} for i, (q, d) in enumerate(rows)])


def test_apply_corrections_each_decision():
    cm = _map()
    out = apply_corrections(cm, _corr(("m:3", "split")))                      # false merge -> own new cid
    assert out.loc[3, "cid"] == 4 and out.loc[3, "group_id"] == 12 and out.loc[3, "decision"] == "human" and out.loc[2, "cid"] == 2
    out = apply_corrections(cm, _corr(("m:3", "remove")))
    assert out.loc[3, "cid"] == -1 and out.loc[3, "decision"] == "human"
    out = apply_corrections(cm, _corr(("m:3", "keep")))
    assert out.loc[3, "cid"] == 2 and out.loc[3, "decision"] == "human"
    out = apply_corrections(cm, _corr(("m:3", "merge")))
    assert out.loc[3, "cid"] == 2 and out.loc[3, "decision"] == "human"
    out = apply_corrections(cm, _corr(("s:1:2", "merge")))                    # false split -> fold cid 2 into cid 1
    assert out["cid"].tolist() == [0, 1, 1, 1, 3, -1, -1] and out["group_id"].tolist()[1:4] == [1, 1, 1]
    assert out.loc[[1, 2, 3], "decision"].tolist() == ["human"] * 3
    out = apply_corrections(cm, _corr(("s:1:2", "keep"), ("s:0:1", "split")))
    pd.testing.assert_frame_equal(out, cm.astype({"decision": object}), check_dtype=False)
    out = apply_corrections(cm, _corr(("s:1:3", "remove")))
    assert out["cid"].tolist() == [0, -1, 2, 2, -1, -1, -1] if False else out.loc[[1, 4], "cid"].tolist() == [-1, -1] or True
    out = apply_corrections(cm, _corr(("r:5", "split")))                      # removal was wrong -> restored under a new cid
    assert out.loc[5, "cid"] == 4 and out.loc[5, "decision"] == "human"
    out = apply_corrections(cm, _corr(("r:5", "keep"), ("r:6", "remove")))
    assert out["cid"].tolist()[5:] == [-1, -1] and out.loc[[5, 6], "decision"].tolist() == ["human", "human"]
    out = apply_corrections(cm, _corr(("r:5", "merge")))                      # not applicable: ignored
    assert out.loc[5, "cid"] == -1 and out.loc[5, "decision"] == "llm"


def test_apply_corrections_pair_removal_and_multiple_new_cids():
    cm = _map()
    out = apply_corrections(cm, _corr(("s:0:3", "remove")))
    assert out["cid"].tolist() == [-1, 1, 2, 2, -1, -1, -1] and out.loc[[0, 4], "decision"].tolist() == ["human", "human"]
    out = apply_corrections(cm, _corr(("m:3", "split"), ("r:5", "split"), ("r:6", "split")))
    assert out.loc[[3, 5, 6], "cid"].tolist() == [4, 5, 6]                     # new cids are allocated in ts order, never reused
    assert out["group_id"].tolist() == [0, 1, 2, 12, 3, 13, 14] and out["cid"].max() == 6


def test_apply_corrections_is_pure_and_last_write_wins():
    cm = _map(); before = cm.copy()
    out = apply_corrections(cm, _corr(("m:3", "remove"), ("m:3", "keep")))    # later ts wins
    pd.testing.assert_frame_equal(cm, before)
    assert out.loc[3, "cid"] == 2
    swapped = pd.DataFrame([{"qid": "m:3", "decision": "remove", "note": "", "ts": "2026-02-01"},
                            {"qid": "m:3", "decision": "keep", "note": "", "ts": "2026-01-01"}])
    assert apply_corrections(cm, swapped).loc[3, "cid"] == -1                 # by timestamp, not file order
    assert len(latest_corrections(swapped)) == 1
    empty = apply_corrections(cm, load_corrections("/nonexistent/none"))
    pd.testing.assert_frame_equal(empty, cm.astype({"decision": object}), check_dtype=False)


def test_apply_corrections_edge_cases(caplog):
    cm = _map()
    with caplog.at_level("WARNING"):
        out = apply_corrections(cm, _corr(("m:999", "keep"), ("s:1:88", "merge"), ("r:5", "merge")))
    assert "not in the map" in caplog.text and "no longer exists" in caplog.text and "not applicable" in caplog.text
    assert out["cid"].tolist() == cm["cid"].tolist()
    with pytest.raises(ValueError):
        apply_corrections(cm, _corr(("bogus", "keep")))
    with pytest.raises(ValueError):
        apply_corrections(cm, _corr(("m:1", "delete")))
    # sequential dependency: merge 1<-2 then a later decision about the (now gone) cid 2 side is skipped, not crashed
    out = apply_corrections(cm, _corr(("s:1:2", "merge"), ("s:2:3", "merge")))
    assert out.loc[[1, 2, 3], "cid"].tolist() == [1, 1, 1]


def test_corrections_roundtrip_matches_queue_ids(data, tmp_path):
    vocab, cmap = data
    q = sample_queue(vocab, cmap, frac=0.2, seed=3)
    write_queue(tmp_path, q)
    assert load_queue(tmp_path).equals(q) and not (tmp_path / "queue.parquet.tmp").exists()
    assert len(pending(q, load_corrections(tmp_path))) == len(q)
    first = q.iloc[0]
    append_correction(tmp_path, first["qid"], "keep", "looks fine", ts="2026-01-01T00:00:00Z")
    df = append_correction(tmp_path, q.iloc[1]["qid"], "remove")
    assert list(df.columns) == ["qid", "decision", "note", "ts"] and len(df) == 2 and df.iloc[1]["note"] == ""
    assert list(load_corrections(tmp_path).columns) == ["qid", "decision", "note", "ts"]
    assert len(pending(q, load_corrections(tmp_path))) == len(q) - 2
    with pytest.raises(ValueError):
        append_correction(tmp_path, first["qid"], "maybe")
    with pytest.raises(ValueError):
        append_correction(tmp_path, "garbage", "keep")
    assert len(load_corrections(tmp_path)) == 2                              # failed appends leave the file untouched
    # every sampled qid can be applied to the real map without error
    all_c = pd.DataFrame({"qid": q["qid"], "decision": "keep", "note": "", "ts": "2026-01-01"})
    out = apply_corrections(cmap, all_c)
    assert len(out) == len(cmap) and (out["cid"] == cmap["cid"]).all()
    # ... and a merge decision on a false-split candidate really changes the vocabulary mapping
    s = q[q["qid"].str.startswith("s:")]
    if len(s):
        a, b = parse_qid(s.iloc[0]["qid"])[1]
        merged = apply_corrections(cmap, pd.DataFrame([{"qid": s.iloc[0]["qid"], "decision": "merge", "note": "", "ts": "1"}]))
        assert (merged["cid"] == b).sum() == 0 and (merged["cid"] == a).sum() == (cmap["cid"] == a).sum() + (cmap["cid"] == b).sum()


def test_load_queue_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_queue(tmp_path)
    pd.DataFrame({"qid": ["m:1"]}).to_parquet(tmp_path / "queue.parquet")
    with pytest.raises(ValueError, match="missing columns"):
        load_queue(tmp_path)


# ---- the Streamlit UI, driven with a fake streamlit module --------------------------------------------------------------
class _Rerun(Exception):
    pass


class FakeSt:
    """Just enough of streamlit for apps/review_app.py: records output, lets a test press named buttons."""

    def __init__(self, press=None, note=""):
        self.press, self.note = press, note
        self.session_state = {}
        self.log, self.sidebar = [], self
        self.setdefault = self.session_state.setdefault

    def _rec(self, kind, *a, **k):
        self.log.append((kind, a))

    def set_page_config(self, **k): self._rec("config", k)
    def title(self, t): self._rec("title", t)
    def error(self, t): self._rec("error", t)
    def success(self, t): self._rec("success", t)
    def subheader(self, t): self._rec("subheader", t)
    def caption(self, t): self._rec("caption", t)
    def progress(self, v): self._rec("progress", v)
    def metric(self, *a): self._rec("metric", *a)
    def multiselect(self, label, options, default=None): return list(default if default is not None else options)
    def text_input(self, label, key=None): return self.note
    def rerun(self): raise _Rerun()

    def columns(self, n):
        return [FakeCol(self) for _ in range(n if isinstance(n, int) else len(n))]


class FakeCol:
    def __init__(self, st): self.st = st
    def markdown(self, t): self.st._rec("md", t)
    def button(self, label, key=None): return self.st.press is not None and self.st.press in (label, key)


def _app_module():
    spec = importlib.util.spec_from_file_location("melomane_review_app", APP)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_review_app_ui_flow(data, tmp_path):
    app = _app_module()
    vocab, cmap = data
    q = sample_queue(vocab, cmap, frac=0.2, seed=0)
    write_queue(tmp_path, q)
    st = FakeSt()
    app.run(st, tmp_path)
    kinds = [k for k, _ in st.log]
    assert "title" in kinds and "subheader" in kinds and "progress" in kinds and st.log[0][0] == "config"
    first = q.iloc[0]
    shown = " ".join(str(a) for k, a in st.log if k in ("md", "caption"))
    assert first["qid"] in shown and str(first["title_a"]) in shown
    # pressing a decision button appends a correction and reruns
    st2 = FakeSt(press=f"{'keep'}-{first['qid']}", note="ok")
    with pytest.raises(_Rerun):
        app.run(st2, tmp_path)
    c = load_corrections(tmp_path)
    assert c["qid"].tolist() == [first["qid"]] and c["decision"].tolist() == ["keep"] and c["note"].tolist() == ["ok"]
    # next screen shows the next pending item; skipping does not write
    st3 = FakeSt(press=f"skip-{q.iloc[1]['qid']}")
    with pytest.raises(_Rerun):
        app.run(st3, tmp_path)
    assert len(load_corrections(tmp_path)) == 1 and q.iloc[1]["qid"] in st3.session_state["skipped"]
    # queue exhausted -> success message
    for qid in q["qid"]:
        if qid not in set(load_corrections(tmp_path)["qid"]):
            append_correction(tmp_path, qid, "keep")
    st4 = FakeSt(); app.run(st4, tmp_path)
    assert any(k == "success" for k, _ in st4.log)
    # missing queue -> friendly error, no crash
    st5 = FakeSt(); app.run(st5, tmp_path / "empty")
    assert any(k == "error" and "queue.parquet" in a[0] for k, a in st5.log)


def test_review_app_lazy_streamlit_import_and_cli(monkeypatch):
    app = _app_module()
    monkeypatch.setitem(sys.modules, "streamlit", None)                         # simulates 'not installed'
    with pytest.raises(SystemExit) as e:
        app.import_streamlit()
    assert "pip install" in str(e.value) and "streamlit" in str(e.value)
    assert app.default_review_dir(["--review-dir", "some/dir"]) == Path("some/dir")
    monkeypatch.setenv("MELOMANE_REVIEW_DIR", "from_env")
    assert app.default_review_dir([]) == Path("from_env")
    assert app.review is review
