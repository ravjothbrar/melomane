import numpy as np, pandas as pd, pytest
from melomane.curation_pipeline import curate, CurationConfig, audit_merges, _reconcile
from melomane.prompts import load_prompt
from pathlib import Path
from fake_llm import FakeLLM

ROOT = Path(__file__).resolve().parents[1]
PROMPTS = {k: load_prompt(ROOT, f"{k}_v1") for k in ("refine", "ultra", "audit")}


def make_tracks():
    rows = [("The Beatles", "Yesterday"), ("The Beatles", "Yesterday - Remastered 2009"), ("The Beatles", "Yesterday (Live)"),
            ("Beatles, The", "Yesterday"), ("Adele", "Hello"), ("Adele", "Hello (Radio Edit)"), ("Adele", "Hello (Remix)"),
            ("Nature Sounds", "Rain Sounds for Sleep"), ("Sleepy", "White Noise 10 hours"), ("Queen", "Bohemian Rhapsody"),
            ("Queen", "Bohemian Rhapsody (Remastered 2011)"), ("Rare Guy", "Obscure Song"), ("Someone", "Another One")]
    df = pd.DataFrame(rows, columns=["artist", "title"]); df["album"] = ""; df["tags"] = ""
    return df


def test_reconcile_enforces_exactly_once():
    rm, gr = _reconcile(5, [1, 1, 9], [[0, 1, 2], [2, 3]])
    assert rm == {1} and sorted(sum(gr, [])) == [0, 2, 3, 4]


def test_curate_with_llm_merges_and_removes():
    t = make_tracks(); ns = np.array([50, 40, 45, 30, 60, 55, 35, 90, 80, 70, 65, 10, 25])
    r = curate(t, ns, CurationConfig(min_sessions=20, level="strict"), FakeLLM(), PROMPTS)
    cm = r.cmap
    # removed non-music
    assert (cm.loc[[7, 8], "cid"] == -1).all()
    # below min_sessions -> out of vocabulary
    assert cm.loc[11, "cid"] == -1
    # Yesterday variants merge (incl. 'The Beatles' vs 'Beatles, The'); live stays separate; remix stays separate
    assert cm.loc[0, "cid"] == cm.loc[1, "cid"] == cm.loc[3, "cid"]
    assert cm.loc[2, "cid"] != cm.loc[0, "cid"] and cm.loc[6, "cid"] != cm.loc[4, "cid"]
    assert cm.loc[4, "cid"] == cm.loc[5, "cid"] and cm.loc[9, "cid"] == cm.loc[10, "cid"]
    counts = [c["n"] for c in r.counts]
    assert all(counts[i] >= counts[i + 1] for i in range(len(counts) - 1)), counts        # monotone non-increasing
    assert r.vocab["cid"].tolist() == list(range(len(r.vocab)))
    assert not r.degraded


def test_curate_without_llm_is_conservative_and_flagged():
    t = make_tracks(); ns = np.full(len(t), 50)
    r = curate(t, ns, CurationConfig(min_sessions=20), llm=None, prompts=None)
    assert r.degraded and "no LLM" in r.degraded[0]
    assert r.cmap.loc[0, "cid"] == r.cmap.loc[3, "cid"]            # exact strict keys still merge ("The Beatles" vs "Beatles, The")
    assert r.cmap.loc[4, "cid"] != r.cmap.loc[6, "cid"]            # remix separate
    assert (r.cmap.loc[[7, 8], "cid"] == -1).all()                  # keyword layer removes obvious non-music


def test_llm_failure_degrades_not_crashes():
    t = make_tracks(); ns = np.full(len(t), 50)
    r = curate(t, ns, CurationConfig(min_sessions=20), FakeLLM(fail_names=("refine",)), PROMPTS)
    assert any("refinement requests failed" in d for d in r.degraded)
    assert len(r.vocab) > 5


def test_blocker_a_no_names():
    t = pd.DataFrame({"artist": [""] * 5, "title": [""] * 5}); ns = np.full(5, 30)
    r = curate(t, ns, CurationConfig(min_sessions=20), FakeLLM(), PROMPTS, has_names=False)
    assert len(r.vocab) == 5 and any("Blocker-A" in d for d in r.degraded)


def test_audit_agreement():
    t = make_tracks(); ns = np.full(len(t), 50)
    r = curate(t, ns, CurationConfig(min_sessions=20), FakeLLM(), PROMPTS)
    a = audit_merges(r.cmap, t, FakeLLM(), PROMPTS, n=20)
    assert a["agreement_merge"] == 1.0


def test_human_corrections_applied():
    t = make_tracks(); ns = np.full(len(t), 50)
    r0 = curate(t, ns, CurationConfig(min_sessions=20), FakeLLM(), PROMPTS)
    corr = pd.DataFrame({"qid": ["m:1"], "decision": ["split"], "note": [""], "ts": ["2026-01-01"]})
    r1 = curate(t, ns, CurationConfig(min_sessions=20), FakeLLM(), PROMPTS, review_corrections=corr)
    assert r1.cmap.loc[1, "cid"] != r1.cmap.loc[0, "cid"] and r1.cmap.loc[1, "decision"] == "human"
    assert len(r1.vocab) == len(r0.vocab) + 1
