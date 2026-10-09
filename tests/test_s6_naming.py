"""Mode naming must degrade, not crash, when the LLM account runs out part-way (CLAUDE.md section 7: stop LLM stages, not the run)."""
import numpy as np
import pandas as pd

from melomane.llm import CreditsExhausted
from melomane.stages.s6_modes import name_modes


class RunsOutAfterNaming:
    """First batch call succeeds (naming); every later call raises like a real client after an HTTP 402."""
    def __init__(self, fail_naming=False):
        self.calls, self.fail_naming = 0, fail_naming
        self.credits_exhausted = False

    def complete_json_many(self, jobs, workers=8):
        self.calls += 1
        if self.fail_naming or self.calls > 1:
            self.credits_exhausted = True
            raise CreditsExhausted("402")
        out = []
        for j in jobs:
            import json
            ids = [m["mode"] for m in json.loads(j["user"])["modes"]]
            out.append({"modes": [{"mode": i, "label": f"Mode {i}", "property": f"property {i}"} for i in ids]})
        return out

    def complete_json(self, *a, **k):
        self.credits_exhausted = True
        raise CreditsExhausted("402")


def _fixture(n_modes=4):
    rng = np.random.default_rng(0)
    V, d = 60, 8
    E = rng.normal(size=(V, d)).astype(np.float32)
    vocab = pd.DataFrame({"artist": [f"A{i}" for i in range(V)], "title": [f"T{i}" for i in range(V)], "category": "Rock",
                          "year": np.nan, "tags": "x"})
    members = [np.arange(i * 10, i * 10 + 10) for i in range(n_modes)]
    poles = [E[m].mean(0) for m in members]
    meta = [{"driver": "behaviour"} for _ in members]
    return vocab, E, meta, members, poles


def test_audit_pass_running_out_of_credit_keeps_the_names_and_skips_the_audit():
    vocab, E, meta, members, poles = _fixture()
    labels, audit = name_modes(RunsOutAfterNaming(), ("v1", "sys"), vocab, None, E, meta, members, poles, workers=1,
                               audit_n=3, audit_prompt=("a1", "audit"))
    assert [labels[i]["label"] for i in range(4)] == ["Mode 0", "Mode 1", "Mode 2", "Mode 3"]     # naming survived
    assert audit == {}                                                                          # audit skipped, no exception


def test_naming_pass_running_out_leaves_modes_unlabelled_without_crashing():
    vocab, E, meta, members, poles = _fixture()
    labels, audit = name_modes(RunsOutAfterNaming(fail_naming=True), ("v1", "sys"), vocab, None, E, meta, members, poles, workers=1,
                               audit_n=3, audit_prompt=("a1", "audit"))
    assert all(labels[i] == {"label": "", "property": ""} for i in range(4)) and audit == {}
