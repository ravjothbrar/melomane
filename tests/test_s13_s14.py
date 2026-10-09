import json
from pathlib import Path
import numpy as np
import pytest

from melomane.pipeline.framework import Ctx
from melomane.stages.s13_paper import Paper, load_all_results, collate_hypotheses, build_markdown_paper, build_latex_paper
from melomane.stages.s14_qa import ReleaseQA, scan_file_for_secrets, check_release_bundles
try:
    from .wshelper import make_ws
except ImportError:
    from wshelper import make_ws


def test_s13_paper_generation(tmp_path: Path):
    ws = make_ws(tmp_path)
    prof = ws.profile("smoke")
    ctx = Ctx(ws, "synth", "smoke", prof, {"corpora": ["synth"]})

    # Create mock result data
    rd = ws.d("results", "synth", "smoke")
    rd.mkdir(parents=True, exist_ok=True)
    summary_data = {
        "summary_by_model": {
            "cooc": {"PR": {"mean": 170.0}, "mean_cos": {"mean": 0.12}, "b_heldout|rho": {"mean": 0.32}},
            "composition": {"PR": {"mean": 180.0}, "mean_cos": {"mean": 0.11}, "b_heldout|rho": {"mean": 0.47}},
            "core": {"PR": {"mean": 95.0}, "mean_cos": {"mean": 0.35}, "b_heldout|rho": {"mean": 0.42}},
        },
        "n_eval_tracks": 1000,
        "n_targets": 20,
    }
    (rd / "S7_summary.json").write_text(json.dumps(summary_data), encoding="utf-8")

    hyp_data = {
        "1": {"claim": "Composition >= Cooc", "status": "supported", "evidence": {}},
        "2": {"claim": "Core lowest PR", "status": "supported", "evidence": {}},
        "3": {"claim": "Regional d", "status": "supported", "evidence": {}},
    }
    (rd / "S7_hypotheses.json").write_text(json.dumps(hyp_data), encoding="utf-8")

    # Run S13
    stage = Paper()
    res = stage.run(ctx)
    assert res.status in ("GREEN", "PARTIAL")

    paper_md = ws.root / "paper" / "melomane_paper.md"
    paper_tex = ws.root / "paper" / "melomane_paper.tex"
    repro_md = ws.root / "paper" / "reproducibility.md"

    assert paper_md.exists()
    assert paper_tex.exists()
    assert repro_md.exists()

    md_text = paper_md.read_text(encoding="utf-8")
    assert "emergent geometry in music-track embeddings" in md_text
    assert "DEMO RUN" in md_text
    assert "{{" not in md_text and "TODO" not in md_text and "TBD" not in md_text
    assert "## 5. Limitations" in md_text and "preprint pending" in md_text
    # the fixture has no external / cultural strata: they must read n/a, never 0.000 / 0.00
    lines = md_text.splitlines()
    head = next(i for i, l in enumerate(lines) if "Baked-in (a)" in l)
    row = lines[head + 2]                                    # first data row of the strata table (Cooc)
    assert row.startswith("| Melomane-Cooc") and row.count("n/a") == 3 and "0.320" in row and "0.000" not in row

    tex_text = paper_tex.read_text(encoding="utf-8")
    assert "\\documentclass" in tex_text
    assert "Melomane-Cooc" in tex_text
    assert tex_text.rstrip().endswith(r"\end{document}")
    for env in ("table", "tabular", "itemize", "abstract"):
        assert tex_text.count(r"\begin{%s}" % env) == tex_text.count(r"\end{%s}" % env), env       # balanced environments
    assert "Limitations" in tex_text


def test_s14_qa_stage(tmp_path: Path):
    ws = make_ws(tmp_path)
    prof = ws.profile("smoke")
    ctx = Ctx(ws, "synth", "smoke", prof, {"corpora": ["synth"]})

    # Run S14
    stage = ReleaseQA()
    res = stage.run(ctx)

    qa_report = ws.reports / "QA.md"
    qa_json = ws.results / "S14_qa.json"

    assert qa_report.exists()
    assert qa_json.exists()

    report_text = qa_report.read_text(encoding="utf-8")
    assert "Melomane Release QA Audit Report" in report_text


def test_secret_scanner(tmp_path: Path):
    f = tmp_path / "secret.txt"
    f.write_text("my_key = sk-or-v1-abcdefghijklmnopqrstuvwxyz1234567890\n", encoding="utf-8")
    hits = scan_file_for_secrets(f, known_secrets=set())
    assert len(hits) > 0
    assert "matched pattern" in hits[0]


def _s2(rows):
    """Shape written by stage S2 (S2_llm_vs_measured.json)."""
    pos = [r for r in rows if r["rho"] > 0 and r["ci_excludes_0"]]
    return {"n_dims_with_counterpart": len(rows), "share_positive_ci": len(pos) / len(rows) if rows else None, "table": rows}


def test_h5_is_read_from_the_keys_stage_s2_actually_writes():
    """Regression: S13 looked for `share_positive_significant` while S2 writes `share_positive_ci`, so H5 was always reported
    'not evaluable' even with a full table of results."""
    from melomane.stages.s13_paper import h5_status
    rows = [{"dim": "energy", "rho": 0.29, "ci_excludes_0": True}, {"dim": "aggression", "rho": -0.37, "ci_excludes_0": True},
            {"dim": "tempo", "rho": -0.05, "ci_excludes_0": False}, {"dim": "bright", "rho": 0.17, "ci_excludes_0": True}]
    h = h5_status(_s2(rows))
    assert h["status"] == "not supported"                                   # 2 of 4 = 50 % < 70 %
    assert h["evidence"]["share_positive_ci"] == 0.5 and h["evidence"]["wrong_sign_dims"] == ["aggression"]
    good = [dict(r, rho=0.3, ci_excludes_0=True) for r in rows]
    assert h5_status(_s2(good))["status"] == "supported"
    assert h5_status({})["status"] == "not evaluable"                       # nothing to compare is the only 'not evaluable'
    assert h5_status({"table": [], "share_positive_ci": None})["status"] == "not evaluable"
    assert collate_hypotheses({"s2": _s2(rows)})["H5"]["status"] == "not supported"


def test_h3_with_a_single_region_is_not_evaluable():
    from melomane.stages.s7_eval import Evaluation
    out = {"summary_by_model": {m: {"PR": {"mean": 1.0}, "mean_cos": {"mean": 0.1}} for m in ("cooc", "composition", "core")},
           "cultural_d": {"composition": {"r1": 1.0}, "cooc": {"r1": 2.0}}}
    assert Evaluation.hypotheses(out, None)["3"]["status"] == "not evaluable"
    out["cultural_d"] = {"composition": {f"r{i}": 2.0 for i in range(4)}, "cooc": {f"r{i}": 1.0 for i in range(4)}}
    assert Evaluation.hypotheses(out, None)["3"]["status"] == "supported"


def test_artist_coverage_flags_bundles_without_artist_names():
    from melomane.api import Melomane
    from melomane.stages.s14_qa import artist_coverage, MIN_ARTIST_COVERAGE
    E = np.random.default_rng(0).normal(size=(10, 4)).astype(np.float32)
    bare = Melomane(E, [{"cid": i, "artist": "", "title": ""} for i in range(10)])
    full = Melomane(E, [{"cid": i, "artist": f"A{i}", "title": f"T{i}"} for i in range(10)])
    assert artist_coverage(bare) == 0.0 < MIN_ARTIST_COVERAGE <= artist_coverage(full) == 1.0


def test_readme_generated_region_is_replaced_in_place_and_idempotent():
    from melomane import paper_text as pt
    base = f"intro prose\n\n{pt.README_BEGIN}\nOLD NUMBERS 0.123\n{pt.README_END}\n\nouttro prose\n"
    data = {"corpus": "onion", "profile": "standard", "vocab_count": 10, "artist_count": 3,
            "s7": {"summary_by_model": {"cooc": {"PR": {"mean": 1.5, "sd": 0.1, "n_seeds": 3}}}}}
    block = pt.readme_results_block(data, {"H1": {"claim": "c", "status": "supported"}})
    once = pt.inject_readme(base, block)
    assert "OLD NUMBERS" not in once and once.startswith("intro prose") and once.rstrip().endswith("outtro prose")
    assert "1.5 ± 0.1" in once and "SUPPORTED" in once and "Do not edit by hand" in once
    assert pt.inject_readme(once, block) == once                       # idempotent
    assert pt.inject_readme("no markers here", block) == "no markers here"


def test_credit_gate_turns_a_402_into_a_question_with_the_resume_command():
    from types import SimpleNamespace
    from melomane.pipeline import StageResult
    from melomane.stages.s1_curation import credit_gate
    ctx = SimpleNamespace(corpus="onion", profile_name="standard")
    res = StageResult()
    credit_gate(ctx, SimpleNamespace(credits_exhausted=False, budget=30.0), res, "S2")
    credit_gate(ctx, None, res, "S2")
    assert res.questions == [] and res.status != "PARTIAL"                      # nothing wrong, nothing said
    credit_gate(ctx, SimpleNamespace(credits_exhausted=True, budget=30.0), res, "S2")
    assert res.status == "PARTIAL" and len(res.questions) == 1
    q = res.questions[0]
    assert "out of credit" in q and "--stage S2 --corpus onion --profile standard --force" in q and "$30" in q
