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
    assert "Melomane: Emergent Geometry" in md_text
    assert "DEMO RUN" in md_text
    assert "{{" not in md_text

    tex_text = paper_tex.read_text(encoding="utf-8")
    assert "\\documentclass" in tex_text
    assert "Melomane-Cooc" in tex_text


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
