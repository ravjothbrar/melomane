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


def _real_shape_data():
    """A small dict shaped like load_all_results() output for a mixed result: Composition wins on (a)/(b), loses on (c), Cooc wins culture."""
    ev = lambda d, lo, hi: {"delta_rho": d, "ci_low": lo, "ci_high": hi, "p_boot": 0.001, "n_targets": 10, "share_targets_positive": 0.9}
    sm = {m: {"PR": {"mean": pr, "sd": 1.0, "n_seeds": 3}, "mean_cos": {"mean": 0.1, "sd": 0.0, "n_seeds": 3},
              "a_baked|rho": {"mean": a, "sd": 0.0, "n_seeds": 3}, "b_heldout|rho": {"mean": b, "sd": 0.0, "n_seeds": 3},
              "c_external|rho": {"mean": c, "sd": 0.0, "n_seeds": 3}, "cultural|d": {"mean": d, "sd": 0.0, "n_seeds": 3}}
          for m, pr, a, b, c, d in (("cooc", 45.0, 0.38, 0.24, 0.55, 1.84), ("composition", 200.0, 0.64, 0.33, 0.50, 0.64), ("core", 176.0, 0.64, 0.33, 0.55, 1.65))}
    return {"corpus": "onion", "profile": "standard", "vocab_count": 1000, "artist_count": 100, "unknown_artist_tracks": 7,
            "s1": {"consolidation_ratio": 1.0, "degraded": ["1300/1333 refinement requests failed; those items kept unmerged"]},
            "s2_stats": {"n_labelled": 200, "coverage": 1.0, "coverage_of_vocab": 0.2, "excluded_dims": ["space"]},
            "s4": {"max_cluster_share": 0.0512, "n_compounds": 3300},
            "s6": {m: {"n_factors_stable": 20, "n_modes": 140, "coherence": {"coherence": 0.3, "random": 0.1, "ratio": 3.0}, "driver_counts": {"acoustic": 140}} for m in ("cooc", "composition", "core")},
            "s7": {"summary_by_model": sm, "n_eval_tracks": 5000, "strata_sizes": {"a_baked": 17}, "paired": {}},
            "s8": {}, "s8_h7": {"status": "supported", "share_by_model": {"cooc": 0.94, "composition": 0.8, "core": 0.94}},
            "s9": {"H8": {"n_tracks": 8000, "status": "supported"},
                   "E6": {"status": "done", "span_years": 15.1, "procrustes": {"disparity": 0.66, "null_disparity_mean": 1.0, "row_cosine": 0.58},
                          "mode_persistence": {"mean_best_jaccard": 0.209, "null_best_jaccard_mean": 0.029, "share_persisting": 0.021}},
                   "E8": {"status": "done", "summary_delta_composition_minus_cooc": {"n_cells": 27, "n_meeting": 27, "min": 0.019, "median": 0.086, "max": 0.103}},
                   "E4": {"status": "done", "H1_equal_token": "not supported"},
                   "E3": {"status": "done", "systems": [{"model": "core_partial30", "pct_hub": 30, "rho_all": 0.257, "rho_cooc_all": 0.254, "rho_hub": 0.26, "rho_cooc_hub": 0.25, "rho_nonhub": 0.256, "rho_cooc_nonhub": 0.254}]}},
            "listening_coverage": {"tracks_with_listening_data": 56512, "tracks_with_audio_features": 109269}}, \
        {"H1": {"claim": "c", "status": "not supported", "evidence": {"a": ev(0.259, 0.25, 0.268), "b": ev(0.094, 0.088, 0.099), "c": ev(-0.043, -0.055, -0.031)}},
         "H2": {"claim": "c", "status": "not supported"}, "H3": {"claim": "c", "status": "not supported"}, "H7": {"claim": "c", "status": "supported"},
         "H8": {"claim": "c", "status": "supported"}}


def test_abstract_reports_the_evidence_not_just_the_h1_verdict():
    """Regression: with H1 'not supported' the abstract said Composition 'did not exceed Cooc on the held-out strata', but it beat Cooc there
    by a wide margin and lost only on the LLM-label stratum."""
    from melomane import paper_text as pt
    data, hyp = _real_shape_data()
    a = pt.abstract(data, hyp)
    assert "did not exceed Cooc on the held-out" not in a
    assert "baked-in" in a and "held-out" in a and "LLM-label" in a and "0.094" in a and "-0.043" in a
    assert "Cooc" in a and "1.84" in a and "0.64" in a                           # the culture result (H3) is stated with numbers
    assert "94%" in a and "80%" in a                                              # steering shares come from S8
    assert "8,000" in a                                                           # the budget check names its subset size


def test_cultural_column_is_filled_and_extensions_and_limitations_are_specific():
    from melomane import paper_text as pt
    data, hyp = _real_shape_data()
    md = pt.render_markdown(pt.build_blocks(data, hyp))
    lines = md.splitlines()
    head = next(i for i, l in enumerate(lines) if "Baked-in (a)" in l)
    assert "1.84" in lines[head + 2] and "n/a" not in lines[head + 2]            # Cooc row: cultural d present (was n/a through a key mismatch)
    ext = [l for l in lines if l.startswith("| E")]
    for tag in ("E3", "E4", "E6", "E8"):
        row = next(l for l in ext if l.startswith(f"| {tag}"))
        assert row.split("|")[3].strip() != "", tag                               # no empty detail cells
    assert "27 of 27" in md and "15.1" in md                                      # sensitivity grid and temporal span are reported
    lim = md.split("## 5. Limitations")[1]
    assert "unconfirmed provenance" not in lim                                    # the source has been identified
    assert "rules" in lim and "200" in lim and "5.1" in lim                       # degraded curation, label coverage and the cluster-size miss
    assert "enrich-music4all" in lim
    assert "(20% of the vocabulary)" in lim and "100% of the vocabulary" not in lim      # coverage is against the vocabulary, not the annotated sample
    assert "ceiling of 7" in lim


def test_reproducibility_manifest_has_real_facts_and_never_prints_none():
    """Regression: the manifest printed 'Host: None (None)', a raw dict for the hardware and an LLM spend of $0.00."""
    from melomane import paper_text as pt
    data = {"corpus": "onion", "profile": "standard", "vocab_count": 53309, "artist_count": 9872, "seeds": 3, "git_commit": "abc1234",
            "train": {"walks_per_node": 4, "epochs": 6, "ii_repeat": 10, "dim": 300}, "llm_model": "google/gemini-3.7-flash",
            "llm_cache_spend_usd": 12.48, "llm_budget_usd": 30, "lock_amendments": 2}
    host = {"platform": "Windows-11", "cpus": 24, "python": "3.12.13", "torch": "2.11.0+cu128", "gpu": "NVIDIA GeForce RTX 5080 Laptop GPU", "gpu_capability": "12.0"}
    out = pt.reproducibility_text(data, host, "2026-10-10 13:30 UTC")
    assert "None" not in out and "{" not in out and "$0.00" not in out
    assert "RTX 5080" in out and "$12.48 of $30 cap" in out and "4 walks/node × 6 epochs" in out and "abc1234" in out and "2 recorded amendment" in out
    bare = pt.reproducibility_text({"corpus": "x", "profile": "y"}, {}, "now")
    assert "None" not in bare and "LLM spend" not in bare                               # missing facts are omitted, not faked
