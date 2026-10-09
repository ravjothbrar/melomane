from melomane.hygiene import public_progress, scrub_local_paths

BS = chr(92)
LOG = f"""# PROGRESS

> **Read this first.** header note

Auto-written by the Melomane pipeline (newest last).

## 2026-10-05 18:54 UTC — S1 (Canonical vocabulary (curation)) [synth/smoke] — GREEN
Ran in 0s. Key numbers: V=599

## 2026-10-06 10:28 UTC — S3 (Behaviour graph (session NPMI)) [onion/standard] — GREEN
Ran in 98s. Key numbers: same_artist_edge_share=0 (superseded run)

## 2026-10-09 16:35 UTC — S3 (Behaviour graph (session NPMI)) [onion/standard] — GREEN
Ran in 62s. Notes: wrote C:{BS}Users{BS}someone{BS}Projects{BS}melomane{BS}paper/ ok

## 2026-10-09 17:00 UTC — S1 (Canonical vocabulary (curation)) [synth/smoke] — GREEN
Ran in 1s. demo

## 2026-10-09 20:00 UTC — S14 (Release QA) [onion/standard] — PARTIAL
Ran in 3s.
"""


def test_public_progress_keeps_the_header_and_only_current_run_entries():
    out = public_progress(LOG, since="2026-10-09 15:00", corpora=("onion/standard",), root=BS.join(["C:", "Users", "someone", "Projects", "melomane"]))
    assert out.startswith("# PROGRESS") and "Read this first" in out                         # header kept
    assert "[synth/smoke]" not in out                                                         # demo-corpus noise dropped, even when recent
    assert "2026-10-06" not in out and "superseded run" not in out                           # earlier onion run dropped
    assert "2026-10-09 16:35" in out and "2026-10-09 20:00" in out                           # current run kept, in order
    assert out.index("16:35") < out.index("20:00")
    assert "Users" + BS not in out and "<repo>" in out                                       # local path scrubbed


def test_public_progress_is_idempotent():
    once = public_progress(LOG, since="2026-10-09 15:00", corpora=("onion/standard",), root="C:/x")
    assert public_progress(once, since="2026-10-09 15:00", corpora=("onion/standard",), root="C:/x") == once


def test_scrub_local_paths_handles_both_slash_styles_and_json_escapes():
    root = BS.join(["C:", "Users", "someone", "Projects", "melomane"])
    s = f'a {root}{BS}data and C:/Users/someone/Projects/melomane/logs and C:{BS}{BS}Users{BS}{BS}someone{BS}{BS}Projects{BS}{BS}melomane{BS}{BS}x'
    out = scrub_local_paths(s, root)
    assert "someone" not in out and out.count("<repo>") == 3


def test_a_supplied_header_replaces_the_logs_own():
    hdr = "# PROGRESS" + chr(10) + chr(10) + "> curated header" + chr(10)
    out = public_progress(LOG, since="2026-10-09 15:00", header=hdr, root="C:/x")
    assert out.startswith(hdr) and "Read this first" not in out and "2026-10-09 16:35" in out
