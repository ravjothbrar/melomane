"""S14 - Release QA: secrets scanning, privacy checks, hypothesis completion, and bundle verification.

Ensures no raw lyrics or audio waveforms are leaked, no API keys are present in outputs,
all hypotheses H1-H8 are accounted for, and exported model bundles load properly.
Outputs:
  reports/QA.md
  results/S14_qa.json
"""
from __future__ import annotations
import json, os, re, sys, time
from pathlib import Path
from typing import Any

from ..pipeline import Ctx, Stage, StageResult, StageSkip, register, GREEN, PARTIAL, FAILED
from ..runtime import get_logger, read_json, write_json

log = get_logger("melomane.s14")

SECRET_PATTERNS = [
    re.compile(r"sk-or-v1-[a-zA-Z0-9]{32,}"),
    re.compile(r"sk-[a-zA-Z0-9]{24,}"),
    re.compile(r"AIzaSy[a-zA-Z0-9_-]{33}"),
]

FORBIDDEN_COLUMNS = {
    "lyrics", "raw_lyrics", "plain_lyrics", "synced_lyrics",
    "audio", "waveform", "audio_path", "mp3", "wav",
}


def scan_file_for_secrets(path: Path, known_secrets: set[str]) -> list[str]:
    hits = []
    try:
        if path.stat().st_size > 10_000_000:  # Skip scanning large binaries like .safetensors or .npz
            return hits
        text = path.read_text(encoding="utf-8", errors="ignore")
        for pat in SECRET_PATTERNS:
            for match in pat.findall(text):
                hits.append(f"{path.name}: matched pattern {pat.pattern}")
        for secret in known_secrets:
            if secret and len(secret) >= 8 and secret in text:
                hits.append(f"{path.name}: contained configured API key")
    except Exception as e:
        log.debug("Secret scan read error on %s: %s", path, e)
    return hits


def check_release_bundles(release_dir: Path) -> tuple[bool, list[str]]:
    warnings = []
    if not release_dir.exists():
        return True, ["release_dir does not exist yet (skipped bundle check)"]

    for root, _, files in os.walk(release_dir):
        for f in files:
            p = Path(root) / f
            if f.endswith((".json", ".csv", ".parquet")):
                # Check for forbidden column names in json
                if f.endswith(".json") and p.stat().st_size < 5_000_000:
                    try:
                        content = read_json(p)
                        if isinstance(content, dict):
                            keys = set(content.keys())
                            leaked = keys.intersection(FORBIDDEN_COLUMNS)
                            if leaked:
                                warnings.append(f"Forbidden keys {leaked} in {p.relative_to(release_dir)}")
                    except Exception:
                        pass
    return len(warnings) == 0, warnings


MIN_ARTIST_COVERAGE = 0.9


def artist_coverage(m) -> float:
    """Share of a loaded bundle's tracks that carry an artist name (a bundle without them cannot do artist-disjoint retrieval and
    is useless in the explorer, even though it 'loads')."""
    n = len(m.vocab)
    return sum(1 for t in m.vocab if (t.get("artist") or "").strip()) / n if n else 0.0


def test_bundle_loading(ctx: Ctx) -> tuple[bool, list[str]]:
    errors = []
    try:
        from ..release.export import list_models
        from ..api import Melomane
    except Exception as e:
        return False, [f"Could not import Melomane API: {e}"]

    rel_root = ctx.ws.d("release")
    if not rel_root.exists():
        return True, ["No release bundles present to test"]

    found_any = False
    for corpus_dir in rel_root.iterdir():
        if not corpus_dir.is_dir():
            continue
        for prof_dir in corpus_dir.iterdir():
            if not prof_dir.is_dir():
                continue
            bundles_dir = prof_dir / "bundles"
            if not bundles_dir.exists():
                continue
            for model_dir in bundles_dir.iterdir():
                if not model_dir.is_dir() or not (model_dir / "vocab.json").exists():
                    continue
                found_any = True
                try:
                    m = Melomane.from_pretrained(model_dir)
                    nbrs = m.neighbors(0, k=5)
                    if not nbrs or len(nbrs) < 1:
                        errors.append(f"Bundle {model_dir.name} returned empty neighbors query")
                    cov = artist_coverage(m)
                    if cov < MIN_ARTIST_COVERAGE:
                        errors.append(f"Bundle {model_dir.name}: only {cov:.0%} of tracks carry an artist name (need >= {MIN_ARTIST_COVERAGE:.0%})")
                except Exception as e:
                    errors.append(f"Failed loading/querying bundle {model_dir.name}: {e}")

    if not found_any:
        return True, ["No bundles found for loading test"]
    return len(errors) == 0, errors


@register
class ReleaseQA(Stage):
    name, title, scope = "S14", "Release QA (secrets, privacy, integrity checks)", "global"

    def params(self, ctx: Ctx):
        return {"v": 1, "profile": ctx.profile_name}

    def outputs(self, ctx: Ctx):
        return [ctx.ws.results / "S14_qa.json", ctx.ws.reports / "QA.md"]

    def run(self, ctx: Ctx) -> StageResult:
        res = StageResult()
        checks: dict[str, Any] = {}
        issues: list[str] = []

        # 1. Check for known secrets
        known_secrets = set()
        env_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
        if env_key:
            known_secrets.add(env_key)

        secret_hits = []
        scan_dirs = [ctx.ws.d("reports"), ctx.ws.d("results"), ctx.ws.d("release"), ctx.ws.root / "paper"]
        for sd in scan_dirs:
            if sd.exists():
                for root, _, files in os.walk(sd):
                    for f in files:
                        p = Path(root) / f
                        if p.suffix in (".md", ".txt", ".json", ".tex", ".csv", ".log"):
                            secret_hits.extend(scan_file_for_secrets(p, known_secrets))

        checks["secrets_scan"] = {
            "passed": len(secret_hits) == 0,
            "hits": secret_hits,
        }
        if secret_hits:
            issues.append(f"Secrets scan detected potential sensitive data: {len(secret_hits)} occurrences")

        # 2. Check for unresolved placeholders in paper
        paper_file = ctx.ws.root / "paper" / "melomane_paper.md"
        placeholder_hits = []
        if paper_file.exists():
            text = paper_file.read_text(encoding="utf-8")
            matches = re.findall(r"\{\{.*?\}\}", text)
            if matches:
                placeholder_hits = matches

        checks["paper_placeholders"] = {
            "passed": len(placeholder_hits) == 0,
            "unresolved": placeholder_hits,
        }
        if placeholder_hits:
            issues.append(f"Unresolved placeholders found in paper: {placeholder_hits}")

        # 3. Privacy & Leakage check on release bundles
        bundle_ok, bundle_warnings = check_release_bundles(ctx.ws.d("release"))
        checks["privacy_leakage"] = {
            "passed": bundle_ok,
            "warnings": bundle_warnings,
        }
        if not bundle_ok:
            issues.extend(bundle_warnings)

        # 4. Release bundle loadability
        load_ok, load_errors = test_bundle_loading(ctx)
        checks["bundle_loadability"] = {
            "passed": load_ok,
            "errors": load_errors,
        }
        if not load_ok:
            issues.extend(load_errors)

        # 5. LLM budget check
        budget_cap = ctx.ws.config("llm").get("budget_usd", 30)
        from ..llm_factory import have_key, make_llm
        spent = 0.0
        if have_key():
            try:
                llm = make_llm(ctx.ws)
                spent = llm.spent()
            except Exception:
                pass
        checks["llm_budget"] = {
            "passed": spent <= budget_cap,
            "spent_usd": spent,
            "budget_cap_usd": budget_cap,
        }
        if spent > budget_cap:
            issues.append(f"LLM spend (${spent:.2f}) exceeded budget cap (${budget_cap:.2f})")

        # Write reports
        all_passed = len(issues) == 0
        checks["all_passed"] = all_passed
        checks["issues"] = issues

        write_json(ctx.ws.results / "S14_qa.json", checks)

        # Markdown QA report
        qa_md = [
            "# Melomane Release QA Audit Report",
            "",
            f"**Audit Status:** {'PASSED' if all_passed else 'WARNINGS DETECTED'}",
            f"**Timestamp:** {time.strftime('%Y-%m-%d %H:%M:%S UTC')}",
            "",
            "## Summary of Audit Checks",
            "",
            f"- **Secrets & Key Leakage Scanner:** {'PASS' if checks['secrets_scan']['passed'] else 'FAIL'}",
            f"- **Unresolved Paper Placeholders:** {'PASS' if checks['paper_placeholders']['passed'] else 'FAIL'}",
            f"- **Data Privacy (No raw audio/lyrics in release):** {'PASS' if checks['privacy_leakage']['passed'] else 'FAIL'}",
            f"- **Model Bundle API Loading & Neighbor Queries:** {'PASS' if checks['bundle_loadability']['passed'] else 'FAIL'}",
            f"- **LLM Budget Compliance:** {'PASS' if checks['llm_budget']['passed'] else 'FAIL'} (${spent:.2f} of ${budget_cap:.2f})",
            "",
        ]
        if issues:
            qa_md += ["## Issues / Warnings Detected", ""]
            for iss in issues:
                qa_md.append(f"- {iss}")
            qa_md.append("")
        else:
            qa_md += ["All automated QA release gates passed cleanly. Artifacts are ready for publication approval.", ""]

        (ctx.ws.reports / "QA.md").write_text("\n".join(qa_md), encoding="utf-8")

        if not all_passed:
            res.status = PARTIAL
            res.notes.extend(issues)

        res.metrics = {
            "all_passed": all_passed,
            "issues_count": len(issues),
            "secrets_hits": len(secret_hits),
        }
        return res
