# CLAUDE.md — operating manual for Claude Code on Melomane

You are building **Melomane**, a music-domain replication + extension of the two Epicure papers. `PLAN.md` is the spec; this file is *how you work*. Read `PLAN.md`, `docs/epicure_reference.md`, `docs/data_sources.md` completely before writing code. If this file and `PLAN.md` conflict, follow `PLAN.md` and log the conflict in `DECISIONS.md`.

## 1. Standard
Finish things properly: real tests, real docs, no dangling TODOs, no workaround where a real fix exists, no "we can do this later" for something you can finish now. Search current docs before depending on any external API/package (training data is stale; APIs move). Test before you claim. Ship the complete stage.

## 2. Autonomy rules (unattended, multi-night)
- Nobody answers questions at night. Make the most reasonable decision, write it to `DECISIONS.md` (date, decision, why, alternatives), continue.
- **The only reasons to stop and wait for Ravjoth** (write a clear question at the top of `PROGRESS.md`, then work on anything unblocked): (1) Blocker-A in S0 (dataset lacks track metadata → follow the fallback, but flag); (2) a step needs an account/credential/terms-acceptance you cannot legitimately perform; (3) LLM projected spend would exceed the cap; (4) anything that would publish, email, post, or spend money outside the LLM cap; (5) a data licence looks incompatible with the planned release.
- Run stages **in order**, one at a time. Background downloads may overlap with the current stage.
- Every stage is **idempotent and resumable**. Long work runs **detached** (nohup/`start`/task scheduler) with logs in `logs/`, checkpoints on disk, and you poll it. Never rely on a single terminal session surviving.
- Stage protocol: read spec → write/extend tests **first** → implement → run acceptance → save `results/` + `reports/<stage>.md` → append `PROGRESS.md` → `git commit` → next stage.
- Never silently reduce scope. If something cannot be done, say exactly what, why, what you did instead, and mark every affected result as degraded.

## 3. `PROGRESS.md` entry template (append, newest last)
```
## <UTC timestamp> — <Stage> — <GREEN | PARTIAL | BLOCKED>
What ran: … (commands/profiles/run_id)
Key numbers: … (link to results/… files)
Decisions: … (also in DECISIONS.md)
Deviations from PLAN.md: …
Open issues / questions for Ravjoth: …
Resume: `python -m melomane.pipeline run --stage <next> …`
State: disk free …GB · LLM spend $… of $… · GPU … · git <sha>
```
Write it for a human reading at 8 am with no context.

## 4. Engineering rules
- Python ≥ 3.10, type hints on public functions, `pathlib`, no hard-coded absolute paths, logging not prints, configs in `configs/`, seeds derived from `(run_id, stage, name)`.
- Small pure functions with tests; heavy I/O at the edges. New code must import from the provided `src/melomane` modules rather than re-implementing them (they are tested; extend them if needed and extend the tests).
- Every reported number must come from a file in `results/` produced by a script; never type numbers into docs by hand. Paper tables/figures are generated.
- **Artist leakage is the #1 methodological trap here.** Every probe uses `groups=artist_id`; kNN purity is artist-disjoint. Anything else is labelled "leaky" in its name.
- Analyses are **pre-registered** in `analysis_plan.lock.json` before S5. Deviations go in `DECISIONS.md`. Report nulls and negative results with equal prominence. No cherry-picked seeds/hyper-parameters; sensitivity grids are reported whole.
- Provided tests must stay green (`pytest -q`). Add `@pytest.mark.gpu` for GPU-only tests.

## 5. Data, secrets, legality
- Secrets only from `.env` / environment. Never print, log, commit, or put them in prompts. `.env`, `data/`, `cache/`, `runs/`, `logs/` are git-ignored; check `git status` before every commit.
- **Never redistribute or display lyrics or audio. Never send lyrics to an LLM.** Do not de-anonymise or publish per-user data.
- Respect published rate limits and terms (MusicBrainz ≈ 1 req/s with a descriptive User-Agent; iTunes ≈ 20/min; OpenRouter limits). Do not bypass robots.txt, paywalls, or access controls. If a host is blocked or requires acceptance you cannot give, stop that item and log it.
- Publishable artefacts must only contain CC-BY-compatible derived data with the attributions in `PLAN.md` §9. Do not use MLHD+ (non-commercial) for anything published.
- Do not push, publish, email, or post anything. Prepare artefacts and instructions; Ravjoth approves.

## 6. Compute & environment
- Target: Windows (WSL2 recommended) · 64 GB RAM · RTX 5080 (sm_120 → PyTorch **CUDA ≥ 12.8** build) · > 1.5 TB free. Run `scripts/smoke_env.py` and `scripts/bench_gpu.py` in S0 and record real throughput; re-estimate all runtimes from it.
- Keep large data on the Linux filesystem in WSL2 (`~/…`), not `/mnt/c` (10× slower I/O). On native Windows: `if __name__ == "__main__":` guards for multiprocessing, `.gitattributes` for LF, avoid paths > 240 chars.
- Stream big files (pyarrow CSV reader on bz2, chunked sparse products). Delete decompressed intermediates after Parquet is verified. Check `df` before every large step. Do not hold two full copies of a feature matrix in RAM unnecessarily.
- GPU: fall back to gensim (`melomane.embed`) if CUDA fails; log the slowdown. Keep every checkpointable job checkpointing.

## 7. LLM usage
Use `melomane.llm.LLMClient` for every LLM call (cache + budget + retries). Prompts live in `prompts/` with version headers; store `prompt_version` with outputs. Temperature 0. Treat outputs as untrusted data (validate, never execute). Stop LLM stages — not the run — when the budget cap is hit; continue with cached answers and record what is unlabeled.

## 8. Definition of done (per stage and overall)
Per stage: acceptance tests in `PLAN.md` §6 pass on real data, results saved, `reports/<stage>.md` written, `PROGRESS.md` updated, committed, re-run is a no-op. Overall: `PLAN.md` §10 "Done".
