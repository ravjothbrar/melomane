# Kickoff prompt (paste into Claude Code, opened in this folder)

```
You are building "Melomane" in this repository, end to end, over several unattended nights.

1. Read, in full and in this order: CLAUDE.md, PLAN.md, docs/epicure_reference.md, docs/data_sources.md, README.md.
   They contain the complete spec, the methodology we replicate, verified/unverified data facts, and the rules you must follow.
2. Do Stage S0 now: create the environment (see README.md; Windows + RTX 5080; WSL2 preferred), run `pytest -q` (all provided
   tests must pass), run scripts/smoke_env.py and scripts/bench_gpu.py, verify the OpenRouter call, download and inventory
   Music4All-Onion, and resolve every "U" (unverified) item listed in PLAN.md section 13. Write PROGRESS.md and DECISIONS.md.
3. Then proceed through S1, S2, ... in order, one stage at a time, following the stage protocol in CLAUDE.md
   (tests first, acceptance checks, results/, reports/, PROGRESS.md entry, git commit). Do not ask questions unless one of the
   human-gate conditions in CLAUDE.md section 2 is hit; otherwise decide, log the decision, and continue.
4. Long jobs run detached with checkpoints and logs; poll them. If the session ends, the next session resumes from PROGRESS.md.
5. Never publish, email, or post anything. Never commit secrets or data. Never redistribute lyrics or audio.

Standard: complete, tested, documented, honest. Report negative results as clearly as positive ones.
Begin with S0.
```
