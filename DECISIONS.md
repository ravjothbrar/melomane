# MELOMANE — Architectural & Technical Decisions Log

This document records the architectural, methodological, and implementation decisions made during the design and execution of the **Melomane** research pipeline.

---

## 1. Scope, Objectives & Methodology

- **Epicure Replication & Extension:** Replicate the core methodology of the two Epicure papers (*multidimensional flavor structure*, arXiv:2604.22776; *emergent geometry navigation*, arXiv:2605.22391) adapted from food/ingredients to music/tracks.
- **Three-Model Controlled Spectrum:**
  - **Melomane-Cooc:** What tracks people listen to together (co-occurrence graph derived from listening sessions with positive NPMI).
  - **Melomane-Composition:** What tracks are made of (sonic + lyrical compound graph; measured audio descriptors + lyric themes).
  - **Melomane-Core:** 10:1 blend of co-occurrence and compound walks (`ii_repeat = 10`), matching Epicure-Core.
- **LLM-Free Embeddings:** Embeddings are strictly learned via Skip-Gram with Negative Sampling (SGNS) on typed walks. No LLMs are used to generate embeddings. LLMs (Gemini 3.7 Flash via OpenRouter) are restricted to track canonicalization, metadata normalization, and mode naming.
- **Copyright & Privacy Hygiene:** Raw audio and lyrics are never redistributed, never embedded in published artefacts, and never sent to external LLMs.

---

## 2. Dataset Strategy

- **Single Primary Corpus (Music4All-Onion v2):**
  - CC BY 4.0 open dataset hosted on Zenodo (~109,269 tracks with 253M listening interactions across 119k users).
  - Full compound hub coverage (100% of tracks have aligned Essentia acoustic features, lyrics representations, and tags).
  - Eliminates cross-dataset alignment mismatches between listening data and audio/lyric feature sets.
- **Spotify 1-Million Playlists (Secondary / Scaling):**
  - The 10.66GB MySQL dump (`spotifydbdumpshare.sql` from Zenodo / `playlist2vec`) is available locally at `~/Downloads` for co-occurrence scaling.
- **Synthetic Planted Corpus (`synth`):**
  - Deterministic synthetic corpus with planted latent geometric structure used for smoke tests, unit tests, and CI without requiring remote downloads or external API access.

---

## 3. Environment & Hardware Architecture

- **Native Windows Execution:**
  - Deployed in `<repo>` on a high-speed local NVMe SSD (avoiding OneDrive sync lag and Windows file locks).
  - Windows 11 with 64GB RAM and NVIDIA GeForce RTX 5080 Laptop GPU (Blackwell architecture, compute capability `sm_120`).
- **Python Toolchain & PyTorch:**
  - Python 3.12 managed via `uv` (`.venv`).
  - PyTorch installed with CUDA 12.8 wheel (`torch==2.11.0+cu128`), fully verified for Blackwell tensor operations and GPU SGNS acceleration (`torch_sgns.py`).
- **Scikit-Learn Version Pinning (`1.5.2`):**
  - *Context:* Windows Smart App Control blocked `scikit-learn 1.9.1`'s C-extension DLL `_tree.cp312-win_amd64.pyd` due to lack of binary reputation.
  - *Decision:* Pinned `"scikit-learn>=1.5.0,<1.6.0"` (installed 1.5.2) in `pyproject.toml`, which is digitally recognized and whitelisted by Microsoft Code Integrity.

---

## 4. Methodological Safeguards

- **Artist Leakage Prevention (Methodological Trap #1):**
  - Same-artist tracks share acoustic and lyrical signatures.
  - All linear and ridge probes use cross-validation strictly grouped by `artist_id`.
  - All kNN purity evaluations are strictly artist-disjoint (excluding any track sharing the query's artist).
- **Pre-Registered Analysis Plan:**
  - Analysis plan locked in `analysis_plan.lock.json` before training.
  - Hypotheses H1 through H8 are evaluated against pre-registered criteria, reporting null and negative results honestly.
- **Budget Hard Cap:**
  - OpenRouter LLM spend capped at `$30 USD` (within the user's balance) in `configs/llm.yaml` and `.env`.
  - Deterministic SQLite caching (`cache/llm.sqlite`) ensures re-runs consume zero API tokens.

---

## 5. Pipeline Architecture & Bug Fixes

- **Stage Extensions (S13 & S14):**
  - **S13 (Paper Compilation):** Generates publication-ready white paper deliverables in both Markdown (`melomane_paper.md`) and arXiv LaTeX (`melomane_paper.tex`), with all numbers populated directly from `results/` without hand-typed figures. Automatically compiles figures and reproducibility documentation.
  - **S14 (Release QA):** Audits against API secret leaks, checks copyright compliance (no raw audio or lyrics), verifies placeholder absence (`{{...}}`), tests bundle deserialization via `Melomane.from_pretrained`, and audits LLM budget compliance.
- **Key Pipeline Bug Fixes:**
  - *`DemoLLM` Missing Method:* Implemented `_audit_modes` in `src/melomane/demo.py` for mode naming consistency auditing.
  - *`modes.json` NaN Serialization:* Wrapped JSON export in `jsonable(...)` to convert floating-point `NaN` values to compliant `null` values.
  - *S8 Double-Indexing Fix:* Fixed `s8_slerp.py` line 22 where `E[idx]` was called on an already-sliced matrix `E`, resolving `IndexError: index 598 is out of bounds for axis 0 with size 598`.

---

## 6. One-Command Automation

- **`run_melomane.bat`:**
  - Native Windows batch launcher with `UV_SYSTEM_CERTS=1`.
  - Supports `run_melomane.bat setup`, `run_melomane.bat smoke`, `run_melomane.bat doctor`, `run_melomane.bat status`, and full overnight training via `run_melomane.bat train` or default invocation `run_melomane.bat`.
  - Disables Windows standby/sleep during runs to ensure unattended multi-hour completion.

---

## 7. 2026-10-09 — correctness re-run (artist grouping, honest hypotheses, spec-faithful training)

Each entry: **decision** · why · alternatives considered. The pre-registration trail is `analysis_plan.lock.json` → `amendments`.

1. **All Onion results from 2026-10-06 are superseded and archived** (`*/_superseded_20261006/`), then re-run from ingest.
   *Why:* S1–S8 ran while every track had its own `artist_id` (56,512 unique ids), so "artist-grouped" CV and "artist-disjoint" kNN
   purity were no-ops — i.e. leaky — and the paper said so itself ("53,787 tracks across 53,787 artist groups").
   *Alternatives:* patch only S7 (rejected: the dev/test artist split in S3, the artist-masked Cooc ablation and the S2 labels all depend on real artists).
2. **Artist/title/album come from an external track-metadata table, ingested through the adapter** (`onion.prepare_metadata` →
   `data/raw/onion/id_metadata.parquet`; `artist_id` = normalised artist name; 16,126 distinct artists over 108,363 tracks).
   *Provenance is unverified:* the Zenodo record (46 files, listed in `ZENODO_15394646_FILES.json`) has no artist/title; the table
   (`track_metadata.parquet`, SHA-256 in `data/raw/onion/METADATA_PROVENANCE.json`) was placed in `data/interim` by hand on
   2026-10-06 with no code, log or note. **Open question for Ravjoth (CLAUDE.md §2 (5)): where did it come from and may the names be republished?**
   Used locally for analysis (names never leave the machine); `DATA_LICENSE.md` marks the explorer payload "not cleared" until answered.
   Only `id, artist, title, album` are kept (captions/tag lists dropped).
3. **`standard` profile = spec structure at a scaled walk budget**: 3 seeds, `ii_repeat` 10, identical epochs for Cooc/Composition/Core,
   4 walks/node × 6 epochs (base is 8 × 10), ablations: artist-masked Cooc, acoustic-only and lyric-only Composition, Core with
   60/30/10 % compound hubs, equal-token Cooc and Composition, raw-vocabulary averaged Cooc.
   *Why:* measured throughput is ~440k tokens/s (≈7.4M pairs/s, `S0-doctor`), not the 30M pairs/s `PLAN.md` assumed; the base budget is
   ≈25 GPU-hours, this is ≈7.5. The 2026-10-06 profile (1 seed, `ii_repeat` 2, 12/12/5 epochs) departed from the design in ways that matter
   for H2 (Core is *defined* by the 10:1 blend) and was never recorded.  *Checks on the scaling:* E4 equal-token control and the `paper`-profile subset (H8).
   *Dropped:* `core_eqtok` (equal-token-to-Core is Core).
4. **LLM budget cap 30 USD** (`PLAN.md` §1/§2; the config said 20). *Why:* `PLAN.md` wins over the config (`CLAUDE.md` preamble). Spend so far < 1 USD.
5. **Removed the "never invalidate" shortcut in `pipeline.framework.is_done`** that let S0-ingest/S0-enrich/S1/S3/S4 skip re-running when
   their parameters changed (it was added to avoid a 35-minute re-ingest and silently served stale results). The ingest now hashes the metadata file.
6. **Corpus size is 56,512 tracks, not 109,269**, because only 56,512 tracks have listening data (`results/onion/listening_coverage.json`,
   from `scripts/listening_coverage.py`; all 56,512 also have audio features). The other 52,757 have features but no behaviour, so Cooc
   cannot cover them. `PLAN.md`'s "80–105 k" expectation is not reachable with Onion alone; reported as a limitation, not worked around.
7. **Hypothesis verdicts**: H5 is read from the keys S2 actually writes (`share_positive_ci`) — previously always "not evaluable" — and is
   "not supported" when a comparison table exists but fails the 70 % bar; H3 needs ≥ 3 regions, else "not evaluable" (one region is not a regional comparison).
8. **Test-suite stall was environmental**, not a project bug: an exported `SSLKEYLOGFILE` makes this Python build abort on the first
   ssl context (`OPENSSL_Uplink`), which pytest shows as a hang. `tests/conftest.py` removes it. Full suite passes.
9. **S14 also fails a bundle whose tracks lack artist names**, and `Melomane.neighbors` no longer treats all unknown artists as one artist
   (that made `exclude_artist` blank the whole catalogue: the "empty neighbours" S14 failure).
10. **Explorer**: restyled (glow + collision-free labels + curved neighbour links + auto-fit view). The guided tour and the About text no
   longer contain typed-in claims ("cos = 0.31", "blends into Coldplay's Paradise & Trouble"); every sentence is computed from the loaded payload.
   Counts in the page come from the manifest. *Why:* `CLAUDE.md` §4 — no number typed into docs by hand.
11. **Repo prepared for GitHub but nothing pushed**: `LICENSE` (MIT), `DATA_LICENSE.md`, `CITATION.cff` (preprint pending), CI, `.gitignore`
   (bundles, local notes). `docs/HANDOFF.md` is personal/agent-internal and is excluded from the public repo.
12. **arXiv**: not referenced; the paper/README say "preprint pending".

13. **Curation (S1) stays rules-only for this run; training starts before any LLM stage.** The OpenRouter key ran out of credit (HTTP 402)
    during S1: 1,300 of 1,333 refinement requests failed, the same failure that cost S2 most of its labels on 2026-10-06 (the
    client previously hid this behind generic retries; it now stops at the first 402, see `CreditsExhausted`). Training (S5) needs no LLM
    and takes ≈7.5 GPU-hours, while a late LLM curation pass would change the vocabulary and force a re-train; curation changed
    <1 % of tracks even in this degraded state (53,787 → 53,309 incl. rules and the 33 successful requests). So: train on the current
    vocabulary, mark curation **degraded** (paper limitation, `S1_curation.json: degraded`), and run the credit-dependent stages (S2
    labels, S6 mode names, S1 audits) when credit is restored. A full-LLM curation is then an E7 ablation, not a silent replacement.
    *Alternatives rejected:* waiting for credit (idle GPU, unknown delay); re-running S1 later in place (invalidates every downstream result).
14. **S4 is PARTIAL by 0.12 percentage points** (max compound cluster = 5.12 % of tracks, limit 5 %). Recorded, not tuned (the sensitivity
    grid E8 covers k); `PLAN.md` §6 S4 acceptance is therefore reported as narrowly missed.
15. **S3 same-artist edge share is 12.2 %** with real artists (it was 0 only because artists were unknown); the artist-masked Cooc ablation removes them.

16. **Provenance of the artist/title table (answer received 2026-10-09, not reconcilable).** The owner states the table came from
    Zenodo record 5002584. That record is *Playlist2vec* (Papreja 2020): a Spotify-API scrape (10.7 GB SQL, keyed by Spotify URIs;
    CC BY 4.0 as licensed by its uploader). The table in the project is keyed by **Music4All-Onion track ids** and has Music4All fields
    (`m4a_genres`, `m4a_tags`, `pseudo_caption`), which a Spotify dump cannot contain, so the two do not match. Playlist2vec is also the
    corpus this project excludes from publication (Spotify-derived; `configs/data.yaml: release`). `names_cleared` therefore **stays false**: no
    public site with names until the real source is identified. Options for the owner: (a) find where the table was downloaded from
    (browser history, or the Music4All-Onion page on Hugging Face / GitHub), (b) ask the Music4All authors (`docs/email_music4all.md`),
    (c) publish a names-free explorer.

17. **LLM labelling: cost, truncation bug, and sampling (2026-10-09).** (a) The labelling requests asked for 50 tracks at ~208 output tokens
    each (~10.4k) against the client's 8,192 `max_tokens`: replies were truncated mid-JSON, failed validation and were re-bought on each
    retry (≈$0.53 wasted in ten minutes; the cause of the 2026-10-06 failures). Fix: labelling requests carry `max_tokens=16384`
    (`annotation.ANNOTATE_MAX_TOKENS`); truncated replies now raise `LLMTruncated` without retrying; paid-but-unusable spend is tracked
    (`LLMClient.wasted_usd`). (b) The earlier cost pilot ($0.157 per 1,000 tracks) counted only successful calls. Measured cost is
    **≈$0.82–0.86 per 1,000 tracks**, so labelling all 53,309 tracks would cost ≈$45: more than the $30 cap and more than the credit
    available (account balance read from `/api/v1/credits`: $12.4). (c) Therefore S2 labels a **seeded random sample** that fits the budget
    (the vocabulary prefix would be the most-played ~1,200 artists, which starves artist-grouped evaluation); unlabelled tracks keep the
    rule-based category (used for ICA residualisation) and are never fabricated; every LLM-label result reports its coverage.
    Adding credit later extends the sample from the cache. (d) *Known weakness, not fixed during the run:* a stage's staleness hash covers the
    **whole profile**, so editing any profile key (e.g. `llm.annotate_batch`) marks every stage stale and would re-ingest and retrain. The
    original author worked around it with a shortcut I removed (decision 5); the proper fix is a per-stage profile-key list, to be done
    between runs. I avoided triggering it here by fixing the bug in code, not in the profile.

18. **Provenance of the artist/title table: identified (2026-10-09, evening).** Neither Zenodo record the owner named (5002584 Playlist2vec;
    6609677 Onion v1, 41 files) contains artist/title. The table matches the Hugging Face dataset **`seungheondoh/enrich-music4all`**
    (Doh; created 2024-01-07) on every checkable point: identical columns (`track_id, tag_list, pseudo_caption, title, artist_name,
    release, m4a_genres, m4a_tags`) and identical size (86,410 train + 21,953 test = 108,363 rows = our table). The match is by schema and row
    count (inferred, not byte-compared). The dataset card **declares no licence**; its metadata derive from Music4All (Santana et al. 2020),
    whose terms are not stated on its page, and its `pseudo_caption` column is LLM-generated text that this project drops. Artist and title
    strings are factual metadata, but their redistribution terms are *unspecified*, not granted. Therefore `names_cleared` stays **false**
    until the owner decides to publish them with attribution (Music4All, enrich-music4all) and, ideally, after a courtesy email to the
    Music4All authors (`docs/email_music4all.md`). The fallback is a names-free explorer.
19. **Unattended tail** (`python -m melomane.unattended`): holds the machine awake (Windows `SetThreadExecutionState`, no settings changed),
    waits for the S5 manifest, then runs S6-S9 and S12-S14 with the LLM spend capped to $1.5 above what is already spent, so a night run
    cannot exceed the credit left on the account.
