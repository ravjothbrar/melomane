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
