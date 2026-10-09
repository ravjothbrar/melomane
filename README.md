# MELOMANE: Emergent Geometry & Multidimensional Structure in Music Embeddings

[![Status: Validated](https://img.shields.io/badge/Pipeline-Validated%20(S0--S14)-brightgreen.svg)]()
[![Hardware: RTX 5080](https://img.shields.io/badge/Hardware-NVIDIA%20RTX%205080%20(sm__120)-76B900.svg)]()
[![Platform: Windows 11](https://img.shields.io/badge/Platform-Windows%2011%20Native-0078D4.svg)]()
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-blue.svg)]()

> **Music-domain replication and extension of the two Epicure papers:**
> - *Epicure I: The Multidimensional Flavor Structure of Ingredients* (arXiv:2604.22776)
> - *Epicure II: Emergent Geometry in Ingredient Embeddings Enables Navigating the Flavor Space* (arXiv:2605.22391)

Melomane adapts the entire Epicure analysis stack to musical tracks: discovering canonical vocabulary, contrasting listening co-occurrence against acoustic and lyrical composition, isolating independent semantic and sonic modes via FastICA, steering playlists via spherical linear interpolation (SLERP), and publishing an automated, reproducible white paper.

---

## Key Features

1. **Controlled Three-Model Spectrum:**
   - **Melomane-Cooc:** What people listen to together (derived from session-level positive NPMI).
   - **Melomane-Composition:** What tracks are physically made of (sonic compound clusters: timbre, spectral shape, rhythm groove, tonal harmony, loudness dynamics + lyrical themes/lexicons).
   - **Melomane-Core:** A 10:1 walk blend of co-occurrence and composition (`ii_repeat = 10`), matching Epicure-Core.
2. **Methodological Rigor & Safeguards:**
   - **Artist Leakage Protection:** Cross-validation strictly grouped by `artist_id`; kNN purity strictly artist-disjoint.
   - **100% LLM-Free Embeddings:** Embeddings are learned strictly via GPU-accelerated Skip-Gram with Negative Sampling (SGNS). OpenRouter (Gemini Flash) is restricted to vocabulary normalization, metadata enrichment, and mode naming with a hard `$30 USD` spend cap.
   - **Copyright & Privacy Hygiene:** Zero raw audio or lyrics are stored or redistributed.
3. **Turnkey Publication Pipeline:**
   - Automated paper generation (`S13`): Generates arXiv LaTeX (`paper/melomane_paper.tex`) and GitHub Markdown (`paper/melomane_paper.md`) with figures (`paper/figures/`) and tables (`paper/tables/`) read dynamically from `results/` without hand-typed figures.
   - Automated Release QA (`S14`): Audits secret leaks, copyright compliance, placeholder absence, and bundle deserialization.

---

## Quickstart (Windows 11 Native)

The repository provides a turnkey batch launcher `run_melomane.bat` that configures environment variables (`UV_SYSTEM_CERTS=1`), manages Python dependencies, prevents system sleep during long training runs, and executes pipeline stages.

### 1. Environment & API Key
Ensure `.env` exists with your OpenRouter key:
```bash
OPENROUTER_API_KEY=sk-or-v1-...
```
*(Spend is capped at `$30 USD` in `configs/llm.yaml` and `.env`)*

### 2. Verify Everything (Smoke Test in < 2 Minutes)
Run the full S0 through S14 pipeline on the synthetic planted corpus:
```cmd
run_melomane.bat smoke
```
This runs doctor checks, builds graphs, trains GPU SGNS embeddings, extracts FastICA modes, executes the evaluation battery, evaluates SLERP arithmetic, computes compression frontiers, and compiles the white paper deliverables.

### 3. Check Status
```cmd
run_melomane.bat status
```

### 4. Run Full Overnight Training
To run the unattended, end-to-end training pipeline on the primary corpus:
```cmd
run_melomane.bat
```
*(or `run_melomane.bat train`)*

This keeps your machine awake, checkpoints every epoch, and writes all results to `results/` and `paper/`.

---

## Command Reference

| Command | Action |
|---|---|
| `run_melomane.bat` | Full turnkey overnight run: setup -> smoke verification -> status -> standard training |
| `run_melomane.bat smoke` | Fast end-to-end pipeline run (S0..S14) on synthetic data |
| `run_melomane.bat doctor` | Diagnostic audit of Python, PyTorch, CUDA 12.8 (RTX 5080 sm_120), and API keys |
| `run_melomane.bat status` | Table of completed and pending stages across corpora |
| `run_melomane.bat setup` | Verify and install Python dependencies via `uv` |
| `run_melomane.bat serve` | Launch local static WebGL SLERP Explorer (port 8765) |

---

## Datasets

- **Primary Dataset (Core):** **Music4All-Onion v2** (Zenodo, CC BY 4.0).
  - 109,269 tracks with 253M listening interactions across 119,000 users.
  - Aligned Essentia acoustic descriptors, lyrics word embeddings, and tags with 100% compound hub coverage.
- **Secondary / Scaling:** Local Spotify 10.66GB MySQL dump (`spotifydbdumpshare.sql` in `~/Downloads` from Zenodo `playlist2vec`) for co-occurrence graph scaling.
- **Synthetic Ground Truth (`synth`):** Planted multi-modal latent structure for deterministic testing and CI.

---

## Project Structure

```
<repo>\
|-- configs/               # Hyperparameter profiles (standard, paper, smoke), families, LLM config
|-- data/                  # Ingested and processed datasets (git-ignored)
|-- paper/                 # White paper deliverables
|   |-- melomane_paper.tex # arXiv LaTeX preprint
|   |-- melomane_paper.md  # GitHub Markdown paper
|   |-- reproducibility.md # System manifest and run provenance
|   |-- figures/           # Auto-generated publication charts (.png)
|   +-- tables/            # Auto-generated CSV and Markdown tables
|-- release/               # Model bundles (safetensors, modes, poles, config) + WebGL explorer
|-- results/               # Single source of truth for all metrics (.json, .parquet)
|-- scripts/               # bootstrap.py and diagnostic benchmarks
|-- src/melomane/          # Core research library
|   |-- pipeline/          # Orchestration engine and runner
|   |-- stages/            # Pipeline stages S0 through S14
|   |-- torch_sgns.py      # CUDA 12.8 GPU-accelerated Skip-Gram with Negative Sampling
|   |-- slerp.py           # Spherical Linear Interpolation arithmetic
|   |-- ica_modes.py       # FastICA and GMM semantic mode extraction
|   +-- probes.py          # Artist-grouped linear and ridge probe battery
|-- tests/                 # 60+ pytest unit and property tests
|-- run_melomane.bat       # Native Windows one-click execution batch script
|-- PLAN.md                # Master specification & build plan (v10)
+-- DECISIONS.md           # Architecture and technical decisions log
```

---

## Hardware & Environment

- **OS:** Windows 11 Home / Pro (64-bit)
- **GPU:** NVIDIA GeForce RTX 5080 Laptop GPU (Blackwell architecture, compute capability `sm_120`)
- **Memory:** 64 GB RAM
- **CUDA / PyTorch:** `torch==2.11.0+cu128` (CUDA 12.8 wheel)
- **Dependency Isolation:** Managed with `uv` and Python 3.12 (`scikit-learn` pinned to 1.5.2 for Windows Code Integrity compatibility).
