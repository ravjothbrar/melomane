# Melomane

**Do the structures that Epicure found in food embeddings also appear in music embeddings?**
Melomane is a music-domain replication and extension of the two Epicure papers
([arXiv:2604.22776](https://arxiv.org/abs/2604.22776), [arXiv:2605.22391](https://arxiv.org/abs/2605.22391)).
It trains three skip-gram models on the same tracks, with the same objective, that differ only in what they are allowed to read, and
tests a set of hypotheses that were written down before training (`analysis_plan.lock.json`).

> **Status: work in progress, preprint pending.** The numbers below are generated from the result files; nothing in this README is
> typed by hand. Read the limitations before drawing conclusions.

| Model | Learns from |
|---|---|
| **Melomane-Cooc** | listening sessions only (which tracks the same people play in one sitting) |
| **Melomane-Composition** | measured audio and lyric features only (clusters of timbre, rhythm, harmony, loudness and lyric descriptors), no listening data |
| **Melomane-Core** | both, with listening weighted ten to one |

The embeddings are LLM-free. A language model is used only to curate and tag tracks and to name the emergent modes. Lyrics and audio are
never redistributed and never sent to a language model.

## What is measured

* **Artist leakage is treated as the main trap.** Tracks by one artist share voice and audience, so every probe is cross-validated with
  folds grouped by artist and nearest-neighbour purity excludes the query's artist.
* **Probes come in three strata**: properties of the very descriptors that build the compound vocabulary (a), properties held out of it
  (voice, emotion, lyric sentiment) (b), and LLM-labelled musical dimensions validated against instrument-measured features (c).
* **Hypotheses H1–H8** are reported whatever the outcome, including the ones that fail.
* **Steering** uses spherical interpolation (SLERP) toward a pole direction, with a quantitative test of how retrieval changes with angle.

## Headline results

<!-- BEGIN GENERATED RESULTS -->
_Not generated yet. Run `python -m melomane run-all --corpus onion --profile standard`; stage S13 fills this section from `results/`._
<!-- END GENERATED RESULTS -->

Full tables, CIs, extensions and the limitations are in [`paper/melomane_paper.md`](paper/melomane_paper.md).

## Explorer

A static, dependency-free WebGL2 explorer lives in [`apps/explorer`](apps/explorer): search a track, see its neighbours in each model,
and rotate the query toward a genre, a mode or an emergent direction. Everything is computed in the browser from a small int8 payload;
the 2-D layout (UMAP) is for display only. It is a large-screen tool: phones are shown an explanation and a copy-link button instead (tablets, laptops, desktops and unfolded foldables work).

```bash
python -m melomane serve --dir release/onion/standard/explorer --port 8765
```

(The payload is built by stage S12. Python's stock `http.server` can drop parallel requests on Windows; use the command above.)

## Reproduce

Windows 11 or Linux, Python ≥ 3.10, an NVIDIA GPU is strongly recommended (CUDA ≥ 12.8 for RTX 50-series).

```bash
uv sync                                            # or: python -m venv .venv && pip install -e ".[dev]"
cp .env.example .env                               # optional: OPENROUTER_API_KEY for the LLM stages
python -m melomane doctor                          # GPU, disk and LLM smoke test
python -m melomane demo                            # whole pipeline on a synthetic corpus (minutes)
python -m melomane run-all --corpus onion --profile standard   # the real run (hours; resumable, idempotent)
python -m pytest -q -m "not gpu"                   # tests
```

The real run downloads Music4All-Onion (≈ 20 GB) from Zenodo. Artist and title strings are **not** part of that record; see
[`DATA_LICENSE.md`](DATA_LICENSE.md) and [`DECISIONS.md`](DECISIONS.md) §7 for how they are handled.

## Data, licences, attribution

* Code: MIT ([`LICENSE`](LICENSE)). Derived artefacts: CC BY 4.0 subject to [`DATA_LICENSE.md`](DATA_LICENSE.md).
* Built on **Music4All-Onion** (Moscati, Parada-Cabaleiro, Deldjoo, Zangerle, Schedl; CIKM 2022; Zenodo 15394646; CC BY 4.0), whose
  listening events come from Last.fm.
* No audio, no lyrics, no per-user data are included.

## Repository map

```
src/melomane/      library: cooc, sessions, compounds, walks, torch_sgns, geometry, slerp, probes, ica_modes, llm, api, paper_text
  adapters/        Onion, MusicBrainz-stack, Jamendo, FMA, playlist adapters
  stages/          S0 … S14 pipeline stages (idempotent, resumable)
  analysis/        paired statistics, WEAT, Procrustes, compression, attribution, direction test, sensitivity grid
apps/explorer/     the static WebGL2 explorer;  apps/mcp_server/  read-only MCP server
configs/           profiles (smoke / standard / paper), LLM and data settings
PLAN.md            the specification;  CLAUDE.md  how the pipeline is operated;  DECISIONS.md  every decision and deviation
results/ reports/  generated; the single source of every reported number
```

## Citing

See [`CITATION.cff`](CITATION.cff). A preprint is pending; this section will be updated when it is posted.
