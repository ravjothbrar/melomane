# PROGRESS

> **Read this first.**
> * This is the run log of the current `onion/standard` run (started 2026-10-09), newest last. It is written by the pipeline after every
>   stage; the committed copy is curated (demo-corpus runs and earlier, superseded runs are omitted, local paths removed).
> * The earlier run (2026-10-05/06) used one artist per track, so its artist-grouped evaluation was a no-op, and a reduced training
>   profile. It is superseded: see `DECISIONS.md` section 7.
> * Known degradations of this run are stated in the stage entries and collected in the paper's limitations: curation is rules-only
>   (the LLM refinement ran out of credit), LLM labels cover a random sample of the vocabulary, and one compound cluster narrowly exceeds
>   the 5 % size limit.

Auto-written by the Melomane pipeline (newest last).

## 2026-10-09 15:28 UTC — S0-ingest (Ingest raw data into the corpus contract) [onion/standard] — GREEN
Ran in 2357s. Key numbers: tracks=56512, sessions=12683839, session_track_links=102532208, families=7, has_measured=True
Resume: `python -m melomane run --stage S0-ingest --corpus onion --profile standard` · disk free 1654 GB

## 2026-10-09 15:28 UTC — S0-enrich (Match tracks to a feature corpus (partial-hub compounds)) [onion/standard] — GREEN
Ran in 0s. Key numbers: -
Notes: feature corpus: nothing to enrich
Resume: `python -m melomane run --stage S0-enrich --corpus onion --profile standard` · disk free 1654 GB

## 2026-10-09 15:35 UTC — S1 (Canonical vocabulary (curation)) [onion/standard] — PARTIAL
Ran in 413s. Key numbers: V=53309, consolidation_ratio=1.009, removed=26, audit_agreement_merge=None
Notes: 1300/1333 refinement requests failed; those items kept unmerged
Resume: `python -m melomane run --stage S1 --corpus onion --profile standard` · disk free 1654 GB

## 2026-10-09 15:36 UTC — S3 (Behaviour graph (session NPMI)) [onion/standard] — GREEN
Ran in 62s. Key numbers: n_sessions=12683611, mean_degree=148.4, same_artist_edge_share=0.1222, n_isolated=662
Resume: `python -m melomane run --stage S3 --corpus onion --profile standard` · disk free 1654 GB

## 2026-10-09 15:37 UTC — S4 (Compound graph (sonic / lyrical families)) [onion/standard] — PARTIAL
Ran in 51s. Key numbers: n_compounds=3300, max_cluster_share=0.05125, tracks_with_compounds=1
Notes: cluster sanity: empty=0, max share=0.051 (limit 0.05)
Resume: `python -m melomane run --stage S4 --corpus onion --profile standard` · disk free 1654 GB

## 2026-10-09 17:43 UTC — S2 (LLM annotation & measured ground truth) [onion/standard] — PARTIAL
Ran in 2099s. Key numbers: pilot_usd_per_1k_tracks=0.8142, coverage=1, spent_usd=10.04, h5_share=0.8333
Deviations: projected LLM spend $61 > remaining $10: annotating a random sample of 8,504 tracks only
Notes: dimensions below reliability 0.6 (excluded from headline probes): ['space']
Resume: `python -m melomane run --stage S2 --corpus onion --profile standard` · disk free 1652 GB

## 2026-10-09 22:49 UTC — S5 (Walks & SGNS training (models, seeds, ablations)) [onion/standard] — GREEN
Ran in 25863s. Key numbers: models_trained=12, recall_cooc=0.573, recall_composition=0.4207, recall_core=0.4627
Resume: `python -m melomane run --stage S5 --corpus onion --profile standard` · disk free 1650 GB

## 2026-10-09 23:04 UTC — S6 (ICA, modes & poles) [onion/standard] — FAILED
Ran in 862s. Key numbers: -
Notes: CreditsExhausted: OpenRouter credits exhausted (HTTP 402 earlier in this run): top up the key's credit; cached answers still work
Resume: `python -m melomane run --stage S6 --corpus onion --profile standard` · disk free 1650 GB

## 2026-10-09 23:09 UTC — S12 (Products (bundles, explorer, MCP config)) [onion/standard] — PARTIAL
Ran in 291s. Key numbers: corpora=1
Notes: onion: bundle export failed: ValueError: embeddings have 53761 rows but vocab has 53309 | onion: explorer build failed: ValueError: cooc_raw: embeddings rows 53761 != vocab 53309
Resume: `python -m melomane run --stage S12 --corpus onion --profile standard` · disk free 1649 GB

## 2026-10-09 23:09 UTC — S13 (White paper generation (tables, figures, MD/LaTeX)) [onion/standard] — GREEN
Ran in 3s. Key numbers: primary_corpus=onion, figures_generated=0, hypotheses_evaluated=1
Notes: Paper generated in <repo>\paper/ (Markdown + LaTeX arXiv preprint)
Resume: `python -m melomane run --stage S13 --corpus onion --profile standard` · disk free 1649 GB

## 2026-10-09 23:09 UTC — S14 (Release QA (secrets, privacy, integrity checks)) [onion/standard] — GREEN
Ran in 19s. Key numbers: all_passed=True, issues_count=0, secrets_hits=0
Resume: `python -m melomane run --stage S14 --corpus onion --profile standard` · disk free 1649 GB

## 2026-10-09 23:22 UTC — S6 (ICA, modes & poles) [onion/standard] — PARTIAL
Ran in 682s. Key numbers: models=7, modes_cooc=140, modes_composition=140, modes_core=140, modes_composition_acoustic=133, modes_composition_lyric=140, modes_cooc_artistmasked=140, modes_cooc_avg=140
Questions for Ravjoth: S6: the OpenRouter key is out of credit (HTTP 402), so the LLM part was skipped (unlabelled items stay unlabelled, results are marked degraded). Top up the key's credit limit (configs/llm.yaml caps spend at $13), then resume with `python -m melomane run --stage S6 --corpus onion --profile standard --force` (cached answers make this cheap).
Resume: `python -m melomane run --stage S6 --corpus onion --profile standard` · disk free 1649 GB
