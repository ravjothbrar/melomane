# MELOMANE — Master Spec & Build Plan (v10)

*Music-domain replication and extension of the two Epicure papers (arXiv:2604.22776, arXiv:2605.22391).*
Planning status: **fully operationalized, pipeline stages S0–S14 validated on Windows RTX 5080 (sm_120) with automated one-command execution via `run_melomane.bat`**.

Companion files: `CLAUDE.md` · `DECISIONS.md` · `docs/epicure_reference.md` · `docs/data_sources.md` · `README.md` (one-command setup).

---

## 1. Summary

**What we build.** A research-grade, fully open pipeline that discovers and validates interpretable structure in music-track embeddings the way Epicure did for food:

1. LLM-augmented **curation** of a raw track vocabulary into canonical tracks (paper 1's 8 stages).
2. Three 300-D skip-gram models on **one controlled spectrum** (paper 2): **Melomane-Cooc** (what people listen to together), **Melomane-Composition** (what tracks are *made of*: measured audio + lyric "compounds"), **Melomane-Core** (10 : 1 blend).
3. The full analysis stack: LLM-labelled dimensions validated against *instrument-measured* features, kNN-purity curation lift, probe battery in three strata, FastICA → GMM **modes**, **SLERP** direction arithmetic, geometry (PR, cosine, NMI), WEAT/Procrustes supplement analyses.
4. Extensions the papers never ran: **compression/quantisation frontier** (small model, ≥ 90 % accuracy), acoustic-vs-lyric attribution, equal-token control, external-embedding agreement (MAEST/MusiCNN/Jukebox), cross-catalog transfer, temporal stability, artist-leakage controls.
5. Products & Publication: Python package + Hugging Face release mirroring Epicure's artefacts, an interactive SLERP explorer, read-only MCP server, and automated white paper compilation (`S13`: arXiv LaTeX `melomane_paper.tex` + Markdown `melomane_paper.md`) with all numbers populated dynamically from `results/`, backed by release QA (`S14`).

**Data.** *Primary Dataset (core, fully open, no signup):* Music4All-Onion v2 — 109,269 tracks with listening logs (253 M records / 119 k users), track-level audio features, lyrics features and tags for the **same** tracks (100% compound hub coverage). *Scaling / Secondary:* Local 10.66GB Spotify `playlist2vec` MySQL dump (`spotifydbdumpshare.sql` in `~/Downloads`). *Track B (scale/recency):* MBID-native open stack. *Track C (generalisation):* MTG-Jamendo + FMA.

**Compute.** Windows 11 Native, 64 GB RAM, NVIDIA RTX 5080 Laptop GPU (sm_120), fast local NVMe SSD (`<repo>`). Only one secret: an OpenRouter key (Gemini 3.7 Flash, hard-capped at `$30 USD`).

**Definition of success.** (a) Every stage's acceptance tests pass on real data; (b) results are reported honestly against pre-registered hypotheses (nulls included); (c) a stranger can reproduce the smoke profile from a fresh clone via `run_melomane.bat smoke`; (d) artefacts and paper are ready to publish.

---

## 2. What Ravjoth must do (the only human actions)

| # | When | Action | Time | Why it can't be automated |
|---|---|---|---|---|
| 1 | Before first run | **Native Windows setup (Done):** Python 3.12 via `uv`, PyTorch `2.11.0+cu128` (NVIDIA driver ≥ 570 for RTX 5080). Verified on GPU. Run `run_melomane.bat setup` or `run_melomane.bat doctor`. | 5 min | Local machine |
| 2 | Before first run | **OpenRouter API Key:** Put key in `.env` as `OPENROUTER_API_KEY`. Spend is strictly hard-capped at `$30 USD` in `configs/llm.yaml`. | 2 min | Payment/identity |
| 3 | Start training | **Double-click or run `run_melomane.bat`** (or `run_melomane.bat train`): runs setup, smoke verification, status check, and launches full unattended overnight training with keep-awake enabled. | 10 sec | Local launcher |
| 4 | End of S12–S14 | **Approve publishing** (GitHub repo, Hugging Face org/repo, static hosting for the explorer). Needs *your* accounts — nothing is published without an explicit "yes". | 20 min | Accounts / consent |

Everything else — downloads, curation, training, evaluation, figures, LaTeX paper compilation, release QA — is automated. No other accounts or API keys are required to build.

---

## 3. What the research changed (v8 → v9)

| Finding | Consequence |
|---|---|
| Both official Spotify datasets (MPD, MSSD) are gone from AIcrowd; Playlist2vec is an unofficial re-scrape | **Dropped as core.** Track A uses Music4All-Onion instead |
| Music4All-Onion is open (CC BY 4.0, Zenodo), has behaviour + audio + lyrics + tags for the *same* 109 k tracks, no rate limits | Solves the core mismatch of the old plan (co-occurrence data and audio/lyrics for *different* tracks) |
| Paper 1's real pipeline has *human-in-the-loop QC* and *embedding averaging*, Paper 2 has WEAT/Procrustes/orthogonal-residual analyses | Stage list rebuilt to mirror them (§6) |
| Paper 2's exact hyper-parameters are public (walks 100, length 50, window 7, 5 neg., lr 0.0025, 20 ep., batch 32,768) | `paper` profile reproduces them; `standard` is the scaled default |
| At 109 k nodes the paper's walk budget ≈ 2,000 GPU-h/model | GPU SGNS backend + scaled profile; equal-token control to prove scaling doesn't change conclusions |
| Essentia has no Windows wheels; Onion/AcousticBrainz ship precomputed Essentia-family features | Essentia is optional; audio never needs downloading for Tracks A/B |
| AcousticBrainz has small CSV feature dumps (29.5 M rows) besides the ~1 TB JSON | Track B starts light, streams JSON only for candidates |
| iTunes API ≈ 20 req/min | Only used client-side (visitor's IP) for preview playback |
| Music is not food: same-artist tracks share labels | **Artist-grouped CV + artist-disjoint kNN purity** everywhere (tested) |
| Spotify Developer terms restrict ML on Spotify content | Irrelevant to Track A (Last.fm-derived, CC BY); Track B avoids Spotify data entirely |
| Lyrics are copyrighted | **Never redistribute or display lyrics.** Explorer shows LLM one-line themes only |
| MLHD+ is non-commercial | Not used for published artefacts |

---

## 4. Design

### 4.1 Faithfulness map

| Epicure element | Melomane |
|---|---|
| Ingredient | Canonical **track** (after consolidating versions/duplicates) |
| Recipe (document) | Listening **session** (gap 30 min, 5–200 distinct tracks) |
| NPMI ingredient graph (≥ 20 recipes, positive NPMI; 203,508 edges) | Session NPMI graph (`min_count` 20 sessions, `min_pair_count` ≥ 5, positive NPMI, top-k 250/track); target average degree 100–250 |
| FlavorDB compounds (2,247 typed; 15 categories; 80,019 edges) | **Sonic + lyrical compounds**: k-means clusters per feature family (§4.4), ≈ 2,800 typed nodes; each track links to its top-3 clusters per family |
| 525/1,790 hub ingredients | 100 % hubs by construction; partial-hub regime as ablation E3 |
| Epicure-Cooc / Chem / Core | **Melomane-Cooc / Composition / Core** (`ii_repeat` 1 / 0 / 10) |
| Food-group residualisation for ICA | **Category (genre-family) residualisation** |
| 8 cuisine macro-regions ("only if distinctive marker") | Cultural/scene macro-regions from LLM tags, same rule |
| LLM taste/texture dimensions (5-level) validated on USDA/FooDB | LLM music dimensions (§7.1) validated on **Onion/Essentia measured features** (tercile centroids, no LLM) |
| Chemical-hub ablation | Direct-vs-indirect compound coverage ablation (E3) |
| Curated-vs-raw kNN-purity lift (6.2×) | Same, artist-disjoint, size-matched bootstrap |
| SLERP with supervised + emergent poles | Same + multi-pole composition + a *quantitative* direction test |
| HF release (safetensors, modes, poles) + Explorer + MCP | Same file conventions (§9) |

### 4.2 Corpora
- **Track A — Music4All-Onion v2 (core).** All headline results. ~80–105 k canonical tracks expected after curation.
- **Track B — MBID-native scale-up (S10).** Same schemas via an adapter; 300 k–1 M tracks; more recent listening (ListenBrainz to 2026). Headline claim only if Track A conclusions replicate.
- **Track C — MTG-Jamendo + FMA (S11).** No behaviour data → cannot train Cooc; used for generalisation of the sonic-compound vocabulary and for tag-prediction transfer. Requires an *identical feature definition* (Essentia/AcousticBrainz-style); if Onion's `id_essentia` keys don't align with AcousticBrainz's, run the transfer inside Track B instead and say so.

### 4.3 Models (all share vocabulary, dim 300, SGNS; only the walk template differs)
| Model | Templates per round | Epicure analogue |
|---|---|---|
| **Melomane-Cooc** | 1 × TT (track–track) | Epicure-Cooc |
| **Melomane-Composition** | TCT[f] for every family f (t–c_f–t–c_f–t…) + 1 cross-family bridge, **no TT** | Epicure-Chem |
| **Melomane-Core** | 10 × TT + all compound templates | Epicure-Core (`ii_repeat` = 10) |
Ablations (same hyper-parameters): Composition-acoustic-only, Composition-lyric-only, Core-acoustic-only, **equal-token-budget** versions of all three, **Cooc-artist-masked** (co-occurrence pairs from the same artist removed), **partial-hub Core/Composition** (30 % of tracks keep compound edges; others reach compounds through TT hops, i.e. N–H–C–H–N).

### 4.4 Compound families (Track A; **freeze in `analysis_plan.lock.json` before any training**)
Stage 0 inventories the real Onion files; the table is the default assignment. Blocks are z-scored, wide blocks → whitened PCA(32) → MiniBatch k-means → top-3 soft edges.
| Family | k | Source files (default) |
|---|---|---|
| `timbre` | 500 | `id_mfcc_stats`, `id_compare_mfcc_stats` |
| `spectral_shape` | 500 | `id_blf_spectral`, `id_blf_spectralcontrast`, `id_blf_deltaspectral`, `id_blf_vardeltaspectral` |
| `rhythm_groove` | 500 | `id_blf_logfluc`, Essentia `rhythm.*` from `id_essentia` |
| `harmony_tonal` | 500 | `id_chroma_bow`, `id_blf_correlation`, Essentia `tonal.*` |
| `dynamics_loudness` | 500 | Essentia loudness/dynamic-complexity descriptors, `id_compare_pcm_stats` |
| `lyric_theme` | 400 | `id_bert` (PCA) + `id_lyrics_word2vec` |
| `lyric_lexicon` | 400 | `id_lyrics_tf-idf` → NMF topics |
**Never used to build compounds (held out for probes/validation):** openSMILE voice (`f0`, `hnr`, `jitter`, `shimmer`, `voice`), `emobase` voice/LSP, `id_gems`, `id_lyrics_sentiment_functionals`, `id_vad_bow`, and all learned embeddings (`id_jukebox`, `id_maest`, `id_musicnn`, `id_ivec*`, `id_incp`, `id_resnet`, `id_vgg19`). Tracks without lyrics (instrumentals) have no lyric edges — logged, reported, side-analysed.

---

## 5. Environment, repo, data contracts

### 5.1 Machine
- **Path 1 (recommended): WSL2 Ubuntu 24.04**, project in `~/melomane`, Python 3.11/3.12 venv (`uv` or `python -m venv`), `pip install -e ".[dev]"`, PyTorch from the **cu128-or-newer** index, NVIDIA driver on Windows exposes the GPU to WSL. Essentia via pip (Linux wheel) — optional.
- **Path 2: native Windows.** Same, minus Essentia. Use `pathlib` everywhere; no shell-isms in Python code; avoid symlinks; long-path support on.
- Stage 0 runs `scripts/smoke_env.py` and `scripts/bench_gpu.py`; if the CUDA build fails on sm_120, try nightly cu128/cu129 once, else fall back to gensim (`melomane.embed`, CPU) and log the slowdown in `PROGRESS.md`.

### 5.2 Repo layout
```
src/melomane/{cooc,sessions,compounds,walks,embed,torch_sgns,geometry,slerp,probes,ica_modes,curation,llm,synth}.py   # provided + tested
src/melomane/{adapters/,pipeline/,analysis/,release/}                       # you build (adapters: onion.py, mbid_stack.py, jamendo.py, fma.py)
apps/{review_app.py (Streamlit), explorer/ (web), mcp_server/}              # products
configs/{profiles.yaml, llm.yaml, families.yaml}
data/{raw,interim,processed}/  cache/llm.sqlite  runs/<run_id>/  results/  reports/  review/  logs/
docs/  tests/  scripts/  PLAN.md  CLAUDE.md  PROGRESS.md  DECISIONS.md  analysis_plan.lock.json
```
`data/`, `cache/`, `runs/`, `logs/` are git-ignored. `results/*.json|parquet` are the single source of truth for every reported number.

### 5.3 Data contracts (Parquet; stable ids)
- `tracks.parquet`: `raw_id, cid (canonical), artist_id, artist_norm, title_norm, album, year, language, n_sessions, n_users, has_lyrics, is_instrumental, dev_split (artist-level 20 % dev / 80 % test)`.
- `consolidation_map.parquet`: `raw_id → cid, decision (rule|llm|human), group_id`.
- `labels.parquet`: per `cid`: LLM dimension levels (0–4 or NaN), region tags (multi-label), category, flags, `label_run`, `confidence`.
- `features_<family>.parquet`: `cid + float32 columns`; `features_heldout.parquet`; `features_manifest.json` (source file, column names, transforms, PCA/k).
- `sessions.npz`: CSR sessions × cid; `cooc_edges.parquet (i,j,npmi)`; `compound_edges.npz`; `compound_meta.parquet (comp_id, family, size)`.
- `runs/<run_id>/{embeddings.npy (fp32), config.json, manifest.json, train_log.jsonl}`.
- Manifests record: git commit, package versions, seeds, data file hashes (sha256), profile, device, wall-clock, token counts.

### 5.4 Determinism
Every random step takes a seed derived from `(run_id, stage, name)`. LLM calls: `temperature 0`, cached by hash of (model, prompts, schema). GPU non-determinism is accepted but **multi-seed** runs (≥ 3 seeds for headline models) quantify it; report mean ± sd.

### 5.5 Operating protocol (unattended, multi-night)
- `python -m melomane.pipeline run --stage S3 --profile standard` — every stage **idempotent and resumable** (skips work whose inputs/params hash matches a finished manifest).
- Long jobs (downloads, sessionising, training) run **detached** with logs in `logs/` and checkpoints (per-epoch for training); poll rather than block.
- After every stage: append to `PROGRESS.md` (what, numbers, decisions, blockers, exact resume command), commit, continue. Never stop to ask unless §13 says a human gate is hit.
- Budget guard: LLM spend read from the SQLite cache; stop LLM stages (not the run) at the cap and continue with cached answers.
- Disk: keep raw downloads, delete decompressed intermediates once Parquet is verified; `df` check before large steps.

---

## 6. Stages

Each stage lists **Do / Outputs / Acceptance (automated, must pass) / Fallback / Estimate (5080 machine)**. Do not start S*n+1* before S*n* is green and logged, except background downloads.

### S0 — Bootstrap, inventory, gates (≈ 0.5 day incl. 20 GB download)
**Do.** Create venv; install; `pytest` (all provided tests green); `scripts/smoke_env.py`; `scripts/bench_gpu.py` (records pairs/s → real runtime table in `reports/s0_bench.md`); OpenRouter smoke test (structured output, `reasoning` off switch, cost field) → write `configs/llm.yaml`; **LLM cost pilot** (label 200 tracks → extrapolate; abort plan if projected total > `LLM_BUDGET_USD`). Download all Onion files resumably (Zenodo API gives md5), verify checksums, parse `README.md`, write `data/raw/onion/INVENTORY.json` (file → shape, dtype, id semantics, 5 sample rows, feature names). Determine: metadata files present? timestamp unit and range? #tracks with lyrics? `processed_lyrics` format? whether `id_essentia` keys resemble AcousticBrainz's? Probe (do not download) the availability of: LRCLIB dump, ListenBrainz dump, AcousticBrainz dump URLs (HTTP HEAD) for Track B.
**Outputs.** `reports/s0_inventory.md`, `reports/s0_bench.md`, `configs/llm.yaml`, `DECISIONS.md` entries.
**Acceptance.** Checksums match; inventory has every file; smoke tests PASS or documented WARN; cost projection ≤ budget.
**Gate — Blocker-A:** if the open Onion release has **no artist/title metadata**, do *not* stall: continue with tag/genre-only labelling for S1–S2 (flag results), draft the follow-up to the Music4All authors, and log a question for Ravjoth. Everything not needing titles proceeds.

### S1 — Canonical vocabulary (paper-1 stages 1–8 analogue) (≈ 0.5 day + optional 1 h human QC)
**Do.** Implement the 8 stages with per-stage counts (report as a table; headline = consolidation ratio, cf. paper's 6.4×):
1. *Simplification* — `curation.canonical_key` (strict policy), exact groups, `candidate_groups` (Jaccard/subset within artist) → LLM batches of ≤ 40.
2. *Refinement* — Gemini JSON: `keep / remove / consolidate(group)`; remove non-music (spoken word, white noise, sleep/nature sounds, DJ continuous mixes, karaoke/tribute).
3. *Ultra-refinement* — long rule prompt (analogue of the 450-line prompt): remix/cover/instrumental/sped-up never merged; classical works & movements; soundtracks; transliteration; live/acoustic policy (**strict default; song-level as ablation**).
4. *Human-in-the-loop QC* — stratified 5–10 % sample → `review/queue.parquet`, `apps/review_app.py` (Streamlit: search, batch accept/edit); `review/corrections.parquet` merged on rerun. Pipeline does **not** wait.
5. *Consolidation* → `cid`, `consolidation_map.parquet`.
6. *Embedding averaging* — after S5: train a **raw-vocabulary Cooc** model (all raw ids), average per `cid` (`geometry.consolidate_embeddings`) → "curated by averaging" embeddings for the paper-1 analyses.
7. *Categorisation* — 16–20 music categories (Rock, Pop, Hip-Hop, R&B/Soul, Electronic/Dance, Metal, Punk/Hardcore, Indie/Alt, Folk/Acoustic, Country, Jazz, Blues, Classical/Score, Reggae/Dub, Latin, World/Regional, Ambient/Chill, Soundtrack…) + flags `has_vocals`, `explicit_lyrics`, `is_instrumental`. Categories are the ICA residualisation groups.
8. *Pairwise similarity* — top-100 cosine neighbours per `cid` (full all-pairs is 6 × 10⁹) + similarity distribution summary.
**Acceptance.** Counts monotone non-increasing; every `cid` has ≥ 1 raw id; no `cid` spans two artists unless flagged as collaboration; LLM audit on a 300-item sample by a *second prompt variant* reports agreement (target ≥ 90 % on merge decisions); unit tests for rule layer pass; re-run is a no-op (cache).
**Fallback.** If no metadata (Blocker-A): skip stages 1–3 title logic, keep stage 5–8 on ids; log.

### S2 — LLM annotation & measured ground truth (≈ 2–4 h; $30–60)
**Do.** Batch 50 tracks/request, JSON schema, `temperature 0`, retry until 100 % coverage; separate yes/no binary pass for extreme levels (paper 1). Dimensions (five levels none/low/moderate/high/very_high unless noted): **energy, aggression/intensity, valence, danceability/groove, acousticness (organic↔electronic production), vocal prominence, tempo feel, production polish (lo-fi↔hi-fi), lyrical explicitness, era (decade ordinal 1950s→2020s)**; **texture**: grit (distortion), density (sparse↔dense), space (dry↔reverberant), punch (transients), warmth, brightness; **culture**: macro-region tags (Anglo-American, British-Irish, Latin American, Iberian, Francophone, Germanic-Central-European, Nordic, East Asian, South Asian, African/Afro-Caribbean, Middle-Eastern/Mediterranean — final list frozen in the lock file), *tagged only with a distinctive marker*, universal/unclear left untagged. Input to the LLM is **artist + title (+ Last.fm tags/genres)** only — never audio features (keeps the labels independent of the measured data).
**Measured counterparts (USDA analogue; no LLM involved):** energy ↔ loudness/onset-rate; brightness ↔ spectral centroid/rolloff; warmth ↔ low-mid band energy; danceability ↔ Essentia danceability; tempo feel ↔ BPM; acousticness ↔ spectral flatness/tonal ratio; vocal prominence ↔ voice statistics; era ↔ release year; valence/arousal ↔ `id_gems`/VAD; aggression ↔ dynamic complexity + loudness. Axes for validation = **tercile centroids of measured values**.
**Reliability (addition).** Re-label a 3 % sample with a second prompt variant and a second seed; report Spearman/κ between runs, and LLM-vs-measured correlations *before* touching embeddings.
**Acceptance.** 100 % coverage; schema-valid; per-dimension reliability ≥ 0.6 Spearman (else flagged and excluded from headline probes); cost ≤ budget; cached rerun is free.

### S3 — Behaviour graph (Cooc) (≈ 1–3 h)
**Do.** Stream `userid_trackid_timestamp` (pyarrow, bz2) → int32 arrays → `sessions.sessionize` (gap 30 min, 5–200 distinct tracks; also produce `user_profiles` variant) → map `raw_id → cid` (dedupe within session) → `cooc.npmi_edges` (chunk 512–2048 to fit RAM; positive NPMI; `min_pair_count` ≥ 5; top-k 250) → `cooc_edges.parquet`. Also build the artist-masked variant. Report: #sessions, tracks/session distribution, degree distribution, #edges, share of same-artist edges (expect high; that is *why* artist-controls exist).
**Acceptance.** Symmetric, positive weights; every kept track has ≥ 1 edge or is logged; NPMI unit tests + a brute-force check on a 2 k-track slice; average degree within 100–250 or a justified alternative in `DECISIONS.md`.

### S4 — Compound graph (≈ 0.5 h)
**Do.** Load families (§4.4), build `features_<family>.parquet`, `fit_compound_graph(k per family, top_m 3, pca_dim 32)`. Report cluster-size distribution, per-family coverage, **cluster–label mutual information** (do compounds carry tag/era information? — diagnostic only, never tuned on), instrumental coverage.
**Acceptance.** Every track has ≥ 1 edge per non-missing family; no empty clusters; compound sizes non-degenerate (max cluster < 5 % of tracks); held-out files verified **absent** from all families (test reads `features_manifest.json`).

### S5 — Walks & training (≈ 1 h per model on `standard` if the 5080 reaches ~30 M pairs/s; measure in S0)
**Do.** Build `TypedGraph`; `generate_walks` per epoch with fresh seeds; `torch_sgns.train_torch_sgns` on GPU (fallback `embed.train_skipgram_file`). Profiles in `configs/profiles.yaml` (`paper`, `standard`, `smoke`). Train Cooc, Composition, Core × **3 seeds**, then ablations (§4.3). Checkpoint each epoch; log loss; write manifest. Use `paper` profile on a **20 k-track subset** to demonstrate that `standard` and `paper` give the same qualitative conclusions (stability check: probe scores and neighbour overlap).
**Acceptance.** Loss decreases monotonically per epoch (±noise); embeddings finite, non-degenerate (PR > 5; not all cosine ≈ 1); neighbour recall@10 between two seeds of the same model ≥ 0.3 (sanity, report actual); the provided synthetic end-to-end test still passes with the installed torch/CUDA stack.

### S6 — ICA, modes, poles (≈ 1–2 h; LLM labels ≈ $2)
**Do.** Per model: `ica_modes.residualise(E, category)` → `stable_ica(n_components=20, n_seeds=10, threshold 0.6)` → `factor_modes(top_quantile 0.75, K 3–7, min 6)` for each stable factor → LLM labels for modes (member titles/artists/tags/categories; **not** lyrics) with a property name; note whether each mode is driven by acoustic families, lyric families, or behaviour (mean compound-family membership of members). Build supervised poles: categories, region tags, era buckets, high-vs-low centroids of each LLM dimension. Save `modes.json`, `factor_poles.npy`, `supervised_poles.json`, `mode_atlas_<model>.parquet` in Epicure's schema.
**Acceptance.** Factor stability table; #modes per model reported (paper: 150/193/200 — do **not** force); mode coherence vs random ≥ 2× (report actual); label audit by second prompt (≥ 80 % agreement on property).

### S7 — Evaluation battery (≈ 2–4 h; CPU-heavy, parallelise)
**Do.** Everything in §7, for each model × seed, written to `results/*.parquet|json` with CIs. Paper-1 block on the averaged-embedding matrix; paper-2 block on the three retrained models. Supplement analyses: WEAT, Procrustes across models, orthogonal-residual SNR ranking, ℓ1 probes (our operationalisations recorded).
**Acceptance.** Every metric has a unit test on synthetic planted data (provided for probes/geometry); results tables regenerate from scripts; no metric computed without `groups=artist_id` unless explicitly labelled "leaky".

### S8 — SLERP & direction arithmetic (≈ 2 h)
**Do.** Implement `Melomane` API (§9) on top of `slerp.py`; qualitative examples (seed × pole × θ ∈ {0,30,60}); **quantitative direction test (new)**: for ≥ 500 (seed, pole) pairs, θ sweep 0–90° step 15: (i) mean cosine of top-10 to the pole, (ii) precision@10 of the pole's tag among retrieved, (iii) seed retention (cosine to seed), (iv) monotonicity (Spearman across θ); multi-pole composition test; orthogonal-residual SNR. Artist-disjoint retrieval (exclude the seed's artist) as the headline.
**Acceptance.** θ = 60° ⇒ cos(q, seed) = 0.5 (unit test); mean precision curve increases with θ for ≥ 80 % of poles in at least one model, else report honestly.

### S9 — Extensions (≈ 0.5–1 day)
- **E1 Compression frontier** — dims {300,150,100,64,32,16} × dtype {fp32,fp16,int8} × {PCA, random projection} per model; probes retention; **size in MB vs retention curve**; pick the smallest config ≥ 95 % / 90 % of full-precision retention and ship it as `melomane-*-small` (109 k × 64 × int8 ≈ 7 MB).
- **E2 Attribution** — acoustic-only / lyric-only / combined Composition; mean ρ per stratum; Shapley-style split of the gain over Cooc.
- **E3 Partial-hub regime** — 100/60/30/10 % of tracks with compound edges; does chemistry propagate through TT hops as in Epicure?
- **E4 Equal-token control** — the three models at equal total token-passes.
- **E5 External-embedding agreement** — neighbour-set overlap (kNN Jaccard) and CKA between each model and MAEST/MusiCNN/Jukebox (audio) and BERT/W2V (lyrics), artist-disjoint.
- **E6 Temporal stability** — Cooc trained on early vs late sessions (if the timestamp span ≥ 3 years): Procrustes disparity, mode persistence.
- **E7 Curation policy ablation** — strict vs song-level consolidation; raw vs curated (kNN-purity lift, size-matched bootstrap 200×, permutation N = 10,000).
- **E8 Sensitivity grid ("multiverse")** — k ∈ {250, 500, 1000}, top-m ∈ {1, 3, 5}, NPMI top-k ∈ {100, 250, 500}, session gap ∈ {10, 30, 60} min on a 20 k-track subset; report the *whole* grid, never the best cell.
**Acceptance.** Each extension emits a results table + figure; E1 artefact loads through the public API.

### S10 — Track B scale-up (multi-day, checkpointed)
**Do.** Adapter `mbid_stack`: (1) MusicBrainz canonical + (artist, title) → recording MBID mapping offline; (2) ListenBrainz listens (JSONL) → sessions on mapped MBIDs; (3) AcousticBrainz CSV features first, low-level JSON streamed and filtered for candidate MBIDs; (4) LRCLIB dump (if it exists) or API at ≤ 1 req/s with caching; (5) MusicBrainz/Discogs genres; (6) run S1–S8 unchanged on the new adapter. Optional temporal design: train ≤ 2019, evaluate listening ≥ 2020.
**Acceptance.** Same schemas validate; coverage report (tracks with behaviour ∧ audio ∧ lyrics); results reported side-by-side with Track A; stop cleanly with a resumable checkpoint at any time.

### S11 — Track C cross-catalog (≈ 0.5 day)
Fit compound vocabulary on the source track; project Jamendo/FMA tracks with the same scalers/PCA/centroids; test (a) cluster-usage entropy, (b) tag prediction from compound membership with cross-validation (chance-corrected), (c) SLERP retrieval sanity by tag. If feature definitions cannot be aligned, run inside Track B and document.

### S12 — Products (Release Bundles, Explorer, MCP Server)
**Do.** Export release bundles under `release/<corpus>/<profile>/` for licensed corpora:
1. `Melomane` model bundles (`embeddings.safetensors`, `vocab.json`, `modes.json`, `factor_poles.npy`, `supervised_poles.json`, `config.json`, `README.md` model card).
2. `-small` quantized variants (int8 / fp16 at compression frontier dim, e.g. 64-D, ~7 MB).
3. Static WebGL SLERP Explorer (HTML/JS/WebGL point cloud with iTunes Search API client-side 30s preview playback).
4. Read-only MCP Server config (`melomane-mcp`) exposing `neighbors`, `slerp`, `list_modes`, `mode_members`, and `describe_track`.
**Acceptance.** Release bundles validate via `Melomane.from_pretrained(path)`; explorer loads locally; model cards contain only numbers read from `results/`.

### S13 — Paper compilation (Markdown + arXiv LaTeX)
**Do.** Compile all experimental findings directly from `results/` into publication-ready documents:
1. `paper/melomane_paper.md` (full GitHub Markdown preprint with embedded figures and tables).
2. `paper/melomane_paper.tex` (arXiv-ready LaTeX preprint with booktabs, hyperref, and proper citations).
3. `paper/reproducibility.md` (hardware environment, GPU/CUDA details, seeds, package versions, OpenRouter spend audit).
4. `paper/tables/*.csv` and `paper/tables/*.md` (Strata probe means, SLERP metrics, H1–H8 status summary).
5. `paper/figures/*.png` (Figure 1: Stratum Means; Figure 2: SLERP Direction Precision; Figure 3: Compression Frontier).
**Acceptance.** Zero hand-typed experimental values; LaTeX builds without undefined references or missing figures; all H1–H8 hypotheses transparently reported.

### S14 — Release QA & Integrity Audits
**Do.** Run automated release verification before external distribution:
1. *Secret Leak Audit:* Scan all code, configs, and outputs for OpenRouter API keys (`sk-or-v1-...`), AWS, or Hugging Face tokens.
2. *Copyright & Privacy Audit:* Ensure zero raw lyrics or audio files exist in release folders. Verify that all distributed texts are LLM one-line themes or normalized metadata.
3. *Placeholder Absence Audit:* Scan paper deliverables (`.md` and `.tex`) for unresolved placeholders (`{{...}}`, `TODO`, `TBD`, `[INSERT]`).
4. *Bundle Deserialization Test:* Instantiate every exported model bundle via `Melomane.from_pretrained` and execute sample kNN queries and SLERP arithmetic.
5. *Budget Compliance:* Audit total OpenRouter LLM spend against the `$30 USD` hard cap.
**Acceptance.** 100% pass across all 5 audits; exit code 0.

---

## 7. Evaluation & pre-registered analysis plan

Write `analysis_plan.lock.json` (with sha256 of `configs/*.yaml` and this section's probe lists) **before S5**; changes after that go in `DECISIONS.md` as deviations.

### 7.1 Probe registry (analogues of the papers' 27 continuous + 8 cuisine probes)
| Stratum | Target count | Contents | Epicure analogue |
|---|---|---|---|
| **(a) baked-in measured** | 14 | scalar summaries of the descriptors that build the compound vocabularies: BPM, onset rate, Essentia danceability, average loudness, dynamic complexity, spectral centroid, spectral flux, zero-crossing rate, MFCC-1/2 means, chord strength, key strength, HPCP entropy, lyric-theme PC1 | 14 baked-in CF sensory categories |
| **(b) held-out measured** | ≈ 14 | mean F0, jitter, shimmer, HNR (openSMILE voice), 9 GEMS emotions, lyric VAD valence/arousal/dominance, lyric sentiment polarity | 5 held-out basic-taste probes |
| **(c) fully external** | ≈ 18 | LLM-knowledge labels (energy, aggression, valence, danceability, acousticness, vocal prominence, tempo feel, polish, explicitness, grit, density, space, punch, warmth, brightness), release year, log-popularity (**flagged behaviour-adjacent; excluded from the H1 test**) | 8 USDA macronutrient probes (+ LLM dims of paper 1) |
| **Cultural** | 8–11 regions | one-vs-rest Cohen's d, kNN purity, soft-NMI | 8 cuisine macro-regions |

### 7.2 Metric definitions (exact)
- **Ordinal probe**: axis = centroid(highest level) − centroid(lowest level) fit on train folds; held-out Spearman ρ; **10-fold × 20 repeats** (paper 1) and **5-fold repeated** (paper 2); **folds grouped by artist**; also report in-sample ρ. Continuous targets use tercile centroids (measured) or a ridge direction (`method="ridge"`) as a second reading.
- **Binary/one-vs-rest**: No→Yes centroid axis on train folds; Cohen's d and Mann-Whitney U on held-out; artist-grouped.
- **kNN purity** (k = 10, cosine): artist-disjoint; lift = purity / label-shuffle baseline; size-matched bootstrap; permutation test.
- **Participation ratio**, **mean pairwise cosine** (exact via ‖Σuᵢ‖²), **NMI** (k-means k = #classes), **soft-NMI** (Monte-Carlo over multi-label draws; operationalisation recorded), **mode coherence** vs same-size random sets.
- **WEAT** (Caliskan et al.): target sets = region groups, attribute sets = high- vs low-energy tracks; effect size d with permutation p. **Procrustes**: orthogonal alignment of model pairs on the shared vocabulary, report residual disparity and per-axis cosine stability. **Orthogonal-residual SNR** for poles: variance of neighbour projections on the pole after removing the seed component vs residual variance.

### 7.3 Hypotheses (directional replication targets; report outcomes whatever they are)
- **H1** Composition ≥ Cooc on the mean Spearman ρ of strata (b) and (c-excluding-popularity); ordering Cooc < Core < Composition on (a),(b). (Epicure: Cooc < Core < Chem on all strata.)
- **H2** Core has the lowest participation ratio and the highest mean pairwise cosine and mode coherence.
- **H3** Composition ≥ Cooc on regional Cohen's d (Epicure: Chem best on all 8 regions).
- **H4** Curated vocabulary > size-matched raw on artist-disjoint kNN purity (lift > 1).
- **H5** LLM dimensions correlate with their measured counterparts (Spearman > 0 with CI excluding 0) for ≥ 70 % of dimensions that have a counterpart.
- **H6** ≥ 90 % of full-precision held-out ρ is retained at ≤ 64 dims (fp16/int8) for Composition.
- **H7** SLERP target-dominance (precision@10 of the pole tag) increases with θ up to 60° for ≥ 80 % of poles.
- **H8** Conclusions on the `standard` profile match those on the `paper` profile (subset).

### 7.4 Statistics
Paired bootstrap over tracks *clustered by artist* (2,000 resamples) for model differences; report effect size + 95 % CI; Holm correction across the probe family; ≥ 3 training seeds per headline model; sensitivity grid (E8) instead of tuning on results; hyper-parameters chosen on **dev artists** only, headline numbers on **test artists**.

---

## 8. LLM specification (OpenRouter → `google/gemini-3.7-flash`)

- Client: `melomane.llm.LLMClient` (tested): JSON-schema structured output, cache, USD cap, retries, parallel `complete_json_many(workers=8)`. Stage 0 fixes `extra_body` (e.g. reasoning off, usage accounting) in `configs/llm.yaml`; if strict schemas are unsupported the client's validate-and-retry loop still guarantees valid JSON.
- Tasks & sizing (109 k tracks; batches of 50): S1 curation ≈ 4–8 k requests; S2 labelling ≈ 2.2 k requests + binary pass; S6 mode labels ≈ 600 small requests; audits ≈ 5 %. **Expected total $40–120**, half with the `:batch` variant if Stage 0 confirms batch submission. Track B adds a similar amount per 100 k tracks.
- Prompt hygiene: prompts live in `prompts/*.md` with a version header; every output stores `prompt_version`; changing a prompt invalidates only its cache keys. No secrets or user data in prompts. Never send lyrics to the LLM (labels come from titles/artists/tags — copyright and cost).
- Safety: model outputs are data, never instructions; strip/validate; unknown categories → `unknown`, not guesses.
- The **embeddings themselves are LLM-free** (state this in the paper, as Epicure does). LLM use is limited to canonicalisation, tagging, mode naming and the optional runtime NL layer.

---

## 9. Products

1. **Python package `melomane`** mirroring `epicure.py`: `Melomane.from_pretrained(path_or_repo)`, `vec`, `neighbors(name, k, exclude_self, exclude_artist=True)`, `slerp(seed, direction|list, theta_deg, k, weights=None)`, `closest_mode`, `mode_members`, `list_supervised_poles`, `list_modes`. Files per model: `embeddings.safetensors`, `vocab.json`, `modes.json`, `factor_poles.npy`, `supervised_poles.json`, `config.json`, `README.md` (model card).
2. **Hugging Face release** (after approval): 3 main models + ablations + `-small` variants; dataset `melomane-corpus-resources` with canonical vocabulary (ids, normalised artist/title, no lyrics), mode atlases, probe tables, direction tables, WEAT/Procrustes. Licence CC BY 4.0 with required attribution to Music4All-Onion (and Last.fm-derived data notice). **No audio, no lyrics, no per-user data.**
3. **Explorer (static web app)**: WebGL point cloud (UMAP, cosine, n_neighbors 32, min_dist 0.004 — *visualisation only*, all numbers computed in 300-D); seed picker; pole picker (supervised + emergent); θ slider 0–90° with live neighbours; model toggle (Cooc / Core / Composition; acoustic-only / lyric-only); compression toggle (full vs `-small`); mode atlas browser; each result shows artist – title, category, mode label, and **LLM one-line theme** (never lyrics); audio preview fetched **client-side from the iTunes Search API** using the visitor's own IP (verify CORS/JSONP behaviour in S12; fall back to "search on…" links). Data payload ≤ 50 MB (int8/fp16 small model + top-k neighbour tables); hosted as static files.
4. **Optional "Ask Melomane" (Gemini Flash via OpenRouter)** — user's idea: natural-language → structured query `{seed, poles[], weights[], theta, k}` (strict JSON schema), plus one-sentence explanations of why neighbours match (from their compound families/modes). Never put the OpenRouter key in client code. Options: (a) serverless proxy (e.g. Cloudflare Worker) holding the key with per-IP rate limit and a hard daily USD cap; (b) bring-your-own-key stored only in the visitor's browser; (c) offline precomputed queries only. Default recommendation: (a)+(c) fallback. Retrieval always runs on the local embeddings; the LLM never sees embeddings.
5. **MCP server** (read-only, mirrors `epicure-mcp`): tools `neighbors`, `slerp`, `list_modes`, `mode_members`, `describe_track`.

---

## 10. Testing, CI, done criteria

- **Provided & green today (sandbox, CPU):** 53 tests — NPMI vs brute force, sampler statistics, walk/edge validity, compound graph (NaN handling, cluster sharing), SLERP paper property (cos 60° = 0.5), PR/cosine exactness, probes incl. **shuffle-null, artist-leakage and artist-disjoint kNN**, ICA recovery of planted sources + modes, sessionisation, canonicalisation rules, LLM client (429 retry, bad-JSON retry, budget cap, cache, parallel), an end-to-end planted-structure test for gensim **and** PyTorch backends. Run `pytest -q` in Stage 0; a red test is a Stage-0 blocker.
- **Add per stage:** unit tests for every new function, golden-file tests for schemas, a `--profile smoke` end-to-end test (< 10 min on the 5080) that runs S3→S8 on the first 5 k tracks, property tests for probes (label shuffle ⇒ ρ ≈ 0), and a regression test that pins headline numbers of the synthetic study within tolerance.
- **CI** (GitHub Actions or local `python -m melomane.tasks ci`): ruff, mypy (loose), pytest (CPU), smoke pipeline on synthetic data. GPU tests marked `@pytest.mark.gpu`.
- **Done = ** S0–S9 acceptance green on real data; results regenerate from `python -m melomane.tasks results`; paper draft builds with generated tables; explorer runs locally against the small model; fresh-clone `python -m melomane.tasks reproduce-smoke` passes; `PROGRESS.md` ends with a clean state.

## 11. Paper plan
Title (working): *Melomane: emergent geometry in music-track embeddings, and what measured composition adds to listening behaviour.* Structure mirrors both papers (Abstract · Intro · Related work: Epicure ×2, FlavorGraph, word2vec/metapath2vec, ICA/SLERP, music embeddings & MIR datasets · Data & curation (per-stage table, consolidation ratio) · Methods (three models, compounds, probes, ICA/modes, SLERP) · Results: geometry, probe strata, cultural structure, modes, SLERP (quantitative), curation lift, LLM-vs-measured validation · Extensions (compression, attribution, partial hubs, equal-token, external agreement, temporal, cross-catalog) · Discussion · Limitations · Ethics & licences · Reproducibility). **All tables/figures generated from `results/`**. Claims allowed: "replicates the *methodology* and tests whether its findings transfer"; not "reproduces Epicure's numbers". Limitations must include: Onion's age/skew (Last.fm-era Western listening), artist structure, LLM label reliability, Onion's aggregated (not frame-level) features, instrumental asymmetry, lyric availability, unofficial/unverified pieces, scaled walk budget.

## 12. Risk register
| Risk | Likelihood | Mitigation |
|---|---|---|
| Onion lacks artist/title | Medium | Gate S0; tag-only labelling; email authors; Track B independent |
| Session NPMI dominated by same-artist/album listening | High | Artist-masked ablation; artist-grouped CV; report share of same-artist edges |
| Compound vocabulary ≈ kNN on features (trivially smooth) | Medium | Report cluster–label MI; E4/E5; multiverse grid; be explicit that Composition = "measured similarity structure" |
| GPU/CUDA issues on RTX 5080 | Medium | smoke test, nightly fallback, gensim CPU fallback |
| LLM label noise | Medium | reliability audit, measured validation, exclude unreliable dims |
| Costs overrun | Low | hard cap, cache, pilot, `:batch` variant |
| AcousticBrainz/ListenBrainz downloads slow or huge | Medium | stream + filter, resume, CSV-first; Track B optional |
| Licence/ToS problems | Low–Med | CC BY sources only for publication; no lyrics/audio/user data released; MLHD+ excluded |
| Session/context loss overnight | High | idempotent stages, `PROGRESS.md`, manifests, detached jobs |
| Windows sleep/thermal | Medium | power settings (Human action 3); checkpoints |

## 13. Stage-0 questions that must be answered from real files (everything marked **U**)
1. Does the Onion record ship artist/title/album/year/language/popularity? (Blocker-A gate)
2. Timestamp unit (s/ms), date span, users with ≥ N sessions, sessions per user.
3. `processed_lyrics` format; % tracks with lyrics; language mix.
4. `id_essentia` descriptor names vs AcousticBrainz low-level keys (Track C alignment).
5. OpenRouter: json_schema support, reasoning off switch, cost field, batch flow, effective rate limits for `google/gemini-3.7-flash`.
6. PyTorch build on the 5080 and measured pairs/s.
7. LRCLIB dump exists/size/terms; ListenBrainz dump size, licence, MBID-mapping fields; AcousticBrainz download throughput.
8. Whether the Music4All authors replied and what terms they gave (affects what we may republish).
