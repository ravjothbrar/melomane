# Epicure reference sheet (what we are replicating, exactly)

Compiled from the two papers' HTML versions and the public Hugging Face artefacts. Every number below is quoted from those sources;
items marked **[supplement]** are referenced by the papers but live in supplements we could not read — implement from the
description and record our operationalisation in the run manifest.

Papers
- **Paper 1** — *Epicure: Multidimensional Flavor Structure in Food Ingredient Embeddings* (arXiv:2604.22776). LLM-augmented curation of
  FlavorGraph's 6,653 ingredients → 1,032 canonical items; embedding *analysis* (no retraining).
- **Paper 2** — *Epicure: Navigating the Emergent Geometry of Food Ingredient Embeddings* (Radzikowski & Chen, arXiv:2605.22391). Retrains
  three skip-gram models on a 1,790-ingredient vocabulary; ICA modes, SLERP, probe battery.
- Artefacts: `Kaikaku/epicure-{cooc,core,chem}` (models), `Kaikaku/epicure-corpus-resources` (tables), `Kaikaku/epicure-explorer` (Space),
  loader `epicure.py`. Paper text says training code/raw data are not released; the HF dataset confirms only derived tables + embeddings.

## Paper 1 — the real 8 stages
| # | Stage | What happens | Count after |
|---|---|---|---|
| 1 | Simplification | "LLM + Jaccard clustering" batches similar raw items; strip brands, quantities, prep, packaging ("Kraft Sharp Cheddar Cheese (8 oz)" → cheddar_cheese) | ~4,800 |
| 2 | Refinement | Gemini 2.5 Flash, structured JSON; keep / remove (non-foods, packaging, compound foods) / consolidate; multi-language translation | ~2,100 |
| 3 | Ultra-refinement | ~450 lines of domain rules in the prompt (bread, rice keeps brown/wild, cultural spices kept distinct, cuts, cheeses…) | ~1,400 |
| 4 | Human-in-the-loop QC | Streamlit editor with search + batch ops; corrects 5–10 % of entries | ~1,050 |
| 5 | Consolidation | sequential canonical node ids + consolidation map original→canonical | 1,032 |
| 6 | Embedding averaging | e_cons = (1/|S|) Σ_{i∈S} e_i (centre of mass) → 1,032 × 300 | 1,032 |
| 7 | Categorisation | Gemini 2.5 Flash, 92-line prompt; 18 culinary categories + vegetarian/vegan flags (culinary function, not taxonomy) | — |
| 8 | Pairwise similarity | all-pairs cosine, 531,996 scores | — |
Headline: 6.4× consolidation with embedding semantics preserved.

### Paper 1 — analyses
- **LLM-labelled dimensions** (Gemini 3.1 Pro production / 2.5 Flash dev; structured JSON, 50 items per request, retry loop to 100 % coverage,
  separate yes/no binary-validation pass for 7 dims): taste/chem — climate zone (ordinal, n=509), NOVA 1–4, umami, Scoville (continuous, n=57),
  sweet, salty, sour, bitter (5-level none…very_high); texture (ISO 11036 / Szczesniak) — hardness, viscosity, crunchiness, chewiness,
  moisture, fattiness; culture — 7 macro-regional cuisines, tagged **only if a distinctive cultural marker** (universal items untagged).
- **Ordinal projection**: axis = centroid(lowest level) → centroid(highest level), fit on training folds only; project held-out items;
  Spearman ρ vs LLM rank. **10-fold CV × 20 random repeats = 200 measurements per dimension.** Paper reports in-sample ρ and ρ_CV
  (e.g. Scoville 0.757 / 0.20†, climate 0.572 / 0.48, umami 0.499 / 0.48, salty 0.400 / 0.38, sweet 0.387 / 0.37, NOVA 0.374 / 0.35; raw-vs-curated Δ).
- **Binary separation**: axis No-centroid → Yes-centroid; Mann-Whitney U + Cohen's d (sour 0.84, bitter 0.91, texture 0.79–1.26).
- **Categorical delta** (climate): within-group mean cosine − cross-group mean cosine; Wilcoxon signed-rank; Cohen's d.
- **Axis geometry**: inter-axis cosine matrices; partial Spearman via OLS residualisation.
- **Cultural clustering**: kNN purity (k = 10, cosine, 300-D) = fraction of an item's neighbours sharing its cuisine tag; random-baseline
  normalised ("6.2× lift"); size-matched bootstrap (200 iterations) curated vs raw; permutation test N = 10,000; curated purity ≈ doubles
  vs size-matched raw (Wilcoxon p = 0.008); cohorts curated-savoury 589 vs raw-savoury 2,916.
- **External measured validation** (the USDA analogue): USDA FoodData Central (Dec 2025, 8,221 foods) + FooDB; 3-layer matching
  (inverted-index rules → Gemini 3,072-d text-embedding cosine ranking → Gemini validation); 712/1,032 (69 %) USDA and 770 (74.6 %) FooDB matches;
  nutrient axes from **tercile centroids of measured values, no LLM involved**; sugars–sweet ρ 0.41, sodium–salty 0.29, glutamate–umami 0.24,
  macronutrients 0.44–0.47.
- **Chemical-hub ablation**: 308 curated ingredients trace to 416 FlavorGraph chemical hubs (direct compound edges); the other 724 get chemistry only
  through metapath propagation; correlations reported separately. Finding: sweetness/saltiness come purely from recipe co-occurrence.

## Paper 2 — the three models
All three: 1,790 canonical ingredients, **300-D skip-gram with negative sampling, PyTorch, walks_per_node = 100, walk_length = 50, context_size = 7,
5 negatives, batch_size = 32,768, lr = 0.0025, 20 epochs, no warm restart** (seeds unspecified).
- Graph: 4.14 M recipes from 11 sources; ingredients in < 20 recipes dropped; **positive-NPMI only → 203,508 I–I edges**. FlavorDB: 2,247 typed compound
  nodes, 80,019 I–C edges; 15 flavour categories (balsamic, citrus, earthy, fatty, floral, fruity, green, meaty, minty, nutty, spicy, vegetable,
  wine-like, woody + residual) — a compound is replicated once per category. 525 of 1,790 ingredients are hubs (direct compound edges); 1,267 non-hubs
  reach chemistry only via N–H–C[x]–H–N.
- Metapath templates: H–C[x]–H (shared same-family compound); N–H–C[x]–H–N (only route for non-hubs); C[x]–H–N–H–C[y] (cross-type bridge).
- **Cooc** = I–I graph only. **Chem** = typed-compound metapaths only, `ii_repeat = 0`. **Core** = compound metapaths + **ten pure I–I templates per walk round**
  (`ii_repeat = 10`) so I–I context is ~10× any single compound template.
- **Results** (Cooc / Core / Chem): participation ratio 173.6 / 94.2 / 183.1 (of 300); mean pairwise cosine ≈ 0.10–0.12 / 0.35 / 0.10–0.12;
  NMI vs USDA food groups 0.20–0.25, vs cuisine macro-regions 0.43–0.46 (soft-NMI); Spearman ρ (5-fold repeated CV, fold-trained direction) —
  14 baked-in CF sensory 0.28 / 0.40 / 0.46; 5 held-out basic-taste 0.32 / 0.42 / 0.47; 8 USDA macronutrient 0.41 / 0.45 / 0.49; cuisine one-vs-rest
  Cohen's d 2.43 / 2.70 / 3.07 (Chem wins all 8 regions); mode coherence 0.611 / 0.833 / 0.703 vs random 0.097 / 0.348 / 0.115.
  Headline: **Chem > Core > Cooc on every one of 27 continuous + 8 cuisine probes, including strata never in the walk schema.**
- **ICA**: sklearn FastICA, 20 factors per model on the *food-group-residualised* embedding; 10 seeds, Hungarian matching, keep the seed with highest mean
  matched-cosine to the others; keep factors with split-half cosine stability > 0.6.
- **Modes**: top-quartile items per factor → PCA-reduced → GMM, BIC over K ∈ {3..7}, ≥ 6 members per mode → unit-mean "pole" in 300-D; LLM labels each mode.
  150 / 193 / 200 modes (Cooc / Core / Chem), coherence 5–6× random.
- **SLERP** (`epicure.py`): unit seed v, unit direction d, d⊥ = d − (d·v)v, q = cos θ · v + sin θ · unit(d⊥), renormalise; θ = 0 → seed, θ = 60° → cos(q, seed) = 0.5;
  seed excluded from results. Supervised poles from tags (cuisine, food group, NOVA); emergent poles = mode centroids. Qualitative top-5 lists at 0/30/60°
  (rice → South Asian 30° → curry leaf, masoor dal, urad dal, chana dal, fenugreek). No quantitative retrieval metric is reported for SLERP — we add one.
- **Released files** (per model): `embeddings.safetensors`, `vocab.json`/`itos.json`, `modes.json` (Core: 193 modes / 44 properties), `factor_poles.npy` ((87, 300) for Core),
  `supervised_poles.json` (113), `config.json`; corpus-resources tables: canonical_vocabulary, mode_atlas_*, direction_arithmetic_full (2,160 rows),
  direction_orthogonal (orthogonal-residual SNR ranking), factor_top_alignments_*, linear_probe_{categorical,continuous}, cross_modal, weat, procrustes_sensory.
  Licence CC BY 4.0. Blog claims a ~2 MB model (1,790 × 300 floats ≈ 2.1 MB) — no compression experiment exists in either paper.
- **[supplement]** WEAT target/attribute sets, Procrustes alignment across models, orthogonal-residual SNR ranking definition, ℓ1-regularised probes, multi-label NMI protocol.
- **Limitations the authors state**: corpus imbalance (half East Asian), hub coverage (525/1,790), LLM dependence in canonicalisation/tagging/labels (embeddings themselves are LLM-free).

## What Melomane changes on purpose (deviation register — keep this updated in `DECISIONS.md`)
1. Track-level items with **artist/album structure** → artist-grouped CV and artist-disjoint kNN purity (no such issue in food).
2. Documents are **listening sessions**, not recipes; extra NPMI guards (`min_pair_count`, `topk`) needed at 10⁵-node scale.
3. Compound layer is **measured/derived from the track's own audio and lyrics**, typed by feature family (7 families ≈ FlavorDB's 15 categories); **100 % hub coverage**
   by construction → the paper's N–H–C–H–N regime is reproduced only as an explicit ablation (partial-hub).
4. Walk budget scaled (`standard` profile) because 109 k nodes × paper budget is ≈ 2,000 GPU-hours; the `paper` profile is used on subsets to prove parity.
5. Gemini 3.7 Flash via OpenRouter replaces Claude/Gemini-2.5/3.1 for LLM steps.
6. Extensions beyond the papers: compression frontier, acoustic-vs-lyric attribution, equal-token control, external-embedding agreement, cross-catalog transfer, temporal stability.
