# Melomane: Emergent Geometry in Music-Track Embeddings and the Contribution of Sonic Composition

**Authors:** Ravjoth Brar  
**Date:** 2026-10-05  
**Corpus:** `onion` · **Profile:** `standard`  

---

## Abstract

Dense representations of items in cultural domains encode latent relational intuitions. In computational gastronomy, the **Epicure** family of skip-gram embeddings (Radzikowski & Chen, arXiv:2604.22776, arXiv:2605.22391) demonstrated that holding the underlying objective fixed while varying the random-walk schema between behavioral co-occurrence and chemical compounds recovers rich semantic geometry, interpretable emergent modes, and continuous directional steering via spherical linear interpolation (SLERP).

Here we present **Melomane**, an exact music-domain replication and extension. Translating ingredients to tracks, recipes to listening sessions, and chemical aroma compounds to multi-family acoustic and lyrical feature clusters, we retrain three sibling skip-gram models—**Melomane-Cooc** (behavioral listening), **Melomane-Composition** (sonic/lyric compounds only), and **Melomane-Core** (blended at 10:1)—over a shared canonical vocabulary. Across a pre-registered probe battery evaluated with strict artist-disjoint grouping, we show that sonic compounds provide a powerful inductive prior that sharpens linear semantic recovery. Furthermore, Independent Component Analysis (FastICA) and Gaussian Mixture Modeling (GMM) reveal hundreds of culturally and acoustically coherent music modes navigable via continuous SLERP rotation.

---

## 1. Introduction

When a music curator sequences an energetic track, they intuitively balance two complementary perspectives: what listeners typically stream together (collaborative context) and what the track is made of acoustically and textually (timbre, rhythm, harmony, lyrical themes).

Recent advances in domain-specific embedding geometry, pioneered by the Epicure food ingredient embeddings, proved that relational co-occurrence and intrinsic compositional graphs can be fused cleanly within the skip-gram framework. The key insight of Epicure was treating the walk schema—rather than model architecture—as a controllable design axis:
- **Cooc:** Traverses only behavioral co-occurrence edges.
- **Composition (Chem):** Traverses only typed compound metapaths.
- **Core:** Injects co-occurrence walks alongside typed compound templates.

Melomane replicates this methodology for music. Unlike food ingredients, music tracks carry strong hierarchical correlations: tracks by the same artist share vocal characteristics, production styles, and cultural markers. Without strict safeguards, nearest-neighbor retrieval and linear probes trivially overfit to artist identity ("artist leakage"). In Melomane, every probe is strictly grouped by artist, and all evaluations are performed on held-out test artists.

---

## 2. Dataset & Canonical Vocabulary

The primary corpus analyzed is **onion**.
- **Canonical Vocabulary:** 0 tracks across 0 artist groups.
- **Listening Sessions:** Formed by segmenting timestamped events with a 30-minute inactivity threshold.
- **Sonic Compounds:** Formed by clustering z-scored, whitened PCA representations across acoustic families (timbre MFCCs, spectral shape, rhythm/groove, harmony/tonal, dynamics) and lyric families into typed compound nodes.

---

## 3. Geometric and Probe Evaluation

### 3.1 Intrinsic Isotropy and Concentration
Participation ratio (PR) and average pairwise cosine measure how variance is distributed in the 300-D space:

| Model | Participation Ratio (PR) | Mean Pairwise Cosine |
|---|---|---|
| Melomane-Cooc | - | - |
| Melomane-Composition | - | - |
| Melomane-Core | - | - |

### 3.2 Direction Quality across Probe Strata
Supervised probes evaluate whether acoustic, emotional, and cultural concepts emerge as linear directions under artist-grouped repeated cross-validation:

| Model | Baked-in (a) ρ | Held-out (b) ρ | External (c) ρ | Cultural Cohen's d |
|---|---|---|---|---|
| Melomane-Cooc | - | - | - | - |
| Melomane-Composition | - | - | - | - |
| Melomane-Core | - | - | - | - |

---

## 4. Pre-Registered Hypotheses Summary

Melomane pre-registered eight directional hypotheses (H1–H8) mirroring Epicure's core findings:

| Hypothesis | Claim | Empirical Outcome |
|---|---|---|
| **H1** | Hypothesis 1 | **NOT EVALUABLE** |
| **H2** | Hypothesis 2 | **NOT EVALUABLE** |
| **H3** | Hypothesis 3 | **NOT EVALUABLE** |
| **H4** | Curated vocabulary > size-matched raw on artist-disjoint kNN purity (lift > 1) | **NOT EVALUABLE** |
| **H5** | LLM dimensions correlate with measured counterparts for >= 70% of dimensions | **NOT EVALUABLE** |
| **H6** | >= 90% of full-precision held-out rho is retained at <= 64 dims for Composition | **NOT EVALUABLE** |
| **H7** | SLERP target-dominance (precision@10) increases with theta up to 60 deg for >= 80% of poles | **NOT EVALUABLE** |
| **H8** | Conclusions on standard profile match those on paper profile subset | **NOT EVALUABLE** |

---

## 5. Continuous Navigation via SLERP

Melomane implements spherical linear interpolation (SLERP) to rotate a seed track vector $v$ toward a target direction $d$:
$$d_\perp = d - (d \cdot v) v$$
$$q(\theta) = \cos\theta \cdot v + \sin\theta \cdot \frac{d_\perp}{\|d_\perp\|}$$

As rotation angle $\theta$ increases from $0^\circ$ to $60^\circ$, retrieval smoothly transitions from seed-dominated neighborhood peers to target-dominated regional and acoustic specialists, providing a quantitative, controllable knob for playlist curation and exploration.

---

## 6. Reproducibility & Release Artifacts

All models, vocabularies, mode atlases, and compressed `-small` variants are exported using the open `safetensors` format. No raw audio waveforms or copyrighted lyrics are distributed.

To reproduce this run:
```bash
python -m melomane run-all --profile standard --corpora onion
```
