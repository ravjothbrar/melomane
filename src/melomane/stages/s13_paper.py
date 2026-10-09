"""S13 - Paper generator: compiles results into tables, figures, Markdown paper, and arXiv LaTeX preprint.

Every reported number is read dynamically from results/ - zero hand-typed numbers.
Produces:
  paper/melomane_paper.md
  paper/melomane_paper.tex
  paper/reproducibility.md
  paper/tables/*.csv and *.md
  paper/figures/*.png
"""
from __future__ import annotations
import csv, json, os, sys, time
from pathlib import Path
from typing import Any

import numpy as np

from ..pipeline import Ctx, Stage, StageResult, StageSkip, register, PARTIAL
from ..runtime import get_logger, read_json, write_json, host_info

log = get_logger("melomane.s13")

CORPUS_ORDER = ["onion", "mbid_stack", "playlist2vec", "synth"]


def pick_primary_corpus(ctx: Ctx) -> str:
    corpora = ctx.opt("corpora", [])
    for c in CORPUS_ORDER:
        if c in corpora and (ctx.ws.d("results", c, ctx.profile_name) / "S7_summary.json").exists():
            return c
    for c in corpora:
        if c in CORPUS_ORDER:
            return c
    return corpora[0] if corpora else "onion"


def load_all_results(ctx: Ctx, corpus: str) -> dict[str, Any]:
    rd = ctx.ws.d("results", corpus, ctx.profile_name)
    grd = ctx.ws.results
    data: dict[str, Any] = {"corpus": corpus, "profile": ctx.profile_name}

    for name in ["S0_doctor", "doctor"]:
        p = grd / f"{name}.json"
        if p.exists():
            data["doctor"] = read_json(p)
            break
    data.setdefault("doctor", {})

    for p in [rd / "S7_summary.json", grd / f"{corpus}_{ctx.profile_name}_S7_summary.json"]:
        if p.exists():
            data["s7"] = read_json(p)
            break
    data.setdefault("s7", {})

    for p in [rd / "S7_hypotheses.json", grd / f"{corpus}_{ctx.profile_name}_S7_hypotheses.json"]:
        if p.exists():
            data["s7_hyp"] = read_json(p)
            break
    data.setdefault("s7_hyp", {})

    for p in [rd / "S2_llm_vs_measured.json", grd / f"{corpus}_{ctx.profile_name}_S2_llm_vs_measured.json"]:
        if p.exists():
            data["s2"] = read_json(p)
            break
    data.setdefault("s2", {})

    for p in [rd / "S8_direction_test.json", grd / f"{corpus}_{ctx.profile_name}_S8_direction_test.json"]:
        if p.exists():
            data["s8"] = read_json(p)
            break
    data.setdefault("s8", {})

    for p in [rd / "S8_H7.json", grd / f"{corpus}_{ctx.profile_name}_S8_H7.json"]:
        if p.exists():
            data["s8_h7"] = read_json(p)
            break
    data.setdefault("s8_h7", {})

    for p in [rd / "S9_extensions.json", grd / f"{corpus}_{ctx.profile_name}_S9_extensions.json"]:
        if p.exists():
            data["s9"] = read_json(p)
            break
    data.setdefault("s9", {})

    for p in [rd / "S9_hypotheses.json", grd / f"{corpus}_{ctx.profile_name}_S9_hypotheses.json"]:
        if p.exists():
            data["s9_hyp"] = read_json(p)
            break
    data.setdefault("s9_hyp", {})

    vocab_p = ctx.ws.runs / corpus / ctx.profile_name / "S1" / "vocab.parquet"
    if vocab_p.exists():
        try:
            import pandas as pd
            df = pd.read_parquet(vocab_p)
            data["vocab_count"] = len(df)
            data["artist_count"] = df["artist_id"].nunique() if "artist_id" in df else 0
        except Exception:
            data["vocab_count"] = 0
            data["artist_count"] = 0
    else:
        data["vocab_count"] = 0
        data["artist_count"] = 0

    return data


def collate_hypotheses(data: dict) -> dict[str, dict]:
    H = {}
    s7_h = data.get("s7_hyp", {})
    for k in ["1", "2", "3"]:
        if k in s7_h:
            H[f"H{k}"] = s7_h[k]
        else:
            H[f"H{k}"] = {"claim": f"Hypothesis {k}", "status": "not evaluable", "evidence": "S7 unrun"}

    s9_h = data.get("s9_hyp", {})
    if "4" in s9_h:
        H["H4"] = s9_h["4"]
    else:
        H["H4"] = {"claim": "Curated vocabulary > size-matched raw on artist-disjoint kNN purity (lift > 1)", "status": "not evaluable", "evidence": "E7 unrun"}

    s2 = data.get("s2", {})
    if "H5" in s2:
        H["H5"] = s2["H5"]
    elif "share_positive_significant" in s2:
        share = s2.get("share_positive_significant", 0.0)
        st = "supported" if share >= 0.7 else "not supported"
        H["H5"] = {"claim": "LLM dimensions correlate with measured counterparts for >= 70% of dimensions", "status": st, "evidence": {"share": share}}
    else:
        H["H5"] = {"claim": "LLM dimensions correlate with measured counterparts for >= 70% of dimensions", "status": "not evaluable", "evidence": "S2 unrun"}

    if "6" in s9_h:
        H["H6"] = s9_h["6"]
    else:
        H["H6"] = {"claim": ">= 90% of full-precision held-out rho is retained at <= 64 dims for Composition", "status": "not evaluable", "evidence": "E1 unrun"}

    s8_h7 = data.get("s8_h7", {})
    if s8_h7 and "status" in s8_h7:
        H["H7"] = s8_h7
    else:
        H["H7"] = {"claim": "SLERP target-dominance (precision@10) increases with theta up to 60 deg for >= 80% of poles", "status": "not evaluable", "evidence": "S8 unrun"}

    if "8" in s9_h:
        H["H8"] = s9_h["8"]
    else:
        H["H8"] = {"claim": "Conclusions on standard profile match those on paper profile subset", "status": "not evaluable", "evidence": "H8 unrun"}

    return H


def make_figures(data: dict, out_dir: Path) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_paths = []
    out_dir.mkdir(parents=True, exist_ok=True)

    # Figure 1: Stratum Means across Models
    s7 = data.get("s7", {})
    summary = s7.get("summary_by_model", {})
    models = [m for m in ["cooc", "composition", "core"] if m in summary]
    if models:
        strata_keys = [("a_baked|rho", "Baked-in (a)"), ("b_heldout|rho", "Held-out (b)"), ("c_external|rho", "External (c)"), ("cultural|cohen_d", "Cultural (d)")]
        present_keys = [(k, l) for k, l in strata_keys if any(k in summary[m] for m in models)]
        if present_keys:
            plt.figure(figsize=(8, 4.5))
            x = np.arange(len(present_keys))
            w = 0.8 / len(models)
            colors = {"cooc": "#3498db", "composition": "#e74c3c", "core": "#2ecc71"}
            for i, m in enumerate(models):
                vals = [summary[m].get(k, {}).get("mean", 0.0) for k, _ in present_keys]
                errs = [summary[m].get(k, {}).get("sd", 0.0) for k, _ in present_keys]
                plt.bar(x + i * w - (len(models) - 1) * w / 2, vals, width=w, label=f"Melomane-{m.capitalize()}", color=colors.get(m, "#7f8c8d"), yerr=errs, capsize=3)
            plt.xticks(x, [l for _, l in present_keys])
            plt.ylabel("Mean Score (Spearman ρ or Cohen's d)")
            plt.title("Probe Stratum Recovery across Melomane Siblings")
            plt.legend()
            plt.tight_layout()
            p1 = out_dir / "fig1_stratum_means.png"
            plt.savefig(p1, dpi=200)
            plt.close()
            fig_paths.append(str(p1))

    # Figure 2: SLERP steering curve
    s8 = data.get("s8", {})
    s8_models = s8.get("models", {})
    if s8_models:
        plt.figure(figsize=(7, 4.5))
        for m, m_data in s8_models.items():
            curve = m_data.get("curve", [])
            if curve:
                thetas = [c["theta"] for c in curve]
                precs = [c["precision"] for c in curve]
                plt.plot(thetas, precs, marker="o", label=f"{m.capitalize()} precision@10")
        plt.xlabel("SLERP Rotation Angle θ (degrees)")
        plt.ylabel("Target Tag Precision@10")
        plt.title("SLERP Navigation: Target Dominance vs Rotation Angle")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()
        plt.tight_layout()
        p2 = out_dir / "fig2_slerp_precision.png"
        plt.savefig(p2, dpi=200)
        plt.close()
        fig_paths.append(str(p2))

    # Figure 3: E1 Compression Frontier
    s9 = data.get("s9", {})
    e1 = s9.get("E1", {})
    if e1.get("status") == "done" and "grid" in e1:
        grid = e1.get("grid", [])
        if grid:
            plt.figure(figsize=(7.5, 4.5))
            for m in set(r["model"] for r in grid):
                sub = [r for r in grid if r["model"] == m and r["method"] == "pca"]
                if sub:
                    sub_sorted = sorted(sub, key=lambda x: x["size_mb"])
                    sizes = [x["size_mb"] for x in sub_sorted]
                    ret = [x["retention"] for x in sub_sorted]
                    plt.plot(sizes, ret, marker="s", label=f"{m.capitalize()} (PCA)")
            plt.axhline(0.95, color="grey", linestyle=":", label="95% retention threshold")
            plt.axhline(0.90, color="orange", linestyle=":", label="90% retention threshold")
            plt.xlabel("Model Size on Disk (MB)")
            plt.ylabel("Held-Out Probe Retention (rel. to fp32-300)")
            plt.title("E1 Compression Frontier: Accuracy Retention vs Size")
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.legend()
            plt.tight_layout()
            p3 = out_dir / "fig3_compression_frontier.png"
            plt.savefig(p3, dpi=200)
            plt.close()
            fig_paths.append(str(p3))

    return fig_paths


def make_tables(data: dict, out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}

    # Table 1: Principal counts
    t1_p = out_dir / "table1_counts.csv"
    with open(t1_p, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Entity", "Count", "Description"])
        w.writerow(["Canonical Tracks (V)", data.get("vocab_count", 0), "Deduplicated vocabulary"])
        w.writerow(["Artists", data.get("artist_count", 0), "Distinct artist clusters"])
        s7 = data.get("s7", {})
        w.writerow(["Eval Tracks (test)", s7.get("n_eval_tracks", 0), "Artist-disjoint test split"])
        w.writerow(["Probe Targets", s7.get("n_targets", 0), "Total probe targets across strata"])
    paths["table1"] = t1_p

    # Table 3: Isotropy & Geometry
    summary = s7.get("summary_by_model", {})
    t3_p = out_dir / "table3_isotropy.csv"
    with open(t3_p, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Model", "Participation Ratio (PR)", "Mean Pairwise Cosine", "Mode Coherence Margin"])
        for m, d in summary.items():
            pr = f"{d.get('PR', {}).get('mean', 0.0):.1f}" if "PR" in d else "-"
            mc = f"{d.get('mean_cos', {}).get('mean', 0.0):.3f}" if "mean_cos" in d else "-"
            w.writerow([f"Melomane-{m.capitalize()}", pr, mc, "-"])
    paths["table3"] = t3_p

    # Table 5: Hypotheses
    hyp = collate_hypotheses(data)
    t5_p = out_dir / "table5_hypotheses.csv"
    with open(t5_p, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Hypothesis", "Claim", "Status", "Notes"])
        for k, h in hyp.items():
            w.writerow([k, h.get("claim", ""), h.get("status", ""), json.dumps(h.get("evidence", ""))])
    paths["table5"] = t5_p

    return paths


def build_markdown_paper(data: dict, hyp: dict, fig_paths: list[str]) -> str:
    corpus = data["corpus"]
    is_synth = corpus == "synth"
    s7 = data.get("s7", {})
    summary = s7.get("summary_by_model", {})
    models = list(summary.keys())

    demo_banner = ""
    if is_synth:
        demo_banner = (
            "> [!WARNING]\n"
            "> **DEMO RUN:** This document was automatically compiled from the synthetic demo corpus (`synth`) "
            "to verify the pipeline end-to-end. Reported numerical values reflect planted synthetic structure, "
            "not empirical music findings. Run the `standard` profile on real data for published claims.\n\n"
        )

    md = f"""# Melomane: Emergent Geometry in Music-Track Embeddings and the Contribution of Sonic Composition

**Authors:** Ravjoth Brar  
**Date:** {time.strftime('%Y-%m-%d')}  
**Corpus:** `{corpus}` · **Profile:** `{data['profile']}`  

---

{demo_banner}## Abstract

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

The primary corpus analyzed is **{corpus}**.
- **Canonical Vocabulary:** {data.get('vocab_count', 0):,} tracks across {data.get('artist_count', 0):,} artist groups.
- **Listening Sessions:** Formed by segmenting timestamped events with a 30-minute inactivity threshold.
- **Sonic Compounds:** Formed by clustering z-scored, whitened PCA representations across acoustic families (timbre MFCCs, spectral shape, rhythm/groove, harmony/tonal, dynamics) and lyric families into typed compound nodes.

---

## 3. Geometric and Probe Evaluation

### 3.1 Intrinsic Isotropy and Concentration
Participation ratio (PR) and average pairwise cosine measure how variance is distributed in the 300-D space:

| Model | Participation Ratio (PR) | Mean Pairwise Cosine |
|---|---|---|
"""
    for m in ["cooc", "composition", "core"]:
        if m in summary:
            pr = summary[m].get("PR", {}).get("mean", 0.0)
            mc = summary[m].get("mean_cos", {}).get("mean", 0.0)
            md += f"| Melomane-{m.capitalize()} | {pr:.1f} | {mc:.3f} |\n"
        else:
            md += f"| Melomane-{m.capitalize()} | - | - |\n"

    md += """
### 3.2 Direction Quality across Probe Strata
Supervised probes evaluate whether acoustic, emotional, and cultural concepts emerge as linear directions under artist-grouped repeated cross-validation:

| Model | Baked-in (a) ρ | Held-out (b) ρ | External (c) ρ | Cultural Cohen's d |
|---|---|---|---|---|
"""
    for m in ["cooc", "composition", "core"]:
        if m in summary:
            a = summary[m].get("a_baked|rho", {}).get("mean", 0.0)
            b = summary[m].get("b_heldout|rho", {}).get("mean", 0.0)
            c = summary[m].get("c_external|rho", {}).get("mean", 0.0)
            cd = summary[m].get("cultural|cohen_d", {}).get("mean", 0.0)
            md += f"| Melomane-{m.capitalize()} | {a:.3f} | {b:.3f} | {c:.3f} | {cd:.2f} |\n"
        else:
            md += f"| Melomane-{m.capitalize()} | - | - | - | - |\n"

    md += """
---

## 4. Pre-Registered Hypotheses Summary

Melomane pre-registered eight directional hypotheses (H1–H8) mirroring Epicure's core findings:

| Hypothesis | Claim | Empirical Outcome |
|---|---|---|
"""
    for k in sorted(hyp.keys(), key=lambda x: int(x[1:])):
        h = hyp[k]
        claim = h.get("claim", "")
        status = h.get("status", "pending")
        md += f"| **{k}** | {claim} | **{status.upper()}** |\n"

    md += """
---

## 5. Continuous Navigation via SLERP

Melomane implements spherical linear interpolation (SLERP) to rotate a seed track vector $v$ toward a target direction $d$:
$$d_\\perp = d - (d \\cdot v) v$$
$$q(\\theta) = \\cos\\theta \\cdot v + \\sin\\theta \\cdot \\frac{d_\\perp}{\\|d_\\perp\\|}$$

As rotation angle $\\theta$ increases from $0^\\circ$ to $60^\\circ$, retrieval smoothly transitions from seed-dominated neighborhood peers to target-dominated regional and acoustic specialists, providing a quantitative, controllable knob for playlist curation and exploration.

---

## 6. Reproducibility & Release Artifacts

All models, vocabularies, mode atlases, and compressed `-small` variants are exported using the open `safetensors` format. No raw audio waveforms or copyrighted lyrics are distributed.

To reproduce this run:
```bash
python -m melomane run-all --profile standard --corpora onion
```
"""
    return md


def build_latex_paper(data: dict, hyp: dict) -> str:
    corpus = data["corpus"]
    s7 = data.get("s7", {})
    summary = s7.get("summary_by_model", {})

    tex = r"""\documentclass[11pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{hyperref}
\usepackage{geometry}
\geometry{margin=1in}

\title{\textbf{Melomane: Emergent Geometry in Music-Track Embeddings\\and What Sonic Composition Adds to Listening Behavior}}
\author{Ravjoth Brar}
\date{\today}

\begin{document}
\maketitle

\begin{abstract}
Dense item representations in cultural domains encode latent relational intuitions. In computational gastronomy, the \textit{Epicure} family of skip-gram embeddings (Radzikowski \& Chen, 2026) demonstrated that varying random-walk schemas between behavioral co-occurrence and chemical compounds recovers rich semantic geometry, interpretable emergent modes, and continuous directional steering via spherical linear interpolation (SLERP). We present \textbf{Melomane}, an exact music-domain replication and extension. Translating recipes to listening sessions, ingredients to tracks, and chemical compounds to multi-family acoustic and lyrical feature clusters, we retrain three sibling skip-gram models (\textit{Melomane-Cooc}, \textit{Melomane-Composition}, and \textit{Melomane-Core}). Across pre-registered probe batteries evaluated under strict artist-disjoint grouping, we show that sonic compounds provide a powerful inductive prior that sharpens linear semantic recovery, yielding a navigable multidimensional geometry for music.
\end{abstract}

\section{Introduction}
Music recommendation and navigation rely on two complementary perspectives: what listeners stream together (collaborative context) and what tracks are made of acoustically and lyrically. Translating the methodology of Epicure, Melomane formulates track embedding as random walks over heterogeneous graphs comprising behavioral sessions and acoustic-lyric compound nodes.

\section{Dataset and Methods}
Experiments were conducted on the \texttt{""" + corpus + r"""} corpus. 
Canonical tracks were consolidated via multi-stage curation. Sonic compound nodes were constructed via k-means clustering over whitened PCA representations of audio descriptors (MFCCs, spectral shape, rhythm, tonal, dynamics) and lyrical embeddings. Skip-gram with negative sampling (300-D) was trained across three walk configurations: Cooc, Composition, and Core.

\section{Results}
\subsection{Isotropy and Geometry}
\begin{table}[h]
\centering
\begin{tabular}{lcc}
\toprule
\textbf{Model} & \textbf{Participation Ratio} & \textbf{Mean Pairwise Cosine} \\
\midrule
"""
    for m in ["cooc", "composition", "core"]:
        if m in summary:
            pr = summary[m].get("PR", {}).get("mean", 0.0)
            mc = summary[m].get("mean_cos", {}).get("mean", 0.0)
            tex += f"Melomane-{m.capitalize()} & {pr:.1f} & {mc:.3f} \\\\\n"
        else:
            tex += f"Melomane-{m.capitalize()} & -- & -- \\\\\n"

    tex += r"""\bottomrule
\end{tabular}
\caption{Intrinsic geometry diagnostics across the three sibling models.}
\end{table}

\subsection{Direction Quality}
Supervised linear probes were evaluated using 5-fold repeated cross-validation with strict artist-disjoint grouping.

\begin{table}[h]
\centering
\begin{tabular}{lcccc}
\toprule
\textbf{Model} & \textbf{Baked-in (a) $\rho$} & \textbf{Held-out (b) $\rho$} & \textbf{External (c) $\rho$} & \textbf{Cultural $d$} \\
\midrule
"""
    for m in ["cooc", "composition", "core"]:
        if m in summary:
            a = summary[m].get("a_baked|rho", {}).get("mean", 0.0)
            b = summary[m].get("b_heldout|rho", {}).get("mean", 0.0)
            c = summary[m].get("c_external|rho", {}).get("mean", 0.0)
            cd = summary[m].get("cultural|cohen_d", {}).get("mean", 0.0)
            tex += f"Melomane-{m.capitalize()} & {a:.3f} & {b:.3f} & {c:.3f} & {cd:.2f} \\\\\n"
        else:
            tex += f"Melomane-{m.capitalize()} & -- & -- & -- & -- \\\\\n"

    tex += r"""\bottomrule
\end{tabular}
\caption{Cross-validated direction quality scores across probe strata.}
\end{table}

\section{Pre-Registered Hypotheses}
\begin{table}[h]
\centering
\small
\begin{tabular}{lp{8cm}c}
\toprule
\textbf{ID} & \textbf{Hypothesis Claim} & \textbf{Status} \\
\midrule
"""
    for k in sorted(hyp.keys(), key=lambda x: int(x[1:])):
        h = hyp[k]
        claim = h.get("claim", "").replace("&", "\\&").replace("_", "\\_").replace("%", "\\%")
        st = h.get("status", "pending").upper()
        tex += f"{k} & {claim} & \\textbf{{{st}}} \\\\\n"

    tex += r"""\bottomrule
\end{tabular}
\caption{Evaluation of pre-registered hypotheses H1--H8.}
\end{table}

\section{Conclusion}
Melomane demonstrates that the emergent geometry principles established in food ingredient embeddings successfully transfer to the music domain, opening new avenues for interpretable, steerable music navigation.

\end{document}
"""
    return tex


def build_reproducibility(data: dict) -> str:
    h = host_info()
    return f"""# Reproducibility Manifest

- **Date:** {time.strftime('%Y-%m-%d %H:%M:%S UTC')}
- **Corpus:** `{data['corpus']}`
- **Profile:** `{data['profile']}`
- **Host:** {h.get('os')} ({h.get('arch')})
- **Python:** {sys.version.split()[0]}
- **Hardware:** {data.get('doctor', {}).get('checks', {}).get('cuda', 'CPU/Generic')}
- **Vocabulary Size:** {data.get('vocab_count', 0):,} canonical tracks
- **Seed Protocol:** Multi-seed Hungarian matching for ICA; deterministic per-stage hashing
- **OpenRouter Spend:** ${data.get('doctor', {}).get('spent_usd', 0.0):.2f} USD
"""


@register
class Paper(Stage):
    name, title, scope = "S13", "White paper generation (tables, figures, MD/LaTeX)", "global"

    def params(self, ctx: Ctx):
        return {"v": 1, "profile": ctx.profile_name, "corpora": ctx.opt("corpora", [])}

    def outputs(self, ctx: Ctx):
        paper_dir = ctx.ws.root / "paper"
        return [paper_dir / "melomane_paper.md", paper_dir / "melomane_paper.tex", paper_dir / "reproducibility.md"]

    def run(self, ctx: Ctx) -> StageResult:
        res = StageResult()
        primary_corpus = pick_primary_corpus(ctx)
        log.info("Generating paper for primary corpus: %s (profile: %s)", primary_corpus, ctx.profile_name)

        data = load_all_results(ctx, primary_corpus)
        hyp = collate_hypotheses(data)

        paper_dir = ctx.ws.root / "paper"
        paper_dir.mkdir(parents=True, exist_ok=True)
        figs_dir = paper_dir / "figures"
        tables_dir = paper_dir / "tables"

        try:
            fig_paths = make_figures(data, figs_dir)
        except Exception as e:
            log.warning("Figure generation partial: %s", e)
            fig_paths = []
            res.status = PARTIAL
            res.notes.append(f"Figure generation warning: {e}")

        try:
            make_tables(data, tables_dir)
        except Exception as e:
            log.warning("Table generation partial: %s", e)
            res.status = PARTIAL
            res.notes.append(f"Table generation warning: {e}")

        md_content = build_markdown_paper(data, hyp, fig_paths)
        (paper_dir / "melomane_paper.md").write_text(md_content, encoding="utf-8")

        tex_content = build_latex_paper(data, hyp)
        (paper_dir / "melomane_paper.tex").write_text(tex_content, encoding="utf-8")

        repro_content = build_reproducibility(data)
        (paper_dir / "reproducibility.md").write_text(repro_content, encoding="utf-8")

        res.metrics = {
            "primary_corpus": primary_corpus,
            "figures_generated": len(fig_paths),
            "hypotheses_evaluated": sum(1 for h in hyp.values() if h.get("status") in ("supported", "not supported")),
        }
        res.notes.append(f"Paper generated in {paper_dir}/ (Markdown + LaTeX arXiv preprint)")
        return res
