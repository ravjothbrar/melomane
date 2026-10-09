"""Results-driven paper text: one structured document, rendered to Markdown and LaTeX.

Rules this module enforces (CLAUDE.md section 4, PLAN.md section 11):

* every number comes from ``data`` (the parsed ``results/`` files) - none is typed here;
* a missing number is printed as ``n/a``, never as ``0.000`` (a missing stratum is not a zero);
* no sentence states a finding unconditionally: findings are phrased from the hypothesis *statuses*, so a null or negative
  result reads as plainly as a positive one;
* the limitations section is always present and carries the caveats the data force on us.

``data`` is what :func:`melomane.stages.s13_paper.load_all_results` returns; ``hyp`` is :func:`collate_hypotheses`.
"""
from __future__ import annotations

import datetime as _dt
import math
from typing import Any, Sequence

MODELS = ("cooc", "composition", "core")
NAME = {"cooc": "Melomane-Cooc", "composition": "Melomane-Composition", "core": "Melomane-Core"}
STRATA = (("a_baked|rho", "Baked-in (a) ρ"), ("b_heldout|rho", "Held-out (b) ρ"), ("c_external|rho", "External (c) ρ"),
          ("cultural|cohen_d", "Cultural d"))
NA = "n/a"


# ---------------------------------------------------------------------------------------------------------------------
# number formatting - missing is "n/a", never zero
# ---------------------------------------------------------------------------------------------------------------------
def _ok(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def num(x: Any, nd: int = 3) -> str:
    return f"{x:.{nd}f}" if _ok(x) else NA


def integer(x: Any) -> str:
    return f"{int(x):,}" if _ok(x) else NA


def pct(x: Any, nd: int = 0) -> str:
    return f"{100 * x:.{nd}f}%" if _ok(x) else NA


def ci(lo: Any, hi: Any, nd: int = 3) -> str:
    return f"[{num(lo, nd)}, {num(hi, nd)}]" if _ok(lo) and _ok(hi) else NA


def mean_sd(d: dict | None, nd: int = 3) -> str:
    """'0.338 ± 0.004 (n=3)' from an aggregate {mean, sd, n_seeds}; ``n/a`` when the stratum was not evaluated."""
    if not d or not _ok(d.get("mean")):
        return NA
    n = d.get("n_seeds")
    sd = d.get("sd")
    if _ok(sd) and n and n > 1:
        return f"{d['mean']:.{nd}f} ± {sd:.{nd}f}"
    return f"{d['mean']:.{nd}f}"


def get(d: Any, *path: Any, default: Any = None) -> Any:
    for p in path:
        if isinstance(d, dict) and p in d:
            d = d[p]
        elif isinstance(d, (list, tuple)) and isinstance(p, int) and -len(d) <= p < len(d):
            d = d[p]
        else:
            return default
    return d


# ---------------------------------------------------------------------------------------------------------------------
# document model: ("h", level, text) ("p", text) ("ul", [text]) ("table", header, rows, caption) ("fig", path, caption) ("eq", latex)
# ("note", text)
# ---------------------------------------------------------------------------------------------------------------------
Block = tuple


def status_counts(hyp: dict) -> dict[str, int]:
    out: dict[str, int] = {}
    for h in hyp.values():
        s = str(h.get("status", "pending")).lower()
        out[s] = out.get(s, 0) + 1
    return out


def _hyp_sentence(k: str, h: dict) -> str:
    return f"{k} ({h.get('status', 'pending')})"


def abstract(data: dict, hyp: dict) -> str:
    """Abstract assembled from the numbers and the statuses.  Each clause appears only if the data support writing it."""
    corp = data.get("corpus", "?")
    V, A = data.get("vocab_count"), data.get("artist_count")
    sc = status_counts(hyp)
    parts = [
        "Melomane transfers the Epicure methodology (Radzikowski & Chen, arXiv:2604.22776, arXiv:2605.22391) from food ingredients to "
        "music tracks: three skip-gram models share one vocabulary and one training objective and differ only in the random-walk schema "
        "(listening sessions, measured audio and lyric feature clusters, or a blend). "
        f"We train them on {integer(V)} tracks by {integer(A)} artists (corpus `{corp}`) and test eight pre-registered hypotheses with "
        "artist-grouped cross-validation."
    ]
    n_sup, n_not, n_ne = sc.get("supported", 0), sc.get("not supported", 0), sc.get("not evaluable", 0)
    parts.append(f"Outcome: {n_sup} supported, {n_not} not supported, {n_ne} not evaluable"
                 + (f", {sum(v for k, v in sc.items() if k not in ('supported', 'not supported', 'not evaluable'))} mixed/inconclusive"
                    if sum(v for k, v in sc.items() if k not in ('supported', 'not supported', 'not evaluable')) else "") + ".")
    h1 = hyp.get("H1", {})
    ev_b = get(h1, "evidence", "b")
    if h1.get("status") == "supported" and isinstance(ev_b, dict):
        parts.append(f"Measured-feature structure is recovered better by the Composition model than by Cooc on held-out acoustic properties "
                     f"(Δρ = {num(ev_b.get('delta_rho'))}, 95% CI {ci(ev_b.get('ci_low'), ev_b.get('ci_high'))}, artist-clustered bootstrap).")
    elif h1.get("status") == "not supported":
        parts.append("Contrary to the Epicure ordering, the Composition model did not exceed Cooc on the held-out strata (H1 not supported).")
    elif h1.get("status") in ("inconclusive", "mixed"):
        parts.append("The Composition-versus-Cooc comparison on the held-out strata was inconclusive (H1).")
    h7 = hyp.get("H7", {})
    if h7.get("status") in ("supported", "not supported"):
        parts.append(f"Steering by spherical interpolation {'does' if h7['status'] == 'supported' else 'does not'} reliably increase target-tag "
                     "precision with rotation angle (H7).")
    parts.append("The embeddings themselves are LLM-free; a language model is used only for curation, tagging and naming. "
                 "This is a preprint-pending write-up generated from the result files.")
    return " ".join(parts)


def build_blocks(data: dict, hyp: dict, fig_paths: Sequence[str] = ()) -> list[Block]:
    corp, prof = data.get("corpus", "?"), data.get("profile", "?")
    s1, s2, s3 = data.get("s1", {}), data.get("s2", {}), data.get("s3", {})
    s4, s5, s6, s7 = data.get("s4", {}), data.get("s5", {}), data.get("s6", {}), data.get("s7", {})
    s8, s9 = data.get("s8", {}), data.get("s9", {})
    summary = s7.get("summary_by_model", {})
    cov = data.get("listening_coverage", {})
    B: list[Block] = []

    B.append(("title", "Melomane: emergent geometry in music-track embeddings, and what measured composition adds to listening behaviour"))
    B.append(("meta", f"Ravjoth Brar · preprint pending · generated {_dt.date.today().isoformat()} from `results/` · corpus `{corp}` · profile `{prof}`"))
    if corp == "synth":
        B.append(("note", "DEMO RUN: compiled from the synthetic corpus to verify the pipeline. Numbers reflect planted structure, not music."))

    B.append(("h", 1, "Abstract"))
    B.append(("p", abstract(data, hyp)))

    B.append(("h", 1, "1. Introduction"))
    B.append(("p", "A track can be placed in a space by what people play around it (collaborative context) or by what it is made of (timbre, rhythm, "
                   "harmony, loudness, lyric themes). Epicure showed for food that holding the skip-gram objective fixed and changing only the "
                   "walk schema between co-occurrence and a typed compound layer yields different but complementary geometries. Melomane runs the "
                   "same controlled comparison for music and asks whether the findings transfer; it does not claim to reproduce Epicure's numbers."))
    B.append(("p", "Music adds a trap that food does not have: tracks by one artist share voice, production and audience, so any probe that can "
                   "see two tracks of the same artist on both sides of a train/test split measures artist identity. Every probe here is grouped by "
                   "artist, and nearest-neighbour purity excludes the query's artist."))

    B.append(("h", 1, "2. Data and curation"))
    rows = [["Tracks in the vocabulary", integer(data.get("vocab_count"))],
            ["Distinct artists (normalised name)", integer(data.get("artist_count"))],
            ["Tracks with audio/lyric features", integer(cov.get("tracks_with_audio_features"))],
            ["Tracks with any listening data", integer(cov.get("tracks_with_listening_data"))],
            ["Listening sessions", integer(s3.get("n_sessions"))],
            ["Mean / median session length", f"{num(s3.get('session_len_mean'), 1)} / {num(s3.get('session_len_median'), 0)}"],
            ["Behaviour-graph edges (mean degree)", f"{integer(s3.get('n_edges_undirected'))} ({num(s3.get('mean_degree'), 1)})"],
            ["Compound nodes", integer(s4.get("n_compounds"))],
            ["Consolidation ratio (curation)", num(s1.get("consolidation_ratio"), 2)]]
    B.append(("table", ["Quantity", "Value"], rows, "Corpus and graph sizes, read from the stage result files."))
    B.append(("p", "Sessions are listening events split at 30-minute gaps (5–200 distinct tracks). Behaviour edges are positive-NPMI pairs "
                   "(minimum 5 co-occurrences, top 250 per track). Compound nodes are k-means clusters per feature family over whitened PCA blocks of "
                   "the measured audio and lyric descriptors; each track links to its three nearest clusters per family. Voice, emotion (GEMS), lyric "
                   "sentiment/VAD and all learned embeddings are held out of the compound vocabulary and used only as probe targets or comparisons."))
    if _ok(cov.get("tracks_with_listening_data")) and _ok(cov.get("tracks_with_audio_features")):
        B.append(("p", f"Only {integer(cov['tracks_with_listening_data'])} of the {integer(cov['tracks_with_audio_features'])} tracks that have audio features "
                       "have any listening data, so the behaviour model (and therefore the three-way comparison) is limited to those."))

    B.append(("h", 1, "3. Models and training"))
    seeds = {m: get(s5, "models", m, "seeds") for m in MODELS}
    train_rows = []
    for m in MODELS:
        e = get(s5, "models", m, default={}) or {}
        train_rows.append([NAME[m], integer(e.get("seeds")), integer(e.get("epochs")), integer(e.get("tokens_per_epoch")),
                           num(e.get("seconds"), 0), num(e.get("neighbour_recall_at10_seed0_vs_1"), 2)])
    B.append(("table", ["Model", "Seeds", "Epochs", "Tokens / epoch", "Seconds (last seed)", "Seed-to-seed neighbour recall@10"], train_rows,
              "Training summary (S5). Seed-to-seed recall is the fraction of top-10 neighbours shared by two seeds of the same model."))
    B.append(("p", "Cooc walks only the behaviour graph; Composition walks only the typed compound templates (no track–track template); Core adds ten "
                   "track–track templates per round to all compound templates. All three share the vocabulary, dimension (300), window, negative "
                   "sampling and learning rate. The walk budget is scaled below the Epicure paper's (see Limitations); the equal-token control and the "
                   "paper-budget subset check test whether that matters."))

    B.append(("h", 1, "4. Results"))
    B.append(("h", 2, "4.1 Geometry"))
    geo = []
    for m in MODELS:
        d = summary.get(m, {})
        geo.append([NAME[m], mean_sd(d.get("PR"), 1), mean_sd(d.get("mean_cos"), 3)])
    B.append(("table", ["Model", "Participation ratio (of 300)", "Mean pairwise cosine"], geo, "Embedding geometry; mean ± sd over seeds."))

    B.append(("h", 2, "4.2 Direction quality by probe stratum"))
    rows = [[NAME[m]] + [mean_sd(summary.get(m, {}).get(k)) for k, _ in STRATA] for m in MODELS]
    B.append(("table", ["Model"] + [l for _, l in STRATA], rows,
              "Mean over probes in each stratum of the artist-grouped, fold-trained direction score (Spearman ρ; Cohen's d for the cultural stratum). "
              "n/a means the stratum had no evaluable probe in this run, not a score of zero."))
    sizes = s7.get("strata_sizes", {})
    if sizes:
        B.append(("p", "Probes per stratum: " + ", ".join(f"{k}: {v}" for k, v in sizes.items()) + f". Evaluation tracks (test artists only): {integer(s7.get('n_eval_tracks'))}."))
    paired = s7.get("paired", {})
    prow = []
    for pair, label in (("composition_minus_cooc", "Composition − Cooc"), ("core_minus_cooc", "Core − Cooc"), ("composition_minus_core", "Composition − Core")):
        for st, stl in (("a_baked", "(a)"), ("b_heldout", "(b)"), ("c_external", "(c)")):
            e = get(paired, pair, "strata", st)
            if e:
                prow.append([label, stl, num(e.get("delta_rho")), ci(e.get("ci_low"), e.get("ci_high")), num(e.get("p_boot"), 3), integer(e.get("n_targets")), pct(e.get("share_targets_positive"))])
    if prow:
        B.append(("table", ["Contrast", "Stratum", "Δρ", "95% CI", "p", "Probes", "Share of probes > 0"], prow,
                  "Paired differences, artist-clustered bootstrap (seed 0 models)."))

    B.append(("h", 2, "4.3 Hypotheses"))
    hrows = []
    for k in sorted(hyp, key=lambda x: int(x[1:])):
        h = hyp[k]
        hrows.append([k, h.get("claim", ""), str(h.get("status", "pending")).upper()])
    B.append(("table", ["", "Claim", "Outcome"], hrows, "Pre-registered hypotheses (analysis_plan.lock.json) and their outcomes, nulls included."))
    notes = []
    h2 = hyp.get("H2", {})
    if h2.get("status") == "not supported" and isinstance(h2.get("evidence"), dict):
        pr = h2["evidence"].get("PR", {})
        notes.append("H2: participation ratio " + ", ".join(f"{NAME[m]} {num(pr.get(m), 1)}" for m in MODELS if m in pr) + ".")
    h5 = hyp.get("H5", {})
    if isinstance(h5.get("evidence"), dict) and h5["evidence"].get("n_dims"):
        e = h5["evidence"]
        notes.append(f"H5: {pct(e.get('share_positive_ci'))} of {e['n_dims']} LLM dimensions with a measured counterpart correlate positively with a CI excluding 0"
                     + (f"; wrong-sign dimensions: {', '.join(e['wrong_sign_dims'])}" if e.get("wrong_sign_dims") else "") + ".")
    if notes:
        B.append(("ul", notes))

    B.append(("h", 2, "4.4 Emergent modes"))
    mrows = []
    for m in MODELS:
        d = s6.get(m)
        if d:
            c = d.get("coherence", {})
            dc = d.get("driver_counts", {})
            mrows.append([NAME[m], integer(d.get("n_factors_stable")), integer(d.get("n_modes")), f"{num(c.get('coherence'), 2)} vs {num(c.get('random'), 2)} (×{num(c.get('ratio'), 1)})",
                          ", ".join(f"{k} {v}" for k, v in dc.items()), pct(get(d, "audit", "agreement"))])
    if mrows:
        B.append(("table", ["Model", "Stable ICA factors", "Modes", "Coherence vs random", "Mode drivers", "Label audit agreement"], mrows,
                  "FastICA on category-residualised embeddings → GMM modes (S6). Drivers: whether a mode's members share behaviour, acoustic or lyric compounds."))

    B.append(("h", 2, "4.5 Steering by SLERP"))
    srows = []
    for m in MODELS:
        d = get(s8, "models", m)
        if d:
            inc = d.get("increasing_share_60", {})
            mono = d.get("monotonicity", {})
            srows.append([NAME[m], integer(d.get("n_pairs")), integer(d.get("n_poles")), f"{pct(inc.get('share'))} {ci(inc.get('ci_low'), inc.get('ci_high'), 2)}",
                          num(mono.get("mean_rho"), 2), num(d.get("snr_median_all"), 2)])
    if srows:
        B.append(("table", ["Model", "Seed×pole pairs", "Poles", "Poles with precision increasing to 60° (95% CI)", "Mean Spearman ρ(θ, precision)", "Median orthogonal-residual SNR"], srows,
                  "Quantitative direction test: precision@10 of the pole's tag among artist-disjoint retrievals as the rotation angle grows (S8)."))
    B.append(("eq", r"d_\perp = d - (d\cdot v)\,v,\qquad q(\theta) = \cos\theta\, v + \sin\theta\, \frac{d_\perp}{\lVert d_\perp\rVert}"))

    B.append(("h", 2, "4.6 LLM labels versus measured features"))
    rows = [[r.get("dim"), r.get("measured"), integer(r.get("n")), num(r.get("rho")), ci(r.get("ci_lo"), r.get("ci_hi"))] for r in (s2.get("table") or [])]
    if rows:
        B.append(("table", ["LLM dimension", "Measured counterpart", "n", "Spearman ρ", "95% CI"], rows,
                  "Agreement between LLM-labelled dimensions (artist + title + tags only) and instrument-measured features, before touching embeddings."))
    else:
        B.append(("p", "No LLM labels with a measured counterpart were available in this run."))

    ext = _extension_rows(s9)
    if ext:
        B.append(("h", 2, "4.7 Extensions"))
        B.append(("table", ["Extension", "Status", "Detail"], ext, "Extensions beyond the original papers (S9); full tables are in results/."))

    for i, p in enumerate(fig_paths):
        cap = {"fig1": "Probe-stratum means by model.", "fig2": "Target-tag precision@10 against rotation angle.", "fig3": "Compression frontier."}
        B.append(("fig", p, next((v for k, v in cap.items() if k in p), "Figure.")))

    B.append(("h", 1, "5. Limitations"))
    lim = [
        "Onion's listening data come from a Last.fm-era, Western-skewed population; findings are about this catalogue, not music in general.",
        "Only tracks with listening data can be in the three-way comparison" + (f" ({integer(cov.get('tracks_with_listening_data'))} of {integer(cov.get('tracks_with_audio_features'))})" if _ok(cov.get("tracks_with_listening_data")) else "") + ".",
        "Artist and title strings come from a separate metadata table of unconfirmed provenance (see DATA_LICENSE.md); curation here is therefore limited to rules and cannot recover versions/remixes beyond normalised (artist, title) keys.",
        "Artist structure: same-artist tracks are strongly correlated; all probes are artist-grouped, but unmodelled structure (label, era, collaborators) can still leak.",
        "LLM labels are noisy; dimensions below the reliability threshold are excluded from headline probes, and the probe strata that depend on them are reported as n/a when empty.",
        "Features are track-level aggregates of 30-second clips, not frame-level audio; instrumental tracks have no lyric edges.",
        "The walk budget is scaled below the Epicure paper's (profile `" + str(prof) + "`); the equal-token control and the paper-budget subset check bound, but do not remove, this caveat.",
        "Probes target properties of the compound inputs themselves (stratum a), properties held out of them (b) and LLM labels (c); only (b) and (c) test generalisation beyond what the Composition model was given.",
    ]
    B.append(("ul", lim))

    B.append(("h", 1, "6. Ethics, licences and data handling"))
    B.append(("p", "The embeddings are learned without any language model. Lyrics and audio are never redistributed and never sent to a language model; "
                   "no per-user data is released. Upstream data: Music4All-Onion (CC BY 4.0; Moscati et al., CIKM 2022), listening events from Last.fm. "
                   "Code is MIT; derived artefacts are CC BY 4.0 subject to DATA_LICENSE.md."))

    B.append(("h", 1, "7. Reproducibility"))
    B.append(("p", "Every figure and table is generated from the files under `results/` by `python -m melomane run-all`; see `paper/reproducibility.md` "
                   "for the environment, seeds and LLM spend, and `analysis_plan.lock.json` (including its amendments) for the pre-registration trail."))
    return B


def _extension_rows(s9: dict) -> list[list[str]]:
    rows: list[list[str]] = []
    labels = {"E1": "E1 compression frontier", "E2": "E2 acoustic vs lyric attribution", "E3": "E3 partial-hub regime", "E4": "E4 equal-token control",
              "E5": "E5 external-embedding agreement", "E6": "E6 temporal stability", "E7": "E7 curation policy", "E8": "E8 sensitivity grid"}
    for k, lab in labels.items():
        e = s9.get(k)
        if not isinstance(e, dict):
            continue
        status = str(e.get("status", "n/a"))
        detail = ""
        if k == "E1" and isinstance(e.get("ship_small"), dict):
            detail = "; ".join(f"{m}: {v.get('dims')}-d {v.get('dtype')} = {num(v.get('size_mb'), 1)} MB, retention {pct(v.get('retention'))}" for m, v in e["ship_small"].items() if isinstance(v, dict))
        elif k == "E2" and isinstance(e.get("by_stratum"), dict):
            p = e["by_stratum"].get("primary") or {}
            detail = f"gain over Cooc split: acoustic {pct(p.get('share_acoustic'))}, lyric {pct(p.get('share_lyric'))}"
        elif status != "done":
            detail = str(e.get("reason") or e.get("note") or "")
        rows.append([lab, status, detail])
    return rows


# ---------------------------------------------------------------------------------------------------------------------
# renderers
# ---------------------------------------------------------------------------------------------------------------------
def _md_cell(s: Any) -> str:
    return str(s).replace("|", "\\|").replace("\n", " ")


def render_markdown(blocks: Sequence[Block]) -> str:
    out: list[str] = []
    for b in blocks:
        t = b[0]
        if t == "title":
            out.append(f"# {b[1]}\n")
        elif t == "meta":
            out.append(f"_{b[1]}_\n")
        elif t == "note":
            out.append(f"> **{b[1]}**\n")
        elif t == "h":
            out.append(f"{'#' * (b[1] + 1)} {b[2]}\n")
        elif t == "p":
            out.append(b[1] + "\n")
        elif t == "ul":
            out.append("\n".join(f"- {x}" for x in b[1]) + "\n")
        elif t == "eq":
            out.append(f"$$ {b[1]} $$\n")
        elif t == "fig":
            out.append(f"![{b[2]}]({b[1]})\n\n*{b[2]}*\n")
        elif t == "table":
            _, head, rows, cap = b
            lines = ["| " + " | ".join(_md_cell(h) for h in head) + " |", "|" + "---|" * len(head)]
            lines += ["| " + " | ".join(_md_cell(c) for c in r) + " |" for r in rows]
            out.append("\n".join(lines) + f"\n\n*{cap}*\n")
    return "\n".join(out)


_TEX = {"&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\^{}", "\\": r"\textbackslash{}",
        "±": r"$\pm$", "×": r"$\times$", "−": "-", "θ": r"$\theta$", "ρ": r"$\rho$", "Δ": r"$\Delta$", "≥": r"$\geq$", "≤": r"$\leq$", "→": r"$\rightarrow$",
        "–": "--", "—": "---", "·": r"$\cdot$", "σ": r"$\sigma$", "°": r"$^\circ$"}


def tex_escape(s: Any) -> str:
    s = str(s).replace("`", "")
    return "".join(_TEX.get(c, c) for c in s)


def render_latex(blocks: Sequence[Block]) -> str:
    title = next((b[1] for b in blocks if b[0] == "title"), "Melomane")
    meta = next((b[1] for b in blocks if b[0] == "meta"), "")
    body: list[str] = []
    in_abstract = False
    for b in blocks:
        t = b[0]
        if t in ("title", "meta"):
            continue
        if t == "h":
            if in_abstract:
                body.append("\\end{abstract}\n")
                in_abstract = False
            if b[2] == "Abstract":
                body.append("\\begin{abstract}")
                in_abstract = True
                continue
            name = b[2].split(". ", 1)[-1] if b[2][:1].isdigit() else b[2]
            body.append(("\\section*{" if b[1] == 1 else "\\subsection*{") + tex_escape(name.split(" ", 1)[-1] if name[:1].isdigit() else name) + "}")
        elif t == "p":
            body.append(tex_escape(b[1]) + "\n")
        elif t == "note":
            body.append("\\noindent\\textbf{" + tex_escape(b[1]) + "}\n")
        elif t == "ul":
            body.append("\\begin{itemize}\n" + "\n".join("\\item " + tex_escape(x) for x in b[1]) + "\n\\end{itemize}\n")
        elif t == "eq":
            body.append("\\[ " + b[1] + " \\]\n")
        elif t == "fig":
            body.append("\\begin{figure}[h]\\centering\\includegraphics[width=0.85\\linewidth]{" + b[1].replace("\\", "/") + "}\\caption{" + tex_escape(b[2]) + "}\\end{figure}\n")
        elif t == "table":
            _, head, rows, cap = b
            spec = "l" + "p{0.45\\linewidth}" * (len(head) > 3 and False) + "c" * (len(head) - 1)
            if len(head) == 3 and head[0] == "":
                spec = "lp{0.62\\linewidth}c"
            body.append("\\begin{table}[h]\\centering\\small\n\\begin{tabular}{" + spec + "}\n\\toprule\n"
                        + " & ".join("\\textbf{" + tex_escape(h) + "}" for h in head) + " \\\\\n\\midrule\n"
                        + "\n".join(" & ".join(tex_escape(c) for c in r) + " \\\\" for r in rows)
                        + "\n\\bottomrule\n\\end{tabular}\n\\caption{" + tex_escape(cap) + "}\n\\end{table}\n")
    if in_abstract:
        body.append("\\end{abstract}\n")
    return ("\\documentclass[11pt,a4paper]{article}\n\\usepackage[utf8]{inputenc}\n\\usepackage[T1]{fontenc}\n\\usepackage{amsmath,amssymb}\n"
            "\\usepackage{booktabs}\n\\usepackage{graphicx}\n\\usepackage{hyperref}\n\\usepackage[margin=1in]{geometry}\n"
            "\\title{\\textbf{" + tex_escape(title) + "}}\n\\author{Ravjoth Brar}\n\\date{}\n\\begin{document}\n\\maketitle\n"
            "\\begin{center}\\small " + tex_escape(meta) + "\\end{center}\n\n" + "\n".join(body) + "\n\\end{document}\n")


# ---------------------------------------------------------------------------------------------------------------------
# README: the only numbers in the README are the ones S13 injects here
# ---------------------------------------------------------------------------------------------------------------------
README_BEGIN, README_END = "<!-- BEGIN GENERATED RESULTS -->", "<!-- END GENERATED RESULTS -->"


def readme_results_block(data: dict, hyp: dict) -> str:
    """Markdown for the README's results section (generated, with provenance line).  Same sources and n/a rules as the paper."""
    blocks = build_blocks(data, hyp)
    keep = []
    want = {"4.1 Geometry", "4.2 Direction quality by probe stratum", "4.3 Hypotheses"}
    take = False
    for b in blocks:
        if b[0] == "h":
            take = b[2] in want
            continue
        if take and b[0] in ("table", "ul"):
            keep.append(b)
        elif take and b[0] == "p":
            keep.append(b)
    head = (f"_Generated by stage S13 from `results/` on {_dt.date.today().isoformat()} · corpus `{data.get('corpus', '?')}` · profile `{data.get('profile', '?')}` · "
            f"{integer(data.get('vocab_count'))} tracks, {integer(data.get('artist_count'))} artists. Do not edit by hand._\n")
    if data.get("corpus") == "synth":
        head += "\n> **Demo corpus** - these numbers come from planted synthetic structure, not music.\n"
    return README_BEGIN + "\n" + head + "\n" + render_markdown(keep).replace("\n#", "\n####") + README_END


def inject_readme(readme: str, block: str) -> str:
    """Replace the generated region of a README (or return it unchanged if the markers are absent)."""
    if README_BEGIN not in readme or README_END not in readme:
        return readme
    a, b = readme.index(README_BEGIN), readme.index(README_END) + len(README_END)
    return readme[:a] + block + readme[b:]
