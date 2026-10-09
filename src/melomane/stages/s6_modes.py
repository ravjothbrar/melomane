"""S6 - ICA factors, GMM modes, LLM mode naming, supervised poles (paper 2 §3 + supplement)."""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
import scipy.sparse as sp

from ..annotation import DIMS
from ..ica_modes import factor_modes, residualise, stable_ica
from ..llm import BudgetExceeded
from ..modes import choose_modes_members_for_labelling, mode_drivers, project_sources, supervised_poles
from ..pipeline import Ctx, Stage, StageResult, register, PARTIAL
from ..probes import mode_coherence
from ..prompts import load_prompt
from ..runtime import get_logger, jsonable, write_json
from .common import available_models, corpus_of, labels_of, load_embeddings, vocab_of
from .s1_curation import get_llm

log = get_logger("melomane.s6")
MODES_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["modes"], "properties": {"modes": {"type": "array", "items": {
    "type": "object", "additionalProperties": False, "required": ["mode", "label", "property"],
    "properties": {"mode": {"type": "integer"}, "label": {"type": "string"}, "property": {"type": "string"}}}}}}
AUDIT_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["answers"], "properties": {"answers": {"type": "array", "items": {
    "type": "object", "additionalProperties": False, "required": ["q", "same"], "properties": {"q": {"type": "integer"}, "same": {"type": "string", "enum": ["yes", "no"]}}}}}}
MODE_MODELS = ["cooc", "composition", "core", "composition_acoustic", "composition_lyric", "cooc_artistmasked", "cooc_avg"]


def name_modes(llm, prompt, vocab, labels, E, modes_meta, members, poles, workers=8, audit_n=0, audit_prompt=None, seed=0):
    """Ask the LLM for label+property per mode (member titles/artists/categories/tags only - never lyrics)."""
    v, system = prompt
    payload = []
    for i, (mem, pole) in enumerate(zip(members, poles)):
        top = choose_modes_members_for_labelling(E, mem, pole, 25)
        payload.append({"mode": i, "drivers": modes_meta[i]["driver"], "members": [
            {"artist": str(vocab.at[int(c), "artist"])[:60], "title": str(vocab.at[int(c), "title"])[:80], "category": str(vocab.at[int(c), "category"]),
             "decade": (f"{int(vocab.at[int(c), 'year']) // 10 * 10}s" if pd.notna(vocab.at[int(c), "year"]) else ""), "tags": str(vocab.at[int(c), "tags"])[:60]} for c in top]})
    chunks = [payload[i:i + 15] for i in range(0, len(payload), 15)]
    jobs = [dict(system=system, user=json.dumps({"modes": ch}, ensure_ascii=False), schema=MODES_SCHEMA, name="modes") for ch in chunks]
    labels_out = {i: {"label": "", "property": ""} for i in range(len(members))}
    try:
        res = llm.complete_json_many(jobs, workers=workers) if jobs else []
    except BudgetExceeded:
        res = []
    for ch, r in zip(chunks, res):
        if isinstance(r, Exception):
            continue
        ids = {m["mode"] for m in ch}
        for m in r.get("modes", []):
            if m.get("mode") in ids:
                labels_out[m["mode"]] = {"label": m["label"], "property": m["property"]}
    audit = {}
    if audit_n and audit_prompt is not None and members:
        # second, independently ordered naming pass -> yes/no agreement judged by a third prompt
        rng = np.random.default_rng(seed)
        pick = rng.choice(len(members), min(audit_n, len(members)), replace=False)
        p2 = []
        for i in pick:
            item = dict(payload[int(i)]); ms = list(item["members"]); rng.shuffle(ms); item["members"] = ms; item["mode"] = int(i)
            p2.append(item)
        r2 = llm.complete_json_many([dict(system=system, user=json.dumps({"modes": p2[j:j + 15]}, ensure_ascii=False), schema=MODES_SCHEMA, name="modes")
                                      for j in range(0, len(p2), 15)], workers=workers)
        second = {}
        for r in r2:
            if isinstance(r, Exception): continue
            for m in r.get("modes", []): second[m["mode"]] = m["property"]
        qs = [{"q": int(i), "a": labels_out[int(i)]["property"], "b": second[int(i)]} for i in pick if int(i) in second and labels_out[int(i)]["property"]]
        if qs:
            ra = llm.complete_json(audit_prompt[1], json.dumps({"pairs": qs}), AUDIT_SCHEMA, name="audit_modes")
            yes = [a["same"] == "yes" for a in ra.get("answers", [])]
            audit = {"n": len(yes), "agreement": float(np.mean(yes)) if yes else None, "prompt_version": audit_prompt[0]}
    return labels_out, audit


@register
class Modes(Stage):
    name, title, deps = "S6", "ICA, modes & poles", ("S5", "S2")

    def params(self, ctx):
        return {"v": 3, "ica": ctx.profile["ica"], "have_llm": bool(ctx.opt("llm") is not None or __import__("os").environ.get("OPENROUTER_API_KEY")), "models": MODE_MODELS}

    def outputs(self, ctx):
        return [ctx.stage_dir("S6") / "S6_done.json"]

    def run(self, ctx: Ctx) -> StageResult:
        prof = ctx.profile; ic = prof["ica"]
        vocab, labels = vocab_of(ctx), labels_of(ctx)
        d3, d4, d6 = ctx.stage_dir("S3"), ctx.stage_dir("S4"), ctx.stage_dir("S6")
        mask = np.load(d3 / "eval_mask.npy")
        z = np.load(d3 / "cooc_edges.npz"); V = len(vocab)
        tt = sp.csr_matrix((z["npmi"], (z["rows"], z["cols"])), shape=(V, V))
        comp = sp.load_npz(d4 / "compound_edges.npz").tocsr() if (d4 / "compound_edges.npz").exists() else None
        kind_cols = None
        if comp is not None:
            meta = pd.read_parquet(d4 / "compound_meta.parquet")
            kind_cols = {k: np.flatnonzero((meta["kind"] == k).to_numpy()) for k in ("acoustic", "lyric")}
        llm = get_llm(ctx)
        P = {"modes": load_prompt(ctx.ws.root, "modes_v1"), "audit": load_prompt(ctx.ws.root, "modes_audit_v1")} if llm is not None else None
        reliable = None
        if ctx.has_result("S2_reliability"):
            ex = set(ctx.load_result("S2_reliability").get("excluded_dims", []))
            reliable = [x for x in DIMS if x not in ex]
        cat = vocab["category"].astype(str).to_numpy()
        res, summary = StageResult(), {}
        models = [m for m in MODE_MODELS if m in available_models(ctx)]
        for m in models:
            E = load_embeddings(ctx, m).astype(np.float64)
            idx = np.flatnonzero(mask)
            out = d6 / m; out.mkdir(parents=True, exist_ok=True)
            Er = residualise(E[idx], cat[idx])
            rng = np.random.default_rng(ctx.seed("S6", m))
            fit_rows = rng.choice(len(idx), min(len(idx), 40000), replace=False)
            r = stable_ica(Er[fit_rows], ic["n_components"], ic["n_seeds"], ic["split_half_threshold"], 5, seed=ctx.seed("S6", m, "ica") % 10_000)
            nF = r.directions.shape[0]
            srcs_masked = project_sources(Er, r.directions) if nF else np.zeros((len(idx), 0))
            factor_src = np.zeros((V, nF)); factor_src[idx] = srcs_masked
            np.savez(out / "ica.npz", directions=r.directions, sources=factor_src.astype(np.float32), stability=r.stability, seed_stability=r.seed_stability, all_split_half=r.all_split_half)
            np.save(out / "factor_poles.npy", r.directions.astype(np.float32))
            members, mode_meta, poles = [], [], []
            for f in range(nF):
                for md in factor_modes(E[idx], srcs_masked[:, f], f, ic["top_quantile"], range(ic["gmm_k_range"][0], ic["gmm_k_range"][-1] + 1), ic["min_members"], seed=ctx.seed("S6", m, f) % 10_000):
                    members.append(idx[md.members]); poles.append(md.pole)
                    mode_meta.append({"factor": int(md.factor), "bic_k": int(md.bic_k), "size": int(len(md.members))})
            coh = mode_coherence(E[idx], [np.searchsorted(idx, mm) for mm in members], seed=ctx.seed("S6", m, "coh")) if members else {"coherence": None, "random": None, "ratio": None}
            drivers = mode_drivers(members, comp, kind_cols, tt, n_random=15, seed=ctx.seed("S6", m, "drv")) if members else []
            for mm, dv in zip(mode_meta, drivers):
                mm.update(dv)
            named = {i: {"label": "", "property": ""} for i in range(len(members))}
            audit = {}
            if llm is not None and members:
                named, audit = name_modes(llm, P["modes"], vocab, labels, E, mode_meta, members, poles, prof["llm"]["mode_label_workers"],
                                          audit_n=max(5, len(members) // 10), audit_prompt=P["audit"], seed=ctx.seed("S6", m, "name"))
            modes_json = {"model": m, "n_modes": len(members), "modes": [
                {"mode_id": i, **mode_meta[i], "members": [int(c) for c in members[i]], "label": named[i]["label"], "property": named[i]["property"],
                 "coherence": float(np.mean((lambda U: (U @ U.T).sum() - len(U)) (E[members[i]] / (np.linalg.norm(E[members[i]], axis=1, keepdims=True) + 1e-9)) / max(len(members[i]) * (len(members[i]) - 1), 1)))}
                for i in range(len(members))]}
            write_json(out / "modes.json", jsonable(modes_json))
            np.save(out / "mode_poles.npy", np.vstack(poles).astype(np.float32) if poles else np.zeros((0, E.shape[1]), np.float32))
            atlas = pd.DataFrame([{"cid": int(c), "mode_id": i, "cos_to_pole": float((E[c] / (np.linalg.norm(E[c]) + 1e-9)) @ poles[i])} for i in range(len(members)) for c in members[i]])
            atlas.to_parquet(out / "mode_atlas.parquet")
            smeta, sp_mat = supervised_poles(E, vocab, labels, mask, reliable_dims=reliable)
            write_json(out / "supervised_poles.json", jsonable({"poles": smeta})); np.save(out / "supervised_poles.npy", sp_mat)
            summary[m] = {"n_factors_stable": int(nF), "split_half": r.stability.tolist(), "n_modes": len(members), "coherence": coh, "audit": audit,
                          "driver_counts": pd.Series([x["driver"] for x in drivers]).value_counts().to_dict() if drivers else {}, "n_supervised_poles": len(smeta)}
        write_json(d6 / "S6_done.json", jsonable(summary))
        ctx.save_result("S6_modes", jsonable(summary))
        (ctx.reports_dir / "s6_modes.md").write_text("# S6 - ICA & modes\n\n| model | stable factors | modes | coherence | random | ratio |\n|---|---|---|---|---|---|\n" + "\n".join(
            f"| {k} | {v['n_factors_stable']} | {v['n_modes']} | {v['coherence']['coherence']} | {v['coherence']['random']} | {v['coherence']['ratio']} |" for k, v in summary.items()) + "\n", encoding="utf-8")
        if llm is None:
            res.status = PARTIAL; res.notes.append("no LLM: modes are unlabelled")
        res.metrics = {"models": len(summary), **{f"modes_{k}": v["n_modes"] for k, v in summary.items()}}
        return res
