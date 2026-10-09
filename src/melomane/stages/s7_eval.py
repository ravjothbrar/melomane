"""S7 - evaluation battery (PLAN §7): probe strata, geometry, cultural structure, kNN purity, paired comparisons, WEAT, Procrustes."""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..analysis.battery import geometry_summary, knn_battery, probe_battery, stratum_means
from ..analysis.procrustes import procrustes_matrix
from ..analysis.weat import weat_region_energy
from ..evaluation import paired_strata
from ..pipeline import Ctx, Stage, StageResult, register, PARTIAL
from ..runtime import get_logger, jsonable, write_json
from .common import available_models, corpus_of, labels_of, model_dir, stratum_targets, vocab_of

log = get_logger("melomane.s7")
MAIN = ["cooc", "composition", "core"]
MIN_REGIONS_H3 = 3                                     # H3 compares regions: fewer than 3 usable tags is not a comparison
PRIMARY = {"ordinal": "centroid", "binary": "centroid", "continuous": "tercile_centroid"}


def eval_rows(ctx: Ctx, vocab: pd.DataFrame, mask: np.ndarray, headline_test_only: bool = True) -> np.ndarray:
    keep = mask.copy()
    if headline_test_only and (vocab["dev_split"] == "test").sum() > 200:
        keep &= (vocab["dev_split"] == "test").to_numpy()
    idx = np.flatnonzero(keep)
    cap = ctx.profile["eval"]["max_eval_tracks"]
    if len(idx) > cap:
        idx = np.sort(np.random.default_rng(ctx.seed("S7rows")).choice(idx, cap, replace=False))
    return idx


def region_targets(vocab: pd.DataFrame, idx: np.ndarray, min_n: int = 60) -> dict:
    reg = vocab["region_tags"].astype(str).to_numpy()[idx]
    names = sorted({x for s in reg for x in s.split("|") if x})
    out = {}
    for r in names:
        m = np.array([r in s.split("|") for s in reg])
        if m.sum() >= min_n and (~m).sum() >= min_n:
            out[f"cultural:{r}"] = {"kind": "binary", "y": m, "stratum": "cultural"}
    return out


def seeds_of(ctx: Ctx, name: str) -> list[int]:
    d = model_dir(ctx, name) / "seeds"
    return sorted(int(p.stem[4:]) for p in d.glob("seed*.npy")) if d.exists() else []


def load_seed(ctx: Ctx, name: str, s: int) -> np.ndarray:
    return np.load(model_dir(ctx, name) / "seeds" / f"seed{s}.npy")


@dataclass
class EvalSet:
    """Everything the evaluation stages share: eval rows (test artists, >=1 cooc edge), artist groups and the probe registry restricted to them."""
    vocab: pd.DataFrame
    labels: pd.DataFrame | None
    corp: object
    idx: np.ndarray
    groups: np.ndarray
    T: dict
    strata: dict

    def targets(self, *strata: str, kinds=("ordinal", "continuous")) -> dict:
        return {k: v for k, v in self.T.items() if v["stratum"] in strata and v["kind"] in kinds}

    def primary(self) -> dict:
        """Held-out (b) + external (c, popularity excluded) targets = the H1 family; falls back to whatever exists."""
        t = self.targets("b_heldout", "c_external")
        return t or self.targets("a_baked", "b_heldout", "c_external", "cultural")


def build_eval(ctx: Ctx) -> EvalSet:
    vocab, labels, corp = vocab_of(ctx), labels_of(ctx), corpus_of(ctx, load_families=False)
    mask = np.load(ctx.stage_dir("S3") / "eval_mask.npy")
    idx = eval_rows(ctx, vocab, mask)
    groups = vocab["artist_id"].to_numpy()[idx]
    T_all = stratum_targets(ctx, vocab, labels, corp)
    excl = set(ctx.load_result("S2_reliability").get("excluded_dims", [])) if ctx.has_result("S2_reliability") else set()
    T_all = {k: v for k, v in T_all.items() if not (k.startswith("c_llm:") and k.split(":")[1] in excl)}
    T = {k: {**v, "y": np.asarray(v["y"])[idx]} for k, v in T_all.items()}
    T.update(region_targets(vocab, idx))
    strata: dict = {}
    for k, v in T.items():
        strata.setdefault(v["stratum"], []).append(k)
    return EvalSet(vocab, labels, corp, idx, groups, T, strata)


@register
class Evaluation(Stage):
    name, title, deps = "S7", "Evaluation battery", ("S5", "S6")

    def params(self, ctx):
        return {"v": 3, "eval": ctx.profile["eval"], "cv": ctx.profile["cv"]}

    def outputs(self, ctx):
        return [ctx.results_dir / "S7_summary.json"]

    def run(self, ctx: Ctx) -> StageResult:
        prof = ctx.profile; ev, cv = prof["eval"], prof["cv"]
        es = build_eval(ctx)
        vocab, labels, idx, groups, T, strata = es.vocab, es.labels, es.idx, es.groups, es.T, es.strata
        res = StageResult()
        models = available_models(ctx)
        models = [m for m in models if m != "cooc_raw"]
        out: dict = {"n_eval_tracks": int(len(idx)), "n_targets": len(T), "strata_sizes": {k: len(v) for k, v in strata.items()}, "models": {}}
        cats = vocab["category"].astype(str).to_numpy()[idx]
        first_region = np.array([(s.split("|")[0] if s else "") for s in vocab["region_tags"].astype(str).to_numpy()[idx]])
        decade = np.array([f"{int(y) // 10 * 10}s" if y == y else "" for y in vocab["year"].to_numpy(float)[idx]])
        label_sets = {"category": {"labels": cats, "tagged": (cats != "Other") & (cats != "")}}
        if (first_region != "").sum() > 300 and len(set(first_region[first_region != ""])) > 1:
            label_sets["region"] = {"labels": first_region, "tagged": first_region != ""}
        if (decade != "").sum() > 300 and len(set(decade[decade != ""])) > 1:
            label_sets["era"] = {"labels": decade, "tagged": decade != ""}
        label_sets = {k: v for k, v in label_sets.items() if v["tagged"].sum() >= 100}
        tables = []
        for m in models:
            seeds = seeds_of(ctx, m) if m in MAIN else seeds_of(ctx, m)[:1]
            if not seeds:
                seeds = [0]
            per_seed = []
            for s in seeds:
                E = (load_seed(ctx, m, s) if seeds_of(ctx, m) else np.load(model_dir(ctx, m) / "embeddings.npy"))[idx]
                bat = probe_battery(E, T, groups, cv["n_splits"], cv["n_repeats"], seed=ctx.seed("S7", m, s) % 10_000, ci_boot=ev["knn_boot"], ridge=(m in MAIN and s == 0))
                bat["model"], bat["seed"] = m, s
                tables.append(bat)
                prim = bat[bat["reading"] == bat["kind"].map(PRIMARY)]
                sm = stratum_means(prim, {t: v["stratum"] for t, v in T.items()}, n_boot=500, seed=0)
                geo = geometry_summary(E)
                knn = knn_battery(E, label_sets, groups, k=ev["knn_k"], n_boot=ev["knn_boot"], seed=ctx.seed("S7knn", m, s) % 10_000) if label_sets else pd.DataFrame()
                per_seed.append({"seed": s, "strata": sm.to_dict("records"), "geometry": geo, "knn": knn.to_dict("records")})
            out["models"][m] = {"seeds": [p["seed"] for p in per_seed], "per_seed": per_seed}
        full = pd.concat(tables, ignore_index=True)
        ctx.stage_dir("S7").mkdir(parents=True, exist_ok=True)
        full.to_parquet(ctx.stage_dir("S7") / "probe_table.parquet")
        out["summary_by_model"] = self.aggregate(out["models"])
        # ---- paired model comparisons (seed 0), artist-clustered bootstrap ----
        paired = {}
        pair_strata = {"a_baked": strata.get("a_baked", []), "b_heldout": strata.get("b_heldout", []),
                       "c_external": [t for t in strata.get("c_external", [])]}
        pair_strata = {k: v for k, v in pair_strata.items() if v}
        Ts = {k: v for k, v in T.items() if v["kind"] != "binary"}
        for a, b in (("composition", "cooc"), ("core", "cooc"), ("composition", "core")):
            if a in models and b in models and pair_strata:
                Ea, Eb = np.load(model_dir(ctx, a) / "embeddings.npy")[idx], np.load(model_dir(ctx, b) / "embeddings.npy")[idx]
                paired[f"{a}_minus_{b}"] = paired_strata(Ea, Eb, Ts, groups, pair_strata, cv["n_splits"], cv["n_repeats"], n_boot=min(ev["boot_clusters"], 1000), seed=ctx.seed("S7pair"))
        out["paired"] = paired
        # ---- cultural: per-region Cohen's d per model (seed 0) ----
        reg_rows = full[(full["stratum"] == "cultural") & (full["seed"] == 0)][["model", "target", "d"]] if (full["stratum"] == "cultural").any() else pd.DataFrame()
        out["cultural_d"] = reg_rows.pivot(index="target", columns="model", values="d").to_dict() if len(reg_rows) else {}
        # ---- supplement: WEAT (region x energy) and Procrustes across models ----
        try:
            if labels is not None and labels["labelled"].any() and (first_region != "").sum() > 300:
                energy = labels["energy"].to_numpy(float)[idx]
                weat = {}
                for m in [x for x in MAIN if x in models]:
                    E = np.load(model_dir(ctx, m) / "embeddings.npy")[idx]
                    weat[m] = weat_region_energy(E, np.where(first_region != "", first_region, None).astype(object), energy, groups=groups, n_perm=ev["weat_perm"], seed=0).to_dict("records")
                out["weat"] = weat
        except Exception as e:  # noqa: BLE001
            res.notes.append(f"WEAT skipped: {type(e).__name__}: {e}")
        if len(models) >= 2:
            sub = np.random.default_rng(0).choice(len(idx), min(len(idx), 15000), replace=False)
            pm = procrustes_matrix({m: np.load(model_dir(ctx, m) / "embeddings.npy")[idx][sub] for m in models if m in MAIN + ["cooc_artistmasked", "composition_acoustic", "composition_lyric"]}, n_null=0)
            out["procrustes"] = {"names": pm["names"], "disparity": pm["disparity"].tolist(), "axis_cosine": pm["axis_cosine"].tolist()}
        out["hypotheses"] = self.hypotheses(out, ctx)
        ctx.save_result("S7_summary", jsonable(out))
        ctx.save_result("S7_hypotheses", out["hypotheses"])
        (ctx.reports_dir / "s7_evaluation.md").write_text(self.report(out), encoding="utf-8")
        res.metrics = {"models": len(models), "targets": len(T), **{f"H{k}": v["status"] for k, v in out["hypotheses"].items()}}
        return res

    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def aggregate(models: dict) -> dict:
        agg = {}
        for m, d in models.items():
            rec = {}
            for ps in d["per_seed"]:
                for r in ps["strata"]:
                    rec.setdefault(f"{r['stratum']}|{r['metric']}", []).append(r["mean"])
                rec.setdefault("PR", []).append(ps["geometry"]["participation_ratio"])
                rec.setdefault("mean_cos", []).append(ps["geometry"]["mean_pairwise_cosine"])
                for k in ps["knn"]:
                    rec.setdefault(f"knn_lift|{k['label_set']}", []).append(k["lift"])
            agg[m] = {k: {"mean": float(np.nanmean(v)), "sd": float(np.nanstd(v)), "n_seeds": len(v)} for k, v in rec.items()}
        return agg

    @staticmethod
    def hypotheses(out: dict, ctx: Ctx) -> dict:
        H = {}
        P, S = out.get("paired", {}), out["summary_by_model"]
        def ev(pair, stratum):
            return (P.get(pair, {}).get("strata", {}) or {}).get(stratum)
        # H1: Composition >= Cooc on (b) and (c-excl-popularity)
        ev_b, ev_c = ev("composition_minus_cooc", "b_heldout"), ev("composition_minus_cooc", "c_external")
        if ev_b or ev_c:
            ok = [e for e in (ev_b, ev_c) if e]
            H["1"] = {"claim": "Composition >= Cooc on held-out (b) and external (c) strata", "status": "supported" if all(e["ci_low"] > 0 for e in ok) else ("not supported" if any(e["ci_high"] < 0 for e in ok) else "inconclusive"),
                      "evidence": {"b": ev_b, "c": ev_c, "a": ev("composition_minus_cooc", "a_baked")}}
        else:
            H["1"] = {"claim": "Composition >= Cooc on (b),(c)", "status": "not evaluable", "evidence": "needs composition + cooc + measured/LLM targets"}
        # H2: Core lowest PR, highest mean cos
        if all(m in S for m in MAIN):
            pr = {m: S[m]["PR"]["mean"] for m in MAIN}; mc = {m: S[m]["mean_cos"]["mean"] for m in MAIN}
            H["2"] = {"claim": "Core has the lowest participation ratio and highest mean pairwise cosine", "status": "supported" if min(pr, key=pr.get) == "core" and max(mc, key=mc.get) == "core" else "not supported", "evidence": {"PR": pr, "mean_cos": mc}}
        # H3: regional Cohen's d Composition >= Cooc
        cd = out.get("cultural_d") or {}
        if "composition" in cd and "cooc" in cd:
            regs = [r for r in cd["composition"] if r in cd["cooc"] and cd["composition"][r] == cd["composition"][r] and cd["cooc"][r] == cd["cooc"][r]]
            share = float(np.mean([cd["composition"][r] >= cd["cooc"][r] for r in regs])) if regs else float("nan")
            if len(regs) < MIN_REGIONS_H3:      # one region is an anecdote, not a regional comparison
                H["3"] = {"claim": "Composition >= Cooc on regional Cohen's d", "status": "not evaluable",
                          "evidence": {"reason": f"only {len(regs)} region(s) with a usable tag (need >= {MIN_REGIONS_H3})", "n_regions": len(regs)}}
            else:
                H["3"] = {"claim": "Composition >= Cooc on regional Cohen's d", "status": "supported" if share >= 0.75 else ("not supported" if share <= 0.25 else "mixed"), "evidence": {"share_regions_composition_ge_cooc": share, "n_regions": len(regs)}}
        else:
            H["3"] = {"claim": "Composition >= Cooc on regional d", "status": "not evaluable", "evidence": "no cultural targets"}
        return H

    @staticmethod
    def report(out: dict) -> str:
        L = ["# S7 - evaluation", "", f"Eval tracks: {out['n_eval_tracks']:,} (test artists); targets: {out['n_targets']}; strata: {out['strata_sizes']}", "", "## Hypotheses (directional replication targets)", ""]
        for k, h in out["hypotheses"].items():
            L.append(f"- **H{k}** {h['claim']}: **{h['status']}**")
        L += ["", "## Stratum means (rho / d) by model (mean across seeds)", ""]
        keys = sorted({k for m in out["summary_by_model"].values() for k in m if "|" in k and not k.startswith("knn")})
        L.append("| model | " + " | ".join(keys) + " |"); L.append("|---|" + "---|" * len(keys))
        for m, d in out["summary_by_model"].items():
            L.append(f"| {m} | " + " | ".join(f"{d[k]['mean']:.3f}" if k in d else "-" for k in keys) + " |")
        return "\n".join(L) + "\n"
