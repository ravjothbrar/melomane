"""S5 - walks & training: Melomane-Cooc / Composition / Core (x seeds) + ablations + raw-vocabulary Cooc (paper-1 averaging)."""
from __future__ import annotations
import json, shutil
import numpy as np
import pandas as pd
import scipy.sparse as sp

from ..cooc import npmi_edges, edges_to_csr
from ..geometry import participation_ratio, mean_pairwise_cosine, consolidate_embeddings
from ..pipeline import Ctx, Stage, StageResult, register, PARTIAL
from ..runtime import get_logger, read_json, write_json
from ..training import GraphInputs, ModelSpec, model_specs, train_one, neighbour_recall, templates_per_round, build_typed_graph
from .common import corpus_of, vocab_of, model_dir

log = get_logger("melomane.s5")


def load_graph_inputs(ctx: Ctx) -> GraphInputs:
    vocab = vocab_of(ctx)
    V = len(vocab)
    d3, d4 = ctx.stage_dir("S3"), ctx.stage_dir("S4")
    z = np.load(d3 / "cooc_edges.npz"); tt = edges_to_csr(z["rows"], z["cols"], z["npmi"], V)
    zm = np.load(d3 / "cooc_edges_artistmasked.npz"); ttm = edges_to_csr(zm["rows"], zm["cols"], zm["npmi"], V)
    gi = GraphInputs(V, tt, ttm)
    if (d4 / "compound_edges.npz").exists():
        import pickle
        gi.comp_edges = sp.load_npz(d4 / "compound_edges.npz").tocsr()
        meta = pd.read_parquet(d4 / "compound_meta.parquet")
        cm = pickle.load(open(d4 / "compound_model.pkl", "rb"))
        gi.family_names = list(cm["family_names"])
        gi.comp_family = np.array([gi.family_names.index(f) for f in meta["family"]], dtype=np.int64)
        gi.family_kind = dict(zip(meta["family"], meta["kind"]))
    return gi


def partial_pcts_of(prof: dict) -> list[int]:
    """Partial-hub percentages (E3) requested by the profile: `compounds.partial_hub_fracs` (fractions) x 100."""
    return sorted({int(round(100 * f)) for f in prof["compounds"].get("partial_hub_fracs", [0.3]) if 0 < f < 1}) or [30]


def sanity(E: np.ndarray, mask: np.ndarray | None = None) -> dict:
    Em = E[mask] if mask is not None else E
    n = min(len(Em), 20000)
    sub = Em[np.random.default_rng(0).choice(len(Em), n, replace=False)]
    return {"finite": bool(np.isfinite(E).all()), "participation_ratio": float(participation_ratio(sub)), "mean_pairwise_cosine": float(mean_pairwise_cosine(sub))}


def train_raw_cooc(ctx: Ctx, corp, cmap: pd.DataFrame, train_cfg: dict, cooc_cfg: dict, seed_key: str, device, out) -> tuple[np.ndarray, np.ndarray, dict]:
    ok = cmap["cid"].to_numpy() >= 0
    raw_ids = cmap["raw_id"].to_numpy()[ok]
    X = corp.sessions[:, raw_ids].tocsr(); X.data[:] = 1.0
    X = X[np.flatnonzero(np.diff(X.indptr) >= 2)]
    r, c, v = npmi_edges(X, cooc_cfg["min_count"], cooc_cfg["min_pair_count"], cooc_cfg["topk"], cooc_cfg["chunk"], True)
    gi = GraphInputs(len(raw_ids), edges_to_csr(r, c, v, len(raw_ids)))
    E, info = train_one(model_specs(train_cfg["ii_repeat"])["cooc"], gi, train_cfg, 0, out, seed_key, device=device)
    return E, raw_ids, info


@register
class Training(Stage):
    name, title, deps = "S5", "Walks & SGNS training (models, seeds, ablations)", ("S3", "S4")

    def params(self, ctx):
        return {"v": 3, "train": ctx.profile["train"], "seeds": ctx.profile["seeds"], "models": ctx.profile["models"], "abl": ctx.profile["ablations"], "pcts": partial_pcts_of(ctx.profile),
                "only": ctx.opt("only_models")}

    def outputs(self, ctx):
        return [ctx.stage_dir("S5") / "S5_done.json"]

    def run(self, ctx: Ctx) -> StageResult:
        prof = ctx.profile
        tc = dict(prof["train"])
        vocab = vocab_of(ctx); V = len(vocab)
        gi = load_graph_inputs(ctx)
        has_comp = gi.comp_edges is not None
        specs = model_specs(tc["ii_repeat"], partial_pcts_of(prof))
        mask = np.load(ctx.stage_dir("S3") / "eval_mask.npy")
        res = StageResult()
        device = ctx.opt("device")
        n_seeds = prof["seeds"]
        key = f"{ctx.corpus}/{ctx.profile_name}"
        ref_templates = None
        if has_comp:
            g_core, _ = build_typed_graph(gi, specs["core"])
            ref_templates = templates_per_round(g_core, specs["core"])
        wanted_main = [m for m in prof["models"] if has_comp or m == "cooc"]
        wanted_abl = [a for a in prof["ablations"] if a in specs and (has_comp or specs[a].compound == "none") and a != "cooc_avg"]
        if ctx.opt("only_models"):
            wanted_main = [m for m in wanted_main if m in ctx.opt("only_models")]; wanted_abl = [a for a in wanted_abl if a in ctx.opt("only_models")]
        if not has_comp:
            res.status = PARTIAL; res.notes.append("no compound graph: Composition/Core and compound ablations skipped (Cooc-family only)")
        summary: dict = {"models": {}, "seed_recall": {}}
        for name in wanted_main + wanted_abl:
            spec = specs[name]
            main = name in wanted_main
            seeds = range(n_seeds if main else 1)
            out = model_dir(ctx, name)
            Es = []
            for s in seeds:
                log.info("training %s seed %d", name, s)
                try:
                    E, info = train_one(spec, gi, tc, s, out, key, ref_templates=ref_templates, device=device)
                except ValueError as e:
                    res.notes.append(f"{name} skipped: {e}"); break
                Es.append(E)
            if not Es:
                continue
            np.save(out / "embeddings.npy", Es[0])
            sm = sanity(Es[0], mask)
            entry = {**sm, "seeds": len(Es), "seconds": info["seconds"], "device": info["device"], "epochs": info["epochs"],
                     "tokens_per_epoch": info["tokens_per_epoch"], "loss_first": (info["loss"] or [None])[0], "loss_last": (info["loss"] or [None])[-1],
                     "loss_decreasing": bool(len(info["loss"]) < 2 or info["loss"][-1] <= info["loss"][0] * 1.02)}
            if len(Es) >= 2:
                entry["neighbour_recall_at10_seed0_vs_1"] = neighbour_recall(Es[0], Es[1], 10, 1500, mask=mask)
                summary["seed_recall"][name] = entry["neighbour_recall_at10_seed0_vs_1"]
            summary["models"][name] = entry
            write_json(out / "manifest.json", {"name": name, "spec": spec.__dict__, "profile": ctx.profile_name, "train": tc, "sanity": entry})
            if not (entry["finite"] and entry["participation_ratio"] > 5 and entry["mean_pairwise_cosine"] < 0.99 and entry["loss_decreasing"]):
                res.status = PARTIAL; res.notes.append(f"{name}: sanity flags {entry}")
        # ---- paper-1 stage 6: raw-vocabulary Cooc averaged onto canonical tracks ----
        if "cooc_avg" in prof["ablations"] and not ctx.opt("only_models"):
            corp = corpus_of(ctx, load_families=False)
            cmap = pd.read_parquet(ctx.stage_dir("S1") / "consolidation_map.parquet")
            Eraw, raw_ids, info = train_raw_cooc(ctx, corp, cmap, tc, prof["cooc"], key + "/raw", device, model_dir(ctx, "cooc_raw"))
            np.save(model_dir(ctx, "cooc_raw") / "embeddings.npy", Eraw); np.save(model_dir(ctx, "cooc_raw") / "raw_ids.npy", raw_ids)
            ok = cmap["cid"].to_numpy() >= 0
            cids = cmap["cid"].to_numpy()[ok]
            Eavg, uid = consolidate_embeddings(Eraw, cids)
            full = np.zeros((V, Eavg.shape[1]), np.float32); full[uid] = Eavg
            (model_dir(ctx, "cooc_avg")).mkdir(parents=True, exist_ok=True)
            np.save(model_dir(ctx, "cooc_avg") / "embeddings.npy", full)
            summary["models"]["cooc_avg"] = {**sanity(full, mask), "raw_vocab": int(len(raw_ids))}
        write_json(ctx.stage_dir("S5") / "S5_done.json", summary)
        ctx.save_result("S5_training", summary)
        lines = ["# S5 - training", "", "| model | seeds | PR | mean cos | seconds | device | loss first->last |", "|---|---|---|---|---|---|---|"]
        for k, e in summary["models"].items():
            lines.append(f"| {k} | {e.get('seeds', 1)} | {e['participation_ratio']:.1f} | {e['mean_pairwise_cosine']:.3f} | {e.get('seconds', 0):.0f} | {e.get('device', '')} | {e.get('loss_first')} -> {e.get('loss_last')} |")
        (ctx.reports_dir / "s5_training.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        res.metrics = {"models_trained": len(summary["models"]), **{f"recall_{k}": v for k, v in summary["seed_recall"].items()}}
        return res
