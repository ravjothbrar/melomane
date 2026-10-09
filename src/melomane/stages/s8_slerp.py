"""S8 - SLERP & direction arithmetic: qualitative examples + quantitative direction test (H7) + multi-pole composition + SNR."""
from __future__ import annotations
import json
import numpy as np
import pandas as pd

from ..analysis.direction_test import increasing_share, monotonicity, multi_pole_composition, sample_seed_pole_pairs, theta_sweep_metrics
from ..analysis.snr import orthogonal_residual_snr, snr_ranking
from ..geometry import unit
from ..modes import pole_member_mask
from ..pipeline import Ctx, Stage, StageResult, register
from ..runtime import get_logger, jsonable, read_json
from ..slerp import nearest, slerp
from .common import available_models, labels_of, load_embeddings, vocab_of

log = get_logger("melomane.s8")
MODELS = ["cooc", "composition", "core", "composition_acoustic", "composition_lyric", "cooc_artistmasked"]


def examples(E: np.ndarray, vocab: pd.DataFrame, idx: np.ndarray, poles_meta: list[dict], P: np.ndarray, n_seeds: int = 8, thetas=(0, 30, 60), k: int = 5) -> list[dict]:
    """Qualitative SLERP examples: artist - title only (never lyrics/audio); artist-disjoint retrieval."""
    U = unit(E.astype(np.float32))
    pop = np.argsort(-vocab["n_sessions"].to_numpy()[idx])[:n_seeds]
    art = vocab["artist_id"].to_numpy()[idx]
    out = []
    for si in pop:
        for pj in np.linspace(0, max(len(poles_meta) - 1, 0), min(3, len(poles_meta))).astype(int) if len(poles_meta) else []:
            row = {"seed": f"{vocab.at[idx[si], 'artist']} - {vocab.at[idx[si], 'title']}", "pole": poles_meta[pj]["name"], "results": {}}
            for th in thetas:
                q = slerp(U[si], P[pj], th)
                ex = np.flatnonzero(art == art[si]).tolist()
                nn = nearest(U, q, k, exclude=ex)
                nn = nn[0] if isinstance(nn, tuple) else nn
                row["results"][str(th)] = [f"{vocab.at[idx[j], 'artist']} - {vocab.at[idx[j], 'title']}" for j in np.asarray(nn).ravel()[:k]]
            out.append(row)
    return out


@register
class Slerp(Stage):
    name, title, deps = "S8", "SLERP & direction arithmetic", ("S6",)

    def params(self, ctx):
        return {"v": 2, "slerp": ctx.profile["slerp"]}

    def outputs(self, ctx):
        return [ctx.results_dir / "S8_direction_test.json"]

    def run(self, ctx: Ctx) -> StageResult:
        cfg = ctx.profile["slerp"]
        vocab, labels = vocab_of(ctx), labels_of(ctx)
        mask = np.load(ctx.stage_dir("S3") / "eval_mask.npy")
        idx = np.flatnonzero(mask)
        art = vocab["artist_id"].to_numpy()[idx]
        res = StageResult(); out = {"models": {}}
        for m in [x for x in MODELS if x in available_models(ctx)]:
            d6 = ctx.stage_dir("S6") / m
            if not (d6 / "supervised_poles.json").exists():
                continue
            E = load_embeddings(ctx, m)[idx]
            meta = read_json(d6 / "supervised_poles.json")["poles"]
            P = np.load(d6 / "supervised_poles.npy")
            entry = {}
            if len(meta) >= 2:
                mem = []
                keep = []
                for i, pm in enumerate(meta):
                    mk = pole_member_mask(pm["name"], vocab, labels)
                    if mk is not None and mk[idx].sum() >= 10:
                        mem.append(mk[idx]); keep.append(i)
                if len(keep) >= 2:
                    M, Pk, mk_meta = np.stack(mem), P[keep], [meta[i] for i in keep]
                    pairs = sample_seed_pole_pairs(M, cfg["n_pairs"], seed=ctx.seed("S8", m))
                    df = theta_sweep_metrics(E, pairs["seeds"], Pk, M, thetas=cfg["thetas"], k=cfg["k"], exclude_groups=art, pole_index=pairs["pole_index"])
                    df["pole_name"] = [mk_meta[int(p)]["name"] for p in df["pole"]]
                    inc = increasing_share(df, up_to=60, by="pole", seed=ctx.seed("S8inc"))
                    mono = monotonicity(df, "precision", by="pole", up_to=60)
                    entry.update({"n_pairs": int(len(pairs["seeds"])), "n_poles": len(keep), "increasing_share_60": {k: v for k, v in inc.items() if k != "per_group"},
                                  "monotonicity": {k: v for k, v in mono.items() if k != "per_group"},
                                  "curve": df.groupby("theta")[["precision", "cos_pole", "cos_seed", "prevalence"]].mean().reset_index().to_dict("records")})
                    ctx.stage_dir("S8").mkdir(parents=True, exist_ok=True)
                    df.to_parquet(ctx.stage_dir("S8") / f"direction_test_{m}.parquet")
                    # multi-pole composition: two poles of different kinds
                    kinds = np.array([p["kind"] for p in mk_meta])
                    rng = np.random.default_rng(ctx.seed("S8combo", m))
                    combos = []
                    for _ in range(min(60, cfg["n_pairs"] // 4)):
                        a, b = rng.choice(len(mk_meta), 2, replace=False)
                        if kinds[a] != kinds[b] and (M[a] & M[b]).sum() >= 5:
                            combos.append((a, b))
                    if len(combos) >= 5:
                        seeds = rng.choice(len(idx), len(combos), replace=False)
                        mp = multi_pole_composition(E, seeds, Pk, np.array(combos), M, thetas=(30, 60), k=cfg["k"], exclude_groups=art)
                        entry["multi_pole"] = {"mean_gain_over_single_theta60": float(mp[mp["theta"] == 60]["gain_over_single"].mean()),
                                               "mean_joint_precision_theta60": float(mp[mp["theta"] == 60]["joint_precision"].mean()), "n_combos": len(combos)}
                    # orthogonal-residual SNR ranking (supplement)
                    s_seeds = rng.choice(len(idx), min(20, len(idx)), replace=False)
                    snr = orthogonal_residual_snr(E, s_seeds, Pk, theta=60.0, k=cfg["k"], exclude_groups=art)
                    rk = snr_ranking(snr, [p["name"] for p in mk_meta])
                    entry["snr_top"] = rk.head(10).to_dict("records"); entry["snr_median_all"] = float(np.nanmedian(snr["snr"]))
                    entry["examples"] = examples(E, vocab, idx, mk_meta, Pk)
            out["models"][m] = entry
        if out["models"]:
            shares = {m: e["increasing_share_60"]["share"] for m, e in out["models"].items() if "increasing_share_60" in e}
            out["H7"] = {"claim": "precision@10 of the pole tag increases with theta up to 60 deg for >= 80% of poles", "share_by_model": shares,
                         "status": ("supported" if any(v >= 0.8 for v in shares.values()) else "not supported") if shares else "not evaluable"}
        else:
            out["H7"] = {"status": "not evaluable", "claim": "no supervised poles available"}
        ctx.save_result("S8_direction_test", jsonable(out)); ctx.save_result("S8_H7", out["H7"])
        (ctx.reports_dir / "s8_slerp.md").write_text("# S8 - SLERP\n\nH7: " + json.dumps(out["H7"], default=str) + "\n", encoding="utf-8")
        res.metrics = {"H7": out["H7"]["status"], **{f"share60_{k}": v for k, v in out["H7"].get("share_by_model", {}).items()}}
        return res
