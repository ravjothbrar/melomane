"""S3 - behaviour graph (Melomane-Cooc input): sessions in cid space -> positive-NPMI edges (+ artist-masked variant)."""
from __future__ import annotations
import numpy as np
import scipy.sparse as sp

from ..cooc import npmi_edges, edges_to_csr
from ..pipeline import Ctx, Stage, StageResult, register, PARTIAL
from ..runtime import get_logger
from .common import corpus_of, vocab_of
from .s1_curation import merge_matrix

log = get_logger("melomane.s3")


@register
class Behaviour(Stage):
    name, title, deps = "S3", "Behaviour graph (session NPMI)", ("S1",)

    def params(self, ctx):
        return {"v": 2, "cooc": ctx.profile["cooc"]}

    def outputs(self, ctx):
        d = ctx.stage_dir("S3")
        return [d / "sessions_cid.npz", d / "cooc_edges.npz", d / "cooc_edges_artistmasked.npz"]

    def run(self, ctx: Ctx) -> StageResult:
        cfg = ctx.profile["cooc"]
        corp, vocab = corpus_of(ctx, load_families=False), vocab_of(ctx)
        import pandas as pd
        cmap = pd.read_parquet(ctx.stage_dir("S1") / "consolidation_map.parquet")
        V = len(vocab)
        M = merge_matrix(cmap, corp.n_tracks, V)
        S = (corp.sessions @ M).tocsr(); S.data[:] = 1.0
        lens = np.diff(S.indptr)
        # sessions were already length-filtered on raw ids by the adapter; after vocabulary selection/consolidation keep any session
        # that can still contribute a pair
        keep = lens >= 2
        S = S[np.flatnonzero(keep)].tocsr()
        sp.save_npz(ctx.stage_dir("S3") / "sessions_cid.npz", S)
        n_docs = S.shape[0]
        res = StageResult()
        tried = []
        mpc = cfg["min_pair_count"]
        for attempt in [mpc, 3, 2]:
            if attempt > mpc and tried:
                continue
            r, c, v = npmi_edges(S, cfg["min_count"], attempt, cfg["topk"], cfg["chunk"], True)
            deg = np.bincount(r, minlength=V)
            tried.append((attempt, float(deg.mean())))
            if deg.mean() >= 40 or attempt == 2:
                break
            mpc = attempt - 1 if attempt > 2 else 2
        if tried[0][0] != tried[-1][0]:
            res.decisions.append(f"min_pair_count lowered {tried[0][0]} -> {tried[-1][0]} to reach mean degree >= 40 (tried {tried})")
        A = edges_to_csr(r, c, v, V)
        assert (A != A.T).nnz == 0 or np.allclose((A - A.T).data, 0), "NPMI graph must be symmetric"
        artist = vocab["artist_id"].to_numpy()
        same = artist[r] == artist[c]
        keepm = ~same
        d = ctx.stage_dir("S3")
        np.savez_compressed(d / "cooc_edges.npz", rows=r.astype(np.int32), cols=c.astype(np.int32), npmi=v.astype(np.float32), n=V)
        np.savez_compressed(d / "cooc_edges_artistmasked.npz", rows=r[keepm].astype(np.int32), cols=c[keepm].astype(np.int32),
                            npmi=v[keepm].astype(np.float32), n=V)
        deg = np.bincount(r, minlength=V)
        mdeg = np.bincount(r[keepm], minlength=V)
        eval_mask = deg > 0
        np.save(d / "eval_mask.npy", eval_mask)
        stats = {"n_sessions": int(n_docs), "n_tracks": int(V), "session_len_mean": float(lens[keep].mean()), "session_len_median": float(np.median(lens[keep])),
                 "session_len_p95": float(np.percentile(lens[keep], 95)), "n_edges_undirected": int(len(r) // 2), "mean_degree": float(deg.mean()),
                 "median_degree": float(np.median(deg)), "n_isolated": int((deg == 0).sum()), "same_artist_edge_share": float(same.mean()),
                 "masked_mean_degree": float(mdeg.mean()), "positive_weights": bool((v > 0).all()), "min_pair_count_used": int(tried[-1][0]),
                 "attempts": tried, "weight_mean": float(v.mean())}
        ctx.save_result("S3_behaviour", stats)
        if not (100 <= stats["mean_degree"] <= 250):
            res.decisions.append(f"mean degree {stats['mean_degree']:.0f} outside the 100-250 target of the plan (corpus/sampling dependent); recorded, not tuned")
        if stats["n_isolated"] > 0.02 * V:
            res.status = PARTIAL
            res.notes.append(f"{stats['n_isolated']} tracks have no co-occurrence edges; evaluations are restricted to tracks with >=1 edge")
        (ctx.reports_dir / "s3_behaviour.md").write_text("# S3 - behaviour graph\n\n" + "\n".join(f"- {k}: {v}" for k, v in stats.items()) + "\n", encoding="utf-8")
        res.metrics = {k: stats[k] for k in ("n_sessions", "mean_degree", "same_artist_edge_share", "n_isolated")}
        return res
