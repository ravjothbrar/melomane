"""S4 - compound graph: k-means clusters per feature family (the sonic / lyrical 'compounds')."""
from __future__ import annotations
import pickle
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics import normalized_mutual_info_score

from ..compounds import fit_compound_graph
from ..pipeline import Ctx, Stage, StageResult, register, PARTIAL
from ..runtime import get_logger, write_json
from .common import corpus_of, vocab_of, labels_of

log = get_logger("melomane.s4")


def family_kind(name: str) -> str:
    return "lyric" if name.startswith("lyric") else "acoustic"


def blocks_to_cid(corp, vocab: pd.DataFrame, cmap: pd.DataFrame, blocks: dict | None = None) -> dict[str, np.ndarray]:
    """Dense (V x d) float32 per family in cid space (NaN rows where the track has no data).  A canonical track takes its
    representative raw entry's features, falling back to the most-played merged member that has them."""
    V = len(vocab)
    raw2cid = cmap["cid"].to_numpy()
    rep = vocab["rep_raw"].to_numpy()
    out = {}
    for name, b in (corp.families if blocks is None else blocks).items():
        cid = raw2cid[b.rows]
        ok = cid >= 0
        rows, cids, X = b.rows[ok], cid[ok], b.X[ok]
        is_rep = (rows == rep[cids]).astype(np.int8)
        order = np.lexsort((-is_rep, cids))                # rep first within each cid
        cids_s, first = np.unique(cids[order], return_index=True)
        D = np.full((V, X.shape[1]), np.nan, dtype=np.float32)
        D[cids_s] = X[order][first]
        out[name] = D
    return out


@register
class Compounds(Stage):
    name, title, deps = "S4", "Compound graph (sonic / lyrical families)", ("S1", "S0-enrich")

    def params(self, ctx):
        return {"v": 2, "compounds": ctx.profile["compounds"], "seed": ctx.seed("S4")}

    def outputs(self, ctx):
        return [ctx.stage_dir("S4") / "features_manifest.json"]

    def run(self, ctx: Ctx) -> StageResult:
        cfg = ctx.profile["compounds"]
        corp, vocab = corpus_of(ctx), vocab_of(ctx)
        d = ctx.stage_dir("S4")
        res = StageResult()
        if not corp.families:
            write_json(d / "features_manifest.json", {"n_families": 0, "note": "no feature blocks: Composition/Core unavailable"})
            res.status = PARTIAL; res.notes.append("corpus has no compound feature blocks: only Cooc-family models will be trained")
            return res
        from ..adapters.onion import check_family_leakage
        leaks = check_family_leakage(corp.families)
        if leaks:
            raise RuntimeError(f"held-out / learned-embedding features leaked into compound families: {leaks}")
        cmap = pd.read_parquet(ctx.stage_dir("S1") / "consolidation_map.parquet")
        feats = blocks_to_cid(corp, vocab, cmap)
        names = list(feats)
        k = {n: (cfg["k_lyric"] if family_kind(n) == "lyric" else cfg["k_acoustic"]) for n in names}
        cg = fit_compound_graph(feats, k=k, top_m=cfg["top_m"], seed=ctx.seed("S4"), pca_dim=cfg["pca_dim"])
        E = sp.csr_matrix(cg.edges)
        sp.save_npz(d / "compound_edges.npz", E)
        size = np.asarray((E > 0).sum(axis=0)).ravel()
        meta = pd.DataFrame({"comp_id": np.arange(cg.n_comp), "family": [names[f] for f in cg.comp_family], "kind": [family_kind(names[f]) for f in cg.comp_family], "size": size})
        meta.to_parquet(d / "compound_meta.parquet")
        with open(d / "compound_model.pkl", "wb") as f:               # scalers/PCA/centroids: reused by Track C projection
            pickle.dump({"family_names": cg.family_names, "centroids": cg.centroids, "scalers": cg.scalers, "k": k, "top_m": cfg["top_m"]}, f)
        cover = {n: float(np.isfinite(feats[n]).all(axis=1).mean()) for n in names}
        # diagnostics: cluster-label MI (never tuned on)
        lab = labels_of(ctx)
        mi = {}
        for fi, n in enumerate(names):
            cols = np.flatnonzero(cg.comp_family == fi)
            sub = E[:, cols]
            has = np.diff(sub.tocsr().indptr) > 0
            if has.sum() < 50:
                continue
            top = np.asarray(sub[has].argmax(axis=1)).ravel()
            m = {"category": float(normalized_mutual_info_score(vocab.loc[has, "category"].astype(str), top))}
            if lab is not None and lab["labelled"].any() and lab["era"].notna().sum() > 50:
                mm = has & lab["era"].notna().to_numpy()
                m["era"] = float(normalized_mutual_info_score(lab.loc[mm, "era"].astype(str), np.asarray(sub[mm].argmax(axis=1)).ravel()))
            mi[n] = m
        write_json(d / "features_manifest.json", {"n_families": len(names), "families": {n: {"k": int(k[n]), "kind": family_kind(n), "columns": int(corp.families[n].X.shape[1]),
                                                                                             "source": corp.families[n].source, "coverage": cover[n]} for n in names},
                                                    "held_out_verified_absent": True, "top_m": cfg["top_m"], "pca_dim": cfg["pca_dim"]})
        big = float(size.max() / len(vocab))
        stats = {"n_compounds": int(cg.n_comp), "coverage": cover, "max_cluster_share": big, "cluster_size_median": float(np.median(size)), "empty_clusters": int((size == 0).sum()),
                 "cluster_label_mi": mi, "lyric_coverage": cover.get(next((n for n in names if family_kind(n) == "lyric"), ""), None),
                 "tracks_with_any_compound": float((np.diff(E.indptr) > 0).mean())}
        ctx.save_result("S4_compounds", stats)
        (ctx.reports_dir / "s4_compounds.md").write_text("# S4 - compounds\n\n" + "\n".join(f"- {k}: {v}" for k, v in stats.items()) + "\n", encoding="utf-8")
        if stats["empty_clusters"] or big > 0.05:
            res.status = PARTIAL; res.notes.append(f"cluster sanity: empty={stats['empty_clusters']}, max share={big:.3f} (limit 0.05)")
        if stats["tracks_with_any_compound"] < 0.999:
            res.notes.append(f"partial-hub corpus: {stats['tracks_with_any_compound']:.1%} of canonical tracks have compound edges")
        res.metrics = {"n_compounds": stats["n_compounds"], "max_cluster_share": big, "tracks_with_compounds": stats["tracks_with_any_compound"]}
        return res
