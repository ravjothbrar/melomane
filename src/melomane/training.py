"""Model specifications, graph assembly and (resumable) SGNS training for Melomane-Cooc / Composition / Core and ablations."""
from __future__ import annotations
import json, os, time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence
import numpy as np
import scipy.sparse as sp

from .runtime import get_logger, write_json, derive_seed
from .walks import TypedGraph, generate_walks

log = get_logger("melomane.training")


@dataclass(frozen=True)
class ModelSpec:
    name: str
    tt: int                       # pure track-track templates per round ('ii_repeat' scaled: 1 = Cooc, 0 = Composition, ii_repeat = Core)
    compound: str = "none"        # none | all | acoustic | lyric
    tt_graph: str = "cooc"        # cooc | artistmasked
    keep_frac: float = 1.0        # partial-hub: fraction of tracks that keep compound edges
    equal_token_to: str | None = None   # name of the reference spec whose token budget to match


def model_specs(ii_repeat: int, partial_pct: "int | Sequence[int]" = 30) -> dict[str, ModelSpec]:
    s = {
        "cooc": ModelSpec("cooc", 1, "none"),
        "composition": ModelSpec("composition", 0, "all"),
        "core": ModelSpec("core", ii_repeat, "all"),
        "composition_acoustic": ModelSpec("composition_acoustic", 0, "acoustic"),
        "composition_lyric": ModelSpec("composition_lyric", 0, "lyric"),
        "core_acoustic": ModelSpec("core_acoustic", ii_repeat, "acoustic"),
        "cooc_artistmasked": ModelSpec("cooc_artistmasked", 1, "none", "artistmasked"),
        "cooc_eqtok": ModelSpec("cooc_eqtok", 1, "none", equal_token_to="core"),
        "composition_eqtok": ModelSpec("composition_eqtok", 0, "all", equal_token_to="core"),
        "core_eqtok": ModelSpec("core_eqtok", ii_repeat, "all", equal_token_to="core"),
    }
    for p in ([partial_pct] if isinstance(partial_pct, (int, float)) else partial_pct):
        p = int(round(p))
        s[f"core_partial{p}"] = ModelSpec(f"core_partial{p}", ii_repeat, "all", keep_frac=p / 100)
    return s


def needs_compounds(spec: ModelSpec) -> bool:
    return spec.compound != "none"


@dataclass
class GraphInputs:
    n_tracks: int
    tt: sp.csr_matrix
    tt_masked: sp.csr_matrix | None = None
    comp_edges: sp.csr_matrix | None = None
    comp_family: np.ndarray | None = None        # family index per compound column
    family_names: list[str] = field(default_factory=list)
    family_kind: dict[str, str] = field(default_factory=dict)


def build_typed_graph(gi: GraphInputs, spec: ModelSpec, seed: int = 0) -> tuple[TypedGraph, int]:
    """Returns (TypedGraph, n_compound_nodes)."""
    tt = gi.tt_masked if (spec.tt_graph == "artistmasked" and gi.tt_masked is not None) else gi.tt
    E, fam = None, None
    if needs_compounds(spec):
        if gi.comp_edges is None:
            raise ValueError(f"model {spec.name} needs compound edges")
        keep_cols = np.arange(gi.comp_edges.shape[1])
        if spec.compound in ("acoustic", "lyric"):
            names = np.array(gi.family_names)[gi.comp_family]
            keep_cols = np.flatnonzero([gi.family_kind[n] == spec.compound for n in names])
            if keep_cols.size == 0:
                raise ValueError(f"model {spec.name}: no {spec.compound} families in this corpus")
        E = gi.comp_edges[:, keep_cols].tocsr()
        fam = gi.comp_family[keep_cols]
        # re-index family ids to 0..F-1 (TypedGraph iterates over the distinct ids present)
        if spec.keep_frac < 1.0:                       # partial-hub regime E3: only a fraction of tracks keep compound edges
            rng = np.random.default_rng(seed)
            has = np.diff(E.indptr) > 0
            drop = has & (rng.random(E.shape[0]) > spec.keep_frac)
            D = sp.diags((~drop).astype(np.float32))
            E = (D @ E).tocsr(); E.eliminate_zeros()
    g = TypedGraph(tt, E, fam)
    return g, (E.shape[1] if E is not None else 0)


def partial_hub_mask(gi: GraphInputs, spec: ModelSpec, seed: int) -> np.ndarray:
    """(n_tracks,) bool: tracks that KEEP their compound edges under `spec` (same RNG stream as build_typed_graph, so the mask is
    exactly the one the trained model saw).  Tracks that never had compound edges are False."""
    if gi.comp_edges is None:
        return np.zeros(gi.n_tracks, bool)
    has = np.diff(gi.comp_edges.tocsr().indptr) > 0
    if spec.keep_frac >= 1.0:
        return has
    rng = np.random.default_rng(seed)
    E = gi.comp_edges.tocsr()
    keep_cols = np.arange(E.shape[1])
    if spec.compound in ("acoustic", "lyric"):
        names = np.array(gi.family_names)[gi.comp_family]
        keep_cols = np.flatnonzero([gi.family_kind[n] == spec.compound for n in names])
        has = np.diff(E[:, keep_cols].tocsr().indptr) > 0
    drop = has & (rng.random(E.shape[0]) > spec.keep_frac)
    return has & ~drop


def templates_per_round(g: TypedGraph, spec: ModelSpec) -> int:
    n = spec.tt
    if needs_compounds(spec):
        n += len(g.families) + (1 if len(g.families) > 1 else 0)
    return n


def rounds_for(spec: ModelSpec, g: TypedGraph, base_rounds: int, ref_templates: int | None) -> int:
    if spec.equal_token_to and ref_templates:
        return max(1, int(round(base_rounds * ref_templates / max(templates_per_round(g, spec), 1))))
    return base_rounds


def neighbour_recall(A: np.ndarray, B: np.ndarray, k: int = 10, n: int = 2000, seed: int = 0, mask: np.ndarray | None = None) -> float:
    """Mean overlap of top-k cosine neighbours between two embeddings of the same vocabulary (seed-stability sanity check)."""
    rng = np.random.default_rng(seed)
    idx_pool = np.flatnonzero(mask) if mask is not None else np.arange(len(A))
    q = rng.choice(idx_pool, min(n, len(idx_pool)), replace=False)
    def top(E):
        U = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
        Pool = U[idx_pool]
        S = U[q] @ Pool.T
        S[np.arange(len(q)), np.searchsorted(idx_pool, q)] = -np.inf
        return idx_pool[np.argpartition(-S, k, axis=1)[:, :k]]
    ta, tb = top(A), top(B)
    return float(np.mean([len(set(a) & set(b)) / k for a, b in zip(ta, tb)]))


def _ckpt_path(out: Path, seed: int) -> Path:
    return out / "ckpt" / f"seed{seed}.pt"


def train_one(spec: ModelSpec, gi: GraphInputs, train_cfg: dict, seed: int, out_dir: Path, base_seed_key: str,
              ref_templates: int | None = None, device: str | None = None, backend: str = "auto", log_fn=None) -> tuple[np.ndarray, dict]:
    """Train (or resume) one model+seed.  Saves `seeds/seed<k>.npy` and a per-epoch checkpoint; returns (E[:n_tracks], info)."""
    log_fn = log_fn or (lambda m: log.info(m))
    out_dir = Path(out_dir); (out_dir / "seeds").mkdir(parents=True, exist_ok=True)
    done = out_dir / "seeds" / f"seed{seed}.npy"
    info_p = out_dir / "seeds" / f"seed{seed}.json"
    if done.exists() and info_p.exists():
        return np.load(done), json.loads(info_p.read_text())
    g, n_comp = build_typed_graph(gi, spec, seed=derive_seed(base_seed_key, spec.name, "partial"))
    rounds = rounds_for(spec, g, train_cfg["walks_per_node"], ref_templates)
    L = train_cfg["walk_length"]
    tpr = templates_per_round(g, spec)
    n_nodes = gi.n_tracks + n_comp
    wseed = derive_seed(base_seed_key, spec.name, seed)

    def make_walks(ep):
        return generate_walks(g, rounds=rounds, walk_length=L, tt_per_round=spec.tt, use_compound=needs_compounds(spec), seed=wseed + 1000 * ep)

    t0 = time.time(); history: list[float] = []
    use_torch = backend in ("auto", "torch")
    E = None
    if use_torch:
        try:
            import torch
            from .torch_sgns import train_torch_sgns, pick_device
            dev = pick_device(device)
            if dev.type == "cpu" and backend == "auto" and os.environ.get("MELOMANE_ALLOW_CPU_TORCH") != "1" and gi.n_tracks > 20000:
                use_torch = False               # CPU torch is slower than gensim at scale: use the gensim fallback
                log_fn("no GPU: using the gensim CPU backend")
            else:
                ck = _ckpt_path(out_dir, seed); ck.parent.mkdir(exist_ok=True)
                resume = None
                if ck.exists():
                    try:
                        st = torch.load(ck, map_location="cpu", weights_only=False)
                        resume = {"state": st["state"], "opt": st["opt"], "epoch": st["epoch"]}; history = st.get("history", [])
                        log_fn(f"resuming {spec.name} seed {seed} at epoch {st['epoch']}")
                    except Exception:
                        resume = None

                def on_epoch(ep, model, opt, loss):
                    history.append(float(loss))
                    torch.save({"state": model.state_dict(), "opt": opt.state_dict(), "epoch": ep, "history": history}, ck.with_suffix(".tmp"))
                    os.replace(ck.with_suffix(".tmp"), ck)
                    with open(out_dir / "train_log.jsonl", "a", encoding="utf-8") as f:
                        f.write(json.dumps({"seed": seed, "epoch": ep, "loss": float(loss), "t": time.time() - t0}) + "\n")

                W = train_torch_sgns(make_walks, n_nodes, dim=train_cfg["dim"], window=train_cfg["window"], negative=train_cfg["negative"],
                                     epochs=train_cfg["epochs"], lr=train_cfg["lr"], walk_block=train_cfg.get("walk_block", 4096),
                                     pair_batch=train_cfg.get("pair_batch", 1 << 18), device=str(dev), seed=wseed % (2 ** 31), log=log_fn,
                                     on_epoch=on_epoch, resume=resume)
                E = W[: gi.n_tracks]; device_used = str(dev)
        except ImportError:
            use_torch = False
    if E is None:
        from .embed import train_skipgram
        _ep = iter(range(10 ** 6))
        E = train_skipgram(lambda: make_walks(next(_ep)), n_nodes, dim=train_cfg["dim"], window=train_cfg["window"], negative=train_cfg["negative"],
                           epochs=train_cfg["epochs"], workers=max(1, (os.cpu_count() or 2) - 1), seed=wseed % (2 ** 31))[: gi.n_tracks]
        device_used = "cpu-gensim"
    E = np.asarray(E, dtype=np.float32)
    info = {"model": spec.name, "seed": seed, "rounds": rounds, "templates_per_round": tpr, "n_compound_nodes": n_comp, "walk_length": L,
            "tokens_per_epoch": int(gi.n_tracks * rounds * tpr * L), "epochs": train_cfg["epochs"], "seconds": time.time() - t0,
            "device": device_used, "loss": history, "spec": spec.__dict__}
    np.save(done, E); info_p.write_text(json.dumps(info), encoding="utf-8")
    ck = _ckpt_path(out_dir, seed)
    if ck.exists():
        ck.unlink()
    return E, info
