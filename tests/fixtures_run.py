"""Fabricates a small but *real-shaped* run directory (``docs/CONTRACTS.md`` layout) for the products-layer tests.

Planted structure comes from ``melomane.synth.make_synth`` (region / energy / era latents); ICA factors, modes and poles are
computed with the real ``melomane.ica_modes``.  Three models mimic the paper's ordering: Cooc sees region + artist, Composition
sees energy/era, Core sees both.

Usage in a test module::

    from fixtures_run import fake_run, fake_bundles      # noqa: F401  (session-scoped pytest fixtures)
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from melomane.geometry import unit
from melomane.ica_modes import factor_modes, residualise, stable_ica
from melomane.synth import make_synth

CATEGORIES = ["rock", "pop", "hip-hop", "electronic", "folk", "jazz"]
REGIONS = ["anglo", "latin", "nordic", "afro", "asian", "euro"]
_ADJ = ["Velvet", "Neon", "Paper", "Golden", "Hollow", "Silent", "Crimson", "Electric", "Lunar", "Broken", "Wild", "Amber",
        "Gentle", "Iron", "Scarlet", "Distant", "Midnight", "Rusty", "Shallow", "Bright"]
_NOUN = ["Foxes", "Harbor", "Engines", "Gardens", "Rivers", "Signals", "Wolves", "Machines", "Orchard", "Echoes", "Tigers",
         "Circus", "Canyon", "Lanterns", "Sparrows", "Anthem", "Comets", "Mirage", "Cathedral", "Boulevard"]
_W1 = ["Dancing", "Falling", "Burning", "Sleeping", "Running", "Drifting", "Waiting", "Breaking", "Singing", "Chasing"]
_W2 = ["Moonlight", "Highway", "Summer", "Shadows", "Rain", "Fire", "Kingdom", "Letters", "Ocean", "Mirror", "Thunder", "Dust"]


@dataclass
class FakeRunInfo:
    """Handle returned by :func:`make_fake_run` (``run_dir`` is what tests pass around)."""
    run_dir: Path
    V: int
    D: int
    models: tuple[str, ...]


def _names(rng: np.random.Generator, n_artists: int, V: int, artist_of: np.ndarray):
    arts, seen = [], set()
    for a in range(n_artists):
        while True:
            nm = f"{_ADJ[rng.integers(len(_ADJ))]} {_NOUN[rng.integers(len(_NOUN))]}"
            if nm not in seen:
                seen.add(nm); break
        arts.append(("The " + nm) if a % 7 == 0 else nm)
    titles, seen_t = [], set()
    for i in range(V):
        while True:
            t = f"{_W1[rng.integers(len(_W1))]} {_W2[rng.integers(len(_W2))]}"
            if (artist_of[i], t) not in seen_t:
                seen_t.add((artist_of[i], t)); break
        titles.append(t + (" - Remastered 2011" if i % 20 == 3 else ""))
    return arts, titles


def make_fake_run(root: Path, V: int = 400, D: int = 32, models: tuple[str, ...] = ("cooc", "composition", "core"),
                  seed: int = 0, n_extra_raw: int = 60, n_factors: int = 6) -> Path:
    """Write a fake run under ``root/fakecorpus/smoke`` and return that directory."""
    rng = np.random.default_rng(seed)
    run = Path(root) / "fakecorpus" / "smoke"
    S = make_synth(n_tracks=V, n_users=150, n_regions=6, seed=seed, listens_per_user=20)

    # ---- artists: 4 tracks each, within one region (so artist structure exists and is testable) ----------------------
    artist_of = np.empty(V, dtype=np.int64); nxt = 0
    for r in range(6):
        ids = rng.permutation(np.flatnonzero(S.region == r))
        for c in range(0, len(ids), 4):
            artist_of[ids[c:c + 4]] = nxt; nxt += 1
    n_art = nxt
    art_names, titles = _names(rng, n_art, V, artist_of)
    for a in range(0, n_art, 12):                     # plant near-duplicates: 'X' and 'X (Live)' by the same artist
        ids = np.flatnonzero(artist_of == a)
        if len(ids) >= 2:
            titles[ids[1]] = titles[ids[0]].split(" - ")[0] + " (Live)"
    cat_idx = np.where(rng.random(V) < 0.9, S.region, rng.integers(0, 6, V))
    category = [CATEGORIES[c] for c in cat_idx]
    year = (1965 + 10 * S.era + rng.integers(0, 10, V)).astype(float)
    year[rng.random(V) < 0.05] = np.nan
    n_sessions = rng.poisson(60, V) + 5
    test_art = set(rng.choice(n_art, n_art // 5, replace=False).tolist())
    region_tags = ["|".join([REGIONS[r]] + ([REGIONS[(r + 1) % 6]] if rng.random() < 0.2 else [])) if S.region_tagged[i] else ""
                   for i, r in enumerate(S.region)]

    # raw ids: reps are a random subset of 0..R-1; extras fold into cids ("merge") or are removed (cid=-1)
    R = V + n_extra_raw
    perm = rng.permutation(R)
    rep_raw = perm[:V]; extras = perm[V:]
    vocab = pd.DataFrame({
        "cid": np.arange(V), "rep_raw": rep_raw, "artist_id": artist_of, "artist": [art_names[a] for a in artist_of],
        "title": titles, "album": [f"Album {a % 13}" for a in artist_of], "year": year, "category": category,
        "n_sessions": n_sessions, "popularity": rng.integers(0, 100, V).astype(float),
        "dev_split": ["test" if a in test_art else "dev" for a in artist_of], "region_tags": region_tags,
        "tags": ["|".join(rng.choice(["chill", "party", "sad", "epic"], 2, replace=False)) for _ in range(V)],
    })
    (run / "S1").mkdir(parents=True, exist_ok=True)
    vocab.to_parquet(run / "S1" / "vocab.parquet")

    n_dup = int(n_extra_raw * 0.75)
    raw_cid = np.full(R, -1, dtype=np.int64); decision = np.array(["none"] * R, dtype=object); group = np.full(R, -1, dtype=np.int64)
    raw_cid[rep_raw] = np.arange(V); group[rep_raw] = np.arange(V)
    for j, rid in enumerate(extras):
        if j < n_dup:
            c = int(rng.integers(0, V)); raw_cid[rid] = c; group[rid] = c
            decision[rid] = ["rule", "llm", "llm", "human"][j % 4]
        else:
            group[rid] = V + j; decision[rid] = ["rule", "llm"][j % 2]
    pd.DataFrame({"raw_id": np.arange(R), "cid": raw_cid, "decision": decision, "group_id": group}) \
        .to_parquet(run / "S1" / "consolidation_map.parquet")

    # ---- embeddings ------------------------------------------------------------------------------------------------
    zE = (S.energy - 2) / 1.4; zD = (S.era - 2) / 1.4
    Creg = rng.normal(size=(6, D)); ax_e = unit(rng.normal(size=D)); ax_d = unit(rng.normal(size=D)); common = unit(rng.normal(size=D))
    Aoff = rng.normal(size=(n_art, D)); Egen = {}
    spec = {"cooc": (2.2, 0.5, 0.3, 0.0), "composition": (0.7, 1.8, 1.0, 0.0), "core": (1.8, 1.4, 0.8, 1.0)}
    for name in models:
        wr, we, wd, wc = spec.get(name, (1.5, 1.0, 0.6, 0.5))
        E = (wr * Creg[S.region] + we * zE[:, None] * ax_e * 2 + wd * zD[:, None] * ax_d * 2 + 0.8 * Aoff[artist_of]
             + wc * 1.5 * common + rng.normal(size=(V, D)))
        Egen[name] = (E * 0.05).astype(np.float32)

    cats_arr = np.array(category); regions_arr = np.array(region_tags, dtype=object)
    for name in models:
        E = Egen[name]
        mdir = run / "S5" / "models" / name
        (mdir / "seeds").mkdir(parents=True, exist_ok=True)
        np.save(mdir / "embeddings.npy", E)
        np.save(mdir / "seeds" / "seed0.npy", E)
        np.save(mdir / "seeds" / "seed1.npy", (E + rng.normal(scale=0.01, size=E.shape)).astype(np.float32))
        (mdir / "manifest.json").write_text(json.dumps({"model": name, "seed": 0, "seeds": [0, 1], "dim": D, "n_items": V,
                                                        "profile": "smoke"}), encoding="utf-8")
        (mdir / "train_log.jsonl").write_text(json.dumps({"epoch": 0, "loss": 1.0}) + "\n", encoding="utf-8")

        # ---- S6: ICA on group-residualised embedding, modes, poles ----------------------------------------------
        s6 = run / "S6" / name; s6.mkdir(parents=True, exist_ok=True)
        ica = stable_ica(residualise(E, cat_idx), n_components=n_factors, n_seeds=3, threshold=0.0, n_splits=2, seed=seed)
        np.savez(s6 / "ica.npz", directions=ica.directions, sources=ica.sources, stability=ica.stability)
        np.save(s6 / "factor_poles.npy", unit(ica.directions).astype(np.float32))
        Uu = unit(E); modes, poles = [], []
        for f in range(ica.sources.shape[1]):
            for m in factor_modes(E, ica.sources[:, f], f, k_range=range(2, 4), min_members=5, pca_dim=6, seed=seed):
                mem = np.sort(m.members)
                top_cat = pd.Series(cats_arr[mem]).mode().iloc[0]
                decade = int(np.nanmedian(year[mem]) // 10 * 10) if np.isfinite(year[mem]).any() else 0
                modes.append({"mode_id": len(modes), "factor": int(f), "bic_k": int(m.bic_k), "size": int(len(mem)),
                              "members": [int(x) for x in mem], "label": f"{top_cat} {decade}s",
                              "property": "category", "driver": name,
                              "coherence": float((Uu[mem] @ m.pole).mean())})
                poles.append(m.pole)
        assert modes, "fixture failed to produce any ICA modes"
        (s6 / "modes.json").write_text(json.dumps({"model": name, "modes": modes}), encoding="utf-8")
        np.save(s6 / "mode_poles.npy", np.asarray(poles, dtype=np.float32))

        sup, sup_vecs = [], []
        def add(kind, label, pos):
            neg = ~pos
            if pos.sum() >= 5 and neg.sum() >= 5:
                sup.append({"name": f"{kind}:{label}", "kind": kind, "n_pos": int(pos.sum()), "n_neg": int(neg.sum())})
                sup_vecs.append(unit(Uu[pos].mean(0) - Uu[neg].mean(0)))
        for c in CATEGORIES:
            add("category", c, cats_arr == c)
        for r in REGIONS:
            add("region", r, np.array([r in str(t).split("|") for t in regions_arr]))
        for dec in sorted({int(y // 10 * 10) for y in year[np.isfinite(year)]}):
            add("era", f"{dec}s", np.isfinite(year) & (year // 10 * 10 == dec))
        add("dimension", "energy_high_vs_low", S.energy >= 3)
        (s6 / "supervised_poles.json").write_text(json.dumps({"poles": sup}), encoding="utf-8")
        np.save(s6 / "supervised_poles.npy", np.asarray(sup_vecs, dtype=np.float32))
        pd.DataFrame([{"mode_id": m["mode_id"], "factor": m["factor"], "label": m["label"], "size": m["size"],
                       "coherence": m["coherence"]} for m in modes]).to_parquet(s6 / "mode_atlas.parquet")
    return run


# ---- pytest fixtures ----------------------------------------------------------------------------------------------------
@pytest.fixture(scope="session")
def fake_run(tmp_path_factory) -> Path:
    """Session-cached fake run directory (V=400, D=32, models cooc/composition/core)."""
    return make_fake_run(tmp_path_factory.mktemp("fakerun"))


@pytest.fixture(scope="session")
def fake_bundles(fake_run, tmp_path_factory) -> dict[str, Path]:
    """Session-cached ``export_all`` output (full + ``-small`` bundles; small = 24-d int8 because D is only 32)."""
    from melomane.release.export import export_all
    return export_all(fake_run, tmp_path_factory.mktemp("bundles"), small_dims=24, retention_queries=100)
