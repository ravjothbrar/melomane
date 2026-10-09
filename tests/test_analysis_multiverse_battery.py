import numpy as np
import pandas as pd
import pytest

from melomane import probes
from melomane.analysis import (run_grid, summarise, cell_seed, probe_battery, stratum_means, geometry_summary, knn_battery)
from melomane.geometry import unit, participation_ratio, mean_pairwise_cosine


# ------------------------------------------------------------------------------------------------- E8 multiverse
def _fn(k, top_m, gap, seed):
    return {"score": 0.5 + 0.001 * k / 250 + 0.01 * top_m - 0.0002 * gap + 1e-6 * (seed % 7), "n_edges": k * top_m}


def test_run_grid_returns_every_cell_and_summarises_whole_grid():
    grid = {"k": [250, 500, 1000], "top_m": [1, 3, 5], "gap": [10, 30, 60]}
    df = run_grid(_fn, grid, seed=0)
    assert len(df) == 27 and list(df.columns[:5]) == ["cell_id", "k", "top_m", "gap", "seed"]
    assert (df.status == "ok").all() and df.cell_id.tolist() == list(range(27))
    assert set(map(tuple, df[["k", "top_m", "gap"]].to_numpy())) == {(a, b, c) for a in grid["k"] for b in grid["top_m"] for c in grid["gap"]}
    assert df.seed.tolist() == [cell_seed(0, i) for i in range(27)] and len(set(df.seed)) == 27
    pd.testing.assert_frame_equal(df, run_grid(_fn, grid, seed=0))                       # deterministic
    assert not run_grid(_fn, grid, seed=1).seed.equals(df.seed)
    s = summarise(df, "score", criterion=0.53)
    assert s["n_cells"] == 27 and s["n_valid"] == 27 and s["min"] == df.score.min() and s["max"] == df.score.max()
    assert s["min"] <= s["q25"] <= s["median"] <= s["q75"] <= s["max"]
    assert abs(s["share_meeting"] - (df.score >= 0.53).mean()) < 1e-12 and s["n_meeting"] == int((df.score >= 0.53).sum())
    assert "27 cells" in s["statement"] and "median" in s["statement"] and "meet" in s["statement"]
    assert set(s["by_param"]) == {"k", "top_m", "gap"} and set(s["by_param"]["top_m"]) == {1, 3, 5}
    assert s["by_param"]["top_m"][5] > s["by_param"]["top_m"][1]
    lo = summarise(df, "score", criterion=lambda v: v < 0.52, higher_is_better=False)
    assert abs(lo["share_meeting"] - (df.score < 0.52).mean()) < 1e-12
    assert summarise(df, "n_edges", criterion=1000, higher_is_better=False)["n_meeting"] == int((df.n_edges <= 1000).sum())


def test_run_grid_records_failures_and_scalar_returns():
    def flaky(a, b, seed):
        if a == 2 and b == "x":
            raise RuntimeError("boom")
        return a * 1.0 if b == "x" else 0.5
    df = run_grid(flaky, {"a": [1, 2, 3], "b": ["x", "y"]}, seed=3)
    assert len(df) == 6 and (df.status == "error").sum() == 1 and "RuntimeError: boom" in df.error.iloc[2 * 1 - 2 + 2]
    assert df.value.isna().sum() == 1
    s = summarise(df, "value", criterion=0.5)
    assert s["n_failed"] == 1 and abs(s["share_meeting"] - 5 / 6) < 1e-12                # failed cells count as NOT meeting
    with pytest.raises(RuntimeError):
        run_grid(flaky, {"a": [2], "b": ["x"]}, raise_errors=True)
    assert len(run_grid(lambda a: a, {"a": [1, 2]}, pass_seed=False)) == 2


# --------------------------------------------------------------------------------------------- S7 probe battery
def _world(n_art=150, per=8, D=32, seed=0):
    rng = np.random.default_rng(seed)
    n = n_art * per; art = np.repeat(np.arange(n_art), per)
    ax = unit(rng.normal(size=(3, D)))
    y_ord = rng.integers(0, 5, n)                                   # planted along ax[0]
    cont = rng.normal(size=n)                                       # planted along ax[1]
    lab = rng.integers(0, 3, n); C = rng.normal(size=(3, D)) * 1.5   # 3 classes (binary target: class 0)
    E = rng.normal(size=(n, D)) + 0.8 * np.outer(y_ord - 2, ax[0]) + 1.0 * np.outer(cont, ax[1]) + C[lab]
    return E, y_ord, cont, lab, art


@pytest.fixture(scope="module")
def world():
    return _world()


def test_probe_battery_recovers_planted_structure_and_shuffle_null(world):
    E, y, cont, lab, art = world
    rng = np.random.default_rng(1)
    targets = {"ord": dict(kind="ordinal", y=y, stratum="external"), "ord_shuf": dict(kind="ordinal", y=rng.permutation(y), stratum="null"),
               "cont": dict(kind="continuous", y=cont, stratum="measured"), "cont_shuf": dict(kind="continuous", y=rng.permutation(cont), stratum="null"),
               "cls0": dict(kind="binary", y=lab == 0, stratum="cultural"), "cls_shuf": dict(kind="binary", y=rng.permutation(lab == 0), stratum="null"),
               "masked": dict(kind="ordinal", y=y, mask=np.arange(len(y)) % 2 == 0, stratum="external")}
    df = probe_battery(E, targets, art, n_splits=5, n_repeats=2, seed=0, ci_boot=100)
    assert {"target", "stratum", "kind", "reading", "metric", "value", "rho", "d", "ci_low", "ci_high", "n", "leaky"} <= set(df.columns)
    assert not df.leaky.any()
    g = lambda t, r: df[(df.target == t) & (df.reading == r)].iloc[0]
    assert g("ord", "centroid").rho > 0.5 and g("ord", "ridge").rho > 0.5
    assert g("cont", "tercile_centroid").rho > 0.4 and g("cont", "ridge").rho > 0.5
    assert g("cls0", "centroid").d > 1.0 and np.isnan(g("cls0", "centroid").rho) and g("cls0", "centroid").metric == "d"
    for t in ("ord_shuf", "cont_shuf"):
        assert abs(g(t, "centroid" if t == "ord_shuf" else "tercile_centroid").rho) < 0.12 and abs(g(t, "ridge").rho) < 0.12
    assert abs(g("cls_shuf", "centroid").d) < 0.3
    for _, r in df.iterrows():                                     # CIs are ordered and pooled-CI brackets sensible values
        assert r.ci_low <= r.ci_high
    r0 = g("ord", "centroid"); assert r0.ci_low > 0.3 and r0.ci_high <= 1
    assert g("ord_shuf", "centroid").ci_high < 0.1 and g("ord_shuf", "centroid").ci_low > -0.25   # CV-null sits at ~0 (slightly negative: leave-fold-out bias)
    assert g("masked", "centroid").n == len(y) // 2 and g("ord", "centroid").n == len(y)
    # the value is exactly what probes.py reports for the same seed / protocol
    assert abs(r0.value - probes.ordinal_projection_cv(E, y, 5, 2, "centroid", 0, None, art)["rho"]) < 1e-12
    assert abs(g("cls0", "centroid").value - probes.binary_separation_cv(E, lab == 0, 5, 2, 0, art)["d"]) < 1e-12
    pd.testing.assert_frame_equal(df, probe_battery(E, targets, art, n_splits=5, n_repeats=2, seed=0, ci_boot=100))


def test_probe_battery_artist_leakage_control():
    rng = np.random.default_rng(0); n_art, per, D = 120, 8, 40
    art = np.repeat(np.arange(n_art), per)
    y = np.repeat(rng.integers(0, 5, n_art), per)                                  # artist-level label
    E = np.repeat(rng.normal(size=(n_art, D)), per, axis=0) * 2.0 + 0.3 * rng.normal(size=(n_art * per, D))
    tg = {"artist_label": dict(kind="ordinal", y=y)}
    leaky = probe_battery(E, tg, None, 5, 3, ci_boot=0)
    honest = probe_battery(E, tg, art, 5, 3, ci_boot=0)
    assert leaky.leaky.all() and not honest.leaky.any()
    lk = leaky[leaky.reading == "centroid"].value.iloc[0]; hn = honest[honest.reading == "centroid"].value.iloc[0]
    assert lk > hn + 0.15 and abs(hn) < 0.35
    assert honest.ci_low.isna().all()                                                # ci_boot=0 => no CI


def test_stratum_means(world):
    E, y, cont, lab, art = world
    targets = {f"o{i}": dict(kind="ordinal", y=y, stratum="baked_in") for i in range(3)}
    targets.update({f"c{i}": dict(kind="continuous", y=cont, stratum="held_out") for i in range(2)})
    targets["b"] = dict(kind="binary", y=lab == 0, stratum="cultural")
    df = probe_battery(E, targets, art, 5, 1, ci_boot=0, ridge=False)
    sm = stratum_means(df, n_boot=200)
    assert set(sm.stratum) == {"baked_in", "held_out", "cultural"}
    r = sm[(sm.stratum == "baked_in")].iloc[0]
    assert r.n_targets == 3 and abs(r["mean"] - df[df.kind == "ordinal"].value.mean()) < 1e-12 and r.metric == "rho"
    assert sm[sm.stratum == "cultural"].iloc[0].metric == "d"                          # rho and d are never mixed
    sm2 = stratum_means(df, {"baked_in": ["o0", "o1"], "held_out": ["c0", "c1"]}, n_boot=100)
    assert set(sm2.stratum) == {"baked_in", "held_out"} and sm2[sm2.stratum == "baked_in"].iloc[0].n_targets == 2
    sm3 = stratum_means(df, {"o0": "x", "o1": "x", "c0": "y"}, n_boot=100)
    assert set(sm3.stratum) == {"x", "y"}


def test_geometry_summary_exact(world):
    E = world[0]
    g = geometry_summary(E)
    assert g["n"] == len(E) and g["dim"] == E.shape[1]
    assert abs(g["participation_ratio"] - participation_ratio(E)) < 1e-9
    U = unit(E); S = U @ U.T; brute = (S.sum() - np.trace(S)) / (len(E) * (len(E) - 1))
    assert abs(g["mean_pairwise_cosine"] - brute) < 1e-10 and abs(g["mean_pairwise_cosine"] - mean_pairwise_cosine(E)) < 1e-12
    assert 1 <= g["participation_ratio"] <= E.shape[1]


def test_knn_battery_lift_size_matched_and_null():
    rng = np.random.default_rng(0); n_art, per, D = 200, 5, 24
    art = np.repeat(np.arange(n_art), per); n = n_art * per
    cl = rng.integers(0, 6, n); C = rng.normal(size=(6, D)) * 1.5
    E = C[cl] + rng.normal(size=(n, D))
    junk = rng.integers(0, 6, n)
    tagged = rng.random(n) < 0.6
    ls = {"planted": cl, "junk": junk, "partial": {"labels": cl, "tagged": tagged}}
    df = knn_battery(E, ls, art, k=10, n_boot=8, seed=0, n_null=4)
    g = lambda s: df[df.label_set == s].iloc[0]
    assert g("planted").lift > 2.0 and g("planted").ci_low > 1.5
    assert abs(g("junk").lift - 1.0) < 0.2 and abs(g("planted").null_lift - 1.0) < 0.25
    assert g("planted").m == g("junk").m == g("partial").m == tagged.sum()               # size-matched across label sets
    assert g("partial").n_tagged == tagged.sum() and g("planted").n_classes == 6
    assert g("planted").baseline > 1 / 6 - 0.03 and g("planted").purity > 2 * g("planted").baseline
    with pytest.raises(ValueError):
        knn_battery(E, ls, art, subsample=n + 1, n_boot=2)
    # artist-disjointness matters: artist twins with artist-level labels only look pure when retrieval is leaky
    Ea = np.repeat(rng.normal(size=(n_art, D)), per, axis=0) + 0.05 * rng.normal(size=(n, D))
    la = np.repeat(rng.integers(0, 4, n_art), per)
    honest = knn_battery(Ea, {"a": la}, art, k=5, n_boot=5, seed=1).lift.iloc[0]
    leaky = knn_battery(Ea, {"a": la}, None, k=5, n_boot=5, seed=1).lift.iloc[0]
    assert leaky > honest + 0.4 and honest < 1.4
