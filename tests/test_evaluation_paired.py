import numpy as np
from melomane.evaluation import paired_strata
from melomane.synth import make_synth


def test_paired_strata_detects_better_model_and_null():
    rng = np.random.default_rng(0)
    n, d = 1500, 24
    y = rng.integers(0, 5, n).astype(float); groups = rng.integers(0, 300, n)
    signal = np.zeros((n, d)); signal[:, 0] = y
    good = signal * 1.0 + rng.normal(size=(n, d))
    bad = rng.normal(size=(n, d))
    targets = {"t1": {"kind": "ordinal", "y": y}, "t2": {"kind": "continuous", "y": y + rng.normal(0, 0.3, n)}}
    r = paired_strata(good, bad, targets, groups, {"S": ["t1", "t2"]}, n_splits=4, n_repeats=2, n_boot=300, seed=0)
    s = r["strata"]["S"]
    assert s["delta_rho"] > 0.2 and s["ci_low"] > 0 and s["share_targets_positive"] == 1.0
    # identical models -> delta exactly 0, CI contains 0
    z = paired_strata(good, good, targets, groups, {"S": ["t1", "t2"]}, n_splits=4, n_repeats=2, n_boot=200, seed=0)["strata"]["S"]
    assert abs(z["delta_rho"]) < 1e-12 and z["ci_low"] <= 0 <= z["ci_high"]
    # per-item decomposition reproduces the pooled spearman
    ra = r["targets"]["t1"]; assert ra["rho_a"] > 0.3 > abs(ra["rho_b"]) + 0.2           # per-item mean = pooled Spearman scale
