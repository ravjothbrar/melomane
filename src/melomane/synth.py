"""Synthetic corpus with PLANTED structure, used to test that the pipeline recovers what we put in.

Latents per track: region (cultural macro-region, imbalanced), energy (ordinal 0-4), era (ordinal 0-4).
Behaviour (users x tracks): taste strongly follows region, weakly energy/era.        -> Cooc should see region.
Acoustic families: driven mostly by energy (+era), weakly by region.                 -> Composition should see energy.
Lyric family: driven by region-linked themes; NaN for instrumentals.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import scipy.sparse as sp


@dataclass
class Synth:
    region: np.ndarray
    energy: np.ndarray
    era: np.ndarray
    genre_group: np.ndarray
    region_tagged: np.ndarray
    has_lyrics: np.ndarray
    users_x_tracks: sp.csr_matrix
    features: dict


def make_synth(n_tracks=2000, n_users=4000, n_regions=6, seed=0, listens_per_user=50,
               region_beta=2.2, energy_beta=0.5, era_beta=0.3, acoustic_region_w=0.35) -> Synth:
    rng = np.random.default_rng(seed)
    p = np.array([0.35, 0.22, 0.15, 0.12, 0.09, 0.07][:n_regions]); p /= p.sum()
    region = rng.choice(n_regions, n_tracks, p=p)
    energy = rng.integers(0, 5, n_tracks); era = rng.integers(0, 5, n_tracks)
    pop = rng.lognormal(0, 0.7, n_tracks)
    # ---- behaviour ----
    u_region = rng.choice(n_regions, n_users, p=p); u_e = rng.integers(0, 5, n_users); u_d = rng.integers(0, 5, n_users)
    rows, cols = [], []
    for u in range(n_users):
        logit = (region_beta * (region == u_region[u]) - energy_beta * np.abs(energy - u_e[u])
                 - era_beta * np.abs(era - u_d[u]))
        w = np.exp(logit) * pop; w /= w.sum()
        m = min(max(10, rng.poisson(listens_per_user)), n_tracks // 2)
        t = rng.choice(n_tracks, m, replace=False, p=w)
        rows.append(np.full(m, u)); cols.append(t)
    X = sp.csr_matrix((np.ones(sum(map(len, rows))), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(n_users, n_tracks))
    # ---- acoustic families ----
    z_e = (energy - 2) / 1.4; z_d = (era - 2) / 1.4
    onehot = np.eye(n_regions)[region]
    def fam(dim, we, wd, wr):
        A = rng.normal(size=(dim, 1)); B = rng.normal(size=(dim, 1)); C = rng.normal(size=(n_regions, dim))
        return (we * z_e[:, None] @ A.T + wd * z_d[:, None] @ B.T + wr * onehot @ C + rng.normal(size=(n_tracks, dim)))
    feats = {"timbre": fam(8, 1.6, 0.5, acoustic_region_w), "rhythm": fam(6, 1.3, 0.2, acoustic_region_w),
             "tonal": fam(6, 0.2, 0.8, acoustic_region_w * 1.3), "dynamics": fam(4, 1.5, 0.3, acoustic_region_w * 0.5)}
    # ---- lyrics (region-linked themes; instrumentals = NaN) ----
    has_lyrics = rng.random(n_tracks) > 0.15
    centers = rng.normal(size=(n_regions, 10)) * 1.8
    lyr = centers[region] + 0.5 * z_e[:, None] * rng.normal(size=(1, 10)) + rng.normal(size=(n_tracks, 10))
    lyr[~has_lyrics] = np.nan
    feats["lyric_theme"] = lyr
    genre_group = np.where(rng.random(n_tracks) < 0.85, region // 2, rng.integers(0, max(n_regions // 2, 1), n_tracks))
    region_tagged = rng.random(n_tracks) < 0.7
    return Synth(region, energy, era, genre_group, region_tagged, has_lyrics, X, feats)
