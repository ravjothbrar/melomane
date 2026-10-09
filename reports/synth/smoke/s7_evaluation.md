# S7 - evaluation

Eval tracks: 472 (test artists); targets: 23; strata: {'a_baked': 2, 'b_heldout': 1, 'c_external': 17, 'c_external_excluded_H1': 1, 'cultural': 2}

## Hypotheses (directional replication targets)

- **H1** Composition >= Cooc on held-out (b) and external (c) strata: **not supported**
- **H2** Core has the lowest participation ratio and highest mean pairwise cosine: **not supported**
- **H3** Composition >= Cooc on regional Cohen's d: **not supported**

## Stratum means (rho / d) by model (mean across seeds)

| model | a_baked|rho | b_heldout|rho | c_external_excluded_H1|rho | c_external|rho | cultural|d |
|---|---|---|---|---|---|
| composition | 0.843 | 0.382 | 0.031 | 0.462 | 2.060 |
| composition_acoustic | 0.844 | 0.088 | 0.003 | 0.463 | 0.788 |
| composition_eqtok | 0.850 | 0.292 | 0.021 | 0.476 | 2.116 |
| composition_lyric | 0.109 | 0.434 | -0.055 | -0.005 | 2.565 |
| cooc | 0.174 | 0.527 | 0.002 | 0.042 | 3.247 |
| cooc_artistmasked | 0.198 | 0.539 | 0.004 | 0.048 | 3.040 |
| cooc_avg | 0.226 | 0.573 | -0.025 | 0.067 | 3.086 |
| cooc_eqtok | 0.631 | 0.541 | -0.050 | 0.289 | 3.108 |
| core | 0.840 | 0.542 | -0.008 | 0.475 | 3.056 |
| core_eqtok | 0.840 | 0.565 | -0.032 | 0.469 | 2.848 |
| core_partial30 | 0.617 | 0.550 | -0.018 | 0.304 | 2.854 |
