# S7 - evaluation

Eval tracks: 41,914 (test artists); targets: 90; strata: {'a_baked': 17, 'b_heldout': 50, 'c_external': 15, 'cultural': 8}

## Hypotheses (directional replication targets)

- **H1** Composition >= Cooc on held-out (b) and external (c) strata: **not supported**
- **H2** Core has the lowest participation ratio and highest mean pairwise cosine: **not supported**
- **H3** Composition >= Cooc on regional Cohen's d: **not supported**

## Stratum means (rho / d) by model (mean across seeds)

| model | a_baked|rho | b_heldout|rho | c_external|rho | cultural|d |
|---|---|---|---|---|
| composition | 0.643 | 0.333 | 0.505 | 0.641 |
| composition_acoustic | 0.644 | 0.236 | 0.498 | 0.491 |
| composition_eqtok | 0.646 | 0.333 | 0.503 | 0.658 |
| composition_lyric | 0.210 | 0.314 | 0.276 | 0.570 |
| cooc | 0.385 | 0.240 | 0.550 | 1.838 |
| cooc_artistmasked | 0.365 | 0.225 | 0.519 | 1.492 |
| cooc_avg | 0.385 | 0.239 | 0.550 | 1.956 |
| cooc_eqtok | 0.384 | 0.239 | 0.550 | 1.892 |
| core | 0.645 | 0.326 | 0.550 | 1.649 |
| core_partial10 | 0.384 | 0.237 | 0.540 | 1.869 |
| core_partial30 | 0.426 | 0.245 | 0.501 | 1.771 |
| core_partial60 | 0.535 | 0.281 | 0.509 | 1.617 |
