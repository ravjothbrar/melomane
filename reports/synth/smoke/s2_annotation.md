# S2 - annotation

Labelled 599/599 (coverage 100.0%); spent $0.00.

## Reliability (second prompt variant)

| dim                  |   n |   spearman |   wkappa |   exact_match |
|:---------------------|----:|-----------:|---------:|--------------:|
| energy               |  59 |          1 |        1 |           nan |
| aggression           |  59 |          1 |        1 |           nan |
| valence              |  59 |          1 |        1 |           nan |
| danceability         |  59 |          1 |        1 |           nan |
| acousticness         |  59 |          1 |        1 |           nan |
| vocal_prominence     |  59 |          1 |        1 |           nan |
| tempo_feel           |  59 |          1 |        1 |           nan |
| production_polish    |  59 |          1 |        1 |           nan |
| lyrical_explicitness |  59 |          1 |        1 |           nan |
| grit                 |  59 |          1 |        1 |           nan |
| density              |  59 |          1 |        1 |           nan |
| space                |  59 |          1 |        1 |           nan |
| punch                |  59 |          1 |        1 |           nan |
| warmth               |  59 |          1 |        1 |           nan |
| brightness           |  59 |          1 |        1 |           nan |
| era                  |  59 |          1 |        1 |           nan |
| category             |  59 |        nan |      nan |             1 |

## LLM vs measured

| dim        | measured      |   n |      rho |    ci_lo |    ci_hi | ci_excludes_0   |
|:-----------|:--------------|----:|---------:|---------:|---------:|:----------------|
| energy     | loudness_mean | 599 | 0.974914 | 0.971273 | 0.975815 | True            |
| aggression | loudness_mean | 599 | 0.834222 | 0.810577 | 0.848534 | True            |
| era        | year          | 599 | 0.979972 | 0.979014 | 0.980034 | True            |
