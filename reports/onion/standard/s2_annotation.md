# S2 - annotation

Labelled 8,504/53,309 (coverage 100.0%); spent $10.04.

## Reliability (second prompt variant)

| dim                  |   n |   spearman |     wkappa |   exact_match |
|:---------------------|----:|-----------:|-----------:|--------------:|
| energy               | 255 |   0.873659 |   0.869577 |    nan        |
| aggression           | 255 |   0.856797 |   0.88886  |    nan        |
| valence              | 255 |   0.845479 |   0.843198 |    nan        |
| danceability         | 255 |   0.836337 |   0.837039 |    nan        |
| acousticness         | 255 |   0.880477 |   0.893208 |    nan        |
| vocal_prominence     | 255 |   0.728453 |   0.864678 |    nan        |
| tempo_feel           | 255 |   0.833274 |   0.828629 |    nan        |
| production_polish    | 255 |   0.776555 |   0.79073  |    nan        |
| lyrical_explicitness | 255 |   0.758338 |   0.844311 |    nan        |
| grit                 | 255 |   0.86293  |   0.871403 |    nan        |
| density              | 255 |   0.789518 |   0.792716 |    nan        |
| space                | 255 |   0.296192 |   0.268529 |    nan        |
| punch                | 255 |   0.845691 |   0.853872 |    nan        |
| warmth               | 255 |   0.781708 |   0.766202 |    nan        |
| brightness           | 255 |   0.702943 |   0.696207 |    nan        |
| era                  | 254 |   0.99403  |   0.985425 |    nan        |
| category             | 255 | nan        | nan        |      0.901961 |

## LLM vs measured

| dim          | measured           |    n |        rho |      ci_lo |      ci_hi | ci_excludes_0   |
|:-------------|:-------------------|-----:|-----------:|-----------:|-----------:|:----------------|
| energy       | loudness           | 8504 |  0.373382  |  0.354996  |  0.390591  | True            |
| aggression   | dynamic_complexity | 8504 | -0.436613  | -0.456268  | -0.41836   | True            |
| danceability | danceability       | 8504 |  0.285048  |  0.263525  |  0.306439  | True            |
| tempo_feel   | bpm                | 8504 |  0.0769876 |  0.0544333 |  0.0978036 | True            |
| punch        | onset_rate         | 8504 |  0.18056   |  0.15928   |  0.203024  | True            |
| brightness   | spectral_centroid  | 8504 |  0.318083  |  0.299779  |  0.335527  | True            |
