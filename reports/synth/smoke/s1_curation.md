# S1 - canonical vocabulary

| stage | tracks |
|---|---|
| 0 raw catalogue | 644 |
| vocabulary (>= 8 sessions, top 4000) | 641 |
| 1a exact-key simplification (strict) | 607 |
| 1b keyword non-music removal | 599 |
| 2 LLM refinement (keep/remove/consolidate) | 599 |
| 3 LLM ultra-refinement (rule-checked merges, restores) | 599 |
| 5 consolidation -> canonical tracks | 599 |

Consolidation ratio (vocabulary candidates / canonical): 1.07x
Audit: {'n': 20, 'prompt_version': 'audit_v1', 'agreement_merge': 1.0, 'n_merge': 20, 'agreement_split': None, 'n_split': 0}
