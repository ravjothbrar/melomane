# S1 - canonical vocabulary

| stage | tracks |
|---|---|
| 0 raw catalogue | 56,512 |
| vocabulary (>= 20 sessions, top 150000) | 53,787 |
| 1a exact-key simplification (strict) | 53,335 |
| 1b keyword non-music removal | 53,294 |
| 2 LLM refinement (keep/remove/consolidate) | 53,291 |
| 3 LLM ultra-refinement (rule-checked merges, restores) | 53,309 |
| 5 consolidation -> canonical tracks | 53,309 |

Consolidation ratio (vocabulary candidates / canonical): 1.01x
Audit: {'n': 40, 'prompt_version': 'audit_v1', 'agreement_merge': None, 'n_merge': 0, 'agreement_split': 1.0, 'n_split': 40}

Degraded: 1300/1333 refinement requests failed; those items kept unmerged