# PROGRESS

Auto-written by the Melomane pipeline (newest last).

## 2026-10-05 18:54 UTC — S0-doctor (Environment, GPU, LLM smoke test) [synth/smoke] — PARTIAL
Ran in 11s. Key numbers: cuda=True, pairs_per_s=4.407e+06, disk_free_gb=1669, llm=no_key
Notes: OPENROUTER_API_KEY not set: LLM stages (curation, annotation, mode naming) will run in degraded rule-only mode
Resume: `python -m melomane run --stage S0-doctor --corpus synth --profile smoke` · disk free 1669 GB

## 2026-10-05 18:54 UTC — S0-ingest (Ingest raw data into the corpus contract) [synth/smoke] — GREEN
Ran in 0s. Key numbers: tracks=644, sessions=1000, session_track_links=52211, families=5, has_measured=True
Notes: DEMO corpus (synthetic, planted structure) - results are not music findings
Resume: `python -m melomane run --stage S0-ingest --corpus synth --profile smoke` · disk free 1669 GB

## 2026-10-05 18:54 UTC — S0-enrich (Match tracks to a feature corpus (partial-hub compounds)) [synth/smoke] — GREEN
Ran in 0s. Key numbers: -
Notes: feature corpus: nothing to enrich
Resume: `python -m melomane run --stage S0-enrich --corpus synth --profile smoke` · disk free 1669 GB

## 2026-10-05 18:54 UTC — S1 (Canonical vocabulary (curation)) [synth/smoke] — GREEN
Ran in 0s. Key numbers: V=599, consolidation_ratio=1.07, removed=8, audit_agreement_merge=1
Resume: `python -m melomane run --stage S1 --corpus synth --profile smoke` · disk free 1669 GB

## 2026-10-05 18:54 UTC — S2 (LLM annotation & measured ground truth) [synth/smoke] — GREEN
Ran in 0s. Key numbers: pilot_usd_per_1k_tracks=0, coverage=1, spent_usd=0, h5_share=1
Resume: `python -m melomane run --stage S2 --corpus synth --profile smoke` · disk free 1669 GB

## 2026-10-05 18:54 UTC — S3 (Behaviour graph (session NPMI)) [synth/smoke] — GREEN
Ran in 0s. Key numbers: n_sessions=1000, mean_degree=46.79, same_artist_edge_share=0.1119, n_isolated=1
Decisions: mean degree 47 outside the 100-250 target of the plan (corpus/sampling dependent); recorded, not tuned
Resume: `python -m melomane run --stage S3 --corpus synth --profile smoke` · disk free 1669 GB

## 2026-10-05 18:54 UTC — S4 (Compound graph (sonic / lyrical families)) [synth/smoke] — PARTIAL
Ran in 1s. Key numbers: n_compounds=190, max_cluster_share=0.1503, tracks_with_compounds=1
Notes: cluster sanity: empty=0, max share=0.150 (limit 0.05)
Resume: `python -m melomane run --stage S4 --corpus synth --profile smoke` · disk free 1669 GB

## 2026-10-05 18:56 UTC — S5 (Walks & SGNS training (models, seeds, ablations)) [synth/smoke] — PARTIAL
Ran in 105s. Key numbers: models_trained=11, recall_cooc=0.2843, recall_composition=0.6667, recall_core=0.6921
Notes: cooc: sanity flags {'finite': True, 'participation_ratio': 3.9413299560546875, 'mean_pairwise_cosine': 0.4861118495464325, 'seeds': 2, 'seconds': 0.6980671882629395, 'device': 'cuda', 'epochs': 3, 'tokens_per_epoch': 71880, 'loss_first': 3.1184239326379237, 'loss_last': 1.828558087348938, 'loss_decreasing': True, 'neighbour_recall_at10_seed0_vs_1': 0.2842809364548495} | cooc_artistmasked: sanity flags {'finite': True, 'participation_ratio': 3.890004873275757, 'mean_pairwise_cosine': 0.48082101345062256, 'seeds': 1, 'seconds': 1.0885426998138428, 'device': 'cuda', 'epochs': 3, 'tokens_per_epoch': 71880, 'loss_first': 3.1215319083287167, 'loss_last': 1.868633319170047, 'loss_decreasing': True}
Resume: `python -m melomane run --stage S5 --corpus synth --profile smoke` · disk free 1669 GB

## 2026-10-05 18:56 UTC — S6 (ICA, modes & poles) [synth/smoke] — FAILED
Ran in 3s. Key numbers: -
Notes: AttributeError: 'DemoLLM' object has no attribute '_audit_modes'
Resume: `python -m melomane run --stage S6 --corpus synth --profile smoke` · disk free 1669 GB

## 2026-10-05 18:56 UTC — S13 (White paper generation (tables, figures, MD/LaTeX)) [synth/smoke] — GREEN
Ran in 0s. Key numbers: primary_corpus=synth, figures_generated=0, hypotheses_evaluated=0
Notes: Paper generated in <repo>\paper/ (Markdown + LaTeX arXiv preprint)
Resume: `python -m melomane run --stage S13 --corpus synth --profile smoke` · disk free 1669 GB

## 2026-10-05 18:56 UTC — S14 (Release QA (secrets, privacy, integrity checks)) [synth/smoke] — GREEN
Ran in 0s. Key numbers: all_passed=True, issues_count=0, secrets_hits=0
Resume: `python -m melomane run --stage S14 --corpus synth --profile smoke` · disk free 1669 GB

## 2026-10-05 18:57 UTC — S6 (ICA, modes & poles) [synth/smoke] — FAILED
Ran in 11s. Key numbers: -
Notes: ValueError: Out of range float values are not JSON compliant: nan
Resume: `python -m melomane run --stage S6 --corpus synth --profile smoke` · disk free 1669 GB

## 2026-10-05 18:58 UTC — S6 (ICA, modes & poles) [synth/smoke] — GREEN
Ran in 15s. Key numbers: models=7, modes_cooc=48, modes_composition=21, modes_core=45, modes_composition_acoustic=24, modes_composition_lyric=49, modes_cooc_artistmasked=46, modes_cooc_avg=44
Resume: `python -m melomane run --stage S6 --corpus synth --profile smoke` · disk free 1669 GB

## 2026-10-05 18:58 UTC — S7 (Evaluation battery) [synth/smoke] — GREEN
Ran in 8s. Key numbers: models=11, targets=23, H1=not supported, H2=not supported, H3=not supported
Resume: `python -m melomane run --stage S7 --corpus synth --profile smoke` · disk free 1669 GB

## 2026-10-05 18:58 UTC — S8 (SLERP & direction arithmetic) [synth/smoke] — FAILED
Ran in 0s. Key numbers: -
Notes: IndexError: index 598 is out of bounds for axis 0 with size 598
Resume: `python -m melomane run --stage S8 --corpus synth --profile smoke` · disk free 1669 GB

## 2026-10-05 19:03 UTC — S0-doctor (Environment, GPU, LLM smoke test) [synth/smoke] — PARTIAL
Ran in 8s. Key numbers: cuda=True, pairs_per_s=4.392e+06, disk_free_gb=1669, llm=no_key
Notes: OPENROUTER_API_KEY not set: LLM stages (curation, annotation, mode naming) will run in degraded rule-only mode
Resume: `python -m melomane run --stage S0-doctor --corpus synth --profile smoke` · disk free 1669 GB

## 2026-10-05 19:03 UTC — S0-ingest (Ingest raw data into the corpus contract) [synth/smoke] — GREEN
Ran in 0s. Key numbers: tracks=644, sessions=1000, session_track_links=52211, families=5, has_measured=True
Notes: DEMO corpus (synthetic, planted structure) - results are not music findings
Resume: `python -m melomane run --stage S0-ingest --corpus synth --profile smoke` · disk free 1669 GB

## 2026-10-05 19:03 UTC — S0-enrich (Match tracks to a feature corpus (partial-hub compounds)) [synth/smoke] — GREEN
Ran in 0s. Key numbers: -
Notes: feature corpus: nothing to enrich
Resume: `python -m melomane run --stage S0-enrich --corpus synth --profile smoke` · disk free 1669 GB

## 2026-10-05 19:03 UTC — S1 (Canonical vocabulary (curation)) [synth/smoke] — GREEN
Ran in 1s. Key numbers: V=599, consolidation_ratio=1.07, removed=8, audit_agreement_merge=1
Resume: `python -m melomane run --stage S1 --corpus synth --profile smoke` · disk free 1669 GB

## 2026-10-05 19:03 UTC — S2 (LLM annotation & measured ground truth) [synth/smoke] — GREEN
Ran in 0s. Key numbers: pilot_usd_per_1k_tracks=0, coverage=1, spent_usd=0, h5_share=1
Resume: `python -m melomane run --stage S2 --corpus synth --profile smoke` · disk free 1669 GB

## 2026-10-05 19:03 UTC — S3 (Behaviour graph (session NPMI)) [synth/smoke] — GREEN
Ran in 0s. Key numbers: n_sessions=1000, mean_degree=46.79, same_artist_edge_share=0.1119, n_isolated=1
Decisions: mean degree 47 outside the 100-250 target of the plan (corpus/sampling dependent); recorded, not tuned
Resume: `python -m melomane run --stage S3 --corpus synth --profile smoke` · disk free 1669 GB

## 2026-10-05 19:03 UTC — S4 (Compound graph (sonic / lyrical families)) [synth/smoke] — PARTIAL
Ran in 1s. Key numbers: n_compounds=190, max_cluster_share=0.1503, tracks_with_compounds=1
Notes: cluster sanity: empty=0, max share=0.150 (limit 0.05)
Resume: `python -m melomane run --stage S4 --corpus synth --profile smoke` · disk free 1669 GB

## 2026-10-05 19:03 UTC — S5 (Walks & SGNS training (models, seeds, ablations)) [synth/smoke] — PARTIAL
Ran in 0s. Key numbers: models_trained=11, recall_cooc=0.2843, recall_composition=0.6667, recall_core=0.6921
Notes: cooc: sanity flags {'finite': True, 'participation_ratio': 3.9413299560546875, 'mean_pairwise_cosine': 0.4861118495464325, 'seeds': 2, 'seconds': 0.6980671882629395, 'device': 'cuda', 'epochs': 3, 'tokens_per_epoch': 71880, 'loss_first': 3.1184239326379237, 'loss_last': 1.828558087348938, 'loss_decreasing': True, 'neighbour_recall_at10_seed0_vs_1': 0.2842809364548495} | cooc_artistmasked: sanity flags {'finite': True, 'participation_ratio': 3.890004873275757, 'mean_pairwise_cosine': 0.48082101345062256, 'seeds': 1, 'seconds': 1.0885426998138428, 'device': 'cuda', 'epochs': 3, 'tokens_per_epoch': 71880, 'loss_first': 3.1215319083287167, 'loss_last': 1.868633319170047, 'loss_decreasing': True}
Resume: `python -m melomane run --stage S5 --corpus synth --profile smoke` · disk free 1669 GB

## 2026-10-05 19:04 UTC — S6 (ICA, modes & poles) [synth/smoke] — GREEN
Ran in 18s. Key numbers: models=7, modes_cooc=48, modes_composition=21, modes_core=45, modes_composition_acoustic=24, modes_composition_lyric=49, modes_cooc_artistmasked=46, modes_cooc_avg=44
Resume: `python -m melomane run --stage S6 --corpus synth --profile smoke` · disk free 1669 GB

## 2026-10-05 19:04 UTC — S7 (Evaluation battery) [synth/smoke] — GREEN
Ran in 8s. Key numbers: models=11, targets=23, H1=not supported, H2=not supported, H3=not supported
Resume: `python -m melomane run --stage S7 --corpus synth --profile smoke` · disk free 1669 GB

## 2026-10-05 19:04 UTC — S8 (SLERP & direction arithmetic) [synth/smoke] — GREEN
Ran in 1s. Key numbers: H7=supported, share60_cooc=0.6923, share60_composition=0.913, share60_core=0.8462, share60_composition_acoustic=0.96, share60_composition_lyric=0.6667, share60_cooc_artistmasked=0.7083
Resume: `python -m melomane run --stage S8 --corpus synth --profile smoke` · disk free 1669 GB

## 2026-10-05 19:04 UTC — S9 (Extensions (E1-E8, H4, H6, H8)) [synth/smoke] — GREEN
Ran in 33s. Key numbers: E1=done, E2=done, E3=done, E4=done, E5=done, E6=not evaluable, E7=done, E8=done
Notes: E6: not evaluable - sessions carry no timestamps (playlists / count-only logs): temporal stability needs dated sessions
Resume: `python -m melomane run --stage S9 --corpus synth --profile smoke` · disk free 1669 GB

## 2026-10-05 19:04 UTC — S13 (White paper generation (tables, figures, MD/LaTeX)) [synth/smoke] — GREEN
Ran in 1s. Key numbers: primary_corpus=synth, figures_generated=2, hypotheses_evaluated=6
Notes: Paper generated in <repo>\paper/ (Markdown + LaTeX arXiv preprint)
Resume: `python -m melomane run --stage S13 --corpus synth --profile smoke` · disk free 1669 GB

## 2026-10-05 19:04 UTC — S14 (Release QA (secrets, privacy, integrity checks)) [synth/smoke] — GREEN
Ran in 0s. Key numbers: all_passed=True, issues_count=0, secrets_hits=0
Resume: `python -m melomane run --stage S14 --corpus synth --profile smoke` · disk free 1669 GB

## 2026-10-05 19:14 UTC — S0-ingest (Ingest raw data into the corpus contract) [synth/smoke] — GREEN
Ran in 0s. Key numbers: tracks=1598, sessions=2500, session_track_links=128443, families=5, has_measured=True
Notes: DEMO corpus (synthetic, planted structure) - results are not music findings
Resume: `python -m melomane run --stage S0-ingest --corpus synth --profile smoke` · disk free 1669 GB

## 2026-10-05 19:14 UTC — S0-enrich (Match tracks to a feature corpus (partial-hub compounds)) [synth/smoke] — GREEN
Ran in 0s. Key numbers: -
Notes: feature corpus: nothing to enrich
Resume: `python -m melomane run --stage S0-enrich --corpus synth --profile smoke` · disk free 1669 GB

## 2026-10-05 19:14 UTC — S1 (Canonical vocabulary (curation)) [synth/smoke] — GREEN
Ran in 1s. Key numbers: V=1498, consolidation_ratio=1.063, removed=8, audit_agreement_merge=1
Resume: `python -m melomane run --stage S1 --corpus synth --profile smoke` · disk free 1668 GB

## 2026-10-05 19:14 UTC — S2 (LLM annotation & measured ground truth) [synth/smoke] — FAILED
Ran in 0s. Key numbers: -
Notes: IndexError: Boolean index has wrong length: 599 instead of 1498
Resume: `python -m melomane run --stage S2 --corpus synth --profile smoke` · disk free 1668 GB

## 2026-10-05 19:14 UTC — S3 (Behaviour graph (session NPMI)) [synth/smoke] — FAILED
Ran in 0s. Key numbers: -
Notes: IndexError: Boolean index has wrong length: 599 instead of 1498
Resume: `python -m melomane run --stage S3 --corpus synth --profile smoke` · disk free 1668 GB

## 2026-10-05 19:14 UTC — S4 (Compound graph (sonic / lyrical families)) [synth/smoke] — FAILED
Ran in 0s. Key numbers: -
Notes: IndexError: Boolean index has wrong length: 599 instead of 1498
Resume: `python -m melomane run --stage S4 --corpus synth --profile smoke` · disk free 1668 GB

## 2026-10-05 19:14 UTC — S0-doctor (Environment, GPU, LLM smoke test) [synth/smoke] — PARTIAL
Ran in 6s. Key numbers: cuda=True, pairs_per_s=5.261e+06, disk_free_gb=1669, llm=no_key
Notes: OPENROUTER_API_KEY not set: LLM stages (curation, annotation, mode naming) will run in degraded rule-only mode
Resume: `python -m melomane run --stage S0-doctor --corpus synth --profile smoke` · disk free 1669 GB

## 2026-10-05 19:15 UTC — S0-doctor (Environment, GPU, LLM smoke test) [synth/smoke] — PARTIAL
Ran in 5s. Key numbers: cuda=True, pairs_per_s=5.767e+06, disk_free_gb=1669, llm=no_key
Notes: OPENROUTER_API_KEY not set: LLM stages (curation, annotation, mode naming) will run in degraded rule-only mode
Resume: `python -m melomane run --stage S0-doctor --corpus synth --profile smoke` · disk free 1669 GB

## 2026-10-05 19:15 UTC — S0-ingest (Ingest raw data into the corpus contract) [synth/smoke] — GREEN
Ran in 0s. Key numbers: tracks=644, sessions=1000, session_track_links=52211, families=5, has_measured=True
Notes: DEMO corpus (synthetic, planted structure) - results are not music findings
Resume: `python -m melomane run --stage S0-ingest --corpus synth --profile smoke` · disk free 1669 GB

## 2026-10-05 19:15 UTC — S0-enrich (Match tracks to a feature corpus (partial-hub compounds)) [synth/smoke] — GREEN
Ran in 0s. Key numbers: -
Notes: feature corpus: nothing to enrich
Resume: `python -m melomane run --stage S0-enrich --corpus synth --profile smoke` · disk free 1669 GB

## 2026-10-05 19:15 UTC — S1 (Canonical vocabulary (curation)) [synth/smoke] — GREEN
Ran in 0s. Key numbers: V=599, consolidation_ratio=1.07, removed=8, audit_agreement_merge=1
Resume: `python -m melomane run --stage S1 --corpus synth --profile smoke` · disk free 1669 GB

## 2026-10-05 19:15 UTC — S2 (LLM annotation & measured ground truth) [synth/smoke] — GREEN
Ran in 0s. Key numbers: pilot_usd_per_1k_tracks=0, coverage=1, spent_usd=0, h5_share=1
Resume: `python -m melomane run --stage S2 --corpus synth --profile smoke` · disk free 1669 GB

## 2026-10-05 19:15 UTC — S3 (Behaviour graph (session NPMI)) [synth/smoke] — GREEN
Ran in 0s. Key numbers: n_sessions=1000, mean_degree=46.79, same_artist_edge_share=0.1119, n_isolated=1
Decisions: mean degree 47 outside the 100-250 target of the plan (corpus/sampling dependent); recorded, not tuned
Resume: `python -m melomane run --stage S3 --corpus synth --profile smoke` · disk free 1669 GB

## 2026-10-05 19:15 UTC — S4 (Compound graph (sonic / lyrical families)) [synth/smoke] — PARTIAL
Ran in 1s. Key numbers: n_compounds=190, max_cluster_share=0.1503, tracks_with_compounds=1
Notes: cluster sanity: empty=0, max share=0.150 (limit 0.05)
Resume: `python -m melomane run --stage S4 --corpus synth --profile smoke` · disk free 1669 GB

## 2026-10-05 19:15 UTC — S5 (Walks & SGNS training (models, seeds, ablations)) [synth/smoke] — PARTIAL
Ran in 0s. Key numbers: models_trained=11, recall_cooc=0.2843, recall_composition=0.6667, recall_core=0.6921
Notes: cooc: sanity flags {'finite': True, 'participation_ratio': 3.9413299560546875, 'mean_pairwise_cosine': 0.4861118495464325, 'seeds': 2, 'seconds': 0.6980671882629395, 'device': 'cuda', 'epochs': 3, 'tokens_per_epoch': 71880, 'loss_first': 3.1184239326379237, 'loss_last': 1.828558087348938, 'loss_decreasing': True, 'neighbour_recall_at10_seed0_vs_1': 0.2842809364548495} | cooc_artistmasked: sanity flags {'finite': True, 'participation_ratio': 3.890004873275757, 'mean_pairwise_cosine': 0.48082101345062256, 'seeds': 1, 'seconds': 1.0885426998138428, 'device': 'cuda', 'epochs': 3, 'tokens_per_epoch': 71880, 'loss_first': 3.1215319083287167, 'loss_last': 1.868633319170047, 'loss_decreasing': True}
Resume: `python -m melomane run --stage S5 --corpus synth --profile smoke` · disk free 1669 GB

## 2026-10-05 19:15 UTC — S6 (ICA, modes & poles) [synth/smoke] — GREEN
Ran in 10s. Key numbers: models=7, modes_cooc=48, modes_composition=21, modes_core=45, modes_composition_acoustic=24, modes_composition_lyric=49, modes_cooc_artistmasked=46, modes_cooc_avg=44
Resume: `python -m melomane run --stage S6 --corpus synth --profile smoke` · disk free 1669 GB

## 2026-10-05 19:15 UTC — S7 (Evaluation battery) [synth/smoke] — GREEN
Ran in 4s. Key numbers: models=11, targets=23, H1=not supported, H2=not supported, H3=not supported
Resume: `python -m melomane run --stage S7 --corpus synth --profile smoke` · disk free 1669 GB

## 2026-10-05 19:15 UTC — S8 (SLERP & direction arithmetic) [synth/smoke] — GREEN
Ran in 0s. Key numbers: H7=supported, share60_cooc=0.6923, share60_composition=0.913, share60_core=0.8462, share60_composition_acoustic=0.96, share60_composition_lyric=0.6667, share60_cooc_artistmasked=0.7083
Resume: `python -m melomane run --stage S8 --corpus synth --profile smoke` · disk free 1669 GB

## 2026-10-05 19:15 UTC — S9 (Extensions (E1-E8, H4, H6, H8)) [synth/smoke] — GREEN
Ran in 6s. Key numbers: E1=done, E2=done, E3=done, E4=done, E5=done, E6=not evaluable, E7=done, E8=done
Notes: E6: not evaluable - sessions carry no timestamps (playlists / count-only logs): temporal stability needs dated sessions
Resume: `python -m melomane run --stage S9 --corpus synth --profile smoke` · disk free 1669 GB

## 2026-10-05 19:15 UTC — S13 (White paper generation (tables, figures, MD/LaTeX)) [synth/smoke] — GREEN
Ran in 0s. Key numbers: primary_corpus=synth, figures_generated=2, hypotheses_evaluated=6
Notes: Paper generated in <repo>\paper/ (Markdown + LaTeX arXiv preprint)
Resume: `python -m melomane run --stage S13 --corpus synth --profile smoke` · disk free 1669 GB

## 2026-10-05 19:15 UTC — S14 (Release QA (secrets, privacy, integrity checks)) [synth/smoke] — GREEN
Ran in 0s. Key numbers: all_passed=True, issues_count=0, secrets_hits=0
Resume: `python -m melomane run --stage S14 --corpus synth --profile smoke` · disk free 1669 GB

## 2026-10-05 19:15 UTC — S0-doctor (Environment, GPU, LLM smoke test) [synth/smoke] — PARTIAL
Ran in 5s. Key numbers: cuda=True, pairs_per_s=5.808e+06, disk_free_gb=1669, llm=no_key
Notes: OPENROUTER_API_KEY not set: LLM stages (curation, annotation, mode naming) will run in degraded rule-only mode
Resume: `python -m melomane run --stage S0-doctor --corpus synth --profile smoke` · disk free 1669 GB

## 2026-10-05 19:41 UTC — S0-doctor (Environment, GPU, LLM smoke test) [synth/smoke] — PARTIAL
Ran in 5s. Key numbers: cuda=True, pairs_per_s=5.593e+06, disk_free_gb=1668, llm=no_key
Notes: OPENROUTER_API_KEY not set: LLM stages (curation, annotation, mode naming) will run in degraded rule-only mode
Resume: `python -m melomane run --stage S0-doctor --corpus synth --profile smoke` · disk free 1668 GB

## 2026-10-05 19:41 UTC — S0-ingest (Ingest raw data into the corpus contract) [synth/smoke] — GREEN
Ran in 0s. Key numbers: tracks=644, sessions=1000, session_track_links=52211, families=5, has_measured=True
Notes: DEMO corpus (synthetic, planted structure) - results are not music findings
Resume: `python -m melomane run --stage S0-ingest --corpus synth --profile smoke` · disk free 1668 GB

## 2026-10-05 19:41 UTC — S0-enrich (Match tracks to a feature corpus (partial-hub compounds)) [synth/smoke] — GREEN
Ran in 0s. Key numbers: -
Notes: feature corpus: nothing to enrich
Resume: `python -m melomane run --stage S0-enrich --corpus synth --profile smoke` · disk free 1668 GB

## 2026-10-05 19:41 UTC — S1 (Canonical vocabulary (curation)) [synth/smoke] — GREEN
Ran in 0s. Key numbers: V=599, consolidation_ratio=1.07, removed=8, audit_agreement_merge=1
Resume: `python -m melomane run --stage S1 --corpus synth --profile smoke` · disk free 1668 GB

## 2026-10-05 19:41 UTC — S2 (LLM annotation & measured ground truth) [synth/smoke] — GREEN
Ran in 0s. Key numbers: pilot_usd_per_1k_tracks=0, coverage=1, spent_usd=0, h5_share=1
Resume: `python -m melomane run --stage S2 --corpus synth --profile smoke` · disk free 1668 GB

## 2026-10-05 19:41 UTC — S3 (Behaviour graph (session NPMI)) [synth/smoke] — GREEN
Ran in 0s. Key numbers: n_sessions=1000, mean_degree=46.79, same_artist_edge_share=0.1119, n_isolated=1
Decisions: mean degree 47 outside the 100-250 target of the plan (corpus/sampling dependent); recorded, not tuned
Resume: `python -m melomane run --stage S3 --corpus synth --profile smoke` · disk free 1668 GB

## 2026-10-05 19:41 UTC — S4 (Compound graph (sonic / lyrical families)) [synth/smoke] — PARTIAL
Ran in 1s. Key numbers: n_compounds=190, max_cluster_share=0.1503, tracks_with_compounds=1
Notes: cluster sanity: empty=0, max share=0.150 (limit 0.05)
Resume: `python -m melomane run --stage S4 --corpus synth --profile smoke` · disk free 1668 GB

## 2026-10-05 19:41 UTC — S5 (Walks & SGNS training (models, seeds, ablations)) [synth/smoke] — PARTIAL
Ran in 0s. Key numbers: models_trained=11, recall_cooc=0.2843, recall_composition=0.6667, recall_core=0.6921
Notes: cooc: sanity flags {'finite': True, 'participation_ratio': 3.9413299560546875, 'mean_pairwise_cosine': 0.4861118495464325, 'seeds': 2, 'seconds': 0.6980671882629395, 'device': 'cuda', 'epochs': 3, 'tokens_per_epoch': 71880, 'loss_first': 3.1184239326379237, 'loss_last': 1.828558087348938, 'loss_decreasing': True, 'neighbour_recall_at10_seed0_vs_1': 0.2842809364548495} | cooc_artistmasked: sanity flags {'finite': True, 'participation_ratio': 3.890004873275757, 'mean_pairwise_cosine': 0.48082101345062256, 'seeds': 1, 'seconds': 1.0885426998138428, 'device': 'cuda', 'epochs': 3, 'tokens_per_epoch': 71880, 'loss_first': 3.1215319083287167, 'loss_last': 1.868633319170047, 'loss_decreasing': True}
Resume: `python -m melomane run --stage S5 --corpus synth --profile smoke` · disk free 1668 GB

## 2026-10-05 19:41 UTC — S6 (ICA, modes & poles) [synth/smoke] — GREEN
Ran in 10s. Key numbers: models=7, modes_cooc=48, modes_composition=21, modes_core=45, modes_composition_acoustic=24, modes_composition_lyric=49, modes_cooc_artistmasked=46, modes_cooc_avg=44
Resume: `python -m melomane run --stage S6 --corpus synth --profile smoke` · disk free 1668 GB

## 2026-10-05 19:42 UTC — S7 (Evaluation battery) [synth/smoke] — GREEN
Ran in 4s. Key numbers: models=11, targets=23, H1=not supported, H2=not supported, H3=not supported
Resume: `python -m melomane run --stage S7 --corpus synth --profile smoke` · disk free 1668 GB

## 2026-10-05 19:42 UTC — S8 (SLERP & direction arithmetic) [synth/smoke] — GREEN
Ran in 0s. Key numbers: H7=supported, share60_cooc=0.6923, share60_composition=0.913, share60_core=0.8462, share60_composition_acoustic=0.96, share60_composition_lyric=0.6667, share60_cooc_artistmasked=0.7083
Resume: `python -m melomane run --stage S8 --corpus synth --profile smoke` · disk free 1668 GB

## 2026-10-05 19:42 UTC — S9 (Extensions (E1-E8, H4, H6, H8)) [synth/smoke] — GREEN
Ran in 6s. Key numbers: E1=done, E2=done, E3=done, E4=done, E5=done, E6=not evaluable, E7=done, E8=done
Notes: E6: not evaluable - sessions carry no timestamps (playlists / count-only logs): temporal stability needs dated sessions
Resume: `python -m melomane run --stage S9 --corpus synth --profile smoke` · disk free 1668 GB

## 2026-10-05 19:42 UTC — S13 (White paper generation (tables, figures, MD/LaTeX)) [synth/smoke] — GREEN
Ran in 0s. Key numbers: primary_corpus=synth, figures_generated=2, hypotheses_evaluated=6
Notes: Paper generated in <repo>\paper/ (Markdown + LaTeX arXiv preprint)
Resume: `python -m melomane run --stage S13 --corpus synth --profile smoke` · disk free 1668 GB

## 2026-10-05 19:42 UTC — S14 (Release QA (secrets, privacy, integrity checks)) [synth/smoke] — GREEN
Ran in 0s. Key numbers: all_passed=True, issues_count=0, secrets_hits=0
Resume: `python -m melomane run --stage S14 --corpus synth --profile smoke` · disk free 1668 GB

## 2026-10-05 20:13 UTC — S0-doctor (Environment, GPU, LLM smoke test) [synth/smoke] — GREEN
Ran in 7s. Key numbers: cuda=True, pairs_per_s=5.457e+06, disk_free_gb=1668, llm=ok
Resume: `python -m melomane run --stage S0-doctor --corpus synth --profile smoke` · disk free 1668 GB

## 2026-10-05 20:24 UTC — S0-doctor (Environment, GPU, LLM smoke test) [synth/smoke] — GREEN
Ran in 6s. Key numbers: cuda=True, pairs_per_s=7.409e+06, disk_free_gb=1668, llm=ok
Resume: `python -m melomane run --stage S0-doctor --corpus synth --profile smoke` · disk free 1668 GB

## 2026-10-05 20:24 UTC — S0-ingest (Ingest raw data into the corpus contract) [synth/smoke] — GREEN
Ran in 0s. Key numbers: tracks=644, sessions=1000, session_track_links=52211, families=5, has_measured=True
Notes: DEMO corpus (synthetic, planted structure) - results are not music findings
Resume: `python -m melomane run --stage S0-ingest --corpus synth --profile smoke` · disk free 1668 GB

## 2026-10-05 20:24 UTC — S0-enrich (Match tracks to a feature corpus (partial-hub compounds)) [synth/smoke] — GREEN
Ran in 0s. Key numbers: -
Notes: feature corpus: nothing to enrich
Resume: `python -m melomane run --stage S0-enrich --corpus synth --profile smoke` · disk free 1668 GB

## 2026-10-05 20:24 UTC — S1 (Canonical vocabulary (curation)) [synth/smoke] — GREEN
Ran in 0s. Key numbers: V=599, consolidation_ratio=1.07, removed=8, audit_agreement_merge=1
Resume: `python -m melomane run --stage S1 --corpus synth --profile smoke` · disk free 1668 GB

## 2026-10-05 20:24 UTC — S2 (LLM annotation & measured ground truth) [synth/smoke] — GREEN
Ran in 0s. Key numbers: pilot_usd_per_1k_tracks=0, coverage=1, spent_usd=0, h5_share=1
Resume: `python -m melomane run --stage S2 --corpus synth --profile smoke` · disk free 1668 GB

## 2026-10-05 20:24 UTC — S3 (Behaviour graph (session NPMI)) [synth/smoke] — GREEN
Ran in 0s. Key numbers: n_sessions=1000, mean_degree=46.79, same_artist_edge_share=0.1119, n_isolated=1
Decisions: mean degree 47 outside the 100-250 target of the plan (corpus/sampling dependent); recorded, not tuned
Resume: `python -m melomane run --stage S3 --corpus synth --profile smoke` · disk free 1668 GB

## 2026-10-05 20:24 UTC — S4 (Compound graph (sonic / lyrical families)) [synth/smoke] — PARTIAL
Ran in 1s. Key numbers: n_compounds=190, max_cluster_share=0.1503, tracks_with_compounds=1
Notes: cluster sanity: empty=0, max share=0.150 (limit 0.05)
Resume: `python -m melomane run --stage S4 --corpus synth --profile smoke` · disk free 1668 GB

## 2026-10-05 20:24 UTC — S5 (Walks & SGNS training (models, seeds, ablations)) [synth/smoke] — PARTIAL
Ran in 0s. Key numbers: models_trained=11, recall_cooc=0.2843, recall_composition=0.6667, recall_core=0.6921
Notes: cooc: sanity flags {'finite': True, 'participation_ratio': 3.9413299560546875, 'mean_pairwise_cosine': 0.4861118495464325, 'seeds': 2, 'seconds': 0.6980671882629395, 'device': 'cuda', 'epochs': 3, 'tokens_per_epoch': 71880, 'loss_first': 3.1184239326379237, 'loss_last': 1.828558087348938, 'loss_decreasing': True, 'neighbour_recall_at10_seed0_vs_1': 0.2842809364548495} | cooc_artistmasked: sanity flags {'finite': True, 'participation_ratio': 3.890004873275757, 'mean_pairwise_cosine': 0.48082101345062256, 'seeds': 1, 'seconds': 1.0885426998138428, 'device': 'cuda', 'epochs': 3, 'tokens_per_epoch': 71880, 'loss_first': 3.1215319083287167, 'loss_last': 1.868633319170047, 'loss_decreasing': True}
Resume: `python -m melomane run --stage S5 --corpus synth --profile smoke` · disk free 1668 GB

## 2026-10-05 20:24 UTC — S6 (ICA, modes & poles) [synth/smoke] — GREEN
Ran in 9s. Key numbers: models=7, modes_cooc=48, modes_composition=21, modes_core=45, modes_composition_acoustic=24, modes_composition_lyric=49, modes_cooc_artistmasked=46, modes_cooc_avg=44
Resume: `python -m melomane run --stage S6 --corpus synth --profile smoke` · disk free 1668 GB

## 2026-10-05 20:24 UTC — S7 (Evaluation battery) [synth/smoke] — GREEN
Ran in 4s. Key numbers: models=11, targets=23, H1=not supported, H2=not supported, H3=not supported
Resume: `python -m melomane run --stage S7 --corpus synth --profile smoke` · disk free 1668 GB

## 2026-10-05 20:24 UTC — S8 (SLERP & direction arithmetic) [synth/smoke] — GREEN
Ran in 0s. Key numbers: H7=supported, share60_cooc=0.6923, share60_composition=0.913, share60_core=0.8462, share60_composition_acoustic=0.96, share60_composition_lyric=0.6667, share60_cooc_artistmasked=0.7083
Resume: `python -m melomane run --stage S8 --corpus synth --profile smoke` · disk free 1668 GB

## 2026-10-05 20:24 UTC — S9 (Extensions (E1-E8, H4, H6, H8)) [synth/smoke] — GREEN
Ran in 6s. Key numbers: E1=done, E2=done, E3=done, E4=done, E5=done, E6=not evaluable, E7=done, E8=done
Notes: E6: not evaluable - sessions carry no timestamps (playlists / count-only logs): temporal stability needs dated sessions
Resume: `python -m melomane run --stage S9 --corpus synth --profile smoke` · disk free 1668 GB

## 2026-10-05 20:24 UTC — S13 (White paper generation (tables, figures, MD/LaTeX)) [synth/smoke] — GREEN
Ran in 0s. Key numbers: primary_corpus=synth, figures_generated=2, hypotheses_evaluated=6
Notes: Paper generated in <repo>\paper/ (Markdown + LaTeX arXiv preprint)
Resume: `python -m melomane run --stage S13 --corpus synth --profile smoke` · disk free 1668 GB

## 2026-10-05 20:24 UTC — S14 (Release QA (secrets, privacy, integrity checks)) [synth/smoke] — GREEN
Ran in 0s. Key numbers: all_passed=True, issues_count=0, secrets_hits=0
Resume: `python -m melomane run --stage S14 --corpus synth --profile smoke` · disk free 1668 GB

## 2026-10-05 20:24 UTC — S0-doctor (Environment, GPU, LLM smoke test) [onion/standard] — GREEN
Ran in 10s. Key numbers: cuda=True, pairs_per_s=7.396e+06, disk_free_gb=1668, llm=ok
Resume: `python -m melomane run --stage S0-doctor --corpus onion --profile standard` · disk free 1668 GB

## 2026-10-05 20:24 UTC — S0-ingest (Ingest raw data into the corpus contract) [onion/standard] — FAILED
Ran in 0s. Key numbers: -
Notes: FileNotFoundError: no listening-event file (userid_trackid_timestamp / userid_trackid_count) in <repo>\data\raw\onion
Resume: `python -m melomane run --stage S0-ingest --corpus onion --profile standard` · disk free 1668 GB

## 2026-10-05 20:24 UTC — S13 (White paper generation (tables, figures, MD/LaTeX)) [onion/standard] — GREEN
Ran in 0s. Key numbers: primary_corpus=onion, figures_generated=0, hypotheses_evaluated=0
Notes: Paper generated in <repo>\paper/ (Markdown + LaTeX arXiv preprint)
Resume: `python -m melomane run --stage S13 --corpus onion --profile standard` · disk free 1668 GB

## 2026-10-05 20:24 UTC — S14 (Release QA (secrets, privacy, integrity checks)) [onion/standard] — GREEN
Ran in 0s. Key numbers: all_passed=True, issues_count=0, secrets_hits=0
Resume: `python -m melomane run --stage S14 --corpus onion --profile standard` · disk free 1668 GB

## 2026-10-05 20:25 UTC — S0-doctor (Environment, GPU, LLM smoke test) [synth/smoke] — GREEN
Ran in 5s. Key numbers: cuda=True, pairs_per_s=7.443e+06, disk_free_gb=1668, llm=ok
Resume: `python -m melomane run --stage S0-doctor --corpus synth --profile smoke` · disk free 1668 GB

## 2026-10-05 20:25 UTC — S0-doctor (Environment, GPU, LLM smoke test) [synth/smoke] — GREEN
Ran in 5s. Key numbers: cuda=True, pairs_per_s=7.674e+06, disk_free_gb=1668, llm=ok
Resume: `python -m melomane run --stage S0-doctor --corpus synth --profile smoke` · disk free 1668 GB
