# Reproducibility manifest

- **Date:** 2026-10-10 12:31 UTC
- **Corpus / profile:** `onion` / `standard`
- **Code version (git):** `41c67b9`
- **Host:** Windows-11-10.0.26200-SP0 (32 CPUs)
- **Python / PyTorch:** 3.12.13 / 2.11.0+cu128
- **GPU:** NVIDIA GeForce RTX 5080 Laptop GPU (compute capability 12.0)
- **Training:** 4 walks/node × 6 epochs, Core `ii_repeat` 10, 3 seeds for the main models, dimension 300
- **Vocabulary:** 53,309 canonical tracks, 9,872 named artists
- **LLM:** `google/gemini-3.7-flash` via OpenRouter (curation, labelling, mode naming only; the embeddings are LLM-free)
- **LLM spend (recorded in the cache):** $12.48 of $30 cap; calls that were paid for but produced no usable answer are not recorded (see DECISIONS.md 17)
- **Pre-registration:** `analysis_plan.lock.json` with 1 recorded amendment(s)
- **Seeds:** per-stage seeds derived from (run id, stage, name); ICA uses multi-seed Hungarian matching
