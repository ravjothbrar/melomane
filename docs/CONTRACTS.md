# Melomane — module & file contracts (v10)

Everything below is authoritative for people/agents adding modules. Python ≥ 3.10, Windows-safe (pathlib, no shell-isms, no symlinks,
`if __name__ == "__main__"` guards, no fork-only multiprocessing). Tests: pytest, deterministic seeds, **no network**, synthetic fixtures
(`melomane.synth.make_synth`, `tests/fakedump.py`). Run tests with `/tmp/mvenv/bin/python -m pytest tests/<file> -q` in the sandbox.

## Core objects
* `melomane.workspace.Workspace` — `.root`, `.data`, `.d(...)`, `.raw/.interim/.cache/.runs/.results/.reports/.release`, `.profile(name)`, `.config(name)`.
* `melomane.corpus.Corpus` / `FamilyBlock` / `write_corpus` / `read_corpus` — adapter output (see module docstring).
  `Corpus.tracks` index = raw_id 0..N-1; required cols `ext_id, artist_id(int64), artist, title`; optional `album, year, popularity, duration_ms, explicit, language, tags, has_lyrics`.
  `Corpus.sessions` = CSR (n_sessions × N) of 0/1 "documents".  `FamilyBlock(rows, X, columns, source)`.
* `melomane.pipeline` — `Stage`, `StageResult`, `register`, `Ctx` (ws, corpus, profile_name, profile, options; `.stage_dir(name)`, `.results_dir`, `.save_result`, `.llm()`).
* `melomane.llm.LLMClient` (via `ctx.llm()`); strict JSON schemas must have `additionalProperties: false` and list every property in `required`.
* Existing tested math: `cooc, sessions, compounds, walks, embed, torch_sgns, geometry, slerp, probes, ica_modes, curation, synth`.

## Run layout (`ws.runs/<corpus>/<profile>/`) — after S1 all ids are **cid** (canonical track index 0..V-1, row order of vocab)
```
S1/vocab.parquet            cols: cid, rep_raw, artist_id, artist, title, album, year, category, n_sessions, popularity, dev_split('dev'|'test'), region_tags('a|b'), tags
S1/consolidation_map.parquet raw_id, cid(-1 = removed/out of vocabulary), decision(rule|llm|human|none), group_id
S3/sessions_cid.npz         CSR sessions × V ;   S3/cooc_edges.npz   (rows, cols, npmi) symmetric ;  S3/cooc_edges_artistmasked.npz
S4/compound_edges.npz       CSR V × n_comp ;     S4/compound_meta.parquet (comp_id, family, size) ;  S4/features_manifest.json
S5/models/<name>/embeddings.npy   float32 (V × D) unit-agnostic (raw SGNS output), row = cid   (headline = seed 0)
S5/models/<name>/seeds/seed<k>.npy, train_log.jsonl, manifest.json
   names: cooc, composition, core, composition_acoustic, composition_lyric, core_acoustic, cooc_artistmasked, cooc_avg (raw-vocab Cooc averaged to cids),
          *_eqtok (equal token budget), core_partial<pct>
S6/<model>/ica.npz (directions n_f×D, sources V×n_f, stability), factor_poles.npy (n_f×D unit), modes.json, mode_poles.npy (n_modes×D unit),
            supervised_poles.json + supervised_poles.npy, mode_atlas.parquet
   modes.json = {"model":..., "modes":[{"mode_id","factor","bic_k","size","members":[cid...],"label","property","driver","coherence"}]}
   supervised_poles.json = {"poles":[{"name","kind"(category|region|era|dimension),"n_pos","n_neg"}]}  (row i of supervised_poles.npy)
```
Results (single source of truth for every reported number): `ws.results/<corpus>/<profile>/<name>.json` (strict JSON, NaN→null).
Release bundle per model (Epicure-like): `embeddings.safetensors`(key `embeddings`), `vocab.json`, `modes.json`, `factor_poles.npy`, `supervised_poles.json`+`.npy`, `config.json`, `README.md`.

## Rules
* Never redistribute/display lyrics or audio; never send lyrics to an LLM; no per-user data in any artefact.
* Every stochastic function takes `seed`. Every analysis that can leak by artist takes `groups=artist_id`.
* No secrets in code/logs.  Numbers in docs come from `results/` only.
* Do not edit files you were not asked to create/extend; if you need a change in a shared file, say so in your final report.
