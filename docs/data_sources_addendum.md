# Data sources addendum — Tracks B and C (adapter-driven verification)

Written 2026-09-29 alongside `src/melomane/adapters/{mbid_stack,jamendo,fma}.py`. It extends `docs/data_sources.md` and does **not** replace it.
Where the two disagree, this file is newer and says so explicitly (section 6).

## 0. How things were checked, and how far to trust it

* Tools: WebFetch/WebSearch only (documentation pages, directory listings, small GitHub raw files). **No dataset file was downloaded, no bulk host was
  probed, and no block was bypassed.** WebFetch returns a *small-model summary* of the page, not raw bytes. Anything below marked **V** was therefore read
  through that summary; where a summary looked inconsistent I say so (section 6). Sizes come from Apache autoindex listings, which round to whole
  K/M/G (powers of 1024), so treat them as +/- 5 %.
* Live API traffic: exactly two GET requests to `lrclib.net/api` (one `/get`, one `/search?q=`) to read the response *schema*; I recorded field names only,
  not lyrics. No other API was called.
* Legend: **V** = read on an official page/listing on 2026-09-29. **V2** = corroborated by a second independent page. **U** = could not verify; the code
  is written defensively and Stage 0 must check it on the real file.
* Nothing here is a claim about numerical behaviour of real files. Every parser is tested on synthetic fixtures built from the documented layouts only
  (`tests/test_mbid_stack.py`, `tests/test_track_c.py`).

## 1. Track B — what was verified

### 1.1 ListenBrainz (behaviour)
| Fact | Value | Status |
|---|---|---|
| Full-dump location | `https://data.metabrainz.org/pub/musicbrainz/listenbrainz/fullexport/` (dirs `listenbrainz-dump-2647-20260901-000002-full`, `…-2663-20260915-000002-full`) | V |
| Files in the 2663 dump | `listenbrainz-listens-dump-2663-20260915-000002-full.tar.zst` **229G** (listed 2026-09-16); `…spark-dump….tar` 220G; `…statistics-dump-20260915-000002.tar.zst` 20G; each with `.md5` + `.sha256` | V |
| Sample dump | `…/listenbrainz/sample/listenbrainz-sample-20250610-094311-full/listenbrainz-sample-dump-20250610-094311.tar.zst`, 235M | V |
| Other dirs | `incremental/` (updated daily), `spark/`, `labs/`, `mlhd/` | V (names only) |
| Archive layout | `listens/YYYY/M.listens` inside the listens dump, **one JSON document per line** in the standard listens format | V |
| Cadence | full dumps twice monthly; incremental dumps exist (docs say "twice weekly" in one place and "daily" in another). Deleted listens are **not** propagated by incrementals (documented caveat) | V |
| Listen JSON | `listened_at` (unix seconds, example 1443521965), `track_metadata.{artist_name, track_name, release_name, additional_info}`; `additional_info.{recording_mbid, artist_mbids, release_mbid, release_group_mbid, duration_ms, ...}` | V |
| `mbid_mapping` | described as server-added, *read-only*, present in **API responses**: `recording_mbid`, `recording_name`, `release_mbid`, `artist_mbids`, `artists[{artist_mbid, artist_credit_name, join_phrase}]`, `caa_id` | V (API docs) |
| Mapping latency | failed matches retried on a doubling schedule (1 d up to 30 d); unmatched listens are remapped continuously, "potentially taking weeks" | V |
| **Whether dump lines contain `mbid_mapping` and `user_name`** | the dump docs show **no sample line**. Parser tries `mbid_mapping.recording_mbid`, then `additional_info.recording_mbid`; user from `user_name`, else `user_id` | **U** |
| **Licence of the listens dump** | not stated on any page I could read (`listenbrainz.org/data` and `/about` returned only meta tags; MetaBrainz datasets page says personal use is always free and asks commercial users to support the project). Read the `COPYING`/README inside the dump | **U** |
| Uncompressed size / listen count | not found. Never extract the tar; the adapter streams it | **U** |

### 1.2 MusicBrainz canonical data (metadata, redirects, offline lookup)
| Fact | Value | Status |
|---|---|---|
| Location | `https://data.metabrainz.org/pub/musicbrainz/canonical_data/` -> `musicbrainz-canonical-dump-20260917-080002/…tar.zst` **2G** (+ `.md5`, `.sha256`); older dir `…20260903-080002` | V |
| Licence | **CC0** | V |
| Cadence | "twice a month, on the 1st and 15th" per the page (listing shows the 3rd and 17th) | V |
| Access | the dataset page says access **requires sign-up** (`metabrainz.org/datasets/signup`); I could not tell whether the file server enforces it. Sign up (free) before automating downloads | V (page) / U (enforcement) |
| Files | `canonical_release_redirect.csv` (release_mbid, canonical_release_mbid, release_group_mbid); `canonical_recording_redirect.csv` (recording_mbid, canonical_recording_mbid, canonical_release_mbid); `canonical_musicbrainz_data.csv` (**id, artist_credit_id, artist_mbids, artist_credit_name, release_mbid, release_name, recording_mbid, recording_name, combined_lookup, score, year**) | V2 (wiki + doc page; the doc page says "10 columns", the wiki lists 11 incl. `year` -> reader accepts both) |
| `combined_lookup` | artist_credit_name + recording_name, punctuation and superfluous whitespace removed, non-ASCII converted to ASCII; `score` gives priority ordering for the Typesense index | V (wording) |
| Scale | "approximately 23 million rows" for the metadata CSV | V (canonical-data-example README) |
| Member names inside the `.tar.zst`; `artist_mbids` encoding (Postgres `{a,b}` / JSON / comma list); **direction** of `score`; whether `year` is the *first-release* year | reader matches by base name, parses all three encodings, `higher_score_wins=True` is a parameter, `first_release_year` is documented as "canonical release year" | **U** |
| ISRCs | **not** in the canonical dump. PostgreSQL `isrc` table columns are (id, recording, isrc, edits_pending, created) (V, `CreateTables.sql`); the `recording` table's column order (id, gid, …) is from memory | isrc V / recording U |
| Other MusicBrainz dumps (2026-09-26) | `fullexport/20260926-002121/`: `mbdump.tar.bz2` 7G, `mbdump-derived.tar.bz2` 495M, `mbdump-edit.tar.bz2` 16G (not needed). `data/json-dumps/20260926-001002/`: `recording.tar.xz` **32M**, `release.tar.xz` 21G, `artist.tar.xz` 2G | V |
| Licences | PostgreSQL dumps: core data CC0; supplementary/derived data CC BY-NC-SA 3.0 | V |
| JSON `recording.tar.xz` line format | not inspected. 32 MB is implausibly small for all recordings, so check what it actually contains before using ISRC/`first-release-date` from it | **U** |

### 1.3 AcousticBrainz (frozen 2022-06-23)
| Fact | Value | Status |
|---|---|---|
| Location | `https://data.metabrainz.org/pub/musicbrainz/acousticbrainz/dumps/` with `acousticbrainz-{lowlevel-json,highlevel-json,lowlevel-features,sample-json}-20220623/` (published 2022-07-06) | V |
| CSV feature dumps | `acousticbrainz-lowlevel-features-20220623-{lowlevel 1G, rhythm 1G, tonal 843M}.tar.zst` + `sha256sums`; **29,460,584 rows** each | V |
| CSV columns | `mbid, submission_offset` + lowlevel: `average_loudness, dynamic_complexity, mfcc_zero_mean`; rhythm: `bpm, bpm_histogram_first_peak_bpm_{mean,median}, bpm_histogram_second_peak_bpm_{mean,median}, danceability, onset_rate`; tonal: `key_key, key_scale, tuning_frequency, tuning_equal_tempered_deviation` | V (acousticbrainz-server `db/dump.py` + download page) |
| Low-level JSON | 30 archives `acousticbrainz-lowlevel-json-20220623-{0..29}.tar.zst`, listed sizes 19-21G each (29 x ~20G + the last 9G) = **~589 GB total** | V |
| JSON layout | `<archive_dir>/lowlevel/<mbid[0:2]>/<mbid[2:3]>/<mbid>-<submission_offset>.json` | V (`dump.py`) |
| Sample dumps | `acousticbrainz-lowlevel-sample-json-20220623-0.tar.zst` **2G** (100 000 items), highlevel sample 85M | V |
| Low-level statistics | `mean, median, min, max, var, dmean, dmean2, dvar, dvar2` (no `stdev`); e.g. `lowlevel.mfcc.{mean,cov,icov}`, `tonal.hpcp.mean` (36 values), `rhythm.bpm` | V |
| Licence | **CC0** ("The data is CC0", MetaBrainz blog 2022-02-16); the download page itself states no licence | V (blog) |
| Member names inside the CSV tar archives | reader takes any `*.csv` member | **U** |
| Extractor version mix | mostly older extractor builds; whether `loudness_ebu128`, `gfcc`, `melbands128` exist in these documents | **U** |

### 1.4 LRCLIB
| Fact | Value | Status |
|---|---|---|
| API | `GET https://lrclib.net/api/get?artist_name&track_name[&album_name&duration]` and `GET /api/search?q=` return JSON with `id, name, trackName, artistName, albumName, duration, instrumental, plainLyrics, syncedLyrics` (+ `hasWordSync`, `lyricsfile` in 2026) | V (live schema read) |
| Dump | author statement (Hacker News, 2024): full DB dumps "uploaded regularly and publicly" as **SQLite3** at `lrclib.net/db-dumps`, ~3 M lyrics at the time, "do anything you want with the database" | V2 (third-party post) |
| Dump page, file names, size, compression, schema, `/api/get-cached` | `lrclib.net/db-dumps` and `/docs` are JavaScript apps my fetch could not render | **U** |
| Rate limit / User-Agent policy | not found. The adapter therefore enforces its own **<= 1 request/s** and refuses generic User-Agents | **U** |

## 2. Track C — what was verified

### 2.1 MTG-Jamendo
* TSV header `TRACK_ID ARTIST_ID ALBUM_ID PATH DURATION TAGS`; tags occupy columns `row[5:]`, format `<category>---<name>` with categories `genre`, `instrument`, `mood/theme`; example row `track_0000948 artist_000087 album_000149 48/948.mp3 212.7 mood/theme---background` (V, `commons.py` + `data/autotagging_moodtheme.tsv`).
* Files: `raw.tsv` 56,639; `raw_30s.tsv` 55,701; `raw_30s_cleantags_50artists.tsv` 55,609; `autotagging.tsv` (baseline; a repo alias of the 50-artists file); `autotagging_{genre 95, instrument 41, moodtheme 59, top50tags}.tsv` (V). **Tag counts:** 195 = 95 + 41 + 59 tags in the per-category files, while the README also cites 87 / 40 / 56 (= 183) as available in the official splits; both figures are in the README, they count different things.
* Download types `audio` (508 GB), `audio-low` (156 GB), `melspecs` (229 GB) and `acousticbrainz` (V). Mirrors: MTG Spain and MTG Finland (freesound CDN), DOI 10.5281/zenodo.3826813.
* **Licence: metadata CC BY-NC-SA 4.0, code Apache-2.0, audio individual CC licences; other use needs written permission from Jamendo S.A.** (V). This is **not** CC-BY-compatible (see section 5).
* U: the TSVs hold ids only (no artist/track names), so `artist`/`title` are the ids (`attrs['has_names']=False`); the layout, extractor version and statistic names of the `acousticbrainz` JSON archives (project page gives no detail, and the "acousticbrainz" type is not listed among the concrete downloads there).

### 2.2 FMA
* `fma_metadata.zip` **342 MiB** = `tracks.csv` (106,574 tracks), `genres.csv` (163 genres), `features.csv` (librosa), `echonest.csv` (Echonest features for **13,129** tracks) (V, README). Sizes: small 7.2 GiB / medium 22 GiB / large 93 GiB / full 879 GiB (V, README; the 2017 paper says 7.4/23/98/917 GiB).
* Loader behaviour (V, `utils.py`): `tracks.csv` `header=[0,1], index_col=0`; `features.csv` and `echonest.csv` `header=[0,1,2]`; genres via `index_col=0`; `tags/genres/genres_all` are Python-literal lists; `set.subset` in {small, medium, large}, `set.split` in {training, validation, test}.
* `features.csv`: **518** librosa 0.5.0 features = 3 chroma sets x 84 + tonnetz 42 + mfcc 140 + spectral contrast 49 + 5 scalar features x 7; seven statistics (mean, std, skew, kurtosis, median, min, max); 2048-sample windows, hop 512 (V, paper + UCI + `features.py`).
* `echonest.csv`: top level `echonest`; second level `audio_features`, `metadata`, `ranks`, `social_features`, `temporal_features`; audio_features include acousticness, danceability, energy, instrumentalness, liveness, speechiness, tempo, valence (V, `creation.ipynb`).
* Licences: metadata CC BY 4.0, code MIT, audio per-artist CC licences (some are NC/ND) (V).
* U: exact `genres.csv` header spelling (`genre_id, #tracks, parent, title, top_level` assumed), exact `tracks.csv` column list (the two notebooks disagree on `language` vs `language_code`; the reader accepts either), all echonest column counts. The WebFetch summaries of the notebooks printed nonsensical shapes ("558", "1,881", "1391 features") - I discarded them and used the paper/UCI figure 518.

## 3. Feature alignment (PLAN S11 / section 13 item 4) — what can and cannot be claimed

* **AcousticBrainz vs Essentia/Onion names.** In the repository's `essentia_descriptor_names.json` (provenance not verified: it is a file already in the repo) all 52 statistic keys end in `.stdev`; AcousticBrainz stores `var`. The adapter derives `stdev = sqrt(var)`. At *descriptor-name level* every acoustic family in `mbid_stack.AB_FAMILY_SPECS` finds all of its descriptors in that list (`align_report(ab_columns, onion_names)` -> all five families `aligned`). **This is a naming statement only.** It does not show numerical comparability: the Onion list also has descriptors that AcousticBrainz's older extractor builds may lack (`lowlevel.gfcc`, `lowlevel.melbands128`, `lowlevel.erbbands*`, `loudness_ebu128.*`), and extractor version/profile differences are U. Only Stage 0 on real Onion and AcousticBrainz files can settle PLAN section 13.4.
* **Onion `id_*` files that have no AcousticBrainz analogue:** `id_blf_*` (block-level features), `id_chroma_bow`, `id_compare_*`/`id_emobase_*` (openSMILE), `id_mfcc_stats` (librosa-style, not Essentia). Track B's families are therefore *Essentia-only variants* of PLAN 4.4's families, not the identical definition. Freeze that difference in `analysis_plan.lock.json`.
* **FMA:** librosa 0.5.0 + Echonest, not Essentia. `fma.align_report` returns `can_use_identical_definition = False`, every family `conceptual_only` (mfcc, spectral_*, zcr, rmse, chroma, tonnetz) or `not_alignable` (`rhythm_groove` has no librosa feature; Echonest tempo/danceability are a proprietary, different definition). Per PLAN S11 the pipeline must state this and run the cross-catalog transfer inside Track B (or refit the vocabulary on FMA features).
* **Jamendo:** if the `acousticbrainz`-type JSON files are Essentia music-extractor output, `jamendo.feature_blocks` builds the same five families; `jamendo.align_report(descriptor_names_seen)` states which are really present. Melspectrograms are learned-model inputs and never alignable.

## 4. Recommended Track B download order (native Windows, 1.5 TB free)

Sizes are the listed values above; plan with `plan_downloads()` and refresh with `probe_sizes()` (HEAD only). Dump directory names rotate: refresh with `parse_apache_index` + `pick_latest`.

| Step | What | Size | Why here |
|---|---|---|---|
| 0 | ListenBrainz **sample** dump | 235 MB | Stage 0: read real dump lines, settle the U rows in 1.1 and PLAN 13.7 |
| 0 | Canonical dump | 2 GB | metadata + redirects; settle `score`, `year`, `artist_mbids` |
| 0 | AcousticBrainz CSV dumps (3 files) | ~2.8 GB | coarse features for all 29.5 M submissions; settle member names |
| 0 | AcousticBrainz **sample** low-level JSON | 2 GB | check descriptor names/versions against Onion (PLAN 13.4) |
| 1 | ListenBrainz **full listens dump** (2663-20260915) | 229 GB | stream, do not extract; `year_range`/`min_ts`/`mbid_filter` limit what is kept. Shards store 32 bytes/listen (listen count U, so re-estimate from the sample) |
| 2 | AcousticBrainz low-level JSON, archives 0-29 | 589 GB total | **Rolling**: download one ~20 GB archive, run `scan_lowlevel_archives`, delete the archive (keep the shard). Peak extra disk ~40 GB instead of 589 GB. Needs the candidate list from steps 0-1 first |
| 3 | LRCLIB SQLite dump | U | prefer over the API for > ~50 k tracks; check size before downloading |
| opt | MusicBrainz `recording.tar.xz` (32 MB) or `mbdump.tar.bz2` (7 GB) | | only for ISRCs; not needed for the core pipeline |

Disk: 229 (listens) + ~5 (canonical/CSV/samples) + 20-40 (rolling AB) + shards (tens of GB, U) is roughly 300-350 GB peak, or ~900 GB if all 30 archives are kept. Both fit in 1.5 TB; delete the listens tar only after the shards are verified (`sha256` files are published beside every dump; `verify_sha256`). Track C: FMA `fma_metadata.zip` 342 MiB (audio not needed for the metadata-only adapter); Jamendo TSVs are tiny, `acousticbrainz` JSON size U.

## 5. Terms, rate limits, licences — what to obey

* **MusicBrainz web service** (only if you ever call it; the adapters are offline): on average **1 request/s per IP**, 300/s global, and a descriptive `User-Agent` `Application/<version> ( contact )`; blank, "Java", "Python-urllib" style agents are throttled (V, rate-limiting page). Prefer dumps.
* **LRCLIB:** descriptive User-Agent, cache everything, <= 1 req/s (self-imposed, policy U), prefer the SQLite dump. Lyrics are third-party copyrighted works: the pipeline reduces them to presence + counts + an aggregate embedding **in memory**; `LrclibCache`/`LrclibDump` never persist text (tests scan the SQLite cache, shards and corpus files for a planted lyric token). The author's "do anything you want with the database" remark is not a licence for the lyrics themselves.
* **ListenBrainz:** licence U (see 1.1). Usernames are public handles; the adapter HMAC-hashes them to 63-bit ints at parse time with a salt kept in `data/private/user_salt.bin` (outside every artefact; changing it is refused on resume). Hashed ids are still per-user data: `sessions_meta.user_id` should be dropped from anything released (CONTRACTS: no per-user data in artefacts). The full dump is authoritative; incrementals do not carry deletions.
* **MusicBrainz canonical + core PostgreSQL data, AcousticBrainz:** CC0 (derived MusicBrainz tables CC BY-NC-SA 3.0).
* **MLHD+:** non-commercial only; not used.
* **LICENCE FLAG (CLAUDE.md section 2 stop-reason 5 / section 5):** MTG-Jamendo *metadata* is **CC BY-NC-SA 4.0** and Jamendo commercial use needs written permission. That is not "CC BY-compatible", so Jamendo-derived tables must not go into a public release bundle without Ravjoth's decision. FMA metadata (CC BY 4.0) is fine; FMA *audio* licences vary per artist.

## 6. Corrections to `docs/data_sources.md`

1. AcousticBrainz full low-level JSON is listed at ~**589 GB** (29 x ~20G + 9G), not "900-1000 GB".
2. ListenBrainz full listens dump is **229 GB** (2026-09-16 listing); the existing note had no figure. A 235 MB sample dump exists (2025-06-10).
3. MusicBrainz canonical dump is **2 GB**; the metadata CSV is ~23 M rows; column `year` exists.
4. Jamendo: the "195 tags" and "87/40/56" figures count different things (see 2.1); metadata licence is **NC-SA**, which the existing table did not flag.
5. FMA: features.csv is 518 columns; echonest.csv covers 13,129 tracks.
6. LRCLIB dump: SQLite, confirmed by a second (third-party) source, but page contents still unread.

## 7. Not verified (plainly)

Real listens-dump line schema (mapping/user keys) and licence; listen counts and uncompressed sizes; canonical CSV inner file names, `artist_mbids` encoding, `score` direction, `year` semantics; AcousticBrainz CSV inner names and extractor-version mix; LRCLIB rate limit, User-Agent policy, dump size/name/schema and `/api/get-cached`; Jamendo `acousticbrainz` archive layout and names in the TSVs; FMA `genres.csv` header and `echonest.csv` column counts; whether the MetaBrainz file server enforces the sign-up; download throughput; any numerical comparability between Onion, AcousticBrainz, Jamendo and FMA features.
