# Data sources dossier

Legend — **V1** = read on the official page/record by me (2026-09-29); **V2** = corroborated by ≥ 2 independent pages; **U** = unverified, Claude Code must check in Stage 0.
My sandbox blocks all data hosts (egress policy), so nothing here was *downloaded* — reachability, checksums and schemas are Stage-0 checks, not assumptions.

## Track A (core) — Music4All-Onion v2 (Zenodo 15394646; v1 = 6609677)
| Fact | Value | Verification |
|---|---|---|
| Licence / access | CC BY 4.0, open (no request form), must cite Moscati et al., CIKM 2022, doi 10.1145/3511808.3557656 | V1 |
| Scale | 109,269 tracks · 119,140 users · 252,984,396 listening records (Last.fm) | V1 |
| Size | 49 files, ≈ 19.9 GB compressed (bz2 TSV; `processed_lyrics.tar.gz`; `README.md` 20.5 kB) | V1 (46 file names seen, 3 not listed → **U**) |
| Audio (track-level aggregates of 30 s clips) | `id_essentia` (407 MB; spectral/time/rhythm/tonal), `id_mfcc_stats`, `id_mfcc_bow`, `id_chroma_bow`, `id_blf_{spectral,spectralcontrast,deltaspectral,vardeltaspectral,logfluc,correlation}`, `id_compare_*` (openSMILE ComParE: audspec, f0, hnr, jitter, shimmer, mfcc, pcm, voice), `id_emobase_*`, `id_ivec{256,512,1024}`, `id_incp` | V1 |
| Learned audio embeddings / emotion | `id_jukebox` (3.9 GB), `id_maest`, `id_musicnn`, `id_gems` (GEMS emotion) | V1 |
| Lyrics | `id_bert` (623 MB), `id_lyrics_word2vec` (300-d), `id_lyrics_tf-idf`, `id_lyrics_sentiment_functionals`, `id_vad_bow`, `processed_lyrics.tar.gz` (22.8 MB — format **U**) | V1 |
| Tags / genres | `id_tags_dict`, `id_tags_tf-idf`, `id_genres_tf-idf` | V1 |
| Video/visual | `id_resnet`, `id_vgg19` | V1 (ignore for Melomane) |
| Listening | `userid_trackid_timestamp.tsv.bz2` (2.2 GB), `userid_trackid_count.tsv.bz2` (551 MB) | V1; timestamp unit & date range **U** |
| Track metadata (artist/title/album/year/language/popularity) | Original Music4All has them; whether the open Onion record ships them is **U** — the three unlisted files may be `id_information`, `id_metadata`, `id_lang` | **U — Stage-0 gate** |
| Original Music4All (30 s clips, raw lyrics, Spotify ids, year, language, 853 genres, 19,541 tags) | by email to contact4music4all@gmail.com; terms not stated on page | V1 |
| Overlap with LFM-2b | 56,512 tracks (≈ 51 %) | V1 (paper) |

**Fallbacks if artist/title are absent** (in order): (1) mine `README.md`/other files for metadata; (2) email the Music4All authors (draft in `docs/email_music4all.md`); (3) proceed with tag/genre-only LLM labelling
(weaker: no per-track world knowledge) and flag every affected result. Track B (MBID-native) is unaffected.

## Track B (scale-up) — MBID-native open stack
| Source | What | Access | Verification |
|---|---|---|---|
| MusicBrainz **Canonical Data Dumps** | one record per equivalent recording/release; zstd CSV; twice monthly; **CC0**; designed for offline (artist, title) → MBID mapping | free download (metabrainz.org/datasets/derived-dumps) | V1 |
| MusicBrainz PostgreSQL dumps | recordings, artist credits, ISRC, release dates, genres/tags | free (metabrainz.org/datasets/postgres-dumps) | V2 |
| **ListenBrainz** dumps | listens as JSONL inside `.tar.zst`, files `listens/YYYY/M.listens`; full dump twice a month + daily incrementals; up to the present | free (listenbrainz.org/data; ftp.musicbrainz.org/pub/musicbrainz/listenbrainz/) ; licence, `recording_mbid` mapping fields, total size **U** | V1 (format) / U |
| **AcousticBrainz** dumps (frozen 2022-06-23) | low-level JSON: 30 archives × ~1 M files (≈ 900–1000 GB compressed for the full set); **CSV feature dumps: 29,460,584 rows** — `lowlevel` (average_loudness, dynamic_complexity, mfcc_zero_mean), `rhythm` (bpm, bpm histogram, danceability, onset_rate), `tonal` (key, scale, tuning); 100 k sample dump | `data.metabrainz.org/pub/musicbrainz/acousticbrainz/dumps/…20220623/`; no signup | V1 (page) — throughput **U** |
| **LRCLIB** | free lyrics API (no key); a **database-dump page exists** (lrclib.net/db-dumps) but its content is a JS app I could not read | API + dump | API V2 / dump **U** |
| Discogs data dumps | releases/artists/labels/masters XML, **CC0**, monthly | data.discogs.com, no signup | V1 |
| **MLHD+** | 27 B Last.fm listens, MBID-cleaned | **non-commercial only** → use for private analysis at most; do not build published artefacts on it | V1 |

Track B plan: stream-filter AcousticBrainz to recordings that survive the ListenBrainz/MusicBrainz intersection; start with the CSV feature dumps (small) and pull low-level JSON only for candidates.

## Track C (cross-catalog check)
| Source | What | Verification |
|---|---|---|
| MTG-Jamendo | > 55,000 full tracks (320 kbps MP3), CC licences, 195 tags (87 genre / 40 instrument / 56 mood-theme), scripted download without account, optional AcousticBrainz features | V1 |
| FMA | 106,574 tracks / 16,341 artists / 161 genres; sizes small 7.2 GiB (8 k × 30 s), medium 22 GiB, large 93 GiB, full 879 GiB; metadata CC BY 4.0, audio under artist licences, research use; pre-computed features in `fma_metadata` | V1 (precomputed-feature file names **U**) |

## Rejected / demoted (do not build on these)
| Source | Why |
|---|---|
| Spotify Million Playlist Dataset (AIcrowd) | page now says the dataset is not available for download; contact Spotify Research (V1, screenshot from user) |
| Spotify Music Streaming Sessions Dataset | same channel, community reports it removed (Mar 2025) (V2) |
| "Playlist2vec" (Zenodo 5002584) | independent researcher's re-scrape of Spotify API data, self-licensed CC-BY; provenance/Spotify-ToS risk; no matching audio/lyrics. Optional private sanity check only |
| Melon Playlist Dataset | hosted via Kakao Arena, unavailable (V1) |
| Spotify audio-features / preview_url | deprecated for new apps (Nov 2024) |
| iTunes Search API as a *bulk* source | ≈ 20 requests/min/IP (V1). Allowed only for **client-side preview playback in the explorer** (visitor's own IP) |
| Essentia on native Windows | PyPI wheels exist for Linux (manylinux x86_64) and macOS only (checked on PyPI JSON) → optional; needs WSL2 |

## LLM & platform facts
| Item | Value | Verification |
|---|---|---|
| Model | OpenRouter `google/gemini-3.7-flash` (released 2026-08-13), 1 M context, $0.75 / $3.75 per M tokens (standard); `google/gemini-3.7-flash:batch` $0.375 / $1.875; a newer `gemini-3.8-flash` also exists | V1 |
| Endpoint | `POST https://openrouter.ai/api/v1/chat/completions`, `Authorization: Bearer`, optional `HTTP-Referer`, `X-Title`; `response_format` support varies by model → use `provider.require_parameters: true` | V1 (docs) |
| json_schema support, `reasoning` off switch, `usage.include` cost field, batch submission format, rate limits | **U — Stage-0 smoke test decides `configs/llm.yaml`** | U |
| PyTorch on RTX 5080 (Blackwell, sm_120) | needs a CUDA ≥ 12.8 build (`cu128`/newer wheels); early 2025–26 reports of failures on old builds; verify with `scripts/smoke_env.py` | V2 |
| Hardware (user) | Windows, 64 GB RAM, RTX 5080, C: 1.54 TB free of 1.86 TB | user-stated |

## Reference implementations to study, not copy
`Kaikaku/epicure-core` (`epicure.py` loader/SLERP), `KAIKAKU-AI/epicure-mcp` (read-only MCP server), `Kaikaku/epicure-explorer` (HF Space). Licence CC BY 4.0 for the derived tables/embeddings; loader-code licence **U**.
