# Data and derived-artefact licence

**Code** is MIT (see `LICENSE`).

**Derived artefacts** (embeddings, mode atlases, probe tables, direction tables, figures, the paper) are offered under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) **subject to the upstream terms below**.

## Required attribution

* **Music4All-Onion** (CC BY 4.0): Moscati, Parada-Cabaleiro, Deldjoo, Zangerle, Schedl, "Music4All-Onion – A Large-Scale Multi-faceted
  Content-Centric Music Recommendation Dataset", CIKM 2022, doi:10.1145/3511808.3557656. Zenodo records 15394646 / 6609677.
  The listening events in that dataset are derived from Last.fm.
* Keep this attribution in anything you redistribute.

## What is *not* in this repository, and must not be added

* audio, lyrics (raw or processed), per-user listening data, or anything that would let a user be identified;
* MLHD+ (non-commercial) or any Spotify-API data.

## Artist and title strings

The Zenodo Onion records contain **no artist or title** fields. The strings shown in the explorer and stored in the bundles come from the
Hugging Face dataset [`seungheondoh/enrich-music4all`](https://huggingface.co/datasets/seungheondoh/enrich-music4all) (identified by matching
its columns and its 108,363 rows; see `DECISIONS.md` 18). That dataset's card **declares no licence**, and its metadata derive from Music4All
(Santana et al., 2020), whose terms are not stated either. Only `track_id`, `title`, `artist_name` and `release` are used; its LLM-generated
captions and tag lists are not.

Until the owner confirms that these names may be republished (with attribution to Music4All and `seungheondoh/enrich-music4all`), the
explorer payload and bundles are **not cleared for public release**; the numeric embeddings alone are.
