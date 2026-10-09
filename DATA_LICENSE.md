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

## Open question: artist and title strings

The Zenodo Onion record contains **no artist or title** fields. The artist/title/album strings shown in the explorer and stored in
the bundles come from a separate track-metadata table whose origin and licence are **not yet confirmed** (its SHA-256 is recorded
in `data/raw/onion/METADATA_PROVENANCE.json`). Until the owner confirms where it came from and that republishing the names is
permitted, treat the explorer payload and bundles as **not cleared for public release** (the numeric embeddings alone are).
