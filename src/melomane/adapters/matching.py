"""Cross-corpus track matching (partial-hub regime: behaviour corpus x feature corpus).

Typical use: ``target`` = a big behavioural corpus without audio/lyrics (Playlist2vec, ~3M tracks), ``source`` = a corpus
with measured features (Music4All-Onion, ~110k tracks).  ``match_tracks(target.tracks, source.tracks)`` finds the tracks
of the two catalogues that are the same recording *as far as the deterministic curation policy can tell*
(``melomane.curation.canonical_key``: remaster / mono / edit / feat-credit noise removed, remix / cover / live kept
apart at level 'strict'); ``enrich`` then transplants the source's feature blocks onto the target's raw ids for matched
tracks only (unmatched tracks are simply absent from the blocks = partial hubs), and ``coverage_report`` says how much
of the target (by track and by listening mass) ends up with compounds.

Matching passes (each pass only sees rows still unmatched):
  1. ``exact``                 identical (normalised artist, normalised title).
  2. ``exact_primary_artist``  identical title and identical *primary* artist (credit cut at feat./ft./with/x/vs/,/;///+
                               so 'Drake feat. Rihanna' == 'Drake').  Score 0.95 (heuristic constant, documented).
  3. ``fuzzy`` (optional)      within one primary-artist block, title token Jaccard / difflib ratio >= threshold; a guard
                               refuses pairs whose differing tokens are version words (remix, live, ...) or digits.
Ambiguity: when a canonical key has several rows on either side, every ``a`` row is paired with the ``b`` row of highest
``popularity`` (ties: lowest raw id) and flagged ``ambiguous=True``.  Fuzzy pairs are flagged ambiguous when the runner-up
candidate scores within 0.02 of the winner.

Scaling: normalisation runs once per *unique* artist string (and, after discarding rows whose artist cannot occur in the
other catalogue, once per unique title) - no per-row Python at 3M rows; all joins are pandas merges; the only per-pair
Python loop is difflib on the pruned fuzzy candidate set.  UNVERIFIED on real data: throughput at 3M x 110k rows and the
match rate between Playlist2vec and Onion (Onion's open release may lack artist/title altogether - see ``onion.py``; then
``match_tracks`` returns an empty frame and warns).
"""
from __future__ import annotations

import difflib
import re

import numpy as np
import pandas as pd
import scipy.sparse as sp

from ..corpus import Corpus, FamilyBlock
from ..curation import normalize_artist, normalize_title
from ..runtime import get_logger

log = get_logger("melomane.adapters.matching")

MATCH_COLUMNS = ["a_raw", "b_raw", "method", "score", "ambiguous"]
_PRIMARY_SPLIT = re.compile(r"\s*(?:\bfeat(?:uring)?\b\.?|\bft\b\.?|\bwith\b|\bvs\b\.?|\bx\b|,|;|/|\+)\s*", re.I)
_VERSION_WORDS = frozenset("""remix mix edit live acoustic instrumental cover karaoke demo version unplugged session tribute vip
bootleg slowed sped nightcore reprise reworked rework dub radio extended original mono stereo pt part""".split())
_SKIP_FUZZY_ARTISTS = frozenset({"", "various artists", "various", "unknown", "unknown artist", "va", "traditional", "soundtrack"})
_PRIMARY_SCORE = 0.95


# ----------------------------------------------------------------------------------------------------------------------
# normalisation
# ----------------------------------------------------------------------------------------------------------------------
def primary_artist_key(name: str) -> str:
    """Normalised first credited artist ('Drake feat. Rihanna' -> 'drake'; 'Hall & Oates' stays whole)."""
    first = _PRIMARY_SPLIT.split(name or "", maxsplit=1)[0]
    return normalize_artist(first)


def _map_unique(s: pd.Series, fn) -> pd.Series:
    """Apply ``fn`` once per distinct string (vectorised gather back to rows)."""
    s = s.fillna("").astype(str)
    codes, uniq = pd.factorize(s, sort=False)
    out = np.empty(len(uniq), dtype=object)
    for i, u in enumerate(uniq):
        out[i] = fn(u)
    return pd.Series(out[codes] if len(uniq) else np.array([], dtype=object), index=s.index)


def _prepare(df: pd.DataFrame, side: str) -> pd.DataFrame:
    for c in ("artist", "title"):
        if c not in df.columns:
            raise ValueError(f"{side}: tracks frame needs an {c!r} column")
    pop = pd.to_numeric(df["popularity"], errors="coerce").fillna(0).to_numpy(np.float64) if "popularity" in df.columns \
        else np.zeros(len(df))
    return pd.DataFrame({"raw": df.index.to_numpy(np.int64), "artist": df["artist"].to_numpy(dtype=object),
                         "title": df["title"].to_numpy(dtype=object), "pop": pop})


def _add_artist_keys(t: pd.DataFrame) -> pd.DataFrame:
    t["ak"] = _map_unique(t["artist"], normalize_artist).to_numpy()
    t["pk"] = _map_unique(t["artist"], primary_artist_key).to_numpy()
    return t


def _add_title_key(t: pd.DataFrame, level: str) -> pd.DataFrame:
    t["tk"] = _map_unique(t["title"], lambda x: normalize_title(x, level)).to_numpy()
    return t


# ----------------------------------------------------------------------------------------------------------------------
# exact passes
# ----------------------------------------------------------------------------------------------------------------------
def _exact_pass(a: pd.DataFrame, b: pd.DataFrame, akey: str, method: str, score: float) -> pd.DataFrame:
    a = a[(a[akey] != "") & (a["tk"] != "")]
    b = b[(b[akey] != "") & (b["tk"] != "")]
    if a.empty or b.empty:
        return pd.DataFrame(columns=MATCH_COLUMNS)
    ka = a[akey].to_numpy(object) + "\x1f" + a["tk"].to_numpy(object)
    kb = b[akey].to_numpy(object) + "\x1f" + b["tk"].to_numpy(object)
    A = pd.DataFrame({"key": ka, "a_raw": a["raw"].to_numpy()})
    B = pd.DataFrame({"key": kb, "b_raw": b["raw"].to_numpy(), "pop": b["pop"].to_numpy()})
    n_a, n_b = A["key"].value_counts(), B["key"].value_counts()
    best = B.sort_values(["key", "pop", "b_raw"], ascending=[True, False, True]).drop_duplicates("key")
    m = A.merge(best[["key", "b_raw"]], on="key", how="inner")
    if m.empty:
        return pd.DataFrame(columns=MATCH_COLUMNS)
    m["ambiguous"] = (m["key"].map(n_a).to_numpy() > 1) | (m["key"].map(n_b).to_numpy() > 1)
    m["method"], m["score"] = method, np.float32(score)
    return m[MATCH_COLUMNS]


# ----------------------------------------------------------------------------------------------------------------------
# fuzzy pass
# ----------------------------------------------------------------------------------------------------------------------
def _guard_ok(ta: str, tb: str) -> bool:
    diff = set(ta.split()) ^ set(tb.split())
    return not any(w in _VERSION_WORDS or any(ch.isdigit() for ch in w) for w in diff)


def _fuzzy_pass(a: pd.DataFrame, b: pd.DataFrame, threshold: float, max_pairs: int, max_block_pairs: int) -> pd.DataFrame:
    a = a[(a["pk"] != "") & (a["tk"] != "") & ~a["pk"].isin(_SKIP_FUZZY_ARTISTS)]
    b = b[(b["pk"] != "") & (b["tk"] != "") & ~b["pk"].isin(_SKIP_FUZZY_ARTISTS)]
    empty = pd.DataFrame(columns=MATCH_COLUMNS)
    if a.empty or b.empty:
        return empty
    # blocking: same primary artist; drop absurdly large blocks (label compilations) to bound the pair count
    prod = a["pk"].value_counts().mul(b["pk"].value_counts(), fill_value=0)
    keep_art = prod[(prod > 0) & (prod <= max_block_pairs)].index
    dropped = int(((prod > max_block_pairs)).sum())
    if dropped:
        log.warning("fuzzy: %d artist blocks exceed %d pairs and are skipped", dropped, max_block_pairs)
    a, b = a[a["pk"].isin(keep_art)], b[b["pk"].isin(keep_art)]
    if a.empty or b.empty:
        return empty
    A = pd.DataFrame({"pk": a["pk"].to_numpy(), "a_raw": a["raw"].to_numpy(), "ta": a["tk"].to_numpy()})
    B = pd.DataFrame({"pk": b["pk"].to_numpy(), "b_raw": b["raw"].to_numpy(), "tb": b["tk"].to_numpy(), "pop": b["pop"].to_numpy()})
    A["na"] = A["ta"].str.count(" ").to_numpy() + 1
    B["nb"] = B["tb"].str.count(" ").to_numpy() + 1
    # candidate pairs = all (a, b) pairs inside one artist block (bounded above by max_block_pairs per artist); the number
    # of shared title tokens comes from a token-level merge and drives Jaccard without any Python loop
    def tokens(df, col, raw):
        t = df[["pk", raw]].copy()
        t["tok"] = df[col].str.split(" ")
        return t.explode("tok").drop_duplicates()
    shared = tokens(A, "ta", "a_raw").merge(tokens(B, "tb", "b_raw"), on=["pk", "tok"]) \
        .groupby(["a_raw", "b_raw"], sort=False).size().rename("shared").reset_index()
    pairs = A[["pk", "a_raw"]].merge(B[["pk", "b_raw"]], on="pk")[["a_raw", "b_raw"]]
    pairs = pairs.merge(shared, on=["a_raw", "b_raw"], how="left")
    pairs["shared"] = pairs["shared"].fillna(0).astype(np.int32)
    if len(pairs) > max_pairs:
        log.warning("fuzzy: %d candidate pairs > max_pairs=%d; keeping those sharing most tokens", len(pairs), max_pairs)
        pairs = pairs.nlargest(max_pairs, "shared")
    pairs = pairs.merge(A[["a_raw", "ta", "na"]], on="a_raw").merge(B[["b_raw", "tb", "nb", "pop"]], on="b_raw")
    pairs["jac"] = pairs["shared"] / (pairs["na"] + pairs["nb"] - pairs["shared"])
    la, lb = pairs["ta"].str.len().to_numpy(), pairs["tb"].str.len().to_numpy()
    ub = 2.0 * np.minimum(la, lb) / np.maximum(la + lb, 1)                    # upper bound of difflib ratio
    need_ratio = (pairs["jac"].to_numpy() < threshold) & (ub >= threshold)
    ratio = np.zeros(len(pairs))
    idx = np.flatnonzero(need_ratio)
    ta_v, tb_v = pairs["ta"].to_numpy(object), pairs["tb"].to_numpy(object)
    for i in idx:                                                            # pruned candidate set only
        sm = difflib.SequenceMatcher(None, ta_v[i], tb_v[i], autojunk=False)
        if sm.real_quick_ratio() >= threshold and sm.quick_ratio() >= threshold:
            ratio[i] = sm.ratio()
    pairs["score"] = np.maximum(pairs["jac"].to_numpy(), ratio)
    pairs = pairs[pairs["score"] >= threshold]
    if pairs.empty:
        return empty
    ok = np.fromiter((_guard_ok(x, y) for x, y in zip(pairs["ta"].to_numpy(object), pairs["tb"].to_numpy(object))),
                     dtype=bool, count=len(pairs))
    pairs = pairs[ok].sort_values(["a_raw", "score", "pop", "b_raw"], ascending=[True, False, False, True])
    if pairs.empty:
        return empty
    rank = pairs.groupby("a_raw", sort=False).cumcount().to_numpy()
    top = pairs[rank == 0].copy()
    runner = pd.Series(pairs["score"].to_numpy()[rank == 1], index=pairs["a_raw"].to_numpy()[rank == 1])
    gap = top["score"] - top["a_raw"].map(runner)
    top["ambiguous"] = gap.le(0.02).fillna(False).to_numpy()
    top["method"], top["score"] = "fuzzy", top["score"].astype(np.float32)
    return top[MATCH_COLUMNS]


# ----------------------------------------------------------------------------------------------------------------------
# public API
# ----------------------------------------------------------------------------------------------------------------------
def match_counts(matches: pd.DataFrame) -> dict:
    """Per-method row counts, ambiguous count and number of distinct a / b tracks."""
    if matches.empty:
        return {"total": 0, "by_method": {}, "ambiguous": 0, "n_a": 0, "n_b": 0}
    return {"total": int(len(matches)), "by_method": {k: int(v) for k, v in matches["method"].value_counts().items()},
            "ambiguous": int(matches["ambiguous"].sum()), "n_a": int(matches["a_raw"].nunique()),
            "n_b": int(matches["b_raw"].nunique())}


def match_tracks(a: pd.DataFrame, b: pd.DataFrame, level: str = "strict", fuzzy: bool = True, threshold: float = 0.85,
                 max_fuzzy_pairs: int = 5_000_000, max_block_pairs: int = 200_000) -> pd.DataFrame:
    """Match tracks of ``a`` (index = raw id; columns artist, title, optional popularity) to tracks of ``b``.

    Returns ``DataFrame[a_raw, b_raw, method, score, ambiguous]`` (each ``a`` row at most once; several ``a`` rows may share
    one ``b`` row when they canonicalise to the same song).  ``result.attrs['counts']`` holds ``match_counts``.
    ``level`` is the curation policy ('strict' | 'song').  ``threshold`` applies to the fuzzy pass only.
    """
    A, B = _prepare(a, "a"), _prepare(b, "b")
    empty = pd.DataFrame({c: pd.Series(dtype=t) for c, t in zip(MATCH_COLUMNS, ["int64", "int64", "object", "float32", "bool"])})
    if (A["artist"].fillna("").astype(str).str.strip() == "").all() or (B["artist"].fillna("").astype(str).str.strip() == "").all():
        log.warning("match_tracks: one side has no artist strings at all (Blocker-A corpus?) -> no matches possible")
        empty.attrs["counts"] = match_counts(empty)
        return empty
    _add_artist_keys(A); _add_artist_keys(B)
    # only rows whose artist can occur on the other side need title normalisation (cuts 3M -> a few 100k)
    a_keys = set(A["ak"]) | set(A["pk"])
    b_keys = set(B["ak"]) | set(B["pk"])
    A = A[A["ak"].isin(b_keys) | A["pk"].isin(b_keys)].copy()
    B = B[B["ak"].isin(a_keys) | B["pk"].isin(a_keys)].copy()
    log.info("match_tracks: %d/%d a-rows and %d/%d b-rows share an artist key with the other side",
             len(A), len(a), len(B), len(b))
    _add_title_key(A, level); _add_title_key(B, level)
    parts = []
    m1 = _exact_pass(A, B, "ak", "exact", 1.0)
    parts.append(m1)
    A2, B2 = A[~A["raw"].isin(m1["a_raw"])], B  # b rows may be reused by several a rows (many-to-one); keep all b
    m2 = _exact_pass(A2, B2, "pk", "exact_primary_artist", _PRIMARY_SCORE)
    parts.append(m2)
    if fuzzy:
        done = pd.concat([m1, m2])["a_raw"] if len(m1) + len(m2) else pd.Series([], dtype="int64")
        A3 = A[~A["raw"].isin(done)]
        used_b = set(pd.concat([m1, m2])["b_raw"]) if len(m1) + len(m2) else set()
        B3 = B[~B["raw"].isin(used_b)]
        parts.append(_fuzzy_pass(A3, B3, threshold, max_fuzzy_pairs, max_block_pairs))
    parts = [p for p in parts if len(p)]
    out = pd.concat(parts, ignore_index=True) if parts else empty
    out = out.astype({"a_raw": "int64", "b_raw": "int64", "score": "float32", "ambiguous": "bool"})
    out = out.sort_values("a_raw", kind="stable").reset_index(drop=True)
    out.attrs["counts"] = match_counts(out)
    log.info("match_tracks: %s", out.attrs["counts"])
    return out


def _dedupe_matches(matches: pd.DataFrame) -> pd.DataFrame:
    if matches.empty:
        return matches
    return matches.sort_values(["a_raw", "score"], ascending=[True, False], kind="stable").drop_duplicates("a_raw")


def _reindex_block(b: FamilyBlock, tgt: np.ndarray, src: np.ndarray, n_src: int) -> FamilyBlock | None:
    if b.rows.size == 0:
        return None
    pos = np.full(n_src, -1, dtype=np.int64)
    pos[b.rows] = np.arange(b.rows.size)
    p = pos[src]
    ok = p >= 0
    if not ok.any():
        return None
    rows, X = tgt[ok], b.X[p[ok]]
    order = np.argsort(rows, kind="stable")
    return FamilyBlock(rows[order].astype(np.int64), np.ascontiguousarray(X[order], dtype=np.float32), list(b.columns), b.source)


def _reindex_frame(df: pd.DataFrame | None, tgt: np.ndarray, src: np.ndarray, n_target: int) -> pd.DataFrame | None:
    if df is None:
        return None
    vals = df.to_numpy(dtype=np.float32, na_value=np.nan)
    out = np.full((n_target, vals.shape[1]), np.nan, dtype=np.float32)
    out[tgt] = vals[src]
    return pd.DataFrame(out, columns=list(df.columns))


def enrich(target: Corpus, source: Corpus, matches: pd.DataFrame, swap: bool = False) -> Corpus:
    """New corpus with ``target``'s tracks/sessions and ``source``'s feature blocks moved onto target raw ids.

    ``matches`` is ``match_tracks(target.tracks, source.tracks)`` (``a`` = target, ``b`` = source), or the reverse if
    ``swap=True``.  Only matched target tracks appear in ``families`` / ``external`` (``FamilyBlock.rows`` are target ids);
    ``measured`` / ``heldout`` have one row per target track with NaN where unmatched.  Neither input is modified (session
    matrix and track table are shared by reference / copied respectively).
    """
    m = matches.rename(columns={"a_raw": "b_raw", "b_raw": "a_raw"}) if swap else matches
    m = _dedupe_matches(m)
    tgt = m["a_raw"].to_numpy(np.int64)
    src = m["b_raw"].to_numpy(np.int64)
    if tgt.size and (tgt.max() >= target.n_tracks or src.max() >= source.n_tracks or tgt.min() < 0 or src.min() < 0):
        raise ValueError("matches reference raw ids outside target/source track ranges")
    fam, ext = {}, {}
    for k, blk in source.families.items():
        r = _reindex_block(blk, tgt, src, source.n_tracks)
        if r is not None:
            fam[k] = r
    for k, blk in source.external.items():
        r = _reindex_block(blk, tgt, src, source.n_tracks)
        if r is not None:
            ext[k] = r
    meta = dict(target.meta)
    meta.update({"enriched_from": source.name, "n_matched_tracks": int(tgt.size),
                 "match_counts": match_counts(m), "has_audio": bool(fam),
                 "partial_hub_note": "compound blocks exist only for tracks matched to '%s'" % source.name})
    return Corpus(f"{target.name}+{source.name}", target.tracks.copy(), target.sessions, fam,
                  _reindex_frame(source.measured, tgt, src, target.n_tracks),
                  _reindex_frame(source.heldout, tgt, src, target.n_tracks), ext, target.sessions_meta, meta)


def coverage_report(target: Corpus, matches: pd.DataFrame, min_sessions: int = 20) -> dict:
    """How much of ``target`` gets compounds: share of tracks, of session-weighted mass (token count in the session
    matrix), the same restricted to 'eligible' tracks (>= ``min_sessions`` sessions, i.e. likely to survive S1) and the
    per-session covered fraction."""
    m = _dedupe_matches(matches)
    n = target.n_tracks
    covered = np.zeros(n, dtype=bool)
    if len(m):
        covered[m["a_raw"].to_numpy(np.int64)] = True
    X = sp.csr_matrix(target.sessions)
    nsess = np.asarray(X.sum(axis=0)).ravel()
    mass = float(nsess.sum())
    elig = nsess >= min_sessions
    frac = None
    if X.shape[0]:
        lens = np.diff(X.indptr).astype(np.float64)
        cov_per = X @ covered.astype(np.float32)
        frac = np.divide(cov_per, lens, out=np.zeros_like(lens), where=lens > 0)
    rep = {
        "n_tracks": int(n), "n_matched": int(covered.sum()), "track_share": float(covered.mean()) if n else 0.0,
        "mass_share": float(nsess[covered].sum() / mass) if mass else 0.0,
        "min_sessions": int(min_sessions), "n_eligible": int(elig.sum()),
        "n_eligible_matched": int((covered & elig).sum()),
        "eligible_track_share": float((covered & elig).sum() / max(int(elig.sum()), 1)),
        "eligible_mass_share": float(nsess[covered & elig].sum() / max(nsess[elig].sum(), 1.0)),
        "per_method": (m["method"].value_counts().astype(int).to_dict() if len(m) else {}),
        "n_ambiguous": int(m["ambiguous"].sum()) if len(m) else 0,
    }
    if frac is not None and frac.size:
        rep.update({"session_mean_covered_fraction": float(frac.mean()), "session_median_covered_fraction": float(np.median(frac)),
                    "share_sessions_half_covered": float((frac >= 0.5).mean())})
    return rep
