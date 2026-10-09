"""Human QC logic for the curation stage (the Streamlit editor ``apps/review_app.py`` is a thin UI over this module).

Queue (``review/queue.parquet``): ``qid, cid_a, artist_a, title_a, cid_b, artist_b, title_b, decision_auto, reason``.
Corrections (``review/corrections.parquet``, append-only): ``qid, decision, note, ts`` with decision in
``{merge, split, remove, keep}``.

A ``qid`` is a stable, self-describing key so corrections can be applied to the consolidation map without the queue:

* ``m:<raw_id>``     an automatic **merge** - raw entry ``raw_id`` was folded into another track (``cid_a == cid_b`` = the
  surviving cid; ``artist_b/title_b`` are the raw entry when raw names are supplied);
* ``s:<cid_a>:<cid_b>``  a near-duplicate pair the pipeline **kept separate** (``decision_auto = 'split'``);
* ``r:<raw_id>``     an automatic **removal** (``cid = -1``: out of vocabulary).

Decision semantics (the desired final state of the item; ``keep`` = confirm the automatic decision, no change):

===========  ==========================  ==========================  ==================================
decision     ``m:`` (merged raw)         ``s:`` (separate pair)      ``r:`` (removed raw)
===========  ==========================  ==========================  ==================================
keep         no change                   no change                   no change
merge        no change                   fold cid_b into cid_a       ignored (warning)
split        raw gets its own new cid    no change                   restored as its own new cid
remove       raw removed (cid -1)        both cids removed (junk)    no change
===========  ==========================  ==========================  ==================================

Every touched row gets ``decision = 'human'``.  New cids are provisional (``max(cid) + 1 ...``): the next S1 run
re-densifies vocab and embeddings.  When one qid has several corrections the latest ``ts`` wins.
"""
from __future__ import annotations

import datetime as _dt
import logging
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

from ..curation import candidate_groups, canonical_key

log = logging.getLogger(__name__)

QUEUE_COLUMNS = ["qid", "cid_a", "artist_a", "title_a", "cid_b", "artist_b", "title_b", "decision_auto", "reason"]
CORRECTION_COLUMNS = ["qid", "decision", "note", "ts"]
DECISIONS = ("merge", "split", "remove", "keep")
_QID = re.compile(r"^(?:(m|r):(\d+)|(s):(\d+):(\d+))$")


def parse_qid(qid: str) -> tuple[str, tuple[int, ...]]:
    """``'m:12'`` -> ``('m', (12,))``; ``'s:3:9'`` -> ``('s', (3, 9))``.  Raises ``ValueError`` on anything else."""
    m = _QID.match(str(qid))
    if not m:
        raise ValueError(f"malformed qid {qid!r}")
    return (m.group(1), (int(m.group(2)),)) if m.group(1) else ("s", (int(m.group(4)), int(m.group(5))))


# --------------------------------------------------------------------------------------------------------------------------
# queue sampling
# --------------------------------------------------------------------------------------------------------------------------
def _empty_queue() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="object" if c not in ("cid_a", "cid_b") else "int64") for c in QUEUE_COLUMNS})


def candidate_pairs(vocab: pd.DataFrame, cmap: pd.DataFrame, raw: pd.DataFrame | None = None) -> pd.DataFrame:
    """Every reviewable automatic decision as a queue row (before sampling), with a ``category`` stratum column.

    ``raw`` (optional, index = raw_id, columns ``artist``, ``title``) supplies names for raw entries that are not in the
    vocabulary; without it they are shown as ``(raw #<id>)``.  Decisions already made by a human are not re-queued."""
    v = vocab.set_index("cid")
    rows: list[dict] = []

    def raw_name(rid: int) -> tuple[str, str]:
        if raw is not None and rid in raw.index:
            return str(raw.at[rid, "artist"]), str(raw.at[rid, "title"])
        return "", f"(raw #{rid})"

    rep = vocab.set_index("cid")["rep_raw"].to_dict() if "rep_raw" in vocab else cmap[cmap["cid"] >= 0].groupby("cid")["raw_id"].min().to_dict()
    auto = cmap[cmap["decision"] != "human"]
    merged = auto[(auto["cid"] >= 0) & (auto["raw_id"].to_numpy() != auto["cid"].map(rep).fillna(-1).to_numpy())]
    for r in merged.itertuples(index=False):
        c = int(r.cid)
        ba, bt = raw_name(int(r.raw_id))
        rows.append({"qid": f"m:{int(r.raw_id)}", "cid_a": c, "artist_a": v.at[c, "artist"], "title_a": v.at[c, "title"],
                     "cid_b": c, "artist_b": ba, "title_b": bt, "decision_auto": "merge",
                     "reason": f"raw #{int(r.raw_id)} folded into cid {c} by {r.decision} (group {int(r.group_id)})",
                     "category": v.at[c, "category"]})
    for r in auto[auto["cid"] < 0].itertuples(index=False):
        ba, bt = raw_name(int(r.raw_id))
        rows.append({"qid": f"r:{int(r.raw_id)}", "cid_a": -1, "artist_a": ba, "title_a": bt, "cid_b": -1, "artist_b": "",
                     "title_b": "", "decision_auto": "remove", "reason": f"raw #{int(r.raw_id)} removed by {r.decision}",
                     "category": "(removed)"})
    keys = [canonical_key(a, t, "song") for a, t in zip(vocab["artist"].astype(str), vocab["title"].astype(str))]
    cid_arr = vocab["cid"].to_numpy()
    for grp in candidate_groups(keys):
        for x in range(len(grp)):
            for y in range(x + 1, len(grp)):
                a, b = sorted((int(cid_arr[grp[x]]), int(cid_arr[grp[y]])))
                rows.append({"qid": f"s:{a}:{b}", "cid_a": a, "artist_a": v.at[a, "artist"], "title_a": v.at[a, "title"],
                             "cid_b": b, "artist_b": v.at[b, "artist"], "title_b": v.at[b, "title"], "decision_auto": "split",
                             "reason": "near-duplicate title by the same artist kept as separate tracks",
                             "category": v.at[a, "category"]})
    if not rows:
        return _empty_queue().assign(category=pd.Series(dtype="object"))
    out = pd.DataFrame(rows)
    out["category"] = out["category"].fillna("(none)").astype(str)
    return out.drop_duplicates("qid").reset_index(drop=True)


def sample_queue(vocab: pd.DataFrame, cmap: pd.DataFrame, frac: float = 0.05, seed: int = 0,
                 raw: pd.DataFrame | None = None, min_per_stratum: int = 1) -> pd.DataFrame:
    """Stratified random sample of ``frac`` of the reviewable automatic decisions.

    Strata = ``decision_auto`` (merge / split / remove) x track category; each non-empty stratum contributes
    ``max(min_per_stratum, ceil(frac * size))`` items (capped at its size) so rare categories and rare decision types are
    never missed.  Deterministic in ``seed``; ``qid`` is stable so re-sampling and resuming never duplicates work.
    Returns the queue columns only, ordered by ``qid``."""
    if not 0.0 < frac <= 1.0:
        raise ValueError("frac must be in (0, 1]")
    cand = candidate_pairs(vocab, cmap, raw)
    if cand.empty:
        return _empty_queue()
    rng = np.random.default_rng(seed)
    picked: list[pd.DataFrame] = []
    for _, grp in sorted(cand.groupby(["decision_auto", "category"], sort=True), key=lambda kv: kv[0]):
        n = min(len(grp), max(min_per_stratum, math.ceil(frac * len(grp))))
        picked.append(grp.iloc[np.sort(rng.choice(len(grp), size=n, replace=False))])
    out = pd.concat(picked).sort_values("qid", kind="stable")
    return out[QUEUE_COLUMNS].reset_index(drop=True)


# --------------------------------------------------------------------------------------------------------------------------
# corrections: storage
# --------------------------------------------------------------------------------------------------------------------------
def _atomic_write(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)


def load_queue(review_dir: str | Path) -> pd.DataFrame:
    p = Path(review_dir) / "queue.parquet"
    if not p.exists():
        raise FileNotFoundError(f"{p} not found: run sample_queue(...) and write it there first")
    df = pd.read_parquet(p)
    missing = [c for c in QUEUE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"queue.parquet is missing columns {missing}")
    return df


def write_queue(review_dir: str | Path, queue: pd.DataFrame) -> Path:
    p = Path(review_dir) / "queue.parquet"
    _atomic_write(queue[QUEUE_COLUMNS], p)
    return p


def load_corrections(review_dir: str | Path) -> pd.DataFrame:
    p = Path(review_dir) / "corrections.parquet"
    if not p.exists():
        return pd.DataFrame({"qid": pd.Series(dtype="object"), "decision": pd.Series(dtype="object"),
                             "note": pd.Series(dtype="object"), "ts": pd.Series(dtype="object")})
    return pd.read_parquet(p)


def append_correction(review_dir: str | Path, qid: str, decision: str, note: str = "", ts: str | None = None) -> pd.DataFrame:
    """Append one decision to ``corrections.parquet`` (atomic replace; earlier rows are never modified)."""
    if decision not in DECISIONS:
        raise ValueError(f"decision must be one of {DECISIONS}, got {decision!r}")
    parse_qid(qid)
    row = pd.DataFrame([{"qid": qid, "decision": decision, "note": note or "",
                         "ts": ts or _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")}])
    df = pd.concat([load_corrections(review_dir), row], ignore_index=True)
    _atomic_write(df[CORRECTION_COLUMNS], Path(review_dir) / "corrections.parquet")
    return df


def pending(queue: pd.DataFrame, corrections: pd.DataFrame) -> pd.DataFrame:
    """Queue rows that have no correction yet (resume point for the reviewer)."""
    return queue[~queue["qid"].isin(set(corrections["qid"]))].reset_index(drop=True)


# --------------------------------------------------------------------------------------------------------------------------
# corrections: application
# --------------------------------------------------------------------------------------------------------------------------
def latest_corrections(corrections: pd.DataFrame) -> pd.DataFrame:
    """One row per qid: the one with the greatest ``ts`` (ties: last written).  Sorted by ``ts``."""
    if corrections.empty:
        return corrections.copy()
    bad = set(corrections["decision"]) - set(DECISIONS)
    if bad:
        raise ValueError(f"unknown decision(s) {sorted(bad)}")
    c = corrections.reset_index(drop=True).assign(_ord=lambda d: np.arange(len(d)))
    c = c.sort_values(["ts", "_ord"], kind="stable").drop_duplicates("qid", keep="last")
    return c.drop(columns="_ord").reset_index(drop=True)


def apply_corrections(cmap: pd.DataFrame, corrections: pd.DataFrame) -> pd.DataFrame:
    """Apply human corrections to a consolidation map (``raw_id, cid, decision, group_id``) and return a new frame.

    See the module docstring for the per-decision semantics.  Rows whose qid no longer matches the map (for example a pair
    already merged by an earlier correction) are skipped with a log message; malformed qids raise ``ValueError``."""
    out = cmap.copy().reset_index(drop=True)
    for col in ("raw_id", "cid", "group_id"):
        out[col] = out[col].astype("int64")
    out["decision"] = out["decision"].astype(object)
    next_cid = int(out["cid"].max()) + 1 if len(out) else 0
    next_group = int(out["group_id"].max()) + 1 if len(out) else 0
    by_raw = {int(r): i for i, r in enumerate(out["raw_id"])}

    def mark(mask: np.ndarray | pd.Series, cid: int | None = None, group: int | None = None) -> None:
        if cid is not None:
            out.loc[mask, "cid"] = cid
        if group is not None:
            out.loc[mask, "group_id"] = group
        out.loc[mask, "decision"] = "human"

    for c in latest_corrections(corrections).itertuples(index=False):
        kind, ids = parse_qid(c.qid)
        d = c.decision
        if kind in ("m", "r"):
            i = by_raw.get(ids[0])
            if i is None:
                log.warning("%s: raw id %d not in the map, skipped", c.qid, ids[0]); continue
            m = out.index == i
            if kind == "m":
                if d in ("keep", "merge"):
                    mark(m)
                elif d == "split":
                    mark(m, cid=next_cid, group=next_group); next_cid += 1; next_group += 1
                elif d == "remove":
                    mark(m, cid=-1)
            else:
                if d in ("keep", "remove"):
                    mark(m)
                elif d == "split":
                    mark(m, cid=next_cid, group=next_group); next_cid += 1; next_group += 1
                else:
                    log.warning("%s: 'merge' is not applicable to a removed entry, skipped", c.qid)
        else:
            a, b = ids
            if d in ("keep", "split"):
                continue                                    # pair stays separate: nothing to change in the map
            if d == "merge":
                mb = out["cid"] == b
                if not mb.any() or not (out["cid"] == a).any():
                    log.warning("%s: one side no longer exists, skipped", c.qid); continue
                ga = int(out.loc[out["cid"] == a, "group_id"].iloc[0])
                mark(mb | (out["cid"] == a), cid=a, group=ga)
            elif d == "remove":
                mark((out["cid"] == a) | (out["cid"] == b), cid=-1)
    return out
