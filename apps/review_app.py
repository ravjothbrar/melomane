"""Streamlit human-QC app for curation decisions (thin UI; all logic lives in ``melomane.release.review``).

Install the optional dependency and run::

    pip install "melomane[review]"            # or: pip install streamlit
    streamlit run apps/review_app.py -- --review-dir path/to/review

Reads ``<review-dir>/queue.parquet`` and appends decisions (merge / split / remove / keep) to
``<review-dir>/corrections.parquet``.  Only artist/title text is shown - never lyrics or audio.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

try:
    from melomane.release import review
except ImportError:                                                  # running from a checkout without `pip install -e .`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from melomane.release import review

HELP_INSTALL = 'The review app needs Streamlit: pip install "melomane[review]"  (or: pip install streamlit)'
LABELS = {"keep": "Keep automatic decision", "merge": "Same track (merge)", "split": "Different tracks (split)",
          "remove": "Remove from vocabulary"}
OFFERED = {"merge": ("keep", "split", "remove"), "split": ("keep", "merge", "remove"), "remove": ("keep", "split")}


def import_streamlit() -> Any:
    """Import streamlit lazily; exit with an install hint if it is missing."""
    try:
        import streamlit as st
    except ImportError:
        raise SystemExit(HELP_INSTALL) from None
    return st


def default_review_dir(argv: list[str] | None = None) -> Path:
    """``--review-dir`` (after ``--`` when launched by streamlit), else ``$MELOMANE_REVIEW_DIR``, else ``./review``."""
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--review-dir", default=os.environ.get("MELOMANE_REVIEW_DIR", "review"))
    args, _ = ap.parse_known_args(sys.argv[1:] if argv is None else argv)
    return Path(args.review_dir)


def run(st: Any, review_dir: Path) -> None:
    """Render one screen: the next pending queue item with decision buttons.  ``st`` is the streamlit module."""
    st.set_page_config(page_title="Melomane review", layout="wide")
    st.title("Melomane curation review")
    try:
        queue = review.load_queue(review_dir)
    except (FileNotFoundError, ValueError) as e:
        st.error(str(e))
        return
    done = review.load_corrections(review_dir)
    todo = review.pending(queue, done)

    kinds = sorted(queue["decision_auto"].unique())
    chosen = st.sidebar.multiselect("Automatic decision", kinds, default=kinds)
    todo = todo[todo["decision_auto"].isin(chosen)]
    skipped = st.session_state.setdefault("skipped", set())
    todo = todo[~todo["qid"].isin(skipped)]
    st.sidebar.metric("Reviewed", f"{len(done)} / {len(queue)}")
    st.progress(len(done) / max(len(queue), 1))

    if todo.empty:
        st.success("Nothing left to review in this selection.")
        return
    row = todo.iloc[0]
    st.subheader(f"{row['decision_auto'].title()} check")
    st.caption(f"{row['qid']}: {row['reason']}")
    left, right = st.columns(2)
    left.markdown(f"**A** — {row['artist_a']} – {row['title_a']}  \n`cid {row['cid_a']}`")
    if row["decision_auto"] == "remove":
        right.markdown("_Removed automatically (out of vocabulary)._")
    else:
        right.markdown(f"**B** — {row['artist_b']} – {row['title_b']}  \n`cid {row['cid_b']}`")
    note = st.text_input("Note (optional)", key=f"note-{row['qid']}")

    def record(decision: str) -> None:
        review.append_correction(review_dir, str(row["qid"]), decision, note)
        st.rerun()

    cols = st.columns(len(OFFERED[row["decision_auto"]]) + 1)
    for col, decision in zip(cols, OFFERED[row["decision_auto"]]):
        if col.button(LABELS[decision], key=f"{decision}-{row['qid']}"):
            record(decision)
    if cols[-1].button("Skip for now", key=f"skip-{row['qid']}"):
        skipped.add(str(row["qid"]))
        st.rerun()


def main() -> None:
    run(import_streamlit(), default_review_dir())


if __name__ == "__main__":                                           # streamlit executes the script as __main__
    main()
