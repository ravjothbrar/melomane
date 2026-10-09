"""S2 under a budget that cannot cover the vocabulary: which tracks get labels, and that labels land on the right rows."""
import numpy as np
import pandas as pd

from melomane.pipeline.framework import run_stages
from melomane.pipeline.runner import make_ctx
from melomane.stages.common import vocab_of
try:
    from .wshelper import make_ws
except ImportError:
    from wshelper import make_ws


def _run(tmp_path, max_tracks=None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    ws = make_ws(tmp_path)
    ctx = make_ctx(ws, "synth", "smoke", {"demo_size": 400})
    if max_tracks is not None:
        ctx.profile["llm"]["annotate_max_tracks"] = max_tracks
    run_stages(ctx, ["S2"])
    return ctx, pd.read_parquet(ctx.stage_dir("S2") / "labels.parquet")


def test_budget_limited_labelling_is_a_random_sample_with_labels_on_the_right_rows(tmp_path):
    ctx_full, full = _run(tmp_path / "full")
    ctx_s, samp = _run(tmp_path / "samp", max_tracks=90)
    V = len(vocab_of(ctx_s))
    lab_full, lab_s = full["labelled"].to_numpy(bool), samp["labelled"].to_numpy(bool)
    assert lab_full.all() and 0 < lab_s.sum() <= 90 and lab_s.sum() < V
    idx = np.flatnonzero(lab_s)
    assert idx.max() > V * 0.6 and idx.min() < V * 0.4          # spread over the whole vocabulary, not a prefix
    # alignment: a sampled track carries exactly the labels it gets when everything is labelled
    for col in ("category", "energy", "valence", "has_vocals"):
        a, b = samp[col].to_numpy()[idx], full[col].to_numpy()[idx]
        same = (pd.Series(a).fillna(-1).to_numpy() == pd.Series(b).fillna(-1).to_numpy())
        assert same.all(), col
    # unlabelled rows stay unlabelled (never fabricated); the rule category still fills them for ICA residualisation
    assert samp.loc[~lab_s, "energy"].isna().all() and (samp.loc[~lab_s, "category"].astype(str) != "").all()
    # the same seed picks the same sample (reproducible, and a resumed run hits the cache)
    _, samp2 = _run(tmp_path / "samp2", max_tracks=90)
    assert (samp2["labelled"].to_numpy(bool) == lab_s).all()


def test_annotation_requests_leave_room_for_a_full_batch_of_labels():
    """Regression: 50 tracks x ~208 output tokens exceeded the default max_tokens (8,192); the truncated reply was retried and re-paid."""
    import pathlib, yaml
    from melomane.annotation import ANNOTATE_MAX_TOKENS, TOKENS_PER_TRACK, _call
    prof = yaml.safe_load((pathlib.Path(__file__).resolve().parents[1] / "configs" / "profiles.yaml").read_text(encoding="utf-8"))
    batch = prof["_base"]["llm"]["annotate_batch"]
    assert batch * TOKENS_PER_TRACK <= ANNOTATE_MAX_TOKENS * 0.75              # at least 25 % headroom at the shipped batch size
    assert _call(None, "sys", [{"id": 0}])["max_tokens"] == ANNOTATE_MAX_TOKENS
