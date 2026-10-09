"""S2 - LLM annotation (paper-1 5-level dimensions) + tag context + reliability + LLM-vs-measured validation.

The LLM sees only artist / title / album / tags (never audio features, never lyrics), so its labels are independent of the
measured data used to validate them (USDA analogue).  Levels are mapped none..very_high -> 0..4; era -> decade ordinal.
"""
from __future__ import annotations
import json, re
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import spearmanr

from .llm import BudgetExceeded
from .runtime import get_logger

log = get_logger("melomane.annotation")

LEVELS = ["none", "low", "moderate", "high", "very_high"]
DIMS = ["energy", "aggression", "valence", "danceability", "acousticness", "vocal_prominence", "tempo_feel", "production_polish",
        "lyrical_explicitness", "grit", "density", "space", "punch", "warmth", "brightness"]
ERAS = ["1950s", "1960s", "1970s", "1980s", "1990s", "2000s", "2010s", "2020s", "unknown"]
CATEGORIES = ["Rock", "Pop", "Hip-Hop", "R&B/Soul", "Electronic/Dance", "Metal", "Punk/Hardcore", "Indie/Alternative", "Folk/Acoustic",
              "Country", "Jazz", "Blues", "Classical/Score", "Reggae/Dub", "Latin", "World/Regional", "Ambient/Chill", "Soundtrack",
              "Gospel/Religious", "Other"]
REGIONS = ["Anglo-American", "British-Irish", "Latin American", "Iberian", "Francophone", "Germanic-Central-European", "Nordic",
           "East Asian", "South Asian", "African/Afro-Caribbean", "Middle-Eastern/Mediterranean"]

ANNOTATE_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["items"],
    "properties": {"items": {"type": "array", "items": {
        "type": "object", "additionalProperties": False,
        "required": ["id", *DIMS, "era", "category", "regions", "confidence", "has_vocals", "is_instrumental", "explicit_lyrics", "is_live"],
        "properties": {"id": {"type": "integer"},
                       **{d: {"type": "string", "enum": LEVELS} for d in DIMS},
                       "era": {"type": "string", "enum": ERAS},
                       "category": {"type": "string", "enum": CATEGORIES},
                       "regions": {"type": "array", "items": {"type": "string", "enum": REGIONS}},
                       "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
                       "has_vocals": {"type": "boolean"}, "is_instrumental": {"type": "boolean"},
                       "explicit_lyrics": {"type": "boolean"}, "is_live": {"type": "boolean"}}}}}}

BINARY_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["answers"],
                 "properties": {"answers": {"type": "array", "items": {"type": "object", "additionalProperties": False,
                                            "required": ["q", "answer"],
                                            "properties": {"q": {"type": "integer"}, "answer": {"type": "string", "enum": ["yes", "no"]}}}}}}

# measured counterparts (USDA analogue): regex candidates over measured/heldout column names (first matching column wins)
MEASURED_COUNTERPART = {
    "energy": [r"loudness", r"onset", r"rms"], "brightness": [r"spectral[_ ]?centroid", r"centroid", r"rolloff"],
    "danceability": [r"danceab"], "tempo_feel": [r"bpm", r"tempo"], "acousticness": [r"flatness", r"tonal", r"acoustic"],
    "vocal_prominence": [r"voice", r"vocal"], "era": [r"^year$", r"release"], "valence": [r"gems.*joy|joy|valence|vad.*valence"],
    "aggression": [r"dynamic[_ ]?complexity", r"loudness"], "warmth": [r"low.*mid|lowmid|warm"], "punch": [r"attack|transient|onset"],
}
LEAK_POLICY = "LLM never sees measured features; counterparts are used only for validation"


def category_from_tags(tags: str) -> str:
    """Deterministic keyword fallback (no LLM): first matching category wins."""
    t = (tags or "").lower()
    rules = [("Metal", r"metal|grindcore|deathcore"), ("Punk/Hardcore", r"punk|hardcore|emo\b"), ("Hip-Hop", r"hip.?hop|rap\b|trap|grime"),
             ("R&B/Soul", r"r&b|rnb|soul|funk|motown"), ("Electronic/Dance", r"electro|house|techno|trance|edm|dubstep|dance|drum and bass|dnb"),
             ("Ambient/Chill", r"ambient|chill|lo-?fi|downtempo"), ("Jazz", r"jazz|bebop|swing"), ("Blues", r"blues"),
             ("Classical/Score", r"classical|baroque|orchestra|opera|piano concerto"), ("Soundtrack", r"soundtrack|score|ost\b"),
             ("Reggae/Dub", r"reggae|dub|ska\b|dancehall"), ("Latin", r"latin|salsa|reggaeton|bachata|cumbia|bossa"),
             ("Country", r"country|bluegrass"), ("Folk/Acoustic", r"folk|acoustic|singer.?songwriter"),
             ("Indie/Alternative", r"indie|alternative|shoegaze|post.?rock"), ("Rock", r"rock|grunge"), ("Pop", r"pop|k-?pop|j-?pop"),
             ("World/Regional", r"world|afro|bollywood|arabic|celtic"), ("Gospel/Religious", r"gospel|christian|worship")]
    for cat, pat in rules:
        if re.search(pat, t):
            return cat
    return "Other"


def playlist_name_tags(sessions_cid: sp.csr_matrix, names: list[str], top: int = 5, min_df: int = 20, max_features: int = 4000) -> list[str]:
    """Pseudo-tags for tracks from the *names of the playlists they occur in* (Playlist2vec has no Last.fm tags):
    per-track top tokens by lift over corpus frequency.  Stop-words and pure-number tokens dropped."""
    from sklearn.feature_extraction.text import CountVectorizer
    cv = CountVectorizer(stop_words="english", min_df=min_df, max_features=max_features, token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9&'\-]{2,}\b",
                         lowercase=True, binary=True)
    try:
        T = cv.fit_transform(names).astype(np.float32).tocsr()                     # sessions x tokens
    except ValueError:
        return [""] * sessions_cid.shape[1]
    vocab = np.array(cv.get_feature_names_out())
    M = (sessions_cid.T.tocsr().astype(np.float32) @ T).tocsr()                     # tracks x tokens (co-occurrence counts)
    tok_freq = np.asarray(T.sum(axis=0)).ravel() + 1.0
    M = M.multiply(1.0 / np.sqrt(tok_freq)[None, :]).tocsr()                        # damp very common tokens
    out = []
    for i in range(M.shape[0]):
        s, e = M.indptr[i], M.indptr[i + 1]
        if e == s:
            out.append(""); continue
        d = M.data[s:e]; ix = M.indices[s:e]
        k = min(top, len(d))
        sel = np.argpartition(-d, k - 1)[:k]
        out.append(" ".join(vocab[ix[sel[np.argsort(-d[sel])]]]))
    return out


def _level(x: str) -> float:
    return float(LEVELS.index(x)) if x in LEVELS else np.nan


def _era(x: str) -> float:
    m = re.match(r"(\d{4})s", str(x))
    return float(int(m.group(1))) if m else np.nan


def items_payload(vocab: pd.DataFrame, idx, tag_col: str = "tags") -> list[dict]:
    out = []
    for k, i in enumerate(idx):
        r = vocab.iloc[i]
        out.append({"id": k, "artist": str(r["artist"])[:80], "title": str(r["title"])[:120], "album": str(r.get("album", "") or "")[:80],
                    "tags": str(r.get(tag_col, "") or "")[:120]})
    return out


def _call(llm, system, its, name="annotate"):
    return dict(system=system, user=json.dumps({"items": its}, ensure_ascii=False), schema=ANNOTATE_SCHEMA, name=name)


def _parse(resp: dict, n: int) -> dict[int, dict]:
    out = {}
    for it in resp.get("items", []):
        i = it.get("id")
        if isinstance(i, int) and 0 <= i < n and i not in out:
            out[i] = it
    return out


def annotate(vocab: pd.DataFrame, llm, system: str, prompt_version: str, batch: int = 50, workers: int = 8,
             tag_col: str = "tags", order_seed: int | None = None) -> tuple[pd.DataFrame, dict]:
    """Label every vocab row.  Missing/invalid items are retried once in a smaller batch; anything still missing is NaN and
    reported in the coverage stats (we never fabricate labels)."""
    V = len(vocab)
    order = np.arange(V) if order_seed is None else np.random.default_rng(order_seed).permutation(V)
    rows: dict[int, dict] = {}
    fails = 0
    budget_hit = False
    for pass_i, bs in enumerate([batch, max(10, batch // 5)]):
        todo = [int(i) for i in order if int(i) not in rows]
        if not todo or budget_hit:
            break
        chunks = [todo[i:i + bs] for i in range(0, len(todo), bs)]
        jobs = [_call(llm, system, items_payload(vocab, c, tag_col)) for c in chunks]
        try:
            res = llm.complete_json_many(jobs, workers=workers)
        except BudgetExceeded:
            budget_hit = True; break
        for c, r in zip(chunks, res):
            if isinstance(r, Exception):
                fails += 1
                if isinstance(r, BudgetExceeded): budget_hit = True
                continue
            for k, it in _parse(r, len(c)).items():
                rows[c[k]] = it
    lab = pd.DataFrame(index=np.arange(V))
    for d in DIMS:
        lab[d] = [_level(rows[i][d]) if i in rows else np.nan for i in range(V)]
    lab["era"] = [_era(rows[i]["era"]) if i in rows else np.nan for i in range(V)]
    lab["category"] = [rows[i]["category"] if i in rows else "" for i in range(V)]
    lab["regions"] = ["|".join(rows[i]["regions"]) if i in rows else "" for i in range(V)]
    lab["confidence"] = [rows[i]["confidence"] if i in rows else "" for i in range(V)]
    for f in ("has_vocals", "is_instrumental", "explicit_lyrics", "is_live"):
        lab[f] = [bool(rows[i][f]) if i in rows else False for i in range(V)]
    lab["labelled"] = [i in rows for i in range(V)]
    lab["prompt_version"] = prompt_version
    stats = {"coverage": float(lab["labelled"].mean()), "n_labelled": int(lab["labelled"].sum()), "failed_requests": fails,
             "budget_hit": budget_hit}
    return lab, stats


def reliability(lab_a: pd.DataFrame, lab_b: pd.DataFrame, idx: np.ndarray) -> pd.DataFrame:
    """Between-run agreement on the re-labelled sample: Spearman (ordinal dims), weighted kappa; category exact-match."""
    from .analysis.stats import weighted_kappa
    rows = []
    for d in DIMS + ["era"]:
        a, b = lab_a.loc[idx, d].to_numpy(float), lab_b.loc[idx, d].to_numpy(float)
        ok = ~(np.isnan(a) | np.isnan(b))
        if ok.sum() < 10 or np.nanstd(a[ok]) == 0 or np.nanstd(b[ok]) == 0:
            rows.append({"dim": d, "n": int(ok.sum()), "spearman": np.nan, "wkappa": np.nan}); continue
        rho = float(spearmanr(a[ok], b[ok]).statistic)
        wk = float(weighted_kappa(a[ok], b[ok])["kappa"])
        rows.append({"dim": d, "n": int(ok.sum()), "spearman": rho, "wkappa": wk})
    a, b = lab_a.loc[idx, "category"], lab_b.loc[idx, "category"]
    rows.append({"dim": "category", "n": int(len(idx)), "spearman": np.nan, "wkappa": np.nan, "exact_match": float((a.values == b.values).mean())})
    return pd.DataFrame(rows)


def find_measured(columns: list[str], dim: str) -> str | None:
    for pat in MEASURED_COUNTERPART.get(dim, []):
        for c in columns:
            if re.search(pat, str(c), re.I):
                return c
    return None


def llm_vs_measured(lab: pd.DataFrame, measured: pd.DataFrame | None, cid_to_raw: np.ndarray, extra: dict[str, np.ndarray] | None = None,
                    seed: int = 0, n_boot: int = 500) -> pd.DataFrame:
    """H5: Spearman(LLM level, measured value) with bootstrap CI, per dimension that has a counterpart."""
    rows = []
    meas = measured.iloc[cid_to_raw].reset_index(drop=True) if measured is not None else pd.DataFrame(index=np.arange(len(lab)))
    if extra:
        for k, v in extra.items():
            meas[k] = v
    rng = np.random.default_rng(seed)
    for d in DIMS + ["era"]:
        col = find_measured(list(meas.columns), d)
        if col is None:
            continue
        a, b = lab[d].to_numpy(float), meas[col].to_numpy(float)
        ok = ~(np.isnan(a) | np.isnan(b))
        if ok.sum() < 30 or np.std(a[ok]) == 0 or np.std(b[ok]) == 0:
            continue
        rho = float(spearmanr(a[ok], b[ok]).statistic)
        ii = np.flatnonzero(ok)
        boots = []
        for _ in range(n_boot):
            s = rng.choice(ii, len(ii))
            if np.std(a[s]) > 0 and np.std(b[s]) > 0:
                boots.append(spearmanr(a[s], b[s]).statistic)
        lo, hi = (np.percentile(boots, [2.5, 97.5]) if boots else (np.nan, np.nan))
        rows.append({"dim": d, "measured": col, "n": int(ok.sum()), "rho": rho, "ci_lo": float(lo), "ci_hi": float(hi),
                     "ci_excludes_0": bool(lo > 0 or hi < 0)})
    return pd.DataFrame(rows)


def binary_pass(vocab: pd.DataFrame, lab: pd.DataFrame, llm, system: str, dims: list[str], batch: int = 60, workers: int = 8) -> pd.DataFrame:
    """Yes/No verification of extreme levels (paper 1).  Returns columns `<dim>_bin` (1 confirmed high/very_high, 0 confirmed none/low,
    NaN unverified/middle)."""
    out = pd.DataFrame(index=lab.index)
    for d in dims:
        ext = lab.index[(lab[d] >= 3) | (lab[d] <= 1)]
        qs = [{"q": k, "track": {"artist": str(vocab.at[i, "artist"])[:80], "title": str(vocab.at[i, "title"])[:120]},
               "dimension": d, "claimed_level": LEVELS[int(lab.at[i, d])]} for k, i in enumerate(ext)]
        res_vals = np.full(len(ext), np.nan)
        chunks = [list(range(i, min(i + batch, len(qs)))) for i in range(0, len(qs), batch)]
        jobs = [dict(system=system, user=json.dumps({"questions": [qs[k] for k in ch]}, ensure_ascii=False), schema=BINARY_SCHEMA, name="binary") for ch in chunks]
        try:
            res = llm.complete_json_many(jobs, workers=workers) if jobs else []
        except BudgetExceeded:
            res = []
        for ch, r in zip(chunks, res):
            if isinstance(r, Exception):
                continue
            for a in r.get("answers", []):
                q = a.get("q")
                if isinstance(q, int) and q in ch and a["answer"] == "yes":
                    res_vals[q] = 1.0 if lab.at[ext[q], d] >= 3 else 0.0
        col = pd.Series(np.nan, index=lab.index); col.loc[ext] = res_vals
        out[f"{d}_bin"] = col
    return out
