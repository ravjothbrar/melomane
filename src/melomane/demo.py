"""Demo/test corpus with PLANTED structure + a deterministic stand-in LLM ("DemoLLM").

`melomane demo` (and the whole test-suite) exercise the *entire* pipeline on this corpus in minutes, so a fresh machine can be
verified end-to-end before the multi-hour real run.  Nothing here is real music data; every result is labelled DEMO.
The DemoLLM never touches the network: it answers the pipeline's structured prompts from tags planted in the track table.
"""
from __future__ import annotations
import hashlib, json, re
import numpy as np
import pandas as pd
import scipy.sparse as sp

from .corpus import Corpus, FamilyBlock
from .curation import canonical_key
from .synth import make_synth

LEVELS = ["none", "low", "moderate", "high", "very_high"]
REGION_NAMES = ["Anglo-American", "Latin American", "East Asian", "Francophone", "South Asian", "Nordic"]
CAT_BY_GROUP = ["Rock", "Hip-Hop", "Electronic/Dance", "Folk/Acoustic", "Jazz", "Pop"]


def make_demo_corpus(n_tracks: int = 2400, n_users: int = 4000, seed: int = 0, dup_frac: float = 0.06, junk: int = 8,
                     name: str = "synth") -> Corpus:
    S = make_synth(n_tracks=n_tracks, n_users=n_users, seed=seed)
    rng = np.random.default_rng(seed + 1)
    N0 = n_tracks
    artist_of = np.zeros(N0, dtype=np.int64)
    for r in np.unique(S.region):                                 # ~14 tracks per artist, artists live inside a region
        ids = np.flatnonzero(S.region == r)
        artist_of[ids] = r * 10_000 + np.arange(len(ids)) // 14
    year = 1950 + 10 * S.era + rng.integers(0, 10, N0)
    titles = [f"Song {i:04d} {['Blue', 'Night', 'River', 'Fire', 'Dust', 'Glass'][i % 6]}" for i in range(N0)]
    artists = [f"Artist {a}" for a in artist_of]
    tags = [f"e{int(S.energy[i])} y{int(year[i])} c:{CAT_BY_GROUP[int(S.genre_group[i]) % 6]}" + (f" r:{REGION_NAMES[int(S.region[i]) % 6]}" if S.region_tagged[i] else "")
            for i in range(N0)]
    rows = {"artist": artists, "title": titles, "album": [f"Album {a % 97}" for a in artist_of], "tags": tags,
            "artist_id": artist_of, "year": year.astype(float), "popularity": rng.integers(0, 100, N0).astype(float)}
    # duplicates ("remaster") that must be consolidated; behaviour is split between original and duplicate
    n_dup = int(dup_frac * N0)
    dup_src = rng.choice(N0, n_dup, replace=False)
    X = S.users_x_tracks.tocsc().astype(np.float32)
    dupcols = []
    for k, s in enumerate(dup_src):
        col = X[:, s].toarray().ravel()
        take = (rng.random(col.size) < 0.5) & (col > 0)
        keep = col.copy(); keep[take] = 0
        dup = np.zeros_like(col); dup[take] = 1
        X[:, s] = sp.csc_matrix(keep.reshape(-1, 1))
        dupcols.append(sp.csc_matrix(dup.reshape(-1, 1)))
    jk = sp.csc_matrix((rng.random((n_users, junk)) < 0.02).astype(np.float32))
    X = sp.hstack([X] + dupcols + [jk]).tocsr()
    df = pd.DataFrame(rows)
    dup_rows = df.iloc[dup_src].copy(); dup_rows["title"] = dup_rows["title"] + " - Remastered 2011"
    jrows = pd.DataFrame({"artist": [f"Noise Maker {i}" for i in range(junk)], "title": [f"White Noise {i * 3 + 1} Hours" for i in range(junk)],
                          "album": "", "tags": "", "artist_id": [900_000 + i for i in range(junk)], "year": np.nan, "popularity": 1.0})
    tracks = pd.concat([df, dup_rows, jrows], ignore_index=True)
    tracks.insert(0, "ext_id", [f"demo{i:06d}" for i in range(len(tracks))])
    tracks["has_lyrics"] = np.r_[S.has_lyrics, S.has_lyrics[dup_src], np.zeros(junk, bool)]
    N = len(tracks)
    fam_rows = np.r_[np.arange(N0), N0 + np.arange(n_dup)]                          # duplicates carry the same features
    src_rows = np.r_[np.arange(N0), dup_src]
    fams = {}
    for k, M in S.features.items():
        fams[k] = FamilyBlock(fam_rows.astype(np.int64), M[src_rows].astype(np.float32), [f"{k}_{j}" for j in range(M.shape[1])], "synthetic")
    meas = pd.DataFrame(index=np.arange(N), dtype=np.float32)
    E = np.full(N, np.nan); E[fam_rows] = S.energy[src_rows] + rng.normal(0, 0.3, len(src_rows))
    meas["loudness_mean"] = E                                                       # energy counterpart (baked-in style)
    D = np.full(N, np.nan); D[fam_rows] = S.era[src_rows] + rng.normal(0, 0.5, len(src_rows)); meas["release_year_est"] = D
    held = pd.DataFrame(index=np.arange(N), dtype=np.float32)
    H = np.full(N, np.nan); H[fam_rows] = rng.normal(size=len(src_rows)) + 0.6 * S.region[src_rows]; held["voice_f0_mean"] = H
    sessions = sp.csr_matrix(X)
    nnz = np.diff(sessions.indptr)
    keep = (nnz >= 5) & (nnz <= 200)
    sessions = sessions[np.flatnonzero(keep)]
    meta = {"source": "DEMO synthetic corpus (planted structure)", "is_demo": True, "document_type": "synthetic session",
            "has_audio": True, "has_lyrics": True, "has_year": True, "notes": ["DEMO data - not music."]}
    c = Corpus(name, tracks, sessions, fams, meas, held, {}, None, meta)
    c.validate()
    c._truth = dict(region=np.r_[S.region, S.region[dup_src], -np.ones(junk, int)], energy=np.r_[S.energy, S.energy[dup_src], -np.ones(junk, int)],
                    era=np.r_[S.era, S.era[dup_src], -np.ones(junk, int)])          # type: ignore[attr-defined]
    return c


class DemoLLM:
    """Deterministic stand-in for LLMClient: answers curation/annotation/mode prompts from planted tags.  `noise` flips a fraction
    of energy labels (reliability tests); `fail_names` forces request failures (degradation tests)."""

    def __init__(self, noise: float = 0.0, fail_names: tuple = (), budget: float = 1e9):
        self.calls: list[str] = []; self.noise = noise; self.fail_names = fail_names; self.budget = budget; self.model = "demo"

    def spent(self) -> float: return 0.0

    def complete_json(self, system, user, schema, name="result", temperature=0.0, max_tokens=8192):
        self.calls.append(name)
        if name in self.fail_names:
            raise RuntimeError("forced failure")
        return getattr(self, f"_{name}")(json.loads(user), system)

    def complete_json_many(self, jobs, workers=8):
        out = []
        for j in jobs:
            try: out.append(self.complete_json(**j))
            except Exception as e: out.append(e)
        return out

    def _refine(self, d, system):
        items = {i["id"]: i for i in d["items"]}
        remove = [k for k, i in items.items() if re.search(r"white noise|rain sounds|meditation", i["title"], re.I)]
        groups, used = [], set(remove)
        for g in d["candidate_groups"]:
            same = [k for k in g if k not in used]
            keyf = lambda k: bool(re.search(r"remix|live|acoustic|instrumental", items[k]["title"], re.I))
            plain = [k for k in same if not keyf(k)]
            groups.append(plain); used.update(plain)
            for k in same:
                if keyf(k): groups.append([k]); used.add(k)
        groups += [[k] for k in items if k not in used]
        return {"remove": remove, "groups": groups}

    def _ultra(self, d, system):
        v = []
        for g in d["groups"]:
            arts = {m["artist"] for m in g["members"]}
            v.append({"group": g["id"], "verdict": "confirm" if len(arts) == 1 else "reject", "partition": []})
        return {"verdicts": v, "restore": []}

    def _audit(self, d, system):
        ans = []
        for p in d["pairs"]:
            a = canonical_key(p["a"]["artist"], p["a"]["title"]); b = canonical_key(p["b"]["artist"], p["b"]["title"])
            ta, tb = set(a[1].split()), set(b[1].split())
            same = a[0] == b[0] and (ta <= tb or tb <= ta) and abs(len(ta) - len(tb)) <= 2 and not re.search(r"remix|live", p["a"]["title"] + p["b"]["title"], re.I)
            ans.append({"pair": p["pair"], "same": "yes" if same else "no"})
        return {"answers": ans}

    def _annotate(self, d, system):
        from .annotation import DIMS
        out = []
        for it in d["items"]:
            h = int(hashlib.md5((it["artist"] + it["title"]).encode()).hexdigest(), 16)
            rng = np.random.default_rng(h % (2 ** 32))
            m = re.search(r"\be(\d)\b", it.get("tags", ""))
            lv = {k: LEVELS[int(rng.integers(0, 5))] for k in DIMS}
            if m:
                lv["energy"] = LEVELS[int(m.group(1))]
                lv["aggression"] = LEVELS[min(4, max(0, int(m.group(1)) + int(rng.integers(-1, 2))))]
            if self.noise and rng.random() < self.noise:
                lv["energy"] = LEVELS[int(rng.integers(0, 5))]
            era = re.search(r"\by(\d{4})\b", it.get("tags", ""))
            lv["era"] = (f"{int(era.group(1)) // 10 * 10}s" if era else "unknown")
            cat = re.search(r"\bc:([A-Za-z/&\-]+)", it.get("tags", ""))
            reg = re.findall(r"\br:([A-Za-z\-/ ]+?)(?=\s+\w:|\s*$)", it.get("tags", ""))
            out.append({"id": it["id"], **lv, "category": cat.group(1) if cat else "Other", "regions": reg, "confidence": "high",
                        "has_vocals": True, "is_instrumental": False, "explicit_lyrics": False, "is_live": False})
        return {"items": out}

    def _binary(self, d, system):
        return {"answers": [{"q": q["q"], "answer": "yes"} for q in d["questions"]]}

    def _modes(self, d, system):
        return {"modes": [{"mode": m["mode"], "label": f"demo mode {m['mode']}", "property": "shared demo structure"} for m in d["modes"]]}

    def _audit_modes(self, d, system):
        return {"answers": [{"q": p["q"], "same": "yes"} for p in d.get("pairs", [])]}
