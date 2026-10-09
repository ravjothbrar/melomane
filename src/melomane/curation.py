"""Deterministic curation helpers (paper 1 stages 1-2 'LLM + Jaccard clustering' batching).

The LLM only ever sees *small candidate groups* produced here; pure-string rules handle the obvious cases so LLM spend
goes on genuinely ambiguous ones.  Policy (documented, ablatable):
  strict     : merge remaster/mono/stereo/edit/explicit/clean/deluxe/anniversary/single-version/feat-credit noise
  song-level : additionally merge live / acoustic / demo / session / unplugged versions of the same composition
Never merged (different recordings a listener distinguishes): remix, cover, instrumental, karaoke, sped-up, slowed, nightcore.
"""
from __future__ import annotations
import re, unicodedata
from collections import defaultdict

_STRICT_NOISE = r"remaster(?:ed)?(?:\s*\d{4})?|\d{4}\s*remaster(?:ed)?|mono|stereo|(?:radio|single|album|original|explicit|clean|short|long|extended|edit|version)(?:\s+(?:edit|version|mix))?|deluxe(?:\s+edition)?|\d+(?:th)?\s*anniversary(?:\s+edition)?|bonus\s*track|remix\s*edit(?=x)x"
_SONG_NOISE = r"live(?:\s+(?:at|in|from)\s+[^)\]]*)?|acoustic(?:\s+version)?|demo(?:\s+version)?|unplugged|session"
_KEEP = re.compile(r"remix|cover|instrumental|karaoke|sped\s*up|slowed|nightcore|tribute|vip|bootleg", re.I)
_BRACKET = re.compile(r"[\(\[]([^)\]]*)[\)\]]")
_FEAT = re.compile(r"\s*[\(\[]?\s*(?:feat\.?|ft\.?|featuring|with)\s+[^)\]]*[\)\]]?", re.I)
_DASH_SUFFIX = re.compile(r"\s+-\s+([^-]+)$")


def _ascii(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()


def _is_noise(text: str, level: str) -> bool:
    t = text.strip().lower()
    if _KEEP.search(t): return False
    pats = [_STRICT_NOISE] + ([_SONG_NOISE] if level == "song" else [])
    return any(re.fullmatch(rf"(?:{p})(?:\s+(?:{p}))*", t) for p in pats)


def normalize_title(title: str, level: str = "strict") -> str:
    t = _FEAT.sub("", title)
    def repl(m): return "" if _is_noise(m.group(1), level) else m.group(0)
    t = _BRACKET.sub(repl, t)
    m = _DASH_SUFFIX.search(t)
    if m and _is_noise(m.group(1), level): t = t[:m.start()]
    t = _ascii(t).lower()
    t = re.sub(r"&", " and ", t); t = re.sub(r"[^a-z0-9]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def normalize_artist(name: str) -> str:
    a = _ascii(name).lower().strip()
    m = re.match(r"^(.*),\s*the$", a)
    if m: a = "the " + m.group(1)
    a = re.sub(r"^the\s+", "", a); a = a.replace("&", " and ")
    a = re.sub(r"[^a-z0-9]+", " ", a)
    return re.sub(r"\s+", " ", a).strip()


def canonical_key(artist: str, title: str, level: str = "strict") -> tuple[str, str]:
    return normalize_artist(artist), normalize_title(title, level)


def exact_groups(records: list[tuple[str, str]], level: str = "strict") -> dict[tuple[str, str], list[int]]:
    g = defaultdict(list)
    for i, (a, t) in enumerate(records): g[canonical_key(a, t, level)].append(i)
    return dict(g)


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a or b else 0.0


def candidate_groups(keys: list[tuple[str, str]], threshold: float = 0.6, max_group: int = 40,
                     token_cap: int = 400) -> list[list[int]]:
    """Near-duplicate candidates: within one normalised artist, titles whose token sets have Jaccard >= threshold, or that
    differ by <= 2 tokens while one is a subset of the other (connected components, capped at `max_group` so one LLM request
    stays small).  Singletons are omitted.

    Scales to artists with tens of thousands of titles: pairs are generated from an inverted token index instead of all
    O(n^2) pairs.  Tokens shared by more than `token_cap` titles of one artist are bucketed by title length as well (so a very
    common word cannot blow up the pair count); this can miss a near-duplicate that shares *only* such a word across
    different lengths - documented approximation, irrelevant for ordinary catalogues."""
    by_artist = defaultdict(list)
    for i, (a, _) in enumerate(keys): by_artist[a].append(i)
    out = []
    for idxs in by_artist.values():
        if len(idxs) < 2: continue
        toks = {i: set(keys[i][1].split()) for i in idxs}
        parent = {i: i for i in idxs}
        def find(x):
            while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
            return x
        inv = defaultdict(list)
        for i in idxs:
            for w in toks[i]: inv[w].append(i)
        seen = set()
        def consider(i, j):
            if i > j: i, j = j, i
            if (i, j) in seen: return
            seen.add((i, j))
            ti, tj = toks[i], toks[j]
            near_subset = (ti <= tj or tj <= ti) and abs(len(ti) - len(tj)) <= 2   # 'yesterday' vs 'yesterday live'
            if keys[i][1] == keys[j][1] or near_subset or jaccard(ti, tj) >= threshold:
                parent[find(i)] = find(j)
        for w, members in inv.items():
            if len(members) < 2: continue
            buckets = [members] if len(members) <= token_cap else list(_by_len(members, toks).values())
            for b in buckets:
                for x in range(len(b)):
                    for y in range(x + 1, len(b)):
                        consider(b[x], b[y])
        by_key = defaultdict(list)                                  # identical (possibly empty) normalised titles
        for i in idxs: by_key[keys[i][1]].append(i)
        for grp in by_key.values():
            for x in range(1, len(grp)): parent[find(grp[x])] = find(grp[0])
        comp = defaultdict(list)
        for i in idxs: comp[find(i)].append(i)
        for c in comp.values():
            if len(c) > 1:
                out += [c[k:k + max_group] for k in range(0, len(c), max_group)]
    return out


def _by_len(members, toks):
    d = defaultdict(list)
    for m in members: d[len(toks[m])].append(m)
    return d
