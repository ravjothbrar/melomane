"""Public Python API over a released Melomane bundle (mirrors Epicure's ``epicure.py`` loader).

::

    from melomane.api import Melomane
    m = Melomane.from_pretrained("bundles/core")
    m.neighbors("Radiohead - Karma Police", k=10)              # artist-disjoint by default
    m.slerp("Radiohead - Karma Police", "category:jazz", 45)   # rotate the seed 45 deg toward a pole
    m.describe("Radiohead - Karma Police")                     # metadata only: never lyrics or audio

Bundle layout (``docs/CONTRACTS.md``): ``embeddings.safetensors`` (key ``embeddings``; fp32 / fp16 / int8 with per-row key
``scale``; optional PCA keys ``pca_components`` [+ ``pca_mean``]), ``vocab.json``, ``modes.json``, ``factor_poles.npy``,
``supervised_poles.json`` + ``.npy``, ``config.json``.  Every result is a plain ``dict`` / ``list`` (JSON-able).

Conventions
* All similarity maths is cosine on the *stored* (dequantised) vectors; poles are stored in the bundle's own space.
* "Artist-disjoint" means: results whose normalised artist name equals the seed's artist are excluded
  (``vocab.json`` deliberately carries no artist ids).
* Emergent poles are addressed as ``"mode:<mode_id>"`` / ``"factor:<i>"``, supervised poles by name
  (``"category:jazz"``, or the bare label ``"jazz"`` when unambiguous), another track as ``"track:<name or cid>"``.
"""
from __future__ import annotations

import difflib
import json
import logging
import re
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from safetensors.numpy import load_file

from .curation import normalize_artist, normalize_title
from .geometry import unit
from .slerp import compose_poles, nearest, slerp as _slerp

log = logging.getLogger(__name__)

#: The only per-track fields a bundle may expose.  Anything else in a vocab file (lyrics, ids, user data) is dropped on load.
VOCAB_FIELDS = ("cid", "artist", "title", "album", "year", "category", "dev_split", "region_tags")
_TOKEN = re.compile(r"[a-z0-9]+")


class TrackNotFound(LookupError):
    """Raised when a name cannot be resolved to exactly one track; ``suggestions`` lists the closest candidates."""

    def __init__(self, message: str, suggestions: list[dict] | None = None) -> None:
        super().__init__(message)
        self.suggestions = suggestions or []


# --------------------------------------------------------------------------------------------------------------------------
# shared helpers (also used by melomane.release.*)
# --------------------------------------------------------------------------------------------------------------------------
def mode_poles_from_members(U: np.ndarray, member_lists: Sequence[Sequence[int]]) -> np.ndarray:
    """Unit-mean pole per mode (``unit(mean(unit rows))``), matching the paper's definition of an emergent pole."""
    out = np.zeros((len(member_lists), U.shape[1]), dtype=np.float32)
    for i, mem in enumerate(member_lists):
        if len(mem):
            out[i] = unit(U[np.asarray(mem, dtype=np.int64)].mean(0))
    return out


def primary_mode_assignment(U: np.ndarray, member_lists: Sequence[Sequence[int]], poles: np.ndarray) -> np.ndarray:
    """For each track the index of the member-mode whose pole is closest (cosine); ``-1`` for tracks in no mode."""
    best = np.full(len(U), -np.inf, dtype=np.float64)
    which = np.full(len(U), -1, dtype=np.int32)
    for m, mem in enumerate(member_lists):
        if not len(mem):
            continue
        mem = np.asarray(mem, dtype=np.int64)
        c = U[mem] @ poles[m]
        better = c > best[mem]
        best[mem[better]] = c[better]
        which[mem[better]] = m
    return which


def dequantise(tensors: dict[str, np.ndarray]) -> np.ndarray:
    """Return float32 (V x d) from a loaded ``embeddings.safetensors`` (fp32, fp16, or int8 + per-row ``scale``)."""
    if "embeddings" not in tensors:
        raise KeyError("embeddings.safetensors has no 'embeddings' tensor")
    x = tensors["embeddings"]
    if x.dtype == np.int8:
        if "scale" not in tensors:
            raise KeyError("int8 embeddings require a per-row 'scale' tensor")
        return x.astype(np.float32) * tensors["scale"].astype(np.float32)[:, None]
    return x.astype(np.float32)


def _clean(v: Any) -> Any:
    """JSON-able scalar: numpy -> python, NaN -> None."""
    if isinstance(v, np.generic):
        v = v.item()
    if isinstance(v, float) and v != v:
        return None
    return v


def _tokens(s: str) -> list[str]:
    return _TOKEN.findall(s)


# --------------------------------------------------------------------------------------------------------------------------
class Melomane:
    """A loaded embedding bundle plus its vocabulary, modes and poles.  Construct via :meth:`from_pretrained`."""

    def __init__(self, embeddings: np.ndarray, vocab: list[dict], *, modes: dict | None = None,
                 factor_poles: np.ndarray | None = None, supervised_meta: dict | None = None,
                 supervised_poles: np.ndarray | None = None, config: dict | None = None,
                 projection: np.ndarray | None = None, projection_mean: np.ndarray | None = None,
                 path: Path | None = None) -> None:
        E = np.ascontiguousarray(embeddings, dtype=np.float32)
        if E.ndim != 2 or len(E) != len(vocab):
            raise ValueError(f"embeddings {E.shape} do not match vocabulary of {len(vocab)} tracks")
        self.E = E
        self.U = unit(E).astype(np.float32)
        self.dim = E.shape[1]
        self.vocab = [{k: _clean(t.get(k)) for k in VOCAB_FIELDS} for t in vocab]
        for i, t in enumerate(self.vocab):
            if t["cid"] is None:
                t["cid"] = i
            if t["cid"] != i:
                raise ValueError(f"vocab row {i} has cid {t['cid']}: vocab must be in cid order")
        self.config = dict(config or {})
        self.path = path
        self.projection = None if projection is None else np.asarray(projection, dtype=np.float32)
        self.projection_mean = None if projection_mean is None else np.asarray(projection_mean, dtype=np.float32)

        # artist codes (normalised artist name) for artist-disjoint retrieval
        codes: dict[str, int] = {}
        self._artist_code = np.empty(len(self.vocab), dtype=np.int32)
        self._artist_norm: list[str] = []
        self._title_norm: list[str] = []
        for i, t in enumerate(self.vocab):
            a = normalize_artist(t["artist"] or "")
            self._artist_norm.append(a)
            self._title_norm.append(normalize_title(t["title"] or ""))
            # an unknown artist (empty name) is not "the same artist" as every other unknown one: give it its own code,
            # otherwise exclude_artist would blank out the entire catalogue of a metadata-free bundle
            self._artist_code[i] = codes.setdefault(a, len(codes)) if a else -(i + 1)
        self._key_index: dict[tuple[str, str], list[int]] = {}
        for i, key in enumerate(zip(self._artist_norm, self._title_norm)):
            self._key_index.setdefault(key, []).append(i)
        self._tok_index: dict[str, list[int]] | None = None

        # modes / poles
        self.modes: list[dict] = list((modes or {}).get("modes", []))
        for m in self.modes:
            mem = np.asarray(m.get("members", []), dtype=np.int64)
            if len(mem) and (mem.min() < 0 or mem.max() >= len(E)):
                raise ValueError(f"mode {m.get('mode_id')} has member ids outside 0..{len(E) - 1}")
            m["members"] = mem.tolist()
        self._mode_pos = {int(m["mode_id"]): i for i, m in enumerate(self.modes)}
        self.mode_poles = mode_poles_from_members(self.U, [m["members"] for m in self.modes]) if self.modes else \
            np.zeros((0, self.dim), np.float32)
        self.factor_poles = np.zeros((0, self.dim), np.float32) if factor_poles is None else \
            np.asarray(factor_poles, dtype=np.float32)
        self._sup_meta = list((supervised_meta or {}).get("poles", []))
        self.supervised_poles = np.zeros((0, self.dim), np.float32) if supervised_poles is None else \
            np.asarray(supervised_poles, dtype=np.float32)
        if len(self._sup_meta) != len(self.supervised_poles):
            raise ValueError("supervised_poles.json and supervised_poles.npy disagree on the number of poles")
        for name, arr in (("factor_poles", self.factor_poles), ("supervised_poles", self.supervised_poles)):
            if len(arr) and arr.shape[1] != self.dim:
                raise ValueError(f"{name} have dim {arr.shape[1]}, embeddings have dim {self.dim}")
        self._primary_mode: np.ndarray | None = None
        self._member_cache: list[set] | None = None

    # ---- loading --------------------------------------------------------------------------------------------------------
    @classmethod
    def from_pretrained(cls, path_or_dir: str | Path) -> "Melomane":
        """Load a bundle directory (or its ``embeddings.safetensors``).  A ``org/name`` hub id is downloaded if
        ``huggingface_hub`` happens to be installed; otherwise a local directory is required."""
        p = Path(path_or_dir)
        if p.is_file():
            p = p.parent
        if not p.is_dir():
            p = cls._maybe_download(str(path_or_dir))
        emb = p / "embeddings.safetensors"
        if not emb.exists():
            raise FileNotFoundError(f"{emb} not found: {p} is not a Melomane release bundle")
        tensors = load_file(str(emb))
        E = dequantise(tensors)
        vocab = json.loads((p / "vocab.json").read_text(encoding="utf-8"))
        if isinstance(vocab, dict):                      # tolerate {"tracks": [...]}
            vocab = vocab.get("tracks", [])
        modes = cls._read_json(p / "modes.json")
        sup_meta = cls._read_json(p / "supervised_poles.json")
        sup = np.load(p / "supervised_poles.npy") if (p / "supervised_poles.npy").exists() else None
        if sup_meta is None or sup is None:
            sup_meta, sup = None, None
        fac = np.load(p / "factor_poles.npy") if (p / "factor_poles.npy").exists() else None
        config = cls._read_json(p / "config.json") or {}
        return cls(E, vocab, modes=modes, factor_poles=fac, supervised_meta=sup_meta, supervised_poles=sup,
                   config=config, projection=tensors.get("pca_components"), projection_mean=tensors.get("pca_mean"), path=p)

    @staticmethod
    def _read_json(path: Path) -> dict | None:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    @staticmethod
    def _maybe_download(repo_id: str) -> Path:
        try:
            from huggingface_hub import snapshot_download          # type: ignore
        except ImportError as e:
            raise FileNotFoundError(
                f"{repo_id!r} is not a local bundle directory (and huggingface_hub is not installed to fetch it)") from e
        return Path(snapshot_download(repo_id))                    # pragma: no cover - network

    # ---- sizes / projection ---------------------------------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.vocab)

    @property
    def model_name(self) -> str:
        return str(self.config.get("variant") or self.config.get("model") or (self.path.name if self.path else "melomane"))

    def project(self, x: np.ndarray, *, is_direction: bool = True) -> np.ndarray:
        """Map full-space vector(s) into this bundle's space (identity if the bundle is not PCA-reduced).  Directions
        (poles) are not mean-shifted; points are."""
        x = np.asarray(x, dtype=np.float32)
        if self.projection is None:
            return x
        if x.shape[-1] != self.projection.shape[1]:
            return x
        if not is_direction and self.projection_mean is not None:
            x = x - self.projection_mean
        return x @ self.projection.T

    # ---- name resolution ------------------------------------------------------------------------------------------------
    def _search_index(self) -> dict[str, list[int]]:
        if self._tok_index is None:
            idx: dict[str, list[int]] = {}
            for i in range(len(self)):
                for tok in set(_tokens(self._artist_norm[i] + " " + self._title_norm[i])):
                    idx.setdefault(tok[:3], []).append(i)
            self._tok_index = idx
        return self._tok_index

    def find(self, query: str, k: int = 10) -> list[dict]:
        """Fuzzy artist/title lookup, normalised like the curation stage (case, accents, 'The', feat., remaster noise).
        Accepts ``"Artist - Title"``, ``"Title"`` or free text.  Returns up to ``k`` dicts with a ``score`` in [0, 1]."""
        q = str(query).strip()
        if not q or k <= 0:
            return []
        a_part = t_part = None
        if " - " in q:
            a_part, t_part = (normalize_artist(s) for s in q.split(" - ", 1))
        qn = re.sub(r"^the\s+", "", normalize_title(q))
        qtoks = _tokens(qn)
        if not qtoks:
            return []
        index = self._search_index()
        cand: dict[int, int] = {}
        for tok in set(qtoks):
            for i in index.get(tok[:3], ()):
                cand[i] = cand.get(i, 0) + 1
        ids = np.fromiter(cand.keys(), dtype=np.int64) if cand else np.arange(len(self))
        if len(ids) > 4000:                                           # keep the best token overlaps only
            counts = np.array([cand[i] for i in ids])
            ids = ids[np.argsort(-counts, kind="stable")[:4000]]
        scored = []
        for i in ids.tolist():
            an, tn = self._artist_norm[i], self._title_norm[i]
            comb = f"{an} {tn}".strip()
            ctoks = set(_tokens(comb))
            contain = sum(1 for t in qtoks if t in ctoks) / len(qtoks)
            ratio = max(difflib.SequenceMatcher(None, qn, comb).ratio(), difflib.SequenceMatcher(None, qn, tn).ratio())
            score = 0.5 * contain + 0.5 * ratio
            if a_part is not None:
                score = 0.5 * difflib.SequenceMatcher(None, a_part, an).ratio() + \
                    0.5 * difflib.SequenceMatcher(None, t_part, tn).ratio()
            scored.append((score, i))
        scored.sort(key=lambda s: (-s[0], s[1]))
        return [dict(self._track(i), score=round(float(s), 4)) for s, i in scored[:k]]

    def resolve(self, name_or_cid: Any) -> int:
        """Track name / ``cid:<n>`` / integer cid -> cid.  Exact ``"Artist - Title"`` (normalised) wins; otherwise the
        best fuzzy match is accepted only if it is clearly ahead of the runner-up, else :class:`TrackNotFound`."""
        if isinstance(name_or_cid, dict) and "cid" in name_or_cid:
            name_or_cid = name_or_cid["cid"]
        if isinstance(name_or_cid, (int, np.integer)) and not isinstance(name_or_cid, bool):
            cid = int(name_or_cid)
            if not 0 <= cid < len(self):
                raise TrackNotFound(f"cid {cid} out of range 0..{len(self) - 1}")
            return cid
        s = str(name_or_cid).strip()
        if re.fullmatch(r"cid:\d+", s):
            return self.resolve(int(s[4:]))
        if " - " in s:
            a, t = s.split(" - ", 1)
            hit = self._key_index.get((normalize_artist(a), normalize_title(t)))
            if hit:
                return hit[0]
        tn = normalize_title(s)
        title_hits = [i for i, t in enumerate(self._title_norm) if t == tn] if tn else []
        if len(title_hits) == 1:
            return title_hits[0]
        cands = self.find(s, k=5)
        if cands and cands[0]["score"] >= 0.72 and (len(cands) == 1 or cands[0]["score"] - cands[1]["score"] >= 0.05):
            return int(cands[0]["cid"])
        msg = f"could not resolve {s!r} to a single track"
        if cands:
            msg += "; closest: " + "; ".join(f"{c['artist']} - {c['title']} (cid {c['cid']})" for c in cands[:3])
        raise TrackNotFound(msg, cands)

    # ---- track views ----------------------------------------------------------------------------------------------------
    def _modes_of_primary(self) -> np.ndarray:
        if self._primary_mode is None:
            self._primary_mode = primary_mode_assignment(self.U, [m["members"] for m in self.modes], self.mode_poles) \
                if self.modes else np.full(len(self), -1, dtype=np.int32)
        return self._primary_mode

    def _mode_label(self, i: int) -> str | None:
        m = int(self._modes_of_primary()[i])
        return None if m < 0 else self.modes[m].get("label")

    def _track(self, i: int, score: float | None = None, **extra: Any) -> dict:
        t = self.vocab[i]
        out = {"cid": i, "artist": t["artist"], "title": t["title"], "category": t["category"], "year": t["year"],
               "mode": self._mode_label(i)}
        if score is not None:
            out["score"] = round(float(score), 4)
        out.update(extra)
        return out

    def vec(self, name_or_cid: Any) -> np.ndarray:
        """The stored embedding row (float32 copy, in this bundle's space)."""
        return self.E[self.resolve(name_or_cid)].copy()

    def describe(self, name: Any) -> dict:
        """Metadata for one track: artist, title, album, year, category, region tags, mode label(s).  Never lyrics or audio."""
        i = self.resolve(name)
        t = self.vocab[i]
        memberships = []
        for m, mode in enumerate(self.modes):
            if i in self._member_sets()[m]:
                memberships.append({"mode_id": mode["mode_id"], "label": mode.get("label"),
                                    "cos": round(float(self.U[i] @ self.mode_poles[m]), 4)})
        memberships.sort(key=lambda d: -d["cos"])
        return {"cid": i, "artist": t["artist"], "title": t["title"], "album": t["album"], "year": t["year"],
                "category": t["category"], "region_tags": t["region_tags"], "mode": self._mode_label(i),
                "modes": memberships, "closest_mode": self.closest_mode(i)}

    def _member_sets(self) -> list[set]:
        if self._member_cache is None:
            self._member_cache = [set(m["members"]) for m in self.modes]
        return self._member_cache

    # ---- retrieval ------------------------------------------------------------------------------------------------------
    def _exclusions(self, seed: int, exclude_self: bool, exclude_artist: bool) -> np.ndarray:
        mask = np.zeros(len(self), dtype=bool)
        if exclude_artist:
            mask |= self._artist_code == self._artist_code[seed]
            mask[seed] = exclude_self
        elif exclude_self:
            mask[seed] = True
        return np.flatnonzero(mask)

    def _search(self, q: np.ndarray, k: int, exclude: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n_ok = len(self) - len(exclude)
        k_eff = min(int(k), n_ok)
        if k_eff <= 0:
            return np.zeros(0, np.int64), np.zeros(0, np.float32)
        if k_eff >= len(self):                          # nearest() needs k < N
            sims = self.U @ unit(q)
            idx = np.argsort(-sims, kind="stable")[:k_eff]
            return idx, sims[idx]
        idx, sims = nearest(self.U, q, k_eff, exclude=exclude.tolist() if len(exclude) else None)
        ok = np.isfinite(sims)
        return idx[ok], sims[ok]

    def neighbors(self, name: Any, k: int = 10, exclude_self: bool = True, exclude_artist: bool = True) -> list[dict]:
        """Top-``k`` cosine neighbours of a track, artist-disjoint by default (the seed's own artist is excluded)."""
        i = self.resolve(name)
        idx, sims = self._search(self.U[i], k, self._exclusions(i, exclude_self, exclude_artist))
        return [self._track(int(j), s) for j, s in zip(idx, sims)]

    # ---- poles / directions ---------------------------------------------------------------------------------------------
    def list_supervised_poles(self) -> list[dict]:
        """Supervised poles: ``[{"name","kind","n_pos","n_neg"}]`` (row order of ``supervised_poles.npy``)."""
        return [{k: _clean(v) for k, v in p.items()} for p in self._sup_meta]

    def list_poles(self) -> list[dict]:
        """Every addressable pole: supervised (by name), ``mode:<id>`` and ``factor:<i>``."""
        out = [{"name": p["name"], "group": "supervised", "kind": p.get("kind")} for p in self._sup_meta]
        out += [{"name": f"mode:{m['mode_id']}", "group": "mode", "kind": "emergent", "label": m.get("label")}
                for m in self.modes]
        out += [{"name": f"factor:{i}", "group": "factor", "kind": "emergent"} for i in range(len(self.factor_poles))]
        return out

    def pole(self, name: str) -> np.ndarray:
        """Unit pole vector by name (see module docstring for the accepted forms)."""
        s = str(name).strip()
        m = re.fullmatch(r"(mode|factor):(\d+)", s)
        if m:
            n = int(m.group(2))
            if m.group(1) == "mode":
                if n not in self._mode_pos:
                    raise KeyError(f"unknown mode id {n}")
                return self.mode_poles[self._mode_pos[n]].copy()
            if n >= len(self.factor_poles):
                raise KeyError(f"unknown factor {n} (bundle has {len(self.factor_poles)})")
            return unit(self.factor_poles[n]).astype(np.float32)
        names = [p["name"] for p in self._sup_meta]
        if s in names:
            return unit(self.supervised_poles[names.index(s)]).astype(np.float32)
        alias = [i for i, n in enumerate(names) if n.split(":", 1)[-1].lower() == s.lower()]
        if len(alias) == 1:
            return unit(self.supervised_poles[alias[0]]).astype(np.float32)
        near = difflib.get_close_matches(s, names, n=3, cutoff=0.4)
        raise KeyError(f"unknown pole {s!r}" + (f"; did you mean {near}?" if near else ""))

    def _direction(self, direction: Any, weights: Sequence[float] | None) -> np.ndarray:
        items = list(direction) if isinstance(direction, (list, tuple)) else [direction]
        if not items:
            raise ValueError("direction list is empty")
        if weights is not None and len(weights) != len(items):
            raise ValueError(f"{len(weights)} weights for {len(items)} directions")
        vecs = []
        for d in items:
            if isinstance(d, str):
                vecs.append(self.vec(d[6:]) if d.startswith("track:") else self.pole(d))
            else:
                a = np.asarray(d, dtype=np.float32)
                if a.ndim != 1:
                    raise ValueError("a direction vector must be 1-D")
                if a.shape[0] != self.dim:
                    if self.projection is not None and a.shape[0] == self.projection.shape[1]:
                        a = self.project(a)
                    else:
                        raise ValueError(f"direction has dim {a.shape[0]}, bundle has dim {self.dim}")
                vecs.append(a)
        return vecs[0] if len(vecs) == 1 else compose_poles(vecs, None if weights is None else list(weights))

    def slerp_vector(self, seed: Any, direction: Any, theta_deg: float, weights: Sequence[float] | None = None) -> np.ndarray:
        """The rotated query ``q`` (unit): the seed turned ``theta_deg`` toward the (composed) direction.
        ``cos(q, seed) = cos(theta)`` exactly."""
        if not 0.0 <= float(theta_deg) <= 180.0:
            raise ValueError("theta_deg must be within [0, 180] (the explorer uses 0-90)")
        return _slerp(self.U[self.resolve(seed)], self._direction(direction, weights), float(theta_deg))

    def slerp(self, seed: Any, direction: Any, theta_deg: float, k: int = 10, weights: Sequence[float] | None = None,
              exclude_artist: bool = True) -> list[dict]:
        """Steer from a seed toward a pole / list of poles by ``theta_deg`` and return the ``k`` nearest tracks
        (seed always excluded; seed's artist too if ``exclude_artist``).  Each row has ``score`` = cos to the query and
        ``cos_seed`` = cos to the seed."""
        i = self.resolve(seed)
        q = self.slerp_vector(i, direction, theta_deg, weights)
        idx, sims = self._search(q, k, self._exclusions(i, True, exclude_artist))
        return [self._track(int(j), s, cos_seed=round(float(self.U[j] @ self.U[i]), 4)) for j, s in zip(idx, sims)]

    # ---- modes ----------------------------------------------------------------------------------------------------------
    def list_modes(self) -> list[dict]:
        """Mode atlas (without member lists): id, label, property, driver, factor, size, coherence."""
        keys = ("mode_id", "label", "property", "driver", "factor", "size", "coherence", "bic_k")
        return [{k: _clean(m.get(k)) for k in keys} for m in self.modes]

    def closest_mode(self, name: Any) -> dict | None:
        """Mode whose pole is nearest (cosine) to the track; ``member`` says whether the track is actually in that mode."""
        if not self.modes:
            return None
        i = self.resolve(name)
        c = self.mode_poles @ self.U[i]
        m = int(np.argmax(c))
        mode = self.modes[m]
        return {"mode_id": mode["mode_id"], "label": mode.get("label"), "cos": round(float(c[m]), 4),
                "member": i in self._member_sets()[m]}

    def mode_members(self, mode_id: int) -> list[dict]:
        """Members of a mode, most pole-like first (``score`` = cos to the mode pole)."""
        if int(mode_id) not in self._mode_pos:
            raise KeyError(f"unknown mode id {mode_id}")
        m = self._mode_pos[int(mode_id)]
        mem = np.asarray(self.modes[m]["members"], dtype=np.int64)
        if not len(mem):
            return []
        c = self.U[mem] @ self.mode_poles[m]
        order = np.argsort(-c, kind="stable")
        return [self._track(int(mem[j]), c[j]) for j in order]
