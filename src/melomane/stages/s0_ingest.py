"""S0-ingest: raw data -> Corpus contract, per corpus (onion | playlist2vec | synth | trackb).  Plus S0-enrich (partial-hub)."""
from __future__ import annotations
import os
from pathlib import Path

from ..corpus import read_corpus, write_corpus, write_enrichment
from ..pipeline import Ctx, Stage, StageResult, StageSkip, register, GREEN, PARTIAL
from ..runtime import disk_free_gb, get_logger, read_json, write_json

log = get_logger("melomane.ingest")


def _limit(ctx: Ctx):
    return ctx.profile.get("limit_tracks") if isinstance(ctx.profile, dict) else None


@register
class Ingest(Stage):
    name, title = "S0-ingest", "Ingest raw data into the corpus contract"

    def params(self, ctx):
        cfg = ctx.ws.config("data").get(ctx.corpus, {})
        p = {"v": 3, "cfg": cfg, "limit": _limit(ctx)}
        if ctx.corpus == "onion":                    # a different/updated metadata table must re-run the ingest (artist ids!)
            from ..adapters import onion
            prov = onion.default_raw_dir(ctx.ws) / "METADATA_PROVENANCE.json"
            p["metadata_sha256"] = read_json(prov).get("source_sha256") if prov.exists() else None
        if ctx.corpus == "synth":
            p["demo"] = ctx.opt("demo_size", 2400)
        return p

    def outputs(self, ctx):
        return [ctx.corpus_dir / "corpus.json"]

    def run(self, ctx: Ctx) -> StageResult:
        ws, cfg = ctx.ws, ctx.ws.config("data")
        res = StageResult()
        if ctx.corpus == "synth":
            from ..demo import make_demo_corpus
            n = ctx.opt("demo_size", 2400)
            c = make_demo_corpus(n_tracks=n, n_users=max(n * 5 // 3, 800), seed=ctx.seed("demo") % 1000)
            write_corpus(c, ctx.corpus_dir)
            res.notes.append("DEMO corpus (synthetic, planted structure) - results are not music findings")
        elif ctx.corpus == "playlist2vec":
            from ..adapters import playlist2vec as p2v
            pc = cfg.get("playlist2vec", {})
            dirs = [Path(d) for d in pc.get("search_dirs", [])] + [ws.d("raw", "playlist2vec", base="data")]
            env = os.environ.get("PLAYLIST2VEC_SQL")
            st = p2v.find_dump(dirs, env)
            if st.path is None:
                raise StageSkip("Playlist2vec dump not found. Save the Zenodo SQL dump anywhere in Downloads or data/raw/playlist2vec "
                                "(or set PLAYLIST2VEC_SQL=path in .env), then re-run.")
            if not st.complete:
                log.info("Playlist2vec download still in progress (%.2f GB) - waiting for it to finish ...", st.size / 1e9)
                st = p2v.wait_for_dump(dirs, env, timeout_h=pc.get("wait_for_download_hours", 24))
                if not st.complete:
                    raise StageSkip(f"Playlist2vec download not complete ({st.size / 1e9:.2f} GB seen); re-run when it has finished")
            if disk_free_gb(ws.data) < 30:
                res.notes.append("low disk space for the Playlist2vec corpus (< 30 GB free)")
            c = p2v.ingest(ws, st.path, pc.get("min_playlist_len", 5), pc.get("max_playlist_len", 300))
            res.decisions.append(f"Playlist2vec used as a documents=playlists corpus from {st.path.name}")
        elif ctx.corpus == "onion":
            from .. import download
            from ..adapters import onion
            oc = cfg.get("onion", {})
            raw = onion.default_raw_dir(ws)
            have = onion.FileIndex(raw) if raw.exists() else None
            listening_file = (have.find_file("userid_trackid_timestamp") or have.find_file("userid_trackid_count")) if have else None
            feature_file = (have.find_file("id_essentia") or have.find_file("id_mfcc_stats")) if have else None
            if have is None or not listening_file or not feature_file:
                if disk_free_gb(ws.data) < oc.get("min_free_gb", 60):
                    raise RuntimeError(f"Music4All-Onion needs ~{oc.get('min_free_gb', 60)} GB free; only {disk_free_gb(ws.data):.0f} GB on the data drive")
                last = None
                for rec in oc.get("zenodo_records", [15394646, 6609677]):
                    try:
                        download.fetch_zenodo(rec, raw)
                        res.decisions.append(f"Music4All-Onion downloaded from Zenodo record {rec}")
                        break
                    except Exception as e:  # noqa: BLE001
                        last = e; log.warning("Zenodo record %s failed: %s", rec, e)
                else:
                    raise RuntimeError(f"could not download Music4All-Onion: {last}")
            inv = onion.inventory(raw, count_rows=False)
            write_json(raw / "INVENTORY.json", inv)
            kw = {}
            if _limit(ctx):
                kw["limit_tracks"] = int(_limit(ctx))
            c = onion.ingest(ws, raw, **kw)
            if c.meta.get("blocker_a"):
                res.status = PARTIAL
                res.questions.append("Blocker-A: the open Onion release has no artist/title. Curation/annotation fall back to tags only. "
                                     "Send docs/email_music4all.md to get the original Music4All metadata (optional).")
        elif ctx.corpus == "mbid_stack":
            p = ctx.corpus_dir / "corpus.json"
            if not p.exists():
                raise StageSkip("mbid_stack corpus not ingested yet: run S10 with --with-track-b")
            c = read_corpus(ctx.corpus_dir)
            res.notes.append("mbid_stack corpus read from S10 ingest output")
        else:
            raise StageSkip(f"no ingest recipe for corpus '{ctx.corpus}' (track B/C are separate stages)")
        res.metrics = {"tracks": c.n_tracks, "sessions": int(c.sessions.shape[0]), "session_track_links": int(c.sessions.nnz),
                       "families": len(c.families), "has_measured": c.measured is not None}
        return res


@register
class Enrich(Stage):
    name, title, deps = "S0-enrich", "Match tracks to a feature corpus (partial-hub compounds)", ("S0-ingest",)

    def params(self, ctx):
        return {"v": 1, "match": ctx.ws.config("data").get("matching", {}), "source": ctx.opt("enrich_source", "onion")}

    def outputs(self, ctx):
        return [ctx.corpus_dir / "enrich.json"]

    def run(self, ctx: Ctx) -> StageResult:
        from ..adapters import matching
        src_name = ctx.opt("enrich_source", "onion")
        if ctx.corpus == src_name or ctx.corpus in ("synth",):
            write_json(ctx.corpus_dir / "enrich.json", {"skipped": "corpus carries its own features"})
            return StageResult(notes=["feature corpus: nothing to enrich"])
        sd = ctx.ws.d("interim", src_name, base="data")
        if not (sd / "corpus.json").exists():
            write_json(ctx.corpus_dir / "enrich.json", {"skipped": f"{src_name} not ingested"})
            raise StageSkip(f"feature corpus '{src_name}' not ingested; {ctx.corpus} runs Cooc-only")
        tgt, src = read_corpus(ctx.corpus_dir, load_families=False), read_corpus(sd)
        if src.meta.get("blocker_a"):
            write_json(ctx.corpus_dir / "enrich.json", {"skipped": "source has no artist/title"})
            raise StageSkip("feature corpus has no artist/title (Blocker-A): cannot match by name; Cooc-only")
        mc = ctx.ws.config("data").get("matching", {})
        m = matching.match_tracks(tgt.tracks, src.tracks, level=mc.get("level", "strict"), fuzzy=mc.get("fuzzy", True), threshold=mc.get("threshold", 0.85))
        enriched = matching.enrich(tgt, src, m)
        write_enrichment(enriched, ctx.corpus_dir)
        rep = matching.coverage_report(tgt, m, min_sessions=20)
        rep["match_counts"] = dict(m.attrs.get("counts", {}))
        write_json(ctx.corpus_dir / "enrich.json", rep)
        res = StageResult(metrics={k: v for k, v in rep.items() if isinstance(v, (int, float))})
        if not len(m):
            res.status = PARTIAL; res.notes.append("no tracks matched: Composition/Core unavailable for this corpus")
        return res
