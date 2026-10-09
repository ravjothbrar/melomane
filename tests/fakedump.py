"""Generates a small mysqldump-format file that mimics the Playlist2vec schema (incl. nasty strings)."""
from __future__ import annotations
import gzip, random
from pathlib import Path
import numpy as np

SCHEMA = open(Path(__file__).with_name("playlist2vec_schema.sql"), encoding="utf-8").read()


def q(s):
    if s is None:
        return "NULL"
    return "'" + s.replace("\\", "\\\\").replace("'", "\\'").replace("\n", "\\n") + "'"


def sid(prefix, i):
    return f"{prefix}{i:016d}"[:22]


def make_dump(path: Path, n_tracks=300, n_playlists=120, n_artists=40, seed=0, gz=False, complete=True, mojibake=False):
    rng = random.Random(seed)
    nasty = ["Don't Stop (Believin')", "Back\\Slash, (Live)", "Comma, Comma; Semi", "Paren) (Weird", "Ünïcödé — 日本語", "Tab\there",
             "Quote\"Double", "Trailing\\", "100% Pure (Remix)"]
    albums = [(sid("al", i), f"Album {i} " + rng.choice(nasty), f"spotify:album:{sid('al', i)}") for i in range(n_tracks // 3)]
    artists = [(sid("ar", i), f"Artist {i}" + (" & Co" if i % 7 == 0 else ""), f"spotify:artist:{sid('ar', i)}") for i in range(n_artists)]
    tracks = []
    for i in range(n_tracks):
        title = rng.choice(nasty) + f" #{i}" if i % 5 == 0 else f"Song {i}"
        tracks.append((sid("tr", i), title, rng.randint(100000, 400000), round(rng.random() * 100, 1),
                       rng.choice(["0", "1", "True", "False"]) if i % 9 else None, f"https://p.scdn.co/mp3-preview/{i}" if i % 3 else None,
                       f"spotify:track:{sid('tr', i)}", albums[i % len(albums)][0]))
    playlists = [(sid("pl", j), f"Playlist {j} " + rng.choice(["chill", "workout", "sad songs", "throwback 90s"]), rng.choice([None, 0, 5, 12345]),
                  f"spotify:playlist:{sid('pl', j)}", 0) for j in range(n_playlists)]
    ta = [(t[0], artists[i % n_artists][0]) for i, t in enumerate(tracks)] + [(tracks[i][0], artists[(i + 1) % n_artists][0]) for i in range(0, n_tracks, 10)]
    tp = []
    for j, p in enumerate(playlists):
        # taste clusters: playlist j draws from tracks congruent to j%4 (structure for downstream tests)
        pool = [i for i in range(n_tracks) if i % 4 == j % 4]
        k = rng.randint(3, 40) if j % 17 else 500 % n_tracks  # a few very short / very long playlists to test the filters
        for i in rng.sample(pool, min(k, len(pool))):
            tp.append((tracks[i][0], p[0]))
    if len(tp) > 3:
        tp.append(tp[0])          # duplicate link row must not double count
    tp.append(("trNOTINTABLE00000000", playlists[0][0]))   # dangling FK: must be dropped, not crash

    def ins(table, rows, per=50):
        out = []
        for i in range(0, len(rows), per):
            chunk = rows[i:i + per]
            out.append(f"INSERT INTO `{table}` VALUES " + ",".join("(" + ",".join(
                q(v) if isinstance(v, str) or v is None else str(v) for v in r) + ")" for r in chunk) + ";")
        return "\n".join(out) + "\n"

    def blk(table, rows):
        return f"LOCK TABLES `{table}` WRITE;\n/*!40000 ALTER TABLE `{table}` DISABLE KEYS */;\n{ins(table, rows)}/*!40000 ALTER TABLE `{table}` ENABLE KEYS */;\nUNLOCK TABLES;\n"

    txt = SCHEMA.split("-- Dump completed")[0]
    # splice data after each CREATE block by appending at end (parser is order-agnostic for CREATE vs INSERT)
    body = blk("album", albums) + blk("artist", artists) + blk("playlist", playlists) + blk("track", tracks) + blk("track_artist1", ta) + blk("track_playlist1", tp)
    txt = txt + body + ("-- Dump completed on 2021-02-11 15:42:39\n" if complete else "")
    if mojibake:
        w = "Ünïcödé — 日本語"; txt = txt.replace(w, w.encode("utf-8").decode("latin-1"))
    data = txt.encode("utf-8")
    if gz:
        path.write_bytes(gzip.compress(data))
    else:
        path.write_bytes(data)
    return dict(albums=albums, artists=artists, tracks=tracks, playlists=playlists, ta=ta, tp=tp)
