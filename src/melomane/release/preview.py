"""Social-preview image (1200 x 630) for the explorer, drawn from the explorer's own payload.

The picture is the real map (the model's 2-D layout, one dot per track, coloured by genre with the explorer's palette) behind the pixel-block
wordmark and the crotchet.  Nothing is invented: it is a deterministic function of ``<explorer>/data``.  Used for the ``og:image`` /
``twitter:image`` tags so links shared on LinkedIn, Hacker News and elsewhere get a proper card.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# the explorer's palette (apps/explorer/math.js GLOW): categories ranked by size get the most distinct hues
GLOW = [(255, 107, 107), (255, 169, 77), (255, 212, 59), (169, 227, 75), (81, 207, 102), (32, 201, 151),
        (34, 184, 207), (77, 171, 247), (116, 143, 252), (151, 117, 250), (218, 119, 242), (247, 131, 172)]

# 5x7 pixel font (same glyphs as the explorer's logo)
GLYPHS = {
    "M": ["X...X", "XX.XX", "X.X.X", "X.X.X", "X...X", "X...X", "X...X"],
    "E": ["XXXXX", "X....", "X....", "XXXX.", "X....", "X....", "XXXXX"],
    "O": [".XXX.", "X...X", "X...X", "X...X", "X...X", "X...X", ".XXX."],
    "A": ["..X..", ".X.X.", "X...X", "X...X", "XXXXX", "X...X", "X...X"],
    "N": ["X...X", "XX..X", "XX..X", "X.X.X", "X..XX", "X..XX", "X...X"],
}
WORD = ["M", "E", "NOTE", "O", "M", "A", "N", "E"]
NOTE_W = 3          # the crotchet occupies three pixel columns in the wordmark


def word_layout() -> tuple[list[tuple[int, int]], int, int]:
    """Filled pixel cells (col, row) of the wordmark with the crotchet slot left empty; returns (cells, total_cols, note_col)."""
    cells, col, note_col = [], 0, 0
    for ch in WORD:
        if ch == "NOTE":
            note_col = col
            col += NOTE_W + 1
            continue
        for r, row in enumerate(GLYPHS[ch]):
            for c, v in enumerate(row):
                if v == "X":
                    cells.append((col + c, r))
        col += 5 + 1
    return cells, col - 1, note_col


def draw_wordmark(ax, left: float, top: float, ps: float, z: int = 5) -> tuple[float, float]:
    """The MELOMANE wordmark (pixel-block letters, purple gradient, retro offset shadow, crotchet in place of the L).

    ``left``/``top`` locate the top-left corner, ``ps`` is the pixel size; returns the (width, height) drawn, in axes units.
    """
    from matplotlib.patches import Ellipse, Rectangle
    cells, cols, note_col = word_layout()

    def box(c, r, color, dx=0.0, dy=0.0, zz=z):
        ax.add_patch(Rectangle((left + c * ps + dx, top - (r + 1) * ps + dy), ps * 0.94, ps * 0.94, color=color, lw=0, zorder=zz))
    for (c, r) in cells:
        box(c, r, "#2e1065", ps * 0.44, -ps * 0.44, z - 1)
        box(c, r, "#5b21b6", ps * 0.226, -ps * 0.226, z - 1)
    grad = ["#f0b6ff", "#e2a0ff", "#d08aff", "#bd6dff", "#a955ff", "#9140f0", "#7b2ede"]
    for (c, r) in cells:
        box(c, r, grad[r], 0, 0, z)
    nx = left + note_col * ps + ps * 0.1
    ax.add_patch(Rectangle((nx + ps * 2.35, top - 7 * ps + ps * 1.0), ps * 0.55, ps * 6.1, color="#ff8be0", lw=0, zorder=z + 1))
    ax.add_patch(Ellipse((nx + ps * 1.45, top - 7 * ps + ps * 0.95), ps * 2.5, ps * 1.8, angle=22, color="#ff8be0", lw=0, zorder=z + 1))
    ax.add_patch(Ellipse((nx + ps * 1.45, top - 7 * ps + ps * 0.95), ps * 3.4, ps * 2.7, angle=22, color="#ff8be0", alpha=0.18, lw=0, zorder=z))
    return cols * ps, 7 * ps


# mode highlights for the README hero: purple-led, one cool accent
HERO_COLOURS = ["#c9a7ff", "#ff6fd8", "#7c8cff", "#5fe3f5", "#ff9ab8", "#e8d9ff"]


def pick_hero_modes(xy: np.ndarray, modes: list[dict], k: int = 6, min_members: int = 400, max_spread: float = 0.45,
                    min_sep: float = 0.28) -> list[dict]:
    """The ``k`` most coherent compact modes whose centroids are well separated on the 2-D map and whose names are distinct.

    Deterministic: a pure function of the payload.  Each result carries ``label``, ``centroid`` (median member position) and ``members``
    (valid payload indices).  ``min_sep`` and ``max_spread`` are in the layout's own units, relative to the layout's overall spread.
    """
    scale = float(np.sqrt(xy.var(0).sum())) or 1.0
    cands = []
    for m in modes:
        mem = np.array([i for i in m["members"] if i < len(xy)], dtype=int)
        label = str(m.get("label", "")).strip()
        if len(mem) < min_members or not label or label.lower() == "unclear":
            continue
        p = xy[mem]
        spread = float(np.sqrt(p.var(0).sum())) / scale
        if spread > max_spread:
            continue
        cands.append({"id": m["id"], "label": label, "members": mem, "centroid": np.median(p, axis=0),
                      "coherence": float(m.get("coherence", 0.0)), "spread": spread})
    cands.sort(key=lambda c: (-c["coherence"], c["id"]))
    chosen: list[dict] = []
    for c in cands:
        if all(np.linalg.norm(c["centroid"] - o["centroid"]) / scale >= min_sep and c["label"].lower() != o["label"].lower() for o in chosen):
            chosen.append(c)
        if len(chosen) == k:
            break
    return chosen


def render_hero(explorer_dir: str | Path, out_png: str | Path, stats: list[tuple[str, str]], model: str = "core",
                caption: str = "Music, mapped by how it is listened to and what it is made of.") -> Path:
    """README hero card (1600 x 900): the real map with its most coherent named modes picked out, and a row of headline numbers.

    ``stats`` is a list of ``(value, caption)`` supplied by the caller from files in ``results/`` or the payload manifest; nothing is typed here.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse, FancyBboxPatch, Rectangle

    d = Path(explorer_dir) / "data"
    man = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    names = [m["name"] for m in man["models"]]
    model = model if model in names else names[0]
    tracks = json.loads((d / "tracks.json").read_text(encoding="utf-8"))
    xy = np.fromfile(d / model / "umap.bin", dtype="<f4").reshape(-1, 2)
    modes = json.loads((d / model / "modes.json").read_text(encoding="utf-8"))
    chosen = pick_hero_modes(xy, modes)

    W, H, DPI = 1600, 900, 100
    BG = "#0d0a18"
    fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI, facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
    yy, xx = np.mgrid[0:H, 0:W]
    glow = np.exp(-(((xx - W * 0.5) / (W * 0.5)) ** 2 + ((yy - H * 0.6) / (H * 0.55)) ** 2))
    bg = np.zeros((H, W, 3)); bg[..., 0] = 0.05 + 0.09 * glow; bg[..., 1] = 0.035 + 0.04 * glow; bg[..., 2] = 0.10 + 0.17 * glow
    ax.imshow(bg, extent=[0, W, 0, H], origin="lower", aspect="auto", zorder=0)

    # map box: aspect-preserving fit of the 2nd-98th percentile frame into the middle of the card
    x0, x1 = np.percentile(xy[:, 0], [1, 99]); y0, y1 = np.percentile(xy[:, 1], [1, 99])
    bx0, bx1, by0, by1 = 360, W - 360, 300, 765
    s = min((bx1 - bx0) / max(x1 - x0, 1e-6), (by1 - by0) / max(y1 - y0, 1e-6))
    cx, cy = (bx0 + bx1) / 2, (by0 + by1) / 2
    px = cx + (xy[:, 0] - (x0 + x1) / 2) * s
    py = cy + (xy[:, 1] - (y0 + y1) / 2) * s
    base = np.array([0.62, 0.55, 0.85])
    ax.scatter(px, py, s=9, c=[base], alpha=0.10, linewidths=0, zorder=1)
    ax.scatter(px, py, s=1.3, c=[base], alpha=0.38, linewidths=0, zorder=2)
    cents = []
    for j, c in enumerate(chosen):
        col = HERO_COLOURS[j % len(HERO_COLOURS)]
        m = c["members"]
        ax.scatter(px[m], py[m], s=60, c=col, alpha=0.05, linewidths=0, zorder=3)
        ax.scatter(px[m], py[m], s=18, c=col, alpha=0.13, linewidths=0, zorder=3)
        ax.scatter(px[m], py[m], s=2.6, c=col, alpha=0.95, linewidths=0, zorder=4)
        cents.append((cx + (c["centroid"][0] - (x0 + x1) / 2) * s, cy + (c["centroid"][1] - (y0 + y1) / 2) * s, col, c["label"]))

    # labels: left or right margin by centroid side; slots assigned by the permutation with the shortest, non-crossing leader lines
    from itertools import permutations

    def _cross(a, b, c, d):
        def o(p, q, r): return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
        return o(a, b, c) * o(a, b, d) < 0 and o(c, d, a) * o(c, d, b) < 0

    for side, group in ((0, [t for t in cents if t[0] < cx]), (1, [t for t in cents if t[0] >= cx])):
        n = len(group)
        if not n:
            continue
        slots = [by1 + 10 - (i + 0.5) * ((by1 - by0 + 20) / n) for i in range(n)]
        elbow = 58 + 22 if side == 0 else W - 58 - 22
        best, best_cost = None, None
        for perm in permutations(range(n)):
            segs = [((elbow, slots[i]), (group[perm[i]][0], group[perm[i]][1])) for i in range(n)]
            cost = sum(np.hypot(a[0] - b[0], a[1] - b[1]) for a, b in segs)
            cost += 4000 * sum(_cross(*segs[i], *segs[j]) for i in range(n) for j in range(i + 1, n))
            if best_cost is None or cost < best_cost:
                best, best_cost = perm, cost
        for i in range(n):
            mx, my, col, label = group[best[i]]
            ly = slots[i]
            w = 44 + 10.4 * len(label)
            lx0 = 58 if side == 0 else W - 58 - w
            ex = lx0 + w if side == 0 else lx0
            ax.plot([ex, ex + (22 if side == 0 else -22), mx], [ly, ly, my], color=col, lw=1.3, alpha=0.7, zorder=5, solid_capstyle="round")
            ax.scatter([mx], [my], s=22, c=col, zorder=6)
            ax.add_patch(FancyBboxPatch((lx0, ly - 17), w, 34, boxstyle="round,pad=0,rounding_size=7", fc="#17112b", ec=col, lw=1.0, alpha=0.96, zorder=6))
            ax.text(lx0 + w / 2, ly, label.upper(), color="#efe8ff", fontsize=11.2, family="monospace", ha="center", va="center", zorder=7)

    # header: the wordmark, large, with the model tag on the right
    draw_wordmark(ax, 64, H - 52, 8.2)
    label = next((m["label"] for m in man["models"] if m["name"] == model), model)
    n_modes = man["models"][names.index(model)].get("n_modes", len(modes))
    ax.text(W - 64, H - 52 - 3.5 * 8.2, f"{label.upper()} MODEL  ·  {len(chosen)} OF {n_modes} EMERGENT MODES SHOWN",
            color="#a99bd0", fontsize=12.5, family="monospace", ha="right", va="center", zorder=7)

    # caption and stats row
    ax.text(W / 2, 222, caption, color="#efe8ff", fontsize=25, fontweight="bold", ha="center", va="center", zorder=7)
    n = len(stats)
    colw = (W - 2 * 160) / max(n, 1)
    for i, (val, cap) in enumerate(stats):
        mx = 160 + colw * (i + 0.5)
        ax.text(mx, 120, val, color="#ffffff", fontsize=44, fontweight="bold", ha="center", va="center", zorder=7)
        ax.text(mx, 70, cap.upper(), color="#a99bd0", fontsize=12.5, family="monospace", ha="center", va="center", zorder=7)
        if i:
            ax.plot([160 + colw * i] * 2, [48, 140], color="#3a2f5c", lw=1.2, zorder=6)
    ax.add_patch(FancyBboxPatch((14, 14), W - 28, H - 28, boxstyle="round,pad=0,rounding_size=26", fc="none", ec="#2a2147", lw=1.6, zorder=8))
    out = Path(out_png)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=DPI * 2, facecolor=fig.get_facecolor())      # 3200 x 1800: crisp on retina displays
    plt.close(fig)
    return out


def render_preview(explorer_dir: str | Path, out_png: str | Path, model: str = "core", tagline: str | None = None) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse, Rectangle

    d = Path(explorer_dir) / "data"
    man = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    names = [m["name"] for m in man["models"]]
    model = model if model in names else names[0]
    tracks = json.loads((d / man["tracks"]["path"].split("/", 1)[-1]).read_text(encoding="utf-8")) if not (d / "tracks.json").exists() else \
        json.loads((d / "tracks.json").read_text(encoding="utf-8"))
    xy = np.fromfile(d / model / "umap.bin", dtype="<f4").reshape(-1, 2)
    cat = np.asarray(tracks["category_idx"])
    counts = np.bincount(cat[cat >= 0], minlength=len(tracks["categories"]))
    rank = np.empty(len(counts), int)
    rank[np.argsort(-counts)] = np.arange(len(counts))
    rgb = np.array([GLOW[rank[c] % len(GLOW)] if c >= 0 else (118, 130, 150) for c in cat], float) / 255.0

    W, H, DPI = 1200, 630, 100
    fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI, facecolor="#0d0a18")
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
    # soft pool of light
    yy, xx = np.mgrid[0:H, 0:W]
    glow = np.exp(-(((xx - W * 0.62) / (W * 0.45)) ** 2 + ((yy - H * 0.5) / (H * 0.7)) ** 2))
    bg = np.zeros((H, W, 3)); bg[..., 0] = 0.05 + 0.10 * glow; bg[..., 1] = 0.035 + 0.05 * glow; bg[..., 2] = 0.10 + 0.20 * glow
    ax.imshow(bg, extent=[0, W, 0, H], origin="lower", aspect="auto", zorder=0)

    # the map, framed like the explorer (2nd-98th percentile), on the right two thirds
    x0, x1 = np.percentile(xy[:, 0], [2, 98]); y0, y1 = np.percentile(xy[:, 1], [2, 98])
    sx = (xy[:, 0] - (x0 + x1) / 2) / max(x1 - x0, 1e-6); sy = (xy[:, 1] - (y0 + y1) / 2) / max(y1 - y0, 1e-6)
    px = W * 0.60 + sx * W * 0.80; py = H * 0.50 + sy * H * 0.98
    ax.scatter(px, py, s=46, c=rgb, alpha=0.05, linewidths=0, zorder=1)        # bloom
    ax.scatter(px, py, s=15, c=rgb, alpha=0.12, linewidths=0, zorder=1)
    ax.scatter(px, py, s=1.6, c=rgb, alpha=0.9, linewidths=0, zorder=2)        # cores

    # legibility panel for the text, fading into the map
    for i in range(40):
        a = 0.62 * (1 - i / 40) ** 1.5
        ax.add_patch(Rectangle((0, 0), 720 - i * 6, H, color="#0d0a18", alpha=a * 0.09, zorder=3, lw=0))

    ps = 12.4
    left, top = 54, H - 190
    draw_wordmark(ax, left, top, ps)

    ax.text(left, top - 7 * ps - 52, tagline or "A map of music, by how it is listened to and what it is made of", color="#e6def7",
            fontsize=19, fontweight="semibold", zorder=7, va="top")
    ax.text(left, top - 7 * ps - 98, f"{man['n_vocab']:,} tracks · three models · steer with spherical interpolation".replace(",", ","),
            color="#a99bd0", fontsize=14.5, zorder=7, va="top")
    ax.text(left, 52, "ravjothbrar.com/melomane", color="#d9b8ff", fontsize=16, fontweight="bold", zorder=7, va="bottom", family="monospace")
    out = Path(out_png)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=DPI, facecolor=fig.get_facecolor())
    plt.close(fig)
    return out
