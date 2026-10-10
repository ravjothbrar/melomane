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

    # wordmark from pixel cells, vertical purple gradient, retro offset shadow
    cells, cols, note_col = word_layout()
    ps = 12.4
    left, top = 54, H - 190
    def box(c, r, color, dx=0.0, dy=0.0, z=5):
        ax.add_patch(Rectangle((left + c * ps + dx, top - (r + 1) * ps + dy), ps * 0.94, ps * 0.94, color=color, lw=0, zorder=z))
    for (c, r) in cells:
        box(c, r, "#2e1065", 5.5, -5.5, 4)
        box(c, r, "#5b21b6", 2.8, -2.8, 4)
    grad = ["#f0b6ff", "#e2a0ff", "#d08aff", "#bd6dff", "#a955ff", "#9140f0", "#7b2ede"]
    for (c, r) in cells:
        box(c, r, grad[r], 0, 0, 5)
    # crotchet in the empty slot: slanted oval head + stem
    nx = left + note_col * ps + ps * 0.1
    stem_x = nx + ps * 2.35
    ax.add_patch(Rectangle((stem_x, top - 7 * ps + ps * 1.0), ps * 0.55, ps * 6.1, color="#ff8be0", lw=0, zorder=6))
    ax.add_patch(Ellipse((nx + ps * 1.45, top - 7 * ps + ps * 0.95), ps * 2.5, ps * 1.8, angle=22, color="#ff8be0", lw=0, zorder=6))
    ax.add_patch(Ellipse((nx + ps * 1.45, top - 7 * ps + ps * 0.95), ps * 3.4, ps * 2.7, angle=22, color="#ff8be0", alpha=0.18, lw=0, zorder=5))

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
