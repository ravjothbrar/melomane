// math.js - pure numerical helpers for the Melomane explorer.  ES module, NO DOM / WebGL / network access, so the exact
// same code runs in the browser and under node (tests/test_explorer.py drives it with `node`).
//
// Conventions
//   * Embedding matrices are row-major Float32Array of length n*d, rows unit-normalised after decodeEmbeddings().
//   * Everything geometric mirrors melomane.slerp / melomane.api:  q = cos(t) v + sin(t) unit(d - (d.v) v).
//   * View = {cx, cy, zoom} over world coordinates in [-1, 1]^2;  scale = zoom * min(w, h) / 2 pixels per world unit.

/** Euclidean norm. */
export function norm(v) {
  let s = 0;
  for (let i = 0; i < v.length; i++) s += v[i] * v[i];
  return Math.sqrt(s);
}

/** Unit-normalised copy (zero vectors are returned unchanged, like geometry.unit's eps guard). */
export function unit(v) {
  const n = Math.max(norm(v), 1e-9);
  const out = new Float32Array(v.length);
  for (let i = 0; i < v.length; i++) out[i] = v[i] / n;
  return out;
}

export function dot(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

/** IEEE half -> float (the browser has no portable Float16Array yet). */
export function halfToFloat(h) {
  const s = (h & 0x8000) ? -1 : 1;
  const e = (h >> 10) & 0x1f;
  const f = h & 0x3ff;
  if (e === 0) return s * Math.pow(2, -14) * (f / 1024);
  if (e === 0x1f) return f ? NaN : s * Infinity;
  return s * Math.pow(2, e - 15) * (1 + f / 1024);
}

/**
 * Decode a payload matrix into unit rows.
 * @param {"int8"|"fp16"|"fp32"} dtype
 * @param {ArrayBuffer} buffer raw bytes (row-major n x d)
 * @param {ArrayBuffer|null} scaleBuffer float32 per-row scales (int8 only)
 * @returns {Float32Array} n*d unit rows
 */
export function decodeEmbeddings(dtype, buffer, scaleBuffer, n, d) {
  const out = new Float32Array(n * d);
  if (dtype === "int8") {
    const q = new Int8Array(buffer, 0, n * d);
    const sc = new Float32Array(scaleBuffer, 0, n);
    for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) out[i * d + j] = q[i * d + j] * sc[i];
  } else if (dtype === "fp16") {
    const h = new Uint16Array(buffer, 0, n * d);
    for (let i = 0; i < n * d; i++) out[i] = halfToFloat(h[i]);
  } else if (dtype === "fp32") {
    out.set(new Float32Array(buffer, 0, n * d));
  } else {
    throw new Error("unknown dtype " + dtype);
  }
  for (let i = 0; i < n; i++) {
    let s = 0;
    for (let j = 0; j < d; j++) s += out[i * d + j] * out[i * d + j];
    const inv = 1 / Math.max(Math.sqrt(s), 1e-9);
    for (let j = 0; j < d; j++) out[i * d + j] *= inv;
  }
  return out;
}

/** SLERP of a seed toward a direction by thetaDeg inside the plane (seed, direction).  cos(q, seed) = cos(theta). */
export function slerp(seed, direction, thetaDeg) {
  const v = unit(seed), dd = unit(direction);
  const proj = dot(dd, v);
  const dperp = new Float64Array(v.length);
  let n = 0;
  for (let i = 0; i < v.length; i++) { dperp[i] = dd[i] - proj * v[i]; n += dperp[i] * dperp[i]; }
  n = Math.sqrt(n);
  if (n < 1e-9) return v;
  const th = (thetaDeg * Math.PI) / 180, c = Math.cos(th), s = Math.sin(th);
  const q = new Float32Array(v.length);
  for (let i = 0; i < v.length; i++) q[i] = c * v[i] + (s * dperp[i]) / n;
  return unit(q);
}

/** Weighted unit-sum of poles, returned as a unit direction (melomane.slerp.compose_poles). */
export function composePoles(poles, weights) {
  const d = poles[0].length, acc = new Float64Array(d);
  for (let p = 0; p < poles.length; p++) {
    const u = unit(poles[p]), w = weights ? weights[p] : 1;
    for (let i = 0; i < d; i++) acc[i] += w * u[i];
  }
  return unit(Float32Array.from(acc));
}

/** 1 for rows that must not be returned: the seed (excludeSelf) and, if excludeArtist, every track by the seed's artist. */
export function exclusionMask(n, seedIdx, artistIdx, excludeSelf = true, excludeArtist = true) {
  const mask = new Uint8Array(n);
  if (excludeArtist && artistIdx) {
    const a = artistIdx[seedIdx];
    for (let i = 0; i < n; i++) if (artistIdx[i] === a) mask[i] = 1;
    mask[seedIdx] = excludeSelf ? 1 : 0;
  } else if (excludeSelf) {
    mask[seedIdx] = 1;
  }
  return mask;
}

/** Top-k rows of U (unit rows) by dot product with q; excluded rows skipped.  Returns {idx: Int32Array, score: Float32Array}. */
export function topK(U, n, d, q, k, mask) {
  const qq = unit(q);
  const idx = new Int32Array(k).fill(-1), sc = new Float64Array(k).fill(-Infinity);
  let filled = 0;
  for (let i = 0; i < n; i++) {
    if (mask && mask[i]) continue;
    let s = 0;
    const o = i * d;
    for (let j = 0; j < d; j++) s += U[o + j] * qq[j];
    if (filled === k && s <= sc[k - 1]) continue;
    let p = filled < k ? filled++ : k - 1;
    while (p > 0 && sc[p - 1] < s) { sc[p] = sc[p - 1]; idx[p] = idx[p - 1]; p--; }
    sc[p] = s; idx[p] = i;
  }
  return { idx: idx.slice(0, filled), score: Float32Array.from(sc.slice(0, filled)) };
}

/**
 * One explorer query: neighbours of a seed, optionally steered toward poles by thetaDeg.
 * @param {object} o {U, n, d, seed, poles: Float32Array[]|[], weights?, thetaDeg, k, artistIdx, excludeArtist}
 * @returns {{q: Float32Array, idx: Int32Array, score: Float32Array}}
 */
export function query(o) {
  const seedVec = o.U.subarray(o.seed * o.d, (o.seed + 1) * o.d);
  let q = unit(seedVec);
  if (o.poles && o.poles.length && o.thetaDeg > 0) {
    const dir = o.poles.length === 1 ? o.poles[0] : composePoles(o.poles, o.weights);
    q = slerp(seedVec, dir, o.thetaDeg);
  }
  const mask = exclusionMask(o.n, o.seed, o.artistIdx, true, o.excludeArtist !== false);
  const r = topK(o.U, o.n, o.d, q, o.k, mask);
  return { q, idx: r.idx, score: r.score };
}

/** Unit-mean pole of a member list (rows of U): the paper's emergent pole. */
export function meanPole(U, d, members) {
  const acc = new Float64Array(d);
  for (const m of members) for (let j = 0; j < d; j++) acc[j] += U[m * d + j];
  return unit(Float32Array.from(acc));
}

// ---- view / picking -------------------------------------------------------------------------------------------------
export function viewScale(view, w, h) { return (view.zoom * Math.min(w, h)) / 2; }

export function worldToScreen(view, w, h, wx, wy) {
  const s = viewScale(view, w, h);
  return [(wx - view.cx) * s + w / 2, h / 2 - (wy - view.cy) * s];
}

export function screenToWorld(view, w, h, sx, sy) {
  const s = viewScale(view, w, h);
  return [(sx - w / 2) / s + view.cx, (h / 2 - sy) / s + view.cy];
}

/** Uniforms for `gl_Position = vec4(world * scale + offset, 0, 1)`. */
export function clipUniforms(view, w, h) {
  const s = viewScale(view, w, h);
  const sx = (s * 2) / w, sy = (s * 2) / h;
  return { sx, sy, ox: -view.cx * sx, oy: -view.cy * sy };
}

/** Zoom by `factor` keeping the world point under screen (sx, sy) fixed. */
export function zoomAt(view, w, h, sx, sy, factor, minZoom = 0.5, maxZoom = 400) {
  const [wx, wy] = screenToWorld(view, w, h, sx, sy);
  const zoom = Math.min(maxZoom, Math.max(minZoom, view.zoom * factor));
  const s = (zoom * Math.min(w, h)) / 2;
  return { zoom, cx: wx - (sx - w / 2) / s, cy: wy - (h / 2 - sy) / s };
}

/** Pan by a pixel delta (drag). */
export function panBy(view, w, h, dxPx, dyPx) {
  const s = viewScale(view, w, h);
  return { zoom: view.zoom, cx: view.cx - dxPx / s, cy: view.cy + dyPx / s };
}

/** Nearest point (screen space) within radiusPx of (sx, sy); `visible` optionally masks points out.  -1 if none. */
export function pickNearest(xy, n, view, w, h, sx, sy, radiusPx, visible) {
  const s = viewScale(view, w, h);
  let best = -1, bd = radiusPx * radiusPx;
  for (let i = 0; i < n; i++) {
    if (visible && !visible[i]) continue;
    const px = (xy[2 * i] - view.cx) * s + w / 2 - sx;
    if (px > radiusPx || px < -radiusPx) continue;
    const py = h / 2 - (xy[2 * i + 1] - view.cy) * s - sy;
    const d2 = px * px + py * py;
    if (d2 <= bd) { bd = d2; best = i; }
  }
  return best;
}

// ---- colours --------------------------------------------------------------------------------------------------------
// Twelve luminous, well-separated hues that stay readable (and glow) on the dark stage; golden-angle hues after that.
const GLOW = [[255, 107, 107], [255, 169, 77], [255, 212, 59], [169, 227, 75], [81, 207, 102], [32, 201, 151],
  [34, 184, 207], [77, 171, 247], [116, 143, 252], [151, 117, 250], [218, 119, 242], [247, 131, 172]];

/** Categorical colour i (bright palette first, golden-angle hues after). */
export function categoricalColor(i) {
  if (i < 0) return [150, 150, 150];
  if (i < GLOW.length) return GLOW[i];
  return hslToRgb(((i * 137.508) % 360) / 360, 0.66, 0.64);
}

export function hslToRgb(h, s, l) {
  const f = (n) => {
    const k = (n + h * 12) % 12, a = s * Math.min(l, 1 - l);
    return Math.round(255 * (l - a * Math.max(-1, Math.min(k - 3, 9 - k, 1))));
  };
  return [f(0), f(8), f(4)];
}

const VIRIDIS = [[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]];
/** Sequential colour for t in [0, 1] (viridis-like). */
export function sequentialColor(t) {
  const x = Math.min(1, Math.max(0, t)) * (VIRIDIS.length - 1), i = Math.min(VIRIDIS.length - 2, Math.floor(x)), f = x - i;
  return VIRIDIS[i].map((c, j) => Math.round(c + (VIRIDIS[i + 1][j] - c) * f));
}

// ---- search ---------------------------------------------------------------------------------------------------------
/** Lower-case, accent-stripped, punctuation-free form used for search (mirrors curation.normalize_* loosely). */
export function normText(s) {
  return String(s ?? "").normalize("NFKD").replace(/[̀-ͯ]/g, "").toLowerCase().replace(/&/g, " and ")
    .replace(/[^a-z0-9]+/g, " ").trim();
}

/** Token search over pre-normalised haystack strings: every query token must prefix-match a word.  Best = earliest hit. */
export function searchTracks(hay, query, limit = 8) {
  const toks = normText(query).split(" ").filter(Boolean);
  if (!toks.length) return [];
  const hits = [];
  for (let i = 0; i < hay.length && hits.length < limit * 50; i++) {
    const words = hay[i].split(" ");
    let score = 0, ok = true;
    for (const t of toks) {
      const p = words.findIndex((w) => w.startsWith(t));
      if (p < 0) { ok = false; break; }
      score += p + (words[p] === t ? 0 : 0.5);
    }
    if (ok) hits.push([score, i]);
  }
  hits.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  return hits.slice(0, limit).map((h) => h[1]);
}

/** "search on..." links only (no bulk API calls: the iTunes Search API allows ~20 requests/min/IP, so we never prefetch). */
export function searchLinks(artist, title) {
  const q = encodeURIComponent(`${artist} ${title}`.trim());
  return [
    { name: "YouTube", url: `https://www.youtube.com/results?search_query=${q}` },
    { name: "Spotify", url: `https://open.spotify.com/search/${q}` },
    { name: "Apple Music", url: `https://music.apple.com/search?term=${q}` },
  ];
}


// ---- label placement ------------------------------------------------------------------------------------------------
/**
 * Greedy, priority-ordered placement of pill labels so that none overlap and none leave the viewport.
 * `cands` come highest-priority first: {id, x, y (anchor point, px), w, h (pill size), required?}.  Each label tries to sit
 * above its anchor, then below, to the right, to the left; the first free spot wins, otherwise the label is dropped
 * (unless `required`, which is always placed at its first in-bounds spot).  Pure function: used by the overlay and unit-tested.
 * @returns {Array<{id, x, y, w, h, x0, y0, anchor}>} placed labels (x0, y0 = top-left of the pill)
 */
export function placeLabels(cands, W, H, { pad = 3, gap = 7, maxLabels = 120, margin = 4 } = {}) {
  const taken = [], out = [];
  const clash = (b) => {
    for (const t of taken) if (b.x < t.x + t.w + pad && b.x + b.w + pad > t.x && b.y < t.y + t.h + pad && b.y + b.h + pad > t.y) return true;
    return false;
  };
  for (const c of cands) {
    if (out.length >= maxLabels && !c.required) continue;
    if (c.x < -c.w || c.x > W + c.w || c.y < -c.h || c.y > H + c.h) continue;
    const spots = [
      { x: c.x - c.w / 2, y: c.y - c.h - gap, a: "above" }, { x: c.x - c.w / 2, y: c.y + gap, a: "below" },
      { x: c.x + gap, y: c.y - c.h / 2, a: "right" }, { x: c.x - c.w - gap, y: c.y - c.h / 2, a: "left" },
    ];
    const inb = (o) => o.x >= margin && o.y >= margin && o.x + c.w <= W - margin && o.y + c.h <= H - margin;
    let pick = null;
    for (const o of spots) if (inb(o) && !clash({ x: o.x, y: o.y, w: c.w, h: c.h })) { pick = o; break; }
    if (!pick && c.fan !== false) {    // crowded: fan out on rings of increasing radius around the anchor (drawn with a leader line)
      for (let k = 1; k <= 4 && !pick; k++) {
        for (let a = 0; a < 12 && !pick; a++) {
          const ang = (a * Math.PI) / 6 + (k % 2) * (Math.PI / 12);
          const o = { x: c.x + Math.cos(ang) * (c.w / 2 + gap + k * 34) - c.w / 2, y: c.y + Math.sin(ang) * (c.h / 2 + gap + k * (c.h + 8)) - c.h / 2, a: "ring", leader: true };
          if (inb(o) && !clash({ x: o.x, y: o.y, w: c.w, h: c.h })) pick = o;
        }
      }
    }
    if (!pick && c.required) pick = spots.find(inb) || spots[0];
    if (!pick) continue;
    taken.push({ x: pick.x, y: pick.y, w: c.w, h: c.h });
    out.push({ ...c, x0: pick.x, y0: pick.y, anchor: pick.a, leader: !!pick.leader });
  }
  return out;
}


// ---- device gate ----------------------------------------------------------------------------------------------------
/**
 * True for a touch-first device whose *physical screen* is phone-sized.  Pure and fail-open: missing information never blocks.
 * Uses the screen (CSS px), not the window, so a small or zoomed desktop browser is never blocked, and the shorter side, so rotating a
 * phone does not unlock it.  A foldable's cover screen (<= ~480 px) is blocked; unfolded (>= ~670 px), tablets, laptops and desktops pass.
 */
export function isPhoneLike({ coarse, screenW, screenH, minDim = 600 } = {}) {
  if (!coarse || !(screenW > 0) || !(screenH > 0)) return false;          // missing / zero / NaN is not information: fail open
  return Math.min(screenW, screenH) < minDim;
}
