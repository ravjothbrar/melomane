// Runs apps/explorer/math.js (pure ES module, no DOM) under node against a built explorer payload.
// usage: node node_explorer_check.mjs <explorer_dir> <cases.json>   -> JSON on stdout
import { readFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const src = readFileSync(join(here, "..", "apps", "explorer", "math.js"), "utf8");
const M = await import("data:text/javascript;base64," + Buffer.from(src).toString("base64"));   // works on any node >= 18

const [dir, casesPath] = process.argv.slice(2);
const cases = JSON.parse(readFileSync(casesPath, "utf8"));
const rd = (p) => { const b = readFileSync(join(dir, p)); return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength); };
const manifest = JSON.parse(readFileSync(join(dir, "data", "manifest.json"), "utf8"));
const tracks = JSON.parse(readFileSync(join(dir, manifest.tracks.path), "utf8"));
const artistIdx = Int32Array.from(tracks.artist_idx);

const models = {};
for (const info of manifest.models) {
  const f = info.files, d = info.dim, n = info.n;
  const meta = JSON.parse(readFileSync(join(dir, f.poles_meta.path), "utf8"));
  const P = new Float32Array(rd(f.poles.path));
  models[info.name] = {
    n, d, U: M.decodeEmbeddings(info.dtype, rd(f.emb.path), f.scale ? rd(f.scale.path) : null, n, d),
    poles: new Map(meta.map((p, i) => [p.name, P.subarray(i * d, (i + 1) * d)])),
  };
}

const results = cases.map((c) => {
  const m = models[c.model];
  const poles = c.poles.map((p) => m.poles.get(p));
  const r = M.query({ U: m.U, n: m.n, d: m.d, seed: c.seed, poles, weights: c.weights ?? undefined, thetaDeg: c.theta, k: c.k,
    artistIdx, excludeArtist: c.exclude_artist });
  return { idx: Array.from(r.idx), score: Array.from(r.score), cosQ: M.dot(M.unit(m.U.subarray(c.seed * m.d, (c.seed + 1) * m.d)), r.q) };
});

// ---- self-checks of the pure maths ------------------------------------------------------------------------------------
const checks = {};
const rnd = (() => { let s = 12345; return () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296) - 0.5; })();
const vec = (d) => Float32Array.from({ length: d }, rnd);
{
  const v = vec(16), d = vec(16), q = M.slerp(v, d, 60);
  checks.slerp_cos60 = Math.abs(M.dot(q, M.unit(v)) - 0.5) < 1e-5 && Math.abs(M.norm(q) - 1) < 1e-5;
  checks.slerp_theta0 = M.slerp(v, d, 0).every((x, i) => Math.abs(x - M.unit(v)[i]) < 1e-6);
  checks.slerp_parallel = M.slerp(v, v, 45).every((x, i) => Math.abs(x - M.unit(v)[i]) < 1e-5);      // d || v -> seed
  checks.slerp_theta90_orthogonal = Math.abs(M.dot(M.slerp(v, d, 90), M.unit(v))) < 1e-5;
  const c = M.composePoles([vec(16), vec(16)], [1, 3]);
  checks.compose_unit = Math.abs(M.norm(c) - 1) < 1e-5;
}
{
  const n = 300, d = 8, U = new Float32Array(n * d);
  for (let i = 0; i < n; i++) U.set(M.unit(vec(d)), i * d);
  const q = vec(d), mask = new Uint8Array(n); mask[7] = 1; mask[100] = 1;
  const r = M.topK(U, n, d, q, 10, mask);
  const all = [...Array(n).keys()].filter((i) => !mask[i]).map((i) => [M.dot(U.subarray(i * d, (i + 1) * d), M.unit(q)), i]).sort((a, b) => b[0] - a[0]);
  checks.topk_matches_bruteforce = r.idx.every((i, j) => i === all[j][1]);
  checks.topk_short = M.topK(U, 3, d, q, 10, null).idx.length === 3;
  const art = new Int32Array(n).map((_, i) => i % 5);
  const mk = M.exclusionMask(n, 12, art, true, true);
  checks.artist_mask = mk[12] === 1 && mk[2] === 1 && mk[3] === 0 && mk.reduce((a, b) => a + b, 0) === n / 5;
  const keep = M.exclusionMask(n, 12, art, false, true);
  checks.artist_mask_keep_self = keep[12] === 0 && keep[2] === 1;
}
{
  const view = { cx: 0.2, cy: -0.1, zoom: 3 }, w = 800, h = 600;
  const [sx, sy] = M.worldToScreen(view, w, h, 0.3, 0.4), [wx, wy] = M.screenToWorld(view, w, h, sx, sy);
  checks.view_roundtrip = Math.abs(wx - 0.3) < 1e-9 && Math.abs(wy - 0.4) < 1e-9;
  const z = M.zoomAt(view, w, h, 123, 456, 1.7), [ax, ay] = M.screenToWorld(view, w, h, 123, 456), [bx, by] = M.screenToWorld(z, w, h, 123, 456);
  checks.zoom_keeps_anchor = Math.abs(ax - bx) < 1e-9 && Math.abs(ay - by) < 1e-9 && Math.abs(z.zoom - 5.1) < 1e-9;
  checks.zoom_clamped = M.zoomAt(view, w, h, 0, 0, 1e9).zoom === 400;
  const p = M.panBy(view, w, h, 50, -20), [px, py] = M.worldToScreen(p, w, h, 0.3, 0.4);
  checks.pan_moves_content = Math.abs(px - (sx + 50)) < 1e-9 && Math.abs(py - (sy - 20)) < 1e-9;
  const u = M.clipUniforms(view, w, h), [qx, qy] = M.worldToScreen(view, w, h, 0.3, 0.4);
  checks.clip_matches_screen = Math.abs(((0.3 * u.sx + u.ox + 1) / 2) * w - qx) < 1e-6 && Math.abs(((1 - (0.4 * u.sy + u.oy)) / 2) * h - qy) < 1e-6;
  const xy = Float32Array.from([0, 0, 0.3, 0.4, 0.31, 0.4]);
  checks.pick_nearest = M.pickNearest(xy, 3, view, w, h, sx + 2, sy, 9, null) === 1 && M.pickNearest(xy, 3, view, w, h, sx + 400, sy, 9, null) === -1
    && M.pickNearest(xy, 3, view, w, h, sx, sy, 9, Uint8Array.from([1, 0, 1])) === -1;
}
{
  checks.half = [M.halfToFloat(0x3c00) === 1, M.halfToFloat(0xc000) === -2, M.halfToFloat(0x7bff) === 65504, M.halfToFloat(0x0001) === Math.pow(2, -24),
    M.halfToFloat(0x0000) === 0, M.halfToFloat(0x7c00) === Infinity, Number.isNaN(M.halfToFloat(0x7e00))].every(Boolean);
  checks.search = (() => {
    const hay = ["radiohead karma police", "the beatles yesterday", "beatles let it be", "queen bohemian rhapsody"].map((s) => M.normText(s));
    return JSON.stringify(M.searchTracks(hay, "Beatles yest")) === "[1]" && M.searchTracks(hay, "beat", 5).length === 2 && M.searchTracks(hay, "zzz").length === 0
      && M.normText("Beyoncé & Jay‑Z!") === "beyonce and jay z";
  })();
  const L = M.searchLinks("A&B", "x y");
  checks.links = L.length === 3 && L.every((l) => l.url.startsWith("https://") && !l.url.includes(" ")) && L[0].url.includes("A%26B%20x%20y");
  checks.colours = M.categoricalColor(0).length === 3 && M.categoricalColor(50).every((c) => c >= 0 && c <= 255) && M.sequentialColor(0)[0] === 68 && M.sequentialColor(1)[0] === 253;
}
console.log(JSON.stringify({ results, checks }));
