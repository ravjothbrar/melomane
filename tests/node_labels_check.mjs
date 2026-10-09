// Pure-function checks for the explorer's label placement + palette (apps/explorer/math.js), no DOM, no payload.
// usage: node node_labels_check.mjs   -> JSON {check: bool} on stdout
import { readFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const src = readFileSync(join(here, "..", "apps", "explorer", "math.js"), "utf8");
const M = await import("data:text/javascript;base64," + Buffer.from(src).toString("base64"));

const overlap = (a, b, pad = 0) => a.x0 < b.x0 + b.w + pad && a.x0 + a.w + pad > b.x0 && a.y0 < b.y0 + b.h + pad && a.y0 + a.h + pad > b.y0;
const W = 800, H = 500, checks = {};

// 1. dense candidate set: nothing overlaps, everything inside the viewport, priority order respected
const rng = (() => { let s = 12345; return () => (s = (s * 1664525 + 1013904223) >>> 0) / 2 ** 32; })();
const cands = Array.from({ length: 300 }, (_, i) => ({ id: i, x: rng() * W, y: rng() * H, w: 50 + rng() * 60, h: 20 }));
const placed = M.placeLabels(cands, W, H, { pad: 3, maxLabels: 300 });
let clash = false;
for (let i = 0; i < placed.length; i++) for (let j = i + 1; j < placed.length; j++) if (overlap(placed[i], placed[j])) clash = true;
checks.no_overlaps = !clash && placed.length > 10;
checks.inside_viewport = placed.every((p) => p.x0 >= 0 && p.y0 >= 0 && p.x0 + p.w <= W && p.y0 + p.h <= H);
checks.fewer_than_all_when_crowded = placed.length < cands.length;
const ids = placed.map((p) => p.id);
checks.priority_order = ids.every((v, i) => i === 0 || v > ids[i - 1]);                  // candidates are given highest-priority first
checks.first_candidate_wins = ids[0] === 0 || !cands.slice(0, 1).every((c) => c.x > 0 && c.y > 30);

// 2. two labels that cannot both fit above their anchors: the second moves to another anchor instead of vanishing
const two = M.placeLabels([{ id: "a", x: 400, y: 200, w: 100, h: 20 }, { id: "b", x: 410, y: 205, w: 100, h: 20 }], W, H);
checks.collision_resolved_by_anchor = two.length === 2 && two[0].anchor !== two[1].anchor && !overlap(two[0], two[1]);

// 3. required labels are always placed (even if crowded); off-screen anchors are dropped
const crowd = [{ id: "x", x: 100, y: 100, w: 700, h: 400 }, { id: "seed", x: 300, y: 250, w: 90, h: 22, required: true }];
const req = M.placeLabels(crowd, W, H);
checks.required_always_placed = req.some((p) => p.id === "seed");
checks.offscreen_dropped = M.placeLabels([{ id: "far", x: -900, y: -900, w: 50, h: 20 }], W, H).length === 0;
checks.max_labels_cap = M.placeLabels(cands, W, H, { maxLabels: 5 }).length <= 5;


// 5. a tight blob (ten anchors within ~40x20 px, 150 px pills): labels fan out on a ring with leader lines instead of vanishing,
//    still without a single overlap, and only displaced labels are flagged as needing a leader
const blob = Array.from({ length: 10 }, (_, i) => ({ id: "n" + i, x: 400 + (i % 5) * 8, y: 250 + Math.floor(i / 5) * 10, w: 150, h: 22 }));
const fan = M.placeLabels(blob, W, H, { pad: 3, maxLabels: 50 });
let fanClash = false;
for (let i = 0; i < fan.length; i++) for (let j = i + 1; j < fan.length; j++) if (overlap(fan[i], fan[j])) fanClash = true;
checks.blob_fans_out = fan.length >= 8 && !fanClash;
checks.blob_inside_viewport = fan.every((p) => p.x0 >= 0 && p.y0 >= 0 && p.x0 + p.w <= W && p.y0 + p.h <= H);
checks.leader_flag_only_when_displaced = fan.every((p) => p.leader === (p.anchor === "ring")) && fan.some((p) => p.leader);

// 4. palette: valid rgb, distinct, bright enough for a dark background
const pal = Array.from({ length: 24 }, (_, i) => M.categoricalColor(i));
const lum = (c) => 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
checks.palette_valid = pal.every((c) => c.length === 3 && c.every((v) => Number.isInteger(v) && v >= 0 && v <= 255));
checks.palette_bright = pal.every((c) => lum(c) > 95);
const d = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
let minD = 1e9; for (let i = 0; i < 12; i++) for (let j = i + 1; j < 12; j++) minD = Math.min(minD, d(pal[i], pal[j]));
checks.first12_distinct = minD > 40;

// 6. device gate: block touch-first phones (and a foldable's small cover screen), allow iPads, laptops, desktops and unfolded foldables.
//    The test is the physical screen (CSS px), never the window, so a shrunk or zoomed desktop browser is not blocked.
const D = (coarse, w, h) => M.isPhoneLike({ coarse, screenW: w, screenH: h });
checks.gate_blocks_phone_portrait = D(true, 390, 844) === true && D(true, 430, 932) === true && D(true, 360, 800) === true;
checks.gate_blocks_phone_landscape = D(true, 844, 390) === true && D(true, 932, 430) === true;     // rotating a phone does not unlock it
checks.gate_blocks_foldable_cover_screen = D(true, 344, 882) === true;
checks.gate_allows_ipads_and_tablets = D(true, 744, 1133) === false && D(true, 820, 1180) === false && D(true, 1024, 1366) === false && D(true, 600, 960) === false;
checks.gate_allows_unfolded_foldables = D(true, 884, 1104) === false && D(true, 673, 841) === false && D(true, 904, 1136) === false;
checks.gate_allows_laptops_and_desktops = D(false, 1920, 1080) === false && D(false, 1440, 900) === false && D(false, 1280, 720) === false;
checks.gate_ignores_zoomed_desktops = D(false, 384, 216) === false;                                // 500 % browser zoom shrinks screen.width, but the pointer is fine
checks.gate_fails_open_without_information = M.isPhoneLike({}) === false && M.isPhoneLike({ coarse: true }) === false && D(true, 0, 0) === false && D(true, NaN, 800) === false;

console.log(JSON.stringify(checks));
