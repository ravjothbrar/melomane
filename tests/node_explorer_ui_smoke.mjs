// Smoke-tests apps/explorer/explorer.js under node with a minimal DOM / WebGL2 stub (no browser available in CI).
// usage: node node_explorer_ui_smoke.mjs <built_explorer_dir>   -> JSON summary on stdout
import { readFileSync, writeFileSync, mkdtempSync, existsSync } from "node:fs";
import { join, dirname } from "node:path";
import { tmpdir } from "node:os";
import { fileURLToPath, pathToFileURL } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const appDir = join(here, "..", "apps", "explorer");
const dir = process.argv[2];

// explorer.js imports "./math.js"; node may treat .js as CommonJS, so stage .mjs copies with a rewritten specifier
const tmp = mkdtempSync(join(tmpdir(), "melo-ui-"));
writeFileSync(join(tmp, "math.mjs"), readFileSync(join(appDir, "math.js"), "utf8"));
writeFileSync(join(tmp, "explorer.mjs"), readFileSync(join(appDir, "explorer.js"), "utf8").replace('"./math.js"', '"./math.mjs"'));
globalThis.__MELOMANE_NO_AUTOSTART__ = true;
const { start } = await import(pathToFileURL(join(tmp, "explorer.mjs")).href);
const M = await import(pathToFileURL(join(tmp, "math.mjs")).href);

// ---- stubs ------------------------------------------------------------------------------------------------------------
const drawCalls = { arrays: 0, elements: 0 };
const glStub = new Proxy({}, {
  get: (t, p) => {
    if (p === "drawArrays") return () => { drawCalls.arrays++; };
    if (p === "drawElements") return () => { drawCalls.elements++; };
    if (typeof p === "string" && /^[A-Z][A-Z0-9_]*$/.test(p)) return 1;           // GL constants
    return () => 1;                                                                // every method "succeeds"
  },
});
class Node {
  constructor(tag) { this.tag = tag; this.children = []; this.attrs = {}; this.listeners = {}; this.style = {}; this._text = ""; this.hidden = false;
    this.value = ""; this.checked = false; this.classList = { add() {}, remove() {}, toggle() {} }; }
  append(...k) { this.children.push(...k); } appendChild(c) { this.children.push(c); return c; }
  replaceChildren(...k) { this.children = [...k]; this._text = ""; }
  set textContent(v) { this._text = String(v); this.children = []; }
  get textContent() { return this._text + this.children.map((c) => c.textContent ?? "").join(""); }
  setAttribute(k, v) { this.attrs[k] = String(v); } getAttribute(k) { return this.attrs[k]; }
  addEventListener(ev, fn) { (this.listeners[ev] ||= []).push(fn); }
  async fire(ev, e = {}) { for (const fn of this.listeners[ev] || []) await fn({ target: this, preventDefault() {}, ...e }); }
  getBoundingClientRect() { return { left: 0, top: 0, width: 800, height: 600 }; }
  get clientWidth() { return 800; } get clientHeight() { return 600; }
  setPointerCapture() {} getContext() { return this.noGL ? null : glStub; }
  walk(fn) { fn(this); for (const c of this.children) if (c.walk) c.walk(fn); }
}
function makeEnv(base, { noGL = false, failFetch = false } = {}) {
  const byId = new Map();
  const document = {
    getElementById: (id) => { if (!byId.has(id)) { const n = new Node("#" + id); if (id === "map") n.noGL = noGL; if (id === "error" || id === "tooltip") n.hidden = true; byId.set(id, n); } return byId.get(id); },
    createElement: (t) => new Node(t), createTextNode: (s) => ({ textContent: String(s) }),
  };
  const history = { hash: "", replaceState(_a, _b, h) { this.hash = h; } };
  const window = { matchMedia: () => ({ matches: false }), devicePixelRatio: 1, requestAnimationFrame: (fn) => queueMicrotask(fn),
    location: { hash: "" }, history, addEventListener() {} };
  const fetch = async (p) => {
    const f = join(base, p);
    if (failFetch || !existsSync(f)) return { ok: false, status: 404 };
    const b = readFileSync(f);
    return { ok: true, status: 200, json: async () => JSON.parse(b.toString("utf8")), arrayBuffer: async () => b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength) };
  };
  return { document, window, fetch, base: "", byId, history, windowObj: window };
}
const tick = () => new Promise((r) => setTimeout(r, 5));
const out = {};

// ---- 1. failure path: unreachable payload gives a helpful message, no throw ---------------------------------------------
{
  const env = makeEnv(dir, { failFetch: true });
  const app = await start(env);
  const err = env.document.getElementById("error");
  out.fail_returns_null = app === null; out.fail_message = err.textContent; out.fail_visible = err.hidden === false;
}
// ---- 2. no WebGL2 --------------------------------------------------------------------------------------------------------
{
  const env = makeEnv(dir + "/", { noGL: true });
  const app = await start(env);
  out.nogl_returns_null = app === null; out.nogl_message = env.document.getElementById("error").textContent;
}
// ---- 3. full interaction ------------------------------------------------------------------------------------------------
const env = makeEnv(dir + "/");
const $ = (id) => env.document.getElementById(id);
const app = await start(env);
await tick();
out.started = !!app;
const fit0 = { ...app.state.view };            // the auto-fitted opening view
out.fit_zoom_sane = fit0.zoom >= 0.6 && fit0.zoom <= 6;
out.toggle_buttons = $("model-toggle").children.length;
out.status = $("status").textContent;
out.attribution = $("attribution").textContent;
out.error_hidden = $("error").hidden === true;
out.pole_groups = $("pole-select").children.map((g) => g.attrs.label);
out.mode_list_items = $("mode-list").children.length;
out.legend_items_category = $("legend").children.length;

// seed by API, then check the rendered panel
const seed = 5;
app.setSeed(seed); await tick();
out.results_rendered = $("results").children.length;
out.results_head = $("results-head").textContent;
out.first_result = $("results").children[0].textContent;
out.state_results_idx = Array.from(app.state.results.idx);
const links = []; $("results").children[0].walk((n) => { if (n.tag === "a") links.push(n.href); });
out.links = links;
out.seed_card = $("seed-card").textContent;

// steer: add a pole, theta 60 via the slider event
const poleName = app.state.models[app.state.modelName].poleMeta.find((p) => p.group === "supervised").name;
$("pole-select").value = poleName; await $("pole-add").fire("click"); await tick();
$("theta").value = "60"; await $("theta").fire("input"); await tick();
out.pole_name = poleName;
out.steered_head = $("results-head").textContent;
out.steered_idx = Array.from(app.state.results.idx);
out.steered_theta = app.state.theta;
out.theta_cos = $("theta-cos").textContent;
out.chips = $("pole-chips").children.length;
out.hash = env.history.hash;

// k / artist toggle / colour-by
$("k-select").value = "5"; await $("k-select").fire("change"); await tick(); out.k5 = $("results").children.length;
$("exclude-artist").checked = false; await $("exclude-artist").fire("change"); await tick(); out.leaky_idx = Array.from(app.state.results.idx);
$("exclude-artist").checked = true; await $("exclude-artist").fire("change");
for (const c of ["mode", "year", "category"]) { $("color-by").value = c; await $("color-by").fire("change"); out["legend_" + c] = $("legend").children.length; }

// model switch (second toggle button), poles that do not exist there are dropped, seed kept
await $("model-toggle").children[1].fire("click"); await tick();
out.model_after_switch = app.state.modelName; out.seed_after_switch = app.state.seed; out.results_after_switch = $("results").children.length;

// mode atlas
await $("tab-modes").fire("click");
out.modes_visible = $("panel-modes").hidden === false && $("panel-results").hidden === true;
app.showMode(0); await tick();
out.mode_detail_children = $("mode-detail").children.length;
out.mode_focus = app.state.modeFocus ? app.state.modeFocus.length : 0;
$("mode-filter").value = "zzzz-nothing"; await $("mode-filter").fire("input"); out.mode_filter_none = $("mode-list").children.length;
$("mode-filter").value = ""; await $("mode-filter").fire("input");
await $("tab-results").fire("click");

// seed search box
$("seed-search").value = "velvet"; await $("seed-search").fire("input"); out.suggestions = $("seed-suggest").children.length;

// pointer interaction: click exactly on a point
const m = app.state.models[app.state.modelName], target = 42;
app.state.view = { cx: m.xy[2 * target], cy: m.xy[2 * target + 1], zoom: 40 };
const [sx, sy] = M.worldToScreen(app.state.view, 800, 600, m.xy[2 * target], m.xy[2 * target + 1]);
const cv = $("map");
await cv.fire("pointermove", { clientX: sx, clientY: sy, pointerId: 1 });
out.hover = app.state.hover; out.tooltip_visible = $("tooltip").hidden === false; out.tooltip_text = $("tooltip").textContent;
await cv.fire("pointerdown", { clientX: sx, clientY: sy, pointerId: 1 });
await cv.fire("pointerup", { clientX: sx, clientY: sy, pointerId: 1 });
out.clicked_seed = app.state.seed;
// drag pans, wheel zooms
const v0 = { ...app.state.view };
await cv.fire("pointerdown", { clientX: 100, clientY: 100, pointerId: 1 });
await cv.fire("pointermove", { clientX: 160, clientY: 130, pointerId: 1 });
await cv.fire("pointerup", { clientX: 160, clientY: 130, pointerId: 1 });
out.drag_moved_view = app.state.view.cx !== v0.cx && app.state.view.cy !== v0.cy && app.state.seed === out.clicked_seed;
await cv.fire("wheel", { clientX: 400, clientY: 300, deltaY: -200 });
out.wheel_zoomed = app.state.view.zoom > v0.zoom;
await $("zoom-reset").fire("click");
const r1 = { ...app.state.view };
await $("zoom-reset").fire("click");
const r2 = { ...app.state.view };
// reset = auto-fit of the *current* model: a sane view inside the layout, and idempotent
out.reset = r1.zoom >= 0.6 && r1.zoom <= 6 && Math.abs(r1.cx) <= 1 && Math.abs(r1.cy) <= 1 && ["cx", "cy", "zoom"].every((k) => r1[k] === r2[k]);
await tick();
out.draw_calls = drawCalls;
console.log(JSON.stringify(out));
