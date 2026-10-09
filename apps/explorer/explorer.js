// explorer.js - Melomane static explorer (no dependencies, no CDN, no runtime network beyond this site's own payload files).
// Everything numeric lives in math.js; this file is the DOM + WebGL2 shell.  All strings from the payload are inserted with
// textContent (never innerHTML).  Audio previews are "search on..." links only.
import * as M from "./math.js";

const VS = `#version 300 es
in vec2 a_pos; in vec4 a_col;
uniform vec2 u_scale; uniform vec2 u_off; uniform float u_size;
out vec4 v_col;
void main() { gl_Position = vec4(a_pos * u_scale + u_off, 0.0, 1.0); gl_PointSize = u_size; v_col = a_col; }`;
const FS = `#version 300 es
precision mediump float;
in vec4 v_col; uniform float u_alpha; out vec4 o;
void main() { vec2 c = gl_PointCoord * 2.0 - 1.0; float r = length(c); if (r > 1.0) discard;
  o = vec4(v_col.rgb, v_col.a * u_alpha * smoothstep(1.0, 0.75, r)); }`;

const clamp = (x, a, b) => Math.min(b, Math.max(a, x));

function makeEl(doc) {
  return function el(tag, attrs = {}, ...kids) {
    const e = doc.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (k === "text") e.textContent = v;
      else if (k === "on") for (const [ev, fn] of Object.entries(v)) e.addEventListener(ev, fn);
      else if (k in { value: 1, checked: 1, disabled: 1, hidden: 1, href: 1, target: 1, rel: 1, type: 1, min: 1, max: 1, step: 1, title: 1 }) e[k] = v;
      else e.setAttribute(k, v);
    }
    for (const kid of kids) e.append(typeof kid === "string" ? doc.createTextNode(kid) : kid);
    return e;
  };
}

/** Boot the explorer.  `env` may inject {document, window, fetch, base} (used by the node smoke test). */
export async function start(env = {}) {
  const doc = env.document ?? document, win = env.window ?? window;
  const fetchFn = env.fetch ?? fetch.bind(globalThis);
  const base = env.base ?? "";
  const $ = (id) => doc.getElementById(id);
  const el = makeEl(doc);
  const dark = () => !!(win.matchMedia && win.matchMedia("(prefers-color-scheme: dark)").matches);

  const ui = {
    status: $("status"), error: $("error"), toggle: $("model-toggle"), seedSearch: $("seed-search"), suggest: $("seed-suggest"),
    seedCard: $("seed-card"), poleSelect: $("pole-select"), poleAdd: $("pole-add"), chips: $("pole-chips"), theta: $("theta"),
    thetaOut: $("theta-out"), thetaCos: $("theta-cos"), kSel: $("k-select"), excl: $("exclude-artist"), colorBy: $("color-by"),
    legend: $("legend"), canvas: $("map"), tooltip: $("tooltip"), stage: $("stage"), results: $("results"), resHead: $("results-head"),
    tabRes: $("tab-results"), tabModes: $("tab-modes"), panelRes: $("panel-results"), panelModes: $("panel-modes"),
    modeFilter: $("mode-filter"), modeList: $("mode-list"), modeDetail: $("mode-detail"), attribution: $("attribution"),
    zin: $("zoom-in"), zout: $("zoom-out"), zreset: $("zoom-reset"),
  };

  const state = {
    manifest: null, tracks: null, hay: null, modelName: null, models: {}, seed: -1, poles: [], theta: 30, k: 10,
    excludeArtist: true, colorBy: "category", view: { cx: 0, cy: 0, zoom: 0.95 }, hover: -1, results: null, modeFocus: null,
    modeFocusId: null,
  };
  const fail = (msg) => { ui.error.hidden = false; ui.error.textContent = msg; ui.status.textContent = ""; };
  const say = (msg) => { ui.status.textContent = msg; };
  const getJSON = async (p) => { const r = await fetchFn(base + p); if (!r.ok) throw new Error(`${p}: HTTP ${r.status}`); return r.json(); };
  const getBuf = async (p) => { const r = await fetchFn(base + p); if (!r.ok) throw new Error(`${p}: HTTP ${r.status}`); return r.arrayBuffer(); };

  // ------------------------------------------------------------------------------------------------------------------
  // data
  // ------------------------------------------------------------------------------------------------------------------
  try {
    say("Loading manifest...");
    state.manifest = await getJSON("data/manifest.json");
    state.tracks = await getJSON(state.manifest.tracks.path);
  } catch (e) {
    fail("Could not load the explorer payload (" + e.message + "). The explorer must be served over http, e.g. " +
      "`python -m melomane.release.explorer serve <dir>`; opening index.html from disk cannot fetch its data.");
    return null;
  }
  const T = state.tracks, N = T.n;
  state.hay = T.title.map((t, i) => M.normText(T.artist[T.artist_idx[i]] + " " + t));
  const artistIdx = Int32Array.from(T.artist_idx);
  let yearMin = Infinity, yearMax = -Infinity;
  for (const y of T.year) if (y != null) { if (y < yearMin) yearMin = y; if (y > yearMax) yearMax = y; }
  if (yearMin > yearMax) { yearMin = 0; yearMax = 1; }
  if (ui.attribution) ui.attribution.textContent = (state.manifest.attribution || []).join(" ") + " Licence: " + state.manifest.license + ".";

  async function loadModel(name) {
    if (state.models[name]) return state.models[name];
    const info = state.manifest.models.find((m) => m.name === name);
    say(`Loading ${info.label}...`);
    const f = info.files;
    const [emb, scale, xy, poles, meta, modes, midx] = await Promise.all([
      getBuf(f.emb.path), f.scale ? getBuf(f.scale.path) : Promise.resolve(null), getBuf(f.umap.path), getBuf(f.poles.path),
      getJSON(f.poles_meta.path), getJSON(f.modes.path), getBuf(f.mode_idx.path)]);
    const d = info.dim;
    const P = new Float32Array(poles);
    const mdl = {
      name, info, d, U: M.decodeEmbeddings(info.dtype, emb, scale, info.n, d), xy: new Float32Array(xy),
      poles: meta.map((_, i) => P.subarray(i * d, (i + 1) * d)), poleMeta: meta, modes, modeIdx: new Int16Array(midx),
      poleByName: new Map(meta.map((p, i) => [p.name, i])), gl: null,
    };
    state.models[name] = mdl;
    return mdl;
  }
  const cur = () => state.models[state.modelName];
  const label = (i) => T.artist[T.artist_idx[i]] + " – " + T.title[i];
  const catName = (i) => (T.category_idx[i] >= 0 ? T.categories[T.category_idx[i]] : "");
  const modeLabel = (i) => { const m = cur().modeIdx[i]; return m >= 0 ? cur().modes[m].label : null; };

  // ------------------------------------------------------------------------------------------------------------------
  // WebGL2
  // ------------------------------------------------------------------------------------------------------------------
  const gl = ui.canvas.getContext("webgl2", { antialias: false, alpha: true });
  if (!gl) { fail("This browser has no WebGL2 support; the point cloud cannot be drawn."); return null; }
  const shader = (type, src) => {
    const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error("shader: " + gl.getShaderInfoLog(s));
    return s;
  };
  const prog = gl.createProgram();
  gl.attachShader(prog, shader(gl.VERTEX_SHADER, VS)); gl.attachShader(prog, shader(gl.FRAGMENT_SHADER, FS));
  gl.bindAttribLocation(prog, 0, "a_pos"); gl.bindAttribLocation(prog, 1, "a_col");
  gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error("link: " + gl.getProgramInfoLog(prog));
  const loc = { scale: gl.getUniformLocation(prog, "u_scale"), off: gl.getUniformLocation(prog, "u_off"),
    size: gl.getUniformLocation(prog, "u_size"), alpha: gl.getUniformLocation(prog, "u_alpha") };
  const posBuf = gl.createBuffer(), colBuf = gl.createBuffer(), idxBuf = gl.createBuffer();
  const vaoBase = gl.createVertexArray(), vaoSet = gl.createVertexArray();
  gl.bindVertexArray(vaoBase);
  gl.bindBuffer(gl.ARRAY_BUFFER, posBuf); gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
  gl.bindBuffer(gl.ARRAY_BUFFER, colBuf); gl.enableVertexAttribArray(1); gl.vertexAttribPointer(1, 4, gl.UNSIGNED_BYTE, true, 0, 0);
  gl.bindVertexArray(vaoSet);
  gl.bindBuffer(gl.ARRAY_BUFFER, posBuf); gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
  gl.disableVertexAttribArray(1);
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, idxBuf);
  gl.bindVertexArray(null);

  let cssW = 1, cssH = 1, dpr = 1, pending = false;
  const raf = win.requestAnimationFrame ? win.requestAnimationFrame.bind(win) : (fn) => setTimeout(fn, 16);
  function requestDraw() { if (!pending) { pending = true; raf(() => { pending = false; draw(); }); } }

  function resize() {
    dpr = win.devicePixelRatio || 1;
    cssW = Math.max(1, ui.canvas.clientWidth || 800); cssH = Math.max(1, ui.canvas.clientHeight || 600);
    ui.canvas.width = Math.round(cssW * dpr); ui.canvas.height = Math.round(cssH * dpr);
    requestDraw();
  }

  function uploadModel() {
    const m = cur();
    gl.bindBuffer(gl.ARRAY_BUFFER, posBuf); gl.bufferData(gl.ARRAY_BUFFER, m.xy, gl.STATIC_DRAW);
    recolour();
  }

  function pointColours() {
    const m = cur(), out = new Uint8Array(N * 4);
    for (let i = 0; i < N; i++) {
      let c = [150, 150, 150];
      if (state.colorBy === "category") c = M.categoricalColor(T.category_idx[i]);
      else if (state.colorBy === "mode") { const k = m.modeIdx[i]; c = k >= 0 ? M.categoricalColor(k) : [170, 170, 170]; }
      else if (state.colorBy === "year") c = T.year[i] == null ? [170, 170, 170] : M.sequentialColor((T.year[i] - yearMin) / Math.max(1, yearMax - yearMin));
      out[4 * i] = c[0]; out[4 * i + 1] = c[1]; out[4 * i + 2] = c[2]; out[4 * i + 3] = 255;
    }
    return out;
  }

  function recolour() {
    gl.bindBuffer(gl.ARRAY_BUFFER, colBuf); gl.bufferData(gl.ARRAY_BUFFER, pointColours(), gl.STATIC_DRAW);
    renderLegend(); requestDraw();
  }

  function drawSet(indices, size, rgba) {
    if (!indices.length) return;
    const halo = dark() ? 1 : 0;
    gl.bindVertexArray(vaoSet);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, idxBuf);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, Uint32Array.from(indices), gl.DYNAMIC_DRAW);
    gl.uniform1f(loc.alpha, 1);
    gl.vertexAttrib4f(1, halo, halo, halo, 1); gl.uniform1f(loc.size, (size + 5) * dpr);
    gl.drawElements(gl.POINTS, indices.length, gl.UNSIGNED_INT, 0);
    gl.vertexAttrib4f(1, rgba[0], rgba[1], rgba[2], 1); gl.uniform1f(loc.size, size * dpr);
    gl.drawElements(gl.POINTS, indices.length, gl.UNSIGNED_INT, 0);
  }

  function draw() {
    const m = cur();
    gl.viewport(0, 0, ui.canvas.width, ui.canvas.height);
    gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT);
    if (!m) return;
    const u = M.clipUniforms(state.view, cssW, cssH);
    gl.useProgram(prog); gl.uniform2f(loc.scale, u.sx, u.sy); gl.uniform2f(loc.off, u.ox, u.oy);
    gl.enable(gl.BLEND); gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    const size = clamp(clamp(320 / Math.sqrt(N), 2.5, 9) * (1 + 0.35 * Math.log2(state.view.zoom)), 2, 20);
    const focus = state.seed >= 0 || state.modeFocus;
    gl.bindVertexArray(vaoBase);
    gl.uniform1f(loc.alpha, focus ? 0.4 : 0.85); gl.uniform1f(loc.size, size * dpr);
    gl.drawArrays(gl.POINTS, 0, N);
    if (state.modeFocus) drawSet(state.modeFocus, size + 1.5, [0.95, 0.62, 0.0]);
    if (state.results) drawSet(Array.from(state.results.idx), size + 3, [0.0, 0.45, 0.70]);
    if (state.hover >= 0) drawSet([state.hover], size + 4, [0.9, 0.9, 0.2]);
    if (state.seed >= 0) drawSet([state.seed], size + 6, [0.84, 0.15, 0.16]);
    gl.bindVertexArray(null);
  }

  // ------------------------------------------------------------------------------------------------------------------
  // queries and panels
  // ------------------------------------------------------------------------------------------------------------------
  function polesForQuery() {
    const m = cur(), vecs = [], w = [];
    for (const p of state.poles) { const i = m.poleByName.get(p.name); if (i !== undefined) { vecs.push(m.poles[i]); w.push(p.weight); } }
    return { vecs, w };
  }

  function runQuery() {
    const m = cur();
    if (!m || state.seed < 0) { state.results = null; renderResults(); requestDraw(); return; }
    const { vecs, w } = polesForQuery();
    const r = M.query({ U: m.U, n: N, d: m.d, seed: state.seed, poles: vecs, weights: w, thetaDeg: state.theta, k: state.k,
      artistIdx, excludeArtist: state.excludeArtist });
    state.results = r;
    renderResults(); requestDraw(); writeHash();
  }

  function searchLinksEl(i) {
    const box = el("span", { class: "links" }, "search on: ");
    for (const l of M.searchLinks(T.artist[T.artist_idx[i]], T.title[i]))
      box.append(el("a", { href: l.url, target: "_blank", rel: "noopener noreferrer", text: l.name }));
    return box;
  }

  function trackMeta(i, extra) {
    const parts = [catName(i), T.year[i] != null ? String(T.year[i]) : "", modeLabel(i) ? "mode: " + modeLabel(i) : "", extra].filter(Boolean);
    return parts.join(" · ");
  }

  function renderResults() {
    ui.results.replaceChildren();
    if (state.seed < 0) { ui.resHead.textContent = "No seed yet."; ui.seedCard.textContent = "Click a point on the map or search to pick a seed track."; return; }
    ui.seedCard.replaceChildren(el("strong", { text: label(state.seed) }), el("div", { class: "muted", text: trackMeta(state.seed) }), searchLinksEl(state.seed));
    const nPoles = polesForQuery().vecs.length;
    ui.resHead.textContent = nPoles && state.theta > 0
      ? `Steered ${state.theta}° toward ${state.poles.map((p) => p.name).join(" + ")}` : "Nearest neighbours of the seed";
    ui.thetaCos.textContent = `(cos to seed = ${Math.cos((state.theta * Math.PI) / 180).toFixed(2)})`;
    const r = state.results;
    if (!r) return;
    r.idx.forEach((i, rank) => {
      const li = el("li", { on: { mouseenter: () => { state.hover = i; requestDraw(); }, mouseleave: () => { state.hover = -1; requestDraw(); } } },
        el("button", { class: "name", type: "button", text: `${rank + 1}. ${label(i)}`, on: { click: () => setSeed(i, { center: true }) } }),
        el("span", { class: "meta", text: trackMeta(i, "cos " + r.score[rank].toFixed(3)) }), searchLinksEl(i));
      ui.results.append(li);
    });
  }

  function setSeed(i, { center = false } = {}) {
    state.seed = i;
    if (center) { const xy = cur().xy; state.view = { cx: xy[2 * i], cy: xy[2 * i + 1], zoom: Math.max(state.view.zoom, 3) }; }
    runQuery();
  }

  function renderLegend() {
    ui.legend.replaceChildren();
    if (state.colorBy === "category") {
      T.categories.forEach((c, i) => { const [r, g, b] = M.categoricalColor(i); ui.legend.append(el("span", {}, el("i", { style: `background:rgb(${r},${g},${b})` }), c)); });
    } else if (state.colorBy === "year") {
      ui.legend.append(el("span", { class: "ramp" }), el("span", { text: String(yearMin) }), el("span", { text: String(yearMax) }));
    } else {
      ui.legend.append(el("span", { text: `${cur().modes.length} modes (hover a point for its mode; grey = in no mode)` }));
    }
  }

  // ---- poles ---------------------------------------------------------------------------------------------------------
  function renderPoleSelect() {
    const m = cur();
    ui.poleSelect.replaceChildren();
    const groups = { supervised: "Supervised", mode: "Emergent: modes", factor: "Emergent: factors" };
    for (const [g, title] of Object.entries(groups)) {
      const og = el("optgroup", { label: title });
      m.poleMeta.filter((p) => p.group === g).forEach((p) => og.append(el("option", { value: p.name, text: p.label ? `${p.name} – ${p.label}` : p.name })));
      if (og.children && og.children.length) ui.poleSelect.append(og);
    }
  }

  function renderChips() {
    ui.chips.replaceChildren();
    state.poles.forEach((p, i) => {
      ui.chips.append(el("li", {},
        el("span", { text: p.name }),
        el("input", { type: "number", min: "0", step: "0.1", value: String(p.weight), "aria-label": `weight of ${p.name}`,
          on: { change: (e) => { p.weight = Math.max(0, parseFloat(e.target.value) || 0); runQuery(); } } }),
        el("button", { type: "button", "aria-label": `remove ${p.name}`, text: "×", on: { click: () => { state.poles.splice(i, 1); renderChips(); runQuery(); } } })));
    });
  }

  function addPole(name) {
    if (!name || state.poles.some((p) => p.name === name)) return;
    state.poles.push({ name, weight: 1 }); renderChips(); runQuery();
  }

  // ---- mode atlas ----------------------------------------------------------------------------------------------------
  function renderModeList() {
    const m = cur(), f = M.normText(ui.modeFilter.value);
    ui.modeList.replaceChildren();
    m.modes.forEach((mode, i) => {
      const text = M.normText([mode.label, mode.property, mode.driver].join(" "));
      if (f && !text.includes(f)) return;
      ui.modeList.append(el("li", {}, el("button", { type: "button", text: `${mode.label} (${mode.n_in_payload}/${mode.size})`, on: { click: () => showMode(i) } })));
    });
  }

  function showMode(i) {
    const m = cur(), mode = m.modes[i];
    state.modeFocus = mode.members; state.modeFocusId = mode.id;
    const pi = m.poleByName.get(`mode:${mode.id}`);
    let members = mode.members.slice();
    if (pi !== undefined) {
      const pole = m.poles[pi], sc = new Map(members.map((t) => [t, M.dot(m.U.subarray(t * m.d, (t + 1) * m.d), pole)]));
      members.sort((a, b) => sc.get(b) - sc.get(a));
    }
    ui.modeDetail.replaceChildren(
      el("strong", { text: mode.label }),
      el("div", { class: "muted", text: `property: ${mode.property ?? "-"} · driver: ${mode.driver ?? "-"} · factor ${mode.factor ?? "-"} · coherence ${mode.coherence != null ? Number(mode.coherence).toFixed(3) : "-"} · ${mode.n_in_payload} of ${mode.size} members shown` }),
      el("div", { class: "row" }, el("button", { type: "button", text: "Steer toward this mode", on: { click: () => addPole(`mode:${mode.id}`) } }),
        el("button", { type: "button", text: "Clear highlight", on: { click: () => { state.modeFocus = null; state.modeFocusId = null; ui.modeDetail.replaceChildren(); requestDraw(); } } })));
    const ol = el("ol", { class: "results" });
    for (const t of members.slice(0, 100))
      ol.append(el("li", { on: { mouseenter: () => { state.hover = t; requestDraw(); }, mouseleave: () => { state.hover = -1; requestDraw(); } } },
        el("button", { class: "name", type: "button", text: label(t), on: { click: () => setSeed(t, { center: true }) } }),
        el("span", { class: "meta", text: trackMeta(t) })));
    ui.modeDetail.append(ol);
    requestDraw();
  }

  function selectTab(which) {
    const res = which === "results";
    ui.panelRes.hidden = !res; ui.panelModes.hidden = res;
    ui.tabRes.setAttribute("aria-selected", String(res)); ui.tabModes.setAttribute("aria-selected", String(!res));
  }

  // ---- model toggle --------------------------------------------------------------------------------------------------
  function renderToggle() {
    ui.toggle.replaceChildren();
    for (const info of state.manifest.models)
      ui.toggle.append(el("button", { type: "button", role: "radio", "aria-checked": String(info.name === state.modelName), text: info.label,
        on: { click: () => switchModel(info.name) } }));
  }

  async function switchModel(name) {
    try { await loadModel(name); } catch (e) { fail("Could not load model " + name + ": " + e.message); return; }
    state.modelName = name;
    state.poles = state.poles.filter((p) => cur().poleByName.has(p.name));
    state.modeFocus = null; state.modeFocusId = null; ui.modeDetail.replaceChildren();
    renderToggle(); renderPoleSelect(); renderChips(); renderModeList(); uploadModel(); runQuery();
    say(`${cur().info.label}: ${N} tracks, ${cur().d}-d ${cur().info.dtype}, ${cur().modes.length} modes`);
  }

  // ---- url hash ------------------------------------------------------------------------------------------------------
  function writeHash() {
    try {
      const p = new URLSearchParams();
      p.set("m", state.modelName); if (state.seed >= 0) p.set("s", String(T.cid[state.seed]));
      p.set("t", String(state.theta)); if (state.poles.length) p.set("p", state.poles.map((x) => `${x.name}*${x.weight}`).join(","));
      win.history.replaceState(null, "", "#" + p.toString());
    } catch (_) { /* hash is a convenience only */ }
  }
  function readHash() {
    const h = (win.location && win.location.hash || "").replace(/^#/, ""), p = new URLSearchParams(h), o = {};
    if (p.get("m")) o.model = p.get("m");
    if (p.get("s") && /^\d+$/.test(p.get("s"))) o.seedCid = parseInt(p.get("s"), 10);
    if (p.get("t") && /^\d+$/.test(p.get("t"))) o.theta = clamp(parseInt(p.get("t"), 10), 0, 90);
    if (p.get("p")) o.poles = p.get("p").split(",").map((s) => { const [name, w] = s.split("*"); return { name, weight: parseFloat(w) || 1 }; });
    return o;
  }

  // ------------------------------------------------------------------------------------------------------------------
  // events
  // ------------------------------------------------------------------------------------------------------------------
  ui.theta.addEventListener("input", () => { state.theta = parseInt(ui.theta.value, 10); ui.thetaOut.textContent = String(state.theta); runQuery(); });
  ui.kSel.addEventListener("change", () => { state.k = parseInt(ui.kSel.value, 10); runQuery(); });
  ui.excl.addEventListener("change", () => { state.excludeArtist = !!ui.excl.checked; runQuery(); });
  ui.colorBy.addEventListener("change", () => { state.colorBy = ui.colorBy.value; if (cur()) recolour(); });
  ui.poleAdd.addEventListener("click", () => addPole(ui.poleSelect.value));
  ui.tabRes.addEventListener("click", () => selectTab("results"));
  ui.tabModes.addEventListener("click", () => selectTab("modes"));
  ui.modeFilter.addEventListener("input", renderModeList);
  ui.seedSearch.addEventListener("input", () => {
    ui.suggest.replaceChildren();
    for (const i of M.searchTracks(state.hay, ui.seedSearch.value, 8))
      ui.suggest.append(el("li", { role: "option" }, el("button", { type: "button", text: label(i), on: { click: () => { ui.suggest.replaceChildren(); ui.seedSearch.value = ""; setSeed(i, { center: true }); } } })));
  });
  const zoomBy = (f) => { state.view = M.zoomAt(state.view, cssW, cssH, cssW / 2, cssH / 2, f); requestDraw(); };
  ui.zin.addEventListener("click", () => zoomBy(1.6));
  ui.zout.addEventListener("click", () => zoomBy(1 / 1.6));
  ui.zreset.addEventListener("click", () => { state.view = { cx: 0, cy: 0, zoom: 0.95 }; requestDraw(); });

  let drag = null;
  const local = (e) => { const r = ui.canvas.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; };
  const pick = (sx, sy) => M.pickNearest(cur().xy, N, state.view, cssW, cssH, sx, sy, 9, null);
  ui.canvas.addEventListener("pointerdown", (e) => {
    const [x, y] = local(e); drag = { x, y, sx: x, sy: y, moved: false };
    if (ui.canvas.setPointerCapture) ui.canvas.setPointerCapture(e.pointerId);
    ui.canvas.classList.add("dragging");
  });
  ui.canvas.addEventListener("pointermove", (e) => {
    const [x, y] = local(e);
    if (drag) {
      if (Math.abs(x - drag.sx) + Math.abs(y - drag.sy) > 4) drag.moved = true;
      if (drag.moved) { state.view = M.panBy(state.view, cssW, cssH, x - drag.x, y - drag.y); drag.x = x; drag.y = y; ui.tooltip.hidden = true; requestDraw(); return; }
    }
    if (!cur()) return;
    const i = pick(x, y);
    if (i !== state.hover) { state.hover = i; requestDraw(); }
    if (i >= 0) {
      ui.tooltip.hidden = false; ui.tooltip.textContent = `${label(i)} — ${trackMeta(i)}`;
      ui.tooltip.style.left = Math.min(x + 12, cssW - 200) + "px"; ui.tooltip.style.top = Math.max(0, y - 30) + "px";
    } else ui.tooltip.hidden = true;
  });
  const endDrag = (e) => {
    ui.canvas.classList.remove("dragging");
    if (drag && !drag.moved && cur()) { const [x, y] = local(e); const i = pick(x, y); if (i >= 0) setSeed(i); }
    drag = null;
  };
  ui.canvas.addEventListener("pointerup", endDrag);
  ui.canvas.addEventListener("pointercancel", () => { drag = null; ui.canvas.classList.remove("dragging"); });
  ui.canvas.addEventListener("pointerleave", () => { state.hover = -1; ui.tooltip.hidden = true; requestDraw(); });
  ui.canvas.addEventListener("wheel", (e) => {
    if (e.preventDefault) e.preventDefault();
    const [x, y] = local(e); state.view = M.zoomAt(state.view, cssW, cssH, x, y, Math.exp(-e.deltaY * 0.0015)); requestDraw();
  }, { passive: false });
  ui.canvas.addEventListener("dblclick", () => { state.view = { cx: 0, cy: 0, zoom: 0.95 }; requestDraw(); });
  if (win.addEventListener) win.addEventListener("resize", resize);
  if (win.ResizeObserver) new win.ResizeObserver(resize).observe(ui.stage);

  // ------------------------------------------------------------------------------------------------------------------
  // go
  // ------------------------------------------------------------------------------------------------------------------
  const h = readHash();
  const first = state.manifest.models.some((m) => m.name === h.model) ? h.model : state.manifest.models[0].name;
  if (h.theta !== undefined) { state.theta = h.theta; ui.theta.value = String(h.theta); ui.thetaOut.textContent = String(h.theta); }
  resize();
  await switchModel(first);
  if (h.poles) { for (const p of h.poles) if (cur().poleByName.has(p.name)) state.poles.push(p); renderChips(); }
  if (h.seedCid !== undefined) { const i = T.cid.indexOf(h.seedCid); if (i >= 0) setSeed(i, { center: true }); }
  renderLegend();
  return { state, setSeed, addPole, switchModel, runQuery, showMode, draw, resize };
}

if (typeof document !== "undefined" && !globalThis.__MELOMANE_NO_AUTOSTART__) {
  start().catch((e) => {
    const box = document.getElementById("error");
    if (box) { box.hidden = false; box.textContent = "Explorer failed to start: " + e.message; }
    console.error(e);
  });
}
