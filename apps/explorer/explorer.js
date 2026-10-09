// explorer.js - Melomane static explorer (no dependencies, no CDN, no runtime network beyond this site's own payload files).
// Everything numeric lives in math.js; this file is the DOM + WebGL2 shell. All strings from the payload are inserted with
// textContent (never innerHTML). Audio previews are "search on..." links only.
import * as M from "./math.js";

const VS = `#version 300 es
in vec2 a_pos; in vec4 a_col;
uniform vec2 u_scale; uniform vec2 u_off; uniform float u_size;
out vec4 v_col;
void main() { gl_Position = vec4(a_pos * u_scale + u_off, 0.0, 1.0); gl_PointSize = u_size; v_col = a_col; }`;
// Two passes share this program.  u_halo = 1: a wide, faint, *additive* glow (dense regions bloom like a nebula);
// u_halo = 0: a crisp anti-aliased disc with a small specular core, composited "over".  Output is premultiplied alpha.
const FS = `#version 300 es
precision mediump float;
in vec4 v_col; uniform float u_alpha; uniform float u_halo; out vec4 o;
void main() {
  vec2 c = gl_PointCoord * 2.0 - 1.0; float r2 = dot(c, c);
  if (r2 > 1.0) discard;
  if (u_halo > 0.5) {
    float g = exp(-r2 * 4.6) * 0.16 * u_alpha;
    o = vec4(v_col.rgb * g, g * 0.55);
  } else {
    float edge = 1.0 - smoothstep(0.70, 1.0, r2);          // (smoothstep with edge0 > edge1 is undefined in GLSL)
    float shine = 1.0 - smoothstep(0.0, 0.50, r2);
    vec3 col = mix(v_col.rgb, vec3(1.0), shine * 0.34);
    float a = edge * v_col.a * u_alpha;
    o = vec4(col * a, a);
  }
}`;

const clamp = (x, a, b) => Math.min(b, Math.max(a, x));

const MODEL_LABELS = {
  cooc: "Cooc (Streaming)",
  composition: "Composition (Audio)",
  core: "Core (Hybrid)"
};

const MODEL_DESCRIPTIONS = {
  cooc: "Reflects popular streaming habits and playlists.",
  composition: "Matches tracks by raw acoustic features and timbre.",
  core: "The master hybrid blend of streaming habits and physical acoustic properties."
};

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

/** Boot the explorer. `env` may inject {document, window, fetch, base} (used by the node smoke test). */
export async function start(env = {}) {
  const doc = env.document ?? document, win = env.window ?? window;
  const fetchFn = env.fetch ?? fetch.bind(globalThis);
  const base = env.base ?? "";
  const $ = (id) => doc.getElementById(id);
  const el = makeEl(doc);
  const dark = () => !!(win.matchMedia && win.matchMedia("(prefers-color-scheme: dark)").matches);

  const ui = {
    status: $("status"), error: $("error"), toggle: $("model-toggle"), modelDesc: $("model-desc"),
    toggleHelp: $("toggle-help"), closeHelp: $("close-help"), helpDrawer: $("help-drawer"),
    btnOpenGuide: $("btn-open-guide"), btnModelsInfo: $("btn-models-info"),
    welcomeModal: $("welcome-modal"), closeWelcome: $("close-welcome"),
    btnStartTsDemo: $("btn-start-ts-demo"), btnDemoTsCp: $("btn-demo-ts-cp"),
    btnExploreNow: $("btn-explore-now"), dontShowWelcome: $("dont-show-welcome"),
    modelsModal: $("models-modal"), closeModelsModal: $("close-models-modal"), closeModelsModalBtn: $("close-models-modal-btn"),
    tourBanner: $("tour-banner"), tourStepBadge: $("tour-step-badge"), tourStepTitle: $("tour-step-title"),
    tourStepText: $("tour-step-text"), tourActionBtn: $("tour-action-btn"), tourPrevBtn: $("tour-prev-btn"),
    tourNextBtn: $("tour-next-btn"), tourExitBtn: $("tour-exit-btn"),
    seedSearch: $("seed-search"), suggest: $("seed-suggest"),
    seedCard: $("seed-card"), showArtists: $("show-artists"),
    poleSelect: $("pole-select"), poleAdd: $("pole-add"), chips: $("pole-chips"), theta: $("theta"),
    thetaOut: $("theta-out"), thetaDesc: $("theta-desc"), thetaCos: $("theta-cos"), kSel: $("k-select"), excl: $("exclude-artist"), colorBy: $("color-by"),
    legend: $("legend"), canvas: $("map"), overlay: $("overlay"), tooltip: $("tooltip"), stage: $("stage"),
    results: $("results"), resHead: $("results-head"),
    tabRes: $("tab-results"), tabModes: $("tab-modes"), panelRes: $("panel-results"), panelModes: $("panel-modes"),
    modeFilter: $("mode-filter"), modeList: $("mode-list"), modeDetail: $("mode-detail"), attribution: $("attribution"),
    zin: $("zoom-in"), zout: $("zoom-out"), zreset: $("zoom-reset"), mapLegend: $("map-legend"), splash: $("splash"), splashMsg: $("splash-msg"),
  };

  const state = {
    manifest: null, tracks: null, hay: null, modelName: null, models: {}, seed: -1, poles: [], theta: 30, k: 10,
    excludeArtist: true, colorBy: "category", view: { cx: 0, cy: 0, zoom: 0.95 }, hover: -1, results: null, modeFocus: null,
    modeFocusId: null,
  };
  const hideSplash = () => { if (ui.splash && ui.splash.classList) ui.splash.classList.add("done"); };
  const fail = (msg) => { if (ui.error) { ui.error.hidden = false; ui.error.textContent = msg; } if (ui.status) ui.status.textContent = ""; hideSplash(); };
  const say = (msg) => { if (ui.status) ui.status.textContent = msg; if (ui.splashMsg) ui.splashMsg.textContent = msg; };
  const getJSON = async (p) => { const r = await fetchFn(base + p); if (!r.ok) throw new Error(`${p}: HTTP ${r.status}`); return r.json(); };
  const getBuf = async (p) => { const r = await fetchFn(base + p); if (!r.ok) throw new Error(`${p}: HTTP ${r.status}`); return r.arrayBuffer(); };

  // ------------------------------------------------------------------------------------------------------------------
  // data
  // ------------------------------------------------------------------------------------------------------------------
  try {
    say("Initializing knowledge graph...");
    state.manifest = await getJSON("data/manifest.json");
    state.tracks = await getJSON(state.manifest.tracks.path);
  } catch (e) {
    fail("Could not load the explorer payload (" + e.message + "). The explorer must be served over http, e.g. " +
      "`python -m melomane.release.explorer serve <dir>`; opening index.html from disk cannot fetch its data.");
    return null;
  }
  const T = state.tracks, N = T.n;
  state.hay = T.title.map((t, i) => M.normText((T.artist[T.artist_idx[i]] || "") + " " + t));
  const artistIdx = Int32Array.from(T.artist_idx);
  let yearMin = Infinity, yearMax = -Infinity;
  for (const y of T.year) if (y != null) { if (y < yearMin) yearMin = y; if (y > yearMax) yearMax = y; }
  if (yearMin > yearMax) { yearMin = 0; yearMax = 1; }
  const counts = { n: N.toLocaleString("en-GB"), n_vocab: (state.manifest.n_vocab || N).toLocaleString("en-GB") };
  if (doc.querySelectorAll) doc.querySelectorAll("[data-fill]").forEach((node) => { node.textContent = counts[node.dataset.fill] ?? node.textContent; });
  if (ui.seedSearch) ui.seedSearch.placeholder = `Search ${counts.n} tracks or artists…`;
  if (ui.attribution) ui.attribution.textContent = (state.manifest.attribution || []).join(" ") + " Licence: " + state.manifest.license + ".";

  async function loadModel(name) {
    if (state.models[name]) return state.models[name];
    const info = state.manifest.models.find((m) => m.name === name);
    say(`Loading ${info.label}...`);
    const f = info.files;
    const [emb, scale, xy, poles, meta, modes, midx, artists] = await Promise.all([
      getBuf(f.emb.path), f.scale ? getBuf(f.scale.path) : Promise.resolve(null), getBuf(f.umap.path), getBuf(f.poles.path),
      getJSON(f.poles_meta.path), getJSON(f.modes.path), getBuf(f.mode_idx.path),
      getJSON("data/" + name + "/artists.json").catch(() => [])
    ]);
    const d = info.dim;
    const P = new Float32Array(poles);
    const mdl = {
      name, info, d, U: M.decodeEmbeddings(info.dtype, emb, scale, info.n, d), xy: new Float32Array(xy),
      poles: meta.map((_, i) => P.subarray(i * d, (i + 1) * d)), poleMeta: meta, modes, modeIdx: new Int16Array(midx),
      poleByName: new Map(meta.map((p, i) => [p.name, i])), artists: artists || [], gl: null,
    };
    state.models[name] = mdl;
    return mdl;
  }
  const cur = () => state.models[state.modelName];
  const label = (i) => {
    const art = T.artist[T.artist_idx[i]] || "Unknown Artist";
    const tit = T.title[i] || `Track #${T.cid[i]}`;
    return `${art} – ${tit}`;
  };
  const catName = (i) => (T.category_idx[i] >= 0 ? T.categories[T.category_idx[i]] : "");
  const modeLabel = (i) => { const m = cur().modeIdx[i]; return m >= 0 ? cur().modes[m].label : null; };

  // ------------------------------------------------------------------------------------------------------------------
  // WebGL2 Stage
  // ------------------------------------------------------------------------------------------------------------------
  const gl = ui.canvas.getContext("webgl2", { antialias: false, alpha: true, premultipliedAlpha: true });
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
    size: gl.getUniformLocation(prog, "u_size"), alpha: gl.getUniformLocation(prog, "u_alpha"), halo: gl.getUniformLocation(prog, "u_halo") };
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
    if (ui.overlay) {
      ui.overlay.width = Math.round(cssW * dpr);
      ui.overlay.height = Math.round(cssH * dpr);
    }
    requestDraw();
  }

  function uploadModel() {
    const m = cur();
    gl.bindBuffer(gl.ARRAY_BUFFER, posBuf); gl.bufferData(gl.ARRAY_BUFFER, m.xy, gl.STATIC_DRAW);
    recolour();
  }

  // ---- colours: palette ranked by category size, so the biggest genres get the most distinct hues ----
  const catCount = new Int32Array(Math.max(1, T.categories.length));
  for (let i = 0; i < N; i++) if (T.category_idx[i] >= 0) catCount[T.category_idx[i]]++;
  const catRank = new Int32Array(catCount.length);
  [...catCount.keys()].sort((a, b) => catCount[b] - catCount[a]).forEach((c, r) => { catRank[c] = r; });
  const catRGB = (c) => (c >= 0 ? M.categoricalColor(catRank[c]) : [118, 130, 150]);
  const rgba = (c, a) => `rgba(${c[0]},${c[1]},${c[2]},${a})`;

  function pointColours() {
    const m = cur(), out = new Uint8Array(N * 4);
    for (let i = 0; i < N; i++) {
      let c = [120, 140, 175];
      if (state.colorBy === "category") c = catRGB(T.category_idx[i]);
      else if (state.colorBy === "mode") { const k = m.modeIdx[i]; c = k >= 0 ? M.categoricalColor(k) : [100, 115, 135]; }
      else if (state.colorBy === "year") c = T.year[i] == null ? [100, 115, 135] : M.sequentialColor((T.year[i] - yearMin) / Math.max(1, yearMax - yearMin));
      out[4 * i] = c[0]; out[4 * i + 1] = c[1]; out[4 * i + 2] = c[2]; out[4 * i + 3] = 255;
    }
    return out;
  }

  function recolour() {
    gl.bindBuffer(gl.ARRAY_BUFFER, colBuf); gl.bufferData(gl.ARRAY_BUFFER, pointColours(), gl.STATIC_DRAW);
    renderLegend(); requestDraw();
  }

  /** Highlight a set of points: a coloured bloom, a white rim, then the coloured core (rgba01 are 0..1 floats). */
  function drawSet(indices, size, rgba01) {
    if (!indices.length) return;
    gl.bindVertexArray(vaoSet);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, idxBuf);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, Uint32Array.from(indices), gl.DYNAMIC_DRAW);
    gl.uniform1f(loc.alpha, 1);
    gl.blendFunc(gl.ONE, gl.ONE); gl.uniform1f(loc.halo, 1);
    gl.vertexAttrib4f(1, rgba01[0], rgba01[1], rgba01[2], 1); gl.uniform1f(loc.size, Math.min(size * 4.4, 64) * dpr);
    gl.drawElements(gl.POINTS, indices.length, gl.UNSIGNED_INT, 0);
    gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA); gl.uniform1f(loc.halo, 0);
    gl.vertexAttrib4f(1, 1, 1, 1, 0.94); gl.uniform1f(loc.size, (size + 2.8) * dpr);
    gl.drawElements(gl.POINTS, indices.length, gl.UNSIGNED_INT, 0);
    gl.vertexAttrib4f(1, rgba01[0], rgba01[1], rgba01[2], 1); gl.uniform1f(loc.size, size * dpr);
    gl.drawElements(gl.POINTS, indices.length, gl.UNSIGNED_INT, 0);
  }

  function draw() {
    const m = cur();
    gl.viewport(0, 0, ui.canvas.width, ui.canvas.height);
    gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT);        // transparent: the stage's CSS gradient shows through
    if (!m) return;
    const u = M.clipUniforms(state.view, cssW, cssH);
    gl.useProgram(prog); gl.uniform2f(loc.scale, u.sx, u.sy); gl.uniform2f(loc.off, u.ox, u.oy);
    gl.enable(gl.BLEND);
    const size = clamp(clamp(440 / Math.sqrt(N), 3.4, 12) * (1 + 0.42 * Math.log2(state.view.zoom)), 3.2, 26);
    const focus = state.seed >= 0 || state.modeFocus;
    gl.bindVertexArray(vaoBase);
    gl.blendFunc(gl.ONE, gl.ONE); gl.uniform1f(loc.halo, 1);                         // pass 1: additive bloom
    gl.uniform1f(loc.alpha, focus ? 0.45 : 1.0); gl.uniform1f(loc.size, Math.min(size * 3.8, 56) * dpr);
    gl.drawArrays(gl.POINTS, 0, N);
    gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA); gl.uniform1f(loc.halo, 0);         // pass 2: crisp cores
    gl.uniform1f(loc.alpha, focus ? 0.62 : 0.96); gl.uniform1f(loc.size, size * dpr);
    gl.drawArrays(gl.POINTS, 0, N);
    if (state.modeFocus) drawSet(state.modeFocus, size + 3, [1.0, 0.85, 0.25]);
    if (state.results) drawSet(Array.from(state.results.idx), size + 5, [0.55, 0.92, 1.0]);
    if (state.hover >= 0) drawSet([state.hover], size + 6, [1.0, 1.0, 1.0]);
    if (state.seed >= 0) drawSet([state.seed], size + 9, [1.0, 0.42, 0.55]);
    gl.bindVertexArray(null);
    drawOverlay(m);
  }

  // ------------------------------------------------------------------------------------------------------------------
  // Overlay (2-D canvas, CSS-pixel coordinates): curved neighbour links + collision-free pill labels
  // ------------------------------------------------------------------------------------------------------------------
  const trunc = (t, n) => (t.length > n ? t.slice(0, n - 1) + "…" : t);
  const textW = (ctx, t) => { const w = ctx.measureText(t).width; return Number.isFinite(w) ? w : t.length * 6.6; };
  function pill(ctx, x, y, w, h, fill, stroke) {
    ctx.beginPath();
    if (ctx.roundRect) ctx.roundRect(x, y, w, h, h / 2.4); else ctx.rect(x, y, w, h);
    ctx.fillStyle = fill; ctx.fill();
    if (stroke) { ctx.lineWidth = 1; ctx.strokeStyle = stroke; ctx.stroke(); }
  }
  const FONT = '"Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif';
  const LABEL_STYLE = {
    seed: { font: `700 13px ${FONT}`, h: 26, fill: "#ffffff", text: "#0b0e16" },
    A:    { font: `600 12px ${FONT}`, h: 23, fill: "rgba(244,246,251,0.96)", text: "#0f1320" },
    B:    { font: `500 11px ${FONT}`, h: 20, fill: "rgba(22,26,38,0.80)", text: "#dde4f3", stroke: "rgba(255,255,255,0.16)" },
    nb:   { font: `600 11.5px ${FONT}`, h: 22, text: "#0b0e16" },
  };

  function drawOverlay(m) {
    if (!ui.overlay) return;
    const ctx = ui.overlay.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cssW, cssH);
    state.labelBoxes = [];
    const s = state.view.zoom * Math.min(cssW, cssH) / 2;
    const sx = (x) => (x - state.view.cx) * s + cssW / 2, sy = (y) => cssH / 2 - (y - state.view.cy) * s;
    const cands = [];
    const zoom = state.view.zoom;

    // seed -> neighbours: curved, genre-coloured links with a soft glow, then one label each
    if (state.seed >= 0 && state.results && state.results.idx.length) {
      const S0 = sx(m.xy[2 * state.seed]), S1 = sy(m.xy[2 * state.seed + 1]);
      const rIdx = state.results.idx, rSc = state.results.score, n = Math.min(rIdx.length, 12);
      ctx.lineCap = "round";
      for (let r = 0; r < n; r++) {
        const nb = rIdx[r], X = sx(m.xy[2 * nb]), Y = sy(m.xy[2 * nb + 1]);
        const dx = X - S0, dy = Y - S1, len = Math.hypot(dx, dy) || 1;
        const col = catRGB(T.category_idx[nb]), sc = clamp(rSc[r], 0, 1);
        ctx.shadowColor = rgba(col, 0.85); ctx.shadowBlur = 9 * dpr;
        ctx.strokeStyle = rgba(col, 0.28 + 0.5 * sc); ctx.lineWidth = 1 + 1.5 * sc;
        ctx.beginPath(); ctx.moveTo(S0, S1);
        ctx.quadraticCurveTo((S0 + X) / 2 - (dy / len) * len * 0.16, (S1 + Y) / 2 + (dx / len) * len * 0.16, X, Y);
        ctx.stroke();
      }
      ctx.shadowBlur = 0;
      ctx.strokeStyle = "rgba(255,255,255,0.9)"; ctx.lineWidth = 1.6;
      ctx.beginPath(); ctx.arc(S0, S1, 8, 0, Math.PI * 2); ctx.stroke();
      ctx.strokeStyle = "rgba(255,255,255,0.22)"; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.arc(S0, S1, 13, 0, Math.PI * 2); ctx.stroke();
      cands.push({ id: "seed", x: S0, y: S1, text: trunc(label(state.seed), 44), style: "seed", required: true });
      for (let r = 0; r < Math.min(n, 8); r++) {
        const nb = rIdx[r];
        cands.push({ id: "nb" + r, x: sx(m.xy[2 * nb]), y: sy(m.xy[2 * nb + 1]), text: `${r + 1}  ${trunc(label(nb), 38)}`, style: "nb", col: catRGB(T.category_idx[nb]) });
      }
    }

    // artist landmarks (most important first; the placer drops whatever would overlap)
    if (!ui.showArtists || ui.showArtists.checked) {
      const maxShow = zoom < 1.3 ? 36 : (zoom < 2.5 ? 80 : 150);
      (m.artists || []).slice(0, maxShow).forEach((art, rank) =>
        cands.push({ id: "a" + rank, x: sx(art.x), y: sy(art.y), text: trunc(art.name, 26), style: rank < 10 ? "A" : "B", track: art.top_track_idx, fan: false }));
    }

    for (const c of cands) {
      const st = LABEL_STYLE[c.style];
      ctx.font = st.font; c.w = Math.ceil(textW(ctx, c.text)) + 18; c.h = st.h;
    }
    const cap = (zoom < 1.3 ? 14 : zoom < 2.5 ? 30 : 60) + (state.seed >= 0 ? 9 : 0);
    const placed = M.placeLabels(cands, cssW, cssH, { pad: 4, gap: 8, maxLabels: cap });
    ctx.lineWidth = 1; ctx.strokeStyle = "rgba(255,255,255,0.38)";      // leader lines for labels that had to move off their point
    for (const c of placed) {
      if (!c.leader) continue;
      ctx.beginPath(); ctx.moveTo(c.x, c.y); ctx.lineTo(clamp(c.x, c.x0, c.x0 + c.w), clamp(c.y, c.y0, c.y0 + c.h)); ctx.stroke();
    }
    for (const order of ["B", "A", "nb", "seed"]) {
      for (const c of placed) {
        if (c.style !== order) continue;
        const st = LABEL_STYLE[c.style];
        if (c.style === "A" || c.style === "B") {                          // anchor dot on the landmark
          ctx.fillStyle = "rgba(255,255,255,0.75)"; ctx.beginPath(); ctx.arc(c.x, c.y, 2.2, 0, Math.PI * 2); ctx.fill();
        }
        ctx.shadowColor = "rgba(0,0,0,0.5)"; ctx.shadowBlur = (c.style === "B" ? 4 : 8) * dpr; ctx.shadowOffsetY = 1;
        pill(ctx, c.x0, c.y0, c.w, c.h, c.style === "nb" ? rgba(c.col, 0.96) : st.fill, st.stroke);
        ctx.shadowBlur = 0; ctx.shadowOffsetY = 0;
        ctx.font = st.font; ctx.fillStyle = st.text; ctx.textAlign = "center"; ctx.textBaseline = "middle";
        ctx.fillText(c.text, c.x0 + c.w / 2, c.y0 + c.h / 2 + 0.5);
        if (c.track !== undefined && c.track !== null) state.labelBoxes.push({ x0: c.x0, y0: c.y0, w: c.w, h: c.h, track: c.track });
      }
    }
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
    const pq = polesForQuery();
    state.results = M.query({
      U: m.U, n: N, d: m.d, seed: state.seed, poles: pq.vecs, weights: pq.w,
      thetaDeg: state.theta, k: state.k, artistIdx, excludeArtist: state.excludeArtist,
    });
    renderResults(); requestDraw(); writeHash();
  }

  function searchLinksEl(i) {
    const box = el("span", { class: "links" });
    const art = T.artist[T.artist_idx[i]] || "";
    const tit = T.title[i] || "";
    for (const l of M.searchLinks(art, tit))
      box.append(el("a", { href: l.url, target: "_blank", rel: "noopener noreferrer", text: l.name + " ↗" }));
    return box;
  }

  function trackMeta(i, extra) {
    const parts = [catName(i), T.year[i] != null ? String(T.year[i]) : "", modeLabel(i) ? "mode: " + modeLabel(i) : "", extra].filter(Boolean);
    return parts.join(" · ");
  }

  function updateThetaDesc() {
    const t = state.theta;
    let desc = "(Subtle Fusion)";
    if (t === 0) desc = "(Original Track Neighbors)";
    else if (t <= 15) desc = "(Mild Flavor Infusion)";
    else if (t <= 35) desc = "(Subtle Fusion)";
    else if (t <= 60) desc = "(Bold Crossover)";
    else desc = "(Total Genre Transformation)";
    if (ui.thetaDesc) ui.thetaDesc.textContent = desc;
  }

  function renderResults() {
    ui.results.replaceChildren();
    if (state.seed < 0) { ui.resHead.textContent = "No seed yet."; ui.seedCard.textContent = "Click a point on the map, search an artist, or try a demo above."; return; }
    ui.seedCard.replaceChildren(el("strong", { text: label(state.seed) }), el("div", { class: "muted", text: trackMeta(state.seed) }), searchLinksEl(state.seed));
    const nPoles = polesForQuery().vecs.length;
    ui.resHead.textContent = nPoles && state.theta > 0
      ? `Steered ${state.theta}° toward ${state.poles.map((p) => p.name).join(" + ")}` : "Nearest neighbours of the seed";
    if (ui.thetaCos) ui.thetaCos.textContent = `(cos to seed = ${Math.cos((state.theta * Math.PI) / 180).toFixed(2)})`;
    updateThetaDesc();
    const r = state.results;
    if (!r) return;
    r.idx.forEach((i, rank) => {
      const li = el("li", { on: { mouseenter: () => { state.hover = i; requestDraw(); }, mouseleave: () => { state.hover = -1; requestDraw(); } } },
        el("button", { class: "name", type: "button", text: `${rank + 1}. ${label(i)}`, on: { click: () => setSeed(i, { center: true }) } }),
        el("span", { class: "meta", text: trackMeta(i, "cos " + r.score[rank].toFixed(3)) }), searchLinksEl(i));
      ui.results.append(li);
    });
  }

  /** View that frames the given tracks (seed + neighbours) with room for their labels. */
  function fitIndices(idxs, margin = 0.7) {
    const xy = cur().xy;
    let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
    for (const t of idxs) { const x = xy[2 * t], y = xy[2 * t + 1]; if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y; }
    const hx = Math.max(0.004, (x1 - x0) / 2), hy = Math.max(0.004, (y1 - y0) / 2), mm = Math.min(cssW, cssH);
    return { cx: (x0 + x1) / 2, cy: (y0 + y1) / 2, zoom: clamp(margin * Math.min(cssW / (hx * mm), cssH / (hy * mm)), 0.6, 60) };
  }

  /** Select a seed.  With `center` the view is re-framed around the seed *and* its neighbours (they are close in the embedding,
   *  but the 2-D layout only approximates that, so they can be far apart on screen). */
  function setSeed(i, { center = false } = {}) {
    state.seed = i;
    runQuery();
    if (center && cur()) {
      state.view = state.results && state.results.idx.length ? fitIndices([i, ...Array.from(state.results.idx)]) : fitIndices([i]);
      requestDraw();
    }
  }

  function renderLegend() {
    ui.legend.replaceChildren();
    renderMapLegend();
    if (state.colorBy === "category") {
      T.categories.forEach((c, i) => { const [r, g, b] = catRGB(i); ui.legend.append(el("span", {}, el("i", { style: `background:rgb(${r},${g},${b});box-shadow:0 0 6px rgba(${r},${g},${b},.8)` }), c)); });
    } else if (state.colorBy === "year") {
      ui.legend.append(el("span", { class: "ramp" }), el("span", { text: String(yearMin) }), el("span", { text: String(yearMax) }));
    } else {
      ui.legend.append(el("span", { text: `${cur().modes.length} emergent modes (hover point for mode; grey = unassigned)` }));
    }
  }

  /** Compact legend floating on the map: the six biggest genres (or the year ramp). */
  function renderMapLegend() {
    const box = ui.mapLegend;
    if (!box) return;
    box.replaceChildren();
    if (state.colorBy === "category") {
      box.append(el("b", { text: "Genre" }));
      [...catCount.keys()].sort((a, b) => catCount[b] - catCount[a]).slice(0, 6).forEach((c) => {
        const [r, g, b] = catRGB(c);
        box.append(el("span", {}, el("i", { style: `background:rgb(${r},${g},${b});box-shadow:0 0 7px rgba(${r},${g},${b},.9)` }), T.categories[c]));
      });
    } else if (state.colorBy === "year") {
      box.append(el("b", { text: "Year" }), el("span", { class: "ramp" }), el("span", { text: `${yearMin} – ${yearMax}` }));
    } else {
      box.append(el("b", { text: "Emergent modes" }), el("span", { text: `${cur().modes.length} modes; grey = unassigned` }));
    }
  }

  // ---- poles ---------------------------------------------------------------------------------------------------------
  function renderPoleSelect() {
    const m = cur();
    ui.poleSelect.replaceChildren();
    const groups = { supervised: "Supervised", mode: "Emergent: modes", factor: "Emergent: factors" };
    for (const [g, title] of Object.entries(groups)) {
      const og = el("optgroup", { label: title });
      m.poleMeta.filter((p) => p.group === g).forEach((p) => {
        og.append(el("option", { value: p.name, text: p.label ? `${p.name} – ${p.label}` : p.name }));
      });
      if (og.children && og.children.length) ui.poleSelect.append(og);
    }
  }

  function renderChips() {
    ui.chips.replaceChildren();
    state.poles.forEach((p, i) => {
      ui.chips.append(el("li", {},
        el("span", { text: p.name, style: "font-weight: 500;" }),
        el("div", { style: "display: flex; gap: 4px; align-items: center;" },
          el("input", { type: "number", min: "0", step: "0.1", value: String(p.weight), "aria-label": `weight of ${p.name}`,
            on: { change: (e) => { p.weight = Math.max(0, parseFloat(e.target.value) || 0); runQuery(); } } }),
          el("button", { type: "button", "aria-label": `remove ${p.name}`, text: "×", on: { click: () => { state.poles.splice(i, 1); renderChips(); runQuery(); } } }))));
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
      const labelText = mode.label || `Mode #${mode.id} (${mode.driver || 'audio'})`;
      ui.modeList.append(el("li", {}, el("button", { type: "button", text: `${labelText} (${mode.n_in_payload}/${mode.size})`, on: { click: () => showMode(i) } })));
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
    const labelText = mode.label || `Mode #${mode.id}`;
    ui.modeDetail.replaceChildren(
      el("strong", { text: labelText }),
      el("div", { class: "muted", style: "font-size:11px; margin:4px 0;", text: `driver: ${mode.driver || "-"} · coherence ${mode.coherence != null ? Number(mode.coherence).toFixed(2) : "-"} · ${mode.n_in_payload} of ${mode.size} members` }),
      el("div", { class: "row" }, el("button", { type: "button", class: "btn-primary", text: "Steer toward mode", on: { click: () => addPole(`mode:${mode.id}`) } }),
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
    for (const info of state.manifest.models) {
      const friendlyName = MODEL_LABELS[info.name] || info.label;
      ui.toggle.append(el("button", { type: "button", role: "radio", "aria-checked": String(info.name === state.modelName), text: friendlyName,
        on: { click: () => switchModel(info.name) } }));
    }
    if (ui.modelDesc) ui.modelDesc.textContent = MODEL_DESCRIPTIONS[state.modelName] || "";
  }

  async function switchModel(name) {
    if (name === state.modelName) return;
    try { await loadModel(name); } catch (e) { fail("Could not load model " + name + ": " + e.message); return; }
    state.modelName = name;
    state.poles = state.poles.filter((p) => cur().poleByName.has(p.name));
    state.modeFocus = null; state.modeFocusId = null; if (ui.modeDetail) ui.modeDetail.replaceChildren();
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
    if (p.get("s")) { const c = parseInt(p.get("s"), 10); if (!isNaN(c)) o.seedCid = c; }
    if (p.get("t")) { const th = parseInt(p.get("t"), 10); if (!isNaN(th)) o.theta = clamp(th, 0, 90); }
    if (p.get("p")) {
      o.poles = p.get("p").split(",").map((s) => {
        const [name, w] = s.split("*"); return { name, weight: Math.max(0, parseFloat(w) || 1) };
      }).filter((x) => x.name);
    }
    return o;
  }

  // ---- events --------------------------------------------------------------------------------------------------------
  ui.theta.addEventListener("input", () => {
    state.theta = parseInt(ui.theta.value, 10);
    ui.thetaOut.textContent = String(state.theta);
    updateThetaDesc();
    runQuery();
  });
  ui.kSel.addEventListener("change", () => { state.k = parseInt(ui.kSel.value, 10); runQuery(); });
  ui.excl.addEventListener("change", () => { state.excludeArtist = !!ui.excl.checked; runQuery(); });
  ui.colorBy.addEventListener("change", () => { state.colorBy = ui.colorBy.value; if (cur()) recolour(); });
  ui.poleAdd.addEventListener("click", () => addPole(ui.poleSelect.value));
  ui.tabRes.addEventListener("click", () => selectTab("results"));
  ui.tabModes.addEventListener("click", () => selectTab("modes"));
  ui.modeFilter.addEventListener("input", renderModeList);

  if (ui.showArtists) {
    ui.showArtists.addEventListener("change", () => requestDraw());
  }

  // Modal interactions
  function dismissWelcome() {
    if (ui.welcomeModal) ui.welcomeModal.hidden = true;
    if (ui.dontShowWelcome && ui.dontShowWelcome.checked) {
      try { win.localStorage?.setItem("melomane_seen_welcome_v2", "1"); } catch {}
    }
  }

  if (ui.closeWelcome) ui.closeWelcome.addEventListener("click", dismissWelcome);
  if (ui.btnExploreNow) ui.btnExploreNow.addEventListener("click", dismissWelcome);
  if (ui.btnOpenGuide) {
    ui.btnOpenGuide.addEventListener("click", () => {
      if (ui.welcomeModal) ui.welcomeModal.hidden = false;
    });
  }

  if (ui.btnModelsInfo) {
    ui.btnModelsInfo.addEventListener("click", () => {
      if (ui.modelsModal) ui.modelsModal.hidden = false;
    });
  }
  if (ui.closeModelsModal) ui.closeModelsModal.addEventListener("click", () => { if (ui.modelsModal) ui.modelsModal.hidden = true; });
  if (ui.closeModelsModalBtn) ui.closeModelsModalBtn.addEventListener("click", () => { if (ui.modelsModal) ui.modelsModal.hidden = true; });

  // Guided walkthrough: every sentence is computed from the loaded payload (nothing typed in)
  let tourStep = 1;

  /** A recognisable seed that is actually in this payload (falls back to the most-listened track). */
  function tourSeed() {
    for (const want of ["taylor swift", "coldplay", "radiohead", "madonna"])
      for (let i = 0; i < N; i++) if ((T.artist[T.artist_idx[i]] || "").toLowerCase() === want) return i;
    return 0;
  }
  const artistOf = (i) => T.artist[T.artist_idx[i]] || "Unknown artist";
  const names = (idx, n = 3) => Array.from(idx).slice(0, n).map(artistOf).join(", ");

  /** Top-k of one model without touching the UI (used to compare models / steered vs unsteered live in the tour text). */
  async function topFor(modelName, seed, poleNames = [], theta = 0, k = 10) {
    const mdl = await loadModel(modelName);
    const vecs = [], w = [];
    for (const p of poleNames) { const i = mdl.poleByName.get(p); if (i !== undefined) { vecs.push(mdl.poles[i]); w.push(1); } }
    return M.query({ U: mdl.U, n: N, d: mdl.d, seed, poles: vecs, weights: w, thetaDeg: theta, k, artistIdx, excludeArtist: true }).idx;
  }
  const shared = (a, b) => { const s = new Set(Array.from(a)); return Array.from(b).filter((x) => s.has(x)).length; };

  function startTsTour() {
    dismissWelcome();
    tourStep = 1;
    if (ui.tourBanner) {
      ui.tourBanner.hidden = false;
      ui.tourBanner.style.display = "flex";
    }
    applyTourStep(1);
  }

  function updateBadgeClass(cls) {
    if (ui.tourStepBadge) ui.tourStepBadge.className = "tour-step-badge " + cls;
  }

  async function applyTourStep(step) {
    tourStep = Math.max(1, Math.min(4, step));
    if (ui.tourPrevBtn) ui.tourPrevBtn.style.display = tourStep > 1 ? "inline-block" : "none";
    const seed = tourSeed(), k = state.k;
    const setText = (badge, text, action) => {
      if (ui.tourStepBadge) ui.tourStepBadge.textContent = badge;
      if (ui.tourStepText) ui.tourStepText.textContent = text;
      if (ui.tourActionBtn) ui.tourActionBtn.textContent = action;
      if (ui.tourNextBtn) ui.tourNextBtn.textContent = tourStep < 4 ? "Next →" : "Finish ✓";
    };
    const hasAmbient = (mdl) => mdl.poleByName.has("category:Ambient/Chill");

    if (tourStep === 1) {
      state.poles = []; renderChips();
      await switchModel("cooc");
      setSeed(seed, { center: true });
      setText("Step 1 of 4", `${label(seed)} in Cooc, which learns only from what people listen to together. Nearest neighbours: ${names(state.results.idx)}.`, "Compare with Composition →");
    } else if (tourStep === 2) {
      state.poles = []; renderChips();
      const cooc = await topFor("cooc", seed, [], 0, k);
      if (state.modelName !== "composition") await switchModel("composition");
      setSeed(seed, { center: true });
      setText("Step 2 of 4", `The same track in Composition, which learns only from measured audio and lyric features: ${names(state.results.idx)}. ${shared(cooc, state.results.idx)} of ${k} neighbours also appear in the Cooc list.`, "Steer it →");
    } else if (tourStep === 3) {
      if (state.modelName !== "composition") await switchModel("composition");
      state.poles = [];
      const pole = hasAmbient(cur()) ? "category:Ambient/Chill" : (cur().poleMeta.find((p) => p.group === "supervised") || {}).name;
      state.theta = 30; ui.theta.value = "30"; ui.thetaOut.textContent = "30"; updateThetaDesc();
      const plain = await topFor("composition", seed, [], 0, k);
      if (pole) addPole(pole);
      setSeed(seed, { center: true });
      const moved = pole ? k - shared(plain, state.results.idx) : 0;
      setText("Step 3 of 4", pole ? `Rotated 30° toward ${pole}: ${names(state.results.idx)}. ${moved} of ${k} neighbours changed; the rest stay close to the seed (cos 30° = ${Math.cos(Math.PI / 6).toFixed(2)}).` : "No steering pole is available in this model.", "Finish →");
    } else {
      setText("Step 4 of 4", "Search any artist above, click a point or a label to re-centre, and drag the angle slider to steer.", "Finish ✓");
    }
  }

  function handleTourAction() {
    if (tourStep === 1) {
      applyTourStep(2);
    } else if (tourStep === 2) {
      applyTourStep(3);
    } else if (tourStep === 3) {
      applyTourStep(4);
    } else {
      endTour();
    }
  }

  function endTour() {
    if (ui.tourBanner) {
      ui.tourBanner.hidden = true;
      ui.tourBanner.style.display = "none";
    }
    tourStep = 1;
  }

  if (ui.btnStartTsDemo) ui.btnStartTsDemo.addEventListener("click", startTsTour);
  if (ui.btnDemoTsCp) ui.btnDemoTsCp.addEventListener("click", startTsTour);
  if (ui.tourActionBtn) ui.tourActionBtn.addEventListener("click", handleTourAction);
  if (ui.tourNextBtn) ui.tourNextBtn.addEventListener("click", () => {
    if (tourStep < 4) applyTourStep(tourStep + 1);
    else endTour();
  });
  if (ui.tourPrevBtn) ui.tourPrevBtn.addEventListener("click", () => {
    if (tourStep > 1) applyTourStep(tourStep - 1);
  });
  if (ui.tourExitBtn) ui.tourExitBtn.addEventListener("click", endTour);

  // Quick Artist Landmark Buttons
  if (doc.querySelectorAll) {
    doc.querySelectorAll(".chip-artist").forEach((btn) => {
      btn.addEventListener("click", () => {
        const artName = btn.dataset.artist;
        if (!artName) return;
        let matchedTrack = -1;
        for (let i = 0; i < N; i++) {
          const a = T.artist[T.artist_idx[i]];
          if (a && a.toLowerCase().includes(artName.toLowerCase())) {
            matchedTrack = i;
            break;
          }
        }
        if (matchedTrack >= 0) {
          setSeed(matchedTrack, { center: true });
        }
      });
    });

    doc.querySelectorAll(".chip-preset").forEach((btn) => {
      btn.addEventListener("click", () => {
        const seed = parseInt(btn.dataset.seed, 10);
        const pole = btn.dataset.pole;
        const theta = parseInt(btn.dataset.theta, 10);
        if (seed >= 0 && seed < N) {
          state.poles = [];
          addPole(pole);
          state.theta = theta;
          ui.theta.value = String(theta);
          ui.thetaOut.textContent = String(theta);
          updateThetaDesc();
          setSeed(seed, { center: true });
        }
      });
    });
  }

  // Keyboard Shortcuts (Spotlight / Esc)
  win.addEventListener("keydown", (e) => {
    if ((e.key === "/" || ((e.metaKey || e.ctrlKey) && e.key === "k")) && doc.activeElement !== ui.seedSearch) {
      e.preventDefault();
      ui.seedSearch.focus();
      ui.seedSearch.select();
    } else if (e.key === "Escape") {
      if (ui.welcomeModal && !ui.welcomeModal.hidden) { ui.welcomeModal.hidden = true; }
      if (ui.modelsModal && !ui.modelsModal.hidden) { ui.modelsModal.hidden = true; }
      if (ui.suggest) ui.suggest.replaceChildren();
      ui.seedSearch.blur();
    }
  });

  if (doc.addEventListener) {
    doc.addEventListener("pointerdown", (e) => {
      if (e.target && e.target.closest && !e.target.closest(".search-wrap") && ui.suggest) {
        ui.suggest.replaceChildren();
      }
    });
  }

  // Spotlight Autocomplete
  ui.seedSearch.addEventListener("input", () => {
    ui.suggest.replaceChildren();
    const query = ui.seedSearch.value.trim();
    if (!query) return;

    const normQ = M.normText(query);
    const matchedArtists = [];
    for (let a = 0; a < T.artist.length && matchedArtists.length < 3; a++) {
      const aName = T.artist[a];
      if (aName && M.normText(aName).includes(normQ)) {
        matchedArtists.push({ id: a, name: aName });
      }
    }

    const maxSuggestions = 8;
    let count = 0;
    matchedArtists.slice(0, 3).forEach((a) => {
      const firstTrack = T.artist_idx.indexOf(a.id);
      if (firstTrack >= 0 && count < maxSuggestions) {
        count++;
        ui.suggest.append(el("li", { role: "option" },
          el("button", {
            type: "button",
            style: "font-weight: 600; color: #38bdf8;",
            text: `Artist · ${a.name} (Zoom In & Show Nearest)`,
            on: { click: () => {
              ui.suggest.replaceChildren();
              ui.seedSearch.value = "";
              setSeed(firstTrack, { center: true });
            }}
          })
        ));
      }
    });

    for (const i of M.searchTracks(state.hay, query, maxSuggestions - count)) {
      if (count >= maxSuggestions) break;
      count++;
      ui.suggest.append(el("li", { role: "option" },
        el("button", { type: "button", text: label(i), on: { click: () => {
          ui.suggest.replaceChildren();
          ui.seedSearch.value = "";
          setSeed(i, { center: true });
        }}})
      ));
    }
  });

  ui.seedSearch.addEventListener("input", () => {            // runs after the suggestion builder above
    if (ui.seedSearch.value.trim() && !ui.suggest.children.length)
      ui.suggest.append(el("li", { class: "empty", role: "option" }, `No match in these ${N.toLocaleString("en-GB")} tracks`));
  });

  /** Frame the dense core of the map (2nd-98th percentile of each axis) so the cloud fills the stage instead of floating in it. */
  function fitView() {
    const m = cur();
    if (!m) return { cx: 0, cy: 0, zoom: 0.95 };
    const xs = new Float32Array(N), ys = new Float32Array(N);
    for (let i = 0; i < N; i++) { xs[i] = m.xy[2 * i]; ys[i] = m.xy[2 * i + 1]; }
    xs.sort(); ys.sort();
    const q = (a, p) => a[Math.min(N - 1, Math.max(0, Math.floor(p * (N - 1))))];
    const x0 = q(xs, 0.02), x1 = q(xs, 0.98), y0 = q(ys, 0.02), y1 = q(ys, 0.98);
    const hx = Math.max(1e-3, (x1 - x0) / 2), hy = Math.max(1e-3, (y1 - y0) / 2), mm = Math.min(cssW, cssH);
    const zoom = clamp(0.92 * Math.min(cssW / (hx * mm), cssH / (hy * mm)), 0.6, 6);
    return { cx: (x0 + x1) / 2, cy: (y0 + y1) / 2, zoom };
  }

  const zoomBy = (f) => { state.view = M.zoomAt(state.view, cssW, cssH, cssW / 2, cssH / 2, f); requestDraw(); };
  ui.zin.addEventListener("click", () => zoomBy(1.6));
  ui.zout.addEventListener("click", () => zoomBy(1 / 1.6));
  ui.zreset.addEventListener("click", () => { state.view = fitView(); requestDraw(); });

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
    if (drag && !drag.moved && cur()) {
      const m = cur();
      const [x, y] = local(e);

      // a click on a landmark pill seeds that artist's most-listened track
      const box = (state.labelBoxes || []).find((bx) => x >= bx.x0 && x <= bx.x0 + bx.w && y >= bx.y0 && y <= bx.y0 + bx.h);
      if (box) { setSeed(box.track, { center: true }); drag = null; return; }

      const i = pick(x, y);
      if (i >= 0) setSeed(i, { center: true });
    }
    drag = null;
  };

  ui.canvas.addEventListener("pointerup", endDrag);
  ui.canvas.addEventListener("pointercancel", () => { drag = null; ui.canvas.classList.remove("dragging"); });
  ui.canvas.addEventListener("pointerleave", () => { state.hover = -1; ui.tooltip.hidden = true; requestDraw(); });
  ui.canvas.addEventListener("wheel", (e) => {
    if (e.preventDefault) e.preventDefault();
    const [x, y] = local(e); state.view = M.zoomAt(state.view, cssW, cssH, x, y, Math.exp(-e.deltaY * 0.0015)); requestDraw();
  }, { passive: false });
  ui.canvas.addEventListener("dblclick", () => { state.view = fitView(); requestDraw(); });
  if (win.addEventListener) win.addEventListener("resize", resize);
  if (win.ResizeObserver) new win.ResizeObserver(resize).observe(ui.stage);

  // ------------------------------------------------------------------------------------------------------------------
  // go
  // ------------------------------------------------------------------------------------------------------------------
  const h = readHash();
  const first = state.manifest.models.some((m) => m.name === h.model) ? h.model : state.manifest.models[0].name;
  if (h.theta !== undefined) { state.theta = h.theta; ui.theta.value = String(h.theta); ui.thetaOut.textContent = String(h.theta); }
  updateThetaDesc();
  resize();
  await switchModel(first);
  state.view = fitView(); requestDraw();
  hideSplash();
  if (h.poles) { for (const p of h.poles) if (cur().poleByName.has(p.name)) state.poles.push(p); renderChips(); }
  if (h.seedCid !== undefined) { const i = T.cid.indexOf(h.seedCid); if (i >= 0) setSeed(i, { center: true }); }
  renderLegend();

  // First-visit welcome modal check
  try {
    const seen = win.localStorage?.getItem("melomane_seen_welcome_v2");
    if (!seen && ui.welcomeModal) {
      ui.welcomeModal.hidden = false;
    }
  } catch {}

  return { state, setSeed, addPole, switchModel, runQuery, showMode, draw, resize };
}

if (typeof document !== "undefined" && !globalThis.__MELOMANE_NO_AUTOSTART__) {
  start().then((app) => { globalThis.__melomane = app; }).catch((e) => {
    const box = document.getElementById("error");
    if (box) { box.hidden = false; box.textContent = "Explorer failed to start: " + e.message; }
    console.error(e);
  });
}
