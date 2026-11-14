/* Frozen Lake Lab: dashboard client.
   Talks to the FastAPI backend: one Server-Sent-Events stream for live training
   snapshots, plain POSTs for commands. No framework, no runtime dependencies. */
"use strict";

const $ = (sel, root = document) => root.querySelector(sel);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const REDUCED = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
const nf = new Intl.NumberFormat("en-US");
const pct = (v, d = 0) => (v * 100).toFixed(d) + "%";
const compact = (n) => (n >= 1e6 ? +(n / 1e6).toFixed(1) + "M" : n >= 1e3 ? +(n / 1e3).toFixed(n >= 1e4 ? 0 : 1) + "k" : String(Math.round(n)));

const ACTIONS = ["Left", "Down", "Right", "Up"];
const ARROWS = ["←", "↓", "→", "↑"];
const ICON = { S: "🚩", F: "❄️", H: "🕳️", G: "🏆" };
const TILE_NAME = { S: "start", F: "frozen ice", H: "hole", G: "goal" };
const AGENT = "🐧";

/* ---------------------------------------------------------------- state */
const S = {
  snap: null, layoutKey: "", tiles: [], agentEl: null, agentState: 0,
  selected: 0, play: null, busy: false, replaying: false,
  auto: false, autoToken: 0, initialised: false,
};
let P = {
  map_name: "4x4", slippery: true, alpha: 0.1, gamma: 0.99, epsilon_start: 1.0,
  epsilon_min: 0.01, epsilon_decay: 0.9995, episodes: 15000, seed: 42, pace: 3000,
};

/* ------------------------------------------------------------------ api */
async function api(path, body) {
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body ?? {}),
  });
  if (!res.ok) {
    let msg = res.statusText;
    try {
      const j = await res.json();
      msg = typeof j.detail === "string" ? j.detail
        : Array.isArray(j.detail) ? j.detail.map((d) => `${d.loc?.slice(-1)}: ${d.msg}`).join("; ") : msg;
    } catch (_) { /* keep statusText */ }
    const err = new Error(msg);
    err.status = res.status;
    throw err;
  }
  return res.json();
}

let toastTimer;
function toast(msg) {
  const el = $("#toast");
  el.textContent = msg;
  el.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (el.hidden = true), 4500);
}
const guard = (fn) => async (...a) => { try { return await fn(...a); } catch (e) { toast(e.message || "Something went wrong."); } };

/* ---------------------------------------------------------------- charts */
class LineChart {
  /* o: {height, yDomain:[a,b]|null, fmtY, area:boolean, label}  */
  constructor(el, o) {
    this.el = el; this.o = o; this.data = null; this.table = false; this.hover = null;
    el.innerHTML = '<div class="plot"></div><div class="tip" hidden></div>';
    this.plot = $(".plot", el); this.tip = $(".tip", el);
    new ResizeObserver(() => this.render()).observe(el);
  }
  set(data) { this.data = data; this.render(); }
  toggleTable() { this.table = !this.table; this.render(); }

  domainY() {
    if (this.o.yDomain) return this.o.yDomain;
    let max = 0;
    for (const s of this.data.series) for (const v of s.y) max = Math.max(max, v);
    if (max <= 0) return [0, 1];
    const pow = Math.pow(10, Math.floor(Math.log10(max)));
    const nice = [1, 1.2, 1.6, 2, 2.4, 3, 4, 6, 8, 10].find((m) => m * pow >= max * 1.05) * pow;
    return [0, nice];
  }

  render() {
    const d = this.data;
    if (!d) { this.plot.innerHTML = ""; return; }
    if (this.table) return this.renderTable();
    const W = Math.max(this.el.clientWidth, 200), H = this.o.height || 210;
    const m = { l: 40, r: 46, t: 12, b: 26 };
    const iw = W - m.l - m.r, ih = H - m.t - m.b;
    const [y0, y1] = this.domainY();
    const xMax = Math.max(d.xMax, 1);
    const X = (x) => m.l + (x / xMax) * iw;
    const Y = (y) => m.t + ih - ((y - y0) / (y1 - y0 || 1)) * ih;
    const fy = this.o.fmtY;

    let out = `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" role="img" aria-label="${this.o.label}">`;
    for (let i = 0; i <= 4; i++) {
      const v = y0 + ((y1 - y0) * i) / 4, y = Y(v);
      out += `<line class="gl" x1="${m.l}" x2="${W - m.r}" y1="${y}" y2="${y}"/><text class="ax" x="${m.l - 8}" y="${y + 4}" text-anchor="end">${fy(v)}</text>`;
    }
    for (let i = 0; i <= 4; i++) {
      const v = (xMax * i) / 4;
      out += `<text class="ax" x="${X(v)}" y="${H - 6}" text-anchor="${i === 0 ? "start" : i === 4 ? "end" : "middle"}">${compact(v)}</text>`;
    }

    d.series.forEach((s, si) => {
      if (!s.x.length) return;
      const pts = s.x.map((x, i) => `${X(x).toFixed(1)},${Y(s.y[i]).toFixed(1)}`);
      if (this.o.area && si === 0) {
        out += `<path d="M${X(s.x[0]).toFixed(1)},${Y(y0)} L${pts.join(" L")} L${X(s.x[s.x.length - 1]).toFixed(1)},${Y(y0)} Z" style="fill:${s.color};opacity:.10"/>`;
      }
      out += `<path class="ln" d="M${pts.join(" L")}" style="stroke:${s.color}"/>`;
    });

    // direct end labels, nudged apart so they never collide
    const ends = d.series.filter((s) => s.x.length).map((s) => {
      const i = s.x.length - 1;
      const y = Y(s.y[i]);
      return { s, x: X(s.x[i]), y, labelY: y, v: s.y[i] };
    });
    ends.sort((a, b) => a.y - b.y);
    for (let i = 1; i < ends.length; i++) if (ends[i].labelY - ends[i - 1].labelY < 16) ends[i].labelY = ends[i - 1].labelY + 16;
    for (const e of ends) {
      out += `<circle cx="${e.x}" cy="${e.y}" r="4.5" style="fill:${e.s.color};stroke:var(--surface);stroke-width:2"/>`;
      out += `<text class="ax" style="fill:var(--ink);font-weight:700;font-size:12px" x="${e.x + 10}" y="${e.labelY + 4}">${fy(e.v)}</text>`;
    }

    // hover layer
    let tipHtml = "", tipX = 0, tipY = 0;
    const base = d.series.find((s) => s.x.length);
    if (this.hover !== null && base && this.hover < base.x.length) {
      const i = this.hover, hx = X(base.x[i]);
      out += `<line x1="${hx}" x2="${hx}" y1="${m.t}" y2="${m.t + ih}" style="stroke:var(--ink-2);stroke-width:1;stroke-dasharray:3 3"/>`;
      tipHtml = `<div style="font-weight:700;margin-bottom:.15rem">Episode ${nf.format(base.x[i])}</div>`;
      d.series.forEach((s) => {
        if (s.y[i] === undefined) return;
        out += `<circle cx="${hx}" cy="${Y(s.y[i])}" r="4.5" style="fill:${s.color};stroke:var(--surface);stroke-width:2"/>`;
        tipHtml += `<div><i style="background:${s.color}"></i>${s.label}: <b>${(this.o.fmtTip || fy)(s.y[i])}</b></div>`;
      });
      tipX = hx; tipY = Y(base.y[i]) - 12;
    }
    out += `<rect x="${m.l}" y="${m.t}" width="${iw}" height="${ih}" fill="transparent" class="hit"/></svg>`;
    this.plot.innerHTML = out;

    const hit = $(".hit", this.plot);
    hit.addEventListener("pointermove", (ev) => {
      const r = hit.getBoundingClientRect();
      const x = ((ev.clientX - r.left) / r.width) * xMax;
      let best = 0, bd = Infinity;
      base.x.forEach((v, i) => { const dd = Math.abs(v - x); if (dd < bd) { bd = dd; best = i; } });
      if (best !== this.hover) { this.hover = best; this.render(); }
    });
    hit.addEventListener("pointerleave", () => { this.hover = null; this.render(); });

    if (tipHtml) {
      this.tip.hidden = false; this.tip.innerHTML = tipHtml;
      const half = this.tip.offsetWidth / 2;
      this.tip.style.left = Math.min(Math.max(tipX, half + 4), W - half - 4) + "px";
      this.tip.style.top = Math.max(tipY, this.tip.offsetHeight + 4) + "px";
    } else this.tip.hidden = true;
  }

  renderTable() {
    const d = this.data, base = d.series.find((s) => s.x.length);
    this.tip.hidden = true;
    if (!base) { this.plot.innerHTML = '<p class="text-sm text-muted py-6">No data yet. Train the agent to fill this in.</p>'; return; }
    const rows = [], n = base.x.length, step = Math.max(1, Math.floor(n / 9));
    for (let i = n - 1; i >= 0 && rows.length < 10; i -= step) rows.push(i);
    let h = `<table class="tbl"><thead><tr><th>Episode</th>${d.series.map((s) => `<th>${s.label}</th>`).join("")}</tr></thead><tbody>`;
    for (const i of rows.reverse()) h += `<tr><td>${nf.format(base.x[i])}</td>${d.series.map((s) => `<td>${s.y[i] === undefined ? "" : (this.o.fmtTip || this.o.fmtY)(s.y[i])}</td>`).join("")}</tr>`;
    this.plot.innerHTML = h + "</tbody></table>";
  }
}

const charts = {
  success: new LineChart($("#chart-success"), { yDomain: [0, 1], fmtY: (v) => pct(v), area: true, label: "Moving average success rate over training episodes" }),
  q: new LineChart($("#chart-q"), { yDomain: null, fmtY: (v) => (v === 0 ? "0" : v >= 10 ? v.toFixed(0) : v >= 0.1 ? v.toFixed(2) : v.toPrecision(1)), fmtTip: (v) => v.toFixed(3), area: true, label: "Average best Q-value over training episodes" }),
  eps: new LineChart($("#chart-eps"), { yDomain: [0, 1], fmtY: (v) => v.toFixed(2), fmtTip: (v) => v.toFixed(3), area: true, label: "Exploration rate epsilon over training episodes" }),
};
document.querySelectorAll(".tbl-toggle").forEach((b) =>
  b.addEventListener("click", () => {
    const on = b.getAttribute("aria-pressed") !== "true";
    b.setAttribute("aria-pressed", on);
    b.textContent = on ? "Chart" : "Table";
    charts[b.dataset.chart].toggleTable();
  }));
$("#legend-success").innerHTML =
  '<span class="inline-flex items-center gap-1.5"><span class="inline-block w-3 h-0.5 rounded" style="background:var(--series-1);height:3px"></span>Last 100 episodes</span>' +
  '<span class="inline-flex items-center gap-1.5"><span class="inline-block w-3 rounded" style="background:var(--series-2);height:3px"></span>Last 500 episodes</span>';

function updateCharts(s) {
  const se = s.series, xMax = s.total_episodes || 1;
  charts.success.set({ xMax, series: [
    { label: "Last 100", color: "var(--series-1)", x: se.episodes, y: se.success_100 },
    { label: "Last 500", color: "var(--series-2)", x: se.episodes, y: se.success_500 },
  ] });
  charts.q.set({ xMax, series: [{ label: "Mean best Q", color: "var(--series-1)", x: s.qconv.map((p) => p[0]), y: s.qconv.map((p) => p[1]) }] });
  charts.eps.set({ xMax, series: [{ label: "Epsilon", color: "var(--series-1)", x: se.episodes, y: se.epsilon }] });
}

/* ----------------------------------------------------------------- board */
function ensureBoard(layout) {
  const key = layout.join("|");
  if (key === S.layoutKey) return;
  S.layoutKey = key;
  const n = layout.length, board = $("#board");
  board.dataset.n = n;
  board.style.setProperty("--n", n);
  board.innerHTML = "";
  S.tiles = [];
  S.selected = 0;
  layout.forEach((row, r) => [...row].forEach((t, c) => {
    const el = document.createElement("button");
    el.type = "button";
    el.className = "tile tile-" + t;
    el.setAttribute("aria-label", `State ${r * n + c}, row ${r + 1}, column ${c + 1}, ${TILE_NAME[t]}`);
    el.setAttribute("aria-selected", "false");
    el.innerHTML = `<span class="heat"></span><span class="trail"></span><span class="icon">${ICON[t]}</span><span class="arrow"></span><span class="qv"></span>`;
    const s = r * n + c;
    el.addEventListener("click", () => select(s));
    board.appendChild(el);
    S.tiles.push(el);
  }));
  const agent = document.createElement("div");
  agent.className = "agent no-anim";
  agent.innerHTML = `<span>${AGENT}</span>`;
  board.appendChild(agent);
  S.agentEl = agent;
  select(0);
  placeAgent(0, false);
}

function placeAgent(state, animate = true) {
  const t = S.tiles[state];
  if (!t || !S.agentEl) return;
  S.agentState = state;
  const a = S.agentEl;
  if (!animate) a.classList.add("no-anim");
  a.style.width = t.offsetWidth + "px";
  a.style.height = t.offsetHeight + "px";
  a.style.transform = `translate(${t.offsetLeft}px, ${t.offsetTop}px)`;
  if (!animate) { void a.offsetWidth; a.classList.remove("no-anim"); }
}
new ResizeObserver(() => placeAgent(S.agentState, false)).observe($("#board"));

function select(s) {
  S.selected = s;
  S.tiles.forEach((t, i) => t.setAttribute("aria-selected", i === s));
  if (S.snap) updateInspector(S.snap);
}

function updateBoard(s) {
  const q = s.q_table, maxes = q.map((r) => Math.max(...r));
  const gmax = Math.max(1e-9, ...maxes), vmax = Math.max(1, ...s.visits);
  S.tiles.forEach((el, i) => {
    const kind = s.layout[Math.floor(i / s.layout.length)][i % s.layout.length];
    const walkable = kind === "F" || kind === "S";
    el.style.setProperty("--v", walkable ? (Math.max(maxes[i], 0) / gmax).toFixed(3) : "0");
    el.style.setProperty("--visit", Math.pow(s.visits[i] / vmax, 0.8).toFixed(3));
    $(".arrow", el).textContent = walkable && s.policy[i] >= 0 ? ARROWS[s.policy[i]] : "";
    $(".qv", el).textContent = walkable && maxes[i] > 0 ? maxes[i].toFixed(2) : "";
  });
}

function updateInspector(s) {
  const n = s.layout.length, i = S.selected, r = Math.floor(i / n), c = i % n;
  const kind = s.layout[r][c];
  $("#insp-title").textContent = `State ${i} (row ${r + 1}, column ${c + 1}), ${TILE_NAME[kind]}`;
  const row = s.q_table[i], max = Math.max(...row), scale = Math.max(...row.map(Math.abs)) || 1;
  const best = row.some((v) => v !== 0) ? row.indexOf(max) : -1;
  let html = "";
  if (kind === "H" || kind === "G") {
    html = '<p class="text-sm text-muted">The episode ends on this tile, so the agent never chooses an action here and nothing is learned from it.</p>';
  } else {
    row.forEach((v, a) => {
      html += `<div><div class="flex justify-between text-sm"><span class="${a === best ? "font-bold text-ink" : "text-ink-2"}">${ARROWS[a]} ${ACTIONS[a]}</span><span class="tabular-nums ${a === best ? "font-bold text-ink" : "text-ink-2"}">${v.toFixed(4)}</span></div><div class="qbar mt-1 ${a === best ? "best" : ""}"><i style="width:${Math.max(0, v / scale) * 100}%"></i></div></div>`;
    });
    if (best < 0) html += '<p class="text-xs text-muted">All zero: this tile has not taught the agent anything yet.</p>';
  }
  $("#insp-bars").innerHTML = html;
}

/* ------------------------------------------------------------- snapshot */
const STATUS_TEXT = { idle: "Idle", training: "Training", done: "Trained", stopped: "Stopped", error: "Error" };

function onSnap(s) {
  const prev = S.snap;
  S.snap = s;
  ensureBoard(s.layout);
  if (!S.initialised) { S.initialised = true; P = { ...P, ...s.params }; syncControls(); }

  const pill = $("#status-pill");
  pill.dataset.s = s.status;
  $("#status-text").textContent = STATUS_TEXT[s.status] || s.status;
  const training = s.status === "training";
  $("#btn-train").disabled = training;
  $("#btn-stop").disabled = !training;
  for (const id of ["#btn-step", "#btn-auto", "#btn-new"]) $(id).disabled = training;

  $("#k-episode").textContent = nf.format(s.episode);
  $("#k-eps").textContent = s.epsilon.toFixed(3);
  $("#k-success").textContent = pct(s.success_100);
  $("#k-eval").textContent = s.evaluation ? pct(s.evaluation.success_rate, 1) : "—";
  $("#k-speed").textContent = training && s.eps_per_sec ? compact(s.eps_per_sec) + "/s" : "—";
  const prog = s.total_episodes ? (s.episode / s.total_episodes) * 100 : 0;
  $("#progress").style.width = prog.toFixed(1) + "%";
  $("#progress-wrap").setAttribute("aria-valuenow", Math.round(prog));

  updateBoard(s);
  updateInspector(s);
  updateCharts(s);

  const was = prev ? prev.status : "idle";
  if (training && was !== "training") {
    stopAuto(); clearFx();
    narrate("Training: the penguin replays its latest attempt as it learns.");
  }
  if (training) replay(s.last_path);
  if (was === "training" && !training) {
    S.play = null;
    guard(playReset)();
    if (s.status === "done" && s.evaluation) {
      narrate(`Trained for ${nf.format(s.episode)} episodes. In a greedy test, ${pct(s.evaluation.success_rate, 1)} of 1,000 runs reached the goal in ${s.evaluation.average_episode_length.toFixed(1)} steps on average.`);
    } else if (s.status === "stopped") narrate(`Stopped after ${nf.format(s.episode)} episodes.`);
    else if (s.status === "error") toast(s.error || "Training failed.");
  }
  if (prev && prev.run_id !== s.run_id && !training) { S.play = null; guard(playReset)(); narrate("Fresh agent. Train it, or step through to see it flounder."); }
}

/* ------------------------------------------------------------ play / fx */
function narrate(t) { $("#narration").textContent = t; }

function clearFx() {
  const a = S.agentEl;
  if (a) a.classList.remove("slip", "sink", "win");
  document.querySelectorAll(".bubble, .confetti").forEach((e) => e.remove());
}

async function playReset() {
  const v = await api("/api/play/reset");
  S.play = v;
  clearFx();
  placeAgent(v.state, false);
}

function bubble(text, state) {
  const t = S.tiles[state], b = document.createElement("div");
  b.className = "bubble"; b.textContent = text;
  b.style.left = t.offsetLeft + t.offsetWidth / 2 + "px";
  b.style.top = t.offsetTop + "px";
  $("#board").appendChild(b);
  setTimeout(() => b.remove(), 1000);
}

function confetti(state) {
  if (REDUCED) return;
  const t = S.tiles[state], board = $("#board");
  const colors = ["#f6c65b", "#5b9bff", "#ff7a45", "#4cc78d", "#ffffff"];
  for (let i = 0; i < 34; i++) {
    const p = document.createElement("i");
    p.className = "confetti";
    p.style.left = t.offsetLeft + t.offsetWidth / 2 + "px";
    p.style.top = t.offsetTop + t.offsetHeight / 2 + "px";
    p.style.background = colors[i % colors.length];
    const ang = Math.random() * Math.PI * 2, dist = 70 + Math.random() * 150;
    p.style.setProperty("--dx", Math.cos(ang) * dist + "px");
    p.style.setProperty("--dy", Math.sin(ang) * dist - 50 + "px");
    p.style.setProperty("--rot", (Math.random() * 720 - 360) + "deg");
    board.appendChild(p);
    setTimeout(() => p.remove(), 1600);
  }
}

function describe(prev, v) {
  const tried = ACTIONS[v.action], moved = v.actual_action == null ? null : ACTIONS[v.actual_action];
  if (v.won) return `Reached the goal in ${v.steps} steps.`;
  if (v.fell) return `Tried ${tried}${v.slipped && moved ? ` but slid ${moved}` : ""} and fell through the ice after ${v.steps} steps.`;
  if (v.truncated) return `Out of moves after ${v.steps} steps.`;
  if (moved === null) return `Tried ${tried} and ${v.slipped ? "slid into the edge" : "bumped the edge"}, staying put.`;
  return v.slipped ? `Tried ${tried} but slid ${moved}.` : `Moved ${moved}.`;
}

async function playStep(action) {
  if (S.busy || !S.snap || S.snap.status === "training") return;
  S.busy = true;
  try {
    if (!S.play || S.play.done) { await playReset(); narrate("New episode. Press Step or use the arrow keys."); return; }
    const prev = S.play;
    clearFx();
    const v = await api("/api/play/step", action == null ? {} : { action });
    S.play = v;
    placeAgent(v.state, true);
    narrate(describe(prev, v));
    await sleep(REDUCED ? 0 : 330);
    if (v.slipped && !v.done) { S.agentEl.classList.add("slip"); bubble("Slipped!", v.state); }
    if (v.fell) { S.agentEl.classList.add("sink"); bubble("Splash!", v.state); }
    if (v.won) { S.agentEl.classList.add("win"); confetti(v.state); }
  } finally { S.busy = false; }
}

async function replay(path) {
  if (S.replaying || !path || !path.length) return;
  S.replaying = true;
  const a = S.agentEl, dt = REDUCED ? 0 : Math.max(20, Math.min(110, 700 / path.length));
  a.style.setProperty("--dur", Math.max(dt * 0.9, 10) + "ms");
  try {
    for (const st of REDUCED ? [path[path.length - 1]] : path) {
      if (!S.snap || S.snap.status !== "training") break;
      placeAgent(st, true);
      await sleep(dt);
    }
  } finally { a.style.removeProperty("--dur"); S.replaying = false; }
}

function stopAuto() {
  S.auto = false; S.autoToken++;
  const b = $("#btn-auto");
  b.setAttribute("aria-pressed", "false"); b.textContent = "Auto-play";
}

async function autoLoop() {
  const token = ++S.autoToken;
  while (S.auto && token === S.autoToken) {
    if (!S.snap || S.snap.status === "training") { await sleep(300); continue; }
    if (!S.play || S.play.done) {
      if (S.play && S.play.done) await sleep(REDUCED ? 600 : 1600);
      if (!S.auto || token !== S.autoToken) break;
      await playReset();
      await sleep(450);
      continue;
    }
    await playStep();
    await sleep(REDUCED ? 300 : 560);
  }
}

$("#btn-step").addEventListener("click", guard(() => playStep()));
$("#btn-new").addEventListener("click", guard(async () => { stopAuto(); await playReset(); narrate("New episode. Press Step or use the arrow keys."); }));
$("#btn-auto").addEventListener("click", () => {
  if (S.auto) { stopAuto(); return; }
  S.auto = true;
  const b = $("#btn-auto");
  b.setAttribute("aria-pressed", "true"); b.textContent = "Pause";
  guard(autoLoop)();
});
$("#board").addEventListener("keydown", (e) => {
  const map = { ArrowLeft: 0, ArrowDown: 1, ArrowRight: 2, ArrowUp: 3 };
  if (!(e.key in map)) return;
  e.preventDefault();
  guard(playStep)(map[e.key]);
});

/* -------------------------------------------------------------- controls */
const SLIDERS = [
  { key: "alpha", label: "Learning rate", sym: "α", min: 0.01, max: 1, step: 0.01, fmt: (v) => v.toFixed(2), help: () => "How far each update moves a Q-value." },
  { key: "gamma", label: "Discount factor", sym: "γ", min: 0.5, max: 0.999, step: 0.001, fmt: (v) => v.toFixed(3), help: () => "How much reward later on is worth today." },
  { key: "epsilon_decay", label: "Epsilon decay", sym: "ε", min: 0.99, max: 0.9999, step: 0.0001, fmt: (v) => v.toFixed(4), help: decayHint },
  { key: "epsilon_min", label: "Minimum epsilon", sym: "ε", min: 0, max: 0.2, step: 0.005, fmt: (v) => v.toFixed(3), help: () => "Exploration the agent never gives up." },
  { key: "episodes", label: "Episodes", sym: "", min: 1000, max: 100000, step: 1000, fmt: (v) => nf.format(v), help: () => "More episodes help most on the 8 × 8 map." },
];

function decayHint() {
  const { epsilon_start: a, epsilon_min: m, epsilon_decay: d, episodes } = P;
  if (m <= 0) return "Epsilon keeps shrinking toward zero.";
  const n = Math.ceil(Math.log(m / a) / Math.log(d));
  return n > episodes ? `Won't reach the minimum within ${nf.format(episodes)} episodes.` : `Reaches the minimum after about ${nf.format(n)} episodes.`;
}

const sliderBox = $("#sliders");
SLIDERS.forEach((d) => {
  const wrap = document.createElement("div");
  wrap.innerHTML = `
    <div class="flex items-baseline justify-between gap-3 mb-2">
      <label for="sl-${d.key}" class="text-sm font-semibold">${d.label}${d.sym ? ` <span class="text-muted font-normal">${d.sym}</span>` : ""}</label>
      <output id="out-${d.key}" for="sl-${d.key}" class="text-sm font-bold tabular-nums"></output>
    </div>
    <input id="sl-${d.key}" class="range" type="range" min="${d.min}" max="${d.max}" step="${d.step}">
    <p id="hint-${d.key}" class="text-xs text-muted mt-1.5"></p>`;
  sliderBox.appendChild(wrap);
  const input = $("input", wrap);
  input.addEventListener("input", () => { P[d.key] = parseFloat(input.value); paintSliders(); });
  input.addEventListener("change", () => scheduleRetrain());
});

function paintSliders() {
  SLIDERS.forEach((d) => {
    const input = $("#sl-" + d.key), v = P[d.key];
    input.value = v;
    input.style.setProperty("--pct", ((v - d.min) / (d.max - d.min)) * 100 + "%");
    $("#out-" + d.key).textContent = d.fmt(v);
    $("#hint-" + d.key).textContent = d.help();
  });
}

function syncControls() {
  paintSliders();
  setSeg("#seg-map", P.map_name);
  setSeg("#seg-speed", String(P.pace));
  $("#sw-slippery").setAttribute("aria-checked", P.slippery);
  $("#seed").value = P.seed;
}

function setSeg(sel, value) {
  document.querySelectorAll(sel + " button").forEach((b) => b.setAttribute("aria-pressed", b.dataset.v === value));
}

let retrainTimer;
function scheduleRetrain() {
  if (!$("#auto-retrain").checked) return;
  clearTimeout(retrainTimer);
  retrainTimer = setTimeout(guard(startTraining), 450);
}

const startTraining = () => api("/api/train", P);
const resetAgent = () => api("/api/reset", P);

$("#seg-map").addEventListener("click", guard(async (e) => {
  const b = e.target.closest("button"); if (!b) return;
  P.map_name = b.dataset.v;
  if (P.map_name === "8x8" && P.episodes < 30000) P.episodes = 30000;
  syncControls();
  stopAuto();
  await resetAgent();
  scheduleRetrain();
}));
$("#sw-slippery").addEventListener("click", guard(async () => {
  P.slippery = !P.slippery;
  syncControls();
  stopAuto();
  await resetAgent();
  scheduleRetrain();
}));
$("#seg-speed").addEventListener("click", (e) => {
  const b = e.target.closest("button"); if (!b) return;
  P.pace = parseInt(b.dataset.v, 10);
  setSeg("#seg-speed", b.dataset.v);
});
$("#seed").addEventListener("change", (e) => {
  const v = parseInt(e.target.value, 10);
  P.seed = Number.isFinite(v) ? Math.max(0, v) : 42;
  e.target.value = P.seed;
  scheduleRetrain();
});
$("#btn-train").addEventListener("click", guard(startTraining));
$("#btn-stop").addEventListener("click", guard(() => api("/api/stop")));
$("#btn-reset").addEventListener("click", guard(async () => { stopAuto(); await resetAgent(); }));

for (const [id, cls] of [["#ov-policy", "show-policy"], ["#ov-heat", "show-heat"], ["#ov-trail", "show-trail"]]) {
  $(id).addEventListener("click", () => {
    const on = $(id).getAttribute("aria-pressed") !== "true";
    $(id).setAttribute("aria-pressed", on);
    $("#board").classList.toggle(cls, on);
  });
}

$("#theme-btn").addEventListener("click", () => {
  const root = document.documentElement;
  const dark = root.dataset.theme ? root.dataset.theme === "dark" : window.matchMedia("(prefers-color-scheme: dark)").matches;
  root.dataset.theme = dark ? "light" : "dark";
  try { localStorage.setItem("fl-theme", root.dataset.theme); } catch (_) { /* storage may be blocked */ }
});

/* --------------------------------------------------------------- stream */
function connect() {
  const es = new EventSource("/api/stream");
  let everConnected = false;
  es.onmessage = (e) => { everConnected = true; $("#conn").textContent = "Live"; onSnap(JSON.parse(e.data)); };
  es.onerror = () => {
    $("#conn").textContent = "Reconnecting…";
    if (!everConnected) toast("Backend not reachable. Run: uvicorn backend:app --reload, then open http://localhost:8000");
  };
}
paintSliders();
connect();
