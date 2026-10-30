"use strict";

const $ = (id) => document.getElementById(id);
const video = $("video");

const CLASS_COLORS = {
  "Goal": "#3fb950", "Sending off": "#f85149", "Yellow card": "#e3b341", "Shots on target": "#4ea1ff",
  "Penalty": "#d2a8ff", "Shots off target": "#79c0ff", "Foul": "#ffa657", "Corner": "#a5d6ff",
  "Offside": "#ff7b72", "Kick-off": "#8b949e", "Substitution": "#7ee787", "Direct free-kick": "#f0883e",
  "Indirect free-kick": "#bc8cff", "Clearance": "#6e7681", "Ball out of play": "#484f58", "Throw-in": "#56d4dd",
  "Goal disallowed": "#ff9bce",
};
const DEFAULT_ON = ["Goal", "Sending off", "Yellow card", "Shots on target", "Penalty"];
const KEY_CLASSES = new Set(["Goal", "Sending off", "Yellow card", "Penalty", "Goal disallowed"]);
const ROLE_COLORS = { home: "#4ea1ff", draw: "#9da7b3", away: "#ffa657" };
const WINDOWS = [60, 180, 600, 1800, 0];
const LEAD_S = 5; // seek this far before an event so you can watch it happen

let S = null;
const ui = { win: 600, live: true, hindsight: false, minConf: 0.3, enabled: new Set(DEFAULT_ON), hover: null, dirty: true };
let lastSide = 0;
let listKeys = {};

// ---- helpers ------------------------------------------------------------------

const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
const pad = (n) => String(Math.floor(n)).padStart(2, "0");
function fmtVideo(t) {
  t = Math.max(0, t);
  return `${Math.floor(t / 3600)}:${pad((t % 3600) / 60)}:${pad(t % 60)}`;
}
function fmtMMSS(t) {
  const s = Math.abs(t);
  return `${t < 0 ? "-" : ""}${pad(s / 60)}:${pad(s % 60)}`;
}
const segmentAt = (t) => (S.clock.segments || []).find((s) => t >= s.start_s && t <= s.end_s);
const inFootage = (t) => { const s = segmentAt(t); return !s || (t >= s.footage[0] && t <= s.footage[1]); };
function matchClock(t) {
  if (S.clock.segments) {
    const s = segmentAt(t);
    if (!s) return { label: "–", half: "" };
    const i = S.clock.segments.indexOf(s) + 1, m = s.minute + (t - s.anchor_video_s) / 60;
    return { label: `~${Math.round(m)}'`, half: `clip ${i}/${S.clock.segments.length}: ${s.label} (${teamName(s.team)})` };
  }
  const { video_kickoff_s: k1, video_second_half_s: k2 } = S.clock;
  if (k2 != null && t >= k2) {
    const m = (t - k2) / 60;
    return { label: m < 45 ? `${Math.floor(45 + m) + 1}'` : `90+${Math.floor(m - 45) + 1}'`, half: `2H ${fmtMMSS(t - k2)}` };
  }
  if (t < k1) return { label: "pre", half: `KO in ${fmtMMSS(k1 - t)}` };
  const m = (t - k1) / 60;
  return { label: m < 45 ? `${Math.floor(m) + 1}'` : `45+${Math.floor(m - 45) + 1}'`, half: `1H ${fmtMMSS(t - k1)}` };
}
function utcAt(t) {
  const c = S.clock;
  if (!c.kickoff_utc) return null;
  if (c.segments) {
    const s = segmentAt(t);
    return s ? new Date(Date.parse(s.anchor_utc) + (t - s.anchor_video_s) * 1000).toISOString().slice(11, 19) : null;
  }
  const k2 = c.video_second_half_s;
  const base = k2 != null && t >= k2 ? Date.parse(c.second_half_utc) + (t - k2) * 1000
                                     : Date.parse(c.kickoff_utc) + (t - c.video_kickoff_s) * 1000;
  return new Date(base).toISOString().slice(11, 19);
}
function upperBound(arr, x) { // first index with arr[i] > x
  let lo = 0, hi = arr.length;
  while (lo < hi) { const mid = (lo + hi) >> 1; if (arr[mid] <= x) lo = mid + 1; else hi = mid; }
  return lo;
}
const teamName = (team) => (team && S.teams && S.teams[team]) || team || "?";
const roleName = (r) => (S.market && S.market.names[r]) || r;
const now = () => video.currentTime || 0;
const knownBy = (t) => (ui.live ? now() : Infinity) >= t;
function scoreAt(arr, t) {
  const i = Math.floor(t * S.fps);
  return i >= 0 && i < arr.length ? arr[i] / 1000 : null;
}
function marketAt(field, role, t) {
  const m = S.market;
  if (!m) return null;
  const i = clamp(Math.floor(t / m.grid_s), 0, m[field][role].length - 1);
  return m[field][role][i];
}
function money(x) { return `${x >= 0 ? "+" : "−"}$${Math.abs(x).toFixed(2)}`; }

// ---- loading --------------------------------------------------------------------

async function init() {
  const sessions = await (await fetch("/api/sessions")).json();
  const sel = $("session");
  sel.innerHTML = sessions.map((s) => `<option value="${s.name}">${s.name}</option>`).join("");
  if (!sessions.length) {
    $("title").textContent = "no sessions yet: run `varflash replay build`";
    return;
  }
  const want = new URLSearchParams(location.search).get("s");
  sel.value = sessions.some((s) => s.name === want) ? want : sessions[0].name;
  sel.onchange = () => { history.replaceState(null, "", `?s=${encodeURIComponent(sel.value)}`); load(sel.value); };
  await load(sel.value);
}

async function load(name) {
  const base = `/sessions/${encodeURIComponent(name)}/`;
  S = await (await fetch(base + "session.json")).json();
  S.base = base;
  S.alerts.sort((a, b) => a.fire_s - b.fire_s);
  S.alertFire = S.alerts.map((a) => a.fire_s);
  S.truth.sort((a, b) => a.video_s - b.video_s);
  S.decisions.sort((a, b) => a.video_s - b.video_s);
  S.tradedAlerts = new Set(S.decisions.filter((d) => d.contracts > 0).map((d) => d.alert_id));
  if (S.market) {
    const tp = S.market.tape, idx = tp.v.map((_, i) => i).sort((a, b) => tp.v[a] - tp.v[b]);
    for (const k of Object.keys(tp)) tp[k] = idx.map((i) => tp[k][i]);
  }
  $("title").textContent = S.title + (S.market ? `  ·  ${S.market.event_ticker}` : "");
  $("note").hidden = !S.note;
  $("note").textContent = S.note;
  $("market-card").hidden = !S.market;
  $("tape-card").hidden = !S.market;
  $("trades-card").hidden = !S.market;
  $("min-conf").value = ui.minConf = S.trade_min_confidence ?? 0.3;
  buildToggles();
  renderScorecard();
  listKeys = {};
  video.src = S.base + S.video;
  ui.dirty = true;
}

function buildToggles() {
  const classes = S.live.model.classes;
  const order = [...DEFAULT_ON.filter((c) => classes.includes(c)), ...classes.filter((c) => !DEFAULT_ON.includes(c))];
  $("class-toggles").innerHTML = order.map((c) => `<label><input type="checkbox" data-cls="${c}" ${ui.enabled.has(c) ? "checked" : ""}>
    <span class="swatch" style="background:${CLASS_COLORS[c] || "#888"}"></span>${c}</label>`).join("");
  for (const cb of $("class-toggles").querySelectorAll("input")) {
    cb.onchange = () => { cb.checked ? ui.enabled.add(cb.dataset.cls) : ui.enabled.delete(cb.dataset.cls); ui.dirty = true; listKeys = {}; };
  }
}

function renderScorecard() {
  const d = S.live.model.delay_s;
  const parts = (S.scorecard || []).map((r) => {
    const lat = r.median_latency_s == null ? "" : `, median ${r.median_latency_s >= 0 ? "+" : ""}${r.median_latency_s.toFixed(0)}s`;
    return `<b>${r.label}</b> ${r.caught}/${r.events} caught, ${r.false_alarms} false${lat}`;
  });
  $("scorecard").innerHTML = `live model (${S.live.model.causal ? `causal, ${d}s look-ahead` : "NOT causal"}), conf ≥ ${S.trade_min_confidence}: ` +
    (parts.join(" · ") || "no reference events");
}

// ---- timeline -------------------------------------------------------------------

const tl = $("tl"), ov = $("overview");
const L = 64, R = 10;
const LANES = { model: [8, 128], truth: 140, alerts: 160, trades: 180, market: [196, 300], axis: 316 };

function fitCanvas(c) {
  const dpr = window.devicePixelRatio || 1, w = c.clientWidth, h = c.clientHeight;
  if (c.width !== Math.round(w * dpr) || c.height !== Math.round(h * dpr)) {
    c.width = Math.round(w * dpr); c.height = Math.round(h * dpr);
  }
  const g = c.getContext("2d");
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  return [g, w, h];
}

function viewRange() {
  if (!ui.win || ui.win >= S.duration_s) return [0, S.duration_s];
  const t0 = now() - 0.7 * ui.win;
  return [t0, t0 + ui.win];
}

function drawTimeline() {
  const [g, W, H] = fitCanvas(tl);
  const [t0, t1] = viewRange();
  const X = (t) => L + ((t - t0) / (t1 - t0)) * (W - L - R);
  const T = (x) => t0 + ((x - L) / (W - L - R)) * (t1 - t0);
  const tNow = now(), horizon = ui.live ? tNow : Infinity;
  g.clearRect(0, 0, W, H);
  g.font = "11px system-ui"; g.textBaseline = "middle";

  // lane frames + labels
  g.fillStyle = "#8a94a3";
  const [m0, m1] = LANES.model, [k0, k1] = LANES.market;
  g.fillText("Model", 8, (m0 + m1) / 2);
  g.fillText("Truth", 8, LANES.truth); g.fillText("Alerts", 8, LANES.alerts); g.fillText("Trades", 8, LANES.trades);
  g.fillText("Kalshi", 8, k0 + 10);
  g.strokeStyle = "#2a3039"; g.lineWidth = 1;
  for (const y of [m0, m1, (m0 + m1) / 2, LANES.truth - 10, LANES.trades + 10, k0, k1]) {
    g.beginPath(); g.moveTo(L, y + 0.5); g.lineTo(W - R, y + 0.5); g.stroke();
  }

  // x axis (video time) and half boundaries
  const span = t1 - t0, steps = [10, 30, 60, 120, 300, 600, 900, 1800];
  const step = steps.find((s) => span / s <= 12) || 3600;
  g.fillStyle = "#8a94a3"; g.textAlign = "center";
  for (let t = Math.ceil(t0 / step) * step; t <= t1; t += step) {
    const x = X(t);
    g.strokeStyle = "#20252c"; g.beginPath(); g.moveTo(x + 0.5, m0); g.lineTo(x + 0.5, k1); g.stroke();
    if (t >= 0) g.fillText(`${fmtVideo(t)} (${matchClock(t).label})`, x, LANES.axis);
  }
  g.textAlign = "left";
  for (const s of S.clock.segments || []) { // frozen lead-in / run-out: market runs, no footage
    g.fillStyle = "#ffffff08";
    for (const [a, b] of [[s.start_s, s.footage[0]], [s.footage[1], s.end_s]]) {
      if (b > t0 && a < t1) g.fillRect(X(Math.max(a, t0)), m0, X(Math.min(b, t1)) - X(Math.max(a, t0)), k1 - m0);
    }
    if (s.start_s >= t0 && s.start_s <= t1) {
      g.strokeStyle = "#8a94a3"; g.beginPath(); g.moveTo(X(s.start_s) + 0.5, m0); g.lineTo(X(s.start_s) + 0.5, k1); g.stroke();
      g.fillStyle = "#8a94a3"; g.fillText(`${s.label} ${teamName(s.team)} · ${s.minute}'`, X(s.start_s) + 4, m0 + 8);
    }
  }
  const k2 = S.clock.segments ? null : S.clock.video_second_half_s;
  for (const [t, lbl] of [[S.clock.segments ? null : S.clock.video_kickoff_s, "KO"], [k2, "2H"]]) {
    if (t == null || t < t0 || t > t1) continue;
    g.strokeStyle = "#6e7681"; g.setLineDash([4, 4]);
    g.beginPath(); g.moveTo(X(t), m0); g.lineTo(X(t), k1); g.stroke(); g.setLineDash([]);
    g.fillStyle = "#6e7681"; g.fillText(lbl, X(t) + 3, m0 + 8);
  }

  // model scores (event-aligned, so a causal model's curve is only known delay_s later)
  const yScore = (v) => m1 - v * (m1 - m0);
  const drawScores = (scores, until, dashed) => {
    g.setLineDash(dashed ? [5, 3] : []);
    for (const cls of ui.enabled) {
      const arr = scores[cls];
      if (!arr) continue;
      const i0 = Math.max(0, Math.floor(t0 * S.fps)), i1 = Math.min(arr.length - 1, Math.ceil(Math.min(t1, until) * S.fps));
      if (i1 <= i0) continue;
      const per = Math.max(1, Math.floor((i1 - i0) / (W - L - R)));
      g.strokeStyle = CLASS_COLORS[cls] || "#888"; g.lineWidth = dashed ? 1 : 1.5; g.globalAlpha = dashed ? 0.7 : 1;
      g.beginPath();
      for (let i = i0; i <= i1; i += per) {
        let v = 0;
        for (let j = i; j < Math.min(i + per, i1 + 1); j++) v = Math.max(v, arr[j]);
        const x = X(i / S.fps), y = yScore(v / 1000);
        i === i0 ? g.moveTo(x, y) : g.lineTo(x, y);
      }
      g.stroke();
    }
    g.setLineDash([]); g.globalAlpha = 1; g.lineWidth = 1;
  };
  if (ui.hindsight && S.hindsight) drawScores(S.hindsight.scores, Infinity, true);
  drawScores(S.live.scores, horizon - S.live.model.delay_s, false);
  if (ui.live && S.live.model.delay_s > 0 && tNow >= t0) { // the not-yet-known stretch of the live curve
    g.fillStyle = "#ffffff0d";
    g.fillRect(X(tNow - S.live.model.delay_s), m0, X(tNow) - X(tNow - S.live.model.delay_s), m1 - m0);
  }

  // truth
  g.textAlign = "center"; g.font = "bold 10px system-ui";
  for (const e of S.truth) {
    if (e.video_s < t0 || e.video_s > t1 || e.video_s > horizon || !ui.enabled.has(e.label)) continue;
    const x = X(e.video_s);
    g.fillStyle = CLASS_COLORS[e.label] || "#888";
    g.beginPath(); g.moveTo(x, LANES.truth - 6); g.lineTo(x - 5, LANES.truth + 5); g.lineTo(x + 5, LANES.truth + 5); g.fill();
    if (KEY_CLASSES.has(e.label)) {
      g.globalAlpha = 0.25; g.strokeStyle = g.fillStyle;
      g.beginPath(); g.moveTo(x + 0.5, m0); g.lineTo(x + 0.5, k1); g.stroke(); g.globalAlpha = 1;
    }
  }
  g.font = "11px system-ui"; g.textAlign = "left";

  // alerts: bar from the incident to when the live model fires
  for (const a of S.alerts) {
    if (a.confidence < ui.minConf || !ui.enabled.has(a.label) || a.fire_s > horizon) continue;
    if (a.fire_s < t0 || a.time_s > t1) continue;
    const c = CLASS_COLORS[a.label] || "#888", y = LANES.alerts;
    g.strokeStyle = c; g.globalAlpha = 0.6;
    g.beginPath(); g.moveTo(X(a.time_s), y); g.lineTo(X(a.fire_s), y); g.stroke(); g.globalAlpha = 1;
    g.beginPath(); g.arc(X(a.fire_s), y, 2.5 + 4 * a.confidence, 0, 2 * Math.PI);
    if (S.market && a.label in { "Goal": 1, "Sending off": 1 } && !a.team) {
      g.strokeStyle = c; g.stroke(); // no reference event -> no team -> not traded
    } else { g.fillStyle = c; g.fill(); }
  }

  // market
  if (S.market) {
    const m = S.market, roles = m.roles;
    const i0 = Math.max(0, Math.floor(t0 / m.grid_s)), last = Math.min(Math.ceil(t1 / m.grid_s), m.prob[roles[0]].length - 1);
    const i1 = ui.live ? Math.min(last, Math.floor(tNow / m.grid_s)) : last;
    let lo = 1, hi = 0;
    for (const r of roles) for (let i = i0; i <= last; i++) {
      for (const v of [m.bid[r][i], m.ask[r][i], m.prob[r][i]]) if (v != null) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
    }
    if (lo > hi) { lo = 0; hi = 1; }
    const padY = Math.max(0.03, (hi - lo) * 0.1); lo = clamp(lo - padY, 0, 1); hi = clamp(hi + padY, 0, 1);
    const Y = (v) => k1 - ((v - lo) / (hi - lo)) * (k1 - k0);
    g.fillStyle = "#8a94a3"; g.textAlign = "right";
    for (const v of [lo, (lo + hi) / 2]) g.fillText(`${Math.round(v * 100)}%`, L - 4, Y(v) - 6);
    g.textAlign = "left";
    for (const r of roles) {
      const c = ROLE_COLORS[r] || "#ccc";
      g.fillStyle = c; g.globalAlpha = 0.13; g.beginPath(); // bid/ask band
      let started = false;
      for (let i = i0; i <= i1; i++) { if (m.ask[r][i] == null) continue; const x = X(i * m.grid_s); started ? g.lineTo(x, Y(m.ask[r][i])) : (g.moveTo(x, Y(m.ask[r][i])), started = true); }
      for (let i = i1; i >= i0; i--) { if (m.bid[r][i] == null) continue; g.lineTo(X(i * m.grid_s), Y(m.bid[r][i])); }
      g.fill(); g.globalAlpha = 1;
      g.strokeStyle = c; g.lineWidth = 1.5; g.beginPath(); started = false;
      for (let i = i0; i <= i1; i++) { const v = m.prob[r][i]; if (v == null) continue; const x = X(i * m.grid_s); started ? g.lineTo(x, Y(v)) : (g.moveTo(x, Y(v)), started = true); }
      g.stroke(); g.lineWidth = 1;
    }
    // fills on the price chart: entry triangle (up = YES, down = NO) -> exit square, coloured by P&L
    for (const f of S.fills) {
      if (f.entry_s == null || f.entry_s > horizon) continue;
      const yesPx = (px) => (f.side === "yes" ? px : 1 - px);
      const xe = X(f.entry_s), ye = Y(yesPx(f.entry_px)), c = ROLE_COLORS[f.role];
      const exited = f.exit_s != null && f.exit_s <= horizon;
      if (exited) {
        g.strokeStyle = f.pnl >= 0 ? "#3fb950" : "#f85149"; g.setLineDash([3, 3]);
        g.beginPath(); g.moveTo(xe, ye); g.lineTo(X(f.exit_s), Y(yesPx(f.exit_px))); g.stroke(); g.setLineDash([]);
        g.fillStyle = g.strokeStyle; g.fillRect(X(f.exit_s) - 3, Y(yesPx(f.exit_px)) - 3, 6, 6);
      }
      g.fillStyle = c; g.strokeStyle = "#fff"; g.beginPath();
      const d = f.side === "yes" ? -1 : 1;
      g.moveTo(xe, ye + d * 7); g.lineTo(xe - 6, ye - d * 4); g.lineTo(xe + 6, ye - d * 4); g.closePath(); g.fill(); g.stroke();
    }
  }

  // decisions row: traded (filled triangle) vs skipped (x)
  for (const d of S.decisions) {
    if (d.video_s == null || d.video_s > horizon || d.video_s < t0 || d.video_s > t1) continue;
    const x = X(d.video_s), y = LANES.trades, c = ROLE_COLORS[d.role] || "#ccc";
    if (d.contracts > 0) {
      const s = d.side === "yes" ? -1 : 1;
      g.fillStyle = c; g.beginPath(); g.moveTo(x, y + s * 6); g.lineTo(x - 5, y - s * 4); g.lineTo(x + 5, y - s * 4); g.fill();
    } else {
      g.strokeStyle = "#6e7681"; g.beginPath(); g.moveTo(x - 4, y - 4); g.lineTo(x + 4, y + 4); g.moveTo(x + 4, y - 4); g.lineTo(x - 4, y + 4); g.stroke();
    }
  }

  // playhead + hover line
  if (tNow >= t0 && tNow <= t1) {
    g.strokeStyle = "#fff"; g.lineWidth = 1.5;
    g.beginPath(); g.moveTo(X(tNow), 2); g.lineTo(X(tNow), k1 + 4); g.stroke(); g.lineWidth = 1;
  }
  if (ui.hover != null) {
    const x = ui.hover;
    g.strokeStyle = "#ffffff55"; g.setLineDash([2, 3]);
    g.beginPath(); g.moveTo(x, 2); g.lineTo(x, k1); g.stroke(); g.setLineDash([]);
  }
  tl._T = T; tl._X = X;
}

function drawOverview() {
  const [g, W, H] = fitCanvas(ov);
  const dur = S.duration_s, X = (t) => 4 + (t / dur) * (W - 8);
  const horizon = ui.live ? now() : Infinity;
  g.clearRect(0, 0, W, H);
  if (S.market) {
    const m = S.market;
    for (const r of m.roles) {
      g.strokeStyle = ROLE_COLORS[r]; g.globalAlpha = 0.6; g.beginPath();
      const arr = m.prob[r], n = Math.min(arr.length - 1, Math.floor(Math.min(horizon, dur) / m.grid_s));
      let started = false;
      for (let i = 0; i <= n; i += 2) { if (arr[i] == null) continue; const x = X(i * m.grid_s), y = H - 4 - arr[i] * (H - 8); started ? g.lineTo(x, y) : (g.moveTo(x, y), started = true); }
      g.stroke(); g.globalAlpha = 1;
    }
  }
  for (const e of S.truth) {
    if (!KEY_CLASSES.has(e.label) || !ui.enabled.has(e.label) || e.video_s > horizon) continue;
    g.fillStyle = CLASS_COLORS[e.label]; g.fillRect(X(e.video_s) - 1, 2, 2, 10);
  }
  for (const d of S.decisions) {
    if (d.contracts <= 0 || d.video_s == null || d.video_s > horizon) continue;
    g.fillStyle = ROLE_COLORS[d.role]; g.fillRect(X(d.video_s) - 1, H - 12, 3, 10);
  }
  const [t0, t1] = viewRange();
  g.strokeStyle = "#4ea1ff"; g.strokeRect(X(Math.max(t0, 0)), 1, Math.max(2, X(Math.min(t1, dur)) - X(Math.max(t0, 0))), H - 2);
  g.strokeStyle = "#fff"; g.beginPath(); g.moveTo(X(now()), 0); g.lineTo(X(now()), H); g.stroke();
  ov._T = (x) => clamp(((x - 4) / (W - 8)) * dur, 0, dur);
}

function tooltip(ev) {
  const rect = tl.getBoundingClientRect(), x = ev.clientX - rect.left, y = ev.clientY - rect.top;
  if (x < L || !tl._T) { $("tooltip").hidden = true; ui.hover = null; ui.dirty = true; return; }
  ui.hover = x; ui.dirty = true;
  const t = tl._T(x), mc = matchClock(t), lines = [`${fmtVideo(t)}  ${mc.label}  ${mc.half}${utcAt(t) ? "  " + utcAt(t) + " UTC" : ""}`];
  const liveKnown = !ui.live || t <= now() - S.live.model.delay_s;
  for (const c of ui.enabled) {
    const v = S.live.scores[c] && scoreAt(S.live.scores[c], t);
    const hv = ui.hindsight && S.hindsight && S.hindsight.scores[c] ? scoreAt(S.hindsight.scores[c], t) : null;
    const h = hv != null ? ` (hindsight ${hv.toFixed(2)})` : "";
    if (v != null) lines.push(`${c}: ${liveKnown ? v.toFixed(2) : "not known yet"}${h}`);
  }
  if (S.market && (!ui.live || t <= now())) {
    lines.push(S.market.roles.map((r) => `${roleName(r)} ${Math.round(100 * (marketAt("prob", r, t) ?? NaN))}%`).join("  "));
  }
  const near = (s) => Math.abs(tl._X(s) - x) < 6;
  for (const e of S.truth) if (near(e.video_s) && ui.enabled.has(e.label) && knownBy(e.video_s)) lines.push(`TRUTH ${e.label} (${teamName(e.team)})`);
  for (const a of S.alerts) if ((near(a.fire_s) || near(a.time_s)) && a.confidence >= ui.minConf && ui.enabled.has(a.label) && knownBy(a.fire_s))
    lines.push(`ALERT ${a.label} conf ${a.confidence.toFixed(2)}, fired +${(a.fire_s - a.time_s).toFixed(0)}s after the incident`);
  for (const d of S.decisions) if (d.video_s != null && near(d.video_s) && knownBy(d.video_s))
    lines.push(d.contracts > 0 ? `BUY ${d.side.toUpperCase()} ${roleName(d.role)} ${d.contracts} @ ${d.price.toFixed(2)}, edge ${d.edge.toFixed(3)}`
                               : `skip ${d.event} (${teamName(d.team)}): best edge ${d.edge.toFixed(3)} < min`);
  const tt = $("tooltip");
  tt.textContent = lines.join("\n"); tt.hidden = false;
  tt.style.left = `${Math.min(x + 12, rect.width - tt.offsetWidth - 4)}px`;
  tt.style.top = `${Math.max(0, y - tt.offsetHeight - 8)}px`;
}

// ---- side panel ------------------------------------------------------------------

function setList(id, key, items) {
  if (listKeys[id] === key) return;
  const fresh = listKeys[id] !== undefined && !video.paused;
  listKeys[id] = key;
  const ol = $(id);
  ol.innerHTML = "";
  items.forEach((it, i) => {
    const li = document.createElement("li");
    li.innerHTML = it.html;
    if (it.cls) li.className = it.cls;
    if (fresh && i === 0 && it.isNew) li.classList.add("fresh");
    if (it.seek != null) li.onclick = () => seek(it.seek);
    ol.appendChild(li);
  });
}

function updateSide() {
  const t = now(), mc = matchClock(t), utc = utcAt(t);
  $("clock").innerHTML = `<b>${mc.label}</b>${mc.half} · video ${fmtVideo(t)} / ${fmtVideo(S.duration_s)}${utc ? ` · ${utc} UTC` : ""}` +
    ` · ${video.playbackRate}×`;
  const horizon = ui.live ? t : Infinity;
  $("frozen").hidden = inFootage(t);

  if (S.market) {
    const m = S.market;
    $("market-sub").textContent = m.result ? `settled: ${roleName(m.result)}` : "";
    $("book").innerHTML = `<tr><th>outcome</th><th>bid</th><th>ask</th><th>implied</th><th>Δ1m</th></tr>` + m.roles.map((r) => {
      const p = marketAt("prob", r, t), p1 = marketAt("prob", r, t - 60), d = p != null && p1 != null ? p - p1 : null;
      const f = (v) => (v == null ? "–" : v.toFixed(2));
      return `<tr><td><span class="swatch" style="background:${ROLE_COLORS[r]}"></span>${roleName(r)}</td><td>${f(marketAt("bid", r, t))}</td>
        <td>${f(marketAt("ask", r, t))}</td><td>${p == null ? "–" : (100 * p).toFixed(1) + "%"}</td>
        <td class="${d > 0.005 ? "pos" : d < -0.005 ? "neg" : ""}">${d == null ? "–" : (d >= 0 ? "+" : "") + (100 * d).toFixed(1)}</td></tr>`;
    }).join("");
  }

  // strategy P&L as of t (open positions marked at the current exit price)
  let realized = 0, open = 0, nOpen = 0, nTrades = 0;
  for (const f of S.fills) {
    if (f.entry_s == null || f.entry_s > horizon) continue;
    nTrades++;
    if (f.exit_s != null && f.exit_s <= horizon) { realized += f.pnl; continue; }
    nOpen++;
    const bid = marketAt("bid", f.role, t), ask = marketAt("ask", f.role, t);
    if (bid != null && ask != null) open += f.contracts * ((f.side === "yes" ? bid : 1 - ask) - f.entry_px);
  }
  const cls = (v) => (v > 0 ? "pos" : v < 0 ? "neg" : "");
  const sum = S.summary;
  $("pnl").innerHTML = !S.market ? `<span>trading</span><span>no Kalshi event in this session</span>` :
    `<span>trades so far</span><span>${nTrades}</span>
     <span>realized</span><span class="${cls(realized)}">${money(realized)}</span>
     <span>open (${nOpen}), marked</span><span class="${cls(open)}">${money(open)}</span>
     <span>full match</span><span>${sum ? `${sum.trades} trades, ${money(sum.pnl)}${sum.hit_rate != null ? `, hit ${Math.round(100 * sum.hit_rate)}%` : ""}` : "–"}</span>`;

  // alerts
  const nAlerts = upperBound(S.alertFire, horizon);
  const shown = [];
  for (let i = nAlerts - 1; i >= 0 && shown.length < 80; i--) {
    const a = S.alerts[i];
    if (a.confidence < ui.minConf || !ui.enabled.has(a.label)) continue;
    const marketEv = a.label === "Goal" || a.label === "Sending off";
    let tail = "";
    if (marketEv && S.market) {
      tail = a.no_trade ? ` · <i>${a.no_trade}</i>` : !a.team ? " · <i>no reference event: not traded</i>"
           : S.tradedAlerts.has(a.id) ? " · <b>traded</b>" : " · no trade";
    }
    const who = a.team ? ` ${teamName(a.team)}` : "";
    shown.push({
      html: `<span class="t">${matchClock(a.fire_s).label}</span><span style="color:${CLASS_COLORS[a.label] || "#ccc"}">${a.label}</span>${who}` +
            ` · ${a.confidence.toFixed(2)} · +${(a.fire_s - a.time_s).toFixed(0)}s${tail}`,
      cls: marketEv ? "" : "dim", seek: a.time_s - LEAD_S, isNew: i === nAlerts - 1,
    });
  }
  setList("alerts", `${nAlerts}|${ui.minConf}|${[...ui.enabled]}|${ui.live}`, shown);

  // trades
  const decs = S.decisions.filter((d) => d.video_s != null && d.video_s <= horizon).reverse();
  setList("trades", `${decs.length}|${S.fills.filter((f) => f.exit_s != null && f.exit_s <= horizon).length}|${ui.live}`, decs.slice(0, 80).map((d, i) => {
    const f = S.fills.find((x) => x.alert_id === d.alert_id && Math.abs((x.entry_s ?? -1) - d.video_s) < 1);
    let html = `<span class="t">${matchClock(d.video_s).label}</span>`;
    if (d.contracts > 0) {
      html += `BUY ${d.side.toUpperCase()} ${roleName(d.role)} ${d.contracts} @ ${d.price.toFixed(2)} · edge ${d.edge.toFixed(3)} · ${d.event} ${teamName(d.team)}`;
      if (f && f.exit_s != null && f.exit_s <= horizon) html += ` → ${f.exit_px.toFixed(2)} <span class="${cls(f.pnl)}">${money(f.pnl)}</span>`;
      else if (f) html += " · open";
    } else {
      html += `skip ${d.event} ${teamName(d.team)} · best edge ${d.edge.toFixed(3)}`;
    }
    return { html, cls: d.contracts > 0 ? "" : "dim", seek: d.video_s - LEAD_S - 10, isNew: i === 0 };
  }));

  // tape
  if (S.market) {
    const tp = S.market.tape, n = upperBound(tp.v, t);
    const items = [];
    for (let i = n - 1; i >= Math.max(0, n - 30); i--) {
      items.push({ html: `<span class="t">${fmtVideo(tp.v[i])}</span><span style="color:${ROLE_COLORS[tp.role[i]]}">${roleName(tp.role[i])}</span>` +
        ` ${tp.price[i].toFixed(2)} ×${tp.count[i]} <span class="t">taker ${tp.side[i]}</span>`, seek: null, isNew: i === n - 1 });
    }
    setList("tape", `${n}`, items);
  }
}

// ---- controls ---------------------------------------------------------------------

function seek(t) {
  video.currentTime = clamp(t, 0, S ? S.duration_s : video.duration || 0);
  ui.dirty = true;
}

function keyTimes() {
  const out = [];
  for (const a of S.alerts) if (a.confidence >= ui.minConf && ui.enabled.has(a.label) && KEY_CLASSES.has(a.label)) out.push(a.time_s);
  for (const e of S.truth) if (ui.enabled.has(e.label) && KEY_CLASSES.has(e.label)) out.push(e.video_s);
  for (const d of S.decisions) if (d.video_s != null) out.push(d.video_s);
  return out.sort((a, b) => a - b);
}
function jumpEvent(dir) {
  const t = now(), times = keyTimes();
  const target = dir > 0 ? times.find((x) => x - LEAD_S > t + 1) : [...times].reverse().find((x) => x - LEAD_S < t - 1);
  if (target != null) seek(target - LEAD_S);
}
function setSpeed(v) { video.playbackRate = v; $("speed").value = String(v); }
function setWindow(w) {
  ui.win = w; ui.dirty = true;
  for (const b of document.querySelectorAll("[data-win]")) b.classList.toggle("on", Number(b.dataset.win) === w);
}

for (const b of document.querySelectorAll("[data-skip]")) b.onclick = () => seek(now() + Number(b.dataset.skip));
for (const b of document.querySelectorAll("[data-win]")) b.onclick = () => setWindow(Number(b.dataset.win));
$("play").onclick = () => (video.paused ? video.play() : video.pause());
$("prev-ev").onclick = () => jumpEvent(-1);
$("next-ev").onclick = () => jumpEvent(1);
$("speed").onchange = (e) => setSpeed(Number(e.target.value));
$("live-mode").onchange = (e) => { ui.live = e.target.checked; ui.dirty = true; listKeys = {}; };
$("show-hindsight").onchange = (e) => { ui.hindsight = e.target.checked; ui.dirty = true; };
$("min-conf").onchange = (e) => { ui.minConf = Number(e.target.value); ui.dirty = true; };
video.onplay = video.onpause = () => { $("play").textContent = video.paused ? "Play" : "Pause"; ui.dirty = true; };
video.onseeked = video.onloadedmetadata = video.onratechange = () => (ui.dirty = true);

tl.onmousemove = tooltip;
tl.onmouseleave = () => { $("tooltip").hidden = true; ui.hover = null; ui.dirty = true; };
tl.onclick = (e) => { const x = e.clientX - tl.getBoundingClientRect().left; if (x >= L && tl._T) seek(tl._T(x)); };
tl.onwheel = (e) => {
  e.preventDefault();
  const cur = ui.win || S.duration_s;
  setWindow(clamp(cur * (e.deltaY > 0 ? 1.25 : 0.8), 20, S.duration_s));
};
let dragging = false;
ov.onmousedown = (e) => { dragging = true; seek(ov._T(e.clientX - ov.getBoundingClientRect().left)); };
window.addEventListener("mousemove", (e) => { if (dragging) seek(ov._T(e.clientX - ov.getBoundingClientRect().left)); });
window.addEventListener("mouseup", () => (dragging = false));

window.addEventListener("keydown", (e) => {
  if (!S || ["INPUT", "SELECT"].includes(document.activeElement.tagName)) return;
  const k = e.key, frame = 1 / 25;
  const actions = {
    " ": () => (video.paused ? video.play() : video.pause()),
    ArrowLeft: () => seek(now() - (e.shiftKey ? 30 : 5)),
    ArrowRight: () => seek(now() + (e.shiftKey ? 30 : 5)),
    j: () => seek(now() - 10), l: () => seek(now() + 10), k: () => (video.paused ? video.play() : video.pause()),
    ",": () => { video.pause(); seek(now() - frame); }, ".": () => { video.pause(); seek(now() + frame); },
    n: () => jumpEvent(1), p: () => jumpEvent(-1),
    "[": () => setSpeed([0.25, 0.5, 1, 2, 4, 8, 16].filter((s) => s < video.playbackRate).pop() ?? 0.25),
    "]": () => setSpeed([0.25, 0.5, 1, 2, 4, 8, 16].find((s) => s > video.playbackRate) ?? 16),
  };
  const fn = actions[k] || (k >= "1" && k <= "5" ? () => setWindow(WINDOWS[Number(k) - 1]) : null);
  if (!fn) return;
  e.preventDefault(); e.stopPropagation();
  fn();
}, true);
window.addEventListener("resize", () => (ui.dirty = true));

function loop(ts) {
  if (S && (ui.dirty || !video.paused)) {
    drawTimeline(); drawOverview();
    if (ui.dirty || ts - lastSide > 150) { updateSide(); lastSide = ts; }
    ui.dirty = false;
  }
  requestAnimationFrame(loop);
}

init().then(() => requestAnimationFrame(loop));
