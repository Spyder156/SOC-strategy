/* SOC Sandpile live dashboard — framework-free.
   Connects to the python server's websocket, buffers events, renders canvas charts. */

const WS_URL = `ws://${location.hostname || "localhost"}:8765`;
const MAXPTS = 2000;

// ---- ring buffers ------------------------------------------------------------
const buf = {
  price: [], xc: [], p: [], equity: [],
  alpha: [], beta: [], gv: [], ga: [],
  rollLoss: [], rollBrier: [],
};
let calib = [];
let lastMetric = null;
let activeTab = "market";

function push(arr, v) { arr.push(v); if (arr.length > MAXPTS) arr.shift(); }

// ---- websocket ---------------------------------------------------------------
function connect() {
  const ws = new WebSocket(WS_URL);
  const dot = document.getElementById("conn");
  const txt = document.getElementById("connText");
  ws.onopen = () => { dot.classList.add("on"); txt.textContent = "live"; txt.className = ""; };
  ws.onclose = () => {
    dot.classList.remove("on"); txt.textContent = "disconnected — retrying";
    setTimeout(connect, 1500);
  };
  ws.onmessage = (e) => handle(JSON.parse(e.data));
}

function handle(m) {
  if (m.type === "config") {
    document.getElementById("symbol").textContent = m.symbol;
    document.getElementById("feedtag").textContent = `feed: ${m.feed}`;
  } else if (m.type === "view") {
    push(buf.price, m.x); push(buf.xc, m.x_c); push(buf.p, m.p); push(buf.equity, m.equity);
    push(buf.alpha, m.params.alpha); push(buf.beta, m.params.beta);
    push(buf.gv, m.params.gamma_v); push(buf.ga, m.params.gamma_a);
    updateMarket(m);
  } else if (m.type === "metric") {
    lastMetric = m; calib = m.calibration || [];
    if (m.roll_logloss != null) push(buf.rollLoss, m.roll_logloss);
    if (m.roll_brier != null) push(buf.rollBrier, m.roll_brier);
    updateChips(m); updateModel(m);
  }
}

// ---- DOM updates -------------------------------------------------------------
const fmt = (v, d = 2) => (v == null || isNaN(v)) ? "—" : Number(v).toFixed(d);
const money = (v) => (v == null) ? "—" : "$" + Math.round(v).toLocaleString();

function updateChips(m) {
  const ret = lastMetricReturn();
  const el = document.getElementById("cReturn");
  if (ret != null) { el.textContent = (ret >= 0 ? "+" : "") + ret.toFixed(2) + "%";
    el.className = "v " + (ret > 0 ? "pos" : ret < 0 ? "neg" : "flat"); }
  document.getElementById("cWin").textContent = fmt(m.winrate * 100, 1) + "%";
  document.getElementById("cBrier").textContent = fmt(m.brier, 3);
  document.getElementById("cN").textContent = (m.n || 0).toLocaleString();
  // narrative
  const story = document.getElementById("story");
  if (m.n < 3000) story.textContent = "Cold start — model is guessing; convergence variables still wild.";
  else if (m.winrate > 0.52) story.textContent = "Converging — win rate above 50%, calibration tightening.";
  else story.textContent = "Learning — watch the convergence variables settle and the calibration line straighten.";
}
let _lastReturn = null;
function lastMetricReturn() { return _lastReturn; }

function updateMarket(m) {
  _lastReturn = m.return_pct;
  const exp = m.exposure || 0;
  const ps = document.getElementById("posState");
  if (exp > 1) { ps.textContent = "LONG"; ps.className = "pos"; }
  else if (exp < -1) { ps.textContent = "SHORT"; ps.className = "neg"; }
  else { ps.textContent = "flat"; ps.className = "flat"; }
  document.getElementById("posExp").textContent = (exp >= 0 ? "" : "-") + money(Math.abs(exp));
  document.getElementById("posGap").textContent = fmt(m.gap, 3);
  const ly = document.getElementById("lastY");
  ly.textContent = m.y === 1 ? "▼ down" : "▲ up";
  ly.className = m.y === 1 ? "neg" : "pos";
}

function updateModel(m) {
  document.getElementById("mWin").textContent = `${fmt(m.winrate*100,1)}% / ${fmt(m.roll_winrate*100,1)}%`;
  document.getElementById("mBase").textContent = fmt(m.base_rate*100,1) + "%";
  document.getElementById("mLoss").textContent = fmt(m.logloss,4);
  document.getElementById("mBrier").textContent = fmt(m.brier,4);
  const last = (a) => a.length ? a[a.length-1] : null;
  document.getElementById("mA").textContent = fmt(last(buf.alpha),3);
  document.getElementById("mB").textContent = fmt(last(buf.beta),3);
  document.getElementById("mGv").textContent = fmt(last(buf.gv),3);
  document.getElementById("mGa").textContent = fmt(last(buf.ga),3);
}

// ---- tabs --------------------------------------------------------------------
document.querySelectorAll(".tab").forEach(t => t.onclick = () => {
  document.querySelectorAll(".tab").forEach(x => x.classList.remove("active"));
  document.querySelectorAll(".view").forEach(x => x.classList.remove("active"));
  t.classList.add("active");
  activeTab = t.dataset.tab;
  document.getElementById(activeTab).classList.add("active");
});

// ---- canvas charts -----------------------------------------------------------
function ctxOf(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth, h = canvas.clientHeight;
  if (canvas.width !== w * dpr || canvas.height !== h * dpr) {
    canvas.width = w * dpr; canvas.height = h * dpr;
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  return { ctx, w, h };
}

function lineChart(id, series, opts = {}) {
  const c = document.getElementById(id); if (!c) return;
  const { ctx, w, h } = ctxOf(c);
  const padL = 44, padR = 8, padT = 8, padB = 16;
  const X0 = padL, X1 = w - padR, Y0 = padT, Y1 = h - padB;

  let lo = opts.min, hi = opts.max;
  if (lo == null || hi == null) {
    lo = Infinity; hi = -Infinity;
    series.forEach(s => s.data.forEach(v => { if (v != null && !isNaN(v)) { if (v < lo) lo = v; if (v > hi) hi = v; } }));
    if (!isFinite(lo)) { lo = 0; hi = 1; }
    if (lo === hi) { lo -= 1; hi += 1; }
    const pad = (hi - lo) * 0.08; lo -= pad; hi += pad;
  }
  const n = Math.max(...series.map(s => s.data.length), 1);
  const xAt = i => X0 + (X1 - X0) * (n <= 1 ? 0 : i / (n - 1));
  const yAt = v => Y1 - (Y1 - Y0) * (v - lo) / (hi - lo);

  // grid + y labels
  ctx.strokeStyle = "#232a36"; ctx.fillStyle = "#7d8590"; ctx.font = "10px monospace"; ctx.lineWidth = 1;
  for (let g = 0; g <= 4; g++) {
    const yy = Y0 + (Y1 - Y0) * g / 4; const val = hi - (hi - lo) * g / 4;
    ctx.beginPath(); ctx.moveTo(X0, yy); ctx.lineTo(X1, yy); ctx.stroke();
    ctx.fillText(val.toFixed(opts.dec ?? 2), 4, yy + 3);
  }
  // reference line (e.g. 0.5 for prob, 0 for params)
  if (opts.ref != null && opts.ref >= lo && opts.ref <= hi) {
    ctx.strokeStyle = "#3a4250"; ctx.setLineDash([4, 4]);
    const yy = yAt(opts.ref); ctx.beginPath(); ctx.moveTo(X0, yy); ctx.lineTo(X1, yy); ctx.stroke();
    ctx.setLineDash([]);
  }
  // series
  series.forEach(s => {
    ctx.strokeStyle = s.color; ctx.lineWidth = 1.4; ctx.beginPath();
    let started = false;
    s.data.forEach((v, i) => {
      if (v == null || isNaN(v)) return;
      const X = xAt(i), Y = yAt(v);
      if (!started) { ctx.moveTo(X, Y); started = true; } else ctx.lineTo(X, Y);
    });
    ctx.stroke();
  });
}

function calibChart(id) {
  const c = document.getElementById(id); if (!c) return;
  const { ctx, w, h } = ctxOf(c);
  const pad = 30, X0 = pad, X1 = w - 10, Y0 = 10, Y1 = h - pad;
  const xAt = v => X0 + (X1 - X0) * v, yAt = v => Y1 - (Y1 - Y0) * v;
  // grid box
  ctx.strokeStyle = "#232a36"; ctx.fillStyle = "#7d8590"; ctx.font = "10px monospace";
  ctx.strokeRect(X0, Y0, X1 - X0, Y1 - Y0);
  // diagonal
  ctx.strokeStyle = "#3a4250"; ctx.setLineDash([4, 4]); ctx.beginPath();
  ctx.moveTo(X0, Y1); ctx.lineTo(X1, Y0); ctx.stroke(); ctx.setLineDash([]);
  ctx.fillText("predicted →", X0 + 4, Y1 + 18); ctx.save();
  ctx.translate(10, Y0 + 30); ctx.rotate(-Math.PI/2); ctx.fillText("realised →", 0, 0); ctx.restore();
  // points + connecting line
  if (!calib.length) return;
  const maxN = Math.max(...calib.map(b => b.n));
  ctx.strokeStyle = "#58a6ff"; ctx.beginPath();
  calib.forEach((b, i) => { const X = xAt(b.pred), Y = yAt(b.realized); i ? ctx.lineTo(X, Y) : ctx.moveTo(X, Y); });
  ctx.stroke();
  calib.forEach(b => {
    const X = xAt(b.pred), Y = yAt(b.realized);
    const r = 2 + 5 * Math.sqrt(b.n / maxN);
    ctx.fillStyle = "#58a6ff"; ctx.beginPath(); ctx.arc(X, Y, r, 0, 7); ctx.fill();
  });
}

// ---- render loop -------------------------------------------------------------
function render() {
  if (activeTab === "market") {
    lineChart("chPrice", [
      { data: buf.price, color: "#e6edf3" },
      { data: buf.xc, color: "#f0883e" },
    ], { dec: 2 });
    lineChart("chProb", [{ data: buf.p, color: "#58a6ff" }], { min: 0, max: 1, ref: 0.5, dec: 2 });
    lineChart("chEquity", [{ data: buf.equity, color: "#3fb950" }], { dec: 0 });
  } else {
    lineChart("chParams", [
      { data: buf.alpha, color: "#58a6ff" },
      { data: buf.beta, color: "#3fb950" },
      { data: buf.gv, color: "#d29922" },
      { data: buf.ga, color: "#f85149" },
    ], { ref: 0, dec: 2 });
    lineChart("chLoss", [
      { data: buf.rollLoss, color: "#f0883e" },
      { data: buf.rollBrier, color: "#58a6ff" },
    ], { dec: 3 });
    calibChart("chCalib");
  }
  requestAnimationFrame(render);
}

connect();
requestAnimationFrame(render);
