/* SOC Universe — multi-stock live monitor. */

const C = { price:"#e6edf3", xc:"#f0883e", accent:"#58a6ff", green:"#3fb950",
            red:"#f85149", yellow:"#d29922", purple:"#bc8cff", grid:"#283040", muted:"#8b949e", panel:"#161b22" };
const MAXPTS = 600;
let SYMS = [], bufs = {}, portBuf = [], bars = 0, lastUni = null;

const $ = id => document.getElementById(id);
const money = v => v == null ? "—" : (v < 0 ? "-$" : "$") + Math.abs(v).toLocaleString(undefined, {maximumFractionDigits: 0});
const push = (a, v) => { a.push(v); if (a.length > MAXPTS) a.shift(); };

// ---- websocket ----
function connect() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/stream`);
  ws.onopen = () => { $("conn").classList.add("on"); const t=$("connText"); t.textContent="live"; t.style.color="var(--green)"; };
  ws.onclose = () => { $("conn").classList.remove("on"); const t=$("connText"); t.textContent="reconnecting…"; t.style.color="var(--red)"; setTimeout(connect, 1500); };
  ws.onmessage = e => handle(JSON.parse(e.data));
}

function handle(m) {
  if (m.type === "config" && m.mode === "multi") { build(m.symbols); }
  else if (m.type === "uni") { update(m); }
  else if (m.type === "status" && m.phase === "warmup") {
    $("story").textContent = `Warming up — training the universe… ${m.seen.toLocaleString()} / ${m.total.toLocaleString()} bars`;
    const t = $("connText"); t.textContent = "warming up"; t.style.color = "var(--yellow)";
  }
}

// ---- build cards + heatmap ----
function build(symbols) {
  SYMS = symbols; bufs = {}; bars = 0; portBuf = [];
  symbols.forEach(s => bufs[s] = { price: [], xc: [], xcnc: [], p: [], y: [], gaprel: [],
                                   alpha: [], beta: [], gv: [], ga: [], mu: [] });
  $("cN").textContent = symbols.length;

  const grid = $("cards"); grid.innerHTML = "";
  symbols.forEach(s => {
    const c = document.createElement("div"); c.className = "scard"; c.id = "card-" + s;
    c.innerHTML = `
      <div class="sc-head"><span class="sc-sym">${s}</span><span class="sc-ret" id="wt-${s}" title="weight = share of budget">—</span></div>
      <canvas class="sc-spark" id="spark-${s}"></canvas>
      <div class="sc-row"><span class="sc-px" id="px-${s}">—</span><span class="sc-pos" id="pos-${s}">flat</span></div>
      <div class="pbar"><div class="pbar-fill" id="pf-${s}"></div><div class="pbar-mid"></div><span class="pbar-txt" id="pt-${s}">—</span></div>
      <div class="sc-meta"><span id="gap-${s}">gap —</span><span id="win-${s}">win —</span></div>
      <div class="phys">
        <div class="pcell"><div class="plab">P(aval) + avalanches</div><canvas id="ph1-${s}"></canvas></div>
        <div class="pcell"><div class="plab">loaded ←gap→ slack</div><canvas id="ph2-${s}"></canvas></div>
        <div class="pcell"><div class="plab">calibration</div><canvas id="ph3-${s}"></canvas></div>
      </div>`;
    grid.appendChild(c);
  });

  const heat = $("heat");
  heat.style.gridTemplateColumns = `40px repeat(${symbols.length}, 1fr)`;
  heat.innerHTML = "";
  heat.appendChild(cell("", "lab"));
  symbols.forEach(s => heat.appendChild(cell(s, "lab")));
  symbols.forEach((ri, i) => {
    heat.appendChild(cell(ri, "lab"));
    symbols.forEach((cj, j) => { const e = cell("", ""); e.id = `h-${i}-${j}`; heat.appendChild(e); });
  });

  const mgrid = $("mcards"); mgrid.innerHTML = "";
  symbols.forEach(s => {
    const c = document.createElement("div"); c.className = "scard";
    c.innerHTML = `
      <div class="sc-head"><span class="sc-sym">${s}</span><span class="sc-meta" id="mn-${s}">n=0</span></div>
      <canvas class="mspark" id="mspark-${s}"></canvas>
      <div class="prow"><span>α <b id="ma-${s}">—</b></span><span>β <b id="mb-${s}">—</b></span>
        <span>γ_v <b id="mgv-${s}">—</b></span><span>γ_a <b id="mga-${s}">—</b></span><span>μ <b id="mmu-${s}">—</b></span></div>
      <div class="prow"><span>win <b id="mw-${s}">—</b></span><span>brier <b id="mbr-${s}">—</b></span><span>logloss <b id="ml-${s}">—</b></span></div>`;
    mgrid.appendChild(c);
  });
}
function cell(txt, cls) { const d = document.createElement("div"); d.className = "cell " + cls; d.textContent = txt; return d; }

// ---- update ----
function update(m) {
  lastUni = m; bars++;
  const ct = $("connText");
  if (m.training) { ct.textContent = "training"; ct.style.color = "var(--yellow)"; }
  else if (ct.textContent !== "live") { ct.textContent = "live"; ct.style.color = "var(--green)"; }
  const sh = $("cSharpe");
  sh.textContent = (m.sharpe != null) ? m.sharpe.toFixed(2) : "—";
  sh.style.color = m.sharpe > 0.05 ? C.green : m.sharpe < -0.05 ? C.red : C.muted;
  const r = m.return_pct;
  const rel = $("cReturn"); rel.textContent = (r>=0?"+":"") + (r||0).toFixed(2) + "%";
  rel.style.color = r>0?C.green:r<0?C.red:C.muted;
  $("cEquity").textContent = money(m.equity);
  $("cBars").textContent = bars.toLocaleString();
  push(portBuf, m.equity);
  $("curPort").textContent = money(m.equity);

  let learning = 0;
  for (const s of SYMS) {
    const st = m.stocks[s]; if (!st) continue;
    push(bufs[s].price, st.x); push(bufs[s].xc, st.x_c);
    if (st.x_c_no_corr != null) push(bufs[s].xcnc, st.x_c_no_corr);
    push(bufs[s].p, st.p); push(bufs[s].y, st.y); push(bufs[s].gaprel, 100 * st.gap / st.x);
    $("px-" + s).textContent = "$" + st.x.toFixed(2);
    const w = st.weight || 0, wl = $("wt-" + s);          // share of the universe budget
    wl.textContent = (w >= 0 ? "+" : "") + (w * 100).toFixed(0) + "%";
    wl.style.color = w > 0.01 ? C.green : w < -0.01 ? C.red : C.muted;
    const pos = $("pos-" + s), e = st.exposure||0;
    if (e>1){pos.textContent="LONG";pos.style.color=C.green;} else if(e<-1){pos.textContent="SHORT";pos.style.color=C.red;} else {pos.textContent="flat";pos.style.color=C.muted;}
    const pf = $("pf-" + s); pf.style.width = (st.p*100).toFixed(0) + "%";
    pf.style.background = st.p>0.5 ? C.red : C.green; pf.style.opacity = 0.45;
    $("pt-" + s).textContent = `P↓ ${st.p.toFixed(3)}`;
    $("gap-" + s).textContent = "gap " + st.gap.toFixed(2);
    $("win-" + s).textContent = "win " + (st.winrate*100).toFixed(1) + "%";
    if (st.brier < 0.25) learning++;

    // model page: convergence params + metrics
    const pr = st.params;
    push(bufs[s].alpha, pr.alpha); push(bufs[s].beta, pr.beta); push(bufs[s].gv, pr.gamma_v);
    push(bufs[s].ga, pr.gamma_a); push(bufs[s].mu, pr.mu);
    $("ma-" + s).textContent = pr.alpha.toFixed(3); $("mb-" + s).textContent = pr.beta.toFixed(3);
    $("mgv-" + s).textContent = pr.gamma_v.toFixed(3); $("mga-" + s).textContent = pr.gamma_a.toFixed(3);
    $("mmu-" + s).textContent = (pr.mu * 100).toFixed(2) + "%";
    $("mn-" + s).textContent = "n=" + (st.n || 0).toLocaleString();
    $("mw-" + s).textContent = (st.winrate * 100).toFixed(1) + "%";
    $("mbr-" + s).textContent = st.brier.toFixed(4);
    $("ml-" + s).textContent = (st.logloss != null) ? st.logloss.toFixed(4) : "—";
  }

  const mat = m.couple || m.corr;   // learned coupling matrix W_ij (peer j -> stock i)
  if (mat) for (let i=0;i<SYMS.length;i++) for (let j=0;j<SYMS.length;j++) {
    const v = mat[i][j], el = $(`h-${i}-${j}`); if (!el) continue;
    el.textContent = (i===j) ? "·" : v.toFixed(3);
    const t = Math.min(1, Math.abs(v) * 12);   // W values are small; scale up for visibility
    el.style.background = i===j ? "transparent" : (v>=0 ? `rgba(88,166,255,${t.toFixed(2)})` : `rgba(248,81,73,${t.toFixed(2)})`);
    el.style.color = t > 0.5 ? "#0d1117" : C.muted;
  }
  $("story").textContent = m.training
    ? `Warming up — training & watching x_c / params converge (NOT trading yet)… ${(m.seen||0).toLocaleString()} / ${(m.warmup||0).toLocaleString()} bars`
    : "Each card: price (white), x_c+coupling (orange), x_c if ALONE (grey). Bottom row: hazard+avalanches · the loading law · calibration.";
}

// ---- canvas sparklines ----
function ctxOf(c) {
  const dpr = window.devicePixelRatio||1, w=c.clientWidth, h=c.clientHeight;
  if (c.width!==w*dpr||c.height!==h*dpr){c.width=w*dpr;c.height=h*dpr;}
  const x=c.getContext("2d"); x.setTransform(dpr,0,0,dpr,0,0); x.clearRect(0,0,w,h); return {x,w,h};
}
function spark(canvas, series, pad) {
  const c=$(canvas); if(!c) return; const {x,w,h}=ctxOf(c);
  let lo=Infinity,hi=-Infinity;
  series.forEach(s=>s.data.forEach(v=>{if(v!=null&&!isNaN(v)){if(v<lo)lo=v;if(v>hi)hi=v;}}));
  if(!isFinite(lo)){return;} if(lo===hi){lo-=1;hi+=1;} const m=(hi-lo)*0.1; lo-=m; hi+=m;
  const n=Math.max(...series.map(s=>s.data.length),1);
  const X=i=>4+(w-8)*(n<=1?1:i/(n-1)), Y=v=>h-4-(h-8)*(v-lo)/(hi-lo);
  series.forEach(s=>{ x.strokeStyle=s.color; x.lineWidth=1.3; x.beginPath(); let st=false;
    s.data.forEach((v,i)=>{ if(v==null||isNaN(v))return; const xx=X(i),yy=Y(v); st?x.lineTo(xx,yy):(x.moveTo(xx,yy),st=true); }); x.stroke(); });
}
// ---- physics plots ----------------------------------------------------------
function drawHazard(id, ps, ys) {                  // P(avalanche) over time + avalanche events
  const c = $(id); if (!c || !ps.length) return; const { x: ctx, w, h } = ctxOf(c);
  const N = Math.min(ps.length, 200), p = ps.slice(-N), y = ys.slice(-N);
  const X = i => 2 + (w - 4) * (N <= 1 ? 1 : i / (N - 1)), Y = v => h - 8 - (h - 12) * v;
  ctx.strokeStyle = "#3a4250"; ctx.setLineDash([3, 3]); ctx.beginPath(); ctx.moveTo(2, Y(0.5)); ctx.lineTo(w - 2, Y(0.5)); ctx.stroke(); ctx.setLineDash([]);
  ctx.strokeStyle = C.accent; ctx.lineWidth = 1.2; ctx.beginPath();
  p.forEach((v, i) => i ? ctx.lineTo(X(i), Y(v)) : ctx.moveTo(X(i), Y(v))); ctx.stroke();
  ctx.strokeStyle = C.red; ctx.lineWidth = 1; ctx.beginPath();
  y.forEach((v, i) => { if (v === 1) { ctx.moveTo(X(i), h - 6); ctx.lineTo(X(i), h - 1); } }); ctx.stroke();
}
function binMean(xv, yv, nb, lo, hi) {
  const sy = Array(nb).fill(0), sx = Array(nb).fill(0), c = Array(nb).fill(0);
  for (let i = 0; i < xv.length; i++) {
    let b = Math.floor((xv[i] - lo) / (hi - lo) * nb); b = b < 0 ? 0 : b >= nb ? nb - 1 : b;
    sy[b] += yv[i]; sx[b] += xv[i]; c[b]++;
  }
  const out = []; for (let b = 0; b < nb; b++) if (c[b] > 3) out.push({ x: sx[b] / c[b], y: sy[b] / c[b] });
  return out;
}
function drawLoading(id, gaps, ys) {               // realised P(avalanche) vs how loaded (gap)
  const c = $(id); if (!c || gaps.length < 40) return; const { x: ctx, w, h } = ctxOf(c);
  const N = Math.min(gaps.length, 800), g = gaps.slice(-N), y = ys.slice(-N);
  let lo = Math.min(...g), hi = Math.max(...g); if (hi - lo < 1e-6) hi = lo + 1;
  const pts = binMean(g, y, 6, lo, hi);
  const X = v => 4 + (w - 8) * (v - lo) / (hi - lo), Y = v => h - 6 - (h - 10) * v;
  ctx.strokeStyle = "#3a4250"; ctx.setLineDash([3, 3]); ctx.beginPath(); ctx.moveTo(4, Y(0.5)); ctx.lineTo(w - 4, Y(0.5)); ctx.stroke(); ctx.setLineDash([]);
  ctx.strokeStyle = C.xc; ctx.fillStyle = C.xc; ctx.lineWidth = 1.3; ctx.beginPath();
  pts.forEach((pt, i) => i ? ctx.lineTo(X(pt.x), Y(pt.y)) : ctx.moveTo(X(pt.x), Y(pt.y))); ctx.stroke();
  pts.forEach(pt => { ctx.beginPath(); ctx.arc(X(pt.x), Y(pt.y), 2, 0, 7); ctx.fill(); });
}
function drawCalib(id, ps, ys) {                   // predicted P vs realised freq (diagonal = honest)
  const c = $(id); if (!c || ps.length < 40) return; const { x: ctx, w, h } = ctxOf(c);
  const N = Math.min(ps.length, 1500), p = ps.slice(-N), y = ys.slice(-N);
  const pts = binMean(p, y, 6, 0, 1);
  const X = v => 4 + (w - 8) * v, Y = v => h - 6 - (h - 10) * v;
  ctx.strokeStyle = "#3a4250"; ctx.setLineDash([3, 3]); ctx.beginPath(); ctx.moveTo(X(0), Y(0)); ctx.lineTo(X(1), Y(1)); ctx.stroke(); ctx.setLineDash([]);
  ctx.fillStyle = C.green;
  pts.forEach(pt => { ctx.beginPath(); ctx.arc(X(pt.x), Y(pt.y), 2.2, 0, 7); ctx.fill(); });
}

let view = "universe";
function render() {
  if (view === "universe") {
    for (const s of SYMS) {
      spark("spark-" + s, [
        {data: bufs[s].price, color: C.price},
        {data: bufs[s].xcnc, color: C.muted},      // x_c with NO cross-asset coupling
        {data: bufs[s].xc, color: C.xc}]);          // x_c WITH coupling (on top)
      drawHazard("ph1-" + s, bufs[s].p, bufs[s].y);
      drawLoading("ph2-" + s, bufs[s].gaprel, bufs[s].y);
      drawCalib("ph3-" + s, bufs[s].p, bufs[s].y);
    }
    spark("chPort", [{data: portBuf, color: C.green}]);
  } else {
    for (const s of SYMS) spark("mspark-" + s, [
      {data: bufs[s].alpha, color: C.accent}, {data: bufs[s].beta, color: C.green},
      {data: bufs[s].gv, color: C.yellow}, {data: bufs[s].ga, color: C.red},
      {data: bufs[s].mu, color: C.purple}]);
  }
  requestAnimationFrame(render);
}

document.querySelectorAll(".tab").forEach(t => t.onclick = () => {
  document.querySelectorAll(".tab").forEach(x => x.classList.remove("active"));
  document.querySelectorAll(".mview").forEach(x => x.classList.remove("active"));
  t.classList.add("active"); view = t.dataset.tab; $(view).classList.add("active");
});

connect();
requestAnimationFrame(render);
