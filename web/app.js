/* SOC Sandpile live dashboard — framework-free, canvas charts with synced crosshair. */

const C = { price:"#e6edf3", xc:"#f0883e", accent:"#58a6ff", green:"#3fb950",
            yellow:"#d29922", red:"#f85149", purple:"#bc8cff", grid:"#283040", muted:"#8b949e" };
const MAXPTS = 1800;
const PAD = { l: 46, r: 58, t: 8, b: 18 };

const buf = { ts:[], price:[], xc:[], xbar:[], p:[], equity:[], exposure:[], gap:[], y:[],
              alpha:[], beta:[], gv:[], ga:[], rollLoss:[], rollBrier:[], rollWin:[] };
let calib = [], lastMetric = null, lastView = null;
let paused = false, hoverFrac = null, hoverCanvas = null, activeTab = "market";

const VIEW_CHARTS = { market:["chPrice","chProb","chEquity","chExposure","chGap"],
                      model:["chParams","chLoss","chWin","chCalib"] };
const ALL_CHARTS = ["chPrice","chProb","chEquity","chExposure","chGap","chParams","chLoss","chWin","chCalib"];

const push = (a, v) => { a.push(v); if (a.length > MAXPTS) a.shift(); };
const $ = id => document.getElementById(id);

// ---- formatting --------------------------------------------------------------
const fmt = (v, d = 2) => (v == null || isNaN(v)) ? "—" : (+v).toFixed(d);
function money(v) {
  if (v == null || isNaN(v)) return "—";
  const s = v < 0 ? "-" : "", a = Math.abs(v);
  if (a >= 1e6) return s + "$" + (a/1e6).toFixed(2) + "M";
  if (a >= 1e3) return s + "$" + (a/1e3).toFixed(1) + "k";
  return s + "$" + a.toFixed(0);
}
function timeLabel(ts) {
  if (ts == null) return "";
  if (ts > 1e6) { const d = new Date(ts*1000);
    return d.toLocaleTimeString([], {hour:"2-digit",minute:"2-digit",second:"2-digit"}); }
  return "t" + Math.round(ts);
}

// ---- websocket ---------------------------------------------------------------
function connect() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/stream`);
  ws.onopen = () => { $("conn").classList.add("on"); const t=$("connText"); t.textContent="live"; t.style.color="var(--green)"; };
  ws.onclose = () => { $("conn").classList.remove("on"); const t=$("connText"); t.textContent="reconnecting…"; t.style.color="var(--red)"; setTimeout(connect, 1500); };
  ws.onmessage = e => handle(JSON.parse(e.data));
}

function handle(m) {
  if (m.type === "config") {
    $("symbol").textContent = m.symbol; $("feedtag").textContent = `feed: ${m.feed}`;
    return;
  }
  if (m.type === "status") {
    if (m.phase === "warmup") {
      $("story").textContent = `Warming up — training the model… ${m.seen.toLocaleString()} / ${m.total.toLocaleString()} bars`;
      const t = $("connText"); t.textContent = "warming up"; t.style.color = "var(--yellow)";
    }
    return;
  }
  if (paused) return;
  if (m.type === "view") {
    lastView = m;
    const ct = $("connText"); if (ct.textContent !== "live") { ct.textContent = "live"; ct.style.color = "var(--green)"; }
    push(buf.ts, m.ts); push(buf.price, m.x); push(buf.xc, m.x_c); push(buf.xbar, m.x_bar); push(buf.p, m.p);
    push(buf.equity, m.equity); push(buf.exposure, m.exposure); push(buf.gap, m.gap); push(buf.y, m.y);
    push(buf.alpha, m.params.alpha); push(buf.beta, m.params.beta);
    push(buf.gv, m.params.gamma_v); push(buf.ga, m.params.gamma_a);
    updateMarket(m);
  } else if (m.type === "metric") {
    lastMetric = m; calib = m.calibration || [];
    if (m.roll_logloss != null) push(buf.rollLoss, m.roll_logloss);
    if (m.roll_brier != null) push(buf.rollBrier, m.roll_brier);
    if (m.roll_winrate != null) push(buf.rollWin, m.roll_winrate);
    updateChips(m); updateModel(m);
  }
}

// ---- DOM readouts ------------------------------------------------------------
function updateChips(m) {
  if (lastView) {
    const r = lastView.return_pct;
    const el = $("cReturn"); el.textContent = (r>=0?"+":"") + (r||0).toFixed(2) + "%";
    el.style.color = r>0?C.green:r<0?C.red:C.muted;
    $("cEquity").textContent = money(lastView.equity);
  }
  $("cWin").textContent = fmt(m.winrate*100,1)+"%";
  $("cBrier").textContent = fmt(m.brier,3);
  $("cLoss").textContent = fmt(m.logloss,3);
  $("cN").textContent = (m.n||0).toLocaleString();
  const s = $("story");
  if (m.n < 3000) s.textContent = "Cold start — model still calibrating; convergence vars unsettled.";
  else if (m.brier < m.base_rate*(1-m.base_rate)) s.textContent = "Discriminating — Brier below base. Check calibration plot for honesty.";
  else s.textContent = "Calibrated but near base rate — little directional edge (expected on tick data).";
}

function updateMarket(m) {
  const exp = m.exposure||0, ps = $("posState");
  if (exp>1){ps.textContent="LONG";ps.className="pos";} else if(exp<-1){ps.textContent="SHORT";ps.className="neg";} else {ps.textContent="flat";ps.className="flat";}
  const ly=$("lastY"); ly.textContent=m.y===1?"▼ down":"▲ up"; ly.className=m.y===1?"neg":"pos";
  const rr=$("rdReturn"); rr.textContent=(m.return_pct>=0?"+":"")+(m.return_pct||0).toFixed(2)+"%"; rr.className=m.return_pct>=0?"pos":"neg";
  $("rdCost").textContent = money(m.total_cost ?? (lastMetric? null:0));
  $("curPrice").textContent = `x=${fmt(m.x,2)}  x_c=${fmt(m.x_c,2)}  gap=${fmt(m.gap,3)}`;
  $("curP").textContent = fmt(m.p,3);
  $("curEq").textContent = money(m.equity);
  $("curExp").textContent = money(m.exposure);
  $("curGap").textContent = fmt(m.gap,3);
  $("curParams").textContent = `α=${fmt(m.params.alpha,2)} β=${fmt(m.params.beta,2)} γv=${fmt(m.params.gamma_v,2)} γa=${fmt(m.params.gamma_a,2)}`;
}

function updateModel(m) {
  $("mWin").textContent = `${fmt(m.winrate*100,1)}% / ${fmt(m.roll_winrate*100,1)}%`;
  $("mBase").textContent = fmt(m.base_rate*100,1)+"%";
  $("mLB").textContent = `${fmt(m.logloss,4)} / ${fmt(m.brier,4)}`;
  const last=a=>a.length?a[a.length-1]:null;
  $("mAB").textContent = `${fmt(last(buf.alpha),3)} / ${fmt(last(buf.beta),3)}`;
  $("mG").textContent = `${fmt(last(buf.gv),3)} / ${fmt(last(buf.ga),3)}`;
  $("curParams").textContent = `α=${fmt(last(buf.alpha),2)} β=${fmt(last(buf.beta),2)} γv=${fmt(last(buf.gv),2)} γa=${fmt(last(buf.ga),2)}`;
  $("curLoss").textContent = `ll=${fmt(last(buf.rollLoss),3)} br=${fmt(last(buf.rollBrier),3)}`;
  $("curWin").textContent = fmt(last(buf.rollWin)*100,1)+"%";
}

// ---- chart specs (single source of truth for draw + tooltip) -----------------
function specFor(id) {
  switch (id) {
    case "chPrice": return { series:[{label:"price",color:C.price,data:buf.price},{label:"x_c",color:C.xc,data:buf.xc},{label:"x̄ base",color:C.muted,data:buf.xbar}], fill:[0,1,"#f0883e22"], dec:2 };
    case "chProb": return { series:[{label:"P(down)",color:C.accent,data:buf.p}], fixed:[0,1], ref:0.5, dec:2 };
    case "chEquity": return { series:[{label:"equity",color:C.green,data:buf.equity}], money:true, dec:0 };
    case "chExposure": return { series:[{label:"exposure",color:C.yellow,data:buf.exposure}], ref:0, money:true, dec:0 };
    case "chGap": return { series:[{label:"gap",color:C.purple,data:buf.gap}], dec:3 };
    case "chParams": return { series:[{label:"α",color:C.accent,data:buf.alpha},{label:"β",color:C.green,data:buf.beta},{label:"γ_v",color:C.yellow,data:buf.gv},{label:"γ_a",color:C.red,data:buf.ga}], ref:0, dec:2 };
    case "chLoss": return { series:[{label:"logloss",color:C.xc,data:buf.rollLoss},{label:"brier",color:C.accent,data:buf.rollBrier}], dec:3 };
    case "chWin": return { series:[{label:"win",color:C.purple,data:buf.rollWin}], fixed:[0,1], ref:0.5, dec:3 };
    default: return null;
  }
}

// ---- canvas helpers ----------------------------------------------------------
function ctxOf(c) {
  const dpr = window.devicePixelRatio || 1, w = c.clientWidth, h = c.clientHeight;
  if (c.width !== w*dpr || c.height !== h*dpr) { c.width = w*dpr; c.height = h*dpr; }
  const ctx = c.getContext("2d"); ctx.setTransform(dpr,0,0,dpr,0,0); ctx.clearRect(0,0,w,h);
  return { ctx, w, h };
}
const valAt = (data, frac) => { if (!data.length) return null; return data[Math.round(frac*(data.length-1))]; };

function drawLine(id) {
  const c = $(id), spec = specFor(id); if (!c || !spec) return;
  const { ctx, w, h } = ctxOf(c);
  const X0=PAD.l, X1=w-PAD.r, Y0=PAD.t, Y1=h-PAD.b;
  let lo, hi;
  if (spec.fixed) { [lo,hi]=spec.fixed; }
  else {
    lo=Infinity; hi=-Infinity;
    spec.series.forEach(s=>s.data.forEach(v=>{ if(v!=null&&!isNaN(v)){ if(v<lo)lo=v; if(v>hi)hi=v; }}));
    if (!isFinite(lo)){lo=0;hi=1;} if (lo===hi){lo-=1;hi+=1;}
    const pad=(hi-lo)*0.1; lo-=pad; hi+=pad;
  }
  const maxLen = Math.max(...spec.series.map(s=>s.data.length),1);
  const xAt=(i,len)=>X0+(X1-X0)*(len<=1?1:i/(len-1));
  const yAt=v=>Y1-(Y1-Y0)*(v-lo)/(hi-lo);
  const yfmt=v=> spec.money?money(v):(+v).toFixed(spec.dec??2);

  // grid + y labels
  ctx.font="9px ui-monospace,monospace"; ctx.textBaseline="middle";
  for (let g=0; g<=4; g++){ const yy=Y0+(Y1-Y0)*g/4, val=hi-(hi-lo)*g/4;
    ctx.strokeStyle=C.grid; ctx.lineWidth=1; ctx.beginPath(); ctx.moveTo(X0,yy); ctx.lineTo(X1,yy); ctx.stroke();
    ctx.fillStyle=C.muted; ctx.textAlign="right"; ctx.fillText(yfmt(val), X0-3, yy); }
  // x time labels
  ctx.textAlign="center"; ctx.textBaseline="alphabetic";
  for (let g=0; g<=3; g++){ const fr=g/3, xx=X0+(X1-X0)*fr;
    ctx.fillStyle=C.muted; ctx.fillText(timeLabel(valAt(buf.ts,fr)), xx, h-5); }
  // ref line
  if (spec.ref!=null && spec.ref>=lo && spec.ref<=hi){ ctx.strokeStyle="#3a4250"; ctx.setLineDash([4,4]);
    const yy=yAt(spec.ref); ctx.beginPath(); ctx.moveTo(X0,yy); ctx.lineTo(X1,yy); ctx.stroke(); ctx.setLineDash([]); }
  // fill between two series
  if (spec.fill){ const [ai,bi,col]=spec.fill, A=spec.series[ai].data, B=spec.series[bi].data, n=Math.min(A.length,B.length);
    if(n>1){ ctx.fillStyle=col; ctx.beginPath();
      for(let i=0;i<n;i++) ctx.lineTo(xAt(i,n), yAt(A[i]));
      for(let i=n-1;i>=0;i--) ctx.lineTo(xAt(i,n), yAt(B[i]));
      ctx.closePath(); ctx.fill(); } }
  // series
  spec.series.forEach(s=>{ ctx.strokeStyle=s.color; ctx.lineWidth=1.4; ctx.beginPath(); let st=false;
    s.data.forEach((v,i)=>{ if(v==null||isNaN(v))return; const X=xAt(i,s.data.length),Y=yAt(v);
      st?ctx.lineTo(X,Y):(ctx.moveTo(X,Y),st=true); }); ctx.stroke(); });
  // last-value tags
  spec.series.forEach(s=>{ if(!s.data.length)return; const v=s.data[s.data.length-1]; if(v==null||isNaN(v))return;
    const Y=Math.max(Y0+6,Math.min(Y1-1,yAt(v))), txt=yfmt(v); ctx.font="9px ui-monospace,monospace";
    const tw=ctx.measureText(txt).width+6; ctx.fillStyle=s.color; ctx.fillRect(X1+2,Y-7,tw,14);
    ctx.fillStyle="#0d1117"; ctx.textAlign="left"; ctx.textBaseline="middle"; ctx.fillText(txt,X1+5,Y); });
  // crosshair (synced across all charts via global hoverFrac)
  if (hoverFrac!=null){ const X=X0+(X1-X0)*hoverFrac; ctx.strokeStyle="#5a6577"; ctx.lineWidth=1;
    ctx.beginPath(); ctx.moveTo(X,Y0); ctx.lineTo(X,Y1); ctx.stroke();
    spec.series.forEach(s=>{ const v=valAt(s.data,hoverFrac); if(v==null||isNaN(v))return;
      ctx.fillStyle=s.color; ctx.beginPath(); ctx.arc(X,yAt(v),3,0,7); ctx.fill(); }); }
}

function drawCalib() {
  const c=$("chCalib"); if(!c) return; const { ctx, w, h }=ctxOf(c);
  const p=28, X0=p, X1=w-12, Y0=10, Y1=h-p;
  const xAt=v=>X0+(X1-X0)*v, yAt=v=>Y1-(Y1-Y0)*v;
  ctx.font="9px ui-monospace,monospace";
  // grid
  ctx.strokeStyle=C.grid; ctx.lineWidth=1; ctx.strokeRect(X0,Y0,X1-X0,Y1-Y0);
  for(let g=1;g<5;g++){ const gx=X0+(X1-X0)*g/5, gy=Y0+(Y1-Y0)*g/5;
    ctx.beginPath();ctx.moveTo(gx,Y0);ctx.lineTo(gx,Y1);ctx.moveTo(X0,gy);ctx.lineTo(X1,gy);ctx.stroke(); }
  // diagonal = perfect calibration
  ctx.strokeStyle="#5a6577"; ctx.setLineDash([4,4]); ctx.beginPath(); ctx.moveTo(X0,Y1); ctx.lineTo(X1,Y0); ctx.stroke(); ctx.setLineDash([]);
  ctx.fillStyle=C.muted; ctx.textAlign="center"; ctx.fillText("predicted P", (X0+X1)/2, h-4);
  ctx.save(); ctx.translate(9,(Y0+Y1)/2); ctx.rotate(-Math.PI/2); ctx.fillText("realised freq",0,0); ctx.restore();
  if(!calib.length) return;
  const maxN=Math.max(...calib.map(b=>b.n));
  ctx.strokeStyle=C.accent; ctx.lineWidth=1.5; ctx.beginPath();
  calib.forEach((b,i)=>{ const X=xAt(b.pred),Y=yAt(b.realized); i?ctx.lineTo(X,Y):ctx.moveTo(X,Y); }); ctx.stroke();
  calib.forEach(b=>{ const X=xAt(b.pred),Y=yAt(b.realized), r=2+5*Math.sqrt(b.n/maxN);
    ctx.fillStyle=C.accent; ctx.beginPath(); ctx.arc(X,Y,r,0,7); ctx.fill(); });
}

// ---- tooltip + hover ---------------------------------------------------------
const tip = $("tip");
function showTip(id, e) {
  if (id === "chCalib") { tip.style.display="none"; return; }
  const spec = specFor(id); if (!spec) return;
  const rows = spec.series.map(s=>{ const v=valAt(s.data,hoverFrac);
    return `<div class="row"><span style="color:${s.color}">${s.label}</span><span>${spec.money?money(v):fmt(v,spec.dec??2)}</span></div>`; }).join("");
  tip.innerHTML = `<div class="tt">${timeLabel(valAt(buf.ts,hoverFrac))}</div>${rows}`;
  tip.style.display="block";
  let x=e.clientX+14, y=e.clientY+14;
  if (x+tip.offsetWidth>window.innerWidth) x=e.clientX-tip.offsetWidth-14;
  if (y+tip.offsetHeight>window.innerHeight) y=e.clientY-tip.offsetHeight-14;
  tip.style.left=x+"px"; tip.style.top=y+"px";
}
function attachHover() {
  ALL_CHARTS.forEach(id=>{ const c=$(id); if(!c) return;
    c.addEventListener("mousemove", e=>{ const r=c.getBoundingClientRect();
      const fr=(e.clientX-r.left-PAD.l)/(r.width-PAD.l-PAD.r);
      hoverFrac=Math.max(0,Math.min(1,fr)); hoverCanvas=id; showTip(id,e); });
    c.addEventListener("mouseleave", ()=>{ hoverFrac=null; hoverCanvas=null; tip.style.display="none"; }); });
}

// ---- tabs + pause ------------------------------------------------------------
document.querySelectorAll(".tab").forEach(t=>t.onclick=()=>{
  document.querySelectorAll(".tab").forEach(x=>x.classList.remove("active"));
  document.querySelectorAll(".view").forEach(x=>x.classList.remove("active"));
  t.classList.add("active"); activeTab=t.dataset.tab; $(activeTab).classList.add("active");
});
$("pauseBtn").onclick=()=>{ paused=!paused; const b=$("pauseBtn");
  b.textContent=paused?"▶ Resume":"⏸ Pause"; b.classList.toggle("paused",paused); };
document.addEventListener("keydown", e=>{ if(e.code==="Space"){ e.preventDefault(); $("pauseBtn").click(); } });

// ---- render loop -------------------------------------------------------------
function render() {
  VIEW_CHARTS[activeTab].forEach(id => id==="chCalib" ? drawCalib() : drawLine(id));
  requestAnimationFrame(render);
}

attachHover();
connect();
requestAnimationFrame(render);
