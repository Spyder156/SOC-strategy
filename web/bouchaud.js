/* SOC Fragility (Bouchaud) dashboard. */
const C = { price:"#e6edf3", accent:"#58a6ff", green:"#3fb950", red:"#f85149",
            xc:"#f0883e", purple:"#bc8cff", grid:"#283040", muted:"#8b949e" };
const MAXPTS = 1200;
let SYMS = [], bufs = {}, eqStrat = [], eqBh = [], hist = null;
const $ = id => document.getElementById(id);
const push = (a, v) => { a.push(v); if (a.length > MAXPTS) a.shift(); };
const money = v => v == null ? "—" : "$" + Math.round(v).toLocaleString();

function connect() {
  const ws = new WebSocket(`ws://${location.host}/stream`);
  ws.onopen = () => { $("conn").classList.add("on"); const t=$("connText"); t.textContent="live"; t.style.color="var(--green)"; };
  ws.onclose = () => { $("conn").classList.remove("on"); const t=$("connText"); t.textContent="reconnecting…"; t.style.color="var(--red)"; setTimeout(connect,1500); };
  ws.onmessage = e => handle(JSON.parse(e.data));
}
function handle(m) {
  if (m.type === "config" && m.mode === "fragility") build(m.symbols);
  else if (m.type === "frag") update(m);
}
function build(symbols) {
  SYMS = symbols; bufs = {}; eqStrat = []; eqBh = []; hist = null;
  symbols.forEach(s => bufs[s] = { price: [], plarge: [], n: [], av: [] });
  const g = $("cards"); g.innerHTML = "";
  symbols.forEach(s => {
    const c = document.createElement("div"); c.className = "scard";
    c.innerHTML = `
      <div class="sc-head"><span class="sc-sym">${s}</span><span class="sc-ret" id="n-${s}">n —</span></div>
      <canvas class="sc-spark" id="px-${s}"></canvas>
      <div class="pbar"><div class="pbar-fill" id="pf-${s}"></div><div class="pbar-mid"></div><span class="pbar-txt" id="pt-${s}">—</span></div>
      <div class="phys">
        <div class="pcell"><div class="plab">P(big move) + avalanches</div><canvas id="ph1-${s}"></canvas></div>
        <div class="pcell"><div class="plab">branching ratio n (→1 critical)</div><canvas id="ph2-${s}"></canvas></div>
      </div>`;
    g.appendChild(c);
  });
}
function update(m) {
  const ct=$("connText"); if(ct.textContent!=="live"){ct.textContent="live";ct.style.color="var(--green)";}
  const f=(v,d=2)=>v==null?"—":(+v).toFixed(d);
  const sh=$("cSharpe"); sh.textContent=f(m.sharpe); sh.style.color=m.sharpe>0.05?C.green:m.sharpe<-0.05?C.red:C.muted;
  $("cBh").textContent=f(m.bh_sharpe);
  $("cExp").textContent=(m.exposure*100).toFixed(0)+"%";
  $("cAv").textContent=(m.n_aval||0).toLocaleString();
  $("curEq").textContent="strat "+money(m.equity)+"  ·  b&h "+money(m.bh_equity);
  push(eqStrat, m.equity); push(eqBh, m.bh_equity); hist = m.size_hist;
  for (const s of SYMS) {
    const st=m.stocks[s]; if(!st) continue;
    push(bufs[s].price, st.price); push(bufs[s].plarge, st.p_large); push(bufs[s].n, st.n); push(bufs[s].av, st.is_av);
    $("n-"+s).textContent="n "+st.n.toFixed(2); $("n-"+s).style.color = st.n>0.5?C.red:st.n>0.2?C.xc:C.muted;
    const pf=$("pf-"+s); pf.style.width=(st.p_large*100).toFixed(0)+"%"; pf.style.background=st.p_large>0.12?C.red:C.green; pf.style.opacity=0.5;
    $("pt-"+s).textContent=`P(big) ${st.p_large.toFixed(3)} · σ ${(st.sigma*1e4).toFixed(0)}bp`;
  }
}
// ---- canvas ----
function ctxOf(c){const dpr=window.devicePixelRatio||1,w=c.clientWidth,h=c.clientHeight;
  if(c.width!==w*dpr||c.height!==h*dpr){c.width=w*dpr;c.height=h*dpr;}
  const x=c.getContext("2d");x.setTransform(dpr,0,0,dpr,0,0);x.clearRect(0,0,w,h);return{x,w,h};}
function line(id, series, opts={}){const c=$(id);if(!c)return;const{x,w,h}=ctxOf(c);
  const P={l:opts.lab?28:4,r:4,t:4,b:4};let lo=opts.min,hi=opts.max;
  if(lo==null){lo=Infinity;hi=-Infinity;series.forEach(s=>s.data.forEach(v=>{if(v!=null&&!isNaN(v)){if(v<lo)lo=v;if(v>hi)hi=v;}}));if(!isFinite(lo)){return;}if(lo===hi){lo-=1;hi+=1;}const m=(hi-lo)*.1;lo-=m;hi+=m;}
  const n=Math.max(...series.map(s=>s.data.length),1);
  const X=i=>P.l+(w-P.l-P.r)*(n<=1?1:i/(n-1)),Y=v=>h-P.b-(h-P.t-P.b)*(v-lo)/(hi-lo);
  if(opts.ref!=null){x.strokeStyle="#3a4250";x.setLineDash([3,3]);x.beginPath();x.moveTo(P.l,Y(opts.ref));x.lineTo(w-P.r,Y(opts.ref));x.stroke();x.setLineDash([]);}
  series.forEach(s=>{x.strokeStyle=s.color;x.lineWidth=1.3;x.beginPath();let st=false;
    s.data.forEach((v,i)=>{if(v==null||isNaN(v))return;const xx=X(i),yy=Y(v);st?x.lineTo(xx,yy):(x.moveTo(xx,yy),st=true);});x.stroke();});
}
function drawHazard(id, ps, av){const c=$(id);if(!c||!ps.length)return;const{x,w,h}=ctxOf(c);
  const N=Math.min(ps.length,200),p=ps.slice(-N),a=av.slice(-N);
  const X=i=>2+(w-4)*(N<=1?1:i/(N-1)),Y=v=>h-8-(h-12)*Math.min(1,v*4);  // p_large small -> scale x4
  x.strokeStyle=C.accent;x.lineWidth=1.2;x.beginPath();p.forEach((v,i)=>i?x.lineTo(X(i),Y(v)):x.moveTo(X(i),Y(v)));x.stroke();
  x.strokeStyle=C.red;x.lineWidth=1;x.beginPath();a.forEach((v,i)=>{if(v===1){x.moveTo(X(i),h-6);x.lineTo(X(i),h-1);}});x.stroke();
}
function drawPower(){const c=$("chPow");if(!c||!hist)return;const{x,w,h}=ctxOf(c);
  const P={l:30,r:8,t:8,b:18};const ctr=hist.centers,cnt=hist.counts;
  const pts=[];for(let i=0;i<ctr.length;i++)if(cnt[i]>0)pts.push([Math.log(ctr[i]),Math.log(cnt[i])]);
  if(pts.length<2)return;
  const xs=pts.map(p=>p[0]),ys=pts.map(p=>p[1]);
  const xl=Math.min(...xs),xh=Math.max(...xs),yl=Math.min(...ys),yh=Math.max(...ys)+0.3;
  const X=v=>P.l+(w-P.l-P.r)*(v-xl)/(xh-xl||1),Y=v=>h-P.b-(h-P.t-P.b)*(v-yl)/(yh-yl||1);
  x.strokeStyle=C.grid;x.strokeRect(P.l,P.t,w-P.l-P.r,h-P.t-P.b);
  x.fillStyle=C.muted;x.font="9px monospace";x.fillText("log size →",w/2-22,h-4);
  // points + line (power law => straight)
  x.strokeStyle=C.xc;x.lineWidth=1.3;x.beginPath();pts.forEach((p,i)=>i?x.lineTo(X(p[0]),Y(p[1])):x.moveTo(X(p[0]),Y(p[1])));x.stroke();
  x.fillStyle=C.xc;pts.forEach(p=>{x.beginPath();x.arc(X(p[0]),Y(p[1]),2.5,0,7);x.fill();});
  // mark the largest-size bin (Dragon-King watch)
  const last=pts[pts.length-1];x.fillStyle=C.red;x.beginPath();x.arc(X(last[0]),Y(last[1]),3.5,0,7);x.fill();
}
function render(){
  for(const s of SYMS){
    line("px-"+s,[{data:bufs[s].price,color:C.price}]);
    drawHazard("ph1-"+s,bufs[s].plarge,bufs[s].av);
    line("ph2-"+s,[{data:bufs[s].n,color:C.purple}],{min:0,max:1,ref:1});
  }
  line("chEq",[{data:eqStrat,color:C.green},{data:eqBh,color:C.muted}]);
  drawPower();
  requestAnimationFrame(render);
}
connect();
requestAnimationFrame(render);
