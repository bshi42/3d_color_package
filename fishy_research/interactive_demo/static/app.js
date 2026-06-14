"use strict";
// =============================================================================
// Live force-directed color+pattern morphospace.
// Nodes = specimens; springs link each to its kNN in the descriptor space (a connected
// similarity graph). A continuous physics sim relaxes it; you can DRAG a node and the rest
// react, and ranking an anchor's neighbours edits "preference springs" (similar -> short,
// dissimilar -> long) so the whole graph re-settles. The dendrogram clusters the live layout.
// =============================================================================

// ---- physics constants (world units; neighbour spacing normalised to WORLD_SPACING) --------
const WORLD_SPACING = 62;     // target screen-world distance between kNN neighbours
const CHARGE = 1600;          // repulsion strength (f = CHARGE*alpha/d^2)
const LINK_K = 0.28;          // structural spring stiffness
const FB_K = 0.55;            // feedback (preference) spring stiffness
const GRAV = 0.018;           // pull toward centre (keeps it bounded)
const VEL_DECAY = 0.62;       // velocity retained each tick (lower = more damping)
const ALPHA_DECAY = 0.022, ALPHA_MIN = 0.0015, REHEAT = 0.85, DRAG_ALPHA = 0.35;
// preference-spring rest lengths (× WORLD_SPACING). SIMILAR items are PULLED, graded near->far but
// never sent past the natural neighbour distance (SIM_FAR < 1) so a "less similar but still similar"
// item is not punished. DISSIMILAR items are PUSHED out (DIS_NEAR > 1).
const SIM_NEAR = 0.18, SIM_FAR = 0.92, DIS_NEAR = 1.30, DIS_FAR = 2.85;
// high-amplitude marks should WIN against the other links: the involved nodes release their
// structural springs (×amp) and the marked-similar pair's mutual repulsion is suppressed (×amp),
// so s=1 actually pulls the two nodes adjacent instead of being out-tugged by the rest of the graph.
const STRUCT_RELEASE = 0.92;
const GRAPH_BASE = 40;        // base graph-exemplar px at the fitted zoom (× graphMul × zoom^EXP)
const ZOOM_IMG_EXP = 0.5;     // <1: images grow SUB-linearly with zoom so zooming reveals more specimens

const CAT = ["#4e79a7","#f28e2b","#e15759","#76b7b2","#59a14f","#edc948","#b07aa1","#ff9da7","#9c755f","#bab0ac"];
const VIRIDIS = [[68,1,84],[59,82,139],[33,144,140],[93,201,99],[253,231,37]];
function viridis(t){ t=Math.max(0,Math.min(1,t)); const x=t*(VIRIDIS.length-1),i=Math.floor(x),f=x-i;
  const a=VIRIDIS[i],b=VIRIDIS[Math.min(i+1,VIRIDIS.length-1)];
  return `rgb(${a[0]+(b[0]-a[0])*f|0},${a[1]+(b[1]-a[1])*f|0},${a[2]+(b[2]-a[2])*f|0})`; }

const S = {
  sid:null, dataset:null, K:0, hasGt:false, names:[],
  nodes:[], edges:[], adj:[], fbSprings:[], pcValues:[], thumbs:{},
  active_pc:0, colorBy:"joint", gtJoint:[], gtFactors:{},
  view:{scale:1,ox:0,oy:0,fitted:false}, hover:null, round:0,
  anchor:null, panel:null, marks:{}, remote:null, cands:[],  // marks: idx->{kind,amp}; cands: ranking set
  alpha:0, raf:null, frozen:false, showEdges:true, graphMul:2.5, fitScale:null, autoFit:true,
  pendingNode:null, dragNode:null, dragMoved:false, panning:false, lastX:0, lastY:0,
};

const $=(id)=>document.getElementById(id);
async function api(method,path,body){ const o={method,headers:{"Content-Type":"application/json"}};
  if(body) o.body=JSON.stringify(body);
  const r=await fetch(path,o); if(!r.ok){ const e=await r.json().catch(()=>({error:r.statusText})); throw new Error(e.error||r.statusText);} return r.json(); }
async function apiBlob(method,path,body){ const o={method,headers:{"Content-Type":"application/json"}};
  if(body) o.body=JSON.stringify(body);
  const r=await fetch(path,o); if(!r.ok) throw new Error("render "+r.status); return r.blob(); }
function toast(m){ const t=$("toast"); t.textContent=m; t.classList.add("show");
  clearTimeout(toast._t); toast._t=setTimeout(()=>t.classList.remove("show"),2200); }
function setStatus(s){ $("status").textContent=s; }

// ----------------------------------------------------------------------------- session
async function loadDatasets(){ const {datasets}=await api("GET","/api/datasets");
  const sel=$("dataset"); sel.innerHTML="";
  datasets.forEach(d=>{const o=document.createElement("option");o.value=d.name;o.textContent=d.title;sel.appendChild(o);}); }

async function createSession(){
  const ds=$("dataset").value;
  setStatus("loading "+ds+" (rendering thumbnails on first load)…"); $("newSession").disabled=true;
  try{
    const r=await api("POST","/api/session",{dataset:ds});
    Object.assign(S,{sid:r.sid,dataset:ds,K:r.K,hasGt:r.has_gt,names:r.names,active_pc:r.active_pc,round:0});
    S.fbSprings=[]; S.thumbs={}; S.anchor=null; S.panel=null; S.marks={}; S.cands=[]; S.gtJoint=[]; S.gtFactors={};
    S.colorBy=r.has_gt?"joint":"uniform"; S.view.fitted=false;
    await loadGraph();
    buildColorBy(); preloadThumbs();
    await newAnchor();
    reheat(1.0); startSim();
    refreshDendro(600);
    setStatus(`${r.title} · ${r.names.length} specimens`);
  }catch(e){ setStatus("error: "+e.message); toast("error: "+e.message); }
  finally{ $("newSession").disabled=false; }
}

async function loadGraph(){
  const g=await api("GET",`/api/session/${S.sid}/graph`);
  S.pcValues=g.pc_values; S.gtJoint=g.gt_joint||[]; S.gtFactors=g.gt_factors||{};
  // normalise to WORLD_SPACING using median structural rest length
  const rests=g.edges.map(e=>e[2]).sort((a,b)=>a-b);
  const med=rests.length?rests[rests.length>>1]:1; const sc=WORLD_SPACING/(med||1);
  S.nodes=g.nodes.map(n=>({i:n.i,name:n.name,x:n.x*sc,y:n.y*sc,vx:0,vy:0,fx:null,fy:null}));
  S.edges=g.edges.map(([a,b,r])=>({a,b,rest:r*sc}));
  S.adj=S.nodes.map(()=>[]);
  S.edges.forEach((e,k)=>{ S.adj[e.a].push(k); S.adj[e.b].push(k); });
  S.spacing=WORLD_SPACING;
  S.structRel=new Float32Array(S.nodes.length); S.simPair=new Array(S.nodes.length).fill(null);
  S.view.fitted=false; S.autoFit=true;
}

function preloadThumbs(){ S.names.forEach((name,i)=>{ if(S.thumbs[i])return;
  const img=new Image(); img.onload=()=>draw(); img.src=`/thumb/${S.dataset}/${encodeURIComponent(name)}.png`; S.thumbs[i]=img; }); }

// ----------------------------------------------------------------------------- controls
function buildColorBy(){ const sel=$("colorBy"), wrap=$("colorByWrap");
  sel.innerHTML="";
  if(!S.hasGt){ S.colorBy="uniform"; if(wrap) wrap.style.display="none"; updateLegend(); return; }
  if(wrap) wrap.style.display="";
  [["joint","ground-truth (joint)"],["belly","belly"],["tail","tail"],["stripe","stripe"],["cheeks","cheeks"]]
    .forEach(([v,t])=>{const o=document.createElement("option");o.value=v;o.textContent=t;sel.appendChild(o);});
  if(!["joint","belly","tail","stripe","cheeks"].includes(S.colorBy)) S.colorBy="joint";
  sel.value=S.colorBy; sel.onchange=()=>{S.colorBy=sel.value;updateLegend();draw();}; updateLegend(); }
// independent size controls: ranking-panel cards (CSS) and graph exemplars (canvas multiplier)
const PANEL_SIZES={ s:{card:64,anchor:84}, m:{card:98,anchor:124}, l:{card:144,anchor:176} };
const GRAPH_SIZES={ s:1.5, m:2.5, l:3.7 };
function setPanelSize(sz){ const c=PANEL_SIZES[sz]||PANEL_SIZES.m;
  document.documentElement.style.setProperty("--cardimg", c.card+"px");
  document.documentElement.style.setProperty("--anchorimg", c.anchor+"px"); }
function setGraphSize(sz){ S.graphMul=GRAPH_SIZES[sz]||GRAPH_SIZES.m; draw(); }
function updateLegend(){ const lg=$("legend"); lg.innerHTML="";
  if(!S.hasGt || S.colorBy==="uniform") return;
  const labs=S.colorBy==="joint"?S.gtJoint:(S.gtFactors[S.colorBy]||[]);
  [...new Set(labs)].sort((a,b)=>a-b).forEach(u=>{ const s=document.createElement("span"); s.className="sw";
    s.innerHTML=`<span class="dot" style="background:${CAT[u%CAT.length]}"></span>${S.colorBy}=${u}`; lg.appendChild(s); }); }

// debounced WYSIWYG dendrogram from the live layout
let _dendroTimer=null;
function refreshDendro(delay=900){ clearTimeout(_dendroTimer);
  _dendroTimer=setTimeout(async()=>{ if(!S.sid)return;
    const positions=S.nodes.map(n=>[n.x,n.y]);
    try{ const d=await api("POST",`/api/session/${S.sid}/dendrogram`,{positions}); buildDendro(d); }
    catch(e){/* ignore transient */} }, delay); }

// ----------------------------------------------------------------------------- save / load feedback
function saveFeedback(){
  if(!S.sid){ toast("load a dataset first"); return; }
  if(!S.fbSprings.length){ toast("no feedback to save"); return; }
  const data={dataset:S.dataset, round:S.round, saved:_explainTick,
    springs:S.fbSprings.map(e=>({a:e.a,b:e.b,kind:e.kind,amp:e.amp,rest:e.rest,k:e.k}))};
  _download(new Blob([JSON.stringify(data)],{type:"application/json"}), `feedback_${S.dataset}_round${S.round}.json`);
  toast(`saved ${S.fbSprings.length} feedback springs`);
}
async function loadFeedbackFile(file){
  if(!S.sid){ toast("load a dataset first"); return; }
  try{
    const data=JSON.parse(await file.text());
    if(data.dataset && data.dataset!==S.dataset){ toast(`feedback is for '${data.dataset}', current dataset is '${S.dataset}'`); return; }
    S.fbSprings=(data.springs||[]).filter(s=>s.a<S.nodes.length&&s.b<S.nodes.length).map(s=>{
      const amp=(s.amp==null?1:s.amp);
      const rest=(s.rest!=null)?s.rest:(s.kind==="near"?(SIM_FAR+(SIM_NEAR-SIM_FAR)*amp):(DIS_NEAR+(DIS_FAR-DIS_NEAR)*amp))*S.spacing;
      return {a:s.a,b:s.b,kind:s.kind,amp,rest,k:(s.k!=null?s.k:FB_K*(0.3+0.7*amp))};
    });
    S.round=data.round||S.fbSprings.length;
    recomputeReleases(); reheat(REHEAT); refreshDendro(1100);
    toast(`loaded ${S.fbSprings.length} feedback springs — reshaping`);
  }catch(e){ toast("load failed: "+e.message); }
}

// ----------------------------------------------------------------------------- explain my grouping
let _explainTick=0;
async function runExplain(){
  if(!S.sid) return;
  const pairs=S.fbSprings.map(e=>({a:e.a,b:e.b,kind:e.kind,amp:(e.amp==null?1:e.amp)}));
  if(!pairs.length){ toast("Give some similar/dissimilar feedback first"); return; }
  toast("analysing your feedback…");
  const positions=S.nodes.map(n=>[n.x,n.y]);
  let d; try{ d=await api("POST",`/api/session/${S.sid}/explain`,{pairs,positions}); }
  catch(e){ toast("explain failed: "+e.message); return; }
  renderExplain(d);
}
function renderExplain(d){
  const panel=$("explainPanel"), body=$("explainBody"); panel.classList.remove("hidden");
  if(!d.ok){ body.innerHTML=`<div style="color:var(--muted);font-size:12px">${d.reason||"not enough feedback"}</div>`; return; }
  _explainTick++;
  const covTxt={good:"Color + pattern explains your grouping",
                weak:"Color + pattern only weakly explains your grouping",
                none:"Color + pattern does NOT explain your grouping — it's likely shape/size or something not measured"}[d.coverage];
  const covCls={good:"cov-good",weak:"cov-weak",none:"cov-none"}[d.coverage];
  const cs=Math.round(d.color_share*100), ps=100-cs;
  const pcs=d.pcs.slice().sort((a,b)=>b.dw-a.dw).filter(p=>p.dw>0.01).slice(0,6);
  const maxdw=pcs.length?pcs[0].dw:1, sumdw=d.pcs.reduce((s,p)=>s+Math.max(0,p.dw),0)||1;
  const bars=pcs.map(p=>{ const share=Math.round(100*Math.max(0,p.dw)/sumdw);
    const sig=p.p<0.05?' <span style="color:var(--green)" title="significant (permutation p&lt;0.05)">★</span>':'';
    return `<div class="pcbar-row"><span class="pcbar-lab">PC${p.pc+1} · ${share}%${sig}</span>
      <span class="pcbar-track"><span class="pcbar-fill" style="width:${Math.round(100*p.dw/maxdw)}%"></span></span></div>`; }).join("");
  const ex=d.exemplars, imgs=a=>a.map(e=>`<img src="/thumb/${S.dataset}/${encodeURIComponent(e.name)}.png" title="${e.name}">`).join("");
  body.innerHTML=
    `<div class="cov-badge ${covCls}">${covTxt}</div>
     <div style="font-size:11px;color:var(--muted);margin:7px 0 4px">predicts held-out judgments
       <b style="color:var(--fg)">${Math.round(d.heldout_acc*100)}%</b> (baseline ${Math.round(d.baseline_acc*100)}%)`+
       (d.place_err!=null?` · places held-out samples within <b style="color:var(--fg)">${Math.round(d.place_err*100)}%</b> of the spread`:``)+
     `</div>
     <div style="font-size:11px;color:var(--muted)">color vs pattern</div>
     <div class="bar"><div class="cseg" style="width:${cs}%"></div><div class="pseg" style="width:${ps}%"></div></div>
     <div style="font-size:11px"><span style="color:#5aa9e6">color ${cs}%</span> / <span style="color:#e6a15a">pattern ${ps}%</span></div>
     <div style="font-size:11px;color:var(--muted);margin-top:9px">PCs that explain your grouping (★ = significant)</div>${bars}
     <div style="font-size:11px;color:var(--muted);margin-top:9px">where it lives on the body</div>
     <img id="explainHeat" src="/api/session/${S.sid}/explain_heatmap.png?t=${_explainTick}" alt="grouping heatmap">
     <div style="font-size:11px;color:var(--muted)">PC${ex.pc+1} extremes — you're separating these…</div>
     <div class="exrow">${imgs(ex.low)}</div>
     <div style="font-size:11px;color:var(--muted);margin-top:3px">…from these:</div>
     <div class="exrow">${imgs(ex.high)}</div>`;
}

// ----------------------------------------------------------------------------- feedback
async function newAnchor(anchorIdx){
  const q=(anchorIdx==null)?"":`?anchor=${anchorIdx}`;
  const p=await api("GET",`/api/session/${S.sid}/panel${q}`);
  S.panel=p; S.anchor=p.anchor; S.remote=p.remote; S.marks={};
  S.cands=[...p.neighbors]; if(p.remote!=null) S.cands.push(p.remote);
  renderPanel(); draw(); updateDendroSelection(); }
// Ctrl/Cmd-click a graph node toggles it in/out of the ranking set (auto picks stay as a start).
function toggleCand(idx){ const i=S.cands.indexOf(idx);
  if(i>=0){ S.cands.splice(i,1); delete S.marks[idx]; if(idx===S.remote) S.remote=null; }
  else S.cands.push(idx);
  renderPanel(); draw(); updateDendroSelection(); }
function renderPanel(){
  $("anchorBox").innerHTML=`<img src="/thumb/${S.dataset}/${encodeURIComponent(S.names[S.anchor])}.png" title="${S.names[S.anchor]}">`;
  const nb=$("neighbors"); nb.innerHTML="";
  S.cands.forEach(idx=>{ const card=document.createElement("div"); card.className="card"; card.dataset.idx=idx;
    let html=`<img src="/thumb/${S.dataset}/${encodeURIComponent(S.names[idx])}.png" title="${S.names[idx]}">`;
    if(idx===S.remote) html+=`<span class="remoteTag">distant check</span>`;
    html+=`<span class="badge"></span><input type="range" class="amp" min="0" max="1" step="0.01" value="0.5" title="how strongly (0–1)">`;
    card.innerHTML=html;
    const slider=card.querySelector(".amp");
    slider.oninput=()=>setAmp(idx,+slider.value);
    slider.onpointerdown=e=>e.stopPropagation(); slider.onclick=e=>e.stopPropagation();
    card.onclick=e=>{ if(!e.target.classList.contains("amp")) cycleCard(idx); };
    card.addEventListener("wheel",e=>{ const m=S.marks[idx]; if(!m) return;   // scroll a marked card = amplitude
      e.preventDefault(); e.stopPropagation(); setAmp(idx, m.amp + (e.deltaY<0?0.05:-0.05)); },{passive:false});
    nb.appendChild(card); });
  updateMarks(); }
// click cycles: unmarked -> SIMILAR (pull) -> DISSIMILAR (push) -> cleared. amplitude defaults to 0.5.
function cycleCard(idx){ const m=S.marks[idx];
  if(!m) S.marks[idx]={kind:"sim",amp:0.5};
  else if(m.kind==="sim") m.kind="dis";       // keep the amplitude when flipping
  else delete S.marks[idx];
  updateMarks(); }
function setAmp(idx,a){ const m=S.marks[idx]; if(!m) return; m.amp=Math.max(0,Math.min(1,a)); updateMarks(); }
function updateMarks(){ document.querySelectorAll("#neighbors .card").forEach(card=>{
    const idx=+card.dataset.idx, m=S.marks[idx];
    card.classList.toggle("sim", !!m && m.kind==="sim");
    card.classList.toggle("dis", !!m && m.kind==="dis");
    const b=card.querySelector(".badge"), s=card.querySelector(".amp");
    if(m){ b.textContent=(m.kind==="sim"?"S ":"D ")+m.amp.toFixed(2); s.value=m.amp; } else b.textContent="";
  });
  $("submitRank").disabled=Object.keys(S.marks).length<1; }

function addPreferenceSprings(anchor){
  // per-item AMPLITUDE sets rest length AND stiffness: SIMILAR amp 1->SIM_NEAR (tight) .. 0->SIM_FAR
  // (neutral); DISSIMILAR amp 0->DIS_NEAR (slight push) .. 1->DIS_FAR (far push). Higher amp also
  // makes the spring stiffer and (with the structural-release/charge-suppression in tick) win.
  for(const k in S.marks){ const idx=+k, m=S.marks[k];
    const rest = (m.kind==="sim") ? (SIM_FAR + (SIM_NEAR-SIM_FAR)*m.amp)*S.spacing
                                  : (DIS_NEAR + (DIS_FAR-DIS_NEAR)*m.amp)*S.spacing;
    S.fbSprings.push({a:anchor, b:idx, rest, kind:(m.kind==="sim"?"near":"far"),
                      amp:m.amp, k:FB_K*(0.3 + 0.7*m.amp)});
  }
  recomputeReleases();
}
function recomputeReleases(){
  const N=S.nodes.length;
  S.structRel=new Float32Array(N); S.simPair=new Array(N).fill(null);
  for(const e of S.fbSprings){ const amp=(e.amp==null?1:e.amp);
    if(amp>S.structRel[e.a]) S.structRel[e.a]=amp;
    if(amp>S.structRel[e.b]) S.structRel[e.b]=amp;
    if(e.kind==="near"){ (S.simPair[e.a]||(S.simPair[e.a]={}))[e.b]=amp;
                         (S.simPair[e.b]||(S.simPair[e.b]={}))[e.a]=amp; }
  }
}

async function submitRanking(tied){
  if(!S.panel) return;
  const ks=Object.keys(S.marks);
  const nsim=ks.filter(k=>S.marks[k].kind==="sim").length, ndis=ks.length-nsim;
  if(!tied && ks.length>=1){ addPreferenceSprings(S.anchor);
    S.round++; reheat(REHEAT); refreshDendro(1100);
    toast(`Reshaped: ${nsim} pulled / ${ndis} pushed (round ${S.round})`); }
  else toast("Tie noted — springs unchanged");
  // log to server (best-effort; layout is client-side)
  api("POST",`/api/session/${S.sid}/feedback`,{anchor:S.anchor,ranked:tied?[]:ks.map(Number),tied:!!tied,remote:null}).catch(()=>{});
  await newAnchor();
}
function resetFeedback(){ S.fbSprings=[]; recomputeReleases(); S.round=0; reheat(REHEAT); refreshDendro(1100); toast("feedback springs cleared"); }
async function resetLayout(){ await loadGraph(); preloadThumbs(); reheat(1.0); startSim(); refreshDendro(1100); toast("layout reseeded from PCA"); }

// ----------------------------------------------------------------------------- physics
function reheat(a){ S.alpha=Math.max(S.alpha,a); if(!S.frozen) startSim(); }
function startSim(){ if(S.raf!=null) return; const loop=()=>{ S.raf=null;
    if(S.frozen){ updateBadge(); return; }
    for(let s=0;s<1;s++) tick();
    if(S.autoFit) S.view.fitted=false;          // keep the settling/reshaping layout framed
    draw(); updateBadge();
    if(S.alpha>ALPHA_MIN || S.dragNode!=null) S.raf=requestAnimationFrame(loop); };
  S.raf=requestAnimationFrame(loop); }
function updateBadge(){ const b=$("simBadge");
  if(S.frozen){ b.innerHTML="paused"; return; }
  b.innerHTML=(S.alpha>ALPHA_MIN)?`<span class="pulse">● settling</span>`:"● settled"; }

function tick(){
  const N=S.nodes.length, nd=S.nodes, a=S.alpha;
  // charge repulsion (O(N^2); fine for N<=~300). Repulsion between a marked-similar pair is
  // suppressed by their amplitude so a strong "similar" can actually bring them adjacent.
  const simPair=S.simPair;
  for(let i=0;i<N;i++){ const ni=nd[i], sp=simPair[i];
    for(let j=i+1;j<N;j++){ const nj=nd[j];
      let dx=nj.x-ni.x, dy=nj.y-ni.y, d2=dx*dx+dy*dy+0.01; const d=Math.sqrt(d2);
      let f=CHARGE*a/d2; if(sp){ const ap=sp[j]; if(ap) f*=(1-ap); }
      const ux=dx/d, uy=dy/d;
      ni.vx-=f*ux; ni.vy-=f*uy; nj.vx+=f*ux; nj.vy+=f*uy; } }
  // structural springs — weakened on nodes that carry a strong preference (they "let go")
  const sr=S.structRel;
  for(const e of S.edges){ const rel=1-STRUCT_RELEASE*(sr[e.a]>sr[e.b]?sr[e.a]:sr[e.b]);
    springForce(nd[e.a],nd[e.b],e.rest,LINK_K*a*rel); }
  // feedback (preference) springs — per-spring stiffness scaled by amplitude
  for(const e of S.fbSprings) springForce(nd[e.a],nd[e.b],e.rest,(e.k||FB_K)*a);
  // gravity + integrate
  for(let i=0;i<N;i++){ const n=nd[i];
    if(n===S.dragNode){ n.x=n.fx; n.y=n.fy; n.vx=0; n.vy=0; continue; }
    n.vx-=GRAV*a*n.x; n.vy-=GRAV*a*n.y;
    n.x+=n.vx; n.y+=n.vy; n.vx*=VEL_DECAY; n.vy*=VEL_DECAY; }
  S.alpha+=(0-S.alpha)*ALPHA_DECAY;
  if(S.dragNode!=null) S.alpha=Math.max(S.alpha,DRAG_ALPHA);
}
function springForce(ni,nj,rest,k){ let dx=nj.x-ni.x, dy=nj.y-ni.y; const d=Math.sqrt(dx*dx+dy*dy)+1e-6;
  const m=k*(d-rest)/d; ni.vx+=m*dx*0.5; ni.vy+=m*dy*0.5; nj.vx-=m*dx*0.5; nj.vy-=m*dy*0.5; }

// ----------------------------------------------------------------------------- rendering
const cv=$("morpho"), ctx=cv.getContext("2d"); let DPR=window.devicePixelRatio||1;
function resize(){ DPR=window.devicePixelRatio||1;
  const w=Math.round(cv.clientWidth*DPR), h=Math.round(cv.clientHeight*DPR);
  if(cv.width===w && cv.height===h) return;          // no change -> don't refit
  cv.width=w; cv.height=h; S.view.fitted=false; draw(); }
window.addEventListener("resize",resize);
// keep the backing store in sync with the displayed size: the dendrogram appearing after load
// shrinks the canvas, and a stale buffer would squish drawing vs. the (unsquished) mouse hit-test,
// putting each node's true hit-target below its painted image.
if(window.ResizeObserver) new ResizeObserver(()=>resize()).observe(cv);
function fitView(){ if(!S.nodes.length)return;
  const xs=S.nodes.map(n=>n.x), ys=S.nodes.map(n=>n.y);
  const minx=Math.min(...xs),maxx=Math.max(...xs),miny=Math.min(...ys),maxy=Math.max(...ys);
  const w=cv.width,h=cv.height,pad=70*DPR;
  const sx=(w-2*pad)/(maxx-minx+1e-6), sy=(h-2*pad)/(maxy-miny+1e-6);
  S.view.scale=Math.min(sx,sy); S.view.ox=w/2-S.view.scale*(minx+maxx)/2; S.view.oy=h/2+S.view.scale*(miny+maxy)/2;
  S.fitScale=S.view.scale; S.view.fitted=true; }
function w2s(x,y){ return [S.view.scale*x+S.view.ox, -S.view.scale*y+S.view.oy]; }
function s2w(sx,sy){ return [(sx-S.view.ox)/S.view.scale, (S.view.oy-sy)/S.view.scale]; }
function colorOf(i){ if(!S.hasGt || S.colorBy==="uniform") return "#7fa8c9";
  const labs=S.colorBy==="joint"?S.gtJoint:(S.gtFactors[S.colorBy]||[]); return CAT[(labs[i]??0)%CAT.length]; }

function draw(){ if(!cv.width)return; ctx.clearRect(0,0,cv.width,cv.height); if(!S.nodes.length)return;
  if(!S.view.fitted) fitView();
  // image size grows SUB-linearly with zoom (zf^EXP) while node spacing grows linearly with zoom,
  // so zooming into a region REVEALS more of its specimens instead of just enlarging the same few.
  const zf=Math.max(0.45,Math.min(9,S.view.scale/(S.fitScale||S.view.scale)));
  const tw=GRAPH_BASE*DPR*S.graphMul*Math.pow(zf,ZOOM_IMG_EXP);
  // edges
  if(S.showEdges){ ctx.lineWidth=1*DPR;
    ctx.strokeStyle="rgba(150,170,200,0.05)"; ctx.beginPath();
    for(const e of S.edges){ const A=S.nodes[e.a],B=S.nodes[e.b]; const [ax,ay]=w2s(A.x,A.y),[bx,by]=w2s(B.x,B.y);
      ctx.moveTo(ax,ay); ctx.lineTo(bx,by); } ctx.stroke();
    // feedback springs, colored
    ctx.lineWidth=1.6*DPR;
    for(const e of S.fbSprings){ const A=S.nodes[e.a],B=S.nodes[e.b]; const [ax,ay]=w2s(A.x,A.y),[bx,by]=w2s(B.x,B.y);
      ctx.strokeStyle=e.kind==="near"?"rgba(90,209,122,0.55)":"rgba(224,96,122,0.5)";
      ctx.beginPath(); ctx.moveTo(ax,ay); ctx.lineTo(bx,by); ctx.stroke(); } }
  // highlight dragged node's structural edges
  if(S.dragNode!=null){ ctx.strokeStyle="rgba(67,198,216,0.5)"; ctx.lineWidth=1.5*DPR; ctx.beginPath();
    for(const k of S.adj[S.dragNode.i]){ const e=S.edges[k]; const A=S.nodes[e.a],B=S.nodes[e.b];
      const [ax,ay]=w2s(A.x,A.y),[bx,by]=w2s(B.x,B.y); ctx.moveTo(ax,ay); ctx.lineTo(bx,by); } ctx.stroke(); }
  // dots
  for(const n of S.nodes){ const [sx,sy]=w2s(n.x,n.y); ctx.fillStyle=colorOf(n.i);
    ctx.beginPath(); ctx.arc(sx,sy,2.4*DPR,0,7); ctx.fill(); }
  // adaptive thumbnails: forced (anchor/panel/hover/drag) + greedy-spaced
  const forced=new Set(); if(S.anchor!=null)forced.add(S.anchor);
  for(const i of S.cands) forced.add(i);
  if(S.hover!=null)forced.add(S.hover); if(S.dragNode!=null)forced.add(S.dragNode.i);
  const placed=[], mind=tw;
  const fits=(sx,sy)=>{ for(const c of placed) if(Math.hypot(c[0]-sx,c[1]-sy)<mind) return false; return true; };
  const place=(i,big)=>{ const n=S.nodes[i]; if(!n)return; const [sx,sy]=w2s(n.x,n.y);
    if(!big&&!fits(sx,sy))return; placed.push([sx,sy]); drawThumb(i,big?tw*1.9:tw,big); };
  forced.forEach(i=>{ if(i!==S.hover&&!(S.dragNode&&i===S.dragNode.i)) place(i,false); });
  for(const n of S.nodes) if(!forced.has(n.i)) place(n.i,false);
  // highlights
  for(const idx of S.cands) ring(idx,tw,"#43c6d8",2.5);
  if(S.anchor!=null){ drawThumb(S.anchor,tw*1.15,false); ring(S.anchor,tw*1.15,"#ffd24a",4); }
  if(S.dragNode!=null) drawThumb(S.dragNode.i,tw*1.6,true);
  if(S.hover!=null) drawThumb(S.hover,tw*1.9,true);
}
function drawThumb(i,tw,big){ const n=S.nodes[i]; if(!n)return; const [sx,sy]=w2s(n.x,n.y);
  const img=S.thumbs[i], col=colorOf(i); const asp=(img&&img.width)?img.width/img.height:2.6;
  const w=tw*Math.sqrt(asp), h=tw/Math.sqrt(asp);
  ctx.strokeStyle=col; ctx.lineWidth=(big?3:1.6)*DPR;
  if(img&&img.complete&&img.naturalWidth){ ctx.drawImage(img,sx-w/2,sy-h/2,w,h); ctx.strokeRect(sx-w/2,sy-h/2,w,h); }
  else { ctx.fillStyle=col; ctx.beginPath(); ctx.arc(sx,sy,3*DPR,0,7); ctx.fill(); } }
function ring(i,tw,color,lw){ const n=S.nodes[i]; if(!n)return; const [sx,sy]=w2s(n.x,n.y);
  const img=S.thumbs[i], asp=(img&&img.width)?img.width/img.height:2.6; const w=tw*Math.sqrt(asp),h=tw/Math.sqrt(asp);
  ctx.strokeStyle=color; ctx.lineWidth=lw*DPR; ctx.strokeRect(sx-w/2-2,sy-h/2-2,w+4,h+4); }
function nodeAt(mx,my){ let best=null,bd=24*DPR; for(const n of S.nodes){ const [sx,sy]=w2s(n.x,n.y);
  const d=Math.hypot(sx-mx,sy-my); if(d<bd){bd=d;best=n.i;} } return best; }

// ----------------------------------------------------------------------------- interaction
cv.addEventListener("mousedown",e=>{ const mx=e.offsetX*DPR,my=e.offsetY*DPR; const hit=nodeAt(mx,my);
  S.dragMoved=false; S.lastX=mx; S.lastY=my; S.pendingCtrl=e.ctrlKey||e.metaKey;
  if(hit!=null){ S.pendingNode=hit; } else { S.panning=true; cv.classList.add("grabbing"); } });
window.addEventListener("mouseup",()=>{
  if(S.pendingNode!=null && !S.dragMoved){
    if(S.pendingCtrl && S.pendingNode!==S.anchor) toggleCand(S.pendingNode);   // ctrl/cmd-click = add/remove candidate
    else newAnchor(S.pendingNode); }                                            // plain click = set anchor
  if(S.dragNode!=null){ S.dragNode.fx=null; S.dragNode.fy=null; reheat(DRAG_ALPHA); }
  S.pendingNode=null; S.dragNode=null; S.panning=false; cv.classList.remove("grabbing"); });
cv.addEventListener("mousemove",e=>{ const mx=e.offsetX*DPR,my=e.offsetY*DPR;
  if(S.panning){ S.autoFit=false; S.view.ox+=mx-S.lastX; S.view.oy+=my-S.lastY; S.lastX=mx; S.lastY=my; draw(); return; }
  if(S.pendingNode!=null){
    if(!S.dragMoved && Math.hypot(mx-S.lastX,my-S.lastY)>3*DPR){ S.dragMoved=true; S.autoFit=false; S.dragNode=S.nodes[S.pendingNode]; reheat(DRAG_ALPHA); }
    if(S.dragNode!=null){ const [wx,wy]=s2w(mx,my); S.dragNode.fx=wx; S.dragNode.fy=wy; if(!S.frozen)startSim(); }
    return; }
  const hit=nodeAt(mx,my); if(hit!==S.hover){ S.hover=hit; draw(); }
  const tip=$("hoverTip"); if(hit!=null){ tip.style.display="block"; tip.style.left=(e.offsetX+12)+"px"; tip.style.top=(e.offsetY+8)+"px"; tip.textContent=S.names[hit]; } else tip.style.display="none"; });
cv.addEventListener("mouseleave",()=>{ S.hover=null; $("hoverTip").style.display="none"; draw(); });
cv.addEventListener("wheel",e=>{ e.preventDefault(); S.autoFit=false; const mx=e.offsetX*DPR,my=e.offsetY*DPR; const f=Math.exp(-e.deltaY*0.0015);
  S.view.ox=mx-(mx-S.view.ox)*f; S.view.oy=my-(my-S.view.oy)*f; S.view.scale*=f; draw(); },{passive:false});

// ----------------------------------------------------------------------------- dendrogram (inline SVG)
// vector branches (crisp at any zoom) + leaf <image> referencing hi-res PNGs (browser samples them
// sharply on zoom). Zoom/pan via a <g> transform; non-scaling strokes keep line widths constant.
const SVGNS="http://www.w3.org/2000/svg", XLINK="http://www.w3.org/1999/xlink";
let ddG=null, ddTx=0, ddTy=0, ddS=1, ddVBW=1, ddVBH=1;
let ddLeaves=[], ddPosI=[];   // ddLeaves: {rect, i, color}; ddPosI[pos] = specimen index — for in-dendrogram selection
let ddMoved=false;            // set while a dendrogram press becomes a pan — mirrors the graph's !dragMoved click guard
function applyDdTransform(){ if(ddG) ddG.setAttribute("transform",`translate(${ddTx} ${ddTy}) scale(${ddS})`); }
function ddMeet(){ const svg=$("dendroSvg"), r=svg.getBoundingClientRect();
  const sc=Math.min(r.width/ddVBW, r.height/ddVBH);
  return {sc, offX:(r.width-ddVBW*sc)/2, offY:(r.height-ddVBH*sc)/2, r}; }
function buildDendro(d){
  const svg=$("dendroSvg"); while(svg.firstChild) svg.removeChild(svg.firstChild);
  ddLeaves=[]; ddPosI=[];
  const W=d.step*d.n, TREE_H=W*0.22, gap=TREE_H*0.06;   // compact tree (less vertical space)
  const leafW=d.step*0.86, leafH=leafW/(d.leaf_aspect||2);
  const VBH=TREE_H+gap+leafH+leafW*0.3;
  ddVBW=W; ddVBH=VBH; ddTx=0; ddTy=0; ddS=1;
  svg.setAttribute("viewBox",`0 0 ${W} ${VBH}`); svg.setAttribute("preserveAspectRatio","xMidYMid meet");
  ddG=document.createElementNS(SVGNS,"g"); svg.appendChild(ddG); applyDdTransform();
  const yT=v=>(1 - v/d.ymax)*TREE_H;
  for(const L of d.links){
    const pl=document.createElementNS(SVGNS,"polyline");
    pl.setAttribute("points", L.x.map((x,k)=>`${x},${yT(L.y[k]).toFixed(2)}`).join(" "));
    pl.setAttribute("fill","none"); pl.setAttribute("stroke",L.color);
    pl.setAttribute("stroke-width",(1.2+L.w*5).toFixed(2));
    pl.setAttribute("vector-effect","non-scaling-stroke");
    pl.setAttribute("stroke-linejoin","round"); pl.setAttribute("stroke-linecap","round");
    ddG.appendChild(pl);
    if(L.span){   // wide transparent overlay: click a branch -> toggle its whole clade
      const hit=document.createElementNS(SVGNS,"polyline");
      hit.setAttribute("points", pl.getAttribute("points"));
      hit.setAttribute("fill","none"); hit.setAttribute("stroke","transparent");
      hit.setAttribute("stroke-width","10"); hit.setAttribute("vector-effect","non-scaling-stroke");
      hit.style.cursor="pointer";
      const span=L.span;
      // A clade isn't a single specimen, so it can't be an anchor: ctrl/cmd-click toggles it as samples.
      hit.addEventListener("click",e=>{ e.stopPropagation(); if(ddMoved) return;
        if(e.ctrlKey||e.metaKey) toggleCluster(span);
        else toast("ctrl-click a branch to select its whole clade"); });
      ddG.appendChild(hit);
    }
  }
  const lyrow=TREE_H+gap;
  for(const lf of d.leaves){
    ddPosI[lf.pos]=lf.i;
    const cx=5+lf.pos*d.step, x=cx-leafW/2;
    // Mirror the 2-D graph: plain click = set anchor, ctrl/cmd-click = add/remove sample.
    // A press that turned into a pan (ddMoved) is not a click — match the graph's !dragMoved guard.
    const onLeaf=e=>{ e.stopPropagation(); if(ddMoved) return;
      if((e.ctrlKey||e.metaKey) && lf.i!==S.anchor) toggleCand(lf.i);
      else newAnchor(lf.i); };
    const im=document.createElementNS(SVGNS,"image");
    const href=`/thumb_hi/${d.dataset}/${encodeURIComponent(lf.name)}.png`;
    im.setAttributeNS(XLINK,"href",href); im.setAttribute("href",href);
    im.setAttribute("x",x.toFixed(2)); im.setAttribute("y",lyrow.toFixed(2));
    im.setAttribute("width",leafW.toFixed(2)); im.setAttribute("height",leafH.toFixed(2));
    im.setAttribute("preserveAspectRatio","xMidYMid meet");
    im.style.cursor="pointer";
    im.addEventListener("click",onLeaf);
    ddG.appendChild(im);
    const rc=document.createElementNS(SVGNS,"rect");
    rc.setAttribute("x",x.toFixed(2)); rc.setAttribute("y",lyrow.toFixed(2));
    rc.setAttribute("width",leafW.toFixed(2)); rc.setAttribute("height",leafH.toFixed(2));
    rc.setAttribute("fill","none"); rc.setAttribute("stroke",lf.color);
    rc.setAttribute("stroke-width","2"); rc.setAttribute("vector-effect","non-scaling-stroke");
    rc.style.cursor="pointer";
    rc.addEventListener("click",onLeaf);
    ddG.appendChild(rc);
    ddLeaves.push({rect:rc, i:lf.i, color:lf.color});
  }
  updateDendroSelection();
  const t=document.createElementNS(SVGNS,"text");
  t.setAttribute("x",(W/2).toFixed(1)); t.setAttribute("y",(TREE_H*0.07).toFixed(1));
  t.setAttribute("text-anchor","middle"); t.setAttribute("font-size",(W*0.016).toFixed(2)); t.setAttribute("fill","#444");
  t.textContent=`${d.n} specimens · merges aligned by depth, thickness = merge distance · round ${d.round} · click a leaf = anchor · ctrl-click a leaf or branch = sample`;
  ddG.appendChild(t);
}
// Highlight dendrogram leaves that are in the candidate set (gold ring), restore cluster color otherwise.
function updateDendroSelection(){
  const sel=new Set(S.cands);
  for(const L of ddLeaves){
    const on=sel.has(L.i);
    L.rect.setAttribute("stroke", on?"#ffce3a":L.color);
    L.rect.setAttribute("stroke-width", on?"3.4":"2");
  }
}
// Click a branch -> toggle its whole clade. If every leaf under it is already selected, deselect them; else add the missing ones.
function toggleCluster(span){
  // Exclude the anchor (it is never a sample) — mirror the leaf/graph `!==S.anchor` guard.
  const ids=[]; for(let p=span[0]; p<=span[1]; p++){ const i=ddPosI[p]; if(i!=null && i!==S.anchor) ids.push(i); }
  if(!ids.length) return;
  const sel=new Set(S.cands);
  const allIn=ids.every(i=>sel.has(i));
  if(allIn){ for(const i of ids){ const k=S.cands.indexOf(i); if(k>=0) S.cands.splice(k,1); delete S.marks[i]; if(i===S.remote) S.remote=null; } }
  else { for(const i of ids) if(!sel.has(i)) S.cands.push(i); }
  renderPanel(); draw(); updateDendroSelection();
}
(function(){ const svg=$("dendroSvg");
  svg.addEventListener("wheel",e=>{ e.preventDefault(); if(!ddG)return; const m=ddMeet();
    const vx=(e.clientX-m.r.left-m.offX)/m.sc, vy=(e.clientY-m.r.top-m.offY)/m.sc;
    const f=Math.exp(-e.deltaY*0.0015);
    ddTx=vx-(vx-ddTx)*f; ddTy=vy-(vy-ddTy)*f; ddS*=f;
    if(ddS<=1){ ddS=1; ddTx=0; ddTy=0; } applyDdTransform(); },{passive:false});
  let drag=false,lx=0,ly=0;
  svg.addEventListener("mousedown",e=>{ drag=true; lx=e.clientX; ly=e.clientY; ddMoved=false; svg.classList.add("grabbing"); e.preventDefault(); });
  window.addEventListener("mouseup",()=>{ if(drag){drag=false; svg.classList.remove("grabbing");} });
  window.addEventListener("mousemove",e=>{ if(!drag||!ddG)return;
    if(!ddMoved && Math.hypot(e.clientX-lx,e.clientY-ly)<=3) return;   // under threshold: not yet a pan, keep the click alive
    ddMoved=true; const m=ddMeet();
    ddTx+=(e.clientX-lx)/m.sc; ddTy+=(e.clientY-ly)/m.sc; lx=e.clientX; ly=e.clientY; applyDdTransform(); });
  svg.addEventListener("dblclick",()=>{ ddTx=0; ddTy=0; ddS=1; applyDdTransform(); });
})();
// ---- download (self-contained SVG with embedded leaf images; or rasterized PNG) ----
const _imgCache={};
async function _embedUrl(href){
  if(_imgCache[href]) return _imgCache[href];
  const bm=await fetch(href).then(r=>r.blob()).then(b=>createImageBitmap(b));
  const maxW=460, sc=Math.min(1,maxW/bm.width);           // cap embed res -> reasonable file size
  const cv=document.createElement("canvas"); cv.width=Math.max(1,Math.round(bm.width*sc)); cv.height=Math.max(1,Math.round(bm.height*sc));
  cv.getContext("2d").drawImage(bm,0,0,cv.width,cv.height);
  const url=cv.toDataURL("image/png"); _imgCache[href]=url; return url;
}
async function buildStandaloneSvg(){
  const svg=$("dendroSvg"), clone=svg.cloneNode(true);
  const g=clone.querySelector("g"); if(g) g.removeAttribute("transform");   // full tree, ignore current zoom
  clone.setAttribute("xmlns",SVGNS); clone.setAttribute("xmlns:xlink",XLINK);
  const vb=(svg.getAttribute("viewBox")||"0 0 100 100").split(/\s+/).map(Number);
  const outW=2200, outH=Math.round(outW*vb[3]/vb[2]);
  clone.setAttribute("width",outW); clone.setAttribute("height",outH);
  const bg=document.createElementNS(SVGNS,"rect");
  bg.setAttribute("x",vb[0]); bg.setAttribute("y",vb[1]); bg.setAttribute("width",vb[2]); bg.setAttribute("height",vb[3]); bg.setAttribute("fill","#fff");
  clone.insertBefore(bg, clone.firstChild);
  await Promise.all([...clone.querySelectorAll("image")].map(async im=>{
    const href=im.getAttribute("href")||im.getAttributeNS(XLINK,"href");
    const data=await _embedUrl(href); im.setAttribute("href",data); im.setAttributeNS(XLINK,"href",data);
  }));
  return {svg:'<?xml version="1.0" encoding="UTF-8"?>\n'+new XMLSerializer().serializeToString(clone), w:outW, h:outH};
}
function _download(blob,name){ const u=URL.createObjectURL(blob); const a=document.createElement("a");
  a.href=u; a.download=name; document.body.appendChild(a); a.click(); a.remove(); setTimeout(()=>URL.revokeObjectURL(u),3000); }
async function downloadDendroSvg(){ if(!ddG) return; toast("building SVG…");
  const {svg}=await buildStandaloneSvg();
  _download(new Blob([svg],{type:"image/svg+xml"}), `dendrogram_${S.dataset}_round${S.round}.svg`); }
async function downloadDendroPng(){ if(!ddG) return; toast("building PNG…");
  const {svg,w,h}=await buildStandaloneSvg();
  const url=URL.createObjectURL(new Blob([svg],{type:"image/svg+xml"})), img=new Image();
  img.onload=()=>{ const cv=document.createElement("canvas"); cv.width=w; cv.height=h;
    const ctx=cv.getContext("2d"); ctx.fillStyle="#fff"; ctx.fillRect(0,0,w,h); ctx.drawImage(img,0,0,w,h);
    cv.toBlob(b=>{ _download(b, `dendrogram_${S.dataset}_round${S.round}.png`); URL.revokeObjectURL(url); },"image/png"); };
  img.onerror=()=>{ toast("PNG export failed"); URL.revokeObjectURL(url); };
  img.src=url; }

// ----------------------------------------------------------------------------- wire up
$("newSession").onclick=createSession;
$("resetFeedback").onclick=()=>S.sid&&resetFeedback();
$("resetLayout").onclick=()=>S.sid&&resetLayout();
$("submitRank").onclick=()=>submitRanking(false);
$("tiedBtn").onclick=()=>submitRanking(true);
$("clearRank").onclick=()=>{S.marks={};updateMarks();};
$("deselectAll").onclick=()=>{ S.cands=[]; S.marks={}; S.remote=null; renderPanel(); draw(); updateDendroSelection(); };
$("newAnchor").onclick=()=>newAnchor();
$("explainBtn").onclick=runExplain;
$("saveFb").onclick=saveFeedback;
$("loadFb").onclick=()=>$("loadFbFile").click();
$("loadFbFile").onchange=e=>{ const f=e.target.files[0]; if(f) loadFeedbackFile(f); e.target.value=""; };
$("showEdges").onchange=e=>{S.showEdges=e.target.checked;draw();};
$("freeze").onchange=e=>{S.frozen=e.target.checked; if(!S.frozen){reheat(0.2);} updateBadge();};
$("dlSvg").onclick=downloadDendroSvg; $("dlPng").onclick=downloadDendroPng;
$("panelSize").onchange=e=>setPanelSize(e.target.value);
$("graphSize").onchange=e=>setGraphSize(e.target.value);
setPanelSize($("panelSize").value); setGraphSize($("graphSize").value);
loadDatasets().then(()=>resize());
