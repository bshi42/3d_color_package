// Tagging morphospace — multi-label labeling -> per-tag classifier -> warm-started UMAP latent space.
const $ = id => document.getElementById(id);
const CAT = ["#4e79a7","#f28e2b","#e15759","#76b7b2","#59a14f","#edc948","#b07aa1","#ff9da7","#9c755f",
             "#1f9e89","#d37295","#8cd17d"];
const WORLD = 900;                       // fixed world extent so the view is stable across UMAP updates

const S = {
  sid:null, dataset:null, names:[], N:0, hasGt:false, gtFactors:{},
  tags:[], labels:{},                    // committed labels {i:[tagidx]}
  coords:[], disp:[], pred:null, batch:[],
  work:{},                               // pending per-specimen tag sets being edited (Set)
  colorBy:"uniform", colorAuto:true, selTag:-1, activeTag:-1, hover:null,  // activeTag = class that click-toggles; colorAuto = follow the default mode until the user picks one
  view:{scale:1, ox:0, oy:0, fitted:false}, thumbs:{}, showThumbs:true, raf:null,
};

async function api(method, path, body){
  const o={method, headers:{"Content-Type":"application/json"}};
  if(body) o.body=JSON.stringify(body);
  const r=await fetch(path,o);
  if(!r.ok){ const e=await r.json().catch(()=>({error:r.statusText})); throw new Error(e.error||r.statusText); }
  return r.json();
}
function toast(m){ const t=$("toast"); t.textContent=m; t.classList.add("show");
  clearTimeout(toast._t); toast._t=setTimeout(()=>t.classList.remove("show"),2200); }
function busy(on){ $("busy").classList.toggle("hidden", !on); }

// ----------------------------------------------------------------- session
async function loadDatasets(){
  const {datasets}=await api("GET","/api/datasets");
  const sel=$("dataset"); sel.innerHTML="";
  datasets.forEach(d=>{ const o=document.createElement("option"); o.value=d.name; o.textContent=d.title; sel.appendChild(o); });
}
async function createSession(){
  const ds=$("dataset").value;
  $("status").textContent="loading "+ds+"…"; $("load").disabled=true;
  try{
    const st=await api("POST","/api/session",{dataset:ds, batch:+$("batchN").value||12});
    S.sid=st.sid; S.thumbs={}; S.view.fitted=false; S.selTag=-1; S.activeTag=-1; S.colorBy="uniform"; S.colorAuto=true;
    applyState(st, true);
    $("status").textContent=`${st.title} · ${st.N} specimens`;
    refreshDendro();
  }catch(e){ $("status").textContent="error: "+e.message; toast(e.message); }
  finally{ $("load").disabled=false; }
}

// merge a server state payload; rebuild UI; tween the morphospace
function applyState(st, initial){
  S.dataset=st.dataset; S.names=st.names; S.N=st.N; S.hasGt=st.has_gt; S.gtFactors=st.gt_factors||{};
  S.tags=st.tags; S.labels=st.labels||{}; S.pred=st.pred; S.batch=st.batch||S.batch;
  const target=normalize(st.coords);
  if(initial || S.disp.length!==target.length){ S.disp=target.map(p=>p.slice()); }
  else { align(target, S.disp); }           // Procrustes-align to current display, then tween
  S.coords=target;
  // reset working edits to committed labels for the current batch
  S.work={}; S.batch.forEach(i=>{ S.work[i]=new Set(S.labels[i]||[]); });
  preloadThumbs();
  buildColorBy(); renderTags(); renderBatch(); updateLegend();
  if(initial) fitView();
  startTween();
}

// scale/center raw UMAP coords into a fixed world box (stable view across updates)
function normalize(raw){
  const n=raw.length; let cx=0,cy=0; raw.forEach(p=>{cx+=p[0];cy+=p[1];}); cx/=n; cy/=n;
  let r=0; raw.forEach(p=>{ r+=Math.hypot(p[0]-cx,p[1]-cy); }); r=(r/n)||1;
  const s=WORLD*0.32/r;
  return raw.map(p=>[(p[0]-cx)*s, (p[1]-cy)*s]);
}
// best rotation+flip of `t` onto reference `ref` (keeps orientation stable -> no jumpy flips)
function align(t, ref){
  let a=0,b=0,c=0,d=0;
  for(let i=0;i<t.length;i++){ a+=t[i][0]*ref[i][0]; b+=t[i][1]*ref[i][1]; c+=t[i][0]*ref[i][1]; d+=t[i][1]*ref[i][0]; }
  // 2x2 cross-cov M=[[a,d],[c,b]]; rotation = U V^T via closed-form for 2D
  const M=[[a,d],[c,b]];
  const ang=Math.atan2(M[1][0]-M[0][1], M[0][0]+M[1][1]);
  const co=Math.cos(ang), si=Math.sin(ang);
  // also test flip (reflection) and pick whichever aligns better
  const cand=[[co,-si,si,co],[co,si,si,-co]];
  let best=null,bs=-1e18;
  for(const [m00,m01,m10,m11] of cand){
    let sc=0; for(let i=0;i<t.length;i++){ const x=m00*t[i][0]+m01*t[i][1], y=m10*t[i][0]+m11*t[i][1]; sc+=x*ref[i][0]+y*ref[i][1]; }
    if(sc>bs){ bs=sc; best=[m00,m01,m10,m11]; }
  }
  const [m00,m01,m10,m11]=best;
  for(let i=0;i<t.length;i++){ const x=m00*t[i][0]+m01*t[i][1], y=m10*t[i][0]+m11*t[i][1]; t[i][0]=x; t[i][1]=y; }
}

// ----------------------------------------------------------------- canvas morphospace
const cv=$("morpho"), ctx=cv.getContext("2d"); let DPR=1;
function resize(){ DPR=window.devicePixelRatio||1; const r=cv.getBoundingClientRect();
  cv.width=Math.round(r.width*DPR); cv.height=Math.round(r.height*DPR); draw(); }
new ResizeObserver(resize).observe($("mWrap"));
function fitView(){ if(!S.coords.length||!cv.width) return;
  let mnx=1e9,mny=1e9,mxx=-1e9,mxy=-1e9; S.coords.forEach(p=>{mnx=Math.min(mnx,p[0]);mny=Math.min(mny,p[1]);mxx=Math.max(mxx,p[0]);mxy=Math.max(mxy,p[1]);});
  const w=cv.width,h=cv.height,m=70*DPR; const sc=Math.min((w-2*m)/(mxx-mnx||1),(h-2*m)/(mxy-mny||1));
  S.view.scale=sc; S.view.ox=w/2-sc*(mnx+mxx)/2; S.view.oy=h/2-sc*(mny+mxy)/2; S.view.fitted=true; }
function w2s(x,y){ return [x*S.view.scale+S.view.ox, y*S.view.scale+S.view.oy]; }

function colorOf(i){
  if(S.colorBy.startsWith("tag:")){ const k=+S.colorBy.slice(4); const p=S.pred?S.pred[i][k]:0;
    return mix("#2b3a4f", CAT[k%CAT.length], p); }
  if((S.colorBy==="predicted"||S.colorBy==="allpred") && S.pred){ let bi=0,bv=-1; S.pred[i].forEach((v,k)=>{if(v>bv){bv=v;bi=k;}});
    return bv>0.5?CAT[bi%CAT.length]:"#56708f"; }   // single-colour fallback (border / hover)
  return "#7fa8c9";
}
function mix(a,b,t){ t=Math.max(0,Math.min(1,t)); const pa=hx(a),pb=hx(b);
  return `rgb(${Math.round(pa[0]+(pb[0]-pa[0])*t)},${Math.round(pa[1]+(pb[1]-pa[1])*t)},${Math.round(pa[2]+(pb[2]-pa[2])*t)})`; }
function hx(c){ if(c[0]==="#"){ const n=parseInt(c.slice(1),16); return [(n>>16)&255,(n>>8)&255,n&255]; } return [120,160,200]; }
// multi-label glyph helpers: every tag whose predicted prob >= 0.5 (sorted by confidence)
function predTags(i,thr=0.5){ if(!S.pred) return []; const o=[]; S.pred[i].forEach((p,t)=>{ if(p>=thr) o.push({t,p}); }); o.sort((a,b)=>b.p-a.p); return o; }
function drawPie(x,y,r,tags){
  if(!tags.length){ ctx.fillStyle="#56708f"; ctx.beginPath(); ctx.arc(x,y,r,0,7); ctx.fill(); return; }
  if(tags.length===1){ ctx.fillStyle=CAT[tags[0].t%CAT.length]; ctx.beginPath(); ctx.arc(x,y,r,0,7); ctx.fill(); return; }
  const tot=tags.reduce((s,o)=>s+o.p,0)||1; let a=-Math.PI/2;            // wedge ∝ confidence
  for(const o of tags){ const a2=a+2*Math.PI*o.p/tot; ctx.fillStyle=CAT[o.t%CAT.length];
    ctx.beginPath(); ctx.moveTo(x,y); ctx.arc(x,y,r,a,a2); ctx.closePath(); ctx.fill(); a=a2; }
}

function draw(){ if(!cv.width) return; ctx.clearRect(0,0,cv.width,cv.height); if(!S.disp.length) return;
  if(!S.view.fitted) fitView();
  const zf=Math.max(.5,Math.min(7,S.view.scale/( (S._fit||S.view.scale) )));
  const tw=42*DPR*Math.pow(zf,0.5);
  // dots — a multi-wedge pie glyph (all predicted classes) in 'allpred' mode, else a single dot
  const allpred = S.colorBy==="allpred" && S.pred;
  const dscale = S.showThumbs?1:1.7;                       // bigger dots when exemplars are hidden (dots-only view)
  for(let i=0;i<S.N;i++){ const [x,y]=w2s(S.disp[i][0],S.disp[i][1]); const r=(allpred?4.4:3)*dscale*DPR;
    if(allpred){ drawPie(x,y,r,predTags(i)); }
    else { ctx.fillStyle=colorOf(i); ctx.beginPath(); ctx.arc(x,y,r,0,7); ctx.fill(); }
    if(S.labels[i] && S.labels[i].length){ ctx.strokeStyle="#fff8"; ctx.lineWidth=1.4*DPR;
      ctx.beginPath(); ctx.arc(x,y,r,0,7); ctx.stroke(); } }
  // adaptive thumbnails (greedy spacing) + hover — skipped entirely when exemplars are toggled off
  if(S.showThumbs){
    const placed=[], mind=tw*1.05;
    const fits=(x,y)=>{ for(const c of placed) if(Math.hypot(c[0]-x,c[1]-y)<mind) return false; return true; };
    const order=[]; for(let i=0;i<S.N;i++) order.push(i);
    if(S.hover!=null) order.sort((a,b)=>(a===S.hover?1:0)-(b===S.hover?1:0));
    for(const i of order){ if(i===S.hover) continue; const [x,y]=w2s(S.disp[i][0],S.disp[i][1]);
      if(fits(x,y)){ placed.push([x,y]); drawThumb(i,x,y,tw,false); } }
    if(S.hover!=null){ const [x,y]=w2s(S.disp[S.hover][0],S.disp[S.hover][1]); drawThumb(S.hover,x,y,tw*1.8,true); }
  }
}
function drawThumb(i,x,y,tw,big){ const img=S.thumbs[i]; const asp=(img&&img.width)?img.width/img.height:2.4;
  const w=tw*Math.sqrt(asp), h=tw/Math.sqrt(asp);
  if(img&&img.complete&&img.naturalWidth){ ctx.drawImage(img,x-w/2,y-h/2,w,h);
    ctx.strokeStyle=colorOf(i); ctx.lineWidth=(big?3:1.6)*DPR; ctx.strokeRect(x-w/2,y-h/2,w,h);
    if(S.colorBy==="allpred" && S.pred){ const tags=predTags(i);   // segmented top bar = all predicted classes
      if(tags.length){ const bw=w/tags.length, bh=Math.max(3*DPR,h*0.13);
        tags.forEach((o,j)=>{ ctx.fillStyle=CAT[o.t%CAT.length]; ctx.fillRect(x-w/2+j*bw, y-h/2, bw, bh); }); } } }
}
function preloadThumbs(){ S.names.forEach((nm,i)=>{ if(S.thumbs[i])return; const im=new Image();
  im.onload=()=>draw(); im.src=`/thumb/${S.dataset}/${encodeURIComponent(nm)}.png`; S.thumbs[i]=im; }); }

function startTween(){ if(S.raf) return; if(!S._fit) S._fit=S.view.scale;
  const step=()=>{ let mov=0;
    for(let i=0;i<S.disp.length;i++){ const dx=S.coords[i][0]-S.disp[i][0], dy=S.coords[i][1]-S.disp[i][1];
      S.disp[i][0]+=dx*0.16; S.disp[i][1]+=dy*0.16; mov=Math.max(mov,Math.abs(dx),Math.abs(dy)); }
    draw();
    if(mov>0.4){ S.raf=requestAnimationFrame(step); } else { S.disp=S.coords.map(p=>p.slice()); draw(); S.raf=null; } };
  S.raf=requestAnimationFrame(step);
}
function nodeAt(mx,my){ let best=null,bd=22*DPR; for(let i=0;i<S.N;i++){ const [x,y]=w2s(S.disp[i][0],S.disp[i][1]);
  const d=Math.hypot(x-mx,y-my); if(d<bd){bd=d;best=i;} } return best; }
// pan / zoom / hover
let pan=false,lx=0,ly=0;
cv.addEventListener("mousedown",e=>{ pan=true; lx=e.offsetX*DPR; ly=e.offsetY*DPR; cv.classList.add("grab"); });
window.addEventListener("mouseup",()=>{ pan=false; cv.classList.remove("grab"); });
cv.addEventListener("mousemove",e=>{ const mx=e.offsetX*DPR,my=e.offsetY*DPR;
  if(pan){ S.view.ox+=mx-lx; S.view.oy+=my-ly; lx=mx; ly=my; draw(); return; }
  const h=nodeAt(mx,my); if(h!==S.hover){ S.hover=h; draw(); }
  const tip=$("tip");
  if(h!=null){ tip.style.display="block"; tip.style.left=(e.offsetX+12)+"px"; tip.style.top=(e.offsetY+12)+"px";
    tip.innerHTML=`<img src="/thumb_hi/${S.dataset}/${encodeURIComponent(S.names[h])}.png">`; }
  else tip.style.display="none";
});
cv.addEventListener("mouseleave",()=>{ S.hover=null; $("tip").style.display="none"; draw(); });
cv.addEventListener("wheel",e=>{ e.preventDefault(); const mx=e.offsetX*DPR,my=e.offsetY*DPR;
  const f=Math.exp(-e.deltaY*0.0015); const wx=(mx-S.view.ox)/S.view.scale, wy=(my-S.view.oy)/S.view.scale;
  S.view.scale*=f; S.view.ox=mx-wx*S.view.scale; S.view.oy=my-wy*S.view.scale; draw(); },{passive:false});

// ----------------------------------------------------------------- color-by
function buildColorBy(){ const sel=$("colorBy"); const cur=S.colorBy; sel.innerHTML="";
  const add=(v,t)=>{ const o=document.createElement("option"); o.value=v; o.textContent=t; sel.appendChild(o); };
  add("uniform","uniform");
  if(S.tags.length){ add("predicted","predicted (top class)"); add("allpred","all predicted classes"); }  // top class first = the default
  S.tags.forEach((t,k)=>add("tag:"+k,"tag: "+t));         // no ground-truth modes — classifier output only
  // until the user picks a mode, default to TOP predicted class once predictions exist (else uniform)
  if(S.colorAuto){ S.colorBy=S.pred?"predicted":"uniform"; }
  else if(![...sel.options].some(o=>o.value===cur)){ S.colorBy=S.tags.length?"predicted":"uniform"; }
  sel.value=[...sel.options].some(o=>o.value===S.colorBy)?S.colorBy:"uniform";
  sel.onchange=()=>{ S.colorBy=sel.value; S.colorAuto=false; S.selTag=S.colorBy.startsWith("tag:")?+S.colorBy.slice(4):-1; renderTags(); updateLegend(); draw(); };
}
function updateLegend(){ const lg=$("legend"); lg.innerHTML="";
  if(S.colorBy==="predicted" || S.colorBy==="allpred"){
    S.tags.forEach((t,k)=>{ const s=document.createElement("span"); s.className="sw";
      s.innerHTML=`<span class="dot" style="background:${CAT[k%CAT.length]}"></span>${t}`; lg.appendChild(s); });
    if(S.colorBy==="allpred"){ const n=document.createElement("span"); n.style.color="#789";
      n.textContent=" · pie / top-bar = every class with P≥0.5 (wedge ∝ confidence)"; lg.appendChild(n); }
  } else if(S.colorBy.startsWith("tag:")){ const k=+S.colorBy.slice(4);
    lg.innerHTML=`<span class="sw">P(<b style="color:${CAT[k%CAT.length]}">${S.tags[k]}</b>) — darker = lower</span>`; }
}

// ----------------------------------------------------------------- tags panel
function tagColor(k){ return CAT[k%CAT.length]; }
function renderTags(){ const box=$("tagList"); box.innerHTML="";
  if(!S.tags.length){ box.innerHTML='<span class="tag-empty">No tags yet — add one below, then click it to make it active and click specimens to apply.</span>'; updateTagHint(); return; }
  S.tags.forEach((t,k)=>{ const c=document.createElement("span"); c.className="tag-chip"+(S.activeTag===k?" active":"");
    c.title="click to make this the active class for labeling";
    c.innerHTML=`<span class="sw" style="background:${tagColor(k)}"></span>${t}<span class="x" title="delete tag">✕</span>`;
    c.addEventListener("click",e=>{ if(e.target.classList.contains("x")) return;
      S.activeTag = (S.activeTag===k ? -1 : k); renderTags(); renderBatch(); });   // toggle active class
    c.querySelector(".x").addEventListener("click",async e=>{ e.stopPropagation();
      busy(true); try{ const st=await api("POST",`/api/session/${S.sid}/remove_tag`,{idx:k});
        // tags shift down on delete -> reindex BOTH the active class and the color-by selection (same idx scheme)
        if(S.activeTag===k) S.activeTag=-1; else if(S.activeTag>k) S.activeTag--;
        if(S.colorBy==="tag:"+k){ S.colorBy="predicted"; S.selTag=-1; }   // fall back to the top-class default
        else if(S.colorBy.startsWith("tag:")){ const j=+S.colorBy.slice(4); if(j>k){ S.colorBy="tag:"+(j-1); S.selTag=j-1; } }
        applyState(st); refreshDendro(); toast("tag removed"); } finally{ busy(false); } });
    box.appendChild(c); });
  updateTagHint();
}
function updateTagHint(){ const h=$("tagHint"); if(!h) return;
  h.textContent = (S.activeTag>=0 && S.activeTag<S.tags.length)
    ? `active: ${S.tags[S.activeTag]} — click specimens to toggle it`
    : "click a tag to activate, then click specimens to apply"; }
async function addTag(){ const inp=$("newTag"); const name=inp.value.trim(); if(!name) return;
  const r=await api("POST",`/api/session/${S.sid}/tag`,{name}); inp.value="";
  S.tags=r.tags; S.activeTag=S.tags.indexOf(name);          // new tag becomes the active class
  buildColorBy(); renderTags(); renderBatch(); }

// ----------------------------------------------------------------- batch labeling
function renderBatch(){ const g=$("batch"); g.innerHTML="";
  const active=S.activeTag>=0 && S.activeTag<S.tags.length;
  S.batch.forEach(i=>{ const card=document.createElement("div"); card.className="card"; card.dataset.i=i;
    const work=S.work[i]||new Set(); const committed=new Set(S.labels[i]||[]);
    const dirty=work.size!==committed.size || [...work].some(t=>!committed.has(t));
    if(dirty) card.classList.add("dirty");
    if(active){ card.classList.add("clickable");                 // a class is active -> click toggles it here
      if(work.has(S.activeTag)){ card.classList.add("hasActive"); card.style.boxShadow=`inset 0 0 0 2px ${tagColor(S.activeTag)}`; } }
    card.innerHTML=`<img src="/thumb/${S.dataset}/${encodeURIComponent(S.names[i])}.png">`+
      `<div class="ctags"></div><div class="cname">${S.names[i]}</div>`;
    const ct=card.querySelector(".ctags");
    [...work].sort((a,b)=>a-b).forEach(t=>{ const ch=document.createElement("span"); ch.className="ctag"+(t===S.activeTag?" act":"");
      ch.innerHTML=`<span class="sw" style="background:${tagColor(t)}"></span>${S.tags[t]||"?"}<span class="x">✕</span>`;
      ch.querySelector(".x").addEventListener("click",e=>{ e.stopPropagation(); work.delete(t); S.work[i]=work; renderBatch(); });
      ct.appendChild(ch); });
    card.addEventListener("click",()=>{                          // click-to-toggle the active class (replaces drag-drop)
      if(S.activeTag<0 || S.activeTag>=S.tags.length){ toast("select a class first — click a tag above"); return; }
      const w=S.work[i]||new Set(); if(w.has(S.activeTag)) w.delete(S.activeTag); else w.add(S.activeTag);
      S.work[i]=w; renderBatch(); });
    g.appendChild(card); });
}
async function rollBatch(){ busy(true);
  try{ const r=await api("POST",`/api/session/${S.sid}/batch`,
        {n:+$("batchN").value||12, only_unlabeled:$("onlyUnlabeled").checked, active:$("informative").checked});
    S.batch=r.batch; S.batch.forEach(i=>{ S.work[i]=new Set(S.labels[i]||[]); }); renderBatch(); }
  finally{ busy(false); } }
async function applyLabels(){ const updates={};
  S.batch.forEach(i=>{ updates[i]=[...(S.work[i]||new Set())]; });
  busy(true);
  try{ const st=await api("POST",`/api/session/${S.sid}/apply`,
        {updates, n:+$("batchN").value||12, active:$("informative").checked});   // server auto-rolls the next batch
    applyState(st); refreshDendro();
    toast($("informative").checked ? "applied — next batch picked by uncertainty" : "labels applied — latent updated"); }
  catch(e){ toast("apply failed: "+e.message); } finally{ busy(false); }
}

// ----------------------------------------------------------------- save / load
function saveLabels(){ const data={dataset:S.dataset, tags:S.tags, labels:S.labels};
  const blob=new Blob([JSON.stringify(data,null,1)],{type:"application/json"});
  const a=document.createElement("a"); a.href=URL.createObjectURL(blob); a.download=`tags_${S.dataset}.json`;
  document.body.appendChild(a); a.click(); a.remove(); toast("labels saved"); }
async function loadLabelsFile(file){ try{ const data=JSON.parse(await file.text());
  busy(true); const st=await api("POST",`/api/session/${S.sid}/load`,{data}); applyState(st); refreshDendro(); toast(`loaded ${st.tags.length} tags`); }
  catch(e){ toast("load failed: "+e.message); } finally{ busy(false); } }

// ----------------------------------------------------------------- latent dendrogram (inline SVG)
const SVGNS="http://www.w3.org/2000/svg", XLINK="http://www.w3.org/1999/xlink";
let ddG=null, ddTx=0, ddTy=0, ddS=1, ddVBW=1, ddVBH=1, ddLast=null;
async function refreshDendro(){ if(!S.sid) return;
  try{ const d=await api("POST",`/api/session/${S.sid}/dendrogram`,{}); ddLast=d; buildDendro(d); }catch(e){} }
function buildDendro(d){ const svg=$("dendroSvg"); while(svg.firstChild) svg.removeChild(svg.firstChild);
  const W=d.step*d.n, TREE_H=W*0.22, gap=TREE_H*0.06, rootPad=TREE_H*0.12;
  const leafW=d.step*0.86, leafH=leafW/(d.leaf_aspect||2);
  const VBH=rootPad+TREE_H+gap+leafH+leafW*0.3;
  ddVBW=W; ddVBH=VBH; ddTx=0; ddTy=0; ddS=1;
  svg.setAttribute("viewBox",`0 0 ${W} ${VBH}`); svg.setAttribute("preserveAspectRatio","xMidYMid meet");
  ddG=document.createElementNS(SVGNS,"g"); svg.appendChild(ddG); applyDd();
  const yT=v=>rootPad+(1-v/d.ymax)*TREE_H;
  for(const L of d.links){ const pl=document.createElementNS(SVGNS,"polyline");
    pl.setAttribute("points", L.x.map((x,k)=>`${x},${yT(L.y[k]).toFixed(2)}`).join(" "));
    pl.setAttribute("fill","none"); pl.setAttribute("stroke",L.color);
    pl.setAttribute("stroke-width",(1.2+L.w*5).toFixed(2)); pl.setAttribute("vector-effect","non-scaling-stroke");
    pl.setAttribute("stroke-linejoin","round"); pl.setAttribute("stroke-linecap","round"); ddG.appendChild(pl); }
  const lyrow=rootPad+TREE_H+gap;
  for(const lf of d.leaves){ const cx=5+lf.pos*d.step, x=cx-leafW/2;
    const im=document.createElementNS(SVGNS,"image"); const href=`/thumb_hi/${d.dataset}/${encodeURIComponent(lf.name)}.png`;
    im.setAttributeNS(XLINK,"href",href); im.setAttribute("href",href);
    im.setAttribute("x",x.toFixed(2)); im.setAttribute("y",lyrow.toFixed(2));
    im.setAttribute("width",leafW.toFixed(2)); im.setAttribute("height",leafH.toFixed(2));
    im.setAttribute("preserveAspectRatio","xMidYMid meet"); ddG.appendChild(im);
    const rc=document.createElementNS(SVGNS,"rect"); rc.setAttribute("x",x.toFixed(2)); rc.setAttribute("y",lyrow.toFixed(2));
    rc.setAttribute("width",leafW.toFixed(2)); rc.setAttribute("height",leafH.toFixed(2)); rc.setAttribute("fill","none");
    rc.setAttribute("stroke",lf.color); rc.setAttribute("stroke-width","2"); rc.setAttribute("vector-effect","non-scaling-stroke"); ddG.appendChild(rc); }
}
function applyDd(){ if(ddG) ddG.setAttribute("transform",`translate(${ddTx} ${ddTy}) scale(${ddS})`); }
(function(){ const svg=$("dendroSvg");
  svg.addEventListener("wheel",e=>{ e.preventDefault(); if(!ddG)return; const r=svg.getBoundingClientRect();
    const sc=Math.min(r.width/ddVBW,r.height/ddVBH), offX=(r.width-ddVBW*sc)/2, offY=(r.height-ddVBH*sc)/2;
    const vx=(e.clientX-r.left-offX)/sc, vy=(e.clientY-r.top-offY)/sc, f=Math.exp(-e.deltaY*0.0015);
    ddTx=vx-(vx-ddTx)*f; ddTy=vy-(vy-ddTy)*f; ddS*=f; if(ddS<=1){ddS=1;ddTx=0;ddTy=0;} applyDd(); },{passive:false});
  let dr=false,lx=0,ly=0;
  svg.addEventListener("mousedown",e=>{dr=true;lx=e.clientX;ly=e.clientY;e.preventDefault();});
  window.addEventListener("mouseup",()=>dr=false);
  window.addEventListener("mousemove",e=>{ if(!dr||!ddG)return; const r=svg.getBoundingClientRect();
    const sc=Math.min(r.width/ddVBW,r.height/ddVBH); ddTx+=(e.clientX-lx)/sc; ddTy+=(e.clientY-ly)/sc; lx=e.clientX; ly=e.clientY; applyDd(); });
})();
function downloadDendroSvg(){ const svg=$("dendroSvg").cloneNode(true); const g=svg.querySelector("g"); if(g) g.removeAttribute("transform");
  svg.setAttribute("xmlns",SVGNS); const blob=new Blob([new XMLSerializer().serializeToString(svg)],{type:"image/svg+xml"});
  const a=document.createElement("a"); a.href=URL.createObjectURL(blob); a.download=`latent_tree_${S.dataset}.svg`; a.click(); }

// ----------------------------------------------------------------- wiring
$("load").onclick=createSession;
$("reset").onclick=()=>{ if(S.sid) createSession(); };
$("addTag").onclick=addTag;
$("newTag").addEventListener("keydown",e=>{ if(e.key==="Enter") addTag(); });
$("reroll").onclick=rollBatch;
$("apply").onclick=applyLabels;
$("saveBtn").onclick=saveLabels;
$("loadBtn").onclick=()=>$("loadFile").click();
$("loadFile").onchange=e=>{ const f=e.target.files[0]; if(f) loadLabelsFile(f); e.target.value=""; };
$("dlSvg").onclick=downloadDendroSvg;
$("showThumbs").onchange=e=>{ S.showThumbs=e.target.checked; draw(); };
// instructions modal: button opens; ✕, backdrop click, or Esc closes
function showHelp(on){ $("helpModal").classList.toggle("hidden", !on); }
$("helpBtn").onclick=()=>showHelp(true);
$("helpClose").onclick=()=>showHelp(false);
$("helpModal").addEventListener("click",e=>{ if(e.target===$("helpModal")) showHelp(false); });
window.addEventListener("keydown",e=>{ if(e.key==="Escape") showHelp(false); });
loadDatasets().then(()=>{ resize(); createSession(); });
