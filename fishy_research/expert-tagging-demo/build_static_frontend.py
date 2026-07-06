"""Regenerate the STATIC tagging demo's frontend from the served demo (single source of truth).

The served `static/app.js` is the canonical UI. The static build is identical except its `api()` dispatches
to a client-side TagEngine instead of `fetch`, and thumbnail URLs are relative. This script reapplies those
two transforms + copies style.css so the two demos never drift. (index.html is maintained separately because
its <title>/script tags differ.)

Run:  python3 build_static_frontend.py
"""
from __future__ import annotations

import hashlib
import re
import shutil
from pathlib import Path

SERVED = Path(__file__).resolve().parent / "static"
STATIC = Path(__file__).resolve().parent / "static_demo"   # the gitignored static build, alongside this script

# cache-busting: every <script>/<link> to these gets a ?v=<hash> stamped into index.html so a freshly-deployed
# index.html can never load a stale cached copy of one of them (the GitHub-Pages bug that broke the demo on
# returning visitors). The version is a content hash, so it changes iff the assets change — zero maintenance.
CACHE_BUST_ASSETS = ["app.js", "engine.js", "style.css", "lib/umap-js.min.js"]


def stamp_cache_bust(dst: Path) -> str:
    h = hashlib.sha1()
    for a in CACHE_BUST_ASSETS:
        p = dst / a
        if p.exists():
            h.update(p.read_bytes())
    ver = h.hexdigest()[:8]
    idx = dst / "index.html"
    html = idx.read_text()
    for a in CACHE_BUST_ASSETS:
        html = re.sub(rf'((?:href|src)="{re.escape(a)})(\?v=[0-9a-f]+)?"', rf'\1?v={ver}"', html)
    idx.write_text(html)
    return ver

SHIM = '''// Tagging morphospace — STATIC build. Same UI as the served demo, but the Python server/engine is replaced
// by a pure client-side TagEngine (engine.js): per-tag logistic + warm-started umap-js + Ward latent tree,
// all in the browser. `api()` below dispatches the old server routes into that engine; data + thumbnails are
// baked (bundle_<ds>.json, thumb/, thumb_hi/). Validated to match the Python engine (logistic err <1e-2).
//
// GENERATED from ../static/app.js by build_static_frontend.py — do not edit by hand.

// ===================== static backend: TagEngine instead of the HTTP server =====================
const BUNDLES = {};                       // ds -> parsed bundle json (lazy)
let ENG = null;                            // the single live TagEngine
const DATASETS = [
  {name:"mussel", title:"Mussel (real, n=31)",      has_gt:false},
  {name:"fishy",  title:"Fishy (synthetic, n=250)", has_gt:false},
];
async function _loadBundle(ds){ if(!BUNDLES[ds]) BUNDLES[ds]=await fetch(`bundle_${ds}.json`).then(r=>r.json()); return BUNDLES[ds]; }
const _yield = ()=>new Promise(r=>setTimeout(r,0));   // let the 'updating…' indicator paint before heavy compute
async function api(method, path, body){
  body = body || {};
  if(path==="/api/datasets") return {datasets:DATASETS};
  if(path==="/api/session"){ const b=await _loadBundle(body.dataset); await _yield();
    ENG=new TagEngine(b, 0); ENG.rollBatch(+body.batch||12, true, true); return Object.assign({sid:"static"}, ENG.state()); }
  if(!ENG) throw new Error("no session");
  const m=path.match(/\\/api\\/session\\/[^/]+\\/(\\w+)$/), op=m?m[1]:"";
  if(op==="tag"){ ENG.addTag(body.name); return {tags:ENG.tags.slice()}; }
  if(op==="remove_tag"){ await _yield(); ENG.removeTag(body.idx|0); return ENG.state(); }
  if(op==="apply"){ await _yield(); ENG.applyLabels(body.updates||{}); ENG.rollBatch(+body.n||12, true, body.active!==false); return ENG.state(); }
  if(op==="batch"){ ENG.rollBatch(+body.n||12, body.only_unlabeled!==false, body.active!==false); return {batch:ENG.batch.slice()}; }
  if(op==="dendrogram"){ return ENG.dendrogram(28); }
  if(op==="load"){ await _yield(); ENG.importState(body.data||{}); return ENG.state(); }
  throw new Error("static demo: unsupported endpoint "+path);
}
// ===================== end static backend =====================

'''

# the served fetch-based api() block, removed verbatim (the shim above defines api())
OLD_API = '''async function api(method, path, body){
  const o={method, headers:{"Content-Type":"application/json"}};
  if(body) o.body=JSON.stringify(body);
  const r=await fetch(path,o);
  if(!r.ok){ const e=await r.json().catch(()=>({error:r.statusText})); throw new Error(e.error||r.statusText); }
  return r.json();
}
'''


def main():
    src = (SERVED / "app.js").read_text()
    assert OLD_API in src, "served api() block not found verbatim — update OLD_API"
    src = src.replace(OLD_API, "")
    src = src.replace("/thumb_hi/${", "thumb_hi/${").replace("/thumb/${", "thumb/${")
    assert '"/thumb' not in src and "`/thumb" not in src, "leftover absolute thumb path"
    (STATIC / "app.js").write_text(SHIM + src)
    shutil.copy(SERVED / "style.css", STATIC / "style.css")
    ver = stamp_cache_bust(STATIC)
    print(f"wrote {STATIC/'app.js'} ({len((SHIM+src).splitlines())} lines)")
    print(f"copied style.css -> {STATIC/'style.css'}")
    print(f"stamped index.html assets with ?v={ver}")


if __name__ == "__main__":
    main()
