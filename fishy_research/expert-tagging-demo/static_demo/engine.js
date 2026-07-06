// Pure client-side port of the Python TagSession (engine.py) — NO server, NO DOM.
// Runs in the browser (global `TagEngine`, uses window.UMAP.UMAP) and in node (module.exports, require umap-js)
// so the exact same logic can be unit-tested offline against the baked bundles.
//
// What it reproduces from engine.py:
//   * per-tag linear logistic classifier  (C=0.5, class_weight='balanced')  -> tag-score latent
//   * warm-started umap-js projection of that latent  (init = previous embedding, ~60 epochs)
//   * cold-start NOVELTY seeding + uncertainty active-learning batch rolls
//   * Ward dendrogram of the current latent (subsampled), depth-aligned, thickness = merge distance
(function (root, factory) {
  if (typeof module === "object" && module.exports) {
    module.exports = factory(function () { return require("umap-js").UMAP; });
  } else {
    root.TagEngine = factory(function () { return (root.UMAP && root.UMAP.UMAP) || root.UMAP; });
  }
})(typeof self !== "undefined" ? self : this, function (getUMAP) {
  "use strict";

  // ----------------------------------------------------------------- tiny rng (LCG; seedable, browser-safe)
  function lcg(seed) { let s = (seed >>> 0) || 1; return () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; }; }

  // ----------------------------------------------------------------- linear algebra
  function solve(A, b) {                       // Gaussian elimination w/ partial pivot; A:n×n (mutated), b:n
    const n = b.length, M = A.map((r, i) => r.slice().concat(b[i]));
    for (let c = 0; c < n; c++) {
      let p = c; for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
      const t = M[c]; M[c] = M[p]; M[p] = t;
      const piv = M[c][c] || 1e-12;
      for (let r = 0; r < n; r++) { if (r === c) continue; const f = M[r][c] / piv;
        for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k]; }
    }
    return M.map((r, i) => r[n] / (r[i] || 1e-12));
  }

  // ----------------------------------------------------------------- logistic regression (IRLS / Newton)
  // sklearn LogisticRegression(C, class_weight='balanced'), NO penalty on intercept.
  // objective: 0.5||w||^2 + C * Σ sw_i * NLL_i ;  returns θ=[w(D), intercept].
  function fitLogistic(X, y, C) {
    const n = X.length, D = X[0].length, P = D + 1;
    // balanced sample weights: n / (2 * count(class))
    let n1 = 0; for (let i = 0; i < n; i++) n1 += y[i]; const n0 = n - n1;
    const w1 = n / (2 * Math.max(1, n1)), w0 = n / (2 * Math.max(1, n0));
    const sw = new Array(n); for (let i = 0; i < n; i++) sw[i] = y[i] ? w1 : w0;
    const phi = X.map(r => r.concat(1));                 // augment intercept
    let th = new Array(P).fill(0);
    for (let it = 0; it < 30; it++) {
      const g = new Array(P).fill(0);
      const H = Array.from({ length: P }, () => new Array(P).fill(0));
      for (let i = 0; i < n; i++) {
        let z = 0; const pr = phi[i]; for (let j = 0; j < P; j++) z += pr[j] * th[j];
        const p = 1 / (1 + Math.exp(-z)), wgt = C * sw[i] * p * (1 - p), r = C * sw[i] * (p - y[i]);
        for (let a = 0; a < P; a++) {
          g[a] += r * pr[a];
          const ha = wgt * pr[a]; const Ha = H[a];
          for (let b = a; b < P; b++) Ha[b] += ha * pr[b];
        }
      }
      for (let a = 0; a < D; a++) { g[a] += th[a]; H[a][a] += 1; }   // L2 on coeffs only
      for (let a = 0; a < P; a++) { H[a][a] += 1e-8; for (let b = 0; b < a; b++) H[a][b] = H[b][a]; }
      const step = solve(H, g);
      let mx = 0; for (let a = 0; a < P; a++) { th[a] -= step[a]; mx = Math.max(mx, Math.abs(step[a])); }
      if (mx < 1e-7) break;
    }
    return th;
  }
  function decisionAll(Z, th) {                 // raw scores Z·w + b for every row
    const D = th.length - 1, b = th[D];
    return Z.map(r => { let z = b; for (let j = 0; j < D; j++) z += r[j] * th[j]; return z; });
  }
  const sig = z => 1 / (1 + Math.exp(-z));

  // ----------------------------------------------------------------- stats helpers
  function colMean(M) { const n = M.length, d = M[0].length, m = new Array(d).fill(0);
    for (const r of M) for (let j = 0; j < d; j++) m[j] += r[j]; return m.map(v => v / n); }
  function zscoreCols(M) { const n = M.length, d = M[0].length, m = colMean(M), sd = new Array(d).fill(0);
    for (const r of M) for (let j = 0; j < d; j++) { const e = r[j] - m[j]; sd[j] += e * e; }
    for (let j = 0; j < d; j++) sd[j] = Math.sqrt(sd[j] / n) + 1e-9;
    return M.map(r => r.map((v, j) => (v - m[j]) / sd[j])); }
  function eucl(a, b) { let s = 0; for (let k = 0; k < a.length; k++) { const d = a[k] - b[k]; s += d * d; } return Math.sqrt(s); }

  // ----------------------------------------------------------------- umap projection (warm-started)
  function umapProject(X, prevEmb, seed) {
    const UMAP = getUMAP();
    const nn = Math.max(2, Math.min(15, X.length - 1));
    if (!prevEmb || prevEmb.length !== X.length) {          // cold: full fit
      return new UMAP({ nComponents: 2, nNeighbors: nn, minDist: 0.12, random: lcg(seed) }).fit(X);
    }
    const EP = 60;
    const u = new UMAP({ nComponents: 2, nNeighbors: nn, minDist: 0.12, random: lcg(seed), nEpochs: EP });
    u.initializeFit(X);
    // inject the previous embedding (scaled/centred to match umap-js's own init magnitude) as the warm start
    const os = u.optimizationState, cur = os.headEmbedding, n = X.length;
    const cm = colMean(cur), pm = colMean(prevEmb);
    let cs = 0, ps = 0; for (let i = 0; i < n; i++) for (let k = 0; k < 2; k++) {
      cs += (cur[i][k] - cm[k]) ** 2; ps += (prevEmb[i][k] - pm[k]) ** 2; }
    const scale = Math.sqrt(cs / n) / (Math.sqrt(ps / n) + 1e-9);
    for (let i = 0; i < n; i++) for (let k = 0; k < 2; k++) {
      const v = (prevEmb[i][k] - pm[k]) * scale + cm[k];
      cur[i][k] = v; if (os.tailEmbedding && os.tailEmbedding !== cur) os.tailEmbedding[i][k] = v;
    }
    while (u.step() < EP) { /* optimise from the warm start */ }
    return u.getEmbedding();
  }

  // ----------------------------------------------------------------- Ward dendrogram (Lance-Williams, any-dim)
  const _PALETTE = ["#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#bcbd22", "#17becf", "#1f77b4"];
  const _GRAY = "#b8b8b8";
  function wardLinkage(P) {                     // P: m×d ; returns m-1 merges ascending in distance
    const n = P.length, sz = new Array(2 * n - 1).fill(0); for (let i = 0; i < n; i++) sz[i] = 1;
    const D = {}; for (let i = 0; i < n; i++) D[i] = {};
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) D[i][j] = eucl(P[i], P[j]);
    const gd = (i, j) => i < j ? D[i][j] : D[j][i];
    const active = new Set(); for (let i = 0; i < n; i++) active.add(i);
    const merges = []; let next = n;
    while (active.size > 1) {
      const arr = [...active]; let bi = arr[0], bj = arr[1], bd = Infinity;
      for (let a = 0; a < arr.length; a++) for (let b = a + 1; b < arr.length; b++) { const d = gd(arr[a], arr[b]); if (d < bd) { bd = d; bi = arr[a]; bj = arr[b]; } }
      const id = next++; sz[id] = sz[bi] + sz[bj]; merges.push({ a: bi, b: bj, dist: bd, id });
      active.delete(bi); active.delete(bj); D[id] = {};
      for (const k of active) { const ni = sz[bi], nj = sz[bj], nk = sz[k], T = ni + nj + nk;
        const dik = gd(bi, k), djk = gd(bj, k);
        const dn = Math.sqrt(((ni + nk) / T) * dik * dik + ((nj + nk) / T) * djk * djk - (nk / T) * bd * bd);
        if (id < k) D[id][k] = dn; else (D[k] = D[k] || {})[id] = dn; }
      active.add(id);
    }
    return merges;
  }
  // build the SVG-renderer payload (mirror server dendrogram(): depth-aligned levels, thickness = merge dist)
  function dendroPayload(latent, names, idx, dataset, leafAspect) {
    const sub = idx.map(i => latent[i]);
    const Q = zscoreCols(sub), n = Q.length;
    const merges = wardLinkage(Q), m = merges.length;
    const childA = {}, childB = {}, parent = {}, mh = {}, leavesOf = {};
    for (let i = 0; i < n; i++) leavesOf[i] = [i];
    merges.forEach((mg, k) => { childA[mg.id] = mg.a; childB[mg.id] = mg.b; parent[mg.a] = mg.id; parent[mg.b] = mg.id;
      mh[mg.id] = k + 1; leavesOf[mg.id] = leavesOf[mg.a].concat(leavesOf[mg.b]); });
    const rootId = merges[m - 1].id, pos = {}; let pp = 0;
    const minIdx = id => Math.min.apply(null, leavesOf[id]);
    (function visit(id) { if (id < n) { pos[id] = pp++; return; } let a = childA[id], b = childB[id];
      if (minIdx(a) > minIdx(b)) { const t = a; a = b; b = t; } visit(a); visit(b); })(rootId);
    const cx = {}, lvl = {};
    function cxOf(id) { if (id in cx) return cx[id];
      if (id < n) { cx[id] = 5 + 10 * pos[id]; lvl[id] = 0; }
      else { cx[id] = (cxOf(childA[id]) + cxOf(childB[id])) / 2; lvl[id] = 1 + Math.max(lvl[childA[id]], lvl[childB[id]]); }
      return cx[id]; }
    for (const mg of merges) cxOf(mg.id);
    const ymax = Math.max.apply(null, merges.map(mg => lvl[mg.id]));
    const td = merges.map(mg => mg.dist), dmin = Math.min.apply(null, td);
    const sd = [...td].sort((a, b) => a - b), dhi = sd[Math.floor(0.95 * (sd.length - 1))];
    const kc = Math.max(2, Math.min(6, Math.max(2, n - 1))), ct = m - kc + 1.5;
    const climb = id => { let c = id; while (parent[c] !== undefined && mh[parent[c]] < ct) c = parent[c]; return c; };
    const roots = merges.filter(mg => mh[mg.id] < ct && (parent[mg.id] === undefined || mh[parent[mg.id]] >= ct)).map(mg => mg.id);
    roots.sort((a, b) => Math.min.apply(null, leavesOf[a].map(l => pos[l])) - Math.min.apply(null, leavesOf[b].map(l => pos[l])));
    const rc = {}; roots.forEach((r, i) => rc[r] = _PALETTE[i % _PALETTE.length]);
    const linkColor = id => mh[id] >= ct ? _GRAY : rc[climb(id)];
    const leafColor = l => { const pm = parent[l]; return (pm === undefined || mh[pm] >= ct) ? _GRAY : rc[climb(pm)]; };
    const links = merges.map(mg => { const id = mg.id, xa = cx[mg.a], xb = cx[mg.b];
      const x = [xa, xa, xb, xb], y = [lvl[mg.a], lvl[id], lvl[id], lvl[mg.b]];
      const w = Math.max(0, Math.min(1, (mg.dist - dmin) / (dhi - dmin + 1e-9)));
      const ps = leavesOf[id].map(l => pos[l]);
      return { x: x.map(v => +v.toFixed(2)), y: y.map(v => +v.toFixed(2)), color: linkColor(id), w: +w.toFixed(3),
               span: [Math.min.apply(null, ps), Math.max.apply(null, ps)] }; });
    const leaves = []; for (let i = 0; i < n; i++) leaves.push({ pos: pos[i], i: idx[i], name: names[idx[i]], color: leafColor(i) });
    leaves.sort((a, b) => a.pos - b.pos);
    return { n, ymax, step: 10, leaf_aspect: leafAspect, links, leaves, dataset };
  }

  // ----------------------------------------------------------------- TagEngine (mirrors TagSession)
  class TagEngine {
    constructor(bundle, seed) {
      this.dataset = bundle.dataset; this.title = bundle.title; this.hasGt = !!bundle.has_gt;
      this.names = bundle.names; this.Z = bundle.Z; this.novelty = bundle.novelty;
      this.leafAspect = bundle.leaf_aspect || 2; this.N = this.names.length; this.seed = seed || 0;
      this.tags = []; this.labels = {};              // labels[i] = array of tag idxs
      this.pred = null; this.prevEmb = null; this.latent = this.Z;
      this.coords = umapProject(this.Z, null, this.seed);   // unsupervised morphospace
      this.prevEmb = this.coords.map(p => p.slice());
      this.batch = [];
    }
    _labeledIdx() { return Object.keys(this.labels).map(Number).filter(i => this.labels[i] && this.labels[i].length).sort((a, b) => a - b); }

    _tagScores() {                                 // (N×T) raw decision scores; constant fallback at tiny n
      const T = this.tags.length, lab = this._labeledIdx();
      const S = Array.from({ length: this.N }, () => new Array(T).fill(0));
      for (let t = 0; t < T; t++) {
        const y = lab.map(i => this.labels[i].indexOf(t) >= 0 ? 1 : 0);
        const pos = y.reduce((s, v) => s + v, 0);
        if (lab.length < 2 || pos === 0 || pos === y.length) {
          const c = y.length ? pos / y.length : 0; for (let i = 0; i < this.N; i++) S[i][t] = c; continue;
        }
        const th = fitLogistic(lab.map(i => this.Z[i]), y, 0.5), sc = decisionAll(this.Z, th);
        for (let i = 0; i < this.N; i++) S[i][t] = sc[i];
      }
      return S;
    }
    recompute() {
      if (!this.tags.length || !this._labeledIdx().length) {
        this.latent = this.Z; this.pred = null;
        this.coords = umapProject(this.Z, this.prevEmb, this.seed);
        this.prevEmb = this.coords.map(p => p.slice()); return;
      }
      const S = this._tagScores(); this.latent = S;
      this.coords = umapProject(S, this.prevEmb, this.seed);
      this.prevEmb = this.coords.map(p => p.slice());
      this.pred = S.map(row => row.map(sig));
    }
    addTag(name) { name = String(name || "").trim(); if (name && this.tags.indexOf(name) < 0) this.tags.push(name); return this.tags.indexOf(name); }
    removeTag(idx) {
      if (idx < 0 || idx >= this.tags.length) return;
      this.tags.splice(idx, 1); const next = {};
      for (const k of Object.keys(this.labels)) {
        const nt = this.labels[k].filter(t => t !== idx).map(t => t < idx ? t : t - 1);
        if (nt.length) next[k] = nt;
      }
      this.labels = next; this.recompute();
    }
    applyLabels(updates) {
      for (const k of Object.keys(updates)) {
        const ts = (updates[k] || []).map(Number).filter(t => t >= 0 && t < this.tags.length);
        if (ts.length) this.labels[k] = Array.from(new Set(ts)).sort((a, b) => a - b); else delete this.labels[k];
      }
      this.recompute();
    }
    rollBatch(n, onlyUnlabeled, active) {
      n = Math.max(1, n | 0);
      const unlabeled = []; for (let i = 0; i < this.N; i++) if (!(this.labels[i] && this.labels[i].length)) unlabeled.push(i);
      if (active && unlabeled.length) {
        if (this.pred) {                            // uncertainty: highest mean per-tag |2p-1| nearness to 0.5
          const u = unlabeled.map(i => { let s = 0; for (const p of this.pred[i]) s += 1 - Math.abs(2 * p - 1); return [i, s / this.pred[i].length]; });
          u.sort((a, b) => b[1] - a[1]); this.batch = u.slice(0, n).map(x => x[0]);
        } else {                                    // cold start: novelty band, randomised within the top band
          const ranked = unlabeled.slice().sort((a, b) => this.novelty[b] - this.novelty[a]);
          const band = ranked.slice(0, Math.max(n, Math.min(2 * n, ranked.length)));
          for (let i = band.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); const t = band[i]; band[i] = band[j]; band[j] = t; }
          this.batch = band.slice(0, Math.min(n, band.length)).sort((a, b) => this.novelty[b] - this.novelty[a]);
        }
        return this.batch;
      }
      let pool = (onlyUnlabeled && unlabeled.length) ? unlabeled.slice() : Array.from({ length: this.N }, (_, i) => i);
      for (let i = pool.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); const t = pool[i]; pool[i] = pool[j]; pool[j] = t; }
      this.batch = pool.slice(0, Math.min(n, pool.length)).sort((a, b) => a - b);
      return this.batch;
    }
    importState(d) {
      if (d.dataset && d.dataset !== this.dataset) throw new Error(`labels are for '${d.dataset}', current dataset is '${this.dataset}'`);
      this.tags = (d.tags || []).map(String);
      this.labels = {};
      const L = d.labels || {};
      for (const k of Object.keys(L)) { const v = (L[k] || []).map(Number); if (v.length) this.labels[k] = v; }
      this.recompute();
    }
    exportState() {
      const labels = {}; for (const k of this._labeledIdx()) labels[k] = this.labels[k].slice().sort((a, b) => a - b);
      return { dataset: this.dataset, tags: this.tags.slice(), labels };
    }
    state() {
      const labels = {}; for (const k of this._labeledIdx()) labels[k] = this.labels[k].slice().sort((a, b) => a - b);
      return {
        dataset: this.dataset, title: this.title, has_gt: this.hasGt, names: this.names, N: this.N,
        tags: this.tags.slice(), labels,
        coords: this.coords.map(p => [+p[0].toFixed(4), +p[1].toFixed(4)]),
        pred: this.pred ? this.pred.map(r => r.map(v => +v.toFixed(3))) : null,
        batch: this.batch.slice(),
      };
    }
    dendrogram(k) {
      k = k || 28; let idx;
      if (this.N <= k + 6) { idx = Array.from({ length: this.N }, (_, i) => i); }
      else {                                        // deterministic subsample (seeded shuffle), then sort
        const r = lcg(this.seed + 1), a = Array.from({ length: this.N }, (_, i) => i);
        for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); const t = a[i]; a[i] = a[j]; a[j] = t; }
        idx = a.slice(0, k).sort((x, y) => x - y);
      }
      return dendroPayload(this.latent, this.names, idx, this.dataset, this.leafAspect);
    }
  }
  TagEngine.fitLogistic = fitLogistic; TagEngine.umapProject = umapProject;   // exposed for tests
  return TagEngine;
});
