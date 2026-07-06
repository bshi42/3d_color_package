# Live color + pattern morphospace — force-directed feedback graph

A browser demo where specimens are nodes in a **live force-directed graph**: springs link each
to its nearest neighbours in the combined **color + pattern** descriptor, a continuous physics
simulation lays it out, and **you shape it by hand**. Drag any node and the rest react; rank an
anchor's neighbours by similarity and the springs retune (similar → pulled together, dissimilar →
pushed apart) so the whole graph re-settles. A dendrogram clusters the live on-screen layout, and a
heatmap shows where any PC lives on the body.

Runs on two datasets:
- **fishy** — synthetic, n=250, has ground truth (belly / tail / stripe / cheeks).
- **mussel** — real, n=31, no ground truth (human-in-the-loop only).

> **Static (server-free) build:** `export_static.py` bakes the mussel dataset into a fully
> frontend-only demo under `static_demo/` (precomputed graph + PCA coords + exemplar PNGs; Ward
> dendrogram and panel sampling ported to JS). It needs only a dumb static host. See
> [`static_demo/README.md`](static_demo/README.md).

## Run

No new dependencies — uses the existing `fishy_research` env (numpy / scipy / sklearn /
matplotlib / Pillow) and the Python **stdlib** web server.

```bash
cd fishy_research
.venv/bin/python interactive_demo/server.py            # http://127.0.0.1:8000
.venv/bin/python interactive_demo/server.py --port 8765 --warm fishy,mussel
```

`--warm` pre-renders specimen thumbnails at boot. First open of a dataset renders thumbnails once
(~25 s for fishy's 250, ~15 s for mussel's 31), cached to `interactive_demo/_cache/thumbs/`.

Open the URL, pick a dataset, click **Load**.

## How to use it

- The graph is **live** — it animates as the physics settles (the badge reads *settling* / *settled*).
- **Drag any node**: it pins to the cursor and its spring neighbours follow; release and the graph
  relaxes. This is the "grab a node, the rest react" behaviour.
- **Pan** with a background drag, **zoom** with the wheel, **hover** to enlarge a thumbnail.
- **Give feedback**: an anchor (gold) and a diverse panel appear on the right. **Click** a panel
  specimen to mark it **similar** (green, pulled closer); **click again** → **dissimilar** (red,
  pushed apart); again → clear. On a marked card, **scroll** or drag the **vertical slider** to set
  the amplitude 0–1 (how strongly). **Apply → reshape** adds the *preference springs* and the graph
  re-settles. At amplitude 1 the involved nodes "let go" of their other links so the pair snaps adjacent.
- **Click a node** to make it the anchor; **Ctrl-click** a node to add/remove it from the ranking set
  (on top of the auto-sample); **New anchor** sweeps to a fresh diverse one.
- **Explain grouping** fits which PCA components account for your feedback (metric learning), with a
  body heatmap, color/pattern split, significance, held-out coverage, and leave-one-out placement.
- **⊞ Recover** learns a low-rank metric from your graded feedback, clusters into *N* groups (GMM),
  recolours the morphospace by recovered group (with the GT-ARI on fishy), and enables **Factor grid** —
  which snaps the specimens into a structured 4-quadrant × split layout so you can *see* the recovered
  groups (a 2-D force layout can't settle into ≥8). This is the EXP-42..45 recipe (graded target
  distances + rank-4 metric + K≈24 PCs); see `RESEARCH_LOG.md`.
- **Save / Load feedback** writes/reads the preference set as JSON (validates the dataset on load).
- **color by** recolours nodes by a ground-truth factor (fishy only; mussel nodes are uniform).
- **Reset feedback** clears the preference springs; **Reset layout** reseeds from the PCA layout.
- The **dendrogram** (inline SVG, depth-aligned, line thickness = merge distance) zooms/pans and
  exports as **SVG/PNG**; tick **rotate 90°** to export a horizontal tree instead (root at the left,
  branches growing rightward, specimen images upright down the right edge). **springs** toggles the edge
  overlay; **freeze** pauses the simulation.
- **hide unlabeled** / **hide labeled** filter the 2-D graph by feedback status — a specimen is
  *labeled* once it is an endpoint of a feedback spring (committed similar/dissimilar). The two are
  mutually exclusive (show all / only labeled / only unlabeled). The filter is purely visual (the
  physics layout is untouched) and hidden nodes are non-interactive, but the **active working set**
  (current anchor, its ranking candidates, a node being dragged) always stays visible so the graph
  never disagrees with the panel; the dendrogram is unaffected.
- **Select in the dendrogram** (same as the graph): **click** a leaf to make it the anchor;
  **Ctrl-click** a leaf to add/remove it from the ranking set, or **Ctrl-click** a branch to toggle its
  whole clade. Selected leaves get a gold ring and stay in sync with the graph (cyan ring) and the panel.

## What's under the hood

- **Descriptor** (`engine.py`): the body is partitioned into regions (k-means on face centroids,
  R=128 fishy / 48 mussel); each region contributes an 11-dim **color + texture** block (5 Lab/chroma
  + 6 band-pass/edge/dark), concatenated into the per-specimen descriptor that PCA + the kNN graph
  run on.
- **Similarity graph**: a kNN graph (k=8) over the standardized descriptor, plus an MST so it's
  **connected** (a tug anywhere can propagate everywhere). Edge rest lengths are high-D distances
  rescaled to the PCA-2D seed, so the layout starts near equilibrium then relaxes.
- **Physics** (`static/app.js`): a compact client-side force solver — many-body repulsion + link
  springs + light gravity, velocity-damped with a decaying `alpha` that reheats on drag/feedback.
  Feedback ranking adds preference springs (rest length grows with rank); the connected graph
  ripples the change outward and re-settles globally.
- **Dendrogram**: server-rendered (matplotlib) Ward linkage on a diverse subset of the **client's
  current 2-D layout** (WYSIWYG), with specimen-thumbnail leaves, refreshed when the graph settles.
- **Renders**: the repo's CPU `render_side_view` (fishy lateral) / `render_textured_rgba` (mussel
  valve) produce the specimen thumbnails.

## Files

```
interactive_demo/
  engine.py        datasets, per-region descriptor, PCA, similarity graph, panels (pure compute)
  server.py        stdlib http.server: JSON API + thumbnail/heatmap/dendrogram PNG rendering
  static/          index.html · app.js (force sim + ranking UI) · style.css
  _cache/          rendered thumbnails, region-label caches, logs  (git-ignored)
  _probe.py        standalone smoke test of the descriptor/render pipeline on both datasets
```

## API

| method | path | purpose |
|---|---|---|
| GET  | `/api/datasets` | list datasets |
| POST | `/api/session` `{dataset}` | create a session → `{sid, K, names, has_gt, evr, …}` |
| GET  | `/api/session/{sid}/graph` | nodes (PCA-seed xy), kNN+MST edges, per-PC values, GT |
| GET  | `/api/session/{sid}/panel?anchor=` | anchor + diverse neighbours + remote check |
| POST | `/api/session/{sid}/dendrogram` `{positions}` | WYSIWYG dendrogram of the live layout (PNG) |
| GET  | `/thumb/{dataset}/{name}.png` | specimen thumbnail |
| POST | `/api/session/{sid}/feedback` (log) · `/reset` | controls |

The layout physics and preference springs run **client-side** (`app.js`); the server is a data +
render service. Static JS/CSS are served uncached for dev (edit + hard-reload to see changes).
