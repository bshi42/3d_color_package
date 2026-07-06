# Tagging Morphospace — multi-label labeling → classifier → latent space

A browser demo where an expert **creates tags** ("green tail", "4 stripes", …), **clicks a tag to make it the
active class**, then **clicks specimens to toggle that class** on them (multi-label). After each batch a per-tag
linear classifier is retrained and its tag-score
**latent space** is projected to 2-D with a **warm-started UMAP** — so the morphospace reacts from the very
first batch and evolves smoothly as more labels arrive. The tree view clusters the same latent. No pairwise
"similarity / anchor" machinery — this replaces the old similarity demo.

Design validated offline as **EXP-46..51** in the `fishy_research` repo (tag feedback ≫ pairwise for label
efficiency; per-tag logistic best placer; warm-started UMAP gives reactive-yet-smooth updates at ~0.2 s/batch).

## Run

This demo lives inside the repo at `fishy_research/expert-tagging-demo/` (alongside `interactive_demo/`). It
uses the `fishy_research` virtualenv (numpy / scipy / scikit-learn / **umap-learn** / matplotlib / Pillow) and
that repo's dataset + descriptor pipeline. `FISHPIPE_ROOT` defaults to the repo root one level up; override it
only if the data lives elsewhere. Thumbnails are reused from the similarity demo's cache, so nothing re-renders.

```bash
cd fishy_research/expert-tagging-demo
../.venv/bin/python server.py --port 8770            # -> http://127.0.0.1:8770
```

Pick a dataset (**mussel** n=31 = instant; **fishy** n=250 = a few seconds), then:

- **Tags panel** — type a name + **Add** (the new tag becomes active); **click** a tag chip to make it the
  **active class** (click again to deactivate); **✕** on a chip deletes the tag everywhere.
- **Label a batch** — set the **batch size**, **↻ Re-roll** (with **only unlabeled** on/off); with a class
  active, **click a specimen card to toggle that class** on/off (**✕** on a card chip removes one), then
  **Apply labels →** to retrain + re-project. Apply then **auto-rolls the next, most-informative batch** — the
  unlabeled specimens the classifier is most unsure about (uncertainty sampling; **informative** toggle,
  EXP-52: reaches ~0.96 ARI by 80 labels vs ~0.81 random).
- **color by** — uniform / all predicted classes / predicted top class / a specific tag's probability
  (classifier output only — no ground-truth colouring).
- **Save / Load labels** — JSON of `{dataset, tags, labels}`.
- The **latent-space tree** (bottom) clusters the classifier's latent; zoom/pan + SVG export.

## Files
```
engine.py                  TagSession: descriptor+PCA, per-tag logistic, warm-started UMAP, latent Ward tree, save/load
server.py                  stdlib http.server JSON API + thumbnail serving (reuses the similarity demo's thumb cache)
static/                    index.html · app.js (morphospace + tagging UI) · style.css   ← the served frontend (source of truth)
build_static_frontend.py   regenerate static_demo/app.js + style.css from static/ (keeps the static build in parity)
export_tagging_static.py   bake static_demo/bundle_<ds>.json + thumbnails (needs the venv)
static_demo/               server-free browser build (see static_demo/README.md). Gitignored except the hand-authored
                           source (engine.js, index.html, lib/, README) — app.js/style.css/bundles/thumbs are regenerated.
```

## Static (server-free) build

A fully client-side version lives in `static_demo/` — same UI, but a JS `TagEngine` (engine.js) replaces the
Python server. Rebuild it from this directory with the venv:

```bash
../.venv/bin/python export_tagging_static.py     # bake bundles + thumbnails
python3 build_static_frontend.py                 # regenerate app.js + style.css from static/
python3 -m http.server 8771 -d static_demo       # serve -> http://127.0.0.1:8771
```
