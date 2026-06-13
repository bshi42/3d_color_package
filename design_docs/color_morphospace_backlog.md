# Backlog — Structural/Pattern Morphospace + Exemplar Slider (deca3)

**Status:** proposed · **Created:** 2026-06-01 · **Implementation method:** RPI (Research → Plan → Implement), one story at a time.
**Validated reference / oracle:** `fishy_research/` (the `fishpipe` package + `RESEARCH_LOG.md` + `docs/RECOMMENDATIONS.md`). Every algorithm below is already prototyped and quantitatively validated there; this backlog ports it into the Slicer module incrementally.

---

## 0. Context, principles, and anchors (read before any story)

**What we're building.** The color morphospace already recovers *hue* clusters (belly/tail) but is blind to *structure* (stripe count, pattern) — which is what biologists care about. We add: (1) a measured **Pattern view**, (2) **automatic cluster coloring**, (3) an **exemplar slider**, (4) optional **region-segmentation** compression, (5) a **rare-variant** overlay. UI cost target: **one toggle (Color | Pattern)** + automatic behavior.

**Guiding principle (validated, EXP-04).** *Discovery ≠ measurement.* Variance-driven reduction (PCA/ICA/UMAP) only surfaces high-variance factors (color). Subtle structure (stripe count) must be **directly measured / described** with a label-free descriptor and surfaced as a **named axis** or via auto-clustering — it will never be a top PC. Full evidence in `fishy_research/RESEARCH_LOG.md`.

**Generality principle (validated, EXP-07) — important.** The structural engine MUST generalize to arbitrary biological patterns (spots, stripes, reticulation, blotches, ocelli…), not just "count stripes". The **general graph-spectral / manifold-harmonic descriptor** does this (no pattern/axis/color assumptions) and actually *beats* a taxon-tuned stripe detector (0.93 vs 0.89) while one descriptor recovered all four different feature types. → **graph-spectral is the PRIMARY structural engine; the body-axis "banding count" detector is an OPTIONAL, taxon-specific plug-in, not the engine.** Build a small **descriptor library** (spectral + patch-adjacency + chromatic novelty) and **auto-rank** descriptors by how much they cluster the population.

**Validated result we are reproducing in-module** (balanced CV accuracy): belly/tail 1.00 (unchanged); **stripe 0.64 → 0.89** via measured banding axis; **cheeks 0.48 → 0.78** via graph-spectral chromatic descriptor.

**The live code path.** Morphospace = inline **MultiRecolor Step 3 (Population Analysis)** + **Step 4 (Morphospace)** in `InterDeCA.py`. `pca_morphospace/PCAMorphospace.py` is **legacy / not wired in** (see G.4). Target the inline path.

**Slider invertibility constraint.** The generative slider does `model.inverse_transform(coord)` → reconstructed colors → paint mesh. It needs (a) an invertible reducer (**PCA/ICA yes, UMAP no**) and (b) a representation that reconstructs to displayable colors (**subsampled-flatten yes; area-histogram is not supported — errors at 6096**). Measured pattern axes are **lossy/non-invertible** → they require the **exemplar slider** (Epic E), not generative reconstruction.

**Reuse, don't rebuild — existing helpers (`InterDeCA.py`):**
| helper | line | reuse for |
|---|---|---|
| `performPopulationAnalysis` | 11527 | add pattern feature branch (~11671) |
| `_calculateFaceAverageColors` | 8809 | per-face colors (parity oracle for A.1) |
| `_buildFaceAdjacencyGraph` | 10549 | face graph for spectral/structural descriptors |
| `_calculateFaceAreas` | 10452 | area weighting |
| `_subsampleFacesUniformly` | 10734 | region/compression + subsample slider rep |
| `rgb_to_lab` | 9427 | color space |
| `ClusteringPipeline` | 6368 | already carries `faceAdjacency`, `subsampledFaceIndices` |
| `applyTextureToModel` / `applyTextureWithSubsamplingOnly` | 7655 / 11155 | exemplar texture display |
| morphospace slider chain | 5694/5709 → 6014 → 11450 | dual-view + exemplar slider |
| Step 3/4 UI | 1041–1198 | minimal new widgets |
| `ColorTheme` | 12008 | consistent styling |
| dep guards | 47–70 | gate scipy/sklearn/umap/skimage |

**RPI note.** Each story below becomes its own RPI story file (Background, Reasoning, Acceptance Criteria, Research, Implementation Plan, Testing Plan). The "Anchors/notes" and "Acceptance criteria" here seed that. Sizes: S ≈ ½–1 day, M ≈ 1–2 days, L ≈ 3–5 days.

**Suggested delivery order / milestones:**
- **M1 (foundation):** A.1, A.2 (graph-spectral — the primary general engine), A.3, A.4, A.5. *(A.6 banding is optional/deferred.)*
- **M2 (first user-visible win):** B.1–B.3, C.1–C.2 → a general Pattern view with auto-colored clusters and auto-ranked structural axes.
- **M3 (interactive):** D.1–D.3, E.1–E.3 → dual-view morphospace + exemplar slider.
- **M4 (depth/robustness):** B.4, C.3, E.4, F.1–F.3, A.6 (optional banding plug-in), G.1–G.4.

Dependency graph (high level): A.1 → A.2 → A.3 ; A → B → {C, D} ; D → E ; A → F ; {B,C,D,E,F} → G.

---

## Epic A — Slicer-independent, tested analysis core

*Goal:* a pure-Python `interdeca_analysis` package (NO slicer/qt/vtk imports at the algorithm layer) that mirrors the validated `fishpipe` logic, callable from `InterDeCALogic`, with headless pytest + a tiny fixture. This de-risks every later epic and makes the math reviewable/testable outside Slicer.

- **A.1 — Package skeleton, fixture, and face-color parity** *(M)*
  - *Why:* Slicer code is hard to test; isolate the algorithms. Establish a ground-truth fixture.
  - *Acceptance:* `color_deca/deca3/interdeca_analysis/` importable with no Slicer deps; a small fixture (≤10 fishy textures + `fishy1.obj`, or a synthetic stand-in) under test data; `pytest` runs headless in the module's Python; a per-face color sampler whose output matches `_calculateFaceAverageColors` within rounding on the fixture (parity test).
  - *Anchors/notes:* mirror `fishpipe/mesh.py` + `data.build_face_colors`; oracle = `_calculateFaceAverageColors` (8809). Provide adapters to accept VTK polydata OR raw arrays.
- **A.2 — Graph-spectral descriptor (manifold harmonics) — PRIMARY structural engine** *(L)*  *(depends A.1)*
  - *Why:* the **general**, pattern-agnostic, UV/topology-independent structural descriptor — handles arbitrary patterns (spots/stripes/reticulation) and is the strongest single engine (EXP-07: stripe up to 0.93, cheeks 0.79). Critical for real cichlid meshes.
  - *Acceptance:* build vertex/face graph Laplacian (reuse/adapt `_buildFaceAdjacencyGraph` 10549), cache the smallest-k eigenbasis per **mesh hash**, project color channels (L\*/a\*/b\*/chroma) → band-energy descriptor; **k configurable** (k≈300 fast / ≈700 sharper — document the eig time, ~80 s at k=700, one-time per mesh); guarded by `SCIPY_AVAILABLE` with graceful fallback; unit test reproduces fishy stripe ≈0.88–0.93 and cheeks ≈0.79 (direct). Port from `fishpipe/spectral.py`.
- **A.3 — General structural descriptor library + auto-ranking** *(M)*  *(depends A.2)*
  - *Why:* one descriptor isn't provably complete; combine complementary **general** descriptors and let the data say which matter — the generalization mechanism for *arbitrary* features.
  - *Acceptance:* a library API combining graph-spectral (A.2) + patch-adjacency (Endler/QCPA transition matrix, reuse `_buildFaceAdjacencyGraph`) + chromatic novelty; plus an **auto-rank** function scoring each descriptor/axis by population **cluster-tendency** (e.g. multimodality / silhouette after whitening) so the top structural axes surface without labels; unit test on fishy ranks the stripe-bearing axis highly. Port/extend `fishpipe/structure.py` + `features.py`.
- **A.6 — (Optional) Taxon-tuned banding detector** *(M)*  *(depends A.1; OPTIONAL / lower priority)*
  - *Why:* a sharp, directly-interpretable "# bands" axis for *striped* taxa — an optional plug-in, **not** the general engine.
  - *Acceptance:* `pattern_traits(...)` returns `banding_count`/`banding_strength`/FFT power (auto principal axis + salient mask, quantile≥0.80 default 0.85); documented as taxon-specific; unit test reproduces fishy stripe ≈0.89; the q=0.70 sensitivity documented. Port from `fishpipe/recommended.py`. *Defer to M4 unless striped-taxon interpretability is needed sooner.*
- **A.4 — Region-segmentation compression** *(S)*  *(depends A.1)*
  - *Why:* owner-requested compression / denoise / area-normalization of small features.
  - *Acceptance:* `region_summary(...)` (KMeans on centroids → R regions; area-weighted per-region color; cached region labels per mesh); test shows compression reduces dims and preserves belly/tail recovery; documents region-count default. Port from `fishpipe/features.region_summary`.
- **A.5 — Auto-clustering + exemplar retrieval helpers** *(S)*  *(depends A.1)*
  - *Why:* shared primitives for Epics C and E.
  - *Acceptance:* `auto_cluster(scores, max_k)` (GMM+BIC) and `nearest_specimen(coord, scores, metric)` (returns index/name + distance); unit tests: fishy color-space auto-cluster finds 4 groups (ARI≈0.99); retrieval returns correct nearest row. Port from `fishpipe/recommended.py`.

## Epic B — Pattern analysis path (Step 3)

*Goal:* add the pattern feature path to `performPopulationAnalysis` and the minimal Color|Pattern toggle.

- **B.1 — Pattern feature branch in `performPopulationAnalysis`** *(M)*  *(depends A.2; A.3)*
  - *Acceptance:* a third feature-vector branch (inserted ~11671) builds the **general structural descriptor** (A.2 graph-spectral, optionally the A.3 library) per specimen; returns the same `result` dict shape (`reduced_data`, `model`, `texture_names`, `n_components`, plus `feature_kind="pattern"`); existing color paths unchanged; dep-guarded.
  - *Anchors:* 11620–11707 feature construction; 11732–11756 reduction.
- **B.2 — "Analysis view: Color | Pattern" toggle (minimal UI)** *(S)*  *(depends B.1)*
  - *Acceptance:* one segmented control / radio pair added near the dim-reduction radios (1041); selection threads into `onCompareTexturesButton` (5493) → `performPopulationAnalysis(..., feature_kind=...)`; default = Color (current behavior). Styled via `ColorTheme`.
- **B.3 — Auto-ranked named structural axes attached to result** *(M)*  *(depends A.3, B.1)*
  - *Acceptance:* the auto-rank function (A.3) scores structural descriptors/axes by population cluster-tendency; the top few are returned as `result["named_axes"]` (general, label-free) for the morphospace axis selectors (D.2). If the optional banding detector (A.6) is present, its `banding_count` is offered as an additional named axis. Default surfaces the data-driven top axes, **not** a hardcoded trait.
- **B.4 — Rare-variant (cheeks) score + overlay data** *(S)*  *(depends A.3)*
  - *Acceptance:* per-specimen chromatic-spectral `novelty_score` in the result; a documented threshold/percentile flags outliers; consumed as a point overlay in C.2/E.4. Honest doc note re: 5%-class limits.

## Epic C — Automatic cluster detection & coloring

*Goal:* color morphospace points by auto-detected clusters so non-technical users SEE segments.

- **C.1 — Auto-cluster the population result** *(S)*  *(depends A.5, B.1)*
  - *Acceptance:* GMM+BIC over the active axes (PCs or named axes); labels stored in `result["clusters"]`; max-k and covariance documented; deterministic seed.
- **C.2 — Color the scatter by cluster + legend + novelty overlay** *(M)*  *(depends C.1, B.4)*
  - *Acceptance:* `createPopulationPlot`/`createMorphospacePlot` (5868) render one plot series per cluster (distinct colors) with a legend; rosy-cheek/novelty outliers drawn with a marked style; replaces the single-series scatter.
  - *Anchors:* current scatter built via vtkMRML plot nodes in `createMorphospacePlot` (5868) / `createPopulationPlot`.
- **C.3 — Cluster summary readout** *(S)*  *(depends C.1)*
  - *Acceptance:* a small table/log in Step 3 listing cluster sizes and member specimen names; exportable.

## Epic D — Morphospace: dual view, measured axes, non-invertible handling

*Goal:* generalize the morphospace to Color|Pattern and to measured axes; gate the generative slider on invertibility.

- **D.1 — Color view uses the invertible subsampled-flatten generative slider** *(S)*  *(depends B.1)*
  - *Why:* keep the existing, working generative slider for the color view; avoid the unsupported area-histogram reconstruction (6096).
  - *Acceptance:* Color view uses the subsampled-flatten representation so `inverse_transform` reconstructs displayable colors; belly/tail morph works as today; no silent error path.
- **D.2 — Axis selectors include measured named axes** *(M)*  *(depends B.3)*
  - *Acceptance:* `multiRecolorXAxisCombo`/`YAxisCombo` (1071/1078) list PCs/ICs **and** named measured axes (banding count/strength); selecting them re-plots; axis labels show names + (for PCs) variance %.
- **D.3 — Detect non-invertible axes → switch to exemplar mode** *(M)*  *(depends D.2, E.1)*
  - *Why:* measured axes and UMAP have no generative inverse.
  - *Acceptance:* when the active representation/axes are non-invertible (measured axis or UMAP), the generative slider is disabled and the UI switches to exemplar display (Epic E) with a clear inline note; replaces the current silent error at 6096. Invertible cases keep the generative slider.

## Epic E — Exemplar slider (retrieval)

*Goal:* slide along any axis → display the nearest real specimen's texture on the model.

- **E.1 — Nearest-specimen retrieval** *(S)*  *(depends A.5)*
  - *Acceptance:* given a morphospace (x,y) in the active axes, return the nearest specimen (index/name + distance) over `reduced_data`/named axes; unit-tested.
- **E.2 — Apply a specimen's texture to the model on demand** *(M)*  *(depends E.1)*
  - *Acceptance:* given a specimen name, load its texture from `textureDir` and apply to the displayed model node (reuse `applyTextureToModel` 7655 / `applyTextureWithSubsamplingOnly` 11155); idempotent and fast enough for slider drag.
- **E.3 — Wire sliders + "Display mode: Morph | Exemplar"** *(M)*  *(depends E.2, D.3)*
  - *Acceptance:* the existing X/Y sliders (5694/5709) in exemplar mode call E.1→E.2 and display the nearest real specimen; a "Display mode" toggle switches Morph vs Exemplar; the current exemplar's name is shown; works for measured axes and UMAP. Update `updateMorphospaceVisualizationXY` (5833) to branch on display mode.
- **E.4 — Scatter markers + specimen readout polish** *(S)*  *(depends E.3, C.2)*
  - *Acceptance:* current point, exemplar, and the rest are visually distinct on the scatter; hovering/selecting reports the specimen name; novelty outliers highlighted.

## Epic F — Region segmentation (compression) option

*Goal:* optional face→region compression for speed/denoise/area-normalization.

- **F.1 — Integrate region segmentation into feature extraction** *(M)*  *(depends A.4, B.1)*
  - *Acceptance:* an optional pre-step compresses faces→regions for both color and pattern feature paths; results equivalent-or-better on fishy; off by default.
- **F.2 — Minimal UI toggle + region count** *(S)*  *(depends F.1)*
  - *Acceptance:* a checkbox + region-count spin (sensible default) near Step 3; threaded through; tooltip explains the speed/rare-feature tradeoff.
- **F.3 — Performance validation on a large real mesh** *(S)*  *(depends F.1)*
  - *Acceptance:* timing/memory before/after on a real cichlid atlas; documented default region count; no regression in recovery.

## Epic G — Validation, generality, docs, cleanup

- **G.1 — End-to-end fishy validation through the module** *(M)*  *(depends B–E)*
  - *Acceptance:* a scripted/headless run reproduces the offline scorecard (belly/tail 1.0, stripe ≈0.89, cheeks ≈0.78) using the in-module core; plus a manual Slicer smoke-test checklist (load atlas+textures → Pattern view → auto clusters → exemplar slider).
- **G.2 — Generality test on `cichlid-synth` (varying meshes)** *(M)*  *(depends A.2, B–E)*
  - *Acceptance:* run the pipeline on `/mnt/data/ml_data/cichlid-synth` (n=25, differing meshes/UVs) where the **graph-spectral path** earns its keep; document what transfers and what needs per-dataset attention.
- **G.3 — User docs + tooltips + methods note** *(S)*  *(depends B–E)*
  - *Acceptance:* concise in-app tooltips for Color|Pattern, measured axes, exemplar mode; a short methods paragraph + figure for the paper (reuse `fig3_before_after`); update module change_log.
- **G.4 — Resolve legacy `PCAMorphospace.py`** *(S)*  *(independent)*
  - *Acceptance:* decide deprecate/remove vs fold-in (it's not wired in); reconcile with `design_docs/user_story_01_remove_obsolete_features.md`; remove dead code or document why it stays.

---

## Open questions for RPI research stages
- Per-specimen vs atlas-space: on real data the morphospace operates on DeCA atlas-space textures (one shared atlas mesh) — confirm the pattern/spectral descriptors are computed in atlas space (they should be; that's the shared-mesh assumption the fishy validation relied on).
- Eigenbasis caching key: hash on atlas mesh identity so the spectral basis is computed once per DeCA run (A.2).
- Exemplar display fidelity: apply the raw specimen texture vs the subsampled/clustered reconstruction — pick per "Display mode" (E.2/E.3).
- Whether the Pattern view should also offer the spectral-coefficient *generative* slider (invertible, smooth pattern morph) as an advanced option, or exemplar-only (simpler UI). Default to exemplar; revisit after M3.
