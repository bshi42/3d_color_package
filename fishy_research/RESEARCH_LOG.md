# Research Log — Improving cluster recovery in the fishy color morphospace

**Owner:** (handed off from Alek) · **Started:** 2026-06-01 · **Status:** ACTIVE

This log is the source of truth. It records, in order: the problem, the known ground
truth, every hypothesis, every experiment (with the exact script + result), conclusions,
and the current state of work-in-flight (so we can resume after an interruption).

---

## 0. Problem statement & success criteria

> **PRIORITY UPDATE (2026-06-01, from project owner):** Color/hue clusters are *easy* and
> already solved (belly/tail = PC1/PC2). The high-value target — especially for biologists —
> is **STRUCTURAL / PATTERN features**: number of stripes, and texture/pattern structure.
> Favor methods that are **CPU-only** (audience has no GPUs), **general**, and interpretable:
> classical CV texture descriptors and **graph/spectral approaches on the mesh** (color as a
> signal on the mesh graph; pattern = spectral structure / "topology"). Light DL (CPU
> inference on rendered views) is acceptable as a comparison, not the primary path.

**Problem.** The Slicer color-DeCA module runs a color-based population analysis
(PCA/ICA/UMAP "morphospace") on a set of textured 3D models. On the synthetic **fishy**
dataset — which was *built* to contain known clusters — the morphospace recovers the
clusters poorly. We want an analysis that lets a **non-technical user (biologist)** *see*
the population's clusters/segments in a simple 2D morphospace, with **minimal UI options**.

**Success criteria (measurable).**
1. Each of the 4 planted factors is *recoverable* from a low-dim embedding — quantified by
   balanced cross-val classification accuracy (fair to the 5% minority class).
2. At least the 3 "majority" factors (belly hue, tail hue, stripe count) are **visually
   separable** in a 2D (or small set of 2D) morphospace panels.
3. The recommended pipeline adds **few or zero** user-facing options.
4. Bonus / stretch: a *single* 2D plot in which all clusters are visible; recovery of the
   5%-prevalence "rosy cheeks" minority class.

---

## 1. Known ground truth (the planted structure)

Source: `color_fishy.py` generator + `FISHY_DATASET_SUMMARY.md`. Per-sample generating
parameters in `/mnt/data/ml_data/fishy/samples.csv` (250 rows).

Fish has 5 color features: base (yellow, whole body), black **stripes**, **belly** (blue),
**tail** (green), minority **rosy cheeks** (red). Four factors are deliberately clustered:

| Factor | Structure | Spatial extent | Difficulty intuition |
|---|---|---|---|
| `belly` hue | 2 blobs (1-D make_blobs) | medium patch | easiest (moderate area, 2 clean hues) |
| `tail` hue | 2 blobs | small patch | smaller area → weaker signal |
| `stripe` count | 4 (80%) / 5 (20%) | thin lines over body | **geometric**, not a color-composition change |
| `cheeks` present | 5% minority | tiny patch | **rare + tiny area** → hardest |

Everything else is continuous Gaussian nuisance: base hue/sat/val noise over the **whole
body** (largest-area variation), stripe spacing/width/offset, belly/tail strength &
translation, plus a 10% blended procedural noise. **Key tension:** PCA maximizes total
variance, and the largest-area variation is the *nuisance* base color, not the planted
factors.

> NOTE on a historical bug (Slack, 2025-11-13): an earlier generator correlated belly &
> tail hue 1:1, collapsing 4 hue clusters → 2. The current dataset (mtime 2025-11-21) is
> post-fix; **EXP-00 must verify belly/tail hue are independent** before trusting anything.

---

## 2. How the module builds the morphospace (reproduced faithfully)

All 250 fishy textures are baked onto the **same** mesh + UV (`fishy1.obj`), so DeCA's
atlas-transfer step is the identity here. The module's per-specimen analysis
(`InterDeCALogic.performPopulationAnalysis`, InterDeCA.py:11527) is:

1. Per face, average the texture color over the face's UV corners → `(N_faces, 3)` RGB
   (`_calculateFaceAverageColors`, :8809). N_faces = 73,728.
2. Build a **per-specimen feature vector**, one of:
   - **area-weighted histogram**: KMeans-quantize colors (Lab) → K clusters; per fish sum
     face *area* into nearest cluster; unit-normalize → `(K,)`.
   - **subsampled flattened**: take N_s faces, Lab colors, flatten → `(N_s*3,)`.
3. `PCA`/`FastICA(n_components)` or `UMAP(2)` on the `(N_specimens, D)` matrix. **No
   feature standardization.** (The older `PCAMorphospace.performPCA` uses per-vertex RGB
   flattened, also unstandardized; `area_weighted` flag there is accepted but never used;
   its `LAB` option is a no-op stub.)

`fishpipe` reproduces step 1 exactly (`mesh.sample_face_colors`) and both feature reps
(`features.spatial_flatten`, `features.area_hist`).

---

## 3. Experiment ledger

> Each experiment: hypothesis → method (script) → result → conclusion. Scripts live in
> `experiments/`; figures + scorecards in `results/`.

### EXP-00 — Sanity: dataset integrity & ground-truth independence  ·  STATUS: ✅ DONE
- **Goal:** confirm 250 textures load; belly/tail hue each form 2 clean, *independent*
  clusters; stripe ~80/20; cheeks ~5%. Guards against the historical 1:1 bug.
- **Result** (`experiments/exp00_sanity.py`, `results/exp00/ground_truth_structure.png`):
  belly 125/125, tail 125/125, **stripe 198/52 (79/21%)**, **cheeks 236/14 (5.6%)**.
  belly_hue⊥tail_hue **Pearson r=+0.11**, cross-tab ~uniform (68/57/57/68) → **independent;
  the historical bug is fixed.** Both hues show 2 cleanly-separated blobs; belly×tail makes
  4 clean quadrants in parameter space. Cache built: `(250, 73728, 3)` uint8, 13 s.
- **Conclusion:** ground-truth labels are reliable. The 4-quadrant belly×tail structure is
  the gold target for a color morphospace.

### EXP-01 — Baseline reproduction of the module morphospace  ·  STATUS: ✅ DONE
- **Hypothesis:** the module's PCA on unstandardized flattened/area-weighted color vectors
  recovers belly hue on PC1 but smears tail/stripe/cheeks (per Slack).
- **Method** (`experiments/exp01_baseline.py`): 3 module-faithful feature reps —
  `spatial_rgb_full` (221k-D), `spatial_lab_sub4000` (12k-D), `area_hist_lab_k24` (24-D) —
  × PCA/ICA/UMAP. Scorecard = balanced 5-fold CV accuracy per factor (+ best-axis AUC,
  silhouette, joint ARI). Figures in `results/exp01/`.
- **Result (balanced CV accuracy; chance≈0.5):**

  | factor | rgb_full PCA | lab_sub PCA | area_hist PCA | lab_sub UMAP |
  |---|---|---|---|---|
  | belly  | **1.00** | **1.00** | **1.00** | **1.00** |
  | tail   | **1.00** | **1.00** | **1.00** | **1.00** |
  | stripe | 0.72 | 0.75 | 0.52 | 0.49 |
  | cheeks | 0.48 | 0.48 | 0.44 | 0.55 |

  `spatial_lab_sub4000` PCA: **PC1=belly, PC2=tail → 4 clean quadrants** on a single 2D
  plot. Joint(belly×tail×stripe) ARI: rgb 0.22 / lab 0.50 / area_hist 0.55.
- **Conclusion (REFRAMES THE PROBLEM):**
  - **belly & tail hue are already perfectly recovered** by the existing pipeline (they
    dominate PC1/PC2). The stale Slack "only 2 clusters" was pre-bug-fix.
  - **The real failures are exactly two factors:** ① **stripe count** (geometric/periodic;
    weak ~0.7 because stripe positions jitter specimen-to-specimen → cross-specimen color
    vectors are misaligned, the "blurry blob"; area_hist≈chance because count is not a
    color-composition change). ② **rosy cheeks** (≈chance: red is a unique hue but covers a
    tiny area on only 14/250 fish → negligible variance, drowned).
  - area_hist (color composition) ties belly/tail but **destroys** stripe (no spatial info).
    Spatial reps keep weak stripe signal. → a good rep needs *both* composition and layout.

### EXP-02 — Proof that stripe count IS recoverable with a structure-aware feature  ·  ✅ DONE
- **Hypothesis:** stripe count is invisible to color PCA only because the feature is wrong;
  a structure-aware (spatial) feature recovers it.
- **Method (inline):** empirically found stripes live in the **dorsal** band (Y>0.04) and are
  periodic along **Z** (the body long axis). Built an area-weighted **darkness (100−L*) vs Z
  profile** over the dorsal faces; classified stripe from (a) raw profile, (b) |FFT| of the
  profile, (c) peak count.
- **Result:** raw profile → **balanced acc 0.905**; |FFT| → **0.905** (baseline color PCA
  was 0.72–0.75). Peak-count confusion: 5 detected peaks ⇒ **100% precision** for 5-stripe
  (33/33), recall 63% (19 five-stripe fish read as 4 → faint 5th band). corr(peaks, count)=0.76.
- **Conclusion:** the structural signal is strongly present and recoverable. BUT this feature
  hardcodes the axis (Z) + region (dorsal) — **not general**. Need a descriptor that finds
  pattern structure without knowing the axis: → graph/spectral on the mesh (EXP-03).

### EXP-03 — Structural / pattern descriptors (graph-spectral, Endler, region-compression)  ·  ✅ DONE
- **Hypotheses:** H3 region-compression area-normalizes rare/small features; H7 mesh-Laplacian
  spectral band-energy is a general shift-invariant pattern descriptor; Endler color-transition
  matrix (lit. top pick) captures stripe structure; connected components count stripes.
- **Methods:** `fishpipe.spectral` (manifold-harmonics band energy, 300 modes, 20 s one-time),
  `fishpipe.structure` (Endler adjacency, component counts, axial banding FFT), `features.region_summary`.
  Literature survey saved to `docs/methods_survey.md`.
- **Results (balanced CV acc unless noted):**
  | descriptor | belly | tail | stripe | cheeks | notes |
  |---|---|---|---|---|---|
  | region_mean R=256 (direct) | 1.0 | 1.0 | **0.93** | 0.66 | structure recoverable… |
  | region_mean R=256 (top-6 PC) | 1.0 | 1.0 | **0.58** | 0.47 | …but **NOT in the morphospace** |
  | spectral all-channels (direct) | 1.0 | 1.0 | 0.84 | **0.75** | best cheeks so far |
  | spectral L-only (direct) | 0.93 | – | 0.81 | – | belly leaks via lightness |
  | Endler adjacency k=6 | – | – | 0.70 | – | weak for *count* (subtle) |
  | connected-components | – | – | **0.46** | – | FAIL: baked stripes don't form clean components |
  | axial banding FFT (auto axis, direct) | – | – | **0.93** | – | general (auto Z), but belly leaks (low-freq) |
  | axial banding (top PCs) | – | – | 0.62 | – | structure buried under color variance again |
- **Conclusion:** region-compression **works** (recovers stripe 0.93 directly, area-normalizes
  cheeks 0.48→0.66) and is a good speed/denoise/compression lever (per owner request). The
  graph-spectral descriptor is the best *general* all-factor descriptor (cheeks 0.75). Endler /
  component-count underperform for *counting* on this baked+noised data. **But every descriptor
  hits the same wall:** structure is recoverable by a *classifier* (~0.9) yet does **not** appear
  in the top PCs because color-composition variance (belly/tail/base) dominates.

### EXP-04 — Can unsupervised reduction surface structure? + best single-scalar stripe axis  ·  ✅ DONE
- **Hypothesis:** ICA (independent sources) on rich combined features gives one clean axis per
  independent factor; or a directly-measured scalar makes stripe visible.
- **Results:**
  - **PCA/ICA on combined [region-color + banding + spectral] (428-D, standardized):** belly &
    tail get clean axes (auc ~1.0); **stripe best single axis only 0.73 (PCA, on the *last* PC8)
    / 0.79 (ICA); cheeks ~0.65.** Unsupervised reduction **cannot** isolate the low-variance,
    jitter-distributed stripe signal — confirmed across PCA and ICA.
  - **Single label-free scalar for stripe** (dorsal axial darkness profile): **peak-count AUC
    0.875** (literal "# detected bands"); FFT dominant-freq 0.55 / centroid 0.52 (4→5 frequency
    shift too subtle); autocorrelation 0.67; dark-area 0.61. Supervised profile classifier 0.905.
- **Conclusion (PIVOTAL):** **Discovery ≠ measurement.** Variance-driven morphospaces (PCA/ICA/
  UMAP) surface only high-variance factors (color). Subtle *structure* (4-vs-5 stripes under
  position jitter) must be **directly MEASURED with a label-free structural detector** (peak-count
  of the auto-body-axis darkness profile → 0.875, interpretable) and offered as a **named axis**.
  This matches biology practice (patternize/QCPA named traits). Ceiling for stripe ≈ 0.88–0.93.

### EXP-05 — Recommended pipeline, end-to-end, with publication figures  ·  ✅ DONE
- **Method** (`experiments/exp05_recommended.py`, `fishpipe/recommended.py`): assemble the
  C5 architecture, all **label-free**, and score baseline vs recommended. Figures in `results/exp05/`.
- **Result (balanced CV accuracy):**

  | factor | baseline color-PCA | **recommended** | how (recommended) |
  |---|---|---|---|
  | belly  | 0.996 | **1.000** | color morphospace (PCA on area histogram) |
  | tail   | 1.000 | **1.000** | color morphospace |
  | stripe | 0.640 | **0.887** | **measured** banding traits (auto-axis dark profile) |
  | cheeks | 0.482 | **0.775** | graph-spectral chromatic descriptor (rare-variant) |

  - **Color view auto-clusters:** GMM+BIC on Color-PC1/PC2 finds **exactly 4 clusters,
    ARI=0.989** vs belly×tail — segments appear with zero tuning.
  - **fig2/fig3:** the **measured "banding count" axis splits the population at 4.5** into a
    4-band group (mostly 4-stripe) and 5-band group (mostly 5-stripe) — *visibly* separable.
    Confirmed (again) that **pattern-feature PCA does NOT** put stripe on PC1/PC2 (the 0.887
    lives in a classifier combination), so the headline plot uses the measured axis, not PCA.
- **Conclusion:** the C5 architecture delivers: color clusters auto-shown, stripe made visible
  via a measured axis, rare variant flagged. Honest ceilings: stripe ≈0.89, cheeks ≈0.78
  (5%-prevalence + tiny area is fundamentally limiting, as Alek noted).

### EXP-06 — Robustness / overfitting / leakage audit  ·  ✅ DONE  (delegated to subagent)
- **Method:** `results/exp06_audit/audit_report.md` (+ `audit_numbers.json`, `audit_scratch.py`).
  Seed sweep, hyperparameter sweep, leakage grep, negative-control label shuffle, figure review.
- **Result:**
  - **Seed-stable:** stripe 0.874 mean (0.863–0.887 over 5 seeds); cheeks 0.787 (0.775–0.794).
  - **Hyperparameter:** stripe robust for salient-mask **quantile ≥ 0.80** (0.89–0.91, n_bins
    barely matters) but **drops to ~0.74 at quantile 0.70** — the one real knob. → default moved
    to **0.85** (re-verified stripe = 0.890). Documented in `recommended.salient_pattern_mask`.
  - **No label leakage:** every feature builder takes only `fcd`/`scores`; GT used only in
    experiments/plotting. Scaler/LogReg fit fold-internally. Population stats are label-free and
    ~97.5% stable on a random half-population (transduction-safe).
  - **Negative control:** shuffled stripe labels → **0.504** (vs real 0.890) → signal is genuine.
  - **Figures:** fig1/fig3 honest & compelling; fig2 right panel was misleading (supervised 0.89
    caption on an intermixed PCA) → **relabeled** to explicitly show "PCA does NOT separate stripe."
- **Conclusion:** headline (stripe 0.64→0.89, cheeks→0.78, color ARI 0.99) is **robust and
  leakage-free**. Residual caveats: mask-quantile knob (documented, default 0.85), banding
  detector is taxon-specific, cheeks rests on 14 specimens (wide CI — "promising, not locked").

### EXP-07 — Generality stress test: general descriptor vs taxon-tuned detector  ·  ✅ DONE
- **Question (from owner):** does the approach generalize to *arbitrary* biological features, or
  is it overfit to "count dark stripes"?
- **Method:** compare the taxon-TUNED banding detector against the GENERAL graph-spectral
  descriptor (no stripe assumptions; vary #modes) and a general structural library
  (spectral + Endler adjacency + component counts). Balanced CV accuracy.
- **Result:**
  | descriptor | stripe | cheeks | taxon assumptions |
  |---|---|---|---|
  | tuned banding (axial dark profile) | 0.890 | – | YES (axial, dark-on-light, 1-D periodic) |
  | **graph-spectral k=300** | 0.881 | 0.794 | **none** |
  | **graph-spectral k=700** | **0.932** | 0.790 | **none** |
  | general library (spec+Endler+comp) | 0.895 | 0.754 | none |
- **Conclusion (REFRAMES the recommendation):** the **banding detector does NOT generalize**
  (assumes axial dark periodic stripes → misses spots/reticulation/blotches/hue-patterns). But
  the **general graph-spectral descriptor generalizes well and is actually the stronger choice**:
  with enough modes it *beats* the tuned detector (0.932 vs 0.890) with **zero** pattern-specific
  assumptions, and ONE descriptor recovered all four structurally-different feature types (hue
  clusters, periodic banding, rare spot). It is also UV/topology-independent (needed for real
  meshes). → **Make graph-spectral the PRIMARY general engine; banding becomes an optional,
  taxon-specific sharpening plug-in.** Residual general limits: (1) visibility ≠ classification
  (still need auto-clustering / named axes to surface structure in 2-D); (2) band energy is
  magnitude-only and no descriptor set is provably complete → use an extensible **descriptor
  library** auto-ranked by population cluster-tendency, with room to add targeted detectors.

### EXP-08 — Real-data generality test: mussel/clam half-shells (n=31)  ·  ✅ DONE
- **Data:** module output `…/mussels/out/2025_12-19_01_21_54/colorAnalysis` — shared atlas
  `atlasModelUV.obj` (122,901 v / 245,798 f) + 31 per-specimen baked textures (`UF_IZ_*`). Same
  shared-atlas+UV structure as fishy → ports directly. NO ground truth → judge by interpretability.
- **New harness:** `fishpipe/dataset.py` (generic shared-atlas loader) with **baking-artifact
  handling via population-median imputation** (per-specimen black-bake faces replaced by that
  face's median across clean specimens — needs no downstream changes). Spectral engine made
  mesh-agnostic (inject mesh+cache; fishy path verified unchanged). Script: `experiments/exp_mussels.py`.
- **Result:**
  - **Artifacts:** 1.89% of (specimen,face) cells imputed; worst = UF_IZ_439322 (8.6%, the big black
    blob). Imputation cut 439322's outlier z **3.02→2.31** while leaving genuine color-outliers
    (507761/781/616) unchanged → artifact handling works, doesn't distort real signal.
  - **COLOR morphospace:** auto-clustered into **2 interpretable groups** — 4 darker/browner shells
    (L*=54, b*=+20: 507399/616/761/781) vs 27 lighter (L*=67). A real periostracum-darkness axis.
  - **PATTERN morphospace (general graph-spectral, k=200, no tuning):** GMM **correctly abstained
    (k=1, no discrete groups)** but PC1 is a clean **continuous** axis: corr **−0.71 with lightness
    contrast** (growth-ring/texture prominence), +0.51 with darkness. Exemplar extremes confirm it
    visually (smooth shells at low end, ringed/high-contrast at high end —
    `results/mussels/pattern_axis_exemplars.png`).
- **Conclusion:** the approach **generalizes to real data**: on a genuinely different dataset
  (real mussel shells, different mesh, baking artifacts) the *same* general engine produced
  interpretable structure — a discrete color group + a continuous pattern (ring-contrast) axis —
  and the auto-clusterer correctly **abstained** rather than inventing spurious pattern clusters.
  Honest caveats: n=31 is small; pattern variation here is continuous (real, or n/modes-limited);
  museum-label text on shells is a minor confound; ~1.8% of faces are imputed in *many* specimens
  → likely shared UV/bake gaps worth flagging to the module team. (k>200 modes may sharpen ring signal.)

### EXP-09 — Orientation-aware (Gabor) descriptor + exemplar pattern morphospace (mussels)  ·  ✅ DONE
- **Motivation:** the 507450 "wavy green" retrieval (EXP-08 follow-up) exposed the magnitude-only
  spectral descriptor's blind spot — it captures pattern *scale* but not *orientation/waviness*.
- **Method:** auto-extract a clean **exterior** crop per specimen (shared UV → fixed image box;
  `exterior.py`), run a **Gabor filter bank** (freq × orientation, on L* and a* channels, black
  masked) → orientation-aware descriptor (`texture2d.py`). Retrieval + an **exemplar morphospace**
  (image-scatter of exterior crops at Gabor-PCA coords). `experiments/exp_mussels_gabor.py`.
- **Result:**
  - **Gabor retrieval** of 507450 pulls in **507738, 507616, 507751** (green-rayed cohort) and drops
    the smooth tan shells — better-targeted than spectral (which mixed in 507380/507608).
  - **New interpretable axis:** 507450 = **rank-1 total Gabor energy** (boldest pattern) AND **most
    *omnidirectional*** (wavy chevrons sweep all orientations); the opposite extreme is **65304**
    (max anisotropy = straight concentric growth rings). The magnitude-only spectral descriptor
    **cannot** make this wavy↔straight distinction. → orientation-aware features add real signal.
  - **Exemplar pattern morphospace** (`results/mussels/pattern_morphospace_exemplars.png`): shells
    placed by exterior texture; 507450 isolated top-right (bold+wavy), 65304 bottom (straight rings),
    green cohort forming a gradient toward 507450. Auto handled artifacts (no artifact specimen as a
    spurious outlier — the outliers are genuine pattern extremes).
- **Conclusion:** confirms the generality caveat fix — add an **orientation-aware descriptor**
  (Gabor, or signed spectral coefficients) to the library for *directional* patterns; it isolates
  the wavy pattern and yields an interpretable directionality axis. The **exemplar morphospace**
  (image-scatter of the auto-extracted exterior) is the visual deliverable for the module (Epic E).
  Caveat: 2D-Gabor runs on the raw texture crop → still sensitive to large bake artifacts; apply
  the per-face artifact mask to the crop for full robustness.

### EXP-10 — Fishy pattern morphospace with side-view exemplars  ·  ✅ DONE
- **Method:** added a tiny CPU renderer (`render.py`: orthographic lateral view, back-face cull,
  painter's sort) → side view of each fish (stripes/belly/tail clearly visible). Rendered all 250,
  ran orientation-aware Gabor on the lateral L* channel (keys on the stripe bars), PCA → morphospace
  with a farthest-point-sampled spread of side-view exemplars (border = true stripe count).
  `experiments/exp_fishy_exemplar_morphospace.py`; figure `results/fishy_pattern/`.
- **Result:** clean, legible exemplar morphospace (real fish, countable stripes). Gabor(L) recovers
  stripe **0.78 direct** but only **0.65 in top-4 PCA** → the 2-D layout shows pattern variation but
  does **not** cleanly split 4 vs 5 stripes (the familiar discovery≠measurement effect; the measured
  banding axis would separate them by construction). Renderer + exemplar plot reusable for the module.

### EXP-11 — Label-free standard + unlabeled auto-discovery of the stripe split  ·  ✅ DONE
- **Methodological correction (owner):** *all user-facing visualizations must assume UNLABELED
  data.* Ground-truth labels are for VALIDATION only (a researcher view, possible only on fishy),
  reported as numbers — never used to color a user-facing plot. (Earlier fishy plots colored by
  true belly/tail/stripe were validation views; the mussel plots were already label-free.)
- **Key distinction:** the measured **banding count is a label-free MEASUREMENT** (computed from
  the darkness profile), not a label — so it's a legal axis on unlabeled data. What's illegal is
  coloring by true stripe count; the label-free substitute is **auto-clustering** + **exemplars**.
- **Does an unlabeled pipeline actually surface the stripe split?** (`exp_fishy_unlabeled_morphospace.py`)
  - **Label-free clusterability ranking** flags banding_count as most clustered: bimodality 0.864,
    BIC(1)−BIC(2)=+2671 (vs color-PC1 0.602/+56, strength 0.474/+16) — surfaced with NO labels.
  - **GMM+BIC auto-chose k=2** on [banding_count, strength]; clusters = {n=190, count≈4.0} and
    {n=60, count≈5.0}. *(Validation only:)* ARI vs true stripe 0.51; cluster1 = 67% true 5-stripe
    vs 6% in cluster0. → the split is auto-discovered label-free; labels only confirm it after.
  - Figure `results/fishy_pattern/fishy_unlabeled_pattern_morphospace.png`: x=measured banding
    count, y=measured banding strength, **color=auto-cluster (no GT)**, side-view render exemplars.
- **Note:** the planted **80/20 (4/5-stripe) imbalance is intentional** (`STRIPE_COUNT_PROB_4=0.8`),
  a deliberate majority/minority factor — which is part of why the unlabeled split is imperfect
  (ARI 0.51): the measured count miscounts some faint 5th bands and the minority is small.

### EXP-12 — REALISTIC morphospace on unknown data (no pattern prior) — the honest result  ·  ✅ DONE
- **Owner critique (correct):** axes labeled "banding count/strength" smuggle in the prior that the
  data *has bands*. On unknown data we'd never compute that. A realistic plot must use only a
  GENERAL, assumption-free descriptor; known labels may appear as **outlines** for validation only.
- **Method:** general Gabor texture descriptor (scales×orientations, no banding prior) → PCA → 2-D
  morphospace, side-view exemplars, outline=true stripe (validation). `exp_fishy_realistic_morphospace.py`.
- **Result (honest):**
  - General Gabor PCA, best top-2 axis vs stripe: **AUC 0.687** (≈ no separation).
  - Stripe IS in the full descriptor (classifier 0.826) but lives in minor-variance directions.
  - **Assumption-free auto-clustering: k=2, ARI vs stripe = −0.002** → the clusters it finds have
    **ZERO** correspondence to stripe count. The outlines are visibly intermixed across the plot.
- **Conclusion (TEMPERS the recommendation — important):**
  - On truly unknown data, the unsupervised pattern morphospace surfaces only the **DOMINANT** axes
    of pattern variation. **Subtle + imbalanced** structure (fishy 4-vs-5, 80/20) does NOT surface
    and is NOT found by auto-clustering — even though a *supervised* classifier recovers it (0.83).
  - The earlier clean fishy "banding-count" split was an artifact of the **domain prior** (we knew to
    measure bands); it is NOT representative of module output on unknown data.
  - **Why mussels "worked" but fishy didn't:** mussel ring-contrast was the *dominant* pattern
    variation, so the general morphospace surfaced it. Fishy stripe count is *not* dominant, so it
    stays hidden. You cannot know in advance which factors a dataset's morphospace will miss.
  - **Honest framing for the module:** the pattern view is a *discovery* tool for dominant structure
    + a label-free **auto-cluster** display; it is NOT a guarantee that every biologically-meaningful
    (possibly subtle/rare) factor becomes visible. Targeted detectors require a hypothesis (prior).
- **Imbalance control (`exp_fishy_realistic_balanced.py`):** the realistic plot's exemplars were FPS-
  sampled (≈80/20, not balanced). Checked the conclusion is not an imbalance artifact: rank-based
  AUC is base-rate-robust (0.687), and a **class-balanced subsample** gives top-2 PCA stripe balacc
  **0.613 ± 0.04** (chance 0.5). A **class-balanced DISPLAY** (18 true-4 + 18 true-5) still shows
  reds/blues fully intermixed → the general morphospace genuinely does not separate stripe, balanced
  or not. (Labels used only for display selection + outlines; axes/clustering stay label-free.)

### EXP-13/14/15 — Unsupervised discovery of low-variance/subspace structure (lit-guided)  ·  ✅ DONE
- **Question:** can a general, label-free, CPU method surface structure that variance-PCA misses,
  WITHOUT a domain prior and WITHOUT fabricating? Literature survey (`docs/` notes) → top methods:
  unsupervised RF (Shi-Horvath addcl1), random-subspace consensus, **projection pursuit by
  clusterability** (dip / Min-Density-Hyperplane), sparse k-means (SPARCL), HDBSCAN for minority,
  and a **significance gate** (dip test, SigClust, prediction-strength, column-permutation null).
- **Methods tested** (`exp_discovery*.py`) on general descriptors (Gabor/spectral/color), each vs a
  **column-shuffled null**:
  - **Projection pursuit (ICA ranked by clusterability)** — the winner. On the combined descriptor it
    ranks clusterable axes far above null (real BIC-gaps 604/355/221 vs null 74/28/21).
  - URF (addcl1) and random-subspace consensus: found belly strongly, stripe only ~0.67-0.73.
  - HDBSCAN / density (for cheeks minority): **failed** (best cluster cheek-purity ≈ base rate).
  - dip-maximizing PP via random hill-climb: my optimizer was too weak (noise-floor only) — inconclusive
    as a method, but the *supervised* check is decisive (below).
- **Significance-gated result (dip test, gate = p<0.05 AND dip>null-max):**
  - **belly, tail → DISCOVERED** (dip 0.12-0.22, p<0.001) in the color block, well above null. Robust.
  - **stripe → NOT significant.** Best unsupervised axis AUC 0.74-0.80 but dip ≈ 0.016 (unimodal). Even
    the *optimal supervised* (LDA) stripe direction has dip only **0.047 ≈ null floor 0.035** → the
    4-vs-5 classes **overlap** (AUC 0.83), so there is essentially **no density gap to discover** in any
    assumption-free descriptor. Fundamental, not a method failure.
  - **cheeks → NOT robustly discoverable.** GMM-BIC liked an axis (AUC 0.89) but dip rejects it (p=0.70,
    blind to a 5% minority) and HDBSCAN/density find nothing → the apparent axis is ~chance alignment of
    only 14 points. n=14 is below what forms a significant mode.
- **Conclusion (the discoverability taxonomy — a real, honest result):**
  1. **Well-separated balanced clusters** (belly/tail) → **unsupervised-discoverable** by gated projection
     pursuit; the null/dip gate confirms them AND rejects noise (no fabrication).
  2. **Subtle OVERLAPPING structure** (stripe 4-vs-5 under jitter) → **NOT discoverable** from general
     features (no density gap; optimal dip ≈ noise floor). Needs a domain-specific *measurement* (prior).
  3. **Rare minority** (cheeks, 14/250) → **NOT discoverable** at this prevalence (no significant mode).
  - **The genuine win:** a **trustworthy, gated discovery pipeline** (projection-pursuit + dip + shuffled
    null) that surfaces real separated structure and *refuses to invent* structure — exactly what makes an
    unsupervised morphospace safe to publish from. It honestly maps the limits rather than faking results.

### EXP-16/17 — Micro-autoencoder + achromatic/chromatic decomposition  ·  ✅ DONE
- **Idea tested (owner):** face-region-reduce + a TINY MLP/CNN/GNN autoencoder ("micro distill",
  few neurons ≈ the ~17 true generative params) to carve out the stripe factor linear methods miss.
  (No torch → sklearn MLP autoencoder; GNN/CNN-learned deferred.)
- **Result on region-reduced fishy (`exp_microae.py`, `exp_lightness_morphospace.py`):**
  - The **micro-AE did NOT beat PCA** (stripe ~0.82-0.83 either way) → the **nonlinearity is not the
    lever**; the AE is a nonlinear PCA and stays variance/reconstruction-driven.
  - **The lever the experiment exposed: ACHROMATIC vs CHROMATIC decomposition.** Analyzing the
    **lightness (L\*) spatial pattern separately** from hue (a\*,b\*) — a general Lab decomposition,
    NOT a banding prior — removes the high-variance belly/tail *hue* nuisance and lifts stripe
    recoverability from **0.64 → 0.768 ± 0.03** balanced (best-axis AUC 0.83). Stripe is confirmed
    *achromatic* (chromatic morphospace stripe only 0.63).
- **Honest caveats (the limit persists, softened):**
  - Stripe lands on **PC6** of the lightness morphospace (belly's lightness contrast + other lightness
    variation still own PC1-5) → still not a *default top-2* axis without knowing which PC is stripe.
  - Stripe-axis **dip 0.018 → still a GRADIENT, not a clean cluster** (4-vs-5 overlap is fundamental).
    Along the stripe-bearing axis the classes form a visible gradient (5→left, 4→right) with overlap.
- **Conclusion:** a **genuine, general, label-free improvement** (achromatic decomposition: 0.64→0.77),
  prompted by testing the NN idea — but it makes stripe a *visible gradient*, not a discoverable
  *cluster*. The neural net itself didn't help; the **representation choice** did. Consistent with C8:
  overlapping structure has no gap to cluster, only a gradient to measure/order.
- **EXP-17b — CONFOUND CAUGHT (owner): the lightness axis is stripe DENSITY/SPACING, not count.**
  Correlating lightness PCs with the ground-truth stripe parameters: **PC6 ↔ stripe_spacing r=−0.61**
  (stronger than ↔ stripe_count −0.49); PC3–PC5 ↔ stripe_width −0.45…−0.61. count⊥spacing in the data
  (r=−0.035) and spacing alone doesn't predict count (AUC 0.54), so the "count AUC 0.83" was **inflated
  by entanglement with the density axis** — regressing spacing OUT of PC6 *raises* count AUC to **0.907**
  (spacing was contaminating, not helping). → The pattern morphospace surfaces the **continuous stripe-
  GEOMETRY nuisances** (spacing/width/offset, high-variance) as its axes; the planted COUNT cluster is
  smeared weakly across them. Isolating count needs a spacing/width/offset-INVARIANT band-count = a prior.
  Reinforces C8: unsupervised axes track the high-variance continuous variation, not the planted cluster.

### EXP-18/19 — GFT visualization + (JPEG-like) graph compression sweep  ·  ✅ DONE
- **EXP-18 (viz):** painted Laplacian eigenmodes on the fish (smooth→oscillatory = "sines/cosines on
  the mesh"), the GFT spectrum of a fish's lightness (FFT-style magnitude plot, low-freq dominated),
  and low-pass reconstruction (blurry→sharp). `results/gft/`. DCT = path-graph GFT, so "graph DCT" ≈ our GFT.
- **EXP-19 (compression, owner-requested):** compress each fish's per-vertex L,a,b via GFT (keep K
  lowest modes / top-energy) and measure factor recovery vs compression. `results/gft/compression_sweep.png`.
  - **belly/tail survive 99% compression** (recoverable from 2-5 modes, low-freq/high-energy).
  - **stripe retained ~0.92-0.96 down to ~40 modes** (lost only at extreme K=2) — I'd WRONGLY predicted
    compression would kill it; it doesn't.
  - **cheeks: inverted-U — moderate compression DENOISES and IMPROVES it 0.48→0.77 @ K≈40**, then degrades.
  - **Phase matters:** signed GFT *coefficients* retain stripe 0.95 vs band *energy* 0.84 → switch the
    descriptor from magnitude band-energy to (compressed) coefficients to keep more structure.
  - **BUT (unsupervised check):** denoised rep still fails the dip gate (stripe/cheeks no gap) and HDBSCAN
    still can't find cheeks (purity 0.04≈base) → compression denoises for SUPERVISED/known-trait analysis
    and compaction, but does **NOT** make subtle/rare structure unsupervised-DISCOVERABLE. C8 holds.
- **Conclusion:** compression is a genuine **efficiency + denoising** win (esp. for weak/rare signals and
  via phase-preserving coefficients), not a discovery tool. Actionable: GFT-coefficient compression (~40
  modes) as a denoise/compaction preprocessing step.

### EXP-20 — Coefficient descriptor + compression adopted; pattern morphospace on it  ·  ✅ DONE
- **Implemented (owner):** `spectral.spectral_coeffs(k=40)` — signed GFT coefficients (phase-preserving)
  compressed to k=40 modes (denoise) — the recommended replacement for band-energy.
- **Net effect (supervised balanced-acc):** band-energy → **coeffs(k=40)**: belly .996→**1.00**,
  tail .996→**1.00**, **stripe .836→.925**, cheeks .748→**.769**. Clear, across-the-board upgrade.
- **Pattern morphospace** (`results/fishy_pattern/fishy_coeff_pattern_morphospace.png`): PCA on the
  achromatic (L) compressed coefficients, label-free, side-view exemplars. **Transparency check (PC↔
  stripe params):** PC1 |r|=0.77 with **spacing** (0.42 count), PC3↔width, PC6↔offset → the dominant
  pattern axes are the continuous **geometry** (spacing/width/offset); auto-cluster top-2 ARI vs stripe
  **= 0.014** (clusters are NOT count). So the better descriptor helps a *classifier* but the
  unsupervised morphospace still orders by stripe **density**, not count — consistent with C8.
- **Conclusion:** improvements adopted (coeffs + compression: stripe retention .84→.93). The pattern
  morphospace honestly shows the dominant pattern-geometry gradient (density) with exemplars; count
  remains a weak gradient, not a clean cluster.

### EXP-21 — Mussel pattern morphospace on compressed signed GFT coefficients  ·  ✅ DONE
- Same recipe as EXP-20 on the real mussels (`exp_mussels_coeff_morphospace.py`,
  `results/mussels/mussels_coeff_pattern_morphospace.png`). Descriptor = `spectral_coeffs(k=40, L)`.
- **Axes (interpretable):** PC2 ↔ **ring-contrast** (|r|=0.54), PC1 ↔ greenness/darkness. Variance more
  distributed than the old band-energy (EV 17/16/13%). The new descriptor gives a richer pattern view.
- **Honest gate:** GMM+BIC proposed k=4, but **dip test p>0.5 on every PC → NO significant clusters**;
  the structure is **continuous** (matches EXP-08). The "2 main groups" are a soft darkness gradient
  (light L≈68 vs dark L≈62); the 2 singletons are pattern *extremes* — **65304 & 507445 (most ringed)**.
- **Conclusion:** the coefficient+compression descriptor transfers cleanly to real data and yields an
  interpretable, denoised pattern morphospace — but mussel pattern is genuinely **continuous**, so the
  trustworthy read is "a ring-contrast/darkness gradient with two ringed outliers," NOT discrete clusters.
  Demonstrates the gate working on real data: GMM over-splits, the dip gate correctly abstains.

### EXP-22 — Segmentation-first + SPATIAL blob analysis (owner-directed; multi-agent workflow)  ·  ✅ DONE
- **Owner steers:** (a) check pattern analysis WITH a color-segmentation step like the Slicer module
  (KMeans quantization + neighbor-average smoothing); (b) go beyond blob COUNT — *spatial* analysis,
  since "same count, different distribution" patterns differ. Built `fishpipe.segment` (faithful
  smooth+KMeans), `fishpipe.blobs` (connected-component blobs, Endler transitions, segmented axial
  peaks), `fishpipe.spatial` (per-color spread/anisotropy/axial-layout/blob-dispersion). Ran an
  exhaustive Workflow: 30-config sweep + 3 deep-dives + adversarial verification + synthesis.
- **Headline (VERIFIED):** **segmentation (K=8, smooth=0, whole-body) → spatial(80-d)+Endler(36-d)
  recovers STRIPE COUNT at 0.929 full / 0.883 ± 0.024 balanced-subsample** (shuffled null 0.488).
  Blob *count* alone fails (0.54 — stripes merge into ~2 connected blobs); the win is the *arrangement*
  (spread/anisotropy/axial-layout + color adjacency), exactly the owner's point.
- **THE KEY ADVANCE — it measures COUNT, not the spacing confound:** the supervised stripe axis
  correlates **r=0.946 with stripe_count vs 0.027 spacing / 0.019 width**; regressing spacing+width OUT
  leaves count-AUC = 1.000 (spacing/width alone predict count at chance). Every *continuous* method
  (GFT/Gabor/lightness-PCA, EXP-17/20) keyed on **spacing**; segmentation+binarization strips the
  amplitude/spacing modulation and leaves the **topological count**. And it needs **no hardcoded body
  axis / dorsal mask** (unlike EXP-02's 0.875 dorsal-Z-FFT) → same accuracy, *more general*.
- **Honest caveats (still hold):**
  - **Visibility wall persists (C8):** in the unsupervised morphospace PC1=belly (21%), PC2=tail (17%);
    stripe-count max-PC |r|=0.095 → a supervised-recoverable **gradient/minority direction, NOT a cluster**.
    Figure `results/fishy_pattern/fishy_segspatial_morphospace.png`: exemplars intermixed (left) vs clean
    supervised LDA count axis (right).
  - **Cheeks NOT recoverable via segmentation:** no KMeans palette grows a dedicated red bin even at K=32
    (reddest a*≈19–25, never >30 for the 14/250 rosy fish); the 0.84 cell was full-data class-weight
    inflation. Honest cheeks signal (balanced 0.719) comes only from the **non-segmentation per-face
    chroma-novelty** detector → cheeks needs a separate rare-local-patch channel.
  - **Mussels:** spatial+Endler PCA axes = darkness / ring-contrast; **dip gate says continuous, no
    significant clusters** (consistent with EXP-08/21).
- **Conclusion (recommendation):** ship **segmentation (K=8, smooth=0) → spatial+Endler** as the general
  pattern front end — it matches/marginally beats the continuous supervised ceiling for belly/tail/stripe
  with **no hardcoded geometry** and provably measures **count, not spacing**; add a **per-face chroma-
  novelty channel** for rare patches (cheeks). Communicate stripe-count as a **supervised-recoverable
  trait axis, not an unsupervised cluster.**

### EXP-23 — Subtracting color-composition variance to surface spatial structure  ·  ✅ DONE
- **Owner Q:** how to "subtract out" the non-spatial (color-only) variance, since belly/tail dominate
  the unsupervised morphospace and bury the spatial count signal?
- **Method (reusable):** **residualize** the spatial+Endler pattern features against the color-
  composition histogram (`features.area_hist`) via partial linear regression (`exp_color_subtract.py`,
  `residualize()`). Also tested achromatic (L*-only) segmentation.
- **Result:**
  - **Color subtraction WORKS:** belly/tail correlation with the top residual PCs drops **0.97/0.87 →
    0.01** — fully removed. Figure `results/fishy_pattern/fishy_color_subtracted_morphospace.png`
    (before: belly/tail clean splits; after: intermixed = gone).
  - **But count does NOT surface:** the residual top PC is **count 0.35 ≈ spacing 0.38** (entangled);
    best top-PC stripe AUC unchanged (0.78→0.78); residual PC1 **dip p=0.99 → gradient, not cluster.**
- **Conclusion — the variance hierarchy:** **color > stripe-geometry-nuisance (spacing/width/offset)
  > count.** Removing the top layer (color) only exposes the next layer (continuous stripe-GEOMETRY),
  not count. Count is the lowest-variance, overlapping factor → still buried, now under geometry not color.
  - *Useful upshot:* residualization gives a clean **pattern-geometry morphospace** (arrangement, free of
    belly/tail color) — a legitimate separate view. To surface COUNT specifically you must also remove the
    spatial-geometry nuisance (unsupervised-inseparable from count's geometry) → so count needs a
    **count-specific measured axis** (segmented axial peak-count, disentangled per EXP-22; mild body-axis
    prior) or stays a **supervised** trait. Confirms C8 at a deeper level: it's not just color burying it.

### EXP-24 — Multi-PC pairwise scan + ICA on the best descriptor (owner-directed)  ·  ✅ DONE
- **Owner Q:** look beyond PC1×PC2 — does stripe count show on lower-variance PC pairs (PC1×PC3,
  PC3×PC4, …)? And ICA? Tested on segmentation spatial+Endler (raw) and the color-subtracted residual.
- **Result — owner's intuition VALIDATED:**
  - Per-PC stripe AUC (raw spatial+endler): PC1=0.55(belly) PC2=0.57(tail) **PC3=0.78 PC4=0.75**, rest ~0.5.
    Count is **absent from the top-2** but lives on **PC3/PC4 (lower-variance, EV≈9%)**.
  - **top-2 PCs balacc 0.48 (chance) → top-4 0.79; BEST PC PAIR = PC3×PC4 → balacc 0.805** (vs 0.48 for
    PC1×PC2). Figures: `fishy_pc_pairgrid_stripe.png` (all pairs), `fishy_pc3pc4_exemplars.png` (5-stripe
    lean −PC3, 4-stripe +PC3 — a visible gradient). Most signal in PC3+PC4 (top-20 only 0.81).
  - Color-subtracted residual: count moves up to **PC1 (AUC 0.78)** (belly/tail freed), caps ~0.77
    (spacing-entangled).
  - **ICA does NOT beat PCA** here: best IC AUC 0.72, leans spacing (0.45) > count (0.33); top-15-IC
    balacc 0.806 = top-PC. (Consistent with EXP-04/14.)
- **Conclusion (refines C8):** stripe count IS visible — as a **gradient (~0.80), not a clean cluster** —
  on the **right lower-variance PC pair (PC3×PC4)**, NOT the top-2. → The morphospace should **auto-rank PC
  pairs by structure and surface the most-structured pair** (or let users browse pairs — the module's
  X/Y axis selectors already allow this). The "buried under belly/tail" finding was specific to the
  *default top-2*; pick the right pair and the geometry/count gradient shows. ICA adds nothing over PCA here.

### EXP-25 — Head-to-head: blob/spatial vs GFT + CORRECTION of the "segmentation measures count" overclaim  ·  ✅ DONE
- **Owner Qs:** is color-subtraction necessary if you browse PC pairs? how does blob/spatial vs GFT compare?
- **Head-to-head (same eval, supervised balanced-acc + supervised count-vs-spacing + best unsup PC pair):**
  | descriptor | belly | tail | stripe | cheeks | sup. stripe dir | best PC pair |
  |---|---|---|---|---|---|---|
  | blob/spatial+endler (116-d) | 1.0 | 1.0 | 0.929 | 0.51 | count 0.95/sp 0.03 | PC3×PC4 0.805 |
  | **GFT coefficients (k=40)** | 1.0 | 1.0 | 0.925 | **0.77** | count 0.99/sp 0.02 | PC3×PC5 0.792 |
  | GFT band-energy | 1.0 | 1.0 | 0.836 | 0.75 | count 0.79/sp 0.14 | PC2×PC6 0.770 |
- **CORRECTION (important, humbling):** EXP-22 framed segmentation+spatial as "the breakthrough that
  measures COUNT where continuous methods key on SPACING." **That was an overclaim.** (a) Segmentation+
  spatial only **TIES** GFT coefficients on stripe (0.929 vs 0.925) — it did not beat the EXP-20 result.
  (b) The "continuous = spacing" claim was only about the **unsupervised top axis**; the **supervised
  direction of GFT coefficients also recovers COUNT cleanly (0.99, spacing 0.02)**. Both descriptors
  contain count disentangled from spacing; the spacing/count confound is an *unsupervised-axis* artifact
  for BOTH. GFT also **beats** blob/spatial on cheeks (0.77 vs 0.51) and is simpler (no K/smoothing/
  threshold; mesh-intrinsic; one eigenbasis).
- **Where blob/spatial still genuinely helps (not exercised by fishy):** explicit, interpretable
  arrangement stats (dispersion, anisotropy, Endler adjacency = QCPA) that distinguish *same count,
  different distribution* — would matter on spotted/reticulated taxa; fishy never varies arrangement at
  fixed count, so no gain here.
- **Q1 verdict:** color-subtraction **not necessary** for recovering count — raw PC3×PC4 (0.805) ≥
  color-subtracted PC1 (0.78); residualization is lossy. Its value is convenience (promote pattern to a
  top axis), scaling with #color-factors.
- **Recommendation update:** **GFT signed coefficients (k≈40) is the best single general pattern
  descriptor** (ties/beats segmentation on every fishy factor, simpler, chromatic). Keep blob/spatial+
  Endler as the **interpretable** companion (named QCPA-style arrangement traits) and for arrangement-
  varying taxa. Surface structure via **auto-ranked PC pairs**, not the default top-2; cheeks still needs
  the per-face chroma-novelty channel.

### EXP-26 — GFT pairgrid (vs blob) + morphospace colored by GENERATING variables  ·  ✅ DONE
- **GFT vs blob pairgrid (both colored by stripe):** look the same — count intermixed on top pairs; best
  pair GFT PC3×PC5 (0.79) ≈ blob PC3×PC4 (0.805). Figures `fishy_pc_pairgrid_stripe_GFT.png` (+ blob).
- **Colored by continuous generating variables (the interesting bit):** the GFT-coeff morphospace cleanly
  recovers the CONTINUOUS texture parameters on dedicated PCs — **belly_hue→PC1 (r=0.93), stripe_spacing→
  PC3 (0.79), stripe_width→PC4 (0.84), tail_hue→PC6 (0.68)** — each a clean gradient; **stripe_offset not
  captured** (shift-invariant/low-var); **stripe_count NOT on any clean PC** (≤0.37, spread PC3/PC5).
  Figure `fishy_pcs_by_genvars.png`: PC3×PC4 plane is a clean **spacing(x)×width(y)** stripe-geometry
  morphospace; count intermixed across it.
- **Conclusion:** crisp demonstration of the variance hierarchy — the morphospace surfaces every
  *high-variance continuous* generating parameter (hue, spacing, width) on its own ordered axis; the
  *discrete, low-variance, overlapping* count is the lone exception. Biologically useful upshot: the
  PC3×PC4 plane is a real, interpretable **stripe spacing × width** morphospace even though count isn't there.

### EXP-27 — Segment-THEN-GFT (GFT of the segmentation one-hot)  ·  ✅ DONE (negative result)
- **Idea:** apply color segmentation (K=8 smooth) BEFORE GFT — project each palette color's per-vertex
  indicator field onto the Laplacian eigenbasis (k=100 modes), concatenate. Binarizing first could, in
  principle, strip amplitude/spacing modulation and leave count.
- **Result:** belly 1.0, tail 0.996, **stripe 0.762, cheeks 0.48** — **WORSE** than both pure approaches
  (continuous GFT coeffs 0.925; segmentation+spatial-blob 0.929). Pairgrid (`fishy_pc_pairgrid_stripe_
  segGFT.png`): belly→PC1, tail→PC2, count buried (best pair PC5×PC7=0.75).
- **Why it hurts:** segmentation **discards the continuous amplitude+phase** that the signed GFT
  coefficients exploit; and the binary indicator's discriminative content (sharp stripe boundaries) is
  **high-frequency**, outside the low-freq modes the eigenbasis captures (λ≤0.028 at 300 modes) — so the
  loss isn't recovered. Information loss > denoising gain, for GFT specifically.
- **Conclusion:** the two front-ends are **complementary but must NOT be chained.** Segmentation helps
  *spatial/blob* analysis (it needs discrete regions for connected-components/adjacency/dispersion) but
  *hurts* GFT (which thrives on the continuous signal). Best descriptors stand alone: continuous **GFT
  signed coefficients** (general, best) or **segmentation→spatial+Endler** (interpretable, tied) — not GFT-of-segmentation.

### EXP-28 — charisma-Figure-5-style plot (Schwartz et al. 2026 MEE) on the mussels  ·  ✅ DONE
- **Owner ask:** reproduce a plot like Fig 5 (charisma colour classifications on the tanager phylogeny:
  tree + per-species colour-presence dots + exemplar image + ancestral colour-wheels + gain/loss).
- **Adaptation (`exp_mussels_charisma_fig5.py`, `results/mussels/mussels_charisma_fig5.png`):** we have
  NO phylogeny for the 31 mussels (catalog numbers, no species IDs/tree), so we substitute a **phenetic
  DENDROGRAM**. Maps charisma's pieces onto our pipeline: K=10 **colour segmentation → per-specimen
  colour-class PRESENCE dots** (filled=present >3% area); **exterior exemplars** per tip;
  **dominant-colour node markers** (illustrative ancestral-state analog).
- **THREE dendrograms, now as SEPARATE FILES (owner ask), doubled resolution (dpi 260):** each
  clustering criterion is emitted as its own standalone charisma-style panel (tree + rotated exemplars +
  colour-presence dots, all keyed to that tree's own optimal leaf order):
  `mussels_dendro_colour.png` (ward on PCA(5) of √ palette area-fractions),
  `mussels_dendro_pattern.png` (ward on PCA(5) of the signed GFT coefficients k=40 L/a/b — the EXP-20
  descriptor, *colour-blind*), and `mussels_dendro_colour_pattern.png` (ward on both standardized +
  concatenated; integrative). 3276×3956 each.
- **Exemplars are now actual textured-MESH renders (owner ask), on the new 2026-06-09 run:** switched
  `dataset.MUSSELS` to `out/2026_06-09_00_08_36/colorAnalysis` (same atlas topology as the Dec run —
  v=122901/f=245798 — but new coordinates+textures; cleared the stale cache to rebuild). Exemplars are
  no longer flat texture crops: `exp_mussels_render_exemplars.py` renders each specimen as the shared
  ATLAS VALVE shaded by its own per-face colours via the CPU `render.render_textured_rgba` (exterior
  side: h=Z length, v=X, depth=Y thickness, **cull −1** → 99% exterior/periostracum faces; transparent
  bg, supersample 3, tight-cropped; shared bounds → auto-aligned, identical silhouette). This is the
  fishy approach (mesh + per-face colours) applied to mussels — a real 3-D shaded valve (growth rings,
  mottling, 507450's zigzag) instead of a UV rectangle. Saved to `results/mussels/exterior_render/`.
- **Landscape alignment (owner ask — valves came out diagonal):** the atlas valve's long axis is not
  aligned with the Z projection axis, so the raw render sat tilted. `render_textured_rgba(align=True)`
  now PCA-rotates the 2-D projection so the silhouette's major axis is HORIZONTAL (frame derived from the
  rotated silhouette + pad). Because the rotation is computed from geometry only it is identical across
  specimens → frames stay registered. Aligned aspect ~1.75 (landscape); re-tuned zoom 0.30→0.45 at the
  0.46 in/row pitch → valves ~87px in a 108px pitch (~21px gap), no overlap. Visual payoff of the
  separate files: under COLOUR
  the colour-presence dot rows grade smoothly (similar palettes adjacent); under PATTERN they scramble
  (pattern clades mix colours) — a direct read of colour⊥pattern disagreement.
- **Build notes (iterated by inspecting the actual plot — per "look at the plots"):** (a) build the
  MUSSEL face-adjacency (segment.smoothing was loading fishy's), (b) replace data-coord Circle/Wedge
  patches (stretched to streaks under the panel's x≠y aspect) with **scatter markers** (points-sized →
  round), (c) fix the earthy-palette colour-namer (was calling tan/olive "green"), (d) **exemplar sizing:**
  zoom 0.34 → 0.22 (was too large) then opened vertical row pitch (figure height N·0.82+3) so shells no
  longer touch top/bottom; shortened the tree title so it stops colliding with the "specimen" header.
- **Honest caveats:** it is a **phenetic similarity tree, NOT a phylogeny**; node markers are clade colour
  summaries, NOT evolutionary reconstructions. The *real* Fig 5 (ancestral-state reconstruction +
  gains/losses) needs an actual phylogeny for the specimens. Otherwise it's a faithful style/structure match.

---

## 4. Conclusions so far
- C8. **Discoverability taxonomy + trustworthy gated pipeline (EXP-13/14/15) — headline.** On UNKNOWN
  data with general (assumption-free) features, what an unsupervised morphospace can honestly surface:
  **(a) well-separated balanced clusters → YES** (gated projection pursuit discovers belly/tail and the
  dip+null gate confirms them); **(b) subtle OVERLAPPING structure → NO** (fishy stripe: no density gap —
  optimal dip ≈ noise floor; needs a domain measurement/prior); **(c) rare minority below ~n_mode → NO**
  (fishy cheeks n=14); **(d) genuinely continuous variation → the gate ABSTAINS** (mussels: 0 significant
  axes, no fabrication). The deliverable is a **trustworthy** pipeline that finds real structure and
  refuses to invent it — not a guarantee that every meaningful factor becomes visible.
- C0. **(reprioritized)** Structural/pattern features (stripe count, texture) are the goal;
  color clusters are already solved. Target CPU-only, general, interpretable descriptors.
- C1. Ground truth is reliable & 4-quadrant separable (belly×tail). Labels trustworthy.
- C2. Existing module pipeline **already solves belly+tail** (PC1/PC2). Remaining problem is
  **narrowly scoped**: recover **stripe count** and **rosy cheeks**, ideally without losing
  belly/tail and without adding UI complexity.
- C3. Two distinct sub-problems with different physics:
  - *stripe*: spatial-pattern/periodicity feature, robust to position jitter (shift-invariant).
  - *cheeks*: rare + tiny-area chromatic novelty → needs area-normalization or max-pooling so
    small vivid regions aren't drowned (this is Alek's unsolved "area normalization" + "low
    prevalence" problem from Slack).
- C4. **Discovery vs measurement (the key result).** A variance-driven morphospace (PCA/ICA/UMAP)
  only shows factors with large between-specimen variance. Color does; subtle structure does not.
  No feature engineering fixes this for the *unsupervised reduction* — stripe stays a minor axis.
  → **Measure structure directly** with label-free detectors and show as named axes.
- C5. **Recommended architecture (simple UI, CPU, interpretable, publication-ready):**
  1. **Color morphospace** (keep existing): PCA/ICA on color composition (area histogram or
     region-mean color) → belly/tail hue clusters, the clean 4-quadrant plot. *No change needed.*
  2. **Pattern view** (NEW, the real gain): a small set of **directly-measured, label-free
     structural traits** as axes — primarily **banding count** (peak-count of the auto-body-axis
     darkness profile, 0.875) and **banding strength** — so stripe-count groups separate *by
     construction*. General (auto principal axis), interpretable ("# bands"), CPU-trivial.
  3. **Rare-variant overlay:** chromatic-novelty / spectral score flags outlier specimens
     (cheeks ~0.75) as marked points beside the morphospace.
  4. **Auto cluster coloring:** GMM/HDBSCAN (with model selection) colors points so biologists
     SEE segments without reading PCs. Single UI control: *View = Color | Pattern*.
  5. **Region segmentation** offered as a compression/denoise/area-normalization option (owner req).
- C6. **Generality caveat:** "banding along the body axis" suits elongate striped taxa. For
  arbitrary patterns the general substrate is the **graph-spectral descriptor** (EXP-03) + a
  small library of label-free structural detectors; the principle (measure, don't hope to
  discover) holds regardless.

## 5. Next hypotheses (to test)
- **H2 (standardization):** z-scoring features before PCA equalizes low-variance structure
  → may surface stripe/cheeks, but risks amplifying base-color & procedural noise. *Cheap, test first.*
- **H3 (spatial region-summary):** per-region mean+max color over R body regions (KMeans on
  3D centroids; same mesh → aligned). Per-region summarization **area-normalizes** small
  regions (helps cheeks) and encodes coarse layout (may help stripe). max-pool catches tiny
  vivid spots (cheeks-red).
- **H4 (stripe spatial-frequency):** longitudinal darkness profile → FFT magnitude / peak
  count = shift-invariant stripe-count estimator. Likely solves stripe; judge generality.
- **H5 (chromatic-novelty for cheeks):** per-specimen area of palette-outlier color (general,
  not hardcoded to red) → rare-variant detector; may live *beside* the morphospace.
- **H6 (assemble):** best rep + auto cluster detection (GMM/HDBSCAN) coloring the morphospace
  → biologist sees colored groups; evaluate the "all clusters on one 2D plot" + simple-UI goal.

## 6. Work in flight / resume point
- ✅ EXP-00…05 complete & logged. Recommended pipeline in `fishpipe/recommended.py`, validated
  by `experiments/exp05_recommended.py` (figs in `results/exp05/`). Methods survey in
  `docs/methods_survey.md`. Recommendation/handoff in `docs/RECOMMENDATIONS.md`.
- IN FLIGHT: a robustness/leakage audit subagent (seed stability, hyperparam sweep of the
  banding mask/n_bins, negative-control label shuffle, figure review) → `results/exp06_audit/`.
  Fold its verdict into §7 when it returns.
- OPEN / future: (a) port the recommended analysis into the Slicer module (`InterDeCA.py`) as a
  "View = Color | Pattern" toggle + auto-cluster coloring; (b) test generality on the real
  cichlid-synth data (`/mnt/data/ml_data/cichlid-synth`, n=25, varying meshes — the graph-
  spectral path matters there since UV/topology differ); (c) push cheeks via better rare-variant
  detection if 5%-class recovery becomes a priority.

## 7. RECOMMENDATION (summary)
See `docs/RECOMMENDATIONS.md`. In one paragraph: **keep the existing color morphospace** (it
already solves hue clusters — auto-GMM finds the 4 belly×tail groups, ARI 0.99). **Add a
"Pattern" view** built from a few **directly-measured, label-free structural traits** — chiefly
a **banding/stripe-count axis** from the darkness profile along the auto-detected body axis
(0.64→0.89) — because subtle structure is invisible to variance-driven PCA/ICA/UMAP and must be
*measured*, not *discovered*. Flag **rare variants** (rosy cheeks) with a graph-spectral chromatic
descriptor (≈0.78). UI cost: one toggle (Color | Pattern) + automatic cluster coloring; everything
else is automatic. Offer **region segmentation** as a compression/denoise option.

---

## 8. Fable continuation (2026-06-12): the SMALL-n program — EXP-29..33

**Owner prompt:** review the prior research, do our own, and find an effective way to find
patterns with **unsupervised** methods on **small sample sizes** (the real data is small:
cichlid-synth n=25, mussels n=31 — the regime where C8 actually bites). A fresh CPU literature
scan (8 method families, all cited in `docs/smalln_methods_survey.md`) plus five new experiments.

> **Reframing (the headline).** C8 is not a limitation of this pipeline — it is a **theorem**
> about high-dimension-low-sample-size (HDLSS) geometry. Jung & Marron 2009: the top PCA axis is
> a consistent estimate of a signal direction **iff** its eigenvalue grows like p^α (α>1); when
> subtle pattern variance < per-face noise variance the leading eigenvector rotates ~90° off the
> true axis ⇒ "supervised-recoverable but not unsupervised-discoverable" is *expected*, provable,
> and predictable in advance. The productive response is a **three-part program**: (A) push what
> discovery CAN reach with a better representation; (B) make every claimed cluster TRUSTWORTHY
> with HDLSS-aware significance + stability so we never fabricate; (C) for what is real but not
> density-discoverable, a MINIMAL-SUPERVISION escape. (A) and (C) are validated below.

### EXP-29 — Small-n regime + plain semi-supervised learning curve  ·  ✅ DONE
- **Method** (`experiments/exp29_smalln_semisup.py`): on the team's best descriptor (GFT coeffs
  k=40), held-out balanced accuracy vs #annotated specimens (25 draws), for belly/stripe/cheeks,
  at n=250 AND n=40. Methods: LDA, LogReg, semi-supervised LabelSpreading. Baseline = GMM+BIC
  (oracle-named) unsupervised ceiling.
- **Result:** belly (well-separated) — unsup already 0.98; ~5-10 labels reach 0.96. **stripe** —
  unsup **0.500** (chance, confirms the wall); even 40 labels only reach **0.75-0.79**, and
  **LabelSpreading UNDERperforms plain LDA** (0.64) because the raw-descriptor kNN graph is
  dominated by color/spacing nuisance, so labels propagate along the wrong manifold. **cheeks** —
  stays ≈chance even with labels (random annotation rarely catches the 14 positives).
- **Conclusion:** plain few-label supervision on a generic high-dim descriptor is **label-
  inefficient** for the subtle factor (the discriminative direction is a subtle combination of
  120 dims). Two levers follow: a representation where the trait is low-dim (EXP-30), and a
  constraint *warp* + smart annotation rather than raw label-spreading (EXP-33).

### EXP-30 — Decoupled texton/VLAD representation (shared codebook over pooled patches)  ·  ✅ DONE
- **Idea (new, lit-endorsed):** there are only 25-250 specimens but each has ~73k faces — so learn
  a texture **vocabulary** from MILLIONS of pooled local patches (huge effective n) and summarize
  each specimen as a denoised histogram/VLAD over that **shared** codebook. A strict generalization
  of the team's per-image KMeans-segmentation + Endler adjacency (shared soft multi-scale codebook
  vs per-image hard color KMeans). New module `fishpipe/textons.py`: per-face multi-scale local
  jet via graph diffusion on the shared mesh (`diffusion_operator` = D⁻¹(A+I)) → MiniBatchKMeans
  codebook → area-weighted BoW or VLAD. CPU: built in 3-5 s.
- **Result (balanced CV acc; `experiments/exp30_textons.py`):**
  | descriptor | belly | tail | stripe | cheeks | note |
  |---|---|---|---|---|---|
  | GFT coeffs k=40 (their best) | 1.00 | 1.00 | 0.925 | 0.769 | baseline |
  | **texton BoW K=128** | 1.00 | 1.00 | **0.969** | 0.54 | **best stripe yet** |
  | texton VLAD K=128 | 1.00 | 0.95 | 0.71 | 0.706 | **cheeks best-axis AUC 0.847** (best seen) |
- **Conclusion:** the shared-codebook BoW **beats** the team's best general descriptor on stripe
  (0.925→0.969); the VLAD residual encoding gives the strongest rare-cheek axis seen (AUC 0.847).
  A genuine, validated descriptor upgrade (validated in EXP-31).

### EXP-31 — Validate the texton stripe gain (the team's audit discipline)  ·  ✅ DONE
- **Method** (`experiments/exp31_texton_validate.py`): seed sweep, shuffled-label null,
  count-vs-spacing disentanglement (their gold standard, EXP-22/25), small-n, label efficiency.
- **Result:** **seed-stable** (stripe 0.970 mean over 5 codebook seeds, 0.962-0.981);
  **null-clean** (shuffled labels → 0.489 vs real 0.981); **MEASURES COUNT not spacing** —
  supervised stripe direction correlates **|0.988| with stripe_count vs |0.018| spacing /
  |0.011| width** (matches/over the best prior descriptor). **Honest limits:** at **n=40** the
  supervised edge washes out (texton 0.641 vs GFT 0.664 — tied, as HDLSS predicts); **label
  efficiency** for stripe ≈ GFT (neither makes stripe a clean cluster — silhouette ≈0.04). So
  texton BoW is the new best **descriptor**, but it does **not** by itself break the discovery
  wall (consistent with C8 / Jung-Marron).

### EXP-32 — Trustworthy small-n gate: find real clusters, refuse to fabricate  ·  ✅ DONE
- **Method** (`fishpipe/gating.py`, `experiments/exp32_trustworthy_gate.py`): SigClust (Gaussian-
  null 2-means cluster-index test with a soft-thresholded eigen-spectrum, ~30 lines numpy) +
  consensus-PAC with a **PCA-covariance-preserving M3C-style null** (per-k p-value incl. k=1 =
  abstain) + per-cluster bootstrap **Jaccard** (Hennig). Run on the 2-D COLOR (area-hist) and
  PATTERN (texton-BoW) morphospaces at n=250 and n=40, validated vs ground truth.
- **Result:**
  | n | descriptor | SigClust p | GMM+BIC k | ARI vs color | ARI vs stripe | verdict |
  |---|---|---|---|---|---|---|
  | 250 | COLOR | 0.002 | 4 | **0.989** | −0.00 | real 4 groups ✓ |
  | 250 | PATTERN | 0.002 | 2 | 0.49 | −0.00 | non-Gaussian but NOT stripe |
  | 40 | COLOR | 0.044 | 5 | 0.834 | 0.02 | color still found ✓ |
  | 40 | PATTERN | **0.168** | 3 | 0.38 | −0.03 | **ABSTAINS** ✓ |
- **Conclusion:** the gate **finds the real color structure (ARI 0.99 at n=250, 0.83 at n=40)**
  and **correctly abstains on the pattern descriptor at n=40** — it surfaces real structure and
  refuses to invent it. **Two honest caveats:** (1) SigClust over-fires on *continuous* non-
  Gaussianity (it rejects "one Gaussian" for the pattern-geometry gradient too), so it is the
  "is there >1 Gaussian" test — pair it with the consensus/Jaccard "are there discrete clusters"
  test. (2) Per-cluster Jaccard is conservative on *adjacent* clusters (the 4 quadrants score
  ~0.63 "questionable" despite ARI 0.99) — read ARI-to-structure alongside it.

### EXP-33 — Minimal-supervision escape: how few constraints break the wall?  ·  ✅ DONE
- **Idea (the practical answer for small n):** for structure that is real but density-gapless,
  let m must-link/cannot-link pairs WARP the descriptor space (constraint metric learning;
  `fishpipe/semisup.py`, a regularized RCA/DCA generalized-eigen warp), then run ORDINARY GMM
  clustering in the warped space. `experiments/exp33_semisup_escape.py`: ARI vs #pairs, random
  vs active selection, stripe + belly control, n=250 and n=40.
- **Result (ARI vs the true split):**
  | descriptor·factor | unsup baseline | 10 | 20 | 30 | 50 | 80 pairs |
  |---|---|---|---|---|---|---|
  | GFT · stripe (random) | 0.026 | 0.05 | 0.16 | 0.24 | 0.33 | **0.42** |
  | texton · stripe (random) | −0.006 | 0.00 | 0.04 | 0.13 | 0.20 | 0.35 |
  | GFT · belly (control, random) | −0.00 | 0.85 | 0.84 | 0.84 | 0.96 | 0.97 |
  | n=40 · stripe · texton (active) | −0.002 | 0.11 | 0.12 | **0.25** | – | – |
- **Conclusion:** the escape **works** — a handful of pairwise constraints converts stripe from
  unsupervised-invisible (ARI ≈0) to recoverable (ARI 0.42 at 80 pairs; **0.25 at 30 pairs even
  at n=40**), with no change to the discovery pipeline (the warp just rotates the space so the
  existing GMM fires). Belly (well-separated) needs ~0 constraints (control). **Surprising honest
  finding:** naive **ACTIVE (uncertainty) selection UNDERperformed RANDOM** here (stripe 0.30 vs
  0.42 at 80) — in the raw space the dominant ambiguity is the *color* boundary, so uncertainty-
  sampling spends the budget on the wrong factor. **Lesson:** when multiple factors compete,
  random pair coverage is the robust default; target-aware active selection (or seeding the warp
  first) is needed before active selection pays off. (Tempers the literature's "active halves the
  budget" claim for this multi-factor setting.)

### C9 — The small-n program (new headline conclusion)
On genuinely small samples (n=25-40), the honest and effective recipe is **not** a cleverer
unsupervised reducer — it is:
1. **Predict** discoverability up front with the HDLSS spike diagnostic (Jung-Marron): subtle/
   overlapping/rare structure will be supervised-only; say so before running.
2. **Represent** better: shared-codebook **texton BoW/VLAD** (EXP-30/31) is the new best general
   descriptor (stripe 0.925→0.969, count-disentangled; best rare-cheek axis) — it raises the
   *supervised* ceiling and denoises, the right use of huge per-specimen patch counts.
3. **Gate** hard: **SigClust + consensus-PAC/M3C null + per-cluster Jaccard** (EXP-32) finds the
   real color clusters (ARI 0.99/0.83) and **abstains** on pattern at n=40 — trustworthy, never
   fabricates. This is what makes an unsupervised morphospace safe to publish from at small n.
4. **Escape** with minimal supervision what the gate (correctly) rejects: ~30 active/random
   pairwise constraints → metric warp → existing gated discovery recovers stripe (EXP-33). This
   turns C8's "undiscoverable" into a quotable expert-effort budget.
5. (Not yet prototyped, lit-recommended): **SNF** multi-view fusion; **Mapper** with a pattern
   lens (SW1PerS) as the biologist-facing morphospace; **patternize-style spatial-homology PCA**
   (the registration lever the team hasn't pulled); **RRPP** effect sizes for confirmatory
   small-n inference. See `docs/smalln_methods_survey.md`.

### EXP-35 — Anchor + rank-the-neighbors triplet feedback loop (owner-proposed)  ·  ✅ DONE
- **Idea (owner):** show an ANCHOR + its k nearest neighbors (+ a REMOTE instance as a check);
  the expert RANKS them most→least similar; MORPH the space to pull similar closer / dissimilar
  apart; repeat. This is *iterative triplet / ordinal metric learning with relevance feedback*
  (Schultz-Joachims 2003; van der Maaten-Weinberger 2012 t-STE; Tamuz 2011 crowd-kernel) — a
  ranking-driven, iterated version of the EXP-33 warp. Rankings carry more bits/query than
  yes/no pairs and humans rank relative similarity more reliably than absolute category.
- **Method** (`fishpipe/semisup.triplet_warp` + `experiments/exp34_anchor_triplet_loop.py`):
  round-0 = ordinary unsupervised PCA morphospace of texton-BoW; each round pick anchors,
  take current-space kNN + a remote, a SIMULATED expert ranks the candidate set by distance in
  a chosen subset of the TRUE generating params, rankings → triplets → linear triplet-metric
  warp (Frobenius-norm pinned each step to prevent the scale-collapse/blow-up failure modes),
  iterate. Three expert models (holistic / pattern-only / color-only) × diverse-FPS vs local
  anchors. Metric: per-factor kNN balanced accuracy each round.
- **Result (stripe kNN balacc; round 0 → 6):**
  | expert / anchors | belly | tail | **stripe** | cheeks |
  |---|---|---|---|---|
  | holistic, diverse | 0.99→1.00 | 0.99→0.99 | **0.56→0.61** | 0.50→0.50 |
  | pattern, diverse | 0.99→0.98 | 0.99→**0.88** | **0.56→0.63** | 0.50→0.50 |
  | color, diverse | 0.99→1.00 | 0.99→1.00 | 0.56→0.55 | 0.50→0.50 |
  | pattern, LOCAL (filter-bubble) | 0.99→0.98 | 0.99→0.85 | 0.56→0.64 | 0.50→0.50 |
- **Conclusion — the scheme works, in the predicted direction, with predicted limits:**
  1. **It surfaces what the expert RANKS.** A pattern-focused ranking lifts stripe neighborhood
     purity **0.56 → ~0.63 — essentially the *supervised*-kNN ceiling for this factor** (kNN-acc
     was ~0.61-0.69 even with full labels, EXP-30/01) — i.e. rankings recover about as much
     same-stripe neighborhood structure as full supervision would, while de-emphasizing color
     (tail 0.99→0.88). A color-focused ranking only amplifies color (stripe flat). **"You get
     what you rank"** — so steer the ranking to the subtle trait (instruct "rank by pattern,
     ignore colour", or residualize colour first), or run a loop per trait.
  2. **It does NOT fabricate** and does NOT rescue the **rare class** (cheeks stays 0.500): a
     5%/n=14 minority rarely enters any anchor's neighbor set, so it cannot be ranked → still
     needs the novelty-ranked-annotation / unbalanced-SigClust path, not this loop.
  3. **Stability is the engineering crux:** naive triplet SGD collapses (shrinks all distances)
     or blows up; pinning ‖L‖ every step fixes both. The **remote-instance check** doubled as a
     scale anchor + (here) enough exploration that the filter-bubble (local-only anchors) barely
     hurt — but on data where a hidden factor is fully orthogonal to the top axes, deliberate
     diverse-anchor sampling is still needed so that factor can ever enter a neighbor set.
- **Verdict:** a genuinely good, low-cognitive-load supervision channel and the natural
  ranking-upgrade of EXP-33 — best used to *sharpen a chosen perceptible trait* and as the
  interactive front-end to the warp, NOT as a rare-class discoverer. Magnitude is incremental,
  bounded by ranking fidelity and how cleanly the trait sits in the descriptor.

### EXP-36 — Spring/energy morphospace + diverse-panel ranking + per-iteration exemplar plots  ·  ✅ DONE
- **Owner spec (faithful):** diverse sample panel (not kNN), rank vs anchor by generating-var
  similarity, **plot the morphospace with fish exemplars every iteration**, transform via a
  **spring/energy** method. EXP-34/35 had skipped the plots, used kNN (not diverse) panels, and
  used a linear warp (not springs). This corrects all three.
  `fishpipe/semisup.spring_embed` (soft-ordinal/t-STE-style: ranked-near pull together, ranked-far
  push apart, tethered to the initial layout); `experiments/exp36_spring_morphospace.py` (renders,
  per-iter PNGs + GIF + montage) and `exp36b_compare_transforms.py` (the comparison figure).
- **Result (`results/exp36/transform_compare.png`, `morphospace_morph.gif`):**
  - The 2-D morphospace **visibly reorganizes AWAY from colour** under a pattern-focused ranking:
    belly kNN 0.996→0.80 (spring) / 0.67 (warp-display); the initial two clean colour blobs
    dissolve. The morph works and is legible.
  - **But the overlapping stripe factor stays intermixed in 2-D** (kNN 0.50-0.59) across initial,
    spring, and warp-display. **Oracle check:** even ranking by *pure* stripe_count with a loose
    tether caps at **0.56 in 2-D** → a genuine **2-D ceiling**, not a tuning failure. 2-D simply
    has no room for a low-variance overlapping factor (reproduces C8 / EXP-12 / Jung-Marron in
    the interactive setting; the rare cheeks stays 0.500 throughout).
- **Two design findings (both useful, one counter-intuitive):**
  1. **Spring (move points) vs linear warp (transform descriptor):** the spring/energy embedding
     is the right tool for the *interactive 2-D display* (smooth, faithful morph) but **cannot
     separate** a factor that needs >2 dims; the **N-D linear warp** (EXP-35) carries more of the
     subtle factor (stripe kNN 0.63 in its N-D space) but is not itself a 2-D picture. → Best
     practice: **learn the metric in N-D from the rankings, embed its 2-D shadow for display.**
  2. **DIVERSE panels HURT the subtle factor.** Warp from *diverse* panels (this exp) gave stripe
     kNN 0.54; warp from *local kNN* panels (EXP-35) gave 0.63. Fine contrasts among *similar*
     fish are where a subtle difference is the deciding factor; diverse panels give coarse
     contrasts dominated by the big (colour) differences, diluting the subtle signal. → Diversity
     helps **exploration** (don't miss a factor entirely) but **local panels help discrimination**
     of a factor already weakly present. **Use diverse ANCHORS for coverage + LOCAL panels for
     fine contrast.**
- **Conclusion:** the owner's anchor+rank+morph loop is sound and the spring embedding gives a
  faithful interactive morphospace, but (a) keep the separating metric in N-D and use 2-D only to
  display; (b) panels should be locally-fine, anchors globally-diverse; (c) a genuinely
  overlapping low-variance factor will not split in any 2-D view regardless of supervision —
  surface it as a measured/ranked AXIS (or in the N-D warped space), not as a 2-D cluster.

### EXP-37 — Region-localized axes + region-SCOPED comparison on a COMBINED morphospace (owner)  ·  ✅ DONE
- **Owner spec:** (1) tell the user WHICH body region a given PC affects most; (2) the user
  compares/ranks scoped to that region (circle it); (3) morph only that PC/region — all on a
  **combined colour+pattern** morphospace. `experiments/exp37_region_scoped.py`.
- **Method:** combined descriptor = standardized [area-hist colour | texton-BoW pattern] → PCA.
  Q1 back-projection: per-face saliency = |corr| across the population between each face's L*/
  chroma and the PC score → render on the fish + mark the largest connected region. Q2: build a
  region-SCOPED descriptor (colour/darkness stats over ROI faces only) and measure recovery,
  label-free ROIs (cheeks=per-face novelty, stripe=cross-pop lightness variance) AND an oracle ROI.
- **Result:**
  - **Q1 WORKS for the factors that are IN the morphospace** (`results/exp37/pc_regions.png`):
    PC2↔belly (r=0.92, region renders **on the belly**), PC3↔tail (r=0.95, **on the tail**);
    stripe appears as **periodic dorsal bands** distributed weakly across PC1/PC4/PC5/PC8 (corr
    ≤0.31); **cheeks tracks NO PC (≤0.12)** — being absent from the morphospace, its region can't
    be back-projected. So PC→region localization is real and readable for dominant factors.
  - **Q2 — the mechanism is sound but ROI-identification + feature choice are the crux:**
    | factor / ROI | sup balacc | unsup kNN | GMM ARI |
    |---|---|---|---|
    | stripe, whole-fish combined | 0.98 | 0.60 | ~0 |
    | stripe, **oracle dorsal ROI + darkness stats** | – | **0.71** | **0.20** |
    | stripe, label-free L-var ROI (too big) + crude colour stats | 0.79 | 0.58 | 0.04 |
    | cheeks, whole-fish | 0.61 | 0.50 | ~0 |
    | cheeks, **oracle cheek ROI + chroma** | – | 0.50* | **0.22** |
    | cheeks, label-free novelty ROI (landed OFF the cheek) | 0.51 | 0.50 | ~0 |
    Scoping to the RIGHT region with a matching feature **removes the masking by dominant colour
    variance** and lifts unsupervised recovery (stripe kNN 0.60→0.71, ARI→0.20; cheeks ARI→0.22) —
    the interactive, generalized form of EXP-02's dorsal-mask win. *(cheeks kNN stays 0.50: 14
    positives can never win a 7-NN vote; GMM-ARI is the right metric and it does move.)
- **Conclusion / verdict:** the owner's idea is **mechanistically sound and worth building**, with
  two honest constraints. (1) **ROI identification is the bottleneck, and it splits by factor
  type:** PC back-projection finds the region for DOMINANT factors (which are already easy), but a
  rare/subtle factor is absent from the top PCs, so there is **no PC region to back-project** — the
  label-free novelty ROI for cheeks landed OFF the cheek (`results/exp37/roi_cheek.png`). This is
  exactly where the **human-in-the-loop step pays off**: a biologist can SEE and circle a red cheek
  or a stripe band the morphospace misses, providing the ROI the algorithm cannot. (2) **Magnitude
  is bounded by the factor's intrinsic separability** — scoping unmasks the factor from colour but
  cannot beat n=14 (cheeks) or the 4-vs-5 overlap (stripe). (3) **Feature must match the region**
  (chroma for cheeks, darkness/texton for stripe); scoping + wrong/crude feature loses.
- **Recommended design:** (a) ship **per-axis back-projection** (per-face |corr| saliency rendered
  + contour) — strong interpretability, works now for the dominant axes; (b) let the user **circle
  a region** (especially for visible rare/subtle features the morphospace misses) → scope the
  descriptor + ranking + morph to that ROI with a region-appropriate feature; (c) decompose the
  global morph into **per-region/per-axis** morphs (each scoped sub-descriptor is low-dim and
  factor-dominated, so the morph along it is easy). This is the principled, interactive successor to
  the hand dorsal-mask, and the right way to inject human spatial attention into the morphospace.

### EXP-38 — Purely UNSUPERVISED localized clustering: tile the body, cluster per region, gate  ·  ✅ DONE
- **Owner Q (pausing the expert loop):** can region-local information improve *purely
  unsupervised* clustering? (final product = a global vs localized flag.) `exp38_localized_clustering.py`.
- **Method:** tile the shared mesh into R=128 spatial regions (KMeans on centroids, cached);
  per region build a per-specimen descriptor [area-wt mean Lab + the MAX-chroma face's (a*,b*)
  — area-independent, catches a rare vivid patch]; GMM-2 cluster WITHIN each region; gate each
  with a best-axis dip test vs a column-shuffled null (95th pct). Label-free throughout; ground
  truth only validates. Compared to a global GMM-2 on the whole-fish combined descriptor.
- **Result:**
  | factor | GLOBAL GMM-2 ARI | best-REGION ARI | region dip > null95? |
  |---|---|---|---|
  | belly | 0.04 | **1.00** | **YES** |
  | tail | 0.00 | **1.00** | **YES** |
  | stripe | −0.00 | **0.51** | no (sub-threshold) |
  | cheeks | 0.00 | **0.21** | no (sub-threshold) |
  - **Localized clustering recovers belly & tail PERFECTLY (ARI 1.0) and gates them significant**,
    where a single global GMM-2 on the entangled combined descriptor gets ~0 — because each region
    *isolates* its factor from the others' variance. It also **partially surfaces stripe (0.51)
    and cheeks (0.21)** — better than any global view (≈0) — but the dip gate correctly marks
    these **sub-threshold** (suggestive, not clean clusters), consistent with C8.
  - **Label-free "structure map"** (`results/exp38/structure_map.png`): painting each region by
    its clusterability (−log10 dip p) lights up the **belly brightest** (its 2 hue groups) with
    secondary structure near tail/head, and leaves the unimodal base-colour body dark — a
    biologist-readable map of WHERE the population has discrete groups, with no labels.
- **v2 — per-region TEXTURE feature + combined mode (owner ask).** Added a per-region banding/
  edge descriptor from the local jet (|band-pass L*| = stripe edges, multiscale band-pass, local
  L* contrast, dark fraction) and a combined color+texture mode. Best-region ARI by mode:
  | factor | color | texture | color+texture | gated-significant? |
  |---|---|---|---|---|
  | belly | 1.00 | 0.95 | 1.00 | YES (color, color+tex) |
  | tail | 1.00 | 0.13 | 1.00 | color YES |
  | **stripe** | 0.51 | **0.54** | 0.54 | **no** (all sub-threshold) |
  | **cheeks** | 0.21 | **0.46** | 0.41 | **no** (all sub-threshold) |
  - **Texture barely moves stripe (0.51→0.54) and it stays sub-threshold.** Why: a per-region
    banding-energy feature measures stripe DENSITY, and a ~576-face tile is too small to COUNT
    stripes; stripe-count is a body-spanning, overlapping 80/20 factor, so it caps at ARI ≈0.5
    UNSUPERVISED regardless of feature (energy or count — cf. EXP-11's measured-count ARI 0.51).
    To reach stripe you need a body-spanning DORSAL region + a peak-COUNT feature (the measured
    trait, EXP-02/05), not many small tiles — and even that is a gradient, not a gated cluster.
  - **Texture helps cheeks (0.21→0.46):** the cheek patch creates a local lightness EDGE the
    band-pass feature detects (best unsupervised cheek ARI seen) — but still sub-threshold (n=14).
  - **Structure map (combined, `structure_map_colortexture.png`):** belly stays brightest; the
    dorsal stripe band does NOT light up because stripe energy is not bimodal — the dip gate
    honestly declines to flag it.
- **Honest caveats:** (1) **multiplicity** — picking best-of-128 regions inflates ARI; the
  dip+null gate is what keeps it honest (belly/tail pass, stripe/cheeks don't). (2) The global
  baseline here (k=2 on the full descriptor) is deliberately weak; a *fair* global (GMM+BIC on
  the colour morphospace) also recovers belly×tail (EXP-32, ARI 0.99) — so the localized win is
  not "beats a good global at belly/tail" but **(a) auto-LOCALIZES structure (interpretable map),
  (b) DISENTANGLES factors (one region = one factor, vs an entangled global split), (c) partially
  reaches stripe/cheeks that no global view gets**. (3) Mean-colour per region captures stripe
  DENSITY/darkness, not count — region 60's 0.51 is suggestive; a per-region texture feature
  (texton/darkness-profile) would be the proper stripe channel.
- **Conclusion / product design:** region-locality DOES help unsupervised analysis — ship the
  **global | localized flag**: *global* = one gated clustering on the whole-fish descriptor;
  *localized* = the per-region scan that (i) emits the label-free clusterability **structure map**,
  (ii) returns, for each region that PASSES the gate, its own clustering + exemplars, and (iii)
  flags sub-threshold-but-suggestive regions as "look here." This is the unsupervised dual of
  EXP-37: there the human circled the region; here the gated scan proposes the regions. Natural
  next step: per-region *texture* descriptors (texton/darkness) so the dorsal scan can reach
  stripe, and a multiplicity correction (SHC/FDR over regions) for a publication-grade map.

### EXP-39 — How region analysis shapes the biologist-facing 2D EXEMPLAR plot  ·  ✅ DONE
- **Owner Q:** how does the region analysis affect a 2D exemplar plot the user inspects?
  `experiments/exp39_region_exemplar_views.py` → one composite figure
  (`results/exp39/region_exemplar_views.png`): a structure-map NAVIGATOR (top) + four exemplar
  morphospaces (global + belly/dorsal/head-cheek region-scoped), fish thumbnails placed by each
  view's PCA-2, borders = label-free GMM+BIC auto-clusters; regions picked LABEL-FREE.
- **Design:** the region scan converts ONE global plot into a **navigable set of region-scoped
  exemplar morphospaces**. Structure map = the navigator (click a bright region); each region
  re-lays-out the SAME 250 fish so that region's variation is the dominant axis. The product's
  global|localized flag = a **view selector**.
- **Honest result — region views foreground a factor as one of three shapes, only one of which
  is a cluster:**
  - **GLOBAL:** auto-clusters = belly (ARI 1.0); belly/tail show, stripe/cheeks invisible.
  - **BELLY region (colour):** auto-clusters track belly (ARI ~0.74 for the label-free-picked
    region) — but global already shows this, so the region view is redundant for the *dominant*
    colour factors.
  - **DORSAL region (texture):** stripe does NOT auto-cluster (ARI ~0.05) — it appears as a
    **gradient** the exemplars order along, not discrete groups.
  - **HEAD/CHEEK region (texture):** cheeks appear as **outliers**, not a cluster (auto-cluster
    ARI ~0) — visually flaggable but not a gated group.
- **Conclusion (the real answer):** an exemplar plot's value is that you can SEE structure even
  when it does not cluster. The region-scoped views surface a region's factor as **clusters
  (colour), a gradient (stripe), or outliers (cheeks)** — and the right user-facing coloring is
  therefore view-dependent: **auto-cluster for the gated colour regions; a label-free measured
  SCALAR (banding energy / max-chroma) for the gradient/outlier regions** so the ordering/outliers
  read without labels. Region analysis does NOT raise auto-cluster ARI for the overlapping/rare
  factors (they are not clusters), but it makes their variation INSPECTABLE via region-scoped
  exemplars — which is exactly what the plot is for. Net product: a global view (dominant colour
  clusters) + a structure-map navigator into region views (each foregrounding its factor as
  cluster/gradient/outlier), with coloring chosen per view.

### EXP-40 — Covariation-based mesh parcellation (owner-proposed): segment by what co-varies  ·  ✅ DONE
- **Owner idea:** global combined PCA → per-region share of each PC's variance (region×PC matrix)
  → merge regions into segments by that signature → re-run analysis on segments.
  `experiments/exp40_covariation_segments.py`. (This is **morphological integration/modularity**
  — Klingenberg — applied to colour+pattern: parcellate the body into modules that co-vary.)
- **Method:** global descriptor = CONCATENATION of per-region color+texture descriptors (128
  regions × 11 d); global PCA (K=20); region r's contribution to PC k = Σ squared loadings on
  region r's block (columns sum to 1 over regions); EV-weighted region signatures → Ward → M=7
  segments; render; per-segment GMM-2 ARI; re-run PCA on a segment-summary descriptor.
- **Result (`results/exp40/segments.png`):**
  - The parcellation is **anatomically coherent and recovers the dominant covariation modules**:
    a **belly module** (segs 0/1, ventral, GMM-2 ARI **1.0/0.97**) and a **tail module** (seg 3,
    caudal end, ARI **1.0**) — data-driven, correctly located, no labels.
  - **Stripe and cheeks do NOT form their own segments** (best per-segment ARI: stripe 0.07,
    cheeks 0.14): they don't drive the top PCs (low-variance/overlapping/rare), so covariation
    parcellation — like every variance-driven method — parcels by the DOMINANT (colour) structure;
    the dorsal/stripe field splits into a few colour-driven segments, not a stripe module.
  - **Re-run PCA on the segment summary** (compact, interpretable): belly **0.98**, tail 0.79,
    stripe **0.84** (supervised; best-axis AUC 0.80, silhouette 0.02 → recoverable, not a cluster),
    cheeks 0.50 (chance). So the segment-summary is a tidy descriptor that supports supervised
    stripe recovery but does not surface cheeks.
- **Conclusion:** the idea **works and is valuable** — it yields a **data-driven atlas of colour/
  pattern covariation modules** (belly, tail, dorsal field, tail-base…), which is (a) a better,
  factor-aligned set of regions for the EXP-38/39 localized views than arbitrary KMeans tiles, and
  (b) a publishable **modularity/integration** analysis in its own right (which body parts vary
  together across the population). The honest limit is the familiar C8 wall: covariation
  parcellation isolates the **prominent** modules (colour), not the subtle/overlapping (stripe) or
  rare (cheeks) factors — those remain measured-axis / targeted problems, not emergent segments.
  **Recommended use:** adopt the covariation segments as the region units for the localized
  morphospace navigator (EXP-38/39), and report the segmentation itself as a colour-pattern
  modularity map; keep stripe as a measured banding axis and cheeks as a flagged rare variant.

### EXP-41 — Exemplar morphospaces on the COVARIATION segments (modules)  ·  ✅ DONE
- `experiments/exp41_segment_exemplars.py` → `results/exp41/segment_exemplars.png`: the EXP-40
  module atlas (navigator) + the segment-summary morphospace + per-module exemplar morphospaces,
  fish thumbnails placed by each view's PCA-2. Module DISPLAY chosen label-free by clusterability
  (top-2 → belly & tail modules) + the most-dorsal module. Coloring is view-adaptive (EXP-39
  finding): discrete **auto-cluster** colors when a view's axis is bimodal (belly & tail modules,
  segment-summary), continuous **PC1 gradient** otherwise (dorsal module). Confirms the assembled
  product: a covariation-module atlas → segment-summary morphospace → per-module exemplar views,
  each foregrounding its factor as clusters (belly/tail) or a gradient (dorsal striping).

### EXP-42..45 — Recovering the 2³=8 factor structure from graded expert feedback  ·  ✅ DONE
Motivation (interactive demo): fishy has 3 ~binary factors (belly, tail, stripe) → 8 true clusters.
Two worries: (i) most pairs share 1–2 of 3 factors → graded "similar"; clean all-3-different "D 1.0"
pairs are rare → not enough repulsion; (ii) a 2-D spring layout cannot settle into 8. All labels
SIMULATED from GT, using the demo's own descriptor (`interactive_demo.engine`).
- **EXP-42** (`exp42_8cluster_feedback.py`): structure + budget. Color (belly/tail) is trivially
  separable (0.99/1.00), **stripe is the bottleneck** (0.84 in the demo's PCA-10, and the 21%
  minority → imbalanced cells `[50,18,48,9,48,9,52,16]`). Pair-Hamming budget: H=0 17%, H=1 42%,
  H=2 33%, **H=3 only 8.4%** → worry (i) confirmed. Cube-MDS stress 2-D 8.3 vs 3-D 3.2 → worry (ii)
  confirmed. Baseline KMeans-k8 ARI 0.23.
- **Fix = graded TARGET DISTANCES** (target d² = Hamming): every non-identical pair separates
  proportionally, so partial-overlap pairs do the pushing — no need for the rare clean-D.
- **EXP-43** (`exp43_repr_metric_ceiling.py`): the demo's **K=10 PCA is the real bottleneck** — the
  FULL color+jet descriptor separates stripe at **0.988**; PCA-10 crushes it to 0.84. **Textons are
  NOT the lever here** (0.88 < 0.988 on fishy stripe). Keeping more PCs lifts the supervised ceiling:
  K=10→0.57, K=20→0.88, K=30→0.96, K=50→1.0. Method: diagonal metric too weak (~0.40); full
  Mahalanobis overfits at high K.
- **EXP-44** (`exp44_recover_grid.py`): **low-rank metric (rank≈4) is the method win** — finds the ~3
  discriminative directions from few labels. Recipe **K≈24 + graded targets + rank-4 metric + GMM(8)**
  → ARI ≈0.84 @200 labels, ≈0.92 @400. **Active** sampling (query within/across the *current* clusters,
  no GT) beats random in the practical range (0.75→0.84 @200). Visualization = **FACTOR GRID** (4 color
  quadrants × split by the recovered weak factor) since 2-D springs can't show 8.
- **EXP-45** (`exp45_harden.py`): noise robustness — per-feature judgment error q: holds well to q≈0.05
  (ARI 0.70/0.83 @200/400), usable to q≈0.1, **more labels buy back noise**, collapses past q≈0.2.
  Clean balanced-bisection 2×2×2 factor grid + diagonal-ordered confusion matrix.
- **Recipe to port into the demo:** raise PCA K 10→~24; reframe feedback as a graded target distance
  (one "how similar 0–1" → rest length); swap the explain/metric diagonal→low-rank; do recovery by
  clustering in the K-space (GMM-8) and SHOW it via a factor grid / color-by-recovered-8, not by hoping
  the 2-D springs separate 8. Caveats: simulated exact-count labels (real experts noisier — see q sweep);
  no GT on mussels to validate recovery there.

### EXP-46 — Multi-label TAG feedback vs pairwise: which classifier best PLACES unlabeled?  ·  ✅ DONE
Idea: instead of "more/less similar", the expert creates tags ("green tail", "4 stripes", …) and drags
them onto specimens (multi-label); train a classifier and watch it PLACE the unlabeled (its latent space).
Shared harness `experiments/_tag_harness.py` (cached PCA-24 coords; 6 factor-ordered tags; eval = per-factor
acc + 8-way ARI on UNLABELED vs # tagged specimens). 5 model families (4 fanned out + verified, + linear
baseline), all leak-free & reproduced:
| model | ari8@40 | ari8@80 | ari8@160 | acc@160 | 2-D latent silhouette@80 |
| linear logistic (per-tag) | **0.68** | **0.86** | **0.89** | **0.98** | 0.28 |
| prototype / NCM | 0.49 | 0.74 | 0.86 | 0.98 | 0.16 |
| MLP 2-D bottleneck | 0.37 | 0.49 | 0.55 | 0.90 | **0.31** |
| label-prop (semi-sup kNN) | 0.19 | 0.23 | 0.42 | 0.87 | 0.01 |
| NCA 2-D embed | 0.23 | 0.20 | 0.40 | 0.83 | 0.04 |
Findings: (1) **TAG feedback is far more label-efficient than pairwise** — linear hits ari8 0.86 at 80
tagged specimens (~240 drags) and per-factor acc 0.93 at 40, vs ~200–400 *pairs* for similar ARI in
EXP-44; and it yields direct per-specimen class predictions, not just a clustering. (2) **Best PLACER =
simple per-tag linear logistic** in the full 24-PC space (factors are ~linearly separable there);
prototype/NCM a close, tiny-budget-robust 2nd. (3) **Semi-supervised label-prop and NCA-2D UNDERperform** —
the descriptor-kNN graph is dominated by color, so propagating smears the subtle stripe factor; a 2-D
bottleneck loses stripe (same dimensionality limit as EXP-43). (4) **Classifier ≠ visualization:** the MLP
2-D bottleneck gives the cleanest 2-D latent (silhouette 0.31) but caps placement; the full-D linear places
best AND its PCA-2-of-tag-scores latent is nearly as clean (0.28). RECOMMENDATION for a demo "tag mode":
full-D linear (or NCM) for placement + recolour/predict, visualise via PCA-2 of the decision scores (or an
MLP bottleneck for a smoother map). Caveats: simulated tagger gives correct, complete (3/specimen) tags;
synthetic fishy is ~linearly separable so linear wins (real data may need the nonlinear model); noise &
partial-tagging not yet swept.

### EXP-47 — Displaying the tag-classifier latent in 2-D: UMAP vs PCA vs t-SNE  ·  ✅ DONE
`experiments/exp47_umap_latent.py`. The per-tag linear classifier's 6-D tag-score space is the latent;
project to 2-D for display. 8-way silhouette / trustworthiness on UNLABELED, by tag budget:
| 2-D method | sil@40 | sil@80 | sil@160 | trustworthiness |
| PCA-2  | 0.19 | 0.28 | 0.37 | ~0.93 |
| **UMAP-2** | **0.38** | **0.57** | **0.56** | **~0.99** |
| t-SNE-2 | 0.38 | 0.52 | 0.55 | ~0.99 |
**UMAP ~doubles the display quality of PCA** (and t-SNE ties it) — the 8 groups resolve into clean
separated islands at ~80 tags, vs a muddy blob under PCA. Unsupervised UMAP on the RAW descriptor (no
feedback) = silhouette **−0.07** (no visible structure) → the tags→classifier-latent is what creates the
structure; UMAP merely lays it out. So the recommended demo "tag mode" display = **UMAP of the classifier's
tag-score latent**, re-projected as tags accumulate. (Caveat: UMAP is stochastic — naive re-projection each
update will jump; use parametric UMAP or embedding alignment / re-run only on Apply.)

### EXP-48 — Non-stochastic, smoothly-evolving UMAP as tags are added  ·  ✅ DONE
`experiments/exp48_aligned_umap.py`. Tag MONOTONICALLY (fixed order, budget = prefix) and embed the SEQUENCE
of tag-score latents with `umap.AlignedUMAP` (identity relations, deterministic random_state=0) so the map
evolves smoothly instead of teleporting. Smoothness (mean per-point frame-to-frame motion, scale-normalized):
naive independent UMAP **1.17** → independent + Procrustes **0.49** → **AlignedUMAP 0.31** (~4x steadier).
8-way silhouette of the aligned frame climbs and STABILISES: 5t −0.23 · 20t 0.14 · 40t 0.34 · 60t 0.45 ·
80t 0.54 · 160t 0.52 — structure emerges from noise and locks in by ~80 tags while clusters stay spatially
anchored (sharpen in place). Outputs `aligned_umap_grid.png` + `aligned_umap.gif`. → demo tag-mode display:
AlignedUMAP over the tag-budget sequence (or parametric UMAP for live single-point `transform`).

### EXP-49 — Parametric UMAP for a LIVE single-update tag morphospace  ·  ✅ DONE
`experiments/exp49_parametric_umap.py` (real `ParametricUMAP`, run in an ISOLATED tf env at /tmp/pumap_env —
NOT the research venv, which is numpy-2.4 / uv-locked and would break under TF; install: `uv venv` +
`tensorflow-cpu tf-keras umap-learn`, run with `TF_USE_LEGACY_KERAS=1`). Fit the encoder ONCE at a tag
budget, then `.transform()` every later latent (true live update, no re-fit). RESULT — *contradicts* the
"won't be as nice as aligned" prior:
| method | motion (smooth) | sil@40 | sil@80 | sil@160 |
| parametric frozen@80 | **0.207** | 0.33 | 0.60 | **0.64** |
| AlignedUMAP | 0.261 | 0.37 | 0.52 | 0.50 |
| independent UMAP | 1.161 | 0.34 | 0.58 | 0.58 |
Parametric is the SMOOTHEST (fixed encoder; only the input latent drifts) AND highest final quality (its map
fit on the mature 80-tag latent generalises cleanly to 120/160 since structure plateaus by ~80 — EXP-48),
AND it's the only one that supports a genuine live single-point update. Only cost: a map frozen@80 is a bit
worse than AlignedUMAP at VERY early budgets (10–20 tags, sil −0.31 vs −0.12) — the under-tagged regime with
no real structure anyway. PRACTICAL recipe: fit the parametric encoder once ~40 tags exist, then all future
updates are instant + stable + sharp; re-fit occasionally if structure shifts. DEMO caveat: real PUMAP needs
TF (conflicts with the numpy-2.4 venv) — a TF-free stand-in (sklearn MLPRegressor: latent→reference-UMAP-2D,
fit once + `.predict` updated points) gives the same deterministic reusable-transform behaviour without TF.

### EXP-50 — Build/perf for the live tag morphospace (CPU, small-n)  ·  ✅ DONE
`experiments/exp50_update_bench.py`. Two-tier design: FREQUENT per-tag update (retrain 6 linear tag
classifiers + project the changed latent through a FROZEN encoder) vs OCCASIONAL re-layout (re-fit the
encoder). Measured (this CPU; project times on a *changed* latent, the realistic case):
| n | clf retrain | UMAP.transform (changed) | MLP.predict (cache) | UMAP refit (re-layout) |
| 250 | 3 ms | 140 ms (sil 0.51) | **0.04 ms** (sil 0.46) | ~0.2 s |
| 1000 | 4 ms | 750 ms (sil 0.74) | **0.11 ms** (sil 0.68) | ~0.9 s |
RECOMMENDATION: server-side Python (sklearn + umap-learn, **no TF/GPU**). Per Apply = logistic retrain (ms)
+ a frozen-encoder projection → send 2-D coords + tag predictions → client tweens nodes. For the expected
scale (hundreds), **UMAP fit-once + `.transform()` per update (~140 ms) is simplest and good enough**; add a
sklearn **MLPRegressor cache** (latent→reference-UMAP-2D, `.predict` ~0.1 ms, ~same quality) only if you want
sub-ms updates or n→thousands — that's the TF-free "parametric UMAP" (matches EXP-49 behaviour, no TF dep).
Re-layout (re-fit UMAP + MLP, ~0.2–1 s) on a button or auto on drift, in a background thread. Pre-warm UMAP's
numba JIT at server start (~10–20 s one-time) so the first interaction isn't slow.

### EXP-51 — React after EVERY batch (from the 1st) AND stay smooth  ·  ✅ DONE
`experiments/exp51_incremental.py`. UX fix to the "fit @80" idea: do NOT freeze — re-fit UMAP each tagging
batch but WARM-START with `init=previous_embedding` + few epochs (n_epochs=60) so points carry over.
Motion (frame-to-frame, lower=steadier) / final silhouette over batches [8,16,24,32,48,64,96,128,160]:
warm-start **0.41 / 0.58** · naive re-fit **1.02** / 0.56 · AlignedUMAP-ref 0.26 / 0.56. So warm-start is
~2.5x steadier than naive at full quality, reacts from batch 1 (8 tags already renders a layout), and clusters
sharpen IN PLACE. AlignedUMAP `.update()` (incremental) is smoothest (0.28) but **7.8 s/update** → unusable
live (offline replay only). RECOMMENDED BUILD for tag mode: per batch = retrain logistic (C=0.5 regularised
for tiny budgets) + warm-started UMAP fit (init=prev, ~60 epochs, ~0.2 s @ n=250) → client TWEENS nodes to
the new coords for smooth perceived motion. No freeze, no TF, no GPU; reactive from the first batch.

### EXP-52 — Active batch selection for the tag loop  ·  ✅ DONE
`experiments/exp52_active_tags.py`. After applying a batch, auto-roll the NEXT batch picked to most help the
classifier. Held-out 8-way ARI vs # labeled (mean of 6 seeds): UNCERTAINTY sampling (pick unlabeled with
highest mean per-tag uncertainty, prob nearest 0.5) ≫ random — 24:0.57/0.39, 40:0.83/0.58, 56:0.92/0.73,
80:0.96/0.81, plateau 0.94 vs 0.85. Uncertainty+diversity HURTS at tiny budgets (24:0.31) though it edges
ahead late (160:0.98) → pure uncertainty is the robust default. IMPLEMENTED in `expert-tagging-demo`:
`TagSession.roll_batch(active=True)` ranks unlabeled by `mean_t(1-|2p-1|)`; the `/apply` endpoint auto-rolls
an active batch (so the next, most-informative batch appears on Apply); falls back to random with no
classifier or when the "informative" toggle is off. API-verified: returned batch == top-k uncertain unlabeled.

### EXP-53 — PCA dims (color vs pattern/spectral) vs accuracy, and does explained variance predict it?
`experiments/exp53_dims_vs_accuracy.py`. Sweep #PCA dims separately for COLOR (per-region Lab blocks) and
PATTERN/SPECTRAL (GFT coeffs k=40, L/a/b); per #dims: per-factor 5-fold logistic acc, GMM cluster ARI,
cumulative EV. Findings:
- COLOR: belly/tail classifier hit 1.0 by **d≈3–4**; belly×tail clustering ARI 0.95 by d≈8. Cumulative EV-95%
  only at **d=15** → accuracy saturates LONG before EV (EV over-counts).
- PATTERN/STRIPE: stripe classifier climbs 0.79→0.96 and only saturates at **d≈30**, while EV-95% is at
  **d=12** → the discriminative signal lives in the LOW-variance PCA tail; **EV UNDER-counts the needed dims
  ~2.5×**. Stripe **never** clusters unsupervised (ARI ≈0 at every dim, up to 60).
- Spearman(EV, acc) is high (+0.86..+0.99) but MISLEADING — both are monotone; the SATURATION dim diverges
  (EV over-counts color, under-counts stripe). So **explained variance does NOT predict the dims you need**,
  and no purely unsupervised metric (EV or clusterability) can flag the low-variance stripe factor — it's
  invisible to variance and to unsupervised clustering (HDLSS / spiked-covariance). Practical: size PCA by the
  SUPERVISED factor of interest (or just keep ~30 dims to preserve the subtle one — matches the demo's K≈24),
  NOT an EV threshold (which would silently discard stripe at ~12–15 dims).

### EXP-54 — GFT mode count `k` vs accuracy (the spectral_coeffs truncation)
`experiments/exp54_k_vs_accuracy.py`. EXP-53 swept PCA dims on a fixed k=40 descriptor; this varies `k` (# low-
freq Laplacian modes/channel) and classifies directly on the k*3 coefficients. Findings:
- COLOR (belly/tail) saturates at **k=1–3** (belly 0.91 / tail 0.96 from a SINGLE mode → ~1.0 by k=3) — color
  is concentrated in the lowest modes (broad gradients).
- STRIPE climbs **0.79(k=1) → 0.96(k=40)**, marginal to 0.976(k=150). Default k=40 is a sensible knee.
- WHERE stripe lives: a per-mode 1-feature probe shows stripe info is THINLY DISTRIBUTED across MID-frequency
  L modes (best single modes ≈ idx 17/28/46/59 at only ~0.84–0.85 vs majority 0.79); no low mode carries it.
  So you need many modes to accumulate the signal, and small-k truncation silently drops stripe — same theme
  as EXP-53 (stripe sits in the spectral tail / mid-band). Keep k≳40 to preserve it.

### EXP-55 — Discovering GFT k without ground truth (3 methods) + ICA vs PCA on the spectral data
`experiments/exp55_*.py`, shared cache `results/exp55/cache.npz` (GFT coeffs CL/Ca/Cb 250x300, eigvals, GT).
HOW TO PICK k:
- CV-on-your-own-LABELS (`label_budget_k`): track the SLOWEST-saturating factor, NOT the mean (mean knee ~4-6
  saturates when belly/tail max → undercounts). Recovering stripe's k≥40 knee needs ~80 labels (M=20 useless,
  M=40 a weak hint); binding constraint = rare-class COUNT (stripe 21% → ~4 positives at M=20). Tends to
  recover "k≥40, take the largest labels still support".
- EV-threshold & clustering-STABILITY (`unsup_k`): BOTH FAIL — pick k≈12-18, lock onto belly (ARI~1.0), MISS
  stripe (ARI~0), leaving 6-8 pts. 99% EV (k=39) only incidentally OK. Don't use variance/clusterability for a
  low-variance factor.
- Unsupervised SPECTRAL structure-floor / eigenvalue elbow (`unsup_floor`): points to keeping MANY modes (~145)
  which safely captures stripe (0.976). Per-mode energy is NON-monotone (no clean freq order). Practical rule:
  over-supplying k is nearly free (accuracy plateaus, doesn't collapse) → default generous k≈50-150, trim via
  label-CV on the hardest factor.
ICA:
- `ica_vs_pca_stripe`: FastICA(whiten) ≡ PCA for these models — BYTE-FOR-BYTE identical accuracy at every n
  (rotation within the top-n PCA subspace; logistic + full-cov GMM are invariant to invertible linear maps;
  subspace residual ~3e-6). So ICA is NOT a stripe-recovery lever (won't get stripe in fewer components).
- `ica_disentangle`: ICA DOES isolate the strong independent COLOR factors into single clean components
  (tail 0.996 vs PCA 0.884; belly 0.980) — good for interpretability — but stripe stays weak/spread
  (best single comp ~0.86 ≈ a raw GFT mode); use the multi-mode classifier for stripe, not one component.

### EXP-56 — Class-imbalance handling for rare factors (stripe 21%, cheeks 5.6%) while tagging
`experiments/exp56_imbalance.py` (balanced accuracy on held-out, mean of 40 seeds; feature = demo PCA-24).
Methods: plain logistic vs `class_weight='balanced'` vs manual SMOTE vs oracle stratified-acquisition+balanced.
- STRIPE (21%): reweighting/SMOTE give a BIG boost in the LOW-label regime — M=40 plain 0.756 → balanced 0.842
  (+0.09); gap fades by M=160 (0.913 vs 0.927). **`balanced` ≈ SMOTE** (no gain from synthetic oversampling).
  strat-acq (oracle minority-aware acquisition) only marginally beats balanced for stripe.
- CHEEKS (5.6%): everyone struggles (0.52→~0.68 even at M=160) — at M≤40 there are only ~1–2 positives, so no
  reweighting/oversampling can conjure signal; the binding lever is ACQUIRING minority examples (active /
  stratified), and 5.6% @ N=250 stays hard regardless.
RECOMMENDATION for the tagging demo: add **`class_weight='balanced'`** to the per-tag logistic (free, helps rare
tags exactly in the early incremental regime; matches SMOTE without the dependency); rely on the EXP-52 active
uncertainty sampling to surface rare-class examples. SMOTE not worth the imblearn dependency here.

### EXP-57 — SMOTE vs ADASYN (vs class_weight, plain) for rare factors
`experiments/exp57_smote_adasyn.py` (balanced-acc + minority recall on held-out, 50 seeds, manual SMOTE/ADASYN
since imblearn pins sklearn). **SMOTE ≈ ADASYN — indistinguishable** (within ~0.002–0.003 at every M, both
stripe 21% and cheeks 5.6%). ADASYN's boundary-density weighting gives no edge because the binding constraint
is the tiny minority COUNT — r_i is estimated from too few points and at low prevalence nearly all minority
points are "hard", so ADASYN collapses to ~uniform = SMOTE. Nuance for cheeks: at the SMALLEST budgets
class_weight='balanced' beats both synths (it upweights a single positive; SMOTE/ADASYN need ≥2 to interpolate),
while synths edge ahead at M≥80. CONCLUSION unchanged: keep `class_weight='balanced'` (free, ties/【wins,
no imblearn dependency); neither SMOTE nor ADASYN worth adding, and ADASYN buys nothing over SMOTE here.

### EXP-59 — Do more color-PCA/ICA components help the rare cheeks factor?
`experiments/exp59_color_components_cheeks.py` (color-only 640-dim descriptor; balanced-acc, repeated
stratified 5-fold CV, class_weight='balanced'). YES more components help cheeks — its signal is in the
LOW-VARIANCE TAIL: cheeks acc recovers only by n≈32-48 while cumulative EV is 96% by n≈12 (EV under-counts
hugely). Striking U-SHAPE: at n≈8-16 cheeks dips BELOW chance (~0.43) — the top color PCs (belly/tail variance)
mislead the rare signal until you go deep enough. Ceiling still low (~0.62 bal-acc) — cheeks is rare AND its
color signal is weak. ICA ≡ PCA (overlapping curves, linear-invariant per EXP-55) AND ICA gives cheeks NO
dedicated component (best single ICA comp 0.62, gap 0.006 — unlike belly/tail which ICA isolates cleanly). →
keep enough PCA dims to include the tail (≥~32 / full descriptor) for rare low-variance factors; ICA not a lever.

### EXP-58 — Improving the rare CHEEKS factor (5.6%) in the incremental tagging pipeline (4 directions)
`experiments/exp58_*.py`, shared cache `results/exp58/cache.npz`. Cheeks plateaus ~0.66 balanced-acc even with
all positives (oracle ceiling) — fundamentally hard; the wins are DISCOVERY SPEED, not final accuracy.
- REPRESENTATION (`cheeks_repr`): cheeks wants MORE dims — full-Xs PCA-60/100 beats demo Z24 at M≥80
  (BA 0.627/0.695 vs 0.614/0.678), monotone in dims (tail matters, cf EXP-59). Supervised feature/region
  selection FAILS (too few positives to rank). Budget-gate: Z24 while M<60, swap to PCA-60/100 at M≥80.
- ACQUISITION (`acquisition`): UNCERTAINTY sampling accumulates positives fastest (3 positives @~39 labels vs
  ~54 random). **NN-expansion / "find more like this" FAILS** (worse than random — the tiny cheeks cluster
  drains, then it wastes budget on near-duplicate negatives). Density-weighting = no gain. The demo's existing
  active uncertainty roll (EXP-52) is already right. Judge by positives-discovered (accuracy plateaus ~0.66).
- COLD-START (`coldstart_discovery`): **the big new win** — unsupervised novelty ranking on Z24 (LOF or
  mean-kNN-distance) hits the FIRST cheeks at rank 1 (vs ~16 labels random) and 3 by labels 3–5 (vs ~49),
  ~10–16x speedup, harvesting ~6/14 cheeks in the first ~16–20 labels, then saturates → switch to active.
  (IsolationForest on raw color-640 is worse than random.)
- CLASSIFIER TRICKS (`loss_threshold_tricks`): NOTHING beats plain `class_weight='balanced'` + 0.5 threshold —
  focal/balanced-bagging tie, threshold-CV overfits (negative @M120), calibration HURTS recall. Binding
  constraint = minority COUNT, not loss machinery → keep the current classifier.
RECIPE: cold-start novelty seeding (round 0) → active uncertainty roll (already in demo) → budget-gated
higher-dim PCA → keep balanced logistic. Highest-value NEW addition = cold-start novelty queue ordering.

### EXP-60 — Do anomaly-detection methods surface the minority classes?
`experiments/exp60_anomaly_minority.py` (ROC-AUC of unsupervised anomaly score vs minority membership on Z24).
YES, but it's about SEPARABILITY not frequency. AUC: cheeks(5.6%) 0.63-0.67 · **5-stripe(21%) 0.71-0.76** ·
random-6% control **0.50**. So (a) both planted minorities are surfaced well above the random-rare control
(which sits at chance → anomaly does NOT surface arbitrary rare sets); (b) the RARER cheeks is surfaced LESS
than the more-common 5-stripe → "rarer ≠ more surfaced"; what matters is whether the class occupies a
lower-density / atypical region (5-stripe = structurally distinct extra-stripe texture). kNN-dist & LOF best;
top-20 cheeks enrichment ~5-6x. Feature space matters: Z24 (compressed PCA) AUC 0.67 >> raw colour-640 0.53
(curse of dimensionality kills distance-based anomaly in raw space) — do anomaly scoring in the compressed
space. Reconciles with EXP-42/53 (stripe not cleanly CLUSTERABLE, ARI~0) — AUC 0.76 is soft 1-D enrichment,
"enriched but overlapping", not separation. Practical: anomaly/novelty ranking is a good cold-start to surface
ANY separable minority (generalises beyond cheeks), complements active learning; won't surface a truly
inseparable minority.

### Resume point / open
- New modules: `fishpipe/textons.py` (shared-codebook descriptor), `fishpipe/gating.py`
  (SigClust + consensus + Jaccard), `fishpipe/semisup.py` (constraint warp + active pairs +
  `triplet_warp` + `spring_embed` for the ranking/relevance-feedback loop). Experiments
  `exp29..38`, results in `results/exp29..38/`. (EXP-35 = anchor+rank linear-warp loop; EXP-36 =
  spring/energy morphospace + diverse panels + per-iteration exemplar plots/GIF; EXP-37 = per-axis
  back-projection to body regions + region-scoped comparison; EXP-38 = purely-unsupervised
  localized per-region gated clustering + label-free structure map; global|localized flag;
  EXP-39 = structure-map navigator + region-scoped exemplar morphospaces / view selector;
  EXP-40 = covariation-based mesh parcellation into colour/pattern modules + segment morphospace;
  EXP-41 = exemplar morphospaces on the covariation modules — atlas navigator + per-module views.)
- HIGH VALUE next: (a) port texton-BoW into the module as the pattern descriptor; (b) implement
  unbalanced SigClust (Keefe-Marron 2023) for the rare-cheek class — the one gate aimed at it;
  (c) the pairwise-constraint UI ("same pattern? yes/no" on rendered pairs) feeding the warp;
  (d) test SNF + Mapper on the real n=25 cichlids and n=31 mussels.
