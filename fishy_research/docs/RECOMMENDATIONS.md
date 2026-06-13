# Recommendation: making population segments visible in the color-DeCA morphospace

**Audience:** the team maintaining the 3D Slicer color-DeCA module. **Validated on:** the
synthetic *fishy* dataset (250 specimens, known ground truth). See `RESEARCH_LOG.md` for the
full experimental trail and `docs/methods_survey.md` for the literature basis.

## TL;DR
The existing color PCA **already solves color/hue clusters** — it just can't show **structure**
(stripe count, pattern), which is exactly what biologists care about. The fix is **not a better
PCA**; it is to **measure structure directly** and offer it as a named axis, because subtle
structure is low-variance and *invisible to any variance-driven reduction* (we confirmed this for
PCA, ICA, and UMAP on several feature sets). Net result on fishy (balanced CV accuracy):

| factor | before (color PCA) | after | mechanism |
|---|---|---|---|
| belly hue (2 clusters) | 0.996 | 1.000 | unchanged — color morphospace |
| tail hue (2 clusters)  | 1.000 | 1.000 | unchanged — color morphospace |
| **stripe count (4/5)** | **0.64** | **0.89–0.93** | **general graph-spectral descriptor** (or optional tuned banding axis) |
| rosy cheeks (5% rare)  | 0.48 | 0.78 | graph-spectral chromatic rare-variant flag |

> **Generality (EXP-07).** The structural engine is a **general graph-spectral descriptor** that
> makes *no* assumptions about stripes/axis/colors — one descriptor recovered all four very
> different feature types (hue clusters, periodic banding, rare spot) and, with enough modes,
> **beats** a taxon-tuned stripe detector (0.93 vs 0.89). The body-axis "banding count" detector
> is **NOT general** (axial, dark-on-light, periodic) → it is an *optional* interpretability
> plug-in for striped taxa, not the engine.

## The single most important finding
**Discovery ≠ measurement.** A morphospace built by maximizing variance (PCA/ICA/UMAP) only
surfaces the *highest-variance* differences. On fishy that is color (belly/tail hue), which
dominates and produces a clean 4-quadrant plot. Stripe count is a **low-variance, position-
jittered** difference: a classifier can recover it (~0.9) from good features, but it **never
lands on the top PCs/ICs** — so it stays invisible in the plot. The only reliable way to show
structure is to **measure it with a label-free detector and plot that measurement as an axis**
(this is also what biology pattern tools like `patternize`/QCPA do — named traits, not abstract PCs).

## Recommended UX (minimal options)
One control: **View = [ Color | Pattern ]**, plus automatic cluster coloring. Nothing else the
user must understand.

1. **Color view** *(keep what exists)* — PCA on the area-weighted color-composition vector;
   auto-cluster with GMM+BIC and color the points. On fishy this auto-finds the 4 hue groups
   (ARI 0.99) with no tuning.
2. **Pattern view** *(new, the real gain)* — built on a **general, label-free structural
   descriptor library** (primary: **graph-spectral / manifold-harmonic band energy** on L\*/a\*/b\*/
   chroma; plus patch-adjacency [Endler/QCPA], component counts, chromatic novelty). This makes
   *no* pattern-type assumption, so it handles stripes, spots, reticulation, blotches, etc.
   - **Surfacing structure:** because subtle structure is low-variance (won't be a top PC), either
     (a) **auto-rank** descriptors by how much they cluster the population (a multimodality score)
     and offer the top few as **named axes**, or (b) let the user pick a named axis. Auto-cluster +
     color the points as above.
   - **Optional taxon plug-ins:** when the system is known (e.g. striped fish), add a directly-
     measured **banding-count axis** for sharpness/interpretability — *optional*, not the engine.
3. **Rare-variant overlay** — a chromatic novelty / graph-spectral chromatic score; draw flagged
   specimens (e.g. rosy cheeks) with a marked outline beside either view.
4. **Region segmentation** *(optional toggle)* — compress faces → K spatial regions before
   analysis. Speeds things up, denoises, and **area-normalizes small features** so rare patches
   aren't drowned (recovered cheeks 0.48→0.66 on its own).

## How it maps onto the code (`fishpipe/recommended.py` is the reference implementation)
- `color_morphospace(fcd)` → PCA on `features.area_hist` (≈ the module's existing area-weighted
  vector). Reuse the module's `performPopulationAnalysis` area-weighted path.
- `pattern_traits(fcd)` → label-free banding traits. Core idea, ~25 lines:
  1. `principal_axis()` = top SVD axis of face centroids (auto body axis; no hardcoding).
  2. `salient_pattern_mask()` = faces whose **lightness varies most across the population**
     (where a shifting pattern lives) — a general, label-free replacement for a hand "dorsal" mask.
  3. area-weighted **darkness (100−L\*) profile** along the axis within that mask →
     **peak-count** (interpretable "# bands") + profile std + low-band FFT power.
- `rare_variant_descriptor(fcd)` → `spectral.spectral_descriptor(channels=('a','b','chroma'))`
  (graph-spectral chromatic band energy; best general rare-color detector, ≈0.78).
- `auto_cluster(scores)` → GMM with BIC model selection for point coloring.

The graph-spectral machinery (`fishpipe/spectral.py`, manifold-harmonics band energy, computed
once per mesh) is the **general, UV-distortion-proof** substrate for the pattern/rare-variant
descriptors — important for the **real cichlid data where meshes/UVs differ** (unlike the toy
fishy where all share one mesh). Recommended to adopt it there.

## Validation (EXP-06 audit, `results/exp06_audit/`)
Robust and leakage-free: shuffled-label **negative control → 0.50** (real 0.89); seed-stable
(stripe 0.863–0.887, cheeks 0.775–0.794); **no ground-truth leakage** (features take only colors,
population stats are label-free & ~97.5% stable on a random half). One real knob: the salient-mask
`quantile` — stable at **≥0.80** (~0.89), degrades to ~0.74 at 0.70; default is **0.85**.

## Honest limitations
- **Stripe ceiling ≈ 0.89** (not 1.0): the 5th stripe is sometimes faint after baking + noise; a
  measured band-count occasionally reads a 5-stripe fish as 4. Precision of "5 bands detected" is
  ~100%; recall is the soft spot.
- **Rosy cheeks ≈ 0.78**: 5% prevalence + tiny surface area is a fundamental statistical limit
  (as flagged in the original Slack thread). Region segmentation + chromatic-spectral helps but
  does not fully solve it. Present it as a *flag*, not a morphospace cluster.
- The body-axis banding detector suits **elongate, striped** taxa. For arbitrary patterns, build
  the pattern axes from the graph-spectral descriptor + a small library of label-free detectors;
  the principle ("measure structure, don't hope PCA discovers it") is what generalizes.
