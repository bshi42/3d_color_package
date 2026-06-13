# Methods survey — structural / pattern descriptors (CPU, interpretable)

Compiled 2026-06-01 from a literature scan (see RESEARCH_LOG EXP-03). Goal: separate
specimens by pattern **structure** (stripe count, texture), CPU-only, interpretable, simple
UI. The shared mesh + UV is our biggest asset: alignment is free, so heavy isometry-invariant
machinery is unnecessary.

## Recommended, in priority order
1. **Endler-style adjacency / color-transition matrix** on the color-segmented patch map
   (Endler 2012; `recolorize`/`patternize`/QCPA ecosystem, van den Berg 2020). Count
   color-class→color-class transitions across adjacent faces → transition matrix. Position-
   invariant, CPU-trivial, **publication-grade in the target field**, reuses the already-solved
   color quantization. More stripes ⇒ more black↔base transitions. *Lowest-risk, highest-acceptance.*
2. **2D Fourier + Gabor band-energy on the rasterized UV texture.** Shared UV ⇒ alignment is
   free; a UV power spectrum / Gabor bank captures stripe spacing + orientation as a few
   interpretable numbers. Much simpler than the mesh Laplacian, likely sufficient.
3. **Connected-component count + autocorrelation period** on a body-axis UV transect → an
   explicit human-readable "number of stripes / stripe spacing" axis to anchor the morphospace.
   (TDA H0 persistence is the robust version; bare component count often suffices.)
4. **Manifold-harmonics (Laplace–Beltrami) band-energy spectrum** of the per-face color signal
   (Vallet & Lévy 2008; ShapeDNA, Reuter 2006). Eigenbasis once on the shared mesh; band-summed
   energy = compact position-invariant vector. The elegant **UV-distortion-proof** upgrade if UV
   seams/distortion are bad. (Implemented in `fishpipe/spectral.py`.)
5. **DINOv2 / ResNet frozen embeddings** (CPU forward pass on standardized UV renders, minutes
   for 250 specimens). Best raw clustering, worst interpretability → benchmark/fallback only.

## Biology ecosystem to match (interpretability/publishability)
- `patternize` (Van Belleghem 2018, MEE) — per-pixel PCA on aligned color rasters (homology via
  registration; our fixed UV = built-in homology). Note: pixel-PCA conflates *where* color is with
  mean color → still need a structure descriptor on top.
- `recolorize` (Weller 2024, Ecol. Lett.) — guided color segmentation (≈ our color quantization).
- Endler adjacency / QCPA — transect-based transition matrices + boundary/patch geometry. Gold standard.
- `pavo`, `colordistance` — color-space metrics (our already-solved problem).

## Counting periodic structure
- Fourier dominant-frequency (peak ∝ stripe count); autocorrelation period; H0 persistent-homology
  component count (robust to irregular spacing). Component-counting on the existing patch map is ~free.

## Key citations
- Vallet & Lévy 2008, *CGF* — Manifold Harmonics.
- Reuter, Wolter & Peinecke 2006 — ShapeDNA (Laplace–Beltrami spectra).
- Sun 2009 (HKS); Aubry 2011 (WKS) — per-point spectral signatures (geometry; borrow band-pass idea only).
- Haralick 1973 (GLCM); Ojala 2002 (LBP); Gabor banks; 2D wavelet/Fourier texture energy.
- Endler 2012, *BJLS* — adjacency analysis. van den Berg 2020, *MEE* — QCPA.
- Van Belleghem 2018, *MEE* — patternize. Weller 2024, *Ecol. Lett.* — recolorize.
- Perea 2018, *Proc. R. Soc. A* — TDA for periodic signals. Oquab 2023 — DINOv2.
