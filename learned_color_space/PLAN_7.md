# PLAN 7: Handcrafted Spectral and Morphometric Features

## Purpose

Build a deterministic, explainable baseline using handcrafted color, texture, and frequency features.

This approach is less flexible than foundation embeddings, but it is transparent and can be surprisingly effective for museum specimen color and texture analysis.

## Core Idea

```text
UV Texture Map
      |
      v
Feature Extraction
color histograms / texture statistics / Gabor / wavelets
      |
      v
Feature Vector
      |
      v
PCA / UMAP / clustering
      |
      v
Explainable Appearance Space
```

## Feature Families

### Color Features

- mean RGB or Lab color
- color histogram
- hue histogram
- saturation and brightness statistics
- dominant colors
- quantiles of lightness and chroma

### Texture Features

- local contrast
- entropy
- edge density
- gray-level co-occurrence matrix features
- local binary patterns

### Frequency Features

- Gabor filter responses
- wavelet features
- stripe orientation estimates
- stripe frequency estimates

### Optional Shape Features

If geometry should be included, add simple morphometric descriptors such as:

- length
- width
- height
- aspect ratio
- curvature summaries
- surface area
- volume

## Inputs

Required:

- UV texture maps
- specimen IDs

Recommended:

- valid texture masks
- metadata for plotting and evaluation
- baseline embeddings from PLAN_1

Optional:

- mesh-derived morphometric measurements

## Outputs

This plan should produce:

- feature table with named columns
- PCA and UMAP coordinates
- feature importance summaries
- clustering results
- nearest-neighbor examples

Suggested output structure:

```text
outputs/
  features/
    handcrafted_texture_features.csv
    handcrafted_texture_features.npy
  tables/
    handcrafted_pca_coordinates.csv
    handcrafted_umap_coordinates.csv
  figures/
    handcrafted_pca.png
    handcrafted_umap.png
    feature_correlations.png
    handcrafted_neighbors/
```

## Implementation Steps

1. Build a texture manifest.

   Store specimen ID, texture path, and optional mask path.

2. Convert textures into perceptual color spaces.

   Use Lab, HSV, or both in addition to RGB.

3. Extract color statistics.

   Compute means, standard deviations, quantiles, and histograms over valid texture pixels.

4. Extract texture statistics.

   Compute contrast, entropy, local binary patterns, and co-occurrence statistics.

5. Extract frequency features.

   Apply Gabor or wavelet filters to estimate stripe, band, and speckle structure.

6. Normalize features.

   Standardize numeric features before PCA, UMAP, clustering, or distance calculations.

7. Visualize and compare.

   Compare handcrafted features against PLAN_1 embeddings and inspect nearest neighbors.

## Evaluation Questions

- Which handcrafted features align with human appearance judgments?
- Do color histograms separate light, dark, brown, gray, and iridescent specimens?
- Do Gabor or wavelet features capture stripe frequency and orientation?
- Are the PCA axes directly interpretable?
- How much worse or better is this than DINOv2 for nearest-neighbor retrieval?

## Success Criteria

Minimum useful result:

- features are deterministic and interpretable
- broad color groups appear in PCA or UMAP
- nearest neighbors are plausible for simple color similarity

Strong result:

- stripe and texture features improve retrieval beyond color histograms alone
- PCA axes can be described in plain appearance terms
- handcrafted features provide useful validation for learned embeddings

## Risks

- Handcrafted features may miss subtle or high-level patterns.
- UV seams and empty texture regions can distort statistics.
- Feature engineering may become time-consuming.
- Distances may be sensitive to scaling choices.

## Mitigations

- Use masks for valid texture pixels.
- Standardize all features before distance calculations.
- Keep the feature set simple at first.
- Compare against PLAN_1 rather than treating this as the only method.
- Use feature names to explain why specimens are near each other.

## Decision Point

Use this plan as an explainable baseline and sanity check. Even if learned embeddings win, handcrafted features can help interpret axes, validate clusters, and identify whether color or texture is driving the learned space.
