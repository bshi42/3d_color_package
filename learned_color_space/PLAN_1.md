# PLAN 1: Pretrained Vision Embedding Baseline

## Purpose

Build the fastest useful baseline for a learned color and texture space using pretrained image foundation models on UV texture maps.

The goal is to determine whether pretrained visual embeddings already organize museum specimen textures in a way that feels human-meaningful:

- light brown specimens near other light brown specimens
- dark gray specimens near other dark gray specimens
- striped or banded specimens forming a coherent region
- intermediate specimens placed between visual groups

This plan should be tried first because it requires little or no model training and can produce a useful appearance map quickly.

## Core Idea

```text
UV Texture Image
        |
        v
Pretrained Vision Encoder
DINOv2 / CLIP / SigLIP
        |
        v
High-Dimensional Embedding
        |
        v
PCA / UMAP / t-SNE
        |
        v
2D or 3D Appearance Space
```

## Recommended Starting Model

Start with DINOv2.

DINOv2 is a strong first choice because it is self-supervised, tends to preserve texture and visual similarity well, and often performs better than CLIP for image-to-image retrieval or clustering tasks.

Secondary models to compare:

- CLIP: useful semantic baseline with a large ecosystem
- SigLIP: modern CLIP-like alternative with strong retrieval behavior
- EVA / EVA-CLIP: potentially high quality, but heavier

## Inputs

Expected input data:

- UV texture maps for each specimen
- specimen identifiers
- optional metadata such as species, collection ID, location, sex, age, or museum catalog fields

Optional later input:

- rendered views of the textured mesh from standard angles
- masks separating specimen pixels from texture-map background

## Outputs

The first milestone should produce:

- one embedding vector per texture map
- a table linking specimen ID to embedding path and metadata
- PCA coordinates
- UMAP coordinates
- plots for quick visual inspection
- nearest-neighbor retrieval examples for each specimen

Suggested output structure:

```text
data/
  textures/
  metadata.csv

outputs/
  embeddings/
    dinov2_embeddings.npy
    clip_embeddings.npy
  tables/
    specimen_embeddings.csv
    pca_coordinates.csv
    umap_coordinates.csv
  figures/
    dinov2_umap.png
    dinov2_pca.png
    nearest_neighbors/
```

## Implementation Steps

1. Inventory texture maps.

   Create a manifest containing specimen ID, texture file path, and any available metadata.

2. Preprocess texture images.

   Normalize image format and size for model input. Preserve enough resolution to retain stripes, bands, speckles, and subtle shell color variation.

3. Extract embeddings.

   Run each texture through DINOv2 first. Save embeddings to disk so dimensionality reduction and plotting can be repeated without rerunning inference.

4. Build dimensionality reductions.

   Run PCA for a linear baseline and UMAP for a nonlinear appearance map. Keep both because PCA is easier to interpret, while UMAP may better expose clusters.

5. Visualize.

   Plot specimens in 2D with thumbnails, metadata coloring, and labels where useful.

6. Evaluate nearest neighbors.

   For selected specimens, retrieve the nearest textures by embedding distance and visually inspect whether neighbors match human color and pattern similarity.

7. Compare encoders.

   Repeat the same process for CLIP and optionally SigLIP. Compare whether DINOv2, CLIP, or SigLIP gives the most human-meaningful neighborhoods.

## Evaluation Questions

- Do visually similar textures land near each other?
- Are color groups preserved?
- Are stripe, band, speckle, and mottling patterns reflected in local neighborhoods?
- Are nearest neighbors useful to a human curator or biologist?
- Does PCA already show interpretable axes such as brightness, brownness, or stripe density?
- Does UMAP create sensible local neighborhoods without over-separating continuous variation?

## Success Criteria

This plan is successful if DINOv2 or another pretrained encoder produces a map where humans recognize meaningful appearance neighborhoods without additional training.

Minimum useful result:

- nearest neighbors are usually plausible
- broad color clusters are visible
- striped or patterned specimens are not randomly mixed with plain specimens

Strong result:

- continuous gradients appear between visual groups
- PCA axes have intuitive appearance meanings
- UMAP neighborhoods support browsing the collection by appearance

## Risks

- CLIP may emphasize semantic categories instead of subtle specimen texture.
- UV texture layout may introduce artifacts that pretrained image models were not trained for.
- Background, seams, empty UV space, or scaling differences may dominate embeddings.
- UMAP can create visually compelling clusters even when global distances are not reliable.

## Mitigations

- Test DINOv2 before CLIP for texture sensitivity.
- Mask unused UV regions if they dominate the image.
- Try standardized rendered views if raw UV maps are visually unnatural.
- Use nearest-neighbor retrieval, not only 2D plots, to judge quality.
- Keep PCA coordinates as a sanity check for global structure.

## Decision Point

After this baseline, decide one of three paths:

- If the map is already useful, build browsing and annotation tools around it.
- If neighborhoods are close but not human-oriented enough, move to PLAN_2.
- If texture maps fail because geometry matters, add mesh-aware or rendered-view embeddings later.
