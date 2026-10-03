# PLAN 5: UV Patch Learning

## Purpose

Learn texture representations from local UV patches instead of treating each full texture map as one image.

This is useful when local structures such as stripes, bands, growth lines, speckles, or mottling are more important than the global layout of the UV map.

## Core Idea

```text
UV Texture Map
      |
      v
Patch Extraction
      |
      v
Patch Encoder
Vision Transformer / CNN / DINOv2
      |
      v
Patch Embeddings
      |
      v
Pooling / Attention
      |
      v
Specimen Texture Embedding
```

The model learns from many small regions of each texture, then aggregates them into one representation per specimen.

## Why This Plan

Full UV maps may contain large empty regions, seams, distortions, or layout artifacts. Patch learning focuses the model on visible local texture.

This approach can be especially valuable for:

- shell growth bands
- fine stripe patterns
- repeated speckles
- local iridescence
- high-frequency color variation

## Inputs

Required:

- UV texture maps
- specimen IDs

Recommended:

- valid texture masks
- patch coordinates
- metadata for later interpretation
- baseline embeddings from PLAN_1

## Outputs

This plan should produce:

- patch-level embeddings
- pooled specimen-level embeddings
- visualizations of specimen-level appearance space
- optional patch maps showing which regions drive similarity
- nearest-neighbor retrieval examples

Suggested output structure:

```text
outputs/
  patches/
    patch_manifest.csv
  embeddings/
    patch_embeddings.npy
    pooled_patch_embeddings.npy
  tables/
    patch_based_coordinates.csv
  figures/
    patch_umap.png
    patch_attention_maps/
    patch_based_neighbors/
```

## Implementation Options

### Option A: Pretrained Patch Features

Extract patches and embed each patch with DINOv2. Pool patch embeddings into one vector per specimen.

This is the simplest version.

### Option B: Multiple Instance Learning

Treat each specimen as a bag of patches and learn attention weights over patches.

This is useful when only some regions contain important pattern information.

### Option C: Self-Supervised Patch Model

Train BYOL, VICReg, DINO, or masked prediction on extracted patches.

This is useful when the collection has many textures and pretrained features miss important local details.

## Implementation Steps

1. Create valid texture masks.

   Exclude empty UV regions and background pixels.

2. Extract patches.

   Sample fixed-size patches from valid regions. Store specimen ID, patch path, and patch coordinates.

3. Filter low-information patches.

   Remove mostly empty, nearly uniform, or invalid patches.

4. Embed patches.

   Start with DINOv2 patch embeddings.

5. Pool patch embeddings per specimen.

   Start with mean pooling and max pooling. Later test attention pooling.

6. Visualize specimen vectors.

   Run PCA and UMAP on pooled patch vectors.

7. Inspect patch influence.

   For a given specimen, identify patches most responsible for nearest-neighbor similarity.

## Evaluation Questions

- Does patch pooling improve sensitivity to stripes and local texture?
- Are nearest neighbors more visually convincing than full-texture embeddings?
- Are important local patterns preserved?
- Do patch embeddings avoid empty UV layout artifacts?
- Does pooling lose too much global color information?

## Success Criteria

Minimum useful result:

- patch-based embeddings are competitive with PLAN_1
- local pattern similarities are better represented
- nearest-neighbor examples reveal shared texture motifs

Strong result:

- patch-based retrieval finds specimens with similar stripe, speckle, or growth-line structure
- patch influence maps are interpretable
- patch embeddings can support both global browsing and local texture search

## Risks

- Patch extraction can create many files and large embedding tables.
- Mean pooling may dilute rare but important patterns.
- Patch location may matter, and naive pooling may discard it.
- UV distortion may make patches inconsistent across specimens.

## Mitigations

- Start with a limited patch count per specimen.
- Save a patch manifest for reproducibility.
- Compare mean, max, and attention pooling.
- Use masks to avoid empty UV regions.
- Keep full-texture PLAN_1 embeddings as a baseline.

## Decision Point

Use this plan if full-image embeddings miss local texture details. If local features help but global appearance is also important, combine patch-based vectors with PLAN_1 full-texture embeddings.
