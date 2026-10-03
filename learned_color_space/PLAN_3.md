# PLAN 3: Self-Supervised Appearance Learning

## Purpose

Train a self-supervised model directly on the specimen texture collection when pretrained embeddings are not sensitive enough to the visual differences that matter.

This plan is for learning collection-specific texture representations without requiring dense human labels.

## Core Idea

```text
UV Texture Images
        |
        v
Self-Supervised Training
BYOL / VICReg / DINO / I-JEPA-style objective
        |
        v
Texture Encoder
        |
        v
Appearance Embedding
        |
        v
PCA / UMAP / retrieval
```

Instead of predicting labels, the model learns from relationships between augmented views, masked regions, or neighboring representations.

## Candidate Methods

### DINO-style Training

Use student-teacher self-distillation on augmented texture crops.

Good when:

- local and global pattern consistency both matter
- you want strong image retrieval behavior
- the dataset is large enough for collection-specific fine-tuning

### BYOL

Train two augmented views of the same texture to produce similar embeddings.

Good when:

- you want a stable, well-known self-supervised baseline
- labels are unavailable
- implementation simplicity matters

### VICReg

Train embeddings with invariance, variance, and covariance regularization.

Good when:

- you want smooth latent spaces
- collapse prevention and clustering behavior are important
- you want a representation that may work well for PCA or UMAP

### I-JEPA-Style Training

Predict the embedding of missing image regions rather than reconstructing pixels.

Good when:

- global structure matters
- you want a more semantic representation than pixel reconstruction
- you can tolerate a more experimental implementation

## Inputs

Required:

- UV texture maps
- specimen IDs

Recommended:

- texture masks for valid UV regions
- metadata for later evaluation
- outputs from PLAN_1 for baseline comparison

## Outputs

This plan should produce:

- a trained self-supervised texture encoder
- one embedding vector per specimen
- PCA and UMAP coordinates
- nearest-neighbor retrieval examples
- comparison against DINOv2, CLIP, or SigLIP from PLAN_1

Suggested output structure:

```text
outputs/
  models/
    self_supervised_texture_encoder.pt
  embeddings/
    self_supervised_embeddings.npy
  tables/
    self_supervised_coordinates.csv
  figures/
    self_supervised_umap.png
    self_supervised_pca.png
    self_supervised_neighbors/
```

## Implementation Steps

1. Build the dataset loader.

   Load texture maps, apply masks if available, and return augmented image views.

2. Define augmentations carefully.

   Use color jitter, crop, resize, blur, and mild geometric transforms. Avoid augmentations that erase biologically meaningful color or stripe information.

3. Start from an existing implementation.

   Prefer a known BYOL, VICReg, or DINO implementation rather than writing the training method from scratch.

4. Train a small or medium encoder.

   Use a ResNet or ViT backbone depending on data size and compute. Start modestly before scaling up.

5. Export embeddings.

   Freeze the encoder and embed every texture map.

6. Visualize and compare.

   Run PCA and UMAP. Compare neighborhoods against PLAN_1 outputs.

7. Inspect failure cases.

   Use nearest-neighbor examples to see whether the model learned shell appearance or only UV layout artifacts.

## Evaluation Questions

- Are nearest neighbors better than pretrained DINOv2?
- Does the model preserve subtle stripe, band, mottling, and speckle patterns?
- Are color gradients smooth?
- Does the embedding avoid collapsing all similar UV layouts together?
- Does self-supervised training improve collection-specific distinctions?

## Success Criteria

Minimum useful result:

- self-supervised neighbors are competitive with PLAN_1
- learned clusters are visually coherent
- the model can embed new textures without retraining

Strong result:

- self-supervised embeddings outperform PLAN_1 on expert review
- the map captures specimen-specific texture details missed by foundation models
- PCA or UMAP reveals smooth, interpretable appearance gradients

## Risks

- The dataset may be too small for training from scratch.
- Augmentations may remove the exact color or texture cues that matter.
- The model may learn UV layout artifacts instead of biological appearance.
- I-JEPA-style training may take longer to implement and tune.

## Mitigations

- Start from PLAN_1 and use it as the baseline.
- Fine-tune a pretrained backbone instead of training from random initialization.
- Keep augmentations conservative.
- Use texture masks where possible.
- Evaluate with nearest-neighbor review throughout training.

## Decision Point

Use this plan if pretrained embeddings are close but not sensitive enough. If this still fails, move toward geometry-aware learning, patch-based texture learning, or explicit reconstruction models.
