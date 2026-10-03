# PLAN 2: Human-Oriented Appearance Space on Top of Foundation Embeddings

## Purpose

Learn a smaller, smoother, more human-oriented appearance space using pretrained embeddings as the input representation.

This plan builds on PLAN_1. Instead of training directly from pixels, it uses DINOv2, CLIP, or SigLIP embeddings as stable visual features, then learns a compact mapping that reflects the way humans describe specimen appearance.

## Core Idea

```text
UV Texture Image
        |
        v
Pretrained Vision Encoder
DINOv2 / CLIP / SigLIP
        |
        v
Foundation Embedding
        |
        v
Small Trainable Model
MLP / metric head / projection head
        |
        v
64-D Human-Oriented Appearance Space
        |
        v
PCA / UMAP / retrieval / browsing
```

## Why This Plan

Training directly from images can be expensive and data-hungry. A small model on top of foundation embeddings is cheaper, easier to retrain, and can incorporate expert feedback.

This is the right next step if PLAN_1 gives reasonable neighborhoods but misses important human judgments, such as:

- two shell textures look similar to experts but are far apart in DINOv2 space
- striped specimens are inconsistently grouped
- color dominates when pattern should matter more
- species or shape cues dominate when texture should matter more
- the map needs interpretable axes such as brightness, brownness, stripe strength, or speckling

## Inputs

Required:

- saved embeddings from PLAN_1
- specimen IDs
- texture paths

Recommended labels:

- base color: brown, tan, gray, black, white, reddish, iridescent
- pattern: striped, spotted, banded, speckled, mottled, plain
- brightness: light, medium, dark
- texture strength: smooth, subtle, high contrast

Optional supervision:

- pairwise similarity judgments
- triplets such as "A is more similar to B than to C"
- curator-defined groups
- known biological or collection metadata

## Outputs

The plan should produce:

- a trained projection model
- compact appearance vectors, such as 32-D or 64-D
- 2D and 3D visualizations of the learned space
- nearest-neighbor retrieval examples
- label prediction metrics where labels exist
- before/after comparison against raw DINOv2 or CLIP embeddings

Suggested output structure:

```text
outputs/
  models/
    appearance_projection.pt
  embeddings/
    learned_appearance_vectors.npy
  tables/
    learned_appearance_coordinates.csv
    label_predictions.csv
  figures/
    learned_umap.png
    learned_pca.png
    before_after_neighbors/
```

## Model Options

### Option A: Multi-Label Classifier Plus Embedding

Train a small MLP to predict human appearance tags from foundation embeddings.

The hidden layer or penultimate layer becomes the learned appearance space.

Good when:

- labels are available
- categories like striped, spotted, brown, or dark gray are important
- interpretability matters

### Option B: Metric Learning Projection

Train a projection head using contrastive, triplet, or supervised contrastive loss.

Good when:

- humans can provide similar/different judgments
- relative similarity matters more than fixed labels
- the goal is high-quality nearest-neighbor browsing

### Option C: Hybrid Model

Use both label prediction loss and metric learning loss.

Good when:

- some labels exist
- pairwise or triplet judgments can also be collected
- the space should support both interpretable axes and perceptual retrieval

## Implementation Steps

1. Reuse PLAN_1 embeddings.

   Load saved DINOv2 embeddings first. Keep the encoder frozen so this phase stays cheap and repeatable.

2. Create an annotation table.

   Store one row per specimen with appearance tags and optional pairwise or triplet judgments.

3. Split the data.

   Use train, validation, and test splits. Avoid leaking near-duplicate specimens across splits if duplicates exist.

4. Train a small MLP.

   Start with a simple projection from the foundation embedding to a 64-D appearance vector. Add label heads or metric-learning losses depending on available supervision.

5. Save learned vectors.

   Export one compact vector per specimen. These vectors become the main learned color and texture space.

6. Visualize.

   Run PCA and UMAP on the learned vectors and compare against the raw foundation embeddings from PLAN_1.

7. Evaluate retrieval.

   Compare nearest neighbors before and after training. Check whether the learned space improves expert-perceived similarity.

8. Iterate on labels.

   Use failure cases to decide what labels or pairwise judgments are missing.

## Evaluation Questions

- Does the learned space improve nearest-neighbor quality over PLAN_1?
- Are human appearance tags predictable from the learned vectors?
- Are visually intermediate specimens placed between stronger examples?
- Does the model avoid overfitting to species, specimen size, or UV layout artifacts?
- Do PCA axes become more interpretable than the raw foundation embedding axes?

## Success Criteria

Minimum useful result:

- learned nearest neighbors are better than raw DINOv2 neighbors
- appearance tags are predictable above a simple baseline
- the learned space remains smooth enough for browsing and interpolation

Strong result:

- experts prefer learned-space retrieval over raw foundation embeddings
- the map exposes recognizable axes like brightness, stripe strength, and base color
- newly added specimens land in sensible positions without retraining from scratch

## Risks

- Labels may be too sparse or inconsistent.
- A small model can overfit if the dataset is small.
- Human labels may collapse continuous variation into overly rigid categories.
- Metric learning can be sensitive to poor triplets or unbalanced examples.

## Mitigations

- Start with frozen embeddings and a small projection model.
- Prefer multi-label annotations over single exclusive classes.
- Keep raw PLAN_1 embeddings as a baseline for every comparison.
- Use nearest-neighbor review as the main qualitative check.
- Add regularization and early stopping.
- Collect more annotations around failure cases rather than labeling everything upfront.

## Decision Point

After this plan, decide whether the learned space is good enough for collection browsing.

If it is not good enough, the next likely directions are:

- train a self-supervised model such as BYOL, VICReg, or DINO on the collection
- add rendered mesh views or mesh embeddings for geometry-aware appearance
- train a VAE or beta-VAE if explicit continuous interpolation is more important than retrieval
