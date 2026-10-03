# PLAN 4: Geometry-Aware Appearance Learning

## Purpose

Learn an appearance space that uses both mesh geometry and texture. This is useful when color pattern alone is not enough, or when specimen shape carries biological meaning that should affect similarity.

## Core Idea

```text
Mesh Geometry             UV Texture
     |                        |
     v                        v
Shape Encoder           Texture Encoder
     |                        |
     v                        v
Shape Embedding         Texture Embedding
     \                        /
      \                      /
       v                    v
       Joint Appearance Embedding
              |
              v
       PCA / UMAP / retrieval
```

The key design choice is whether the final space should represent pure appearance, pure morphology, or a controlled mixture of both.

## Candidate Geometry Encoders

- Point-MAE: masked autoencoder for point clouds
- Point-BERT: BERT-style point cloud representation learning
- PointNeXt: strong point cloud encoder
- MeshMAE: masked autoencoder directly on mesh structure
- MeshCNN: mesh-edge-based learning

## Inputs

Required:

- high-resolution meshes
- UV texture maps
- specimen IDs

Recommended:

- normalized mesh orientation and scale
- sampled point clouds from meshes
- rendered views of textured meshes
- metadata for biological or taxonomic evaluation

## Outputs

This plan should produce:

- shape embeddings
- texture embeddings
- combined shape-plus-texture embeddings
- visualizations of the joint space
- nearest-neighbor retrieval examples
- ablation plots comparing texture-only, shape-only, and combined similarity

Suggested output structure:

```text
outputs/
  embeddings/
    shape_embeddings.npy
    texture_embeddings.npy
    joint_geometry_texture_embeddings.npy
  tables/
    geometry_aware_coordinates.csv
  figures/
    geometry_texture_umap.png
    texture_only_vs_joint_neighbors/
```

## Implementation Options

### Option A: Late Fusion

Compute texture and shape embeddings separately, concatenate them, and run PCA or UMAP.

This is the easiest starting point.

```text
joint = concat(texture_embedding, shape_embedding)
```

### Option B: Weighted Fusion

Apply weights before concatenation so texture or geometry can dominate depending on the task.

```text
joint = concat(texture_weight * texture_embedding,
               shape_weight * shape_embedding)
```

This is useful for interactively testing how much shape should matter.

### Option C: Learned Fusion

Train a small MLP that maps texture and shape embeddings into a shared appearance vector.

This is useful when labels, pairwise judgments, or known biological groups are available.

## Implementation Steps

1. Normalize meshes.

   Ensure meshes share consistent scale, orientation, and coordinate conventions.

2. Create geometry representations.

   Sample point clouds or convert meshes into the format required by the chosen geometry encoder.

3. Extract texture embeddings.

   Reuse PLAN_1 DINOv2, CLIP, or SigLIP embeddings.

4. Extract shape embeddings.

   Use a pretrained point-cloud or mesh model if available. If not, start with deterministic morphometric descriptors.

5. Combine embeddings.

   Begin with late fusion and test multiple texture-versus-shape weights.

6. Visualize and evaluate.

   Run PCA and UMAP on texture-only, shape-only, and combined spaces.

7. Review nearest neighbors.

   Compare whether combined neighbors are more biologically meaningful than texture-only neighbors.

## Evaluation Questions

- Does shape improve retrieval quality?
- Does geometry overpower the color and texture signal?
- Are shape-only clusters biologically meaningful?
- Can texture and shape weights be tuned for different research questions?
- Do combined embeddings place intermediate specimens between groups?

## Success Criteria

Minimum useful result:

- shape-only and texture-only spaces can be compared
- joint embeddings produce plausible nearest neighbors
- the effect of geometry is visible and controllable

Strong result:

- combined embeddings outperform texture-only embeddings in expert review
- biologically meaningful morphology and appearance patterns emerge together
- users can adjust shape-versus-texture influence depending on the analysis

## Risks

- Geometry preprocessing may dominate project complexity.
- Mesh resolution, holes, scans, or alignment differences may affect embeddings.
- Shape may swamp subtle color variation.
- Pretrained geometry models may not transfer well to shells or museum specimens.

## Mitigations

- Start with late fusion, not an end-to-end multimodal model.
- Keep texture-only PLAN_1 embeddings as a baseline.
- Normalize shape embeddings and texture embeddings before combining them.
- Evaluate multiple fusion weights.
- Use simple morphometric descriptors if learned geometry models are too heavy.

## Decision Point

Use this plan if shape is part of the scientific question. If the goal is strictly color and surface texture, keep geometry separate as metadata or an optional retrieval mode.
