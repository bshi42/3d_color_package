# PLAN 6: Neural Appearance PCA with Autoencoders

## Purpose

Train an autoencoder or variational autoencoder that learns a continuous latent space of specimen appearance.

This is the closest approach to a neural version of PCA: each texture is encoded into a compact vector, and nearby latent positions should decode to similar-looking textures.

## Core Idea

```text
UV Texture
    |
    v
Encoder
    |
    v
Latent Vector
    |
    v
Decoder
    |
    v
Reconstructed Texture
```

For a beta-VAE, the latent space is encouraged to separate factors such as:

- brightness
- brownness
- stripe strength
- speckling
- contrast
- mottling

## Candidate Models

### Autoencoder

Learns a compressed representation by reconstructing the input texture.

Good when:

- reconstruction quality matters
- you want a simple baseline
- interpretability is secondary

### Variational Autoencoder

Learns a probabilistic latent space that supports smooth sampling and interpolation.

Good when:

- continuity matters
- interpolation between specimens is important
- you want a generative appearance manifold

### Beta-VAE

Adds stronger regularization to encourage disentangled latent factors.

Good when:

- interpretable latent dimensions are desirable
- you want axes like stripe amount, brightness, or base color to separate

## Inputs

Required:

- UV texture maps
- specimen IDs

Recommended:

- texture masks
- standardized image sizes
- enough examples to train a generative model
- baseline embeddings from PLAN_1 for comparison

## Outputs

This plan should produce:

- trained encoder and decoder
- latent vector per specimen
- reconstruction examples
- interpolation examples
- PCA and UMAP of latent vectors
- optional generated textures sampled from the latent space

Suggested output structure:

```text
outputs/
  models/
    texture_autoencoder.pt
    texture_beta_vae.pt
  embeddings/
    vae_latents.npy
  figures/
    reconstructions/
    interpolations/
    vae_latent_umap.png
    latent_traversals/
```

## Implementation Steps

1. Prepare texture images.

   Resize textures to a consistent resolution. Apply masks if available.

2. Train a plain autoencoder first.

   Confirm that the model can reconstruct broad color and pattern features.

3. Train a VAE.

   Add a probabilistic latent space and inspect whether interpolation is smooth.

4. Train a beta-VAE variant.

   Increase the beta regularization gradually and watch the tradeoff between disentanglement and reconstruction quality.

5. Export latent vectors.

   Use encoder outputs as the neural appearance coordinates.

6. Visualize latent space.

   Run PCA and UMAP. Inspect whether latent dimensions or PCA directions correspond to human-recognizable factors.

7. Create interpolation and traversal figures.

   Interpolate between specimen pairs and vary single latent dimensions to understand what each direction controls.

## Evaluation Questions

- Are reconstructions faithful enough to preserve color and texture?
- Are interpolations smooth and visually plausible?
- Do latent dimensions correspond to interpretable appearance factors?
- Does the latent space produce meaningful nearest neighbors?
- Does reconstruction loss overemphasize pixel details at the expense of human similarity?

## Success Criteria

Minimum useful result:

- reconstructions preserve major color and pattern information
- latent nearest neighbors are plausible
- interpolation between similar specimens is smooth

Strong result:

- latent traversals reveal interpretable axes
- beta-VAE separates factors such as brightness, stripe amount, and speckling
- the latent space supports both browsing and generative exploration

## Risks

- Autoencoders may optimize pixel reconstruction rather than perceptual similarity.
- VAE outputs may be blurry.
- Beta regularization may remove details needed for shell texture.
- The dataset may be too small for a robust generative model.
- UV layout artifacts may be reconstructed and encoded.

## Mitigations

- Use perceptual losses or feature-space losses in addition to pixel loss.
- Start with a small latent dimension and compare several sizes.
- Keep masks to reduce background reconstruction.
- Compare nearest neighbors against PLAN_1.
- Treat generated textures as exploratory, not as scientific measurements.

## Decision Point

Use this plan if continuous interpolation and a generative appearance manifold are central goals. If retrieval and clustering are more important, PLAN_1, PLAN_2, or PLAN_5 may be more practical.
