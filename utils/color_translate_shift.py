#!/usr/bin/env python3
"""
Global translation+scaling color normalization in CIE Lab.

- Reads all PNGs from INPUT_DIR (recursively).
- Computes pooled mean/std in Lab over NON-BLACK pixels only (black = RGB==(0,0,0)).
- For each image, computes its own mean/std over non-black pixels.
- Applies per-channel affine: (x - mu_img) * (std_pool/std_img) + mu_pool
  to all non-black pixels, leaves black pixels as pure black.
- Preserves alpha if present.
- Writes PNGs with the same relative path into OUTPUT_DIR.

Edit INPUT_DIR and OUTPUT_DIR constants below to point to your folders.
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image
from skimage.color import rgb2lab, lab2rgb
from tqdm import tqdm

# ====== CONFIG: set your folders here ======
INPUT_DIR  = Path("/media/alek/e6852e67-f061-4723-a0d3-c6271961077a/ml_data/color-modeling-pub/3D_Fish/3D_Fish/deca_out/2025_09-28_20_39_38/colorAnalysis/atlasTextures")   # <-- change me
OUTPUT_DIR = Path("/media/alek/e6852e67-f061-4723-a0d3-c6271961077a/ml_data/color-modeling-pub/3D_Fish/3D_Fish/deca_out/2025_09-28_20_39_38/colorAnalysis/atlasTextures_shifted")  # <-- change me
# Optional: processing
ALPHA      = 1.0   # strength of transform (0..1). 1.0 = full, <1 = gentler
EPS        = 1e-8  # numerical stability
# ===========================================

def _is_png(p: Path) -> bool:
    return p.suffix.lower() == ".png"

def _load_image_rgb_and_alpha(path: Path) -> Tuple[np.ndarray, np.ndarray | None]:
    """Return (rgb_float_[H,W,3], alpha_uint8_or_None). RGB in [0,1] sRGB."""
    img = Image.open(path).convert("RGBA")  # handle alpha uniformly
    rgba = np.array(img, dtype=np.uint8)
    rgb = rgba[..., :3].astype(np.float32) / 255.0
    alpha = rgba[..., 3]  # uint8
    # If the file was RGB (no alpha), Pillow made alpha=255; we still preserve it.
    return rgb, alpha

def _save_image_rgb_alpha(path: Path, rgb: np.ndarray, alpha: np.ndarray | None):
    """rgb float [0,1], optional alpha uint8; saves PNG."""
    rgb8 = np.clip(np.round(rgb * 255.0), 0, 255).astype(np.uint8)
    if alpha is not None:
        rgba = np.concatenate([rgb8, alpha[..., None]], axis=-1)
        Image.fromarray(rgba, mode="RGBA").save(path)
    else:
        Image.fromarray(rgb8, mode="RGB").save(path)

def _black_mask(rgb: np.ndarray) -> np.ndarray:
    """True where pixel is strictly black in 8-bit space."""
    # Compare against exact zeros in 8-bit space to be strict.
    return (np.round(rgb * 255.0).astype(np.uint8).sum(axis=-1) == 0)

def _lab_clip(lab: np.ndarray) -> np.ndarray:
    """Clip Lab to safe ranges."""
    out = lab.copy()
    out[..., 0] = np.clip(out[..., 0], 0.0, 100.0)   # L*
    out[..., 1] = np.clip(out[..., 1], -128.0, 127.0)  # a*
    out[..., 2] = np.clip(out[..., 2], -128.0, 127.0)  # b*
    return out

def _accumulate_mean_std_online(sum_vec, sumsq_vec, count, samples_2d):
    """Update running sums given (N,3) samples."""
    if samples_2d.size == 0:
        return sum_vec, sumsq_vec, count
    sum_vec   += samples_2d.sum(axis=0)
    sumsq_vec += (samples_2d ** 2).sum(axis=0)
    count     += samples_2d.shape[0]
    return sum_vec, sumsq_vec, count

def _finalize_mean_std(sum_vec, sumsq_vec, count):
    if count == 0:
        # Degenerate: fallback to neutral stats
        mu  = np.array([50.0, 0.0, 0.0], dtype=np.float64)
        std = np.array([1.0, 1.0, 1.0], dtype=np.float64)
        return mu, std
    mu = sum_vec / count
    var = np.maximum(sumsq_vec / count - mu**2, 0.0)
    std = np.sqrt(var) + EPS
    return mu, std

def discover_pngs(root: Path):
    return [p for p in root.rglob("*.png") if p.name != "average_texture.png"]

def compute_pooled_stats(paths):
    sum_vec   = np.zeros(3, dtype=np.float64)
    sumsq_vec = np.zeros(3, dtype=np.float64)
    count     = 0

    for p in tqdm(paths, desc="Pass 1: pooling stats"):
        rgb, _ = _load_image_rgb_and_alpha(p)
        mask_black = _black_mask(rgb)
        if mask_black.all():
            continue
        lab = rgb2lab(rgb)  # float Lab
        samples = lab[~mask_black].reshape(-1, 3).astype(np.float64)
        sum_vec, sumsq_vec, count = _accumulate_mean_std_online(sum_vec, sumsq_vec, count, samples)

    mu_pool, std_pool = _finalize_mean_std(sum_vec, sumsq_vec, count)
    return mu_pool, std_pool

def transform_image(path: Path, mu_pool, std_pool, alpha: float):
    rgb, alpha_ch = _load_image_rgb_and_alpha(path)
    H, W, _ = rgb.shape
    mask_black = _black_mask(rgb)

    lab = rgb2lab(rgb).astype(np.float64)
    if (~mask_black).any():
        region = lab[~mask_black].reshape(-1, 3)

        # per-image stats on non-black pixels
        mu_img = region.mean(axis=0)
        std_img = region.std(axis=0) + EPS

        # affine per channel
        region_t = (region - mu_img) * (std_pool / std_img) + mu_pool
        # strength blending
        region_out = region + alpha * (region_t - region)

        lab_out = lab.copy()
        lab_out[~mask_black] = region_out.reshape((-1, 3))
    else:
        lab_out = lab  # nothing to do

    lab_out = _lab_clip(lab_out)
    rgb_out = lab2rgb(lab_out)  # float [0,1]

    # keep black pixels strictly black in output
    rgb_out[mask_black] = 0.0

    return rgb_out, alpha_ch

def main():
    assert INPUT_DIR.exists(), f"INPUT_DIR does not exist: {INPUT_DIR}"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    in_paths = discover_pngs(INPUT_DIR)
    if not in_paths:
        print(f"No PNG files found under {INPUT_DIR}")
        return

    # 1) pooled stats over non-black pixels (Lab)
    mu_pool, std_pool = compute_pooled_stats(in_paths)
    print(f"Pooled Lab mean: {mu_pool.round(4)}  std: {std_pool.round(4)}")

    # 2) transform each image
    for p in tqdm(in_paths, desc="Pass 2: transforming"):
        rgb_out, alpha_ch = transform_image(p, mu_pool, std_pool, alpha=ALPHA)

        # write preserving relative structure
        rel = p.relative_to(INPUT_DIR)
        out_path = (OUTPUT_DIR / rel).with_suffix(".png")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        _save_image_rgb_alpha(out_path, rgb_out, alpha_ch)

    print(f"Done. Wrote outputs to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
