#!/usr/bin/env python3
"""
Global translation+scaling on non-hue dimensions (L* and C*) in CIE L*a*b* space.

- Reads PNGs from INPUT_DIR (recursively).
- Ignores black pixels (RGB==0,0,0) for stats and leaves them black in outputs.
- Computes pooled mean/std of L* and C* across all images (non-black pixels only).
- For each image, computes its own L*, C* mean/std (non-black).
- Transforms only L* and C*: (x - mu_img) * (std_pool/std_img) + mu_pool.
- Keeps hue (h = atan2(b,a)) unchanged.
- Preserves alpha channel if present.
- Writes to OUTPUT_DIR preserving relative paths.
"""

from __future__ import annotations
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image
from skimage.color import rgb2lab, lab2rgb
from tqdm import tqdm

# ====== CONFIG ======
INPUT_DIR  = Path("/media/alek/e6852e67-f061-4723-a0d3-c6271961077a/ml_data/color-modeling-pub/3D_Fish/3D_Fish/deca_out/2025_09-28_20_39_38/colorAnalysis/atlasTextures")   # <-- change me
OUTPUT_DIR = Path("/media/alek/e6852e67-f061-4723-a0d3-c6271961077a/ml_data/color-modeling-pub/3D_Fish/3D_Fish/deca_out/2025_09-28_20_39_38/colorAnalysis/atlasTextures_shifted")  # <-- change me
ALPHA      = 1.0       # 0..1 strength toward pooled stats
EPS        = 1e-8       # numerical stability
# ====================

def _is_png(p: Path) -> bool:
    return p.suffix.lower() == ".png"

def _load_image_rgb_and_alpha(path: Path) -> Tuple[np.ndarray, np.ndarray | None]:
    """Return (rgb_float_[H,W,3], alpha_uint8). RGB in [0,1] sRGB."""
    img = Image.open(path).convert("RGBA")
    rgba = np.array(img, dtype=np.uint8)
    rgb = rgba[..., :3].astype(np.float32) / 255.0
    alpha = rgba[..., 3]  # uint8
    return rgb, alpha

def _save_image_rgb_alpha(path: Path, rgb: np.ndarray, alpha: np.ndarray | None):
    rgb8 = np.clip(np.round(rgb * 255.0), 0, 255).astype(np.uint8)
    if alpha is not None:
        rgba = np.concatenate([rgb8, alpha[..., None]], axis=-1)
        Image.fromarray(rgba, mode="RGBA").save(path)
    else:
        Image.fromarray(rgb8, mode="RGB").save(path)

def _black_mask(rgb: np.ndarray) -> np.ndarray:
    """True where pixel is strictly black in 8-bit."""
    return (np.round(rgb * 255.0).astype(np.uint8).sum(axis=-1) == 0)

def _lab_clip(lab: np.ndarray) -> np.ndarray:
    out = lab.copy()
    out[..., 0] = np.clip(out[..., 0], 0.0, 100.0)   # L*
    out[..., 1] = np.clip(out[..., 1], -128.0, 127.0)  # a*
    out[..., 2] = np.clip(out[..., 2], -128.0, 127.0)  # b*
    return out

def _accumulate(sum_vec, sumsq_vec, count, samples_2d):
    # samples_2d shape: (N,2) for [L, C]
    if samples_2d.size == 0:
        return sum_vec, sumsq_vec, count
    sum_vec   += samples_2d.sum(axis=0)
    sumsq_vec += (samples_2d ** 2).sum(axis=0)
    count     += samples_2d.shape[0]
    return sum_vec, sumsq_vec, count

def _finalize(sum_vec, sumsq_vec, count):
    if count == 0:
        mu  = np.array([50.0, 20.0], dtype=np.float64)   # neutral-ish fallback
        sd  = np.array([1.0,  1.0 ], dtype=np.float64)
        return mu, sd
    mu = sum_vec / count
    var = np.maximum(sumsq_vec / count - mu**2, 0.0)
    sd = np.sqrt(var) + EPS
    return mu, sd

def discover_pngs(root: Path):
    return [p for p in root.rglob("*.png")]

def compute_pooled_LC(paths):
    sum_vec   = np.zeros(2, dtype=np.float64)  # [L, C]
    sumsq_vec = np.zeros(2, dtype=np.float64)
    count     = 0

    for p in tqdm(paths, desc="Pass 1: pooling L*,C* stats"):
        rgb, _ = _load_image_rgb_and_alpha(p)
        mask_black = _black_mask(rgb)
        if mask_black.all():
            continue
        lab = rgb2lab(rgb).astype(np.float64)
        nb = ~mask_black
        if not nb.any():
            continue
        L = lab[..., 0][nb]
        a = lab[..., 1][nb]
        b = lab[..., 2][nb]
        C = np.sqrt(a*a + b*b)
        LC = np.stack([L, C], axis=1)
        sum_vec, sumsq_vec, count = _accumulate(sum_vec, sumsq_vec, count, LC)

    mu_pool, sd_pool = _finalize(sum_vec, sumsq_vec, count)  # [mu_L, mu_C], [sd_L, sd_C]
    return mu_pool, sd_pool

def transform_image_LC_only(path: Path, mu_pool, sd_pool, alpha: float):
    rgb, alpha_ch = _load_image_rgb_and_alpha(path)
    mask_black = _black_mask(rgb)

    lab = rgb2lab(rgb).astype(np.float64)
    nb = ~mask_black
    if nb.any():
        L = lab[..., 0][nb]
        a = lab[..., 1][nb]
        b = lab[..., 2][nb]
        C = np.sqrt(a*a + b*b)            # chroma
        h = np.arctan2(b, a)              # hue (kept constant)

        # per-image stats on L and C for non-black pixels
        mu_img = np.array([L.mean(), C.mean()], dtype=np.float64)
        sd_img = np.array([L.std(),  C.std()],  dtype=np.float64) + EPS

        # target LC via per-channel affine toward pooled stats
        L_t = (L - mu_img[0]) * (sd_pool[0] / sd_img[0]) + mu_pool[0]
        C_t = (C - mu_img[1]) * (sd_pool[1] / sd_img[1]) + mu_pool[1]

        # strength blend
        L_out = L + alpha * (L_t - L)
        C_out = C + alpha * (C_t - C)

        # enforce valid ranges
        L_out = np.clip(L_out, 0.0, 100.0)
        C_out = np.maximum(C_out, 0.0)  # chroma can't be negative

        # recompose with ORIGINAL hue
        a_out = C_out * np.cos(h)
        b_out = C_out * np.sin(h)

        lab_out = lab.copy()
        lab_out[..., 0][nb] = L_out
        lab_out[..., 1][nb] = a_out
        lab_out[..., 2][nb] = b_out
    else:
        lab_out = lab

    lab_out = _lab_clip(lab_out)
    rgb_out = lab2rgb(lab_out)
    rgb_out[mask_black] = 0.0  # keep pure black untouched

    return rgb_out, alpha_ch

def main():
    assert INPUT_DIR.exists(), f"INPUT_DIR does not exist: {INPUT_DIR}"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    paths = discover_pngs(INPUT_DIR)
    if not paths:
        print(f"No PNG files found under {INPUT_DIR}")
        return

    mu_pool, sd_pool = compute_pooled_LC(paths)
    print(f"Pooled L*,C* mean: {mu_pool.round(4)}  std: {sd_pool.round(4)}")

    for p in tqdm(paths, desc="Pass 2: transforming"):
        rgb_out, alpha_ch = transform_image_LC_only(p, mu_pool, sd_pool, alpha=ALPHA)
        rel = p.relative_to(INPUT_DIR)
        out_path = (OUTPUT_DIR / rel).with_suffix(".png")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        _save_image_rgb_alpha(out_path, rgb_out, alpha_ch)

    print(f"Done. Wrote outputs to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
