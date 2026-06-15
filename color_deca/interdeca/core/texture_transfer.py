"""Texture transfer file helpers."""

from pathlib import Path

import imageio.v2 as imageio
import numpy as np


def calculate_average_texture(texture_dir: Path) -> Path:
    out_textures_dir = Path(texture_dir)
    atlas_texture = out_textures_dir / "average_texture.png"
    if atlas_texture.exists():
        return atlas_texture

    pngs = sorted(path for path in out_textures_dir.glob("*.png") if path.name != "average_texture.png")
    if not pngs:
        raise FileNotFoundError(f"No PNG textures found in {out_textures_dir}")

    images = [imageio.imread(png) for png in pngs]
    average = np.mean(images, axis=0).astype(np.uint8)
    imageio.imwrite(atlas_texture, average)
    return atlas_texture


class TextureTransferService:
    """Texture-transfer helpers that can run outside Slicer."""

    def calculate_average_texture(self, texture_dir: Path) -> Path:
        return calculate_average_texture(texture_dir)
