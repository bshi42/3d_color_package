"""Pre-render an actual textured-MESH exterior view per mussel specimen (not a texture crop).

Each specimen is the shared atlas valve coloured by its own per-face sampled colours
(render.render_textured_rgba), viewed on the exterior (periostracum) side, transparent
background, tight-cropped to the silhouette. Because every specimen shares the atlas mesh the
silhouette is identical, so the renders are auto-aligned and equal-sized. Saves RGBA PNGs to
results/mussels/exterior_render/ for the charisma-Fig.5 dendrograms and any exemplar morphospace.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from fishpipe import config, dataset, render

DS = dataset.MUSSELS
OUTDIR = config.RESULTS_DIR / "mussels" / "exterior_render"
LONG_AXIS = 700     # saved long-axis px (downscaled from the supersampled render -> smooth edges)


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    mesh = DS.load_mesh()
    fcd, _ = dataset.build_face_colors(DS)
    # align=True frames every specimen from the shared atlas silhouette -> auto-registered, landscape
    for i, name in enumerate(fcd.names):
        im = render.render_textured_rgba(mesh, fcd.colors[i], px=440, supersample=3, align=True)
        pim = Image.fromarray(im, "RGBA")
        scale = LONG_AXIS / max(pim.size)
        pim = pim.resize((round(pim.width * scale), round(pim.height * scale)), Image.LANCZOS)
        pim.save(OUTDIR / f"{name}.png")
        if i % 10 == 0 or i == len(fcd.names) - 1:
            print(f"  rendered {i + 1}/{len(fcd.names)}: {name}  {pim.size}")
    print(f"saved {len(fcd.names)} exterior-mesh renders -> {OUTDIR}")


if __name__ == "__main__":
    main()
