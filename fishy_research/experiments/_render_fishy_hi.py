"""Cache high-resolution fishy side-view renders for publication-grade exemplar figures."""
import numpy as np
from PIL import Image
from fishpipe import config, data, render

HI = config.CACHE_DIR / "fishy_renders_hi"; HI.mkdir(exist_ok=True)
fcd = data.build_face_colors(verbose=False); mesh = data.load_mesh()
b = (mesh.vertices[:, 2].min(), mesh.vertices[:, 2].max(),
     mesh.vertices[:, 1].min(), mesh.vertices[:, 1].max())
for i, n in enumerate(fcd.names):
    p = HI / f"{n}.png"
    if not p.exists():
        Image.fromarray(render.render_side_view(mesh, fcd.colors[i], px=520, bounds=b)).save(p)
    if i % 50 == 0:
        print(f"  {i+1}/{len(fcd.names)}")
print(f"done -> {HI}")
