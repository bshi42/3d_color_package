"""EXP-18 — Visualize what a GFT 'looks like' on the fishy mesh.

(A) The graph Fourier BASIS: paint Laplacian eigenvectors on the fish — low modes are smooth,
    high modes oscillate (the surface analog of sines/cosines).
(B) The GFT of a real signal: the spectrum |x_hat| vs graph-frequency for one fish's lightness,
    and the low-pass RECONSTRUCTION at increasing #modes (Fourier partial sums = blurry -> sharp).
"""
from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image, ImageDraw

from fishpipe import config, data, spectral, render

RES = config.RESULTS_DIR / "gft"
RES.mkdir(parents=True, exist_ok=True)


def vec_to_face_colors(mesh, vec, cmap="coolwarm"):
    v = vec / (np.abs(vec).max() + 1e-12)            # [-1,1]
    rgb = cm.get_cmap(cmap)(v * 0.5 + 0.5)[:, :3]    # per-vertex RGB
    return (rgb[mesh.face_v].mean(1) * 255).astype(np.uint8)


def gray_face_colors(mesh, Lvert):
    g = np.clip((Lvert - 30) / 60.0, 0, 1)           # stretch contrast around the fish's L range
    rgb = cm.get_cmap("viridis")(g)[:, :3]
    return (rgb[mesh.face_v].mean(1) * 255).astype(np.uint8)


def main():
    mesh = data.load_mesh()
    fcd = data.build_face_colors()
    w, U = spectral.laplacian_eigenbasis(300)        # eigvals, eigvecs (Nv,300), cached
    b = (mesh.vertices[:, 2].min(), mesh.vertices[:, 2].max(),
         mesh.vertices[:, 1].min(), mesh.vertices[:, 1].max())

    # ---------- (A) basis: eigenmodes painted on the fish ----------
    modes = [1, 3, 8, 30, 100, 280]
    cell_w = 0
    imgs = []
    for m in modes:
        im = render.render_side_view(mesh, vec_to_face_colors(mesh, U[:, m]), px=180, bounds=b,
                                     bg=(1, 1, 1))
        imgs.append((m, im)); cell_w = max(cell_w, im.shape[1])
    cw, ch = cell_w, imgs[0][1].shape[0] + 26
    sheet = Image.new("RGB", (3 * cw, 2 * ch), (255, 255, 255)); d = ImageDraw.Draw(sheet)
    for i, (m, im) in enumerate(imgs):
        x, y = (i % 3) * cw, (i // 3) * ch
        sheet.paste(Image.fromarray(im), (x, y + 22))
        d.text((x + 6, y + 4), f"eigenmode {m}  (graph-freq sqrt(lambda)={np.sqrt(w[m]):.3f})", fill=(0, 0, 0))
    sheet.save(RES / "gft_eigenmodes.png")
    print("saved gft_eigenmodes.png — low modes smooth, high modes oscillate")

    # ---------- (B) spectrum + reconstruction of one fish's lightness ----------
    vlab = spectral.vertex_lab(fcd)                  # (N, Nv, 3) cached
    idx = 0                                          # a 5-stripe fish
    L = vlab[idx, :, 0].astype(np.float64)
    Lc = L - L.mean()
    coeff = U.T @ Lc                                 # GFT: x_hat (300,)

    fig, axs = plt.subplots(1, 2, figsize=(13, 4))
    axs[0].plot(np.sqrt(w), np.abs(coeff), lw=0.8)
    axs[0].set_xlabel("graph frequency  sqrt(lambda)"); axs[0].set_ylabel("|GFT coefficient|")
    axs[0].set_title("GFT spectrum of fish #0 lightness\n(low-freq = belly/tail/base blobs dominate)")
    axs[1].bar(range(len(coeff)), np.abs(coeff), width=1.0)
    axs[1].set_xlabel("eigenmode index (low->high freq)"); axs[1].set_ylabel("|coeff|")
    axs[1].set_title("same, by mode index")
    fig.tight_layout(); fig.savefig(RES / "gft_spectrum.png", dpi=130); plt.close(fig)
    print("saved gft_spectrum.png")

    # reconstruction at increasing #modes (low-pass partial sums)
    cuts = [3, 10, 40, 150, 300]
    imgs2 = [("original", gray_face_colors(mesh, L))]
    for k in cuts:
        Lk = U[:, :k] @ coeff[:k] + L.mean()
        imgs2.append((f"{k} modes", gray_face_colors(mesh, Lk)))
    cw2 = max(im.shape[1] for _, im in imgs2); ch2 = imgs2[0][1].shape[0] + 24
    sheet2 = Image.new("RGB", (len(imgs2) * cw2, ch2), (255, 255, 255)); d2 = ImageDraw.Draw(sheet2)
    for i, (lbl, im) in enumerate(imgs2):
        sheet2.paste(Image.fromarray(im), (i * cw2, 20)); d2.text((i * cw2 + 6, 4), lbl, fill=(0, 0, 0))
    sheet2.save(RES / "gft_reconstruction.png")
    print("saved gft_reconstruction.png — adding graph-Fourier modes: blurry -> sharp (like FFT partial sums)")


if __name__ == "__main__":
    main()
