"""Fishy — pattern morphospace with side-view exemplars.

Renders a lateral view of every fish, builds an orientation-aware (Gabor on L*) pattern
descriptor that keys on the stripe bars, and lays the fish out in a 2-D morphospace with a
spread of side-view exemplars (border colored by true stripe count, 4 vs 5).
"""
from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image

from fishpipe import config, data, embed, metrics, render, texture2d

RES = config.RESULTS_DIR / "fishy_pattern"
RDIR = config.CACHE_DIR / "fishy_renders"
RES.mkdir(parents=True, exist_ok=True)
RDIR.mkdir(parents=True, exist_ok=True)


def farthest_point_sample(pts, k, seed=0):
    rng = np.random.default_rng(seed)
    idx = [int(rng.integers(len(pts)))]
    d = np.linalg.norm(pts - pts[idx[0]], axis=1)
    for _ in range(k - 1):
        j = int(np.argmax(d)); idx.append(j)
        d = np.minimum(d, np.linalg.norm(pts - pts[j], axis=1))
    return sorted(set(idx))


def main():
    fcd = data.build_face_colors(); gt = data.load_ground_truth(); names = fcd.names
    mesh = data.load_mesh()
    b = (mesh.vertices[:, 2].min(), mesh.vertices[:, 2].max(),
         mesh.vertices[:, 1].min(), mesh.vertices[:, 1].max())

    # ---- render all 250 side views (cache to disk) ----
    print("rendering side views...")
    imgs = []
    for i, n in enumerate(names):
        p = RDIR / f"{n}.png"
        if p.exists():
            imgs.append(np.asarray(Image.open(p).convert("RGB")))
        else:
            im = render.render_side_view(mesh, fcd.colors[i], px=150, bounds=b)
            Image.fromarray(im).save(p); imgs.append(im)
        if i % 50 == 0:
            print(f"  {i+1}/{len(names)}")

    # ---- orientation-aware pattern descriptor on the L* (stripe) channel ----
    G = texture2d.gabor_descriptor_set(imgs, channels=("L",), size=(96, 192),
                                       freqs=(0.08, 0.18, 0.35), n_orient=6)
    ys = gt.labels["stripe"]
    print(f"Gabor(L) descriptor {G.shape}; stripe balacc from descriptor: "
          f"{metrics.factor_recoverability(G, ys)['balanced_acc']:.3f}")
    emb = embed.pca(G, n_components=4, standardize=True)
    sc = emb.scores[:, :2]
    print("stripe recover from Gabor PCA(top4):",
          round(metrics.factor_recoverability(emb.scores, ys)['balanced_acc'], 3))

    # ---- exemplar morphospace ----
    keep = farthest_point_sample((sc - sc.mean(0)) / sc.std(0), 48)
    fig, ax = plt.subplots(figsize=(17, 12))
    for i in keep:
        im = Image.fromarray(imgs[i]); im.thumbnail((120, 120))
        col = "#d62728" if ys[i] == 1 else "#1f77b4"     # 5 vs 4 stripes
        ax.add_artist(AnnotationBbox(OffsetImage(np.asarray(im), zoom=1.0), (sc[i, 0], sc[i, 1]),
                     frameon=True, pad=0.05, bboxprops=dict(edgecolor=col, lw=2.0)))
    pad = 1.0
    ax.set_xlim(sc[keep, 0].min() - pad, sc[keep, 0].max() + pad)
    ax.set_ylim(sc[keep, 1].min() - pad, sc[keep, 1].max() + pad)
    ax.set_xlabel("Pattern-PC1"); ax.set_ylabel("Pattern-PC2")
    ax.set_title("Fishy — PATTERN morphospace with side-view exemplars "
                 "(orientation-aware Gabor on stripes; red border=5 stripes, blue=4)")
    fig.tight_layout()
    fig.savefig(RES / "fishy_pattern_morphospace_exemplars.png", dpi=120)
    plt.close(fig)
    print(f"saved {RES/'fishy_pattern_morphospace_exemplars.png'}")


if __name__ == "__main__":
    main()
