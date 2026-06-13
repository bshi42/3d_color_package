"""Mussels — exemplar pattern morphospace: SPECTRAL (no Gabor) vs GABOR, side by side.

Control for "does orientation-awareness matter": same exterior-image scatter, axes from the
magnitude-only graph-spectral descriptor vs the orientation-aware Gabor descriptor.
"""
from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image

from fishpipe import config, dataset, embed, spectral, texture2d

RES = config.RESULTS_DIR / "mussels"
EXT = RES / "exterior"
DS = dataset.MUSSELS
HILITE = {"UF_IZ_507450": "red", "UF_IZ_65304": "blue",
          "UF_IZ_507738": "lime", "UF_IZ_507616": "lime", "UF_IZ_507751": "lime"}


def s(n):
    return n.replace("UF_IZ_", "")


def panel(ax, sc, names, title):
    for i, n in enumerate(names):
        im = Image.open(EXT / (n + ".png")).convert("RGB")
        im.thumbnail((80, 80))
        col = HILITE.get(n, "0.4")
        ax.add_artist(AnnotationBbox(
            OffsetImage(np.asarray(im), zoom=1.0), (sc[i, 0], sc[i, 1]),
            frameon=True, pad=0.1,
            bboxprops=dict(edgecolor=col, lw=(2.5 if n in HILITE else 0.5))))
        if n in HILITE:
            ax.annotate(s(n), (sc[i, 0], sc[i, 1]), fontsize=7, xytext=(0, -30),
                        textcoords="offset points", ha="center", color=col)
    pad = 1.0
    ax.set_xlim(sc[:, 0].min() - pad, sc[:, 0].max() + pad)
    ax.set_ylim(sc[:, 1].min() - pad, sc[:, 1].max() + pad)
    ax.set_title(title); ax.set_xlabel("PC1"); ax.set_ylabel("PC2")


def main():
    fcd, _ = dataset.build_face_colors(DS)
    names = fcd.names
    crops = [np.asarray(Image.open(EXT / (n + ".png")).convert("RGB")) for n in names]
    mesh = DS.load_mesh()

    Xs = spectral.spectral_descriptor(fcd, k=200, n_bands=14, channels=("L", "a", "b", "chroma"),
                                      mesh=mesh, cache_dir=DS.cache_dir)
    Xg = texture2d.gabor_descriptor_set(crops, channels=("L", "a"))
    sc_s = embed.pca(Xs, 2, standardize=True).scores
    sc_g = embed.pca(Xg, 2, standardize=True).scores

    fig, axes = plt.subplots(1, 2, figsize=(22, 10))
    panel(axes[0], sc_s, names, "SPECTRAL pattern morphospace (magnitude-only, no Gabor)")
    panel(axes[1], sc_g, names, "GABOR pattern morphospace (orientation-aware)")
    fig.suptitle("Exemplar pattern morphospace — does orientation-awareness change the layout?  "
                 "(red=507450 wavy, blue=65304 straight-rings, green=green-rayed cohort)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = RES / "morphospace_compare.png"
    fig.savefig(out, dpi=120); plt.close(fig)
    # quick quantitative diff: where do the key shells sit (z-distance from centroid)?
    for tag, sc in [("SPECTRAL", sc_s), ("GABOR", sc_g)]:
        z = np.linalg.norm((sc - sc.mean(0)) / sc.std(0), axis=1)
        print(f"{tag} isolation (z from centroid): "
              f"507450={z[names.index('UF_IZ_507450')]:.2f}  65304={z[names.index('UF_IZ_65304')]:.2f}  "
              f"median={np.median(z):.2f}")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
