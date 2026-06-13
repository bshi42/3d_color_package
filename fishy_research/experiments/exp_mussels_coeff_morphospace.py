"""EXP-21 — Mussel PATTERN morphospace on compressed signed GFT coefficients (achromatic).

Same recipe as the fishy pattern morphospace (EXP-20) applied to the real mussel data:
descriptor = signed GFT coefficients of the lightness channel, compressed to k=40 modes
(denoise). Label-free; no ground truth, so axes are characterized by interpretable scalars
(darkness, ring-contrast, greenness) and points are the shell-exterior images, colored by
auto-detected cluster.
"""
from __future__ import annotations

import os
import numpy as np
import warnings

warnings.filterwarnings("ignore")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from fishpipe import config, dataset, features, spectral

RES = config.RESULTS_DIR / "mussels"
EXT = RES / "exterior"
DS = dataset.MUSSELS
CLUSTER_COLORS = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#666666"]


def s(n):
    return n.replace("UF_IZ_", "")


def main():
    fcd, _ = dataset.build_face_colors(DS)
    names = fcd.names
    mesh = DS.load_mesh()
    lab = features.rgb_to_lab(fcd.colors)
    darkness = -lab[..., 0].mean(1); contrast = lab[..., 0].std(1); greenness = -lab[..., 1].mean(1)

    # new descriptor: compressed signed GFT coefficients, achromatic (lightness) = pattern
    Xp = spectral.spectral_coeffs(fcd, k=40, channels=("L",), n_basis=200,
                                  mesh=mesh, cache_dir=DS.cache_dir)
    emb = PCA(6, random_state=0).fit_transform(StandardScaler().fit_transform(Xp))
    print(f"mussel lightness-pattern coeffs {Xp.shape}; PCA EV%={np.round(PCA(6).fit(StandardScaler().fit_transform(Xp)).explained_variance_ratio_[:4]*100,1)}")
    print("\nWhat each pattern PC encodes (|corr| with interpretable scalars):")
    print(f"{'PC':>4} {'darkness':>9} {'ring-contrast':>13} {'greenness':>10}")
    for k in range(4):
        print(f"PC{k+1:>2} " + " ".join(f"{abs(pearsonr(emb[:, k], v)[0]):>9.2f}"
              for v in [darkness, contrast, greenness]))

    Z = StandardScaler().fit_transform(emb[:, :2])
    gm = min((GaussianMixture(kk, random_state=0).fit(Z) for kk in range(1, 5)), key=lambda g: g.bic(Z))
    cl = gm.predict(Z); ncl = len(set(cl))
    print(f"\nauto-cluster on top-2 pattern PCs: k={ncl}  sizes={np.bincount(cl).tolist()}")

    sc = emb[:, :2]
    fig, ax = plt.subplots(figsize=(13, 10))
    for i, n in enumerate(names):
        im = Image.open(EXT / (n + ".png")).convert("RGB"); im.thumbnail((95, 95))
        ax.add_artist(AnnotationBbox(OffsetImage(np.asarray(im)), (sc[i, 0], sc[i, 1]), frameon=True,
                     pad=0.05, bboxprops=dict(edgecolor=CLUSTER_COLORS[cl[i] % len(CLUSTER_COLORS)], lw=2.4)))
        ax.annotate(s(n), (sc[i, 0], sc[i, 1]), fontsize=6, xytext=(0, -32),
                    textcoords="offset points", ha="center", color="0.3")
    for c in range(ncl):
        ax.scatter([], [], c=CLUSTER_COLORS[c % len(CLUSTER_COLORS)], label=f"auto-cluster {c} (n={(cl==c).sum()})", s=80)
    pad = 1.0
    ax.set_xlim(sc[:, 0].min() - pad, sc[:, 0].max() + pad)
    ax.set_ylim(sc[:, 1].min() - pad, sc[:, 1].max() + pad)
    ax.set_xlabel("Lightness-pattern PC1 (compressed signed GFT coeffs, k=40)")
    ax.set_ylabel("Lightness-pattern PC2")
    ax.set_title("Mussels — PATTERN morphospace on compressed signed GFT coefficients (achromatic, denoised).\n"
                 "Label-free; exemplars are shell exteriors, colored by auto-detected cluster.")
    ax.legend(loc="best"); fig.tight_layout()
    out = RES / "mussels_coeff_pattern_morphospace.png"
    fig.savefig(out, dpi=120); plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
