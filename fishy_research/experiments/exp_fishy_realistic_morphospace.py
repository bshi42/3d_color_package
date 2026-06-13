"""Fishy — REALISTIC pattern morphospace (what the module would show on UNKNOWN data).

Axes come from a GENERAL, assumption-free texture descriptor (Gabor energy over scales x
orientations) — NO 'banding count' / no prior that the data has stripes. PCA -> 2D. Side-view
render exemplars. Outlines = true stripe count, used ONLY as a post-hoc validation overlay.
We report, honestly, whether the unsupervised morphospace separates stripe and what auto-
clustering finds with no pattern prior.
"""
from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image
from sklearn.metrics import adjusted_rand_score, roc_auc_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from fishpipe import config, data, embed, metrics, texture2d

RES = config.RESULTS_DIR / "fishy_pattern"
RDIR = config.CACHE_DIR / "fishy_renders"


def fps(pts, k, seed=0):
    r = np.random.default_rng(seed); idx = [int(r.integers(len(pts)))]
    d = np.linalg.norm(pts - pts[idx[0]], axis=1)
    for _ in range(k - 1):
        j = int(np.argmax(d)); idx.append(j); d = np.minimum(d, np.linalg.norm(pts - pts[j], axis=1))
    return sorted(set(idx))


def main():
    fcd = data.build_face_colors(); gt = data.load_ground_truth(); names = fcd.names
    ys = gt.labels["stripe"]
    imgs = [np.asarray(Image.open(RDIR / (n + ".png")).convert("RGB")) for n in names]

    # GENERAL assumption-free texture descriptor (no banding prior)
    G = texture2d.gabor_descriptor_set(imgs, channels=("L", "a"), size=(96, 192))
    emb = embed.pca(G, n_components=4, standardize=True)
    sc = emb.scores[:, :2]

    # --- honest validation: does the general morphospace separate stripe? ---
    auc2 = max(roc_auc_score(ys, sc[:, 0]), 1 - roc_auc_score(ys, sc[:, 0]),
               roc_auc_score(ys, sc[:, 1]), 1 - roc_auc_score(ys, sc[:, 1]))
    print(f"general Gabor PCA — best single top-2 axis AUC vs stripe = {auc2:.3f}")
    print(f"  stripe balacc from full Gabor descriptor (classifier) = "
          f"{metrics.factor_recoverability(G, ys)['balanced_acc']:.3f}")
    # what does assumption-free auto-clustering find? (no pattern prior)
    Z = StandardScaler().fit_transform(sc)
    gm = min((GaussianMixture(k, random_state=0).fit(Z) for k in range(1, 6)), key=lambda g: g.bic(Z))
    cl = gm.predict(Z); k = len(set(cl))
    print(f"  auto-cluster on general morphospace: k={k}; ARI vs stripe (validation) = "
          f"{adjusted_rand_score(ys, cl):.3f}")

    # --- the plot ---
    keep = fps(Z, 48)
    fig, ax = plt.subplots(figsize=(16, 11))
    for i in keep:
        im = Image.fromarray(imgs[i]); im.thumbnail((115, 115))
        col = "#d62728" if ys[i] == 1 else "#1f77b4"
        ax.add_artist(AnnotationBbox(OffsetImage(np.asarray(im)), (sc[i, 0], sc[i, 1]),
                     frameon=True, pad=0.05, bboxprops=dict(edgecolor=col, lw=2.0)))
    ax.scatter([], [], edgecolors="#d62728", facecolors="none", label="true 5-stripe (validation)", s=90)
    ax.scatter([], [], edgecolors="#1f77b4", facecolors="none", label="true 4-stripe (validation)", s=90)
    pad = 1.0
    ax.set_xlim(sc[keep, 0].min() - pad, sc[keep, 0].max() + pad)
    ax.set_ylim(sc[keep, 1].min() - pad, sc[keep, 1].max() + pad)
    ax.set_xlabel("Pattern-PC1  (general texture descriptor — no pattern-type assumptions)")
    ax.set_ylabel("Pattern-PC2  (general texture descriptor)")
    ax.set_title("Fishy — REALISTIC pattern morphospace (general descriptor, what the module computes on "
                 "UNKNOWN data)\nexemplars = side-view renders; outline = true stripe count (post-hoc validation only)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    out = RES / "fishy_realistic_pattern_morphospace.png"
    fig.savefig(out, dpi=120); plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
