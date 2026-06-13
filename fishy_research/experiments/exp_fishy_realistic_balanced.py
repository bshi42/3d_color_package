"""Fishy realistic morphospace — CLASS-BALANCED DISPLAY + balanced separability check.

Same general (assumption-free) Gabor axes as exp_fishy_realistic_morphospace, but the displayed
exemplars are balanced equally across true stripe classes (labels used for DISPLAY SELECTION
only — a validation aid, like the outlines), so the visual isn't dominated by the 80% majority.
Also reports balanced-subsample separability so the 'does not separate' claim is imbalance-fair.
"""
from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from fishpipe import config, data, embed, texture2d

RES = config.RESULTS_DIR / "fishy_pattern"
RDIR = config.CACHE_DIR / "fishy_renders"
GCACHE = config.CACHE_DIR / "fishy_gabor_La.npy"


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

    if GCACHE.exists():
        G = np.load(GCACHE)
    else:
        G = texture2d.gabor_descriptor_set(imgs, channels=("L", "a"), size=(96, 192))
        np.save(GCACHE, G)
    sc = embed.pca(G, n_components=4, standardize=True).scores[:, :2]

    # ----- balanced separability (so the conclusion isn't an imbalance artifact) -----
    print(f"full-data top-2 axis AUC vs stripe (rank-based, imbalance-robust) = "
          f"{max(roc_auc_score(ys, sc[:,0]), 1-roc_auc_score(ys, sc[:,0]), roc_auc_score(ys, sc[:,1]), 1-roc_auc_score(ys, sc[:,1])):.3f}")
    rng = np.random.default_rng(0)
    pos = np.where(ys == 1)[0]; neg = np.where(ys == 0)[0]
    accs = []
    for t in range(20):  # 20 balanced subsamples: all 52 fives + 52 random fours
        sub = np.concatenate([pos, rng.choice(neg, len(pos), replace=False)])
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
        pred = cross_val_predict(clf, sc[sub], ys[sub], cv=StratifiedKFold(5, shuffle=True, random_state=t))
        accs.append(balanced_accuracy_score(ys[sub], pred))
    print(f"balanced-subsample top-2 PCA stripe balacc = {np.mean(accs):.3f} ± {np.std(accs):.3f} (chance 0.5)")

    # ----- class-balanced DISPLAY (equal true-4 and true-5, spread within each) -----
    Z = StandardScaler().fit_transform(sc)
    sel = [int(neg[i]) for i in fps(Z[neg], 18)] + [int(pos[i]) for i in fps(Z[pos], 18)]
    fig, ax = plt.subplots(figsize=(15, 11))
    for i in sel:
        im = Image.fromarray(imgs[i]); im.thumbnail((120, 120))
        col = "#d62728" if ys[i] == 1 else "#1f77b4"
        ax.add_artist(AnnotationBbox(OffsetImage(np.asarray(im)), (sc[i, 0], sc[i, 1]),
                     frameon=True, pad=0.05, bboxprops=dict(edgecolor=col, lw=2.4)))
    ax.scatter([], [], edgecolors="#d62728", facecolors="none", label="true 5-stripe", s=90)
    ax.scatter([], [], edgecolors="#1f77b4", facecolors="none", label="true 4-stripe", s=90)
    pad = 1.0
    ax.set_xlim(sc[sel, 0].min() - pad, sc[sel, 0].max() + pad)
    ax.set_ylim(sc[sel, 1].min() - pad, sc[sel, 1].max() + pad)
    ax.set_xlabel("Pattern-PC1 (general texture descriptor — no pattern-type assumptions)")
    ax.set_ylabel("Pattern-PC2 (general texture descriptor)")
    ax.set_title("Fishy realistic pattern morphospace — CLASS-BALANCED DISPLAY (18 true-4 + 18 true-5)\n"
                 "general axes (no banding prior); outline=true stripe. If reds/blues still intermix, "
                 "the morphospace genuinely doesn't separate stripe.")
    ax.legend(loc="upper right")
    fig.tight_layout()
    out = RES / "fishy_realistic_balanced_display.png"
    fig.savefig(out, dpi=120); plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
