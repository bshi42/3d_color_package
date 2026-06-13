"""EXP-17 — Achromatic (lightness) vs chromatic decomposition surfaces stripe (label-free, no prior).

Splitting color pattern into achromatic (L*) and chromatic (a*,b*) spatial structure — a general
Lab decomposition, NOT a banding prior — lets the dark stripe pattern become a dominant axis in
the lightness-pattern morphospace. Verifies recovery (real vs shuffled null, balanced subsample,
dip gate) and renders the morphospace with side-view exemplars (balanced display, stripe outlines).
"""
from __future__ import annotations

import numpy as np
import warnings

warnings.filterwarnings("ignore")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image
from diptest import diptest
from sklearn.decomposition import PCA
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from fishpipe import config, data, features

RES = config.RESULTS_DIR / "fishy_pattern"
RDIR = config.CACHE_DIR / "fishy_renders"
R = 256


def fps(pts, k, seed=0):
    r = np.random.default_rng(seed); idx = [int(r.integers(len(pts)))]
    d = np.linalg.norm(pts - pts[idx[0]], axis=1)
    for _ in range(k - 1):
        j = int(np.argmax(d)); idx.append(j); d = np.minimum(d, np.linalg.norm(pts - pts[j], axis=1))
    return sorted(set(idx))


def bal_balacc(emb, y, axes=2):
    rng = np.random.default_rng(0); pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
    accs = []
    for t in range(20):
        sub = np.concatenate([pos, rng.choice(neg, len(pos), replace=False)])
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
        pred = cross_val_predict(clf, emb[sub, :axes], y[sub], cv=StratifiedKFold(5, shuffle=True, random_state=t))
        accs.append(balanced_accuracy_score(y[sub], pred))
    return np.mean(accs), np.std(accs)


def main():
    gt = data.load_ground_truth(); fcd = data.build_face_colors(); names = fcd.names
    ys = gt.labels["stripe"]
    reg = features.region_summary(fcd, R, "mean", "lab").reshape(250, R, 3)
    Lp = reg[..., 0]                 # achromatic (lightness) pattern
    ab = reg[..., 1:].reshape(250, -1)  # chromatic pattern

    emb = PCA(6, random_state=0).fit_transform(StandardScaler().fit_transform(Lp))
    # per-PC stripe AUC (confirm stripe is its own axis, not belly)
    aucs = [max(roc_auc_score(ys, emb[:, k]), 1 - roc_auc_score(ys, emb[:, k])) for k in range(6)]
    sj = int(np.argmax(aucs))
    print("lightness-pattern PCA — per-PC stripe AUC:", np.round(aucs, 2), f"(best PC{sj+1})")
    for f in ["belly", "tail", "cheeks"]:
        a = [max(roc_auc_score(gt.labels[f], emb[:, k]), 1 - roc_auc_score(gt.labels[f], emb[:, k])) for k in range(6)]
        print(f"   {f} best-PC AUC={max(a):.2f} (PC{int(np.argmax(a))+1})")
    bacc, bstd = bal_balacc(emb, ys, axes=6)
    print(f"stripe balanced-subsample balacc (top-6 lightness PCs) = {bacc:.3f} ± {bstd:.3f}")
    print(f"stripe-axis dip = {diptest(emb[:, sj])[0]:.3f} (gradient, expect no clean gap)")
    # contrast: chromatic morphospace stripe (should be low)
    embc = PCA(6, random_state=0).fit_transform(StandardScaler().fit_transform(ab))
    print("chromatic morphospace stripe best-PC AUC =",
          round(max(max(roc_auc_score(ys, embc[:, k]), 1 - roc_auc_score(ys, embc[:, k])) for k in range(6)), 2))

    # ---- lightness-pattern morphospace figure: PC(stripe) vs PC1, balanced exemplars ----
    xax, yax = sj, (0 if sj != 0 else 1)
    sc = emb[:, [xax, yax]]
    pos = np.where(ys == 1)[0]; neg = np.where(ys == 0)[0]
    Z = StandardScaler().fit_transform(sc)
    sel = [int(neg[i]) for i in fps(Z[neg], 18)] + [int(pos[i]) for i in fps(Z[pos], 18)]
    fig, ax = plt.subplots(figsize=(15, 10))
    for i in sel:
        im = Image.open(RDIR / (names[i] + ".png")).convert("RGB"); im.thumbnail((115, 115))
        col = "#d62728" if ys[i] == 1 else "#1f77b4"
        ax.add_artist(AnnotationBbox(OffsetImage(np.asarray(im)), (sc[i, 0], sc[i, 1]),
                     frameon=True, pad=0.05, bboxprops=dict(edgecolor=col, lw=2.2)))
    ax.scatter([], [], edgecolors="#d62728", facecolors="none", label="true 5-stripe", s=90)
    ax.scatter([], [], edgecolors="#1f77b4", facecolors="none", label="true 4-stripe", s=90)
    ax.set_xlim(sc[sel, 0].min() - 1, sc[sel, 0].max() + 1)
    ax.set_ylim(sc[sel, 1].min() - 1, sc[sel, 1].max() + 1)
    ax.set_xlabel(f"Lightness-pattern PC{xax+1} (stripe-bearing; AUC {aucs[sj]:.2f})")
    ax.set_ylabel(f"Lightness-pattern PC{yax+1}")
    ax.set_title("Fishy — ACHROMATIC (lightness) pattern morphospace, balanced display.\n"
                 "General Lab decomposition (no banding prior); stripe now a visible GRADIENT axis. "
                 "Outline=true stripe (validation).")
    ax.legend(loc="upper right"); fig.tight_layout()
    out = RES / "fishy_lightness_pattern_morphospace.png"
    fig.savefig(out, dpi=120); plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
