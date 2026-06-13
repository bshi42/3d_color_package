"""EXP-23 — Subtracting color-composition variance from the spatial pattern morphospace.

Residualize the spatial+Endler pattern features against the color-composition histogram
(partial regression) so the morphospace shows ARRANGEMENT variance, not 'which colors'. Shows
(a) it cleanly removes belly/tail, (b) what then dominates (the continuous stripe-geometry
nuisances), (c) whether stripe COUNT becomes visible (it stays a low-variance gradient).
"""
from __future__ import annotations

import numpy as np
import warnings

warnings.filterwarnings("ignore")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from fishpipe import config, data, segment, spatial, blobs, features

RES = config.RESULTS_DIR / "fishy_pattern"


def residualize(S, C):
    """Partial out C from S (remove the part of S linearly predictable from C)."""
    Ss = StandardScaler().fit_transform(S)
    Cs = StandardScaler().fit_transform(C)
    return Ss - LinearRegression().fit(Cs, Ss).predict(Cs)


def main():
    gt = data.load_ground_truth(); fcd = data.build_face_colors(); df = gt.params; ys = gt.labels["stripe"]
    seg = segment.segment(fcd, n_colors=8, smooth_iters=0)
    SE = np.concatenate([StandardScaler().fit_transform(b) for b in
                         [spatial.spatial_descriptor(fcd, seg), blobs.endler_transitions(fcd, seg)]], axis=1)
    C = features.area_hist(fcd, 12, "lab")             # color composition (non-spatial)

    base = PCA(6, random_state=0).fit_transform(StandardScaler().fit_transform(SE))
    resid = PCA(6, random_state=0).fit_transform(residualize(SE, C))

    def desc(emb, k):
        return (abs(pearsonr(emb[:, k], gt.labels["belly"])[0]), abs(pearsonr(emb[:, k], gt.labels["tail"])[0]),
                abs(pearsonr(emb[:, k], df.stripe_count)[0]), abs(pearsonr(emb[:, k], df.stripe_spacing)[0]),
                abs(pearsonr(emb[:, k], df.stripe_width)[0]))

    print("axis  belly tail count spacing width   (BASELINE spatial+endler)")
    for k in range(4):
        print(f"PC{k+1}  " + " ".join(f"{v:.2f}" for v in desc(base, k)))
    print("\naxis  belly tail count spacing width   (COLOR-SUBTRACTED)")
    for k in range(4):
        print(f"PC{k+1}  " + " ".join(f"{v:.2f}" for v in desc(resid, k)))

    # figure: before vs after, colored by belly / tail / stripe
    fig, axs = plt.subplots(2, 3, figsize=(15, 9))
    for col, (emb, tag) in enumerate([(base, "BEFORE (spatial+endler)"), (resid, "AFTER color-subtracted")]):
        pass
    rows = [(base, "BEFORE: spatial+endler"), (resid, "AFTER: color-composition subtracted")]
    factors = ["belly", "tail", "stripe"]
    cmaps = {"belly": "coolwarm", "tail": "PiYG", "stripe": "bwr"}
    for r, (emb, tag) in enumerate(rows):
        for c, f in enumerate(factors):
            ax = axs[r, c]
            sca = ax.scatter(emb[:, 0], emb[:, 1], c=gt.labels[f], cmap=cmaps[f], s=22,
                             edgecolors="k" if f == "stripe" else "none", linewidths=0.3)
            ax.set_title(f"{tag}\ncolored by {f}", fontsize=9)
            ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    fig.suptitle("Subtracting color-composition variance: belly/tail vanish, but stripe-geometry "
                 "(spacing/width) — not count — then dominates", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = RES / "fishy_color_subtracted_morphospace.png"
    fig.savefig(out, dpi=120); plt.close(fig)
    print(f"\nsaved {out}")

    # honest: best-axis stripe AUC before vs after; dip on residual
    from diptest import diptest
    print("\nbest top-4-PC stripe AUC: before=%.2f after=%.2f" % (
        max(max(roc_auc_score(ys, base[:, k]), 1 - roc_auc_score(ys, base[:, k])) for k in range(4)),
        max(max(roc_auc_score(ys, resid[:, k]), 1 - roc_auc_score(ys, resid[:, k])) for k in range(4))))
    print("residual PC1 dip p=%.3f (count is a gradient, not a clustered gap)" % diptest(resid[:, 0])[1])


if __name__ == "__main__":
    main()
