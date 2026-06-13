"""Fishy — LABEL-FREE pattern morphospace with side-view exemplars.

Everything in the PLOT uses only label-free information (measurements + auto-clusters +
exemplar renders), exactly as it would be on a real unlabeled dataset. Ground-truth labels
are used ONLY to print a separate validation line — never to color the plot.
"""
from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from fishpipe import config, data, recommended

RES = config.RESULTS_DIR / "fishy_pattern"
RDIR = config.CACHE_DIR / "fishy_renders"
rng = np.random.default_rng(0)
CLUSTER_COLORS = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a"]


def fps(pts, k, seed=0):
    r = np.random.default_rng(seed); idx = [int(r.integers(len(pts)))]
    d = np.linalg.norm(pts - pts[idx[0]], axis=1)
    for _ in range(k - 1):
        j = int(np.argmax(d)); idx.append(j); d = np.minimum(d, np.linalg.norm(pts - pts[j], axis=1))
    return sorted(set(idx))


def main():
    fcd = data.build_face_colors(); names = fcd.names
    pt = recommended.pattern_traits(fcd)
    bc, bs = pt["banding_count"], pt["banding_strength"]

    # ---- LABEL-FREE: measured pattern-trait space + auto-cluster ----
    X = StandardScaler().fit_transform(np.column_stack([bc, bs]))
    best = min((GaussianMixture(k, random_state=0).fit(X) for k in range(1, 5)), key=lambda g: g.bic(X))
    clusters = best.predict(X)
    k = len(set(clusters))

    # order cluster ids by mean banding count (so colors read low->high stripes)
    order = np.argsort([bc[clusters == c].mean() for c in range(k)])
    remap = {c: i for i, c in enumerate(order)}
    clusters = np.array([remap[c] for c in clusters])

    # ---- validation ONLY (not used in the plot) ----
    gt = data.load_ground_truth()
    print(f"auto k={k}; (validation) ARI vs true stripe={adjusted_rand_score(gt.labels['stripe'], clusters):.3f}")
    for c in range(k):
        m = clusters == c
        print(f"  cluster{c}: n={m.sum()} mean_banding_count={bc[m].mean():.2f}  "
              f"(validation: true 5-stripe frac={gt.labels['stripe'][m].mean():.2f})")

    # ---- plot: x=measured banding count (jittered), y=banding strength, color=AUTO cluster ----
    xj = bc + rng.normal(0, 0.07, len(bc))
    coords = StandardScaler().fit_transform(np.column_stack([xj, bs]))
    # CLASS-BALANCED exemplars: equal number per AUTO-cluster (label-free), spread within each
    per = 20
    sel = []
    for c in range(k):
        ci = np.where(clusters == c)[0]
        sub = fps(coords[ci], min(per, len(ci)))
        sel += [int(ci[s]) for s in sub]
    fig, ax = plt.subplots(figsize=(16, 10))
    for i in sel:
        im = Image.open(RDIR / (names[i] + ".png")).convert("RGB"); im.thumbnail((110, 110))
        ax.add_artist(AnnotationBbox(OffsetImage(np.asarray(im)), (xj[i], bs[i]), frameon=True, pad=0.05,
                     bboxprops=dict(edgecolor=CLUSTER_COLORS[clusters[i]], lw=2.2)))
    # legend by auto-cluster
    for c in range(k):
        ax.scatter([], [], c=CLUSTER_COLORS[c], label=f"auto-cluster {c} (n={(clusters==c).sum()})", s=80)
    ax.set_xlim(xj[sel].min() - 0.4, xj[sel].max() + 0.4)
    ax.set_ylim(bs[sel].min() - 1.5, bs[sel].max() + 1.5)
    ax.set_xlabel("measured banding count  (label-free)")
    ax.set_ylabel("measured banding strength  (label-free)")
    ax.set_title("Fishy — LABEL-FREE pattern morphospace, CLASS-BALANCED exemplars "
                 "(equal per AUTO-cluster; no ground truth used)\nexemplars are CPU side-view renders")
    ax.legend(loc="upper right")
    fig.tight_layout()
    out = RES / "fishy_unlabeled_pattern_morphospace_balanced.png"
    fig.savefig(out, dpi=120); plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
