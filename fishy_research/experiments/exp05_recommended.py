"""EXP-05 — The recommended analysis, end-to-end, with publication-style figures.

Demonstrates: (1) a COLOR morphospace that auto-discovers the belly×tail quadrants;
(2) a measured PATTERN view that makes stripe count visible (where color PCA can't);
(3) a rare-variant novelty overlay; (4) a baseline→recommended scorecard.
All label-free (ground truth used ONLY to color/score the plots, never to fit).
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import adjusted_rand_score

from fishpipe import config, data, embed, features, metrics, recommended

RESULTS = config.RESULTS_DIR / "exp05"
RESULTS.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(0)


def main():
    gt = data.load_ground_truth()
    fcd = data.build_face_colors()
    out = {}

    # ----- COLOR morphospace + auto cluster -----
    cm = recommended.color_morphospace(fcd)
    clusters = recommended.auto_cluster(cm.scores[:, :2], max_k=8)
    joint_bt = data.joint_label(gt, ("belly", "tail"))
    ari = adjusted_rand_score(joint_bt, clusters)
    out["color_auto_clusters"] = int(len(np.unique(clusters)))
    out["color_ARI_vs_bellytail"] = float(ari)

    # ----- PATTERN traits (measured, label-free) -----
    pt = recommended.pattern_traits(fcd)
    ys = gt.labels["stripe"]
    pat_balacc = metrics.factor_recoverability(pt["pattern_vector"], ys)["balanced_acc"]
    out["pattern_stripe_balacc"] = float(pat_balacc)

    # ----- RARE VARIANT (cheeks): general chromatic-spectral descriptor + scalar overlay -----
    nv = recommended.novelty_score(fcd)
    rv = recommended.rare_variant_descriptor(fcd)
    out["novelty_cheeks_balacc"] = float(
        metrics.factor_recoverability(rv, gt.labels["cheeks"])["balanced_acc"]
    )
    out["novelty_scalar_cheeks_balacc"] = float(
        metrics.factor_recoverability(nv.reshape(-1, 1), gt.labels["cheeks"])["balanced_acc"]
    )

    # ===== Figure 1: color morphospace (auto-cluster vs truth) =====
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    sc = ax[0].scatter(cm.scores[:, 0], cm.scores[:, 1], c=clusters, cmap="tab10", s=24)
    ax[0].set_title(f"COLOR morphospace — auto-detected clusters (k={out['color_auto_clusters']})")
    ax[0].set_xlabel("Color-PC1"); ax[0].set_ylabel("Color-PC2")
    j = ax[1].scatter(cm.scores[:, 0], cm.scores[:, 1], c=joint_bt, cmap="tab10", s=24)
    ax[1].set_title(f"same plot — true belly×tail (ARI={ari:.2f})")
    ax[1].set_xlabel("Color-PC1"); ax[1].set_ylabel("Color-PC2")
    fig.suptitle("Color view: hue clusters are auto-discovered with no tuning", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(RESULTS / "fig1_color_morphospace.png", dpi=140); plt.close(fig)

    # ===== Figure 2: measured pattern view (stripe becomes visible) =====
    jit = rng.normal(0, 0.06, len(ys))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    for cls, lab, col in [(0, "4 stripes", "#1f77b4"), (1, "5 stripes", "#d62728")]:
        m = ys == cls
        ax[0].scatter(pt["banding_count"][m] + jit[m], pt["banding_strength"][m], s=26,
                      alpha=0.7, label=lab, color=col)
    ax[0].set_xlabel("measured banding count (# dark bands)")
    ax[0].set_ylabel("banding strength (profile contrast)")
    ax[0].set_title("PATTERN view — measured traits separate stripe count")
    ax[0].legend()
    pat_emb = embed.pca(pt["pattern_vector"], 2, standardize=True)
    for cls, lab, col in [(0, "4 stripes", "#1f77b4"), (1, "5 stripes", "#d62728")]:
        m = ys == cls
        ax[1].scatter(pat_emb.scores[m, 0], pat_emb.scores[m, 1], s=26, alpha=0.7, label=lab, color=col)
    ax[1].set_title("PCA of the SAME traits — does NOT separate stripe\n"
                    "(unsupervised axes bury it; cf. measured axis at left)")
    ax[1].set_xlabel("Pattern-PC1"); ax[1].set_ylabel("Pattern-PC2"); ax[1].legend()
    fig.suptitle("Pattern view: stripe count is MEASURED (left), not DISCOVERED by PCA (right)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(RESULTS / "fig2_pattern_view.png", dpi=140); plt.close(fig)

    # ===== Figure 3: the money shot — baseline can't show stripe, recommended can =====
    X_base = features.spatial_flatten(fcd, color_space="lab", subsample=4000)
    base = embed.pca(X_base, 4, standardize=False)
    base_stripe = metrics.factor_recoverability(base.scores, ys)["balanced_acc"]
    out["baseline_stripe_balacc"] = float(base_stripe)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    for cls, lab, col in [(0, "4 stripes", "#1f77b4"), (1, "5 stripes", "#d62728")]:
        m = ys == cls
        ax[0].scatter(base.scores[m, 0], base.scores[m, 1], s=24, alpha=0.7, label=lab, color=col)
    ax[0].set_title(f"BEFORE: color PCA morphospace\nstripe invisible (balacc={base_stripe:.2f})")
    ax[0].set_xlabel("PC1"); ax[0].set_ylabel("PC2"); ax[0].legend()
    for cls, lab, col in [(0, "4 stripes", "#1f77b4"), (1, "5 stripes", "#d62728")]:
        m = ys == cls
        ax[1].scatter(pt["banding_count"][m] + jit[m], pt["banding_strength"][m], s=24,
                      alpha=0.7, label=lab, color=col)
    ax[1].axvline(4.5, ls="--", c="grey", lw=1)
    ax[1].set_title(f"AFTER: measured banding axis\nstripe separated (balacc={pat_balacc:.2f})")
    ax[1].set_xlabel("measured banding count (# dark bands)")
    ax[1].set_ylabel("banding strength"); ax[1].legend()
    fig.suptitle("Structure becomes visible only when MEASURED, not discovered by color PCA", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(RESULTS / "fig3_before_after.png", dpi=140); plt.close(fig)

    # ===== scorecard table =====
    print("\n=== BASELINE vs RECOMMENDED (balanced CV accuracy; chance≈0.5) ===")
    base_card = metrics.scorecard(base.scores, gt.labels)
    print(f"{'factor':8s} {'baseline(colorPCA)':>20s} {'recommended':>14s}")
    rec = {
        "belly": metrics.factor_recoverability(cm.scores, gt.labels["belly"])["balanced_acc"],
        "tail": metrics.factor_recoverability(cm.scores, gt.labels["tail"])["balanced_acc"],
        "stripe": pat_balacc,
        "cheeks": out["novelty_cheeks_balacc"],
    }
    for f in config.CLUSTER_FACTORS:
        print(f"{f:8s} {base_card[f]['balanced_acc']:>20.3f} {rec[f]:>14.3f}")
    out["recommended_scorecard"] = rec
    out["baseline_scorecard"] = {f: float(base_card[f]["balanced_acc"]) for f in config.CLUSTER_FACTORS}

    with open(RESULTS / "scorecard.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nSaved figures + scorecard to {RESULTS}")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
