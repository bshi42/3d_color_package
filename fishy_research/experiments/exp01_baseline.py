"""EXP-01 — Baseline reproduction of the module morphospace.

Reproduces the module's per-specimen color analysis and quantifies how poorly it recovers
the 4 planted factors. Three feature representations × PCA/ICA/UMAP, scored objectively.
"""
from __future__ import annotations

import json
import time

import numpy as np

from fishpipe import config, data, embed, features, metrics, plotting

RESULTS = config.RESULTS_DIR / "exp01"
RESULTS.mkdir(parents=True, exist_ok=True)


def run_block(name, X, gt, results):
    print(f"\n### {name}  (X shape={X.shape})")
    summary = {}
    # PCA (no standardization — exactly as module)
    emb = embed.pca(X, n_components=6, standardize=False)
    card = metrics.scorecard(emb.scores, gt.labels)
    ev = emb.explained_variance
    print(metrics.format_scorecard(card, f"[PCA, no-standardize]  EV(PC1..3)="
                                   f"{np.round(ev[:3]*100,1)}%"))
    plotting.morphospace_grid(emb.scores, gt, f"{name} — PCA (PC1 vs PC2)",
                              RESULTS / f"{name}_pca.png", axis_label="PC")
    summary["pca"] = card
    summary["pca_ev"] = ev[:6].tolist()
    # joint ARI (belly×tail×stripe)
    joint = data.joint_label(gt)
    summary["pca_ari_joint"] = metrics.ari_for_joint(emb.scores[:, :3], joint)
    print(f"   joint(belly×tail×stripe) ARI via KMeans on PC1-3: {summary['pca_ari_joint']:.3f}")
    results[name] = summary
    return emb


def main():
    t0 = time.time()
    gt = data.load_ground_truth()
    fcd = data.build_face_colors()
    results = {}

    # M1: spatial RGB flatten, ALL faces (PCAMorphospace path, per-face)
    X_rgb = features.spatial_flatten(fcd, color_space="rgb", subsample=None)
    run_block("spatial_rgb_full", X_rgb, gt, results)

    # M2: spatial Lab flatten, subsampled 4000 faces (module subsampled path)
    X_lab = features.spatial_flatten(fcd, color_space="lab", subsample=4000)
    emb_lab = run_block("spatial_lab_sub4000", X_lab, gt, results)

    # M3: area-weighted Lab cluster histogram, K=24 (module area-weighted path)
    X_hist = features.area_hist(fcd, n_clusters=24, color_space="lab")
    run_block("area_hist_lab_k24", X_hist, gt, results)

    # Also: ICA + UMAP on the Lab-subsampled rep (module offers these)
    print("\n### nonlinear / ICA on spatial_lab_sub4000")
    ica = embed.ica(X_lab, n_components=6, standardize=False)
    print(metrics.format_scorecard(metrics.scorecard(ica.scores, gt.labels), "[ICA]"))
    plotting.morphospace_grid(ica.scores, gt, "spatial_lab_sub4000 — ICA (IC1 vs IC2)",
                              RESULTS / "spatial_lab_sub4000_ica.png", axis_label="IC")
    um = embed.umap_embed(X_lab, n_components=2, standardize=False)
    print(metrics.format_scorecard(metrics.scorecard(um.scores, gt.labels), "[UMAP]"))
    plotting.morphospace_grid(um.scores, gt, "spatial_lab_sub4000 — UMAP",
                              RESULTS / "spatial_lab_sub4000_umap.png", axis_label="UMAP")

    # pairwise PC grid for the best linear rep, to see which PC pair holds each factor
    for f in config.CLUSTER_FACTORS:
        plotting.pairwise_pc_grid(emb_lab.scores, gt, f,
                                  f"spatial_lab_sub4000 PCA — colored by {f}",
                                  RESULTS / f"pairgrid_lab_{f}.png", max_pcs=4)

    with open(RESULTS / "scorecards.json", "w") as fh:
        json.dump(results, fh, indent=2, default=float)
    print(f"\nDONE in {time.time()-t0:.1f}s. Results in {RESULTS}")


if __name__ == "__main__":
    main()
