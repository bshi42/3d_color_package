"""EXP-02/03/04 (consolidated) — structure-descriptor comparison + discovery-vs-measurement.

Reproduces the key tables behind the central finding: many descriptors let a *classifier*
recover stripe count (~0.84-0.93), but no unsupervised reduction puts it on the top axes —
only a *measured* label-free banding scalar makes it visible. Prints one table.
"""
from __future__ import annotations

import numpy as np

from fishpipe import data, features, structure, spectral, embed, metrics, recommended


def bal(X, y):
    return metrics.factor_recoverability(X, y)["balanced_acc"]


def main():
    gt = data.load_ground_truth()
    fcd = data.build_face_colors()
    ys, yc = gt.labels["stripe"], gt.labels["cheeks"]

    print("\n# A. Descriptor comparison — STRIPE recoverability (balanced CV acc)")
    print(f"{'descriptor':32s} {'direct':>8s} {'top6-PCA':>9s}")
    rows = [
        ("baseline color flatten (Lab sub)", features.spatial_flatten(fcd, "lab", 4000)),
        ("color area-histogram K24", features.area_hist(fcd, 24, "lab")),
        ("region-mean color R=256", features.region_summary(fcd, 256, "mean", "lab")),
        ("graph-spectral (all channels)", spectral.spectral_descriptor(fcd, 300, 12)),
        ("graph-spectral (L only)", spectral.spectral_descriptor(fcd, 300, 12, ("L",))),
        ("Endler adjacency k=6", structure.endler_adjacency(fcd, 6)),
        ("connected-component counts", structure.component_counts(fcd, 6)),
        ("axial banding FFT (auto axis)", structure.axial_banding_descriptor(fcd)),
        ("MEASURED pattern_vector", recommended.pattern_traits(fcd)["pattern_vector"]),
    ]
    for name, X in rows:
        direct = bal(X, ys)
        pca = bal(embed.pca(X, 6, standardize=True).scores, ys)
        print(f"{name:32s} {direct:>8.3f} {pca:>9.3f}")

    print("\n# B. Discovery (unsupervised top axis) cannot surface stripe")
    Xrich = np.concatenate([
        features.region_summary(fcd, 128, "mean", "lab"),
        structure.axial_banding_descriptor(fcd),
        spectral.spectral_descriptor(fcd, 300, 12, ("L",)),
    ], axis=1)
    for method, emb in [("PCA", embed.pca(Xrich, 8, True)), ("ICA", embed.ica(Xrich, 8, True))]:
        aucs = []
        from sklearn.metrics import roc_auc_score
        for d in range(emb.scores.shape[1]):
            a = roc_auc_score(ys, emb.scores[:, d])
            aucs.append(max(a, 1 - a))
        print(f"  {method}: best single-axis stripe AUC = {max(aucs):.3f} (axis {int(np.argmax(aucs))})")

    print("\n# C. MEASUREMENT (label-free) makes stripe visible")
    pt = recommended.pattern_traits(fcd)
    from sklearn.metrics import roc_auc_score
    bc = pt["banding_count"]
    a = roc_auc_score(ys, bc)
    print(f"  banding_count single-scalar AUC = {max(a, 1 - a):.3f}  (interpretable '# bands')")
    print(f"  pattern_vector classifier balacc = {bal(pt['pattern_vector'], ys):.3f}")

    print("\n# D. Rare variant (cheeks) detectors")
    print(f"  novelty scalar          : {bal(recommended.novelty_score(fcd).reshape(-1,1), yc):.3f}")
    print(f"  graph-spectral chromatic: {bal(recommended.rare_variant_descriptor(fcd), yc):.3f}")


if __name__ == "__main__":
    main()
