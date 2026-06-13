"""EXP-14 — Projection pursuit by CLUSTERABILITY on a combined general descriptor.

Hypothesis: low-variance structure (stripe, cheeks) is invisible to variance-PCA but IS a
clusterable projection. Compute many ICA components, rank them by a label-free clusterability
score (GMM BIC gap), and check whether each planted factor appears among the TOP-ranked
clusterable axes — with its clusterability RANK reported (so it would be surfaced without
labels). Column-shuffled null guards against fabrication.
"""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import FastICA
from sklearn.metrics import roc_auc_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from fishpipe import config, data, features, spectral

GCACHE = config.CACHE_DIR / "fishy_gabor_La.npy"


def combined_descriptor(fcd):
    blocks = [
        features.area_hist(fcd, 24, "lab"),                       # color composition
        np.load(GCACHE),                                          # Gabor texture (general)
        spectral.spectral_descriptor(fcd, k=300, n_bands=12),     # graph-spectral (general)
    ]
    return np.concatenate([StandardScaler().fit_transform(b) for b in blocks], axis=1)


def clusterability(s):
    """GMM BIC gap: BIC(1)-BIC(2) on a 1-D axis (positive => bimodal/clustered). Label-free."""
    s = StandardScaler().fit_transform(s.reshape(-1, 1))
    return GaussianMixture(1, random_state=0).fit(s).bic(s) - GaussianMixture(2, random_state=0).fit(s).bic(s)


def pp_axes(X, n=20, seed=0):
    Xs = StandardScaler().fit_transform(X)
    ica = FastICA(n_components=min(n, *Xs.shape), random_state=seed, max_iter=3000, tol=1e-3)
    S = ica.fit_transform(Xs)
    cl = np.array([clusterability(S[:, j]) for j in range(S.shape[1])])
    order = np.argsort(cl)[::-1]
    return S[:, order], cl[order]


def main():
    gt = data.load_ground_truth()
    fcd = data.build_face_colors()
    X = combined_descriptor(fcd)
    print(f"combined general descriptor: {X.shape}\n")

    S, cl = pp_axes(X)
    rng = np.random.default_rng(7)
    Xn = np.column_stack([rng.permutation(X[:, j]) for j in range(X.shape[1])])
    Sn, cln = pp_axes(Xn)

    print("clusterability (BIC gap) of ranked axes — real vs null (top 10):")
    print("  real:", np.round(cl[:10], 0))
    print("  null:", np.round(cln[:10], 0))

    print("\nPer factor: best-aligned clusterable axis (rank among ranked axes), AUC real vs null:")
    K = 8
    for f in ["belly", "tail", "stripe", "cheeks"]:
        y = gt.labels[f]
        aucs = np.array([max(roc_auc_score(y, S[:, j]), 1 - roc_auc_score(y, S[:, j])) for j in range(S.shape[1])])
        best = int(np.argmax(aucs))
        aucs_n = np.array([max(roc_auc_score(y, Sn[:, j]), 1 - roc_auc_score(y, Sn[:, j])) for j in range(Sn.shape[1])])
        in_topK = "YES" if best < K else f"no(rank {best})"
        print(f"  {f:7s}: AUC real={aucs[best]:.3f} (axis rank {best}, clusterability={cl[best]:.0f}) "
              f"| best-in-top{K}? {in_topK} | null best-axis AUC={aucs_n.max():.3f}")


if __name__ == "__main__":
    main()
