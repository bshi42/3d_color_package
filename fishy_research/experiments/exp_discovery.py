"""EXP-13 — Can UNSUPERVISED methods surface low-variance subspace structure PCA misses?

Tests structure-discovery methods on the general Gabor descriptor (which CONTAINS the stripe
signal — supervised classifier 0.83 — but not in its top variance directions). Every method is
also run on a COLUMN-SHUFFLED NULL (same marginals, destroyed joint structure) so we can tell
real discovery from fabrication. Honest metric per factor: recovery(real) vs recovery(null).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import kurtosis, skew
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering
from sklearn.decomposition import FastICA
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import MDS
from sklearn.metrics import adjusted_rand_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from fishpipe import config, data

GCACHE = config.CACHE_DIR / "fishy_gabor_La.npy"


def column_shuffle(X, seed=0):
    rng = np.random.default_rng(seed)
    return np.column_stack([rng.permutation(X[:, j]) for j in range(X.shape[1])])


# ---------------- structure-discovery methods (all label-free) ----------------
def urf_affinity(X, n_trees=600, seed=0):
    """Shi-Horvath unsupervised RF: real vs marginal-null, leaf co-occurrence proximity."""
    Xn = column_shuffle(X, seed + 1)
    Xall = np.vstack([X, Xn]); y = np.r_[np.ones(len(X)), np.zeros(len(Xn))]
    rf = RandomForestClassifier(n_estimators=n_trees, random_state=seed, n_jobs=-1).fit(Xall, y)
    leaves = rf.apply(X)
    prox = np.mean(leaves[:, None, :] == leaves[None, :, :], axis=2)
    return prox  # similarity in [0,1]


def consensus_subspace_affinity(X, B=400, m_frac=0.3, krange=(2, 7), seed=0):
    rng = np.random.default_rng(seed); n, p = X.shape
    Xs = StandardScaler().fit_transform(X)
    co = np.zeros((n, n))
    for b in range(B):
        feats = rng.choice(p, max(2, int(m_frac * p)), replace=False)
        k = int(rng.integers(*krange))
        lab = KMeans(k, n_init=3, random_state=b).fit_predict(Xs[:, feats])
        co += (lab[:, None] == lab[None, :])
    return co / B


def projection_pursuit_axes(X, n_axes=10, seed=0):
    """ICA components ranked by clusterability (2-component vs 1 separation = |kurtosis|+bimodality)."""
    Xs = StandardScaler().fit_transform(X)
    ica = FastICA(n_components=min(n_axes, *Xs.shape), random_state=seed, max_iter=2000)
    S = ica.fit_transform(Xs)
    scores = []
    for j in range(S.shape[1]):
        s = S[:, j]; n = len(s)
        bc = (skew(s) ** 2 + 1) / (kurtosis(s) + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3)))
        scores.append(bc)
    order = np.argsort(scores)[::-1]
    return S[:, order], np.array(scores)[order]


# ---------------- evaluation ----------------
def affinity_recovery(aff, gt):
    """Embed affinity (2-D) + 2-cluster it; report best-axis AUC and ARI per factor."""
    d = 1 - (aff - aff.min()) / (aff.max() - aff.min() + 1e-9)
    np.fill_diagonal(d, 0)
    emb = MDS(n_components=3, dissimilarity="precomputed", random_state=0,
              normalized_stress="auto").fit_transform(d)
    out = {}
    for f, y in gt.labels.items():
        auc = max(max(roc_auc_score(y, emb[:, k]), 1 - roc_auc_score(y, emb[:, k])) for k in range(3))
        out[f] = auc
    # 2-way clustering ARI for stripe (and joint)
    cl = AgglomerativeClustering(2).fit_predict(emb)
    out["stripe_ARI"] = adjusted_rand_score(gt.labels["stripe"], cl)
    return out


def axes_recovery(S, gt):
    out = {}
    for f, y in gt.labels.items():
        out[f] = max(max(roc_auc_score(y, S[:, k]), 1 - roc_auc_score(y, S[:, k])) for k in range(S.shape[1]))
    return out


def main():
    gt = data.load_ground_truth()
    X = np.load(GCACHE)  # general Gabor descriptor (250, 36)
    print(f"representation: Gabor descriptor {X.shape} (contains stripe at 0.83 supervised)\n")

    for name, fn in [("URF (Shi-Horvath)", urf_affinity),
                     ("consensus-subspace", consensus_subspace_affinity)]:
        real = affinity_recovery(fn(X), gt)
        null = affinity_recovery(fn(column_shuffle(X, 99)), gt)
        print(f"== {name} ==  (best-axis AUC per factor; chance 0.5)")
        for f in ["belly", "tail", "stripe", "cheeks"]:
            print(f"    {f:7s}: real={real[f]:.3f}  null={null[f]:.3f}  Δ={real[f]-null[f]:+.3f}")
        print(f"    stripe 2-cluster ARI: real={real['stripe_ARI']:.3f}  null={null['stripe_ARI']:.3f}\n")

    S, sc = projection_pursuit_axes(X)
    real = axes_recovery(S, gt)
    Sn, _ = projection_pursuit_axes(column_shuffle(X, 99))
    null = axes_recovery(Sn, gt)
    print("== projection-pursuit (ICA ranked by clusterability) ==  (best-axis AUC)")
    for f in ["belly", "tail", "stripe", "cheeks"]:
        print(f"    {f:7s}: real={real[f]:.3f}  null={null[f]:.3f}  Δ={real[f]-null[f]:+.3f}")


if __name__ == "__main__":
    main()
