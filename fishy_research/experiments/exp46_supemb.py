"""EXP-46 — SUPERVISED 2-D EMBEDDING for the cleanest visualisable latent.

Idea: each labeled specimen gets an 'applied-tag class' = the joint (tuple) of its applied tags.
Fit a 2-D supervised projection (NCA, fallback LDA) on the LABELED specimens only, transform ALL Z
to a 2-D latent. tag_scores for ALL specimens come from nearest labeled prototype in that 2-D latent
(per-class prototype, then for each tag aggregate the soft membership of classes that carry the tag).

Stays factor-agnostic: the 'applied-tag class' is just the bit-pattern of the 6 multi-label tags; we never
group the 6 tags into the 3 factors and never touch unlabeled true labels.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "experiments")
import _tag_harness as H  # noqa: E402

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import NearestCentroid

try:
    from sklearn.neighbors import NeighborhoodComponentsAnalysis
    _HAVE_NCA = True
except Exception:
    _HAVE_NCA = False

RESULTS = Path("/home/alek/projects/3d_color_package/fishy_research/results/exp46")
RESULTS.mkdir(parents=True, exist_ok=True)
T = 6  # number of tags


def _supervised_2d(Z, lab, Ylab):
    """Return (latent_all (N,2), classes list-of-6bit-tuples, class_of_lab (n,))."""
    Xl = Z[lab]
    # 'applied-tag class' = bit pattern of the 6 tags for each labeled specimen
    keys = [tuple(int(v) for v in row) for row in Ylab]
    uniq = sorted(set(keys))
    key2id = {k: i for i, k in enumerate(uniq)}
    y = np.array([key2id[k] for k in keys], int)
    n_classes = len(uniq)

    latent = None
    # NCA needs >=2 classes and (ideally) >= n_components+1 samples per ... try, else LDA, else PCA-2.
    if n_classes >= 2:
        if _HAVE_NCA:
            try:
                nca = NeighborhoodComponentsAnalysis(
                    n_components=2, init="pca", max_iter=200, random_state=0)
                nca.fit(Xl, y)
                latent = nca.transform(Z)
            except Exception:
                latent = None
        if latent is None:
            try:
                nc = min(2, n_classes - 1)
                lda = LinearDiscriminantAnalysis(n_components=nc)
                lda.fit(Xl, y)
                Lt = lda.transform(Z)
                if Lt.shape[1] == 1:  # only 2 classes -> pad with a PCA axis for a usable 2-D view
                    from sklearn.decomposition import PCA
                    p2 = PCA(2, random_state=0).fit(Xl).transform(Z)
                    Lt = np.column_stack([Lt[:, 0], p2[:, 1]])
                latent = Lt
            except Exception:
                latent = None
    if latent is None:
        from sklearn.decomposition import PCA
        latent = PCA(2, random_state=0).fit(Xl).transform(Z)

    return np.asarray(latent, float), uniq, y


def model_fn(Z, lab, Ylab):
    N = len(Z)
    latent, classes, y = _supervised_2d(Z, lab, Ylab)

    # nearest labeled-class prototype in the 2-D latent -> hard class assignment for ALL specimens
    nc = NearestCentroid()
    nc.fit(latent[lab], y)
    pred_class = nc.predict(latent)  # (N,) class id per specimen

    # class -> its 6-bit tag pattern; tag_scores = the tag pattern of the assigned class.
    # Make scores soft via distance to each class centroid so within-factor argmax is meaningful.
    cents = nc.centroids_  # (n_classes, 2)
    # soft membership: inverse-distance softmax over classes
    d2 = ((latent[:, None, :] - cents[None, :, :]) ** 2).sum(-1)  # (N, n_classes)
    # temperature: median pairwise scale of labeled latent
    scale = np.median(np.sqrt(d2[lab].min(1) + 1e-9)) + 1e-6
    W = np.exp(-d2 / (2 * scale * scale))
    W = W / (W.sum(1, keepdims=True) + 1e-12)  # (N, n_classes)

    # tag membership matrix: (n_classes, 6) from the class bit-patterns
    M = np.array(classes, float)  # (n_classes, 6)
    tag_scores = W @ M  # (N, 6) soft per-tag scores

    # ensure labeled specimens reflect their own applied tags exactly (no leakage of unlabeled truth)
    tag_scores[lab] = Ylab.astype(float)

    _ = pred_class
    return {"tag_scores": tag_scores, "latent": latent}


def main():
    from sklearn.metrics import silhouette_score
    Z, F, joint, names = H.load()

    res = H.eval_model(model_fn)
    print("== supemb (NCA/LDA 2-D supervised embedding) ==")
    print("  budget |  acc_mean  acc_stripe  ari8")
    for n, m in res.items():
        print(f"  {n:5d}  |   {m['acc_mean']:.3f}     {m['acc_stripe']:.3f}     {m['ari8']:.3f}")

    # latent silhouette at n_labeled=80 (build latent for ALL, silhouette vs joint)
    lab80 = H.labeled_set(len(Z), 80, 0)
    out80 = model_fn(Z, lab80, H.tags_for(F)[lab80])
    L80 = out80["latent"]
    sil = float(silhouette_score(L80, joint))
    print(f"  silhouette(latent n80, joint) = {sil:.3f}")

    p1 = RESULTS / "latent_supemb_n40.png"
    p2 = RESULTS / "latent_supemb_n160.png"
    H.viz(model_fn, 40, p1, "supemb · 40 tagged")
    H.viz(model_fn, 160, p2, "supemb · 160 tagged")
    print(f"  figs: {p1}  {p2}")
    print(f"  NCA available: {_HAVE_NCA}")
    return res, sil


if __name__ == "__main__":
    main()
