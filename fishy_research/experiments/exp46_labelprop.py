"""EXP-46 — Semi-supervised label propagation tag model.

Exploits the UNLABELED structure: build one kNN affinity graph over all Z (z-scored PCA coords)
and, for each of the 6 multi-label tags independently, run LabelSpreading (normalized graph
propagation) seeded ONLY by the labeled specimens. The propagated per-tag probability for ALL
250 specimens becomes tag_scores (N,6).

Factor-agnostic: each tag is treated as an independent binary target. We do NOT group the 6 tags
into 3 factors anywhere in the model (the harness does that only at eval time). For a labeled
specimen, tag t is either present (Y=1) or absent (Y=0) — both are valid graph seeds, so each
tag's LabelSpreading is seeded with the labeled specimens' 0/1 value for that tag.

Latent: PCA-2 of the propagated tag_scores — the "classifier latent space" that reflects what the
propagation actually decided per tag. (A spectral / Laplacian-eigenmaps embedding of the same kNN
graph was also tried, but it captures the dominant manifold geometry rather than the subtle
multi-factor tag structure and scores worse on the 8-way silhouette proxy: ~-0.18 vs ~+0.01, so we
return latent=None and let the harness use PCA-2 of tag_scores consistently.)
"""
from __future__ import annotations

import sys
import warnings

import numpy as np
from sklearn.manifold import SpectralEmbedding
from sklearn.metrics import silhouette_score
from sklearn.semi_supervised import LabelSpreading

sys.path.insert(0, "experiments")
import _tag_harness as H  # noqa: E402

warnings.filterwarnings("ignore")

KNN_K = 10        # neighbors for the affinity graph
ALPHA = 0.2       # LabelSpreading clamping (0.2 -> keep 80% of seed influence)
NAME = "labelprop-knn"


def _spectral_latent(Z, k=KNN_K, seed=0):
    """2-D Laplacian eigenmaps of the kNN affinity over all Z. None if it degenerates."""
    try:
        se = SpectralEmbedding(
            n_components=2, affinity="nearest_neighbors", n_neighbors=k, random_state=seed
        )
        L = se.fit_transform(Z)
        if not np.all(np.isfinite(L)) or np.allclose(L.std(0), 0):
            return None
        return L
    except Exception:
        return None


def model_fn(Z, labeled_idx, Ytags_labeled):
    Z = np.asarray(Z, float)
    N, T = Z.shape[0], Ytags_labeled.shape[1]
    labeled_idx = np.asarray(labeled_idx)
    Y = np.asarray(Ytags_labeled, int)

    tag_scores = np.full((N, T), 0.5, float)
    for t in range(T):
        yy = -np.ones(N, dtype=int)          # -1 = unlabeled
        yy[labeled_idx] = Y[:, t]            # seed 0/1 for tagged specimens
        classes = np.unique(yy[yy >= 0])
        if classes.size < 2:
            # only one class seen among the labeled set for this tag -> propagate that constant
            tag_scores[:, t] = float(classes[0]) if classes.size == 1 else 0.5
            continue
        ls = LabelSpreading(kernel="knn", n_neighbors=KNN_K, alpha=ALPHA)
        ls.fit(Z, yy)
        # P(tag present); label_distributions_ columns follow ls.classes_
        col = int(np.where(ls.classes_ == 1)[0][0])
        tag_scores[:, t] = ls.label_distributions_[:, col]

    # latent=None -> harness (and our reporting) use PCA-2 of tag_scores, the better-separated
    # "classifier latent space". _spectral_latent(Z) is available but scores worse (see docstring).
    return {"tag_scores": tag_scores, "latent": None}


def main():
    Z, F, joint, _ = H.load()

    res = H.eval_model(model_fn)
    print(f"== {NAME} ==")
    print("  budget |  acc_mean  acc_stripe  ari8")
    for n, m in res.items():
        print(f"  {n:5d}  |   {m['acc_mean']:.3f}     {m['acc_stripe']:.3f}     {m['ari8']:.3f}")

    # latent quality proxy at n_labeled=80, on the SAME thing viz would plot
    lab80 = H.labeled_set(len(Z), 80, 0)
    out80 = model_fn(Z, lab80, H.tags_for(F)[lab80])
    L = out80.get("latent")
    if L is None:
        from sklearn.decomposition import PCA
        L = PCA(2, random_state=0).fit_transform(np.asarray(out80["tag_scores"], float))
        latent_kind = "PCA-2 of tag_scores (fallback)"
    else:
        latent_kind = "spectral embedding of kNN graph"
    sil = float(silhouette_score(np.asarray(L, float), joint))
    print(f"\n  latent silhouette (n=80, {latent_kind}) vs 8-way joint = {sil:.4f}")

    p1 = H.RESULTS / f"latent_{NAME}_n40.png"
    p2 = H.RESULTS / f"latent_{NAME}_n160.png"
    H.viz(model_fn, 40, p1, f"{NAME} · 40 tagged")
    H.viz(model_fn, 160, p2, f"{NAME} · 160 tagged")
    print(f"  figures -> {p1}\n            {p2}")

    return res, sil, latent_kind, str(p1), str(p2)


if __name__ == "__main__":
    main()
