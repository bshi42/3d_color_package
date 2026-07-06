"""EXP-46 — Nearest-Class-Mean / PROTOTYPE multi-label tag model.

For each of the 6 tags independently:
  prototype_t = mean of Z over labeled specimens carrying tag t.
  tag_scores[i, t] = -||Z[i] - prototype_t||^2   (negative squared euclidean distance)

A tag with 0 labeled positives gets a constant large negative score (no prototype),
so its within-factor sibling wins by default — handled gracefully.

Latent: project Z onto the top-2 PCs of the 6 prototype vectors (factor-agnostic).
If <2 prototypes exist, fall back to PCA-2 of the per-specimen 6 tag_scores.

Factor-agnostic: the 6 tags are treated as independent multi-label targets; the
3-factor grouping is never used inside the model.
"""
import sys

import numpy as np
from sklearn.decomposition import PCA

sys.path.insert(0, "experiments")
import _tag_harness as H

T = 6
NEG = -1e9  # score for a tag with no labeled positives (loses every comparison)


def model_fn(Z, labeled_idx, Ytags_labeled):
    Z = np.asarray(Z, float)
    N, K = Z.shape
    Yl = np.asarray(Ytags_labeled, int)
    Zl = Z[labeled_idx]

    protos = np.zeros((T, K))
    has_proto = np.zeros(T, bool)
    for t in range(T):
        m = Yl[:, t] == 1
        if m.any():
            protos[t] = Zl[m].mean(0)
            has_proto[t] = True

    # tag_scores[i,t] = -||Z[i]-proto_t||^2 ; absent prototype -> NEG
    scores = np.full((N, T), NEG, float)
    for t in range(T):
        if has_proto[t]:
            d2 = ((Z - protos[t]) ** 2).sum(1)
            scores[:, t] = -d2

    # Latent: top-2 PCs of the prototype vectors, project all Z onto them.
    latent = None
    P = protos[has_proto]
    if P.shape[0] >= 2:
        pca = PCA(2, random_state=0).fit(P)
        latent = pca.transform(Z)
    else:
        latent = PCA(2, random_state=0).fit_transform(scores)

    return {"tag_scores": scores, "latent": latent}


if __name__ == "__main__":
    from sklearn.metrics import silhouette_score

    res = H.eval_model(model_fn)
    print("== prototype (NCM) ==")
    print("  budget |  acc_mean  acc_belly  acc_tail  acc_stripe  ari8")
    for n, m in res.items():
        print(f"  {n:5d}  |   {m['acc_mean']:.3f}    {m['acc_belly']:.3f}    "
              f"{m['acc_tail']:.3f}    {m['acc_stripe']:.3f}     {m['ari8']:.3f}")

    # Silhouette of the latent at n_labeled=80, on the SAME thing viz would plot.
    Z, F, joint, _ = H.load()
    N = len(Z)
    Y = H.tags_for(F)
    lab = H.labeled_set(N, 80, 0)
    out = model_fn(Z, lab, Y[lab])
    L = out.get("latent")
    if L is None:
        L = PCA(2, random_state=0).fit_transform(np.asarray(out["tag_scores"], float))
    L = np.asarray(L, float)
    sil = silhouette_score(L, joint)
    print(f"\nlatent silhouette (n_labeled=80, 8-way joint): {sil:.4f}")

    p1 = str(H.RESULTS / "latent_prototype_n40.png")
    p2 = str(H.RESULTS / "latent_prototype_n160.png")
    H.viz(model_fn, 40, p1, "prototype (NCM) · 40 tagged")
    H.viz(model_fn, 160, p2, "prototype (NCM) · 160 tagged")
    print(f"figures:\n  {p1}\n  {p2}")
