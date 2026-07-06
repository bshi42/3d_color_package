"""EXP-47 — Displaying the tag-classifier latent in 2-D: UMAP vs PCA vs t-SNE.

EXP-46: the best placer is a per-tag linear classifier; its 6-D tag-score space is the "classifier latent".
PCA-2 of it is only a linear shadow (8-way silhouette ~0.28). Question: does UMAP lay that latent out in 2-D
much more cleanly for DISPLAY? Compare PCA-2 / UMAP-2 / t-SNE-2 of the tag-score latent at several tag
budgets, plus an unsupervised UMAP of the raw descriptor as a no-feedback reference.

Run:  .venv/bin/python experiments/exp47_umap_latent.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE, trustworthiness
from sklearn.metrics import silhouette_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / ""))
import _tag_harness as H  # noqa: E402
import umap  # noqa: E402

RESULTS = H.RESULTS
BUDGETS = [40, 80, 160]


def tag_scores(Z, lab, Yl):
    """Per-tag linear logistic decision scores for ALL specimens -> the classifier latent (N, 6)."""
    S = np.zeros((len(Z), 6))
    for t in range(6):
        y = Yl[:, t]
        if len(np.unique(y)) < 2:
            S[:, t] = float(y.mean()); continue
        S[:, t] = LogisticRegression(max_iter=2000).fit(Z[lab], y).decision_function(Z)
    return S


def proj(method, X):
    if method == "PCA":
        return PCA(2, random_state=0).fit_transform(X)
    if method == "UMAP":
        return umap.UMAP(n_neighbors=20, min_dist=0.1, random_state=0).fit_transform(X)
    if method == "t-SNE":
        return TSNE(2, perplexity=30, init="pca", random_state=0).fit_transform(X)


def quality(X, E, joint):
    return (round(float(silhouette_score(E, joint)), 3),
            round(float(trustworthiness(X, E, n_neighbors=12)), 3))


def main():
    Z, F, joint, _ = H.load()
    N = len(Z); Y = H.tags_for(F)
    methods = ["PCA", "UMAP", "t-SNE"]
    colors = [plt.cm.tab10(j) for j in joint]

    # unsupervised reference: UMAP of the raw 24-PC descriptor (no feedback)
    E0 = proj("UMAP", Z); s0 = quality(Z, E0, joint)
    print(f"unsupervised UMAP(raw 24-PC descriptor): silhouette={s0[0]} trustworthiness={s0[1]}")

    fig, axes = plt.subplots(len(BUDGETS), len(methods), figsize=(11, 11))
    print("\nclassifier-latent (tag scores) -> 2-D:")
    print("  budget | method |  silhouette  trustworthiness")
    for r, n in enumerate(BUDGETS):
        lab = H.labeled_set(N, n, 0)
        S = tag_scores(Z, lab, Y[lab])
        for c, m in enumerate(methods):
            E = proj(m, S); sil, tw = quality(S, E, joint)
            print(f"  {n:5d}  | {m:6s} |    {sil:.3f}        {tw:.3f}")
            ax = axes[r, c]
            ax.scatter(E[:, 0], E[:, 1], s=18, c=colors, edgecolors="none", alpha=.85)
            ax.scatter(E[lab, 0], E[lab, 1], s=55, facecolors="none", edgecolors="k", linewidths=.9)
            ax.set_title(f"{m} · {n} tags · sil={sil}", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("2-D display of the tag-classifier latent (colour = true 8-way; ringed = tagged)", y=1.0)
    fig.tight_layout(); fig.savefig(RESULTS / "umap_latent_compare.png", dpi=125); plt.close(fig)

    # standalone unsupervised reference figure
    fig2, ax = plt.subplots(figsize=(4.6, 4.4))
    ax.scatter(E0[:, 0], E0[:, 1], s=20, c=colors, edgecolors="none", alpha=.85)
    ax.set_title(f"unsupervised UMAP, no feedback · sil={s0[0]}", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    fig2.tight_layout(); fig2.savefig(RESULTS / "umap_unsupervised.png", dpi=130); plt.close(fig2)
    print(f"\nfigures -> {RESULTS}")


if __name__ == "__main__":
    main()
