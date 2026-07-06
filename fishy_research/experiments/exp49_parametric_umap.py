"""EXP-49 — Parametric UMAP for a LIVE single-update tag morphospace (vs AlignedUMAP).

EXP-48 used AlignedUMAP, which re-optimises the whole budget SEQUENCE — great for replay, but it needs all
frames, so it can't place a single new update live. Parametric UMAP learns a reusable encoder f: latent->2D;
you fit it ONCE and then f(updated_latent) places points with no re-fit (deterministic, naturally stable).
This tests the realistic interactive path: fit f at one tag budget, then `.transform()` the latents as more
tags are added — and asks the cost (does a frozen map go stale / look worse than AlignedUMAP?).

Run in the isolated TF env:
  TF_USE_LEGACY_KERAS=1 TF_CPP_MIN_LOG_LEVEL=3 /tmp/pumap_env/bin/python experiments/exp49_parametric_umap.py
(Loads results/exp46/cache.npz directly so it needs no fishpipe/engine.)
"""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score
from umap import UMAP, AlignedUMAP
from umap.parametric_umap import ParametricUMAP

tf.random.set_seed(0); np.random.seed(0)
RES = Path(__file__).resolve().parent.parent / "results" / "exp46"
BUDGETS = [10, 20, 40, 60, 80, 120, 160]
UKW = dict(n_neighbors=20, min_dist=0.12, random_state=0)


def tags_for(F):
    Y = np.zeros((len(F), 6), int)
    for c in range(3):
        Y[np.arange(len(F)), 2 * c + F[:, c]] = 1
    return Y


def tag_scores(Z, lab, Yl):
    S = np.zeros((len(Z), 6), "float32")
    for t in range(6):
        y = Yl[:, t]
        if len(np.unique(y)) < 2:
            S[:, t] = float(y.mean()); continue
        S[:, t] = LogisticRegression(max_iter=2000).fit(Z[lab], y).decision_function(Z)
    return S


def _norm(E):
    E = np.asarray(E, float); E = E - E.mean(0)
    return E / (np.sqrt((E ** 2).sum(1).mean()) + 1e-9)


def motion(frames):
    F = [_norm(e) for e in frames]
    return float(np.mean([np.linalg.norm(F[i + 1] - F[i], axis=1).mean() for i in range(len(F) - 1)]))


def fit_pumap(X):
    pu = ParametricUMAP(n_components=2, verbose=False, random_state=0,
                        n_neighbors=20, min_dist=0.12)
    pu.fit(X.astype("float32"))
    return pu


def main():
    d = np.load(RES / "cache.npz", allow_pickle=True)
    Z, F, joint = d["Z"].astype("float32"), d["F"], d["joint"]
    N = len(Z); Y = tags_for(F)
    order = np.random.default_rng(0).permutation(N)
    labsets = [np.sort(order[:n]) for n in BUDGETS]
    latents = [tag_scores(Z, lab, Y[lab]) for lab in labsets]

    def sils(seq):
        return [round(float(silhouette_score(e, joint)), 3) for e in seq]

    # --- Parametric UMAP: FIT ONCE at a budget, then .transform() every budget's latent (the live path) ---
    seqs = {}
    for b0 in (40, 80):
        pu = fit_pumap(latents[BUDGETS.index(b0)])
        seqs[f"parametric (frozen@{b0})"] = [np.asarray(pu.transform(S)) for S in latents]
    # --- references: AlignedUMAP (whole sequence) + independent UMAP per budget ---
    rel = [{i: i for i in range(N)} for _ in range(len(latents) - 1)]
    seqs["AlignedUMAP"] = [np.asarray(e) for e in AlignedUMAP(**UKW).fit(latents, relations=rel).embeddings_]
    seqs["independent UMAP"] = [np.asarray(UMAP(**UKW).fit_transform(S)) for S in latents]

    print("method                 | motion |  silhouette per budget " + str(BUDGETS))
    rowsil = {}
    for name, seq in seqs.items():
        s = sils(seq); rowsil[name] = s
        print(f"  {name:20s} | {motion(seq):.3f}  | {s}")

    # figure: parametric(frozen@80) small-multiples + silhouette-vs-budget comparison
    colors = [plt.cm.tab10(j) for j in joint]
    pseq = seqs["parametric (frozen@80)"]
    PS = np.concatenate(pseq); xl = (PS[:, 0].min() - 1, PS[:, 0].max() + 1); yl = (PS[:, 1].min() - 1, PS[:, 1].max() + 1)
    fig = plt.figure(figsize=(14, 6.4))
    for k, (n, e, lab) in enumerate(zip(BUDGETS, pseq, labsets)):
        ax = fig.add_subplot(2, 4, k + 1)
        ax.scatter(e[:, 0], e[:, 1], s=14, c=colors, edgecolors="none", alpha=.85)
        ax.scatter(e[lab, 0], e[lab, 1], s=36, facecolors="none", edgecolors="k", linewidths=.7)
        ax.set_title(f"{n} tags · sil={sils([e])[0]}", fontsize=9); ax.set_xlim(*xl); ax.set_ylim(*yl)
        ax.set_xticks([]); ax.set_yticks([])
    axc = fig.add_subplot(2, 4, 8)
    for name, s in rowsil.items():
        axc.plot(BUDGETS, s, "o-", ms=3, label=name)
    axc.set_xlabel("# tags"); axc.set_ylabel("8-way silhouette"); axc.set_ylim(-0.3, 0.7)
    axc.legend(fontsize=6.5); axc.set_title("quality vs budget", fontsize=9)
    fig.suptitle("Parametric UMAP (fit once @80 tags, then .transform as tags are added) — frozen map is "
                 "smooth but cannot re-sharpen", y=1.0)
    fig.tight_layout(); fig.savefig(RES / "parametric_umap.png", dpi=120); plt.close(fig)
    print(f"\nfigure -> {RES/'parametric_umap.png'}")


if __name__ == "__main__":
    main()
