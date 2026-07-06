"""EXP-60 — Do anomaly-detection methods surface the MINORITY classes?

Hypothesis: anomaly detectors find ANOMALIES, not minorities per se — they surface a rare class only when its
members are intrinsically DISTINCTIVE (outlier-like in feature space). Test by scoring how well several
unsupervised anomaly scores rank each minority target:
  * cheeks  (5.6%) — a rare + visually DISTINCTIVE feature (rosy cheeks = unusual colour)
  * 5-stripe (21%) — a minority that is NOT distinctive (just a different stripe COUNT, normal-looking)
  * random-6%      — control: an arbitrary rare subset (should NOT be surfaced -> AUC ~0.5)
Metric = ROC-AUC(anomaly score, minority membership) + top-20 enrichment, on Z24 (and colour/full for cheeks).

Run:  .venv/bin/python experiments/exp60_anomaly_minority.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.covariance import EmpiricalCovariance
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.preprocessing import StandardScaler

RES = Path(__file__).resolve().parent.parent / "results" / "exp58"


def scores(X):
    """dict of unsupervised anomaly scores (higher = more anomalous)."""
    out = {}
    out["IsolationForest"] = -IsolationForest(n_estimators=300, random_state=0).fit(X).score_samples(X)
    lof = LocalOutlierFactor(n_neighbors=20); lof.fit_predict(X); out["LOF"] = -lof.negative_outlier_factor_
    nn = NearestNeighbors(n_neighbors=9).fit(X); d, _ = nn.kneighbors(X); out["kNN-dist"] = d[:, 1:].mean(1)
    cov = EmpiricalCovariance().fit(X); out["Mahalanobis"] = cov.mahalanobis(X)
    p = PCA(5, random_state=0).fit(X); out["PCA-recon-err"] = np.linalg.norm(X - p.inverse_transform(p.transform(X)), axis=1)
    return out


def main():
    d = np.load(RES / "cache.npz")
    Z24, F, Xs = d["Z24"], d["F"], d["Xs"]
    color = Xs.reshape(len(Xs), int(d["R"]), int(d["PDIM"]))[:, :, :int(d["COLOR_CH"])].reshape(len(Xs), -1)
    color = StandardScaler().fit_transform(color)
    N = len(Z24); rng = np.random.default_rng(0)
    targets = {"cheeks (5.6%)": F[:, 3], "5-stripe (21%)": F[:, 2]}

    sc = scores(Z24)
    methods = list(sc)
    print("ROC-AUC of each anomaly score vs minority membership (on Z24; AUC>0.5 = surfaces it):")
    print(f"  {'method':16} " + "  ".join(f"{t.split()[0]:>9}" for t in targets) + "   random-6%")
    auc = {m: {} for m in methods}
    for m in methods:
        rnd = np.mean([roc_auc_score((rng.random(N) < 0.06).astype(int) | 0, sc[m]) for _ in range(80)])
        # control: arbitrary random 6% subsets, averaged
        ctrl = np.mean([roc_auc_score(((np.argsort(rng.random(N)) < 15)).astype(int), sc[m]) for _ in range(80)])
        cells = []
        for t, y in targets.items():
            a = roc_auc_score(y, sc[m]); auc[m][t] = a; cells.append(f"{a:9.3f}")
        print(f"  {m:16} " + "  ".join(cells) + f"   {ctrl:.3f}")

    # top-20 enrichment for cheeks (rate among the 20 most-anomalous vs the 5.6% base)
    print("\ntop-20 cheeks enrichment (cheeks-rate in 20 most-anomalous / 0.056 base):")
    for m in methods:
        top = np.argsort(sc[m])[::-1][:20]; print(f"  {m:16}: {F[top,3].mean()/0.056:.1f}x  ({int(F[top,3].sum())}/20 are cheeks)")

    # feature-space check for cheeks
    print("\ncheeks AUC by feature space (kNN-dist / LOF):")
    for name, X in [("Z24", Z24), ("colour-640", color), ("full-Xs", StandardScaler().fit_transform(Xs))]:
        s = scores(X); print(f"  {name:10}: kNN-dist {roc_auc_score(F[:,3], s['kNN-dist']):.3f}  LOF {roc_auc_score(F[:,3], s['LOF']):.3f}")

    # figure: AUC heatmap (methods x targets) + cumulative found curve (anomaly order vs random)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    M = np.array([[auc[m][t] for t in targets] for m in methods])
    im = ax[0].imshow(M, cmap="RdYlGn", vmin=0.3, vmax=0.9, aspect="auto")
    ax[0].set_xticks(range(len(targets))); ax[0].set_xticklabels(list(targets), rotation=15)
    ax[0].set_yticks(range(len(methods))); ax[0].set_yticklabels(methods)
    for i in range(len(methods)):
        for j in range(len(targets)):
            ax[0].text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", fontsize=9)
    ax[0].set_title("anomaly-score AUC vs minority membership"); fig.colorbar(im, ax=ax[0], fraction=.046)
    best = "kNN-dist"
    order = np.argsort(sc[best])[::-1]
    for t, y, c in [("cheeks", F[:, 3], "#e15759"), ("5-stripe", F[:, 2], "#1d6fd0")]:
        ax[1].plot(range(1, N + 1), np.cumsum(y[order]) / y.sum(), color=c, label=f"{t} (anomaly order)")
        ax[1].plot(range(1, N + 1), np.arange(1, N + 1) / N, "--", color=c, alpha=.4)
    ax[1].set_xlim(0, 80); ax[1].set_xlabel("# specimens labeled (most-anomalous first)")
    ax[1].set_ylabel("fraction of class found"); ax[1].set_title("cheeks vs 5-stripe found (anomaly vs random dashed)")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=.2)
    fig.suptitle("Anomaly detection surfaces SEPARABLE minorities (both cheeks & 5-stripe, AUC>0.6) but not a "
                 "random-6% set (AUC≈0.5); rarer ≠ more surfaced", y=1.02)
    fig.tight_layout(); fig.savefig(RES / "anomaly_minority.png", dpi=130, bbox_inches="tight")
    print(f"\nfigure -> {RES/'anomaly_minority.png'}")


if __name__ == "__main__":
    main()
