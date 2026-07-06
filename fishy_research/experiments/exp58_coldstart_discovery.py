"""EXP-58 — COLD-START discovery of the FIRST cheeks positive via UNSUPERVISED novelty ranking.

QUESTION
  Before ANY cheeks specimen is tagged, can a label-free 'unusualness' score surface the first
  'rosy cheeks' positive faster than random labeling? cheeks is RARE (14/250 = 5.6%) and LOCALIZED.
  At 5.6% base rate, random labeling needs ~1/0.056 ≈ 17.9 labels in expectation to hit the first
  positive. Can any unsupervised scorer beat that?

WHAT WE DO (labels NEVER used as input; GT used ONLY to score the produced ranking)
  Rank all 250 specimens by 'unusualness' under several scorers, on two label-free feature spaces:
    - Z24    : the demo's current PCA-24 space
    - color  : the 5 color channels of the full descriptor, Xs.reshape(250,128,11)[:,:,:5] -> (250,640)
  Scorers (all unsupervised, higher = more unusual):
    IF   IsolationForest        (anomaly = -score_samples)
    LOF  LocalOutlierFactor     (novelty=False, -negative_outlier_factor_)
    kNN  mean distance to k nearest neighbors
    PCA  reconstruction error from a low-rank PCA fit on all 250 (energy left out)
    MAH  Mahalanobis distance to the global mean (shrunk covariance)

  Expert labels in ranked order (most-unusual first). We report, per scorer:
    - E[#labels to FIRST cheeks positive]   = 1 + (rank index of first cheeks, 0-based)
    - E[#labels to 3rd cheeks positive]      = 1 + (rank index of 3rd cheeks)
    - enrichment in top-20 = (cheeks rate among top-20 ranked) / 0.056
  RANDOM baseline: same quantities averaged over >=300 shuffles of the label order (exact, no scorer).

  Plot: cumulative #cheeks found vs #specimens labeled, one curve per scorer + random (mean +/- band).

Run:  .venv/bin/python experiments/exp58_coldstart_discovery.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results" / "exp58"
RES.mkdir(parents=True, exist_ok=True)
CACHE = RES / "cache.npz"

N_SHUFFLE = 2000          # random-order shuffles (>=300 required)
KNN_K = 10
PCA_RANK = 10
LOF_K = 20
N_SEEDS_IF = 10           # average IsolationForest over seeds (it is stochastic)
TARGET = 3                # cheeks column in F


def load():
    d = np.load(CACHE, allow_pickle=True)
    Xs = d["Xs"]                      # (250,1408)
    Z24 = d["Z24"]                    # (250,24)
    F = d["F"]                        # (250,4)
    R, PDIM, CC = int(d["R"]), int(d["PDIM"]), int(d["COLOR_CH"])
    color = Xs.reshape(-1, R, PDIM)[:, :, :CC].reshape(Xs.shape[0], -1)  # (250,640)
    return Z24.astype(np.float64), color.astype(np.float64), F[:, TARGET].astype(int)


# ---- scorers: return per-specimen unusualness (higher = more unusual) ----

def score_if(X, rng):
    n = X.shape[0]
    acc = np.zeros(n)
    for s in range(N_SEEDS_IF):
        m = IsolationForest(n_estimators=300, random_state=int(rng.integers(1 << 30)))
        m.fit(X)
        acc += -m.score_samples(X)      # higher = more anomalous
    return acc / N_SEEDS_IF


def score_lof(X, _rng):
    k = min(LOF_K, X.shape[0] - 1)
    m = LocalOutlierFactor(n_neighbors=k)
    m.fit_predict(X)
    return -m.negative_outlier_factor_  # higher = more outlying


def score_knn(X, _rng):
    k = min(KNN_K, X.shape[0] - 1)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    dist, _ = nn.kneighbors(X)
    return dist[:, 1:].mean(axis=1)     # drop self (col 0)


def score_pca(X, _rng):
    r = min(PCA_RANK, X.shape[1], X.shape[0] - 1)
    p = PCA(n_components=r).fit(X)
    Xr = p.inverse_transform(p.transform(X))
    return ((X - Xr) ** 2).sum(axis=1)  # reconstruction energy


def score_mahalanobis(X, _rng):
    cov = LedoitWolf().fit(X)
    return cov.mahalanobis(X)           # squared Mahalanobis distance


SCORERS = {
    "IsolationForest": score_if,
    "LOF": score_lof,
    "kNN-dist": score_knn,
    "PCA-recon": score_pca,
    "Mahalanobis": score_mahalanobis,
}


def order_metrics(order, y):
    """Given a label order (array of specimen indices, most-unusual first), compute discovery metrics."""
    pos_ranks = np.where(y[order] == 1)[0]   # 0-based positions of cheeks in this order
    n_pos = len(pos_ranks)
    first = int(pos_ranks[0]) + 1 if n_pos >= 1 else np.nan
    third = int(pos_ranks[2]) + 1 if n_pos >= 3 else np.nan
    top20 = int(y[order[:20]].sum())
    cum = np.cumsum(y[order])               # cumulative cheeks found vs #labeled
    return first, third, top20, cum


def main():
    rng = np.random.default_rng(0)
    Z, color, y = load()
    n = len(y)
    n_pos = int(y.sum())
    base = y.mean()
    print(f"n={n}  cheeks={n_pos}  base rate={base:.4f}  random E[first]≈{1/base:.1f}")

    spaces = {"Z24": StandardScaler().fit_transform(Z),
              "color640": StandardScaler().fit_transform(color)}

    rows = []          # (label, first, third, top20, enrichment)
    curves = {}        # label -> cumulative array

    for sname, X in spaces.items():
        for scname, fn in SCORERS.items():
            s = fn(X, rng)
            order = np.argsort(-s)          # most unusual first; ties broken by index (stable)
            first, third, top20, cum = order_metrics(order, y)
            enr = (top20 / 20) / base
            label = f"{scname}/{sname}"
            rows.append((label, first, third, top20, enr))
            curves[label] = cum
            print(f"  {label:24s} first={first!s:>4}  3rd={third!s:>5}  "
                  f"top20={top20}  enrichment={enr:.2f}x")

    # ---- random baseline over many shuffles ----
    firsts, thirds, top20s = [], [], []
    cum_rand = np.zeros((N_SHUFFLE, n))
    for i in range(N_SHUFFLE):
        order = rng.permutation(n)
        f, t, t20, cum = order_metrics(order, y)
        firsts.append(f); thirds.append(t); top20s.append(t20)
        cum_rand[i] = cum
    rand_first = np.nanmean(firsts)
    rand_third = np.nanmean(thirds)
    rand_top20 = np.mean(top20s)
    rand_enr = (rand_top20 / 20) / base
    rand_mean = cum_rand.mean(axis=0)
    rand_lo = np.percentile(cum_rand, 5, axis=0)
    rand_hi = np.percentile(cum_rand, 95, axis=0)
    print(f"  RANDOM (n={N_SHUFFLE} shuffles)     first={rand_first:.1f}  "
          f"3rd={rand_third:.1f}  top20={rand_top20:.2f}  enrichment={rand_enr:.2f}x")

    # ---- figure ----
    x = np.arange(1, n + 1)
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(15, 6))

    # left: full curves
    axL.fill_between(x, rand_lo, rand_hi, color="0.8", alpha=0.6,
                     label="random 5-95%")
    axL.plot(x, rand_mean, color="0.35", lw=2.2, ls="--", label="random mean")
    axL.plot(x, np.minimum(x, n_pos), color="k", lw=1.0, ls=":", label="perfect (all cheeks first)")
    cmap = plt.cm.tab10(np.linspace(0, 1, len(curves)))
    for (label, cum), c in zip(curves.items(), cmap):
        axL.plot(x, cum, lw=1.6, color=c, label=label)
    axL.axhline(n_pos, color="0.5", lw=0.8)
    axL.set_xlabel("# specimens labeled (in ranked order)")
    axL.set_ylabel("cumulative cheeks positives found")
    axL.set_title(f"Cold-start cheeks discovery (n={n}, {n_pos} positives, base {base*100:.1f}%)")
    axL.legend(fontsize=7, loc="lower right", ncol=2)
    axL.grid(alpha=0.3)

    # right: zoom on first 40 labels (the cold-start budget)
    axR.fill_between(x, rand_lo, rand_hi, color="0.8", alpha=0.6)
    axR.plot(x, rand_mean, color="0.35", lw=2.2, ls="--", label="random mean")
    for (label, cum), c in zip(curves.items(), cmap):
        axR.plot(x, cum, lw=1.8, color=c, marker="o", ms=3, label=label)
    axR.axvline(rand_first, color="0.35", ls=":", lw=1)
    axR.set_xlim(0.5, 40.5)
    ymax = max(c[:40].max() for c in curves.values())
    axR.set_ylim(-0.3, ymax + 0.5)
    axR.set_xlabel("# specimens labeled (in ranked order)")
    axR.set_ylabel("cumulative cheeks positives found")
    axR.set_title("Zoom: first 40 labels (cold-start budget)")
    axR.legend(fontsize=7, loc="upper left", ncol=2)
    axR.grid(alpha=0.3)

    fig.tight_layout()
    out = RES / "coldstart_discovery.png"
    fig.savefig(out, dpi=130)
    print("saved", out)

    # ---- summary table sorted by E[first] ----
    rows.append(("RANDOM", rand_first, rand_third, rand_top20, rand_enr))
    rows.sort(key=lambda r: (np.inf if r[1] != r[1] else r[1]))
    print("\n=== ranked by E[#labels to FIRST cheeks] ===")
    print(f"{'scorer/space':26s} {'first':>6} {'3rd':>7} {'top20':>6} {'enrich':>7}")
    for label, first, third, t20, enr in rows:
        fs = f"{first:.1f}" if isinstance(first, float) else str(first)
        ts = f"{third:.1f}" if isinstance(third, float) else str(third)
        print(f"{label:26s} {fs:>6} {ts:>7} {t20:>6.1f} {enr:>6.2f}x")


if __name__ == "__main__":
    main()
