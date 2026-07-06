"""EXP-45 — Hardening the recovery recipe: label-noise robustness + a clean factor grid.

Recipe (EXP-42..44): K=24 PCs, graded target distances (=Hamming over the 3 GT factors), rank-4 metric
learning, GMM(8); active within/across-cluster sampling. Here:
  (a) NOISE ROBUSTNESS — the real expert is not a perfect match-counter. Model each of the 3 per-feature
      same/different judgments as independently wrong with rate q, sweep q, and see how recovery degrades
      and whether more labels buy it back.
  (b) CLEAN FACTOR GRID — lay the 8 recovered clusters on a balanced 2x2x2 grid (recursive balanced
      bisection of the cluster centroids) so the "4 colour quadrants, each split by the weak factor" reads
      cleanly, validated by colouring points with the TRUE label.

Run:  .venv/bin/python experiments/exp45_harden.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "interactive_demo"))
import engine  # noqa: E402

RESULTS = HERE.parent / "results" / "exp45"
RESULTS.mkdir(parents=True, exist_ok=True)
K_PCS, RANK = 24, 4
zc = engine._zscore_cols


def lowrank_metric(Z, pairs, t, r=RANK, iters=600, lr=0.05, lam=1e-3, seed=0):
    rng = np.random.default_rng(seed); K = Z.shape[1]; L = rng.standard_normal((r, K)) * 0.3
    Dz = np.array([Z[i] - Z[j] for (i, j) in pairs]); t = np.asarray(t, float)
    mL = vL = 0.0; b1, b2, eps = 0.9, 0.999, 1e-8
    for it in range(1, iters + 1):
        Y = Dz @ L.T; e = (Y ** 2).sum(1) - t
        g = 4 * ((e[:, None] * Y).T @ Dz) / len(pairs) + 2 * lam * L
        mL = b1 * mL + (1 - b1) * g; vL = b2 * vL + (1 - b2) * g * g
        L = L - lr * (mL / (1 - b1 ** it)) / (np.sqrt(vL / (1 - b2 ** it)) + eps)
    return L


def noisy_targets(F, pairs, q, rng):
    """Expert judges each of the 3 per-feature same/different comparisons, wrong w.p. q (independent)."""
    out = []
    for (i, j) in pairs:
        diff = (F[i] != F[j]).astype(int)
        flip = rng.random(3) < q
        out.append(int(np.logical_xor(diff, flip).sum()))
    return out


def rand_pairs(N, n, rng):
    seen = set(); out = []
    while len(out) < n:
        i, j = rng.integers(N, size=2)
        if i != j and (i, j) not in seen and (j, i) not in seen:
            out.append((int(i), int(j))); seen.add((i, j))
    return out


def recover(Z, F, joint, pairs, q, rng):
    t = noisy_targets(F, pairs, q, rng)
    Y = Z @ lowrank_metric(Z, pairs, t).T
    pred = GaussianMixture(8, n_init=3, random_state=0).fit_predict(Y)
    return Y, pred, adjusted_rand_score(joint, pred)


# --------------------------------------------------------------------------- (a) noise robustness
def noise_sweep(Z, F, joint):
    N = len(joint); qs = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3]; reps = 6
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    print("  per-feature error rate q -> ARI (mean of 6):")
    for budget in (200, 400):
        ys = []
        for q in qs:
            a = []
            for d in range(reps):
                rng = np.random.default_rng(1000 * d + 1)
                a.append(recover(Z, F, joint, rand_pairs(N, budget, rng), q, rng)[2])
            ys.append(np.mean(a))
        print(f"    {budget} labels: " + "  ".join(f"q={q:.2f}:{y:.3f}" for q, y in zip(qs, ys)))
        ax.plot(qs, ys, "o-", label=f"{budget} labels")
    ax.set_xlabel("per-feature judgment error rate q"); ax.set_ylabel("8-cluster recovery (ARI)")
    ax.set_title("Robustness to noisy expert labels"); ax.set_ylim(0, 1); ax.legend()
    fig.tight_layout(); fig.savefig(RESULTS / "noise_robustness.png", dpi=130); plt.close(fig)


# --------------------------------------------------------------------------- (b) clean factor grid
def balanced_code(cen):
    """3-bit code per cluster via recursive BALANCED median bisection on the top local PC -> clean 2x2x2."""
    code = np.zeros((len(cen), 3), int)

    def split(idx, level):
        if level == 3 or len(idx) <= 1:
            return
        X = cen[idx]; Xc = X - X.mean(0)
        u = np.linalg.svd(Xc, full_matrices=False)[2][0]
        order = np.argsort(Xc @ u); hi = set(np.asarray(idx)[order[len(idx) // 2:]])
        lo_idx, hi_idx = [], []
        for k in idx:
            if k in hi:
                code[k, level] = 1; hi_idx.append(k)
            else:
                lo_idx.append(k)
        split(lo_idx, level + 1); split(hi_idx, level + 1)

    split(list(range(len(cen))), 0)
    return code


def factor_grid(Z, F, joint, q=0.1, n=400):
    rng = np.random.default_rng(7)
    Y, pred, ari = recover(Z, F, joint, rand_pairs(len(joint), n, rng), q, rng)
    cen = np.array([Y[pred == c].mean(0) if (pred == c).any() else Y.mean(0) for c in range(8)])
    code = balanced_code(cen)
    col = code[:, 0] * 2 + code[:, 1]            # 4 quadrants (the two strong factors)
    row = code[:, 2]                              # the weak factor -> top/bottom within a quadrant

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    cmap = plt.cm.tab10
    for c in range(8):
        gx, gy = col[c] % 2, col[c] // 2
        cx = gx * 3.2; cy = gy * 3.2 + (row[c] - 0.5) * 1.25
        pts = np.where(pred == c)[0]
        jit = rng.normal(0, 0.17, (len(pts), 2))
        axes[0].scatter(cx + jit[:, 0], cy + jit[:, 1], s=26,
                        c=[cmap(joint[p]) for p in pts], edgecolors="k", linewidths=0.3)
    for gx in (0, 1):
        for gy in (0, 1):
            axes[0].axvline(gx * 3.2 + 1.6, color="#ddd", lw=.8) if gx == 0 else None
    axes[0].axvline(1.6, color="#ccc", lw=1)
    axes[0].set_title(f"Recovered FACTOR GRID  (q={q}, {n} labels, ARI={ari:.2f})\n"
                      "position = recovered cluster · colour = TRUE 8-way label")
    axes[0].set_xticks([]); axes[0].set_yticks([])
    axes[0].set_xlabel("2 strong (colour) factors → 4 quadrants  ·  weak factor → top/bottom")

    M = np.zeros((8, 8), int)
    for p in range(len(joint)):
        M[pred[p], joint[p]] += 1
    # row-permute the confusion matrix to its best diagonal for readability
    from scipy.optimize import linear_sum_assignment
    ri, ci = linear_sum_assignment(-M)
    perm = np.argsort(ci)
    axes[1].imshow(M[ri][:, :], cmap="Greys")
    axes[1].set_title("recovered cluster vs true 8-way\n(rows ordered to diagonal)")
    axes[1].set_xlabel("true joint label"); axes[1].set_ylabel("recovered cluster")
    Mr = M[ri]
    for i in range(8):
        for j in range(8):
            if Mr[i, j]:
                axes[1].text(j, i, Mr[i, j], ha="center", va="center",
                             color="white" if Mr[i, j] > Mr.max() / 2 else "black", fontsize=8)
    fig.tight_layout(); fig.savefig(RESULTS / "factor_grid.png", dpi=130); plt.close(fig)
    purity = M.max(0).sum() / M.sum()
    print(f"  clean factor grid at q={q}: ARI={ari:.3f}  cluster purity={purity:.3f}")


def main():
    sess = engine.Session("fishy"); ld = sess.ld
    F = np.stack([ld.gt.labels[f] for f in ("belly", "tail", "stripe")], 1)
    joint = F[:, 0] * 4 + F[:, 1] * 2 + F[:, 2]
    Z = zc(PCA(K_PCS, random_state=0).fit_transform(StandardScaler().fit_transform(ld.Xs)))
    print(f"recipe: K={K_PCS} PCs, rank-{RANK} metric, GMM-8\n\n(a) noise robustness:")
    noise_sweep(Z, F, joint)
    print("\n(b) clean factor grid:")
    factor_grid(Z, F, joint)
    print(f"\nfigures -> {RESULTS}")


if __name__ == "__main__":
    main()
