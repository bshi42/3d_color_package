"""EXP-44 — The winning recipe + the 8-cluster visualisation ("factor grid").

From EXP-42/43 the recipe that recovers the 2^3 fishy structure from SIMULATED graded feedback:
  * keep MORE PCA dims (K=24; the demo's K=10 throws stripe away: 0.99 -> 0.84 separable),
  * graded target distances (target d^2 = Hamming over the 3 factors),
  * a LOW-RANK metric (rank 4; diagonal too weak, full Mahalanobis overfits at high K),
  * cluster with GMM(8).

This script (a) checks label efficiency for random vs a realistic active schedule that uses only the
CURRENT learned clusters (no GT), and (b) renders the recovered structure as a FACTOR GRID — 4 colour
quadrants each split into 2 by the recovered (weak) factor — which is how the demo could *show* 8 even
though a 2-D spring layout can't settle into 8.

Run:  .venv/bin/python experiments/exp44_recover_grid.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "interactive_demo"))
import engine  # noqa: E402

RESULTS = HERE.parent / "results" / "exp44"
RESULTS.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(0)
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


def oracle_t(F, pairs):
    return [int(np.abs(F[i] - F[j]).sum()) for (i, j) in pairs]


def rand_pairs(N, n, exclude=()):
    seen = set(exclude); out = []
    while len(out) < n:
        i, j = RNG.integers(N, size=2)
        if i != j and (i, j) not in seen and (j, i) not in seen:
            out.append((int(i), int(j))); seen.add((i, j))
    return out


def active_pairs(N, n, Y, labeled):
    """Realistic active step (no GT): half pairs WITHIN the current 8 clusters (refine fuzzy splits),
    half ACROSS clusters (keep the coarse structure), drawn from the current metric space Y."""
    lab = GaussianMixture(8, n_init=2, random_state=0).fit_predict(Y)
    seen = set(labeled) | {(j, i) for (i, j) in labeled}; out = []
    tries = 0
    while len(out) < n and tries < n * 400:
        i, j = RNG.integers(N, size=2); tries += 1
        if i == j or (i, j) in seen:
            continue
        want_within = len(out) % 2 == 0
        if (lab[i] == lab[j]) == want_within:
            out.append((int(i), int(j))); seen.add((i, j))
    return out


def recover(Z, F, joint, pairs):
    L = lowrank_metric(Z, pairs, oracle_t(F, pairs))
    Y = Z @ L.T
    pred = GaussianMixture(8, n_init=3, random_state=0).fit_predict(Y)
    return Y, pred, adjusted_rand_score(joint, pred)


# --------------------------------------------------------------------------- (a) sampling efficiency
def sampling_curve(Z, F, joint):
    N = len(joint); budgets = [50, 100, 150, 200, 300, 400]; reps = 6
    res = {"random": [], "active": []}
    for n in budgets:
        ar = []; aa = []
        for d in range(reps):
            ar.append(recover(Z, F, joint, rand_pairs(N, n))[2])
            # active: seed with 40 random, then top up actively in 2 rounds
            P = rand_pairs(N, min(40, n))
            while len(P) < n:
                Y = Z @ lowrank_metric(Z, P, oracle_t(F, P), seed=d).T
                P += active_pairs(N, min(40, n - len(P)), Y, P)
            aa.append(recover(Z, F, joint, P[:n])[2])
        res["random"].append(np.mean(ar)); res["active"].append(np.mean(aa))
        print(f"  n={n:3d}: random ARI={np.mean(ar):.3f}   active ARI={np.mean(aa):.3f}")
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    ax.plot(budgets, res["random"], "o-", label="random pairs")
    ax.plot(budgets, res["active"], "s-", label="active (within/across current clusters)")
    ax.set_xlabel("# simulated graded labels"); ax.set_ylabel("8-cluster recovery (ARI)")
    ax.set_title(f"Recovery vs label budget  (K={K_PCS} PCs, rank-{RANK} metric, GMM-8)")
    ax.set_ylim(0, 1); ax.legend(); fig.tight_layout()
    fig.savefig(RESULTS / "sampling_curve.png", dpi=130); plt.close(fig)


# --------------------------------------------------------------------------- (b) factor grid
def factor_grid(Z, F, joint, n=400):
    """Recover 8 clusters, then arrange them as 4 colour quadrants x 2 (the weak factor) and scatter
    members. Colour points by TRUE joint label to show the grid recovers the planted structure."""
    Y, pred, ari = recover(Z, F, joint, rand_pairs(len(joint), n))
    cen = np.array([Y[pred == c].mean(0) for c in range(8)])
    # hierarchy of the 8 cluster centroids: 4 coarse groups (colour-like), each split into 2 (stripe-like)
    coarse = AgglomerativeClustering(4, linkage="ward").fit_predict(cen)
    quad = {g: (i % 2, i // 2) for i, g in enumerate(sorted(set(coarse)))}   # 2x2 placement
    # within each coarse group, order its (usually 2) clusters along their largest spread -> top/bottom
    sub = {}
    for g in set(coarse):
        cs = [c for c in range(8) if coarse[c] == g]
        order = sorted(cs, key=lambda c: cen[c] @ (cen[cs[0]] - cen[cs[-1]] + 1e-9))
        for r, c in enumerate(order):
            sub[c] = r

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    cmap = plt.cm.tab10
    for c in range(8):
        gx, gy = quad[coarse[c]]; sy = sub[c]
        cx, cy = gx * 3.0, gy * 3.0 + (sy - 0.5) * 1.1
        pts = np.where(pred == c)[0]
        jit = RNG.normal(0, 0.18, (len(pts), 2))
        axes[0].scatter(cx + jit[:, 0], cy + jit[:, 1], s=26,
                        c=[cmap(joint[p]) for p in pts], edgecolors="k", linewidths=0.3)
        axes[0].text(cx, cy + 0.75, f"#{c}", ha="center", fontsize=8, color="#555")
    axes[0].set_title(f"Recovered FACTOR GRID  (ARI={ari:.2f})\n"
                      "position = recovered cluster · colour = TRUE 8-way label")
    axes[0].set_xticks([]); axes[0].set_yticks([])
    axes[0].set_xlabel("4 coarse (colour) groups  ×  2 fine (stripe) rows")

    # confusion: recovered cluster vs true joint label
    M = np.zeros((8, 8), int)
    for p in range(len(joint)):
        M[pred[p], joint[p]] += 1
    axes[1].imshow(M, cmap="Greys"); axes[1].set_title("recovered cluster (rows) vs true 8-way (cols)")
    axes[1].set_xlabel("true joint label"); axes[1].set_ylabel("recovered cluster")
    for i in range(8):
        for j in range(8):
            if M[i, j]:
                axes[1].text(j, i, M[i, j], ha="center", va="center",
                             color="white" if M[i, j] > M.max() / 2 else "black", fontsize=8)
    fig.tight_layout(); fig.savefig(RESULTS / "factor_grid.png", dpi=130); plt.close(fig)
    print(f"  factor grid: ARI={ari:.3f}  ({n} labels)")


def main():
    sess = engine.Session("fishy"); ld = sess.ld
    F = np.stack([ld.gt.labels[f] for f in ("belly", "tail", "stripe")], 1)
    joint = F[:, 0] * 4 + F[:, 1] * 2 + F[:, 2]
    Z = zc(PCA(K_PCS, random_state=0).fit_transform(StandardScaler().fit_transform(ld.Xs)))
    print(f"recipe: K={K_PCS} PCs, rank-{RANK} metric, GMM-8\n\n(a) sampling efficiency:")
    sampling_curve(Z, F, joint)
    print("\n(b) factor-grid visualisation:")
    factor_grid(Z, F, joint)
    print(f"\nfigures -> {RESULTS}")


if __name__ == "__main__":
    main()
