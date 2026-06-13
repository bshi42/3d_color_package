"""EXP-15 — Per-block projection pursuit + dip-test significance gate.

Idea: run discovery WITHIN interpretable feature blocks (color composition vs texture vs
spectral — a general 'what colors' vs 'how arranged' split, no taxon prior), so low-variance
texture structure isn't drowned by high-variance color. Each candidate axis must pass a
SIGNIFICANCE GATE: Hartigan dip test p<0.05 (genuinely multimodal) AND dip larger than the most
multimodal axis obtainable on a column-shuffled null. Honest: which factors survive, per block.
"""
from __future__ import annotations

import numpy as np
from diptest import diptest
from sklearn.decomposition import FastICA
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from fishpipe import config, data, features, spectral

GCACHE = config.CACHE_DIR / "fishy_gabor_La.npy"


def get_blocks(fcd):
    return {
        "color(area-hist)": features.area_hist(fcd, 24, "lab"),
        "texture(Gabor)": np.load(GCACHE),
        "spectral(graph)": spectral.spectral_descriptor(fcd, k=300, n_bands=12),
    }


def pp_dip(X, n=20, seed=0):
    Xs = StandardScaler().fit_transform(X)
    S = FastICA(min(n, *Xs.shape), random_state=seed, max_iter=4000, tol=1e-3,
                whiten="unit-variance").fit_transform(Xs)
    dp = np.array([diptest(S[:, j]) for j in range(S.shape[1])])  # (m,2): dip,pval
    order = np.argsort(dp[:, 1])  # smallest p (most multimodal) first
    return S[:, order], dp[order, 0], dp[order, 1]


def main():
    gt = data.load_ground_truth()
    fcd = data.build_face_colors()
    rng = np.random.default_rng(3)

    for bname, X in get_blocks(fcd).items():
        S, dips, pv = pp_dip(X)
        Xn = np.column_stack([rng.permutation(X[:, j]) for j in range(X.shape[1])])
        Sn, dipsn, pvn = pp_dip(Xn)
        null_dip = dipsn.max()           # most-multimodal axis achievable on shuffled data
        print(f"\n=== block: {bname}  (gate: dip p<0.05 AND dip>null_max={null_dip:.3f}) ===")
        for f in ["belly", "tail", "stripe", "cheeks"]:
            y = gt.labels[f]
            aucs = np.array([max(roc_auc_score(y, S[:, j]), 1 - roc_auc_score(y, S[:, j]))
                             for j in range(S.shape[1])])
            j = int(np.argmax(aucs))
            passed = (pv[j] < 0.05) and (dips[j] > null_dip)
            print(f"  {f:7s}: bestAUC={aucs[j]:.3f}  dip={dips[j]:.3f} p={pv[j]:.3f}  "
                  f"dip-rank={j}  ->  {'DISCOVERED' if passed else 'not significant'}")


if __name__ == "__main__":
    main()
