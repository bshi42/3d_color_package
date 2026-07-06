"""EXP-43 — How high can 8-cluster recovery go? Representation x metric x clustering.

EXP-42 showed graded-feedback metric learning lifts ARI 0.23->0.39, but the supervised ceiling on the
demo's color+texture PCA is only ~0.57 (stripe is the bottleneck: 0.84 separable, 21% minority).
Two levers from the program:
  * REPRESENTATION: swap the weak local-jet texture for the texton BoW (known to beat GFT on stripe).
  * METHOD: a full Mahalanobis metric (closed-form LS + PSD projection) vs the diagonal NNLS one, and
    GMM / Ward vs KMeans (imbalanced, non-spherical 8 cells).

Everything from simulated graded labels (target squared distance = Hamming over the 3 GT factors).

Run:  .venv/bin/python experiments/exp43_repr_metric_ceiling.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from scipy.optimize import nnls

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "interactive_demo"))
import engine  # noqa: E402
from fishpipe import config, textons  # noqa: E402

FACTORS = ("belly", "tail", "stripe")
RNG = np.random.default_rng(0)
PDIM, COLOR_CH = engine.PDIM, engine.COLOR_CH


# --------------------------------------------------------------------------- representations
def representations(sess):
    """Return {name: Z(N,K)} for the demo's current space and a texton-augmented one."""
    ld = sess.ld
    Xs = ld.Xs                                                   # standardized color+texture blocks
    R = len(ld.rids)
    color_cols = np.concatenate([np.arange(i * PDIM, i * PDIM + COLOR_CH) for i in range(R)])
    color = Xs[:, color_cols]                                    # color-only blocks
    print("  building texton BoW (this can take ~1 min the first time)…", flush=True)
    tex = textons.texton_descriptor(ld.fcd, K=128, mode="vlad", cache_dir=ld.cache_dir)
    aug = np.hstack([StandardScaler().fit_transform(color), StandardScaler().fit_transform(tex)])

    def pca10(X):
        return engine._zscore_cols(PCA(10, random_state=0).fit_transform(StandardScaler().fit_transform(X)))

    return {"current(color+jet)": sess.Z0, "texton-aug(color+textons)": pca10(aug)}


# --------------------------------------------------------------------------- metric learners
def diag_metric(Z, pairs, t):
    G = np.array([(Z[i] - Z[j]) ** 2 for (i, j) in pairs])
    w, _ = nnls(G, np.asarray(t, float))
    w = w * (len(w) / w.sum()) if w.sum() > 1e-9 else np.ones(Z.shape[1])
    return Z * np.sqrt(w)[None, :]


def full_metric(Z, pairs, t):
    """Full Mahalanobis M>=0 fit to squared-distance targets (LS, then PSD eigen-clip)."""
    K = Z.shape[1]; iu = np.triu_indices(K); off = iu[0] != iu[1]
    rows = []
    for (i, j) in pairs:
        d = Z[i] - Z[j]; v = np.outer(d, d)[iu].astype(float); v[off] *= 2.0
        rows.append(v)
    m, *_ = np.linalg.lstsq(np.array(rows), np.asarray(t, float), rcond=None)
    M = np.zeros((K, K)); M[iu] = m; M = M + M.T - np.diag(np.diag(M))
    wv, V = np.linalg.eigh(M); wv = np.clip(wv, 0, None)
    T = (V * np.sqrt(wv)) @ V.T                                  # M^{1/2}; y = T z preserves d^2_M
    return Z @ T.T


def cluster_ari(Y, joint, k=8):
    out = {}
    out["KMeans"] = adjusted_rand_score(joint, KMeans(k, n_init=10, random_state=0).fit_predict(Y))
    out["GMM"] = adjusted_rand_score(joint, GaussianMixture(k, covariance_type="full",
                                                            n_init=3, random_state=0).fit_predict(Y))
    out["Ward"] = adjusted_rand_score(joint, AgglomerativeClustering(k, linkage="ward").fit_predict(Y))
    return out


def sample(joint, n):
    N = len(joint); out = []
    while len(out) < n:
        i, j = RNG.integers(N, size=2)
        if i != j:
            out.append((int(i), int(j)))
    return out


def main():
    sess = engine.Session("fishy")
    gt = sess.ld.gt
    F = np.stack([gt.labels[f] for f in FACTORS], 1)
    joint = F[:, 0] * 4 + F[:, 1] * 2 + F[:, 2]
    reps = representations(sess)

    print("\n=== per-representation diagnostics ===")
    for name, Z in reps.items():
        sep = [round(cross_val_score(LogisticRegression(max_iter=2000), Z, F[:, c], cv=5).mean(), 3)
               for c in range(3)]
        axes = np.stack([LDA().fit(Z, F[:, c]).decision_function(Z) for c in range(3)], 1)
        ceil = adjusted_rand_score(joint, KMeans(8, n_init=10, random_state=0).fit_predict(
            engine._zscore_cols(axes)))
        unsup = adjusted_rand_score(joint, KMeans(8, n_init=10, random_state=0).fit_predict(Z))
        print(f"  {name:30s}  sep[belly,tail,stripe]={sep}  unsup-k8 ARI={unsup:.3f}  "
              f"SUPERVISED ceiling ARI={ceil:.3f}")

    print("\n=== graded-feedback recovery: representation x metric x clustering ===")
    budget, draws = 200, 5
    for name, Z in reps.items():
        print(f"\n  {name}")
        for mname, mfn in [("diagonal", diag_metric), ("full-Mahalanobis", full_metric)]:
            agg = {"KMeans": [], "GMM": [], "Ward": []}
            for d in range(draws):
                pairs = sample(joint, budget)
                t = [int(np.abs(F[i] - F[j]).sum()) for (i, j) in pairs]
                Y = mfn(Z, pairs, t)
                for c, v in cluster_ari(Y, joint).items():
                    agg[c].append(v)
            cells = "  ".join(f"{c}={np.mean(v):.3f}" for c, v in agg.items())
            print(f"    {mname:16s} ({budget} labels): {cells}")


if __name__ == "__main__":
    main()
