"""EXP-42 — Recovering the 8-cluster (2^3 factor) structure from expert similarity feedback.

Fishy has 3 ~binary planted factors (belly hue, tail hue, stripe count) -> 2^3 = 8 true clusters.
Two questions the demo's spring-feedback scheme raises:

  (1) FEEDBACK BUDGET / REPULSION. The natural expert label for a pair is "how many of the 3 binary
      features match" = 3 - Hamming(i,j). Most pairs share 1-2 features (graded "similar", e.g. S~0.66 /
      S~0.33); only all-3-different pairs (1/8 of pairs) are unambiguously "dissimilar" (D 1.0). Is there
      enough repelling signal to push 8 groups apart?

  (2) RECOVERY + VISUALISATION. A 2-D spring layout cannot hold 8 well-separated clusters (the 3-bit code
      is a cube -> needs 3 independent directions). Quantify, and test whether GRADED feedback recast as
      TARGET DISTANCES (feedback-driven metric learning) recovers the 8 clusters in the metric space, which
      we can then visualise deliberately rather than hoping springs settle into 8.

All "expert" labels are SIMULATED from the known GT. Uses the SAME per-region descriptor + PCA the
interactive demo uses (interactive_demo.engine), so conclusions transfer to the demo.

Run:  .venv/bin/python experiments/exp42_8cluster_feedback.py
"""
from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import nnls
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import MDS
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.model_selection import cross_val_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "interactive_demo"))
import engine  # noqa: E402

RESULTS = HERE.parent / "results" / "exp42"
RESULTS.mkdir(parents=True, exist_ok=True)
FACTORS = ("belly", "tail", "stripe")
RNG = np.random.default_rng(0)


# --------------------------------------------------------------------------- data
def load():
    sess = engine.Session("fishy")
    Z = sess.Z0                                    # (N, K) z-scored PCA coords (the demo's space)
    gt = sess.ld.gt
    F = np.stack([gt.labels[f] for f in FACTORS], 1).astype(int)   # (N, 3) binary factor labels
    joint = np.zeros(len(F), dtype=int)
    for c in range(3):
        joint = joint * 2 + F[:, c]                # 0..7  three-bit code
    return sess, Z, F, joint


def banner(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


# --------------------------------------------------------------------------- (1) structure
def phase_structure(Z, F, joint):
    banner("PHASE 1 — is the 8-cluster structure there, and where is it hard?")
    N, K = Z.shape
    cells = np.bincount(joint, minlength=8)
    print(f"N={N}  K(PCs)={K}")
    print(f"8-cell counts (belly,tail,stripe code 0..7): {cells.tolist()}  "
          f"(min {cells.min()}, max {cells.max()})")

    print("\nper-factor linear separability in the demo's PCA space (5-fold logistic CV acc):")
    for c, f in enumerate(FACTORS):
        acc = cross_val_score(LogisticRegression(max_iter=2000), Z, F[:, c], cv=5).mean()
        print(f"  {f:7s}: {acc:.3f}")

    print("\nUNSUPERVISED clustering of the raw PCA space (no feedback):")
    for k, lab, name in [(4, joint // 2, "color only (belly,tail) k=4"),
                         (8, joint, "all 3 factors      k=8")]:
        km = KMeans(k, n_init=10, random_state=0).fit_predict(Z)
        print(f"  KMeans {name}: ARI={adjusted_rand_score(lab, km):.3f}")
    return cells


# --------------------------------------------------------------------------- (2) feedback budget
def hamming_budget(F):
    banner("PHASE 2 — feedback budget: how much 'repel' signal exists?")
    N = len(F)
    H = squareform(pdist(F, metric="cityblock")).astype(int)     # Hamming over the 3 bits, 0..3
    iu = np.triu_indices(N, 1)
    h = H[iu]
    frac = np.bincount(h, minlength=4) / len(h)
    print("Hamming distance over the 3 factors, ALL pairs:")
    for d in range(4):
        sim = 1 - d / 3
        kind = "S" if d <= 1 else ("borderline" if d == 2 else "D")
        print(f"  H={d}  ({frac[d]*100:5.1f}% of pairs)  expert label ~ "
              f"{'S' if d<=1 else 'D'} {sim if d<=1 else d/3:.2f}   [{kind}]")
    print(f"\n-> only {frac[3]*100:.1f}% of pairs are all-3-different (the only clean 'D 1.00').")
    print(f"-> {(frac[2]+frac[3])*100:.1f}% of pairs differ in >=2 factors (candidate repulsion if we")
    print("   are willing to call H>=2 'dissimilar' rather than only H==3).")

    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.bar(range(4), frac, color=["#2ca02c", "#7fc97f", "#f0a868", "#d62728"])
    ax.set_xticks(range(4)); ax.set_xlabel("# of 3 factors that differ (Hamming)")
    ax.set_ylabel("fraction of pairs"); ax.set_title("Feedback budget: pair label distribution")
    for d in range(4):
        ax.text(d, frac[d] + .01, f"{frac[d]*100:.0f}%", ha="center")
    fig.tight_layout(); fig.savefig(RESULTS / "hamming_budget.png", dpi=130); plt.close(fig)
    return H


# --------------------------------------------------------------------------- (2b) 2D can't hold 8
def embedding_limit(F, joint):
    banner("PHASE 3 — can 2-D hold 8 clusters? (ideal-distance MDS stress)")
    # 8 cluster centroids in the IDEAL factor space = the 3-bit cube; pairwise = Hamming.
    codes = np.array([[ (c >> b) & 1 for b in (2, 1, 0)] for c in range(8)], float)
    Dcube = squareform(pdist(codes, "cityblock"))
    for dim in (2, 3):
        mds = MDS(dim, dissimilarity="precomputed", random_state=0, normalized_stress="auto")
        mds.fit(Dcube)
        print(f"  MDS of the 8-vertex cube into {dim}-D: normalized stress = {mds.stress_:.4f}")
    print("  (stress ~0 means the distances fit; 2-D keeps a large residual -> 8 clusters cannot be")
    print("   laid out in 2-D without collapsing some apart-pairs together.)")


# --------------------------------------------------------------------------- (3) recovery via graded metric
def learn_metric(Z, pairs, target_d2):
    """Non-negative diagonal metric w>=0 with d2_w(i,j)=sum_k w_k (z_ik-z_jk)^2 ~ target_d2 (LS)."""
    G = np.array([(Z[i] - Z[j]) ** 2 for (i, j) in pairs])        # (n_pairs, K)
    w, _ = nnls(G, np.asarray(target_d2, float))
    s = w.sum()
    return w * (len(w) / s) if s > 1e-9 else np.ones(Z.shape[1])


def sample_pairs(joint, n, mode):
    """Simulated query pairs. 'random' = uniform pairs; 'within_color' = pairs that SHARE the two
    color factors (so they isolate the hard stripe factor)."""
    N = len(joint)
    color = joint // 2
    out = []
    tries = 0
    while len(out) < n and tries < n * 200:
        i, j = RNG.integers(N, size=2)
        tries += 1
        if i == j:
            continue
        if mode == "within_color" and color[i] != color[j]:
            continue
        out.append((int(i), int(j)))
    return out


def phase_recovery(Z, F, joint):
    banner("PHASE 4 — recover 8 clusters from GRADED feedback (target-distance metric learning)")
    base = adjusted_rand_score(joint, KMeans(8, n_init=10, random_state=0).fit_predict(Z))
    print(f"baseline (no feedback) KMeans k=8 ARI on raw PCA: {base:.3f}\n")

    budgets = [20, 40, 80, 160, 320]
    reps = 5
    curves = {}
    for mode in ("random", "within_color"):
        print(f"sampling = {mode}")
        means = []
        for n in budgets:
            aris = []
            for r in range(reps):
                pairs = sample_pairs(joint, n, mode)
                # simulated expert: target squared distance = Hamming over the 3 factors
                td2 = [int(np.abs(F[i] - F[j]).sum()) for (i, j) in pairs]
                w = learn_metric(Z, pairs, td2)
                Zw = Z * np.sqrt(w)[None, :]
                km = KMeans(8, n_init=10, random_state=r).fit_predict(Zw)
                aris.append(adjusted_rand_score(joint, km))
            m = float(np.mean(aris))
            means.append(m)
            print(f"  {n:4d} labels: ARI(k=8) = {m:.3f}  (+/-{np.std(aris):.3f})")
        curves[mode] = means

    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    for mode, ys in curves.items():
        ax.plot(budgets, ys, "o-", label=mode)
    ax.axhline(base, ls="--", c="gray", label="no feedback")
    ax.set_xlabel("# simulated pair labels"); ax.set_ylabel("8-cluster recovery (ARI)")
    ax.set_title("Graded-feedback metric learning -> KMeans k=8")
    ax.legend(); ax.set_ylim(0, 1); fig.tight_layout()
    fig.savefig(RESULTS / "recovery_curve.png", dpi=130); plt.close(fig)
    return curves


def main():
    sess, Z, F, joint = load()
    phase_structure(Z, F, joint)
    hamming_budget(F)
    embedding_limit(F, joint)
    phase_recovery(Z, F, joint)
    print(f"\nfigures -> {RESULTS}")


if __name__ == "__main__":
    main()
