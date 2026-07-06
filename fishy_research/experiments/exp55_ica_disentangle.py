"""EXP-55: Does ICA put STRIPE in a DEDICATED component (disentanglement)?

Fit FastICA(n_components=n, whiten='unit-variance', max_iter=2000, random_state=0)
for n in {8,16} on the standardized k=100-mode descriptor; also PCA(n).
For each component, compute single-feature discriminability for stripe/belly/tail:
  - 5-fold 1-feature logistic accuracy
  - |point-biserial corr| with the factor
Report BEST single-component stripe accuracy for ICA vs PCA.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA, PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold

SEED = 0
np.random.seed(SEED)
K = 100

d = np.load("results/exp55/cache.npz", allow_pickle=True)
CL, Ca, Cb = d["CL"], d["Ca"], d["Cb"]
F = d["F"]  # (250,3) [belly, tail, stripe] GT — score only
FACTORS = ["belly", "tail", "stripe"]

# k-mode spectral descriptor
X = np.concatenate([CL[:, :K], Ca[:, :K], Cb[:, :K]], axis=1)
X = StandardScaler().fit_transform(X)  # (250, 3K)
print("descriptor X:", X.shape)


def point_biserial(feat, y):
    # |point-biserial correlation| = |pearson(feat, binary y)|
    f = (feat - feat.mean()) / (feat.std() + 1e-12)
    yy = (y - y.mean()) / (y.std() + 1e-12)
    return abs(float(np.mean(f * yy)))


def single_feature_acc(feat, y):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    clf = LogisticRegression(max_iter=2000)
    scores = cross_val_score(clf, feat.reshape(-1, 1), y, cv=cv, scoring="accuracy")
    return float(scores.mean())


def analyze(Z, label):
    """Z: (250, n) component scores. Returns dict of per-component acc & |pbc| per factor."""
    n = Z.shape[1]
    # standardize each component (logistic + pbc are scale-invariant-ish but keep clean)
    Zs = StandardScaler().fit_transform(Z)
    out = {fac: {"acc": np.zeros(n), "pbc": np.zeros(n)} for fac in FACTORS}
    for j in range(n):
        for fi, fac in enumerate(FACTORS):
            y = F[:, fi]
            out[fac]["acc"][j] = single_feature_acc(Zs[:, j], y)
            out[fac]["pbc"][j] = point_biserial(Zs[:, j], y)
    return out


results = {}
for n in (8, 16):
    ica = FastICA(n_components=n, whiten="unit-variance", max_iter=2000, random_state=SEED)
    Zi = ica.fit_transform(X)
    pca = PCA(n_components=n, random_state=SEED)
    Zp = pca.fit_transform(X)
    results[("ICA", n)] = analyze(Zi, f"ICA n={n}")
    results[("PCA", n)] = analyze(Zp, f"PCA n={n}")
    print(f"\n=== n={n} ===")
    for method in ("ICA", "PCA"):
        r = results[(method, n)]
        for fac in FACTORS:
            acc = r[fac]["acc"]
            pbc = r[fac]["pbc"]
            jbest = int(np.argmax(acc))
            # "spread": how many components beat a threshold; gap between best and 2nd best
            srt = np.sort(acc)[::-1]
            gap = srt[0] - srt[1]
            n_high = int((acc >= 0.80).sum())
            print(f"  {method} {fac:6s}: best_acc={acc[jbest]:.3f} (comp {jbest}, |pbc|={pbc[jbest]:.3f}) "
                  f"| 2nd={srt[1]:.3f} gap={gap:.3f} | #comps>=0.80={n_high}")

# ---- Plot: per-component stripe discriminability (sorted) ICA vs PCA at n=16 ----
n = 16
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
stripe_idx = FACTORS.index("stripe")

# Left: sorted stripe single-feature accuracy
ax = axes[0]
acc_ica = np.sort(results[("ICA", n)]["stripe"]["acc"])[::-1]
acc_pca = np.sort(results[("PCA", n)]["stripe"]["acc"])[::-1]
xs = np.arange(1, n + 1)
ax.plot(xs, acc_ica, "o-", color="tab:red", label=f"ICA (best={acc_ica[0]:.3f})")
ax.plot(xs, acc_pca, "s-", color="tab:blue", label=f"PCA (best={acc_pca[0]:.3f})")
ax.axhline(0.85, color="gray", ls="--", lw=1, label="raw single GFT mode ~0.85 (EXP-54)")
ax.axhline(52 / 250, color="black", ls=":", lw=1, label=f"stripe base rate {52/250:.2f}")
ax.set_xlabel("component rank (sorted by stripe acc)")
ax.set_ylabel("5-fold 1-feature logistic accuracy (STRIPE)")
ax.set_title(f"STRIPE discriminability per component (n={n})")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# Right: sorted |point-biserial corr|
ax = axes[1]
pbc_ica = np.sort(results[("ICA", n)]["stripe"]["pbc"])[::-1]
pbc_pca = np.sort(results[("PCA", n)]["stripe"]["pbc"])[::-1]
ax.plot(xs, pbc_ica, "o-", color="tab:red", label=f"ICA (max |pbc|={pbc_ica[0]:.3f})")
ax.plot(xs, pbc_pca, "s-", color="tab:blue", label=f"PCA (max |pbc|={pbc_pca[0]:.3f})")
ax.set_xlabel("component rank (sorted by |pbc|)")
ax.set_ylabel("|point-biserial corr| with STRIPE")
ax.set_title(f"STRIPE |corr| per component (n={n})")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

fig.suptitle("ICA vs PCA: does ICA concentrate STRIPE into one dedicated component?", fontsize=12)
fig.tight_layout()
fig.savefig("results/exp55/ica_disentangle.png", dpi=120)
print("\nsaved results/exp55/ica_disentangle.png")

# Summary table for stripe concentration
print("\n--- STRIPE concentration summary ---")
for n in (8, 16):
    for method in ("ICA", "PCA"):
        acc = np.sort(results[(method, n)]["stripe"]["acc"])[::-1]
        print(f"n={n:2d} {method}: top3 stripe acc = {acc[0]:.3f}, {acc[1]:.3f}, {acc[2]:.3f}")
