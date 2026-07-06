"""EXP-55: ICA vs PCA on spectral descriptor -- does ICA recover stripe with FEWER components?

Build standardized k=100-mode descriptor. Sweep n_components.
For each n: fit FastICA(n) and PCA(n) on SAME descriptor.
Measure per-factor (belly,tail,stripe) 5-fold logistic accuracy + GMM ARI
(stripe k=2, belly x tail k=4). Plot stripe accuracy vs n for ICA vs PCA.
Report n where each hits 0.95 of its max stripe accuracy.
"""
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FastICA, PCA
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import cross_val_score
from sklearn.metrics import adjusted_rand_score
from sklearn.exceptions import ConvergenceWarning

SEED = 0
K = 100
NS = [2, 3, 4, 6, 8, 12, 16, 20, 30, 40]
FACTORS = ["belly", "tail", "stripe"]

rng = np.random.RandomState(SEED)

# ---- Load + build k=100 descriptor ----
d = np.load("results/exp55/cache.npz", allow_pickle=True)
CL, Ca, Cb = d["CL"], d["Ca"], d["Cb"]
F = d["F"]  # (250,3) GT: belly, tail, stripe -- SCORE ONLY
desc = np.concatenate([CL[:, :K], Ca[:, :K], Cb[:, :K]], axis=1)
X = StandardScaler().fit_transform(desc)  # (250, 300)
print(f"descriptor X: {X.shape}, factor sums {F.sum(0)}")

belly = F[:, 0]; tail = F[:, 1]; stripe = F[:, 2]
# belly x tail 4-class label for GMM (k=4)
bt = belly * 2 + tail


def logacc(Z, y):
    clf = LogisticRegression(max_iter=5000, random_state=SEED)
    return cross_val_score(clf, Z, y, cv=5, scoring="accuracy").mean()


def gmm_ari(Z, y, k):
    gm = GaussianMixture(n_components=k, covariance_type="full",
                         n_init=5, reg_covar=1e-3, random_state=SEED)
    pred = gm.fit_predict(Z)
    return adjusted_rand_score(y, pred)


results = {m: {f: [] for f in FACTORS} for m in ["ICA", "PCA"]}
results["ICA"]["ari_stripe"] = []; results["PCA"]["ari_stripe"] = []
results["ICA"]["ari_bt"] = []; results["PCA"]["ari_bt"] = []
ica_nonconverged = []

for n in NS:
    # ---- PCA ----
    Zp = PCA(n_components=n, random_state=SEED).fit_transform(X)
    # ---- ICA (catch convergence) ----
    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter("always")
        ica = FastICA(n_components=n, whiten="unit-variance",
                      max_iter=2000, random_state=SEED)
        Zi = ica.fit_transform(X)
        converged = not any(issubclass(w.category, ConvergenceWarning) for w in wlist)
    n_iter = getattr(ica, "n_iter_", None)
    if not converged:
        ica_nonconverged.append((n, n_iter))
    # standardize component representations before downstream (ICA scale arbitrary)
    Zp = StandardScaler().fit_transform(Zp)
    Zi = StandardScaler().fit_transform(Zi)

    for f, y in zip(FACTORS, [belly, tail, stripe]):
        results["PCA"][f].append(logacc(Zp, y))
        results["ICA"][f].append(logacc(Zi, y))
    results["PCA"]["ari_stripe"].append(gmm_ari(Zp, stripe, 2))
    results["ICA"]["ari_stripe"].append(gmm_ari(Zi, stripe, 2))
    results["PCA"]["ari_bt"].append(gmm_ari(Zp, bt, 4))
    results["ICA"]["ari_bt"].append(gmm_ari(Zi, bt, 4))
    print(f"n={n:2d} conv={converged} iter={n_iter} | "
          f"ICA stripe={results['ICA']['stripe'][-1]:.3f} PCA stripe={results['PCA']['stripe'][-1]:.3f} | "
          f"ICA ari2={results['ICA']['ari_stripe'][-1]:.3f} PCA ari2={results['PCA']['ari_stripe'][-1]:.3f}")

NSa = np.array(NS)


def n_at_95(vals):
    vals = np.array(vals)
    mx = vals.max()
    thr = 0.95 * mx
    idx = np.argmax(vals >= thr)  # first index reaching threshold
    return NSa[idx], mx, thr


ica_stripe = results["ICA"]["stripe"]
pca_stripe = results["PCA"]["stripe"]
n_ica, mx_ica, thr_ica = n_at_95(ica_stripe)
n_pca, mx_pca, thr_pca = n_at_95(pca_stripe)
print(f"\nSTRIPE 0.95-of-max: ICA n={n_ica} (max={mx_ica:.3f}, thr={thr_ica:.3f}); "
      f"PCA n={n_pca} (max={mx_pca:.3f}, thr={thr_pca:.3f})")
print(f"ICA non-converged (n, n_iter): {ica_nonconverged}")

# ---- Plot ----
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
ax = axes[0]
ax.plot(NSa, pca_stripe, "o-", color="#1f77b4", label="PCA stripe acc", lw=2)
ax.plot(NSa, ica_stripe, "s-", color="#d62728", label="ICA stripe acc", lw=2)
ax.axhline(thr_pca, color="#1f77b4", ls=":", alpha=0.6, label=f"0.95*max PCA={thr_pca:.3f}")
ax.axhline(thr_ica, color="#d62728", ls=":", alpha=0.6, label=f"0.95*max ICA={thr_ica:.3f}")
ax.axvline(n_pca, color="#1f77b4", ls="--", alpha=0.5)
ax.axvline(n_ica, color="#d62728", ls="--", alpha=0.5)
ax.scatter([n_pca], [pca_stripe[list(NSa).index(n_pca)]], s=160, facecolors="none",
           edgecolors="#1f77b4", lw=2, zorder=5)
ax.scatter([n_ica], [ica_stripe[list(NSa).index(n_ica)]], s=160, facecolors="none",
           edgecolors="#d62728", lw=2, zorder=5)
ax.set_xlabel("n_components"); ax.set_ylabel("stripe 5-fold logistic accuracy")
ax.set_title(f"STRIPE recovery: ICA vs PCA (k={K} desc)\n0.95-of-max @ ICA n={n_ica}, PCA n={n_pca}")
ax.legend(fontsize=8, loc="lower right"); ax.grid(alpha=0.3)

ax = axes[1]
for f, mk in zip(FACTORS, ["o", "^", "s"]):
    ax.plot(NSa, results["PCA"][f], mk + "-", color="#1f77b4", alpha=0.5)
    ax.plot(NSa, results["ICA"][f], mk + "-", color="#d62728", alpha=0.5)
ax.plot([], [], color="#1f77b4", label="PCA")
ax.plot([], [], color="#d62728", label="ICA")
ax.plot([], [], "ko-", alpha=0.4, label="belly (o)")
ax.plot([], [], "k^-", alpha=0.4, label="tail (^)")
ax.plot([], [], "ks-", alpha=0.4, label="stripe (s)")
ax.set_xlabel("n_components"); ax.set_ylabel("5-fold logistic accuracy")
ax.set_title("All factors: ICA (red) vs PCA (blue)")
ax.legend(fontsize=8, loc="lower right"); ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("results/exp55/ica_vs_pca_stripe.png", dpi=130)
print("saved results/exp55/ica_vs_pca_stripe.png")

# Summary table
print("\n n  | ICA_belly PCA_belly | ICA_tail PCA_tail | ICA_stripe PCA_stripe | ICAari2 PCAari2 | ICAari4 PCAari4")
for i, n in enumerate(NS):
    print(f"{n:3d} | {results['ICA']['belly'][i]:.3f}    {results['PCA']['belly'][i]:.3f}    | "
          f"{results['ICA']['tail'][i]:.3f}   {results['PCA']['tail'][i]:.3f}   | "
          f"{results['ICA']['stripe'][i]:.3f}     {results['PCA']['stripe'][i]:.3f}     | "
          f"{results['ICA']['ari_stripe'][i]:.3f}  {results['PCA']['ari_stripe'][i]:.3f}  | "
          f"{results['ICA']['ari_bt'][i]:.3f}  {results['PCA']['ari_bt'][i]:.3f}")
