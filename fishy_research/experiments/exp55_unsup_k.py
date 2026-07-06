"""EXP-55: Do standard UNSUPERVISED model-selection rules pick the right k?

Two rules, NO labels used to pick k:
  (1) explained-variance threshold: smallest k where cumulative PCA EV of the
      k_max=100 descriptor reaches 90/95/99%.
  (2) clustering stability: for each k, subsample 80% several times, GMM-cluster
      the k-mode descriptor, measure mean pairwise ARI across subsamples on the
      shared (overlapping) samples. Pick k where stability plateaus.

THEN (scoring only, GT never used to pick k): stripe accuracy at those k via
cross-validated LogisticRegression on the k-mode descriptor.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import adjusted_rand_score

SEED = 0
np.random.seed(SEED)

cache = np.load("results/exp55/cache.npz", allow_pickle=True)
CL, Ca, Cb = cache["CL"], cache["Ca"], cache["Cb"]
F = cache["F"]            # (250,3) belly, tail, stripe -- SCORING ONLY
stripe = F[:, 2]
N = CL.shape[0]


def descriptor(k):
    """k-mode spectral descriptor: standardized concat of first-k modes per channel."""
    X = np.concatenate([CL[:, :k], Ca[:, :k], Cb[:, :k]], axis=1)
    return StandardScaler().fit_transform(X)


# ---------------------------------------------------------------------------
# RULE 1: explained-variance threshold on the k_max=100 descriptor
# ---------------------------------------------------------------------------
KMAX = 100
Xmax = descriptor(KMAX)                       # (250, 300)
pca = PCA(n_components=min(N, Xmax.shape[1]), random_state=SEED).fit(Xmax)
cum_ev = np.cumsum(pca.explained_variance_ratio_)

def k_for_ev(th):
    return int(np.searchsorted(cum_ev, th) + 1)   # # components to reach threshold

k_ev = {th: k_for_ev(th) for th in (0.90, 0.95, 0.99)}

# ---------------------------------------------------------------------------
# RULE 2: clustering stability vs k
# ---------------------------------------------------------------------------
KS = [1, 2, 3, 5, 8, 10, 12, 15, 20, 30, 40, 60, 80, 100]
N_REPEATS = 12
FRAC = 0.80
N_CLUST = 2          # we are probing for binary structure (stripe present/absent)

rng = np.random.default_rng(SEED)
# fixed subsample index sets, shared across all k so ARI is comparable
subsets = [rng.choice(N, size=int(FRAC * N), replace=False) for _ in range(N_REPEATS)]

stab_mean = []
stab_std = []
for k in KS:
    Xk = descriptor(k)
    labels = []
    for idx in subsets:
        gm = GaussianMixture(n_components=N_CLUST, covariance_type="full",
                             random_state=SEED, n_init=2, reg_covar=1e-4)
        lab = gm.fit_predict(Xk[idx])
        full = np.full(N, -1)
        full[idx] = lab
        labels.append(full)
    # pairwise ARI on overlapping samples
    aris = []
    for i in range(N_REPEATS):
        for j in range(i + 1, N_REPEATS):
            shared = (labels[i] >= 0) & (labels[j] >= 0)
            if shared.sum() > 10:
                aris.append(adjusted_rand_score(labels[i][shared], labels[j][shared]))
    stab_mean.append(np.mean(aris))
    stab_std.append(np.std(aris))
stab_mean = np.array(stab_mean)
stab_std = np.array(stab_std)

# plateau pick: first k whose stability is within 2% of the running max and
# whose forward change is small (<0.01 abs to the next probed k)
def pick_plateau(ks, vals, tol=0.01):
    run_max = np.maximum.accumulate(vals)
    for i in range(len(ks) - 1):
        if vals[i] >= run_max[i] - 0.02 and abs(vals[i + 1] - vals[i]) < tol:
            return ks[i]
    return ks[int(np.argmax(vals))]

k_stab = pick_plateau(KS, stab_mean)
k_stab_argmax = KS[int(np.argmax(stab_mean))]

# ---------------------------------------------------------------------------
# SCORING (GT used only here): stripe accuracy at the picked k
# ---------------------------------------------------------------------------
def stripe_acc(k):
    Xk = descriptor(k)
    clf = LogisticRegression(max_iter=2000, C=1.0)
    sc = cross_val_score(clf, Xk, stripe, cv=5, scoring="accuracy")
    return sc.mean(), sc.std()

picked_ks = sorted(set(list(k_ev.values()) + [k_stab, k_stab_argmax]))
acc_at = {k: stripe_acc(k) for k in picked_ks}
# reference accuracy at large k to show what stripe COULD reach
acc_ref = {k: stripe_acc(k) for k in (40, 100)}

# stripe accuracy curve across all probed KS (for context / plotting)
acc_curve = np.array([stripe_acc(k)[0] for k in KS])

# ---------------------------------------------------------------------------
# REPORT
# ---------------------------------------------------------------------------
print("=== RULE 1: cumulative PCA EV (k_max=100 descriptor) ===")
for th in (0.90, 0.95, 0.99):
    print(f"  {int(th*100)}% EV -> k = {k_ev[th]}  (stripe acc {stripe_acc(k_ev[th])[0]:.3f})")
print("=== RULE 2: clustering stability (GMM 2-comp, 80%% subsample x%d) ===" % N_REPEATS)
for k, m, s in zip(KS, stab_mean, stab_std):
    print(f"  k={k:3d}  meanARI={m:.3f} +/- {s:.3f}   stripe_acc={dict(zip(KS,acc_curve))[k]:.3f}")
print(f"  plateau pick k = {k_stab}  (argmax k = {k_stab_argmax})")
print("=== SCORING (GT, stripe accuracy) ===")
for k in picked_ks:
    m, s = acc_at[k]
    print(f"  picked k={k:3d}: stripe acc = {m:.3f} +/- {s:.3f}")
for k in (40, 100):
    m, s = acc_ref[k]
    print(f"  REF    k={k:3d}: stripe acc = {m:.3f} +/- {s:.3f}")

# ---------------------------------------------------------------------------
# PLOT
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
ks_ev = np.arange(1, len(cum_ev) + 1)
ax.plot(ks_ev, cum_ev, "-", color="tab:blue", label="cumulative PCA EV")
for th, c in zip((0.90, 0.95, 0.99), ("tab:green", "tab:orange", "tab:red")):
    ax.axhline(th, ls=":", color=c, lw=1)
    ax.axvline(k_ev[th], ls="--", color=c, lw=1.2,
               label=f"{int(th*100)}% EV -> k={k_ev[th]}")
ax.set_xlim(0, 60)
ax.set_xlabel("k (# PCA components of k_max=100 descriptor)")
ax.set_ylabel("cumulative explained variance")
ax.set_title("RULE 1: explained-variance threshold")
ax.legend(loc="lower right", fontsize=8)
ax.grid(alpha=0.3)

ax2 = axes[1]
ax2.errorbar(KS, stab_mean, yerr=stab_std, marker="o", color="tab:purple",
             label="GMM stability (mean pairwise ARI)")
ax2.axvline(k_stab, ls="--", color="tab:purple", lw=1.2,
            label=f"stability plateau -> k={k_stab}")
ax2.plot(KS, acc_curve, marker="s", color="tab:gray", alpha=0.8,
         label="stripe acc (GT, scoring only)")
ax2.axhline(0.79, ls=":", color="k", lw=0.8, alpha=0.5)
ax2.set_xscale("log")
ax2.set_xlabel("k (modes per channel)")
ax2.set_ylabel("score")
ax2.set_title("RULE 2: clustering stability vs k  (+ GT stripe acc)")
ax2.legend(loc="best", fontsize=8)
ax2.grid(alpha=0.3)

fig.suptitle("EXP-55: unsupervised k-selection rules vs GT stripe accuracy", fontsize=12)
fig.tight_layout()
fig.savefig("results/exp55/unsup_k.png", dpi=130)
print("saved results/exp55/unsup_k.png")
