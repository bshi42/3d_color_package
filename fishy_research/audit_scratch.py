"""EXP-06 audit: overfitting / robustness / leakage checks on the recommended pipeline.

Writes notes + figures under results/exp06_audit/. Does NOT modify the package.
"""
from __future__ import annotations

import json

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from fishpipe import config, data, embed, features, metrics, recommended, structure
from fishpipe.features import rgb_to_lab

OUT = config.RESULTS_DIR / "exp06_audit"
OUT.mkdir(parents=True, exist_ok=True)

results = {}

gt = data.load_ground_truth()
fcd = data.build_face_colors(verbose=False)
y_stripe = gt.labels["stripe"]
y_cheeks = gt.labels["cheeks"]


def cv_balacc(X, y, seed, n_splits=5):
    """Re-implementation of the metric's CV: standardize + balanced LogReg, stratified KFold."""
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    classes, counts = np.unique(y, return_counts=True)
    n_splits = int(min(n_splits, counts.min()))
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced"),
    )
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    pred = cross_val_predict(model, X, y, cv=cv)
    return float(balanced_accuracy_score(y, pred))


# ===========================================================================
# TASK 1 — SEED / STABILITY
# ===========================================================================
print("=== TASK 1: SEED/STABILITY ===")
pt = recommended.pattern_traits(fcd)
pv = pt["pattern_vector"]
rv = recommended.rare_variant_descriptor(fcd)

# the metric's own CV (seed=0 hardcoded inside)
stripe_metric = metrics.factor_recoverability(pv, y_stripe)["balanced_acc"]
cheeks_metric = metrics.factor_recoverability(rv, y_cheeks)["balanced_acc"]

seeds = [0, 1, 2, 7, 123]
stripe_seeds = [cv_balacc(pv, y_stripe, s) for s in seeds]
# cheeks minority count = 14 -> min class count caps n_splits at 5 fine
cheeks_seeds = [cv_balacc(rv, y_cheeks, s) for s in seeds]

results["task1_stability"] = {
    "stripe_metric_own_cv": stripe_metric,
    "stripe_seeds": dict(zip(map(str, seeds), stripe_seeds)),
    "stripe_mean": float(np.mean(stripe_seeds)),
    "stripe_min": float(np.min(stripe_seeds)),
    "stripe_max": float(np.max(stripe_seeds)),
    "cheeks_metric_own_cv": cheeks_metric,
    "cheeks_seeds": dict(zip(map(str, seeds), cheeks_seeds)),
    "cheeks_mean": float(np.mean(cheeks_seeds)),
    "cheeks_min": float(np.min(cheeks_seeds)),
    "cheeks_max": float(np.max(cheeks_seeds)),
}
print("stripe metric own cv:", round(stripe_metric, 4))
print("stripe seeds:", [round(x, 4) for x in stripe_seeds])
print("cheeks metric own cv:", round(cheeks_metric, 4))
print("cheeks seeds:", [round(x, 4) for x in cheeks_seeds])


# ===========================================================================
# TASK 2 — HYPERPARAM SENSITIVITY for pattern_traits
# Copy of pattern_traits logic, parametrized by quantile (mask) and n_bins.
# ===========================================================================
print("\n=== TASK 2: HYPERPARAM SENSITIVITY ===")


def pattern_vector_param(fcd, quantile=0.80, n_bins=64, smooth=3):
    """Replica of recommended.pattern_traits, parametrized on mask quantile + n_bins.

    Returns the 'pattern_vector' = [std(prof), dark_fraction, band_power(2:8)].
    """
    mask = recommended.salient_pattern_mask(fcd, quantile=quantile)
    axis = structure.principal_axis()
    t = fcd.centroids @ axis
    edges = np.linspace(t[mask].min(), t[mask].max(), n_bins + 1)
    bid = np.clip(np.digitize(t, edges) - 1, 0, n_bins - 1)
    areas = fcd.areas
    abin = np.bincount(bid[mask], weights=areas[mask], minlength=n_bins)
    abin[abin == 0] = 1.0

    L = rgb_to_lab(fcd.colors)[..., 0]
    dark = 100.0 - L
    N = L.shape[0]
    prof = np.zeros((N, n_bins))
    for i in range(N):
        prof[i] = np.bincount(bid[mask], weights=(areas * dark[i])[mask], minlength=n_bins) / abin
    # (prof_s computed for parity with original; only band_power/std/dark_frac used in vector)
    _ = uniform_filter1d(prof, size=max(1, smooth), axis=1, mode="nearest")
    fft = np.abs(np.fft.rfft(prof - prof.mean(axis=1, keepdims=True), axis=1))
    band_power = fft[:, 2:8]
    return np.column_stack([prof.std(axis=1), (prof > 50).mean(axis=1), band_power])


# sanity: replica at defaults should match the package number
pv_replica = pattern_vector_param(fcd, quantile=0.80, n_bins=64)
replica_check = metrics.factor_recoverability(pv_replica, y_stripe)["balanced_acc"]
print("replica @ defaults (q=0.80,n_bins=64):", round(replica_check, 4),
      "| package:", round(stripe_metric, 4))

quantiles = [0.70, 0.80, 0.90]
nbins_list = [48, 64, 96]
sweep = {}
for q in quantiles:
    for nb in nbins_list:
        pvq = pattern_vector_param(fcd, quantile=q, n_bins=nb)
        ba = metrics.factor_recoverability(pvq, y_stripe)["balanced_acc"]
        sweep[f"q{q}_nb{nb}"] = float(ba)
        print(f"  q={q} n_bins={nb}: balacc={ba:.4f}")

sweep_vals = list(sweep.values())
results["task2_sweep"] = {
    "replica_check_default": float(replica_check),
    "grid": sweep,
    "mean": float(np.mean(sweep_vals)),
    "min": float(np.min(sweep_vals)),
    "max": float(np.max(sweep_vals)),
}


# ===========================================================================
# TASK 4 — NEGATIVE CONTROL (placed before task 3 narrative; computed here)
# ===========================================================================
print("\n=== TASK 4: NEGATIVE CONTROL (shuffled stripe labels) ===")
shuf = {}
rng = np.random.default_rng(0)
shuffled_scores = []
for s in range(10):
    y_perm = y_stripe.copy()
    rng.shuffle(y_perm)
    ba = metrics.factor_recoverability(pv, y_perm)["balanced_acc"]
    shuffled_scores.append(ba)
results["task4_negative_control"] = {
    "shuffled_balacc_runs": [float(x) for x in shuffled_scores],
    "shuffled_mean": float(np.mean(shuffled_scores)),
    "shuffled_min": float(np.min(shuffled_scores)),
    "shuffled_max": float(np.max(shuffled_scores)),
    "real_balacc": float(stripe_metric),
}
print("shuffled stripe balacc (10 perms): mean",
      round(np.mean(shuffled_scores), 4),
      "range", round(np.min(shuffled_scores), 4), "-", round(np.max(shuffled_scores), 4))
print("real stripe balacc:", round(stripe_metric, 4))


with open(OUT / "audit_numbers.json", "w") as fh:
    json.dump(results, fh, indent=2)
print("\nSaved", OUT / "audit_numbers.json")


# ===========================================================================
# TASK 3 addendum — quantify the population-statistic transduction concern,
# and auto_cluster k-stability.
# ===========================================================================
print("\n=== TASK 3 addendum: transduction & auto_cluster stability ===")

# (a) salient_pattern_mask uses cross-specimen lightness VARIANCE over all 250 fish;
#     novelty_score uses cross-specimen per-face MEDIAN. These are unsupervised (no labels)
#     but use the whole population to build features, then the SAME population is scored in CV.
#     Check: does the salient mask depend on labels? It does not. Does building it on a
#     train-only subset change the recovered stripe signal materially? Quick split-half test.
import numpy as np
from fishpipe.features import rgb_to_lab as _r2l

L = _r2l(fcd.colors)[..., 0]
rng2 = np.random.default_rng(0)
idx = rng2.permutation(len(y_stripe))
half = idx[: len(idx) // 2]
var_full = L.var(axis=0)
var_half = L[half].var(axis=0)
mask_full = var_full >= np.quantile(var_full, 0.80)
mask_half = var_half >= np.quantile(var_half, 0.80)
overlap = (mask_full & mask_half).sum() / mask_full.sum()
print(f"salient-mask overlap full-pop vs half-pop (q=0.80): {overlap:.3f}")
results["task3_transduction"] = {"salient_mask_fullvshalf_overlap": float(overlap)}

# (b) auto_cluster k across random_state on color morphospace PC1-2
cm = recommended.color_morphospace(fcd)
ks = [int(len(np.unique(recommended.auto_cluster(cm.scores[:, :2], max_k=8, random_state=s))))
      for s in range(6)]
print("auto_cluster k over random_state 0..5 (max_k=8):", ks)
# ARI vs belly x tail for each
from sklearn.metrics import adjusted_rand_score as _ari
jbt = data.joint_label(gt, ("belly", "tail"))
aris = [float(_ari(jbt, recommended.auto_cluster(cm.scores[:, :2], max_k=8, random_state=s)))
        for s in range(6)]
print("auto_cluster ARI vs belly x tail over seeds:", [round(a, 4) for a in aris])
results["task3_transduction"]["auto_cluster_k_seeds"] = ks
results["task3_transduction"]["auto_cluster_ari_seeds"] = aris

with open(OUT / "audit_numbers.json", "w") as fh:
    json.dump(results, fh, indent=2)
print("\nUpdated", OUT / "audit_numbers.json")
