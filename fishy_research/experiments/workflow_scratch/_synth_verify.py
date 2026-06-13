"""Adversarial verification of the headline STRIPE result from the seg-config sweep.

Best config (from completed sweep incl. K=8): K=8, smooth=0, region=whole.
spatial+endler -> stripe recoverability (class-weighted full-data CV) = 0.929.

This script checks the result is not an imbalance artifact:
  (a) balanced-subsample stripe balacc: equalize 4-stripe (count=4, n=198) vs
      5-stripe (count=5, n=52) by sampling 52 negatives, equal-class 5-fold CV,
      averaged over many random draws.  (true honest separability)
  (b) column-shuffled null: permute the stripe labels, re-run full-data CV.
      Should collapse to ~0.5.
  (c) count-vs-spacing transparency: supervised LDA projection of spatial+endler,
      correlation with stripe_count vs stripe_spacing vs stripe_width; plus
      regress-spacing-out test on the count signal.

All output printed as one JSON line.
"""
from __future__ import annotations

import json
import warnings

warnings.filterwarnings("ignore")
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from fishpipe import data, segment, blobs, spatial

K, SMOOTH, REGION = 8, 0, "whole"
MAF = 0.0005

gt = data.load_ground_truth()
fcd = data.build_face_colors()
df = gt.params
y = gt.labels["stripe"].astype(int)
count = df.stripe_count.values.astype(float)
spacing = df.stripe_spacing.values.astype(float)
width = df.stripe_width.values.astype(float)

seg = segment.segment(fcd, n_colors=K, smooth_iters=SMOOTH, per_specimen=False)
endl = blobs.endler_transitions(fcd, seg)
spat = spatial.spatial_descriptor(fcd, seg, region_mask=None, min_area_frac=MAF)
# match seg_eval.py: per-block StandardScaler then concat
spat_endl = np.concatenate(
    [StandardScaler().fit_transform(b) for b in [spat, endl]], axis=1
)

out = {"config": {"K": K, "smooth": SMOOTH, "region": REGION},
       "n_pos5": int(y.sum()), "n_neg4": int((y == 0).sum())}


def full_balacc(X, yv, seed=0):
    """Class-weighted logistic, stratified 5-fold balanced acc on full data."""
    model = make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    pred = cross_val_predict(model, X, yv, cv=cv)
    return float(balanced_accuracy_score(yv, pred))


def balanced_subsample_balacc(X, yv, n_draws=50, seed0=0):
    """Equalize classes (52 vs 52), equal-class 5-fold CV, avg over draws."""
    pos = np.where(yv == 1)[0]
    neg = np.where(yv == 0)[0]
    n = min(len(pos), len(neg))
    accs = []
    for d in range(n_draws):
        rng = np.random.default_rng(seed0 + d)
        sel = np.concatenate([rng.choice(pos, n, replace=False),
                              rng.choice(neg, n, replace=False)])
        Xs, ys = X[sel], yv[sel]
        model = make_pipeline(StandardScaler(),
                              LogisticRegression(max_iter=2000))  # balanced already
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=d)
        pred = cross_val_predict(model, Xs, ys, cv=cv)
        accs.append(balanced_accuracy_score(ys, pred))
    return float(np.mean(accs)), float(np.std(accs))


def shuffled_null_balacc(X, yv, n_draws=50, seed0=100):
    """Permute labels, full-data class-weighted CV balacc. Should be ~0.5."""
    accs = []
    for d in range(n_draws):
        rng = np.random.default_rng(seed0 + d)
        yp = rng.permutation(yv)
        accs.append(full_balacc(X, yp, seed=d))
    return float(np.mean(accs)), float(np.std(accs))


# --- headline reproduction (class-weighted, full data) ---
out["full_balacc_spatial_endler"] = full_balacc(spat_endl, y)
out["full_balacc_spatial"] = full_balacc(spat, y)
out["full_balacc_endler"] = full_balacc(endl, y)

# --- (a) balanced subsample ---
m, s = balanced_subsample_balacc(spat_endl, y, n_draws=50)
out["balanced_subsample_spatial_endler"] = {"mean": m, "std": s}
m2, s2 = balanced_subsample_balacc(spat, y, n_draws=50)
out["balanced_subsample_spatial"] = {"mean": m2, "std": s2}

# --- (b) shuffled null ---
mn, sn = shuffled_null_balacc(spat_endl, y, n_draws=50)
out["shuffled_null_spatial_endler"] = {"mean": mn, "std": sn}

# --- (c) count vs spacing transparency ---
proj = LinearDiscriminantAnalysis().fit(spat_endl, y).transform(spat_endl)[:, 0]
out["lda_corr"] = {
    "count": float(abs(np.corrcoef(proj, count)[0, 1])),
    "spacing": float(abs(np.corrcoef(proj, spacing)[0, 1])),
    "width": float(abs(np.corrcoef(proj, width)[0, 1])),
}


def auc_sign(yv, x):
    a = roc_auc_score(yv, x)
    return float(max(a, 1 - a))


# count-AUC of the LDA projection, baseline
out["count_auc_baseline"] = auc_sign(y, proj)


# regress spacing/width out of the projection -> does count signal survive?
def residualize(target, nuis):
    Z = np.column_stack([np.ones_like(nuis[0])] + list(nuis))
    beta, *_ = np.linalg.lstsq(Z, target, rcond=None)
    return target - Z @ beta


proj_no_spacing = residualize(proj, [spacing])
proj_no_both = residualize(proj, [spacing, width])
out["count_auc_minus_spacing"] = auc_sign(y, proj_no_spacing)
out["count_auc_minus_both"] = auc_sign(y, proj_no_both)

# can spacing/width ALONE predict count? (should be ~chance)
out["spacing_alone_count_auc"] = auc_sign(y, spacing)
out["width_alone_count_auc"] = auc_sign(y, width)

# param-level orthogonality
out["param_corr_count_spacing"] = float(np.corrcoef(count, spacing)[0, 1])

# --- prior-best continuous reference (supervised coeff morphospace ~0.925) ---
# We compare segmentation balanced-subsample honest separability to that headline.
out["prior_continuous_supervised_corr"] = 0.925

print(json.dumps(out, indent=None))
