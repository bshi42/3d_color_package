"""DEEP-DIVE: 'same count, different distribution'.

On fishy the planted stripe factor is COUNT (4 vs 5, 198/52 split). stripe_spacing
and stripe_width are INDEPENDENT continuous nuisances (corr with count ~ -0.03).

Q: does the SPATIAL descriptor (K=8, smooth=1, salient region) recover stripe COUNT
specifically, or does its discriminative projection actually track spacing/width/density
(a descriptor-level entanglement)?

Tests:
 1. Fit supervised LDA from spatial(+endler) -> stripe. Correlate the LDA projection with
    stripe_count vs stripe_spacing vs stripe_width (|pearson|).
 2. Entanglement: linearly regress stripe_spacing OUT of the projection; recompute count-AUC.
    Rise => the spacing component was HURTING count (entanglement); Fall => spacing carried
    genuine count info (i.e. count signal lives partly in density).
 3. Sub-feature block ablation: spatial is per-color [area, 3 spread-eig, anisotropy,
    axial_mean, axial_spread, axial_fftpeak, n_blobs, nn_cv] (10 stats x K). Zero each block
    (across all colors) and measure the drop in stripe recoverability (CV balanced acc + LDA
    count-AUC). Also single-block-only (keep just that block).
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import json
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr

from fishpipe import data, segment, blobs, metrics, recommended, spatial

K = 8
SMOOTH = 1
MIN_AREA_FRAC = 0.0005

# spatial per-color stat layout (order matches spatial_descriptor)
STAT_NAMES = ["area", "spread_eig0", "spread_eig1", "spread_eig2", "anisotropy",
              "axial_mean", "axial_spread", "axial_fftpeak", "n_blobs", "nn_cv"]
N_STATS = len(STAT_NAMES)


def auc_abs(y, x):
    a = roc_auc_score(y, x)
    return float(max(a, 1 - a))


def lda_proj(X, y):
    return LinearDiscriminantAnalysis().fit(X, y).transform(X)[:, 0]


def regress_out(target, nuisance):
    """Remove the linear component of `nuisance` from `target` (1-D each)."""
    n = (nuisance - nuisance.mean())
    t = (target - target.mean())
    beta = (n @ t) / (n @ n)
    return t - beta * n


def main():
    gt = data.load_ground_truth()
    fcd = data.build_face_colors(verbose=False)
    df = gt.params
    y = gt.labels["stripe"]                      # 0=count4, 1=count5
    count = df["stripe_count"].to_numpy().astype(float)
    spacing = df["stripe_spacing"].to_numpy()
    width = df["stripe_width"].to_numpy()

    rmask = recommended.salient_pattern_mask(fcd, 0.80)
    seg = segment.segment(fcd, n_colors=K, smooth_iters=SMOOTH)

    spat = spatial.spatial_descriptor(fcd, seg, region_mask=rmask, min_area_frac=MIN_AREA_FRAC)
    endl = blobs.endler_transitions(fcd, seg)
    spat_z = StandardScaler().fit_transform(spat)
    endl_z = StandardScaler().fit_transform(endl)
    spat_endl = np.concatenate([spat_z, endl_z], axis=1)

    out = {}

    # ---- parameter-level confound sanity ----
    out["param_confound"] = {
        "corr_count_spacing": float(pearsonr(count, spacing)[0]),
        "corr_count_width": float(pearsonr(count, width)[0]),
        "corr_spacing_width": float(pearsonr(spacing, width)[0]),
    }

    # ---- Test 1: LDA projection correlations ----
    def proj_corrs(X, tag):
        p = lda_proj(X, y)
        return {
            "tag": tag,
            "abscorr_count": float(abs(pearsonr(p, count)[0])),
            "abscorr_spacing": float(abs(pearsonr(p, spacing)[0])),
            "abscorr_width": float(abs(pearsonr(p, width)[0])),
            "count_auc": auc_abs(y, p),
        }, p

    res_spat, p_spat = proj_corrs(spat_z, "spatial")
    res_se, p_se = proj_corrs(spat_endl, "spatial+endler")
    out["lda_projection_corr"] = {"spatial": res_spat, "spatial+endler": res_se}

    # ---- Test 2: regress spacing (and width) out of the projection -> count-AUC ----
    def entangle(p, tag):
        base = auc_abs(y, p)
        p_no_sp = regress_out(p, spacing)
        p_no_w = regress_out(p, width)
        p_no_both = regress_out(regress_out(p, spacing), width)
        return {
            "tag": tag,
            "count_auc_base": base,
            "count_auc_minus_spacing": auc_abs(y, p_no_sp),
            "count_auc_minus_width": auc_abs(y, p_no_w),
            "count_auc_minus_both": auc_abs(y, p_no_both),
            "delta_minus_spacing": auc_abs(y, p_no_sp) - base,
            "delta_minus_both": auc_abs(y, p_no_both) - base,
        }

    out["entanglement"] = {
        "spatial": entangle(p_spat, "spatial"),
        "spatial+endler": entangle(p_se, "spatial+endler"),
    }

    # Also: can spacing/width ALONE predict count? (is there hidden info in the nuisance?)
    out["nuisance_predicts_count"] = {
        "spacing_alone_auc": auc_abs(y, spacing),
        "width_alone_auc": auc_abs(y, width),
    }

    # ---- Test 3: sub-feature block ablation ----
    # column index for stat s, color c in spat is c*N_STATS + s
    K_eff = spat.shape[1] // N_STATS
    assert K_eff == K, (spat.shape, K)

    def block_cols(stat_idx):
        return [c * N_STATS + stat_idx for c in range(K)]

    # baseline recoverability on full spatial (CV balanced acc) + LDA count-AUC
    base_bal = metrics.factor_recoverability(spat_z, y)["balanced_acc"]
    base_auc = auc_abs(y, lda_proj(spat_z, y))

    ablate_zero = {}   # zero out one block
    keep_only = {}     # keep only one block
    for s, name in enumerate(STAT_NAMES):
        cols = block_cols(s)
        # zero-out: copy, zero the block columns (z-scored space)
        Xz = spat_z.copy()
        Xz[:, cols] = 0.0
        bal_zero = metrics.factor_recoverability(Xz, y)["balanced_acc"]
        auc_zero = auc_abs(y, lda_proj(Xz, y))
        ablate_zero[name] = {
            "bal_acc": float(bal_zero),
            "drop_bal": float(base_bal - bal_zero),
            "count_auc": float(auc_zero),
            "drop_auc": float(base_auc - auc_zero),
        }
        # keep-only this block
        Xk = spat_z[:, cols]
        bal_k = metrics.factor_recoverability(Xk, y)["balanced_acc"]
        # LDA needs >=1 col; fine
        auc_k = auc_abs(y, lda_proj(Xk, y)) if Xk.shape[1] >= 1 else float("nan")
        keep_only[name] = {"bal_acc": float(bal_k), "count_auc": float(auc_k)}

    out["block_ablation"] = {
        "baseline": {"bal_acc": float(base_bal), "count_auc": float(base_auc)},
        "zero_out_block": ablate_zero,
        "keep_only_block": keep_only,
    }

    # ---- which colors are the "stripe" (dark) colors, for context ----
    out["palette_L_sorted"] = np.round(np.sort(seg.centroids_lab[:, 0]), 1).tolist()
    out["dark_order"] = seg.dark_order.tolist()

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
