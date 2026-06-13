"""Flexible evaluator for segmentation-based pattern features (fishy). Emits JSON.

Usage: uv run python experiments/seg_eval.py --K 6 --smooth 1 [--per_specimen] [--region salient]
Builds one segmentation config, computes a suite of segmentation/blob features, and reports
supervised recoverability of every factor + the stripe-count-specific blob/peak scalars + a
column-shuffled null. Designed to be fanned out across configs by a Workflow.
"""
from __future__ import annotations

import argparse
import json
import warnings

warnings.filterwarnings("ignore")
import numpy as np
from sklearn.metrics import roc_auc_score

from fishpipe import data, segment, blobs, metrics, recommended


def auc(y, x):
    a = roc_auc_score(y, x)
    return float(max(a, 1 - a))


def bal(X, y):
    return float(metrics.factor_recoverability(X, y)["balanced_acc"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=6)
    ap.add_argument("--smooth", type=int, default=1)
    ap.add_argument("--per_specimen", action="store_true")
    ap.add_argument("--region", choices=["whole", "salient"], default="whole")
    ap.add_argument("--min_area_frac", type=float, default=0.0005)
    args = ap.parse_args()

    gt = data.load_ground_truth(); fcd = data.build_face_colors(); df = gt.params
    labels = gt.labels
    rmask = recommended.salient_pattern_mask(fcd, 0.80) if args.region == "salient" else None

    from fishpipe import spatial
    from sklearn.preprocessing import StandardScaler

    seg = segment.segment(fcd, n_colors=args.K, smooth_iters=args.smooth, per_specimen=args.per_specimen)
    bd = blobs.blob_descriptor(fcd, seg, min_area_frac=args.min_area_frac, region_mask=rmask)
    endl = blobs.endler_transitions(fcd, seg)
    spat = spatial.spatial_descriptor(fcd, seg, region_mask=rmask, min_area_frac=args.min_area_frac)
    spat_endl = np.concatenate([StandardScaler().fit_transform(b) for b in [spat, endl]], axis=1)

    out = {"config": vars(args), "palette_L": np.round(np.sort(seg.centroids_lab[:, 0]), 1).tolist()}

    # feature-set recoverability (supervised balanced acc)
    fsets = {"blob_features": bd["features"], "blob_count": bd["count"],
             "blob_area": bd["total_area"], "endler": endl,
             "spatial": spat, "spatial+endler": spat_endl}
    out["recoverability"] = {
        fn: {f: bal(X, labels[f]) for f in ["belly", "tail", "stripe", "cheeks"]}
        for fn, X in fsets.items()
    }

    # stripe-count-specific scalars
    stripe_scalars = {}
    for nd in [1, 2, 3]:
        dbc = blobs.dark_blob_count(fcd, seg, n_dark=nd, region_mask=rmask, min_area_frac=args.min_area_frac)
        ap_ = blobs.axial_dark_peaks(fcd, seg, n_dark=nd, region_mask=rmask)
        stripe_scalars[f"dark_blob_count_n{nd}"] = {"auc": auc(labels["stripe"], dbc),
                                                    "mean4": float(dbc[labels["stripe"] == 0].mean()),
                                                    "mean5": float(dbc[labels["stripe"] == 1].mean())}
        stripe_scalars[f"axial_peaks_n{nd}"] = {"auc": auc(labels["stripe"], ap_),
                                                "corr_count": float(np.corrcoef(ap_, df.stripe_count)[0, 1]),
                                                "corr_spacing": float(np.corrcoef(ap_, df.stripe_spacing)[0, 1])}
    out["stripe_scalars"] = stripe_scalars

    # cheeks: best palette-color blob total-area / count (does a rare red bin isolate it?)
    yc = labels["cheeks"]
    a_star = seg.centroids_lab[:, 1]
    out["cheeks"] = {
        "best_color_area_auc": max(auc(yc, bd["total_area"][:, c]) for c in range(args.K)),
        "best_color_count_auc": max(auc(yc, bd["count"][:, c]) for c in range(args.K)),
        "reddest_a": float(a_star.max()),
    }

    # transparency: does the spatial+endler stripe signal encode COUNT or SPACING?
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    try:
        proj = LinearDiscriminantAnalysis().fit(spat_endl, labels["stripe"]).transform(spat_endl)[:, 0]
        out["stripe_transparency"] = {
            "lda_corr_count": float(abs(np.corrcoef(proj, df.stripe_count)[0, 1])),
            "lda_corr_spacing": float(abs(np.corrcoef(proj, df.stripe_spacing)[0, 1])),
            "lda_corr_width": float(abs(np.corrcoef(proj, df.stripe_width)[0, 1])),
        }
    except Exception:
        out["stripe_transparency"] = None

    print(json.dumps(out))


if __name__ == "__main__":
    main()
