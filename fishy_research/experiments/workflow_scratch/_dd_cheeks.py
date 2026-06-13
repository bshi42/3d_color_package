"""DEEP-DIVE: recover the rare rosy-cheeks minority (14/250) via segmentation.

Methods:
  (a) high-K palette segmentation (K in 12,16,24,32): does a dedicated red bin appear
      (centroid a*>30)? cheeks recoverability from blob total_area/count over all bins.
  (b) per_specimen=True segmentation (red isolated locally then matched to shared palette).
  (c) rare-color detector: per-face chroma-novelty vs per-face population median;
      area of high-novelty faces (a few compact scalar features).
  (d) spatial.spatial_descriptor on the reddest bin.

For each: cheeks recoverability (balanced_acc, logistic CV) AND honest checks:
  - balanced-subsample balacc: equalize 14 pos vs 14 sampled neg, CV balacc, avg over draws.
  - column-shuffled null balacc: permute labels, recompute full-data balacc, avg over draws (~0.5).
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")
import json
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis  # noqa
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from fishpipe import data, segment, blobs, spatial, metrics
from fishpipe.features import rgb_to_lab
from fishpipe.structure import principal_axis

RNG = np.random.default_rng(0)


def _logit_pipe():
    return make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=2000, class_weight="balanced"))


def cv_balacc(X, y, seed=0):
    """Stratified 5-fold (capped by minority count) balanced acc, full data."""
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        return float("nan")
    n_splits = int(min(5, counts.min()))
    if n_splits < 2:
        return float("nan")
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    pred = cross_val_predict(_logit_pipe(), X, y, cv=cv)
    return float(balanced_accuracy_score(y, pred))


def balanced_subsample_balacc(X, y, n_draws=25, seed=0):
    """Honest check 1: equalize classes (14 pos vs 14 sampled neg), CV balacc, avg over draws.

    Removes the prevalence prior that class_weight=balanced can exploit. With 14 vs 14,
    chance is genuinely 0.5 and the classifier must separate on features alone.
    """
    rng = np.random.default_rng(seed)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    npos = len(pos)
    accs = []
    for d in range(n_draws):
        sel_neg = rng.choice(neg, size=npos, replace=False)
        idx = np.concatenate([pos, sel_neg])
        Xs, ys = X[idx], y[idx]
        # 5-fold on 14v14 -> at least 2 per class per fold
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1000 + d)
        pred = cross_val_predict(_logit_pipe(), Xs, ys, cv=cv)
        accs.append(balanced_accuracy_score(ys, pred))
    return float(np.mean(accs)), float(np.std(accs))


def shuffled_null_balacc(X, y, n_draws=25, seed=0):
    """Honest check 2: permute labels, recompute full-data CV balacc, avg over draws (~0.5)."""
    rng = np.random.default_rng(seed + 7)
    accs = []
    for d in range(n_draws):
        yp = rng.permutation(y)
        accs.append(cv_balacc(X, yp, seed=2000 + d))
    return float(np.mean(accs)), float(np.std(accs))


def evaluate(name, X, y):
    if X.ndim == 1:
        X = X[:, None]
    full = cv_balacc(X, y)
    bsub_m, bsub_s = balanced_subsample_balacc(X, y)
    null_m, null_s = shuffled_null_balacc(X, y)
    return {"method": name, "ncols": int(X.shape[1]),
            "full_balacc": round(full, 3),
            "bal_subsample_balacc": round(bsub_m, 3), "bal_subsample_std": round(bsub_s, 3),
            "null_balacc": round(null_m, 3), "null_std": round(null_s, 3)}


def main():
    gt = data.load_ground_truth()
    fcd = data.build_face_colors(verbose=False)
    y = gt.labels["cheeks"].astype(int)
    print(f"# cheeks: {int(y.sum())} pos / {len(y)}", flush=True)

    results = []

    # ---------------- (a) high-K palette ----------------
    print("\n=== (a) high-K palette ===", flush=True)
    for K in (12, 16, 24, 32):
        seg = segment.segment(fcd, n_colors=K, smooth_iters=1)
        a_star = seg.centroids_lab[:, 1]
        b_star = seg.centroids_lab[:, 2]
        reddest = int(np.argmax(a_star))
        red_bin_exists = bool(a_star.max() > 30)
        bd = blobs.blob_descriptor(fcd, seg, min_area_frac=0.0005)
        # full blob feature stack across all bins
        r_stack = evaluate(f"a_K{K}_blobfeatures", bd["features"], y)
        # the single reddest bin's area+count (2 cols) -- the targeted signal
        red2 = np.column_stack([bd["total_area"][:, reddest], bd["count"][:, reddest]])
        r_red = evaluate(f"a_K{K}_reddest_area+count", red2, y)
        for r in (r_stack, r_red):
            r["reddest_a"] = round(float(a_star.max()), 1)
            r["red_bin_exists_a>30"] = red_bin_exists
            r["reddest_bin"] = reddest
            r["reddest_ab"] = [round(float(a_star[reddest]), 1), round(float(b_star[reddest]), 1)]
        results += [r_stack, r_red]
        print(f"K={K:2d} reddest a*={a_star.max():5.1f} (a>30:{red_bin_exists}) "
              f"blobfeat full={r_stack['full_balacc']} bsub={r_stack['bal_subsample_balacc']} "
              f"null={r_stack['null_balacc']} | red2 full={r_red['full_balacc']} "
              f"bsub={r_red['bal_subsample_balacc']} null={r_red['null_balacc']}", flush=True)

    # ---------------- (b) per_specimen segmentation ----------------
    print("\n=== (b) per_specimen segmentation ===", flush=True)
    for K in (12, 24):
        seg = segment.segment(fcd, n_colors=K, smooth_iters=1, per_specimen=True)
        a_star = seg.centroids_lab[:, 1]
        reddest = int(np.argmax(a_star))
        bd = blobs.blob_descriptor(fcd, seg, min_area_frac=0.0005)
        r_stack = evaluate(f"b_perspec_K{K}_blobfeatures", bd["features"], y)
        red2 = np.column_stack([bd["total_area"][:, reddest], bd["count"][:, reddest]])
        r_red = evaluate(f"b_perspec_K{K}_reddest_area+count", red2, y)
        for r in (r_stack, r_red):
            r["reddest_a"] = round(float(a_star.max()), 1)
            r["red_bin_exists_a>30"] = bool(a_star.max() > 30)
        results += [r_stack, r_red]
        print(f"K={K:2d} reddest a*={a_star.max():5.1f} "
              f"blobfeat full={r_stack['full_balacc']} bsub={r_stack['bal_subsample_balacc']} "
              f"null={r_stack['null_balacc']} | red2 full={r_red['full_balacc']} "
              f"bsub={r_red['bal_subsample_balacc']} null={r_red['null_balacc']}", flush=True)

    # ---------------- (c) rare-color novelty detector ----------------
    # per-face chroma (a*,b*) novelty vs per-face population median; area of high-novelty faces.
    print("\n=== (c) rare-color chroma-novelty detector ===", flush=True)
    lab = rgb_to_lab(fcd.colors)            # (N,Nf,3)
    ab = lab[..., 1:]                       # chroma channels only -> ignores stripes/lightness
    med = np.median(ab, axis=0)             # (Nf,2) per-face population median
    nov = np.linalg.norm(ab - med[None], axis=2)   # (N,Nf) per-face chroma novelty
    areas = fcd.areas
    tot = areas.sum()
    # redness-directed novelty: only count faces that are novel AND red-shifted (a* above pop median)
    a_dev = lab[..., 1] - med[:, 0][None]   # (N,Nf) signed a* deviation (red = positive)
    feats_c = [
        nov.max(axis=1),                                            # max chroma novelty
        np.percentile(nov, 99, axis=1),                             # robust top novelty
        (nov * areas[None]).sum(axis=1) / tot,                      # area-weighted mean novelty
    ]
    for t in (15.0, 25.0, 40.0):
        feats_c.append(((nov > t) * areas[None]).sum(axis=1) / tot)         # area frac novel
        feats_c.append((((nov > t) & (a_dev > 0)) * areas[None]).sum(1) / tot)  # area frac red-novel
    Xc = np.column_stack(feats_c)
    r_c = evaluate("c_chroma_novelty_area", Xc, y)
    results.append(r_c)
    # also the single best scalar: area fraction of red-novel faces at t=25
    red_novel_25 = (((nov > 25.0) & (a_dev > 0)) * areas[None]).sum(1) / tot
    r_c1 = evaluate("c_rednovel_area_t25_scalar", red_novel_25, y)
    results.append(r_c1)
    print(f"novelty-area full={r_c['full_balacc']} bsub={r_c['bal_subsample_balacc']} "
          f"null={r_c['null_balacc']} | scalar full={r_c1['full_balacc']} "
          f"bsub={r_c1['bal_subsample_balacc']} null={r_c1['null_balacc']}", flush=True)

    # ---------------- (d) spatial descriptor on reddest bin ----------------
    print("\n=== (d) spatial_descriptor on reddest bin ===", flush=True)
    for K in (16, 24):
        seg = segment.segment(fcd, n_colors=K, smooth_iters=1)
        a_star = seg.centroids_lab[:, 1]
        reddest = int(np.argmax(a_star))
        spat = spatial.spatial_descriptor(fcd, seg, min_area_frac=0.0005)  # (N, K*10)
        # slice out the reddest bin's 10 spatial stats
        red_spat = spat[:, reddest * 10:(reddest + 1) * 10]
        r_d = evaluate(f"d_spatial_reddest_K{K}", red_spat, y)
        r_d["reddest_a"] = round(float(a_star.max()), 1)
        results.append(r_d)
        # also full spatial stack
        r_dall = evaluate(f"d_spatial_all_K{K}", spat, y)
        results.append(r_dall)
        print(f"K={K:2d} reddest_spatial(10) full={r_d['full_balacc']} "
              f"bsub={r_d['bal_subsample_balacc']} null={r_d['null_balacc']} | "
              f"all_spatial full={r_dall['full_balacc']} bsub={r_dall['bal_subsample_balacc']} "
              f"null={r_dall['null_balacc']}", flush=True)

    # ---------------- summary ----------------
    print("\n=== SUMMARY (ranked by honest balanced-subsample balacc) ===", flush=True)
    # honest ranking: balanced-subsample balacc, but require null ~0.5 (<=0.58) to be trustworthy
    ranked = sorted(results, key=lambda r: r["bal_subsample_balacc"], reverse=True)
    for r in ranked:
        trust = "OK" if r["null_balacc"] <= 0.58 else "NULL-INFLATED"
        print(f"{r['method']:36s} full={r['full_balacc']:.3f} "
              f"bsub={r['bal_subsample_balacc']:.3f}+-{r['bal_subsample_std']:.3f} "
              f"null={r['null_balacc']:.3f}  [{trust}]", flush=True)

    best = ranked[0]
    print("\n=== BEST METHOD ===", flush=True)
    print(json.dumps(best), flush=True)


if __name__ == "__main__":
    main()
