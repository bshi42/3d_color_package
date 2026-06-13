"""EXP-31 — Validate the texton-BoW stripe gain (0.969): is it real, seed-stable, measuring
COUNT (not the spacing confound), and does it hold / help at small n + few labels?

Mirrors the team's audit discipline (EXP-06/22/25): seed sweep, shuffled-label null,
count-vs-spacing disentanglement, small-n subsample, semi-supervised label efficiency.
"""
from __future__ import annotations

import json
import warnings
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from fishpipe import data, spectral, textons

warnings.filterwarnings("ignore")


def balacc(X, y, clf="logreg", seed=0):
    n_splits = int(min(5, np.bincount(y).min()))
    if n_splits < 2:
        return float("nan")
    model = make_pipeline(StandardScaler(),
                          LogisticRegression(max_iter=2000, class_weight="balanced")
                          if clf == "logreg" else LDA())
    cv = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
    return balanced_accuracy_score(y, cross_val_predict(model, X, y, cv=cv))


def texton_bow(fcd, K=128, seed=0, idx=None):
    """Texton BoW restricted to specimens `idx` (codebook fit only on those — honest small-n)."""
    P = textons.diffusion_operator()
    sub = fcd if idx is None else data.FaceColorData(
        colors=fcd.colors[idx], areas=fcd.areas, centroids=fcd.centroids,
        names=[fcd.names[i] for i in idx])
    jet = textons.local_jet(sub, scales=(2, 4), P=P)
    scaler, km = textons.build_codebook(jet, K=K, seed=seed)
    return textons.encode(sub, jet, scaler, km, mode="bow")


def main():
    fcd = data.build_face_colors(verbose=False)
    gt = data.load_ground_truth()
    y = gt.labels["stripe"]
    rep = {}

    # 1) seed stability of the stripe gain
    seeds = [0, 1, 2, 3, 4]
    accs = [balacc(texton_bow(fcd, 128, s), y) for s in seeds]
    rep["seed_sweep_stripe_balacc"] = [round(a, 3) for a in accs]
    print(f"1) texton-BoW K=128 stripe balacc over codebook seeds {seeds}: "
          f"{[round(a,3) for a in accs]}  mean={np.mean(accs):.3f}")

    # 2) shuffled-label null
    X = texton_bow(fcd, 128, 0)
    rng = np.random.default_rng(0)
    null = [balacc(X, rng.permutation(y)) for _ in range(5)]
    rep["shuffled_null_stripe_balacc"] = [round(a, 3) for a in null]
    print(f"2) shuffled-label null (real {accs[0]:.3f}): {[round(a,3) for a in null]}  "
          f"mean={np.mean(null):.3f}")

    # 3) count-vs-spacing disentanglement (the team's gold standard, EXP-22/25)
    #    supervised stripe direction; correlate its projection with the true generating params.
    Xs = StandardScaler().fit_transform(X)
    w = LDA().fit(Xs, y).coef_.ravel()
    proj = Xs @ w
    params = gt.params
    def corr(a, b):
        return float(np.corrcoef(a, b)[0, 1])
    c_count = corr(proj, params["stripe_count"].to_numpy())
    c_space = corr(proj, params["stripe_spacing"].to_numpy())
    c_width = corr(proj, params["stripe_width"].to_numpy())
    rep["stripe_dir_corr"] = {"count": round(c_count, 3), "spacing": round(c_space, 3),
                              "width": round(c_width, 3)}
    print(f"3) supervised stripe-direction correlation:  |count|={abs(c_count):.3f}  "
          f"|spacing|={abs(c_space):.3f}  |width|={abs(c_width):.3f}   "
          f"({'measures COUNT' if abs(c_count) > 2*max(abs(c_space),abs(c_width)) else 'entangled'})")

    # 4) small-n: stripe balacc at n=40 (codebook + eval both restricted), texton vs GFT
    print("4) small-n (n=40) stripe balacc, texton-BoW vs GFT coeffs, 8 subsamples:")
    rng = np.random.default_rng(1)
    tb, gf = [], []
    for _ in range(8):
        # keep prevalence-ish: 8 five-stripe + 32 four-stripe
        pos = rng.choice(np.where(y == 1)[0], 8, replace=False)
        neg = rng.choice(np.where(y == 0)[0], 32, replace=False)
        sub = np.sort(np.concatenate([pos, neg]))
        ys = y[sub]
        tb.append(balacc(texton_bow(fcd, 128, 0, idx=sub), ys))
        Xg = spectral.spectral_coeffs(fcd, k=40, channels=("L", "a", "b"))[sub]
        gf.append(balacc(Xg, ys))
    rep["smalln40_stripe"] = {"texton_bow": round(float(np.nanmean(tb)), 3),
                              "gft": round(float(np.nanmean(gf)), 3)}
    print(f"     texton-BoW: {np.nanmean(tb):.3f} ± {np.nanstd(tb):.3f}    "
          f"GFT: {np.nanmean(gf):.3f} ± {np.nanstd(gf):.3f}")

    # 5) label efficiency: held-out balacc vs #labels, texton-BoW vs GFT (stripe)
    print("5) label efficiency (held-out balacc vs #labels), stripe:")
    Xt = texton_bow(fcd, 128, 0)
    Xg = spectral.spectral_coeffs(fcd, k=40, channels=("L", "a", "b"))
    ms = [10, 20, 40, 80]
    eff = {"texton_bow": {}, "gft": {}}
    for name, XX in [("texton_bow", Xt), ("gft", Xg)]:
        Xss = StandardScaler().fit_transform(XX)
        for m in ms:
            rng = np.random.default_rng(7)
            accs2 = []
            for _ in range(25):
                pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
                lab = np.concatenate([rng.choice(pos, max(2, m // 5), replace=False),
                                      rng.choice(neg, m - max(2, m // 5), replace=False)])
                test = np.array([i for i in range(len(y)) if i not in set(lab)])
                clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xss[lab], y[lab])
                accs2.append(balanced_accuracy_score(y[test], clf.predict(Xss[test])))
            eff[name][m] = round(float(np.mean(accs2)), 3)
        print(f"     {name:>11}: " + "  ".join(f"m={m}:{eff[name][m]:.3f}" for m in ms))
    rep["label_efficiency_stripe"] = eff

    out = data.config.RESULTS_DIR / "exp31"
    out.mkdir(exist_ok=True)
    (out / "texton_validate.json").write_text(json.dumps(rep, indent=2))
    print(f"\nsaved -> {out/'texton_validate.json'}")


if __name__ == "__main__":
    main()
