"""EXP-33 — The minimal-supervision escape: how few expert pairwise judgments turn the
unsupervised-invisible stripe-count split into a real, clusterable density gap?

For the factor that is supervised-recoverable but has NO unsupervised density gap (stripe),
we let m must-link/cannot-link pairs WARP the descriptor space (constraint metric learning),
then run ORDINARY unsupervised GMM clustering in the warped space and score ARI vs the true
split. Curve: ARI vs #constraints, RANDOM vs ACTIVE (uncertainty) pair selection, at n=250
and n=40. Belly (well-separated) is the positive control: needs ~0 constraints.
"""
from __future__ import annotations

import json
import warnings
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from fishpipe import data, spectral, textons, semisup

warnings.filterwarnings("ignore")


def reduce(X, d=30):
    return PCA(n_components=min(d, X.shape[1] - 1, X.shape[0] - 1),
               random_state=0).fit_transform(StandardScaler().fit_transform(X))


def baseline_ari(X, y, k=2):
    Z = StandardScaler().fit_transform(X)
    lab = GaussianMixture(k, covariance_type="full", random_state=0).fit_predict(Z)
    return adjusted_rand_score(y, lab)


def curve(X, y, budgets, selection, n_trials=20, n_dim=4, k=2):
    n = X.shape[0]
    out = []
    for m in budgets:
        aris = []
        for t in range(n_trials):
            rng = np.random.default_rng(1000 * t + m)
            if selection == "random":
                pairs = semisup.random_pairs(n, m, rng)
                ml, cl = semisup._pairs_from_labels(pairs, y)
            else:  # active: grow incrementally, re-warping every chunk
                queried, ml, cl = [], [], []
                L = np.eye(X.shape[1])[:, :n_dim]
                chunk = max(5, m // 5)
                while len(queried) < m:
                    need = min(chunk, m - len(queried))
                    newp = semisup.active_pairs(X, L, need, queried, rng)
                    if not newp:
                        break
                    queried += newp
                    nml, ncl = semisup._pairs_from_labels(newp, y)
                    ml += nml; cl += ncl
                    if ml and cl:
                        L = semisup.learn_warp(X, ml, cl, n_dim=n_dim)
            if not ml or not cl:
                aris.append(0.0); continue
            L = semisup.learn_warp(X, ml, cl, n_dim=n_dim)
            lab = semisup.cluster_in_warp(X, L, k=k)
            aris.append(adjusted_rand_score(y, lab))
        out.append(float(np.mean(aris)))
    return out


def main():
    fcd = data.build_face_colors(verbose=False)
    gt = data.load_ground_truth()

    descs = {
        "GFT-coeffs": reduce(spectral.spectral_coeffs(fcd, k=40, channels=("L", "a", "b"))),
    }
    P = textons.diffusion_operator()
    jet = textons.local_jet(fcd, scales=(2, 4), P=P)
    sc, km = textons.build_codebook(jet, K=128, seed=0)
    descs["texton-BoW"] = reduce(textons.encode(fcd, jet, sc, km, mode="bow"))

    budgets = [10, 20, 30, 50, 80]
    report = {}
    for dname, Xfull in descs.items():
        report[dname] = {}
        for factor in ("stripe", "belly"):
            y = gt.labels[factor]
            base = baseline_ari(Xfull, y)
            rnd = curve(Xfull, y, budgets, "random")
            act = curve(Xfull, y, budgets, "active")
            report[dname][factor] = {"baseline_unsup_ARI": round(base, 3),
                                     "budgets": budgets,
                                     "random_ARI": [round(a, 3) for a in rnd],
                                     "active_ARI": [round(a, 3) for a in act]}
            print(f"\n[{dname}] factor={factor}  (unsup baseline ARI={base:.3f})")
            print(f"    {'#pairs':>8} " + " ".join(f"{m:>6d}" for m in budgets))
            print(f"    {'random':>8} " + " ".join(f"{a:6.3f}" for a in rnd))
            print(f"    {'active':>8} " + " ".join(f"{a:6.3f}" for a in act))

    # small-n: stripe on texton-BoW at n=40
    print("\n--- small-n (n=40), stripe, texton-BoW ---")
    rng = np.random.default_rng(0)
    ch = gt.labels["cheeks"]; y = gt.labels["stripe"]
    idx = np.array(sorted(list(rng.choice(np.where(y == 1)[0], 8, replace=False)) +
                          list(rng.choice(np.where(y == 0)[0], 32, replace=False))))
    Xsub = reduce(textons.encode(
        data.FaceColorData(fcd.colors[idx], fcd.areas, fcd.centroids, [fcd.names[i] for i in idx]),
        textons.local_jet(data.FaceColorData(fcd.colors[idx], fcd.areas, fcd.centroids,
                          [fcd.names[i] for i in idx]), scales=(2, 4), P=P),
        *textons.build_codebook(textons.local_jet(
            data.FaceColorData(fcd.colors[idx], fcd.areas, fcd.centroids, [fcd.names[i] for i in idx]),
            scales=(2, 4), P=P), K=128, seed=0), mode="bow"), d=20)
    ysub = y[idx]
    b40 = [10, 20, 30]
    base40 = baseline_ari(Xsub, ysub)
    act40 = curve(Xsub, ysub, b40, "active", n_trials=20, n_dim=3)
    print(f"    baseline unsup ARI={base40:.3f}")
    print(f"    {'#pairs':>8} " + " ".join(f"{m:>6d}" for m in b40))
    print(f"    {'active':>8} " + " ".join(f"{a:6.3f}" for a in act40))
    report["smalln40_stripe_texton"] = {"baseline": round(base40, 3), "budgets": b40,
                                        "active_ARI": [round(a, 3) for a in act40]}

    out = data.config.RESULTS_DIR / "exp33"
    out.mkdir(exist_ok=True)
    (out / "semisup_escape.json").write_text(json.dumps(report, indent=2))
    print(f"\nsaved -> {out/'semisup_escape.json'}")


if __name__ == "__main__":
    main()
