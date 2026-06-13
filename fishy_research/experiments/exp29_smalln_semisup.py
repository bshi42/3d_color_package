"""EXP-29 — Small-n regime + the minimal-supervision escape hatch.

The team proved (C8): subtle OVERLAPPING structure (stripe 4-vs-5) and rare minorities
(cheeks) are supervised-recoverable (~0.93 / ~0.77) but NOT unsupervised-discoverable
(no density gap). The honest consequence for SMALL real datasets (n=25-31) is: pure
unsupervised discovery of these factors is provably impossible, so the practical question
becomes — *how little expert input flips them from invisible to recoverable?*

This experiment quantifies the **semi-supervised learning curve**: a biologist annotates a
trait on m specimens; we recover it on the rest. We compare:
  (a) pure unsupervised baseline (GMM+BIC auto-cluster, best-matching-cluster balanced acc),
  (b) supervised-from-m-labels (LDA / LogReg), and
  (c) semi-supervised label-spreading (uses the unlabeled geometry too),
as a function of m, with 25 random annotation draws, for belly (easy), stripe (hard/subtle),
cheeks (rare). Run at full n=250 AND subsampled to n=40 (the real small-n regime).

All evaluation is on HELD-OUT specimens the annotator never saw → leakage-free.
"""
from __future__ import annotations

import json
import warnings

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.semi_supervised import LabelSpreading

from fishpipe import data, spectral

warnings.filterwarnings("ignore")
RNG = np.random.default_rng(0)


def descriptor(fcd):
    """Their best general label-free descriptor: signed GFT coeffs k=40 (L,a,b)."""
    return spectral.spectral_coeffs(fcd, k=40, channels=("L", "a", "b"))


def unsup_balacc(Xs, y, max_k=6, seed=0):
    """Pure-unsupervised ceiling: GMM+BIC clustering, then assign each cluster to the
    majority TRUE class (oracle cluster->label map) and score balanced accuracy. This is
    GENEROUS to unsupervised (uses labels only to name clusters, the best case)."""
    best, best_bic = None, np.inf
    for k in range(2, max_k + 1):
        gm = GaussianMixture(k, covariance_type="full", random_state=seed).fit(Xs)
        b = gm.bic(Xs)
        if b < best_bic:
            best_bic, best = b, gm
    cl = best.predict(Xs)
    # map each cluster to majority true label
    pred = np.zeros_like(y)
    for c in np.unique(cl):
        m = cl == c
        pred[m] = np.bincount(y[m]).argmax()
    return balanced_accuracy_score(y, pred)


def semisup_curve(X, y, ms, n_draws=25, seed=0):
    """For each m, draw a stratified labeled set, evaluate on the rest. Returns dict
    method -> (len(ms),) mean balanced accuracy on held-out."""
    rng = np.random.default_rng(seed)
    Xs = StandardScaler().fit_transform(X)
    classes = np.unique(y)
    out = {"lda": [], "logreg": [], "labelspread": []}
    for m in ms:
        accs = {k: [] for k in out}
        for _ in range(n_draws):
            # stratified labeled set: at least 1 per class, total ~m
            lab_idx = []
            for c in classes:
                ci = np.where(y == c)[0]
                take = max(1, int(round(m * len(ci) / len(y))))
                take = min(take, len(ci) - 1)  # leave >=1 held out per class
                lab_idx.extend(rng.choice(ci, size=take, replace=False))
            lab_idx = np.array(sorted(set(lab_idx)))
            test_idx = np.array([i for i in range(len(y)) if i not in set(lab_idx)])
            if len(np.unique(y[lab_idx])) < 2 or len(np.unique(y[test_idx])) < 2:
                continue
            # (b) supervised from m labels
            for name, clf in [
                ("lda", LinearDiscriminantAnalysis()),
                ("logreg", LogisticRegression(max_iter=2000, class_weight="balanced")),
            ]:
                try:
                    clf.fit(Xs[lab_idx], y[lab_idx])
                    accs[name].append(balanced_accuracy_score(y[test_idx], clf.predict(Xs[test_idx])))
                except Exception:
                    pass
            # (c) semi-supervised: unlabeled get -1, label-spreading over full geometry
            yy = -np.ones(len(y), dtype=int)
            yy[lab_idx] = y[lab_idx]
            try:
                ls = LabelSpreading(kernel="knn", n_neighbors=7, alpha=0.2)
                ls.fit(Xs, yy)
                accs["labelspread"].append(
                    balanced_accuracy_score(y[test_idx], ls.transduction_[test_idx])
                )
            except Exception:
                pass
        for k in out:
            out[k].append(float(np.mean(accs[k])) if accs[k] else float("nan"))
    return {k: np.array(v) for k, v in out.items()}


def main():
    fcd = data.build_face_colors(verbose=False)
    gt = data.load_ground_truth()
    X = descriptor(fcd)
    Xs_full = StandardScaler().fit_transform(X)

    factors = ["belly", "stripe", "cheeks"]
    report = {}

    for n_total in (250, 40):
        if n_total < 250:
            # subsample preserving prevalence per factor is impossible jointly; subsample
            # at random but keep enough minority for the rare factor by stratifying on cheeks
            idx = []
            ch = gt.labels["cheeks"]
            n_ch = max(3, int(round(n_total * ch.mean())))
            idx.extend(RNG.choice(np.where(ch == 1)[0], size=min(n_ch, ch.sum()), replace=False))
            rest = np.where(ch == 0)[0]
            idx.extend(RNG.choice(rest, size=n_total - len(idx), replace=False))
            idx = np.array(sorted(idx))
        else:
            idx = np.arange(250)

        Xn = X[idx]
        print(f"\n{'='*72}\nn_total = {len(idx)}\n{'='*72}")
        report[n_total] = {}
        ms = [5, 10, 20, 40] if n_total >= 100 else [4, 8, 12, 20]
        for f in factors:
            y = gt.labels[f][idx]
            if len(np.unique(y)) < 2 or y.sum() < 2:
                print(f"  {f}: too few minority at this n, skip")
                continue
            Xs = StandardScaler().fit_transform(Xn)
            uns = unsup_balacc(Xs, y)
            curve = semisup_curve(Xn, y, ms, n_draws=25)
            report[n_total][f] = {
                "prevalence": float(y.mean()),
                "unsupervised_ceiling": uns,
                "ms": ms,
                **{k: v.tolist() for k, v in curve.items()},
            }
            print(f"\n  factor={f}  (prev={y.mean():.2f})   unsup GMM+BIC ceiling (oracle-named) = {uns:.3f}")
            print(f"    {'m labels':>10} " + " ".join(f"{m:>7d}" for m in ms))
            for meth in ("lda", "logreg", "labelspread"):
                print(f"    {meth:>10} " + " ".join(f"{v:7.3f}" for v in curve[meth]))

    out = data.config.RESULTS_DIR / "exp29"
    out.mkdir(exist_ok=True)
    (out / "smalln_semisup.json").write_text(json.dumps(report, indent=2))
    print(f"\nsaved -> {out/'smalln_semisup.json'}")


if __name__ == "__main__":
    main()
