"""EXP-58 — classifier/loss/threshold tricks BEYOND plain class_weight='balanced'.

TARGET = cheeks (col 3 of F): RARE (14/250 = 5.6%) and LOCALIZED. Feature space = the demo's CURRENT
Z24 (PCA-24, already standardized). Baseline = LogisticRegression(class_weight='balanced'). We test five
candidate upgrades and report the DELTA in balanced-accuracy and minority-F1 on the HELD-OUT rest, averaged
over many seeds (the only honest unit at 5.6%):

  (B) balanced        : LogisticRegression(class_weight='balanced')              [BASELINE]
  (1) thresh_cv       : balanced + decision-threshold picked by CV on the labeled M to max balanced-acc
  (2) calibrated      : CalibratedClassifierCV(balanced, isotonic/sigmoid, cv) + 0.5 threshold
  (2t) calib+thresh   : calibrated probabilities + CV-tuned threshold
  (3) focal           : iterative sample reweighting ~ (1-p_correct)^gamma on TOP of class_weight (hard-pos focus)
  (4) bal_bagging     : ensemble of plain logistic on class-balanced bootstraps, averaged probs, 0.5 thresh
  (4t) bagging+thresh : bal_bagging probs + CV-tuned threshold
  (5) one_class_lof   : LocalOutlierFactor(novelty) fit on labeled POSITIVES only (used once >=k positives)

Decision rule everywhere: predict positive if p (or score) >= thr.  Threshold tuning uses ONLY the labeled M
via stratified-ish CV (never the held-out test).

Run:  .venv/bin/python experiments/exp58_loss_threshold_tricks.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import LocalOutlierFactor

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results" / "exp58"
RES.mkdir(parents=True, exist_ok=True)
CACHE = RES / "cache.npz"

MS = [40, 80, 120]
SEEDS = 60
C_REG = 0.5


# ----------------------------------------------------------------------------- threshold helper
def best_threshold_cv(make_clf, Xl, yl, rng, n_grid=41):
    """Pick decision threshold maximising mean CV balanced-acc on the labeled set.

    Out-of-fold probabilities -> grid over [0,1]. Falls back to 0.5 if too few positives for CV.
    """
    npos = int(yl.sum())
    if npos < 2:
        return 0.5
    k = min(3, npos)                       # at most #pos folds so each fold can hold a positive
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=int(rng.integers(1 << 30)))
    oof = np.full(len(yl), np.nan)
    for tr, va in skf.split(Xl, yl):
        if len(np.unique(yl[tr])) < 2:
            continue
        clf = make_clf().fit(Xl[tr], yl[tr])
        oof[va] = clf.predict_proba(Xl[va])[:, 1]
    m = ~np.isnan(oof)
    if m.sum() < 2 or len(np.unique(yl[m])) < 2:
        return 0.5
    grid = np.linspace(0.05, 0.95, n_grid)
    scores = [balanced_accuracy_score(yl[m], (oof[m] >= t).astype(int)) for t in grid]
    return float(grid[int(np.argmax(scores))])


# ----------------------------------------------------------------------------- focal-ish reweighting
def focal_fit(Xl, yl, rng, gamma=2.0, rounds=3):
    """Logistic with class_weight='balanced' then up-weight hard (mis-predicted) samples ~ (1-p_true)^gamma.

    Iterative: refit with multiplicative sample_weight that emphasises hard positives. Mild, deterministic.
    """
    n = len(yl)
    base = np.where(yl == 1, n / (2.0 * max(yl.sum(), 1)), n / (2.0 * max((yl == 0).sum(), 1)))
    w = base.copy()
    clf = LogisticRegression(C=C_REG, max_iter=3000)
    for _ in range(rounds):
        clf.fit(Xl, yl, sample_weight=w)
        p = clf.predict_proba(Xl)[:, 1]
        p_true = np.where(yl == 1, p, 1 - p)
        w = base * (1.0 + (1.0 - p_true) ** gamma)
    clf.fit(Xl, yl, sample_weight=w)
    return clf


# ----------------------------------------------------------------------------- balanced bagging
class BalancedBagging:
    def __init__(self, n_est=25, rng=None):
        self.n_est = n_est
        self.rng = rng or np.random.default_rng(0)
        self.models = []

    def fit(self, X, y):
        pos = np.where(y == 1)[0]
        neg = np.where(y == 0)[0]
        npos = len(pos)
        self.models = []
        for _ in range(self.n_est):
            bp = self.rng.choice(pos, npos, replace=True)
            bn = self.rng.choice(neg, npos, replace=True)        # undersample majority to balance
            idx = np.concatenate([bp, bn])
            m = LogisticRegression(C=C_REG, max_iter=2000).fit(X[idx], y[idx])
            self.models.append(m)
        return self

    def predict_proba(self, X):
        ps = np.mean([m.predict_proba(X)[:, 1] for m in self.models], axis=0)
        return np.column_stack([1 - ps, ps])


# ----------------------------------------------------------------------------- one-class on positives
def one_class_scores(Xl, yl, Xte, k=5):
    """LOF novelty fit on labeled POSITIVES; return per-test novelty prob proxy in [0,1]."""
    Xp = Xl[yl == 1]
    if len(Xp) < max(3, k + 1):
        return None
    lof = LocalOutlierFactor(n_neighbors=min(k, len(Xp) - 1), novelty=True)
    lof.fit(Xp)
    s = lof.decision_function(Xte)        # higher = more inlier (more like a positive)
    # squash to [0,1] for thresholding
    return 1.0 / (1.0 + np.exp(-(s - np.median(s))))


# ----------------------------------------------------------------------------- one run
def run_seed(Z, y, M, method, seed):
    rng = np.random.default_rng(seed)
    lab = rng.choice(len(y), M, replace=False)
    te = np.setdiff1d(np.arange(len(y)), lab)
    Xl, yl = Z[lab], y[lab]
    Xte, yte = Z[te], y[te]
    npos = int(yl.sum())
    if npos < 1 or len(np.unique(yl)) < 2:
        return None                       # degenerate seed: no positive labeled -> skip (same for all methods)

    def mk_bal():
        return LogisticRegression(C=C_REG, class_weight="balanced", max_iter=3000)

    if method == "balanced":
        clf = mk_bal().fit(Xl, yl)
        p = clf.predict_proba(Xte)[:, 1]; thr = 0.5
    elif method == "thresh_cv":
        clf = mk_bal().fit(Xl, yl)
        p = clf.predict_proba(Xte)[:, 1]
        thr = best_threshold_cv(mk_bal, Xl, yl, rng)
    elif method == "calibrated":
        thr = 0.5
        if npos >= 2:
            cv = min(3, npos)
            cc = CalibratedClassifierCV(mk_bal(), method="sigmoid", cv=cv)
            try:
                cc.fit(Xl, yl); p = cc.predict_proba(Xte)[:, 1]
            except Exception:
                p = mk_bal().fit(Xl, yl).predict_proba(Xte)[:, 1]
        else:
            p = mk_bal().fit(Xl, yl).predict_proba(Xte)[:, 1]
    elif method == "calib_thresh":
        if npos >= 2:
            cv = min(3, npos)
            try:
                cc = CalibratedClassifierCV(mk_bal(), method="sigmoid", cv=cv).fit(Xl, yl)
                p = cc.predict_proba(Xte)[:, 1]
                # tune threshold on labeled via same calibrated estimator (CV inside)
                thr = best_threshold_cv(
                    lambda: CalibratedClassifierCV(mk_bal(), method="sigmoid", cv=min(2, npos)), Xl, yl, rng)
            except Exception:
                clf = mk_bal().fit(Xl, yl); p = clf.predict_proba(Xte)[:, 1]
                thr = best_threshold_cv(mk_bal, Xl, yl, rng)
        else:
            clf = mk_bal().fit(Xl, yl); p = clf.predict_proba(Xte)[:, 1]; thr = 0.5
    elif method == "focal":
        clf = focal_fit(Xl, yl, rng)
        p = clf.predict_proba(Xte)[:, 1]; thr = 0.5
    elif method == "bal_bagging":
        bb = BalancedBagging(n_est=25, rng=rng).fit(Xl, yl)
        p = bb.predict_proba(Xte)[:, 1]; thr = 0.5
    elif method == "bagging_thresh":
        bb = BalancedBagging(n_est=25, rng=rng).fit(Xl, yl)
        p = bb.predict_proba(Xte)[:, 1]
        thr = best_threshold_cv(lambda: BalancedBagging(n_est=15, rng=rng), Xl, yl, rng)
    elif method == "one_class_lof":
        s = one_class_scores(Xl, yl, Xte)
        if s is None:                     # not enough positives -> fall back to baseline
            clf = mk_bal().fit(Xl, yl); p = clf.predict_proba(Xte)[:, 1]; thr = 0.5
        else:
            p = s
            # tune threshold on labeled positives+sample of negatives
            sl = one_class_scores(Xl, yl, Xl)
            if sl is not None and len(np.unique(yl)) > 1:
                grid = np.linspace(0.05, 0.95, 41)
                sc = [balanced_accuracy_score(yl, (sl >= t).astype(int)) for t in grid]
                thr = float(grid[int(np.argmax(sc))])
            else:
                thr = 0.5
    else:
        raise ValueError(method)

    pred = (p >= thr).astype(int)
    return (
        balanced_accuracy_score(yte, pred),
        f1_score(yte, pred, pos_label=1, zero_division=0),
        recall_score(yte, pred, pos_label=1, zero_division=0),
    )


METHODS = ["balanced", "thresh_cv", "calibrated", "calib_thresh", "focal",
           "bal_bagging", "bagging_thresh", "one_class_lof"]
LABELS = {
    "balanced": "class_weight=balanced (BASE)",
    "thresh_cv": "+ threshold-CV",
    "calibrated": "+ calibration (sigmoid)",
    "calib_thresh": "+ calib + threshold-CV",
    "focal": "+ focal reweight",
    "bal_bagging": "balanced bagging",
    "bagging_thresh": "balanced bagging + thr-CV",
    "one_class_lof": "one-class LOF (pos-only)",
}


def main():
    d = np.load(CACHE, allow_pickle=True)
    Z = d["Z24"].astype("float32")
    y = d["F"][:, 3].astype(int)          # cheeks
    print(f"Z24 {Z.shape}  cheeks positives {int(y.sum())}/{len(y)} = {y.mean():.3%}")
    print(f"SEEDS={SEEDS}  metric = balanced-acc | F1 | recall on held-out, mean over valid seeds\n")

    # results[method][M] = (bal, f1, rec, n_valid)
    res = {m: {} for m in METHODS}
    for M in MS:
        print(f"=== M={M} (expected ~{M*y.mean():.1f} positives labeled) ===")
        print("  method                         |  bal-acc   F1     recall   (Δbal vs base)  nseeds")
        base_bal = None
        for meth in METHODS:
            rows = [run_seed(Z, y, M, meth, 7000 * s + 11) for s in range(SEEDS)]
            rows = [r for r in rows if r is not None]
            n = len(rows)
            bal = float(np.mean([r[0] for r in rows]))
            f1 = float(np.mean([r[1] for r in rows]))
            rec = float(np.mean([r[2] for r in rows]))
            res[meth][M] = (bal, f1, rec, n)
            if meth == "balanced":
                base_bal = bal
            dlt = bal - base_bal
            print(f"  {LABELS[meth]:30s} |  {bal:.3f}  {f1:.3f}  {rec:.3f}    {dlt:+.3f}        {n}")
        print()

    # ---- deltas table (vs balanced) for F1 too
    print("=== DELTA vs baseline (balanced) — balanced-acc / F1 ===")
    for meth in METHODS:
        if meth == "balanced":
            continue
        parts = []
        for M in MS:
            db = res[meth][M][0] - res["balanced"][M][0]
            df = res[meth][M][1] - res["balanced"][M][1]
            parts.append(f"M{M}: dbal={db:+.3f} dF1={df:+.3f}")
        print(f"  {LABELS[meth]:30s} | " + "  ".join(parts))

    # ----------------------------------------------------------------------- plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    xs = np.arange(len(MS))
    cmap = plt.get_cmap("tab10")
    for ax, (mi, mname) in zip(axes, [(0, "balanced-acc"), (1, "minority F1")]):
        width = 0.1
        for i, meth in enumerate(METHODS):
            vals = [res[meth][M][mi] for M in MS]
            ax.bar(xs + (i - len(METHODS) / 2) * width, vals, width,
                   label=LABELS[meth], color=cmap(i % 10))
        ax.set_xticks(xs); ax.set_xticklabels([f"M={M}" for M in MS])
        ax.set_ylabel(mname + " (held-out)")
        ax.set_title(mname)
        ax.grid(alpha=.2, axis="y")
        if mi == 0:
            ax.set_ylim(0.45, max(0.9, max(res[m][MS[-1]][0] for m in METHODS) + 0.05))
            ax.axhline(0.5, color="#bbb", ls=":", lw=.8)
    axes[0].legend(fontsize=7, ncol=1, loc="upper left")
    fig.suptitle(
        f"EXP-58 cheeks (5.6%) on Z24 — loss/threshold tricks vs class_weight='balanced'  "
        f"(mean of {SEEDS} seeds)", y=1.02)
    fig.tight_layout()
    out = RES / "loss_threshold_tricks.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nfigure -> {out}")


if __name__ == "__main__":
    main()
