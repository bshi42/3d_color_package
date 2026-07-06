"""EXP-55 — Discovering k from a LIMITED set of EXPERT LABELS (realistic, no GT on the rest).

EXP-54 picked k using the FULL 250-label ground truth. Realistic users only have M expert labels.
Question: pick M random specimens; for each k sweep, run 5-fold CV of a per-factor logistic ON THOSE M
labeled specimens (features = k-mode descriptor restricted to the M rows). Does the knee of the
stripe CV-on-M curve recover the true k~40, and what M is needed?

CRITICAL nuance tested+reported: the MEAN over factors saturates early (belly/tail dominate) and would
mislead a user to a small k. The user must track the SLOWEST factor (stripe).

Data: results/exp55/cache.npz — CL,Ca,Cb (250,300) signed GFT coeffs; F (250,3) GT [belly,tail,stripe].
GT used ONLY to score / label, never as an input feature.

Run:  .venv/bin/python experiments/exp55_label_budget_k.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results" / "exp55"
RES.mkdir(parents=True, exist_ok=True)

KS = [1, 2, 3, 4, 6, 8, 12, 16, 20, 24, 30, 40, 50, 60, 80, 100]
BUDGETS = [20, 40, 80]
N_DRAWS = 12               # >= 8 random M-draws, averaged
FACTORS = ("belly", "tail", "stripe")
SEED = 0


def load():
    d = np.load(RES / "cache.npz", allow_pickle=True)
    return d["CL"].astype(np.float64), d["Ca"].astype(np.float64), d["Cb"].astype(np.float64), d["F"]


def descriptor(CL, Ca, Cb, k, rows):
    """k-mode spectral descriptor restricted to `rows`, standardized using ONLY those rows
    (a user with M labels has no access to the rest, so scaling must be fit on the M)."""
    X = np.concatenate([CL[rows, :k], Ca[rows, :k], Cb[rows, :k]], axis=1)
    return StandardScaler().fit_transform(X)


def cv_on_subset(X, y, n_splits=5):
    """5-fold stratified CV accuracy. Stratified so each fold sees both classes for rare stripe.
    If a class has fewer than n_splits members in this draw, shrink folds to that count (>=2)."""
    counts = np.bincount(y, minlength=2)
    nsp = int(min(n_splits, counts[counts > 0].min()))
    if nsp < 2:
        return np.nan
    skf = StratifiedKFold(n_splits=nsp, shuffle=True, random_state=0)
    return float(cross_val_score(LogisticRegression(max_iter=3000), X, y, cv=skf).mean())


def knee_k(ks, curve):
    """First k whose CV reaches within 1% (abs) of the curve's max, AND >=0.02 above the k=1 value
    (i.e. a real rise). Returns None if no meaningful rise."""
    curve = np.asarray(curve, float)
    if np.nanmax(curve) - curve[0] < 0.02:
        return None
    thr = np.nanmax(curve) - 0.01
    for kk, v in zip(ks, curve):
        if v >= thr:
            return kk
    return ks[-1]


def main():
    rng = np.random.default_rng(SEED)
    CL, Ca, Cb, F = load()
    N = CL.shape[0]
    Fd = {f: F[:, i].astype(int) for i, f in enumerate(FACTORS)}
    base = {f: float(max(Fd[f].mean(), 1 - Fd[f].mean())) for f in FACTORS}  # majority-class baseline
    print("majority baselines:", {f: round(base[f], 3) for f in FACTORS}, "stripe pos frac",
          round(Fd["stripe"].mean(), 3))

    # ---- full-250-label reference (CV on all rows; standardized on all rows) ----
    ref = {f: [] for f in FACTORS}
    allrows = np.arange(N)
    for k in KS:
        for f in FACTORS:
            X = descriptor(CL, Ca, Cb, k, allrows)
            ref[f].append(cv_on_subset(X, Fd[f]))
    print("\n=== FULL-250 reference (CV vs k) ===")
    print("  k  | " + "  ".join(f"{f:>6}" for f in FACTORS) + "  |  mean")
    for i, k in enumerate(KS):
        m = np.nanmean([ref[f][i] for f in FACTORS])
        print(f"{k:4d} | " + "  ".join(f"{ref[f][i]:6.3f}" for f in FACTORS) + f"  |  {m:5.3f}")

    # ---- limited-label budgets ----
    # results[M][f] -> (n_draws, n_k) ; also track mean-over-factors curve per draw
    results = {M: {f: np.full((N_DRAWS, len(KS)), np.nan) for f in FACTORS} for M in BUDGETS}
    mean_curve = {M: np.full((N_DRAWS, len(KS)), np.nan) for M in BUDGETS}
    knees = {M: {f: [] for f in FACTORS} for M in BUDGETS}
    knees_mean = {M: [] for M in BUDGETS}

    for M in BUDGETS:
        for d in range(N_DRAWS):
            rows = rng.choice(N, size=M, replace=False)
            # cache descriptors per k (same rows used across factors within a draw)
            for ik, k in enumerate(KS):
                X = descriptor(CL, Ca, Cb, k, rows)
                for f in FACTORS:
                    results[M][f][d, ik] = cv_on_subset(X, Fd[f][rows])
            # per-draw knees
            for f in FACTORS:
                kk = knee_k(KS, results[M][f][d])
                if kk is not None:
                    knees[M][f].append(kk)
            mean_curve[M][d] = np.nanmean([results[M][f][d] for f in FACTORS], axis=0)
            kkm = knee_k(KS, mean_curve[M][d])
            if kkm is not None:
                knees_mean[M].append(kkm)

    # ---- print budget tables ----
    for M in BUDGETS:
        print(f"\n=== M={M} labels  (mean over {N_DRAWS} draws of 5-fold CV-on-M) ===")
        print("  k  | " + "  ".join(f"{f:>6}" for f in FACTORS) + "  |  mean(factors)")
        for ik, k in enumerate(KS):
            vals = {f: np.nanmean(results[M][f][:, ik]) for f in FACTORS}
            mv = np.nanmean(mean_curve[M][:, ik])
            print(f"{k:4d} | " + "  ".join(f"{vals[f]:6.3f}" for f in FACTORS) + f"  |  {mv:5.3f}")
        for f in FACTORS:
            ks_found = knees[M][f]
            if ks_found:
                print(f"  knee[{f}] median over draws = {int(np.median(ks_found))} "
                      f"(IQR {int(np.percentile(ks_found,25))}-{int(np.percentile(ks_found,75))}), "
                      f"rise-detected in {len(ks_found)}/{N_DRAWS} draws")
            else:
                print(f"  knee[{f}] = no rise detected (curve flat) in any draw")
        if knees_mean[M]:
            print(f"  knee[MEAN-over-factors] median = {int(np.median(knees_mean[M]))}  <-- the MISLEADING signal")

    # reference knees
    ref_knee = {f: knee_k(KS, ref[f]) for f in FACTORS}
    ref_mean_knee = knee_k(KS, np.nanmean([ref[f] for f in FACTORS], axis=0))
    print("\nFULL-250 reference knees:", ref_knee, " mean-curve knee:", ref_mean_knee)

    # ---------------- figure ----------------
    cm = {"belly": "#4e79a7", "tail": "#59a14f", "stripe": "#e15759"}
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.9), sharey=True)

    # panels 0..2 : per-budget, all factors + mean, with full-250 stripe ref dashed
    for j, M in enumerate(BUDGETS):
        ax = axes[j]
        for f in FACTORS:
            mu = np.nanmean(results[M][f], axis=0)
            sd = np.nanstd(results[M][f], axis=0)
            ax.plot(KS, mu, "o-", color=cm[f], label=f, ms=4)
            ax.fill_between(KS, mu - sd, mu + sd, color=cm[f], alpha=0.12)
        ax.plot(KS, np.nanmean(mean_curve[M], axis=0), "s--", color="#7b3fa0",
                label="mean(factors)", ms=4, lw=1.4)
        ax.plot(KS, ref["stripe"], ":", color="#e15759", alpha=0.55, lw=2,
                label="stripe (full-250 ref)")
        ax.axvline(40, color="#999", ls="--", lw=1)
        ax.text(41, 0.52, "true k~40", color="#777", fontsize=8, rotation=90, va="bottom")
        for f in FACTORS:
            ax.axhline(base[f], color=cm[f], ls=":", lw=0.8, alpha=0.5)
        ax.set_xscale("log")
        ax.set_xlabel("k = # GFT modes per channel (log)")
        ax.set_title(f"M = {M} expert labels")
        ax.grid(alpha=0.2)
        if j == 0:
            ax.set_ylabel("5-fold CV accuracy (on the M labels)")
        ax.set_ylim(0.5, 1.02)
        ax.legend(fontsize=7, loc="lower right")

    # panel 3 : the stripe curve at each budget vs the full-250 reference (the headline)
    ax = axes[3]
    bud_c = {20: "#f1a340", 40: "#d95f02", 80: "#8c2d04"}
    for M in BUDGETS:
        mu = np.nanmean(results[M]["stripe"], axis=0)
        sd = np.nanstd(results[M]["stripe"], axis=0)
        ax.plot(KS, mu, "o-", color=bud_c[M], label=f"stripe, M={M}", ms=4)
        ax.fill_between(KS, mu - sd, mu + sd, color=bud_c[M], alpha=0.12)
    ax.plot(KS, ref["stripe"], "k-", lw=2, label="stripe, full-250")
    ax.axvline(40, color="#999", ls="--", lw=1)
    ax.text(41, 0.52, "true k~40", color="#777", fontsize=8, rotation=90, va="bottom")
    ax.axhline(base["stripe"], color="#888", ls=":", lw=0.9, label=f"majority {base['stripe']:.2f}")
    ax.set_xscale("log")
    ax.set_xlabel("k = # GFT modes per channel (log)")
    ax.set_title("hardest factor (stripe): budget vs full-GT")
    ax.grid(alpha=0.2)
    ax.set_ylim(0.5, 1.02)
    ax.legend(fontsize=7, loc="lower right")

    fig.suptitle("EXP-55: discovering k from M expert labels — track the SLOWEST factor (stripe), "
                 "not the mean", y=1.02, fontsize=12)
    fig.tight_layout()
    out = RES / "label_budget_k.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nfigure -> {out}")

    # ---- summary numbers for the writeup ----
    print("\n=== SUMMARY ===")
    for M in BUDGETS:
        st = np.nanmean(results[M]["stripe"], axis=0)
        k1, k40, kbest = st[0], st[KS.index(40)], np.nanmax(st)
        kbest_k = KS[int(np.nanargmax(st))]
        skm = int(np.median(knees_mean[M])) if knees_mean[M] else None
        sks = int(np.median(knees[M]["stripe"])) if knees[M]["stripe"] else None
        print(f"M={M:3d}: stripe CV  k=1 {k1:.3f} -> k=40 {k40:.3f} (best {kbest:.3f}@k={kbest_k}); "
              f"stripe-knee~{sks}  vs  MEAN-knee~{skm}")


if __name__ == "__main__":
    main()
