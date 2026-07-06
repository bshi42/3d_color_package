"""EXP-56 — Class-imbalance handling for rare factors (stripe 21%, rosy cheeks 5.6%) while incrementally
tagging. Does reweighting / oversampling / SMOTE recover the rare factor with fewer labels?

Methods (implemented directly; imblearn's SMOTE/ROS are these, but its sklearn pin conflicts w/ sklearn 1.8):
  plain      — LogisticRegression(C=0.5)
  balanced   — class_weight='balanced' (inverse-frequency reweighting)
  smote      — synthetic minority oversampling (kNN interpolation) to balance, then plain
  strat-acq* — ORACLE: acquire the M labels class-stratified (minority guaranteed) + balanced — shows that
               *getting* minority examples beats reweighting (active learning approximates this).
Metric: balanced accuracy + minority recall on the HELD-OUT rest, mean over many seeds (rare class -> noisy).
Feature space = the demo's PCA-24 (engine Z).

Run:  .venv/bin/python experiments/exp56_imbalance.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, recall_score
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "interactive_demo"))
import engine  # noqa: E402

RES = HERE.parent / "results" / "exp56"
RES.mkdir(parents=True, exist_ok=True)
MS = [20, 30, 40, 60, 80, 120, 160]
SEEDS = 40


def smote_balance(X, y, rng, k=5):
    out_X, out_y = [X], [y]
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        need = (y != c).sum() - len(idx) if (y != c).sum() > len(idx) else 0
        if need <= 0 or len(idx) < 2:
            if need > 0 and len(idx) == 1:                  # 1 sample -> just duplicate
                out_X.append(np.repeat(X[idx], need, 0)); out_y.append(np.full(need, c))
            continue
        Xc = X[idx]
        D = np.linalg.norm(Xc[:, None] - Xc[None], axis=2); np.fill_diagonal(D, np.inf)
        nn = np.argsort(D, 1)[:, :min(k, len(idx) - 1)]
        a = rng.integers(0, len(idx), need); b = nn[a, rng.integers(0, nn.shape[1], need)]
        lam = rng.random((need, 1))
        out_X.append(Xc[a] + lam * (Xc[b] - Xc[a])); out_y.append(np.full(need, c))
    return np.vstack(out_X), np.concatenate(out_y)


def run(Z, y, M, method, rng):
    N = len(y)
    if method == "strat-acq":                               # oracle stratified acquisition
        pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
        npos = max(1, int(round(M * y.mean())))
        lab = np.concatenate([rng.choice(pos, min(npos, len(pos)), replace=False),
                              rng.choice(neg, M - min(npos, len(pos)), replace=False)])
    else:
        lab = rng.choice(N, M, replace=False)
    te = np.setdiff1d(np.arange(N), lab)
    Xl, yl = Z[lab], y[lab]
    if len(np.unique(yl)) < 2:
        return 0.5, 0.0                                     # no minority seen -> predict majority
    if method == "balanced" or method == "strat-acq":
        clf = LogisticRegression(C=0.5, class_weight="balanced", max_iter=2000)
    elif method == "smote":
        Xl, yl = smote_balance(Xl, yl, rng); clf = LogisticRegression(C=0.5, max_iter=2000)
    else:
        clf = LogisticRegression(C=0.5, max_iter=2000)
    p = clf.fit(Xl, yl).predict(Z[te])
    return balanced_accuracy_score(y[te], p), recall_score(y[te], p, pos_label=1, zero_division=0)


def main():
    ld = engine.load_dataset("fishy")
    Z = engine._zscore_cols(PCA(24, random_state=0).fit_transform(StandardScaler().fit_transform(ld.Xs))).astype("float32")
    factors = {"stripe (21%)": np.asarray(ld.gt.labels["stripe"]),
               "cheeks (5.6%)": np.asarray(ld.gt.labels["cheeks"])}
    methods = ["plain", "balanced", "smote", "strat-acq"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, (fname, y) in zip(axes, factors.items()):
        print(f"\n=== {fname} ===  (balanced accuracy on held-out, mean of {SEEDS} seeds)")
        print("  M  | " + "  ".join(f"{m:>9}" for m in methods))
        for m in MS:
            row = {}
            for meth in methods:
                bal = [run(Z, y, m, meth, np.random.default_rng(1000 * s + 7))[0] for s in range(SEEDS)]
                row[meth] = float(np.mean(bal))
            print(f"  {m:3d} | " + "  ".join(f"{row[meth]:9.3f}" for meth in methods))
        # full curves for the plot
        curves = {meth: [float(np.mean([run(Z, y, m, meth, np.random.default_rng(1000 * s + 7))[0]
                                        for s in range(SEEDS)])) for m in MS] for meth in methods}
        for meth in methods:
            ls = "--" if meth == "strat-acq" else "-"
            ax.plot(MS, curves[meth], ls, marker="o", ms=3, label=meth)
        ax.axhline(0.5, color="#bbb", ls=":", lw=.8)
        ax.set_xlabel("# labeled specimens"); ax.set_ylabel("balanced accuracy (held-out)")
        ax.set_ylim(0.45, 1.0); ax.set_title(fname); ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.suptitle("Imbalance handling for rare factors as labels accumulate (oracle strat-acq dashed)", y=1.02)
    fig.tight_layout(); fig.savefig(RES / "imbalance.png", dpi=130, bbox_inches="tight")
    print(f"\nfigure -> {RES/'imbalance.png'}")


if __name__ == "__main__":
    main()
