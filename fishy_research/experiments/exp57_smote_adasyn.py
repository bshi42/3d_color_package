"""EXP-57 — SMOTE vs ADASYN (vs class_weight + plain) for the rare factors while tagging.

EXP-56 found SMOTE ≈ class_weight='balanced'. This adds ADASYN: same kNN interpolation as SMOTE, but the
PER-MINORITY synthetic count is weighted by local majority density r_i (fraction of majority among each
minority point's kNN) — so it over-samples the HARD/boundary minority points instead of uniformly. Implemented
directly (imblearn's sklearn pin conflicts with sklearn 1.8). Balanced accuracy + minority recall on held-out,
mean of many seeds; feature space = demo PCA-24.

Run:  .venv/bin/python experiments/exp57_smote_adasyn.py
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

RES = HERE.parent / "results" / "exp57"
RES.mkdir(parents=True, exist_ok=True)
MS = [20, 30, 40, 60, 80, 120, 160]
SEEDS = 50


def _minmaj(y):
    cls, cnt = np.unique(y, return_counts=True)
    return cls[np.argmin(cnt)], cls[np.argmax(cnt)]


def smote(X, y, rng, k=5):
    minc, majc = _minmaj(y); Xm = X[y == minc]; nmin = len(Xm)
    G = int((y == majc).sum() - nmin)
    if nmin < 2 or G <= 0:
        return X, y
    D = np.linalg.norm(Xm[:, None] - Xm[None], axis=2); np.fill_diagonal(D, np.inf)
    nn = np.argsort(D, 1)[:, :min(k, nmin - 1)]
    a = rng.integers(0, nmin, G); b = nn[a, rng.integers(0, nn.shape[1], G)]
    lam = rng.random((G, 1))
    return np.vstack([X, Xm[a] + lam * (Xm[b] - Xm[a])]), np.concatenate([y, np.full(G, minc)])


def adasyn(X, y, rng, k=5):
    minc, majc = _minmaj(y); Xm = X[y == minc]; nmin = len(Xm)
    G = int((y == majc).sum() - nmin)
    if nmin < 2 or G <= 0:
        return X, y
    r = np.array([(y[np.argsort(np.linalg.norm(X - Xm[i], axis=1))[1:k + 1]] == majc).mean() for i in range(nmin)])
    if r.sum() == 0:
        r[:] = 1.0                                    # all interior -> uniform (= SMOTE)
    gi = np.round(r / r.sum() * G).astype(int)        # MORE synthetics for boundary (high-r) minority points
    D = np.linalg.norm(Xm[:, None] - Xm[None], axis=2); np.fill_diagonal(D, np.inf)
    nn = np.argsort(D, 1)[:, :min(k, nmin - 1)]
    syn = []
    for i in range(nmin):
        for _ in range(gi[i]):
            j = nn[i, rng.integers(0, nn.shape[1])]; lam = rng.random()
            syn.append(Xm[i] + lam * (Xm[j] - Xm[i]))
    if not syn:
        return X, y
    return np.vstack([X, np.array(syn)]), np.concatenate([y, np.full(len(syn), minc)])


def run(Z, y, M, method, rng):
    lab = rng.choice(len(y), M, replace=False); te = np.setdiff1d(np.arange(len(y)), lab)
    Xl, yl = Z[lab], y[lab]
    if len(np.unique(yl)) < 2:
        return 0.5, 0.0
    if method == "balanced":
        clf = LogisticRegression(C=0.5, class_weight="balanced", max_iter=2000)
    else:
        if method == "smote":
            Xl, yl = smote(Xl, yl, rng)
        elif method == "adasyn":
            Xl, yl = adasyn(Xl, yl, rng)
        clf = LogisticRegression(C=0.5, max_iter=2000)
    p = clf.fit(Xl, yl).predict(Z[te])
    return balanced_accuracy_score(y[te], p), recall_score(y[te], p, pos_label=1, zero_division=0)


def main():
    ld = engine.load_dataset("fishy")
    Z = engine._zscore_cols(PCA(24, random_state=0).fit_transform(StandardScaler().fit_transform(ld.Xs))).astype("float32")
    factors = {"stripe (21%)": np.asarray(ld.gt.labels["stripe"]),
               "cheeks (5.6%)": np.asarray(ld.gt.labels["cheeks"])}
    methods = ["plain", "balanced", "smote", "adasyn"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, (fname, y) in zip(axes, factors.items()):
        print(f"\n=== {fname} === balanced-acc | (minority recall) on held-out, mean of {SEEDS} seeds")
        print("  M  | " + "  ".join(f"{m:>16}" for m in methods))
        curves = {m: [] for m in methods}
        for M in MS:
            cells = []
            for meth in methods:
                res = [run(Z, y, M, meth, np.random.default_rng(2000 * s + 3)) for s in range(SEEDS)]
                bal = float(np.mean([r[0] for r in res])); rec = float(np.mean([r[1] for r in res]))
                curves[meth].append(bal); cells.append(f"{bal:.3f} ({rec:.2f})")
            print(f"  {M:3d} | " + "  ".join(f"{c:>16}" for c in cells))
        for meth in methods:
            ax.plot(MS, curves[meth], "o-", ms=3, label=meth)
        ax.axhline(0.5, color="#bbb", ls=":", lw=.8); ax.set_ylim(0.45, 1.0)
        ax.set_xlabel("# labeled specimens"); ax.set_ylabel("balanced accuracy (held-out)")
        ax.set_title(fname); ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.suptitle("SMOTE vs ADASYN vs class_weight vs plain (rare factors as labels accumulate)", y=1.02)
    fig.tight_layout(); fig.savefig(RES / "smote_vs_adasyn.png", dpi=130, bbox_inches="tight")
    print(f"\nfigure -> {RES/'smote_vs_adasyn.png'}")


if __name__ == "__main__":
    main()
