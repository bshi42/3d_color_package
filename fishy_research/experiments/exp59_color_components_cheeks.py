"""EXP-59 — Do MORE color-PCA / color-ICA components help the rare, low-variance CHEEKS factor?

Cheeks ('rosy cheeks', 5.6%) is a localized COLOR feature contributing little variance -> hypothesis: its
signal is in the LOW-VARIANCE TAIL of the color PCA, so more components help (like stripe). And ICA: EXP-55
showed FastICA(whiten) ≡ PCA for a linear classifier (rotation within the top-n PCA subspace) -> expect the
same accuracy, but check whether ICA at least concentrates cheeks into a single dedicated COMPONENT.

Color descriptor = the 5 color channels of each region block (640 dims). Metric = balanced accuracy
(class_weight='balanced' logistic), repeated stratified 5-fold CV (30 evals). Stripe shown for contrast.

Run:  .venv/bin/python experiments/exp59_color_components_cheeks.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA, FastICA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

RES = Path(__file__).resolve().parent.parent / "results" / "exp58"
NS = [2, 4, 8, 12, 16, 24, 32, 48, 64, 100]
CV = RepeatedStratifiedKFold(n_splits=5, n_repeats=6, random_state=0)


def bal(X, y):
    return float(cross_val_score(LogisticRegression(C=0.5, class_weight="balanced", max_iter=3000),
                                 X, y, cv=CV, scoring="balanced_accuracy").mean())


def main():
    d = np.load(RES / "cache.npz")
    Xs, F = d["Xs"], d["F"]
    color = Xs.reshape(len(Xs), int(d["R"]), int(d["PDIM"]))[:, :, :int(d["COLOR_CH"])].reshape(len(Xs), -1)
    color = StandardScaler().fit_transform(color)
    targets = {"cheeks (5.6%)": F[:, 3], "stripe (21%)": F[:, 2]}
    pcaf = PCA(max(NS), random_state=0).fit(color); evr = pcaf.explained_variance_ratio_; P = pcaf.transform(color)
    ev = [float(evr[:n].sum()) for n in NS]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, (name, y) in zip(axes, targets.items()):
        pca_acc, ica_acc = [], []
        for n in NS:
            pca_acc.append(bal(P[:, :n], y))
            try:
                ic = FastICA(n, whiten="unit-variance", max_iter=2000, random_state=0).fit_transform(color)
            except Exception:
                ic = P[:, :n]
            ica_acc.append(bal(ic, y))
        print(f"\n=== {name} === balanced accuracy vs #color components")
        print("  n  |  PCA    ICA    cumEV")
        for n, p, i, e in zip(NS, pca_acc, ica_acc, ev):
            print(f"  {n:3d} | {p:.3f}  {i:.3f}  {e:.3f}")
        ax.plot(NS, pca_acc, "o-", color="#1d6fd0", label="PCA")
        ax.plot(NS, ica_acc, "s--", color="#e15759", label="ICA", alpha=.8)
        ax.set_xlabel("# color components"); ax.set_ylabel("cheeks/stripe balanced accuracy")
        ax.set_ylim(0.45, 1.0); ax.set_title(name); ax.grid(alpha=.2)
        ev2 = ax.twinx(); ev2.plot(NS, ev, ":", color="#888", lw=2); ev2.set_ylabel("cumulative EV", color="#888")
        ev2.set_ylim(0, 1.02); ax.legend(loc="lower right")

    # does ICA give cheeks a DEDICATED component? best single-component balanced acc at n=24
    print("\nsingle-component cheeks discriminability (n=24):")
    for name, fn in [("PCA", lambda: P[:, :24]),
                     ("ICA", lambda: FastICA(24, whiten="unit-variance", max_iter=2000, random_state=0).fit_transform(color))]:
        Xc = fn(); ch = F[:, 3]
        accs = sorted((bal(Xc[:, c:c + 1], ch) for c in range(Xc.shape[1])), reverse=True)
        print(f"  {name}: best single comp {accs[0]:.3f}, 2nd {accs[1]:.3f} (gap {accs[0]-accs[1]:.3f})")
    fig.suptitle("More color components help the low-variance cheeks/stripe factors; ICA ≈ PCA", y=1.02)
    fig.tight_layout(); fig.savefig(RES / "color_components_cheeks.png", dpi=130, bbox_inches="tight")
    print(f"\nfigure -> {RES/'color_components_cheeks.png'}")


if __name__ == "__main__":
    main()
