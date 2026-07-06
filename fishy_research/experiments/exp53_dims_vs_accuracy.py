"""EXP-53 — PCA dimensionality (color vs pattern/spectral) vs classifier & clustering accuracy,
and whether an UNSUPERVISED metric (explained variance) predicts the dims you actually need.

Two descriptors, swept independently over #PCA dims:
  * COLOR            = the per-region Lab/chroma block (5 channels x R)         -> belly / tail factors
  * PATTERN/SPECTRAL = GFT spectral coefficients (k=40, L,a,b channels)         -> stripe factor (lightness)
For each #dims measure: per-factor 5-fold logistic accuracy, GMM clustering ARI, and CUMULATIVE explained
variance. Hypothesis (from EXP-43): color is high-variance so EV tracks accuracy; the subtle stripe factor
lives in LOW-variance PCs, so stripe accuracy keeps climbing well past where EV saturates -> EV under-counts
the dims you need.

Run:  .venv/bin/python experiments/exp53_dims_vs_accuracy.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "interactive_demo"))
import engine  # noqa: E402
from fishpipe import spectral  # noqa: E402

RES = HERE.parent / "results" / "exp53"
RES.mkdir(parents=True, exist_ok=True)
DIMS = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 60]


def descriptors():
    ld = engine.load_dataset("fishy")
    R = len(ld.rids)
    ccols = np.concatenate([np.arange(i * engine.PDIM, i * engine.PDIM + engine.COLOR_CH) for i in range(R)])
    Xcolor = StandardScaler().fit_transform(ld.Xs[:, ccols])                       # per-region color blocks
    Xspec = StandardScaler().fit_transform(spectral.spectral_coeffs(ld.fcd, k=40, channels=("L", "a", "b")))
    L = {f: np.asarray(ld.gt.labels[f]) for f in ("belly", "tail", "stripe")}
    joint4 = L["belly"] * 2 + L["tail"]
    # each descriptor: X, the factors to CLASSIFY, and the structure to CLUSTER (label + #clusters)
    return {
        "color (Lab blocks)": (Xcolor, ["belly", "tail"], ("belly×tail", joint4, 4)),
        "pattern/spectral (GFT)": (Xspec, ["stripe"], ("stripe", L["stripe"], 2)),
    }, L


def sweep(X, facs, L, clab, ck):
    maxd = min(max(DIMS), X.shape[1] - 1, X.shape[0] - 1)
    dims = [d for d in DIMS if d <= maxd]
    pca = PCA(maxd, random_state=0).fit(X)
    evr = pca.explained_variance_ratio_
    P = pca.transform(X)
    out = {"dims": dims, "ev_cum": [float(evr[:d].sum()) for d in dims],
           "acc": {f: [] for f in facs}, "ari": []}
    for d in dims:
        Pd = P[:, :d]
        for f in facs:
            out["acc"][f].append(float(cross_val_score(LogisticRegression(max_iter=2000), Pd, L[f], cv=5).mean()))
        pred = GaussianMixture(ck, n_init=3, random_state=0).fit_predict(Pd)
        out["ari"].append(float(adjusted_rand_score(clab, pred)))
    return out


def sat_dim(dims, vals, frac=0.97):
    """smallest #dims reaching `frac` of the curve's max."""
    mx = max(vals); thr = frac * mx
    for d, v in zip(dims, vals):
        if v >= thr:
            return d
    return dims[-1]


def main():
    descs, L = descriptors()
    res = {name: sweep(X, facs, L, clab, ck) for name, (X, facs, (cn, clab, ck)) in descs.items()}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colmap = {"belly": "#4e79a7", "tail": "#59a14f", "stripe": "#e15759"}
    print("=" * 76)
    for ax, (name, (X, facs, (cn, clab, ck))) in zip(axes, descs.items()):
        r = res[name]; dims = r["dims"]
        for f in facs:
            ax.plot(dims, r["acc"][f], "o-", color=colmap[f], label=f"{f} · classifier acc")
        ax.plot(dims, r["ari"], "s--", color="#9c27b0", alpha=.7, label=f"{cn} · cluster ARI")
        ax.set_xlabel("# PCA dimensions"); ax.set_ylabel("accuracy / ARI"); ax.set_ylim(0, 1.02)
        ax.set_title(name); ax.grid(alpha=.2)
        ev = ax.twinx(); ev.plot(dims, r["ev_cum"], ":", color="#888", lw=2.2, label="cumulative EV")
        ev.set_ylabel("cumulative explained variance", color="#888"); ev.set_ylim(0, 1.02)
        d_ev = sat_dim(dims, r["ev_cum"], 0.95)
        ev.axhline(0.95, color="#bbb", lw=.7, ls=":")
        ax.legend(loc="lower right", fontsize=8)
        print(f"\n{name}:  cumulative EV reaches 95% at d={d_ev}")
        for f in facs:
            da = sat_dim(dims, r["acc"][f]); rho = spearmanr(r["ev_cum"], r["acc"][f]).correlation
            print(f"  classifier {f:7s}: final {r['acc'][f][-1]:.3f}, 97%-of-max by d={da}  (vs EV-95% d={d_ev})  "
                  f"spearman(EV,acc)={rho:+.2f}")
        print(f"  cluster {cn:9s}: final ARI {r['ari'][-1]:.3f}, 97%-of-max by d={sat_dim(dims, r['ari'])}")
    fig.suptitle("PCA dims vs accuracy, with cumulative explained variance (dotted grey, 95% line)", y=1.02)
    fig.tight_layout(); fig.savefig(RES / "dims_vs_accuracy.png", dpi=130, bbox_inches="tight")

    print("\n" + "=" * 76 + "\nclassifier accuracy at max dims (all factors x both descriptors):")
    print(f"{'descriptor':26s} {'belly':>7} {'tail':>7} {'stripe':>7}")
    for name, (X, facs, _) in descs.items():
        accs = {f: float(cross_val_score(LogisticRegression(max_iter=2000),
                PCA(min(40, X.shape[1] - 1), random_state=0).fit_transform(X), L[f], cv=5).mean())
                for f in ("belly", "tail", "stripe")}
        print(f"{name:26s} " + " ".join(f"{accs[f]:7.3f}" for f in ('belly', 'tail', 'stripe')))
    print(f"\nfigure -> {RES/'dims_vs_accuracy.png'}")


if __name__ == "__main__":
    main()
