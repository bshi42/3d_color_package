"""EXP-58 — Best FEATURE REPRESENTATION for the rare/localized 'cheeks' factor.

TARGET = cheeks (F col 3): RARE (14/250 = 5.6%), LOCALIZED ('rosy cheeks').
At budgets M in {40,80,120} (random labeled subset; class_weight='balanced' logistic; balanced-acc +
minority recall/F1 on the HELD-OUT rest; >=40 seeds, averaged) we compare representation families:

  (1) Z24            — the demo's CURRENT PCA-24 space.
  (2) PCA(Xs, d)     — d in {8,16,24,40,60,100}: does cheeks need MORE dims (like stripe)?
  (3) full-Xs        — all 1408 standardized features, L2-regularized.
  (4) color-only     — color channels (0:5) of every region -> (250, 128*5=640), and color PCA-24.
  (5) supervised sel — from the LABELED M only, score every Xs feature by |t-stat| / mutual_info,
                       OR score every 11-dim region-block; keep top-k features / top region-blocks,
                       classify on those. k in {20,50,150}; region-blocks: top {8,16,32} regions.

All PCA / selection fit ONLY on the labeled M (no leakage from held-out or GT). GT used ONLY to label/score.

Run:  .venv/bin/python experiments/exp58_cheeks_repr.py
Out:  results/exp58/cheeks_repr.png  (+ printed OBSERVED table)
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, recall_score, f1_score

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results" / "exp58"
RES.mkdir(parents=True, exist_ok=True)

CACHE = RES / "cache.npz"
MS = [40, 80, 120]
SEEDS = 60
TARGET = 3  # cheeks
R, PDIM, COLOR_CH = 128, 11, 5


def load():
    d = np.load(CACHE, allow_pickle=True)
    return d["Xs"].astype(np.float64), d["Z24"].astype(np.float64), d["F"].astype(int)


def clf():
    return LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)


def fit_eval(Xtr, ytr, Xte, yte):
    """Standardize on train, fit balanced logistic, score on held-out."""
    mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-9
    Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd
    m = clf().fit(Xtr, ytr)
    p = m.predict(Xte)
    return (balanced_accuracy_score(yte, p),
            recall_score(yte, p, pos_label=1, zero_division=0),
            f1_score(yte, p, pos_label=1, zero_division=0))


# ---- representation builders. Each returns (Xtr, Xte) given raw train/test Xs + labels. ----
def rep_pca(dim):
    def f(Xs_tr, Xs_te, ytr, Z_tr, Z_te):
        p = PCA(n_components=min(dim, Xs_tr.shape[0] - 1, Xs_tr.shape[1])).fit(Xs_tr)
        return p.transform(Xs_tr), p.transform(Xs_te)
    return f


def rep_full(Xs_tr, Xs_te, ytr, Z_tr, Z_te):
    return Xs_tr, Xs_te


def rep_z24(Xs_tr, Xs_te, ytr, Z_tr, Z_te):
    return Z_tr, Z_te


def _color_only(Xs):
    return Xs.reshape(-1, R, PDIM)[:, :, :COLOR_CH].reshape(Xs.shape[0], -1)


def rep_color(Xs_tr, Xs_te, ytr, Z_tr, Z_te):
    return _color_only(Xs_tr), _color_only(Xs_te)


def rep_color_pca(dim):
    def f(Xs_tr, Xs_te, ytr, Z_tr, Z_te):
        Ctr, Cte = _color_only(Xs_tr), _color_only(Xs_te)
        p = PCA(n_components=min(dim, Ctr.shape[0] - 1, Ctr.shape[1])).fit(Ctr)
        return p.transform(Ctr), p.transform(Cte)
    return f


def _tstat(X, y):
    pos = X[y == 1]; neg = X[y == 0]
    if pos.shape[0] < 2 or neg.shape[0] < 2:
        return np.zeros(X.shape[1])
    num = pos.mean(0) - neg.mean(0)
    den = np.sqrt(pos.var(0, ddof=1) / pos.shape[0] + neg.var(0, ddof=1) / neg.shape[0] + 1e-9)
    return np.abs(num / den)


def rep_sel_t(k):
    def f(Xs_tr, Xs_te, ytr, Z_tr, Z_te):
        s = _tstat(Xs_tr, ytr)
        idx = np.argsort(s)[::-1][:k]
        return Xs_tr[:, idx], Xs_te[:, idx]
    return f


def rep_sel_mi(k):
    def f(Xs_tr, Xs_te, ytr, Z_tr, Z_te):
        # standardize before MI for stability
        mu = Xs_tr.mean(0); sd = Xs_tr.std(0) + 1e-9
        Ztr = (Xs_tr - mu) / sd
        mi = mutual_info_classif(Ztr, ytr, discrete_features=False, random_state=0)
        idx = np.argsort(mi)[::-1][:k]
        return Xs_tr[:, idx], Xs_te[:, idx]
    return f


def rep_sel_region(topr):
    """Score each of 128 region-blocks by summed |t| over its 11 channels; keep top region-blocks (all chans)."""
    def f(Xs_tr, Xs_te, ytr, Z_tr, Z_te):
        s = _tstat(Xs_tr, ytr).reshape(R, PDIM)
        rscore = s.sum(1)
        ridx = np.argsort(rscore)[::-1][:topr]
        cols = np.concatenate([np.arange(r * PDIM, r * PDIM + PDIM) for r in ridx])
        return Xs_tr[:, cols], Xs_te[:, cols]
    return f


REPS = [
    ("Z24 (current)",   rep_z24),
    ("PCA-8",           rep_pca(8)),
    ("PCA-16",          rep_pca(16)),
    ("PCA-24",          rep_pca(24)),
    ("PCA-40",          rep_pca(40)),
    ("PCA-60",          rep_pca(60)),
    ("PCA-100",         rep_pca(100)),
    ("full-Xs(1408)",   rep_full),
    ("color-640",       rep_color),
    ("colorPCA-24",     rep_color_pca(24)),
    ("sel|t| k=20",     rep_sel_t(20)),
    ("sel|t| k=50",     rep_sel_t(50)),
    ("sel|t| k=150",    rep_sel_t(150)),
    ("selMI k=50",      rep_sel_mi(50)),
    ("selReg top8",     rep_sel_region(8)),
    ("selReg top16",    rep_sel_region(16)),
    ("selReg top32",    rep_sel_region(32)),
]


def main():
    Xs, Z24, F = load()
    y = F[:, TARGET]
    N = len(y)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    print(f"N={N}  cheeks pos={len(pos_idx)} ({len(pos_idx)/N:.1%})")

    names = [n for n, _ in REPS]
    # ba[rep, budget, seed], plus rec/f1
    BA = np.full((len(REPS), len(MS), SEEDS), np.nan)
    REC = np.full_like(BA, np.nan)
    F1 = np.full_like(BA, np.nan)

    # track which Xs features/regions supervised selection picks (for the strongest sel rep)
    sel_feat_counts = np.zeros(Xs.shape[1])
    sel_region_counts = np.zeros(R)

    for si in range(SEEDS):
        rng = np.random.default_rng(1000 + si)
        for mi_, M in enumerate(MS):
            # stratified-ish labeled subset: ensure >=1 positive in the labeled set so balanced LR can train
            npos = max(1, int(round(M * len(pos_idx) / N)))
            tr_pos = rng.choice(pos_idx, size=min(npos, len(pos_idx)), replace=False)
            n_neg = M - len(tr_pos)
            tr_neg = rng.choice(neg_idx, size=n_neg, replace=False)
            tr = np.concatenate([tr_pos, tr_neg])
            te = np.setdiff1d(np.arange(N), tr)
            ytr, yte = y[tr], y[te]
            if ytr.sum() < 1 or yte.sum() < 1:
                continue
            Xs_tr, Xs_te = Xs[tr], Xs[te]
            Z_tr, Z_te = Z24[tr], Z24[te]
            for ri, (nm, fn) in enumerate(REPS):
                Atr, Ate = fn(Xs_tr, Xs_te, ytr, Z_tr, Z_te)
                ba, rc, f1 = fit_eval(Atr, ytr, Ate, yte)
                BA[ri, mi_, si] = ba; REC[ri, mi_, si] = rc; F1[ri, mi_, si] = f1
            # record selection picks at M=80 using |t| k=50
            if M == 80:
                s = _tstat(Xs_tr, ytr)
                idx = np.argsort(s)[::-1][:50]
                sel_feat_counts[idx] += 1
                sel_region_counts[(idx // PDIM)] += 1

    BAm = np.nanmean(BA, 2); BAs = np.nanstd(BA, 2)
    RECm = np.nanmean(REC, 2); F1m = np.nanmean(F1, 2)

    # ---- printed OBSERVED table ----
    print("\n=== Balanced accuracy (mean over seeds) ===")
    hdr = "rep".ljust(18) + "".join(f"M={m}".rjust(11) for m in MS)
    print(hdr)
    order = np.argsort(-BAm[:, -1])  # sort by M=120 BA
    for ri in order:
        row = names[ri].ljust(18)
        for mi_ in range(len(MS)):
            row += f"{BAm[ri,mi_]:.3f}±{BAs[ri,mi_]:.02f}".rjust(11)
        print(row)

    print("\n=== Minority recall / F1 (M=80) ===")
    for ri in order:
        print(f"{names[ri].ljust(18)} rec={RECm[ri,1]:.3f}  f1={F1m[ri,1]:.3f}")

    # which channels do the top-selected features fall in?
    chan_of = (np.arange(Xs.shape[1]) % PDIM)
    chan_hist = np.array([sel_feat_counts[chan_of == c].sum() for c in range(PDIM)])
    print("\n=== Supervised |t| selection (M=80) channel mass (0-4 color, 5-10 texture) ===")
    print("counts:", np.round(chan_hist).astype(int))
    color_frac = chan_hist[:COLOR_CH].sum() / (chan_hist.sum() + 1e-9)
    print(f"color-channel fraction of selected features: {color_frac:.1%}")
    top_regions = np.argsort(sel_region_counts)[::-1][:12]
    print("top regions by selection frequency:", top_regions.tolist())

    # ---- figure ----
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    # panel A: BA by representation x budget (grouped bars)
    ax = axes[0]
    x = np.arange(len(REPS))
    w = 0.26
    for j, M in enumerate(MS):
        ax.bar(x + (j - 1) * w, BAm[:, j], w, yerr=BAs[:, j] / np.sqrt(SEEDS),
               label=f"M={M}", capsize=2)
    ax.axhline(0.5, color="k", lw=0.8, ls=":")
    z24ba = BAm[0, -1]
    ax.axhline(z24ba, color="C0", lw=0.8, ls="--", alpha=0.6)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("balanced accuracy (held-out)")
    ax.set_ylim(0.45, max(0.9, np.nanmax(BAm) + 0.05))
    ax.set_title(f"EXP-58 cheeks (5.6%): balanced-acc by representation x budget\n"
                 f"({SEEDS} seeds, balanced logistic; dashed=Z24@M120)")
    ax.legend(); ax.grid(axis="y", alpha=0.3)

    # panel B: PCA-dim sweep (does cheeks need more dims?) vs color/full/sel best
    ax = axes[1]
    pca_names = ["PCA-8", "PCA-16", "PCA-24", "PCA-40", "PCA-60", "PCA-100"]
    pca_dims = [8, 16, 24, 40, 60, 100]
    pidx = [names.index(n) for n in pca_names]
    for j, M in enumerate(MS):
        ax.plot(pca_dims, BAm[pidx, j], "-o", label=f"PCA M={M}")
    # reference lines: Z24, best sel, color
    for nm, col in [("Z24 (current)", "k"), ("color-640", "C4"), ("sel|t| k=50", "C5"), ("full-Xs(1408)", "C6")]:
        ri = names.index(nm)
        ax.axhline(BAm[ri, -1], ls="--", lw=1, color=col, alpha=0.7, label=f"{nm}@M120")
    ax.set_xlabel("PCA dimensionality"); ax.set_ylabel("balanced acc")
    ax.set_title("Does cheeks need MORE PCA dims?\n(lines=PCA sweep, dashed=other reps @M120)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # panel C: where supervised selection lives (region map + channel mass)
    ax = axes[2]
    ax.bar(np.arange(PDIM), chan_hist, color=["C2"] * COLOR_CH + ["C3"] * (PDIM - COLOR_CH))
    ax.set_xticks(np.arange(PDIM))
    ax.set_xticklabels([f"c{c}" for c in range(COLOR_CH)] + [f"t{c}" for c in range(PDIM - COLOR_CH)],
                       fontsize=8)
    ax.set_xlabel("descriptor channel (green=color, red=texture)")
    ax.set_ylabel("selection mass (|t| top-50, summed over seeds)")
    ax.set_title(f"Supervised selection is COLOR-dominated\ncolor frac={color_frac:.0%}; "
                 f"top regions {top_regions[:6].tolist()}")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = RES / "cheeks_repr.png"
    plt.savefig(out, dpi=120)
    print(f"\nsaved figure -> {out}")

    # summary winners
    best_ri = order[0]
    print(f"\nWINNER @M120: {names[best_ri]}  BA={BAm[best_ri,-1]:.3f}  (Z24={z24ba:.3f})")
    for j, M in enumerate(MS):
        bi = np.argmax(BAm[:, j])
        print(f"  best @M={M}: {names[bi]} BA={BAm[bi,j]:.3f}  | Z24={BAm[0,j]:.3f}")


if __name__ == "__main__":
    main()
