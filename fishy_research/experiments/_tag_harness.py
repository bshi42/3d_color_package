"""Shared harness for EXP-46 — multi-label TAG feedback (vs pairwise similarity).

New interaction idea: instead of "more/less similar", the expert dynamically creates tags ("green tail",
"purple belly", "4 stripes", ...) and drags them onto specimens (multi-label). A classifier is trained and
we watch how it PLACES the unlabeled specimens (the classifier's latent space) as more tags are applied.

This module gives every candidate model the SAME data, labeling simulation, and evaluation, so results are
comparable. A model is a function:

    model_fn(Z, labeled_idx, Ytags_labeled) -> {"tag_scores": (N, T) array, "latent": (N, 2) or None}

  * Z              (N, K) descriptor coords (demo's PCA space) — the input features for ALL specimens
  * labeled_idx    indices the expert has tagged so far
  * Ytags_labeled  (len(labeled_idx), T) binary tag matrix for those specimens (multi-label)
  * returns per-specimen tag scores for ALL N specimens and (optionally) a 2-D latent embedding.

The harness maps the T tags back to the 3 known factors only at EVAL time (the model stays factor-agnostic,
like free-form tags), and reports placement quality on the UNLABELED specimens vs label budget.

Tags are factor-ordered: [belly0,belly1, tail0,tail1, stripe0,stripe1]  (factor_of_tag = [0,0,1,1,2,2]).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "interactive_demo"))
RESULTS = HERE.parent / "results" / "exp46"
RESULTS.mkdir(parents=True, exist_ok=True)
CACHE = RESULTS / "cache.npz"

FACTORS = ("belly", "tail", "stripe")
TAG_NAMES = ["belly:A", "belly:B", "tail:A", "tail:B", "stripe:4", "stripe:5"]
FACTOR_OF_TAG = np.array([0, 0, 1, 1, 2, 2])
K_PCS = 24
BUDGETS = [5, 10, 20, 40, 80, 160]


def load():
    """(Z (N,K z-scored PCA), F (N,3) factor labels, joint (N,) 0..7, names). Cached to npz."""
    if CACHE.exists():
        d = np.load(CACHE, allow_pickle=True)
        return d["Z"], d["F"], d["joint"], d["names"]
    import engine
    s = engine.Session("fishy")
    Z = engine._zscore_cols(PCA(K_PCS, random_state=0).fit_transform(StandardScaler().fit_transform(s.ld.Xs)))
    F = np.stack([s.ld.gt.labels[f] for f in FACTORS], 1).astype(int)
    joint = F[:, 0] * 4 + F[:, 1] * 2 + F[:, 2]
    names = np.array(s.ld.names)
    np.savez(CACHE, Z=Z, F=F, joint=joint, names=names)
    return Z, F, joint, names


def tags_for(F):
    """(N, 6) binary multi-label tag matrix: each specimen has its 3 true factor-value tags."""
    N = len(F)
    Y = np.zeros((N, 6), int)
    for c in range(3):
        Y[np.arange(N), 2 * c + F[:, c]] = 1
    return Y


def labeled_set(N, n, seed):
    return np.sort(np.random.default_rng(seed).choice(N, min(n, N), replace=False))


def _factor_pred(tag_scores):
    """(N,6) tag scores -> (N,3) predicted factor values via within-factor argmax."""
    s = np.asarray(tag_scores, float)
    return np.stack([(s[:, 2 * c + 1] > s[:, 2 * c]).astype(int) for c in range(3)], 1)


def eval_model(model_fn, budgets=BUDGETS, reps=6, seed=0):
    """Mean over reps of: per-factor accuracy on UNLABELED, and 8-way ARI of predicted joint (unlabeled)."""
    Z, F, joint, _ = load()
    N = len(Z); Y = tags_for(F)
    rows = {}
    for n in budgets:
        accs = np.zeros((reps, 3)); aris = np.zeros(reps)
        for r in range(reps):
            lab = labeled_set(N, n, seed + r)
            out = model_fn(Z, lab, Y[lab])
            pred = _factor_pred(out["tag_scores"])
            un = np.setdiff1d(np.arange(N), lab)
            accs[r] = (pred[un] == F[un]).mean(0)
            pj = pred[:, 0] * 4 + pred[:, 1] * 2 + pred[:, 2]
            aris[r] = adjusted_rand_score(joint[un], pj[un])
        rows[n] = {"acc_belly": round(float(accs[:, 0].mean()), 3),
                   "acc_tail": round(float(accs[:, 1].mean()), 3),
                   "acc_stripe": round(float(accs[:, 2].mean()), 3),
                   "acc_mean": round(float(accs.mean()), 3),
                   "ari8": round(float(aris.mean()), 3)}
    return rows


def viz(model_fn, n_labeled, path, title=""):
    """Save the model's 2-D latent (or PCA-2 of tag scores), points coloured by TRUE 8-way label,
    labeled specimens ringed. Shows how unlabeled specimens get placed."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    Z, F, joint, _ = load()
    N = len(Z); Y = tags_for(F)
    lab = labeled_set(N, n_labeled, 0)
    out = model_fn(Z, lab, Y[lab])
    L = out.get("latent")
    if L is None:
        L = PCA(2, random_state=0).fit_transform(np.asarray(out["tag_scores"], float))
    L = np.asarray(L, float)
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.scatter(L[:, 0], L[:, 1], s=22, c=[plt.cm.tab10(j) for j in joint], edgecolors="none", alpha=.85)
    ax.scatter(L[lab, 0], L[lab, 1], s=70, facecolors="none", edgecolors="k", linewidths=1.1, label="tagged")
    ax.set_title(title or f"latent space · {n_labeled} tagged"); ax.set_xticks([]); ax.set_yticks([])
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def run_model(model_fn, name):
    """Convenience: eval + viz at two budgets, print a compact table, return the metrics dict."""
    res = eval_model(model_fn)
    print(f"== {name} ==")
    print("  budget |  acc_mean  acc_stripe  ari8")
    for n, m in res.items():
        print(f"  {n:5d}  |   {m['acc_mean']:.3f}     {m['acc_stripe']:.3f}     {m['ari8']:.3f}")
    safe = name.lower().replace(" ", "_").replace("/", "_")
    viz(model_fn, 40, RESULTS / f"latent_{safe}_n40.png", f"{name} · 40 tagged")
    viz(model_fn, 160, RESULTS / f"latent_{safe}_n160.png", f"{name} · 160 tagged")
    return res
