"""EXP-52 — Active batch selection for the tag loop: which next batch is most informative?

After applying a batch the demo should auto-roll the NEXT batch picked to help the classifier most. Compare
random vs uncertainty sampling vs uncertainty+diversity (batch-mode), on the held-out 8-way placement (ARI)
vs labeled count. Active learning can backfire (EXP-44), so check before implementing.

Run:  .venv/bin/python experiments/exp52_active_tags.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import _tag_harness as H  # noqa: E402

RES = H.RESULTS
B0, STEP, MAX = 8, 8, 160
REPS = 6


def fit_pred(Z, lab, Ylab):
    """per-tag logistic -> (scores N x6, probs N x6)."""
    S = np.zeros((len(Z), 6))
    for t in range(6):
        y = Ylab[:, t]
        if len(lab) < 2 or len(np.unique(y)) < 2:
            S[:, t] = float(y.mean()) if len(y) else 0.0; continue
        S[:, t] = LogisticRegression(C=0.5, max_iter=2000).fit(Z[lab], y).decision_function(Z)
    return S, 1.0 / (1.0 + np.exp(-S))


def ari8(Z, lab, Y, joint):
    S, _ = fit_pred(Z, lab, Y[lab])
    un = np.setdiff1d(np.arange(len(Z)), lab)
    pred = GaussianMixture(8, n_init=2, random_state=0).fit_predict(S)
    return adjusted_rand_score(joint[un], pred[un])


def pick(strategy, Z, lab, prob, B, rng):
    N = len(Z); un = np.setdiff1d(np.arange(N), lab)
    if strategy == "random" or prob is None:
        return rng.choice(un, min(B, len(un)), replace=False)
    u = (1.0 - np.abs(2 * prob - 1.0)).mean(1)[un]          # mean per-tag uncertainty
    if strategy == "uncertainty":
        return un[np.argsort(u)[::-1][:B]]
    # uncertainty + diversity: greedy, pick high-uncertainty but far from already-picked (in Z)
    chosen = []
    cand = list(un[np.argsort(u)[::-1]])                    # uncertain-first
    score = {int(i): float(uu) for i, uu in zip(un, u)}
    while len(chosen) < B and cand:
        if not chosen:
            c = cand[0]
        else:
            d = {i: min(np.linalg.norm(Z[i] - Z[j]) for j in chosen) for i in cand}
            md = (np.median(list(d.values())) or 1.0)
            c = max(cand, key=lambda i: score[i] * (d[i] / md))
        chosen.append(c); cand.remove(c)
    return np.array(chosen)


def main():
    Z, F, joint, _ = H.load(); Y = H.tags_for(F); N = len(Z)
    budgets = list(range(B0, MAX + 1, STEP))
    strategies = ["random", "uncertainty", "uncertainty+diversity"]
    curves = {s: np.zeros((REPS, len(budgets))) for s in strategies}
    for s in strategies:
        for r in range(REPS):
            rng = np.random.default_rng(100 * r + 1)
            lab = list(rng.choice(N, B0, replace=False))     # same cold-start seed per (s,r) via rng
            prob = None
            for bi, _ in enumerate(budgets):
                curves[s][r, bi] = ari8(Z, np.array(lab), Y, joint)
                _, prob = fit_pred(Z, np.array(lab), Y[np.array(lab)])
                nxt = pick(s, Z, np.array(lab), prob, STEP, rng)
                lab += [int(x) for x in nxt]
    print("labeled | " + "  ".join(f"{s[:12]:>12}" for s in strategies))
    for bi, n in enumerate(budgets):
        print(f"  {n:4d}  | " + "  ".join(f"{curves[s][:, bi].mean():12.3f}" for s in strategies))

    fig, ax = plt.subplots(figsize=(6, 4))
    for s in strategies:
        ax.plot(budgets, curves[s].mean(0), "o-", ms=3, label=s)
    ax.set_xlabel("# labeled specimens"); ax.set_ylabel("held-out 8-way ARI"); ax.set_ylim(0, 1)
    ax.set_title("Active batch selection for tag labeling"); ax.legend()
    fig.tight_layout(); fig.savefig(RES / "active_tags.png", dpi=130)
    print(f"\nfigure -> {RES/'active_tags.png'}")


if __name__ == "__main__":
    main()
