"""EXP-58 — ACQUISITION strategy to accumulate the RARE 'cheeks' class (maps to the demo batch-roll).

TARGET = cheeks (F col 3): RARE (14/250 = 5.6%) and localized. The binding constraint at small budgets is the
minority COUNT — you cannot learn 'cheeks' until the loop has actually labeled a few positives. So we score the
acquisition loop on BOTH axes vs #labels: (1) #cheeks positives DISCOVERED (binding constraint), and
(2) held-out cheeks BALANCED ACCURACY of a class_weight='balanced' logistic on Z24.

Incremental loop: seed ~8 random, add batches of 8 up to ~120 labeled. Classifier refit each batch on Z24.
Strategies:
  (a) random
  (b) uncertainty          : smallest |p-0.5| among unlabeled
  (c) nn_expansion         : query-by-example — once a cheeks positive is labeled, prioritize the unlabeled
                             nearest-neighbours (Z24) of labeled positives; fall back to uncertainty otherwise
  (d) density_uncertainty  : uncertainty * local density (kNN) — uncertain AND in a dense region
  (e) oracle_stratified    : upper bound — preferentially grab true positives (uses GT; label/eval only)

Held-out = the 'rest' (unlabeled pool) at each step. Metric averaged over MANY seeds. Plots BOTH curves.

Run:  .venv/bin/python experiments/exp58_acquisition.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, recall_score, f1_score
from sklearn.neighbors import NearestNeighbors

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RES = ROOT / "results" / "exp58"
RES.mkdir(parents=True, exist_ok=True)
CACHE = RES / "cache.npz"

TARGET = 3            # cheeks
SEED0, STEP, MAXLAB = 8, 8, 120
N_SEEDS = 60
KNN_DENSITY = 10      # neighbours for local-density estimate
TEST_POS, TEST_NEG = 4, 71   # frozen per-seed stratified test split (4 of 14 positives held out -> 10 acquirable)


def load():
    d = np.load(CACHE, allow_pickle=True)
    Z = d["Z24"].astype(np.float64)
    y = d["F"][:, TARGET].astype(int)
    return Z, y


def fit_clf(Z, lab, y):
    """balanced logistic on the labeled set; None if only one class present."""
    yl = y[lab]
    if len(np.unique(yl)) < 2:
        return None
    return LogisticRegression(class_weight="balanced", max_iter=2000, C=1.0).fit(Z[lab], yl)


def proba_pos(clf, Z):
    return clf.predict_proba(Z)[:, 1]


def density(Z):
    """local density = 1 / mean distance to KNN_DENSITY neighbours (in Z24)."""
    nn = NearestNeighbors(n_neighbors=KNN_DENSITY + 1).fit(Z)
    dist, _ = nn.kneighbors(Z)
    return 1.0 / (dist[:, 1:].mean(1) + 1e-9)


def pick(strategy, Z, lab, y, clf, nn_full, dens, rng, pool):
    """return up to STEP indices (from `pool`, i.e. acquirable, not in the frozen test set) to label next."""
    labset = set(int(i) for i in lab)
    un = np.array([i for i in pool if i not in labset])
    if len(un) == 0:
        return np.array([], int)
    B = min(STEP, len(un))

    if strategy == "random":
        return rng.choice(un, B, replace=False)

    if strategy == "oracle_stratified":
        # upper bound: grab true positives first (GT used only to choose labels), fill with random
        pos = un[y[un] == 1]
        neg = un[y[un] == 0]
        rng.shuffle(pos); rng.shuffle(neg)
        return np.concatenate([pos, neg])[:B]

    # everything below needs a classifier; until we have 2 classes, fall back to random
    if clf is None:
        return rng.choice(un, B, replace=False)
    p = proba_pos(clf, Z)
    unc = 1.0 - np.abs(2.0 * p - 1.0)         # high = uncertain (near 0.5)

    if strategy == "uncertainty":
        return un[np.argsort(unc[un])[::-1][:B]]

    if strategy == "density_uncertainty":
        score = unc[un] * dens[un]
        return un[np.argsort(score)[::-1][:B]]

    if strategy == "nn_expansion":
        pos_lab = [i for i in lab if y[i] == 1]
        if len(pos_lab) == 0:
            # no positive found yet -> behave like uncertainty
            return un[np.argsort(unc[un])[::-1][:B]]
        # rank unlabeled by nearest distance to ANY labeled positive (query-by-example)
        d, _ = nn_full.kneighbors(Z[un], n_neighbors=1) if False else (None, None)
        # explicit min-distance to labeled positives
        P = Z[pos_lab]
        # pairwise: for each unlabeled, min dist to a labeled positive
        d2 = ((Z[un][:, None, :] - P[None, :, :]) ** 2).sum(2).min(1)
        nn_order = un[np.argsort(d2)]           # closest to a known positive first
        return nn_order[:B]

    raise ValueError(strategy)


def evaluate(clf, Z, y, test):
    """cheeks balanced-acc + minority recall/F1 on a FIXED held-out test split.

    Using a frozen test set (rather than the shrinking unlabeled pool) keeps the comparison fair: otherwise a
    strategy that pulls positives OUT of the eval pool deflates its own score, and an oracle that exhausts the
    14 positives leaves zero positives to score.
    """
    if clf is None or len(test) == 0 or y[test].sum() == 0:
        return np.nan, np.nan, np.nan
    pred = clf.predict(Z[test])
    ba = balanced_accuracy_score(y[test], pred)
    rec = recall_score(y[test], pred, pos_label=1, zero_division=0)
    f1 = f1_score(y[test], pred, pos_label=1, zero_division=0)
    return ba, rec, f1


def main():
    Z, y = load()
    N = len(Z)
    dens = density(Z)
    nn_full = NearestNeighbors(n_neighbors=min(30, N)).fit(Z)
    budgets = list(range(SEED0, MAXLAB + 1, STEP))
    strategies = ["random", "uncertainty", "nn_expansion", "density_uncertainty", "oracle_stratified"]

    # curves[strategy] -> arrays (N_SEEDS, len(budgets))
    disc = {s: np.full((N_SEEDS, len(budgets)), np.nan) for s in strategies}   # #positives discovered
    ba = {s: np.full((N_SEEDS, len(budgets)), np.nan) for s in strategies}     # held-out balanced-acc
    rec = {s: np.full((N_SEEDS, len(budgets)), np.nan) for s in strategies}    # held-out minority recall
    f1 = {s: np.full((N_SEEDS, len(budgets)), np.nan) for s in strategies}

    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    for s in strategies:
        for r in range(N_SEEDS):
            rng = np.random.default_rng(1000 * r + 7)
            # FROZEN per-seed test split (shared across strategies): hold out a stratified ~30% incl. some
            # positives so balanced-acc is always defined and never depleted by the acquisition strategy.
            split_rng = np.random.default_rng(1000 * r + 7)
            test_pos = split_rng.choice(pos_idx, size=TEST_POS, replace=False)
            test_neg = split_rng.choice(neg_idx, size=TEST_NEG, replace=False)
            test = np.concatenate([test_pos, test_neg])
            pool = np.array([i for i in range(N) if i not in set(int(x) for x in test)])
            # cold-start seed drawn from the POOL only (never from the frozen test set)
            lab = list(split_rng.choice(pool, SEED0, replace=False))
            for bi, n in enumerate(budgets):
                lab_arr = np.array(lab[:n])
                clf = fit_clf(Z, lab_arr, y)
                disc[s][r, bi] = int(y[lab_arr].sum())
                b, rc, fc = evaluate(clf, Z, y, test)
                ba[s][r, bi], rec[s][r, bi], f1[s][r, bi] = b, rc, fc
                if n >= MAXLAB:
                    break
                nxt = pick(s, Z, lab_arr, y, clf, nn_full, dens, rng, pool)
                lab += [int(x) for x in nxt]

    # ---- report ----
    def col(d, bi):
        return d
    print(f"cheeks positives total: {int(y.sum())}/{N}  ({100*y.mean():.1f}%)   seeds={N_SEEDS}")
    print("\n#CHEEKS POSITIVES DISCOVERED  (mean over seeds)")
    print("labeled | " + "  ".join(f"{s[:11]:>11}" for s in strategies))
    for bi, n in enumerate(budgets):
        print(f"  {n:4d}  | " + "  ".join(f"{np.nanmean(disc[s][:, bi]):11.2f}" for s in strategies))

    print("\nHELD-OUT cheeks BALANCED-ACC  (mean over seeds)")
    print("labeled | " + "  ".join(f"{s[:11]:>11}" for s in strategies))
    for bi, n in enumerate(budgets):
        print(f"  {n:4d}  | " + "  ".join(f"{np.nanmean(ba[s][:, bi]):11.3f}" for s in strategies))

    print("\nHELD-OUT cheeks minority RECALL  (mean over seeds)")
    print("labeled | " + "  ".join(f"{s[:11]:>11}" for s in strategies))
    for bi, n in enumerate(budgets):
        print(f"  {n:4d}  | " + "  ".join(f"{np.nanmean(rec[s][:, bi]):11.3f}" for s in strategies))

    # budget to reach >=2 and >=3 positives (the binding constraint)
    print("\nMEAN #labels to reach K discovered positives (binding constraint)")
    for K in (1, 2, 3):
        print(f"  K={K}: ", end="")
        for s in strategies:
            reached = []
            for r in range(N_SEEDS):
                row = disc[s][r]
                hit = np.where(row >= K)[0]
                reached.append(budgets[hit[0]] if len(hit) else np.nan)
            print(f"{s[:11]}={np.nanmean(reached):5.1f}  ", end="")
        print()

    # ---- figure: both curves ----
    colors = {"random": "#888888", "uncertainty": "#1f77b4", "nn_expansion": "#d62728",
              "density_uncertainty": "#2ca02c", "oracle_stratified": "#9467bd"}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    for s in strategies:
        m = np.nanmean(disc[s], 0); se = np.nanstd(disc[s], 0) / np.sqrt(N_SEEDS)
        a1.plot(budgets, m, "o-", ms=4, color=colors[s], label=s)
        a1.fill_between(budgets, m - se, m + se, color=colors[s], alpha=0.12)
    a1.axhline(int(y.sum()) - TEST_POS, ls=":", c="k", lw=1, alpha=.5,
               label=f"all acquirable positives ({int(y.sum()) - TEST_POS})")
    a1.set_xlabel("# labeled specimens"); a1.set_ylabel("# cheeks positives discovered")
    a1.set_title("Accumulating the rare class (binding constraint)"); a1.legend(fontsize=8); a1.grid(alpha=.25)

    for s in strategies:
        m = np.nanmean(ba[s], 0); se = np.nanstd(ba[s], 0) / np.sqrt(N_SEEDS)
        a2.plot(budgets, m, "o-", ms=4, color=colors[s], label=s)
        a2.fill_between(budgets, m - se, m + se, color=colors[s], alpha=0.12)
    a2.axhline(0.5, ls=":", c="k", lw=1, alpha=.5, label="chance")
    a2.set_xlabel("# labeled specimens"); a2.set_ylabel("held-out cheeks balanced-acc")
    a2.set_title("Recovering cheeks (balanced LR on Z24)"); a2.legend(fontsize=8); a2.grid(alpha=.25)
    a2.set_ylim(0.45, 1.0)

    fig.suptitle(f"EXP-58 acquisition for rare 'cheeks' ({int(y.sum())}/{N}=5.6%) — {N_SEEDS} seeds, batches of {STEP}", y=1.02)
    fig.tight_layout()
    out = RES / "acquisition.png"
    fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"\nfigure -> {out}")


if __name__ == "__main__":
    main()
