"""EXP-34 — Anchor+rank-neighbors triplet feedback loop (owner-proposed).

Loop: from the current (initially purely-unsupervised) morphospace, show an ANCHOR + its k
nearest neighbors + one REMOTE instance; a simulated expert RANKS them by similarity; convert
the ranking to triplets; MORPH the space (triplet metric learning); repeat. We measure whether
each factor's neighborhood structure improves over rounds, under three expert models, and we
deliberately test the FILTER-BUBBLE failure mode (local-only anchors) vs diverse-anchor sampling.

Simulated expert = ranks by Euclidean distance in a chosen subset of the TRUE generating params
(the oracle for 'what the expert perceives'):
  * holistic : all factor params (belly,tail,stripe,cheeks + continuous) -> ranks by everything
  * pattern  : stripe-weighted, color excluded -> 'rank by pattern, ignore colour'
  * color    : belly/tail hue only -> filter-bubble / wrong-factor demo
"""
from __future__ import annotations

import json
import warnings
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import balanced_accuracy_score

from fishpipe import data, textons, semisup

warnings.filterwarnings("ignore")


def perceptual_space(gt, mode):
    p = gt.params
    z = lambda c: StandardScaler().fit_transform(p[[c]].to_numpy()).ravel()
    if mode == "holistic":
        cols = ["belly_hue", "tail_hue", "stripe_count", "rosy_cheeks_present",
                "stripe_spacing", "stripe_width", "base_color_hue"]
        return np.column_stack([z(c) for c in cols])
    if mode == "pattern":
        # rank by pattern: stripe count dominant, some geometry; NO belly/tail hue
        return np.column_stack([3.0 * z("stripe_count"), z("stripe_spacing"), z("stripe_width")])
    if mode == "color":
        return np.column_stack([z("belly_hue"), z("tail_hue")])
    raise ValueError(mode)


def knn_balacc(Z, y, k=7):
    n_splits = int(min(5, np.bincount(y).min()))
    if n_splits < 2:
        return float("nan")
    cv = StratifiedKFold(n_splits, shuffle=True, random_state=0)
    pred = cross_val_predict(KNeighborsClassifier(k), Z, y, cv=cv)
    return balanced_accuracy_score(y, pred)


def fps_anchors(Z, n_anchors, rng):
    """Farthest-point sampling for DIVERSE anchors (anti-filter-bubble)."""
    n = Z.shape[0]
    idx = [int(rng.integers(n))]
    d = np.linalg.norm(Z - Z[idx[0]], axis=1)
    while len(idx) < n_anchors:
        i = int(np.argmax(d)); idx.append(i)
        d = np.minimum(d, np.linalg.norm(Z - Z[i], axis=1))
    return idx


def run_loop(X, gt, mode, rounds=6, k=6, anchors_per_round=25, diverse=True, seed=0):
    rng = np.random.default_rng(seed)
    P = perceptual_space(gt, mode)                       # expert's oracle metric
    n, p = X.shape
    # round-0 space = ordinary unsupervised PCA morphospace
    L0 = PCA(n_components=min(15, p - 1), random_state=0).fit(StandardScaler().fit_transform(X)).components_.T
    Xs = StandardScaler().fit_transform(X)
    L = L0.copy()
    factors = ["belly", "tail", "stripe", "cheeks"]
    history = {f: [] for f in factors}
    all_trips = []
    for r in range(rounds + 1):
        Z = Xs @ L
        for f in factors:
            history[f].append(round(knn_balacc(Z, gt.labels[f]), 3))
        if r == rounds:
            break
        # choose anchors
        if diverse:
            anchors = fps_anchors(Z, anchors_per_round, rng)
        else:  # filter-bubble: random local anchors only
            anchors = list(rng.choice(n, anchors_per_round, replace=False))
        for a in anchors:
            d = np.linalg.norm(Z - Z[a], axis=1)
            nn = np.argsort(d)[1:k + 1]                  # k nearest in CURRENT space
            remote = int(np.argmax(d))                   # the remote check
            # expert ranks the candidate set by perceptual distance to anchor
            cand = list(nn)
            order = sorted(cand, key=lambda j: np.linalg.norm(P[j] - P[a]))
            all_trips += semisup.triplets_from_ranking(a, order, remote=remote)
        # morph: re-learn warp from ALL accumulated triplets, anchored to the PCA prior
        L = semisup.triplet_warp(Xs, all_trips, n_dim=L0.shape[1], L0=L0,
                                 n_epochs=30, lr=0.02, reg=2e-2, seed=seed)
    return history, len(all_trips)


def main():
    fcd = data.build_face_colors(verbose=False)
    gt = data.load_ground_truth()
    P = textons.diffusion_operator()
    jet = textons.local_jet(fcd, scales=(2, 4), P=P)
    sc, km = textons.build_codebook(jet, K=128, seed=0)
    X = textons.encode(fcd, jet, sc, km, mode="bow")

    report = {}
    configs = [("holistic", True), ("pattern", True), ("color", True), ("pattern", False)]
    for mode, diverse in configs:
        hist, ntrip = run_loop(X, gt, mode, rounds=6, diverse=diverse)
        tag = f"{mode}{'' if diverse else ' (LOCAL anchors=filter-bubble)'}"
        report[tag] = {"n_triplets": ntrip, **hist}
        print(f"\n=== expert='{mode}'  anchors={'diverse(FPS)' if diverse else 'LOCAL'}  "
              f"({ntrip} triplets total) ===")
        print(f"  round:        " + " ".join(f"{r:>6d}" for r in range(len(hist['belly']))))
        for f in ["belly", "tail", "stripe", "cheeks"]:
            print(f"  {f:>7} kNN: " + " ".join(f"{v:6.3f}" for v in hist[f]))

    out = data.config.RESULTS_DIR / "exp34"
    out.mkdir(exist_ok=True)
    (out / "anchor_triplet_loop.json").write_text(json.dumps(report, indent=2))
    print(f"\nsaved -> {out/'anchor_triplet_loop.json'}")
    print("\n(round 0 = ordinary unsupervised PCA morphospace; later rounds = after morphing)")


if __name__ == "__main__":
    main()
