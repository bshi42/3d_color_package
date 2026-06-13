"""EXP-36b — Spring/energy 2D embedding vs. N-D metric-warp-then-2D-display, from the SAME
diverse-panel rankings. Definitive comparison of the two ways to 'morph the morphospace'.

Top row colored by true STRIPE (4 vs 5 — the overlapping factor we want to surface), bottom row
by BELLY (the dominant color factor). Columns: initial unsupervised | pure 2D spring | 2D display
of the N-D linear warp. kNN balanced-acc annotated (2D for display; the warp's N-D value too).
"""
from __future__ import annotations

import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import balanced_accuracy_score

from fishpipe import config, data, textons, semisup

warnings.filterwarnings("ignore")
RES = config.RESULTS_DIR / "exp36"; RES.mkdir(parents=True, exist_ok=True)


def fps(Y, k, rng, start=None):
    n = len(Y); i0 = int(rng.integers(n)) if start is None else start
    idx = [i0]; d = np.linalg.norm(Y - Y[i0], axis=1)
    while len(idx) < k:
        j = int(np.argmax(d)); idx.append(j); d = np.minimum(d, np.linalg.norm(Y - Y[j], axis=1))
    return idx


def kb(Y, y, k=7):
    return balanced_accuracy_score(y, cross_val_predict(
        KNeighborsClassifier(k), Y, y, cv=StratifiedKFold(5, shuffle=True, random_state=0)))


def main():
    fcd = data.build_face_colors(verbose=False); gt = data.load_ground_truth()
    Pop = textons.diffusion_operator(); jet = textons.local_jet(fcd, scales=(2, 4), P=Pop)
    sck, km = textons.build_codebook(jet, K=128, seed=0)
    X = StandardScaler().fit_transform(textons.encode(fcd, jet, sck, km, mode="bow"))
    Y0 = PCA(2, random_state=0).fit_transform(X); Y0 = (Y0 - Y0.mean(0)) / Y0.std(0)

    z = lambda c: StandardScaler().fit_transform(gt.params[[c]].to_numpy()).ravel()
    Pmetric = np.column_stack([3.0 * z("stripe_count"), z("stripe_spacing"), z("stripe_width")])
    st, be = gt.labels["stripe"], gt.labels["belly"]

    # accumulate diverse-panel, pattern-ranked triplets (same set feeds both transforms)
    rng = np.random.default_rng(0); trips = []
    for _ in range(8):
        for a in fps(Y0, 12, rng):
            pan = fps(Y0, 10, rng, start=a)[1:]; cand, rem = pan[:-1], pan[-1]
            order = sorted(cand, key=lambda j: np.linalg.norm(Pmetric[j] - Pmetric[a]))
            trips += semisup.triplets_from_ranking(a, order, remote=rem)

    # (1) pure 2D spring
    Ysp = semisup.spring_embed(Y0.copy(), trips, n_iter=600, lr=0.01, margin=0.4, lam=0.01)
    Ysp = (Ysp - Ysp.mean(0)) / (Ysp.std(0) + 1e-9)
    # (2) N-D linear warp (same triplets) then 2D PCA display
    L0 = PCA(12, random_state=0).fit(X).components_.T
    L = semisup.triplet_warp(X, trips, n_dim=12, L0=L0, n_epochs=40, lr=0.02, reg=2e-2)
    Xw = X @ L
    Yw = PCA(2, random_state=0).fit_transform(Xw); Yw = (Yw - Yw.mean(0)) / Yw.std(0)

    cols = [("INITIAL (unsup PCA)", Y0, Y0),
            ("pure 2D SPRING", Ysp, Ysp),
            ("N-D warp -> 2D display", Yw, Xw)]
    print("stripe kNN: initial2D %.3f | spring2D %.3f | warp-2D %.3f | warp-ND %.3f"
          % (kb(Y0, st), kb(Ysp, st), kb(Yw, st), kb(Xw, st)))
    print("belly  kNN: initial2D %.3f | spring2D %.3f | warp-2D %.3f"
          % (kb(Y0, be), kb(Ysp, be), kb(Yw, be)))

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    for c, (ttl, Y2, Ynd) in enumerate(cols):
        sk2 = kb(Y2, st); bk2 = kb(Y2, be)
        for r, (lab, name, cmap) in enumerate([(st, "stripe (red=5,blue=4)", ["#1f77b4", "#d62728"]),
                                               (be, "belly color group", ["#2ca02c", "#9467bd"])]):
            ax = axes[r, c]
            ax.scatter(Y2[:, 0], Y2[:, 1], c=[cmap[v] for v in lab], s=30, edgecolor="k", linewidth=0.3)
            k2 = sk2 if r == 0 else bk2
            extra = ""
            if c == 2 and r == 0:
                extra = f"\n(N-D warp space kNN={kb(Xw, st):.2f})"
            ax.set_title(f"{ttl}\n{name} — 2D kNN={k2:.2f}{extra}", fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Morphing the fishy morphospace from diverse-panel pattern rankings: "
                 "spring vs N-D-warp\n(stripe is the overlapping factor we try to surface; "
                 "belly is the dominant color factor)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(RES / "transform_compare.png", dpi=120); plt.close(fig)
    print(f"saved -> {RES/'transform_compare.png'}")


if __name__ == "__main__":
    main()
