"""EXP-50 — How to build the live tag morphospace FAST on common hardware (CPU).

Splits the work into the FREQUENT per-tag update (must be snappy) vs the OCCASIONAL re-layout (can be ~1-3s).
Frequent update = retrain the 6 linear tag classifiers + project the new tag-score latent through a FROZEN
encoder. Benchmarks three TF-free frozen-encoder options for the per-update projection (speed + 8-way
silhouette), plus the occasional re-fit cost, at n=250 and tiled-up sizes to show scaling.

Run:  .venv/bin/python experiments/exp50_update_bench.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score
from sklearn.neural_network import MLPRegressor
from umap import UMAP

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import _tag_harness as H  # noqa: E402

UKW = dict(n_neighbors=20, min_dist=0.12, random_state=0)


def tag_scores(Z, lab, Yl):
    S = np.zeros((len(Z), 6), "float32")
    for t in range(6):
        y = Yl[:, t]
        if len(np.unique(y)) < 2:
            S[:, t] = float(y.mean()); continue
        S[:, t] = LogisticRegression(max_iter=2000).fit(Z[lab], y).decision_function(Z)
    return S


def t_ms(fn, reps=5):
    fn()  # warm up (numba JIT etc.)
    t = time.perf_counter()
    for _ in range(reps):
        out = fn()
    return (time.perf_counter() - t) / reps * 1000, out


def tile(Z, F, k):
    """Replicate the dataset k-fold with small jitter to benchmark scaling at larger n."""
    rng = np.random.default_rng(0)
    Zb = np.vstack([Z + rng.normal(0, 0.05, Z.shape) for _ in range(k)]).astype("float32")
    Fb = np.vstack([F] * k)
    return Zb, Fb


def bench(Z, F, n_label):
    N = len(Z); Y = H.tags_for(F); joint = F[:, 0] * 4 + F[:, 1] * 2 + F[:, 2]
    lab = H.labeled_set(N, n_label, 0)

    t_clf, S = t_ms(lambda: tag_scores(Z, lab, Y[lab]))           # the classifier retrain (per update)
    # a DIFFERENT latent (more tags) to project through the FROZEN encoder — the realistic per-update case
    S1 = tag_scores(Z, H.labeled_set(N, min(2 * n_label, N), 1), Y[H.labeled_set(N, min(2 * n_label, N), 1)])

    # OCCASIONAL re-fit: fit the reference UMAP + train the MLP cache (on S)
    um = UMAP(**UKW)
    t_umap_fit, Yref = t_ms(lambda: UMAP(**UKW).fit_transform(S), reps=1)
    um.fit(S)
    mlp = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=400, random_state=0)
    t_mlp_fit, _ = t_ms(lambda: mlp.fit(S, Yref), reps=1)

    # FREQUENT per-update PROJECTION options (frozen encoder applied to the CHANGED latent S1)
    t_umap_tf, Yt = t_ms(lambda: um.transform(S1))
    t_mlp_pred, Yp = t_ms(lambda: mlp.predict(S1))
    t_pca, Yq = t_ms(lambda: PCA(2, random_state=0).fit_transform(S1))

    def sil(E):
        return round(float(silhouette_score(np.asarray(E), joint)), 3)

    return {
        "N": N, "n_label": n_label, "clf_ms": round(t_clf, 1),
        "refit": {"umap_fit_ms": round(t_umap_fit, 1), "mlp_fit_ms": round(t_mlp_fit, 1)},
        "per_update_projection": {
            "UMAP.transform": {"ms": round(t_umap_tf, 1), "sil": sil(Yt)},
            "MLP.predict": {"ms": round(t_mlp_pred, 2), "sil": sil(Yp)},
            "PCA-2": {"ms": round(t_pca, 2), "sil": sil(Yq)},
        },
        "umap_fit_reference_sil": sil(Yref),
    }


def main():
    Z, F, joint, _ = H.load()
    import json
    print("== n=250 (fishy), 80 tags ==")
    print(json.dumps(bench(Z, F, 80), indent=2))
    for k in (4, 8):                                             # ~1000, ~2000 specimens
        Zb, Fb = tile(Z, F, k)
        print(f"\n== n={len(Zb)} (tiled x{k}), {min(80*k,len(Zb)//3)} tags ==")
        r = bench(Zb, Fb, min(80 * k, len(Zb) // 3))
        print(json.dumps({kk: r[kk] for kk in ("N", "clf_ms", "refit", "per_update_projection")}, indent=2))


if __name__ == "__main__":
    main()
