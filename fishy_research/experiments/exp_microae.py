"""EXP-16 — Micro autoencoder on region-reduced fishy: can a tiny nonlinear model surface stripe?

Face-region-reduce (73728 faces -> R regions), then train a TINY MLP autoencoder (small
bottleneck, exploiting that fishy has only ~17 true generative params), and check whether ANY
latent dim (or projection-pursuit axis on the latent) recovers stripe — vs a linear PCA baseline
and a column-shuffled null. Label-free; labels used only to score recovery.
"""
from __future__ import annotations

import numpy as np
import warnings

warnings.filterwarnings("ignore")
from diptest import diptest
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from fishpipe import config, data, features


def ae_latent(X, bottleneck=6, hidden=24, seeds=(0, 1, 2)):
    """Tiny MLP autoencoder; return bottleneck activations (averaged over seeds, concatenated)."""
    Xs = StandardScaler().fit_transform(X)
    lats = []
    for s in seeds:
        mlp = MLPRegressor(hidden_layer_sizes=(hidden, bottleneck, hidden), activation="tanh",
                           alpha=1e-2, max_iter=3000, random_state=s, learning_rate_init=3e-3)
        mlp.fit(Xs, Xs)
        a = np.tanh(Xs @ mlp.coefs_[0] + mlp.intercepts_[0])
        a = np.tanh(a @ mlp.coefs_[1] + mlp.intercepts_[1])      # bottleneck activations
        lats.append(StandardScaler().fit_transform(a))
    return np.concatenate(lats, axis=1)


def best_axis(emb, y):
    return max(max(roc_auc_score(y, emb[:, k]), 1 - roc_auc_score(y, emb[:, k])) for k in range(emb.shape[1]))


def stripe_dip(emb, y):
    aucs = [max(roc_auc_score(y, emb[:, k]), 1 - roc_auc_score(y, emb[:, k])) for k in range(emb.shape[1])]
    j = int(np.argmax(aucs)); return diptest(emb[:, j])[0]


def main():
    gt = data.load_ground_truth(); fcd = data.build_face_colors()
    R = 256
    reps = {
        "region-Lab": features.region_summary(fcd, R, "mean", "lab"),
        "region-L-only": features.region_summary(fcd, R, "mean", "lab").reshape(250, R, 3)[..., 0],
    }
    rng = np.random.default_rng(0)
    for rname, X in reps.items():
        Xn = np.column_stack([rng.permutation(X[:, j]) for j in range(X.shape[1])])
        print(f"\n=== representation: {rname} {X.shape} ===")
        methods = {
            "PCA(6)": lambda M: PCA(6, random_state=0).fit_transform(StandardScaler().fit_transform(M)),
            "micro-AE(b=6)": lambda M: ae_latent(M, bottleneck=6),
            "micro-AE(b=12)": lambda M: ae_latent(M, bottleneck=12),
        }
        for mname, fn in methods.items():
            emb = fn(X); embn = fn(Xn)
            row = " ".join(f"{f}={best_axis(emb, gt.labels[f]):.2f}/{best_axis(embn, gt.labels[f]):.2f}"
                           for f in ["belly", "tail", "stripe", "cheeks"])
            sd, sdn = stripe_dip(emb, gt.labels["stripe"]), stripe_dip(embn, gt.labels["stripe"])
            print(f"  {mname:14s} (real/null AUC): {row}   | stripe-axis dip real={sd:.3f} null={sdn:.3f}")


if __name__ == "__main__":
    main()
