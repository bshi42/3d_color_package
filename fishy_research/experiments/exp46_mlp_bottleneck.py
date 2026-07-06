"""EXP-46 — Tiny MLP bottleneck (numpy). The literal 'classifier latent space'.

Architecture (no torch available, so a small hand-written numpy net):
    Z(24) --W1--> ReLU(hidden=16) --W2--> Linear(2) [= latent] --W3--> sigmoid(6) [tag scores]

Trained with multi-label binary cross-entropy on the labeled specimens only, full-batch
Adam, modest epochs, small L2 weight decay (keeps tiny budgets from overfitting). The 2-unit
bottleneck layer is returned as the latent embedding for ALL 250 specimens; the 6 sigmoid
outputs are the per-tag scores. Factor-agnostic: 6 independent multi-label targets.

Deterministic: a fixed seed derived from the labeled budget, so each eval rep / budget is
reproducible (the harness re-samples labeled sets with its own seeds; our net seed is fixed).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import silhouette_score

sys.path.insert(0, "experiments")
import _tag_harness as H  # noqa: E402

HIDDEN = 16
LATENT = 2
N_TAGS = 6
EPOCHS = 400
LR = 0.05
WD = 1e-3          # L2 weight decay
SEED = 0


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def _init(rng, fan_in, fan_out):
    # He-ish init for ReLU stacks, small for the rest; scaled by 1/sqrt(fan_in).
    return (rng.standard_normal((fan_in, fan_out)) * np.sqrt(2.0 / fan_in)).astype(np.float64)


def _train_mlp(Xl, Yl, in_dim, seed):
    """Full-batch Adam training of the 3-layer net on labeled data. Returns weight dict."""
    rng = np.random.default_rng(seed)
    W1 = _init(rng, in_dim, HIDDEN);   b1 = np.zeros(HIDDEN)
    W2 = _init(rng, HIDDEN, LATENT);   b2 = np.zeros(LATENT)
    W3 = _init(rng, LATENT, N_TAGS);   b3 = np.zeros(N_TAGS)
    params = {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "W3": W3, "b3": b3}

    # Adam state
    m = {k: np.zeros_like(v) for k, v in params.items()}
    v = {k: np.zeros_like(v) for k, v in params.items()}
    b1a, b2a, eps = 0.9, 0.999, 1e-8
    n = max(len(Xl), 1)

    for t in range(1, EPOCHS + 1):
        # ---- forward ----
        z1 = Xl @ params["W1"] + params["b1"]
        a1 = np.maximum(z1, 0.0)                 # ReLU hidden
        lat = a1 @ params["W2"] + params["b2"]   # linear 2-D latent
        logits = lat @ params["W3"] + params["b3"]
        p = _sigmoid(logits)

        # ---- backward (BCE summed over 6 tags, mean over samples) ----
        dlogits = (p - Yl) / n                   # (n,6)
        gW3 = lat.T @ dlogits + WD * params["W3"]
        gb3 = dlogits.sum(0)
        dlat = dlogits @ params["W3"].T          # (n,2)
        gW2 = a1.T @ dlat + WD * params["W2"]
        gb2 = dlat.sum(0)
        da1 = dlat @ params["W2"].T
        dz1 = da1 * (z1 > 0)
        gW1 = Xl.T @ dz1 + WD * params["W1"]
        gb1 = dz1.sum(0)
        grads = {"W1": gW1, "b1": gb1, "W2": gW2, "b2": gb2, "W3": gW3, "b3": gb3}

        # ---- Adam step ----
        for k in params:
            m[k] = b1a * m[k] + (1 - b1a) * grads[k]
            v[k] = b2a * v[k] + (1 - b2a) * (grads[k] ** 2)
            mhat = m[k] / (1 - b1a ** t)
            vhat = v[k] / (1 - b2a ** t)
            params[k] -= LR * mhat / (np.sqrt(vhat) + eps)
    return params


def _forward_all(Z, params):
    a1 = np.maximum(Z @ params["W1"] + params["b1"], 0.0)
    lat = a1 @ params["W2"] + params["b2"]
    p = _sigmoid(lat @ params["W3"] + params["b3"])
    return p, lat


def model_fn(Z, labeled_idx, Ytags_labeled):
    Z = np.asarray(Z, float)
    Xl = Z[labeled_idx]
    Yl = np.asarray(Ytags_labeled, float)
    # Deterministic: net seed fixed (does not peek at unlabeled labels).
    params = _train_mlp(Xl, Yl, Z.shape[1], SEED)
    tag_scores, latent = _forward_all(Z, params)
    return {"tag_scores": tag_scores, "latent": latent}


def main():
    name = "mlp_bottleneck"
    res = H.eval_model(model_fn)
    print(f"== {name} ==")
    print("  budget |  acc_mean  acc_stripe  ari8")
    for n, mrow in res.items():
        print(f"  {n:5d}  |   {mrow['acc_mean']:.3f}     {mrow['acc_stripe']:.3f}     {mrow['ari8']:.3f}")

    # Latent silhouette proxy at n_labeled=80 over ALL specimens vs the true 8-way joint.
    Z, F, joint, _ = H.load()
    lab80 = H.labeled_set(len(Z), 80, 0)
    out80 = model_fn(Z, lab80, H.tags_for(F)[lab80])
    L = out80["latent"]
    sil = float(silhouette_score(L, joint))
    print(f"  latent silhouette (n=80, all specimens, 8-way joint): {sil:.4f}")

    # Figures
    p1 = H.RESULTS / "latent_mlp_bottleneck_n40.png"
    p2 = H.RESULTS / "latent_mlp_bottleneck_n160.png"
    H.viz(model_fn, 40, p1, "mlp_bottleneck . 40 tagged")
    H.viz(model_fn, 160, p2, "mlp_bottleneck . 160 tagged")
    print("  figures:", p1, p2)
    return res, sil


if __name__ == "__main__":
    main()
