"""EXP-54 — Number of GFT modes `k` (the spectral_coeffs truncation) vs accuracy.

EXP-53 swept PCA dims on a FIXED k=40 GFT descriptor. Here we vary `k` itself = how many low-frequency
Laplacian modes are kept per channel, and classify directly on the k*channels coefficients (no PCA). Also:
which individual mode index carries the stripe signal (validates "stripes live in mid-frequency L modes").

Run:  .venv/bin/python experiments/exp54_k_vs_accuracy.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "interactive_demo"))
import engine  # noqa: E402
from fishpipe import spectral  # noqa: E402

RES = HERE.parent / "results" / "exp54"
RES.mkdir(parents=True, exist_ok=True)
KMAX = 300
KS = [1, 2, 3, 4, 6, 8, 12, 16, 20, 24, 30, 40, 50, 60, 80, 100, 150, 200, 300]


def coeffs_full():
    """Per-channel signed GFT coefficient matrices C[ch] = (N, KMAX), modes ordered low->high freq."""
    ld = engine.load_dataset("fishy")
    _, U = spectral.laplacian_eigenbasis(KMAX)                 # (Nv, KMAX) low-frequency modes
    vlab = spectral.vertex_lab(ld.fcd)                         # (N, Nv, 3)
    sig = {"L": vlab[..., 0], "a": vlab[..., 1], "b": vlab[..., 2]}
    C = {ch: (s - s.mean(1, keepdims=True)).astype(np.float64) @ U for ch, s in sig.items()}
    F = {f: np.asarray(ld.gt.labels[f]) for f in ("belly", "tail", "stripe")}
    return C, F


def cv(X, y):
    return float(cross_val_score(LogisticRegression(max_iter=3000), StandardScaler().fit_transform(X), y, cv=5).mean())


def main():
    C, F = coeffs_full()
    chans = ("L", "a", "b")
    print("k (modes/channel) |  belly   tail   stripe   (full L,a,b descriptor, dims=k*3)")
    curves = {f: [] for f in F}
    for k in KS:
        X = np.concatenate([C[ch][:, :k] for ch in chans], 1)
        for f in F:
            curves[f].append(cv(X, F[f]))
        print(f"  {k:4d}            |  " + "  ".join(f"{curves[f][-1]:.3f}" for f in ("belly", "tail", "stripe")))

    # per-mode stripe discriminability on the L channel: 1-feature logistic CV acc for each mode index
    stripe = F["stripe"]
    permode = [cv(C["L"][:, m:m + 1], stripe) for m in range(60)]

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    cm = {"belly": "#4e79a7", "tail": "#59a14f", "stripe": "#e15759"}
    for f in ("belly", "tail", "stripe"):
        ax[0].plot(KS, curves[f], "o-", color=cm[f], label=f)
    ax[0].set_xscale("log"); ax[0].set_xlabel("k = # GFT modes per channel (log)"); ax[0].set_ylabel("5-fold logistic accuracy")
    ax[0].set_ylim(0.5, 1.02); ax[0].set_title("accuracy vs k"); ax[0].grid(alpha=.2); ax[0].legend()
    base = 1 - stripe.mean()
    ax[1].bar(range(60), permode, color="#e15759"); ax[1].axhline(base, color="#888", ls=":", label=f"majority {base:.2f}")
    ax[1].set_xlabel("L-channel mode index (0 = lowest freq)"); ax[1].set_ylabel("stripe acc from that single mode")
    ax[1].set_ylim(0.5, 1.0); ax[1].set_title("which modes carry stripe?"); ax[1].legend()
    fig.suptitle("GFT k vs accuracy, and the stripe 'frequency band'", y=1.02)
    fig.tight_layout(); fig.savefig(RES / "k_vs_accuracy.png", dpi=130, bbox_inches="tight")

    top = np.argsort(permode)[::-1][:6]
    print(f"\nbelly/tail saturate by k≈{KS[next(i for i,k in enumerate(KS) if curves['belly'][i]>=0.99)]} ; "
          f"stripe best {max(curves['stripe']):.3f} at k={KS[int(np.argmax(curves['stripe']))]}")
    print(f"most stripe-discriminative single L modes (index: acc): " +
          ", ".join(f"{int(m)}:{permode[m]:.2f}" for m in top))
    print(f"figure -> {RES/'k_vs_accuracy.png'}")


if __name__ == "__main__":
    main()
