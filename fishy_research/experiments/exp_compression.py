"""EXP-19 — How does (JPEG-like) graph compression affect factor recoverability?

Compress each fish's per-vertex color via the GFT (DCT-on-a-graph): (a) low-pass = keep the K
lowest-frequency coefficients; (b) top-energy = keep the K largest-magnitude coefficients
(JPEG-quantization analog). Sweep compression strength and measure supervised balanced-acc
recovery of each factor. Hypothesis: high-energy factors (belly/tail) survive heavy compression;
the low-energy stripe signal is discarded first.
"""
from __future__ import annotations

import numpy as np
import warnings

warnings.filterwarnings("ignore")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fishpipe import config, data, spectral, metrics

RES = config.RESULTS_DIR / "gft"
RES.mkdir(parents=True, exist_ok=True)


def gft_coeffs(vlab, U):
    """Per-specimen GFT coefficients for L,a,b -> dict of (N, n_modes)."""
    out = {}
    for c, name in enumerate(["L", "a", "b"]):
        sig = vlab[:, :, c].astype(np.float64)
        sig = sig - sig.mean(axis=1, keepdims=True)
        out[name] = sig @ U                       # (N, n_modes)
    return out


def lowpass_feat(coeffs, K):
    return np.concatenate([coeffs[c][:, :K] for c in coeffs], axis=1)


def topenergy_feat(coeffs, K):
    """JPEG-like: per specimen keep K largest-|coeff| (others zeroed), as a fixed-length vector
    by masking (mask varies per specimen but the matrix stays dense with zeros)."""
    feats = []
    for c in coeffs:
        X = coeffs[c].copy()
        for i in range(X.shape[0]):
            thr_idx = np.argsort(np.abs(X[i]))[::-1][K:]
            X[i, thr_idx] = 0.0
        feats.append(X)
    return np.concatenate(feats, axis=1)


def main():
    gt = data.load_ground_truth(); fcd = data.build_face_colors()
    w, U = spectral.laplacian_eigenbasis(300)
    vlab = spectral.vertex_lab(fcd)
    coeffs = gft_coeffs(vlab, U)
    n_modes = U.shape[1]

    Ks = [2, 5, 10, 20, 40, 80, 150, 300]
    factors = ["belly", "tail", "stripe", "cheeks"]
    print(f"{'mode':>5} {'compress%':>9} " + " ".join(f"{f:>8}" for f in factors))
    rec_lp = {f: [] for f in factors}
    for K in Ks:
        X = lowpass_feat(coeffs, K)
        row = {f: metrics.factor_recoverability(X, gt.labels[f])["balanced_acc"] for f in factors}
        for f in factors:
            rec_lp[f].append(row[f])
        comp = 100 * (1 - (K * 3) / (n_modes * 3))
        print(f"{K:>5} {comp:>8.1f}% " + " ".join(f"{row[f]:>8.3f}" for f in factors))

    # plot
    fig, ax = plt.subplots(figsize=(8, 5))
    for f in factors:
        ax.plot(Ks, rec_lp[f], "-o", label=f)
    ax.axhline(0.5, ls="--", c="grey", lw=0.8, label="chance")
    ax.set_xscale("log"); ax.set_xlabel("# GFT modes kept per channel  (left = more compression)")
    ax.set_ylabel("recoverability (balanced acc)")
    ax.set_title("Effect of GFT (graph-DCT) compression on factor recoverability\n"
                 "belly/tail (low-freq, high-energy) survive; stripe needs more modes")
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(RES / "compression_sweep.png", dpi=130); plt.close(fig)
    print(f"\nsaved {RES/'compression_sweep.png'}")

    # JPEG-like top-energy at a couple of levels, for contrast
    print("\nJPEG-like (top-energy keep) — does keeping highest-energy coeffs help stripe?")
    for K in [10, 40]:
        X = topenergy_feat(coeffs, K)
        print(f"  keep top-{K} energy: " +
              " ".join(f"{f}={metrics.factor_recoverability(X, gt.labels[f])['balanced_acc']:.3f}" for f in factors))


if __name__ == "__main__":
    main()
