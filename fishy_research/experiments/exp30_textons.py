"""EXP-30 — Decoupled texton/VLAD representation vs the GFT-coefficient descriptor.

Tests whether learning a texture vocabulary from millions of pooled local patches (huge
effective n) and summarizing each specimen as a VLAD/BoW over that shared codebook gives a
more recoverable and/or more LABEL-EFFICIENT per-specimen descriptor than the team's best
general descriptor (signed GFT coeffs k=40). Reports the standard recoverability scorecard
and a small-sample stability check.
"""
from __future__ import annotations

import time
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from fishpipe import data, spectral, textons, metrics


def main():
    fcd = data.build_face_colors(verbose=False)
    gt = data.load_ground_truth()

    print("Reference: GFT signed coeffs k=40 (team's best general descriptor)")
    Xg = spectral.spectral_coeffs(fcd, k=40, channels=("L", "a", "b"))
    print(metrics.format_scorecard(metrics.scorecard(Xg, gt.labels), ""))

    for mode in ("bow", "vlad"):
        for K in (64, 128):
            t0 = time.time()
            P = textons.diffusion_operator()
            jet = textons.local_jet(fcd, scales=(2, 4), P=P)
            scaler, km = textons.build_codebook(jet, K=K, seed=0)
            X = textons.encode(fcd, jet, scaler, km, mode=mode)
            dt = time.time() - t0
            # PCA-reduce VLAD (high-dim) for a fair morphospace-style comparison
            Xr = PCA(n_components=min(40, X.shape[1] - 1), random_state=0).fit_transform(
                StandardScaler().fit_transform(X))
            print(f"\n--- texton {mode.upper()} K={K}  ({X.shape[1]}-d -> PCA40)  built in {dt:.1f}s ---")
            print(metrics.format_scorecard(metrics.scorecard(Xr, gt.labels), ""))


if __name__ == "__main__":
    main()
