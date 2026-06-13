"""EXP-20 — Pattern morphospace on compressed signed GFT coefficients (EXP-19 improvements).

(1) descriptor = signed GFT coefficients (not band energy); (2) compressed to k=40 modes (denoise).
Builds the fishy ACHROMATIC (lightness) pattern morphospace on this representation, label-free,
with side-view exemplars; reports the net-effect scorecard and — transparently — what each PC
actually encodes (count vs spacing vs width), so the axes aren't mislabeled.
"""
from __future__ import annotations

import numpy as np
import warnings

warnings.filterwarnings("ignore")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, roc_auc_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from fishpipe import config, data, spectral, metrics

RES = config.RESULTS_DIR / "fishy_pattern"
RDIR = config.CACHE_DIR / "fishy_renders"


def fps(pts, k, seed=0):
    r = np.random.default_rng(seed); idx = [int(r.integers(len(pts)))]
    d = np.linalg.norm(pts - pts[idx[0]], axis=1)
    for _ in range(k - 1):
        j = int(np.argmax(d)); idx.append(j); d = np.minimum(d, np.linalg.norm(pts - pts[j], axis=1))
    return sorted(set(idx))


def main():
    gt = data.load_ground_truth(); fcd = data.build_face_colors(); names = fcd.names; df = gt.params

    # ---- (1)+(2) net-effect scorecard: coefficients(k=40) vs band-energy ----
    print("Net effect — supervised balanced-acc (factor recoverability):")
    print(f"{'rep':28s} {'belly':>7} {'tail':>7} {'stripe':>7} {'cheeks':>7}")
    reps = {
        "band-energy (old, all ch)": spectral.spectral_descriptor(fcd, k=300, n_bands=12),
        "coeffs k=40 (new, L,a,b)": spectral.spectral_coeffs(fcd, k=40, channels=("L", "a", "b")),
        "coeffs k=40 (L only=pattern)": spectral.spectral_coeffs(fcd, k=40, channels=("L",)),
    }
    for rn, X in reps.items():
        print(f"  {rn:26s} " + " ".join(
            f"{metrics.factor_recoverability(X, gt.labels[f])['balanced_acc']:>7.3f}"
            for f in ["belly", "tail", "stripe", "cheeks"]))

    # ---- ACHROMATIC pattern morphospace on compressed L coefficients ----
    Xp = spectral.spectral_coeffs(fcd, k=40, channels=("L",))
    emb = PCA(6, random_state=0).fit_transform(StandardScaler().fit_transform(Xp))
    ys = gt.labels["stripe"]
    # transparency: what does each PC encode?
    print("\nWhat each lightness-pattern PC encodes (|corr| with stripe params):")
    print(f"{'PC':>4} {'count':>7} {'spacing':>8} {'width':>7} {'offset':>7}  -> stripe AUC")
    for k in range(6):
        cc = [abs(pearsonr(emb[:, k], df[p])[0]) for p in
              ["stripe_count", "stripe_spacing", "stripe_width", "stripe_longitudinal_offset"]]
        auc = max(roc_auc_score(ys, emb[:, k]), 1 - roc_auc_score(ys, emb[:, k]))
        print(f"PC{k+1:>2} " + " ".join(f"{c:>7.2f}" for c in cc) + f"   {auc:.2f}")
    # label-free auto-cluster
    Z = StandardScaler().fit_transform(emb[:, :2])
    gm = min((GaussianMixture(kk, random_state=0).fit(Z) for kk in range(1, 5)), key=lambda g: g.bic(Z))
    cl = gm.predict(Z)
    print(f"\nauto-cluster on top-2 lightness-pattern PCs: k={len(set(cl))}, "
          f"ARI vs stripe (validation)={adjusted_rand_score(ys, cl):.3f}")

    # ---- figure: top-2 PCs, side-view exemplars, stripe outlines ----
    sc = emb[:, :2]
    Zf = StandardScaler().fit_transform(sc)
    sel = [int(np.where(ys == 0)[0][i]) for i in fps(Zf[ys == 0], 18)] + \
          [int(np.where(ys == 1)[0][i]) for i in fps(Zf[ys == 1], 18)]
    fig, ax = plt.subplots(figsize=(15, 10))
    for i in sel:
        im = Image.open(RDIR / (names[i] + ".png")).convert("RGB"); im.thumbnail((115, 115))
        col = "#d62728" if ys[i] == 1 else "#1f77b4"
        ax.add_artist(AnnotationBbox(OffsetImage(np.asarray(im)), (sc[i, 0], sc[i, 1]),
                     frameon=True, pad=0.05, bboxprops=dict(edgecolor=col, lw=2.2)))
    ax.scatter([], [], edgecolors="#d62728", facecolors="none", label="true 5-stripe", s=90)
    ax.scatter([], [], edgecolors="#1f77b4", facecolors="none", label="true 4-stripe", s=90)
    ax.set_xlim(sc[sel, 0].min() - 1, sc[sel, 0].max() + 1)
    ax.set_ylim(sc[sel, 1].min() - 1, sc[sel, 1].max() + 1)
    ax.set_xlabel("Lightness-pattern PC1 (compressed signed GFT coeffs, k=40)")
    ax.set_ylabel("Lightness-pattern PC2")
    ax.set_title("Fishy — PATTERN morphospace on compressed signed GFT coefficients (achromatic, denoised).\n"
                 "Label-free axes (no banding prior); balanced exemplars; outline=true stripe (validation).")
    ax.legend(loc="upper right"); fig.tight_layout()
    out = RES / "fishy_coeff_pattern_morphospace.png"
    fig.savefig(out, dpi=120); plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
