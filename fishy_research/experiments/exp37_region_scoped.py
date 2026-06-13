"""EXP-37 — Region-localized axes + region-SCOPED comparison on a COMBINED color+pattern
morphospace (owner-proposed).

Three questions:
 (Q1) For a PC of the combined morphospace, WHERE on the body does it live? -> per-face saliency
      map (|corr| across population between each face's colour and the PC score) -> render +
      dominant region. Validate the region localizes to the right anatomy.
 (Q2) If the expert/analysis SCOPES to that region, does the factor become recoverable -- in
      particular does scoping crack the rare CHEEKS (area-normalization) and subtle STRIPE,
      UNSUPERVISED? (label-free ROI: cheeks via per-face novelty, stripe via lightness-variance.)
 (Q3) premise for per-axis morphing: a region-scoped sub-descriptor is low-dim & factor-dominated.
"""
from __future__ import annotations

import json, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.sparse as sp
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import balanced_accuracy_score, adjusted_rand_score
from sklearn.mixture import GaussianMixture

from fishpipe import config, data, features, textons, gating
from fishpipe.features import rgb_to_lab

warnings.filterwarnings("ignore")
RES = config.RESULTS_DIR / "exp37"; RES.mkdir(parents=True, exist_ok=True)


def combined_descriptor(fcd):
    color = features.area_hist(fcd, n_clusters=24, color_space="lab")          # composition
    P = textons.diffusion_operator(); jet = textons.local_jet(fcd, scales=(2, 4), P=P)
    sck, km = textons.build_codebook(jet, K=128, seed=0)
    pattern = textons.encode(fcd, jet, sck, km, mode="bow")                    # texture
    Xc = StandardScaler().fit_transform(color)
    Xp = StandardScaler().fit_transform(pattern)
    return np.hstack([Xc, Xp])                                                 # combined


def face_saliency(score, Lf, Cf):
    """|corr| across specimens between PC score and each face's L* and chroma; max of the two."""
    def corr(F):
        s = score - score.mean()
        Fc = F - F.mean(0)
        num = (s[:, None] * Fc).mean(0)
        den = s.std() * Fc.std(0) + 1e-9
        return np.abs(num / den)
    return np.maximum(corr(Lf), corr(Cf))


def largest_region(mask, A):
    """Largest connected component of `mask` faces under face adjacency A."""
    n_comp, lab = sp.csgraph.connected_components(A[mask][:, mask], directed=False)
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return mask
    sizes = np.bincount(lab)
    keep = idx[lab == sizes.argmax()]
    out = np.zeros(len(mask), bool); out[keep] = True
    return out


def heatmap_faces(scalar):
    s = (scalar - scalar.min()) / (np.ptp(scalar) + 1e-9)
    return (cm.inferno(s)[:, :3] * 255).astype(np.uint8)


def knn(X, y, k=7):
    ns = int(min(5, np.bincount(y).min()))
    if ns < 2 or X.shape[1] == 0:
        return float("nan")
    return balanced_accuracy_score(y, cross_val_predict(
        KNeighborsClassifier(k), X, y, cv=StratifiedKFold(ns, shuffle=True, random_state=0)))


def balacc(X, y):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    ns = int(min(5, np.bincount(y).min()))
    if ns < 2:
        return float("nan")
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
    return balanced_accuracy_score(y, cross_val_predict(m, X, y, cv=StratifiedKFold(ns, shuffle=True, random_state=0)))


def gmm_ari(X, y):
    Xs = StandardScaler().fit_transform(X)
    lab = GaussianMixture(2, covariance_type="full", random_state=0).fit_predict(Xs)
    return adjusted_rand_score(y, lab)


def roi_color_stats(fcd, roi):
    """Per-specimen colour summary over the ROI faces only (renormalized): [meanL, mean a*,
    mean b*, mean chroma, 95pct chroma, area-frac high-chroma]. Low-dim, interpretable."""
    lab = rgb_to_lab(fcd.colors)[:, roi, :]                     # (N, |roi|, 3)
    chroma = np.sqrt(lab[..., 1] ** 2 + lab[..., 2] ** 2)
    return np.column_stack([lab[..., 0].mean(1), lab[..., 1].mean(1), lab[..., 2].mean(1),
                            chroma.mean(1), np.percentile(chroma, 95, axis=1),
                            (chroma > 25).mean(1)])


def main():
    fcd = data.build_face_colors(verbose=False); gt = data.load_ground_truth()
    mesh = data.load_mesh()
    names = fcd.names
    A = sp.load_npz(config.CACHE_DIR / "face_adjacency.npz").tocsr()
    Lf = rgb_to_lab(fcd.colors)[..., 0]                         # (N, Nf)
    abf = rgb_to_lab(fcd.colors)[..., 1:]
    Cf = np.sqrt((abf ** 2).sum(-1))                            # (N, Nf) chroma

    X = combined_descriptor(fcd)
    pca = PCA(n_components=8, random_state=0).fit(StandardScaler().fit_transform(X))
    scores = pca.transform(StandardScaler().fit_transform(X))
    b = (mesh.vertices[:, 2].min(), mesh.vertices[:, 2].max(),
         mesh.vertices[:, 1].min(), mesh.vertices[:, 1].max())

    # ---- Q1: back-project each PC to a body region; correlate PC with each factor ----
    report = {"PC_localization": {}}
    fig, axes = plt.subplots(2, 4, figsize=(20, 7))
    for pc in range(8):
        sal = face_saliency(scores[:, pc], Lf, Cf)
        roi = largest_region(sal >= np.quantile(sal, 0.90), A)
        # correlate PC with factors (validation)
        corrs = {f: round(float(abs(np.corrcoef(scores[:, pc], gt.labels[f])[0, 1])), 2)
                 for f in ("belly", "tail", "stripe", "cheeks")}
        # render saliency map with ROI outline (brighten ROI)
        hm = heatmap_faces(sal)
        img = render_with_roi(mesh, hm, roi, b)
        ax = axes[pc // 4, pc % 4]; ax.imshow(img); ax.axis("off")
        top = max(corrs, key=corrs.get)
        ax.set_title(f"PC{pc+1} (EV {pca.explained_variance_ratio_[pc]*100:.0f}%)  "
                     f"~{top}={corrs[top]}\nregion={roi.sum()} faces", fontsize=9)
        report["PC_localization"][f"PC{pc+1}"] = {"corr": corrs, "roi_faces": int(roi.sum())}
    fig.suptitle("Q1: where each combined-morphospace PC lives (per-face |corr| with PC score; "
                 "cyan = dominant region). Title: factor it most tracks.", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(RES / "pc_regions.png", dpi=300); plt.close(fig)
    print("Q1 PC->factor correlations:")
    for k, v in report["PC_localization"].items():
        print(f"  {k}: {v['corr']}  (roi {v['roi_faces']} faces)")

    # ---- Q2: region-scoped recovery, label-free ROIs ----
    # cheeks ROI (label-free): faces where a MINORITY is extreme -> per-face chroma spread above median
    nov = features.population_novelty_map(fcd)                  # (N, Nf) per-face anomaly
    cheek_sal = np.percentile(nov, 96, axis=0) - np.median(nov, axis=0)   # minority-high faces
    cheek_roi = largest_region(cheek_sal >= np.quantile(cheek_sal, 0.985), A)
    # stripe ROI (label-free): faces with highest cross-population lightness variance (salient mask)
    stripe_sal = Lf.var(0)
    stripe_roi = stripe_sal >= np.quantile(stripe_sal, 0.85)

    # render the two ROIs for the figure
    for nm, sal, roi in [("cheek", cheek_sal, cheek_roi), ("stripe", stripe_sal, stripe_roi)]:
        Image.fromarray(render_with_roi(mesh, heatmap_faces(sal), roi, b)).save(RES / f"roi_{nm}.png")

    print("\nQ2: whole-fish vs REGION-SCOPED recovery (label-free ROI):")
    print(f"  {'factor/ROI':<22}{'sup balacc':>11}{'unsup kNN':>11}{'GMM ARI':>9}{'dip p':>8}{'SigClust p':>11}")
    rows = {}
    for fac, roi, roiname in [("cheeks", cheek_roi, f"cheek-novelty ({cheek_roi.sum()}f)"),
                              ("stripe", stripe_roi, f"stripe-Lvar ({stripe_roi.sum()}f)")]:
        y = gt.labels[fac]
        Xroi = roi_color_stats(fcd, roi)
        # also ROI texton histogram (richer pattern within ROI)
        for label, Xd in [("whole-fish combined", X),
                          (f"SCOPED:{roiname}", Xroi)]:
            from diptest import diptest
            Xs = StandardScaler().fit_transform(Xd)
            ax = PCA(min(6, Xs.shape[1]), random_state=0).fit_transform(Xs)
            dipp = min(diptest(ax[:, j])[1] for j in range(ax.shape[1]))
            sg = gating.sigclust(ax, n_sim=300)["pvalue"]
            r = {"sup_balacc": round(balacc(Xd, y), 3), "unsup_knn": round(knn(ax, y), 3),
                 "gmm_ari": round(gmm_ari(ax, y), 3), "dip_p": round(float(dipp), 3),
                 "sigclust_p": round(sg, 3)}
            rows[f"{fac}|{label}"] = r
            print(f"  {fac+': '+label:<22}{r['sup_balacc']:>11}{r['unsup_knn']:>11}"
                  f"{r['gmm_ari']:>9}{r['dip_p']:>8}{r['sigclust_p']:>11}")
    report["region_scoped"] = rows
    (RES / "region_scoped.json").write_text(json.dumps(report, indent=2))
    print(f"\nsaved -> {RES} (pc_regions.png, roi_cheek.png, roi_stripe.png, region_scoped.json)")


def render_with_roi(mesh, face_rgb, roi, bounds):
    """Render the heatmap; tint ROI faces cyan-ish so the region is visible."""
    from fishpipe import render
    fc = face_rgb.copy()
    fc[roi] = (0.2 * fc[roi] + 0.8 * np.array([0, 220, 220])).astype(np.uint8)
    return render.render_side_view(mesh, fc, px=480, bounds=bounds)


if __name__ == "__main__":
    main()
