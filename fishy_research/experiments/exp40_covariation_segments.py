"""EXP-40 — Covariation-based mesh parcellation (owner-proposed).

(1) Build the GLOBAL descriptor as a concatenation of per-region color+texture descriptors.
(2) Global PCA -> each PC's squared loadings PARTITION by region, giving a region x PC
    'contribution' matrix C[r,k] = fraction of PC k's direction living in region r (cols sum to 1).
(3) Each region = a signature vector over the K PCs -> cluster regions -> MERGE into segments
    = a data-driven atlas of body parts that CO-VARY across the population (morphological
    integration/modularity, Klingenberg, applied to colour+pattern).
(4) Re-run analysis on the segments: per-segment clustering + a segment-summary morphospace.

Validates whether segments align with anatomy (belly/tail/dorsal/head) and whether the
segment-based view recovers factors better/more-interpretably than arbitrary tiles (EXP-38).
"""
from __future__ import annotations

import json, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from PIL import Image
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score

from fishpipe import config, data, textons, render
from fishpipe.features import rgb_to_lab, _region_labels

warnings.filterwarnings("ignore")
RES = config.RESULTS_DIR / "exp40"; RES.mkdir(parents=True, exist_ok=True)
FACTORS = ["belly", "tail", "stripe", "cheeks"]
PDIM = 11  # per-region descriptor dim (5 color + 6 texture)


def region_desc(lab, chroma, jet, faces):
    af, bf = lab[:, faces, 1], lab[:, faces, 2]
    am = np.argmax(chroma[:, faces], 1)
    color = np.column_stack([lab[:, faces, 0].mean(1), af.mean(1), bf.mean(1),
                             np.take_along_axis(af, am[:, None], 1)[:, 0],
                             np.take_along_axis(bf, am[:, None], 1)[:, 0]])
    bp = np.abs(jet[:, faces, 6]); ls = jet[:, faces, 8]; dark = 100 - lab[:, faces, 0]
    tex = np.column_stack([bp.mean(1), bp.std(1), np.percentile(bp, 90, 1),
                           ls.mean(1), np.abs(jet[:, faces, 7]).mean(1), (dark > 50).mean(1)])
    return np.hstack([color, tex])


def gmm2_ari(X, y):
    return adjusted_rand_score(y, GaussianMixture(2, covariance_type="full", random_state=0)
                              .fit_predict(StandardScaler().fit_transform(X)))


def main():
    fcd = data.build_face_colors(verbose=False); gt = data.load_ground_truth()
    mesh = data.load_mesh()
    lab = rgb_to_lab(fcd.colors); chroma = np.sqrt(lab[..., 1] ** 2 + lab[..., 2] ** 2)
    P = textons.diffusion_operator(); jet = textons.local_jet(fcd, scales=(2, 4), P=P)
    R = 128; reg = _region_labels(R); cents = fcd.centroids
    region_ids = [r for r in range(R) if (reg == r).sum() >= 20]

    # (1) concat per-region descriptors -> global descriptor
    blocks = {r: region_desc(lab, chroma, jet, np.where(reg == r)[0]) for r in region_ids}
    X = np.hstack([blocks[r] for r in region_ids])                # (N, len*PDIM)
    Xs = StandardScaler().fit_transform(X)

    # (2) global PCA; region x PC contribution = sum of squared loadings on region's block
    K = 20
    pca = PCA(K, random_state=0).fit(Xs)
    W = pca.components_                                            # (K, n_features)
    evr = pca.explained_variance_ratio_
    C = np.zeros((len(region_ids), K))
    for i, r in enumerate(region_ids):
        blk = slice(i * PDIM, (i + 1) * PDIM)
        C[i] = (W[:, blk] ** 2).sum(1)                            # cols ~sum to 1 over regions

    # (3) cluster region signatures (EV-weighted) -> segments
    sig = C * np.sqrt(evr)[None, :]                               # weight by PC importance
    sig = sig / (np.linalg.norm(sig, axis=1, keepdims=True) + 1e-9)
    M = 7
    Z = linkage(sig, method="ward")
    seg_of_region = fcluster(Z, M, criterion="maxclust")          # 1..M
    seg = -np.ones(R, dtype=int)
    for i, r in enumerate(region_ids):
        seg[r] = seg_of_region[i] - 1

    # render the segmentation
    seg_rgb = (np.array([cm.tab10(s % 10)[:3] if s >= 0 else (0.1, 0.1, 0.1)
                         for s in seg]) * 255).astype(np.uint8)
    face_rgb = seg_rgb[reg]
    b = (mesh.vertices[:, 2].min(), mesh.vertices[:, 2].max(),
         mesh.vertices[:, 1].min(), mesh.vertices[:, 1].max())
    Image.fromarray(render.render_side_view(mesh, face_rgb, px=1000, bounds=b)).save(RES / "segments.png")

    # (4) per-segment analysis: descriptor over each segment, GMM-2 ARI vs factors
    print(f"{M} covariation segments; per-segment GMM-2 ARI vs each factor + segment anatomy:")
    print(f"  {'seg':>3}{'#faces':>8}{'meanZ':>7}{'meanY':>7}  {'belly':>6}{'tail':>6}{'stripe':>7}{'cheeks':>7}  driving PCs")
    seg_report = {}
    for s in range(M):
        faces = np.where(seg[reg] == s)[0]
        if len(faces) < 20:
            continue
        Xseg = region_desc(lab, chroma, jet, faces)
        aris = {f: round(gmm2_ari(Xseg, gt.labels[f]), 2) for f in FACTORS}
        # which PCs this segment drives most (mean contribution over its regions)
        rs = [i for i, r in enumerate(region_ids) if seg[r] == s]
        toppc = (np.argsort(C[rs].mean(0))[::-1][:3] + 1).tolist()
        zc, yc = cents[faces, 2].mean(), cents[faces, 1].mean()
        print(f"  {s:>3}{len(faces):>8}{zc:>7.2f}{yc:>7.2f}  {aris['belly']:>6}{aris['tail']:>6}"
              f"{aris['stripe']:>7}{aris['cheeks']:>7}  PC{toppc}")
        seg_report[s] = {"n_faces": int(len(faces)), "ari": aris, "top_pcs": toppc,
                         "centroid_zy": [round(float(zc), 2), round(float(yc), 2)]}

    # step-4b: rerun PCA on a SEGMENT-SUMMARY descriptor (each specimen -> M segments x feats)
    seg_summary = np.hstack([region_desc(lab, chroma, jet, np.where(seg[reg] == s)[0])
                             for s in range(M) if (seg[reg] == s).sum() >= 20])
    Ys = PCA(6, random_state=0).fit_transform(StandardScaler().fit_transform(seg_summary))
    from fishpipe import metrics
    print("\nSegment-summary morphospace recoverability (balanced acc):")
    print(metrics.format_scorecard(metrics.scorecard(Ys, gt.labels)))

    (RES / "covariation_segments.json").write_text(json.dumps(
        {"M": M, "segments": seg_report,
         "explained_var_top5": [round(float(e), 3) for e in evr[:5]]}, indent=2))
    print(f"\nsaved segmentation render + json -> {RES}")


if __name__ == "__main__":
    main()
