"""EXP-41 — Exemplar morphospaces on the EXP-40 COVARIATION SEGMENTS (modules), not arbitrary tiles.

Composite figure: the module atlas (navigator) + the segment-summary morphospace + per-module
exemplar morphospaces. Fish thumbnails placed by each view's PCA-2; borders colored by a
LABEL-FREE continuous scale (PC1 of that view) so the structure reads whether it's clusters
(belly/tail = two PC1 extremes) or a gradient (dorsal striping). Titles report which factor PC1
tracks (validation only).
"""
from __future__ import annotations

import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
import matplotlib.cm as cm
from PIL import Image
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from fishpipe import config, data, textons, render
from fishpipe.features import rgb_to_lab, _region_labels

warnings.filterwarnings("ignore")
RDIR = config.CACHE_DIR / "fishy_renders_hi"
RES = config.RESULTS_DIR / "exp41"; RES.mkdir(parents=True, exist_ok=True)
PDIM = 11


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


def fps(Y, k, seed=0):
    rng = np.random.default_rng(seed); n = len(Y); idx = [int(rng.integers(n))]
    d = np.linalg.norm(Y - Y[idx[0]], axis=1)
    while len(idx) < k:
        j = int(np.argmax(d)); idx.append(j); d = np.minimum(d, np.linalg.norm(Y - Y[j], axis=1))
    return sorted(set(idx))


from diptest import diptest
from fishpipe.recommended import auto_cluster
from sklearn.metrics import adjusted_rand_score
CLUST_PAL = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]


def panel(ax, X, imgs, gt, title, n_ex=24):
    Y = PCA(2, random_state=0).fit_transform(StandardScaler().fit_transform(X))
    Yn = (Y - Y.mean(0)) / (Y.std(0) + 1e-9)
    pc1 = Yn[:, 0]; c01 = (pc1 - pc1.min()) / (np.ptp(pc1) + 1e-9)
    corr = {f: abs(np.corrcoef(pc1, gt.labels[f])[0, 1]) for f in ("belly", "tail", "stripe", "cheeks")}
    top = max(corr, key=corr.get)
    # cluster vs gradient: if a top axis is clearly bimodal -> show auto-clusters; else PC1 gradient
    dip = min(diptest(Yn[:, j])[1] for j in range(2))   # any axis bimodal -> discrete clusters
    clustered = dip < 0.05
    if clustered:
        cl = auto_cluster(Y, max_k=4)
        cols = [CLUST_PAL[c % len(CLUST_PAL)] for c in cl]
        sub = f"auto-clusters (discrete); PC1~{top} |r|={corr[top]:.2f}"
    else:
        cols = [cm.viridis(v) for v in c01]
        sub = f"PC1 gradient (border); tracks {top} |r|={corr[top]:.2f}"
    keep = fps(Yn, n_ex)
    for i in keep:
        im = Image.fromarray(imgs[i]).copy(); im.thumbnail((300, 300))
        ax.add_artist(AnnotationBbox(OffsetImage(np.asarray(im), zoom=0.26), (Yn[i, 0], Yn[i, 1]),
                      frameon=True, pad=0.03, bboxprops=dict(edgecolor=cols[i], lw=2.6)))
    pad = 0.6
    ax.set_xlim(Yn[keep, 0].min() - pad, Yn[keep, 0].max() + pad)
    ax.set_ylim(Yn[keep, 1].min() - pad, Yn[keep, 1].max() + pad)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{title}\n{sub}", fontsize=9.5)


def main():
    fcd = data.build_face_colors(verbose=False); gt = data.load_ground_truth()
    mesh = data.load_mesh(); names = fcd.names
    imgs = [np.asarray(Image.open(RDIR / f"{n}.png").convert("RGB")) for n in names]
    lab = rgb_to_lab(fcd.colors); chroma = np.sqrt(lab[..., 1] ** 2 + lab[..., 2] ** 2)
    P = textons.diffusion_operator(); jet = textons.local_jet(fcd, scales=(2, 4), P=P)
    R = 128; reg = _region_labels(R); cents = fcd.centroids
    rids = [r for r in range(R) if (reg == r).sum() >= 20]

    # --- EXP-40 covariation segments ---
    blocks = {r: region_desc(lab, chroma, jet, np.where(reg == r)[0]) for r in rids}
    Xcat = StandardScaler().fit_transform(np.hstack([blocks[r] for r in rids]))
    pca = PCA(20, random_state=0).fit(Xcat); evr = pca.explained_variance_ratio_
    C = np.vstack([(pca.components_[:, i * PDIM:(i + 1) * PDIM] ** 2).sum(1) for i in range(len(rids))])
    sig = C * np.sqrt(evr)[None, :]; sig /= (np.linalg.norm(sig, axis=1, keepdims=True) + 1e-9)
    M = 7
    seg_of = fcluster(linkage(sig, "ward"), M, "maxclust") - 1
    seg = -np.ones(R, int)
    for i, r in enumerate(rids):
        seg[r] = seg_of[i]

    # module atlas render
    seg_rgb = (np.array([cm.tab10(s % 10)[:3] if s >= 0 else (0.1, 0.1, 0.1) for s in seg]) * 255).astype(np.uint8)
    bnd = (mesh.vertices[:, 2].min(), mesh.vertices[:, 2].max(),
           mesh.vertices[:, 1].min(), mesh.vertices[:, 1].max())
    atlas = render.render_side_view(mesh, seg_rgb[reg], px=620, bounds=bnd)

    def anat(s):
        f = np.where(seg[reg] == s)[0]; z, y = cents[f, 2].mean(), cents[f, 1].mean()
        zside = "caudal" if z < -0.15 else ("anterior" if z > 0.18 else "mid")
        yside = "ventral" if y < -0.03 else ("dorsal" if y > 0.03 else "flank")
        return f"{yside}/{zside}"

    # pick modules to DISPLAY by clusterability (label-free): the 2 most-clustered (->belly/tail)
    # + the most-dorsal module (the stripe field, shown as a gradient).
    big = [s for s in range(M) if (seg[reg] == s).sum() >= 500]
    def mod_dip(s):
        Xs = StandardScaler().fit_transform(region_desc(lab, chroma, jet, np.where(seg[reg] == s)[0]))
        A = PCA(2, random_state=0).fit_transform(Xs)
        return max(diptest(A[:, j])[0] for j in range(2))
    dips = {s: mod_dip(s) for s in big}
    clustered2 = sorted(big, key=lambda s: -dips[s])[:2]
    dorsal = max((s for s in big if s not in clustered2),
                 key=lambda s: cents[np.where(seg[reg] == s)[0], 1].mean())
    sizes = [(s, int((seg[reg] == s).sum())) for s in clustered2 + [dorsal]]

    # segment-summary descriptor (each specimen -> all modules' summaries)
    seg_summary = np.hstack([region_desc(lab, chroma, jet, np.where(seg[reg] == s)[0])
                             for s in range(M) if (seg[reg] == s).sum() >= 20])

    fig = plt.figure(figsize=(18, 11))
    gs = GridSpec(2, 4, height_ratios=[0.8, 1.3], hspace=0.22, wspace=0.1)
    axA = fig.add_subplot(gs[0, :2]); axA.imshow(atlas); axA.axis("off")
    axA.set_title("Covariation-MODULE atlas (label-free); per-module exemplar views below", fontsize=11)
    panel(fig.add_subplot(gs[0, 2:]), seg_summary, imgs, gt, "SEGMENT-SUMMARY morphospace (all modules)")
    for cell, (s, n) in zip([gs[1, 1], gs[1, 2], gs[1, 3]], sizes):
        Xseg = region_desc(lab, chroma, jet, np.where(seg[reg] == s)[0])
        panel(fig.add_subplot(cell), Xseg, imgs, gt, f"MODULE {s} ({anat(s)}, {n}f)")
    # leftmost bottom cell: a 2nd module if available, else blank note
    if len(sizes) >= 4:
        pass
    fig.suptitle("EXP-41 — exemplar morphospaces on COVARIATION segments (borders = PC1, label-free)", fontsize=13)
    fig.savefig(RES / "segment_exemplars.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("modules shown:", [(s, anat(s), n) for s, n in sizes])
    print(f"saved -> {RES/'segment_exemplars.png'}")


if __name__ == "__main__":
    main()
