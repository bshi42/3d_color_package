"""Mussels — charisma-Figure-5-style dendrograms (Schwartz et al. 2026, MEE).

Figure 5 = phylogeny + per-species colour-presence dots + exemplar image + ancestral colour
wheels. We have NO phylogeny for these specimens, so we substitute PHENETIC DENDROGRAMS
(hierarchical clustering). We emit THREE SEPARATE FILES, one per clustering criterion, each a
complete standalone panel (tree + exemplars + colour-presence dots, ordered by that tree):
  colour            -> mussels_dendro_colour.png          (ward on palette-composition PCs)
  pattern           -> mussels_dendro_pattern.png         (ward on signed-GFT PCs, colour-blind)
  colour+pattern    -> mussels_dendro_colour_pattern.png  (ward on both concatenated; integrative)
Exemplars are actual textured-MESH exterior renders (exp_mussels_render_exemplars.py), not texture
crops — the shared atlas valve shaded by each specimen's per-face colours. Real data, no labels.
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
from scipy.cluster.hierarchy import linkage, dendrogram
from skimage import color as skcolor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from fishpipe import config, dataset, features, segment, spectral

DS = dataset.MUSSELS
EXT = config.RESULTS_DIR / "mussels" / "exterior_render"   # actual textured-MESH renders (not crops)
OUT = config.RESULTS_DIR / "mussels"
K = 10
PRESENT = 0.03      # area fraction to call a colour "present"
DPI = 260           # doubled resolution
ZOOM = 0.45         # exemplar thumbnail zoom (tuned so the landscape valve fills the row pitch w/o overlap)
ROW = 0.46          # vertical inches per row


def named_color(lab):
    """Coarse, earthy-palette-appropriate name for a Lab centroid (legend only)."""
    L, a, b = lab
    C = np.hypot(a, b)
    if L < 22:
        return "black"
    if C < 9:
        return "white" if L > 82 else ("light-grey" if L > 55 else "grey")
    h = np.degrees(np.arctan2(b, a)) % 360
    if a < -2 and 90 < h < 165:
        return "olive" if L < 60 else "green"
    if h < 95:                                # reddish-to-yellow (the mussel earth tones)
        if L < 50:
            return "dark-brown"
        if C < 28:
            return "tan" if L > 62 else "brown"
        return "yellow"
    return "olive"


def load_render(path, h=340):
    """Load a pre-rendered exterior-MESH valve (RGBA, transparent bg, already landscape)."""
    im = Image.open(path).convert("RGBA")
    im.thumbnail((h, h))                       # fit long (horizontal) axis to h
    return np.asarray(im)


def mussel_adjacency():
    import scipy.sparse as sp
    cache = DS.cache_dir / "face_adjacency.npz"
    if cache.exists():
        return sp.load_npz(cache)
    fv = DS.load_mesh().face_v
    Nf = fv.shape[0]
    e = np.sort(np.vstack([fv[:, [0, 1]], fv[:, [1, 2]], fv[:, [2, 0]]]), axis=1)
    fid = np.tile(np.arange(Nf), 3)
    o = np.lexsort((e[:, 1], e[:, 0]))
    es, fs = e[o], fid[o]
    same = np.all(es[1:] == es[:-1], axis=1)
    r, c = fs[:-1][same], fs[1:][same]
    A = sp.coo_matrix((np.ones(2 * len(r)), (np.r_[r, c], np.r_[c, r])), shape=(Nf, Nf)).tocsr()
    A.data[:] = 1.0
    sp.save_npz(cache, A)
    return A


def members(Z, node, N):
    if node < N:
        return [node]
    k = node - N
    return members(Z, int(Z[k, 0]), N) + members(Z, int(Z[k, 1]), N)


def draw_tree(axT, Z, N, frac, order_c, cent_rgb, title):
    """Horizontal dendrogram (leaves on the right) with clade-dominant-colour node squares.
    Returns ypos: leaf -> row index (this tree's optimal ordering)."""
    leaves = dendrogram(Z, no_plot=True)["leaves"]
    ypos = {leaf: r for r, leaf in enumerate(leaves)}
    nodeY, nodeX = dict(ypos), {l: 0.0 for l in range(N)}
    maxd = Z[:, 2].max()
    for k in range(len(Z)):
        c1, c2 = int(Z[k, 0]), int(Z[k, 1]); d = Z[k, 2]
        y1, y2 = nodeY[c1], nodeY[c2]; x1, x2 = nodeX[c1], nodeX[c2]
        for (x, y) in [(x1, y1), (x2, y2)]:
            axT.plot([-x, -d], [y, y], "-", color="0.35", lw=1.1)
        axT.plot([-d, -d], [y1, y2], "-", color="0.35", lw=1.1)
        nodeY[N + k] = (y1 + y2) / 2; nodeX[N + k] = d
        if d > maxd * 0.16:                            # node marker = clade dominant colour
            mem = members(Z, N + k, N)
            dom = order_c[np.argmax([frac[mem].mean(0)[c] for c in order_c])]
            axT.scatter(-d, (y1 + y2) / 2, s=90, marker="s",
                        color=cent_rgb[dom], edgecolors="0.3", lw=0.6, zorder=5)
    axT.set_xlim(-maxd * 1.05, maxd * 0.05); axT.set_ylim(-1, N); axT.axis("off")
    axT.set_title(title, fontsize=13, fontweight="bold")
    return ypos


def draw_exemplars(axI, ypos, fcd, names, N):
    for leaf, r in ypos.items():
        im = load_render(EXT / (fcd.names[leaf] + ".png"))
        axI.add_artist(AnnotationBbox(OffsetImage(im, zoom=ZOOM, dpi_cor=False), (0.62, r),
                       frameon=False, zorder=3))
        axI.annotate(names[leaf], (0.0, r), fontsize=8, va="center", ha="left")
    axI.set_xlim(0, 1.25); axI.set_ylim(-1, N); axI.axis("off")
    axI.set_title("specimen", fontsize=11)


def draw_dots(axD, ypos, frac, order_c, cent_rgb, cnames, N):
    for leaf, r in ypos.items():
        for col, c in enumerate(order_c):
            present = frac[leaf, c] >= PRESENT
            axD.scatter(col, r, s=160, marker="o",
                        facecolor=cent_rgb[c] if present else "white",
                        edgecolors="0.45", lw=0.9, zorder=2)
    for col, c in enumerate(order_c):
        axD.scatter(col, N, s=160, marker="o", facecolor=cent_rgb[c], edgecolors="0.3")
        axD.text(col, N + 0.6, cnames[col], rotation=40, fontsize=8, ha="left", va="bottom")
    axD.set_xlim(-0.6, K - 0.4); axD.set_ylim(-1, N); axD.axis("off")
    axD.set_title("colour-class presence\n(filled = present, >3% surface area)", fontsize=11)


def main():
    fcd, _ = dataset.build_face_colors(DS)
    names = [n.replace("UF_IZ_", "") for n in fcd.names]
    seg = segment.segment(fcd, n_colors=K, smooth_iters=1, mesh_adjacency=mussel_adjacency())
    areas = fcd.areas
    N = len(names)
    frac = np.zeros((N, K))
    for i in range(N):
        for c in range(K):
            frac[i, c] = areas[seg.labels[i] == c].sum() / areas.sum()
    cent_lab = seg.centroids_lab
    cent_rgb = np.clip(skcolor.lab2rgb(cent_lab[None])[0], 0, 1)
    order_c = np.argsort(cent_lab[:, 0])              # palette columns dark -> light
    cnames = [named_color(cent_lab[c]) for c in order_c]

    # ---- three feature spaces -> three ward dendrograms ----
    colorPC = PCA(5, random_state=0).fit_transform(StandardScaler().fit_transform(np.sqrt(frac)))
    gft = spectral.spectral_coeffs(fcd, k=40, channels=("L", "a", "b"), n_basis=200,
                                   mesh=DS.load_mesh(), cache_dir=DS.cache_dir)
    patternPC = PCA(5, random_state=0).fit_transform(StandardScaler().fit_transform(gft))
    feat_color = StandardScaler().fit_transform(colorPC)
    feat_pattern = StandardScaler().fit_transform(patternPC)
    feat_both = np.concatenate([feat_color, feat_pattern], axis=1)

    panels = [
        ("colour", linkage(feat_color, "ward"), "COLOUR", "palette composition"),
        ("pattern", linkage(feat_pattern, "ward"), "PATTERN", "GFT spatial arrangement, colour-blind"),
        ("colour_pattern", linkage(feat_both, "ward"), "COLOUR + PATTERN",
         "integrative: colour-PCs + GFT pattern-PCs"),
    ]

    for key, Z, short, long in panels:
        fig = plt.figure(figsize=(16, N * ROW + 3))
        gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 0.85, 1.7], wspace=0.04)
        axT, axI, axD = (fig.add_subplot(gs[0, k]) for k in range(3))
        ypos = draw_tree(axT, Z, N, frac, order_c, cent_rgb, short)
        draw_exemplars(axI, ypos, fcd, names, N)
        draw_dots(axD, ypos, frac, order_c, cent_rgb, cnames, N)
        fig.suptitle(f"Mussels — {short} phenetic dendrogram ({long})\n"
                     "NOT a phylogeny; node square = clade-dominant colour. Real data, no labels.",
                     fontsize=13)
        out = OUT / f"mussels_dendro_{key}.png"
        fig.savefig(out, dpi=DPI, bbox_inches="tight"); plt.close(fig)
        print(f"saved {out}")
    print(f"(N={N}, K={K}, dpi={DPI})  palette={cnames}")


if __name__ == "__main__":
    main()
