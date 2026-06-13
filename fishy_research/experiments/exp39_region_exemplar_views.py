"""EXP-39 — How the region analysis shapes the biologist-facing 2D EXEMPLAR plot.

The region scan turns ONE global morphospace into a NAVIGABLE SET of region-scoped morphospaces:
the label-free structure map is the navigator (click a region that lights up), and each region
gets its own 2D exemplar plot in which THAT region's factor is the dominant axis. So the same
250 fish reorganize to show belly-colour groups (belly region), a striping gradient (dorsal
region), or the rare red outliers (cheek region) — structure the single global plot hides.

Builds one composite figure: structure-map navigator (top) + global morphospace + 3 region-scoped
morphospaces (bottom), each with side-view fish exemplars, points/borders colored by the LABEL-FREE
GMM+BIC auto-cluster of that view (titles report ARI vs the matching factor, validation only).
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
from diptest import diptest
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score

from fishpipe import config, data, textons, render
from fishpipe.features import rgb_to_lab, _region_labels
from fishpipe.recommended import auto_cluster

warnings.filterwarnings("ignore")
RDIR = config.CACHE_DIR / "fishy_renders_hi"
RES = config.RESULTS_DIR / "exp39"; RES.mkdir(parents=True, exist_ok=True)
PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]


def fps(Y, k, seed=0):
    rng = np.random.default_rng(seed); n = len(Y); idx = [int(rng.integers(n))]
    d = np.linalg.norm(Y - Y[idx[0]], axis=1)
    while len(idx) < k:
        j = int(np.argmax(d)); idx.append(j); d = np.minimum(d, np.linalg.norm(Y - Y[j], axis=1))
    return sorted(set(idx))


def color_desc(lab, chroma, faces):
    af, bf = lab[:, faces, 1], lab[:, faces, 2]
    am = np.argmax(chroma[:, faces], 1)
    return np.column_stack([lab[:, faces, 0].mean(1), af.mean(1), bf.mean(1),
                            np.take_along_axis(af, am[:, None], 1)[:, 0],
                            np.take_along_axis(bf, am[:, None], 1)[:, 0]])


def texture_desc(jet, lab, faces):
    bp = np.abs(jet[:, faces, 6]); ls = jet[:, faces, 8]; dark = 100 - lab[:, faces, 0]
    return np.column_stack([bp.mean(1), bp.std(1), np.percentile(bp, 90, 1),
                            ls.mean(1), (dark > 50).mean(1)])


def morpho_panel(ax, X, imgs, gt, title, factor, seed=0):
    Xs = StandardScaler().fit_transform(X)
    Y = PCA(2, random_state=0).fit_transform(Xs)
    Yn = (Y - Y.mean(0)) / (Y.std(0) + 1e-9)
    lab = auto_cluster(Y, max_k=4)                               # label-free coloring
    ari = adjusted_rand_score(gt.labels[factor], lab)
    keep = fps(Yn, 26, seed)
    for i in keep:
        im = Image.fromarray(imgs[i]).copy(); im.thumbnail((300, 300))
        ax.add_artist(AnnotationBbox(OffsetImage(np.asarray(im), zoom=0.26), (Yn[i, 0], Yn[i, 1]),
                      frameon=True, pad=0.03,
                      bboxprops=dict(edgecolor=PALETTE[lab[i] % len(PALETTE)], lw=2.2)))
    pad = 0.6
    ax.set_xlim(Yn[keep, 0].min() - pad, Yn[keep, 0].max() + pad)
    ax.set_ylim(Yn[keep, 1].min() - pad, Yn[keep, 1].max() + pad)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{title}\nauto-clusters vs {factor}: ARI={ari:.2f}", fontsize=10)


def main():
    fcd = data.build_face_colors(verbose=False); gt = data.load_ground_truth()
    mesh = data.load_mesh(); names = fcd.names
    imgs = [np.asarray(Image.open(RDIR / f"{n}.png").convert("RGB")) for n in names]
    lab = rgb_to_lab(fcd.colors); chroma = np.sqrt(lab[..., 1] ** 2 + lab[..., 2] ** 2)
    P = textons.diffusion_operator(); jet = textons.local_jet(fcd, scales=(2, 4), P=P)
    R = 128; reg = _region_labels(R); cents = fcd.centroids

    # pick regions by LABEL-FREE criteria (the structure-map navigator picks these for the user)
    dipc = np.zeros(R); Lvar = np.zeros(R); cy = np.zeros(R); cz = np.zeros(R)
    for r in range(R):
        f = np.where(reg == r)[0]
        if len(f) < 20:
            dipc[r] = -1; continue
        Xs = StandardScaler().fit_transform(color_desc(lab, chroma, f))
        dipc[r] = max(diptest(PCA(2, random_state=0).fit_transform(Xs)[:, j])[0] for j in range(2))
        Lvar[r] = lab[:, f, 0].var(0).mean(); cy[r] = cents[f, 1].mean(); cz[r] = cents[f, 2].mean()
    belly_r = int(np.argmax(dipc))                               # most bimodal colour region
    dorsal_r = int(np.argmax(np.where(cy > np.median(cy), Lvar, -1)))   # dorsal + high L-variance
    # head/cheek region: most anterior (extreme z) region with high texture-edge clusterability
    head_mask = cz < np.percentile(cz, 20)
    tex_dip = np.array([max(diptest(PCA(2, random_state=0).fit_transform(
        StandardScaler().fit_transform(texture_desc(jet, lab, np.where(reg == r)[0])))[:, j])[0]
        for j in range(2)) if (reg == r).sum() >= 20 else -1 for r in range(R)])
    cheek_r = int(np.argmax(np.where(head_mask, tex_dip, -1)))

    # combined global descriptor for the global panel
    from fishpipe import features
    Xg = np.hstack([StandardScaler().fit_transform(features.area_hist(fcd, 24, "lab")),
                    StandardScaler().fit_transform(
                        textons.encode(fcd, jet, *textons.build_codebook(jet, K=128, seed=0), mode="bow"))])

    fig = plt.figure(figsize=(19, 11))
    gs = GridSpec(2, 4, height_ratios=[0.8, 1.3], hspace=0.22, wspace=0.12)
    # navigator: structure map (color clusterability) with the 3 picked regions outlined
    axn = fig.add_subplot(gs[0, :])
    score = (dipc - dipc[dipc > -1].min()); score[dipc < 0] = 0
    fscore = np.zeros(len(reg))
    for r in range(R):
        fscore[reg == r] = score[r]
    hm = (cm.viridis((fscore - fscore.min()) / (np.ptp(fscore) + 1e-9))[:, :3] * 255).astype(np.uint8)
    for rr, c in [(belly_r, [0, 220, 220]), (dorsal_r, [255, 120, 0]), (cheek_r, [255, 0, 200])]:
        hm[reg == rr] = (0.25 * hm[reg == rr] + 0.75 * np.array(c)).astype(np.uint8)
    b = (mesh.vertices[:, 2].min(), mesh.vertices[:, 2].max(),
         mesh.vertices[:, 1].min(), mesh.vertices[:, 1].max())
    axn.imshow(render.render_side_view(mesh, hm, px=620, bounds=b)); axn.axis("off")
    axn.set_title("NAVIGATOR: label-free structure map (bright=bimodal colour structure). "
                  "Outlined: belly-region (cyan), dorsal-region (orange), head/cheek-region (magenta).",
                  fontsize=11)

    panels = [
        (gs[1, 0], Xg, "GLOBAL morphospace", "belly"),
        (gs[1, 1], color_desc(lab, chroma, np.where(reg == belly_r)[0]), "BELLY-region (color)", "belly"),
        (gs[1, 2], texture_desc(jet, lab, np.where(reg == dorsal_r)[0]), "DORSAL-region (texture)", "stripe"),
        (gs[1, 3], texture_desc(jet, lab, np.where(reg == cheek_r)[0]), "HEAD/CHEEK-region (texture)", "cheeks"),
    ]
    for cell, X, ttl, fac in panels:
        morpho_panel(fig.add_subplot(cell), X, imgs, gt, ttl, fac)

    fig.suptitle("EXP-39 — region analysis turns ONE global plot into navigable region-scoped "
                 "exemplar morphospaces (borders = label-free auto-clusters)", fontsize=13)
    fig.savefig(RES / "region_exemplar_views.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"picked regions: belly={belly_r} dorsal={dorsal_r} cheek={cheek_r}")
    print(f"saved -> {RES/'region_exemplar_views.png'}")


if __name__ == "__main__":
    main()
