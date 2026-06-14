"""De-risking probe for the interactive morphospace demo.

Runs the FULL intended pipeline on BOTH datasets and times every stage:
 per-region block descriptor -> PCA -> region-saliency C -> one spring-morph round
 -> thumbnail render -> region-heatmap render. Saves sample PNGs to eyeball orientation.

Verifies the recon spec's risky claims empirically before we write the app:
  * mussel loads + descriptor builds (HDLSS n=31), local_jet on 245798 faces is fast enough
  * region attribution C[r,k] is well-formed for both
  * spring_embed morphs a single PC axis sanely
  * thumbnails render right-side-up (fishy: render_side_view, mussel: render_textured_rgba)
"""
from __future__ import annotations

import sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from fishpipe import config, data, textons, render, features
from fishpipe.dataset import MUSSELS, build_face_colors as ds_build

OUT = Path(__file__).resolve().parent / "_probe_out"
OUT.mkdir(exist_ok=True)
PDIM = 11


def clk(t0, msg):
    print(f"    [{time.time()-t0:6.1f}s] {msg}")


def region_labels_for(fcd, R, cache_dir, seed=0):
    p = cache_dir / f"region_labels_{R}.npy"
    if p.exists():
        return np.load(p)
    lab = KMeans(n_clusters=R, n_init=4, random_state=seed).fit_predict(fcd.centroids).astype(np.int32)
    np.save(p, lab)
    return lab


def region_desc(lab, chroma, jet, faces):
    af, bf = lab[:, faces, 1], lab[:, faces, 2]
    Lf = lab[:, faces, 0]
    am = np.argmax(chroma[:, faces], axis=1)
    color = np.column_stack([Lf.mean(1), af.mean(1), bf.mean(1),
                             af[np.arange(len(am)), am], bf[np.arange(len(am)), am]])
    bp = jet[:, faces, 6]; ms = jet[:, faces, 7]; ls = jet[:, faces, 8]
    tex = np.column_stack([np.abs(bp).mean(1), bp.std(1), np.percentile(np.abs(bp), 90, 1),
                           ls.mean(1), np.abs(ms).mean(1), (Lf < 20).mean(1)])
    return np.hstack([color, tex])


def build_descriptor(fcd, R, cache_dir):
    t0 = time.time()
    reg = region_labels_for(fcd, R, cache_dir)
    clk(t0, f"region_labels R={R} -> reg{reg.shape} ({len(np.unique(reg))} regions)")
    lab = features.rgb_to_lab(fcd.colors)
    chroma = np.sqrt(lab[..., 1] ** 2 + lab[..., 2] ** 2)
    clk(t0, f"rgb_to_lab -> {lab.shape}")
    jet = textons.local_jet(fcd, scales=(2, 4), cache_dir=cache_dir)
    clk(t0, f"local_jet -> {jet.shape}")
    rids = [r for r in range(R) if (reg == r).sum() >= 20]
    blocks = [region_desc(lab, chroma, jet, np.where(reg == r)[0]) for r in rids]
    X = np.hstack(blocks)
    clk(t0, f"descriptor X -> {X.shape}  (rids={len(rids)})")
    return X, reg, rids


def run(tag, fcd, mesh, cache_dir, R, thumb_idx, render_fn):
    print(f"\n=== {tag}  (N={len(fcd.names)}, Nf={fcd.colors.shape[1]}) ===")
    X, reg, rids = build_descriptor(fcd, R, cache_dir)
    Xs = StandardScaler().fit_transform(X)
    K = min(10, len(fcd.names) - 1)
    t0 = time.time()
    pca = PCA(K, random_state=0).fit(Xs)
    Z = Xs @ pca.components_.T
    clk(t0, f"PCA K={K}; evr[:6]={np.round(pca.explained_variance_ratio_[:6],3)}")

    # region saliency C[r,k]
    W = pca.components_
    C = np.zeros((len(rids), K))
    for i in range(len(rids)):
        blk = slice(i * PDIM, (i + 1) * PDIM)
        C[i] = (W[:, blk] ** 2).sum(1)
    # color vs pattern share of PC0
    col_share = sum((W[0, i*PDIM:i*PDIM+5] ** 2).sum() for i in range(len(rids)))
    pat_share = sum((W[0, i*PDIM+5:(i+1)*PDIM] ** 2).sum() for i in range(len(rids)))
    print(f"    PC0 color-share={col_share:.2f} pattern-share={pat_share:.2f}")

    # one spring-morph round on PC0 with a synthetic ranking
    from fishpipe import semisup
    t0 = time.time()
    y0 = Z[:, 0:1].copy()
    order = np.argsort(y0[:, 0])
    anchor = int(order[len(order)//2])
    # diverse neighbors via farthest-point on y0
    cand = list(order[::max(1, len(order)//8)])[:6]
    cand = [c for c in cand if c != anchor][:5]
    ranked = sorted(cand, key=lambda j: abs(y0[j,0]-y0[anchor,0]))  # near->far (proxy)
    trips = semisup.triplets_from_ranking(anchor, ranked)
    y = semisup.spring_embed(y0, trips, n_iter=400, lr=0.01, margin=0.4, lam=0.06)
    y = (y - y.mean()) / (y.std() + 1e-9)
    moved = np.abs((y[:,0]-y[:,0].mean()) - (y0[:,0]-y0[:,0].mean())/ (y0[:,0].std()+1e-9))
    clk(t0, f"spring_embed {len(trips)} trips; mean|Δ(norm PC0)|={moved.mean():.3f} (anchor={anchor})")

    # thumbnail render
    b = (mesh.vertices[:, 2].min(), mesh.vertices[:, 2].max(),
         mesh.vertices[:, 1].min(), mesh.vertices[:, 1].max())
    t0 = time.time()
    img = render_fn(mesh, fcd.colors[thumb_idx], b)
    Image.fromarray(img[..., :3] if img.shape[-1] == 4 else img).save(OUT / f"{tag}_thumb.png")
    clk(t0, f"thumbnail render -> {tag}_thumb.png  {img.shape}")

    # region heatmap for PC0
    t0 = time.time()
    score_r = np.zeros(reg.max() + 1)
    for i, r in enumerate(rids):
        score_r[r] = C[i, 0]
    fscore = score_r[reg]
    s = (fscore - fscore.min()) / (np.ptp(fscore) + 1e-9)
    face_rgb = (cm.inferno(s)[:, :3] * 255).astype(np.uint8)
    himg = render_fn(mesh, face_rgb, b)
    Image.fromarray(himg[..., :3] if himg.shape[-1] == 4 else himg).save(OUT / f"{tag}_heatmap_pc0.png")
    clk(t0, f"heatmap render -> {tag}_heatmap_pc0.png  {himg.shape}")


def fishy_render(mesh, face_rgb, b):
    return render.render_side_view(mesh, face_rgb, px=240, bounds=b)


def mussel_render(mesh, face_rgb, b):
    # mussel-tuned textured view (transparent bg, landscape-aligned)
    return render.render_textured_rgba(mesh, face_rgb, px=240, supersample=2, crop=True, align=True)


def main():
    # fishy
    fcd_f = data.build_face_colors(verbose=False)
    mesh_f = data.load_mesh()
    run("fishy", fcd_f, mesh_f, config.CACHE_DIR, R=128, thumb_idx=0, render_fn=fishy_render)

    # mussel
    fcd_m, _ = ds_build(MUSSELS, force=False, verbose=False)
    mesh_m = MUSSELS.load_mesh()
    run("mussel", fcd_m, mesh_m, MUSSELS.cache_dir, R=48, thumb_idx=0, render_fn=mussel_render)

    print(f"\nsaved sample PNGs -> {OUT}")


if __name__ == "__main__":
    main()
