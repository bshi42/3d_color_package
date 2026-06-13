"""Real-data generality test — mussel/clam half-shells (module output, n=31).

No ground truth, so we judge by INTERPRETABILITY: auto-cluster the color and the (general)
graph-spectral pattern morphospaces, then render per-cluster EXEMPLAR montages so a human can
see whether each cluster is a coherent shell type. Also checks robustness to baking artifacts
(black patches) via population-median imputation.
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from fishpipe import config, dataset, embed, features, recommended, spectral

RES = config.RESULTS_DIR / "mussels"
RES.mkdir(parents=True, exist_ok=True)
DS = dataset.MUSSELS
TEX = DS.texture_dir


def montage_by_cluster(names, labels, title, out):
    """Grid of specimen thumbnails grouped (one row block) per cluster."""
    cell = 150
    order = np.argsort(labels)
    clusters = sorted(set(labels))
    cols = max(int(np.bincount(labels).max()), 1)
    rows = len(clusters)
    sheet = Image.new("RGB", (cols * cell + 60, rows * cell), (30, 30, 30))
    from PIL import ImageDraw

    d = ImageDraw.Draw(sheet)
    for r, c in enumerate(clusters):
        members = [n for n, l in zip(names, labels) if l == c]
        d.text((4, r * cell + cell // 2), f"C{c}\nn={len(members)}", fill=(255, 255, 0))
        for j, nm in enumerate(members):
            p = os.path.join(TEX, nm + ".png")
            if not os.path.exists(p):
                continue
            im = Image.open(p).convert("RGB").resize((cell - 6, cell - 18))
            x, y = 60 + j * cell, r * cell
            sheet.paste(im, (x + 2, y + 2))
            d.text((x + 4, y + cell - 14), nm.replace("UF_IZ_", ""), fill=(200, 255, 200))
    sheet.save(out)
    return out


def scatter(scores, labels, names, title, out):
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(scores[:, 0], scores[:, 1], c=labels, cmap="tab10", s=80, edgecolors="k")
    for i, nm in enumerate(names):
        ax.annotate(nm.replace("UF_IZ_", ""), (scores[i, 0], scores[i, 1]), fontsize=6, alpha=0.7)
    ax.set_title(title); ax.set_xlabel("Dim1"); ax.set_ylabel("Dim2")
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)
    return out


def main():
    print("== MUSSELS (n=31 real shells) ==")
    fcd, artifact = dataset.build_face_colors(DS)
    names = fcd.names
    print(f"mesh faces={fcd.colors.shape[1]}, specimens={len(names)}")
    print(f"total artifact faces imputed: {int(artifact.sum())} "
          f"({100*artifact.mean():.2f}% of all (specimen,face) cells)")

    # ---------- COLOR morphospace ----------
    Xc = features.area_hist(fcd, n_clusters=24, color_space="lab")
    emc = embed.pca(Xc, n_components=min(6, len(names) - 1), standardize=True)
    lab_c = recommended.auto_cluster(emc.scores[:, :2], max_k=5)
    print(f"\nCOLOR: PCA EV(1-3)%={np.round(emc.explained_variance[:3]*100,1)}; "
          f"auto-clusters={len(set(lab_c))} sizes={np.bincount(lab_c).tolist()}")
    scatter(emc.scores, lab_c, names, "Mussels — COLOR morphospace (auto-clustered)",
            RES / "color_morphospace.png")
    montage_by_cluster(names, lab_c, "color clusters", RES / "color_clusters_montage.png")

    # ---------- PATTERN morphospace (general graph-spectral engine) ----------
    mesh = DS.load_mesh()
    Xp = spectral.spectral_descriptor(fcd, k=200, n_bands=14, channels=("L", "a", "b", "chroma"),
                                      mesh=mesh, cache_dir=DS.cache_dir)
    emp = embed.pca(Xp, n_components=min(6, len(names) - 1), standardize=True)
    lab_p = recommended.auto_cluster(emp.scores[:, :2], max_k=5)
    print(f"PATTERN: spectral dim={Xp.shape[1]}; auto-clusters={len(set(lab_p))} "
          f"sizes={np.bincount(lab_p).tolist()}")
    scatter(emp.scores, lab_p, names, "Mussels — PATTERN morphospace (graph-spectral, auto-clustered)",
            RES / "pattern_morphospace.png")
    montage_by_cluster(names, lab_p, "pattern clusters", RES / "pattern_clusters_montage.png")

    # ---------- artifact robustness ----------
    print("\nARTIFACT ROBUSTNESS:")
    per_spec = artifact.sum(axis=1)
    worst = np.argsort(per_spec)[::-1][:4]
    # raw (no imputation): re-insert black at artifact faces, recompute color morphospace
    raw = fcd.colors.copy()
    for s in range(len(names)):
        raw[s, artifact[s]] = 0
    from fishpipe.data import FaceColorData
    fcd_raw = FaceColorData(colors=raw, areas=fcd.areas, centroids=fcd.centroids, names=names)
    Xc_raw = features.area_hist(fcd_raw, n_clusters=24, color_space="lab")
    em_raw = embed.pca(Xc_raw, n_components=4, standardize=True)
    cen_i = emc.scores[:, :2].mean(0); cen_r = em_raw.scores[:, :2].mean(0)
    for j in worst:
        di = np.linalg.norm(emc.scores[j, :2] - cen_i)
        dr = np.linalg.norm(em_raw.scores[j, :2] - cen_r)
        z_i = di / np.linalg.norm(emc.scores[:, :2] - cen_i, axis=1).std()
        z_r = dr / np.linalg.norm(em_raw.scores[:, :2] - cen_r, axis=1).std()
        print(f"  {names[j]:18s}: {per_spec[j]:6d} artifact faces | outlier z: "
              f"imputed={z_i:.2f}  raw(black)={z_r:.2f}")
    print("  (raw z >> imputed z ⇒ imputation removed the artifact-driven outlier)")
    print(f"\nFigures in {RES}")


if __name__ == "__main__":
    main()
