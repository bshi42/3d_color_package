"""Mussels — orientation-aware (Gabor) pattern descriptor + EXEMPLAR pattern morphospace.

(b) Tests whether a Gabor filter bank (orientation-aware) isolates the wavy green-ray pattern
    of 507450 better than the magnitude-only graph-spectral descriptor.
Also renders a pattern morphospace whose points are the shells' EXTERIOR images.
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image
from sklearn.preprocessing import StandardScaler

import imageio.v2 as imageio
from fishpipe import config, dataset, embed, exterior, spectral, texture2d

RES = config.RESULTS_DIR / "mussels"
EXT = RES / "exterior"
EXT.mkdir(parents=True, exist_ok=True)
DS = dataset.MUSSELS
TEX = str(DS.texture_dir)


def short(n):
    return n.replace("UF_IZ_", "")


def nn(X, q, k=6):
    Z = StandardScaler().fit_transform(X)
    d = np.linalg.norm(Z - Z[q], axis=1)
    return [i for i in np.argsort(d) if i != q][:k], d


def main():
    fcd, _ = dataset.build_face_colors(DS)
    names = fcd.names
    q = names.index("UF_IZ_507450")

    # ---- extract + cache exterior crops ----
    crops = []
    for n in names:
        p = EXT / (n + ".png")
        if p.exists():
            crops.append(np.asarray(Image.open(p).convert("RGB")))
        else:
            c = exterior.crop_box_frac(imageio.imread(os.path.join(TEX, n + ".png")))
            Image.fromarray(c).save(p)
            crops.append(c)
    print(f"exterior crops: {len(crops)} (e.g. {crops[0].shape})")

    # ---- (b) Gabor descriptor + directionality ----
    G = texture2d.gabor_descriptor_set(crops, channels=("L", "a"))
    print(f"Gabor descriptor: {G.shape}")
    # orientation anisotropy (how directional / wavy) per specimen
    aniso = []
    for c in crops:
        _, prof = texture2d.gabor_features(c, channels=("L", "a"), return_orient_profile=True)
        prof = prof / (prof.sum() + 1e-9)
        aniso.append(prof.max() - prof.min())          # peaky orientation profile => directional
    aniso = np.array(aniso)

    print("\n== 507450 retrieval comparison ==")
    g_nn, gd = nn(G, q)
    print("  GABOR (orientation-aware) nearest:", [(short(names[i]), round(gd[i], 2)) for i in g_nn])
    mesh = DS.load_mesh()
    Xp = spectral.spectral_descriptor(fcd, k=200, n_bands=14, channels=("L", "a", "b", "chroma"),
                                      mesh=mesh, cache_dir=DS.cache_dir)
    s_nn, sd = nn(Xp, q)
    print("  SPECTRAL (magnitude-only) nearest:", [(short(names[i]), round(sd[i], 2)) for i in s_nn])
    print(f"\n  507450 directionality(anisotropy) rank: "
          f"{(aniso >= aniso[q]).sum()} of {len(names)} (1=most directional/wavy)")
    print("  top-5 most directional shells:", [short(names[i]) for i in np.argsort(aniso)[::-1][:5]])

    # ---- exemplar pattern morphospace (Gabor PCA, points = exterior images) ----
    emb = embed.pca(G, n_components=2, standardize=True)
    sc = emb.scores
    fig, ax = plt.subplots(figsize=(13, 10))
    for i, n in enumerate(names):
        im = Image.open(EXT / (n + ".png")).convert("RGB")
        im.thumbnail((90, 90))
        ab = AnnotationBbox(OffsetImage(np.asarray(im), zoom=1.0), (sc[i, 0], sc[i, 1]),
                            frameon=True, pad=0.1,
                            bboxprops=dict(edgecolor=("red" if i == q else "0.4"),
                                           lw=(2.5 if i == q else 0.6)))
        ax.add_artist(ab)
        ax.annotate(short(n), (sc[i, 0], sc[i, 1]), fontsize=6, xytext=(0, -34),
                    textcoords="offset points", ha="center", color="red" if i == q else "0.3")
    pad = 1.0
    ax.set_xlim(sc[:, 0].min() - pad, sc[:, 0].max() + pad)
    ax.set_ylim(sc[:, 1].min() - pad, sc[:, 1].max() + pad)
    ax.set_xlabel("Pattern-PC1 (Gabor)"); ax.set_ylabel("Pattern-PC2 (Gabor)")
    ax.set_title("Mussels — PATTERN morphospace with shell-exterior exemplars "
                 "(orientation-aware Gabor; 507450 in red)")
    fig.tight_layout()
    fig.savefig(RES / "pattern_morphospace_exemplars.png", dpi=130)
    plt.close(fig)
    print(f"\nsaved {RES/'pattern_morphospace_exemplars.png'}")


if __name__ == "__main__":
    main()
