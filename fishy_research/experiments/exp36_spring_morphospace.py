"""EXP-36 — The owner's spec, faithfully: diverse ranking panel + spring/energy morphospace
+ per-iteration exemplar plots.

Differences from EXP-34/35 (which I should have done the first time):
  * DIVERSE candidate panel (farthest-point sample across the CURRENT space), not the anchor's
    kNN — directly fixes the filter-bubble problem.
  * SPRING / ENERGY transform (`semisup.spring_embed`: ranked-similar pull together, ranked-
    dissimilar push apart, tethered to the initial layout), not a linear metric warp.
  * PLOTS the initial morphospace with fish exemplars AND the transformed morphospace after
    every iteration (+ a montage and a GIF).
Simulated expert ranks a diverse panel vs the anchor by similarity in the TRUE generating
variables (pattern-focused: stripe-dominant, colour down-weighted, to surface the hidden factor).
"""
from __future__ import annotations

import json, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image
import imageio.v2 as imageio
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import balanced_accuracy_score

from fishpipe import config, data, textons, semisup

warnings.filterwarnings("ignore")
RDIR = config.CACHE_DIR / "fishy_renders"
RES = config.RESULTS_DIR / "exp36"; RES.mkdir(parents=True, exist_ok=True)


def fps(Y, k, rng, start=None):
    n = len(Y); i0 = int(rng.integers(n)) if start is None else start
    idx = [i0]; d = np.linalg.norm(Y - Y[i0], axis=1)
    while len(idx) < k:
        j = int(np.argmax(d)); idx.append(j)
        d = np.minimum(d, np.linalg.norm(Y - Y[j], axis=1))
    return idx


def perceptual(gt):
    p = gt.params
    z = lambda c: StandardScaler().fit_transform(p[[c]].to_numpy()).ravel()
    # pattern-focused expert: stripe count dominant + geometry; colour down-weighted
    return np.column_stack([3.0 * z("stripe_count"), z("stripe_spacing"), z("stripe_width"),
                            0.3 * z("belly_hue"), 0.3 * z("tail_hue")])


def knn_balacc(Y, y, k=7):
    ns = int(min(5, np.bincount(y).min()))
    if ns < 2:
        return float("nan")
    return balanced_accuracy_score(y, cross_val_predict(
        KNeighborsClassifier(k), Y, y, cv=StratifiedKFold(ns, shuffle=True, random_state=0)))


def plot(Y, imgs, ys, it, n_exemplars=40, seed=0):
    rng = np.random.default_rng(seed)
    Yn = (Y - Y.mean(0)) / (Y.std(0) + 1e-9)
    keep = fps(Yn, n_exemplars, rng)
    fig, ax = plt.subplots(figsize=(13, 10))
    for i in keep:
        im = Image.fromarray(imgs[i]).copy(); im.thumbnail((90, 90))
        col = "#d62728" if ys[i] == 1 else "#1f77b4"     # red=5 stripes, blue=4 (validation only)
        ax.add_artist(AnnotationBbox(OffsetImage(np.asarray(im), zoom=1.0), (Yn[i, 0], Yn[i, 1]),
                      frameon=True, pad=0.04, bboxprops=dict(edgecolor=col, lw=2.0)))
    pad = 0.6
    ax.set_xlim(Yn[keep, 0].min() - pad, Yn[keep, 0].max() + pad)
    ax.set_ylim(Yn[keep, 1].min() - pad, Yn[keep, 1].max() + pad)
    ttl = ("INITIAL unsupervised morphospace (texton-BoW PCA)" if it == 0
           else f"after iteration {it}  (diverse-panel rankings, spring morph)")
    ax.set_title(f"Fishy morphospace — {ttl}\nborder: red=5 stripes, blue=4 (validation overlay)")
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    path = RES / f"morphospace_iter{it}.png"
    fig.savefig(path, dpi=110); plt.close(fig)
    return path


def main():
    fcd = data.build_face_colors(verbose=False)
    gt = data.load_ground_truth()
    names = fcd.names
    imgs = [np.asarray(Image.open(RDIR / f"{n}.png").convert("RGB")) for n in names]

    # descriptor -> initial 2D unsupervised morphospace
    Pop = textons.diffusion_operator()
    jet = textons.local_jet(fcd, scales=(2, 4), P=Pop)
    sck, km = textons.build_codebook(jet, K=128, seed=0)
    X = StandardScaler().fit_transform(textons.encode(fcd, jet, sck, km, mode="bow"))
    Y0 = PCA(n_components=2, random_state=0).fit_transform(X)
    Y0 = (Y0 - Y0.mean(0)) / Y0.std(0)

    Pmetric = perceptual(gt)
    ys_stripe = gt.labels["stripe"]
    rng = np.random.default_rng(0)

    Y = Y0.copy()
    all_trips = []
    paths = [plot(Y, imgs, ys_stripe, 0)]
    hist = {f: [round(knn_balacc(Y, gt.labels[f]), 3)] for f in ("belly", "tail", "stripe", "cheeks")}
    print(f"iter 0 (initial):  " + "  ".join(f"{f}={hist[f][0]:.3f}" for f in hist))

    n_iters, n_anchors, panel = 6, 10, 9
    for it in range(1, n_iters + 1):
        # diverse anchors across the CURRENT space
        anchors = fps(Y, n_anchors, rng)
        for a in anchors:
            # DIVERSE panel (not kNN!): farthest-point sample around/over the space from a
            panel_idx = fps(Y, panel + 1, rng, start=a)[1:]
            remote = panel_idx[-1]
            cand = panel_idx[:-1]
            order = sorted(cand, key=lambda j: np.linalg.norm(Pmetric[j] - Pmetric[a]))
            all_trips += semisup.triplets_from_ranking(a, order, remote=remote)
        # spring morph (warm-start from current Y, tether to the INITIAL layout Y0)
        Y = semisup.spring_embed(Y0.copy(), all_trips, n_iter=500, lr=0.008, margin=0.4, lam=0.04)
        Y = (Y - Y.mean(0)) / (Y.std(0) + 1e-9)
        for f in hist:
            hist[f].append(round(knn_balacc(Y, gt.labels[f]), 3))
        paths.append(plot(Y, imgs, ys_stripe, it))
        print(f"iter {it}: ({len(all_trips)} triplets)  " +
              "  ".join(f"{f}={hist[f][-1]:.3f}" for f in hist))

    # montage + gif
    frames = [np.asarray(Image.open(p).convert("RGB")) for p in paths]
    h = min(f.shape[0] for f in frames); w = min(f.shape[1] for f in frames)
    frames = [f[:h, :w] for f in frames]
    imageio.mimsave(RES / "morphospace_morph.gif", frames, duration=1.2, loop=0)
    cols = len(frames)
    fig, axes = plt.subplots(1, cols, figsize=(4 * cols, 4))
    for ax, f, i in zip(axes, frames, range(cols)):
        ax.imshow(f); ax.set_title(f"iter {i}", fontsize=9); ax.axis("off")
    fig.tight_layout(); fig.savefig(RES / "morphospace_montage.png", dpi=90); plt.close(fig)

    (RES / "spring_morphospace.json").write_text(json.dumps(hist, indent=2))
    print(f"\nsaved frames + gif + montage -> {RES}")
    print("stripe kNN balacc over iters:", hist["stripe"])


if __name__ == "__main__":
    main()
