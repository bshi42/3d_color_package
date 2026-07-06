"""EXP-48 — A NON-stochastic, smoothly-evolving UMAP of the tag-classifier latent.

EXP-47: UMAP displays the tag-score latent ~2x better than PCA, but plain UMAP is stochastic — re-running
it as the user tags more specimens makes the whole map teleport. Fix: tag MONOTONICALLY (only add tags) and
embed the SEQUENCE of latents with `umap.AlignedUMAP`, which aligns successive embeddings so the map evolves
smoothly (same specimen stays put; structure just sharpens). Compare smoothness to naive independent UMAP
(re-fit each budget) and independent-UMAP + post-hoc Procrustes alignment. Outputs small-multiples + a GIF.

Run:  .venv/bin/python experiments/exp48_aligned_umap.py
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.linalg import orthogonal_procrustes
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score
from umap import UMAP, AlignedUMAP

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import _tag_harness as H  # noqa: E402

RESULTS = H.RESULTS
BUDGETS = [5, 10, 20, 40, 60, 80, 120, 160]


def tag_scores(Z, lab, Yl):
    S = np.zeros((len(Z), 6))
    for t in range(6):
        y = Yl[:, t]
        if len(np.unique(y)) < 2:
            S[:, t] = float(y.mean()); continue
        S[:, t] = LogisticRegression(max_iter=2000).fit(Z[lab], y).decision_function(Z)
    return S


def _norm(E):                       # zero-mean, unit-RMS (scale/shift invariant frame for comparison)
    E = E - E.mean(0)
    return E / (np.sqrt((E ** 2).sum(1).mean()) + 1e-9)


def motion(frames):                 # mean per-point frame-to-frame displacement (scale-normalized)
    F = [_norm(e) for e in frames]
    return float(np.mean([np.linalg.norm(F[i + 1] - F[i], axis=1).mean() for i in range(len(F) - 1)]))


def procrustes_chain(frames):       # align each frame to the previous (best rigid fit) — a simpler fix
    out = [_norm(frames[0])]
    for e in frames[1:]:
        e = _norm(e); R, _ = orthogonal_procrustes(e, out[-1]); out.append(e @ R)
    return out


def main():
    Z, F, joint, _ = H.load()
    N = len(Z); Y = H.tags_for(F)
    order = np.random.default_rng(0).permutation(N)          # fixed tagging order -> monotone budgets
    labsets = [np.sort(order[:n]) for n in BUDGETS]
    latents = [tag_scores(Z, lab, Y[lab]) for lab in labsets]

    # 1) AlignedUMAP over the monotone sequence (identity relations: same 250 specimens each slice)
    rel = [{i: i for i in range(N)} for _ in range(len(latents) - 1)]
    au = AlignedUMAP(n_neighbors=20, min_dist=0.12, alignment_regularisation=0.02,
                     alignment_window_size=2, random_state=0).fit(latents, relations=rel)
    aligned = [np.asarray(e) for e in au.embeddings_]

    # 2) naive independent UMAP per budget (what plain re-running gives), + Procrustes-chained version
    indep = [np.asarray(UMAP(n_neighbors=20, min_dist=0.12, random_state=0).fit_transform(S)) for S in latents]
    proc = procrustes_chain(indep)

    print("smoothness (mean per-point frame-to-frame motion, lower = steadier):")
    print(f"  naive independent UMAP : {motion(indep):.3f}")
    print(f"  independent + Procrustes: {motion(proc):.3f}")
    print(f"  AlignedUMAP            : {motion(aligned):.3f}")
    print("\n8-way silhouette of the AlignedUMAP frame, per tag budget:")
    sils = []
    for n, e in zip(BUDGETS, aligned):
        s = round(float(silhouette_score(e, joint)), 3); sils.append(s)
        print(f"  {n:4d} tags: sil={s}")

    colors = [plt.cm.tab10(j) for j in joint]
    AL = np.concatenate(aligned)
    xlim = (AL[:, 0].min() - 1, AL[:, 0].max() + 1); ylim = (AL[:, 1].min() - 1, AL[:, 1].max() + 1)

    # small-multiples
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.6))
    for ax, n, e, s, lab in zip(axes.ravel(), BUDGETS, aligned, sils, labsets):
        ax.scatter(e[:, 0], e[:, 1], s=16, c=colors, edgecolors="none", alpha=.85)
        ax.scatter(e[lab, 0], e[lab, 1], s=42, facecolors="none", edgecolors="k", linewidths=.8)
        ax.set_title(f"{n} tags · sil={s}", fontsize=10); ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("AlignedUMAP of the tag-classifier latent evolving as tags are added "
                 "(colour = true 8-way; ringed = tagged)", y=1.0)
    fig.tight_layout(); fig.savefig(RESULTS / "aligned_umap_grid.png", dpi=120); plt.close(fig)

    # GIF (interpolate a couple of in-between frames for smoothness)
    frames = []
    seq = aligned
    for k in range(len(seq) - 1):
        for a in np.linspace(0, 1, 6, endpoint=False):
            E = (1 - a) * seq[k] + a * seq[k + 1]
            n = int(round((1 - a) * BUDGETS[k] + a * BUDGETS[k + 1]))
            fig, ax = plt.subplots(figsize=(4.6, 4.4))
            ax.scatter(E[:, 0], E[:, 1], s=18, c=colors, edgecolors="none", alpha=.9)
            ax.set_title(f"~{n} tags", fontsize=11); ax.set_xlim(*xlim); ax.set_ylim(*ylim)
            ax.set_xticks([]); ax.set_yticks([])
            fig.tight_layout()
            buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=90); buf.seek(0)
            frames.append(Image.open(buf).convert("RGB")); plt.close(fig)
    for _ in range(10):
        frames.append(frames[-1])
    frames[0].save(RESULTS / "aligned_umap.gif", save_all=True, append_images=frames[1:],
                   duration=90, loop=0)
    print(f"\nfigures + GIF -> {RESULTS}")


if __name__ == "__main__":
    main()
