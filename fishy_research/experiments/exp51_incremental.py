"""EXP-51 — React after EVERY batch, from the first one, while staying smooth.

UX requirement: the classifier + morphospace must update after each tagging batch starting with the first
(no "wait until 80 tags" freeze). Naive re-fit-each-batch UMAP reacts but teleports (EXP-47/48 motion ~1.2).
Fix: re-fit UMAP each batch but WARM-START it with `init=previous_embedding` (deterministic seed) so points
carry over -> reactive immediately AND smooth. Compare to naive re-fit and to AlignedUMAP (reference).

Run:  .venv/bin/python experiments/exp51_incremental.py
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score
from umap import UMAP, AlignedUMAP

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import _tag_harness as H  # noqa: E402

RES = H.RESULTS
# incremental batches: tag a handful at a time, react after EACH (the first is small)
BUDGETS = [8, 16, 24, 32, 48, 64, 96, 128, 160]
UKW = dict(n_neighbors=15, min_dist=0.12, random_state=0)
WARM_EPOCHS = 60        # fewer epochs on the warm restart -> stays near the prior layout (steadier)


def tag_scores(Z, lab, Yl):
    S = np.zeros((len(Z), 6), "float32")
    for t in range(6):
        y = Yl[:, t]
        if len(np.unique(y)) < 2:
            S[:, t] = float(y.mean()); continue
        # light regularisation (C=0.5) — important at TINY budgets so the first batch isn't wild
        S[:, t] = LogisticRegression(C=0.5, max_iter=2000).fit(Z[lab], y).decision_function(Z)
    return S


def _norm(E):
    E = np.asarray(E, float); E = E - E.mean(0)
    return E / (np.sqrt((E ** 2).sum(1).mean()) + 1e-9)


def motion(frames):
    F = [_norm(e) for e in frames]
    return float(np.mean([np.linalg.norm(F[i + 1] - F[i], axis=1).mean() for i in range(len(F) - 1)]))


def main():
    Z, F, joint, _ = H.load()
    N = len(Z); Y = H.tags_for(F)
    order = np.random.default_rng(0).permutation(N)
    labsets = [np.sort(order[:n]) for n in BUDGETS]
    latents = [tag_scores(Z, lab, Y[lab]) for lab in labsets]

    # WARM-STARTED incremental UMAP: re-fit each batch, init = previous embedding
    warm = []
    prev = None
    for S in latents:
        if prev is None:
            e = UMAP(init="spectral", **UKW).fit_transform(S)                 # first batch: cold start
        else:
            init = (prev - prev.mean(0)) / (prev.std() + 1e-9)
            e = UMAP(init=init, n_epochs=WARM_EPOCHS, **UKW).fit_transform(S)  # warm + few epochs
        warm.append(np.asarray(e)); prev = np.asarray(e)
    # naive (no warm start) + AlignedUMAP reference
    naive = [np.asarray(UMAP(**UKW).fit_transform(S)) for S in latents]
    rel = [{i: i for i in range(N)} for _ in range(len(latents) - 1)]
    aligned = [np.asarray(e) for e in AlignedUMAP(**UKW).fit(latents, relations=rel).embeddings_]

    def sils(seq):
        return [round(float(silhouette_score(e, joint)), 3) for e in seq]

    print("batches (tagged):", BUDGETS)
    print(f"  warm-start init=prev : motion={motion(warm):.3f}  sil={sils(warm)}")
    print(f"  naive re-fit         : motion={motion(naive):.3f}  sil={sils(naive)}")
    print(f"  AlignedUMAP (ref)    : motion={motion(aligned):.3f}  sil={sils(aligned)}")
    print(f"\n-> reacts from batch 1: sil after first batch ({BUDGETS[0]} tags) = {sils(warm)[0]}")

    # small-multiples of the warm-started evolution
    colors = [plt.cm.tab10(j) for j in joint]
    AW = np.concatenate(warm); xl = (AW[:, 0].min() - 1, AW[:, 0].max() + 1); yl = (AW[:, 1].min() - 1, AW[:, 1].max() + 1)
    fig, axes = plt.subplots(3, 3, figsize=(11, 11))
    for ax, n, e, lab, s in zip(axes.ravel(), BUDGETS, warm, labsets, sils(warm)):
        ax.scatter(e[:, 0], e[:, 1], s=14, c=colors, edgecolors="none", alpha=.85)
        ax.scatter(e[lab, 0], e[lab, 1], s=40, facecolors="none", edgecolors="k", linewidths=.7)
        ax.set_title(f"{n} tags · sil={s}", fontsize=9); ax.set_xlim(*xl); ax.set_ylim(*yl)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Warm-started UMAP: reacts after every batch (from the 1st) and stays smooth", y=1.0)
    fig.tight_layout(); fig.savefig(RES / "incremental_grid.png", dpi=120); plt.close(fig)

    # GIF
    frames = []
    for k in range(len(warm) - 1):
        for a in np.linspace(0, 1, 6, endpoint=False):
            E = (1 - a) * warm[k] + a * warm[k + 1]; n = int(round((1 - a) * BUDGETS[k] + a * BUDGETS[k + 1]))
            fig, ax = plt.subplots(figsize=(4.6, 4.4))
            ax.scatter(E[:, 0], E[:, 1], s=16, c=colors, edgecolors="none", alpha=.9)
            ax.set_title(f"{n} tags", fontsize=11); ax.set_xlim(*xl); ax.set_ylim(*yl); ax.set_xticks([]); ax.set_yticks([])
            fig.tight_layout(); buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=88); buf.seek(0)
            frames.append(Image.open(buf).convert("RGB")); plt.close(fig)
    frames += [frames[-1]] * 8
    frames[0].save(RES / "incremental.gif", save_all=True, append_images=frames[1:], duration=95, loop=0)
    print(f"\nfigures + GIF -> {RES}")


if __name__ == "__main__":
    main()
