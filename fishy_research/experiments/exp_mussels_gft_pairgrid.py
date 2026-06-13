"""Large-format mussel pairgrid with shell-exterior exemplars, on the best GFT descriptor.

Descriptor = signed GFT coefficients (k=40, L/a/b) -> PCA(5) -> 5x5 pairgrid. Every off-diagonal
panel places the 31 shell exteriors at their (PC_j, PC_i) coords. Thumbnails are tight-cropped
with TRANSPARENT background (no black boxes), lightly DECLUTTERED to reduce overlap (a faint dot
marks each shell's true position), tinted dot = ring-contrast. Real data, no labels.
"""
from __future__ import annotations

import argparse
import math
import numpy as np
import warnings

warnings.filterwarnings("ignore")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from fishpipe import config, dataset, features, spectral

DS = dataset.MUSSELS
EXT = config.RESULTS_DIR / "mussels" / "exterior"


def transparent_thumb(path, thumb_h, thr=14):
    a = np.asarray(Image.open(path).convert("RGB"))
    mask = a.max(2) >= thr
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    a = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    m = (a.max(2) >= thr).astype(np.uint8) * 255
    im = Image.fromarray(np.dstack([a, m]), "RGBA")
    im.thumbnail((thumb_h, thumb_h * 2))      # portrait valve
    return np.asarray(im)


def declutter(P, rad, iters=200):
    """Repulsion: push pairs closer than rad apart (normalized [0,1] coords)."""
    Q = P.astype(float).copy()
    for _ in range(iters):
        moved = False
        for a in range(len(Q)):
            d = Q - Q[a]
            dist = np.hypot(d[:, 0], d[:, 1])
            for b in np.where((dist < rad) & (dist > 1e-9))[0]:
                if b <= a:
                    continue
                push = (rad - dist[b]) / 2.0
                u = d[b] / dist[b]
                Q[a] -= u * push; Q[b] += u * push; moved = True
        Q = np.clip(Q, 0.02, 0.98)
        if not moved:
            break
    return Q


def grid_snap(P, slack=1.10):
    """Assign each point to the nearest FREE grid cell (optimal assignment) -> zero overlap.
    Returns decluttered coords and the grid dim G."""
    n = len(P)
    G = int(math.ceil(math.sqrt(n) * slack))
    g = (np.arange(G) + 0.5) / G
    cx, cy = np.meshgrid(g, g)
    cells = np.column_stack([cx.ravel(), cy.ravel()])
    cost = np.linalg.norm(P[:, None] - cells[None], axis=2)
    r, c = linear_sum_assignment(cost)
    out = np.zeros((n, 2)); out[r] = cells[c]
    return out, G


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout", choices=["declutter", "strong", "grid"], default="grid")
    ap.add_argument("--npc", type=int, default=5)
    ap.add_argument("--panel", type=float, default=12.0)
    ap.add_argument("--thumb", type=int, default=70)
    ap.add_argument("--dpi", type=int, default=110)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    NPC, PANEL_IN, DPI = args.npc, args.panel, args.dpi
    fcd, _ = dataset.build_face_colors(DS)
    names = fcd.names
    mesh = DS.load_mesh()
    X = spectral.spectral_coeffs(fcd, k=40, channels=("L", "a", "b"), n_basis=200,
                                 mesh=mesh, cache_dir=DS.cache_dir)
    pca = PCA(NPC, random_state=0).fit(StandardScaler().fit_transform(X))
    E = pca.transform(StandardScaler().fit_transform(X))
    ev = pca.explained_variance_ratio_ * 100
    contrast = features.rgb_to_lab(fcd.colors)[..., 0].std(1)
    norm = Normalize(contrast.min(), contrast.max()); cmap = cm.get_cmap("plasma")

    n = len(names)
    G = int(math.ceil(math.sqrt(n) * 1.10))
    NATIVE = 220                                    # native thumbnail height (px); zoom set after measuring
    thumbs = [transparent_thumb(EXT / (nm + ".png"), NATIVE) for nm in names]

    lim = [(E[:, k].min(), E[:, k].max()) for k in range(NPC)]
    span = [hi - lo for lo, hi in lim]
    m = 0.06 if args.layout == "grid" else 0.09

    fig, axs = plt.subplots(NPC, NPC, figsize=(PANEL_IN * NPC, PANEL_IN * NPC), dpi=DPI)
    placements = {}
    for i in range(NPC):
        for j in range(NPC):
            ax = axs[i, j]
            if i == j:
                ax.text(0.5, 0.5, f"PC{i+1}\n{ev[i]:.0f}% var", ha="center", va="center",
                        fontsize=46, fontweight="bold"); ax.set_xticks([]); ax.set_yticks([])
                continue
            xn = (E[:, j] - lim[j][0]) / (span[j] + 1e-9)
            yn = (E[:, i] - lim[i][0]) / (span[i] + 1e-9)
            P = np.column_stack([xn, yn])
            Q = grid_snap(P)[0] if args.layout == "grid" else declutter(P, (NATIVE * 0.5) / (PANEL_IN * DPI))
            placements[(i, j)] = (lim[j][0] + Q[:, 0] * span[j], lim[i][0] + Q[:, 1] * span[i])
            ax.set_xlim(lim[j][0] - span[j] * m, lim[j][1] + span[j] * m)
            ax.set_ylim(lim[i][0] - span[i] * m, lim[i][1] + span[i] * m)
            ax.set_xlabel(f"PC{j+1}", fontsize=16); ax.set_ylabel(f"PC{i+1}", fontsize=16)
            ax.tick_params(labelsize=8)

    # MEASURE actual rendered panel size, then size thumbnails to the true grid cell
    fig.canvas.draw()
    ah_px = axs[0, 1].get_window_extent().height
    cell_px = ah_px / G
    zoom = (cell_px * 0.72) / NATIVE if args.layout == "grid" else (args.thumb / NATIVE)
    print(f"measured axes height={ah_px:.0f}px, cell={cell_px:.0f}px, thumb zoom={zoom:.3f} -> ~{zoom*NATIVE:.0f}px")

    for (i, j), (xd, yd) in placements.items():
        ax = axs[i, j]
        for k in range(n):
            ax.plot([E[k, j], xd[k]], [E[k, i], yd[k]], "-", color="0.82", lw=0.4, zorder=1)
            ax.scatter(E[k, j], E[k, i], s=20, color=cmap(norm(contrast[k])), zorder=2,
                       edgecolors="k", linewidths=0.3)
            if thumbs[k] is not None:
                ax.add_artist(AnnotationBbox(OffsetImage(thumbs[k], zoom=zoom, dpi_cor=False),
                              (xd[k], yd[k]), frameon=False, zorder=3))

    sm = cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    cbar = fig.colorbar(sm, ax=axs, fraction=0.010, pad=0.008)
    cbar.set_label("ring-contrast (std L*) — dot color", fontsize=20)
    fig.suptitle(f"Mussels — GFT pattern morphospace (PC1-{NPC} pairgrid), {args.layout} layout; "
                 "transparent exemplars, dot = true position. Real data, no labels.", fontsize=28)
    suffix = args.out or args.layout
    out = config.RESULTS_DIR / "mussels" / f"mussels_gft_pairgrid_{suffix}.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight"); plt.close(fig)
    print(f"saved {out}  EV%={np.round(ev,1)}  ~{int(PANEL_IN*NPC*DPI)}px  layout={args.layout} cell={cell_px:.0f}px")


if __name__ == "__main__":
    main()
