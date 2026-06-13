"""EXP-00 — Sanity: dataset integrity & ground-truth independence.

Confirms the 250 textures load, builds the per-face color cache, and verifies the planted
structure (belly/tail hue independence guards against the historical 1:1-correlation bug).
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fishpipe import config, data

RESULTS = config.RESULTS_DIR / "exp00"
RESULTS.mkdir(parents=True, exist_ok=True)


def main():
    print("== EXP-00 sanity ==")
    gt = data.load_ground_truth()
    df = gt.params
    print(f"ground-truth rows: {len(df)}")

    # factor distributions
    for f in config.CLUSTER_FACTORS:
        vals, counts = np.unique(gt.labels[f], return_counts=True)
        print(f"  {f:8s}: {dict(zip(vals.tolist(), counts.tolist()))}")

    # belly/tail hue independence
    bh, th = df["belly_hue"].to_numpy(), df["tail_hue"].to_numpy()
    r = np.corrcoef(bh, th)[0, 1]
    print(f"\nbelly_hue vs tail_hue Pearson r = {r:+.3f} (want ~0; ~±1 => historical bug)")
    # cross-tab of belly/tail cluster labels — should be ~uniform 4 cells if independent
    ct = np.zeros((2, 2), int)
    for b, t in zip(gt.labels["belly"], gt.labels["tail"]):
        ct[b, t] += 1
    print("belly×tail label cross-tab (rows=belly,cols=tail):\n", ct)

    # build / load per-face color cache
    print("\nbuilding per-face color cache (samples all 250 textures)...")
    fcd = data.build_face_colors(force=False)
    print(f"  face colors: {fcd.colors.shape} dtype={fcd.colors.dtype}")
    print(f"  face areas : {fcd.areas.shape}  total surface area={fcd.areas.sum():.4f}")
    print(f"  mean per-specimen mean RGB: {fcd.colors.reshape(len(fcd.names),-1,3).mean(axis=(0,1))}")

    # figure: belly/tail hue histograms colored by recovered cluster + scatter
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, name, hue in [(axs[0], "belly", bh), (axs[1], "tail", th)]:
        for cls in (0, 1):
            ax.hist(hue[gt.labels[name] == cls], bins=30, alpha=0.7, label=f"cluster {cls}")
        ax.set_title(f"{name}_hue (2 planted blobs)")
        ax.set_xlabel("hue"); ax.legend(fontsize=8)
    axs[2].scatter(bh, th, c=gt.labels["belly"] * 2 + gt.labels["tail"], cmap="tab10", s=14)
    axs[2].set_title(f"belly vs tail hue (r={r:+.2f})")
    axs[2].set_xlabel("belly_hue"); axs[2].set_ylabel("tail_hue")
    fig.tight_layout()
    out = RESULTS / "ground_truth_structure.png"
    fig.savefig(out, dpi=130); plt.close(fig)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
