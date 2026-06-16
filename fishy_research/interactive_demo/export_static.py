"""Bake a fully static (server-free) demo of the MUSSEL dataset.

Dumps everything the browser needs into `static_demo/`:
  * bundle.json  — session meta, the kNN+MST similarity graph (PCA-seed positions, rest lengths),
                   per-PC values, the raw z-scored PCA coords Z0 (for client-side panel sampling),
                   the New-anchor sweep seed, and the leaf aspect ratio.
  * thumb/mussel/{name}.png      — 150px colour thumbnails (canvas exemplars)
  * thumb_hi/mussel/{name}.png   — 300px hi-res renders (crisp dendrogram leaves)

The force physics, feedback springs, dendrogram (Ward, client-ported), selection, hide toggles and
SVG/PNG export all run in the browser, so the result needs only a dumb static file host.

Run:  .venv/bin/python interactive_demo/export_static.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent          # interactive_demo/
sys.path.insert(0, str(HERE))
import engine                                    # noqa: E402  (sets up fishpipe import path)
from fishpipe import render                      # noqa: E402

DATASET = "mussel"
OUT = HERE / "static_demo"
THUMB_PX = 150
HI_PX = 300


def _render(ld, face_rgb, px):
    if ld.spec.renderer == "valve":
        return render.render_textured_rgba(ld.mesh, face_rgb, px=px, supersample=2, crop=True, align=True)
    return render.render_side_view(ld.mesh, face_rgb, px=px, bounds=ld.bounds)


def main():
    sess = engine.Session(DATASET)
    ld = sess.ld
    tdir = OUT / "thumb" / DATASET
    hdir = OUT / "thumb_hi" / DATASET
    tdir.mkdir(parents=True, exist_ok=True)
    hdir.mkdir(parents=True, exist_ok=True)

    leaf_aspect = None
    for i, name in enumerate(ld.names):
        Image.fromarray(_render(ld, ld.fcd.colors[i], THUMB_PX)).save(tdir / f"{name}.png")
        hi = _render(ld, ld.fcd.colors[i], HI_PX)
        Image.fromarray(hi).save(hdir / f"{name}.png")
        if leaf_aspect is None:
            leaf_aspect = round(float(hi.shape[1]) / float(hi.shape[0]), 3)
        print(f"  rendered {i + 1}/{len(ld.names)}  {name}", flush=True)

    bundle = {
        "dataset": DATASET,
        "title": ld.spec.title,
        "K": sess.K,
        "active_pc": sess.active_pc,
        "has_gt": ld.spec.has_gt,
        "names": list(ld.names),
        "evr": [round(float(v), 5) for v in sess.evr],
        "graph": sess.graph(),                                       # nodes / edges / pc_values
        "Z0": [[round(float(v), 5) for v in row] for row in sess.Z0],  # for client panel sampling
        "leaf_aspect": leaf_aspect,
        "anchor0": int(sess.pick_anchor()),                          # first "New anchor"
    }
    (OUT / "bundle.json").write_text(json.dumps(bundle))
    print(f"\nwrote {OUT / 'bundle.json'}  ·  {len(ld.names)} thumbnails  ·  K={sess.K}  anchor0={bundle['anchor0']}")


if __name__ == "__main__":
    main()
