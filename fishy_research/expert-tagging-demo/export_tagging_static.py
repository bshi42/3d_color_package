"""Bake a fully static (server-free) build of the TAGGING demo.

Writes ./static_demo/: per-dataset bundle_<ds>.json (the demo's PCA-24 coords Z, the cold-start novelty score,
names, leaf aspect) + thumb/<ds>/*.png (150px) + thumb_hi/<ds>/*.png (300px). At run time the browser does
everything (per-tag logistic + umap-js projection + JS Ward tree + active/novelty roll) — see the static
frontend. Nothing here is needed at run time; this is the one-time Python bake.

Run (from this dir, with the repo venv):  ../.venv/bin/python export_tagging_static.py
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import engine   # noqa: E402
import server   # noqa: E402  (reuse its thumbnail renderer/cache)

DST = HERE / "static_demo"          # the gitignored static build, alongside this script


def main():
    DST.mkdir(parents=True, exist_ok=True)
    for ds in ("mussel", "fishy"):
        s = engine.TagSession(ds)
        names = list(s.ld.names)
        td = DST / "thumb" / ds; hd = DST / "thumb_hi" / ds
        td.mkdir(parents=True, exist_ok=True); hd.mkdir(parents=True, exist_ok=True)
        asp = 2.0
        for i, nm in enumerate(names):
            (td / f"{nm}.png").write_bytes(server._thumb_png(ds, nm, hi=False))
            hb = server._thumb_png(ds, nm, hi=True)
            (hd / f"{nm}.png").write_bytes(hb)
            if i == 0:
                a = np.asarray(Image.open(io.BytesIO(hb))); asp = round(float(a.shape[1]) / float(a.shape[0]), 3)
            if (i + 1) % 50 == 0:
                print(f"  {ds}: rendered {i + 1}/{len(names)} thumbs", flush=True)
        bundle = {"dataset": ds, "title": s.ld.spec.title, "has_gt": False, "names": names,
                  "Z": [[round(float(v), 4) for v in r] for r in s.Z],
                  "novelty": [round(float(v), 4) for v in s.novelty], "leaf_aspect": asp}
        (DST / f"bundle_{ds}.json").write_text(json.dumps(bundle))
        print(f"{ds}: baked Z {len(s.Z)}x{len(s.Z[0])} · {len(names)} thumbs · leaf_aspect {asp}", flush=True)
    print(f"\nwrote -> {DST}")


if __name__ == "__main__":
    main()
