#!/usr/bin/env python3
"""Batch-downsample textured OBJ meshes via headless Blender.

For every ``<name>.obj`` in MODELS_DIR with a matching ``<name>.<ext>`` texture
in TEXTURES_DIR, produces a downsampled mesh (+ texture) under OUTPUT_DIR,
mirroring the input layout::

    OUTPUT_DIR/
        Models/<name>.obj      downsampled mesh (triangulated, with UVs)
        Texture/<name>.png     texture for the downsampled mesh
        logs/<name>.log        full Blender output for that model

Two modes (MODE constant below):

* ``decimate``     Quadric edge-collapse (Blender Decimate modifier). UVs are
                   interpolated so the ORIGINAL texture keeps working and is
                   copied bit-exact. Fast; vertex density stays adaptive
                   (denser in high-curvature areas).
* ``remesh_bake``  QuadriFlow remesh to a near-uniform vertex distribution,
                   Smart-UV-project new UVs, then Cycles-bake the original
                   texture onto the new mesh (new PNG at BAKE_RESOLUTION**2).
                   Slower; use when uniform vertex spacing matters more than
                   keeping the original texture file untouched.

Requires only Blender (5.x; tested with 5.1.2) on PATH — no Python deps.
Run:  python3 downsample_meshes.py
Then check results with verify_downsample.py (see README).
"""

import json
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ============================== CONSTANTS ====================================

MODELS_DIR = Path("/mnt/data/ml_data/color-modeling-pub/3D_Fish (Copy)/3D_Fish/Models")
TEXTURES_DIR = Path("/mnt/data/ml_data/color-modeling-pub/3D_Fish (Copy)/3D_Fish/Texture")

# Target size of the downsampled mesh, in vertices. Face count follows
# (~2x vertices for a closed triangulated surface).
TARGET_VERTICES = 25_000

# "decimate" (keep original texture, adaptive density)
# or "remesh_bake" (uniform vertex distribution, texture baked to a new PNG).
MODE = "decimate"

OUTPUT_DIR = MODELS_DIR.parent / f"Downsampled_{MODE}"

# remesh_bake only: side of the square baked texture, in pixels.
BAKE_RESOLUTION = 2048

# How many Blender jobs to run at once. Each job is itself multi-threaded
# (bakes use all cores), so keep this small.
PARALLEL_JOBS = 2

# Skip models whose output .obj already exists (set True to redo everything).
OVERWRITE = False

# Only process models whose stem is in this set; empty set = process all.
ONLY_MODELS: set[str] = set()

BLENDER = "blender"  # executable; absolute path also fine
TEXTURE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
BLENDER_TIMEOUT_S = 3600  # per model

# =============================================================================

WORKER = Path(__file__).parent / "blender_worker.py"


def find_texture(stem: str) -> Path | None:
    """Locate <stem>.<ext> in TEXTURES_DIR, matching the extension case-insensitively."""
    for cand in sorted(TEXTURES_DIR.glob("*")):
        if cand.is_file() and cand.stem == stem and cand.suffix.lower() in TEXTURE_EXTS:
            return cand
    return None


def process_one(obj_path: Path, tex_path: Path) -> dict:
    stem = obj_path.stem
    out_obj = OUTPUT_DIR / "Models" / f"{stem}.obj"
    # remesh_bake always writes a fresh PNG; decimate copies the source bit-exact.
    tex_ext = ".png" if MODE == "remesh_bake" else tex_path.suffix.lower()
    out_tex = OUTPUT_DIR / "Texture" / f"{stem}{tex_ext}"
    log_path = OUTPUT_DIR / "logs" / f"{stem}.log"

    # Skip only when the whole output pair is present (an interrupted run may
    # have left an OBJ with no texture).
    if out_obj.exists() and out_tex.exists() and not OVERWRITE:
        return {"model": stem, "status": "skipped (exists)"}

    job = {
        "mode": MODE,
        "input_obj": str(obj_path),
        "input_texture": str(tex_path),
        "output_obj": str(out_obj),
        "output_texture": str(out_tex),
        "target_vertices": TARGET_VERTICES,
        "bake_resolution": BAKE_RESOLUTION,
    }
    cmd = [BLENDER, "-b", "--factory-startup", "--python", str(WORKER), "--", json.dumps(job)]

    t0 = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=BLENDER_TIMEOUT_S)
    except subprocess.TimeoutExpired as exc:
        # Persist whatever Blender emitted before the kill (esp. useful for hangs).
        log_path.write_text((exc.stdout or "") + "\n--- STDERR ---\n" + (exc.stderr or "")
                            + f"\n--- KILLED: TIMEOUT after {BLENDER_TIMEOUT_S}s ---\n")
        return {"model": stem, "status": f"TIMEOUT after {BLENDER_TIMEOUT_S}s — see {log_path}"}
    log_path.write_text(proc.stdout + "\n--- STDERR ---\n" + proc.stderr)

    # The worker prints a single machine-readable result line. A truncated or
    # interleaved line must not crash the batch — treat it as "no result".
    result = None
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT_JSON:"):
            try:
                result = json.loads(line[len("RESULT_JSON:"):])
            except json.JSONDecodeError:
                result = None
    if proc.returncode != 0 or result is None or not result.get("ok"):
        detail = (result or {}).get("error", f"rc={proc.returncode}")
        return {"model": stem, "status": f"FAILED ({detail}) — see {log_path}"}

    if MODE == "decimate":
        # Decimation keeps UVs valid for the original texture: copy it as-is.
        try:
            shutil.copy2(tex_path, out_tex)
        except OSError as e:
            return {"model": stem, "status": f"FAILED (texture copy: {e}) — see {log_path}"}

    return {
        "model": stem,
        "status": "ok",
        "seconds": round(time.time() - t0, 1),
        "verts": f"{result.get('orig_vertices')} -> {result.get('vertices')}",
        "faces": f"{result.get('orig_faces')} -> {result.get('faces')}",
    }


def main() -> int:
    objs = sorted(p for p in MODELS_DIR.glob("*.obj"))
    if ONLY_MODELS:
        objs = [p for p in objs if p.stem in ONLY_MODELS]
    if not objs:
        print(f"No .obj files found in {MODELS_DIR}", file=sys.stderr)
        return 1

    pairs, missing = [], []
    for obj in objs:
        tex = find_texture(obj.stem)
        (pairs if tex else missing).append((obj, tex))
    for obj, _ in missing:
        print(f"WARNING: no texture found for {obj.name} — skipping", file=sys.stderr)

    for sub in ("Models", "Texture", "logs"):
        (OUTPUT_DIR / sub).mkdir(parents=True, exist_ok=True)

    print(f"Downsampling {len(pairs)} model(s) to ~{TARGET_VERTICES} vertices "
          f"(mode={MODE}, {PARALLEL_JOBS} parallel jobs)\n")

    results = []
    with ThreadPoolExecutor(max_workers=PARALLEL_JOBS) as pool:
        futures = {pool.submit(process_one, obj, tex): obj for obj, tex in pairs}
        for fut in as_completed(futures):
            try:
                r = fut.result()
            except Exception as e:  # one bad model must not abort the batch
                r = {"model": futures[fut].stem, "status": f"FAILED (unexpected: {e})"}
            results.append(r)
            extra = f"  {r.get('seconds', '')}s  verts {r.get('verts', '')}" if r["status"] == "ok" else ""
            print(f"[{len(results)}/{len(pairs)}] {r['model']}: {r['status']}{extra}")

    ok = [r for r in results if r["status"] == "ok"]
    skipped = [r for r in results if r["status"].startswith("skipped")]
    failed = [r for r in results if r["status"].startswith(("FAILED", "TIMEOUT"))]
    print(f"\nDone: {len(ok)} processed, {len(skipped)} skipped, {len(failed)} failed. "
          f"Output: {OUTPUT_DIR}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
