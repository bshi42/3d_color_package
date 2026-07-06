# Textured mesh downsampling

Batch-downsample large textured OBJ fish scans to a configurable vertex budget
while keeping them **volumetrically faithful**, **watertight-equivalent in
topology** (still a proper triangle mesh), and **textured**, so they run through
the color-analysis pipeline much faster.

On the 7 sample fish (129k–597k vertices), downsampling to **~25k vertices**
runs in **4–16 s per model** and reproduces the original surface to within
**~0.02 % of the bounding-box diagonal** (p95) and the original color to within
a few RGB levels — see [Results](#results) and `examples/`.

## Files

| File | Role |
|------|------|
| `downsample_meshes.py` | **Entry point.** Edit the CONSTANTS block, run it. |
| `blender_worker.py` | Runs inside headless Blender (invoked per model by the entry point — not run directly). |
| `verify_downsample.py` | Standalone QA: scores a downsampled mesh vs its original (geometry + color). |
| `examples/` | Rendered comparisons + `verification_summary.json` for the 7 sample fish. |

## Requirements

- **Blender 5.x** on `PATH` (tested on 5.1.2). The worker needs no Blender add-ons
  for `decimate`; `remesh_bake` uses the bundled Cycles add-on (auto-enabled).
- For `verify_downsample.py` only: Python with `numpy`, `scipy`, `Pillow`,
  `trimesh`, `rtree` (`pip install trimesh rtree`). The downsampler itself has
  **no Python dependencies** beyond the standard library — all mesh work happens
  inside Blender.

## Usage

Edit the CONSTANTS at the top of `downsample_meshes.py`:

```python
MODELS_DIR      = ".../3D_Fish/Models"      # dir of <name>.obj
TEXTURES_DIR    = ".../3D_Fish/Texture"     # dir of <name>.png  (paired by stem)
TARGET_VERTICES = 25_000                     # desired output vertex count
MODE            = "decimate"                 # "decimate" | "remesh_bake"
BAKE_RESOLUTION = 2048                        # remesh_bake only: baked PNG side (px)
PARALLEL_JOBS   = 2                           # concurrent Blender processes
```

Then:

```bash
python3 downsample_meshes.py
```

Output mirrors the input layout under `MODELS_DIR/../Downsampled_<mode>/`:

```
Downsampled_decimate/
    Models/<name>.obj      downsampled mesh (triangulated, UVs preserved)
    Texture/<name>.png     texture for the mesh
    logs/<name>.log        full Blender output for that model
```

Models and textures are paired **by filename stem** (`Mchenga_m1.obj` ↔
`Mchenga_m1.png`). A model with no matching texture is skipped with a warning.
Re-running skips models whose output already exists unless `OVERWRITE = True`.

### Verifying results

```bash
python3 verify_downsample.py \
    ORIG/Models/Mchenga_m1.obj  ORIG/Texture/Mchenga_m1.png \
    Downsampled_decimate/Models/Mchenga_m1.obj \
    Downsampled_decimate/Texture/Mchenga_m1.png \
    --samples 40000 --json report.json
```

Prints per-category **PASS/WARN** verdicts (volume, bbox, surface distance,
color) and, with `--json`, the full metric tree.

## The two modes

Both hit the target vertex count and keep the mesh textured; they differ in
**vertex layout** and **what happens to the texture**.

### `decimate` (default — recommended)

Quadric edge-collapse (Blender's Decimate modifier). UVs are interpolated
through the collapse, so the **original texture PNG is reused bit-for-bit** (just
copied to the output folder). Vertex density stays **adaptive**: dense where the
surface bends (fins, head), sparse on flat flanks — this is why it reproduces
the original shape so tightly at a given budget.

- ✅ Highest geometric fidelity per vertex; texture is untouched (no resampling loss).
- ✅ Fast (a few seconds), no Cycles needed.
- ➖ Vertex spacing is non-uniform (follows curvature).

### `remesh_bake` (use when uniform spacing matters)

Rebuilds the surface with near-**uniform** vertex spacing (QuadriFlow when the
mesh is manifold; a voxel remesh fallback for these non-manifold scans), makes
fresh UVs via Smart UV Project, then **bakes** the original texture onto the new
UVs as a new PNG (`BAKE_RESOLUTION²`, Cycles DIFFUSE/COLOR — albedo only, no
lighting).

- ✅ Even vertex distribution across the surface.
- ➖ Voxel remeshing rounds off thin membranes (fish fins) → larger worst-case
  surface deviation (~2–3 % of bbox diagonal vs ~0.05 % for `decimate`).
- ➖ Texture is resampled through the bake (a few extra RGB levels of error) and
  the new UV atlas leaves unused margin in the PNG.

**Rule of thumb:** if the pipeline samples geometry/color at points on the
surface, prefer `decimate` — it is strictly closer to the original. Choose
`remesh_bake` only if a downstream step specifically needs uniform vertex spacing
(e.g. spectral/graph methods sensitive to sampling density).

See the wireframe rows in `examples/compare_*.png` for the density difference —
the textured rows are visually indistinguishable across both modes.

## Results

Downsampling the 7 sample fish to ~25k vertices (verifier, 40k surface samples).
`surf p95/max` = symmetric surface distance as % of the original bbox diagonal;
`RGB Δ` = euclidean color delta on a 0–255 scale.

**`decimate`** (volume within 0.03 %, surface p95 ≤ 0.018 %):

| model | v: orig → ds | vol ratio | surf p95 % | surf max % | RGB Δ mean | RGB Δ p95 |
|-------|-------------:|----------:|-----------:|-----------:|-----------:|----------:|
| Mchenga_m1    | 128,814 → 24,917 | 0.9999 | 0.014 | 0.042 | 7.75 | 24.90 |
| Nimbo_f1      | 597,151 → 24,990 | 0.9997 | 0.018 | 0.090 | 5.51 | 19.66 |
| Nimbo_f2      | 366,165 → 25,000 | 0.9998 | 0.016 | 0.073 | 4.65 | 15.98 |
| yellowhead_m1 | 391,586 → 24,994 | 0.9997 | 0.018 | 0.065 | 4.78 | 16.90 |
| yellowhead_m2 | 392,392 → 25,000 | 0.9997 | 0.018 | 0.078 | 4.94 | 17.12 |
| yellowhead_m4 | 288,371 → 25,002 | 0.9998 | 0.016 | 0.056 | 3.22 | 11.38 |
| yellowhead_m5 | 383,412 → 24,996 | 0.9997 | 0.017 | 0.067 | 5.66 | 19.81 |

**`remesh_bake`** (uniform spacing; larger worst-case deviation on thin fins):

| model | v: orig → ds | vol ratio | surf p95 % | surf max % | RGB Δ mean | RGB Δ p95 |
|-------|-------------:|----------:|-----------:|-----------:|-----------:|----------:|
| Mchenga_m1    | 128,814 → 24,541 | 0.9940 | 0.187 | 2.043 | 7.28 | 26.35 |
| Nimbo_f1      | 597,151 → 24,742 | 0.9999 | 0.237 | 1.794 | 8.56 | 34.76 |
| Nimbo_f2      | 366,165 → 24,819 | 0.9983 | 0.223 | 3.724 | 7.07 | 22.28 |
| yellowhead_m1 | 391,586 → 24,725 | 0.9977 | 0.196 | 2.146 | 6.17 | 20.43 |
| yellowhead_m2 | 392,392 → 24,657 | 1.0007 | 0.180 | 1.981 | 5.89 | 16.59 |
| yellowhead_m4 | 288,371 → 24,814 | 0.9954 | 0.217 | 2.283 | 5.10 | 17.58 |
| yellowhead_m5 | 383,412 → 24,753 | 0.9953 | 0.208 | 3.337 | 7.32 | 25.04 |

(The color `WARN` at RGB p95 ≈ 20–35 is expected: it measures how much the
nearest-surface color shifts under a 5–24× vertex reduction, not texture
corruption — mean delta stays ~5–8 of 255. `decimate` keeps the texture exact.)

## Notes / gotchas baked into the code

- Input paths with spaces/parentheses are passed as argv (never shell-interpolated).
- `/mnt/data` inputs are treated as read-only; outputs go to a sibling folder.
- OBJ axis convention round-trips (`-Z` forward / `Y` up), so downsampled meshes
  stay in the original coordinate frame the pipeline expects.
- Output OBJs are pure triangles, UVs preserved, and carry **no `mtllib`** — the
  texture is paired by filename, matching the input convention.
- QuadriFlow silently `CANCELLED`s on non-manifold scans (leaves geometry
  unchanged); the worker detects this and falls back to a voxel remesh.
