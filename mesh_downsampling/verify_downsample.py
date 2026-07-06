#!/usr/bin/env python3
"""Verification harness: score a downsampled textured OBJ mesh against its original.

Reports mesh counts, geometric fidelity (volume, bbox, symmetric sampled
surface distance) and texture/color fidelity (per-sample RGB delta via
barycentric-interpolated UV lookups), with PASS/WARN verdicts per category.

Usage:
    python3 verify_downsample.py <orig_obj> <orig_texture> \
        <downsampled_obj> <downsampled_texture> [--samples N] [--json out.json]

Notes on correctness:
- OBJ wedge UVs: meshes are loaded with trimesh(process=False), which
  duplicates vertices per UV-corner so mesh.visual.uv aligns 1:1 with
  mesh.vertices.  A separate position-merged copy is used for
  watertightness.  Volume (divergence theorem) is face-based and
  unaffected by duplication; it is only strictly valid for watertight,
  consistently-wound meshes (flagged in output).
- Texture V-flip: OBJ vt has origin at bottom-left, PIL at top-left, so
  texture row = (1 - v) * (H - 1).  Validated empirically on the fish
  textures (UV v-range occupies only the flipped half that contains
  image content) and by the orig-vs-orig self test.
- Closest-point queries use an exact, tolerance-free vectorized
  point-triangle routine (Ericson) with a cKDTree over triangle
  centroids.  trimesh.proximity is NOT used: its absolute tolerances
  misclassify closest-point regions on meshes with very small triangles
  (this dataset is meters-scale with ~1e-4 m triangles), producing
  errors up to the triangle size.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

import trimesh

Image.MAX_IMAGE_PIXELS = None  # allow very large textures

# ---------------------------------------------------------------------------
# Thresholds for PASS/WARN verdicts
# ---------------------------------------------------------------------------
THRESH_VOLUME_REL = 0.02          # |volume ratio - 1| <= 2%
THRESH_BBOX_REL = 0.01            # max per-axis extent diff <= 1% of orig extent
THRESH_DIST_P95_FRAC = 0.005      # p95 surface distance <= 0.5% of orig bbox diag
THRESH_RGB_P95 = 20.0             # p95 euclidean RGB delta <= 20 (0-255 scale)

_EVAL_CHUNK = 2_000_000           # rows per vectorized closest-point chunk


# ---------------------------------------------------------------------------
# Exact point-triangle closest point (vectorized Ericson, no tolerances)
# ---------------------------------------------------------------------------
def _closest_point_on_triangles(tri: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Exact closest point on triangle i for point i.  tri (n,3,3), p (n,3)."""
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    ab, ac, ap = b - a, c - a, p - a
    d1 = np.einsum("ij,ij->i", ab, ap)
    d2 = np.einsum("ij,ij->i", ac, ap)
    bp = p - b
    d3 = np.einsum("ij,ij->i", ab, bp)
    d4 = np.einsum("ij,ij->i", ac, bp)
    cp = p - c
    d5 = np.einsum("ij,ij->i", ab, cp)
    d6 = np.einsum("ij,ij->i", ac, cp)
    vc = d1 * d4 - d3 * d2
    vb = d5 * d2 - d1 * d6
    va = d3 * d6 - d5 * d4

    out = np.empty_like(p)
    done = np.zeros(len(p), dtype=bool)

    def take(mask: np.ndarray, value: np.ndarray) -> None:
        m = mask & ~done
        if m.any():
            out[m] = value[m]
            done[m] = True

    with np.errstate(divide="ignore", invalid="ignore"):
        take((d1 <= 0) & (d2 <= 0), a)                                # vertex A
        take((d3 >= 0) & (d4 <= d3), b)                               # vertex B
        t = (d1 / (d1 - d3))[:, None]
        take((vc <= 0) & (d1 >= 0) & (d3 <= 0), a + t * ab)           # edge AB
        take((d6 >= 0) & (d5 <= d6), c)                               # vertex C
        t = (d2 / (d2 - d6))[:, None]
        take((vb <= 0) & (d2 >= 0) & (d6 <= 0), a + t * ac)           # edge AC
        t = ((d4 - d3) / ((d4 - d3) + (d5 - d6)))[:, None]
        take((va <= 0) & (d4 - d3 >= 0) & (d5 - d6 >= 0),
             b + t * (c - b))                                          # edge BC
        denom = va + vb + vc
        v = (vb / denom)[:, None]
        w = (vc / denom)[:, None]
        take(np.ones(len(p), dtype=bool), a + v * ab + w * ac)        # interior

    # Degenerate triangles (zero-area) can leave NaNs: fall back to edges.
    bad = ~np.isfinite(out).all(axis=1)
    if bad.any():
        idx = np.where(bad)[0]
        best_d = np.full(len(idx), np.inf)
        best_p = np.zeros((len(idx), 3))
        for e0, e1 in ((0, 1), (1, 2), (2, 0)):
            s0, s1 = tri[idx, e0], tri[idx, e1]
            seg = s1 - s0
            denom = np.einsum("ij,ij->i", seg, seg)
            t = np.zeros(len(idx))
            nz = denom > 0
            t[nz] = np.einsum("ij,ij->i", p[idx][nz] - s0[nz], seg[nz]) / denom[nz]
            cand = s0 + np.clip(t, 0.0, 1.0)[:, None] * seg
            d = np.linalg.norm(cand - p[idx], axis=1)
            better = d < best_d
            best_d[better] = d[better]
            best_p[better] = cand[better]
        out[idx] = best_p
    return out


def closest_on_mesh(points: np.ndarray, mesh: trimesh.Trimesh, k: int = 8):
    """Exact closest point on `mesh` for each query point.

    Returns (closest_points (n,3), distances (n,), triangle_ids (n,)).
    Exactness: any triangle whose surface comes within the current best
    distance d of a point must have its centroid within d + r_tri of the
    point, so a KD-tree ball query of radius d_upper + max(r_tri) plus a
    per-triangle r_tri filter yields a complete candidate set.
    """
    tris = np.asarray(mesh.triangles, dtype=np.float64)
    pts = np.asarray(points, dtype=np.float64)
    n, m = len(pts), len(tris)
    cent = tris.mean(axis=1)
    r_tri = np.linalg.norm(tris - cent[:, None, :], axis=2).max(axis=1)
    r_max = float(r_tri.max())
    tree = cKDTree(cent)

    # Phase 1: upper bound from exact distance to k nearest-centroid triangles.
    kk = min(k, m)
    _, knn = tree.query(pts, k=kk, workers=-1)
    knn = knn.reshape(n, kk)
    cp1 = _closest_point_on_triangles(
        tris[knn.ravel()], np.repeat(pts, kk, axis=0))
    d1 = np.linalg.norm(cp1 - np.repeat(pts, kk, axis=0), axis=1).reshape(n, kk)
    j = d1.argmin(axis=1)
    rows = np.arange(n)
    d_upper = d1[rows, j]

    # Phase 2: complete candidate set via ball query.
    eps = 1e-12 + 1e-9 * r_max
    balls = tree.query_ball_point(pts, d_upper + r_max + eps, workers=-1)
    lens = np.fromiter((len(b) for b in balls), dtype=np.int64, count=n)
    cand_tid = np.concatenate([np.asarray(b, dtype=np.int64) for b in balls])
    seg = np.repeat(rows, lens)

    # Per-triangle filter: only triangles with |cent-p| <= d_upper + r_tri.
    cd = np.linalg.norm(cent[cand_tid] - pts[seg], axis=1)
    keep = cd <= d_upper[seg] + r_tri[cand_tid] + eps
    cand_tid, seg = cand_tid[keep], seg[keep]

    d = np.empty(len(cand_tid))
    cp = np.empty((len(cand_tid), 3))
    for s in range(0, len(cand_tid), _EVAL_CHUNK):
        sl = slice(s, s + _EVAL_CHUNK)
        cp[sl] = _closest_point_on_triangles(tris[cand_tid[sl]], pts[seg[sl]])
        d[sl] = np.linalg.norm(cp[sl] - pts[seg[sl]], axis=1)

    # Segment argmin (phase-1 best triangle is guaranteed to be a candidate,
    # so every segment is non-empty).
    order = np.lexsort((d, seg))
    firsts = np.searchsorted(seg[order], rows)
    best = order[firsts]
    return cp[best], d[best], cand_tid[best]


# ---------------------------------------------------------------------------
# Loading / counting
# ---------------------------------------------------------------------------
def obj_file_counts(path: str) -> dict:
    """Count v / vt / f records straight from the OBJ file."""
    with open(path, "rb") as fh:
        data = b"\n" + fh.read()
    return {
        "v": data.count(b"\nv "),
        "vt": data.count(b"\nvt "),
        "f": data.count(b"\nf "),
    }


def load_mesh(path: str) -> trimesh.Trimesh:
    """Load an OBJ preserving wedge-UV vertex duplication (process=False)."""
    mesh = trimesh.load(path, process=False)
    if isinstance(mesh, trimesh.Scene):
        geoms = list(mesh.geometry.values())
        mesh = geoms[0] if len(geoms) == 1 else trimesh.util.concatenate(geoms)
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"{path}: loaded {type(mesh)}, expected a triangle mesh")
    return mesh


def load_texture(path: str, _cache: dict = {}) -> np.ndarray:
    key = os.path.realpath(path)
    if key not in _cache:
        _cache[key] = np.asarray(
            Image.open(path).convert("RGB"), dtype=np.uint8)
    return _cache[key]


# ---------------------------------------------------------------------------
# UV / color sampling
# ---------------------------------------------------------------------------
def get_uv(mesh: trimesh.Trimesh, path: str) -> np.ndarray:
    uv = getattr(mesh.visual, "uv", None)
    if uv is None or len(uv) != len(mesh.vertices):
        raise ValueError(f"{path}: no per-vertex UVs found "
                         f"(visual={type(mesh.visual).__name__})")
    return np.asarray(uv, dtype=np.float64)


def barycentric(tri: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Barycentric coords of p (assumed on/near triangle plane); clipped+renormed."""
    bary = trimesh.triangles.points_to_barycentric(tri, p)
    bary = np.clip(bary, 0.0, 1.0)
    s = bary.sum(axis=1, keepdims=True)
    s[s == 0] = 1.0
    return bary / s


def sample_texture_bilinear(uv: np.ndarray, tex: np.ndarray) -> np.ndarray:
    """Bilinear texture lookup.  OBJ vt origin bottom-left -> flip V for PIL rows."""
    h, w = tex.shape[:2]
    u, v = uv[:, 0], uv[:, 1]
    # wrap only out-of-range coords (OBJ repeat convention), keep [0,1] exact
    u = np.where((u >= 0) & (u <= 1), u, u % 1.0)
    v = np.where((v >= 0) & (v <= 1), v, v % 1.0)
    x = u * (w - 1)
    y = (1.0 - v) * (h - 1)          # V-flip: OBJ bottom-left vs PIL top-left
    x0 = np.clip(np.floor(x).astype(np.int64), 0, w - 1)
    y0 = np.clip(np.floor(y).astype(np.int64), 0, h - 1)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    fx = (x - x0)[:, None]
    fy = (y - y0)[:, None]
    c00 = tex[y0, x0].astype(np.float64)
    c10 = tex[y0, x1].astype(np.float64)
    c01 = tex[y1, x0].astype(np.float64)
    c11 = tex[y1, x1].astype(np.float64)
    return (c00 * (1 - fx) * (1 - fy) + c10 * fx * (1 - fy)
            + c01 * (1 - fx) * fy + c11 * fx * fy)


def colors_at(mesh: trimesh.Trimesh, uv: np.ndarray, tex: np.ndarray,
              points: np.ndarray, tids: np.ndarray) -> np.ndarray:
    """RGB at surface points via barycentric-interpolated wedge UVs."""
    tri = np.asarray(mesh.triangles, dtype=np.float64)[tids]
    bary = barycentric(tri, points)
    uv_face = uv[np.asarray(mesh.faces)[tids]]           # (n, 3, 2)
    uv_pt = np.einsum("ij,ijk->ik", bary, uv_face)       # (n, 2)
    return sample_texture_bilinear(uv_pt, tex)


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------
def dist_stats(d: np.ndarray, diag: float) -> dict:
    return {
        "mean": float(d.mean()),
        "p95": float(np.percentile(d, 95)),
        "max": float(d.max()),
        "mean_pct_diag": float(d.mean() / diag * 100),
        "p95_pct_diag": float(np.percentile(d, 95) / diag * 100),
        "max_pct_diag": float(d.max() / diag * 100),
    }


def delta_stats(x: np.ndarray) -> dict:
    return {
        "mean": float(x.mean()),
        "median": float(np.median(x)),
        "p95": float(np.percentile(x, 95)),
        "max": float(x.max()),
    }


def verdict(ok: bool) -> str:
    return "PASS" if ok else "WARN"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run(orig_obj: str, orig_tex_path: str, ds_obj: str, ds_tex_path: str,
        n_samples: int) -> dict:
    t_start = time.time()
    report: dict = {
        "inputs": {
            "orig_obj": orig_obj, "orig_texture": orig_tex_path,
            "downsampled_obj": ds_obj, "downsampled_texture": ds_tex_path,
            "samples": n_samples,
        }
    }

    print(f"[load] {orig_obj}", flush=True)
    orig = load_mesh(orig_obj)
    print(f"[load] {ds_obj}", flush=True)
    ds = load_mesh(ds_obj)
    orig_counts = obj_file_counts(orig_obj)
    ds_counts = obj_file_counts(ds_obj)

    # ---- 1. Counts -------------------------------------------------------
    counts = {
        "orig": {**orig_counts, "triangles": int(len(orig.faces))},
        "downsampled": {**ds_counts, "triangles": int(len(ds.faces))},
        "reduction_factor_vertices":
            orig_counts["v"] / max(ds_counts["v"], 1),
        "reduction_factor_faces":
            len(orig.faces) / max(len(ds.faces), 1),
    }
    report["counts"] = counts

    # ---- 2. Geometry ------------------------------------------------------
    def merged(m: trimesh.Trimesh) -> trimesh.Trimesh:
        c = m.copy()
        c.merge_vertices(merge_tex=True, merge_norm=True)
        return c

    orig_m, ds_m = merged(orig), merged(ds)
    diag = float(np.linalg.norm(orig.extents))
    vol_o, vol_d = float(orig_m.volume), float(ds_m.volume)
    vol_valid_o = bool(orig_m.is_watertight and orig_m.is_winding_consistent)
    vol_valid_d = bool(ds_m.is_watertight and ds_m.is_winding_consistent)
    vol_ratio = abs(vol_d) / abs(vol_o) if vol_o else float("nan")

    ext_o, ext_d = np.asarray(orig.extents), np.asarray(ds.extents)
    bbox_rel = np.abs(ext_d - ext_o) / np.where(ext_o > 0, ext_o, 1.0)

    geometry = {
        "orig": {"watertight": bool(orig_m.is_watertight),
                 "winding_consistent": bool(orig_m.is_winding_consistent),
                 "volume": vol_o, "volume_valid": vol_valid_o,
                 "bbox_extents": ext_o.tolist()},
        "downsampled": {"watertight": bool(ds_m.is_watertight),
                        "winding_consistent": bool(ds_m.is_winding_consistent),
                        "volume": vol_d, "volume_valid": vol_valid_d,
                        "bbox_extents": ext_d.tolist()},
        "volume_ratio_ds_over_orig": vol_ratio,
        "bbox_extents_abs_diff": np.abs(ext_d - ext_o).tolist(),
        "bbox_extents_rel_diff": bbox_rel.tolist(),
        "orig_bbox_diagonal": diag,
    }

    # Symmetric sampled surface distance.
    print(f"[sample] {n_samples} surface points per mesh", flush=True)
    pts_o, fidx_o = trimesh.sample.sample_surface(orig, n_samples, seed=0)
    pts_d, _ = trimesh.sample.sample_surface(ds, n_samples, seed=1)
    pts_o = np.asarray(pts_o, dtype=np.float64)
    pts_d = np.asarray(pts_d, dtype=np.float64)

    print("[distance] orig -> downsampled", flush=True)
    cp_od, d_od, tid_od = closest_on_mesh(pts_o, ds)
    print("[distance] downsampled -> orig", flush=True)
    _, d_do, _ = closest_on_mesh(pts_d, orig)

    geometry["surface_distance"] = {
        "orig_to_ds": dist_stats(d_od, diag),
        "ds_to_orig": dist_stats(d_do, diag),
        "symmetric": dist_stats(np.concatenate([d_od, d_do]), diag),
    }
    report["geometry"] = geometry

    # ---- 3. Texture / color fidelity --------------------------------------
    print("[color] sampling textures", flush=True)
    uv_o = get_uv(orig, orig_obj)
    uv_d = get_uv(ds, ds_obj)
    tex_o = load_texture(orig_tex_path)
    tex_d = load_texture(ds_tex_path)

    rgb_o = colors_at(orig, uv_o, tex_o, pts_o, np.asarray(fidx_o))
    rgb_d = colors_at(ds, uv_d, tex_d, cp_od, tid_od)
    dc = np.abs(rgb_o - rgb_d)                       # (n,3) per-channel
    de = np.linalg.norm(rgb_o - rgb_d, axis=1)       # (n,) euclidean

    report["color"] = {
        "texture_size_orig": list(tex_o.shape[:2][::-1]),
        "texture_size_downsampled": list(tex_d.shape[:2][::-1]),
        "per_channel_abs_delta": {
            ch: delta_stats(dc[:, i]) for i, ch in enumerate("RGB")
        },
        "euclidean_delta": delta_stats(de),
    }

    # ---- 4. Verdicts -------------------------------------------------------
    vol_ok = np.isfinite(vol_ratio) and abs(vol_ratio - 1.0) <= THRESH_VOLUME_REL
    bbox_ok = bool(bbox_rel.max() <= THRESH_BBOX_REL)
    sym_p95_frac = geometry["surface_distance"]["symmetric"]["p95_pct_diag"] / 100
    dist_ok = sym_p95_frac <= THRESH_DIST_P95_FRAC
    rgb_ok = report["color"]["euclidean_delta"]["p95"] <= THRESH_RGB_P95

    vol_note = "" if (vol_valid_o and vol_valid_d) else \
        " [volume validity limited: mesh(es) not watertight/consistently wound]"
    verdicts = {
        "volume": f"{verdict(vol_ok)}: volume ratio {vol_ratio:.4f} "
                  f"(threshold |ratio-1| <= {THRESH_VOLUME_REL:.0%}){vol_note}",
        "bbox": f"{verdict(bbox_ok)}: max bbox extent rel diff "
                f"{bbox_rel.max():.4%} (threshold <= {THRESH_BBOX_REL:.0%})",
        "surface_distance": f"{verdict(dist_ok)}: symmetric p95 distance "
                            f"{sym_p95_frac:.4%} of bbox diag "
                            f"(threshold <= {THRESH_DIST_P95_FRAC:.2%})",
        "color": f"{verdict(rgb_ok)}: p95 euclidean RGB delta "
                 f"{report['color']['euclidean_delta']['p95']:.2f} "
                 f"(threshold <= {THRESH_RGB_P95:.0f})",
    }
    report["verdicts"] = verdicts
    report["overall"] = verdict(vol_ok and bbox_ok and dist_ok and rgb_ok)
    report["elapsed_seconds"] = round(time.time() - t_start, 1)
    return report


def print_report(r: dict) -> None:
    c, g = r["counts"], r["geometry"]
    sd = g["surface_distance"]
    print("\n================ DOWNSAMPLE VERIFICATION REPORT ================")
    print(f"orig: {r['inputs']['orig_obj']}")
    print(f"ds:   {r['inputs']['downsampled_obj']}")
    print("\n-- counts --")
    print(f"  vertices  orig {c['orig']['v']:>9,}  ds {c['downsampled']['v']:>9,}"
          f"  reduction x{c['reduction_factor_vertices']:.2f}")
    print(f"  triangles orig {c['orig']['triangles']:>9,}"
          f"  ds {c['downsampled']['triangles']:>9,}"
          f"  reduction x{c['reduction_factor_faces']:.2f}")
    print("\n-- geometry --")
    for name in ("orig", "downsampled"):
        m = g[name]
        print(f"  {name:>11}: watertight={m['watertight']}"
              f" winding_ok={m['winding_consistent']}"
              f" volume={m['volume']:.6e} (valid={m['volume_valid']})")
    print(f"  volume ratio ds/orig: {g['volume_ratio_ds_over_orig']:.6f}")
    print("  bbox extent rel diff: "
          + ", ".join(f"{x:.4%}" for x in g["bbox_extents_rel_diff"]))
    print(f"  bbox diagonal (orig): {g['orig_bbox_diagonal']:.6f}")
    for k in ("orig_to_ds", "ds_to_orig", "symmetric"):
        s = sd[k]
        print(f"  dist {k:>10}: mean {s['mean']:.3e} ({s['mean_pct_diag']:.4f}%)"
              f"  p95 {s['p95']:.3e} ({s['p95_pct_diag']:.4f}%)"
              f"  max {s['max']:.3e} ({s['max_pct_diag']:.4f}%)")
    print("\n-- color (0-255) --")
    for ch, s in r["color"]["per_channel_abs_delta"].items():
        print(f"  |d{ch}|: mean {s['mean']:.3f}  median {s['median']:.3f}"
              f"  p95 {s['p95']:.3f}  max {s['max']:.3f}")
    e = r["color"]["euclidean_delta"]
    print(f"  |dRGB| euclid: mean {e['mean']:.3f}  median {e['median']:.3f}"
          f"  p95 {e['p95']:.3f}  max {e['max']:.3f}")
    print("\n-- verdicts --")
    for k, v in r["verdicts"].items():
        print(f"  {k}: {v}")
    print(f"  OVERALL: {r['overall']}   ({r['elapsed_seconds']}s)")
    print("=================================================================")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Score a downsampled textured mesh against its original.")
    ap.add_argument("orig_obj")
    ap.add_argument("orig_texture")
    ap.add_argument("downsampled_obj")
    ap.add_argument("downsampled_texture")
    ap.add_argument("--samples", type=int, default=50000,
                    help="surface sample count per mesh (default 50000)")
    ap.add_argument("--json", default=None, help="write full report JSON here")
    args = ap.parse_args(argv)

    for p in (args.orig_obj, args.orig_texture,
              args.downsampled_obj, args.downsampled_texture):
        if not os.path.isfile(p):
            ap.error(f"input file not found: {p}")

    report = run(args.orig_obj, args.orig_texture,
                 args.downsampled_obj, args.downsampled_texture, args.samples)
    print_report(report)
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(report, fh, indent=2)
        print(f"[json] wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
