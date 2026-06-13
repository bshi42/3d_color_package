#!/usr/bin/env python3
"""
Landmark canonicalization for 3D Slicer Markups (Fiducials) JSON.

What it does per file:
1) Load control points (label -> 3D position)
2) Plot original landmarks (group-colored)
3) Compute a "spring-pull" rotation (SVD / Wahba-style) so:
     FRONT labels pull toward +X
     BACK  labels pull toward -X
     TOP   labels pull toward +Z
     BOTTOM labels pull toward -Z
4) Apply rotation
5) Scale so distance between FRONT centroid and BACK centroid == 1
6) Translate so BACK centroid is at origin
7) Plot transformed landmarks

Notes:
- If a group has <1 point, it is ignored for that part.
- If FRONT or BACK is missing, scaling/translation steps will raise an error (by design).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt


# =========================
# USER CONSTANTS
# =========================

LANDMARK_FILES: List[str] = [
    '/media/alek/e6852e67-f061-4723-a0d3-c6271961077a/ml_data/color-modeling-pub/3D_Fish/3D_Fish/LMS/Mchenga_m1.mrk.json',
    '/media/alek/e6852e67-f061-4723-a0d3-c6271961077a/ml_data/color-modeling-pub/3D_Fish/3D_Fish/LMS/Nimbo_f1.mrk.json',
    '/media/alek/e6852e67-f061-4723-a0d3-c6271961077a/ml_data/color-modeling-pub/3D_Fish/3D_Fish/LMS/Nimbo_f2.mrk.json',
    '/media/alek/e6852e67-f061-4723-a0d3-c6271961077a/ml_data/color-modeling-pub/3D_Fish/3D_Fish/LMS/yellowhead_m1.mrk.json',
    '/media/alek/e6852e67-f061-4723-a0d3-c6271961077a/ml_data/color-modeling-pub/3D_Fish/3D_Fish/LMS/yellowhead_m2.mrk.json',
    '/media/alek/e6852e67-f061-4723-a0d3-c6271961077a/ml_data/color-modeling-pub/3D_Fish/3D_Fish/LMS/yellowhead_m4.mrk.json',
    '/media/alek/e6852e67-f061-4723-a0d3-c6271961077a/ml_data/color-modeling-pub/3D_Fish/3D_Fish/LMS/yellowhead_m5.mrk.json',
]

MESH_FILES: List[str] = [
    '/media/alek/e6852e67-f061-4723-a0d3-c6271961077a/ml_data/color-modeling-pub/3D_Fish/3D_Fish/Models/Mchenga_m1.obj',
    '/media/alek/e6852e67-f061-4723-a0d3-c6271961077a/ml_data/color-modeling-pub/3D_Fish/3D_Fish/Models/Nimbo_f1.obj',
    '/media/alek/e6852e67-f061-4723-a0d3-c6271961077a/ml_data/color-modeling-pub/3D_Fish/3D_Fish/Models/Nimbo_f2.obj',
    '/media/alek/e6852e67-f061-4723-a0d3-c6271961077a/ml_data/color-modeling-pub/3D_Fish/3D_Fish/Models/yellowhead_m1.obj',
    '/media/alek/e6852e67-f061-4723-a0d3-c6271961077a/ml_data/color-modeling-pub/3D_Fish/3D_Fish/Models/yellowhead_m2.obj',
    '/media/alek/e6852e67-f061-4723-a0d3-c6271961077a/ml_data/color-modeling-pub/3D_Fish/3D_Fish/Models/yellowhead_m4.obj',
    '/media/alek/e6852e67-f061-4723-a0d3-c6271961077a/ml_data/color-modeling-pub/3D_Fish/3D_Fish/Models/yellowhead_m5.obj',
]

TOP_LANDMARKS: List[str] = [
    'TFH', 'TFT'
]

BOTTOM_LANDMARKS: List[str] = [
    'LBWL', 'RBWR', 'RBWL', 'LBWR', 'AH'
]

FRONT_LANDMARKS: List[str] = [
    'FIT', 'FOB', 'FIL', 'FIR', 'FIB'
]

BACK_LANDMARKS: List[str] = [
    'TT', 'TV', 'TA',
]


# =========================
# IMPLEMENTATION
# =========================

AXIS = {
    "front":  np.array([+1.0, 0.0, 0.0], dtype=float),
    "back":   np.array([-1.0, 0.0, 0.0], dtype=float),
    "top":    np.array([0.0, 0.0, +1.0], dtype=float),
    "bottom": np.array([0.0, 0.0, -1.0], dtype=float),
}


@dataclass(frozen=True)
class LandmarkSet:
    labels: List[str]
    P: np.ndarray  # (N,3)


def load_slicer_markups_json(path: str) -> LandmarkSet:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)

    markups = data.get("markups", [])
    if not markups:
        raise ValueError(f"No 'markups' found in {path}")

    # Find first Fiducial markup
    fid = None
    for m in markups:
        if m.get("type") == "Fiducial":
            fid = m
            break
    if fid is None:
        raise ValueError(f"No Fiducial markup found in {path}")

    cps = fid.get("controlPoints", [])
    labels, pts = [], []
    for cp in cps:
        if cp.get("positionStatus") != "defined":
            continue
        lab = cp.get("label")
        pos = cp.get("position")
        if lab is None or pos is None:
            continue
        labels.append(str(lab))
        pts.append([float(pos[0]), float(pos[1]), float(pos[2])])

    if not labels:
        raise ValueError(f"No defined control points found in {path}")

    P = np.asarray(pts, dtype=float)
    return LandmarkSet(labels=labels, P=P)


def indices_for_labels(all_labels: List[str], wanted: List[str]) -> List[int]:
    wanted_set = set(wanted)
    return [i for i, lab in enumerate(all_labels) if lab in wanted_set]


def centroid(P: np.ndarray) -> np.ndarray:
    if P.shape[0] == 0:
        raise ValueError("Cannot compute centroid of empty set")
    return P.mean(axis=0)


def spring_pull_rotation(
    P: np.ndarray,
    labels: List[str],
    top_labels: List[str],
    bottom_labels: List[str],
    front_labels: List[str],
    back_labels: List[str],
    weights: Dict[str, float] | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (R, c0) where:
      - c0 is the pivot/origin used to form v_i = p_i - c0
      - R is a proper rotation matrix (det=+1) that best aligns those v_i
        toward their requested axes in least-squares/Wahba sense.

    We use a semantic pivot to avoid global-centroid bias:
      c0 = average( midpoint(front/back centroids), midpoint(top/bottom centroids) )
    falling back to available midpoints, then global centroid if needed.
    """
    if weights is None:
        weights = {}
    w_top = float(weights.get("top", 1.0))
    w_bottom = float(weights.get("bottom", 1.0))
    w_front = float(weights.get("front", 1.0))
    w_back = float(weights.get("back", 1.0))

    idx_top = indices_for_labels(labels, top_labels)
    idx_bottom = indices_for_labels(labels, bottom_labels)
    idx_front = indices_for_labels(labels, front_labels)
    idx_back = indices_for_labels(labels, back_labels)

    # --- semantic pivot c0 ---
    pivots = []

    if idx_front and idx_back:
        cF = centroid(P[idx_front])
        cB = centroid(P[idx_back])
        pivots.append(0.5 * (cF + cB))

    if idx_top and idx_bottom:
        cT = centroid(P[idx_top])
        cD = centroid(P[idx_bottom])
        pivots.append(0.5 * (cT + cD))

    if pivots:
        c0 = np.mean(np.stack(pivots, axis=0), axis=0)
    else:
        # fallback: global centroid if user didn't provide any groups
        c0 = centroid(P)

    # --- build constraints ---
    # Each constrained landmark contributes: M += w * outer(a, v)
    M = np.zeros((3, 3), dtype=float)

    def add_constraints(idxs: List[int], axis: np.ndarray, w: float) -> None:
        nonlocal M
        for i in idxs:
            v = P[i] - c0
            M += w * np.outer(axis, v)

    add_constraints(idx_front, AXIS["front"], w_front)
    add_constraints(idx_back, AXIS["back"], w_back)
    add_constraints(idx_top, AXIS["top"], w_top)
    add_constraints(idx_bottom, AXIS["bottom"], w_bottom)

    # If M is near-zero, rotation is ill-posed
    if np.linalg.norm(M) < 1e-12:
        raise ValueError(
            "Rotation constraints are ill-posed (no/group constraints or degenerate geometry). "
            "Provide more TOP/BOTTOM/FRONT/BACK landmarks."
        )

    U, _, Vt = np.linalg.svd(M)
    D = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        D[2, 2] = -1.0
    R = U @ D @ Vt  # proper rotation
    return R, c0


def apply_transform(P: np.ndarray, R: np.ndarray, t: np.ndarray, s: float) -> np.ndarray:
    # Transform is: P' = s * (P @ R^T) + t
    return s * (P @ R.T) + t


def group_color(label: str) -> str:
    if label in TOP_LANDMARKS:
        return "tab:orange"
    if label in BOTTOM_LANDMARKS:
        return "tab:blue"
    if label in FRONT_LANDMARKS:
        return "tab:green"
    if label in BACK_LANDMARKS:
        return "tab:red"
    return "0.5"


def plot_landmarks_3d(ax, P: np.ndarray, labels: List[str], title: str) -> None:
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # scatter colored points
    colors = [group_color(l) for l in labels]
    ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=30, c=colors, depthshade=True)

    # label text (optional; can be noisy)
    for (x, y, z), lab in zip(P, labels):
        ax.text(x, y, z, lab, fontsize=7)

    # equal-ish axes
    mins = P.min(axis=0)
    maxs = P.max(axis=0)
    ctr = 0.5 * (mins + maxs)
    span = float(np.max(maxs - mins))
    if span <= 0:
        span = 1.0
    half = 0.5 * span
    ax.set_xlim(ctr[0] - half, ctr[0] + half)
    ax.set_ylim(ctr[1] - half, ctr[1] + half)
    ax.set_zlim(ctr[2] - half, ctr[2] + half)


def main() -> None:
    if not LANDMARK_FILES:
        raise SystemExit("Set LANDMARK_FILES at the top of the script.")

    for fpath in LANDMARK_FILES:
        lm = load_slicer_markups_json(fpath)
        P0 = lm.P.copy()

        # Rotation (spring-pull)
        R, c0 = spring_pull_rotation(
            P=P0,
            labels=lm.labels,
            top_labels=TOP_LANDMARKS,
            bottom_labels=BOTTOM_LANDMARKS,
            front_labels=FRONT_LANDMARKS,
            back_labels=BACK_LANDMARKS,
            weights={"front": 2.0, "back": 2.0, "top": 1.0, "bottom": 1.0},
        )

        # Apply rotation around pivot: Pr = (P - c0) rotated, then put back at same place for now
        Pr = (P0 - c0) @ R.T

        # Scale so dist(front centroid, back centroid) == 1 (in rotated space)
        idx_front = indices_for_labels(lm.labels, FRONT_LANDMARKS)
        idx_back = indices_for_labels(lm.labels, BACK_LANDMARKS)
        if not idx_front or not idx_back:
            raise ValueError("Scaling/translation requires BOTH FRONT_LANDMARKS and BACK_LANDMARKS to be non-empty.")

        cF = centroid(Pr[idx_front])
        cB = centroid(Pr[idx_back])
        d = float(np.linalg.norm(cF - cB))
        if d <= 1e-12:
            raise ValueError("Front/back centroids are coincident or too close; cannot scale to unit length.")
        s = 1.0 / d

        Ps = s * Pr

        # Translate so rear/back centroid is at origin
        cB_s = centroid(Ps[idx_back])
        Pt = Ps - cB_s  # now back centroid at origin

        # Plot
        fig = plt.figure(figsize=(12, 6))
        fig.suptitle(Path(fpath).name)

        ax1 = fig.add_subplot(1, 2, 1, projection="3d")
        plot_landmarks_3d(ax1, P0, lm.labels, "Original landmarks")

        ax2 = fig.add_subplot(1, 2, 2, projection="3d")
        plot_landmarks_3d(ax2, Pt, lm.labels, "Rotated + scaled + translated (back centroid @ origin, front-back dist=1)")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
