"""
Merged Blender Python script: TPS-warped volumetric brush coloring.

Combines:
  (A) Landmark canonicalization (Procrustes/spring-pull orientation, scaling, translation)
  (B) Volumetric brush coloring with baking to texture

Pipeline:
  1. Load reference landmarks -> compute canonical transform T_ref -> L_ref_can
  2. For each specimen: load landmarks -> compute T_i -> L_i_can -> build TPS_i(L_ref_can -> L_i_can)
  3. For N_SAMPLES iterations: pick random specimen, generate brushes in ref canonical space,
     warp through TPS_i, transform to specimen world space, apply to mesh, bake, save.
"""

import bpy
import colorsys
import copy
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Blender's snap Python disables user site-packages; add them explicitly
# so that pip-installed packages (pandas, scikit-learn, etc.) are visible.
import site
_user_site = site.getusersitepackages()
if _user_site not in sys.path:
    sys.path.append(_user_site)
sys.path.append("/snap/blender/common/BlenderPython")

import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs

# ═══════════════════════════════════════════════════════════════════════════════
# USER CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

LANDMARK_FILES: List[str] = [
    # [0] is the REFERENCE specimen
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

TOP_LANDMARKS: List[str] = ['TFH', 'TFT']
BOTTOM_LANDMARKS: List[str] = ['LBWL', 'RBWR', 'RBWL', 'LBWR', 'AH']
FRONT_LANDMARKS: List[str] = ['FIT', 'FOB', 'FIL', 'FIR', 'FIB']
BACK_LANDMARKS: List[str] = ['TT', 'TV', 'TA']

TPS_LAMBDA: float = 1e-6

OUTPUT_DIR = "/tmp/ml_data/fishy_tps"
IMAGE_SIZE = 2048
SEED = 49
N_SAMPLES = 1

ADD_NOISE = True
NOISE_AMOUNT = 0.10
NOISE_SCALE = 8.0

# Palette
BLUE = colorsys.rgb_to_hsv(39 / 255., 70 / 255., 144 / 255.)
YELLOW = colorsys.rgb_to_hsv(242 / 255., 255 / 255., 73 / 255.)
GREEN = colorsys.rgb_to_hsv(21 / 255., 127 / 255., 31 / 255.)
BLACK = colorsys.rgb_to_hsv(0., 0., 0.)
RED = colorsys.rgb_to_hsv(255 / 255., 102 / 255., 99 / 255.)

BASE_COLOR_HUE_STD = 0.015
BASE_COLOR_SAT_STD = 0.05
BASE_COLOR_VAL_STD = 0.05

NOMINAL_FIRST_STRIPE_START = (0.8, 0.02, 0.07)
NOMINAL_FIRST_STRIPE_END = (0.8, 0.1, -0.08)

STRIPE_COUNT_PROB_4 = 0.8

NOMINAL_STRIPE_SPACING = -0.08
STRIPE_SPACING_STD = 0.01

NOMINAL_STRIPE_WIDTH = 0.025
STRIPE_WIDTH_STD = 0.005

NOMINAL_STRIPE_LONGITUDINAL_OFFSET = 0.00
STRIPE_LONGITUDINAL_OFFSET_STD = 0.01

BELLY_HUE_SHIFT_RANGE = (-0.1, 0.1)
BELLY_HUE_SHIFT_STD = 0.015
BELLY_STRENGTH_STD = 0.05
BELLY_TRANSLATION_STD = 0.05

TAIL_HUE_SHIFT_RANGE = (-0.1, 0.1)
TAIL_HUE_SHIFT_STD = 0.015
TAIL_STRENGTH_STD = 0.05

ROSY_CHEEKS_PROB = 0.99
ROSY_CHEEKS_HUE_STD = 0.015
ROSY_CHEEKS_STRENGTH_STD = 0.05
ROSY_CHEEKS_TRANSLATION_Y_STD = 0.05
ROSY_CHEEKS_TRANSLATION_Z_STD = 0.05


# ═══════════════════════════════════════════════════════════════════════════════
# LANDMARK LOADING
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class LandmarkSet:
    labels: List[str]
    P: np.ndarray  # (N, 3)


def load_slicer_markups_json(path: str) -> LandmarkSet:
    """Load a 3D Slicer Markups Fiducials JSON file."""
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    markups = data.get("markups", [])
    if not markups:
        raise ValueError(f"No 'markups' found in {path}")
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
    return LandmarkSet(labels=labels, P=np.asarray(pts, dtype=np.float64))


def indices_for_labels(all_labels: List[str], wanted: List[str]) -> List[int]:
    wanted_set = set(wanted)
    return [i for i, lab in enumerate(all_labels) if lab in wanted_set]


def centroid(P: np.ndarray) -> np.ndarray:
    if P.shape[0] == 0:
        raise ValueError("Cannot compute centroid of empty set")
    return P.mean(axis=0)




# ═══════════════════════════════════════════════════════════════════════════════
# CANONICAL TRANSFORM (spring-pull rotation + scale + translate)
# ═══════════════════════════════════════════════════════════════════════════════

AXIS = {
    "front":  np.array([+1.0, 0.0, 0.0], dtype=np.float64),
    "back":   np.array([-1.0, 0.0, 0.0], dtype=np.float64),
    "top":    np.array([0.0, 0.0, +1.0], dtype=np.float64),
    "bottom": np.array([0.0, 0.0, -1.0], dtype=np.float64),
}


def spring_pull_rotation(
    P: np.ndarray,
    labels: List[str],
    top_labels: List[str],
    bottom_labels: List[str],
    front_labels: List[str],
    back_labels: List[str],
    weights: Optional[Dict[str, float]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (R, c0) where:
      - c0 is the semantic pivot
      - R is a proper rotation (det=+1) via SVD / Wahba solution
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

    # Semantic pivot
    pivots = []
    if idx_front and idx_back:
        pivots.append(0.5 * (centroid(P[idx_front]) + centroid(P[idx_back])))
    if idx_top and idx_bottom:
        pivots.append(0.5 * (centroid(P[idx_top]) + centroid(P[idx_bottom])))
    c0 = np.mean(np.stack(pivots, axis=0), axis=0) if pivots else centroid(P)

    # Build cross-correlation matrix M
    M = np.zeros((3, 3), dtype=np.float64)

    def add_constraints(idxs, axis, w):
        nonlocal M
        for i in idxs:
            v = P[i] - c0
            M += w * np.outer(axis, v)

    add_constraints(idx_front, AXIS["front"], w_front)
    add_constraints(idx_back, AXIS["back"], w_back)
    add_constraints(idx_top, AXIS["top"], w_top)
    add_constraints(idx_bottom, AXIS["bottom"], w_bottom)

    if np.linalg.norm(M) < 1e-12:
        raise ValueError("Rotation constraints are ill-posed.")

    U, _, Vt = np.linalg.svd(M)
    D = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        D[2, 2] = -1.0
    R = U @ D @ Vt
    return R, c0


@dataclass
class CanonicalTransform:
    """Stores all parameters of the world->canonical mapping."""
    R: np.ndarray           # (3,3) rotation
    c0: np.ndarray          # (3,) pivot used for rotation
    scale: float            # uniform scale factor
    translation: np.ndarray # (3,) post-scale translation (= -cB_scaled)


def compute_canonical_transform(lm: LandmarkSet) -> Tuple[CanonicalTransform, np.ndarray]:
    """
    Compute canonical transform and return (transform, canonical_landmarks).

    Steps:
      1. Spring-pull rotation around semantic pivot
      2. Scale so front-back centroid distance == 1
      3. Translate so back centroid is at origin
    """
    P0 = lm.P.copy()

    R, c0 = spring_pull_rotation(
        P=P0, labels=lm.labels,
        top_labels=TOP_LANDMARKS, bottom_labels=BOTTOM_LANDMARKS,
        front_labels=FRONT_LANDMARKS, back_labels=BACK_LANDMARKS,
        weights={"front": 2.0, "back": 2.0, "top": 1.0, "bottom": 1.0},
    )

    # Rotate around pivot
    Pr = (P0 - c0) @ R.T

    # Scale
    idx_front = indices_for_labels(lm.labels, FRONT_LANDMARKS)
    idx_back = indices_for_labels(lm.labels, BACK_LANDMARKS)
    if not idx_front or not idx_back:
        raise ValueError("Need both FRONT and BACK landmarks for scaling/translation.")
    cF = centroid(Pr[idx_front])
    cB = centroid(Pr[idx_back])
    d = float(np.linalg.norm(cF - cB))
    if d <= 1e-12:
        raise ValueError("Front/back centroids coincident; cannot scale.")
    scale = 1.0 / d
    Ps = scale * Pr

    # Translate so back centroid at origin
    cB_s = centroid(Ps[idx_back])
    translation = -cB_s
    P_can = Ps + translation

    ct = CanonicalTransform(R=R, c0=c0, scale=scale, translation=translation)
    return ct, P_can


# ═══════════════════════════════════════════════════════════════════════════════
# TRANSFORM UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def world_to_canonical(P: np.ndarray, ct: CanonicalTransform) -> np.ndarray:
    """Transform points from world space to canonical space.
    P_can = scale * (P - c0) @ R.T + translation
    """
    return ct.scale * ((P - ct.c0) @ ct.R.T) + ct.translation


def canonical_to_world(P_can: np.ndarray, ct: CanonicalTransform) -> np.ndarray:
    """Transform points from canonical space back to world space.
    P = (P_can - translation) / scale @ R + c0
    """
    return ((P_can - ct.translation) / ct.scale) @ ct.R + ct.c0


# ═══════════════════════════════════════════════════════════════════════════════
# TPS3D — 3-D Thin-Plate Spline
# ═══════════════════════════════════════════════════════════════════════════════

class TPS3D:
    """3-D Thin-Plate Spline warp (regularized).

    Kernel: U(r) = r^2 * log(r + eps)
    Solves the augmented (K+4)x(K+4) system with Tikhonov regularization
    on the kernel block only.  All arithmetic in float64.
    """

    def __init__(self):
        self.W: Optional[np.ndarray] = None   # (K, 3)
        self.A: Optional[np.ndarray] = None   # (4, 3)
        self.ctrl: Optional[np.ndarray] = None # (K, 3)

    @staticmethod
    def _U(r: np.ndarray) -> np.ndarray:
        eps = 1e-12
        return r ** 2 * np.log(r + eps)

    def fit(self, X: np.ndarray, Y: np.ndarray, lambda_reg: float = 1e-6) -> None:
        """Fit TPS from source control points X to target Y.

        X, Y : (K, 3) — matched landmark arrays
        lambda_reg : Tikhonov regularization weight
        """
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        K = X.shape[0]
        assert X.shape == (K, 3) and Y.shape == (K, 3)
        if K < 4:
            raise ValueError(f"TPS requires >= 4 control points, got {K}")

        # Check that control points span 3D (not coplanar).
        # SVD of centered coordinates; require 3rd singular value > eps (rank 3).
        X_centered = X - X.mean(axis=0, keepdims=True)
        sv_geom = np.linalg.svd(X_centered, compute_uv=False)
        if sv_geom[2] < 1e-10:
            raise ValueError(
                "Control points are (near-)coplanar. "
                "TPS in 3D requires points that span all three dimensions."
            )

        self.ctrl = X.copy()
        P_mat = np.hstack([np.ones((K, 1), dtype=np.float64), X])  # (K, 4)

        # Kernel matrix
        diff = X[:, None, :] - X[None, :, :]   # (K, K, 3)
        dists = np.sqrt((diff ** 2).sum(axis=2))  # (K, K)
        K_mat = self._U(dists)                     # (K, K)

        # Augmented system
        L = np.zeros((K + 4, K + 4), dtype=np.float64)
        L[:K, :K] = K_mat + lambda_reg * np.eye(K, dtype=np.float64)
        L[:K, K:K + 4] = P_mat
        L[K:K + 4, :K] = P_mat.T
        # bottom-right 4x4 stays zero

        rhs = np.zeros((K + 4, 3), dtype=np.float64)
        rhs[:K] = Y

        params = np.linalg.solve(L, rhs)
        self.W = params[:K]
        self.A = params[K:]

    def evaluate(self, P: np.ndarray) -> np.ndarray:
        """Warp arbitrary points P through the fitted TPS.

        P : (N, 3) -> returns (N, 3)
        """
        P = np.asarray(P, dtype=np.float64)
        N = P.shape[0]
        diff = P[:, None, :] - self.ctrl[None, :, :]  # (N, K, 3)
        dists = np.sqrt((diff ** 2).sum(axis=2))        # (N, K)
        U_vals = self._U(dists)                          # (N, K)
        P_aug = np.hstack([np.ones((N, 1), dtype=np.float64), P])  # (N, 4)
        return U_vals @ self.W + P_aug @ self.A

    def local_scale(self, P: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        """Estimate isotropic local scale factor at each point in P.

        Probes 6 axis-aligned offsets (±eps along x, y, z), measures how
        distances change through the warp, and returns the mean stretch
        per point as a (N,) array.
        """
        P = np.asarray(P, dtype=np.float64)
        N = P.shape[0]
        scales = np.zeros(N, dtype=np.float64)
        for axis in range(3):
            offset = np.zeros((N, 3), dtype=np.float64)
            offset[:, axis] = eps
            fwd = self.evaluate(P + offset)
            bwd = self.evaluate(P - offset)
            stretch = np.linalg.norm(fwd - bwd, axis=1) / (2.0 * eps)
            scales += stretch
        return scales / 3.0  # average over 3 axes


# ═══════════════════════════════════════════════════════════════════════════════
# SPECIMEN PREPROCESSING & CACHE
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SpecimenCache:
    """Pre-computed data for one specimen."""
    index: int
    landmark_file: str
    mesh_file: str
    transform: CanonicalTransform
    landmarks_canonical: np.ndarray  # (M, 3) all landmarks in canonical space
    labels: List[str]
    tps: TPS3D
    n_landmarks_used: int
    tps_rmse: float


def preprocess_specimens() -> Tuple[CanonicalTransform, np.ndarray, List[str], List[SpecimenCache]]:
    """
    Load all specimens, compute canonical transforms, build TPS warps.

    Returns:
        ref_ct          : reference canonical transform
        ref_lm_can      : (M, 3) reference landmarks in canonical space
        ref_labels       : list of reference landmark labels
        specimens       : list of SpecimenCache (one per specimen including reference)
    """
    assert len(LANDMARK_FILES) == len(MESH_FILES), \
        f"LANDMARK_FILES ({len(LANDMARK_FILES)}) and MESH_FILES ({len(MESH_FILES)}) must have same length."

    # --- Reference specimen (index 0) ---
    ref_lm = load_slicer_markups_json(LANDMARK_FILES[0])
    ref_ct, ref_lm_can = compute_canonical_transform(ref_lm)
    ref_labels = ref_lm.labels
    print(f"[ref] {Path(LANDMARK_FILES[0]).name}: {len(ref_labels)} landmarks, "
          f"canonical transform computed.")

    specimens: List[SpecimenCache] = []

    for i in range(len(LANDMARK_FILES)):
        lm_i = load_slicer_markups_json(LANDMARK_FILES[i])
        ct_i, lm_i_can = compute_canonical_transform(lm_i)

        # Find shared labels
        shared = [lab for lab in ref_labels if lab in set(lm_i.labels)]
        if len(shared) < 4:
            raise ValueError(
                f"Specimen {i} ({Path(LANDMARK_FILES[i]).name}): only {len(shared)} shared "
                f"labels with reference — need >= 4."
            )

        # Extract matched landmark subsets in canonical space
        ref_idx = [ref_labels.index(lab) for lab in shared]
        spec_idx = [lm_i.labels.index(lab) for lab in shared]
        X = ref_lm_can[ref_idx]   # reference canonical
        Y = lm_i_can[spec_idx]    # specimen canonical

        # Fit TPS: reference canonical -> specimen canonical
        tps_i = TPS3D()
        tps_i.fit(X, Y, lambda_reg=TPS_LAMBDA)

        # Validate
        Y_pred = tps_i.evaluate(X)
        rmse = float(np.sqrt(np.mean((Y_pred - Y) ** 2)))
        print(f"[specimen {i}] {Path(LANDMARK_FILES[i]).name}: "
              f"{len(shared)} shared landmarks, TPS RMSE = {rmse:.6f}")

        specimens.append(SpecimenCache(
            index=i,
            landmark_file=LANDMARK_FILES[i],
            mesh_file=MESH_FILES[i],
            transform=ct_i,
            landmarks_canonical=lm_i_can,
            labels=lm_i.labels,
            tps=tps_i,
            n_landmarks_used=len(shared),
            tps_rmse=rmse,
        ))

    return ref_ct, ref_lm_can, ref_labels, specimens


# ═══════════════════════════════════════════════════════════════════════════════
# COLORING — parameter sampling & brush generation (from color_fishy.py)
# ═══════════════════════════════════════════════════════════════════════════════

def sample_space(n_samples: int, random_state: int = 42) -> List[dict]:
    """Sample n_samples parameter sets for fishy coloration."""
    np.random.seed(random_state)
    base_color_hue = np.random.normal(YELLOW[0], BASE_COLOR_HUE_STD, n_samples) % 1.0
    base_color_sat = np.clip(np.random.normal(0, BASE_COLOR_SAT_STD, n_samples) + YELLOW[1], 0, 1)
    base_color_val = np.clip(np.random.normal(0, BASE_COLOR_VAL_STD, n_samples) + YELLOW[2], 0, 1)
    stripe_count = np.random.choice([4, 5], n_samples, p=[STRIPE_COUNT_PROB_4, 1 - STRIPE_COUNT_PROB_4])
    stripe_spacing = np.random.normal(NOMINAL_STRIPE_SPACING, STRIPE_SPACING_STD, n_samples)
    stripe_width = np.random.normal(NOMINAL_STRIPE_WIDTH, STRIPE_WIDTH_STD, n_samples)
    stripe_longitudinal_offset = np.random.normal(
        NOMINAL_STRIPE_LONGITUDINAL_OFFSET, STRIPE_LONGITUDINAL_OFFSET_STD, n_samples)
    belly_hue_shift = make_blobs(
        n_samples=n_samples, n_features=1, centers=2,
        cluster_std=BELLY_HUE_SHIFT_STD, center_box=BELLY_HUE_SHIFT_RANGE)[0][:, 0]
    belly_hue = (BLUE[0] + belly_hue_shift) % 1.0
    belly_strength = np.clip(np.random.normal(1.0 - BELLY_STRENGTH_STD, BELLY_STRENGTH_STD, n_samples), 0, 1)
    belly_translation = np.random.normal(0, BELLY_TRANSLATION_STD, n_samples)
    tail_hue_shift = make_blobs(
        n_samples=n_samples, n_features=1, centers=2,
        cluster_std=TAIL_HUE_SHIFT_STD, center_box=TAIL_HUE_SHIFT_RANGE)[0][:, 0]
    tail_hue = (GREEN[0] + tail_hue_shift) % 1.0
    tail_strength = np.clip(np.random.normal(1.0 - TAIL_STRENGTH_STD, TAIL_STRENGTH_STD, n_samples), 0, 1)
    rosy_cheeks_present = np.random.choice([0, 1], n_samples, p=[1 - ROSY_CHEEKS_PROB, ROSY_CHEEKS_PROB])
    rosy_cheeks_hue = np.random.normal(RED[0], ROSY_CHEEKS_HUE_STD, n_samples) % 1.0
    rosy_cheeks_strength = np.clip(
        np.random.normal(1.0 - ROSY_CHEEKS_STRENGTH_STD, ROSY_CHEEKS_STRENGTH_STD, n_samples), 0, 1)
    rosy_cheeks_translation_y = np.random.normal(0, ROSY_CHEEKS_TRANSLATION_Y_STD, n_samples)
    rosy_cheeks_translation_z = np.random.normal(0, ROSY_CHEEKS_TRANSLATION_Z_STD, n_samples)

    return [{
        "base_color_hue": base_color_hue[i],
        "base_color_sat": base_color_sat[i],
        "base_color_val": base_color_val[i],
        "stripe_count": stripe_count[i],
        "stripe_spacing": stripe_spacing[i],
        "stripe_width": stripe_width[i],
        "stripe_longitudinal_offset": stripe_longitudinal_offset[i],
        "belly_hue": belly_hue[i],
        "belly_strength": belly_strength[i],
        "belly_translation": belly_translation[i],
        "tail_hue": tail_hue[i],
        "tail_strength": tail_strength[i],
        "rosy_cheeks_present": rosy_cheeks_present[i],
        "rosy_cheeks_hue": rosy_cheeks_hue[i],
        "rosy_cheeks_strength": rosy_cheeks_strength[i],
        "rosy_cheeks_translation_y": rosy_cheeks_translation_y[i],
        "rosy_cheeks_translation_z": rosy_cheeks_translation_z[i],
    } for i in range(n_samples)]


def generate_stripes(stripe_spacing, stripe_width, stripe_longitudinal_offset, stripe_count):
    stripes = []
    for i in range(stripe_count):
        stripe = {
            "type": "capsule",
            "a": NOMINAL_FIRST_STRIPE_START,
            "b": NOMINAL_FIRST_STRIPE_END,
            "sigma": stripe_width,
            "color": BLACK,
            "strength": 1.0,
        }
        stripe["b"] = (stripe["b"][0] + i * stripe_spacing + stripe_longitudinal_offset,
                        stripe["b"][1],
                        stripe["b"][2])
        stripe["a"] = (stripe["a"][0] + i * stripe_spacing + stripe_longitudinal_offset,
                        stripe["a"][1],
                        stripe["a"][2])
        stripes.append(stripe)
        mirrored = copy.deepcopy(stripe)
        mirrored["a"] = (stripe["a"][0], -stripe["a"][1], stripe["a"][2])
        mirrored["b"] = (stripe["b"][0], -stripe["b"][1], stripe["b"][2])
        stripes.append(mirrored)
    return stripes


def generate_base_color(base_color_hue, base_color_sat, base_color_val):
    rgb = colorsys.hsv_to_rgb(base_color_hue, base_color_sat, base_color_val)
    return [{"type": "sphere", "center": (0.5, 0.0, 0.0),
             "solid_radius": 1, "sigma": 1, "color": rgb, "strength": 1.0}]


def generate_belly(belly_hue, belly_strength, belly_translation):
    rgb = colorsys.hsv_to_rgb(belly_hue, BLUE[1], BLUE[2])
    return [{"type": "sphere", "center": (0.30 + belly_translation, 0.0, -0.55),
             "solid_radius": 0.4, "sigma": 0.05, "color": rgb, "strength": belly_strength}]


def generate_tail(tail_hue, tail_strength):
    rgb = colorsys.hsv_to_rgb(tail_hue, GREEN[1], GREEN[2])
    return [{"type": "sphere", "center": (0.0, 0.0, 0.0),
             "solid_radius": 0.1, "sigma": 0.03, "color": rgb, "strength": tail_strength}]


def generate_rosy_cheeks(rosy_cheeks_present, rosy_cheeks_hue, rosy_cheeks_strength,
                         rosy_cheeks_translation_y, rosy_cheeks_translation_z):
    rgb = colorsys.hsv_to_rgb(rosy_cheeks_hue, RED[1], RED[2])
    if rosy_cheeks_present == 0:
        return []
    return [
        {"type": "sphere",
         "center": (0.8 + rosy_cheeks_translation_y, 0.1, -0.1 + rosy_cheeks_translation_z),
         "solid_radius": 0.035, "sigma": 0.01, "color": rgb, "strength": rosy_cheeks_strength},
        {"type": "sphere",
         "center": (0.8 + rosy_cheeks_translation_y, -0.1, -0.1 + rosy_cheeks_translation_z),
         "solid_radius": 0.035, "sigma": 0.01, "color": rgb, "strength": rosy_cheeks_strength},
    ]


def generate_fishy_coloration(
    base_color_hue, base_color_sat, base_color_val,
    stripe_count, stripe_spacing, stripe_width, stripe_longitudinal_offset,
    belly_hue, belly_strength, belly_translation,
    tail_hue, tail_strength,
    rosy_cheeks_present, rosy_cheeks_hue, rosy_cheeks_strength,
    rosy_cheeks_translation_y, rosy_cheeks_translation_z,
):
    """Generate all brushes for one fish coloration (in REFERENCE CANONICAL SPACE)."""
    fishy = []
    fishy += generate_base_color(base_color_hue, base_color_sat, base_color_val)
    fishy += generate_belly(belly_hue, belly_strength, belly_translation)
    fishy += generate_tail(tail_hue, tail_strength)
    fishy += generate_rosy_cheeks(rosy_cheeks_present, rosy_cheeks_hue,
                                  rosy_cheeks_strength, rosy_cheeks_translation_y,
                                  rosy_cheeks_translation_z)
    fishy += generate_stripes(stripe_spacing, stripe_width, stripe_longitudinal_offset, stripe_count)
    return fishy


# ═══════════════════════════════════════════════════════════════════════════════
# APPLY BRUSHES (from color_fishy.py — unchanged)
# ═══════════════════════════════════════════════════════════════════════════════

def gaussian_tail(d: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian falloff: 1 at d=0, decays outward."""
    return np.exp(-0.5 * (d / (sigma + 1e-12)) ** 2)


def apply_brushes(
    P: np.ndarray,
    brushes: List[dict],
    base_rgb=None,
    add_noise: bool = False,
    noise_amount: float = 0.10,
    noise_scale: float = 8.0,
) -> np.ndarray:
    """
    P: (N,3) world-space vertex positions
    brushes: list[dict] with type in {"sphere","capsule"}
    Returns: (N,3) RGB in [0,1]
    """
    N = P.shape[0]
    if base_rgb is None:
        col = np.zeros((N, 3), dtype=np.float64)
    else:
        col = np.tile(np.asarray(base_rgb, dtype=np.float64), (N, 1))

    if add_noise:
        g = (np.sin(P[:, 0] * noise_scale * 3.1) +
             np.sin(P[:, 1] * noise_scale * 2.3 + 1.7) +
             np.sin(P[:, 2] * noise_scale * 4.0 + 3.4)) / 3.0
        g = (g - g.min()) / (g.max() - g.min() + 1e-12)
        noise_rgb = np.stack([g, g, g], axis=1)
        col = (1 - noise_amount) * col + noise_amount * noise_rgb

    def dist_point_to_segment_batch(P_, a_, b_):
        ab = b_ - a_
        denom = np.dot(ab, ab)
        if denom < 1e-20:
            return np.linalg.norm(P_ - a_, axis=1)
        t = np.clip(((P_ - a_) @ ab) / denom, 0.0, 1.0)
        closest = a_[None, :] + t[:, None] * ab[None, :]
        return np.linalg.norm(P_ - closest, axis=1)

    for b in brushes:
        strength = float(b.get("strength", 1.0))
        if b["type"] == "sphere":
            c = np.array(b["center"], dtype=np.float64)
            d_center = np.linalg.norm(P - c, axis=1)
            solid_r = float(b.get("solid_radius", 0.0))
            sigma = float(b["sigma"])
            alpha = np.zeros_like(d_center)
            inside = d_center <= solid_r
            alpha[inside] = 1.0
            if np.any(~inside):
                d_tail = np.maximum(d_center - solid_r, 0.0)
                alpha[~inside] = gaussian_tail(d_tail[~inside], sigma)

        elif b["type"] == "capsule":
            a_pt = np.array(b["a"], dtype=np.float64)
            b_pt = np.array(b["b"], dtype=np.float64)
            d_axis = dist_point_to_segment_batch(P, a_pt, b_pt)
            solid_r = float(b.get("solid_radius", 0.0))
            sigma = float(b["sigma"])
            alpha = np.zeros_like(d_axis)
            inside = d_axis <= solid_r
            alpha[inside] = 1.0
            if np.any(~inside):
                d_tail = np.maximum(d_axis - solid_r, 0.0)
                alpha[~inside] = gaussian_tail(d_tail[~inside], sigma)
        else:
            continue

        a = np.clip(alpha * strength, 0.0, 1.0)
        brush_rgb = np.array(b["color"], dtype=np.float64)
        col = col * (1.0 - a[:, None]) + brush_rgb[None, :] * a[:, None]

    return np.clip(col, 0.0, 1.0)


# ═══════════════════════════════════════════════════════════════════════════════
# BRUSH WARPING (reference canonical → specimen world)
# ═══════════════════════════════════════════════════════════════════════════════

def warp_brushes(
    brushes: List[dict],
    tps: TPS3D,
    specimen_ct: CanonicalTransform,
) -> List[dict]:
    """
    Warp brushes from REFERENCE CANONICAL space to SPECIMEN WORLD space.

    Steps per brush:
      1. Collect control points (center for sphere; a,b for capsule)
      2. Warp through TPS (ref canonical → specimen canonical)
      3. Transform canonical → specimen world via inverse canonical transform
      4. Scale sigma / solid_radius by the local scale factor of the full
         ref-canonical → specimen-world chain:
           local_scale_tps  (from TPS Jacobian)  ×  (1 / specimen_ct.scale)

    Returns new brush list with warped geometry and scaled radii.
    """
    # The canonical→world step has a uniform scale of 1/ct.scale (it undoes
    # the normalization).  The TPS step may have spatially-varying scale which
    # we estimate via finite differences.
    can_to_world_scale = 1.0 / specimen_ct.scale

    warped = []
    for b in brushes:
        wb = copy.deepcopy(b)
        if b["type"] == "sphere":
            pts = np.array([b["center"]], dtype=np.float64)  # (1, 3)
            tps_scale = tps.local_scale(pts)  # (1,)
            total_scale = float(tps_scale[0]) * can_to_world_scale
            pts_spec_can = tps.evaluate(pts)
            pts_world = canonical_to_world(pts_spec_can, specimen_ct)
            wb["center"] = tuple(pts_world[0].tolist())
            wb["sigma"] = float(b["sigma"]) * total_scale
            if "solid_radius" in b:
                wb["solid_radius"] = float(b["solid_radius"]) * total_scale
        elif b["type"] == "capsule":
            pts = np.array([b["a"], b["b"]], dtype=np.float64)  # (2, 3)
            tps_scale = tps.local_scale(pts)  # (2,)
            avg_tps_scale = float(tps_scale.mean())
            total_scale = avg_tps_scale * can_to_world_scale
            pts_spec_can = tps.evaluate(pts)
            pts_world = canonical_to_world(pts_spec_can, specimen_ct)
            wb["a"] = tuple(pts_world[0].tolist())
            wb["b"] = tuple(pts_world[1].tolist())
            wb["sigma"] = float(b["sigma"]) * total_scale
            if "solid_radius" in b:
                wb["solid_radius"] = float(b["solid_radius"]) * total_scale
        else:
            # Unknown brush type — pass through unchanged
            pass
        warped.append(wb)
    return warped


# ═══════════════════════════════════════════════════════════════════════════════
# BLENDER / MESH HANDLING
# ═══════════════════════════════════════════════════════════════════════════════

def cleanup_scene() -> None:
    """Remove all objects, meshes, materials, and images from the scene."""
    # Remove objects
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    # Purge orphan data
    for block in bpy.data.meshes:
        bpy.data.meshes.remove(block, do_unlink=True)
    for block in bpy.data.materials:
        bpy.data.materials.remove(block, do_unlink=True)
    for block in bpy.data.images:
        bpy.data.images.remove(block, do_unlink=True)


def import_obj(mesh_path: str):
    """Import an OBJ file and return the imported object."""
    bpy.ops.wm.read_factory_settings(use_empty=True)
    # Use forward_axis='Y', up_axis='Z' to preserve raw OBJ coordinates,
    # which already match the Slicer landmark coordinate system.
    bpy.ops.wm.obj_import(filepath=mesh_path, forward_axis='Y', up_axis='Z')
    obj = bpy.context.selected_objects[0]
    bpy.context.view_layer.objects.active = obj
    return obj


def ensure_uv(obj) -> None:
    """Ensure the object has at least one UV map."""
    me = obj.data
    if not me.uv_layers:
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.uv.smart_project(island_margin=0.02)
        bpy.ops.object.mode_set(mode='OBJECT')


def world_vertex_positions(obj) -> np.ndarray:
    """Extract (N,3) world-space vertex positions."""
    mw = obj.matrix_world
    verts = obj.data.vertices
    return np.array([mw @ v.co for v in verts], dtype=np.float64)


def write_vertex_colors(obj, rgb: np.ndarray) -> None:
    """Write per-vertex RGB to a CORNER-domain color attribute 'SynColor'."""
    me = obj.data
    layer_name = "SynColor"
    if layer_name in me.color_attributes:
        ca = me.color_attributes[layer_name]
    else:
        ca = me.color_attributes.new(name=layer_name, domain='CORNER', type='BYTE_COLOR')
    data = ca.data
    loops = me.loops
    for li, loop in enumerate(loops):
        vi = loop.vertex_index
        r, g, b = rgb[vi]
        data[li].color = (float(r), float(g), float(b), 1.0)
    me.attributes.active_color = ca


def ensure_bake_material(obj):
    """Create emission material reading SynColor attribute; return (image, tex_node)."""
    mat = bpy.data.materials.new(name="Syn_Bake_Mat")
    mat.use_nodes = True
    nt = mat.node_tree
    nodes = nt.nodes
    links = nt.links
    for n in list(nodes):
        nodes.remove(n)

    out = nodes.new("ShaderNodeOutputMaterial")
    emit = nodes.new("ShaderNodeEmission")
    attr = nodes.new("ShaderNodeAttribute")
    attr.attribute_name = "SynColor"
    links.new(attr.outputs["Color"], emit.inputs["Color"])
    links.new(emit.outputs["Emission"], out.inputs["Surface"])

    img = bpy.data.images.new("SynTex", width=IMAGE_SIZE, height=IMAGE_SIZE,
                              alpha=False, float_buffer=False)
    tex = nodes.new("ShaderNodeTexImage")
    tex.image = img
    nodes.active = tex  # bake target

    obj.data.materials.clear()
    obj.data.materials.append(mat)
    return img, tex


def bake_to_image(obj, img, output_image_path: str) -> None:
    """Bake EMIT pass to the image and save as PNG."""
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'GPU'
    scene.render.bake.margin = 16
    scene.render.bake.target = 'IMAGE_TEXTURES'

    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.bake(type='EMIT')

    img.filepath_raw = output_image_path
    img.file_format = 'PNG'
    img.save()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    os.makedirs(os.path.join(OUTPUT_DIR, "images"), exist_ok=True)

    # ── 1. Preprocess all specimens (landmarks, canonical transforms, TPS) ──
    _ref_ct, _ref_lm_can, _ref_labels, specimens = preprocess_specimens()
    n_specimens = len(specimens)

    # ── 2. Sample coloring parameters ──
    samples = sample_space(N_SAMPLES, random_state=SEED)

    # ── 3. Deterministic random specimen assignment per sample ──
    rng = np.random.RandomState(SEED)
    specimen_indices = rng.randint(0, n_specimens, size=N_SAMPLES)

    # ── 4. Cache: import each specimen OBJ once, store obj + P_world ──
    #    Key = specimen index → (obj, P_world)
    #    We re-import when the specimen changes to keep scene clean.
    cached_specimen_idx: Optional[int] = None
    cached_obj = None
    cached_P_world: Optional[np.ndarray] = None

    csv_rows: List[dict] = []

    for j in range(N_SAMPLES):
        spec_idx = int(specimen_indices[j])
        spec = specimens[spec_idx]
        sample = samples[j]

        # ── 4a. Import specimen mesh if not already loaded ──
        if cached_specimen_idx != spec_idx:
            # Clean up previous specimen
            cleanup_scene()
            cached_obj = import_obj(spec.mesh_file)
            ensure_uv(cached_obj)
            cached_P_world = world_vertex_positions(cached_obj)
            cached_specimen_idx = spec_idx
            print(f"[sample {j}] Imported specimen {spec_idx}: "
                  f"{Path(spec.mesh_file).name} ({cached_P_world.shape[0]} verts)")

        # ── 5. Generate brushes in REFERENCE CANONICAL SPACE ──
        brushes_ref_can = generate_fishy_coloration(**sample)

        # ── 6. Warp brushes: ref canonical → specimen world ──
        brushes_world = warp_brushes(brushes_ref_can, spec.tps, spec.transform)

        # ── 7. Apply brushes to specimen world-space vertices ──
        rgb = apply_brushes(
            cached_P_world, brushes_world,
            add_noise=ADD_NOISE, noise_amount=NOISE_AMOUNT, noise_scale=NOISE_SCALE,
        )

        # ── 8. Write vertex colors, bake, save PNG ──
        write_vertex_colors(cached_obj, rgb)
        img, tex = ensure_bake_material(cached_obj)
        mesh_stem = Path(spec.mesh_file).stem
        output_path = os.path.join(OUTPUT_DIR, "images", f"{mesh_stem}_fishy_{j:06d}.png")
        bake_to_image(cached_obj, img, output_path)
        print(f"  -> {output_path}")

        # ── 9. Collect CSV row ──
        row = dict(sample)
        row["specimen_index"] = spec_idx
        row["mesh_file"] = spec.mesh_file
        row["landmark_file"] = spec.landmark_file
        row["tps_lambda"] = TPS_LAMBDA
        row["n_landmarks_used"] = spec.n_landmarks_used
        row["tps_rmse"] = spec.tps_rmse
        csv_rows.append(row)

    # ── 10. Write global CSV ──
    df = pd.DataFrame(csv_rows)
    csv_path = os.path.join(OUTPUT_DIR, "samples.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nDone. {N_SAMPLES} samples written to {OUTPUT_DIR}")
    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()