"""Regression tests for atlas anchor-frame and sequence preflight metrics."""

import ast
import math
from pathlib import Path

import numpy as np


SOURCE = Path(__file__).parent / "color_deca" / "deca3" / "InterDeCA.py"


def _load_helpers():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    wanted = {
        "_growth_axis_quality",
        "_anchor_frame",
        "_anatomical_anchor_frame",
        "_select_generic_anchor_indices",
        "_select_generic_side_index",
        "_robust_modified_z",
    }
    nodes = [node for node in tree.body
             if isinstance(node, ast.FunctionDef) and node.name in wanted]
    namespace = {"np": np, "math": math}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace


def _proper_rotation(seed=7):
    matrix = np.random.default_rng(seed).normal(size=(3, 3))
    rotation, _ = np.linalg.qr(matrix)
    if np.linalg.det(rotation) < 0:
        rotation[:, -1] *= -1
    return rotation


def test_anchor_frame_is_pose_and_scale_invariant():
    helper = _load_helpers()["_anatomical_anchor_frame"]
    points = np.array([
        [0.0, 0.0, 0.0],   # beak
        [2.0, 3.0, 0.0],   # anterior hinge
        [-2.0, 3.0, 0.0],  # posterior hinge
        [0.4, 1.2, 0.8],
        [-0.7, 2.1, 0.3],
    ])
    expected = helper(points, 0, 1, 2)

    rotation = _proper_rotation()
    transformed = (points @ rotation) * 4.2 + np.array([11.0, -5.0, 8.0])
    actual = helper(transformed, 0, 1, 2)

    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_generic_anchor_selection_works_with_fish_style_labels():
    helpers = _load_helpers()
    select_anchors = helpers["_select_generic_anchor_indices"]
    select_side = helpers["_select_generic_side_index"]
    frame = helpers["_anchor_frame"]
    labels = ["snout", "caudal_base", "dorsal_fin", "left_eye", "jaw", "operculum"]
    points = np.array([
        [-4.0, 0.0, 0.0],
        [4.0, 0.0, 0.0],
        [0.0, 3.0, 0.0],
        [0.0, 0.5, 2.0],
        [-1.0, -0.5, 0.2],
        [1.5, 0.8, -0.4],
    ])

    configurations = []
    for seed, scale in enumerate((0.7, 1.0, 2.3, 4.1)):
        rotation = _proper_rotation(seed + 20)
        translation = np.array([seed * 3.0, -seed, seed * 0.5])
        configurations.append(points @ rotation * scale + translation)

    anchors = select_anchors(configurations)
    assert tuple(labels[index] for index in anchors) == (
        "dorsal_fin", "caudal_base", "snout")

    normalized_frames = [frame(configuration, *anchors)
                         for configuration in configurations]
    for actual in normalized_frames[1:]:
        np.testing.assert_allclose(actual, normalized_frames[0], atol=1e-12)
    side_index = select_side(normalized_frames, anchors)
    assert labels[side_index] == "left_eye"


def test_generic_anchor_selection_rejects_collinear_landmarks():
    helper = _load_helpers()["_select_generic_anchor_indices"]
    points = np.column_stack((np.arange(5, dtype=float), np.zeros((5, 2))))
    try:
        helper(points)
    except ValueError as error:
        assert "non-collinear" in str(error)
    else:
        raise AssertionError("Collinear landmarks must not define an anchor frame")


def test_growth_axis_quality_detects_scrambled_order_and_reversal():
    helper = _load_helpers()["_growth_axis_quality"]
    points = np.zeros((9, 3), dtype=float)
    points[0] = [-1.0, 0.0, 0.0]  # beak
    points[1:9, 0] = np.arange(8, dtype=float)

    good = helper(points, 0, range(1, 9))
    assert good["closest_to_beak"] == 0
    assert good["spacing_cv"] == 0.0
    assert good["min_turn_cosine"] == 1.0
    assert good["path_to_direct"] == 1.0

    scrambled = points.copy()
    scrambled[1:9] = points[[3, 2, 4, 1, 5, 6, 7, 8]]
    bad = helper(scrambled, 0, range(1, 9))
    assert bad["closest_to_beak"] != 0
    assert bad["min_turn_cosine"] <= 0.0
    assert bad["spacing_cv"] > 0.15


def test_robust_zscore_flags_extreme_anchor_geometry():
    helper = _load_helpers()["_robust_modified_z"]
    values = np.array([1.00, 1.02, 0.99, 1.01, 1.00, 4.00])
    scores = np.abs(helper(values))
    assert scores[-1] > 6.0
    assert np.max(scores[:-1]) < 3.0


def test_generate_new_atlas_invokes_preflight():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    generate = next(node for node in ast.walk(tree)
                    if isinstance(node, ast.FunctionDef)
                    and node.name == "generateNewAtlas")
    calls = [node for node in ast.walk(generate)
             if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Attribute)
             and node.func.attr == "validateAtlasDataset"]
    assert calls, "Atlas generation must run dataset preflight before TPS"


def test_preflight_has_generic_anchor_fallback():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    validate = next(node for node in ast.walk(tree)
                    if isinstance(node, ast.FunctionDef)
                    and node.name == "validateAtlasDataset")
    calls = [node for node in ast.walk(validate)
             if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name)
             and node.func.id == "_select_generic_anchor_indices"]
    assert calls, "Datasets without mussel labels must receive automatic anchors"
