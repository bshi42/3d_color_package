"""Regression tests for dataset-file filtering and mesh-scale validation.

InterDeCA normally runs inside 3D Slicer, so importing the complete module in a
normal Python test process would require Slicer's VTK/Qt bindings.  These
tests load the two dependency-free helpers from the source file instead.
"""

import ast
from pathlib import Path


SOURCE = Path(__file__).parent / "color_deca" / "deca3" / "InterDeCA.py"


def _load_helpers():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    wanted = {
        "_is_visible_dataset_file",
        "_relative_landmark_surface_distance",
    }
    nodes = [node for node in tree.body
             if isinstance(node, ast.FunctionDef) and node.name in wanted]
    namespace = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace


def _class_method(name):
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == name:
                    return child
    raise AssertionError(f"Could not find {name} in {SOURCE}")


def test_appledouble_entries_are_not_dataset_files():
    helpers = _load_helpers()
    is_visible = helpers["_is_visible_dataset_file"]

    filenames = [
        "UF_IZ_180660.obj",
        "UF_IZ_180660.mrk.json",
        "._UF_IZ_180660.obj",
        "._UF_IZ_180660.mrk.json",
        ".DS_Store",
    ]

    assert [name for name in filenames if is_visible(name)] == filenames[:2]
    assert not is_visible("")

    for importer in ("importLandmarks", "importMeshes"):
        calls = [node for node in ast.walk(_class_method(importer))
                 if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Name)
                 and node.func.id == "_is_visible_dataset_file"]
        assert calls, f"{importer} must filter dot-prefixed dataset files"


def test_landmark_surface_distance_is_normalized_to_mesh_diagonal():
    helpers = _load_helpers()
    relative_distance = helpers["_relative_landmark_surface_distance"]

    assert relative_distance(0.2, 10.0) == 0.02
    assert relative_distance(0.21, 10.0) > 0.02
    assert relative_distance(0.0, 0.0) == 0.0
    assert relative_distance(1.0, 0.0) == float("inf")
