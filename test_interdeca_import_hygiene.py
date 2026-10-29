"""Regression tests for dataset selection, import order, and mesh scale.

InterDeCA normally runs inside 3D Slicer, so importing the complete module in a
normal Python test process would require Slicer's VTK/Qt bindings. These tests
load the relevant helpers and methods from the source, using a small data-group
stand-in to check the import contract without running scientific algorithms.
"""

import ast
import os
from pathlib import Path
from types import SimpleNamespace

import pytest


SOURCE = Path(__file__).parent / "color_deca" / "deca3" / "InterDeCA.py"


def _load_helpers():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    wanted = {
        "_is_visible_dataset_file",
        "_dataset_file_paths",
        "_relative_landmark_surface_distance",
        "LANDMARK_FILE_EXTENSIONS",
        "MODEL_FILE_EXTENSIONS",
    }
    nodes = [node for node in tree.body
             if (isinstance(node, ast.FunctionDef) and node.name in wanted)
             or (isinstance(node, ast.Assign) and any(
                 isinstance(target, ast.Name) and target.id in wanted
                 for target in node.targets))]
    namespace = {"os": os, "Path": Path}
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


class _DataGroup:
    def __init__(self):
        self.inputs = []
        self.updated = False

    def AddInputData(self, data):
        self.inputs.append(data)

    def Update(self):
        self.updated = True

    def GetOutput(self):
        assert self.updated
        return self.inputs


def _load_logic(removed_nodes):
    namespace = _load_helpers()
    namespace.update({
        "vtk": SimpleNamespace(vtkMultiBlockDataGroupFilter=_DataGroup),
        "slicer": SimpleNamespace(mrmlScene=SimpleNamespace(
            RemoveNode=removed_nodes.append)),
    })
    methods = {}
    for name in ("importLandmarks", "importMeshes", "_dataset_file_map"):
        method = _class_method(name)
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(SOURCE), "exec"), namespace)
        methods[name] = namespace[name]
    return type("InterDeCALogic", (), methods)()


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


def test_dataset_file_paths_preserve_priority_case_and_compound_names(monkeypatch):
    helpers = _load_helpers()
    monkeypatch.setattr(os, "listdir", lambda _: [
        "beta.ply", "alpha.VTK", "alpha.obj", "beta.STL", "beta.obj",
        "alpha.PLY", "beta.obj.rcInfo", "._hidden.obj", ".DS_Store",
        "notes.txt", "specimen.with.dots.VTP", "Case.obj", "case.obj",
    ])

    paths = helpers["_dataset_file_paths"]("/dataset", helpers["MODEL_FILE_EXTENSIONS"])

    assert paths == {
        "beta": "/dataset/beta.obj",
        "alpha": "/dataset/alpha.obj",
        "specimen.with.dots": "/dataset/specimen.with.dots.VTP",
        "Case": "/dataset/Case.obj",
        "case": "/dataset/case.obj",
    }
    assert list(paths) == ["beta", "alpha", "specimen.with.dots", "Case", "case"]


def test_dataset_file_paths_keep_first_equal_priority_match(monkeypatch):
    helpers = _load_helpers()
    monkeypatch.setattr(os, "listdir", lambda _: ["sample.OBJ", "sample.obj"])

    assert helpers["_dataset_file_paths"]("/dataset", helpers["MODEL_FILE_EXTENSIONS"]) == {
        "sample": "/dataset/sample.OBJ",
    }


def test_landmark_suffix_stripping_retains_each_callers_contract(monkeypatch):
    helpers = _load_helpers()
    monkeypatch.setattr(os, "listdir", lambda _: [
        "sample.fcsv", "sample.json", "sample.MRK.JSON", "layers.json.fcsv",
        "._ignored.mrk.json", "notes.txt",
    ])
    select = helpers["_dataset_file_paths"]
    extensions = helpers["LANDMARK_FILE_EXTENSIONS"]

    assert select("/dataset", extensions) == {
        "sample": "/dataset/sample.MRK.JSON",
        "layers.json": "/dataset/layers.json.fcsv",
    }
    assert select("/dataset", extensions, strip_suffixes=(".mrk", ".json", ".fcsv")) == {
        "sample": "/dataset/sample.MRK.JSON",
        "layers": "/dataset/layers.json.fcsv",
    }


def test_landmark_import_loads_preferred_files_in_sorted_subject_order(monkeypatch):
    monkeypatch.setattr(os, "listdir", lambda _: [
        "beta.fcsv", "._hidden.mrk.json", "alpha.json", "beta.MRK.JSON",
        "layers.json.fcsv", "notes.txt",
    ])
    logic = _load_logic([])
    logic.fiducialNodeToPolyData = lambda path: path

    names, group = logic.importLandmarks("/dataset")

    assert names == ["alpha", "beta", "layers"]
    assert group == ["/dataset/alpha.json", "/dataset/beta.MRK.JSON", "/dataset/layers.json.fcsv"]


@pytest.mark.parametrize("restrict_to,coordinate_system,expected_names", [
    (None, "RAS", ["alpha", "beta"]),
    (["beta", "missing", "alpha", "beta"], "LPS", ["beta", "alpha", "beta"]),
    ([], "RAS", []),
])
def test_mesh_import_preserves_order_coordinates_and_cleanup(
        monkeypatch, restrict_to, coordinate_system, expected_names):
    monkeypatch.setattr(os, "listdir", lambda _: [
        "beta.ply", "._hidden.obj", "alpha.obj", "beta.OBJ", "notes.txt",
    ])
    removed_nodes = []
    loaded = []
    logic = _load_logic(removed_nodes)

    def load_model(path, system):
        loaded.append((path, system))
        return SimpleNamespace(GetPolyData=lambda: path)

    logic._load_model_with_cs = load_model
    names, group = logic.importMeshes(
        "/dataset", ["ply"], restrict_to=restrict_to, coordinateSystem=coordinate_system)

    expected_paths = [f"/dataset/{name}.{'OBJ' if name == 'beta' else 'obj'}"
                      for name in expected_names]
    assert names == expected_names
    assert group == expected_paths
    assert loaded == [(path, coordinate_system) for path in expected_paths]
    assert [node.GetPolyData() for node in removed_nodes] == expected_paths


def test_preflight_file_map_uses_matching_suffix_and_rejects_unknown_kinds(monkeypatch):
    monkeypatch.setattr(os, "listdir", lambda _: [
        "sample.fcsv", "sample.mrk.json", "layers.json.fcsv", "sample.ply", "sample.obj",
    ])
    logic = _load_logic([])

    assert logic._dataset_file_map("/dataset", "landmark") == {
        "sample": "/dataset/sample.mrk.json",
        "layers.json": "/dataset/layers.json.fcsv",
    }
    assert logic._dataset_file_map("/dataset", "model") == {"sample": "/dataset/sample.obj"}
    with pytest.raises(ValueError, match="Unsupported dataset file kind: texture"):
        logic._dataset_file_map("/dataset", "texture")


def test_landmark_surface_distance_is_normalized_to_mesh_diagonal():
    helpers = _load_helpers()
    relative_distance = helpers["_relative_landmark_surface_distance"]

    assert relative_distance(0.2, 10.0) == 0.02
    assert relative_distance(0.21, 10.0) > 0.02
    assert relative_distance(0.0, 0.0) == 0.0
    assert relative_distance(1.0, 0.0) == float("inf")
