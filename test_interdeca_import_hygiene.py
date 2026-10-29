"""Regression tests for dataset selection, import order, and mesh scale.

InterDeCA normally runs inside 3D Slicer, so importing the complete module in a
normal Python test process would require Slicer's VTK/Qt bindings. These tests
load the relevant helpers and methods from the source, using a small data-group
stand-in to check the import contract without running scientific algorithms.
"""

import ast
import logging
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
        "_remove_temporary_scene_nodes",
        "LANDMARK_FILE_EXTENSIONS",
        "MODEL_FILE_EXTENSIONS",
    }
    nodes = [node for node in tree.body
             if (isinstance(node, ast.FunctionDef) and node.name in wanted)
             or (isinstance(node, ast.Assign) and any(
                 isinstance(target, ast.Name) and target.id in wanted
                 for target in node.targets))]
    namespace = {"os": os, "Path": Path, "logging": logging}
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
    for name in ("importLandmarks", "importMeshes", "_dataset_file_map", "_load_model_with_cs"):
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


@pytest.mark.parametrize("failure_stage", ["polydata", "group"])
def test_mesh_import_removes_loaded_node_when_collection_fails(monkeypatch, failure_stage):
    monkeypatch.setattr(os, "listdir", lambda _: ["sample.obj"])
    removed_nodes = []
    logic = _load_logic(removed_nodes)
    failure = RuntimeError("cannot collect mesh")

    def fail(*args):
        raise failure

    node = SimpleNamespace(GetPolyData=fail if failure_stage == "polydata" else lambda: "mesh")
    logic._load_model_with_cs = lambda *args: node
    if failure_stage == "group":
        monkeypatch.setattr(_DataGroup, "AddInputData", fail)

    with pytest.raises(RuntimeError) as caught:
        logic.importMeshes("/dataset", ["obj"])

    assert caught.value is failure
    assert removed_nodes == [node]


@pytest.mark.parametrize("failure_stage", ["coordinate", "model", "filename", "read", "rejected", None])
@pytest.mark.parametrize("legacy_storage", [False, True])
def test_model_loader_cleans_up_scene_nodes(monkeypatch, failure_stage, legacy_storage):
    removed_nodes = []
    logic = _load_logic(removed_nodes)
    calls = []
    failure = RuntimeError("storage operation failed")
    model = object()

    def record(stage, value):
        calls.append((stage, value))
        if failure_stage == stage:
            raise failure

    storage = SimpleNamespace(
        SetFileName=lambda path: record("filename", path),
        ReadData=lambda node: (record("read", node), failure_stage != "rejected")[1],
        SetCoordinateSystem=lambda value: record("coordinate", value),
    )
    if not legacy_storage:
        storage.SetCoordinateSystemToLPS = lambda: record("coordinate", "LPS")
        storage.SetCoordinateSystemToRAS = lambda: record("coordinate", "RAS")

    def add_node(class_name):
        if class_name == "vtkMRMLModelStorageNode":
            return storage
        assert class_name == "vtkMRMLModelNode"
        record("model", model)
        return model

    scene = logic._load_model_with_cs.__func__.__globals__["slicer"].mrmlScene
    monkeypatch.setattr(scene, "AddNewNodeByClass", add_node, raising=False)

    if failure_stage is None:
        assert logic._load_model_with_cs("/dataset/sample.obj", "lps") is model
    else:
        with pytest.raises(RuntimeError) as caught:
            logic._load_model_with_cs("/dataset/sample.obj", "lps")
        if failure_stage == "rejected":
            assert str(caught.value) == "Failed to read model: /dataset/sample.obj"
        else:
            assert caught.value is failure

    assert calls[0] == ("coordinate", 1 if legacy_storage else "LPS")
    if failure_stage in ("filename", "read", "rejected"):
        assert removed_nodes == [storage, model]
    else:
        assert removed_nodes == [storage]


def test_cleanup_keeps_original_error_and_attempts_remaining_nodes(caplog):
    logic = _load_logic([])
    namespace = logic._load_model_with_cs.__func__.__globals__
    attempted = []

    def remove(node):
        attempted.append(node)
        if node == "storage":
            raise RuntimeError("cleanup failed")

    namespace["slicer"].mrmlScene.RemoveNode = remove
    original_error = ValueError("model import failed")
    with pytest.raises(ValueError) as caught:
        try:
            raise original_error
        finally:
            namespace["_remove_temporary_scene_nodes"]("storage", "model", suppress_errors=True)

    assert caught.value is original_error
    assert attempted == ["storage", "model"]
    assert "cleanup failed" in caplog.text


def test_cleanup_failure_without_import_error_is_reported():
    logic = _load_logic([])
    namespace = logic._load_model_with_cs.__func__.__globals__

    def remove(node):
        raise RuntimeError("cleanup failed")

    namespace["slicer"].mrmlScene.RemoveNode = remove
    with pytest.raises(RuntimeError, match="cleanup failed"):
        namespace["_remove_temporary_scene_nodes"]("storage")


@pytest.mark.parametrize("model_cleanup_fails", [False, True])
def test_model_loader_releases_unreturned_model_when_storage_cleanup_fails(model_cleanup_fails):
    removed_nodes = []
    logic = _load_logic(removed_nodes)
    scene = logic._load_model_with_cs.__func__.__globals__["slicer"].mrmlScene
    model = object()
    storage = SimpleNamespace(
        SetCoordinateSystemToRAS=lambda: None,
        SetFileName=lambda path: None,
        ReadData=lambda node: True,
    )
    scene.AddNewNodeByClass = lambda name: storage if name == "vtkMRMLModelStorageNode" else model
    storage_error = RuntimeError("storage cleanup failed")

    def remove(node):
        removed_nodes.append(node)
        if node is storage:
            raise storage_error
        if model_cleanup_fails:
            raise RuntimeError("model cleanup also failed")

    scene.RemoveNode = remove
    with pytest.raises(RuntimeError) as caught:
        logic._load_model_with_cs("/dataset/sample.obj")

    assert caught.value is storage_error
    assert removed_nodes == [storage, model]


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
