"""Shared NamedTuple type tests."""

from pathlib import Path

import pytest

from color_deca.interdeca.types import (
    AtlasTextureWorkflowConfig,
    BlenderBakeParameters,
    BlenderUvParameters,
    DatasetInfo,
    DependencyStatus,
    WorkflowPaths,
)


def test_named_tuple_records_retain_fields_and_are_immutable():
    status = DependencyStatus(
        sklearn_available=True,
        umap_available=False,
        scipy_available=True,
        skimage_available=False,
    )
    assert status.sklearn_available is True
    with pytest.raises(AttributeError):
        status.sklearn_available = False


def test_core_record_types_construct_with_path_values(tmp_path):
    dataset = DatasetInfo(
        root=tmp_path,
        models_dir=tmp_path / "models",
        landmarks_dir=tmp_path / "landmarks",
        textures_dir=tmp_path / "textures",
        models={"a": tmp_path / "models" / "a.obj"},
        landmarks={"a": tmp_path / "landmarks" / "a.mrk.json"},
        textures={"a": tmp_path / "textures" / "a.png"},
        matched_subjects=["a"],
        ignored_textures=[],
    )
    assert dataset.root == tmp_path
    assert dataset.matched_subjects == ["a"]

    paths = WorkflowPaths(
        output=tmp_path,
        atlas=tmp_path / "ATLAS",
        color=tmp_path / "colorAnalysis",
        aligned_lms=tmp_path / "ATLAS" / "alignedLMs",
        aligned_models=tmp_path / "ATLAS" / "alignedModels",
        temp_lms=tmp_path / "ATLAS" / "tempAlignedLMs",
        temp_models=tmp_path / "ATLAS" / "tempAlignedModels",
        resampled_models=tmp_path / "ATLAS" / "resampledModels",
        resampled_uv=tmp_path / "colorAnalysis" / "resampledOBJ_withUV",
        baked_textures=tmp_path / "colorAnalysis" / "atlasTextures",
    )
    assert paths.output == tmp_path

    config = AtlasTextureWorkflowConfig(
        dataset_root=tmp_path,
        output_root=tmp_path / "out",
        blender_executable=Path("/bin/blender"),
        uv_parameters=BlenderUvParameters(),
        bake_parameters=BlenderBakeParameters(),
    )
    assert config.uv_parameters.smart_angle == 66.0
    assert config.bake_parameters.bake_size == 2048
