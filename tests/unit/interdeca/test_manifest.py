"""Workflow manifest tests."""

import json
from pathlib import Path

from color_deca.interdeca.core.manifest import WorkflowManifestBuilder, write_json_manifest
from color_deca.interdeca.types import (
    AtlasTextureWorkflowResult,
    BlenderBakeParameters,
    BlenderUvParameters,
    WorkflowCounts,
)


def test_workflow_manifest_builder_outputs_expected_keys(tmp_path):
    result = AtlasTextureWorkflowResult(
        dataset_root=tmp_path / "dataset",
        output_root=tmp_path / "out",
        atlas_model=tmp_path / "out" / "colorAnalysis" / "atlasModelUV.obj",
        atlas_landmarks=tmp_path / "out" / "colorAnalysis" / "atlasLM.mrk.json",
        atlas_model_ply=tmp_path / "out" / "colorAnalysis" / "atlasModel.ply",
        resampled_model_dir=tmp_path / "out" / "ATLAS" / "resampledModels",
        resampled_uv_dir=tmp_path / "out" / "colorAnalysis" / "resampledOBJ_withUV",
        baked_texture_dir=tmp_path / "out" / "colorAnalysis" / "atlasTextures",
        subjects=["a", "b"],
        counts=WorkflowCounts(
            subjects=2,
            aligned_landmarks=2,
            aligned_ply_models=2,
            resampled_ply_models=2,
            resampled_uv_objs=2,
            baked_textures=2,
        ),
        bake_parameters=BlenderBakeParameters(),
        uv_parameters=BlenderUvParameters(),
        blender_executable=Path("/bin/blender"),
        warnings=[],
    )

    manifest = WorkflowManifestBuilder().build_full_manifest(result)

    assert manifest["tier"] == "full"
    assert manifest["subjects"] == ["a", "b"]
    assert manifest["counts"]["subjects"] == 2
    assert manifest["bake_size"] == 2048


def test_write_json_manifest_is_sorted_and_indented(tmp_path):
    path = tmp_path / "manifest.json"

    write_json_manifest(path, {"b": 1, "a": 2})

    assert path.read_text(encoding="utf-8") == '{\n  "a": 2,\n  "b": 1\n}'
    assert json.loads(path.read_text(encoding="utf-8")) == {"a": 2, "b": 1}
