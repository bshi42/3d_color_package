"""Workflow path tests."""

from color_deca.interdeca.core.paths import WorkflowPathBuilder, build_workflow_paths, ensure_workflow_paths


def test_build_workflow_paths_uses_exact_legacy_names(tmp_path):
    paths = build_workflow_paths(tmp_path / "full")

    assert paths.aligned_lms.relative_to(paths.output).as_posix() == "ATLAS/alignedLMs"
    assert paths.aligned_models.relative_to(paths.output).as_posix() == "ATLAS/alignedModels"
    assert paths.temp_lms.relative_to(paths.output).as_posix() == "ATLAS/tempAlignedLMs"
    assert paths.temp_models.relative_to(paths.output).as_posix() == "ATLAS/tempAlignedModels"
    assert paths.resampled_models.relative_to(paths.output).as_posix() == "ATLAS/resampledModels"
    assert paths.resampled_uv.relative_to(paths.output).as_posix() == "colorAnalysis/resampledOBJ_withUV"
    assert paths.baked_textures.relative_to(paths.output).as_posix() == "colorAnalysis/atlasTextures"


def test_ensure_workflow_paths_creates_expected_directories(tmp_path):
    paths = ensure_workflow_paths(build_workflow_paths(tmp_path / "full"))

    for path in paths:
        assert path.is_dir()


def test_workflow_path_builder_ensure_returns_paths(tmp_path):
    paths = WorkflowPathBuilder().ensure(tmp_path / "full")

    assert paths.output.is_dir()
    assert paths.baked_textures.is_dir()
