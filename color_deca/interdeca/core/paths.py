"""Workflow output path construction for InterDeCA."""

from pathlib import Path

from color_deca.interdeca.types import WorkflowPaths


def build_workflow_paths(output_root: Path) -> WorkflowPaths:
    output = Path(output_root).resolve()
    atlas = output / "ATLAS"
    color = output / "colorAnalysis"
    return WorkflowPaths(
        output=output,
        atlas=atlas,
        color=color,
        aligned_lms=atlas / "alignedLMs",
        aligned_models=atlas / "alignedModels",
        temp_lms=atlas / "tempAlignedLMs",
        temp_models=atlas / "tempAlignedModels",
        resampled_models=atlas / "resampledModels",
        resampled_uv=color / "resampledOBJ_withUV",
        baked_textures=color / "atlasTextures",
    )


def ensure_workflow_paths(paths: WorkflowPaths) -> WorkflowPaths:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)
    return paths


class WorkflowPathBuilder:
    """Build and optionally create the standard InterDeCA output tree."""

    def build(self, output_root: Path) -> WorkflowPaths:
        return build_workflow_paths(output_root)

    def ensure(self, output_root: Path) -> WorkflowPaths:
        return ensure_workflow_paths(self.build(output_root))
