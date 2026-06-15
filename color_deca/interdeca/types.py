"""Shared immutable record types for the clean-room InterDeCA package."""

from pathlib import Path
from typing import Any, NamedTuple


class DependencyStatus(NamedTuple):
    sklearn_available: bool
    umap_available: bool
    scipy_available: bool
    skimage_available: bool


class DatasetInfo(NamedTuple):
    root: Path
    models_dir: Path
    landmarks_dir: Path
    textures_dir: Path
    models: dict[str, Path]
    landmarks: dict[str, Path]
    textures: dict[str, Path]
    matched_subjects: list[str]
    ignored_textures: list[str]


class WorkflowPaths(NamedTuple):
    output: Path
    atlas: Path
    color: Path
    aligned_lms: Path
    aligned_models: Path
    temp_lms: Path
    temp_models: Path
    resampled_models: Path
    resampled_uv: Path
    baked_textures: Path


class BlenderUvParameters(NamedTuple):
    merge_dist: float = 0.0001
    smart_angle: float = 66.0
    island_margin: float = 0.002


class BlenderBakeParameters(NamedTuple):
    bake_size: int = 2048
    bake_extrusion: float = 0.001
    bake_margin_px: int = 2
    merge_dist: float = 0.0001


class WorkflowCounts(NamedTuple):
    subjects: int
    aligned_landmarks: int
    aligned_ply_models: int
    resampled_ply_models: int
    resampled_uv_objs: int
    baked_textures: int


class AtlasTextureWorkflowConfig(NamedTuple):
    dataset_root: Path
    output_root: Path
    blender_executable: Path
    uv_parameters: BlenderUvParameters
    bake_parameters: BlenderBakeParameters
    atlas_model: Path | None = None
    atlas_landmarks: Path | None = None


class AtlasTextureWorkflowResult(NamedTuple):
    dataset_root: Path
    output_root: Path
    atlas_model: Path
    atlas_landmarks: Path
    atlas_model_ply: Path
    resampled_model_dir: Path
    resampled_uv_dir: Path
    baked_texture_dir: Path
    subjects: list[str]
    counts: WorkflowCounts
    bake_parameters: BlenderBakeParameters
    uv_parameters: BlenderUvParameters
    blender_executable: Path
    warnings: list[str]


class FolderManifest(NamedTuple):
    root: Path
    files: dict[str, str]


class FolderComparisonResult(NamedTuple):
    only_left: list[str]
    only_right: list[str]
    changed: dict[str, tuple[str, str]]


class ColorSamplingResult(NamedTuple):
    face_colors: Any
    color_space: str
    face_indices: Any | None = None


class PopulationAnalysisResult(NamedTuple):
    reduced_data: Any
    texture_names: list[str]
    method: str
    model: Any | None = None


class MeshSelectionResult(NamedTuple):
    selected_vertices: list[int]
    total_vertices: int
    selected_cells: list[int]
