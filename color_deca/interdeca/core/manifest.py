"""Deterministic manifest helpers for InterDeCA workflows."""

import json
from pathlib import Path
from typing import Mapping

from color_deca.interdeca.types import AtlasTextureWorkflowResult


def write_json_manifest(path: Path, payload: Mapping[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


class WorkflowManifestBuilder:
    """Build full-workflow manifest payloads."""

    def build_full_manifest(self, result: AtlasTextureWorkflowResult) -> dict[str, object]:
        return {
            "tier": "full",
            "dataset_root": str(result.dataset_root),
            "output_root": str(result.output_root),
            "atlas_model": str(result.atlas_model),
            "atlas_landmarks": str(result.atlas_landmarks),
            "atlas_model_ply": str(result.atlas_model_ply),
            "resampled_model_dir": str(result.resampled_model_dir),
            "resampled_uv_dir": str(result.resampled_uv_dir),
            "baked_texture_dir": str(result.baked_texture_dir),
            "subjects": result.subjects,
            "counts": result.counts._asdict(),
            "bake_size": result.bake_parameters.bake_size,
            "bake_extrusion": result.bake_parameters.bake_extrusion,
            "bake_margin_px": result.bake_parameters.bake_margin_px,
            "merge_dist": result.uv_parameters.merge_dist,
            "smart_angle": result.uv_parameters.smart_angle,
            "island_margin": result.uv_parameters.island_margin,
            "blender_executable": str(result.blender_executable),
            "warnings": result.warnings,
        }
