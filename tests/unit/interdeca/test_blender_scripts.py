"""Blender script generation tests."""

from pathlib import Path

from color_deca.interdeca.core.blender_scripts import BlenderScriptFactory
from color_deca.interdeca.types import BlenderBakeParameters, BlenderUvParameters


def test_prepare_atlas_script_contains_legacy_uv_operations():
    script = BlenderScriptFactory().build_prepare_atlas_script()

    assert "bpy.ops.wm.read_factory_settings(use_empty=True)" in script
    assert "forward_axis='Y', up_axis='Z'" in script
    assert "bpy.ops.uv.smart_project" in script
    assert "export_triangulated_mesh=True" in script


def test_bake_script_contains_selected_to_active_bake_settings():
    script = BlenderScriptFactory().build_bake_texture_script()

    assert "b.use_selected_to_active = True" in script
    assert "b.cage_extrusion = extru" in script
    assert "b.margin = margin" in script
    assert "bpy.ops.object.bake(type='DIFFUSE')" in script


def test_script_argument_builders_are_deterministic():
    factory = BlenderScriptFactory()

    assert factory.prepare_atlas_args(
        Path("in.obj"),
        Path("out.obj"),
        BlenderUvParameters(merge_dist=0.1, smart_angle=45.0, island_margin=0.2),
    ) == ["in.obj", "out.obj", "0.1", "45.0", "0.2"]

    assert factory.bake_texture_args(
        Path("src.ply"),
        Path("target.obj"),
        Path("in.png"),
        Path("out.png"),
        BlenderBakeParameters(bake_size=256, bake_extrusion=0.5, bake_margin_px=3, merge_dist=0.1),
    ) == ["src.ply", "target.obj", "in.png", "out.png", "256", "0.5", "3", "0.1"]
