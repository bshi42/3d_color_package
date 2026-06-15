#!/usr/bin/env python3
"""Single-file CLI and Slicer-side driver for InterDeCA E2E tests."""

import importlib.util
import json
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import NamedTuple

import click

try:
    import slicer
except ModuleNotFoundError:
    slicer = None

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
END_TO_END_DIR = THIS_DIR.parent / "end_to_end"
if str(END_TO_END_DIR) not in sys.path:
    sys.path.insert(0, str(END_TO_END_DIR))

from util import (
    ApplicationPaths,
    check,
    count_files,
    discover_dataset,
    landmark_subject_id,
    log,
    slicer_env,
)


class ProgramArgs(NamedTuple):
    repo_root: Path
    dataset: Path
    output: Path
    blender_executable: Path
    atlas_model: Path | None
    atlas_landmarks: Path | None
    bake_size: int
    bake_extrusion: float
    bake_margin_px: int
    merge_dist: float
    smart_angle: float
    island_margin: float


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = REPO_ROOT / "data" / "All_Clams"
DEFAULT_SLICER = ApplicationPaths.get_slicer()
DEFAULT_BLENDER = ApplicationPaths.get_blender()


def preamble(repo_root):
    repo_root = Path(repo_root).resolve()
    paths = [
        repo_root,
        repo_root / "color_deca",
        repo_root / "ATLAS" / "BUILDER",
        repo_root / "ATLAS" / "PREDICT",
        repo_root / "ATLAS" / "DATABASE",
        repo_root / "ATLAS" / "SEGMENTATION",
    ]
    for path in reversed(paths):
        path_str = str(path)
        if path.is_dir() and path_str not in sys.path:
            sys.path.insert(0, path_str)

    from color_deca.interdeca import facade as InterDeCA

    return InterDeCA


def require_slicer():
    check(slicer is not None, "Slicer-side E2E code must be run inside 3D Slicer")


def require_slicer_dependencies():
    require_slicer()


def clear_scene():
    require_slicer()
    try:
        slicer.mrmlScene.Clear(0)
    except TypeError:
        slicer.mrmlScene.Clear()


def load_model(logic, model_path):
    path = str(model_path)
    if hasattr(logic, "_load_model_with_cs"):
        node = logic._load_model_with_cs(path, "RAS")
    else:
        node = slicer.util.loadModel(path)
    check(node is not None, f"Could not load model: {path}")
    node.CreateDefaultDisplayNodes()
    polydata = node.GetPolyData()
    check(polydata is not None, f"Loaded model has no polydata: {path}")
    check(polydata.GetNumberOfPoints() > 0, f"Loaded model has no points: {path}")
    check(polydata.GetNumberOfCells() > 0, f"Loaded model has no cells: {path}")
    return node


def assert_all_clams_contract(dataset):
    if dataset["root"].name != "All_Clams":
        return
    check(len(dataset["models"]) == 32, "All_Clams should have 32 model subjects")
    check(len(dataset["landmarks"]) == 32, "All_Clams should have 32 landmark subjects")
    check(len(dataset["textures"]) == 33, "All_Clams should have 33 texture subjects")
    check(len(dataset["matched_subjects"]) == 32, "All_Clams should have 32 matched subjects")
    check(
        "UF_IZ_439322" in dataset["ignored_textures"],
        "All_Clams extra texture UF_IZ_439322 should be ignored",
    )


def assert_dependency_guardrail(InterDeCA):
    sklearn_installed = importlib.util.find_spec("sklearn") is not None
    if sklearn_installed:
        check(
            InterDeCA.SKLEARN_AVAILABLE,
            "sklearn is importable but InterDeCA.SKLEARN_AVAILABLE is False",
        )


def prepare_full_output(output_root):
    output_root.mkdir(parents=True, exist_ok=True)
    atlas_dir = output_root / "ATLAS"
    color_dir = output_root / "colorAnalysis"
    paths = {
        "output": output_root,
        "atlas": atlas_dir,
        "color": color_dir,
        "aligned_lms": atlas_dir / "alignedLMs",
        "aligned_models": atlas_dir / "alignedModels",
        "temp_lms": atlas_dir / "tempAlignedLMs",
        "temp_models": atlas_dir / "tempAlignedModels",
        "resampled_models": atlas_dir / "resampledModels",
        "resampled_uv": color_dir / "resampledOBJ_withUV",
        "baked_textures": color_dir / "atlasTextures",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def generate_or_load_atlas(args, logic, dataset, paths):
    if args.atlas_model and args.atlas_landmarks:
        log("Loading provided atlas model and landmarks")
        return load_model(logic, Path(args.atlas_model)), slicer.util.loadMarkups(str(args.atlas_landmarks))

    log("Generating atlas from matched full-data fixtures")
    closest_subject_path = logic.getClosestToMeanPath(str(dataset["landmarks_dir"]))
    check(closest_subject_path is not None, "Could not determine closest subject to mean")
    closest_subject = landmark_subject_id(Path(closest_subject_path))
    log(f"Closest subject to mean: {closest_subject}")

    base_lms = logic.getLandmarkFileByID(str(dataset["landmarks_dir"]), closest_subject)
    base_model = logic.getModelFileByID(str(dataset["models_dir"]), closest_subject)
    check(base_lms is not None, f"Could not load base landmarks for {closest_subject}")
    check(base_model is not None, f"Could not load base model for {closest_subject}")

    logic.runAlign(
        base_model,
        base_lms,
        str(dataset["models_dir"]),
        str(dataset["landmarks_dir"]),
        str(paths["temp_models"]),
        str(paths["temp_lms"]),
        False,
    )
    atlas_model, atlas_lms = logic.runMean(str(paths["temp_lms"]), str(paths["temp_models"]))
    check(atlas_model is not None and atlas_lms is not None, "Atlas generation returned None")
    return atlas_model, atlas_lms


def normalize_atlas_landmark_display(atlas_lms):
    atlas_lms.CreateDefaultDisplayNodes()
    display_node = atlas_lms.GetDisplayNode()
    if display_node:
        display_node.SetSelectedColor(1.0, 0.5000076295109483, 0.5000076295109483)


def assert_model_point_count(logic, atlas_model_node, obj_paths):
    atlas_points = atlas_model_node.GetPolyData().GetNumberOfPoints()
    for obj_path in sorted(obj_paths):
        node = load_model(logic, obj_path)
        try:
            points = node.GetPolyData().GetNumberOfPoints()
            check(points == atlas_points, f"Point-count mismatch for {obj_path}: {points} != {atlas_points}")
        finally:
            slicer.mrmlScene.RemoveNode(node)


def run_full(args, InterDeCA, dataset):
    clear_scene()
    log("Running full tier")
    assert_dependency_guardrail(InterDeCA)
    assert_all_clams_contract(dataset)
    logic = InterDeCA.InterDeCALogic()
    output_root = Path(args.output).resolve()
    paths = prepare_full_output(output_root)
    subjects = dataset["matched_subjects"]

    blender = Path(args.blender_executable).expanduser()
    check(blender.is_file(), f"Blender executable not found: {blender}")

    atlas_model, atlas_lms = generate_or_load_atlas(args, logic, dataset, paths)
    check(atlas_lms is not None, "Atlas landmarks are None")

    atlas_preuv_obj = paths["atlas"] / "atlasModel_preUV.obj"
    atlas_uv_obj = paths["color"] / "atlasModelUV.obj"
    logic._save_model_with_cs(atlas_model, str(atlas_preuv_obj), "RAS")

    log("Preparing atlas UVs with Blender")
    logic.blender_prepare_atlas(
        str(blender),
        str(atlas_preuv_obj),
        str(atlas_uv_obj),
        merge_dist=args.merge_dist,
        smart_angle=args.smart_angle,
        island_margin=args.island_margin,
    )
    atlas_uv_model = load_model(logic, atlas_uv_obj)
    tcoords = atlas_uv_model.GetPolyData().GetPointData().GetTCoords()
    check(tcoords is not None and tcoords.GetNumberOfTuples() > 0, "UV atlas has no texture coordinates")

    atlas_lm_path = paths["color"] / "atlasLM.mrk.json"
    atlas_ply_path = paths["color"] / "atlasModel.ply"
    normalize_atlas_landmark_display(atlas_lms)
    check(slicer.util.saveNode(atlas_lms, str(atlas_lm_path)), f"Could not save atlas landmarks: {atlas_lm_path}")
    logic._save_model_with_cs(atlas_uv_model, str(atlas_ply_path), "RAS")

    log("Running rigid alignment to atlas")
    logic.runAlign(
        atlas_uv_model,
        atlas_lms,
        str(dataset["models_dir"]),
        str(dataset["landmarks_dir"]),
        str(paths["aligned_models"]),
        str(paths["aligned_lms"]),
        False,
    )

    log("Running dense correspondence and resampling")
    logic.runDCAlign(
        str(atlas_uv_obj),
        str(atlas_lm_path),
        str(paths["aligned_models"]),
        str(paths["aligned_lms"]),
        str(paths["output"]),
        False,
        atlas_uv_template_obj=str(atlas_uv_obj),
    )

    log("Baking atlas-space textures with Blender")
    baked = logic.blender_bake_all(
        blender_exe=str(blender),
        alignedDir=str(paths["aligned_models"]),
        resampledUVDir=str(paths["resampled_uv"]),
        texturesDir=str(dataset["textures_dir"]),
        outDir=str(paths["baked_textures"]),
        bake_size=int(args.bake_size),
        bake_extrusion=float(args.bake_extrusion),
        bake_margin_px=int(args.bake_margin_px),
        merge_dist=float(args.merge_dist),
    )
    logic._calculate_average_texture(str(paths["baked_textures"]))

    expected = len(subjects)
    counts = {
        "subjects": expected,
        "aligned_landmarks": count_files(paths["aligned_lms"], ".mrk.json"),
        "aligned_ply_models": count_files(paths["aligned_models"], ".ply"),
        "resampled_ply_models": count_files(paths["resampled_models"], ".ply"),
        "resampled_uv_objs": count_files(paths["resampled_uv"], ".obj"),
        "baked_textures": len([p for p in paths["baked_textures"].glob("*.png") if p.name != "average_texture.png"]),
    }
    check(counts["aligned_landmarks"] == expected, "Aligned landmark count mismatch")
    check(counts["aligned_ply_models"] == expected, "Aligned PLY model count mismatch")
    check(counts["resampled_ply_models"] == expected, "Resampled PLY model count mismatch")
    check(counts["resampled_uv_objs"] == expected, "Resampled UV OBJ count mismatch")
    check(counts["baked_textures"] == expected, "Baked texture count mismatch")
    check((paths["baked_textures"] / "average_texture.png").is_file(), "average_texture.png was not created")
    check(len(baked) == expected, "Blender bake return count mismatch")
    assert_model_point_count(logic, atlas_uv_model, paths["resampled_uv"].glob("*.obj"))

    manifest = {
        "tier": "full",
        "dataset_root": str(dataset["root"]),
        "output_root": str(paths["output"]),
        "atlas_model": str(atlas_uv_obj),
        "atlas_landmarks": str(atlas_lm_path),
        "atlas_model_ply": str(atlas_ply_path),
        "resampled_model_dir": str(paths["resampled_models"]),
        "resampled_uv_dir": str(paths["resampled_uv"]),
        "baked_texture_dir": str(paths["baked_textures"]),
        "subjects": subjects,
        "counts": counts,
        "bake_size": int(args.bake_size),
        "bake_extrusion": float(args.bake_extrusion),
        "bake_margin_px": int(args.bake_margin_px),
        "merge_dist": float(args.merge_dist),
        "smart_angle": float(args.smart_angle),
        "island_margin": float(args.island_margin),
        "blender_executable": str(blender),
    }
    manifest_path = output_root / "interdeca_full_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    log(f"Wrote full-tier manifest: {manifest_path}")
    return manifest


def write_result(output_root, status, tier, details, started_at, error=None):
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": status,
        "tier": tier,
        "started_at": started_at,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "details": details,
    }
    if error:
        payload["error"] = error
    result_path = output_root / "interdeca_e2e_result.json"
    result_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    log(f"Wrote result: {result_path}")


def run_program(args: ProgramArgs) -> int:
    started_at = time.strftime("%Y-%m-%dT%H:%M:%S")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)

    try:
        require_slicer_dependencies()
        InterDeCA = preamble(args.repo_root)
        dataset = discover_dataset(args.dataset)
        log(f"Dataset: {dataset['root']}")
        log(f"Matched subjects: {len(dataset['matched_subjects'])}")

        details = run_full(args, InterDeCA, dataset)

        write_result(output, "passed", "full", details, started_at)
        log("Full tier passed")
        return 0
    except Exception as exc:
        traceback.print_exc()
        write_result(output, "failed", "full", {}, started_at, error=str(exc))
        return 1


def run_host(
    repo_root,
    slicer_executable,
    blender_executable,
    dataset,
    output,
    atlas_model,
    atlas_landmarks,
    bake_size,
    bake_extrusion,
    bake_margin_px,
    merge_dist,
    smart_angle,
    island_margin,
    show_main_window,
) -> int:
    repo_root = Path(repo_root).resolve()
    runner = Path(__file__).resolve()
    slicer_executable = Path(slicer_executable).expanduser()
    if not slicer_executable.is_file():
        click.echo(f"Slicer executable not found: {slicer_executable}", err=True)
        return 2

    output_root = Path(output).expanduser() if output else Path(
        tempfile.mkdtemp(prefix="interdeca_e2e_")
    )
    output_root.mkdir(parents=True, exist_ok=True)
    env = slicer_env(repo_root)

    tier_output = output_root / "full"
    tier_output.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(slicer_executable),
        "--testing",
        "--disable-settings",
        "--ignore-slicerrc",
    ]
    if not show_main_window:
        cmd.append("--no-main-window")

    cmd.extend(
        [
            "--python-script",
            str(runner),
            "--",
            "--inside-slicer",
            "--repo-root",
            str(repo_root),
            "--dataset",
            str(Path(dataset).expanduser()),
            "--output",
            str(tier_output),
            "--blender",
            str(Path(blender_executable).expanduser()),
            "--bake-size",
            str(bake_size),
            "--bake-extrusion",
            str(bake_extrusion),
            "--bake-margin-px",
            str(bake_margin_px),
            "--merge-dist",
            str(merge_dist),
            "--smart-angle",
            str(smart_angle),
            "--island-margin",
            str(island_margin),
        ]
    )

    if atlas_model:
        cmd.extend(["--atlas-model", str(Path(atlas_model).expanduser())])
    if atlas_landmarks:
        cmd.extend(["--atlas-landmarks", str(Path(atlas_landmarks).expanduser())])
    click.echo("\n[InterDeCA E2E] Running full tier")
    click.echo(f"[InterDeCA E2E] Output: {tier_output}")
    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        click.echo(
            f"[InterDeCA E2E] Full tier failed with exit code {result.returncode}",
            err=True,
        )

    click.echo(f"\n[InterDeCA E2E] Artifacts: {output_root}")
    return result.returncode


def run_inside_slicer(
    repo_root,
    blender_executable,
    dataset,
    output,
    atlas_model,
    atlas_landmarks,
    bake_size,
    bake_extrusion,
    bake_margin_px,
    merge_dist,
    smart_angle,
    island_margin,
) -> int:
    require_slicer()
    if output is None:
        raise click.ClickException("Slicer subprocess mode requires --output.")

    args = ProgramArgs(
        repo_root=repo_root,
        dataset=dataset,
        output=output,
        blender_executable=blender_executable,
        atlas_model=atlas_model,
        atlas_landmarks=atlas_landmarks,
        bake_size=bake_size,
        bake_extrusion=bake_extrusion,
        bake_margin_px=bake_margin_px,
        merge_dist=merge_dist,
        smart_angle=smart_angle,
        island_margin=island_margin,
    )
    return run_program(args)


@click.command(name="interdeca-e2e")
@click.pass_context
@click.option("--inside-slicer", is_flag=True, hidden=True)
@click.option(
    "--repo-root",
    default=REPO_ROOT,
    show_default=True,
    type=click.Path(path_type=Path),
)
@click.option(
    "--slicer",
    "slicer_executable",
    envvar="SLICER_EXECUTABLE",
    default=DEFAULT_SLICER,
    show_default=True,
    type=click.Path(path_type=Path),
)
@click.option(
    "--blender",
    "blender_executable",
    envvar="BLENDER_EXECUTABLE",
    default=DEFAULT_BLENDER,
    show_default=True,
    type=click.Path(path_type=Path),
)
@click.option(
    "--dataset",
    envvar="INTERDECA_E2E_DATASET",
    default=DEFAULT_DATASET,
    show_default=True,
    type=click.Path(path_type=Path),
)
@click.option(
    "--output",
    envvar="INTERDECA_E2E_OUTPUT",
    default=None,
    type=click.Path(path_type=Path),
    help="Output directory for artifacts. Defaults to a temporary directory.",
)
@click.option("--atlas-model", default=None, type=click.Path(path_type=Path))
@click.option("--atlas-landmarks", default=None, type=click.Path(path_type=Path))
@click.option("--bake-size", type=int, default=2048, show_default=True)
@click.option("--bake-extrusion", type=float, default=0.001, show_default=True)
@click.option("--bake-margin-px", type=int, default=2, show_default=True)
@click.option("--merge-dist", type=float, default=0.0001, show_default=True)
@click.option("--smart-angle", type=float, default=66.0, show_default=True)
@click.option("--island-margin", type=float, default=0.002, show_default=True)
@click.option("--show-main-window", is_flag=True, help="Run Slicer with the main window visible.")
def main(
    ctx,
    inside_slicer: bool,
    repo_root: Path,
    slicer_executable: Path,
    blender_executable: Path,
    dataset: Path,
    output: Path | None,
    atlas_model: Path | None,
    atlas_landmarks: Path | None,
    bake_size: int,
    bake_extrusion: float,
    bake_margin_px: int,
    merge_dist: float,
    smart_angle: float,
    island_margin: float,
    show_main_window: bool,
) -> None:
    if inside_slicer:
        ctx.exit(
            run_inside_slicer(
                repo_root,
                blender_executable,
                dataset,
                output,
                atlas_model,
                atlas_landmarks,
                bake_size,
                bake_extrusion,
                bake_margin_px,
                merge_dist,
                smart_angle,
                island_margin,
            )
        )

    ctx.exit(
        run_host(
            repo_root,
            slicer_executable,
            blender_executable,
            dataset,
            output,
            atlas_model,
            atlas_landmarks,
            bake_size,
            bake_extrusion,
            bake_margin_px,
            merge_dist,
            smart_angle,
            island_margin,
            show_main_window,
        )
    )


if __name__ == "__main__":
    main()
