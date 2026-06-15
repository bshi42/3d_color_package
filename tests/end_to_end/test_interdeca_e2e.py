"""End-to-end tests for the InterDeCA Slicer workflow."""

import json
import subprocess
import sys
from pathlib import Path

from util import FolderComparison



REPO_ROOT = Path(__file__).resolve().parents[2]
DRIVER = REPO_ROOT / "tests" / "slicer_e2e" / "run_interdeca_e2e.py"
SMALL_CLAMS = REPO_ROOT / "tests" / "data" / "small_clams"
EXPECTED_SMALL_CLAMS = REPO_ROOT / "tests" / "output" / "slicer_small_clams" / "2026_06-05_15_14_52"
SMALL_CLAM_SUBJECTS = ["UF_IZ_507781", "UF_IZ_507782", "UF_IZ_507879"]


def test_e2e_driver_targets_clean_room_facade():
    driver_source = DRIVER.read_text(encoding="utf-8")

    assert "from color_deca.interdeca import facade as InterDeCA" in driver_source
    assert 'repo_root / "color_deca" / "deca3"' not in driver_source
    assert "import InterDeCA" not in driver_source


def assert_input_fixture_exists():
    assert SMALL_CLAMS.is_dir()
    for subject in SMALL_CLAM_SUBJECTS:
        assert (SMALL_CLAMS / "models" / f"{subject}.obj").is_file()
        assert (SMALL_CLAMS / "landmarks" / f"{subject}.mrk.json").is_file()

    assert (SMALL_CLAMS / "textures" / "UF_IZ_507781.png").is_file()
    assert (SMALL_CLAMS / "textures" / "UF_IZ_507782.tiff").is_file()
    assert (SMALL_CLAMS / "textures" / "UF_IZ_507879.png").is_file()


def assert_full_output_contract(output_root):
    full_output = output_root / "full"
    assert (full_output / "interdeca_e2e_result.json").is_file()
    assert (full_output / "interdeca_full_manifest.json").is_file()

    assert (full_output / "colorAnalysis" / "atlasLM.mrk.json").is_file()
    assert (full_output / "colorAnalysis" / "atlasModel.ply").is_file()
    assert (full_output / "colorAnalysis" / "atlasModelUV.obj").is_file()
    assert (full_output / "colorAnalysis" / "atlasTextures" / "average_texture.png").is_file()

    for subject in SMALL_CLAM_SUBJECTS:
        assert (full_output / "ATLAS" / "alignedLMs" / f"{subject}_align.mrk.json").is_file()
        assert (full_output / "ATLAS" / "alignedModels" / f"{subject}_align.ply").is_file()
        assert (full_output / "ATLAS" / "resampledModels" / f"{subject}_resampled.ply").is_file()
        assert (full_output / "colorAnalysis" / "resampledOBJ_withUV" / f"{subject}_resampled.obj").is_file()
        assert (full_output / "colorAnalysis" / "atlasTextures" / f"{subject}.png").is_file()

    result = json.loads((full_output / "interdeca_e2e_result.json").read_text(encoding="utf-8"))
    assert result["status"] == "passed"
    assert result["tier"] == "full"

    manifest = json.loads((full_output / "interdeca_full_manifest.json").read_text(encoding="utf-8"))
    assert manifest["tier"] == "full"
    assert manifest["dataset_root"] == str(SMALL_CLAMS.resolve())
    assert manifest["subjects"] == SMALL_CLAM_SUBJECTS
    assert manifest["counts"] == {
        "aligned_landmarks": 3,
        "aligned_ply_models": 3,
        "baked_textures": 3,
        "resampled_ply_models": 3,
        "resampled_uv_objs": 3,
        "subjects": 3,
    }


def test_small_clams(tmp_path):
    assert DRIVER.is_file()
    assert EXPECTED_SMALL_CLAMS.is_dir()
    assert_input_fixture_exists()

    output_root = tmp_path / "interdeca_small_clams"
    cmd = [
        sys.executable,
        str(DRIVER),
        "--dataset",
        str(SMALL_CLAMS),
        "--output",
        str(output_root),
    ]

    result = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    assert_full_output_contract(output_root)

    observed_output = output_root / "full"
    comparison = FolderComparison()
    same, comparison_result = comparison.compare_folders(
        left=EXPECTED_SMALL_CLAMS,
        right=observed_output,
        exclude=["ATLAS/temp*", "interdeca_*.json"],
    )

    assert same, f"Output folders differ: {comparison_result}"
