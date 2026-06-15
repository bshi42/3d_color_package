"""Dataset discovery tests."""

from pathlib import Path

from color_deca.interdeca.core.dataset import DatasetDiscovery, landmark_subject_id, pick_files


REPO_ROOT = Path(__file__).resolve().parents[3]
SMALL_CLAMS = REPO_ROOT / "tests" / "data" / "small_clams"


def test_small_clams_dataset_discovery_matches_subjects():
    dataset = DatasetDiscovery().discover(SMALL_CLAMS)

    assert dataset.matched_subjects == ["UF_IZ_507781", "UF_IZ_507782", "UF_IZ_507879"]
    assert dataset.models_dir == SMALL_CLAMS / "models"
    assert dataset.landmarks_dir == SMALL_CLAMS / "landmarks"
    assert dataset.textures_dir == SMALL_CLAMS / "textures"


def test_pick_files_uses_extension_priority(tmp_path):
    models = tmp_path / "models"
    models.mkdir()
    (models / "subject.ply").write_text("ply", encoding="utf-8")
    (models / "subject.obj").write_text("obj", encoding="utf-8")

    picked = pick_files(models, "model")

    assert picked["subject"] == models / "subject.obj"


def test_landmark_subject_id_strips_full_mrk_json_suffix():
    assert landmark_subject_id(Path("UF_IZ_507781.mrk.json")) == "UF_IZ_507781"
