"""Folder comparison utility tests."""

from tests.util.folder_comparison import FolderComparison


def test_folder_comparison_matches_identical_folders(tmp_path):
    left = tmp_path / "left"
    right = tmp_path / "right"
    left.mkdir()
    right.mkdir()
    (left / "a.txt").write_text("same", encoding="utf-8")
    (right / "a.txt").write_text("same", encoding="utf-8")

    same, result = FolderComparison().compare_folders(left, right)

    assert same is True
    assert result is None


def test_folder_comparison_reports_missing_and_changed_files(tmp_path):
    left = tmp_path / "left"
    right = tmp_path / "right"
    left.mkdir()
    right.mkdir()
    (left / "only-left.txt").write_text("left", encoding="utf-8")
    (right / "only-right.txt").write_text("right", encoding="utf-8")
    (left / "changed.txt").write_text("left", encoding="utf-8")
    (right / "changed.txt").write_text("right", encoding="utf-8")

    same, result = FolderComparison().compare_folders(left, right)

    assert same is False
    assert result is not None
    assert result.only_left == ["only-left.txt"]
    assert result.only_right == ["only-right.txt"]
    assert list(result.changed) == ["changed.txt"]


def test_folder_comparison_exclude_patterns(tmp_path):
    left = tmp_path / "left"
    right = tmp_path / "right"
    left.mkdir()
    right.mkdir()
    (left / "a.txt").write_text("left", encoding="utf-8")
    (right / "a.txt").write_text("right", encoding="utf-8")

    same, result = FolderComparison().compare_folders(left, right, exclude=["a.txt"])

    assert same is True
    assert result is None
