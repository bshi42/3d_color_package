"""Shared helpers for the InterDeCA Slicer E2E harness."""
import sys
import os
from pathlib import Path
from typing import Literal, NamedTuple, Tuple
import fnmatch
import hashlib

import click


class ApplicationPaths:

    @staticmethod
    def get_slicer():
        # check if SLICER_PATH environment variable is set
        slicer_path = os.environ.get("SLICER_PATH")
        if slicer_path:
            return Path(slicer_path)
        
        # otherwise, use default path based on on platform

        # if on macOS, check the default Slicer app path
        if sys.platform == "darwin":
            return Path("/Applications/Slicer.app/Contents/MacOS/Slicer")
    
    @staticmethod
    def get_blender():
        # check if BLENDER_PATH environment variable is set
        blender_path = os.environ.get("BLENDER_PATH")
        if blender_path:
            return Path(blender_path)
        
        # otherwise, use default path based on on platform

        # if on macOS, check the default Blender app path
        if sys.platform == "darwin":
            return Path("/opt/homebrew/bin/blender")



MODEL_EXTENSIONS = {".obj", ".ply", ".stl", ".vtk", ".vtp"}
TEXTURE_EXTENSIONS = {".png", ".tiff", ".tif", ".jpg", ".jpeg"}
LANDMARK_EXTENSIONS = {".fcsv", ".json"}
CHUNK_SIZE = 1024 * 1024

Tier = Literal["smoke", "analysis", "selection", "full"]


class E2EFailure(AssertionError):
    pass


def fail(message):
    raise E2EFailure(message)


def check(condition, message):
    if not condition:
        fail(message)


def log(message):
    print(f"[InterDeCA E2E] {message}", flush=True)


def project_site_packages(repo_root):
    venv = repo_root / ".venv"
    candidates = sorted(venv.glob("lib/python*/site-packages"))
    if candidates:
        return candidates[0]
    return None


def slicer_env(repo_root):
    env = os.environ.copy()
    site_packages = project_site_packages(repo_root)
    if site_packages and (site_packages / "click").is_dir():
        existing = env.get("PYTHONPATH")
        env["PYTHONPATH"] = (
            str(site_packages)
            if not existing
            else os.pathsep.join([str(site_packages), existing])
        )
    return env


def find_first_dir(root, names):
    for name in names:
        candidate = root / name
        if candidate.is_dir():
            return candidate
    fail(f"None of these directories exist under {root}: {', '.join(names)}")


def landmark_subject_id(path):
    name = path.name
    lower = name.lower()
    if lower.endswith(".mrk.json"):
        return name[: -len(".mrk.json")]
    return path.stem


def file_subject_id(path):
    return path.stem


def pick_files(directory, kind):
    if kind == "model":
        priority = {".obj": 0, ".ply": 1, ".stl": 2, ".vtp": 3, ".vtk": 4}
    elif kind == "texture":
        priority = {".png": 0, ".tiff": 1, ".tif": 2, ".jpg": 3, ".jpeg": 4}
    elif kind == "landmark":
        priority = {".mrk.json": 0, ".json": 1, ".fcsv": 2}
    else:
        fail(f"Unknown file kind: {kind}")

    picked = {}
    for path in sorted(directory.iterdir()):
        if path.name.startswith(".") or not path.is_file():
            continue
        lower = path.name.lower()
        if kind == "landmark":
            if lower.endswith(".mrk.json"):
                ext_key = ".mrk.json"
                subject_id = landmark_subject_id(path)
            elif path.suffix.lower() in LANDMARK_EXTENSIONS:
                ext_key = path.suffix.lower()
                subject_id = landmark_subject_id(path)
            else:
                continue
        else:
            ext_key = path.suffix.lower()
            if kind == "model" and ext_key not in MODEL_EXTENSIONS:
                continue
            if kind == "texture" and ext_key not in TEXTURE_EXTENSIONS:
                continue
            subject_id = file_subject_id(path)

        rank = priority[ext_key]
        if subject_id not in picked or rank < picked[subject_id][0]:
            picked[subject_id] = (rank, path)

    return {subject_id: path for subject_id, (_, path) in picked.items()}


def discover_dataset(dataset_root):
    root = Path(dataset_root).resolve()
    check(root.is_dir(), f"Dataset root does not exist: {root}")

    models_dir = find_first_dir(root, ["models", "Models", "models_small"])
    landmarks_dir = find_first_dir(root, ["landmarks", "LMS"])
    textures_dir = find_first_dir(root, ["textures", "Textures"])

    models = pick_files(models_dir, "model")
    landmarks = pick_files(landmarks_dir, "landmark")
    textures = pick_files(textures_dir, "texture")
    matched = sorted(set(models) & set(landmarks) & set(textures))

    check(models, f"No model files found in {models_dir}")
    check(landmarks, f"No landmark files found in {landmarks_dir}")
    check(textures, f"No texture files found in {textures_dir}")
    check(matched, f"No matched model/landmark/texture subjects found in {root}")

    return {
        "root": root,
        "models_dir": models_dir,
        "landmarks_dir": landmarks_dir,
        "textures_dir": textures_dir,
        "models": models,
        "landmarks": landmarks,
        "textures": textures,
        "matched_subjects": matched,
        "ignored_textures": sorted(set(textures) - set(matched)),
    }


def choose_subject(dataset, requested_subject=None):
    matched = dataset["matched_subjects"]
    if requested_subject:
        check(requested_subject in matched, f"Requested subject is not matched: {requested_subject}")
        return requested_subject

    def model_size(subject_id):
        return dataset["models"][subject_id].stat().st_size

    return min(matched, key=model_size)


def count_files(path, suffix):
    return len([p for p in path.iterdir() if p.is_file() and p.name.lower().endswith(suffix)])


class FolderComparisonResult(NamedTuple):
    only_left: list[str]
    only_right: list[str]
    changed: dict[str, Tuple[str, str]]

class FolderComparison:
    CHUNK_SIZE = 1024 * 1024

    def sha256_file(self, path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as file_obj:
            for chunk in iter(lambda: file_obj.read(self.CHUNK_SIZE), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def folder_manifest(self, root: Path, exclude: list[str] = list()) -> dict[str, str]:
        manifest: dict[str, str] = {}
        for path in sorted(root.rglob("*")):
            if path.is_dir():
                continue
            if not path.is_file():
                raise click.ClickException(f"Unsupported filesystem entry: {path}")

            relative_path = path.relative_to(root).as_posix()
            if any(fnmatch.fnmatch(relative_path, pattern) for pattern in exclude):
                continue
            manifest[relative_path] = self.sha256_file(path)

        return manifest

    def compare_folders(self, left: Path, right: Path, exclude: list[str] = list()) -> tuple[bool, FolderComparisonResult | None]:

        # get folder manifests for both folders
        left_manifest = self.folder_manifest(left, exclude)
        right_manifest = self.folder_manifest(right, exclude)

        # compare file sets
        left_files = set(left_manifest)
        right_files = set(right_manifest)

        only_left = sorted(left_files - right_files)
        only_right = sorted(right_files - left_files)
        common = sorted(left_files & right_files)

        # compare common file contents
        changed = [
            path
            for path in common
            if left_manifest[path] != right_manifest[path]
        ]

        # same files with same content
        if not only_left and not only_right and not changed:
            return True, None

        # differences found
        return False, FolderComparisonResult(
            only_left=only_left,
            only_right=only_right,
            changed={path: (left_manifest[path], right_manifest[path]) for path in changed},
        )
