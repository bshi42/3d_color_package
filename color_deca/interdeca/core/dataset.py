"""Dataset discovery and subject matching for InterDeCA workflows."""

from pathlib import Path

from color_deca.interdeca.types import DatasetInfo


MODEL_EXTENSIONS = {".obj", ".ply", ".stl", ".vtk", ".vtp"}
TEXTURE_EXTENSIONS = {".png", ".tiff", ".tif", ".jpg", ".jpeg"}
LANDMARK_EXTENSIONS = {".fcsv", ".json"}


def landmark_subject_id(path: Path) -> str:
    name = path.name
    lower = name.lower()
    if lower.endswith(".mrk.json"):
        return name[: -len(".mrk.json")]
    return path.stem


def file_subject_id(path: Path) -> str:
    return path.stem


def find_first_dir(root: Path, names: list[str]) -> Path:
    for name in names:
        candidate = root / name
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"None of these directories exist under {root}: {', '.join(names)}")


def pick_files(directory: Path, kind: str) -> dict[str, Path]:
    if kind == "model":
        priority = {".obj": 0, ".ply": 1, ".stl": 2, ".vtp": 3, ".vtk": 4}
    elif kind == "texture":
        priority = {".png": 0, ".tiff": 1, ".tif": 2, ".jpg": 3, ".jpeg": 4}
    elif kind == "landmark":
        priority = {".mrk.json": 0, ".json": 1, ".fcsv": 2}
    else:
        raise ValueError(f"Unknown file kind: {kind}")

    picked: dict[str, tuple[int, Path]] = {}
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


class DatasetDiscovery:
    """Discover InterDeCA datasets with deterministic subject matching."""

    def discover(self, dataset_root: Path) -> DatasetInfo:
        root = Path(dataset_root).resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Dataset root does not exist: {root}")

        models_dir = find_first_dir(root, ["models", "Models", "models_small"])
        landmarks_dir = find_first_dir(root, ["landmarks", "LMS"])
        textures_dir = find_first_dir(root, ["textures", "Textures"])

        models = pick_files(models_dir, "model")
        landmarks = pick_files(landmarks_dir, "landmark")
        textures = pick_files(textures_dir, "texture")
        matched_subjects = sorted(set(models) & set(landmarks) & set(textures))

        if not models:
            raise FileNotFoundError(f"No model files found in {models_dir}")
        if not landmarks:
            raise FileNotFoundError(f"No landmark files found in {landmarks_dir}")
        if not textures:
            raise FileNotFoundError(f"No texture files found in {textures_dir}")
        if not matched_subjects:
            raise ValueError(f"No matched model/landmark/texture subjects found in {root}")

        return DatasetInfo(
            root=root,
            models_dir=models_dir,
            landmarks_dir=landmarks_dir,
            textures_dir=textures_dir,
            models=models,
            landmarks=landmarks,
            textures=textures,
            matched_subjects=matched_subjects,
            ignored_textures=sorted(set(textures) - set(matched_subjects)),
        )
