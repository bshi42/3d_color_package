"""Folder comparison helpers based on relative paths and SHA-256 digests."""

import fnmatch
import hashlib
from pathlib import Path

from color_deca.interdeca.types import FolderComparisonResult, FolderManifest


class FolderComparison:
    CHUNK_SIZE = 1024 * 1024

    def sha256_file(self, path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as file_obj:
            for chunk in iter(lambda: file_obj.read(self.CHUNK_SIZE), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def folder_manifest(self, root: Path, exclude: list[str] | None = None) -> FolderManifest:
        exclude = exclude or []
        files: dict[str, str] = {}
        for path in sorted(root.rglob("*")):
            if path.is_dir():
                continue
            if not path.is_file():
                raise ValueError(f"Unsupported filesystem entry: {path}")

            relative_path = path.relative_to(root).as_posix()
            if any(fnmatch.fnmatch(relative_path, pattern) for pattern in exclude):
                continue
            files[relative_path] = self.sha256_file(path)

        return FolderManifest(root=root, files=files)

    def compare_folders(
        self,
        left: Path,
        right: Path,
        exclude: list[str] | None = None,
    ) -> tuple[bool, FolderComparisonResult | None]:
        left_manifest = self.folder_manifest(left, exclude)
        right_manifest = self.folder_manifest(right, exclude)

        left_files = set(left_manifest.files)
        right_files = set(right_manifest.files)

        only_left = sorted(left_files - right_files)
        only_right = sorted(right_files - left_files)
        common = sorted(left_files & right_files)
        changed = [
            path
            for path in common
            if left_manifest.files[path] != right_manifest.files[path]
        ]

        if not only_left and not only_right and not changed:
            return True, None

        return False, FolderComparisonResult(
            only_left=only_left,
            only_right=only_right,
            changed={path: (left_manifest.files[path], right_manifest.files[path]) for path in changed},
        )
