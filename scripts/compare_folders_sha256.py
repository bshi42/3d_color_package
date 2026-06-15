#!/usr/bin/env python3
"""Compare two folders recursively by relative file names and SHA-256 digests."""

import fnmatch
import hashlib
from pathlib import Path

import click


CHUNK_SIZE = 1024 * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def folder_manifest(root: Path, exclude: tuple[str, ...] = ()) -> dict[str, str]:
    manifest: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_dir():
            continue
        if not path.is_file():
            raise click.ClickException(f"Unsupported filesystem entry: {path}")

        relative_path = path.relative_to(root).as_posix()
        if any(fnmatch.fnmatch(relative_path, pattern) for pattern in exclude):
            continue
        manifest[relative_path] = sha256_file(path)
    return manifest


@click.command()
@click.argument("left", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("right", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--exclude", multiple=True, metavar="PATTERN", help="Glob pattern to exclude (relative path). Can be repeated.")
def main(left: Path, right: Path, exclude: tuple[str, ...]) -> None:
    """Compare LEFT and RIGHT recursively by relative file path and SHA-256."""
    left = left.resolve()
    right = right.resolve()

    left_manifest = folder_manifest(left, exclude)
    right_manifest = folder_manifest(right, exclude)

    left_files = set(left_manifest)
    right_files = set(right_manifest)

    only_left = sorted(left_files - right_files)
    only_right = sorted(right_files - left_files)
    common = sorted(left_files & right_files)
    changed = [
        path
        for path in common
        if left_manifest[path] != right_manifest[path]
    ]

    if not only_left and not only_right and not changed:
        click.echo(f"Folders match: {left} == {right}")
        raise SystemExit(0)

    click.echo(f"Folders differ: {left} != {right}")

    if only_left:
        click.echo("\nOnly in left:")
        for path in only_left:
            click.echo(f"  {path}")

    if only_right:
        click.echo("\nOnly in right:")
        for path in only_right:
            click.echo(f"  {path}")

    if changed:
        click.echo("\nSHA-256 differs:")
        for path in changed:
            click.echo(f"  {path}")
            click.echo(f"    left:  {left_manifest[path]}")
            click.echo(f"    right: {right_manifest[path]}")

    raise SystemExit(1)


if __name__ == "__main__":
    main()
