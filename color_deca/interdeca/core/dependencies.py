"""Optional dependency detection for InterDeCA core services."""

from importlib import util

from color_deca.interdeca.types import DependencyStatus


def _has_module(module_name: str) -> bool:
    return util.find_spec(module_name) is not None


def detect_dependencies() -> DependencyStatus:
    """Detect optional analysis dependencies independently."""
    return DependencyStatus(
        sklearn_available=_has_module("sklearn"),
        umap_available=_has_module("umap"),
        scipy_available=_has_module("scipy"),
        skimage_available=_has_module("skimage"),
    )
