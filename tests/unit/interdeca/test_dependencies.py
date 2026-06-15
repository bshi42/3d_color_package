"""Dependency detection tests."""

from color_deca.interdeca.core import dependencies


def test_dependency_detection_checks_sklearn_independently_from_umap(monkeypatch):
    def fake_find_spec(module_name):
        if module_name == "sklearn":
            return object()
        if module_name == "umap":
            return None
        if module_name == "scipy":
            return object()
        if module_name == "skimage":
            return None
        return None

    monkeypatch.setattr(dependencies.util, "find_spec", fake_find_spec)
    status = dependencies.detect_dependencies()

    assert status.sklearn_available is True
    assert status.umap_available is False
    assert status.scipy_available is True
    assert status.skimage_available is False
