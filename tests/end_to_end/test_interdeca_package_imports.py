"""Import checks for the standard-Python InterDeCA package boundary."""

import importlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def test_standard_interdeca_package_import_does_not_import_slicer_modules():
    blocked_modules = {"qt", "ctk", "slicer"}
    before = set(sys.modules)

    importlib.import_module("color_deca.interdeca")
    importlib.import_module("color_deca.interdeca.core")
    importlib.import_module("color_deca.interdeca.vtk_ops")

    newly_imported = set(sys.modules) - before
    assert blocked_modules.isdisjoint(newly_imported)


def test_theme_module_imports_without_slicer_qt_until_used():
    module = importlib.import_module("color_deca.interdeca.ui.theme")
    assert hasattr(module, "ColorTheme")
