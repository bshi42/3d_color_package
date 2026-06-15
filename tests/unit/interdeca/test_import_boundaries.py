"""Import boundary tests for the clean-room InterDeCA package."""

import importlib
import sys


def test_core_imports_do_not_import_slicer_modules():
    blocked_modules = {"qt", "ctk", "slicer"}
    before = set(sys.modules)

    importlib.import_module("color_deca.interdeca")
    importlib.import_module("color_deca.interdeca.core")
    importlib.import_module("color_deca.interdeca.vtk_ops")

    newly_imported = set(sys.modules) - before
    assert blocked_modules.isdisjoint(newly_imported)
