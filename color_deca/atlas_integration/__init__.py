"""
ATLAS Integration Module for InterDeCA

Provides wrapper classes that bridge ATLAS functionality (BUILDER, PREDICT, DATABASE)
with InterDeCA's color analysis workflows. This module handles:
- Atlas generation with texture preservation
- Automated landmark transfer
- Statistical Shape Model (SSM) database management
- Dense correspondence generation

Note: ATLAS modules must be available in the Slicer extension path.
"""

__version__ = "1.0.0"
__author__ = "InterDeCA Development Team"

# Version info for dependency checking
ATLAS_MIN_VERSION = "1.0.0"
REQUIRED_PYTHON_PACKAGES = ["tiny3d", "biocpd", "scipy"]
