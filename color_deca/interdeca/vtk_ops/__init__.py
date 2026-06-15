"""VTK-based InterDeCA operations that avoid Slicer MRML dependencies."""

__all__ = []
"""VTK-only InterDeCA operations."""

from color_deca.interdeca.vtk_ops.color_sampling import ColorSamplingService
from color_deca.interdeca.vtk_ops.mesh_geometry import MeshGeometryService
from color_deca.interdeca.vtk_ops.mesh_selection import MeshSelectionService

__all__ = [
    "ColorSamplingService",
    "MeshGeometryService",
    "MeshSelectionService",
]
