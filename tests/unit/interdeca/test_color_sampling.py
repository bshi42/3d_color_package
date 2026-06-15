"""Color sampling tests."""

import numpy as np
import vtk

from color_deca.interdeca.vtk_ops.color_sampling import ColorSamplingService


def make_uv_triangle_polydata():
    points = vtk.vtkPoints()
    points.InsertNextPoint(0.0, 0.0, 0.0)
    points.InsertNextPoint(1.0, 0.0, 0.0)
    points.InsertNextPoint(0.0, 1.0, 0.0)

    triangle = vtk.vtkTriangle()
    triangle.GetPointIds().SetId(0, 0)
    triangle.GetPointIds().SetId(1, 1)
    triangle.GetPointIds().SetId(2, 2)

    cells = vtk.vtkCellArray()
    cells.InsertNextCell(triangle)

    tcoords = vtk.vtkFloatArray()
    tcoords.SetNumberOfComponents(2)
    tcoords.SetName("TCoords")
    tcoords.InsertNextTuple((0.0, 0.0))
    tcoords.InsertNextTuple((1.0, 0.0))
    tcoords.InsertNextTuple((0.0, 1.0))

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetPolys(cells)
    polydata.GetPointData().SetTCoords(tcoords)
    return polydata


def test_face_average_colors_use_clamped_flipped_uv_pixel_mapping():
    texture = np.array(
        [
            [[10, 0, 0], [20, 0, 0]],
            [[30, 0, 0], [40, 0, 0]],
        ],
        dtype=np.uint8,
    )

    result = ColorSamplingService().calculate_face_average_colors(
        make_uv_triangle_polydata(),
        texture,
        "RGB",
    )

    np.testing.assert_array_equal(result.face_colors, np.array([[80.0 / 3.0, 0.0, 0.0]]))
