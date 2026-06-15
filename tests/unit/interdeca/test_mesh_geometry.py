"""VTK mesh geometry tests."""

import numpy as np
import vtk

from color_deca.interdeca.vtk_ops.mesh_geometry import MeshGeometryService


def make_triangle_polydata():
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

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetPolys(cells)
    return polydata


def test_calculate_triangle_face_area_exactly():
    areas = MeshGeometryService().calculate_face_areas(make_triangle_polydata())

    np.testing.assert_allclose(areas, np.array([0.5]))


def test_extract_face_connectivity_for_triangle():
    faces = MeshGeometryService().extract_face_connectivity(make_triangle_polydata())

    np.testing.assert_array_equal(faces, np.array([[0, 1, 2]]))
