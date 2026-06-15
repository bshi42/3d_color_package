"""VTK mesh geometry utilities copied from InterDeCA logic boundaries."""

import numpy as np
from vtk.util import numpy_support as vtk_np


class MeshGeometryService:
    """Geometry helpers for VTK polydata."""

    def extract_face_connectivity(self, polydata):
        polys = polydata.GetPolys()
        if polys is None or polys.GetNumberOfCells() == 0:
            return None
        raw_array = vtk_np.vtk_to_numpy(polys.GetData())
        number_of_faces = polys.GetNumberOfCells()
        if len(raw_array) == number_of_faces * 4:
            return raw_array.reshape(number_of_faces, 4)[:, 1:4]
        if len(raw_array) == number_of_faces * 5:
            return raw_array.reshape(number_of_faces, 5)[:, 1:5]
        return None

    def calculate_face_areas(self, polydata):
        try:
            number_of_faces = polydata.GetNumberOfCells()
            faces = self.extract_face_connectivity(polydata)
            points_np = vtk_np.vtk_to_numpy(polydata.GetPoints().GetData())

            if faces is not None and faces.shape[1] == 3:
                p0 = points_np[faces[:, 0]]
                p1 = points_np[faces[:, 1]]
                p2 = points_np[faces[:, 2]]
                cross = np.cross(p1 - p0, p2 - p0)
                return 0.5 * np.linalg.norm(cross, axis=1)

            if faces is not None and faces.shape[1] == 4:
                p0 = points_np[faces[:, 0]]
                p1 = points_np[faces[:, 1]]
                p2 = points_np[faces[:, 2]]
                p3 = points_np[faces[:, 3]]
                area1 = 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1)
                area2 = 0.5 * np.linalg.norm(np.cross(p2 - p0, p3 - p0), axis=1)
                return area1 + area2

            face_areas = np.zeros(number_of_faces)
            for face_id in range(number_of_faces):
                cell = polydata.GetCell(face_id)
                if cell.GetNumberOfPoints() >= 3:
                    points = [
                        np.array(polydata.GetPoint(cell.GetPointId(index)))
                        for index in range(cell.GetNumberOfPoints())
                    ]
                    v1 = points[1] - points[0]
                    v2 = points[2] - points[0]
                    area = 0.5 * np.linalg.norm(np.cross(v1, v2))
                    if len(points) == 4:
                        v3 = points[3] - points[0]
                        area += 0.5 * np.linalg.norm(np.cross(v2, v3))
                    face_areas[face_id] = area
            return face_areas
        except Exception:
            return None

    def face_adjacency(self, polydata) -> dict[int, set[int]]:
        adjacency = {face_id: set() for face_id in range(polydata.GetNumberOfCells())}
        edge_to_faces: dict[tuple[int, int], list[int]] = {}
        for face_id in range(polydata.GetNumberOfCells()):
            cell = polydata.GetCell(face_id)
            point_ids = [cell.GetPointId(index) for index in range(cell.GetNumberOfPoints())]
            for index, point_id in enumerate(point_ids):
                edge = tuple(sorted((point_id, point_ids[(index + 1) % len(point_ids)])))
                edge_to_faces.setdefault(edge, []).append(face_id)

        for face_ids in edge_to_faces.values():
            if len(face_ids) < 2:
                continue
            for face_id in face_ids:
                adjacency[face_id].update(other for other in face_ids if other != face_id)
        return adjacency
