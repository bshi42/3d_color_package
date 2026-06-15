"""VTK-only mesh selection helpers."""

from color_deca.interdeca.types import MeshSelectionResult


class MeshSelectionService:
    """Small, Slicer-free mesh selection algorithms."""

    def select_vertices_in_xy_polygon(self, polydata, polygon_points: list[tuple[float, float]]) -> MeshSelectionResult:
        selected_vertices = []
        for point_id in range(polydata.GetNumberOfPoints()):
            x_coord, y_coord, _ = polydata.GetPoint(point_id)
            if self._point_in_polygon(x_coord, y_coord, polygon_points):
                selected_vertices.append(point_id)

        selected_set = set(selected_vertices)
        selected_cells = []
        for cell_id in range(polydata.GetNumberOfCells()):
            cell = polydata.GetCell(cell_id)
            if all(cell.GetPointId(index) in selected_set for index in range(cell.GetNumberOfPoints())):
                selected_cells.append(cell_id)

        return MeshSelectionResult(
            selected_vertices=selected_vertices,
            total_vertices=polydata.GetNumberOfPoints(),
            selected_cells=selected_cells,
        )

    def _point_in_polygon(self, x_coord: float, y_coord: float, polygon_points: list[tuple[float, float]]) -> bool:
        inside = False
        previous_index = len(polygon_points) - 1
        for current_index, point in enumerate(polygon_points):
            xi, yi = point
            xj, yj = polygon_points[previous_index]
            intersects = (yi > y_coord) != (yj > y_coord) and (
                x_coord < (xj - xi) * (y_coord - yi) / (yj - yi) + xi
            )
            if intersects:
                inside = not inside
            previous_index = current_index
        return inside
