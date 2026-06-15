"""Texture color sampling utilities for VTK meshes."""

import colorsys

import numpy as np
from vtk.util import numpy_support as vtk_np

from color_deca.interdeca.types import ColorSamplingResult
from color_deca.interdeca.vtk_ops.mesh_geometry import MeshGeometryService


class ColorSamplingService:
    """Calculate per-face colors from UV-mapped texture images."""

    def __init__(self, geometry: MeshGeometryService | None = None):
        self.geometry = geometry or MeshGeometryService()

    def calculate_face_average_colors(self, polydata, texture_image, color_space: str) -> ColorSamplingResult:
        return ColorSamplingResult(
            face_colors=self.face_average_colors(polydata, texture_image, color_space),
            color_space=color_space,
            face_indices=None,
        )

    def calculate_sampled_face_average_colors(
        self,
        polydata,
        texture_image,
        face_indices,
        color_space: str,
    ) -> ColorSamplingResult:
        return ColorSamplingResult(
            face_colors=self.sampled_face_average_colors(polydata, texture_image, face_indices, color_space),
            color_space=color_space,
            face_indices=face_indices,
        )

    def face_average_colors(self, polydata, texture_image, color_space: str):
        try:
            tcoords = polydata.GetPointData().GetTCoords()
            if not tcoords:
                return None

            tcoords_np = vtk_np.vtk_to_numpy(tcoords)
            height, width = texture_image.shape[:2]
            faces = self.geometry.extract_face_connectivity(polydata)
            if faces is not None and faces.shape[1] == 3:
                face_colors = self._vectorized_face_colors(faces, tcoords_np, texture_image, width, height)
                if color_space == "HSV":
                    face_colors = self._rgb_rows_to_hsv_features(face_colors)
                return face_colors

            return self._loop_face_colors(polydata, tcoords_np, texture_image, width, height, color_space)
        except Exception:
            return None

    def sampled_face_average_colors(self, polydata, texture_image, face_indices, color_space: str):
        try:
            tcoords = polydata.GetPointData().GetTCoords()
            if not tcoords:
                return None

            tcoords_np = vtk_np.vtk_to_numpy(tcoords)
            height, width = texture_image.shape[:2]
            all_faces = self.geometry.extract_face_connectivity(polydata)
            if all_faces is not None and all_faces.shape[1] == 3:
                sampled_faces = all_faces[face_indices]
                face_colors = self._vectorized_face_colors(sampled_faces, tcoords_np, texture_image, width, height)
                if color_space == "HSV":
                    face_colors = self._rgb_rows_to_hsv_features(face_colors)
                return face_colors

            face_colors = self.face_average_colors(polydata, texture_image, color_space)
            return face_colors[face_indices] if face_colors is not None else None
        except Exception:
            return None

    def _vectorized_face_colors(self, faces, tcoords_np, texture_image, width: int, height: int):
        u = np.clip(tcoords_np[:, 0], 0, 1)
        v = np.clip(1.0 - tcoords_np[:, 1], 0, 1)
        px = np.clip((u * (width - 1)).astype(int), 0, width - 1)
        py = np.clip((v * (height - 1)).astype(int), 0, height - 1)
        vertex_colors = texture_image[py, px, :3].astype(np.float64)
        return np.mean(vertex_colors[faces], axis=1)

    def _loop_face_colors(self, polydata, tcoords_np, texture_image, width: int, height: int, color_space: str):
        number_of_faces = polydata.GetPolys().GetNumberOfCells()
        face_colors_list = []

        for face_index in range(number_of_faces):
            cell = polydata.GetCell(face_index)
            vertex_indices = [cell.GetPointId(point_index) for point_index in range(cell.GetNumberOfPoints())]
            face_tex_coords = tcoords_np[vertex_indices].copy()
            face_tex_coords[:, 1] = 1.0 - face_tex_coords[:, 1]
            pixel_coords = np.clip(face_tex_coords, 0, 1) * [width - 1, height - 1]
            pixel_coords = pixel_coords.astype(int)
            face_pixel_colors = texture_image[pixel_coords[:, 1], pixel_coords[:, 0], :3]
            average_color = np.mean(face_pixel_colors, axis=0)

            if color_space == "HSV":
                rgb_normalized = average_color / 255.0
                hsv = colorsys.rgb_to_hsv(rgb_normalized[0], rgb_normalized[1], rgb_normalized[2])
                hue_radians = hsv[0] * 2 * np.pi
                hue_cos = np.cos(hue_radians)
                hue_sin = np.sin(hue_radians)
                average_color = np.array([hue_cos, hue_sin, hsv[1] * 100, hsv[2] * 100])

            face_colors_list.append(average_color)

        return np.array(face_colors_list)

    def _rgb_rows_to_hsv_features(self, face_colors):
        rgb_norm = face_colors / 255.0
        red, green, blue = rgb_norm[:, 0], rgb_norm[:, 1], rgb_norm[:, 2]
        maxc = np.maximum(np.maximum(red, green), blue)
        minc = np.minimum(np.minimum(red, green), blue)
        diff = maxc - minc

        hue = np.zeros(len(face_colors))
        mask_r = (maxc == red) & (diff > 0)
        mask_g = (maxc == green) & (diff > 0)
        mask_b = (maxc == blue) & (diff > 0)
        hue[mask_r] = ((green[mask_r] - blue[mask_r]) / diff[mask_r]) % 6.0
        hue[mask_g] = ((blue[mask_g] - red[mask_g]) / diff[mask_g]) + 2.0
        hue[mask_b] = ((red[mask_b] - green[mask_b]) / diff[mask_b]) + 4.0
        hue = hue / 6.0

        saturation = np.where(maxc > 0, diff / maxc, 0.0)
        hue_radians = hue * 2 * np.pi
        hue_cos = np.cos(hue_radians)
        hue_sin = np.sin(hue_radians)
        return np.column_stack([hue_cos, hue_sin, saturation * 100, maxc * 100])
