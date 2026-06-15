"""Slicer model and markup IO adapters."""

from pathlib import Path

import slicer


class SlicerModelIO:
    """Coordinate-system-safe Slicer model IO."""

    def save_model_with_cs(self, model_node, file_path: Path, coordinate_system: str = "RAS") -> None:
        if model_node is None:
            raise ValueError(f"Model node is None, cannot save to {file_path}")

        storage = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelStorageNode")
        storage.SetFileName(str(file_path))
        coordinate_system = (coordinate_system or "RAS").upper()
        try:
            if coordinate_system == "RAS":
                storage.SetCoordinateSystemToRAS()
            else:
                storage.SetCoordinateSystemToLPS()
        except AttributeError:
            storage.SetCoordinateSystem(0 if coordinate_system == "RAS" else 1)

        model_node.SetAndObserveStorageNodeID(storage.GetID())
        ok = storage.WriteData(model_node)
        slicer.mrmlScene.RemoveNode(storage)
        if not ok:
            raise RuntimeError(f"Failed to write model: {file_path}")

    def load_model_with_cs(self, file_path: Path, coordinate_system: str = "RAS"):
        storage = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelStorageNode")
        coordinate_system = (coordinate_system or "RAS").upper()
        try:
            if coordinate_system == "RAS":
                storage.SetCoordinateSystemToRAS()
            else:
                storage.SetCoordinateSystemToLPS()
        except AttributeError:
            storage.SetCoordinateSystem(0 if coordinate_system == "RAS" else 1)

        model_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode")
        storage.SetFileName(str(file_path))
        ok = storage.ReadData(model_node)
        slicer.mrmlScene.RemoveNode(storage)
        if not ok:
            slicer.mrmlScene.RemoveNode(model_node)
            raise RuntimeError(f"Failed to read model: {file_path}")
        return model_node
