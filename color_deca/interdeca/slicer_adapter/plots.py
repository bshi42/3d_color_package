"""Slicer plot service boundary."""

import slicer


class SlicerPlotService:
    """Create Slicer plot nodes."""

    def create_table_node(self, name: str):
        return slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTableNode", name)
