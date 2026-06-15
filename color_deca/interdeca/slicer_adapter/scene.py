"""Slicer scene helpers."""

import slicer


class SlicerSceneService:
    """Small wrapper around common MRML scene operations."""

    def clear_scene(self) -> None:
        try:
            slicer.mrmlScene.Clear(0)
        except TypeError:
            slicer.mrmlScene.Clear()

    def remove_node(self, node) -> None:
        if node is not None:
            slicer.mrmlScene.RemoveNode(node)
