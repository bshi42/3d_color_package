"""ATLAS shape service boundary for clean-room InterDeCA."""


class AtlasBridgeAdapter:
    """Boundary around atlas_integration.shape_bridge."""

    def bridge(self):
        raise NotImplementedError("ATLAS bridge behavior has not been copied yet")


class AtlasShapeService:
    """Placeholder boundary for copied ATLAS shape workflow behavior."""

    def run_align(self, *args, **kwargs):
        raise NotImplementedError("ATLAS alignment behavior has not been copied yet")
