"""Application workflow service boundary for InterDeCA."""

from color_deca.interdeca.types import AtlasTextureWorkflowConfig, AtlasTextureWorkflowResult


class AtlasTextureWorkflow:
    """Full atlas and texture-transfer workflow boundary."""

    def run(self, config: AtlasTextureWorkflowConfig) -> AtlasTextureWorkflowResult:
        raise NotImplementedError("Atlas texture workflow behavior has not been copied yet")
