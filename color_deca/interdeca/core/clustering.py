"""Multi-texture clustering service boundary."""

from typing import NamedTuple


class ClusteringPipeline(NamedTuple):
    texture_files: list[str]
    shared_palette: object | None = None
    metadata: dict[str, object] | None = None


class MultiTextureClusteringService:
    """Placeholder boundary for copied MultiRecolor clustering behavior."""

    def perform_multi_texture_clustering(self, *args, **kwargs):
        raise NotImplementedError("Multi-texture clustering behavior has not been copied yet")
