"""UI composition root for the clean-room InterDeCA package."""

from color_deca.interdeca.ui.deca_tab import DeCATabController
from color_deca.interdeca.ui.mesh_selection_tab import MeshSelectionController
from color_deca.interdeca.ui.multi_recolor_tab import MultiRecolorController
from color_deca.interdeca.ui.package_management import PackageManagementController
from color_deca.interdeca.ui.visualization import VisualizationController


class InterDeCAUiComposition:
    """Wire InterDeCA UI controllers together."""

    def __init__(self, widget, logic, dependency_service=None):
        self.deca_tab = DeCATabController(widget, logic)
        self.mesh_selection = MeshSelectionController(widget, logic)
        self.visualization = VisualizationController(widget, logic)
        self.multi_recolor = MultiRecolorController(widget, logic)
        self.package_management = PackageManagementController(widget, dependency_service)
