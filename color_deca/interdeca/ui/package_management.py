"""Package management controller boundary."""


class PackageManagementController:
    """Owns dependency status and install UI controls."""

    def __init__(self, widget, dependency_service):
        self.widget = widget
        self.dependency_service = dependency_service
