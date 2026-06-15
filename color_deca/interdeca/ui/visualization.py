"""Visualization controller boundary."""


class VisualizationController:
    """Owns visualization and display controls."""

    def __init__(self, widget, logic):
        self.widget = widget
        self.logic = logic
