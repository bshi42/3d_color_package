"""DeCA workflow tab controller boundary."""


class DeCATabController:
    """Owns DeCA workflow tab UI and delegates workflow execution."""

    def __init__(self, widget, logic):
        self.widget = widget
        self.logic = logic
