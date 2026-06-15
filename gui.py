import sys
import os
import pyvista as pv
from pyvistaqt import QtInteractor
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QCheckBox, QSplitter, QPushButton, QLabel, QFileDialog
)
from PyQt5.QtCore import Qt


class ViewerWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("3D Color Viewer")
        self.resize(1400, 700)

        # Internal state
        self.mesh_folder = ""
        self.texture_folder = ""
        self.pairs = []        # list of (obj_path, texture_path) matched by basename
        self.current_index = 0
        self.actor = None
        self.texture = None

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Splitter divides the three panels so the user can resize them
        splitter = QSplitter()
        main_layout.addWidget(splitter)

        # ── Left panel: folder pickers + load ──────────────────────────────
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(8, 8, 8, 8)
        left_layout.setSpacing(6)

        # Button + path label for the mesh folder
        self.select_mesh_btn = QPushButton("Select Mesh Folder")
        self.mesh_folder_label = QLabel("No folder selected")
        self.mesh_folder_label.setWordWrap(True)
        left_layout.addWidget(self.select_mesh_btn)
        left_layout.addWidget(self.mesh_folder_label)

        # Button + path label for the texture folder
        self.select_texture_btn = QPushButton("Select Texture Folder")
        self.texture_folder_label = QLabel("No folder selected")
        self.texture_folder_label.setWordWrap(True)
        left_layout.addWidget(self.select_texture_btn)
        left_layout.addWidget(self.texture_folder_label)

        # Load button — triggers basename matching across the two folders
        self.load_btn = QPushButton("Load")
        left_layout.addWidget(self.load_btn)

        # Status text reporting how many pairs were matched
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        left_layout.addWidget(self.status_label)
        left_layout.addStretch()

        splitter.addWidget(left_panel)

        # ── Center panel: 3D viewport only ────────────────────────────────
        center_panel = QWidget()
        center_layout = QVBoxLayout(center_panel)
        center_layout.setContentsMargins(8, 8, 8, 8)
        center_layout.setSpacing(0)

        # PyVista 3D viewport embedded as a Qt widget
        self.plotter = QtInteractor(self)
        center_layout.addWidget(self.plotter)

        splitter.addWidget(center_panel)

        # ── Right panel: checkboxes, specimen name, index counter, prev/next ─
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(8, 8, 8, 8)
        right_layout.setSpacing(8)
        right_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Checkboxes for toggling mesh display options
        self.edges_cb = QCheckBox("Show Edges")
        self.texture_cb = QCheckBox("Show Texture")
        self.texture_cb.setChecked(True)
        right_layout.addWidget(self.edges_cb)
        right_layout.addWidget(self.texture_cb)

        # Name of the currently displayed specimen
        self.specimen_label = QLabel("—")
        self.specimen_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.specimen_label.setWordWrap(True)

        # "current / total" counter
        self.index_label = QLabel("")
        self.index_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Arrow buttons — disabled until pairs are loaded
        self.prev_btn = QPushButton("◀  Prev")
        self.next_btn = QPushButton("Next  ▶")
        self.prev_btn.setEnabled(False)
        self.next_btn.setEnabled(False)

        right_layout.addStretch()
        right_layout.addWidget(self.specimen_label)
        right_layout.addWidget(self.index_label)
        right_layout.addWidget(self.prev_btn)
        right_layout.addWidget(self.next_btn)
        right_layout.addStretch()

        splitter.addWidget(right_panel)
        splitter.setSizes([220, 1000, 180])

        # Wire up all buttons and checkboxes
        self.select_mesh_btn.clicked.connect(self.select_mesh_folder)
        self.select_texture_btn.clicked.connect(self.select_texture_folder)
        self.load_btn.clicked.connect(self.load_pairs)
        self.prev_btn.clicked.connect(self.go_prev)
        self.next_btn.clicked.connect(self.go_next)
        self.edges_cb.toggled.connect(self.toggle_edges)
        self.texture_cb.toggled.connect(self.toggle_texture)

    def select_mesh_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Mesh Folder", os.getcwd())
        if folder:
            self.mesh_folder = folder
            self.mesh_folder_label.setText(folder)

    def select_texture_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Texture Folder", os.getcwd())
        if folder:
            self.texture_folder = folder
            self.texture_folder_label.setText(folder)

    def load_pairs(self):
        if not self.mesh_folder or not self.texture_folder:
            self.status_label.setText("Select both folders first.")
            return

        # Index every .obj file in the mesh folder by its basename (no extension)
        obj_files = {
            os.path.splitext(f)[0]: os.path.join(self.mesh_folder, f)
            for f in os.listdir(self.mesh_folder)
            if f.lower().endswith(".obj")
        }

        # For each .obj, look for a texture with the same basename in the texture folder
        image_exts = (".png", ".jpg", ".jpeg")
        self.pairs = []
        for basename, obj_path in sorted(obj_files.items()):
            for ext in image_exts:
                tex_path = os.path.join(self.texture_folder, basename + ext)
                if os.path.exists(tex_path):
                    self.pairs.append((obj_path, tex_path))
                    break

        if not self.pairs:
            self.status_label.setText("No matching pairs found.")
            return

        self.status_label.setText(f"{len(self.pairs)} pairs matched.")
        self.current_index = 0
        self.prev_btn.setEnabled(True)
        self.next_btn.setEnabled(True)
        self._display_current()

    # ── Display the specimen at self.current_index ─────────────────────────

    def _display_current(self):
        obj_path, tex_path = self.pairs[self.current_index]

        # Clear the old mesh and load the new one, preserving checkbox states
        self.plotter.clear()
        mesh = pv.read(obj_path)
        self.texture = pv.read_texture(tex_path)
        tex = self.texture if self.texture_cb.isChecked() else None
        self.actor = self.plotter.add_mesh(mesh, texture=tex, show_edges=self.edges_cb.isChecked())
        self.plotter.reset_camera()
        self.plotter.render()

        # Update the navigation labels on the right panel
        basename = os.path.splitext(os.path.basename(obj_path))[0]
        self.specimen_label.setText(basename)
        self.index_label.setText(f"{self.current_index + 1} / {len(self.pairs)}")

    # ── Navigation ────────────────────────────────────────────────────────

    def go_prev(self):
        if self.pairs:
            self.current_index = (self.current_index - 1) % len(self.pairs)
            self._display_current()

    def go_next(self):
        if self.pairs:
            self.current_index = (self.current_index + 1) % len(self.pairs)
            self._display_current()

    # ── Viewport toggles ──────────────────────────────────────────────────

    def toggle_edges(self, state):
        # Show or hide the wireframe lines along each polygon edge
        if self.actor:
            self.actor.GetProperty().SetEdgeVisibility(state)
            self.plotter.render()

    def toggle_texture(self, state):
        # Swap the texture in or out; passing None shows the plain mesh color
        if self.actor:
            self.actor.SetTexture(self.texture if state else None)
            self.plotter.render()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ViewerWindow()
    window.show()
    sys.exit(app.exec_())
