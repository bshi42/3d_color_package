"""
InterDeCA (Interactive Dense Correspondence Analysis) module for 3D Slicer.

Extends the original DeCA module with advanced color analysis capabilities,
texture baking through Blender integration, and interactive visualization tools
for studying color patterns in biological specimens.

Key features:
- Blender integration for UV mapping and texture baking
- Color space analysis (RGB/HSV) with dimensionality reduction
- Multi-texture clustering for comparative color analysis
- Interactive shape interpolation and visualization
- Face-level color quantization and pattern analysis

Dependencies:
- Optional: sklearn, umap-learn, scikit-image for advanced analysis
- External: Blender for texture processing
"""

import os
import unittest
import vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *
import logging
import fnmatch
import  numpy as np
import random
import math
from datetime import datetime
import re
import csv
import vtk.util.numpy_support as vtk_np
from pathlib import Path
import shutil
import imageio # slicer.util.pip_install('imageio')
import glob
import colorsys


# macOS may add AppleDouble/resource-fork entries (for example, ``._sample.obj``)
# alongside the actual dataset files. They are not independent specimens and
# must never be considered by dataset enumeration code.
def _is_visible_dataset_file(file_name):
  return bool(file_name) and not file_name.startswith('.')


LANDMARK_SURFACE_WARNING_RELATIVE_THRESHOLD = 0.02
ATLAS_ANCHOR_LABELS = ('beak', 'hinge_anterior', 'hinge_posterior')
ATLAS_SIDE_LABEL = 'anterior_adductor_dorsal'
GROWTH_AXIS_LABELS = tuple(f'max_growth_axis_{i:03d}' for i in range(1, 9))


def _relative_landmark_surface_distance(distance, mesh_diagonal):
  """Return landmark-to-surface distance as a fraction of mesh size."""
  if mesh_diagonal > 0:
    return distance / mesh_diagonal
  return float('inf') if distance > 0 else 0.0


def _growth_axis_quality(points, beak_index, growth_indices):
  """Calculate ordering and spacing diagnostics for an eight-point axis."""
  points = np.asarray(points, dtype=float)
  growth_indices = list(growth_indices)
  if points.ndim != 2 or points.shape[1] != 3:
    raise ValueError("Landmark points must have shape (N, 3)")
  if len(growth_indices) != 8:
    raise ValueError("Growth axis must contain exactly eight landmarks")

  growth = points[growth_indices]
  steps = np.linalg.norm(np.diff(growth, axis=0), axis=1)
  if np.any(steps <= 0):
    return {
      'closest_to_beak': int(np.argmin(np.linalg.norm(growth - points[beak_index], axis=1))),
      'spacing_cv': float('inf'),
      'min_turn_cosine': -1.0,
      'path_to_direct': float('inf'),
    }

  directions = np.diff(growth, axis=0)
  turn_cosines = [
    np.dot(first, second) / (np.linalg.norm(first) * np.linalg.norm(second))
    for first, second in zip(directions[:-1], directions[1:])
  ]
  direct = np.linalg.norm(growth[-1] - growth[0])
  return {
    'closest_to_beak': int(np.argmin(np.linalg.norm(growth - points[beak_index], axis=1))),
    'spacing_cv': float(np.std(steps) / np.mean(steps)),
    'min_turn_cosine': float(min(turn_cosines)),
    'path_to_direct': float(np.sum(steps) / direct) if direct > 0 else float('inf'),
  }


def _anatomical_anchor_frame(points, beak_index, anterior_hinge_index,
                             posterior_hinge_index):
  """Express landmarks in a scale-normalized frame defined by three anchors."""
  points = np.asarray(points, dtype=float)
  beak = points[beak_index]
  anterior = points[anterior_hinge_index]
  posterior = points[posterior_hinge_index]
  hinge_axis = anterior - posterior
  hinge_length = np.linalg.norm(hinge_axis)
  hinge_midpoint = 0.5 * (anterior + posterior)
  beak_to_hinge = hinge_midpoint - beak
  beak_hinge_length = np.linalg.norm(beak_to_hinge)
  if hinge_length <= 0 or beak_hinge_length <= 0:
    raise ValueError("Anchor landmarks are coincident")

  x_axis = hinge_axis / hinge_length
  y_axis = beak_to_hinge - np.dot(beak_to_hinge, x_axis) * x_axis
  y_length = np.linalg.norm(y_axis)
  if y_length <= 1e-12 * max(hinge_length, beak_hinge_length):
    raise ValueError("Anchor landmarks are collinear")
  y_axis /= y_length
  z_axis = np.cross(x_axis, y_axis)
  z_axis /= np.linalg.norm(z_axis)
  scale = math.sqrt(hinge_length ** 2 + beak_hinge_length ** 2)
  axes = np.column_stack((x_axis, y_axis, z_axis))
  return (points - beak) @ axes / scale


def _robust_modified_z(values):
  """Return median/MAD modified z-scores, with stable zero-MAD handling."""
  values = np.asarray(values, dtype=float)
  median = np.median(values, axis=0)
  absolute_deviation = np.abs(values - median)
  mad = np.median(absolute_deviation, axis=0)
  fallback = np.maximum(np.median(np.abs(values), axis=0) * 1e-9, 1e-12)
  scale = np.where(mad > 0, mad, fallback)
  return 0.6744897501960817 * (values - median) / scale



# Attempts to import optional machine learning libraries
try:
    from sklearn.decomposition import PCA, FastICA  # Imports dimensionality reduction algorithms
    from sklearn.manifold import TSNE  # Imports t-SNE for non-linear dimensionality reduction
    from sklearn.cluster import KMeans, MiniBatchKMeans  # Imports clustering algorithms for color analysis
    import umap  # Imports UMAP for advanced manifold learning
    SKLEARN_AVAILABLE = True  # Sets flag indicating scikit-learn is available
    UMAP_AVAILABLE = True  # Sets flag indicating UMAP is available
except ImportError:
    SKLEARN_AVAILABLE = False  # Disables scikit-learn dependent features
    UMAP_AVAILABLE = False  # Disables UMAP dependent features
    print("Warning: sklearn and/or umap not available. Colors EDA functionality will be limited.")

# Attempts to import scipy for hierarchical clustering
try:
    from scipy.cluster import hierarchy
    from scipy.spatial.distance import pdist, squareform
    from scipy.optimize import linear_sum_assignment
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available. Hierarchical clustering functionality will be limited.")

# Attempts to import scikit-image for color quantization
try:
    from skimage import color as skimage_color  # Imports color space conversion utilities
    from skimage.color import deltaE_ciede2000  # Imports perceptual color difference metric
    SKIMAGE_AVAILABLE = True  # Sets flag for advanced color features
except ImportError:
    SKIMAGE_AVAILABLE = False  # Disables color quantization features
    print("Warning: scikit-image not available. Color quantization functionality will be limited.")

# ATLAS Integration - replaces DeCA for shape correspondence
# Color analysis features (Blender, EDA, quantization) are preserved as-is
import sys
import os

# Import ATLAS shape bridge for dense correspondence operations
try:
    # Add parent directory to path to find atlas_integration module
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    
    from atlas_integration.shape_bridge import get_shape_bridge
    
    # Get global shape bridge instance (provides DeCA-compatible API)
    atlasShapeBridge = get_shape_bridge()
    
    if atlasShapeBridge.is_atlas_available():
        print('Successfully loaded ATLAS shape correspondence modules!')
    else:
        print('ATLAS modules not found - using fallback VTK implementations')
    
    print(f'ATLAS Shape Bridge initialized: {atlasShapeBridge}')
    
except ImportError as e:
    print(f'Could not import ATLAS integration: {e}')
    print('Please ensure atlas_integration module is in the correct path')
    atlasShapeBridge = None

# For backward compatibility during migration
decaLogic = None  # No longer using DeCA

def checkAndOfferPackageInstallation():
    """
    Checks for missing optional packages and offers to install them.
    Returns a tuple of (missing_packages, all_available)
    """
    import datetime

    print(f"\n--- InterDeCA Package Check ({datetime.datetime.now().strftime('%H:%M:%S')}) ---")

    missing_packages = []
    available_packages = []
    package_info = {
        'numpy': {'import_test': lambda: __import__('numpy'), 'pip_name': 'numpy'},
        'sklearn': {'import_test': lambda: __import__('sklearn'), 'pip_name': 'scikit-learn'},
        'umap': {'import_test': lambda: __import__('umap'), 'pip_name': 'umap-learn'},
        'skimage': {'import_test': lambda: __import__('skimage'), 'pip_name': 'scikit-image'},
        'imageio': {'import_test': lambda: __import__('imageio'), 'pip_name': 'imageio'}
    }

    # Checks which packages are missing by attempting imports
    for package_name, info in package_info.items():
        try:
            info['import_test']()  # Attempts to import the package
            available_packages.append(info['pip_name'])  # Adds to available list if import succeeds
            print(f"✓ {info['pip_name']} - Available")
        except ImportError:
            # Builds package info dictionary when import fails
            missing_packages.append({
                'name': package_name,
                'pip_name': info['pip_name'],
                'description': {  # Maps package names to user-friendly descriptions
                    'numpy': 'Core numerical computing library (usually included with Slicer)',
                    'sklearn': 'Required for PCA, t-SNE, and clustering in Colors EDA',
                    'umap': 'Required for UMAP dimensionality reduction in Colors EDA',
                    'skimage': 'Required for advanced color quantization and analysis',
                    'imageio': 'Required for texture and image processing'
                }.get(package_name, 'Optional package for enhanced functionality')
            })
            print(f"✗ {info['pip_name']} - Missing")  # Indicates missing package

    print(f"Package Status: {len(available_packages)} available, {len(missing_packages)} missing")
    if available_packages:
        print(f"Available: {', '.join(available_packages)}")
    if missing_packages:
        print(f"Missing: {', '.join([pkg['pip_name'] for pkg in missing_packages])}")
    print("--- End Package Check ---\n")

    return missing_packages, available_packages

def installMissingPackages(missing_packages):
    """
    Installs the specified missing packages using Slicer's pip functionality.

    Args:
        missing_packages: List of package dictionaries with 'pip_name' keys
    """
    import subprocess
    import datetime

    success_count = 0
    failed_packages = []
    installed_packages = []

    # Log installation start
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n{'='*60}")
    print(f"InterDeCA Package Installation Started - {timestamp}")
    print(f"{'='*60}")
    print(f"Packages to install: {len(missing_packages)}")
    for package in missing_packages:
        print(f"  • {package['pip_name']}: {package['description']}")
    print(f"{'='*60}\n")

    # Shows progress dialog to track installation status
    progressDialog = slicer.util.createProgressDialog(
        windowTitle="Installing Packages",
        maximum=len(missing_packages)  # Sets maximum value for progress bar
    )

    try:
        for i, package in enumerate(missing_packages):
            package_name = package['pip_name']
            progressDialog.labelText = f"Installing {package_name}..."
            progressDialog.value = i
            slicer.app.processEvents()

            print(f"[{i+1}/{len(missing_packages)}] Installing {package_name}...")

            try:
                # Uses Slicer's pip_install utility to install package
                slicer.util.pip_install(package_name)  # Executes pip install command
                success_count += 1  # Increments success counter
                installed_packages.append(package_name)  # Tracks successfully installed packages
                print(f"✓ Successfully installed {package_name}")

                # Verifies installation by attempting to import the newly installed package
                try:
                    if package_name == 'scikit-learn':
                        import sklearn  # Attempts sklearn import
                        print(f"  → Verified sklearn version: {sklearn.__version__}")  # Confirms version
                    elif package_name == 'umap-learn':
                        import umap  # Attempts umap import
                        print(f"  → Verified umap-learn installation")
                    elif package_name == 'scikit-image':
                        import skimage  # Attempts skimage import
                        print(f"  → Verified skimage version: {skimage.__version__}")
                    elif package_name == 'imageio':
                        import imageio  # Attempts imageio import
                        print(f"  → Verified imageio version: {imageio.__version__}")
                except ImportError as verify_error:
                    print(f"  ⚠ Warning: Could not verify {package_name} import: {verify_error}")  # Warns if verification fails

            except Exception as e:
                print(f"✗ Failed to install {package_name}: {e}")
                failed_packages.append(package_name)

        progressDialog.close()

        # Log installation summary
        end_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n{'='*60}")
        print(f"InterDeCA Package Installation Summary - {end_timestamp}")
        print(f"{'='*60}")

        if success_count > 0:
            print(f"✓ Successfully installed {success_count} package(s):")
            for pkg in installed_packages:
                print(f"    • {pkg}")

        if failed_packages:
            print(f"\n✗ Failed to install {len(failed_packages)} package(s):")
            for pkg in failed_packages:
                print(f"    • {pkg}")

        print(f"\nTotal packages processed: {len(missing_packages)}")
        print(f"Success rate: {(success_count/len(missing_packages)*100):.1f}%")

        if success_count > 0:
            print(f"\n⚠ IMPORTANT: Restart Slicer to use newly installed packages")

        print(f"{'='*60}\n")

        # Show results to user
        if success_count > 0:
            success_msg = f"Successfully installed {success_count} package(s)."
            if failed_packages:
                success_msg += f"\n\nFailed to install: {', '.join(failed_packages)}"
                success_msg += "\nYou may need to restart Slicer for changes to take effect."
            else:
                success_msg += "\n\nPlease restart Slicer to use the new functionality."

            success_msg += f"\n\nCheck the Python console for detailed installation log."
            slicer.util.infoDisplay(success_msg, windowTitle="Installation Complete")
        else:
            slicer.util.errorDisplay(
                f"Failed to install packages: {', '.join(failed_packages)}\n\nCheck the Python console for details.",
                windowTitle="Installation Failed"
            )

    except Exception as e:
        progressDialog.close()
        print(f"\n✗ INSTALLATION ERROR: {e}")
        print(f"Check the Python console for details.\n")
        slicer.util.errorDisplay(f"Installation process failed: {e}\n\nCheck the Python console for details.", windowTitle="Installation Error")

#
# InterDeCA
#

SHOW_VISUALIZE_RESULTS = False

class InterDeCA(ScriptedLoadableModule):
  """
  Module class for Interactive Dense Correspondence Analysis (InterDeCA).

  Extends DeCA functionality with color-based morphometric analysis tools,
  providing workflows for texture processing, color pattern analysis,
  and interactive visualization of biological specimens.

  Base class documentation:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, parent):
    """
    Initializes the InterDeCA module with metadata and configuration.

    Args:
      parent: Parent object from 3D Slicer framework
    """
    ScriptedLoadableModule.__init__(self, parent)

    # Sets module metadata for 3D Slicer's module browser
    self.parent.title = "InterDeCA"  # Interactive Dense Correspondence Analysis
    self.parent.categories = ["SlicerMorph.DeCA Toolbox"]
    self.parent.dependencies = []  # Original DeCA module loaded separately
    self.parent.contributors = ["Sara Rolfe (SCRI)"]

    # Provides user-facing documentation
    self.parent.helpText = """
      This module provides several flexible workflows for finding and analyzing dense correspondence points between models,
      with enhanced support for color analysis and texture processing.
      """
    self.parent.helpText += self.getDefaultModuleDocumentationLink()

    # Acknowledges funding sources
    self.parent.acknowledgementText = """This extension was developed by funding from National Institutes of Health (OD032627 and HD104435) to A. Murat Maga (SCRI)
      """

#
# InterDeCAWidget
#

class InterDeCAWidget(ScriptedLoadableModuleWidget):
  """
  GUI widget for the InterDeCA module.

  Creates and manages an enhanced user interface with multiple tabs:
  - DeCA: Standard dense correspondence with texture support
  - DeCAL: Dense landmarking
  - Visualize: Interactive results visualization with interpolation
  - Colors EDA: Color pattern analysis with dimensionality reduction
  - Recolor: Single texture application and quantization
  - MultiRecolor: Multi-texture clustering analysis

  Base class documentation:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def setup(self):
    """
    Builds the module's user interface.

    Creates a tabbed interface with six main workflows:
    - DeCA: Dense correspondence with Blender integration
    - DeCAL: Dense landmarking with subsampling
    - Visualize: Heatmaps and shape interpolation
    - Colors EDA: Statistical color analysis
    - Recolor: Texture application and quantization
    - MultiRecolor: Comparative multi-texture analysis
    """
    ScriptedLoadableModuleWidget.setup(self)

    # Suppress VTK warnings about painting lines with <2 points (harmless markup visualization warnings)
    try:
      import vtk
      vtkOutput = vtk.vtkOutputWindow()
      vtkOutput.SetInstance(None)
    except:
      pass  # If this fails, warnings will still appear but won't affect functionality

    # Initializes variables for interpolation visualization
    self.interpolatedModelNode = None  # Stores temporary model for interpolation display
    self.selectedOriginalModelNode = None  # Stores selected resampled model reference
    self.lastDeCAAlignedModelsPath = None  # Caches path to aligned models directory

    # Initializes progress tracking components
    self.progressBar = None  # Stores progress bar widget reference

    # Initializes persistent data storage for settings
    self.settings = qt.QSettings()  # Creates QSettings object for persistent storage
    self.settings.beginGroup("InterDeCA")  # Groups all InterDeCA settings together
    self.progressLabel = None  # Stores progress label widget reference
    self.cancelButton = None  # Stores cancel button widget reference
    self.currentOperation = None  # Tracks current operation for cancellation

    # Sets up tabs to organize complex workflow into logical sections
    tabsWidget = qt.QTabWidget()  # Creates main tab widget container
    self.tabsWidget = tabsWidget  # Stores reference for later access
    DeCATab = qt.QWidget()  # Creates widget for DeCA workflow
    DeCATabLayout = qt.QFormLayout(DeCATab)  # Sets form layout for organized controls

    # Conditionally creates visualize tab based on feature flag
    if SHOW_VISUALIZE_RESULTS:  # Checks if visualization feature is enabled
      visualizeTab = qt.QWidget()  # Creates widget for visualization tools
      visualizeTabLayout = qt.QFormLayout(visualizeTab)  # Sets form layout for controls

    meshSelectionTab = qt.QWidget()
    meshSelectionTabLayout = qt.QFormLayout(meshSelectionTab)
    multiRecolorTab = qt.QWidget()
    multiRecolorTabLayout = qt.QFormLayout(multiRecolorTab)

    tabsWidget.addTab(DeCATab, "ATLAS")
    tabsWidget.addTab(meshSelectionTab, "Mesh Selection")
    if SHOW_VISUALIZE_RESULTS:
      tabsWidget.addTab(visualizeTab, "Visualize Results")
    tabsWidget.addTab(multiRecolorTab, "MultiRecolor")

    # Add PCA Morphospace tab if module is available


    self.layout.addWidget(tabsWidget)

    # Package Management Section
    packageWidget = ctk.ctkCollapsibleButton()
    packageWidget.text = "Package Management"
    packageWidget.collapsed = True
    packageWidgetLayout = qt.QFormLayout(packageWidget)
    self.layout.addWidget(packageWidget)

    # Package status label
    self.packageStatusLabel = qt.QLabel()
    packageWidgetLayout.addRow("Status:", self.packageStatusLabel)

    # Install missing packages button
    self.installPackagesButton = qt.QPushButton("Check & Install Missing Packages")
    self.installPackagesButton.toolTip = "Check for missing optional packages and offer to install them"
    self.installPackagesButton.connect('clicked(bool)', self.onInstallPackagesClicked)
    packageWidgetLayout.addRow(self.installPackagesButton)

    # Quick install all button
    self.installAllButton = qt.QPushButton("Install All Recommended Packages")
    self.installAllButton.toolTip = "Install all recommended packages for full InterDeCA functionality"
    self.installAllButton.connect('clicked(bool)', self.onInstallAllPackagesClicked)
    packageWidgetLayout.addRow(self.installAllButton)

    # Update package status after all widgets are created
    self.updatePackageStatus()

    ################################### DeCA Tab ###################################

    #
    # Input Directories Section
    #
    self.inputDirCollapsibleButton = ctk.ctkCollapsibleButton()
    self.inputDirCollapsibleButton.text = "Input Directories"
    self.inputDirCollapsibleButton.collapsed = False
    self.inputDirCollapsibleButton.setStyleSheet(ColorTheme.getHeaderStyle())
    DeCATabLayout.addRow(self.inputDirCollapsibleButton)
    inputDirLayout = qt.QFormLayout(self.inputDirCollapsibleButton)
    
    # Model directory
    self.meshDirectoryDC=ctk.ctkPathLineEdit()
    self.meshDirectoryDC.filters = ctk.ctkPathLineEdit.Dirs
    self.meshDirectoryDC.setToolTip("Select directory containing models")
    
    # Add validation status label for models
    self.meshValidationLabelDC = qt.QLabel()
    self.meshValidationLabelDC.setStyleSheet(ColorTheme.getLabelStyle())
    self.meshValidationLabelDC.setText("No directory selected")
    
    meshDirWidget = qt.QWidget()
    meshDirLayout = qt.QHBoxLayout(meshDirWidget)
    meshDirLayout.setContentsMargins(0, 0, 0, 0)
    meshDirLayout.addWidget(self.meshDirectoryDC)
    meshDirLayout.addWidget(self.meshValidationLabelDC)
    inputDirLayout.addRow("Models: ", meshDirWidget)

    # Landmark directory
    self.landmarkDirectoryDC=ctk.ctkPathLineEdit()
    self.landmarkDirectoryDC.filters = ctk.ctkPathLineEdit.Dirs
    self.landmarkDirectoryDC.setToolTip("Select directory containing landmarks")
    
    # Add validation status label for landmarks
    self.landmarkValidationLabelDC = qt.QLabel()
    self.landmarkValidationLabelDC.setStyleSheet(ColorTheme.getLabelStyle())
    self.landmarkValidationLabelDC.setText("No directory selected")
    
    landmarkDirWidget = qt.QWidget()
    landmarkDirLayout = qt.QHBoxLayout(landmarkDirWidget)
    landmarkDirLayout.setContentsMargins(0, 0, 0, 0)
    landmarkDirLayout.addWidget(self.landmarkDirectoryDC)
    landmarkDirLayout.addWidget(self.landmarkValidationLabelDC)
    inputDirLayout.addRow("Landmarks: ", landmarkDirWidget)

    # Textures directory
    self.textureDirectoryDC = ctk.ctkPathLineEdit()
    self.textureDirectoryDC.filters = ctk.ctkPathLineEdit.Dirs
    self.textureDirectoryDC.setToolTip("Directory with subject PNG textures (file name must match subject ID).")
    
    # Add validation status label for textures
    self.textureValidationLabelDC = qt.QLabel()
    self.textureValidationLabelDC.setStyleSheet(ColorTheme.getValidationLabelStyle('neutral'))
    self.textureValidationLabelDC.setText("No directory selected")
    
    textureDirWidget = qt.QWidget()
    textureDirLayout = qt.QHBoxLayout(textureDirWidget)
    textureDirLayout.setContentsMargins(0, 0, 0, 0)
    textureDirLayout.addWidget(self.textureDirectoryDC)
    textureDirLayout.addWidget(self.textureValidationLabelDC)
    inputDirLayout.addRow("Textures (PNG): ", textureDirWidget)
    
    # Output directory
    self.outputDirectoryDC=ctk.ctkPathLineEdit()
    self.outputDirectoryDC.filters = ctk.ctkPathLineEdit.Dirs
    self.outputDirectoryDC.setToolTip("Select directory for DeCA output")
    inputDirLayout.addRow("Output directory: ", self.outputDirectoryDC)


    # Add spacing
    DeCATabLayout.addRow(" ", qt.QLabel())

    #
    # Atlas Override Section
    #
    self.atlasOverrideCollapsibleButton = ctk.ctkCollapsibleButton()
    self.atlasOverrideCollapsibleButton.text = "Atlas Override (Optional)"
    self.atlasOverrideCollapsibleButton.collapsed = True
    self.atlasOverrideCollapsibleButton.setStyleSheet(ColorTheme.getHeaderStyle())
    DeCATabLayout.addRow(self.atlasOverrideCollapsibleButton)
    atlasOverrideLayout = qt.QFormLayout(self.atlasOverrideCollapsibleButton)

    # Atlas Model
    self.atlasModelOverride = ctk.ctkPathLineEdit()
    self.atlasModelOverride.filters = ctk.ctkPathLineEdit.Files
    self.atlasModelOverride.nameFilters = ["*.ply", "*.stl", "*.obj", "*.vtk", "*.vtp"]
    self.atlasModelOverride.setToolTip("Select a pre-calculated atlas model to skip generation")
    atlasOverrideLayout.addRow("Atlas Model: ", self.atlasModelOverride)

    # Atlas Landmarks
    self.atlasLandmarkOverride = ctk.ctkPathLineEdit()
    self.atlasLandmarkOverride.filters = ctk.ctkPathLineEdit.Files
    self.atlasLandmarkOverride.nameFilters = ["*.mrk.json", "*.fcsv", "*.json"]
    self.atlasLandmarkOverride.setToolTip("Select corresponding landmarks for the atlas model")
    atlasOverrideLayout.addRow("Atlas Landmarks: ", self.atlasLandmarkOverride)

    # --- Blender integration ---
    # Configures external Blender processing for UV mapping and texture baking
    self.blenderGroup = ctk.ctkCollapsibleButton()
    self.blenderGroup.text = "Blender (cleanup, UV, bake)"
    self.blenderGroup.collapsed = True
    self.blenderGroup.setStyleSheet(ColorTheme.getHeaderStyle())
    DeCATabLayout.addRow(self.blenderGroup)
    blForm = qt.QFormLayout(self.blenderGroup)

    self.blenderExeEdit = ctk.ctkPathLineEdit()
    self.blenderExeEdit.filters = ctk.ctkPathLineEdit().Files
    self.blenderExeEdit.setToolTip("Path to Blender executable (blender, blender.exe).")
    blForm.addRow("Blender executable:", self.blenderExeEdit)

    self.blMergeDistSpin = qt.QDoubleSpinBox()
    self.blMergeDistSpin.setDecimals(6); self.blMergeDistSpin.setRange(0.0, 1e3); self.blMergeDistSpin.setValue(0.0001)
    self.blMergeDistSpin.setToolTip("Merge by Distance threshold in model units (used on atlas & resampled before bake).")
    blForm.addRow("Merge by distance:", self.blMergeDistSpin)

    self.blSmartAngleSpin = qt.QDoubleSpinBox()
    self.blSmartAngleSpin.setRange(1.0, 179.0); self.blSmartAngleSpin.setValue(66.0)
    self.blSmartAngleSpin.setToolTip("Smart UV Project angle limit (degrees).")
    blForm.addRow("Smart UV angle (deg):", self.blSmartAngleSpin)

    self.blIslandMarginSpin = qt.QDoubleSpinBox()
    self.blIslandMarginSpin.setDecimals(4); self.blIslandMarginSpin.setRange(0.0, 0.05); self.blIslandMarginSpin.setValue(0.002)
    self.blIslandMarginSpin.setToolTip("Smart UV island margin (UV units).")
    blForm.addRow("Island margin (UV):", self.blIslandMarginSpin)

    self.bakeSizeSpin = qt.QSpinBox()
    self.bakeSizeSpin.setRange(128, 8192); self.bakeSizeSpin.setSingleStep(128); self.bakeSizeSpin.setValue(2048)
    self.bakeSizeSpin.setToolTip("Baked texture size (square).")
    blForm.addRow("Bake size (px):", self.bakeSizeSpin)

    self.bakeExtrusionSpin = qt.QDoubleSpinBox()
    self.bakeExtrusionSpin.setDecimals(6); self.bakeExtrusionSpin.setRange(0.0, 10.0); self.bakeExtrusionSpin.setValue(0.001)
    self.bakeExtrusionSpin.setToolTip("Selection→Active bake cage extrusion distance (Blender units).")
    blForm.addRow("Bake extrusion:", self.bakeExtrusionSpin)

    self.bakeMarginPxSpin = qt.QSpinBox()
    self.bakeMarginPxSpin.setRange(0, 64); self.bakeMarginPxSpin.setValue(2)
    self.bakeMarginPxSpin.setToolTip("Bake dilation margin (pixels).")
    blForm.addRow("Bake margin (px):", self.bakeMarginPxSpin)

    # Adds spacing between sections for visual clarity
    DeCATabLayout.addRow(" ", qt.QLabel())  # Creates empty row as separator

    ################################### Mesh Selection Tab ###################################
    #
    # Mesh Region Selection
    #
    # Region selection section for selecting and analyzing mesh regions
    self.regionSelectionWidget = ctk.ctkCollapsibleButton()
    self.regionSelectionWidget.text = "Mesh Region Selection"
    self.regionSelectionWidget.collapsed = False  # Expanded by default in dedicated tab
    self.regionSelectionWidget.setStyleSheet(ColorTheme.getHeaderStyle())
    meshSelectionTabLayout.addRow(self.regionSelectionWidget)
    regionLayout = qt.QFormLayout(self.regionSelectionWidget)

    # Target mesh selector for region selection
    self.regionMeshSelector = slicer.qMRMLNodeComboBox()
    self.regionMeshSelector.setStyleSheet(ColorTheme.getComboBoxStyle())
    self.regionMeshSelector.nodeTypes = (("vtkMRMLModelNode"), "")
    self.regionMeshSelector.setToolTip("Select the mesh to perform region selection on")
    self.regionMeshSelector.selectNodeUponCreation = False
    self.regionMeshSelector.noneEnabled = True
    self.regionMeshSelector.addEnabled = False
    self.regionMeshSelector.removeEnabled = False
    self.regionMeshSelector.showHidden = False
    self.regionMeshSelector.setMRMLScene(slicer.mrmlScene)
    regionLayout.addRow("Target Mesh:", self.regionMeshSelector)

    # --- Landmark Method Controls ---
    self.landmarkFrame = qt.QFrame()
    self.landmarkLayout = qt.QFormLayout()
    self.landmarkFrame.setLayout(self.landmarkLayout)
    self.landmarkFrame.setVisible(True)  # Visible by default

    # Markup selector for closed curves
    self.selectionMarkupSelector = slicer.qMRMLNodeComboBox()
    self.selectionMarkupSelector.setStyleSheet(ColorTheme.getComboBoxStyle())
    self.selectionMarkupSelector.nodeTypes = ["vtkMRMLMarkupsClosedCurveNode"]
    self.selectionMarkupSelector.setToolTip("Select a closed curve to define the region boundary")
    self.selectionMarkupSelector.selectNodeUponCreation = True
    self.selectionMarkupSelector.addEnabled = False
    self.selectionMarkupSelector.removeEnabled = False
    self.selectionMarkupSelector.noneEnabled = True
    self.selectionMarkupSelector.addEnabled = True
    self.selectionMarkupSelector.removeEnabled = False
    self.selectionMarkupSelector.showHidden = False
    self.selectionMarkupSelector.setMRMLScene(slicer.mrmlScene)
    self.landmarkLayout.addRow("Selection Markup:", self.selectionMarkupSelector)

    # Select only one side checkbox
    self.mirrorSelectionCheckbox = qt.QCheckBox()
    self.mirrorSelectionCheckbox.setToolTip("When enabled, only vertices on the same side as landmarks will be selected. When disabled, vertices from both sides forming the complete area will be selected.")
    self.mirrorSelectionCheckbox.setChecked(True)  # Default to one side only
    self.mirrorSelectionCheckbox.setMinimumHeight(25)  # Make checkbox bigger
    self.mirrorSelectionCheckbox.setStyleSheet("QCheckBox::indicator { width: 20px; height: 20px; }")
    self.landmarkLayout.addRow("Select only one side:", self.mirrorSelectionCheckbox)

    # Add spacing after checkbox
    self.landmarkLayout.addRow(" ", qt.QLabel())

    # Draw and Clear buttons in a horizontal layout
    drawClearWidget = qt.QWidget()
    drawClearLayout = qt.QHBoxLayout(drawClearWidget)
    drawClearLayout.setContentsMargins(0, 0, 0, 0)
    drawClearLayout.setSpacing(10)

    self.createMarkupButton = qt.QPushButton("Draw Curve")
    self.createMarkupButton.setToolTip("Draw a closed curve to define the selection region")
    self.createMarkupButton.setStyleSheet(ColorTheme.getButtonStyle('secondary'))
    self.createMarkupButton.setMinimumHeight(40)
    self.createMarkupButton.connect('clicked(bool)', self.onCreateOrPlaceMarkup)
    drawClearLayout.addWidget(self.createMarkupButton)

    self.clearSelectionButton = qt.QPushButton("Clear Selection")
    self.clearSelectionButton.setToolTip("Clear the current region selection")
    self.clearSelectionButton.enabled = False
    self.clearSelectionButton.setStyleSheet(ColorTheme.getButtonStyle('neutral'))
    self.clearSelectionButton.setMinimumHeight(40)
    drawClearLayout.addWidget(self.clearSelectionButton)

    self.landmarkLayout.addRow(drawClearWidget)

    # Apply selection button - primary action
    self.applyLandmarkSelectionButton = qt.QPushButton("Apply Selection")
    self.applyLandmarkSelectionButton.setToolTip("Apply region selection using the closed curve")
    self.applyLandmarkSelectionButton.enabled = False
    self.applyLandmarkSelectionButton.setStyleSheet(ColorTheme.getButtonStyle('primary'))
    self.applyLandmarkSelectionButton.setMinimumHeight(45)
    self.landmarkLayout.addRow(self.applyLandmarkSelectionButton)

    # Add spacing before export section
    self.landmarkLayout.addRow(" ", qt.QLabel())

    # Export region name input
    self.exportRegionNameEdit = qt.QLineEdit()
    self.exportRegionNameEdit.setPlaceholderText("Enter name for exported region")
    self.exportRegionNameEdit.setToolTip("Name for the exported region model")
    self.exportRegionNameEdit.setText("SelectedRegion")
    self.exportRegionNameEdit.setMinimumHeight(30)
    self.landmarkLayout.addRow("Export Name:", self.exportRegionNameEdit)

    # Export landmark selection button
    self.exportLandmarkSelectionButton = qt.QPushButton("Export Selected Region as Model")
    self.exportLandmarkSelectionButton.setToolTip("Export the selected region as a separate model")
    self.exportLandmarkSelectionButton.enabled = False
    self.exportLandmarkSelectionButton.setStyleSheet(ColorTheme.getButtonStyle('secondary'))
    self.exportLandmarkSelectionButton.setMinimumHeight(40)
    self.landmarkLayout.addRow(self.exportLandmarkSelectionButton)

    regionLayout.addRow(self.landmarkFrame)

    # Add spacing before selection info
    regionLayout.addRow(" ", qt.QLabel())

    # Selection info label - more prominent display
    self.selectionInfoLabel = qt.QLabel("No region selected")
    self.selectionInfoLabel.setStyleSheet("""
        QLabel {
            background-color: #3a3a3a;
            border: 1px solid #555;
            border-radius: 4px;
            padding: 8px;
            font-style: italic;
        }
    """)
    self.selectionInfoLabel.setMinimumHeight(35)
    self.selectionInfoLabel.setAlignment(qt.Qt.AlignCenter)
    regionLayout.addRow(self.selectionInfoLabel)

    ################################### DeCA Tab (continued) ###################################
    #
    # Progress tracking widgets
    #
    self.progressWidgetDC = qt.QWidget()  # Creates container widget for progress controls
    self.progressWidgetDC.setVisible(False)  # Hides progress widget initially
    progressLayout = qt.QVBoxLayout(self.progressWidgetDC)  # Sets vertical layout for progress elements
    progressLayout.setContentsMargins(0, 0, 0, 0)  # Removes margins for compact display

    self.progressBarDC = qt.QProgressBar()  # Creates progress bar widget
    self.progressBarDC.setRange(0, 100)  # Sets percentage range
    self.progressBarDC.setValue(0)  # Initializes at 0%
    progressLayout.addWidget(self.progressBarDC)  # Adds to layout


    self.cancelButtonDC = qt.QPushButton("Cancel Operation")  # Creates cancel button
    self.cancelButtonDC.setMaximumWidth(120)  # Limits button width
    self.cancelButtonDC.setVisible(False)  # Hides initially until operation starts
    progressLayout.addWidget(self.cancelButtonDC)  # Adds to progress layout

    DeCATabLayout.addRow("Progress: ", self.progressWidgetDC)

    #
    # Run ATLAS Button
    #
    self.applyButtonDC = qt.QPushButton("Run ATLAS and Texture Transfer")  # Creates main execution button
    self.applyButtonDC.toolTip = "Run ATLAS shape correspondence and texture transfer"  # Sets helpful tooltip
    self.applyButtonDC.enabled = False  # Disables until required inputs are provided
    self.applyButtonDC.setStyleSheet(ColorTheme.getButtonStyle('primary'))  # Applies primary button styling
    DeCATabLayout.addRow(self.applyButtonDC)  # Adds button to form layout


    #
    # Log Information
    #
    self.logInfoDC = qt.QPlainTextEdit()  # Creates text area for log output
    self.logInfoDC.setPlaceholderText("DeCA log information")  # Shows placeholder when empty
    self.logInfoDC.setReadOnly(True)  # Prevents user editing of log content
    DeCATabLayout.addRow(self.logInfoDC)  # Adds log widget to layout

    # Connects UI signals to handler functions
    self.meshDirectoryDC.connect('validInputChanged(bool)', self.onParameterSelectDC)  # Validates mesh directory input
    self.meshDirectoryDC.connect('currentPathChanged(QString)', self.onMeshDirectoryChangedDC)  # Handles mesh path changes
    self.landmarkDirectoryDC.connect('validInputChanged(bool)', self.onParameterSelectDC)  # Validates landmark directory
    self.landmarkDirectoryDC.connect('currentPathChanged(QString)', self.onLandmarkDirectoryChangedDC)  # Handles landmark path changes
    self.outputDirectoryDC.connect('validInputChanged(bool)', self.onParameterSelectDC)  # Validates output directory
    self.outputDirectoryDC.connect('currentPathChanged(QString)', self.onOutputDirectoryChangedDC)  # Handles output path changes
    self.applyButtonDC.connect('clicked(bool)', self.onDCApplyButton)  # Connects run button to execution handler
    self.textureDirectoryDC.connect('validInputChanged(bool)', self.onParameterSelectDC)  # Validates texture directory
    self.textureDirectoryDC.connect('currentPathChanged(QString)', self.onTextureDirectoryChangedDC)  # Handles texture path changes
    self.blenderExeEdit.connect('validInputChanged(bool)', self.onParameterSelectDC)  # Validates Blender executable path
    self.cancelButtonDC.connect('clicked(bool)', self.onCancelOperationDC)  # Connects cancel button to handler
    self.atlasModelOverride.connect('validInputChanged(bool)', self.onParameterSelectDC)
    self.atlasLandmarkOverride.connect('validInputChanged(bool)', self.onParameterSelectDC)

    # Restores previously saved directory paths from persistent settings
    self.restoreSavedDirectories()  # Loads saved paths for user convenience
    self.autoDetectBlender()  # Auto-detect Blender if not restored


    ################################### Visualize Tab ###################################
    if SHOW_VISUALIZE_RESULTS:
      # Layout within the tab
      visualizeWidget=ctk.ctkCollapsibleButton()
      visualizeWidgetLayout = qt.QFormLayout(visualizeWidget)
      visualizeWidget.text = "Visualize Results"
      visualizeWidget.setStyleSheet(ColorTheme.getHeaderStyle())
      visualizeTabLayout.addRow(visualizeWidget)

    #
      # Visualization mode selection
      #
      self.visualizationModeGroup = qt.QGroupBox("Visualization Mode")
      self.visualizationModeGroupLayout = qt.QHBoxLayout(self.visualizationModeGroup)
      visualizeWidgetLayout.addRow(self.visualizationModeGroup)

      self.visualizeHeatmapRadio = qt.QRadioButton("Heatmap")
      self.visualizeHeatmapRadio.setChecked(True)
      self.visualizeInterpolationRadio = qt.QRadioButton("Shape Interpolation")

      self.visualizationModeButtonGroup = qt.QButtonGroup()
      self.visualizationModeButtonGroup.addButton(self.visualizeHeatmapRadio)
      self.visualizationModeButtonGroup.addButton(self.visualizeInterpolationRadio)
      self.visualizationModeGroupLayout.addWidget(self.visualizeHeatmapRadio)
      self.visualizationModeGroupLayout.addWidget(self.visualizeInterpolationRadio)


      #
      # --- Frame for Heatmap Visualization ---
      #
      self.heatmapFrame = qt.QFrame()
      self.heatmapFrameLayout = qt.QFormLayout(self.heatmapFrame)
      visualizeWidgetLayout.addRow(self.heatmapFrame)

      # Select output model
      self.meshSelect = slicer.qMRMLNodeComboBox()
      self.meshSelect.setStyleSheet(ColorTheme.getComboBoxStyle())
      self.meshSelect.nodeTypes = (("vtkMRMLModelNode"), "")
      self.meshSelect.setToolTip("Select model node with result arrays")
      self.meshSelect.selectNodeUponCreation = False
      self.meshSelect.noneEnabled = True
      self.meshSelect.addEnabled = False
      self.meshSelect.removeEnabled = False
      self.meshSelect.showHidden = False
      self.meshSelect.setMRMLScene(slicer.mrmlScene)
      self.heatmapFrameLayout.addRow("Result Model: ", self.meshSelect)

      # Select Subject ID
      self.subjectIDBox=qt.QComboBox()
      self.subjectIDBox.setStyleSheet(ColorTheme.getComboBoxStyle())
      self.subjectIDBox.enabled = False
      self.heatmapFrameLayout.addRow("Subject ID: ", self.subjectIDBox)

      # --- Frame for Interpolation Visualization (bring back) ---
      self.interpolationFrame = qt.QFrame()
      self.interpolationFrameLayout = qt.QFormLayout(self.interpolationFrame)
      self.interpolationFrame.setVisible(False)  # hidden by default
      visualizeWidgetLayout.addRow(self.interpolationFrame)

      # Atlas model (target of interpolation)
      self.atlasModelSelect = slicer.qMRMLNodeComboBox()
      self.atlasModelSelect.setStyleSheet(ColorTheme.getComboBoxStyle())
      self.atlasModelSelect.nodeTypes = (("vtkMRMLModelNode"), "")
      self.atlasModelSelect.setToolTip("Select the atlas or mean shape model")
      self.atlasModelSelect.selectNodeUponCreation = False
      self.atlasModelSelect.noneEnabled = True
      self.atlasModelSelect.addEnabled = False
      self.atlasModelSelect.removeEnabled = False
      self.atlasModelSelect.showHidden = False
      self.atlasModelSelect.setMRMLScene(slicer.mrmlScene)
      self.interpolationFrameLayout.addRow("Atlas Model: ", self.atlasModelSelect)

      # Directory & file selector for a resampled subject
      self.visOriginalModelDirSelector = ctk.ctkPathLineEdit()
      self.visOriginalModelDirSelector.filters = ctk.ctkPathLineEdit.Dirs
      self.visOriginalModelDirSelector.setToolTip("Select the directory of resampled models")
      self.interpolationFrameLayout.addRow("Resampled Model Directory:", self.visOriginalModelDirSelector)

      self.visOriginalModelFileSelector = qt.QComboBox()
      self.visOriginalModelFileSelector.setStyleSheet(ColorTheme.getComboBoxStyle())
      self.visOriginalModelFileSelector.setToolTip("Select a resampled subject model from the directory above")
      self.visOriginalModelFileSelector.enabled = False
      self.interpolationFrameLayout.addRow("Resampled Subject Model:", self.visOriginalModelFileSelector)

      # Interpolation slider
      self.interpolationSlider = ctk.ctkSliderWidget()
      self.interpolationSlider.minimum = 0.0
      self.interpolationSlider.maximum = 1.0
      self.interpolationSlider.singleStep = 0.01  # Set after min/max to avoid bounds issues
      self.interpolationSlider.value = 0.0
      self.interpolationSlider.setToolTip("Interpolate between original model (0.0) and atlas model (1.0)")
      self.interpolationSlider.enabled = False
      self.interpolationFrameLayout.addRow("Interpolation (Original to Atlas):", self.interpolationSlider)


      self.previewTextureCombo = qt.QComboBox()
      self.previewTextureCombo.setStyleSheet(ColorTheme.getComboBoxStyle())
      self.previewTextureCombo.setToolTip("Preview a xbaked atlas-space PNG on the atlas model.")
      visualizeWidgetLayout.addRow("Preview baked texture:", self.previewTextureCombo)
      self.previewTextureCombo.connect("currentIndexChanged(int)", self.onPreviewTextureSelected)
    
      # Add spacing before the visualization button
      visualizeWidgetLayout.addRow(" ", qt.QLabel())


      #
      # Start Visualization Button (at bottom)
      #
      self.startVisualizationButton = qt.QPushButton("Start Visualization")
      self.startVisualizationButton.toolTip = "Prepare the 3D scene for visualization and show markups"
      self.startVisualizationButton.setStyleSheet(ColorTheme.getButtonStyle('secondary'))
      visualizeWidgetLayout.addRow(self.startVisualizationButton)

      self.lastBakedTexturesPath = None
      self.tabsWidget.connect('currentChanged(int)', self.onTabChanged)
    

      # Connections
      self.meshSelect.connect("currentNodeChanged(vtkMRMLNode*)", self.onVisualizeMeshSelect)
      self.subjectIDBox.connect("currentIndexChanged(int)", self.onSubjectIDSelect)
      self.visualizeHeatmapRadio.connect("toggled(bool)", self.onVisualizationModeChanged)
      # MODIFIED Connections for new widgets
      self.atlasModelSelect.connect("currentNodeChanged(vtkMRMLNode*)", self.onInterpolationInputChanged)
      self.visOriginalModelDirSelector.connect("currentPathChanged(QString)", self.onVisOriginalModelDirChanged)
      self.visOriginalModelFileSelector.connect("currentIndexChanged(int)", self.onVisOriginalModelFileSelected)
      self.interpolationSlider.connect("valueChanged(double)", self.onInterpolationSliderChanged)
      self.tabsWidget.connect('currentChanged(int)', self.onTabChanged)

    # Step 1: Multi-texture clustering section
    clusteringWidget = ctk.ctkCollapsibleButton()
    clusteringWidget.setSizePolicy(qt.QSizePolicy.Preferred, qt.QSizePolicy.Maximum)
    clusteringWidget.setStyleSheet(ColorTheme.getHeaderStyle())
    clusteringWidgetLayout = qt.QFormLayout(clusteringWidget)
    clusteringWidgetLayout.setVerticalSpacing(4)
    clusteringWidgetLayout.setHorizontalSpacing(8)
    clusteringWidgetLayout.setFormAlignment(qt.Qt.AlignTop)
    clusteringWidgetLayout.setLabelAlignment(qt.Qt.AlignLeft | qt.Qt.AlignVCenter)
    clusteringWidget.text = "Step 1: Multi-Texture Clustering"
    multiRecolorTabLayout.addRow(clusteringWidget)

    # Atlas model selector for MultiRecolor
    self.multiRecolorAtlasModelSelect = slicer.qMRMLNodeComboBox()
    self.multiRecolorAtlasModelSelect.setStyleSheet(ColorTheme.getComboBoxStyle())
    self.multiRecolorAtlasModelSelect.nodeTypes = (("vtkMRMLModelNode"), "")
    self.multiRecolorAtlasModelSelect.setToolTip("Select the atlas model for multi-texture analysis")
    self.multiRecolorAtlasModelSelect.setMRMLScene(slicer.mrmlScene)
    clusteringWidgetLayout.addRow("Model: ", self.multiRecolorAtlasModelSelect)

    # Texture directory selector for MultiRecolor
    self.multiRecolorTextureDirectorySelector = ctk.ctkPathLineEdit()
    self.multiRecolorTextureDirectorySelector.filters = ctk.ctkPathLineEdit.Dirs
    self.multiRecolorTextureDirectorySelector.setToolTip("Select directory containing texture images")
    clusteringWidgetLayout.addRow("Texture Directory: ", self.multiRecolorTextureDirectorySelector)

    # Mode selection: Clustering vs Subsample-only
    self.multiRecolorModeGroup = qt.QButtonGroup()
    self.multiRecolorClusteringRadio = qt.QRadioButton("Clustering")
    self.multiRecolorClusteringRadio.setChecked(True)  # Default to clustering
    self.multiRecolorClusteringRadio.setToolTip("Perform full clustering with color quantization")
    self.multiRecolorSubsampleOnlyRadio = qt.QRadioButton("Subsample and Average Only")
    self.multiRecolorSubsampleOnlyRadio.setToolTip("Only subsample and average face colors, skip clustering")
    self.multiRecolorModeGroup.addButton(self.multiRecolorClusteringRadio, 0)
    self.multiRecolorModeGroup.addButton(self.multiRecolorSubsampleOnlyRadio, 1)

    modeLayout = qt.QHBoxLayout()
    modeLayout.addWidget(self.multiRecolorClusteringRadio)
    modeLayout.addWidget(self.multiRecolorSubsampleOnlyRadio)
    modeWidget = qt.QWidget()
    modeWidget.setLayout(modeLayout)
    clusteringWidgetLayout.addRow("Mode: ", modeWidget)

    # Normalize luminosity checkbox
    self.multiRecolorNormalizeLuminosityCheckbox = qt.QCheckBox()
    self.multiRecolorNormalizeLuminosityCheckbox.setChecked(False)
    self.multiRecolorNormalizeLuminosityCheckbox.setToolTip("Normalize L* and C* across all textures to reduce lighting variation")
    clusteringWidgetLayout.addRow("Normalize Luminosity: ", self.multiRecolorNormalizeLuminosityCheckbox)

    # Initial clusters for multi-texture analysis
    self.multiRecolorInitialClustersSpin = qt.QSpinBox()
    self.multiRecolorInitialClustersSpin.setRange(2, 64)
    self.multiRecolorInitialClustersSpin.setValue(24)  # Default value for initial clustering
    self.multiRecolorInitialClustersSpin.setToolTip("Number of initial color clusters per texture (2-64)")
    clusteringWidgetLayout.addRow("Initial Clusters: ", self.multiRecolorInitialClustersSpin)

    # Consolidated clusters for multi-texture analysis
    self.multiRecolorConsolidatedClustersSpin = qt.QSpinBox()
    self.multiRecolorConsolidatedClustersSpin.setRange(2, 64)
    self.multiRecolorConsolidatedClustersSpin.setValue(8)  # Default value for consolidated clustering
    self.multiRecolorConsolidatedClustersSpin.setToolTip("Number of consolidated clusters after hierarchical merging (2-64, must be ≤ Initial Clusters)")
    clusteringWidgetLayout.addRow("Consolidated Clusters: ", self.multiRecolorConsolidatedClustersSpin)

    # Number of faces for subsampling
    self.multiRecolorNumSubsampledFacesSpin = qt.QSpinBox()
    self.multiRecolorNumSubsampledFacesSpin.setRange(100, 100000)
    self.multiRecolorNumSubsampledFacesSpin.setValue(10000)  # Default value for subsampling
    self.multiRecolorNumSubsampledFacesSpin.setSingleStep(1000)  # Increment by 1000
    self.multiRecolorNumSubsampledFacesSpin.setToolTip("Number of faces to subsample for clustering (uniformly distributed by surface distance)")
    clusteringWidgetLayout.addRow("Number of Faces (Subsampling): ", self.multiRecolorNumSubsampledFacesSpin)

    # Disable subsampling checkbox
    self.multiRecolorDisableSubsamplingCheckbox = qt.QCheckBox()
    self.multiRecolorDisableSubsamplingCheckbox.setChecked(False)
    self.multiRecolorDisableSubsamplingCheckbox.setToolTip("If checked, use all faces for clustering instead of subsampling")
    clusteringWidgetLayout.addRow("Disable Subsampling (Use All Faces): ", self.multiRecolorDisableSubsamplingCheckbox)

    # Neighbor Average checkbox
    self.multiRecolorNeighborAverageCheckbox = qt.QCheckBox()
    self.multiRecolorNeighborAverageCheckbox.setChecked(False)
    self.multiRecolorNeighborAverageCheckbox.setToolTip("If checked, use average color of neighboring faces instead of individual face color for clustering")
    clusteringWidgetLayout.addRow("Neighbor Average: ", self.multiRecolorNeighborAverageCheckbox)

    # Cluster button
    self.clusterButton = qt.QPushButton("Cluster")
    self.clusterButton.setToolTip("Process all textures and create color clusters")
    self.clusterButton.enabled = False
    self.clusterButton.setStyleSheet(ColorTheme.getButtonStyle('primary'))
    clusteringWidgetLayout.addRow(self.clusterButton)

    # Progress bar for clustering
    self.clusteringProgressBar = qt.QProgressBar()
    self.clusteringProgressBar.setVisible(False)
    clusteringWidgetLayout.addRow("Progress: ", self.clusteringProgressBar)

    # Clustering log info
    self.clusteringLogInfo = qt.QTextEdit()
    self.clusteringLogInfo.setMaximumHeight(100)
    self.clusteringLogInfo.setReadOnly(True)
    clusteringWidgetLayout.addRow("Log: ", self.clusteringLogInfo)

    # Step 2: Individual visualization section
    individualWidget = ctk.ctkCollapsibleButton()
    individualWidget.setSizePolicy(qt.QSizePolicy.Preferred, qt.QSizePolicy.Maximum)
    individualWidget.setStyleSheet(ColorTheme.getHeaderStyle())
    individualWidgetLayout = qt.QFormLayout(individualWidget)
    individualWidgetLayout.setVerticalSpacing(4)
    individualWidgetLayout.setHorizontalSpacing(8)
    individualWidgetLayout.setFormAlignment(qt.Qt.AlignTop)
    individualWidgetLayout.setLabelAlignment(qt.Qt.AlignLeft | qt.Qt.AlignVCenter)
    individualWidget.text = "Step 2: Individual Visualization"
    multiRecolorTabLayout.addRow(individualWidget)

    # Texture selector dropdown
    self.individualTextureSelector = qt.QComboBox()
    self.individualTextureSelector.setStyleSheet(ColorTheme.getComboBoxStyle())
    self.individualTextureSelector.setToolTip("Select a texture to visualize with the clustered palette")
    self.individualTextureSelector.enabled = False
    individualWidgetLayout.addRow("Select Texture: ", self.individualTextureSelector)

    # Raw texture option for individual visualization
    self.individualRawTextureCheckbox = qt.QCheckBox()
    self.individualRawTextureCheckbox.setChecked(False)
    self.individualRawTextureCheckbox.setEnabled(False)
    self.individualRawTextureCheckbox.setToolTip("Display original texture without face averaging or quantization")
    individualWidgetLayout.addRow("Raw Texture: ", self.individualRawTextureCheckbox)

    # Apply texture button for individual visualization
    self.applyIndividualTextureButton = qt.QPushButton("Apply Texture")
    self.applyIndividualTextureButton.setToolTip("Apply selected texture with clustered palette")
    self.applyIndividualTextureButton.enabled = False
    self.applyIndividualTextureButton.setStyleSheet(ColorTheme.getButtonStyle('primary'))
    individualWidgetLayout.addRow(self.applyIndividualTextureButton)

    # Progress bar for individual visualization
    self.individualProgressBar = qt.QProgressBar()
    self.individualProgressBar.setVisible(False)
    individualWidgetLayout.addRow("Progress: ", self.individualProgressBar)

    # Individual visualization log info
    self.individualLogInfo = qt.QTextEdit()
    self.individualLogInfo.setMaximumHeight(80)
    self.individualLogInfo.setReadOnly(True)
    individualWidgetLayout.addRow("Log: ", self.individualLogInfo)

    # Step 3: Population analysis section
    populationWidget = ctk.ctkCollapsibleButton()
    populationWidget.setSizePolicy(qt.QSizePolicy.Preferred, qt.QSizePolicy.Maximum)
    populationWidget.setStyleSheet(ColorTheme.getHeaderStyle())
    populationWidgetLayout = qt.QFormLayout(populationWidget)
    populationWidgetLayout.setVerticalSpacing(4)
    populationWidgetLayout.setHorizontalSpacing(8)
    populationWidgetLayout.setFormAlignment(qt.Qt.AlignTop)
    populationWidgetLayout.setLabelAlignment(qt.Qt.AlignLeft | qt.Qt.AlignVCenter)
    populationWidget.text = "Step 3: Population Analysis"
    multiRecolorTabLayout.addRow(populationWidget)

    # Dimensionality reduction method selection
    self.dimReductionMethodGroup = qt.QButtonGroup()
    self.pcaRadioButton = qt.QRadioButton("PCA")
    self.pcaRadioButton.setChecked(True)  # Default selection
    self.umapRadioButton = qt.QRadioButton("UMAP")
    self.icaRadioButton = qt.QRadioButton("ICA")
    self.dimReductionMethodGroup.addButton(self.pcaRadioButton, 0)
    self.dimReductionMethodGroup.addButton(self.umapRadioButton, 1)
    self.dimReductionMethodGroup.addButton(self.icaRadioButton, 2)

    dimReductionLayout = qt.QHBoxLayout()
    dimReductionLayout.addWidget(self.pcaRadioButton)
    dimReductionLayout.addWidget(self.umapRadioButton)
    dimReductionLayout.addWidget(self.icaRadioButton)
    dimReductionWidget = qt.QWidget()
    dimReductionWidget.setLayout(dimReductionLayout)
    populationWidgetLayout.addRow("Dimensionality Reduction: ", dimReductionWidget)

    # Number of PCs for PCA (similar to PCA Morphospace tab)
    self.multiRecolorNumPCsSpin = qt.QSpinBox()
    self.multiRecolorNumPCsSpin.setMinimum(2)
    self.multiRecolorNumPCsSpin.setMaximum(20)
    self.multiRecolorNumPCsSpin.setValue(2)
    self.multiRecolorNumPCsSpin.setToolTip("Number of principal components to compute (only for PCA)")
    self.multiRecolorNumPCsSpin.connect("valueChanged(int)", self.onMultiRecolorNumPCsChanged)
    populationWidgetLayout.addRow("Number of PCs: ", self.multiRecolorNumPCsSpin)

    # PC axis selection for plotting
    axisLayout = qt.QHBoxLayout()

    axisLayout.addWidget(qt.QLabel("X-axis:"))
    self.multiRecolorXAxisCombo = qt.QComboBox()
    self.multiRecolorXAxisCombo.setToolTip("Select PC for X-axis")
    self.multiRecolorXAxisCombo.enabled = False
    self.multiRecolorXAxisCombo.setStyleSheet(ColorTheme.getComboBoxStyle())
    axisLayout.addWidget(self.multiRecolorXAxisCombo)

    axisLayout.addWidget(qt.QLabel("Y-axis:"))
    self.multiRecolorYAxisCombo = qt.QComboBox()
    self.multiRecolorYAxisCombo.setToolTip("Select PC for Y-axis")
    self.multiRecolorYAxisCombo.enabled = False
    self.multiRecolorYAxisCombo.setStyleSheet(ColorTheme.getComboBoxStyle())
    axisLayout.addWidget(self.multiRecolorYAxisCombo)

    axisWidget = qt.QWidget()
    axisWidget.setLayout(axisLayout)
    populationWidgetLayout.addRow("Plot Axes: ", axisWidget)

    # Compare textures button
    self.compareTexturesButton = qt.QPushButton("Compare Textures")
    self.compareTexturesButton.setToolTip("Analyze all textures and create population comparison plot")
    self.compareTexturesButton.enabled = False
    self.compareTexturesButton.setStyleSheet(ColorTheme.getButtonStyle('primary'))
    populationWidgetLayout.addRow(self.compareTexturesButton)

    # Progress bar for population analysis
    self.populationProgressBar = qt.QProgressBar()
    self.populationProgressBar.setVisible(False)
    populationWidgetLayout.addRow("Progress: ", self.populationProgressBar)

    # Population analysis log info
    self.populationLogInfo = qt.QTextEdit()
    self.populationLogInfo.setMaximumHeight(80)
    self.populationLogInfo.setReadOnly(True)
    populationWidgetLayout.addRow("Log: ", self.populationLogInfo)

    # Connect MultiRecolor UI events
    self.multiRecolorAtlasModelSelect.connect("currentNodeChanged(vtkMRMLNode*)", self.onMultiRecolorParameterChanged)
    self.multiRecolorTextureDirectorySelector.connect("currentPathChanged(QString)", self.onMultiRecolorParameterChanged)
    self.multiRecolorClusteringRadio.connect("toggled(bool)", self.onMultiRecolorModeChanged)
    self.multiRecolorSubsampleOnlyRadio.connect("toggled(bool)", self.onMultiRecolorModeChanged)
    self.multiRecolorNormalizeLuminosityCheckbox.connect("toggled(bool)", self.onMultiRecolorParameterChanged)
    self.multiRecolorDisableSubsamplingCheckbox.connect("toggled(bool)", self.onMultiRecolorDisableSubsamplingChanged)
    self.multiRecolorInitialClustersSpin.connect("valueChanged(int)", self.onMultiRecolorClusterCountChanged)
    self.multiRecolorConsolidatedClustersSpin.connect("valueChanged(int)", self.onMultiRecolorClusterCountChanged)
    self.clusterButton.connect('clicked(bool)', self.onClusterButton)
    self.individualTextureSelector.connect("currentTextChanged(const QString &)", self.onIndividualTextureChanged)
    self.applyIndividualTextureButton.connect('clicked(bool)', self.onApplyIndividualTextureButton)
    self.compareTexturesButton.connect('clicked(bool)', self.onCompareTexturesButton)
    self.multiRecolorXAxisCombo.connect('currentIndexChanged(int)', self.onMultiRecolorAxisChanged)
    self.multiRecolorYAxisCombo.connect('currentIndexChanged(int)', self.onMultiRecolorAxisChanged)

    # Initialize PC axis selectors with default values
    self.onMultiRecolorNumPCsChanged()

    # Initialize storage for population analysis result
    self.multiRecolorPopulationResult = None

    # ========== Step 4: Morphospace ==========
    morphospaceWidget = ctk.ctkCollapsibleButton()
    morphospaceWidget.text = "Step 4: Morphospace"
    morphospaceWidget.collapsed = True
    multiRecolorTabLayout.addWidget(morphospaceWidget)
    morphospaceWidgetLayout = qt.QFormLayout(morphospaceWidget)

    # Starting texture selector
    self.morphospaceTextureCombo = qt.QComboBox()
    self.morphospaceTextureCombo.setToolTip("Select a texture as the starting point in PCA space")
    self.morphospaceTextureCombo.enabled = False
    self.morphospaceTextureCombo.setStyleSheet(ColorTheme.getComboBoxStyle())
    morphospaceWidgetLayout.addRow("Starting Texture: ", self.morphospaceTextureCombo)

    # X-axis slider (controlled by X-axis PC from Step 3)
    self.morphospaceXSlider = qt.QSlider(qt.Qt.Horizontal)
    self.morphospaceXSlider.setMinimum(0)
    self.morphospaceXSlider.setMaximum(1000)  # Will be set dynamically based on data
    self.morphospaceXSlider.setValue(500)
    self.morphospaceXSlider.setTickInterval(100)
    self.morphospaceXSlider.setTickPosition(qt.QSlider.TicksBelow)
    self.morphospaceXSlider.enabled = False
    self.morphospaceXSlider.setToolTip("Slide to explore color variation along X-axis PC")
    self.morphospaceXSlider.connect('valueChanged(int)', self.onMorphospaceXSliderChanged)
    morphospaceWidgetLayout.addRow("X-axis Position: ", self.morphospaceXSlider)

    # X-axis value label
    self.morphospaceXLabel = qt.QLabel("X: 0.00")
    morphospaceWidgetLayout.addRow("", self.morphospaceXLabel)

    # Y-axis slider (controlled by Y-axis PC from Step 3)
    self.morphospaceYSlider = qt.QSlider(qt.Qt.Horizontal)
    self.morphospaceYSlider.setMinimum(0)
    self.morphospaceYSlider.setMaximum(1000)  # Will be set dynamically based on data
    self.morphospaceYSlider.setValue(500)
    self.morphospaceYSlider.setTickInterval(100)
    self.morphospaceYSlider.setTickPosition(qt.QSlider.TicksBelow)
    self.morphospaceYSlider.enabled = False
    self.morphospaceYSlider.setToolTip("Slide to explore color variation along Y-axis PC")
    self.morphospaceYSlider.connect('valueChanged(int)', self.onMorphospaceYSliderChanged)
    morphospaceWidgetLayout.addRow("Y-axis Position: ", self.morphospaceYSlider)

    # Y-axis value label
    self.morphospaceYLabel = qt.QLabel("Y: 0.00")
    morphospaceWidgetLayout.addRow("", self.morphospaceYLabel)

    # Visualize button
    self.visualizeMorphospaceButton = qt.QPushButton("Visualize Morphospace")
    self.visualizeMorphospaceButton.toolTip = "Start morphospace visualization with selected texture"
    self.visualizeMorphospaceButton.enabled = False
    self.visualizeMorphospaceButton.setStyleSheet(ColorTheme.getButtonStyle('primary'))
    self.visualizeMorphospaceButton.connect('clicked(bool)', self.onVisualizeMorphospace)
    morphospaceWidgetLayout.addRow(self.visualizeMorphospaceButton)

    # Morphospace log info
    self.morphospaceLogInfo = qt.QTextEdit()
    self.morphospaceLogInfo.setMaximumHeight(80)
    self.morphospaceLogInfo.setReadOnly(True)
    morphospaceWidgetLayout.addRow("Log: ", self.morphospaceLogInfo)

    # Connect morphospace events
    self.morphospaceTextureCombo.connect('currentIndexChanged(int)', self.onMorphospaceTextureChanged)

    # Initialize morphospace state variables
    self.morphospaceCurrentPoint = None  # Current point in PCA space
    self.morphospaceStartingPoint = None  # Starting point from selected texture
    self.morphospaceMovingPointSeries = None  # Plot series for the moving point
    self.morphospaceMovingPointTable = None  # Table for the moving point
    self.morphospaceXRange = None  # (min, max) for X-axis PC
    self.morphospaceYRange = None  # (min, max) for Y-axis PC

    # Add vertical spacer so extra space goes below content
    multiRecolorTabLayout.addItem(qt.QSpacerItem(0, 0, qt.QSizePolicy.Minimum, qt.QSizePolicy.Expanding))

    # Initialize MultiRecolor state variables
    self.multiRecolorClusterCenters = None
    self.multiRecolorClusteringPipeline = None
    self.multiRecolorFaceAreas = None
    self.multiRecolorTextureFiles = []
    self.faceAreasCache = {}  # Cache face areas by model node ID
    
    # Restore saved texture directories from previous sessions
    self.restoreTextureDirectories()

    # Setup keyboard shortcuts
    self.setupKeyboardShortcuts()

    # Collapse Data Probe panel by default
    try:
      mainWindow = slicer.util.mainWindow()
      if mainWindow:
        dataProbeWidget = mainWindow.findChild('DataProbeCollapsibleWidget')
        if dataProbeWidget:
          dataProbeWidget.collapsed = True
    except Exception as e:
      print(f"Note: Could not collapse Data Probe panel: {e}")

    # --- Mesh Selection Connections ---
    self.regionMeshSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onRegionSelectionInputChanged)
    self.selectionMarkupSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onMarkupNodeChanged)
    self.mirrorSelectionCheckbox.connect("toggled(bool)", self.onRegionSelectionInputChanged)
    
    self.createMarkupButton.connect('clicked(bool)', self.onCreateOrPlaceMarkup)
    self.clearSelectionButton.connect('clicked(bool)', self.onClearSelection)
    self.applyLandmarkSelectionButton.connect('clicked(bool)', self.onApplyLandmarkSelection)
    self.exportLandmarkSelectionButton.connect('clicked(bool)', self.onExportLandmarkSelection)

    # Initialize button states
    self.onRegionSelectionInputChanged()

  def setupKeyboardShortcuts(self):
    """Setup keyboard shortcuts for the module"""
    try:
      # Create ESC shortcut to exit placement mode
      self.escapeShortcut = qt.QShortcut(qt.QKeySequence("Escape"), slicer.util.mainWindow())
      self.escapeShortcut.connect('activated()', self.onEscapeKey)
      print("Keyboard shortcuts installed: ESC to exit placement mode")
    except Exception as e:
      print(f"Note: Could not setup keyboard shortcuts: {e}")

  def onEscapeKey(self):
    """Handle ESC key press to exit placement mode"""
    try:
      # Check if we're in placement mode
      interactionNode = slicer.app.applicationLogic().GetInteractionNode()

      if interactionNode.GetCurrentInteractionMode() == interactionNode.Place:
        # Get the active placement node (the curve being placed)
        selectionNode = slicer.app.applicationLogic().GetSelectionNode()
        activePlaceNodeID = selectionNode.GetActivePlaceNodeID()

        # Exit placement mode
        interactionNode.SetCurrentInteractionMode(interactionNode.ViewTransform)

        # Get the markup node that was being placed
        markupNode = None
        if activePlaceNodeID:
          markupNode = slicer.mrmlScene.GetNodeByID(activePlaceNodeID)

        # Also check the selector in case it's set there
        if not markupNode:
          markupNode = self.selectionMarkupSelector.currentNode()

        if markupNode:
          numPoints = markupNode.GetNumberOfControlPoints()

          if markupNode.GetClassName() == "vtkMRMLMarkupsClosedCurveNode":
            # Make sure the selector is updated with this node
            if self.selectionMarkupSelector.currentNode() != markupNode:
              self.selectionMarkupSelector.setCurrentNode(markupNode)

            if numPoints < 3:
              slicer.util.warningDisplay(f"Closed curve only has {numPoints} points. Add at least 3 points for selection.")
    except Exception as e:
      print(f"Error handling ESC key: {e}")
      import traceback
      traceback.print_exc()

  def cleanup(self):
    """Cleanup when module is unloaded"""
    # Remove keyboard shortcuts
    if hasattr(self, 'escapeShortcut') and self.escapeShortcut:
      self.escapeShortcut.disconnect('activated()')
      self.escapeShortcut.setParent(None)
      self.escapeShortcut = None

    # Clean up markup observers
    if hasattr(self, '_currentMarkupNode') and self._currentMarkupNode:
      if hasattr(self, '_markupObserver'):
        self._currentMarkupNode.RemoveObserver(self._markupObserver)
      if hasattr(self, '_markupEndInteractionObserver'):
        self._currentMarkupNode.RemoveObserver(self._markupEndInteractionObserver)
      if hasattr(self, '_markupPointAddedObserver'):
        self._currentMarkupNode.RemoveObserver(self._markupPointAddedObserver)
      if hasattr(self, '_markupPointRemovedObserver'):
        self._currentMarkupNode.RemoveObserver(self._markupPointRemovedObserver)

  def restoreTextureDirectories(self):
    """Restore texture directory paths from saved settings"""
    # Restore DeCA texture directory (only if empty)
    savedDecaTexture = self.settings.value("textureDirectoryDC", "")
    if savedDecaTexture and not self.textureDirectoryDC.currentPath:
      self.textureDirectoryDC.setCurrentPath(savedDecaTexture)
    
    # Restore MultiRecolor texture directory
    savedMultiRecolor = self.settings.value("multiRecolorTextureDirectory", "")
    if savedMultiRecolor and not self.multiRecolorTextureDirectorySelector.currentPath:
      self.multiRecolorTextureDirectorySelector.setCurrentPath(savedMultiRecolor)
  
  def saveTextureDirectory(self, key, path):
    """Save a texture directory path to settings"""
    if path:
      self.settings.setValue(key, path)


  def autoDetectBlender(self):
    """Automatically detect and set Blender executable path if not already set."""
    try:
      # Only auto-detect if the field is empty
      if not self.blenderExeEdit.currentPath:
        logic = InterDeCALogic()
        blender_path = logic.findBlenderExecutable()
        if blender_path:
          self.blenderExeEdit.setCurrentPath(blender_path)
          print(f"Auto-detected Blender at: {blender_path}")
        else:
          print("Blender not found during auto-detection. Will attempt installation when needed.")
    except Exception as e:
      print(f"Error during Blender auto-detection: {e}")

  ################################### GUI Support Functions
  
  def validateDirectory(self, directory, extensions, label, dirType="files"):
    """Validate directory contents and update status label"""
    if not directory or not os.path.isdir(directory):
      label.setText("No directory selected")
      label.setStyleSheet(ColorTheme.getValidationLabelStyle('neutral'))
      return False, 0
    
    try:
      files = os.listdir(directory)
      matching_files = []
      for f in files:
        if not f.startswith('.'):
          ext = os.path.splitext(f)[1].lower()
          if ext in extensions:
            matching_files.append(f)
      
      count = len(matching_files)
      if count == 0:
        label.setText("No files found")
        label.setStyleSheet(ColorTheme.getValidationLabelStyle('error'))
        return False, 0
      else:
        label.setText(f"{count} found")
        label.setStyleSheet(ColorTheme.getValidationLabelStyle('success'))
        return True, count
    except Exception as e:
      label.setText(f"Error reading directory")
      label.setStyleSheet(ColorTheme.getValidationLabelStyle('error'))
      return False, 0
  
  def onMeshDirectoryChangedDC(self, directory):
    """Validates mesh directory when path changes.

    Checks for supported 3D model file formats and updates validation status.

    Args:
      directory: Path to the directory containing mesh files
    """
    model_extensions = ['.ply', '.stl', '.obj', '.vtk', '.vtp']  # Defines supported model formats
    self.validateDirectory(directory, model_extensions, self.meshValidationLabelDC, "models")  # Validates directory contents
    # Updates texture matching if textures are already loaded
    if self.textureDirectoryDC.currentPath:  # Checks if textures exist
      self.validateTextureMatching()
    # Save the directory path
    self.settings.setValue("meshDirectory", directory)
    self.onParameterSelectDC()
  
  def onLandmarkDirectoryChangedDC(self, directory):
    """Validates landmark directory when path changes.

    Checks for supported landmark file formats and updates validation status.

    Args:
      directory: Path to the directory containing landmark files
    """
    landmark_extensions = ['.fcsv', '.json', '.mrk.json']  # Defines supported landmark formats
    self.validateDirectory(directory, landmark_extensions, self.landmarkValidationLabelDC, "landmarks")  # Validates directory contents
    # Updates texture matching if landmarks are paired with textures
    if self.textureDirectoryDC.currentPath:  # Checks if textures exist
      self.validateTextureMatching()
    # Save the directory path
    self.settings.setValue("landmarkDirectory", directory)
    self.onParameterSelectDC()
  
  def onTextureDirectoryChangedDC(self, directory):
    """Validate texture directory when changed"""
    texture_extensions = ['.png', '.tiff', '.tif']
    valid, count = self.validateDirectory(directory, texture_extensions, self.textureValidationLabelDC, "textures")

    # If we have textures and models/landmarks, check for matches
    if valid and count > 0:
      self.validateTextureMatching()

    # Save the directory path for persistence
    self.saveTextureDirectory("textureDirectoryDC", directory)
    self.onParameterSelectDC()

  def restoreSavedDirectories(self):
    """Restore previously saved directory paths"""
    # Restore mesh directory
    savedMeshDir = self.settings.value("meshDirectory", "")
    if savedMeshDir and os.path.exists(savedMeshDir):
      self.meshDirectoryDC.setCurrentPath(savedMeshDir)

    # Restore landmark directory
    savedLandmarkDir = self.settings.value("landmarkDirectory", "")
    if savedLandmarkDir and os.path.exists(savedLandmarkDir):
      self.landmarkDirectoryDC.setCurrentPath(savedLandmarkDir)

    # Restore texture directory
    savedTextureDir = self.settings.value("textureDirectory", "")
    if savedTextureDir and os.path.exists(savedTextureDir):
      self.textureDirectoryDC.setCurrentPath(savedTextureDir)

    # Restore output directory
    savedOutputDir = self.settings.value("outputDirectory", "")
    if savedOutputDir and os.path.exists(savedOutputDir):
      self.outputDirectoryDC.setCurrentPath(savedOutputDir)

  def onOutputDirectoryChangedDC(self, directory):
    """Save output directory when changed"""
    # Save the directory path
    self.settings.setValue("outputDirectory", directory)
    self.onParameterSelectDC()

  def validateTextureMatching(self):
    """Check if texture files match available models/landmarks"""
    textureDir = self.textureDirectoryDC.currentPath
    modelDir = self.meshDirectoryDC.currentPath
    landmarkDir = self.landmarkDirectoryDC.currentPath
    
    if not (textureDir and os.path.isdir(textureDir)):
      return
    
    # Get texture file basenames (without extension)
    texture_files = []
    for f in os.listdir(textureDir):
      if f.lower().endswith(('.png', '.tiff', '.tif')) and not f.startswith('.'):
        texture_files.append(os.path.splitext(f)[0])
    
    if not texture_files:
      return
    
    # Get model basenames if available
    model_basenames = set()
    if modelDir and os.path.isdir(modelDir):
      for f in os.listdir(modelDir):
        if not f.startswith('.') and os.path.splitext(f)[1].lower() in ['.ply', '.stl', '.obj', '.vtk', '.vtp']:
          model_basenames.add(os.path.splitext(f)[0])
    
    # Get landmark basenames if available
    landmark_basenames = set()
    if landmarkDir and os.path.isdir(landmarkDir):
      for f in os.listdir(landmarkDir):
        if not f.startswith('.') and os.path.splitext(f)[1].lower() in ['.fcsv', '.json']:
          # Handle .mrk.json files
          base = f
          while os.path.splitext(base)[1].lower() in ['.mrk', '.json', '.fcsv']:
            base = os.path.splitext(base)[0]
          landmark_basenames.add(base)
    
    # Check matches
    if model_basenames or landmark_basenames:
      subject_basenames = model_basenames.union(landmark_basenames)
      matching_textures = [t for t in texture_files if t in subject_basenames]
      
      if matching_textures:
        match_count = len(matching_textures)
        total_subjects = len(subject_basenames)
        self.textureValidationLabelDC.setText(f"{len(texture_files)} found")
        if match_count == total_subjects:
          self.textureValidationLabelDC.setStyleSheet(ColorTheme.getValidationLabelStyle('success'))
        else:
          self.textureValidationLabelDC.setStyleSheet(ColorTheme.getValidationLabelStyle('warning'))
      else:
        self.textureValidationLabelDC.setText(f"{len(texture_files)} found (no matches)")
        self.textureValidationLabelDC.setStyleSheet(ColorTheme.getValidationLabelStyle('error'))
  
  def updateProgressDC(self, value, text="", showCancel=False):
    """Update progress bar (text moved to logs)"""
    if not self.progressWidgetDC.isVisible():
      self.progressWidgetDC.setVisible(True)

    self.progressBarDC.setValue(value)
    # Add progress text to log 
    if text:
      self.logInfoDC.appendPlainText(text)

    self.cancelButtonDC.setVisible(showCancel)

    # Process events to update UI
    slicer.app.processEvents()
  
  def resetProgressDC(self):
    """Reset progress indicators"""
    self.progressWidgetDC.setVisible(False)
    self.progressBarDC.setValue(0)
    self.cancelButtonDC.setVisible(False)
    self.currentOperation = None
  
  def onCancelOperationDC(self):
    """Handle operation cancellation"""
    if self.currentOperation:
      self.logInfoDC.appendPlainText("Operation cancelled by user")
      self.resetProgressDC()
      self.applyButtonDC.enabled = True

  def updatePackageStatus(self):
    """Update the package status display"""
    try:
      missing_packages, all_available = checkAndOfferPackageInstallation()
      
      if not missing_packages:
        status_text = f"✅ All packages available ({len(all_available)} installed)"
        self.packageStatusLabel.setStyleSheet("color: green; font-weight: bold;")
        self.installPackagesButton.setText("Check Packages")
        self.installPackagesButton.enabled = True
        self.installAllButton.enabled = False
      else:
        status_text = f"⚠️ {len(missing_packages)} packages missing, {len(all_available)} available"
        self.packageStatusLabel.setStyleSheet("color: orange; font-weight: bold;")
        self.installPackagesButton.setText(f"Install {len(missing_packages)} Missing Packages")
        self.installPackagesButton.enabled = True
        self.installAllButton.enabled = True
      
      self.packageStatusLabel.setText(status_text)
      
    except Exception as e:
      self.packageStatusLabel.setText("❌ Error checking packages")
      self.packageStatusLabel.setStyleSheet("color: red; font-weight: bold;")
      print(f"Error updating package status: {e}")

  def onInstallPackagesClicked(self):
    """Handle install packages button click"""
    try:
      import subprocess
      import sys
      
      # Get missing packages
      missing_packages, _ = checkAndOfferPackageInstallation()
      
      if not missing_packages:
        qt.QMessageBox.information(None, "Package Status", 
                                 "All recommended packages are already installed!")
        self.updatePackageStatus()
        return
      
      # Create confirmation dialog
      package_names = [pkg['pip_name'] for pkg in missing_packages]
      message = f"Install the following packages?\n\n{', '.join(package_names)}\n\nThis may take a few minutes."
      
      reply = qt.QMessageBox.question(None, "Install Packages", message,
                                    qt.QMessageBox.Yes | qt.QMessageBox.No)
      
      if reply == qt.QMessageBox.Yes:
        # Disable buttons during installation
        self.installPackagesButton.enabled = False
        self.installAllButton.enabled = False
        self.installPackagesButton.setText("Installing...")
        
        # Install packages with detailed logging
        import datetime
        print(f"\n--- Package Installation Started ({datetime.datetime.now().strftime('%H:%M:%S')}) ---")
        print(f"Installing {len(missing_packages)} missing packages...")
        
        for i, pkg in enumerate(missing_packages, 1):
          try:
            print(f"\n[{i}/{len(missing_packages)}] Installing {pkg['pip_name']}...")
            print(f"  📋 Description: {pkg['description']}")
            print(f"  🔧 Command: pip install {pkg['pip_name']}")
            
            result = subprocess.run([sys.executable, "-m", "pip", "install", pkg['pip_name']], 
                                  capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
              print(f"  ✅ Successfully installed {pkg['pip_name']}")
              if result.stdout:
                # Show relevant installation info (not full verbose output)
                lines = result.stdout.strip().split('\n')
                for line in lines:
                  if 'Successfully installed' in line or 'Requirement already satisfied' in line:
                    print(f"     {line}")
            else:
              print(f"  ❌ Failed to install {pkg['pip_name']}")
              if result.stderr:
                print(f"     Error: {result.stderr.strip()}")
          
          except subprocess.TimeoutExpired:
            print(f"  ⏰ Timeout installing {pkg['pip_name']} (exceeded 5 minutes)")
          except Exception as e:
            print(f"  ❌ Error installing {pkg['pip_name']}: {e}")
        
        print(f"\n--- Package Installation Completed ({datetime.datetime.now().strftime('%H:%M:%S')}) ---")
        print("🔄 IMPORTANT: Please restart 3D Slicer for new packages to be recognized!")
        print("   After restart, check the package status to confirm availability.")
        
        # Update status and re-enable buttons
        self.updatePackageStatus()
        
        # Show completion message with restart instruction
        message = ("Package installation completed!\n\n"
                  "📋 Check the Python console for detailed installation logs.\n\n"
                  "🔄 IMPORTANT: Please restart 3D Slicer for the new packages\n"
                  "    to be properly recognized and available for use.\n\n"
                  "After restarting, the package status will update automatically.")
        
        qt.QMessageBox.information(None, "Installation Complete - Restart Required", message)
      
    except Exception as e:
      print(f"Error in package installation: {e}")
      qt.QMessageBox.critical(None, "Installation Error", 
                            f"Failed to install packages: {str(e)}")
      self.updatePackageStatus()

  def onInstallAllPackagesClicked(self):
    """Handle install all packages button click"""
    try:
      import subprocess
      import sys
      
      # List of all recommended packages
      all_packages = ['numpy', 'scikit-learn', 'umap-learn', 'scikit-image', 'imageio']
      
      message = f"Install all recommended packages for full InterDeCA functionality?\n\n{', '.join(all_packages)}\n\nThis may take several minutes."
      
      reply = qt.QMessageBox.question(None, "Install All Packages", message,
                                    qt.QMessageBox.Yes | qt.QMessageBox.No)
      
      if reply == qt.QMessageBox.Yes:
        # Disable buttons during installation
        self.installPackagesButton.enabled = False
        self.installAllButton.enabled = False
        self.installAllButton.setText("Installing All...")
        
        # Install all packages with detailed logging
        import datetime
        print(f"\n--- Full Package Installation Started ({datetime.datetime.now().strftime('%H:%M:%S')}) ---")
        print(f"Installing all {len(all_packages)} recommended packages for InterDeCA...")
        
        for i, pkg_name in enumerate(all_packages, 1):
          try:
            print(f"\n[{i}/{len(all_packages)}] Installing {pkg_name}...")
            print(f"  🔧 Command: pip install {pkg_name}")
            
            result = subprocess.run([sys.executable, "-m", "pip", "install", pkg_name], 
                                  capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
              print(f"  ✅ Successfully installed {pkg_name}")
              if result.stdout:
                # Show relevant installation info
                lines = result.stdout.strip().split('\n')
                for line in lines:
                  if 'Successfully installed' in line or 'Requirement already satisfied' in line:
                    print(f"     {line}")
            else:
              print(f"  ❌ Failed to install {pkg_name}")
              if result.stderr:
                print(f"     Error: {result.stderr.strip()}")
          
          except subprocess.TimeoutExpired:
            print(f"  ⏰ Timeout installing {pkg_name} (exceeded 5 minutes)")
          except Exception as e:
            print(f"  ❌ Error installing {pkg_name}: {e}")
        
        print(f"\n--- Full Package Installation Completed ({datetime.datetime.now().strftime('%H:%M:%S')}) ---")
        print("🔄 IMPORTANT: Please restart 3D Slicer for new packages to be recognized!")
        print("   After restart, all InterDeCA features will be fully available.")
        
        # Update status and re-enable buttons
        self.updatePackageStatus()
        
        # Show completion message with restart instruction
        message = ("All recommended packages installation completed!\n\n"
                  "📋 Check the Python console for detailed installation logs.\n\n"
                  "🔄 IMPORTANT: Please restart 3D Slicer for the new packages\n"
                  "    to be properly recognized and available for use.\n\n"
                  "After restarting, all InterDeCA features will be fully available.")
        
        qt.QMessageBox.information(None, "Installation Complete - Restart Required", message)
      
    except Exception as e:
      print(f"Error in package installation: {e}")
      qt.QMessageBox.critical(None, "Installation Error", 
                            f"Failed to install packages: {str(e)}")
      self.updatePackageStatus()
      # Note: Actual cancellation logic would depend on the specific operation

  # --- Region Selection Methods ---
  
  def onRegionSelectionInputChanged(self):
    """Update button states when region selection inputs change"""
    hasModel = bool(self.regionMeshSelector.currentNode())

    # Update selection button
    hasMarkup = bool(self.selectionMarkupSelector.currentNode())
    self.applyLandmarkSelectionButton.enabled = hasModel and hasMarkup
    self.exportLandmarkSelectionButton.enabled = hasModel and hasMarkup

    # Update selection method display
    if hasMarkup:
      self._updateSelectionMethodDisplay()
  

  def onMarkupNodeChanged(self):
    """Handle markup node change and set up observers for point selection updates"""
    # Clean up previous observers
    if hasattr(self, '_currentMarkupNode') and self._currentMarkupNode:
      if hasattr(self, '_markupObserver'):
        self._currentMarkupNode.RemoveObserver(self._markupObserver)
      if hasattr(self, '_markupEndInteractionObserver'):
        self._currentMarkupNode.RemoveObserver(self._markupEndInteractionObserver)
      if hasattr(self, '_markupPointAddedObserver'):
        self._currentMarkupNode.RemoveObserver(self._markupPointAddedObserver)

    # Set up observers for new markup node
    markupNode = self.selectionMarkupSelector.currentNode()
    if markupNode:
      self._currentMarkupNode = markupNode
      # Observe point modifications (added, removed, or moved)
      self._markupObserver = markupNode.AddObserver(slicer.vtkMRMLMarkupsNode.PointModifiedEvent, self._onMarkupPointModified)
      # Observe when user finishes interacting with points (for auto-selection)
      self._markupEndInteractionObserver = markupNode.AddObserver(slicer.vtkMRMLMarkupsNode.PointEndInteractionEvent, self._onMarkupPointEndInteraction)
      # Observe when a new point is added (for auto-selection on placement)
      self._markupPointAddedObserver = markupNode.AddObserver(slicer.vtkMRMLMarkupsNode.PointPositionDefinedEvent, self._onMarkupPointAdded)
      # Also observe PointRemovedEvent just to be safe
      self._markupPointRemovedObserver = markupNode.AddObserver(slicer.vtkMRMLMarkupsNode.PointRemovedEvent, self._onMarkupPointModified)

    # Update button text and display based on markup type
    self._updateDrawButtonText()
    self._updateSelectionMethodDisplay()

  def _updateDrawButtonText(self):
    """Update the Draw button tooltip based on current state"""
    markupNode = self.selectionMarkupSelector.currentNode()
    if not markupNode:
      self.createMarkupButton.setToolTip("Create a new closed curve for drawing the selection region")
    else:
      self.createMarkupButton.setToolTip("Add more control points to the closed curve")

  def _onMarkupPointModified(self, caller, event):
    """Handle markup point modification events"""
    self._updateSelectionMethodDisplay()

  def _onMarkupPointAdded(self, caller, event):
    """Handle when a new point position is defined (placed on the mesh)"""
    # Just update the display, don't auto-apply for closed curves
    # Users will manually click "Apply Landmark Selection" when ready
    self._updateSelectionMethodDisplay()

  def _onMarkupPointEndInteraction(self, caller, event):
    """Handle when user finishes interacting with a markup point"""
    # Just update the display, don't auto-apply
    # Users will manually click "Apply Landmark Selection" when ready
    self._updateSelectionMethodDisplay()
  

  def onFastSurfacePaint(self):
    """Fast surface-based selection using model scalar overlays - no volume conversion needed"""
    modelNode = self.regionMeshSelector.currentNode()
    if not modelNode:
      slicer.util.errorDisplay("Please select a mesh first.")
      return

    try:
      # Create a copy of the model for selection overlay
      modelCopy = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode")
      modelCopy.SetName(f"{modelNode.GetName()}_SelectionOverlay")

      # Copy the mesh data
      modelPolyData = modelNode.GetPolyData()
      if not modelPolyData:
        slicer.util.errorDisplay("Model has no mesh data.")
        return

      # Create a copy of the polydata
      copiedPolyData = vtk.vtkPolyData()
      copiedPolyData.DeepCopy(modelPolyData)

      # Add scalar array for selection painting
      numPoints = copiedPolyData.GetNumberOfPoints()
      selectionArray = vtk.vtkFloatArray()
      selectionArray.SetName("Selection")
      selectionArray.SetNumberOfComponents(1)
      selectionArray.SetNumberOfTuples(numPoints)
      selectionArray.Fill(0.0)  # Initialize with no selection

      copiedPolyData.GetPointData().SetScalars(selectionArray)
      modelCopy.SetAndObservePolyData(copiedPolyData)

      # Set up display for interactive painting
      displayNode = modelCopy.GetDisplayNode()
      if not displayNode:
        displayNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelDisplayNode")
        modelCopy.SetAndObserveDisplayNodeID(displayNode.GetID())

      # Configure for scalar-based coloring
      displayNode.SetScalarVisibility(True)
      displayNode.SetActiveScalarName("Selection")
      displayNode.SetScalarRangeFlag(displayNode.UseManualScalarRange)
      displayNode.SetScalarRange(0.0, 1.0)

      # Set up color map for selection visualization
      colorNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLColorTableNode")
      colorNode.SetTypeToUser()
      colorNode.SetNumberOfColors(256)
      colorNode.SetName("SelectionColors")

      # Create color map: transparent for 0, red for 1
      for i in range(256):
        value = i / 255.0
        if value < 0.1:  # Unselected
          colorNode.SetColor(i, 0.8, 0.8, 0.8, 0.3)  # Light gray, transparent
        else:  # Selected
          colorNode.SetColor(i, 1.0, 0.0, 0.0, 0.8)  # Red, opaque

      displayNode.SetAndObserveColorNodeID(colorNode.GetID())

      # Hide original model to avoid confusion
      originalDisplay = modelNode.GetDisplayNode()
      if originalDisplay:
        originalDisplay.SetVisibility(False)

      # Store references
      self.currentSurfaceModel = modelCopy
      self.currentOriginalModel = modelNode

      # Set up interactive painting using Markups
      self._setupSurfacePainting(modelCopy)

      slicer.util.infoDisplay(
        "Fast surface selection ready!\n\n"
        "• Place markup points on the model to select regions\n"
        "• Use the radius slider to control selection size\n"
        "• Click 'Export Selected Region' when done\n\n"
        "This method is much faster than volume conversion!"
      )

    except Exception as e:
      slicer.util.errorDisplay(f"Error setting up fast surface painting: {str(e)}")
      print(f"Fast surface painting error: {e}")

  def _setupSurfacePainting(self, modelNode):
    """Setup interactive surface painting using markups"""
    # Create markup points for painting
    markupNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode")
    markupNode.SetName(f"{modelNode.GetName()}_PaintPoints")

    # Configure markup display
    markupDisplay = markupNode.GetDisplayNode()
    if markupDisplay:
      markupDisplay.SetGlyphType(markupDisplay.Sphere3D)
      markupDisplay.SetGlyphScale(2.0)
      markupDisplay.SetSelectedColor(1.0, 0.0, 0.0)  # Red
      markupDisplay.SetOpacity(0.8)

    # Add painting controls to UI
    if not hasattr(self, 'surfacePaintingFrame'):
      self._createSurfacePaintingControls()

    self.surfacePaintingFrame.setVisible(True)
    self.currentPaintMarkup = markupNode

    # Connect markup modification to painting
    markupNode.AddObserver(slicer.vtkMRMLMarkupsNode.PointModifiedEvent, self._onPaintPointModified)
    markupNode.AddObserver(slicer.vtkMRMLMarkupsNode.PointAddedEvent, self._onPaintPointAdded)

  def _createSurfacePaintingControls(self):
    """Create UI controls for surface painting"""
    self.surfacePaintingFrame = qt.QFrame()
    self.surfacePaintingFrame.setFrameStyle(qt.QFrame.StyledPanel)
    self.surfacePaintingLayout = qt.QFormLayout(self.surfacePaintingFrame)

    # Paint radius control
    self.paintRadiusSlider = ctk.ctkDoubleSlider()
    self.paintRadiusSlider.minimum = 1.0
    self.paintRadiusSlider.maximum = 20.0
    self.paintRadiusSlider.value = 5.0
    self.paintRadiusSlider.setToolTip("Radius of painting brush in mm")
    self.surfacePaintingLayout.addRow("Paint Radius (mm):", self.paintRadiusSlider)

    # Clear selection button
    self.clearSurfaceSelectionButton = qt.QPushButton("Clear Selection")
    self.clearSurfaceSelectionButton.setStyleSheet(ColorTheme.getButtonStyle('secondary'))
    self.clearSurfaceSelectionButton.connect('clicked(bool)', self._onClearSurfaceSelection)
    self.surfacePaintingLayout.addRow(self.clearSurfaceSelectionButton)

    # Surface painting frame is now standalone (segmentEditorLayout removed)
    self.surfacePaintingFrame.setVisible(False)

  def _onPaintPointAdded(self, caller, event):
    """Handle new paint point added"""
    self._updateSurfaceSelection()

  def _onPaintPointModified(self, caller, event):
    """Handle paint point moved"""
    self._updateSurfaceSelection()

  def _updateSurfaceSelection(self):
    """Update surface selection based on markup points"""
    if not hasattr(self, 'currentSurfaceModel') or not hasattr(self, 'currentPaintMarkup'):
      return

    modelNode = self.currentSurfaceModel
    markupNode = self.currentPaintMarkup
    polyData = modelNode.GetPolyData()

    if not polyData:
      return

    # Get selection array
    selectionArray = polyData.GetPointData().GetScalars("Selection")
    if not selectionArray:
      return

    # Clear previous selection
    selectionArray.Fill(0.0)

    # Paint around each markup point
    radius = self.paintRadiusSlider.value
    numPoints = polyData.GetNumberOfPoints()

    for i in range(markupNode.GetNumberOfControlPoints()):
      if markupNode.GetNthControlPointVisibility(i):
        markupPos = [0, 0, 0]
        markupNode.GetNthControlPointPosition(i, markupPos)

        # Find points within radius
        for ptId in range(numPoints):
          point = polyData.GetPoint(ptId)
          distance = vtk.vtkMath.Distance2BetweenPoints(point, markupPos)

          if distance <= radius * radius:
            selectionArray.SetValue(ptId, 1.0)  # Mark as selected

    # Update display
    selectionArray.Modified()
    polyData.Modified()
    modelNode.Modified()

  def _onClearSurfaceSelection(self):
    """Clear all surface selection"""
    if hasattr(self, 'currentSurfaceModel'):
      polyData = self.currentSurfaceModel.GetPolyData()
      if polyData:
        selectionArray = polyData.GetPointData().GetScalars("Selection")
        if selectionArray:
          selectionArray.Fill(0.0)
          selectionArray.Modified()
          polyData.Modified()
          self.currentSurfaceModel.Modified()

    if hasattr(self, 'currentPaintMarkup'):
      self.currentPaintMarkup.RemoveAllControlPoints()
  
  
  def onApplyLandmarkSelection(self):
    """Apply region selection using landmarks and radius"""
    modelNode = self.regionMeshSelector.currentNode()
    markupNode = self.selectionMarkupSelector.currentNode()
    
    if not (modelNode and markupNode):
      slicer.util.errorDisplay("Please select both a mesh and markup points.")
      return
    
    if markupNode.GetNumberOfControlPoints() == 0:
      slicer.util.errorDisplay("No markup points found. Please add some points first.")
      return

    try:
      # Get selected landmark points
      selectedPoints = self._getSelectedPoints(markupNode)

      # If no points are explicitly selected, use all available points
      if not selectedPoints:
        numPoints = markupNode.GetNumberOfControlPoints()
        selectedPoints = list(range(numPoints))

      # Check "Select only one side" checkbox state up front
      selectOneSideOnly = self.mirrorSelectionCheckbox.isChecked()

      # Use all selected points for region selection
      selectedVertices = self.selectMeshRegionBySelectedPoints(
        modelNode, markupNode, selectedPoints,
        filterToLandmarkSide=selectOneSideOnly)

      # Check if any vertices were selected
      if len(selectedVertices) == 0:
        slicer.util.warningDisplay("No vertices selected. Try adjusting the curve or adding more landmarks.")
        return

      # If "Select only one side" is CHECKED, filter to only include vertices on the same side as landmarks
      if selectOneSideOnly:
        selectedVertices = self.filterVerticesToSameSide(modelNode, markupNode, selectedVertices)
        if len(selectedVertices) == 0:
          slicer.util.warningDisplay("No vertices found on the same side as landmarks. Try adjusting the selection.")
          return
      else:
        # If "Select only one side" is UNCHECKED, use full area bounded by landmarks
        pass
      # Merge with existing selection if present (on the same model)
      if (hasattr(self, 'selectedRegionVertices') and hasattr(self, 'selectedRegionModel')
          and self.selectedRegionModel == modelNode):
        existingSet = set(self.selectedRegionVertices)
        newSet = set(selectedVertices)
        mergedSet = existingSet | newSet
        selectedVertices = list(mergedSet)
        print(f"[Selection] Merged with previous selection: {len(existingSet)} + {len(newSet)} -> {len(mergedSet)} vertices")

      # Store selected vertices and model
      self.selectedRegionVertices = selectedVertices
      self.selectedRegionModel = modelNode

      # Clear previous visualization before re-drawing
      self.clearRegionSelection(modelNode)

      # Get face indices that contain vertices from the selected region
      polyData = modelNode.GetPolyData()
      nTotalFaces = polyData.GetNumberOfCells()
      regionFaces = []
      regionVertexSet = set(selectedVertices)

      for faceIdx in range(nTotalFaces):
        cell = polyData.GetCell(faceIdx)
        pointIds = cell.GetPointIds()
        numPoints = pointIds.GetNumberOfIds()

        # Check if all vertices of this face are in the region
        allInRegion = True
        for j in range(numPoints):
          if pointIds.GetId(j) not in regionVertexSet:
            allInRegion = False
            break

        if allInRegion:
          regionFaces.append(faceIdx)

      self.selectedRegionFaces = set(regionFaces)

      # Visualize the selection with filtered vertices
      self.visualizeRegionSelection(modelNode, selectedVertices)

      # Update info label
      numVertices = len(selectedVertices)
      totalVertices = modelNode.GetPolyData().GetNumberOfPoints()
      percentage = (numVertices / totalVertices) * 100 if totalVertices > 0 else 0
      self.selectionInfoLabel.setText(f"Selected: {numVertices}/{totalVertices} vertices ({percentage:.1f}%)")

      # Enable clear and export buttons
      self.clearSelectionButton.enabled = True
      self.exportLandmarkSelectionButton.enabled = True

      # Switch to 3D-only view to see the region selection
      try:
        layoutManager = slicer.app.layoutManager()
        if layoutManager:
          layoutManager.setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutOneUp3DView)
      except Exception as e:
        logging.debug("Could not switch to 3D view: %s", e)

    except Exception as e:
      slicer.util.errorDisplay(f"Error during landmark selection: {str(e)}")
  
  def onExportLandmarkSelection(self):
    """Export the selected region as a new model"""
    modelNode = self.regionMeshSelector.currentNode()
    markupNode = self.selectionMarkupSelector.currentNode()
    
    if not (modelNode and markupNode):
      slicer.util.errorDisplay("Please select both a mesh and markup points.")
      return
    
    if markupNode.GetNumberOfControlPoints() == 0:
      slicer.util.errorDisplay("No markup points found. Please add some points first.")
      return
    
    try:
      # Get selected landmark points
      selectedPoints = self._getSelectedPoints(markupNode)

      # If no points are explicitly selected, use all available points
      if not selectedPoints:
        numPoints = markupNode.GetNumberOfControlPoints()
        selectedPoints = list(range(numPoints))

      # Use all selected points for region selection
      selectedVertices = self.selectMeshRegionBySelectedPoints(
        modelNode, markupNode, selectedPoints,
        filterToLandmarkSide=self.mirrorSelectionCheckbox.isChecked())

      if not selectedVertices:
        slicer.util.warningDisplay("No vertices were selected. Try adjusting the curve or landmark positions.")
        return

      # Get the export name from the text field
      exportName = self.exportRegionNameEdit.text.strip()
      if not exportName:
        exportName = "SelectedRegion"

      # Create new model from selected vertices
      newModelNode = self.createModelFromSelectedVertices(modelNode, selectedVertices, exportName)
      
      if newModelNode:
        # Enable clear button
        self.clearSelectionButton.enabled = True
        
        # Update info
        numVertices = len(selectedVertices)
        self.selectionInfoLabel.setText(f"Exported model: {numVertices} vertices")
        
        slicer.util.infoDisplay(
          f"Successfully exported selected region!\n"
          f"New model: {newModelNode.GetName()}\n"
          f"Vertices: {numVertices}")
      else:
        slicer.util.errorDisplay("Failed to create model from selected vertices.")
        
    except Exception as e:
      slicer.util.errorDisplay(f"Error exporting landmark selection: {str(e)}")
      logging.error("Error exporting landmark selection: %s", e)
  
  def onCreateOrPlaceMarkup(self):
    """Create or activate placement for closed curve"""
    markupNode = self.selectionMarkupSelector.currentNode()

    if markupNode:
      # A curve already exists - enter placement mode to add more points
      selectionNode = slicer.app.applicationLogic().GetSelectionNode()
      selectionNode.SetActivePlaceNodeID(markupNode.GetID())

      interactionNode = slicer.app.applicationLogic().GetInteractionNode()
      interactionNode.SetPlaceModePersistence(True)
      interactionNode.SetCurrentInteractionMode(interactionNode.Place)

      slicer.util.infoDisplay("Click on the mesh to add more curve points.\nPress ESC when done.")
    else:
      # No curve selected - create a new one
      self.onCreateSelectionCurve()

  def onCreateSelectionCurve(self):
    """Create a new closed curve markup for drawing selection region"""
    # Create a new closed curve node
    curveNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsClosedCurveNode")
    curveNode.SetName("SelectionCurve")

    # Set the curve as the current selection
    self.selectionMarkupSelector.setCurrentNode(curveNode)

    # Set the active markup node for placement
    selectionNode = slicer.app.applicationLogic().GetSelectionNode()
    selectionNode.SetActivePlaceNodeID(curveNode.GetID())

    # Enter persistent place mode (allows placing multiple points continuously)
    interactionNode = slicer.app.applicationLogic().GetInteractionNode()
    interactionNode.SetPlaceModePersistence(True)  # Keep placing mode active after each point
    interactionNode.SetCurrentInteractionMode(interactionNode.Place)

    slicer.util.infoDisplay("Click on the mesh to add curve points.\nThe curve will automatically close.\nPress ESC when done placing points.")

  def onClearSelection(self):
    """Clear the current region selection"""
    modelNode = self.regionMeshSelector.currentNode()
    if modelNode:
      self.clearRegionSelection(modelNode)
      self.selectionInfoLabel.setText("No region selected")
      self.clearSelectionButton.enabled = False
      self.exportLandmarkSelectionButton.enabled = False
      # Clear stored region data
      if hasattr(self, 'selectedRegionVertices'):
        delattr(self, 'selectedRegionVertices')
      if hasattr(self, 'selectedRegionModel'):
        delattr(self, 'selectedRegionModel')
      if hasattr(self, 'selectedRegionFaces'):
        delattr(self, 'selectedRegionFaces')
  
  
  def _updateSelectionMethodDisplay(self):
    """Update the selection method display based on markup type and point count"""
    markupNode = self.selectionMarkupSelector.currentNode()
    hasMarkup = bool(markupNode)
    
    # Try to get model from regionMeshSelector if available
    regionModelNode = self.regionMeshSelector.currentNode() if hasattr(self, 'regionMeshSelector') else None
    hasModel = bool(regionModelNode)
    
    # Check calling context if needed (not implemented yet, relying on selectors)
    
    if hasattr(self, 'applyLandmarkSelectionButton'):
        newState = hasModel and hasMarkup

        self.applyLandmarkSelectionButton.enabled = newState
            
    if hasattr(self, 'exportLandmarkSelectionButton'):
        self.exportLandmarkSelectionButton.enabled = hasModel and hasMarkup
  
  def _getSelectedPoints(self, markupNode):
    """Get list of selected landmark point indices"""
    selectedPoints = []
    numPoints = markupNode.GetNumberOfControlPoints()
    
    for i in range(numPoints):
      if markupNode.GetNthControlPointSelected(i):
        selectedPoints.append(i)
    
    return selectedPoints
  
  def _findClickedLandmark(self):
    """Fallback method to find which landmark was clicked"""
    markupNode = self.selectionMarkupSelector.currentNode()
    if not markupNode:
      return
    
    # Get the 3D view and mouse position
    threeDWidget = slicer.app.layoutManager().threeDWidget(0)
    threeDView = threeDWidget.threeDView()
    interactor = threeDView.interactor()
    
    # Get mouse position in display coordinates
    mousePos = interactor.GetEventPosition()
    
    # Convert to world coordinates
    renderer = threeDView.renderWindow().GetRenderers().GetFirstRenderer()
    worldPos = [0, 0, 0, 0]
    renderer.SetDisplayPoint(mousePos[0], mousePos[1], 0)
    renderer.DisplayToWorld()
    worldPos = renderer.GetWorldPoint()
    
    # Check each landmark point to see if click is near it
    closestIndex = -1
    minDistance = float('inf')
    tolerance = 10.0  # 10 unit tolerance for landmark selection
    
    for i in range(markupNode.GetNumberOfControlPoints()):
      landmarkPos = [0, 0, 0]
      markupNode.GetNthControlPointPosition(i, landmarkPos)
      
      # Calculate distance from click to landmark
      distance = vtk.vtkMath.Distance2BetweenPoints(worldPos[:3], landmarkPos)
      
      if distance < minDistance:
        minDistance = distance
        closestIndex = i
    
    # If we found a close enough landmark, toggle its selection
    if closestIndex >= 0 and minDistance < (tolerance * tolerance):
      markupNode = self.selectionMarkupSelector.currentNode()
      if markupNode:
        # Toggle the selection state of the landmark
        currentState = markupNode.GetNthControlPointSelected(closestIndex)
        markupNode.SetNthControlPointSelected(closestIndex, not currentState)
        
        # Update the display
        self._updateSelectedPointsDisplay()
        
        action = "Selected" if not currentState else "Deselected"
        slicer.util.infoDisplay(f"{action} landmark point {closestIndex}")
      # Keep the click-select mode active for selecting other landmarks
      # User can click the button again to disable it
    else:
      # Show feedback that no landmark was found near the click
      slicer.util.warningDisplay("No landmark point found near the clicked position. Try clicking closer to a landmark.")
  
  def _selectMeshRegionByPolygonAreaFloodFill(self, modelNode, markupNode, selectedPointIndices, filterToLandmarkSide=True):
    """Select mesh vertices in the region bounded by landmarks using geodesic flood fill.

    Args:
      modelNode: The model node containing the mesh data.
      markupNode: The markup node supplying landmark positions.
      selectedPointIndices: Indices of landmarks to consider.
      filterToLandmarkSide: If True, restrict selection to the landmark side of the mesh.
    """
    import numpy as np
    from collections import deque

    # Get mesh data
    polyData = modelNode.GetPolyData()
    points = polyData.GetPoints()
    numPoints = points.GetNumberOfPoints()

    # Build adjacency list for mesh connectivity
    # Connect all vertices within each face (not just consecutive pairs)
    adjacency = [set() for _ in range(numPoints)]

    numCells = polyData.GetNumberOfCells()
    for i in range(numCells):
      cell = polyData.GetCell(i)
      pointIds = cell.GetPointIds()
      numCellPoints = pointIds.GetNumberOfIds()

      # Get all vertex indices in this face
      faceVertices = [pointIds.GetId(j) for j in range(numCellPoints)]

      # Connect all pairs of vertices in this face
      for j in range(numCellPoints):
        for k in range(j + 1, numCellPoints):
          v1 = faceVertices[j]
          v2 = faceVertices[k]
          adjacency[v1].add(v2)
          adjacency[v2].add(v1)

    # Get landmark positions and find closest vertices
    landmarkVertices = []
    landmarkPositions = []
    for pointIndex in selectedPointIndices:
      point = [0, 0, 0]
      markupNode.GetNthControlPointPosition(pointIndex, point)
      landmarkPositions.append(np.array(point))

      # Find closest mesh vertex to this landmark
      closestVertex = -1
      minDist = float('inf')
      for i in range(numPoints):
        vertex = points.GetPoint(i)
        dist = vtk.vtkMath.Distance2BetweenPoints(point, vertex)
        if dist < minDist:
          minDist = dist
          closestVertex = i

      if closestVertex >= 0:
        landmarkVertices.append(closestVertex)

    if len(landmarkVertices) < 3:
      slicer.util.warningDisplay("Not enough landmark vertices found to define a region.")
      return []

    # Compute the centroid of landmark positions (not vertices)
    centroid = np.mean(landmarkPositions, axis=0)

    # Find multiple seed points near the centroid to start flood fill
    seedCandidates = []
    for i in range(numPoints):
      vertex = np.array(points.GetPoint(i))
      dist = np.linalg.norm(vertex - centroid)
      seedCandidates.append((i, dist))

    # Sort by distance and take closest candidates
    seedCandidates.sort(key=lambda x: x[1])

    # Build edge-to-face mapping to detect mesh boundaries
    edgeToFaces = {}
    for i in range(polyData.GetNumberOfCells()):
      cell = polyData.GetCell(i)
      pointIds = cell.GetPointIds()
      numCellPoints = pointIds.GetNumberOfIds()

      for j in range(numCellPoints):
        p1 = pointIds.GetId(j)
        p2 = pointIds.GetId((j + 1) % numCellPoints)
        edge = tuple(sorted([p1, p2]))
        if edge not in edgeToFaces:
          edgeToFaces[edge] = []
        edgeToFaces[edge].append(i)

    # Compute geodesic distances from each landmark vertex
    landmarkSet = set(landmarkVertices)

    # Identify which boundary edges are part of the landmark perimeter
    # These should be allowed to cross
    landmarkBoundaryEdges = set()
    for i in range(len(landmarkVertices)):
      v1 = landmarkVertices[i]
      v2 = landmarkVertices[(i + 1) % len(landmarkVertices)]
      edge = tuple(sorted([v1, v2]))
      if edge in edgeToFaces:
        landmarkBoundaryEdges.add(edge)

    # Identify all boundary vertices (vertices that have at least one boundary edge)
    boundaryVertices = set()
    for edge, faces in edgeToFaces.items():
      if len(faces) == 1:
        boundaryVertices.add(edge[0])
        boundaryVertices.add(edge[1])

    # Calculate approximate max distance (use distance between furthest landmarks)
    maxLandmarkDist = 0
    for i in range(len(landmarkPositions)):
      for j in range(i+1, len(landmarkPositions)):
        dist = np.linalg.norm(landmarkPositions[i] - landmarkPositions[j])
        maxLandmarkDist = max(maxLandmarkDist, dist)

    # Find interior vertices near landmarks to start the flood fill
    # The landmark vertices themselves are on the boundary, so we need to find vertices
    # that are inside the region bounded by the landmarks

    # Strategy: Find all vertices that are neighbors of multiple landmarks
    # or find vertices near the centroid
    interiorSeeds = set()

    # Add all non-boundary neighbors of landmarks as seeds
    for landmarkVert in landmarkVertices:
      for neighbor in adjacency[landmarkVert]:
        # Check if this neighbor is NOT on the boundary
        if neighbor not in boundaryVertices:
          interiorSeeds.add(neighbor)
        else:
          # Even if on boundary, add if it's close to another landmark
          for otherLandmark in landmarkVertices:
            if otherLandmark != landmarkVert and neighbor in adjacency[otherLandmark]:
              interiorSeeds.add(neighbor)
              break

    # If we found no interior seeds, try vertices close to the centroid
    if len(interiorSeeds) == 0:
      # Find closest vertices to centroid (ignore boundary status)
      candidateSeeds = []
      for i in range(numPoints):
        vertex = np.array(points.GetPoint(i))
        dist = np.linalg.norm(vertex - centroid)
        if dist < maxLandmarkDist * 0.5:  # Within landmark region
          candidateSeeds.append((i, dist))

      # Take closest candidates
      candidateSeeds.sort(key=lambda x: x[1])
      interiorSeeds = set([v for v, d in candidateSeeds[:20]])  # Take more seeds

    # Start from interior seeds, not landmarks
    selectedVertices = set(interiorSeeds) | set(landmarkVertices)
    queue = deque(interiorSeeds)
    visited = set(interiorSeeds) | set(landmarkVertices)

    # Debug: check adjacency for first seed
    expansions = 0
    maxQueueSize = len(queue)
    verticesProcessed = 0

    while queue:
      currentVertex = queue.popleft()
      selectedVertices.add(currentVertex)
      verticesProcessed += 1

      # Explore all neighbors - ignore boundary edge constraints
      neighborsAdded = 0
      for neighbor in adjacency[currentVertex]:
        if neighbor in visited:
          continue

        visited.add(neighbor)
        queue.append(neighbor)
        neighborsAdded += 1
        expansions += 1

      maxQueueSize = max(maxQueueSize, len(queue))

    return list(selectedVertices)

  def _detectLandmarkSide(self, polyData, landmarkPositions):
    """Detect which side of the mesh contains the most landmarks.

    Returns:
      tuple: (mirrorAxis, landmarkSideSign, meshCenter, ranges, midlineThreshold)
    """
    import numpy as np

    points = polyData.GetPoints()
    numPoints = polyData.GetNumberOfPoints()

    # Calculate mesh center and ranges
    allPoints = np.array([points.GetPoint(i) for i in range(numPoints)])
    minCoords = np.min(allPoints, axis=0)
    maxCoords = np.max(allPoints, axis=0)
    meshCenter = (minCoords + maxCoords) / 2.0
    ranges = maxCoords - minCoords

    # Auto-detect bilateral symmetry axis (smallest range)
    mirrorAxis = np.argmin(ranges)

    # Count landmarks on each side
    positiveSideCount = sum(1 for lm in landmarkPositions if (lm[mirrorAxis] - meshCenter[mirrorAxis]) >= 0)
    negativeSideCount = len(landmarkPositions) - positiveSideCount
    landmarkSideSign = 1 if positiveSideCount >= negativeSideCount else -1

    # Calculate midline threshold (5% of range to allow slight overlap)
    midlineThreshold = ranges[mirrorAxis] * 0.05

    return mirrorAxis, landmarkSideSign, meshCenter, ranges, midlineThreshold

  def selectMeshRegionByPolygonArea(self, modelNode, markupNode, selectedPointIndices, filterToLandmarkSide=True):
    """Select mesh vertices using polygon + texture similarity refinement.

    Args:
      filterToLandmarkSide: When True, restricts selection to the landmark side.
    """
    import numpy as np
    from scipy.spatial import Delaunay

    # Get mesh data
    polyData = modelNode.GetPolyData()
    points = polyData.GetPoints()
    numPoints = points.GetNumberOfPoints()

    # Get landmark positions
    landmarkPositions = []
    for pointIndex in selectedPointIndices:
      point = [0, 0, 0]
      markupNode.GetNthControlPointPosition(pointIndex, point)
      landmarkPositions.append(point)
    landmarkPositions = np.array(landmarkPositions)

    if filterToLandmarkSide:
      # Detect which side has most landmarks
      mirrorAxis, landmarkSideSign, meshCenter, ranges, midlineThreshold = self._detectLandmarkSide(polyData, landmarkPositions)

      logging.debug(
        "Mesh ranges: X=%.4f, Y=%.4f, Z=%.4f; mirror axis=%s; selected side=%s",
        ranges[0], ranges[1], ranges[2], ['X', 'Y', 'Z'][mirrorAxis],
        'positive' if landmarkSideSign == 1 else 'negative')

      # Pre-filter vertices: opposite side and midline only
      validVertexIndices = []
      oppositeSideCount = midlineCount = 0

      for i in range(numPoints):
        vertex = points.GetPoint(i)
        vertexSide = vertex[mirrorAxis] - meshCenter[mirrorAxis]
        vertexSideSign = 1 if vertexSide >= 0 else -1

        if vertexSideSign != landmarkSideSign and abs(vertexSide) >= midlineThreshold:
          oppositeSideCount += 1
          continue

        if abs(vertexSide) < midlineThreshold:
          midlineCount += 1

        validVertexIndices.append(i)

      print(f"[Selection] Pre-filtered vertices: {len(validVertexIndices)}/{numPoints} (excluded {oppositeSideCount} opposite side, {midlineCount} midline)")
    else:
      validVertexIndices = list(range(numPoints))
      logging.debug("Select-only-one-side disabled; using full vertex set of size %d", numPoints)

    # STEP 1: Polygon-based selection using PCA + Delaunay (only on valid vertices)
    center = np.mean(landmarkPositions, axis=0)
    centered = landmarkPositions - center

    # Compute covariance matrix for PCA
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Sort by eigenvalues (largest first)
    idx = eigenvalues.argsort()[::-1]
    eigenvectors = eigenvectors[:, idx]

    # Project landmarks onto 2D plane
    landmarks2D = centered @ eigenvectors[:, :2]

    # Create Delaunay triangulation
    try:
      delaunay = Delaunay(landmarks2D)
    except Exception as e:
      logging.warning("Could not create Delaunay triangulation: %s", e)
      return self._fallbackSpatialSelection(modelNode, markupNode, selectedPointIndices)

    # Get polygon selection (only check valid vertices)
    polygonVertices = set()
    
    # Pre-calculate convex hull edges for buffering
    hull_edges = delaunay.convex_hull
    p1 = landmarks2D[hull_edges[:, 0]]
    p2 = landmarks2D[hull_edges[:, 1]]
    v_segments = p2 - p1
    c2 = np.sum(v_segments * v_segments, axis=1)
    
    # Calculate 5% buffer distance in 2D space
    min_b = np.min(landmarks2D, axis=0)
    max_b = np.max(landmarks2D, axis=0)
    buffer_dist = np.max(max_b - min_b) * 0.05
    
    # Vectorized check of all vertices
    v_indices = np.array(validVertexIndices)
    if len(v_indices) > 0:
      v_arr = np.array([points.GetPoint(i) for i in validVertexIndices])
      v_cent = v_arr - center
      v2_arr = v_cent @ eigenvectors[:, :2]
      
      in_simplex = delaunay.find_simplex(v2_arr) >= 0
      
      # Add vertices that are strictly inside
      for idx, is_in in zip(validVertexIndices, in_simplex):
        if is_in:
          polygonVertices.add(idx)
          
      # For vertices outside, check if they are within buffer distance of the convex hull
      outside_mask = ~in_simplex
      if np.any(outside_mask) and len(hull_edges) > 0:
        out_v2 = v2_arr[outside_mask]
        out_idx = v_indices[outside_mask]
        
        # Calculate min distance to any segment for each outside point
        w = out_v2[:, np.newaxis, :] - p1[np.newaxis, :, :]
        c1 = np.sum(w * v_segments[np.newaxis, :, :], axis=2)
        
        dist_p1 = np.linalg.norm(w, axis=2)
        w2 = out_v2[:, np.newaxis, :] - p2[np.newaxis, :, :]
        dist_p2 = np.linalg.norm(w2, axis=2)
        
        c2_safe = np.where(c2 == 0, 1e-10, c2)
        b = c1 / c2_safe[np.newaxis, :]
        pb = p1[np.newaxis, :, :] + b[:, :, np.newaxis] * v_segments[np.newaxis, :, :]
        dist_pb = np.linalg.norm(out_v2[:, np.newaxis, :] - pb, axis=2)
        
        mask1 = c1 <= 0
        mask2 = c2[np.newaxis, :] <= c1
        mask_mid = ~(mask1 | mask2)
        
        dist = np.zeros_like(c1)
        dist[mask1] = dist_p1[mask1]
        dist[mask2] = dist_p2[mask2]
        dist[mask_mid] = dist_pb[mask_mid]
        
        min_dist = np.min(dist, axis=1)
        close_enough = min_dist <= buffer_dist
        
        for idx in out_idx[close_enough]:
          polygonVertices.add(int(idx))

    print(f"[Selection] Polygon selection retained {len(polygonVertices)} vertices (inc buffer)")

    # Texture similarity refinement on boundary vertices (optional)
    colorArray = polyData.GetPointData().GetScalars()
    if not colorArray:
      logging.debug("No texture data available; returning polygon selection only")
      return list(polygonVertices)

    # Find landmark vertices and get their colors
    landmarkVertices = []
    for lmPos in landmarkPositions:
      closestVertex = -1
      minDist = float('inf')
      for i in range(numPoints):
        vertex = points.GetPoint(i)
        dist = vtk.vtkMath.Distance2BetweenPoints(lmPos, vertex)
        if dist < minDist:
          minDist = dist
          closestVertex = i
      if closestVertex >= 0:
        landmarkVertices.append(closestVertex)

    # Get reference color from landmarks
    landmarkColors = []
    numComponents = colorArray.GetNumberOfComponents()
    for vIdx in landmarkVertices:
      if numComponents == 1:
        color = [colorArray.GetValue(vIdx)]
      else:
        color = [colorArray.GetComponent(vIdx, c) for c in range(numComponents)]
      landmarkColors.append(color)

    landmarkColors = np.array(landmarkColors)
    meanLandmarkColor = np.mean(landmarkColors, axis=0)
    stdLandmarkColor = np.std(landmarkColors, axis=0) + 1e-6

    logging.debug("Reference color mean: %s", meanLandmarkColor)

    # Build mesh adjacency to find boundary vertices
    adjacency = [set() for _ in range(numPoints)]
    for i in range(polyData.GetNumberOfCells()):
      cell = polyData.GetCell(i)
      pointIds = cell.GetPointIds()
      numCellPoints = pointIds.GetNumberOfIds()
      for j in range(numCellPoints):
        p1 = pointIds.GetId(j)
        p2 = pointIds.GetId((j + 1) % numCellPoints)
        adjacency[p1].add(p2)
        adjacency[p2].add(p1)

    # Find boundary vertices (vertices on edge of polygon selection)
    boundaryVertices = set()
    for vIdx in polygonVertices:
      for neighbor in adjacency[vIdx]:
        if neighbor not in polygonVertices:
          boundaryVertices.add(vIdx)
          break

    logging.debug("Boundary vertices identified: %d", len(boundaryVertices))

    # Filter boundary vertices using texture similarity
    refinedVertices = polygonVertices.copy()
    for vIdx in boundaryVertices:
      # Get vertex color
      if numComponents == 1:
        vertexColor = np.array([colorArray.GetValue(vIdx)])
      else:
        vertexColor = np.array([colorArray.GetComponent(vIdx, c) for c in range(numComponents)])

      # Calculate color similarity
      colorDiff = np.abs(vertexColor - meanLandmarkColor) / stdLandmarkColor
      colorSimilarity = 1.0 / (1.0 + np.mean(colorDiff))

      # Remove boundary vertex if color is too different (threshold)
      if colorSimilarity < 0.7:  # Adjust threshold as needed
        refinedVertices.discard(vIdx)

    logging.debug("After texture refinement: %d vertices", len(refinedVertices))
    return list(refinedVertices)

  def _fallbackSpatialSelection(self, modelNode, markupNode, selectedPointIndices):
    """Fallback to spatial selection if no texture data available"""
    import numpy as np

    polyData = modelNode.GetPolyData()
    points = polyData.GetPoints()

    landmarkPositions = []
    for pointIndex in selectedPointIndices:
      point = [0, 0, 0]
      markupNode.GetNthControlPointPosition(pointIndex, point)
      landmarkPositions.append(point)

    landmarkPositions = np.array(landmarkPositions)
    center = np.mean(landmarkPositions, axis=0)
    maxDist = max([np.linalg.norm(pos - center) for pos in landmarkPositions])

    selectedVertices = set()
    for i in range(points.GetNumberOfPoints()):
      vertex = np.array(points.GetPoint(i))
      dist = np.linalg.norm(vertex - center)
      if dist <= maxDist * 1.2:
        selectedVertices.add(i)

    return list(selectedVertices)

  def mapCutModelToOriginalVertices(self, originalModel, cutModel):
    """Map vertices from cut model back to original model indices"""
    originalPolyData = originalModel.GetPolyData()
    cutPolyData = cutModel.GetPolyData()

    originalPoints = originalPolyData.GetPoints()
    cutPoints = cutPolyData.GetPoints()

    selectedVertices = []

    # For each vertex in cut model, find its index in original model
    for i in range(cutPoints.GetNumberOfPoints()):
      cutPoint = cutPoints.GetPoint(i)

      # Find matching point in original (within small tolerance)
      for j in range(originalPoints.GetNumberOfPoints()):
        originalPoint = originalPoints.GetPoint(j)
        dist = vtk.vtkMath.Distance2BetweenPoints(cutPoint, originalPoint)
        if dist < 1e-10:  # Very small tolerance for floating point comparison
          selectedVertices.append(j)
          break

    return selectedVertices

  def selectMeshRegionByExistingCurve(self, modelNode, curveNode):
    """Use existing closed curve markup with Slicer's Dynamic Modeler Curve Cut"""

    # Debug: check inputs
    print(f"[CurveCut] Model: {modelNode.GetName()}, polydata points: {modelNode.GetPolyData().GetNumberOfPoints() if modelNode.GetPolyData() else 'None'}")
    print(f"[CurveCut] Curve: {curveNode.GetName()}, control points: {curveNode.GetNumberOfControlPoints()}")
    print(f"[CurveCut] Curve class: {curveNode.GetClassName()}")

    # 1. Setup Dynamic Modeler
    dynamicModelerNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLDynamicModelerNode")
    dynamicModelerNode.SetToolName("Curve cut")

    # Debug: print available reference roles for this tool
    tool = slicer.modules.dynamicmodeler.logic().GetDynamicModelerTool(dynamicModelerNode)
    if tool:
      print(f"[CurveCut] Tool name: {tool.GetName()}")
      print(f"[CurveCut] Number of input nodes: {tool.GetNumberOfInputNodes()}")
      for i in range(tool.GetNumberOfInputNodes()):
        print(f"[CurveCut]   Input {i}: role='{tool.GetNthInputNodeReferenceRole(i)}', name='{tool.GetNthInputNodeName(i)}'")
      print(f"[CurveCut] Number of output nodes: {tool.GetNumberOfOutputNodes()}")
      for i in range(tool.GetNumberOfOutputNodes()):
        print(f"[CurveCut]   Output {i}: role='{tool.GetNthOutputNodeReferenceRole(i)}', name='{tool.GetNthOutputNodeName(i)}'")
    else:
      print("[CurveCut] WARNING: Could not get tool object")

    dynamicModelerNode.SetNodeReferenceID("CurveCut.InputModel", modelNode.GetID())
    dynamicModelerNode.SetNodeReferenceID("CurveCut.InputCurve", curveNode.GetID())

    # Create an "inside point" at the centroid of the curve control points
    # This tells the Dynamic Modeler which side of the curve is "inside"
    insidePointNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", "TempInsidePoint")
    centroid = [0.0, 0.0, 0.0]
    for i in range(curveNode.GetNumberOfControlPoints()):
      pos = curveNode.GetNthControlPointPosition(i)
      centroid[0] += pos[0]
      centroid[1] += pos[1]
      centroid[2] += pos[2]
    n = curveNode.GetNumberOfControlPoints()
    centroid = [c / n for c in centroid]
    insidePointNode.AddControlPoint(centroid, "inside")
    print(f"[CurveCut] Inside point: [{centroid[0]:.3f}, {centroid[1]:.3f}, {centroid[2]:.3f}]")
    dynamicModelerNode.SetNodeReferenceID("CurveCut.InsidePoint", insidePointNode.GetID())

    # 2. Create output model nodes
    insideModel = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode")
    insideModel.SetName("TempInsideModel")
    outsideModel = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode")
    outsideModel.SetName("TempOutsideModel")
    dynamicModelerNode.SetNodeReferenceID("CurveCut.OutputInside", insideModel.GetID())
    dynamicModelerNode.SetNodeReferenceID("CurveCut.OutputOutside", outsideModel.GetID())

    # 3. Execute the cut
    try:
      slicer.modules.dynamicmodeler.logic().RunDynamicModelerTool(dynamicModelerNode)

      # 4. Debug: check output
      insidePolyData = insideModel.GetPolyData()
      outsidePolyData = outsideModel.GetPolyData()
      print(f"[CurveCut] Inside polydata: {insidePolyData}, Outside polydata: {outsidePolyData}")
      if insidePolyData:
        print(f"[CurveCut] Inside points: {insidePolyData.GetNumberOfPoints()}, cells: {insidePolyData.GetNumberOfCells()}")
      if outsidePolyData:
        print(f"[CurveCut] Outside points: {outsidePolyData.GetNumberOfPoints()}, cells: {outsidePolyData.GetNumberOfCells()}")
      if not insidePolyData and not outsidePolyData:
        print("[CurveCut] ERROR: Dynamic Modeler produced no output polydata")

      # 6. Get vertex indices from the inside model
      selectedVertices = self.mapCutModelToOriginalVertices(modelNode, insideModel)

      numPoints = curveNode.GetNumberOfControlPoints()
      print(f"[CurveCut] Curve Cut with {numPoints} control points produced {len(selectedVertices)} vertices")

      # 7. Cleanup temporary nodes
      slicer.mrmlScene.RemoveNode(insideModel)
      slicer.mrmlScene.RemoveNode(outsideModel)
      slicer.mrmlScene.RemoveNode(insidePointNode)
      slicer.mrmlScene.RemoveNode(dynamicModelerNode)

      return selectedVertices

    except Exception as e:
      print(f"[CurveCut] Exception: {e}")
      # Cleanup on failure
      slicer.mrmlScene.RemoveNode(insideModel)
      slicer.mrmlScene.RemoveNode(outsideModel)
      slicer.mrmlScene.RemoveNode(insidePointNode)
      slicer.mrmlScene.RemoveNode(dynamicModelerNode)
      raise e

  def selectMeshRegionBySelectedPoints(self, modelNode, markupNode, selectedPointIndices, filterToLandmarkSide=True):
    """Select mesh vertices inside a closed curve using Polygon Area method."""
    numPoints = markupNode.GetNumberOfControlPoints()
    if numPoints < 3:
      print(f"[Selection] Rejected - need at least 3 control points (got {numPoints})")
      return []

    selectedPointIndices = list(range(numPoints))
    print(f"[Selection] Algorithm: Polygon Area (closed curve with {numPoints} control points)")

    result = self.selectMeshRegionByPolygonArea(
      modelNode, markupNode, selectedPointIndices,
      filterToLandmarkSide=filterToLandmarkSide)
    print(f"[Selection] Polygon Area produced {len(result)} vertices")
    return result
  
  def selectMirroredRegion(self, modelNode, markupNode, selectedPointIndices):
    """Select the mirrored region by mirroring landmarks/curve and applying selection.
    
    This ensures we get a complete region on the opposite side, not just boundary vertices.
    Only proceeds if the mirrored landmarks form a complete area (not just an outline).
    
    Args:
      modelNode: The model node containing the mesh
      markupNode: The markup node containing landmarks or curve
      selectedPointIndices: List of selected point indices
    
    Returns:
      List of mirrored vertex indices
    """
    try:
      # Create a mirrored copy of the markup node
      mirroredMarkupNode = self._createMirroredMarkup(markupNode, modelNode)
      
      if not mirroredMarkupNode:
        return []
      
      # Check if mirrored landmarks form a complete area (not just an outline)
      if not self._landmarksFormCompleteArea(mirroredMarkupNode, modelNode):
        logging.debug("Mirrored landmarks only form an outline; skipping mirrored selection")
        slicer.mrmlScene.RemoveNode(mirroredMarkupNode)
        return []
      
      # Apply the same selection method to the mirrored landmarks/curve
      mirroredVertices = self.selectMeshRegionBySelectedPoints(modelNode, mirroredMarkupNode, selectedPointIndices)
      
      # Clean up the temporary mirrored markup node
      slicer.mrmlScene.RemoveNode(mirroredMarkupNode)
      
      return mirroredVertices
    except Exception as e:
      logging.error("Error selecting mirrored region: %s", e)
      return []
  
  def _landmarksFormCompleteArea(self, markupNode, modelNode):
    """Check if landmarks form a complete area (not just an outline).
    
    For a closed curve, this checks if the landmarks form a polygon that can
    enclose an area, not just a line. For fiducial landmarks, this checks if
    they form a connected region with interior points.
    
    Args:
      markupNode: The markup node containing landmarks or curve
      modelNode: The model node containing the mesh
    
    Returns:
      True if landmarks form a complete area, False if they only form an outline
    """
    numControlPoints = markupNode.GetNumberOfControlPoints()
    
    if numControlPoints < 3:
      return False  # Need at least 3 points to form a polygon
    
    # Get all landmark positions
    landmarkPositions = []
    for i in range(numControlPoints):
      pos = markupNode.GetNthControlPointPosition(i)
      landmarkPositions.append(pos)
    
    landmarkPositions = np.array(landmarkPositions)
    
    # For a closed curve, check if landmarks form a proper polygon
    if markupNode.GetClassName() == "vtkMRMLMarkupsClosedCurveNode":
      # Check if the polygon has area (not just a line)
      # Calculate the area of the polygon formed by landmarks
      area = self._calculatePolygonArea(landmarkPositions)
      
      # If area is too small, it's likely just an outline
      # Use a threshold based on the bounding box of landmarks
      if len(landmarkPositions) > 0:
        bboxSize = np.max(landmarkPositions, axis=0) - np.min(landmarkPositions, axis=0)
        maxBboxSize = np.max(bboxSize)
        minArea = (maxBboxSize * 0.1) ** 2  # At least 10% of bbox size squared
        
        if area < minArea:
          logging.debug(
            "Landmarks outline area %.6f below minimum %.6f", area, minArea)
          return False
      
      return True
    
    # For fiducial landmarks, check if they form a connected region
    # by checking if they're spread out (not just on a line)
    if numControlPoints >= 3:
      # Calculate the convex hull area or check if points are collinear
      # If points are mostly collinear, they form a line, not an area
      try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(landmarkPositions)
        # For 3D points, use the volume (which is actually area for 2D projection)
        # For 2D points, use the area
        if len(landmarkPositions[0]) == 3:
          # 3D points - calculate area of 2D projection
          hullArea = self._calculatePolygonArea(landmarkPositions[hull.vertices])
        else:
          hullArea = hull.volume
        
        # Check if area is significant
        bboxSize = np.max(landmarkPositions, axis=0) - np.min(landmarkPositions, axis=0)
        maxBboxSize = np.max(bboxSize)
        minArea = (maxBboxSize * 0.1) ** 2
        
        if hullArea < minArea:
          logging.debug(
            "Landmark hull area %.6f below minimum %.6f", hullArea, minArea)
          return False
      except ImportError:
        # scipy not available, use polygon area calculation
        area = self._calculatePolygonArea(landmarkPositions)
        bboxSize = np.max(landmarkPositions, axis=0) - np.min(landmarkPositions, axis=0)
        maxBboxSize = np.max(bboxSize)
        minArea = (maxBboxSize * 0.1) ** 2
        
        if area < minArea:
          logging.debug(
            "Landmarks outline area %.6f below minimum %.6f", area, minArea)
          return False
      except:
        # If convex hull fails, assume they form an area
        pass
    
    return True
  
  def _calculatePolygonArea(self, points):
    """Calculate the area of a polygon defined by points.
    
    Uses the shoelace formula for 2D projection (using X-Y plane).
    
    Args:
      points: numpy array of shape (n, 3) with polygon vertices
    
    Returns:
      Area of the polygon
    """
    if len(points) < 3:
      return 0.0
    
    # Project to 2D (use X-Y plane, or find the plane with largest spread)
    # Find the axis with smallest spread (this is the normal direction)
    spreads = np.max(points, axis=0) - np.min(points, axis=0)
    normalAxis = np.argmin(spreads)
    
    # Use the other two axes for 2D projection
    if normalAxis == 0:
      projPoints = points[:, [1, 2]]
    elif normalAxis == 1:
      projPoints = points[:, [0, 2]]
    else:
      projPoints = points[:, [0, 1]]
    
    # Shoelace formula
    n = len(projPoints)
    area = 0.0
    for i in range(n):
      j = (i + 1) % n
      area += projPoints[i, 0] * projPoints[j, 1]
      area -= projPoints[j, 0] * projPoints[i, 1]
    
    return abs(area) / 2.0
  
  def _createMirroredMarkup(self, markupNode, modelNode):
    """Create a mirrored copy of the markup node.
    
    Args:
      markupNode: The original markup node
      modelNode: The model node to determine mirroring center
    
    Returns:
      A new mirrored markup node, or None on error
    """
    try:
      # Get mesh center for mirroring
      polyData = modelNode.GetPolyData()
      points = polyData.GetPoints()
      allPoints = []
      for i in range(polyData.GetNumberOfPoints()):
        point = points.GetPoint(i)
        allPoints.append(point)
      
      allPoints = np.array(allPoints)
      minCoords = np.min(allPoints, axis=0)
      maxCoords = np.max(allPoints, axis=0)
      center = (minCoords + maxCoords) / 2.0
      
      # Mirror across X-axis (axis 0)
      mirrorAxis = 0
      
      # Create a mirrored closed curve
      mirroredMarkupNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsClosedCurveNode", "MirroredCurve")
      
      # Mirror all control points
      numControlPoints = markupNode.GetNumberOfControlPoints()
      for i in range(numControlPoints):
        pos = markupNode.GetNthControlPointPosition(i)
        mirroredPos = list(pos)
        mirroredPos[mirrorAxis] = 2 * center[mirrorAxis] - mirroredPos[mirrorAxis]
        mirroredMarkupNode.AddControlPoint(mirroredPos, markupNode.GetNthControlPointLabel(i))
      
      return mirroredMarkupNode
    except Exception as e:
      print(f"Error creating mirrored markup: {e}")
      return None
  
  def findMirroredVertices(self, modelNode, selectedVertices):
    """Find mirrored vertices on the opposite side of the 3D mesh.
    
    Args:
      modelNode: The model node containing the mesh
      selectedVertices: List of selected vertex indices
    
    Returns:
      List of mirrored vertex indices
    """
    polyData = modelNode.GetPolyData()
    points = polyData.GetPoints()
    numPoints = polyData.GetNumberOfPoints()
    
    if numPoints == 0:
      return []
    
    # Calculate the center of the mesh to determine the symmetry plane
    # Typically, meshes are symmetric about the YZ plane (X=0) or XZ plane (Y=0)
    # We'll determine the symmetry plane by finding the axis with the largest spread
    allPoints = []
    for i in range(numPoints):
      point = points.GetPoint(i)
      allPoints.append(point)
    
    allPoints = np.array(allPoints)
    
    # Calculate bounding box and center
    minCoords = np.min(allPoints, axis=0)
    maxCoords = np.max(allPoints, axis=0)
    center = (minCoords + maxCoords) / 2.0
    
    # Use X-axis (axis 0) for bilateral symmetry mirroring
    # This is the standard for biological models (left-right symmetry)
    mirrorAxis = 0  # 0=X, 1=Y, 2=Z

    # Create a set for fast lookup of selected vertices
    selectedSet = set(selectedVertices)
    
    # For each selected vertex, find its mirrored counterpart
    mirroredVertices = []
    mirroredPositions = []
    
    for vertexId in selectedVertices:
      point = points.GetPoint(vertexId)
      # Mirror the point across the symmetry plane
      mirroredPoint = list(point)
      mirroredPoint[mirrorAxis] = 2 * center[mirrorAxis] - mirroredPoint[mirrorAxis]
      mirroredPositions.append((vertexId, mirroredPoint))
    
    # Find the closest vertex to each mirrored position
    for originalVertexId, mirroredPos in mirroredPositions:
      closestVertex = -1
      minDist = float('inf')
      
      # Search for the closest vertex to the mirrored position
      for i in range(numPoints):
        if i in selectedSet:
          continue  # Skip already selected vertices
        
        point = points.GetPoint(i)
        dist = vtk.vtkMath.Distance2BetweenPoints(mirroredPos, point)
        
        # Use a threshold to avoid selecting vertices that are too far
        # This threshold is based on the average edge length
        if dist < minDist:
          minDist = dist
          closestVertex = i
      
      # Only add if we found a reasonably close vertex
      # Use a threshold based on the mesh size
      spreads = maxCoords - minCoords
      meshSize = np.max(spreads)
      threshold = (meshSize * 0.01) ** 2  # 1% of mesh size as threshold
      
      if closestVertex >= 0 and minDist < threshold:
        if closestVertex not in mirroredVertices:
          mirroredVertices.append(closestVertex)
    
    # Filter mirrored vertices to only include those that form a connected region
    if len(mirroredVertices) > 0:
      mirroredVertices = self._filterToConnectedRegion(modelNode, mirroredVertices)

    return mirroredVertices
  
  def _filterToConnectedRegion(self, modelNode, vertices):
    """Filter vertices to only include those that form a connected region.
    
    Args:
      modelNode: The model node containing the mesh
      vertices: List of vertex indices
    
    Returns:
      Filtered list of vertices that form a connected region
    """
    if len(vertices) == 0:
      return vertices
    
    polyData = modelNode.GetPolyData()
    numPoints = polyData.GetNumberOfPoints()
    
    # Build adjacency list for mesh connectivity
    adjacency = [set() for _ in range(numPoints)]
    
    for i in range(polyData.GetNumberOfCells()):
      cell = polyData.GetCell(i)
      pointIds = cell.GetPointIds()
      numCellPoints = pointIds.GetNumberOfIds()
      
      # Connect all pairs of vertices in this face
      for j in range(numCellPoints):
        for k in range(j + 1, numCellPoints):
          v1 = pointIds.GetId(j)
          v2 = pointIds.GetId(k)
          adjacency[v1].add(v2)
          adjacency[v2].add(v1)
    
    # Find the largest connected component
    vertexSet = set(vertices)
    visited = set()
    components = []
    
    for vertexId in vertices:
      if vertexId in visited:
        continue
      
      # BFS to find connected component
      component = []
      queue = [vertexId]
      visited.add(vertexId)
      
      while queue:
        current = queue.pop(0)
        component.append(current)
        
        # Check neighbors
        for neighbor in adjacency[current]:
          if neighbor in vertexSet and neighbor not in visited:
            visited.add(neighbor)
            queue.append(neighbor)
      
      if len(component) > 0:
        components.append(component)
    
    # Return the largest connected component
    if len(components) == 0:
      return []
    
    largestComponent = max(components, key=len)
    
    # Only return if the component has a reasonable size (at least 10% of original vertices)
    minSize = max(1, len(vertices) // 10)
    if len(largestComponent) < minSize:
      return []

    # Grow the region from the connected component to form a complete area
    # This ensures we get the full region, not just boundary vertices
    grownRegion = self._growRegionFromVertices(modelNode, largestComponent, adjacency)

    return grownRegion
  
  def _growRegionFromVertices(self, modelNode, seedVertices, adjacency):
    """Grow a region from seed vertices by including all connected neighbors.
    
    This ensures we get a complete area, not just boundary vertices.
    
    Args:
      modelNode: The model node containing the mesh
      seedVertices: List of seed vertex indices
      adjacency: Pre-built adjacency list
    
    Returns:
      List of vertices in the grown region
    """
    if len(seedVertices) == 0:
      return []
    
    polyData = modelNode.GetPolyData()
    points = polyData.GetPoints()
    
    # Get the side of the mesh where seed vertices are located
    # This ensures we only grow on the correct side
    seedSet = set(seedVertices)
    allPoints = []
    for i in range(polyData.GetNumberOfPoints()):
      point = points.GetPoint(i)
      allPoints.append(point)
    
    allPoints = np.array(allPoints)
    minCoords = np.min(allPoints, axis=0)
    maxCoords = np.max(allPoints, axis=0)
    center = (minCoords + maxCoords) / 2.0
    
    # Determine which side the seed vertices are on (X-axis for bilateral symmetry)
    mirrorAxis = 0
    seedPositions = [points.GetPoint(v) for v in seedVertices]
    avgSeedPos = np.mean(seedPositions, axis=0)
    seedSide = avgSeedPos[mirrorAxis] - center[mirrorAxis]
    seedSideSign = 1 if seedSide >= 0 else -1
    
    # BFS to grow region from seed vertices
    region = set(seedVertices)
    queue = list(seedVertices)
    visited = set(seedVertices)
    
    # Limit growth to prevent selecting the entire mesh
    maxGrowthFactor = 3  # Grow up to 3x the original size
    maxRegionSize = len(seedVertices) * maxGrowthFactor
    
    while queue and len(region) < maxRegionSize:
      current = queue.pop(0)
      
      # Check neighbors
      for neighbor in adjacency[current]:
        if neighbor in visited:
          continue
        
        # Only include neighbors on the same side as seed vertices
        neighborPoint = points.GetPoint(neighbor)
        neighborSide = neighborPoint[mirrorAxis] - center[mirrorAxis]
        neighborSideSign = 1 if neighborSide >= 0 else -1
        
        if neighborSideSign == seedSideSign:
          visited.add(neighbor)
          region.add(neighbor)
          queue.append(neighbor)
    
    return list(region)
  
  def filterVerticesToSameSide(self, modelNode, markupNode, selectedVertices):
    """Filter selected vertices to only include those on the same side as the landmarks/curve.
    
    When mirror checkbox is unchecked, ALL vertices on the opposite side are filtered out,
    regardless of whether they form a complete area or not.
    
    Args:
      modelNode: The model node containing the mesh
      markupNode: The markup node containing landmarks or curve
      selectedVertices: List of selected vertex indices
    
    Returns:
      Filtered list of vertex indices on the same side as the landmarks/curve
    """
    polyData = modelNode.GetPolyData()
    points = polyData.GetPoints()
    
    if len(selectedVertices) == 0:
      return selectedVertices
    
    # Get landmark/curve positions to determine which side they're on
    landmarkPositions = []
    numControlPoints = markupNode.GetNumberOfControlPoints()
    for i in range(numControlPoints):
      pos = markupNode.GetNthControlPointPosition(i)
      landmarkPositions.append(pos)
    
    if len(landmarkPositions) == 0:
      return selectedVertices

    landmarkPositions = np.array(landmarkPositions)

    # Use helper function to detect side
    mirrorAxis, landmarkSideSign, center, ranges, midlineThreshold = self._detectLandmarkSide(polyData, landmarkPositions)

    # Filter vertices to same side, excluding midline
    filteredVertices = []
    oppositeSideCount = midlineCount = 0

    for vertexId in selectedVertices:
      point = points.GetPoint(vertexId)
      vertexSide = point[mirrorAxis] - center[mirrorAxis]
      vertexSideSign = 1 if vertexSide >= 0 else -1

      if vertexSideSign != landmarkSideSign and abs(vertexSide) >= midlineThreshold:
        oppositeSideCount += 1
        continue

      if abs(vertexSide) < midlineThreshold:
        midlineCount += 1

      filteredVertices.append(vertexId)

    return filteredVertices

  def visualizeRegionSelection(self, modelNode, selectedVertices):
    """Visualize the selected region by coloring vertices"""
    polyData = modelNode.GetPolyData()
    
    # Remove any existing RegionSelection array first
    if polyData.GetPointData().GetArray("RegionSelection"):
      polyData.GetPointData().RemoveArray("RegionSelection")
    
    # Create a scalar array for selection visualization
    selectionArray = vtk.vtkIntArray()
    selectionArray.SetName("RegionSelection")
    selectionArray.SetNumberOfTuples(polyData.GetNumberOfPoints())
    selectionArray.Fill(0)  # 0 = not selected
    
    # Mark selected vertices
    for vertexId in selectedVertices:
      if vertexId < polyData.GetNumberOfPoints():
        selectionArray.SetValue(vertexId, 1)

    # Add array to mesh
    polyData.GetPointData().AddArray(selectionArray)
    polyData.GetPointData().SetActiveScalars("RegionSelection")
    polyData.Modified()
    
    # Set up display
    displayNode = modelNode.GetDisplayNode()
    if displayNode:
      displayNode.SetScalarVisibility(True)
      displayNode.SetActiveScalarName("RegionSelection")
      displayNode.SetScalarRange(0, 1)
      
      # Use a simple color map: gray for unselected, red for selected
      displayNode.SetAndObserveColorNodeID('vtkMRMLColorTableNodeFileColdToHotRainbow.txt')
      displayNode.Modified()
    
    modelNode.Modified()
    slicer.app.processEvents()  # Force UI update
  
  def createModelFromSelectedVertices(self, modelNode, selectedVertices, exportName=None):
    """Create a new model from selected vertices

    Args:
      modelNode: The source model node
      selectedVertices: List of vertex indices to include
      exportName: Optional custom name for the exported model
    """
    try:
      # Get the original mesh data
      originalPolyData = modelNode.GetPolyData()
      points = originalPolyData.GetPoints()
      polys = originalPolyData.GetPolys()
      
      # Create new polydata
      newPolyData = vtk.vtkPolyData()
      newPoints = vtk.vtkPoints()
      newPolys = vtk.vtkCellArray()
      
      # Create mapping from old vertex IDs to new vertex IDs
      vertexMapping = {}
      newVertexId = 0
      
      # Add selected vertices to new mesh
      uniqueSelectedVertices = list(dict.fromkeys(selectedVertices))
      for vertexId in uniqueSelectedVertices:
        vertexMapping[vertexId] = newVertexId
        point = points.GetPoint(vertexId)
        newPoints.InsertNextPoint(point)
        newVertexId += 1
      
      # Add faces that only contain selected vertices
      polys.InitTraversal()
      cell = vtk.vtkIdList()
      originalCellIds = []
      currentCellId = 0
      while polys.GetNextCell(cell):
        # Check if all vertices of this cell are selected
        allVerticesSelected = True
        for i in range(cell.GetNumberOfIds()):
          if cell.GetId(i) not in vertexMapping:
            allVerticesSelected = False
            break
        
        # If all vertices are selected, add the face to new mesh
        if allVerticesSelected:
          newCell = vtk.vtkIdList()
          for i in range(cell.GetNumberOfIds()):
            newCell.InsertNextId(vertexMapping[cell.GetId(i)])
          newPolys.InsertNextCell(newCell)
          originalCellIds.append(currentCellId)
        currentCellId += 1
      
      # Set up the new polydata
      newPolyData.SetPoints(newPoints)
      newPolyData.SetPolys(newPolys)

      # Copy over point data arrays (including texture coordinates)
      originalPointData = originalPolyData.GetPointData()
      if originalPointData:
        newPointData = newPolyData.GetPointData()

        originalTCoords = originalPointData.GetTCoords()
        # Copy generic point data arrays, skipping TCoords (handled separately)
        for arrayIndex in range(originalPointData.GetNumberOfArrays()):
          sourceArray = originalPointData.GetArray(arrayIndex)
          if not sourceArray:
            continue
          if originalTCoords and sourceArray is originalTCoords:
            continue

          newArray = sourceArray.NewInstance()
          newArray.SetName(sourceArray.GetName() or "")

          if hasattr(newArray, "SetNumberOfComponents"):
            newArray.SetNumberOfComponents(sourceArray.GetNumberOfComponents())

          if hasattr(newArray, "SetNumberOfTuples") and hasattr(newArray, "SetTuple"):
            newArray.SetNumberOfTuples(len(vertexMapping))
            for originalId, mappedId in vertexMapping.items():
              newArray.SetTuple(mappedId, sourceArray.GetTuple(originalId))
          elif hasattr(newArray, "SetNumberOfTuples"):
            newArray.SetNumberOfTuples(len(vertexMapping))
            for originalId, mappedId in vertexMapping.items():
              for component in range(sourceArray.GetNumberOfComponents()):
                newArray.SetComponent(mappedId, component, sourceArray.GetComponent(originalId, component))
          elif hasattr(newArray, "SetNumberOfValues"):
            newArray.SetNumberOfValues(len(vertexMapping))
            for originalId, mappedId in vertexMapping.items():
              newArray.SetValue(mappedId, sourceArray.GetValue(originalId))

          newPointData.AddArray(newArray)

        # Copy texture coordinates explicitly so they remain active
        if originalTCoords:
          newTCoords = originalTCoords.NewInstance()
          newTCoords.SetNumberOfComponents(originalTCoords.GetNumberOfComponents())
          newTCoords.SetName(originalTCoords.GetName() or "TextureCoordinates")
          newTCoords.SetNumberOfTuples(len(vertexMapping))
          for originalId, mappedId in vertexMapping.items():
            newTCoords.SetTuple(mappedId, originalTCoords.GetTuple(originalId))
          newPointData.SetTCoords(newTCoords)

        # Preserve active scalars if present
        if originalPointData.GetScalars():
          scalarName = originalPointData.GetScalars().GetName()
          if scalarName:
            newPointData.SetActiveScalars(scalarName)

      # Copy cell data arrays corresponding to retained faces
      originalCellData = originalPolyData.GetCellData()
      if originalCellData and originalCellIds:
        newCellData = newPolyData.GetCellData()
        for arrayIndex in range(originalCellData.GetNumberOfArrays()):
          sourceArray = originalCellData.GetArray(arrayIndex)
          if not sourceArray:
            continue

          newArray = sourceArray.NewInstance()
          newArray.SetName(sourceArray.GetName() or "")

          if hasattr(newArray, "SetNumberOfComponents"):
            newArray.SetNumberOfComponents(sourceArray.GetNumberOfComponents())

          if hasattr(newArray, "SetNumberOfTuples") and hasattr(newArray, "SetTuple"):
            newArray.SetNumberOfTuples(len(originalCellIds))
            for newId, originalId in enumerate(originalCellIds):
              newArray.SetTuple(newId, sourceArray.GetTuple(originalId))
          elif hasattr(newArray, "SetNumberOfTuples"):
            newArray.SetNumberOfTuples(len(originalCellIds))
            for newId, originalId in enumerate(originalCellIds):
              for component in range(sourceArray.GetNumberOfComponents()):
                newArray.SetComponent(newId, component, sourceArray.GetComponent(originalId, component))
          elif hasattr(newArray, "SetNumberOfValues"):
            newArray.SetNumberOfValues(len(originalCellIds))
            for newId, originalId in enumerate(originalCellIds):
              newArray.SetValue(newId, sourceArray.GetValue(originalId))

          newCellData.AddArray(newArray)

        if originalCellData.GetScalars():
          scalarName = originalCellData.GetScalars().GetName()
          if scalarName:
            newCellData.SetActiveScalars(scalarName)

      newPolyData.BuildCells()
      newPolyData.BuildLinks()
      
      # Create new model node
      newModelNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode")
      newModelNode.SetAndObservePolyData(newPolyData)

      # Set name - use custom name if provided, otherwise use default
      if exportName:
        newModelNode.SetName(exportName)
      else:
        originalName = modelNode.GetName()
        newModelNode.SetName(f"{originalName}_SelectedRegion")
      
      # Copy display properties from original model
      originalDisplayNode = modelNode.GetDisplayNode()
      if originalDisplayNode:
        newDisplayNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelDisplayNode")
        newDisplayNode.SetColor(originalDisplayNode.GetColor())
        newDisplayNode.SetOpacity(originalDisplayNode.GetOpacity())
        newDisplayNode.SetBackfaceCulling(originalDisplayNode.GetBackfaceCulling())
        newDisplayNode.SetFrontfaceCulling(originalDisplayNode.GetFrontfaceCulling())
        newDisplayNode.SetScalarVisibility(originalDisplayNode.GetScalarVisibility())
        try:
          newDisplayNode.SetInterpolateTexture(originalDisplayNode.GetInterpolateTexture())
        except AttributeError:
          pass
        if originalDisplayNode.GetColorNodeID():
          newDisplayNode.SetAndObserveColorNodeID(originalDisplayNode.GetColorNodeID())
        textureConnection = None
        try:
          textureConnection = originalDisplayNode.GetTextureImageDataConnection()
        except AttributeError:
          textureConnection = None
        if textureConnection:
          newDisplayNode.SetTextureImageDataConnection(textureConnection)
        else:
          try:
            textureImageData = originalDisplayNode.GetTextureImageData()
          except AttributeError:
            textureImageData = None
          if textureImageData:
            newDisplayNode.SetTextureImageData(textureImageData)
        newModelNode.SetAndObserveDisplayNodeID(newDisplayNode.GetID())
      
      # Update the new model
      newModelNode.Modified()
      
      return newModelNode
      
    except Exception as e:
      print(f"Error creating model from selected vertices: {e}")
      return None
  
  def clearRegionSelection(self, modelNode):
    """Clear the region selection visualization"""
    polyData = modelNode.GetPolyData()
    
    # Remove the selection array
    if polyData.GetPointData().GetArray("RegionSelection"):
      polyData.GetPointData().RemoveArray("RegionSelection")
      polyData.Modified()
    
    # Reset display
    displayNode = modelNode.GetDisplayNode()
    if displayNode:
      displayNode.SetScalarVisibility(False)
    
    modelNode.Modified()
  
  def onStartVisualizationButton(self):
    """Start visualization by preparing the scene and showing models"""
    try:
      # Hide markups for clean visualization
      self._hideMarkupsForVisualization(remove=False)
      
      # Hide unwanted models (planes, reference objects, etc.)
      self._hideUnwantedModels()
      
      # Ensure only relevant ATLAS models are visible
      self._ensureModelsAreVisible()
      
      # Update the button text to indicate visualization is active
      self.startVisualizationButton.setText("Visualization Active")
      self.startVisualizationButton.setStyleSheet(ColorTheme.getButtonStyle('primary'))
      
      print("Visualization started - markups hidden and models made visible")
      
    except Exception as e:
      print(f"Error starting visualization: {e}")

  def setUpDeCADir(self, outDir, DeCALOption=False):
    dateTimeStamp = datetime.now().strftime('%Y_%m-%d_%H_%M_%S')
    outputFolderDC = os.path.join(outDir, dateTimeStamp)
    fileNameDictionary = {}
    try:
      os.makedirs(outputFolderDC)

      # Create main subdirectories
      atlasSubDir = os.path.join(outputFolderDC, "ATLAS")
      colorAnalysisSubDir = os.path.join(outputFolderDC, "colorAnalysis")
      os.makedirs(atlasSubDir)
      os.makedirs(colorAnalysisSubDir)

      # ATLAS shape correspondence data goes in ATLAS subdirectory
      alignedLMFolderDC = os.path.join(atlasSubDir, "alignedLMs")
      os.makedirs(alignedLMFolderDC)
      alignedModelFolderDC = os.path.join(atlasSubDir, "alignedModels")
      os.makedirs(alignedModelFolderDC)

      tempLMFolderDC = os.path.join(atlasSubDir, "tempAlignedLMs")
      os.makedirs(tempLMFolderDC)
      tempModelFolderDC = os.path.join(atlasSubDir, "tempAlignedModels")
      os.makedirs(tempModelFolderDC)

      # Resampled models (without UVs) are ATLAS output
      resampledModelFolderDC = os.path.join(atlasSubDir, "resampledModels")
      os.makedirs(resampledModelFolderDC)

      # initialize the filename dictionary
      fileNameDictionary['output'] = str(outputFolderDC)
      fileNameDictionary['atlasSubDir'] = str(atlasSubDir)
      fileNameDictionary['colorAnalysisSubDir'] = str(colorAnalysisSubDir)
      fileNameDictionary['alignedLMs'] = str(alignedLMFolderDC)
      fileNameDictionary['alignedModels'] = str(alignedModelFolderDC)
      fileNameDictionary['resampledModels'] = str(resampledModelFolderDC)
      fileNameDictionary['tempAlignedLMs'] = str(tempLMFolderDC)
      fileNameDictionary['tempAlignedModels'] = str(tempModelFolderDC)

      if DeCALOption:
        ATLASOutputFolder = os.path.join(atlasSubDir, "ATLASOutput")
        os.makedirs(ATLASOutputFolder)
        fileNameDictionary['ATLASOutput'] = str(ATLASOutputFolder)
    except:
      logging.debug('Result directory failed: Could not create output folder')
    return fileNameDictionary


  def onToggleAtlasDCL(self):
    if self.calculateAtlasOptionDCL.checked == True:
      self.atlasCollapsibleButtonDCL.collapsed = True
      self.atlasCollapsibleButtonDCL.enabled = False
    else:
      self.atlasCollapsibleButtonDCL.collapsed = False
      self.atlasCollapsibleButtonDCL.enabled = True
    self.onParameterSelectDCL()

  # --- Functions for Visualize Tab ---

  def onSubjectIDSelect(self):
    # This function is part of the original Heatmap mode and is unchanged
    try:
      if self.resultNode is None:
        print("Error: No result node selected")
        return
        
      subjectID = self.subjectIDBox.currentText
      displayNode = self.resultNode.GetDisplayNode()
      if displayNode is None:
        print("Error: Selected model has no display node")
        return
        
      displayNode.SetActiveScalarName(subjectID)
      displayNode.SetAndObserveColorNodeID('vtkMRMLColorTableNodeFilePlasma.txt')
      print(subjectID)
    except Exception as e:
      print(f"Error: {str(e)}")

  def onVisualizeMeshSelect(self):
    # This function is part of the original Heatmap mode and is unchanged
    if bool(self.meshSelect.currentNode()):
      self.resultNode = self.meshSelect.currentNode()
      
      # Check if the node has a display node before accessing it
      displayNode = self.resultNode.GetDisplayNode()
      if displayNode is not None:
        displayNode.SetVisibility(True)
        displayNode.SetScalarVisibility(True)
      else:
        print("Warning: Selected model has no display node")
        return
      
      # Check if the node has polydata before accessing it
      polyData = self.resultNode.GetPolyData()
      if polyData is None:
        print("Warning: Selected model has no polydata")
        return
        
      resultData = polyData.GetPointData()
      self.subjectIDBox.enabled = True
      self.subjectIDBox.clear() # Clear previous items
      arrayNumber = resultData.GetNumberOfArrays()
      if arrayNumber > 0:
        for i in range(resultData.GetNumberOfArrays()):
          arrayName = resultData.GetArrayName(i)
          self.subjectIDBox.addItem(arrayName)
      else:
        self.subjectIDBox.clear()
        self.subjectIDBox.enabled = False

  def onVisualizationModeChanged(self):
    # NEW: Switches between Heatmap and Interpolation frames
    isHeatmapMode = self.visualizeHeatmapRadio.isChecked()
    self.heatmapFrame.setVisible(isHeatmapMode)
    self.interpolationFrame.setVisible(not isHeatmapMode)

    # Clean up the other mode's visualization
    if isHeatmapMode and self.interpolatedModelNode:
        self.interpolatedModelNode.SetDisplayVisibility(False)
    elif not isHeatmapMode and self.meshSelect.currentNode():
        self.meshSelect.currentNode().GetDisplayNode().SetScalarVisibility(False)

  def onVisOriginalModelDirChanged(self, directory):
    # NEW: Populates the file selector combobox when a directory is chosen
    self.visOriginalModelFileSelector.clear()
    self.visOriginalModelFileSelector.enabled = False
    if not os.path.isdir(directory):
      return

    modelExtensions = ['*.ply', '*.stl', '*.obj', '*.vtk', '*.vtp']
    foundFiles = []
    for ext in modelExtensions:
        foundFiles.extend(fnmatch.filter(os.listdir(directory), ext))

    if foundFiles:
        self.visOriginalModelFileSelector.addItems(sorted(foundFiles))
        self.visOriginalModelFileSelector.enabled = True

  def onVisOriginalModelFileSelected(self, index):
    # NEW: Loads the selected model file and enables the interpolation slider
    if self.selectedOriginalModelNode:
        slicer.mrmlScene.RemoveNode(self.selectedOriginalModelNode)
        self.selectedOriginalModelNode = None

    directory = self.visOriginalModelDirSelector.currentPath
    fileName = self.visOriginalModelFileSelector.currentText
    if not fileName:
      self.onInterpolationInputChanged()
      return

    filePath = os.path.join(directory, fileName)
    try:
        self.selectedOriginalModelNode = slicer.util.loadModel(filePath)
    except Exception as e:
        slicer.util.errorDisplay(f"Failed to load model: {filePath}\n\n{e}")
        self.selectedOriginalModelNode = None

    self.onInterpolationInputChanged()


  def onInterpolationInputChanged(self):
    # MODIFIED: Provides a much more detailed error message for debugging
    atlasNode = self.atlasModelSelect.currentNode()
    originalNode = self.selectedOriginalModelNode

    if atlasNode and originalNode:
        atlasPolyData = atlasNode.GetPolyData()
        originalPolyData = originalNode.GetPolyData()
        
        if not (atlasPolyData and originalPolyData):
            self.interpolationSlider.enabled = False
            return
            
        atlasPointCount = atlasPolyData.GetNumberOfPoints()
        originalPointCount = originalPolyData.GetNumberOfPoints()

        # Check for matching point counts
        if atlasPointCount != originalPointCount:
            # Construct a detailed error message
            errorMsg = (f"Vertex count mismatch! Cannot interpolate.\n\n"
                        f"Atlas Model: '{atlasNode.GetName()}' has {atlasPointCount} vertices.\n"
                        f"Aligned Model: '{originalNode.GetName()}' has {originalPointCount} vertices.\n\n"
                        f"Please ensure both models are from the exact same DeCA analysis run.")
            slicer.util.warningDisplay(errorMsg, windowTitle="Interpolation Error")
            print(errorMsg) # Also print to Python console for easy copy/paste
            self.interpolationSlider.enabled = False
            return
            
        self.interpolationSlider.enabled = True
        # Trigger an initial update
        self.onInterpolationSliderChanged(self.interpolationSlider.value)
    else:
        self.interpolationSlider.enabled = False

  def onInterpolationSliderChanged(self, value):
    # MODIFIED: Uses self.selectedOriginalModelNode instead of a combobox
    atlasNode = self.atlasModelSelect.currentNode()
    originalNode = self.selectedOriginalModelNode # <-- The change is here

    if not (atlasNode and originalNode):
        return

    # Get polydata from nodes
    atlasPolyData = atlasNode.GetPolyData()
    originalPolyData = originalNode.GetPolyData()

    # Get points as numpy arrays
    atlasPoints_np = vtk_np.vtk_to_numpy(atlasPolyData.GetPoints().GetData())
    originalPoints_np = vtk_np.vtk_to_numpy(originalPolyData.GetPoints().GetData())

    # Perform linear interpolation
    interpolatedPoints_np = (1.0 - value) * originalPoints_np + value * atlasPoints_np

    # Create the output node if it doesn't exist
    if not self.interpolatedModelNode:
        self.interpolatedModelNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", "InterpolatedModel")
        newPolyData = vtk.vtkPolyData()
        newPolyData.SetPolys(originalPolyData.GetPolys())
        self.interpolatedModelNode.SetAndObservePolyData(newPolyData)
        self.interpolatedModelNode.CreateDefaultDisplayNodes()
        if originalNode.GetDisplayNode():
            color = originalNode.GetDisplayNode().GetColor()
            self.interpolatedModelNode.GetDisplayNode().SetColor(color)

    # Update the points of the interpolated model
    points_vtk = vtk.vtkPoints()
    points_vtk.SetData(vtk_np.numpy_to_vtk(interpolatedPoints_np, deep=True))
    self.interpolatedModelNode.GetPolyData().SetPoints(points_vtk)
    self.interpolatedModelNode.GetPolyData().Modified()
    self.interpolatedModelNode.SetDisplayVisibility(True)

  def onParameterSelectDC(self):
    inputPathsSelected = bool(self.meshDirectoryDC.currentPath and self.landmarkDirectoryDC.currentPath and self.outputDirectoryDC.currentPath)
    self.applyButtonDC.enabled = bool(inputPathsSelected)

  def onParameterSelectDCL(self):
    atlasPathSelected = bool(self.DCLBaseModelSelector.currentPath and self.DCLBaseLMSelector.currentPath) or self.calculateAtlasOptionDCL.checked
    inputPathsSelected = bool(self.meshDirectoryDCL.currentPath and self.landmarkDirectoryDCL.currentPath and self.OutputDirectoryDCL.currentPath)
    self.getAtlasButton.enabled = bool(atlasPathSelected and inputPathsSelected)

  def onPointSelectionSelect(self):
    self.subsetApplyButton.enabled = bool(self.DCLLandmarkDirectory.currentPath and self.pointSelection.currentNode())

  def onDCLLandmarkDirectorySelect(self):
    self.subsetApplyButton.enabled = bool(self.DCLLandmarkDirectory.currentPath and self.pointSelection.currentNode())

  def onGenerateAtlasButton(self):
    """
    Generates or loads an atlas model for DeCAL analysis.

    Handles two workflows:
    1. Loading existing atlas: Uses pre-computed atlas model and landmarks
    2. Generating new atlas: Creates average shape from specimen collection

    Outputs atlas files to the DeCA output directory for use in dense
    correspondence analysis.
    """
    # Initializes the InterDeCA logic processor
    logic = InterDeCALogic()

    # Sets up the output directory structure with DeCAL-specific folders
    self.folderNames = self.setUpDeCADir(self.OutputDirectoryDCL.currentPath, True)

    # Validates that directory creation was successful
    if self.folderNames == {}:
      self.logInfoDCL.appendPlainText(f'Output folders could not be created in {self.OutputDirectoryDCL.currentPath}')
      return

    # Stores paths to original data for reference during processing
    self.folderNames['originalLMs'] = self.landmarkDirectoryDCL.currentPath
    self.folderNames['originalModels'] = self.meshDirectoryDCL.currentPath

    # Determines whether to load existing atlas or generate new one
    if self.loadAtlasOptionDCL.checked:
      # Loads existing atlas model from file
      try:
        atlasModelPath = self.DCLBaseModelSelector.currentPath
        self.atlasModel = slicer.util.loadModel(atlasModelPath)
      except:
        self.logInfoDCL.appendPlainText(f"Can't load model from: {atlasModelPath}")
        return

      # Loads corresponding atlas landmarks
      try:
        atlasLMPath = self.DCLBaseLMSelector.currentPath
        self.atlasLMs = slicer.util.loadMarkups(atlasLMPath)
      except:
        print("Can't load from: ", atlasLMPath)
        self.logInfoDCL.appendPlainText(f"Can't load landmarks from: {atlasLMPath}")
        return
    else:
      # Generates new atlas from specimen collection using Procrustes alignment
      removeScale = True  # Removes scale differences during alignment
      self.atlasModel, self.atlasLMs = self.generateNewAtlas(removeScale, self.logInfoDCL)

    if self.atlasModel is None or self.atlasLMs is None:
      self.logInfoDCL.appendPlainText(
        "Atlas generation did not complete. Review the preflight messages above."
      )
      return

    # Saves the atlas model to the colorAnalysis directory for later use
    atlasModelPath = os.path.join(self.folderNames['colorAnalysisSubDir'], 'atlasModel.ply')
    self.logInfoDCL.appendPlainText(f"Saving atlas model to {atlasModelPath}")
    slicer.util.saveNode(self.atlasModel, atlasModelPath)

    # Saves the atlas landmarks alongside the model
    atlasLMPath = os.path.join(self.folderNames['output'], 'atlasLM.mrk.json')
    self.logInfoDCL.appendPlainText(f"Saving atlas landmarks to {atlasLMPath}")
    slicer.util.saveNode(self.atlasLMs, atlasLMPath)

    # Enables the next step button for point number calculation
    self.getPointNumberButton.enabled = True

  def onTabChanged(self, index):
    if SHOW_VISUALIZE_RESULTS and self.tabsWidget.tabText(index) == "Visualize Results":
      # Only update the preview list, don't automatically change the scene
      self.updateBakedPreviewList()
      # Reset the visualization button state
      self.resetVisualizationButton()
    
  def resetVisualizationButton(self):
    """Reset the Start Visualization button to its initial state"""
    if not SHOW_VISUALIZE_RESULTS:
      return
    self.startVisualizationButton.setText("Start Visualization")
    self.startVisualizationButton.setStyleSheet(ColorTheme.getButtonStyle('secondary'))

  def generateNewAtlas(self, removeScale, log):
    """
    Creates an unbiased atlas model from a collection of specimens.

    Implements the following workflow:
    1. Identifies specimen closest to mean shape using Procrustes analysis
    2. Performs rigid alignment of all specimens to this template
    3. Computes average shape from aligned specimens
    4. Cleans up temporary files

    Args:
      removeScale: Boolean to normalize scale during alignment
      log: Qt text widget for progress reporting

    Returns:
      tuple: (atlasModel, atlasLMs) - Generated atlas and average landmarks
    """
    logic = InterDeCALogic()

    log.appendPlainText("Running mesh/landmark preflight validation...")
    try:
      def reportProgress(current, total, subject):
        if current == total or current % 10 == 0:
          log.appendPlainText(f"Preflight: checked {current}/{total} specimens")

      preflight = logic.validateAtlasDataset(
        self.folderNames['originalModels'],
        self.folderNames['originalLMs'],
        coordinateSystem='LPS',
        progressCallback=reportProgress,
      )
    except Exception as error:
      log.appendPlainText(f"Error: Dataset preflight failed unexpectedly: {error}")
      return None, None

    log.appendPlainText(
      f"Preflight checked {preflight['checked']} paired specimens: "
      f"{len(preflight['severe'])} severe issue(s), "
      f"{len(preflight['warnings'])} warning(s)."
    )
    for warning in preflight['warnings'][:20]:
      log.appendPlainText(f"WARNING: {warning}")
    if len(preflight['warnings']) > 20:
      log.appendPlainText(
        f"WARNING: {len(preflight['warnings']) - 20} additional warning(s) omitted."
      )
    if not preflight['passed']:
      for issue in preflight['severe'][:30]:
        log.appendPlainText(f"ERROR: {issue}")
      if len(preflight['severe']) > 30:
        log.appendPlainText(
          f"ERROR: {len(preflight['severe']) - 30} additional severe issue(s) omitted."
        )
      log.appendPlainText(
        "Atlas generation stopped because TPS would be unsafe. "
        "Correct the reported mesh/landmark pairs and try again."
      )
      return None, None

    # Determines which specimen is closest to the mean shape configuration
    # This specimen will serve as the initial template for alignment
    try:
      closestFileName = logic.getClosestToMeanPath(self.folderNames['originalLMs'])
      if closestFileName is None:
        log.appendPlainText("Error: Could not determine closest sample to mean")
        return None, None

      # Extracts the subject ID by removing landmark file extensions
      # Handles multiple extension formats (.fcsv, .mrk.json, etc.)
      subjectID = closestFileName
      fileNameBase = Path(subjectID)
      while fileNameBase.suffix in {'.fcsv', '.mrk', '.json'}:
        fileNameBase = fileNameBase.with_suffix('')
      subjectID = str(fileNameBase)

      # Reports the selected template specimen for user verification
      log.appendPlainText(f"Closest sample to mean: {closestFileName}")
      log.appendPlainText(f"Using subject ID: {subjectID}")
    except Exception as e:
      log.appendPlainText(f"Error finding closest sample to mean: {e}")
      return None, None

    # Resolve the actual landmark/model files by subject ID (handles any extension)
    tempBaseLMs = logic.getLandmarkFileByID(self.folderNames['originalLMs'], subjectID)
    if tempBaseLMs is None:
      log.appendPlainText(f"Can't find landmarks for '{subjectID}' in {self.folderNames['originalLMs']}")
      return None, None

    tempBaseModel = logic.getModelFileByID(self.folderNames['originalModels'], subjectID)
    if tempBaseModel is None:
      log.appendPlainText(f"Can't find model for '{subjectID}' in {self.folderNames['originalModels']}")
      return None, None

    log.appendPlainText(f"Rigid Alignment to: {subjectID}")
    try:
      logic.runAlign(tempBaseModel, tempBaseLMs,
                    self.folderNames['originalModels'], self.folderNames['originalLMs'],
                    self.folderNames['tempAlignedModels'], self.folderNames['tempAlignedLMs'],
                    removeScale)  # Keep parameter since this method still needs it
    except ValueError as errorText:
      log.appendPlainText(str(errorText))
      return None, None

    log.appendPlainText("Generating the average template")
    atlasModel, atlasLMs = logic.runMean(self.folderNames['tempAlignedLMs'],
                                        self.folderNames['tempAlignedModels'])

    slicer.mrmlScene.RemoveNode(tempBaseModel)
    slicer.mrmlScene.RemoveNode(tempBaseLMs)
    shutil.rmtree(self.folderNames['tempAlignedModels'])
    shutil.rmtree(self.folderNames['tempAlignedLMs'])
    return atlasModel, atlasLMs


  def onGetPointNumberButton(self):
    """
    Calculates the number of points in the subsampled atlas.

    Uses the spacing tolerance value to downsample the atlas model and
    reports the resulting point count. This helps users understand the
    density of correspondence points before running DeCAL.
    """
    # Creates logic instance for point calculation
    logic = InterDeCALogic()

    # Subsamples the atlas based on the specified spacing tolerance
    subsampledTemplate, pointNumber = logic.runCheckPoints(self.atlasModel, self.spacingTolerance.value)

    # Reports the point count to help users assess correspondence density
    self.logInfoDCL.appendPlainText(f'The subsampled template has a total of {pointNumber} points.')

    # Enables the DeCAL execution button now that point count is known
    self.DCLApplyButton.enabled = True

  def onTabChanged(self, index):
    if SHOW_VISUALIZE_RESULTS and self.tabsWidget.tabText(index) == "Visualize Results":
      # Only update the preview list, don't automatically change the scene
      self.updateBakedPreviewList()
      # Reset the visualization button state
      self.resetVisualizationButton()

  def updateBakedPreviewList(self):
    if not SHOW_VISUALIZE_RESULTS:
      return
    self.previewTextureCombo.blockSignals(True)
    self.previewTextureCombo.clear()
    d = self.lastBakedTexturesPath
    if d and os.path.isdir(d):
      items = [f for f in os.listdir(d) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.tif'))]
      items.sort()
      self.previewTextureCombo.addItems(items)
    self.previewTextureCombo.blockSignals(False)

  def onPreviewTextureSelected(self, idx):
    if idx < 0: return
    if not hasattr(self, 'atlasModel') or self.atlasModel is None:
      slicer.util.errorDisplay("Atlas model is not in the scene.")
      return
    sid = self.previewTextureCombo.currentText
    png = os.path.join(self.lastBakedTexturesPath or "", sid)
    if not os.path.isfile(png): return
    InterDeCALogic().applyTextureToModel(self.atlasModel, png)

  def onDCApplyButton(self):
    """Executes main DeCA analysis workflow when Run button is clicked.

    Performs the following steps:
    1. Validates input directories and parameters
    2. Creates output directory structure
    3. Runs DeCA alignment algorithm
    4. Processes textures through Blender if available
    5. Generates atlas model and correspondence maps
    6. Updates visualization and reports results
    """
    logic = InterDeCALogic()  # Creates logic instance for analysis

    # Starts progress tracking for user feedback
    self.applyButtonDC.enabled = False  # Disables button during processing
    self.updateProgressDC(0, "Initializing DeCA analysis...")  # Shows initial progress

    # Creates folder structure for organized output
    self.folderNames = self.setUpDeCADir(self.outputDirectoryDC.currentPath, False)  # Sets up directory hierarchy
    if not self.folderNames:
      self.logInfoDC.appendPlainText(f'Output folders could not be created in {self.outputDirectoryDC.currentPath}')  # Reports creation failure
      self.resetProgressDC()
      self.applyButtonDC.enabled = True
      return
    self.folderNames['originalLMs']  = self.landmarkDirectoryDC.currentPath
    self.folderNames['originalModels'] = self.meshDirectoryDC.currentPath
    self.lastDeCAAlignedModelsPath = self.folderNames['resampledModels']  # for Visualize tab

    # ---- 0) Load a representative model for 3D visualization ----
    self.updateProgressDC(5, "Loading representative model for visualization...")
    try:
      # Find a representative model from the model directory
      modelDir = self.meshDirectoryDC.currentPath
      if os.path.exists(modelDir):
        model_files = []
        for f in os.listdir(modelDir):
          if not f.startswith('.') and f.lower().endswith(('.ply', '.stl', '.obj', '.vtk', '.vtp')):
            model_files.append(f)

        if model_files:
          # Use the first available model as representative
          representative_model = model_files[0]
          model_path = os.path.join(modelDir, representative_model)

          # Load the model
          self.representativeModel = slicer.util.loadModel(model_path)
          if self.representativeModel:
            self.representativeModel.SetName("Representative Model (Preview)")
            # Make it semi-transparent and visible
            displayNode = self.representativeModel.GetDisplayNode()
            if displayNode:
              displayNode.SetVisibility(True)
              displayNode.SetOpacity(0.7)  # Semi-transparent
            self.logInfoDC.appendPlainText(f"Loaded representative model: {representative_model}")
          else:
            self.logInfoDC.appendPlainText("Warning: Could not load representative model")
        else:
          self.logInfoDC.appendPlainText("Warning: No valid model files found in directory")
      else:
        self.logInfoDC.appendPlainText("Warning: Model directory not found")
    except Exception as e:
      self.logInfoDC.appendPlainText(f"Warning: Failed to load representative model: {e}")

    # ---- 1) Generate atlas or Load Existing Atlas ----
    atlasOverridePath = self.atlasModelOverride.currentPath
    atlasLandmarksOverridePath = self.atlasLandmarkOverride.currentPath
    
    if atlasOverridePath and atlasLandmarksOverridePath:
      self.updateProgressDC(10, "Loading external atlas...")
      self.logInfoDC.appendPlainText(f"Atlas Override enabled. Loading atlas from: {atlasOverridePath}")
      
      try:
        self.atlasModel = slicer.util.loadModel(atlasOverridePath)
        self.atlasLMs = slicer.util.loadMarkups(atlasLandmarksOverridePath)
        
        if self.atlasModel and self.atlasLMs:
             self.logInfoDC.appendPlainText("Successfully loaded external Atlas Model and Landmarks.")
        else:
             self.logInfoDC.appendPlainText("Error: Failed to load external Atlas Model or Landmarks.")
             self.atlasModel = None
             self.atlasLMs = None
             
      except Exception as e:
        self.logInfoDC.appendPlainText(f"Error loading external atlas: {e}")
        self.atlasModel = None
        self.atlasLMs = None
        
    else:
      # Always generating atlas and using rigid body alignment (no scaling)
      self.updateProgressDC(10, "Generating atlas...")
      self.atlasModel, self.atlasLMs = self.generateNewAtlas(False, self.logInfoDC)

    # Check if atlas generation was successful
    if self.atlasModel is None or self.atlasLMs is None:
      self.logInfoDC.appendPlainText("Failed to generate atlas. Please check the data and try again.")
      self.resetProgressDC()
      self.applyButtonDC.enabled = True
      return

    # Save an intermediate atlas file (RAS) so Blender can read it
    self.updateProgressDC(30, "Preparing atlas for UV mapping...")
    atlas_preuv_obj = os.path.join(self.folderNames['atlasSubDir'], 'atlasModel_preUV.obj')
    logic._save_model_with_cs(self.atlasModel, atlas_preuv_obj, 'RAS')

    # ---- 2) Blender cleanup + Smart UV ----
    self.updateProgressDC(40, "Processing atlas with Blender...")
    blender_exe    = self.blenderExeEdit.currentPath
    merge_dist     = float(self.blMergeDistSpin.value)
    smart_angle    = float(self.blSmartAngleSpin.value)
    island_margin  = float(self.blIslandMarginSpin.value)

    # Auto-detect/install Blender if path is not set or invalid
    if not (blender_exe and os.path.isfile(blender_exe) and os.access(blender_exe, os.X_OK)):
      self.logInfoDC.appendPlainText("Blender path not set or invalid. Attempting automatic detection/installation...")
      blender_exe = logic.getBlenderExecutable(lambda msg: self.logInfoDC.appendPlainText(msg))

      if blender_exe:
        # Update the UI field with the found/installed path
        self.blenderExeEdit.setCurrentPath(blender_exe)
        self.logInfoDC.appendPlainText(f"Using Blender at: {blender_exe}")
      else:
        self.logInfoDC.appendPlainText("Failed to find or install Blender automatically. Please set the path manually.")
      return
    atlas_uv_obj = os.path.join(self.folderNames['colorAnalysisSubDir'], 'atlasModelUV.obj')
    try:
      logic.blender_prepare_atlas(blender_exe, atlas_preuv_obj, atlas_uv_obj,
                                  merge_dist=merge_dist, smart_angle=smart_angle, island_margin=island_margin)
      self.logInfoDC.appendPlainText(f"Atlas cleaned & UV’d in Blender → {atlas_uv_obj}")
    except Exception as e:
      self.logInfoDC.appendPlainText(f"Blender atlas UV step failed: {e}")
      return

    # Reload UV'd atlas back into Slicer (replace old atlas node)
    try:
      slicer.mrmlScene.RemoveNode(self.atlasModel)
    except Exception:
      pass
    self.atlasModel = logic._load_model_with_cs(atlas_uv_obj, 'RAS')

    # Ensure the new atlas model is visible but clean (no texture initially)
    if self.atlasModel:
      self.atlasModel.SetName("ATLAS Model")
      displayNode = self.atlasModel.GetDisplayNode()
      if displayNode:
        displayNode.SetVisibility(True)
        displayNode.SetOpacity(1.0)
        # Clear any texture that might have been applied during UV processing
        try:
          displayNode.SetTextureImageDataConnection(None)
        except AttributeError:
          pass  # No texture to remove or method doesn't exist
        displayNode.SetScalarVisibility(False)

    median_dist = logic._median_landmark_to_surface_dist(self.atlasModel, self.atlasLMs)
    mesh_diagonal = float(self.atlasModel.GetPolyData().GetLength())
    relative_dist = _relative_landmark_surface_distance(median_dist, mesh_diagonal)
    if relative_dist > LANDMARK_SURFACE_WARNING_RELATIVE_THRESHOLD:
      self.logInfoDC.appendPlainText(
        f"WARNING: Landmarks are far from the surface "
        f"({median_dist:.3g}; {relative_dist:.1%} of mesh diagonal)"
      )

    # Save atlas landmarks & a copy of the atlas (PLY) for provenance
    atlasLMPath   = os.path.join(self.folderNames['colorAnalysisSubDir'], 'atlasLM.mrk.json')
    slicer.util.saveNode(self.atlasLMs, atlasLMPath)
    atlasPlyPath  = os.path.join(self.folderNames['colorAnalysisSubDir'], 'atlasModel.ply')
    logic._save_model_with_cs(self.atlasModel, atlasPlyPath, 'RAS')

    # ---- 3) Rigid alignment of subjects to atlas (Slicer) ----
    self.updateProgressDC(60, "Rigid alignment to atlas...")
    try:
      self.logInfoDC.appendPlainText("Rigid alignment to atlas")
      logic.runAlign(self.atlasModel, self.atlasLMs,
                     self.folderNames['originalModels'], self.folderNames['originalLMs'],
                     self.folderNames['alignedModels'], self.folderNames['alignedLMs'],
                     False)  # Always use rigid body alignment (no scaling)
    except ValueError as errorText:
      self.logInfoDC.appendPlainText(str(errorText))
      self.resetProgressDC()
      self.applyButtonDC.enabled = True
      return

    # ---- 4) DeCA resampling (Slicer). Also create OBJ copies that reuse atlas UV (for Blender bake) ----
    self.updateProgressDC(70, "Calculating point correspondences...")
    try:
      self.logInfoDC.appendPlainText("Calculating point correspondences to atlas")
      logic.runDCAlign(
        atlas_uv_obj, atlasLMPath,
        self.folderNames['alignedModels'],
        self.folderNames['alignedLMs'],
        self.folderNames['output'],
        False,  # Never create error checking output
        atlas_uv_template_obj=atlas_uv_obj  # NEW: used to stamp the same UVs onto resampled OBJ copies
      )
    except Exception as e:
      self.logInfoDC.appendPlainText(f"DeCA resampling failed: {e}")
      self.resetProgressDC()
      self.applyButtonDC.enabled = True
      return

    # ---- 5) Blender bake (selection→active) from aligned → resampled(OBJ with atlas UV) ----
    self.updateProgressDC(80, "Setting up texture baking...")
    self.lastBakedTexturesPath = os.path.join(self.folderNames['colorAnalysisSubDir'], "atlasTextures")
    os.makedirs(self.lastBakedTexturesPath, exist_ok=True)

    texturesDir = self.textureDirectoryDC.currentPath
    if os.path.isdir(texturesDir):
      self.updateProgressDC(85, "Baking textures with Blender...")
      try:
        made = logic.blender_bake_all(
          blender_exe=blender_exe,
          alignedDir=self.folderNames['alignedModels'],
          resampledUVDir=os.path.join(self.folderNames['colorAnalysisSubDir'], "resampledOBJ_withUV"),
          texturesDir=texturesDir,
          outDir=self.lastBakedTexturesPath,
          bake_size=int(self.bakeSizeSpin.value),
          bake_extrusion=float(self.bakeExtrusionSpin.value),
          bake_margin_px=int(self.bakeMarginPxSpin.value),
          merge_dist=merge_dist
        )
        self.updateProgressDC(95, "Calculating average texture...")
        logic._calculate_average_texture(self.lastBakedTexturesPath)
        self.logInfoDC.appendPlainText(f"Baked {len(made)} textures to {self.lastBakedTexturesPath}")
      except Exception as e:
        self.logInfoDC.appendPlainText(f"Blender baking failed: {e}")
    else:
      self.logInfoDC.appendPlainText("No textures directory set → skipping bake.")

    # ---- 6) Finalize atlas model display ----
    self.updateProgressDC(95, "Finalizing atlas model display...")

    # Remove the representative model now that we have the final atlas
    if hasattr(self, 'representativeModel') and self.representativeModel:
      slicer.mrmlScene.RemoveNode(self.representativeModel)
      self.representativeModel = None

    # Hide all landmarks for clean visualization
    self._hideAllLandmarks()

    # Hide all other models except the ATLAS Model
    self._hideOtherModels()

    # Ensure the atlas model is prominent and clean (no texture/heatmap)
    if hasattr(self, 'atlasModel') and self.atlasModel:
      # Make sure atlas model has a clear name
      self.atlasModel.SetName("ATLAS Model")

      # Ensure the model node itself is visible first
      self.atlasModel.SetDisplayVisibility(True)
      self.atlasModel.SetHideFromEditors(False)  # Make sure it shows in module lists

      # Get or create display node
      displayNode = self.atlasModel.GetDisplayNode()
      if not displayNode:
        # Create a new display node if one doesn't exist
        self.atlasModel.CreateDefaultDisplayNodes()
        displayNode = self.atlasModel.GetDisplayNode()

      if displayNode:
        # Force visibility in multiple ways
        displayNode.SetVisibility(True)
        displayNode.SetVisibility2D(True)
        displayNode.SetVisibility3D(True)
        displayNode.SetOpacity(1.0)  # Full opacity

        # Make sure it's not clipped or hidden
        displayNode.SetClipping(False)

        # Ensure it's in the scene
        if not slicer.mrmlScene.IsNodePresent(displayNode):
          slicer.mrmlScene.AddNode(displayNode)

        # DISABLE scalar coloring to show clean model without texture/heatmap
        displayNode.SetScalarVisibility(False)

        # Remove any texture that might be applied
        try:
          displayNode.SetTextureImageDataConnection(None)
        except AttributeError:
          # Try alternative method for removing texture
          try:
            displayNode.SetAndObserveTextureImageData(None)
          except AttributeError:
            pass  # No texture to remove or method doesn't exist

        # Set a neutral color for the model
        displayNode.SetColor(0.8, 0.8, 0.8)  # Light gray

        # Force the display node to update
        displayNode.Modified()

      # Force the model node to update and ensure it's in the scene
      if not slicer.mrmlScene.IsNodePresent(self.atlasModel):
        slicer.mrmlScene.AddNode(self.atlasModel)

      self.atlasModel.Modified()

      self.logInfoDC.appendPlainText("ATLAS Model displayed without texture or landmarks")
    else:
      self.logInfoDC.appendPlainText("Warning: Atlas model not found or not properly created")

    # Force the 3D view to center on the models
    slicer.util.resetSliceViews()

    # Make sure the 3D view is active and centered
    layoutManager = slicer.app.layoutManager()
    threeDWidget = layoutManager.threeDWidget(0)
    threeDView = threeDWidget.threeDView()
    threeDView.resetCamera()
    threeDView.resetFocalPoint()

    # Forces a render update to refresh display
    slicer.app.processEvents()  # Processes pending UI events

    # ---- 7) Fill Visualize dropdown ----
    self.updateProgressDC(100, "Finalizing results...")  # Shows completion progress
    if SHOW_VISUALIZE_RESULTS:
      self.updateBakedPreviewList()  # Updates preview list with results

    # ---- 8) Update UI after ATLAS completion ----
    self.updateUIAfterDeCACompletion()  # Refreshes UI elements

    # Reports success and resets UI state
    self.logInfoDC.appendPlainText("ATLAS analysis completed successfully!")  # Shows success message
    self.resetProgressDC()  # Resets progress bar
    self.applyButtonDC.enabled = True  # Re-enables run button

  def maximize3DViewer(self, logWidget=None):
    """
    Maximizes the 3D viewer by setting the layout to 3D-only view.

    This method tries multiple approaches to find and set a 3D-only layout:
    1. First tries common layout constants from vtkMRMLLayoutNode
    2. Falls back to trying common numeric layout IDs
    3. Logs results to the provided log widget or self.logInfoDC

    Args:
      logWidget: Optional QPlainTextEdit widget for logging. If None, uses self.logInfoDC

    Returns:
      bool: True if successfully set 3D layout, False otherwise

    Example usage from other methods:
      # Use with default log widget (self.logInfoDC)
      success = self.maximize3DViewer()

      # Use with specific log widget
      success = self.maximize3DViewer(self.recolorLogInfo)

      # Use from external code (if you have a reference to the widget instance)
      widget = slicer.modules.interdeca.widgetRepresentation().self()
      success = widget.maximize3DViewer()
    """
    if logWidget is None:
      logWidget = getattr(self, 'logInfoDC', None)  # Uses default log widget if none provided

    try:
      layoutManager = slicer.app.layoutManager()
      if layoutManager:
        # Set layout to 3D only view - try different approaches
        try:
          # Method 1: Try common layout constants
          layout_constants_to_try = [
            'SlicerLayoutThreeDOnlyView',
            'SlicerLayoutOneUp3DView',
            'SlicerLayout3DView',
            'SlicerLayoutThreeDView'
          ]

          layout_set = False
          for const_name in layout_constants_to_try:
            try:
              layout_id = getattr(slicer.vtkMRMLLayoutNode, const_name)
              layoutManager.setLayout(layout_id)
              self._logToWidget(logWidget, f"Maximized 3D viewer for optimal visualization ({const_name})")
              layout_set = True
              break
            except AttributeError:
              continue

          if not layout_set:
            # Method 2: Try common numeric layout IDs for 3D-only views
            layout_ids_to_try = [6, 4, 5, 7]  # Common 3D layout IDs
            for layout_id in layout_ids_to_try:
              try:
                layoutManager.setLayout(layout_id)
                self._logToWidget(logWidget, f"Maximized 3D viewer for optimal visualization (layout ID {layout_id})")
                layout_set = True
                break
              except:
                continue

          if not layout_set:
            self._logToWidget(logWidget, "Warning: Could not find 3D-only layout, keeping current layout")
            return False

          return True

        except Exception as inner_e:
          self._logToWidget(logWidget, f"Warning: Error setting 3D layout: {inner_e}")
          return False
      else:
        self._logToWidget(logWidget, "Warning: Could not access layout manager to maximize 3D view")
        return False
    except Exception as e:
      self._logToWidget(logWidget, f"Warning: Error maximizing 3D viewer: {e}")
      return False

  def _logToWidget(self, logWidget, message):
    """
    Helper method to log messages to either QTextEdit or QPlainTextEdit widgets.

    Args:
      logWidget: Either QTextEdit or QPlainTextEdit widget
      message: String message to log
    """
    if logWidget is None:
      return

    try:
      # Try QPlainTextEdit method first
      if hasattr(logWidget, 'appendPlainText'):
        logWidget.appendPlainText(message)
      # Fall back to QTextEdit method
      elif hasattr(logWidget, 'append'):
        logWidget.append(message)
    except Exception as e:
      print(f"Warning: Could not log message to widget: {e}")

  def maximizePlotViewer(self, logWidget=None):
    """
    Maximizes the plot viewer by setting the layout to plot-focused view.

    This method tries multiple approaches to find and set a plot-focused layout:
    1. First tries common plot layout constants from vtkMRMLLayoutNode
    2. Falls back to trying common numeric layout IDs for plot views
    3. Logs results to the provided log widget or self.logInfoDC
    4. Verifies that plot widget is accessible after layout change

    Args:
      logWidget: Optional QTextEdit or QPlainTextEdit widget for logging. If None, uses self.logInfoDC

    Returns:
      bool: True if successfully set plot layout and plot widget is accessible, False otherwise

    Example usage from other methods:
      # Use with default log widget (self.logInfoDC)
      success = self.maximizePlotViewer()

      # Use with specific log widget (supports both QTextEdit and QPlainTextEdit)
      success = self.maximizePlotViewer(self.colorsEDALogInfo)  # QPlainTextEdit
      success = self.maximizePlotViewer(self.populationLogInfo)  # QTextEdit

      # Use from external code (if you have a reference to the widget instance)
      widget = slicer.modules.interdeca.widgetRepresentation().self()
      success = widget.maximizePlotViewer()
    """
    if logWidget is None:
      logWidget = getattr(self, 'logInfoDC', None)

    try:
      layoutManager = slicer.app.layoutManager()
      if layoutManager:
        # Set layout to plot-focused view - try different approaches
        try:
          # Method 1: Try the most reliable plot layout constants
          plot_layout_constants_to_try = [
            'SlicerLayoutOneUpPlotView',      # Single plot view - most reliable
            'SlicerLayoutFourUpPlotView',     # Four-up plot view
            'SlicerLayoutPlotView',           # Generic plot view
            'SlicerLayoutTabbedSliceView'     # Tabbed view that supports plots
          ]

          layout_set = False
          for const_name in plot_layout_constants_to_try:
            try:
              layout_id = getattr(slicer.vtkMRMLLayoutNode, const_name)
              layoutManager.setLayout(layout_id)

              # Give Slicer time to update the layout
              slicer.app.processEvents()

              # Verify that plot widget is accessible
              plotWidget = layoutManager.plotWidget(0)
              if plotWidget is not None:
                self._logToWidget(logWidget, f"Maximized plot viewer for optimal visualization ({const_name})")
                layout_set = True
                break
              else:
                # Layout was set but plot widget is not accessible, try next option
                continue
            except AttributeError:
              continue

          if not layout_set:
            # Method 2: Try well-known numeric layout IDs for plot views
            # These are based on Slicer's standard layout definitions
            plot_layout_ids_to_try = [
              24, 25, 26, 27, 28, 29  # Extended range of plot layout IDs
            ]
            for layout_id in plot_layout_ids_to_try:
              try:
                layoutManager.setLayout(layout_id)

                # Give Slicer time to update the layout
                slicer.app.processEvents()

                # Verify that plot widget is accessible
                plotWidget = layoutManager.plotWidget(0)
                if plotWidget is not None:
                  self._logToWidget(logWidget, f"Maximized plot viewer for optimal visualization (layout ID {layout_id})")
                  layout_set = True
                  break
              except:
                continue

          if not layout_set:
            self._logToWidget(logWidget, "Warning: Could not find plot-compatible layout, keeping current layout")
            return False

          return True

        except Exception as inner_e:
          self._logToWidget(logWidget, f"Warning: Error setting plot layout: {inner_e}")
          return False
      else:
        self._logToWidget(logWidget, "Warning: Could not access layout manager to maximize plot view")
        return False
    except Exception as e:
      self._logToWidget(logWidget, f"Warning: Error maximizing plot viewer: {e}")
      return False

  def switchToOptimalViewLayout(self, viewType="3D", logWidget=None):
    """
    Switches to the optimal layout for the specified view type.

    This is a convenience method that calls the appropriate maximization method
    based on the requested view type.

    Args:
      viewType: String indicating the desired view type. Options: "3D", "plot"
      logWidget: Optional QPlainTextEdit widget for logging. If None, uses self.logInfoDC

    Returns:
      bool: True if successfully switched to the requested layout, False otherwise

    Example usage:
      # Switch to 3D view
      success = self.switchToOptimalViewLayout("3D")

      # Switch to plot view with custom logging
      success = self.switchToOptimalViewLayout("plot", self.colorsEDALogInfo)
    """
    if viewType.lower() == "3d":
      return self.maximize3DViewer(logWidget)
    elif viewType.lower() == "plot":
      return self.maximizePlotViewer(logWidget)
    else:
      if logWidget is None:
        logWidget = getattr(self, 'logInfoDC', None)
      self._logToWidget(logWidget, f"Warning: Unknown view type '{viewType}'. Supported types: '3D', 'plot'")
      return False

  def createPopulationPlot(self, reducedData, textureNames, method, x_axis_idx=0, y_axis_idx=1, variance_explained=None, displayInLayout=True):
    """
    Create a population analysis plot using Slicer's plotting functionality
    with equal X/Y numeric ranges.

    This method handles the UI aspects of plotting and should be called from
    the Widget class after the Logic class has prepared the data.

    Args:
      reducedData: numpy array of coordinates for each texture (n_textures x n_components)
      textureNames: list of texture names corresponding to the data points
      method: string indicating the dimensionality reduction method ("PCA", "ICA", or "UMAP")
      x_axis_idx: index of component to plot on X-axis (default: 0 for PC1/IC1)
      y_axis_idx: index of component to plot on Y-axis (default: 1 for PC2/IC2)
      variance_explained: array of variance explained ratios (for PCA only)
      displayInLayout: if True, switch to plot layout and display the chart (default: True)
                       Set to False when creating chart for use in a custom layout

    Returns:
      dict with success status and plot node information
    """
    try:
      # Validate axis indices
      max_idx = reducedData.shape[1] - 1
      x_axis_idx = min(x_axis_idx, max_idx)
      y_axis_idx = min(y_axis_idx, max_idx)

      # --- series ---
      plotSeriesNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLPlotSeriesNode")
      plotSeriesNode.SetName(f"{method}")

      # Create axis labels based on method
      if method == "PCA":
        x_label = f"PC{x_axis_idx + 1}"
        y_label = f"PC{y_axis_idx + 1}"
      elif method == "ICA":
        x_label = f"IC{x_axis_idx + 1}"
        y_label = f"IC{y_axis_idx + 1}"
      else:  # UMAP or other
        x_label = f"{method} Component {x_axis_idx + 1}"
        y_label = f"{method} Component {y_axis_idx + 1}"

      xArray = vtk.vtkFloatArray(); xArray.SetName(x_label)
      yArray = vtk.vtkFloatArray(); yArray.SetName(y_label)
      xArray.SetNumberOfTuples(len(reducedData))
      yArray.SetNumberOfTuples(len(reducedData))
      labelsArray = vtk.vtkStringArray(); labelsArray.SetName("Texture Names")
      labelsArray.SetNumberOfTuples(len(reducedData))

      for i, (point, name) in enumerate(zip(reducedData, textureNames)):
        xArray.SetValue(i, float(point[x_axis_idx]))
        yArray.SetValue(i, float(point[y_axis_idx]))
        labelsArray.SetValue(i, name)

      tableNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTableNode")
      tableNode.SetName(f"MultiRecolor_Population_Data_{method}")
      tableNode.AddColumn(xArray); tableNode.AddColumn(yArray); tableNode.AddColumn(labelsArray)

      plotSeriesNode.SetAndObserveTableNodeID(tableNode.GetID())
      plotSeriesNode.SetXColumnName(xArray.GetName())
      plotSeriesNode.SetYColumnName(yArray.GetName())
      plotSeriesNode.SetLabelColumnName(labelsArray.GetName())
      plotSeriesNode.SetPlotType(slicer.vtkMRMLPlotSeriesNode.PlotTypeScatter)
      plotSeriesNode.SetMarkerStyle(slicer.vtkMRMLPlotSeriesNode.MarkerStyleCircle)
      plotSeriesNode.SetMarkerSize(8)
      plotSeriesNode.SetLineStyle(slicer.vtkMRMLPlotSeriesNode.LineStyleNone)
      plotSeriesNode.SetColor(0.2, 0.6, 0.8)

      # --- chart ---
      plotChartNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLPlotChartNode")
      plotChartNode.SetName(f"MultiRecolor_Population_Chart_{method}")
      plotChartNode.AddAndObservePlotSeriesNodeID(plotSeriesNode.GetID())
      plotChartNode.SetTitle(f"Multi-Texture Population Analysis ({method})")

      # Set axis titles with variance explained if available
      if method == "PCA" and variance_explained is not None and len(variance_explained) > max(x_axis_idx, y_axis_idx):
        x_var = variance_explained[x_axis_idx] * 100
        y_var = variance_explained[y_axis_idx] * 100
        plotChartNode.SetXAxisTitle(f"PC{x_axis_idx + 1} ({x_var:.1f}%)")
        plotChartNode.SetYAxisTitle(f"PC{y_axis_idx + 1} ({y_var:.1f}%)")
      else:
        plotChartNode.SetXAxisTitle(x_label)
        plotChartNode.SetYAxisTitle(y_label)

      # Calculate axis ranges with proper validation
      if len(reducedData) > 0 and reducedData.shape[1] >= 2:
        x_min, x_max = float(np.min(reducedData[:,x_axis_idx])), float(np.max(reducedData[:,x_axis_idx]))
        y_min, y_max = float(np.min(reducedData[:,y_axis_idx])), float(np.max(reducedData[:,y_axis_idx]))

        # Ensure valid ranges (avoid NaN, inf, or identical min/max)
        if np.isfinite(x_min) and np.isfinite(x_max) and np.isfinite(y_min) and np.isfinite(y_max):
          x_center = (x_min + x_max) / 2.0
          y_center = (y_min + y_max) / 2.0
          x_span = x_max - x_min
          y_span = y_max - y_min

          # Use the larger span for both axes to create equal scaling
          span = max(x_span, y_span)
          span = max(span, 1e-6)  # Avoid zero span
          span *= 1.2  # Add 20% padding

          # Set equal ranges centered on the data
          x_range_min = x_center - span/2.0
          x_range_max = x_center + span/2.0
          y_range_min = y_center - span/2.0
          y_range_max = y_center + span/2.0

          # Disable auto-range BEFORE setting manual ranges
          if hasattr(plotChartNode, "SetXAxisRangeAuto"):
            plotChartNode.SetXAxisRangeAuto(False)
          if hasattr(plotChartNode, "SetYAxisRangeAuto"):
            plotChartNode.SetYAxisRangeAuto(False)

          # Set the ranges
          plotChartNode.SetXAxisRange(x_range_min, x_range_max)
          plotChartNode.SetYAxisRange(y_range_min, y_range_max)
          plotChartNode.Modified()
        else:
          # Data contains invalid values, use auto-range
          self._logToWidget(self.populationLogInfo, "Warning: Invalid data ranges detected, using auto-range")
      else:
        # No data or insufficient dimensions, use auto-range
        self._logToWidget(self.populationLogInfo, "Warning: Insufficient data for manual range setting, using auto-range")

      # Only switch layout and display if displayInLayout is True
      # When called from morphospace, we skip this to avoid layout conflicts
      if displayInLayout:
        # Maximize plot viewer and show the plot
        self._logToWidget(self.populationLogInfo, "Attempting to maximize plot viewer...")
        plotLayoutSuccess = self.maximizePlotViewer(self.populationLogInfo)
        if not plotLayoutSuccess:
          self._logToWidget(self.populationLogInfo, "Warning: Could not set optimal plot layout, using current layout")

        # Show the plot in the plot view
        layoutManager = slicer.app.layoutManager()
        if layoutManager is not None:
          # Try to get plot widget and display the chart
          plotWidget = layoutManager.plotWidget(0)
          if plotWidget is not None:
            plotViewNode = plotWidget.mrmlPlotViewNode()
            if plotViewNode is not None:
              plotViewNode.SetPlotChartNodeID(plotChartNode.GetID())
              self._logToWidget(self.populationLogInfo, "Plot successfully displayed in plot viewer")

              # Force the plot widget to fit the view
              try:
                plotWidget.fitToContent()
              except:
                pass  # fitToContent might not be available in all Slicer versions
            else:
              self._logToWidget(self.populationLogInfo, "Warning: Plot view node not available")
          else:
            self._logToWidget(self.populationLogInfo, "Warning: Could not access plot widget - plot will be available in Data module")

            # Fallback: Try to switch to a different layout that might work better
            try:
              # Try the tabbed slice view which often has plot capabilities
              layoutManager.setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutTabbedSliceView)
              slicer.app.processEvents()
              plotWidget = layoutManager.plotWidget(0)
              if plotWidget is not None:
                plotViewNode = plotWidget.mrmlPlotViewNode()
                if plotViewNode is not None:
                  plotViewNode.SetPlotChartNodeID(plotChartNode.GetID())
                  self._logToWidget(self.populationLogInfo, "Plot displayed using fallback layout")
            except:
              pass
        else:
          self._logToWidget(self.populationLogInfo, "Warning: Layout manager not available")

      return {"success": True, "chart_node": plotChartNode, "series_node": plotSeriesNode, "table_node": tableNode}

    except Exception as e:
      self._logToWidget(self.populationLogInfo, f"Error creating population plot: {e}")
      import traceback; traceback.print_exc()
      return {"success": False}

  def updateUIAfterDeCACompletion(self):
    """
    Update UI components after DeCA completes successfully.

    This function:
    1. Preselects the "ATLAS Model" in Colors EDA, Recolor, and MultiRecolor tabs
    2. Auto-populates texture directories with the baked textures path
    3. Switches to the Recolor tab automatically
    4. Auto-selects average_texture.png
    5. Auto-applies the texture
    6. Maximizes the 3D viewer
    """
    self.logInfoDC.appendPlainText("Starting UI automation after ATLAS completion...")

    try:
      # Find the "ATLAS Model" in the scene
      atlasModel = None
      try:
        for model in slicer.util.getNodesByClass('vtkMRMLModelNode'):
          if model.GetName() == "ATLAS Model":
            atlasModel = model
            break
      except Exception as e:
        self.logInfoDC.appendPlainText(f"Warning: Error searching for ATLAS Model: {e}")

      if atlasModel:
        try:
          # Preselect the ATLAS Model in all relevant tabs
          self.multiRecolorAtlasModelSelect.setCurrentNode(atlasModel)
          self.regionMeshSelector.setCurrentNode(atlasModel)

          self.logInfoDC.appendPlainText("Auto-selected 'ATLAS Model' in MultiRecolor and Mesh Selection tabs")
        except Exception as e:
          self.logInfoDC.appendPlainText(f"Warning: Error setting atlas model selection: {e}")
      else:
        self.logInfoDC.appendPlainText("Warning: Could not find 'ATLAS Model' for auto-selection")

      # Auto-populate texture directories with baked textures path
      try:
        if hasattr(self, 'lastBakedTexturesPath') and self.lastBakedTexturesPath and os.path.isdir(self.lastBakedTexturesPath):
          # Set texture directory in Colors EDA tab
          self.bakedTexturesDirectorySelector.setCurrentPath(self.lastBakedTexturesPath)

          # Set texture directory in Recolor tab
          self.recolorTexturesDirectorySelector.setCurrentPath(self.lastBakedTexturesPath)

          # Set texture directory in MultiRecolor tab
          self.multiRecolorTextureDirectorySelector.setCurrentPath(self.lastBakedTexturesPath)

          self.logInfoDC.appendPlainText(f"Auto-populated texture directories with: {self.lastBakedTexturesPath}")
        else:
          self.logInfoDC.appendPlainText("Warning: No baked textures directory available for auto-population")
      except Exception as e:
        self.logInfoDC.appendPlainText(f"Warning: Error setting texture directories: {e}")

      # Switch to the Recolor tab automatically
      try:
        # Find the index of the Recolor tab using a more robust approach
        recolorTabIndex = -1

        # Try different ways to get tab count and text to handle Qt API variations
        try:
          # Method 1: Try as method call
          tabCount = self.tabsWidget.count()
        except TypeError:
          try:
            # Method 2: Try as property
            tabCount = self.tabsWidget.count
          except:
            # Method 3: Fallback - manually check known tabs
            tabCount = 5  # We know there are typically 5 tabs: DeCA, Colors EDA, Recolor, MultiRecolor (+ optionally Visualize)

        for i in range(tabCount):
          try:
            # Try to get tab text
            try:
              tabText = self.tabsWidget.tabText(i)
            except TypeError:
              # If tabText is callable, call it
              tabText = self.tabsWidget.tabText(i)()

            if tabText == "Recolor":
              recolorTabIndex = i
              break
          except Exception:
            # Skip this tab if we can't get its text
            continue

        if recolorTabIndex >= 0:
          self.tabsWidget.setCurrentIndex(recolorTabIndex)
          self.logInfoDC.appendPlainText("Automatically switched to Recolor tab")
        else:
          # Fallback: try to switch to tab index 2 (which should be Recolor based on the setup)
          # Tab order: 0=DeCA, 1=Colors EDA (or Visualize if enabled), 2=Recolor
          fallbackIndex = 2 if not SHOW_VISUALIZE_RESULTS else 3
          try:
            self.tabsWidget.setCurrentIndex(fallbackIndex)
            self.logInfoDC.appendPlainText(f"Switched to tab index {fallbackIndex} (likely Recolor tab)")
          except:
            self.logInfoDC.appendPlainText("Warning: Could not find or switch to Recolor tab")
      except Exception as e:
        self.logInfoDC.appendPlainText(f"Warning: Error switching to Recolor tab: {e}")

      # ---- Additional Recolor Tab Automation ----
      try:
        # Wait a moment for the tab switch and UI updates to complete
        slicer.app.processEvents()

        # Select "average_texture.png" in the texture selector
        try:
          # Find "average_texture.png" in the texture selector
          averageTextureIndex = -1

          # Handle Qt API variations for count
          try:
            itemCount = self.recolorTextureSelector.count()
          except TypeError:
            try:
              itemCount = self.recolorTextureSelector.count
            except:
              itemCount = 0  # Fallback

          for i in range(itemCount):
            try:
              itemText = self.recolorTextureSelector.itemText(i)
              if itemText == "average_texture.png":
                averageTextureIndex = i
                break
            except:
              continue  # Skip this item if we can't get its text

          if averageTextureIndex >= 0:
            self.recolorTextureSelector.setCurrentIndex(averageTextureIndex)
            self.logInfoDC.appendPlainText("Auto-selected 'average_texture.png' in Recolor tab")
          else:
            self.logInfoDC.appendPlainText("Warning: 'average_texture.png' not found in texture selector")
        except Exception as e:
          self.logInfoDC.appendPlainText(f"Warning: Error selecting average_texture.png: {e}")

        # Automatically run "Apply Texture"
        try:
          # Wait a moment for UI updates to propagate
          slicer.app.processEvents()

          # Check if the apply button is enabled (should be after selecting texture and atlas)
          if hasattr(self, 'applyRecolorButton') and self.applyRecolorButton.enabled:
            self.logInfoDC.appendPlainText("Auto-applying average texture to atlas model...")
            # Trigger the apply recolor button
            self.onApplyRecolorButton()
          else:
            # Try to enable the button by triggering parameter validation
            try:
              self.onRecolorParameterChanged()
              slicer.app.processEvents()
              if self.applyRecolorButton.enabled:
                self.logInfoDC.appendPlainText("Auto-applying average texture to atlas model (after enabling button)...")
                self.onApplyRecolorButton()
              else:
                self.logInfoDC.appendPlainText("Warning: Apply Recolor button not enabled - skipping auto-apply")
            except Exception as e2:
              self.logInfoDC.appendPlainText(f"Warning: Could not enable Apply Recolor button: {e2}")
        except Exception as e:
          self.logInfoDC.appendPlainText(f"Warning: Error auto-applying texture: {e}")

        # Maximize the 3D viewer using the refactored method
        self.maximize3DViewer()

      except Exception as e:
        self.logInfoDC.appendPlainText(f"Warning: Error during additional Recolor tab automation: {e}")

    except Exception as e:
      self.logInfoDC.appendPlainText(f"Error during UI update after DeCA completion: {e}")
      import traceback
      self.logInfoDC.appendPlainText(f"Traceback: {traceback.format_exc()}")

    # Final completion message
    self.logInfoDC.appendPlainText("UI automation after ATLAS completion finished.")


  def onDCLApplyButton(self):
    logic = InterDeCALogic()
    # rigidly align to template
    self.logInfoDCL.appendPlainText(f"Rigid alignment to the atlas")
    removeScale = True
    try:
      logic.runAlign(self.atlasModel, self.atlasLMs, self.folderNames['originalModels'], self.folderNames['originalLMs'],self.folderNames['alignedModels'], self.folderNames['alignedLMs'], removeScale)
    except ValueError as errorText:
      self.logInfoDCL.appendPlainText(str(errorText))
      return
    # generate point correspondences
    self.logInfoDCL.appendPlainText(f"Calculating point correspondences")
    atlasDenseLandmarks = logic.runDeCAL(self.atlasModel, self.atlasLMs, self.folderNames['alignedModels'],
    self.folderNames['alignedLMs'], self.folderNames['DeCALOutput'], self.spacingTolerance.value)
    # setup for optional subsetting
    self.pointSelection.setCurrentNode(atlasDenseLandmarks)
    self.DCLLandmarkDirectory.setCurrentPath(self.folderNames['DeCALOutput'])

  def onSubsetApplyButton(self):
    logic = InterDeCALogic()
    topDir = os.path.dirname(self.DCLLandmarkDirectory.currentPath)
    lmDirectorySubset = os.path.join(topDir, "DeCALSubset")
    os.makedirs(lmDirectorySubset)
    atlasNode = self.pointSelection.currentNode()
    lmDirectorySubset = logic.runSubsetLandmarks(atlasNode, self.DCLLandmarkDirectory.currentPath, lmDirectorySubset)

  def _hideMarkupsForVisualization(self, remove=False):
    """
    Hide (or optionally delete) all markups so the 3D view is clean in Visualize.
    Works for fiducials, curves, lines, etc.  Non-destructive by default.
    """
    try:
      markups = list(slicer.util.getNodesByClass('vtkMRMLMarkupsNode'))
      if not markups:  # fallback for older Slicer builds
        markups = list(slicer.util.getNodesByClass('vtkMRMLMarkupsFiducialNode'))
    except Exception:
      markups = []

    for n in markups:
      try:
        dn = n.GetDisplayNode()
        if dn:
          dn.SetVisibility(False)
        if remove:
          slicer.mrmlScene.RemoveNode(n)
      except Exception:
        pass

  def _hideUnwantedModels(self):
    """Hide models that are likely not DeCA-related (planes, reference objects, etc.)"""
    for m in slicer.util.getNodesByClass('vtkMRMLModelNode'):
      try:
        node_name = m.GetName().lower()
        # Hide models that are likely reference objects, planes, or debugging aids
        unwanted_keywords = ['plane', 'axis', 'reference', 'coordinate', 'grid', 'debug', 'temp', 'tmp']
        if any(keyword in node_name for keyword in unwanted_keywords):
          dn = m.GetDisplayNode()
          if dn: 
            dn.SetVisibility(False)
      except Exception:
        pass

  def _ensureModelsAreVisible(self):
    for m in slicer.util.getNodesByClass('vtkMRMLModelNode'):
      try:
        dn = m.GetDisplayNode()
        if dn: dn.SetVisibility(True)
      except Exception:
        pass

  def _hideAllLandmarks(self):
    """Hide all landmark/markup nodes for clean visualization"""
    try:
      # Get all markup nodes (landmarks, fiducials, etc.)
      markups = list(slicer.util.getNodesByClass('vtkMRMLMarkupsNode'))
      if not markups:  # fallback for older Slicer builds
        markups = list(slicer.util.getNodesByClass('vtkMRMLMarkupsFiducialNode'))
    except Exception:
      markups = []

    for markup in markups:
      try:
        displayNode = markup.GetDisplayNode()
        if displayNode:
          displayNode.SetVisibility(False)
      except Exception:
        pass

  def _hideOtherModels(self):
    """Remove all models except the ATLAS Model to prevent clutter"""
    models_to_remove = []
    for model in slicer.util.getNodesByClass('vtkMRMLModelNode'):
      try:
        # Only keep the ATLAS Model
        if model.GetName() != "ATLAS Model":
          models_to_remove.append(model)
      except Exception:
        pass

    # Remove the unwanted models
    for model in models_to_remove:
      try:
        slicer.mrmlScene.RemoveNode(model)
      except Exception:
        pass

  ################################### MultiRecolor Event Handlers ###################################

  def onMultiRecolorParameterChanged(self):
    """Enable/disable buttons based on MultiRecolor parameter selection"""
    atlasSelected = bool(self.multiRecolorAtlasModelSelect.currentNode())
    textureDirectorySelected = bool(self.multiRecolorTextureDirectorySelector.currentPath and
                                   os.path.isdir(self.multiRecolorTextureDirectorySelector.currentPath))

    # Enable cluster button if atlas and texture directory are selected
    self.clusterButton.enabled = atlasSelected and textureDirectorySelected
    
    # Save the texture directory for persistence
    if self.multiRecolorTextureDirectorySelector.currentPath:
      self.saveTextureDirectory("multiRecolorTextureDirectory", self.multiRecolorTextureDirectorySelector.currentPath)

    # Update texture file list when directory changes
    if textureDirectorySelected:
      self.updateMultiRecolorTextureList()

  def onMultiRecolorDisableSubsamplingChanged(self, checked):
    """Enable/disable subsampling spinbox based on checkbox"""
    self.multiRecolorNumSubsampledFacesSpin.setEnabled(not checked)

  def onMultiRecolorModeChanged(self):
    """Handle mode change between clustering and subsample-only"""
    isClusteringMode = self.multiRecolorClusteringRadio.isChecked()

    # Enable/disable clustering-specific controls based on mode
    self.multiRecolorInitialClustersSpin.setEnabled(isClusteringMode)
    self.multiRecolorConsolidatedClustersSpin.setEnabled(isClusteringMode)
    self.multiRecolorNormalizeLuminosityCheckbox.setEnabled(isClusteringMode)

  def onMultiRecolorClusterCountChanged(self):
    """Validate that consolidated clusters <= initial clusters"""
    initialClusters = self.multiRecolorInitialClustersSpin.value
    consolidatedClusters = self.multiRecolorConsolidatedClustersSpin.value

    # If consolidated > initial, adjust consolidated to match initial
    if consolidatedClusters > initialClusters:
      self.multiRecolorConsolidatedClustersSpin.setValue(initialClusters)

  def onMultiRecolorNumPCsChanged(self):
    """Update PC axis selectors when number of PCs changes"""
    numPCs = self.multiRecolorNumPCsSpin.value

    # Store current selections
    currentXAxis = self.multiRecolorXAxisCombo.currentIndex
    currentYAxis = self.multiRecolorYAxisCombo.currentIndex

    # Clear and repopulate axis selectors
    self.multiRecolorXAxisCombo.clear()
    self.multiRecolorYAxisCombo.clear()

    for i in range(numPCs):
      pc_label = f"PC{i+1}"
      self.multiRecolorXAxisCombo.addItem(pc_label)
      self.multiRecolorYAxisCombo.addItem(pc_label)

    # Restore selections if valid, otherwise default to PC1 and PC2
    if currentXAxis >= 0 and currentXAxis < numPCs:
      self.multiRecolorXAxisCombo.setCurrentIndex(currentXAxis)
    else:
      self.multiRecolorXAxisCombo.setCurrentIndex(0)  # PC1

    if currentYAxis >= 0 and currentYAxis < numPCs:
      self.multiRecolorYAxisCombo.setCurrentIndex(currentYAxis)
    else:
      self.multiRecolorYAxisCombo.setCurrentIndex(min(1, numPCs - 1))  # PC2 if available

  def onMultiRecolorAxisChanged(self):
    """Update plot when axis selection changes"""
    # Only update if we have a stored result from a previous analysis
    if not hasattr(self, 'multiRecolorPopulationResult') or self.multiRecolorPopulationResult is None:
      return

    result = self.multiRecolorPopulationResult

    # Only update for PCA, ICA, and UMAP (all support variable axes)
    if result.get("method") not in ["PCA", "ICA", "UMAP"]:
      return

    # Get selected axes
    x_axis_idx = self.multiRecolorXAxisCombo.currentIndex
    y_axis_idx = self.multiRecolorYAxisCombo.currentIndex

    # Get variance explained if available (only for PCA)
    variance_explained = None
    method = result.get("method", "PCA")
    if method == "PCA" and "pca_model" in result:
      variance_explained = result["pca_model"].explained_variance_ratio_

    # Update the plot with new axes
    comp_label = "PC" if method == "PCA" else "IC" if method == "ICA" else "Component"
    self.populationLogInfo.append(f"Updating plot to show {comp_label}{x_axis_idx+1} vs {comp_label}{y_axis_idx+1}...")
    plotResult = self.createPopulationPlot(
      result["reduced_data"],
      result["texture_names"],
      result["method"],
      x_axis_idx=x_axis_idx,
      y_axis_idx=y_axis_idx,
      variance_explained=variance_explained
    )

    if plotResult.get("success", False):
      self.populationLogInfo.append(f"Plot updated successfully")
    else:
      self.populationLogInfo.append("Plot update failed")

  def updateMultiRecolorTextureList(self):
    """Update the list of texture files for MultiRecolor"""
    textureDir = self.multiRecolorTextureDirectorySelector.currentPath
    if not textureDir or not os.path.exists(textureDir):
      self.multiRecolorTextureFiles = []
      return

    # Find all image files in the directory
    imageExtensions = ['.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp']
    textureFiles = []

    for filename in os.listdir(textureDir):
      if any(filename.lower().endswith(ext) for ext in imageExtensions):
        textureFiles.append(filename)

    self.multiRecolorTextureFiles = sorted(textureFiles)
    self.clusteringLogInfo.append(f"Found {len(self.multiRecolorTextureFiles)} texture files")

    # Update individual texture selector
    self.individualTextureSelector.clear()
    self.individualTextureSelector.addItems(self.multiRecolorTextureFiles)

  def onClusterButton(self):
    """Handle Step 1: Multi-texture clustering or subsample-only mode"""
    try:
      qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)
      self.clusteringProgressBar.setVisible(True)
      self.clusteringProgressBar.setValue(0)
      self.clusteringLogInfo.clear()

      atlasModel = self.multiRecolorAtlasModelSelect.currentNode()
      textureDir = self.multiRecolorTextureDirectorySelector.currentPath
      numSubsampledFaces = 0 if self.multiRecolorDisableSubsamplingCheckbox.isChecked() else self.multiRecolorNumSubsampledFacesSpin.value
      useNeighborAverage = self.multiRecolorNeighborAverageCheckbox.isChecked()

      if not atlasModel:
        self.clusteringLogInfo.append("Error: No atlas model selected")
        return

      if not textureDir or not os.path.exists(textureDir):
        self.clusteringLogInfo.append("Error: Invalid texture directory")
        return

      # Exclude average_texture from clustering input
      clusteringTextureFiles = [f for f in self.multiRecolorTextureFiles if not f.lower().startswith('average_texture')]
      excluded = len(self.multiRecolorTextureFiles) - len(clusteringTextureFiles)
      if excluded > 0:
        self.clusteringLogInfo.append(f"Found {len(self.multiRecolorTextureFiles)} texture files, {len(clusteringTextureFiles)} used for clustering (average_texture omitted)")

      if not clusteringTextureFiles:
        self.clusteringLogInfo.append("Error: No texture files found")
        return

      # Check which mode is selected
      isClusteringMode = self.multiRecolorClusteringRadio.isChecked()

      if isClusteringMode:
        # Full clustering mode
        self._performClusteringMode(atlasModel, textureDir, numSubsampledFaces, useNeighborAverage, clusteringTextureFiles)
      else:
        # Subsample-only mode
        self._performSubsampleOnlyMode(atlasModel, textureDir, numSubsampledFaces, useNeighborAverage, clusteringTextureFiles)

      self.clusteringProgressBar.setVisible(False)
      qt.QApplication.restoreOverrideCursor()

    except Exception as e:
      self.clusteringProgressBar.setVisible(False)
      qt.QApplication.restoreOverrideCursor()
      self.clusteringLogInfo.append(f"Error: {str(e)}")
      slicer.util.errorDisplay(f"Multi-texture clustering failed: {str(e)}")
      import traceback
      traceback.print_exc()

  def _performClusteringMode(self, atlasModel, textureDir, numSubsampledFaces, useNeighborAverage, textureFiles=None):
    """Perform full clustering mode with color quantization"""
    if textureFiles is None:
      textureFiles = self.multiRecolorTextureFiles
    initialClusters = self.multiRecolorInitialClustersSpin.value
    consolidatedClusters = self.multiRecolorConsolidatedClustersSpin.value
    normalizeLuminosity = self.multiRecolorNormalizeLuminosityCheckbox.isChecked()

    self.clusteringLogInfo.append(f"Starting per-texture clustering pipeline...")
    self.clusteringLogInfo.append(f"Initial clusters: {initialClusters}, Consolidated: {consolidatedClusters}")
    self.clusteringLogInfo.append(f"Subsampled faces: {numSubsampledFaces}")
    self.clusteringLogInfo.append(f"Luminosity normalization: {'enabled' if normalizeLuminosity else 'disabled'}")
    self.clusteringLogInfo.append(f"Neighbor average: {'enabled' if useNeighborAverage else 'disabled'}")
    self.clusteringLogInfo.append(f"Processing {len(textureFiles)} textures...")

    logic = InterDeCALogic()

    # Get cached face areas
    cachedFaceAreas = self.getCachedFaceAreas(atlasModel)
    if cachedFaceAreas is None:
      self.clusteringLogInfo.append("Error: Failed to get face areas")
      return

    # Run the multi-texture clustering pipeline
    result = logic.performMultiTextureClustering(
      atlasModel, textureDir, textureFiles,
      initialClusters, consolidatedClusters,
      numSubsampledFaces=numSubsampledFaces,
      normalizeLuminosity=normalizeLuminosity,
      useNeighborAverage=useNeighborAverage,
      faceAreas=cachedFaceAreas,
      progressCallback=self.updateClusteringProgress,
      logCallback=self.logClusteringMessage
    )

    if result.get("success", False):
      self.multiRecolorClusteringPipeline = result["pipeline"]
      self.multiRecolorFaceAreas = cachedFaceAreas  # Use cached areas

      # For backward compatibility, store shared palette as cluster centers
      if self.multiRecolorClusteringPipeline.sharedPalette is not None:
        self.multiRecolorClusterCenters = logic.lab_to_rgb(
          self.multiRecolorClusteringPipeline.sharedPalette
        ).astype(np.uint8)
        self.clusteringLogInfo.append(f"Computed shared palette from {len(self.multiRecolorClusteringPipeline.textureFiles)} textures")
      elif self.multiRecolorClusteringPipeline.referenceCentroids is not None:
        # Fallback to reference centroids if shared palette not available
        self.multiRecolorClusterCenters = logic.lab_to_rgb(
          self.multiRecolorClusteringPipeline.referenceCentroids
        ).astype(np.uint8)
        self.clusteringLogInfo.append("Warning: Using reference centroids (shared palette not computed)")
      else:
        self.multiRecolorClusterCenters = None

      self.clusteringLogInfo.append("Multi-texture clustering pipeline completed successfully!")
      self.clusteringLogInfo.append(f"Processed {len(self.multiRecolorClusteringPipeline.textureFiles)} textures")
      self.clusteringLogInfo.append(f"Created {consolidatedClusters} consolidated clusters (from {initialClusters} initial clusters)")

      # Enable Step 2 controls
      self.individualTextureSelector.setEnabled(True)
      self.onIndividualTextureChanged()  # Update button states

      # Enable Step 3 controls
      self.compareTexturesButton.enabled = True

    else:
      self.clusteringLogInfo.append("Multi-texture clustering failed - check log for details")

  def _performSubsampleOnlyMode(self, atlasModel, textureDir, numSubsampledFaces, useNeighborAverage, textureFiles=None):
    """Perform subsample-only mode without clustering"""
    if textureFiles is None:
      textureFiles = self.multiRecolorTextureFiles
    self.clusteringLogInfo.append(f"Starting subsample and average mode...")
    self.clusteringLogInfo.append(f"Subsampled faces: {numSubsampledFaces}")
    self.clusteringLogInfo.append(f"Neighbor average: {'enabled' if useNeighborAverage else 'disabled'}")
    self.clusteringLogInfo.append(f"Processing {len(textureFiles)} textures...")

    logic = InterDeCALogic()

    # Get cached face areas
    cachedFaceAreas = self.getCachedFaceAreas(atlasModel)
    if cachedFaceAreas is None:
      self.clusteringLogInfo.append("Error: Failed to get face areas")
      return

    # Run subsample-only pipeline
    result = logic.performSubsampleOnly(
      atlasModel, textureFiles,
      numSubsampledFaces=numSubsampledFaces,
      useNeighborAverage=useNeighborAverage,
      faceAreas=cachedFaceAreas,
      progressCallback=self.updateClusteringProgress,
      logCallback=self.logClusteringMessage
    )

    if result.get("success", False):
      self.multiRecolorClusteringPipeline = result["pipeline"]
      self.multiRecolorFaceAreas = cachedFaceAreas
      # In subsample-only mode, we don't have cluster centers
      self.multiRecolorClusterCenters = None

      self.clusteringLogInfo.append("Subsample and average completed successfully!")
      self.clusteringLogInfo.append(f"Processed {len(self.multiRecolorClusteringPipeline.textureFiles)} textures")

      # Enable Step 2 controls
      self.individualTextureSelector.setEnabled(True)
      self.onIndividualTextureChanged()  # Update button states

      # Enable Step 3 controls
      self.compareTexturesButton.enabled = True

    else:
      self.clusteringLogInfo.append("Subsample and average failed - check log for details")

  def onIndividualTextureChanged(self):
    """Handle texture selection change in Step 2"""
    textureSelected = bool(self.individualTextureSelector.currentText)
    clustersAvailable = self.multiRecolorClusterCenters is not None
    isSubsampleOnlyMode = self.multiRecolorSubsampleOnlyRadio.isChecked()

    # Enable apply button if texture is selected and either:
    # - clusters are available (clustering mode), or
    # - we're in subsample-only mode (no clusters needed)
    self.applyIndividualTextureButton.enabled = textureSelected and (clustersAvailable or isSubsampleOnlyMode)
    self.individualRawTextureCheckbox.setEnabled(textureSelected)

  def onApplyIndividualTextureButton(self):
    """Handle Step 2: Individual texture visualization"""
    try:
      qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)
      self.individualProgressBar.setVisible(True)
      self.individualProgressBar.setValue(0)
      self.individualLogInfo.clear()

      atlasModel = self.multiRecolorAtlasModelSelect.currentNode()
      textureDir = self.multiRecolorTextureDirectorySelector.currentPath
      selectedTexture = self.individualTextureSelector.currentText
      useRawTexture = self.individualRawTextureCheckbox.isChecked()

      if not atlasModel:
        self.individualLogInfo.append("Error: No atlas model selected")
        return

      if not selectedTexture:
        self.individualLogInfo.append("Error: No texture selected")
        return

      texturePath = os.path.join(textureDir, selectedTexture)
      if not os.path.exists(texturePath):
        self.individualLogInfo.append(f"Error: Texture file not found: {texturePath}")
        return

      self.individualLogInfo.append(f"Applying texture: {selectedTexture}")

      logic = InterDeCALogic()

      # Check if we're in subsample-only mode
      isSubsampleOnlyMode = self.multiRecolorSubsampleOnlyRadio.isChecked()

      if useRawTexture:
        # Apply raw texture directly without any processing
        self.individualLogInfo.append("Using: Raw texture (no processing)")
        success = logic.applyTextureToModel(atlasModel, texturePath)
      elif isSubsampleOnlyMode:
        # Apply with subsampling but no color quantization
        if self.multiRecolorClusteringPipeline is None:
          self.individualLogInfo.append("Error: No pipeline available. Run Step 1 first.")
          return

        self.individualLogInfo.append("Using: Subsampled face averaging (no color quantization)")
        success = logic.applyTextureWithSubsamplingOnly(
          atlasModel, texturePath,
          clusteringPipeline=self.multiRecolorClusteringPipeline,
          faceAreas=self.multiRecolorFaceAreas,
          progressCallback=self.updateIndividualProgress,
          logCallback=self.logIndividualMessage
        )
      else:
        # Apply with clustered palette
        if self.multiRecolorClusterCenters is None:
          self.individualLogInfo.append("Error: No cluster centers available. Run clustering first.")
          return

        self.individualLogInfo.append("Using: Shared palette from clustering")
        success = logic.applyIndividualTextureWithClusteredPalette(
          atlasModel, texturePath,
          clusterCenters=self.multiRecolorClusterCenters,
          clusteringPipeline=self.multiRecolorClusteringPipeline,
          faceAreas=self.multiRecolorFaceAreas,
          progressCallback=self.updateIndividualProgress,
          logCallback=self.logIndividualMessage
        )

      if success:
        self.individualLogInfo.append("Individual texture visualization completed successfully!")
      else:
        self.individualLogInfo.append("Individual texture visualization failed - check log for details")

      self.individualProgressBar.setVisible(False)
      qt.QApplication.restoreOverrideCursor()

      # Maximize the 3D view
      self.maximize3DViewer(self.individualLogInfo)

    except Exception as e:
      self.individualProgressBar.setVisible(False)
      qt.QApplication.restoreOverrideCursor()
      self.individualLogInfo.append(f"Error: {str(e)}")
      slicer.util.errorDisplay(f"Individual texture visualization failed: {str(e)}")
      import traceback
      traceback.print_exc()

  def onCompareTexturesButton(self):
    """Handle Step 3: Population analysis"""
    try:
      qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)
      self.populationProgressBar.setVisible(True)
      self.populationProgressBar.setValue(0)
      self.populationLogInfo.clear()

      atlasModel = self.multiRecolorAtlasModelSelect.currentNode()
      textureDir = self.multiRecolorTextureDirectorySelector.currentPath

      if not atlasModel:
        self.populationLogInfo.append("Error: No atlas model selected")
        return

      # Check if we have either cluster centers (clustering mode) or pipeline (subsample-only mode)
      isSubsampleOnlyMode = self.multiRecolorSubsampleOnlyRadio.isChecked()
      if self.multiRecolorClusterCenters is None and not isSubsampleOnlyMode:
        self.populationLogInfo.append("Error: No cluster centers available. Run clustering first.")
        return

      if self.multiRecolorClusteringPipeline is None:
        self.populationLogInfo.append("Error: No pipeline available. Run Step 1 first.")
        return

      if not self.multiRecolorTextureFiles:
        self.populationLogInfo.append("Error: No texture files found")
        return

      # Get selected dimensionality reduction method
      if self.pcaRadioButton.isChecked():
        dimReductionMethod = "PCA"
      elif self.umapRadioButton.isChecked():
        dimReductionMethod = "UMAP"
      elif self.icaRadioButton.isChecked():
        dimReductionMethod = "ICA"
      else:
        dimReductionMethod = "PCA"  # Default fallback

      # Get number of components (used for PCA and ICA)
      n_components = self.multiRecolorNumPCsSpin.value

      self.populationLogInfo.append(f"Starting population analysis with {dimReductionMethod}...")
      if dimReductionMethod in ["PCA", "ICA"]:
        self.populationLogInfo.append(f"Computing {n_components} components...")
      self.populationLogInfo.append(f"Analyzing {len(self.multiRecolorTextureFiles)} textures...")

      logic = InterDeCALogic()

      # Perform population analysis (data preparation only) - using pipeline if available
      result = logic.performPopulationAnalysis(
        atlasModel, textureDir, self.multiRecolorTextureFiles,
        clusterCenters=self.multiRecolorClusterCenters,
        clusteringPipeline=self.multiRecolorClusteringPipeline,
        faceAreas=self.multiRecolorFaceAreas,
        dimReductionMethod=dimReductionMethod,
        n_components=n_components,
        progressCallback=self.updatePopulationProgress,
        logCallback=self.logPopulationMessage
      )

      if result.get("success", False):
        self.populationLogInfo.append("Population analysis data preparation completed successfully!")

        # Enable and populate axis selectors if PCA, ICA, or UMAP was used
        if dimReductionMethod in ["PCA", "ICA", "UMAP"] and "n_components" in result:
          n_comps = result["n_components"]
          self.multiRecolorXAxisCombo.clear()
          self.multiRecolorYAxisCombo.clear()

          # Use appropriate labels based on method
          if dimReductionMethod == "PCA":
            component_label = "PC"
          elif dimReductionMethod == "ICA":
            component_label = "IC"
          else:  # UMAP
            component_label = "UMAP"

          for i in range(n_comps):
            comp_label = f"{component_label}{i+1}"
            self.multiRecolorXAxisCombo.addItem(comp_label)
            self.multiRecolorYAxisCombo.addItem(comp_label)

          # Set default axes (Component 1 vs Component 2)
          self.multiRecolorXAxisCombo.setCurrentIndex(0)
          if n_comps > 1:
            self.multiRecolorYAxisCombo.setCurrentIndex(1)

          # Enable axis selectors
          self.multiRecolorXAxisCombo.enabled = True
          self.multiRecolorYAxisCombo.enabled = True

          # Store the result for re-plotting with different axes
          self.multiRecolorPopulationResult = result

          # Enable morphospace controls (Step 4)
          self.morphospaceTextureCombo.clear()
          for textureName in result["texture_names"]:
            self.morphospaceTextureCombo.addItem(textureName)
          self.morphospaceTextureCombo.enabled = True

          self.populationLogInfo.append("Step 4: Morphospace is now available")

        # Get selected axes
        x_axis_idx = self.multiRecolorXAxisCombo.currentIndex if self.multiRecolorXAxisCombo.enabled else 0
        y_axis_idx = self.multiRecolorYAxisCombo.currentIndex if self.multiRecolorYAxisCombo.enabled else 1

        # Get variance explained if available (only for PCA)
        variance_explained = None
        method = result.get("method", "PCA")
        if method == "PCA" and "pca_model" in result:
          variance_explained = result["pca_model"].explained_variance_ratio_

        # Create the plot in the UI layer
        self.populationLogInfo.append("Creating population analysis plot...")
        plotResult = self.createPopulationPlot(
          result["reduced_data"],
          result["texture_names"],
          result["method"],
          x_axis_idx=x_axis_idx,
          y_axis_idx=y_axis_idx,
          variance_explained=variance_explained
        )

        if plotResult.get("success", False):
          self.populationLogInfo.append(f"Created {dimReductionMethod} plot with {len(self.multiRecolorTextureFiles)} texture points")
        else:
          self.populationLogInfo.append("Plot creation failed - data will be available in Data module")
      else:
        self.populationLogInfo.append("Population analysis failed - check log for details")

      self.populationProgressBar.setVisible(False)
      qt.QApplication.restoreOverrideCursor()

    except Exception as e:
      self.populationProgressBar.setVisible(False)
      qt.QApplication.restoreOverrideCursor()
      self.populationLogInfo.append(f"Error: {str(e)}")
      slicer.util.errorDisplay(f"Population analysis failed: {str(e)}")
      import traceback
      traceback.print_exc()

  def onMorphospaceTextureChanged(self):
    """Handle texture selection change in morphospace"""
    # Enable visualize button if a texture is selected and we have PCA, ICA, or UMAP results
    hasTexture = self.morphospaceTextureCombo.currentIndex >= 0
    hasReducer = self.multiRecolorPopulationResult is not None and self.multiRecolorPopulationResult.get("method") in ["PCA", "ICA", "UMAP"]
    self.visualizeMorphospaceButton.enabled = hasTexture and hasReducer

    # If morphospace is already active and texture changes, update the starting point
    if hasReducer and hasTexture and self.morphospaceStartingPoint is not None:
      try:
        # Get the new texture's PCA coordinates
        textureIndex = self.morphospaceTextureCombo.currentIndex
        textureName = self.morphospaceTextureCombo.currentText
        result = self.multiRecolorPopulationResult
        textureNames = result["texture_names"]
        reducedData = result["reduced_data"]

        if textureName in textureNames:
          textureIdx = textureNames.index(textureName)
          self.morphospaceStartingPoint = reducedData[textureIdx].copy()

          # Get the X and Y axis indices from Step 3
          x_axis_idx = self.multiRecolorXAxisCombo.currentIndex if self.multiRecolorXAxisCombo.enabled else 0
          y_axis_idx = self.multiRecolorYAxisCombo.currentIndex if self.multiRecolorYAxisCombo.enabled else 1

          # Update slider positions to match the new starting point
          if self.morphospaceXRange is not None and self.morphospaceYRange is not None:
            x_min, x_max = self.morphospaceXRange
            y_min, y_max = self.morphospaceYRange

            x_start = self.morphospaceStartingPoint[x_axis_idx]
            y_start = self.morphospaceStartingPoint[y_axis_idx]

            # Convert starting coordinates to slider values (0-1000)
            x_slider_value = int(((x_start - x_min) / (x_max - x_min)) * 1000) if x_max != x_min else 500
            y_slider_value = int(((y_start - y_min) / (y_max - y_min)) * 1000) if y_max != y_min else 500

            self.morphospaceXSlider.blockSignals(True)
            self.morphospaceYSlider.blockSignals(True)
            self.morphospaceXSlider.setValue(x_slider_value)
            self.morphospaceYSlider.setValue(y_slider_value)
            self.morphospaceXSlider.blockSignals(False)
            self.morphospaceYSlider.blockSignals(False)

            # Update labels
            self.morphospaceXLabel.text = f"X: {x_start:.2f}"
            self.morphospaceYLabel.text = f"Y: {y_start:.2f}"

          # Update current point to the new starting point
          self.morphospaceCurrentPoint = self.morphospaceStartingPoint.copy()

          # Update plot and apply colors
          self.updateMorphospacePoint()
          self.applyMorphospaceColors(self.morphospaceCurrentPoint)

          self.morphospaceLogInfo.append(f"Switched to texture: {textureName}")
      except Exception as e:
        self.morphospaceLogInfo.append(f"Error switching texture: {str(e)}")

  def onMorphospaceXSliderChanged(self, value):
    """Handle X-axis slider value changes in morphospace"""
    if self.morphospaceCurrentPoint is None or self.morphospaceXRange is None:
      return

    # Convert slider value (0-1000) to actual PCA coordinate
    x_min, x_max = self.morphospaceXRange
    x_coord = x_min + (value / 1000.0) * (x_max - x_min)

    # Update the X coordinate label
    self.morphospaceXLabel.text = f"X: {x_coord:.2f}"

    # Update visualization
    self.updateMorphospaceVisualizationXY()

  def onMorphospaceYSliderChanged(self, value):
    """Handle Y-axis slider value changes in morphospace"""
    if self.morphospaceCurrentPoint is None or self.morphospaceYRange is None:
      return

    # Convert slider value (0-1000) to actual PCA coordinate
    y_min, y_max = self.morphospaceYRange
    y_coord = y_min + (value / 1000.0) * (y_max - y_min)

    # Update the Y coordinate label
    self.morphospaceYLabel.text = f"Y: {y_coord:.2f}"

    # Update visualization
    self.updateMorphospaceVisualizationXY()

  def onVisualizeMorphospace(self):
    """Start morphospace visualization with selected texture"""
    try:
      # Check if we have PCA, ICA, or UMAP results
      if not self.multiRecolorPopulationResult or self.multiRecolorPopulationResult.get("method") not in ["PCA", "ICA", "UMAP"]:
        self.morphospaceLogInfo.append("Error: Run PCA, ICA, or UMAP population analysis first (Step 3)")
        return

      # Get selected texture
      textureIndex = self.morphospaceTextureCombo.currentIndex
      if textureIndex < 0:
        self.morphospaceLogInfo.append("Error: Select a starting texture")
        return

      textureName = self.morphospaceTextureCombo.currentText

      self.morphospaceLogInfo.clear()
      self.morphospaceLogInfo.append(f"Starting morphospace visualization with texture: {textureName}")

      # Get the PCA coordinates for the selected texture
      result = self.multiRecolorPopulationResult
      textureNames = result["texture_names"]
      reducedData = result["reduced_data"]

      # Find the index of the selected texture
      try:
        textureIdx = textureNames.index(textureName)
      except ValueError:
        self.morphospaceLogInfo.append(f"Error: Texture {textureName} not found in PCA results")
        return

      # Store the starting point
      self.morphospaceStartingPoint = reducedData[textureIdx].copy()
      self.morphospaceCurrentPoint = self.morphospaceStartingPoint.copy()

      # Get the X and Y axis indices from Step 3
      x_axis_idx = self.multiRecolorXAxisCombo.currentIndex if self.multiRecolorXAxisCombo.enabled else 0
      y_axis_idx = self.multiRecolorYAxisCombo.currentIndex if self.multiRecolorYAxisCombo.enabled else 1

      # Calculate min/max ranges for X and Y axes based on all data
      x_coords = reducedData[:, x_axis_idx]
      y_coords = reducedData[:, y_axis_idx]

      self.morphospaceXRange = (float(np.min(x_coords)), float(np.max(x_coords)))
      self.morphospaceYRange = (float(np.min(y_coords)), float(np.max(y_coords)))

      self.morphospaceLogInfo.append(f"Starting coordinates: X={self.morphospaceStartingPoint[x_axis_idx]:.2f}, Y={self.morphospaceStartingPoint[y_axis_idx]:.2f}")
      self.morphospaceLogInfo.append(f"X-axis range: [{self.morphospaceXRange[0]:.2f}, {self.morphospaceXRange[1]:.2f}]")
      self.morphospaceLogInfo.append(f"Y-axis range: [{self.morphospaceYRange[0]:.2f}, {self.morphospaceYRange[1]:.2f}]")

      # Enable sliders
      self.morphospaceXSlider.enabled = True
      self.morphospaceYSlider.enabled = True

      # Set slider positions to match the starting point
      x_min, x_max = self.morphospaceXRange
      y_min, y_max = self.morphospaceYRange

      x_start = self.morphospaceStartingPoint[x_axis_idx]
      y_start = self.morphospaceStartingPoint[y_axis_idx]

      # Convert starting coordinates to slider values (0-1000)
      x_slider_value = int(((x_start - x_min) / (x_max - x_min)) * 1000) if x_max != x_min else 500
      y_slider_value = int(((y_start - y_min) / (y_max - y_min)) * 1000) if y_max != y_min else 500

      self.morphospaceXSlider.blockSignals(True)
      self.morphospaceYSlider.blockSignals(True)
      self.morphospaceXSlider.setValue(x_slider_value)
      self.morphospaceYSlider.setValue(y_slider_value)
      self.morphospaceXSlider.blockSignals(False)
      self.morphospaceYSlider.blockSignals(False)

      # Update labels
      self.morphospaceXLabel.text = f"X: {x_start:.2f}"
      self.morphospaceYLabel.text = f"Y: {y_start:.2f}"

      # Create or update the plot with the moving point
      self.createMorphospacePlot()

      # Display the chart using ShowChartInLayout (this links chart to a plot view)
      if hasattr(self, 'morphospaceChartNode') and self.morphospaceChartNode:
        try:
          plotsLogic = slicer.modules.plots.logic()
          plotsLogic.ShowChartInLayout(self.morphospaceChartNode)
          slicer.app.processEvents()
        except Exception:
          pass  # Silently handle if ShowChartInLayout fails

      # Switch to our custom 3D + Plot layout for side-by-side view
      logic = InterDeCALogic()
      layout_id, plotViewNode = logic.setMorphospaceLayout()
      
      # Ensure chart is set on the plot view node in our custom layout
      if plotViewNode and hasattr(self, 'morphospaceChartNode') and self.morphospaceChartNode:
        if plotViewNode.GetPlotChartNodeID() != self.morphospaceChartNode.GetID():
          plotViewNode.SetPlotChartNodeID(self.morphospaceChartNode.GetID())
          slicer.app.processEvents()
          slicer.util.forceRenderAllViews()

      # Apply the starting texture colors to the model
      self.applyMorphospaceColors(self.morphospaceCurrentPoint)

      self.morphospaceLogInfo.append("Morphospace visualization ready. Use sliders to explore PCA space.")

    except Exception as e:
      self.morphospaceLogInfo.append(f"Error: {str(e)}")
      import traceback
      traceback.print_exc()

  def updateMorphospaceVisualizationXY(self):
    """Update morphospace visualization based on X and Y slider positions"""
    if not self.multiRecolorPopulationResult or self.morphospaceXRange is None or self.morphospaceYRange is None:
      return

    try:
      result = self.multiRecolorPopulationResult

      # Get the X and Y axis indices from Step 3
      x_axis_idx = self.multiRecolorXAxisCombo.currentIndex if self.multiRecolorXAxisCombo.enabled else 0
      y_axis_idx = self.multiRecolorYAxisCombo.currentIndex if self.multiRecolorYAxisCombo.enabled else 1

      # Convert slider values to actual PCA coordinates
      x_min, x_max = self.morphospaceXRange
      y_min, y_max = self.morphospaceYRange

      x_coord = x_min + (self.morphospaceXSlider.value / 1000.0) * (x_max - x_min)
      y_coord = y_min + (self.morphospaceYSlider.value / 1000.0) * (y_max - y_min)


      self.morphospaceCurrentPoint = self.morphospaceStartingPoint.copy()
      self.morphospaceCurrentPoint[x_axis_idx] = x_coord
      self.morphospaceCurrentPoint[y_axis_idx] = y_coord

      # Update the plot
      self.updateMorphospacePoint()

      # Apply colors to the model
      self.applyMorphospaceColors(self.morphospaceCurrentPoint)

    except Exception as e:
      self.morphospaceLogInfo.append(f"Error updating visualization: {str(e)}")
      import traceback
      traceback.print_exc()

  def createMorphospacePlot(self):
    """Create or update the morphospace plot with the moving point"""
    try:
      result = self.multiRecolorPopulationResult
      if not result:
        return

      # Get the current axes
      x_axis_idx = self.multiRecolorXAxisCombo.currentIndex if self.multiRecolorXAxisCombo.enabled else 0
      y_axis_idx = self.multiRecolorYAxisCombo.currentIndex if self.multiRecolorYAxisCombo.enabled else 1

      # Check if we already have a chart node for morphospace
      if hasattr(self, 'morphospaceChartNode') and self.morphospaceChartNode:
        chartNode = self.morphospaceChartNode
        # Remove old moving point series if it exists
        if hasattr(self, 'morphospaceMovingPointSeries') and self.morphospaceMovingPointSeries:
          chartNode.RemovePlotSeriesNodeID(self.morphospaceMovingPointSeries.GetID())
      else:
        # Get variance explained (only for PCA)
        variance_explained = None
        method = result.get("method", "PCA")
        if method == "PCA" and "pca_model" in result:
          variance_explained = result["pca_model"].explained_variance_ratio_

        # Create the base plot (same as population analysis)
        # Pass displayInLayout=False to avoid layout switching - we'll display in our custom layout
        plotResult = self.createPopulationPlot(
          result["reduced_data"],
          result["texture_names"],
          result["method"],
          x_axis_idx=x_axis_idx,
          y_axis_idx=y_axis_idx,
          variance_explained=variance_explained,
          displayInLayout=False
        )

        if not plotResult.get("success", False):
          self.morphospaceLogInfo.append("Failed to create plot")
          return

        # Get the chart node
        chartNode = plotResult.get("chart_node")
        if not chartNode:
          return

        # Store the chart node for later use (e.g., displaying in layout)
        self.morphospaceChartNode = chartNode

      # Create a new series for the moving point
      self.morphospaceMovingPointTable = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTableNode")
      self.morphospaceMovingPointTable.SetName("Morphospace_MovingPoint_Data")

      # Create arrays for the moving point with appropriate labels
      method = result.get("method", "PCA")
      if method == "PCA":
        x_comp_label = f"PC{x_axis_idx + 1}"
        y_comp_label = f"PC{y_axis_idx + 1}"
      elif method == "ICA":
        x_comp_label = f"IC{x_axis_idx + 1}"
        y_comp_label = f"IC{y_axis_idx + 1}"
      else:
        x_comp_label = f"{method}_C{x_axis_idx + 1}"
        y_comp_label = f"{method}_C{y_axis_idx + 1}"

      xArray = vtk.vtkFloatArray()
      xArray.SetName(x_comp_label)
      yArray = vtk.vtkFloatArray()
      yArray.SetName(y_comp_label)
      labelArray = vtk.vtkStringArray()
      labelArray.SetName("Label")

      # Add the current point
      xArray.InsertNextValue(float(self.morphospaceCurrentPoint[x_axis_idx]))
      yArray.InsertNextValue(float(self.morphospaceCurrentPoint[y_axis_idx]))
      labelArray.InsertNextValue("Current")

      self.morphospaceMovingPointTable.AddColumn(xArray)
      self.morphospaceMovingPointTable.AddColumn(yArray)
      self.morphospaceMovingPointTable.AddColumn(labelArray)

      # Create series for the moving point
      self.morphospaceMovingPointSeries = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLPlotSeriesNode")
      self.morphospaceMovingPointSeries.SetName("MovingPoint")
      self.morphospaceMovingPointSeries.SetAndObserveTableNodeID(self.morphospaceMovingPointTable.GetID())
      self.morphospaceMovingPointSeries.SetXColumnName(x_comp_label)
      self.morphospaceMovingPointSeries.SetYColumnName(y_comp_label)
      self.morphospaceMovingPointSeries.SetLabelColumnName("Label")
      self.morphospaceMovingPointSeries.SetPlotType(slicer.vtkMRMLPlotSeriesNode.PlotTypeScatter)
      self.morphospaceMovingPointSeries.SetMarkerStyle(slicer.vtkMRMLPlotSeriesNode.MarkerStyleSquare)
      self.morphospaceMovingPointSeries.SetMarkerSize(12)
      self.morphospaceMovingPointSeries.SetLineStyle(slicer.vtkMRMLPlotSeriesNode.LineStyleNone)
      self.morphospaceMovingPointSeries.SetColor(1.0, 0.0, 0.0)  # Red color for moving point

      # Add to chart
      chartNode.AddAndObservePlotSeriesNodeID(self.morphospaceMovingPointSeries.GetID())

      self.morphospaceLogInfo.append("Plot created with moving point")

    except Exception as e:
      self.morphospaceLogInfo.append(f"Error creating morphospace plot: {str(e)}")
      import traceback
      traceback.print_exc()

  def updateMorphospacePoint(self):
    """Update the position of the moving point in the plot"""
    if not self.morphospaceMovingPointTable or self.morphospaceCurrentPoint is None:
      return

    try:
      # Get the current axes
      x_axis_idx = self.multiRecolorXAxisCombo.currentIndex if self.multiRecolorXAxisCombo.enabled else 0
      y_axis_idx = self.multiRecolorYAxisCombo.currentIndex if self.multiRecolorYAxisCombo.enabled else 1

      # Get the new coordinates
      new_x = float(self.morphospaceCurrentPoint[x_axis_idx])
      new_y = float(self.morphospaceCurrentPoint[y_axis_idx])

      # Update the table with new coordinates
      table = self.morphospaceMovingPointTable.GetTable()
      xArray = table.GetColumn(0)
      yArray = table.GetColumn(1)

      if xArray and yArray and table.GetNumberOfRows() > 0:
        xArray.SetValue(0, new_x)
        yArray.SetValue(0, new_y)
        
        # Mark arrays and table as modified
        xArray.Modified()
        yArray.Modified()
        table.Modified()
        self.morphospaceMovingPointTable.Modified()

        # Force plot update by removing and re-adding the series
        if hasattr(self, 'morphospaceMovingPointSeries') and self.morphospaceMovingPointSeries:
          if hasattr(self, 'morphospaceChartNode') and self.morphospaceChartNode:
            seriesId = self.morphospaceMovingPointSeries.GetID()
            # Remove the series
            self.morphospaceChartNode.RemovePlotSeriesNodeID(seriesId)
            # Re-add the series (this forces the plot to re-read the data)
            self.morphospaceChartNode.AddAndObservePlotSeriesNodeID(seriesId)

    except Exception as e:
      self.morphospaceLogInfo.append(f"Error updating point: {str(e)}")
      import traceback
      traceback.print_exc()

  def applyMorphospaceColors(self, pca_coordinates):
    """Apply colors to the model based on PCA/ICA coordinates using inverse transform"""
    try:
      result = self.multiRecolorPopulationResult
      if not result:
        return

      # Get the model (works for both PCA and ICA)
      model = result.get("model") or result.get("pca_model")
      if not model:
        return

      method = result.get("method", "PCA")

      # Get the atlas model
      atlasModel = self.multiRecolorAtlasModelSelect.currentNode()
      if not atlasModel:
        self.morphospaceLogInfo.append("Error: No atlas model selected")
        return

      # Clear any texture that might be applied to the model
      displayNode = atlasModel.GetDisplayNode()
      if displayNode:
        try:
          displayNode.SetTextureImageDataConnection(None)
        except AttributeError:
          try:
            displayNode.SetAndObserveTextureImageData(None)
          except AttributeError:
            pass  # No texture to remove

      # Apply inverse transform to get the color vector
      # color_vector = mean + coordinates @ components (works for both PCA and ICA)
      color_vector = model.inverse_transform(pca_coordinates.reshape(1, -1))[0]

      # Get the clustering pipeline to know how colors were structured
      if not self.multiRecolorClusteringPipeline:
        self.morphospaceLogInfo.append("Error: No clustering pipeline available")
        return

      # Check if subsampling was used
      useSubsampling = result.get("use_subsampling", False)

      logic = InterDeCALogic()

      if useSubsampling:
        # Subsampling approach: color_vector is (n_subsampled_faces * 3,)
        # Reshape to (n_subsampled_faces, 3) for Lab colors
        subsampledFaceIndices = self.multiRecolorClusteringPipeline.subsampledFaceIndices
        n_subsampled = len(subsampledFaceIndices)

        if len(color_vector) != n_subsampled * 3:
          self.morphospaceLogInfo.append(f"Error: Color vector size mismatch. Expected {n_subsampled * 3}, got {len(color_vector)}")
          return

        # Reshape to (n_subsampled, 3) - these are Lab colors
        subsampledColorsLab = color_vector.reshape(n_subsampled, 3)

        # Clip Lab values to valid ranges
        # L: 0-100, a: -128 to 127, b: -128 to 127
        subsampledColorsLab[:, 0] = np.clip(subsampledColorsLab[:, 0], 0, 100)
        subsampledColorsLab[:, 1] = np.clip(subsampledColorsLab[:, 1], -128, 127)
        subsampledColorsLab[:, 2] = np.clip(subsampledColorsLab[:, 2], -128, 127)

        # Apply the subsampled colors to the model
        success = logic.applyMorphospaceColorsSubsampled(
          atlasModel, subsampledColorsLab,
          self.multiRecolorClusteringPipeline
        )
      else:
        # Cluster-based approach: color_vector is (n_clusters,) area weights
        # This is NOT colors, but area-weighted cluster assignments
        # We need to reconstruct cluster colors from this
        n_clusters = self.multiRecolorClusteringPipeline.consolidatedClusters

        if len(color_vector) != n_clusters:
          self.morphospaceLogInfo.append(f"Error: Color vector size mismatch. Expected {n_clusters}, got {len(color_vector)}")
          return

        # The color_vector represents area weights, not colors
        # We need to convert this back to cluster colors somehow
        # This approach doesn't make sense for morphospace - we should always use subsampling
        self.morphospaceLogInfo.append("Error: Morphospace requires subsampling to be enabled in Step 1")
        self.morphospaceLogInfo.append("Please re-run Step 1 with a non-zero 'Subsampled Faces' value")
        return

      if success:
        self.morphospaceLogInfo.append("Colors applied to model")
      else:
        self.morphospaceLogInfo.append("Failed to apply colors")

    except Exception as e:
      self.morphospaceLogInfo.append(f"Error applying colors: {str(e)}")
      import traceback
      traceback.print_exc()

  def updateClusteringProgress(self, value):
    """Update progress bar for clustering operations"""
    self.clusteringProgressBar.setValue(int(value))
    slicer.app.processEvents()

  def logClusteringMessage(self, message):
    """Log message to the clustering log"""
    self.clusteringLogInfo.append(message)
    slicer.app.processEvents()

  def updateIndividualProgress(self, value):
    """Update progress bar for individual visualization operations"""
    self.individualProgressBar.setValue(int(value))
    slicer.app.processEvents()

  def logIndividualMessage(self, message):
    """Log message to the individual visualization log"""
    self.individualLogInfo.append(message)
    slicer.app.processEvents()

  def updatePopulationProgress(self, value):
    """Update progress bar for population analysis operations"""
    self.populationProgressBar.setValue(int(value))
    slicer.app.processEvents()

  def logPopulationMessage(self, message):
    """Log message to the population analysis log"""
    self.populationLogInfo.append(message)
    slicer.app.processEvents()

  def getCachedFaceAreas(self, modelNode):
    """Get cached face areas for a model node, or calculate and cache them"""
    modelId = modelNode.GetID()

    if modelId not in self.faceAreasCache:
      self.clusteringLogInfo.append("Calculating and caching face areas...")
      logic = InterDeCALogic()
      polyData = modelNode.GetPolyData()
      if not polyData:
        self.clusteringLogInfo.append("Error: No polydata in model node")
        return None

      faceAreas = logic._calculateFaceAreas(polyData)
      if faceAreas is not None:
        self.faceAreasCache[modelId] = faceAreas
        self.clusteringLogInfo.append(f"Cached face areas for {polyData.GetNumberOfCells()} faces")
      else:
        self.clusteringLogInfo.append("Warning: Failed to calculate face areas")
        return None
    else:
      self.clusteringLogInfo.append("Using cached face areas")

    return self.faceAreasCache[modelId]

  # ================================ PLOT CREATION METHODS ================================
  # These methods handle UI aspects of plotting and were moved from Logic class

  def _createStandardPlot(self, reducedData, algorithm, colorSpace, modelName="", useRegion=False):
    """Create a standard single-series scatter plot"""
    try:
      # Create a scatter plot node
      plotSeriesNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLPlotSeriesNode")
      plotSeriesNode.SetName(f"Colors_EDA_{algorithm}_{colorSpace}")

      # Create arrays for the plot data
      xArray = vtk.vtkFloatArray()
      xArray.SetName(f"{algorithm}_Component_1")
      xArray.SetNumberOfTuples(reducedData.shape[0])

      yArray = vtk.vtkFloatArray()
      yArray.SetName(f"{algorithm}_Component_2")
      yArray.SetNumberOfTuples(reducedData.shape[0])

      # Fill arrays with data
      for i in range(reducedData.shape[0]):
        xArray.SetValue(i, reducedData[i, 0])
        yArray.SetValue(i, reducedData[i, 1])

      # Create table for the plot
      tableNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTableNode")
      tableNode.SetName(f"Colors_EDA_Data_{algorithm}_{colorSpace}")
      tableNode.AddColumn(xArray)
      tableNode.AddColumn(yArray)

      # Set up the plot series
      plotSeriesNode.SetAndObserveTableNodeID(tableNode.GetID())
      plotSeriesNode.SetXColumnName(xArray.GetName())
      plotSeriesNode.SetYColumnName(yArray.GetName())
      plotSeriesNode.SetPlotType(slicer.vtkMRMLPlotSeriesNode.PlotTypeScatter)
      plotSeriesNode.SetMarkerStyle(slicer.vtkMRMLPlotSeriesNode.MarkerStyleCircle)
      plotSeriesNode.SetMarkerSize(6)
      plotSeriesNode.SetLineStyle(slicer.vtkMRMLPlotSeriesNode.LineStyleNone)

      # Create plot chart
      plotChartNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLPlotChartNode")
      plotChartNode.SetName(f"Colors_EDA_Chart_{algorithm}_{colorSpace}")
      plotChartNode.AddAndObservePlotSeriesNodeID(plotSeriesNode.GetID())

      # Build title with model name and region info
      regionText = " (Selected Region)" if useRegion else ""
      title = f"{modelName}: {algorithm} on {colorSpace} Face Colors{regionText}"
      plotChartNode.SetTitle(title)
      plotChartNode.SetXAxisTitle(f"{algorithm} Component 1")
      plotChartNode.SetYAxisTitle(f"{algorithm} Component 2")

      # Show in plot view
      layoutManager = slicer.app.layoutManager()
      plotWidget = layoutManager.plotWidget(0)
      plotViewNode = plotWidget.mrmlPlotViewNode()
      plotViewNode.SetPlotChartNodeID(plotChartNode.GetID())

      return {"success": True, "chartNode": plotChartNode}

    except Exception as e:
      print(f"Error creating standard plot: {e}")
      return {"success": False, "chartNode": None}

  def _createColoredScatterPlot(self, reducedData, originalColorData, algorithm, colorSpace, enhanceColors=False, modelName="", useRegion=False):
    """
    Render colored scatter by quantizing hue into bins and creating one series per bin.
    This avoids the 'single color per series' limitation in Slicer plots.
    """
    try:
        import colorsys

        # 1) Compute hue (deg), sat, val from your 4D HSV repr
        hue_cos = originalColorData[:, 0]
        hue_sin = originalColorData[:, 1]
        hue_deg = (np.degrees(np.arctan2(hue_sin, hue_cos)) + 360.0) % 360.0  # [0,360)
        sat = np.clip(originalColorData[:, 2] / 100.0, 0.0, 1.0)
        val = np.clip(originalColorData[:, 3] / 100.0, 0.0, 1.0)

        # 2) Optional visibility boost
        if enhanceColors:
            sat = np.maximum(sat, 0.7)
            val = np.maximum(val, 0.8)

        min_hue = 0.0
        max_hue = 360.0
        min_hue = np.minimum(min_hue, np.min(hue_deg))
        max_hue = np.maximum(max_hue, np.max(hue_deg))

        # 3) Bin hues
        n_bins = 36  # 10° per bin; bump to 72 if you want finer gradation
        edges = np.linspace(min_hue, max_hue, n_bins + 1, endpoint=True)
        centers = (edges[:-1] + edges[1:]) / 2.0
        bin_idx = np.clip(np.digitize(hue_deg, edges, right=False) - 1, 0, n_bins - 1)

        # 4) Make a chart and populate one series per bin
        plotChartNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLPlotChartNode")
        plotChartNode.SetName(f"Colors_EDA_Chart_{algorithm}_{colorSpace}")

        # Build title with model name, region info, and color enhancement
        regionText = " (Selected Region)" if useRegion else ""
        colorText = " (Enhanced Colors)" if enhanceColors else " (Actual Colors)"
        title = f"{modelName}: {algorithm} on {colorSpace} Face Colors{regionText}{colorText}"
        plotChartNode.SetTitle(title)
        plotChartNode.SetXAxisTitle(f"{algorithm} Component 1")
        plotChartNode.SetYAxisTitle(f"{algorithm} Component 2")
        plotChartNode.SetLegendVisibility(False)

        # Build series for occupied bins only (keeps node count tight)
        for k in range(n_bins):
            mask = (bin_idx == k)
            if not np.any(mask):
                continue

            X = reducedData[mask, 0]
            Y = reducedData[mask, 1]

            # Representative color for the bin: use bin center hue and the mean sat/val of points in the bin
            mean_sat = float(np.mean(sat[mask]))
            mean_val = float(np.mean(val[mask]))
            r, g, b = colorsys.hsv_to_rgb(centers[k] / 360.0, mean_sat, mean_val)

            # Build table
            xArray = vtk.vtkFloatArray(); xArray.SetName(f"{algorithm}_Component_1"); xArray.SetNumberOfTuples(X.shape[0])
            yArray = vtk.vtkFloatArray(); yArray.SetName(f"{algorithm}_Component_2"); yArray.SetNumberOfTuples(Y.shape[0])
            for i in range(X.shape[0]):
                xArray.SetValue(i, float(X[i])); yArray.SetValue(i, float(Y[i]))

            tableNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTableNode")
            tableNode.SetName(f"Colors_EDA_Data_{algorithm}_{colorSpace}_bin{k:02d}")
            tableNode.AddColumn(xArray); tableNode.AddColumn(yArray)

            seriesNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLPlotSeriesNode")
            seriesNode.SetName(f"Colors_EDA_{algorithm}_{colorSpace}_bin{k:02d}")
            seriesNode.SetAndObserveTableNodeID(tableNode.GetID())
            seriesNode.SetXColumnName(xArray.GetName())
            seriesNode.SetYColumnName(yArray.GetName())
            seriesNode.SetPlotType(slicer.vtkMRMLPlotSeriesNode.PlotTypeScatter)
            seriesNode.SetMarkerStyle(slicer.vtkMRMLPlotSeriesNode.MarkerStyleCircle)
            seriesNode.SetMarkerSize(6)
            seriesNode.SetLineStyle(slicer.vtkMRMLPlotSeriesNode.LineStyleNone)
            seriesNode.SetColor(float(r), float(g), float(b))

            plotChartNode.AddAndObservePlotSeriesNodeID(seriesNode.GetID())

        # Show chart
        layoutManager = slicer.app.layoutManager()
        plotWidget = layoutManager.plotWidget(0)
        plotViewNode = plotWidget.mrmlPlotViewNode()
        plotViewNode.SetPlotChartNodeID(plotChartNode.GetID())

        return {"success": True, "chartNode": plotChartNode}

    except Exception as e:
        print(f"Error creating colored scatter plot: {e}")
        return {"success": False, "chartNode": None}

  def createColorsEDAPlot(self, reducedData, specimenNames, nFaces, colorSpace, algorithm, originalColorData=None, enhanceColors=False, modelName="", useRegion=False):
    """
    Create a Colors EDA plot using Slicer's plotting functionality.

    This method handles the UI aspects of plotting and should be called from
    the Widget class after the Logic class has prepared the data.

    Args:
        reducedData: numpy array of shape (N_samples, 2)
        specimenNames: list of specimen names
        nFaces: number of faces per specimen
        colorSpace: "RGB" or "HSV"
        algorithm: "PCA", "ICA", or "UMAP"
        originalColorData: numpy array of original color data for hue-based coloring (optional)
        enhanceColors: whether to enhance colors for visibility
        modelName: name of the model being analyzed
        useRegion: whether region-based analysis is active

    Returns:
        dict: {"success": bool, "chartNode": node} if successful
    """
    try:
      # Create enhanced plot with color information when HSV data is available
      if colorSpace == "HSV" and originalColorData is not None and originalColorData.shape[1] >= 4:
        return self._createColoredScatterPlot(reducedData, originalColorData, algorithm, colorSpace, enhanceColors, modelName, useRegion)

      # Standard single-series plot for RGB or when no color data available
      return self._createStandardPlot(reducedData, algorithm, colorSpace, modelName, useRegion)

    except Exception as e:
      print(f"Error creating Colors EDA plot: {e}")
      return {"success": False, "chartNode": None}


#
# ClusteringPipeline
#

class ClusteringPipeline:
  """
  Stores all configuration and results from multi-texture clustering pipeline

  This class encapsulates the complete clustering workflow including:
  - Luminosity normalization transforms (if enabled)
  - Per-texture initial and consolidated clustering results
  - Cluster reordering mappings for cross-texture consistency
  """

  def __init__(self, initialClusters, consolidatedClusters, normalizeLuminosity=False, useNeighborAverage=False):
    """
    Initialize clustering pipeline configuration

    Args:
        initialClusters: number of initial clusters per texture
        consolidatedClusters: number of consolidated clusters
        normalizeLuminosity: whether luminosity normalization is enabled
        useNeighborAverage: whether to use neighbor average colors for clustering
    """
    self.initialClusters = initialClusters
    self.consolidatedClusters = consolidatedClusters
    self.normalizeLuminosity = normalizeLuminosity
    self.useNeighborAverage = useNeighborAverage

    # Luminosity normalization
    self.pooledLCStats = None  # (mu_pool, sd_pool) if normalization enabled
    self.perTextureLCStats = {}  # texture -> (mu_img, sd_img)

    # Per-texture clustering results
    self.perTextureInitialLabels = {}  # texture -> initial cluster labels
    self.perTextureConsolidatedLabels = {}  # texture -> consolidated cluster labels
    self.perTextureInitialCentroids = {}  # texture -> initial centroids
    self.perTextureConsolidatedCentroids = {}  # texture -> consolidated centroids
    self.perTextureConsolidationMapping = {}  # texture -> initial->consolidated mapping
    self.perTextureReorderMapping = {}  # texture -> reorder mapping

    # Reference centroids (from first texture)
    self.referenceCentroids = None
    self.referenceTexture = None

    # Shared palette (averaged across all textures)
    self.sharedPalette = None  # Will be computed after all textures are processed

    # Subsampling information
    self.subsampledFaceIndices = None  # Indices of subsampled faces
    self.nearestNeighborMapping = None  # Mapping from all faces to nearest sampled face

    # List of processed textures in order
    self.textureFiles = []

  def addTextureResults(self, textureFile, clusteringResult, lcStats=None, reorderMapping=None):
    """
    Add clustering results for a texture

    Args:
        textureFile: texture filename
        clusteringResult: dict from _clusterAndConsolidateSingleTexture
        lcStats: (mu_img, sd_img) if normalization enabled
        reorderMapping: reorder mapping if not first texture
    """
    self.textureFiles.append(textureFile)

    if lcStats is not None:
      self.perTextureLCStats[textureFile] = lcStats

    self.perTextureInitialLabels[textureFile] = clusteringResult["initialLabels"]
    self.perTextureConsolidatedLabels[textureFile] = clusteringResult["consolidatedLabels"]
    self.perTextureInitialCentroids[textureFile] = clusteringResult["initialCentroids"]
    self.perTextureConsolidatedCentroids[textureFile] = clusteringResult["consolidatedCentroids"]
    self.perTextureConsolidationMapping[textureFile] = clusteringResult["consolidationMapping"]

    if reorderMapping is not None:
      self.perTextureReorderMapping[textureFile] = reorderMapping

    # Set reference from first texture
    if self.referenceCentroids is None:
      self.referenceCentroids = clusteringResult["consolidatedCentroids"].copy()
      self.referenceTexture = textureFile

  def computeSharedPalette(self):
    """
    Compute a shared palette by averaging all consolidated centroids across textures

    This creates a common color palette that represents the typical colors across
    all textures, which can be used for consistent visualization in Step 2.

    Returns:
        numpy array of shared palette centroids in Lab space (consolidatedClusters, 3)
    """
    if len(self.textureFiles) == 0:
      return None

    # Collect all consolidated centroids (after reordering)
    allCentroids = []

    for textureFile in self.textureFiles:
      centroids = self.perTextureConsolidatedCentroids[textureFile]

      # Apply reordering if available
      if textureFile in self.perTextureReorderMapping:
        reorderMapping = self.perTextureReorderMapping[textureFile]
        reorderedCentroids = np.zeros_like(centroids)
        for oldIdx, newIdx in enumerate(reorderMapping):
          reorderedCentroids[newIdx] = centroids[oldIdx]
        centroids = reorderedCentroids

      allCentroids.append(centroids)

    # Average centroids across all textures for each cluster
    allCentroids = np.array(allCentroids)  # (numTextures, consolidatedClusters, 3)
    sharedPalette = allCentroids.mean(axis=0)  # (consolidatedClusters, 3)

    return sharedPalette

  def getConsolidatedCentroidsForTexture(self, textureFile):
    """Get consolidated centroids for a specific texture (after reordering if applicable)"""
    if textureFile not in self.perTextureConsolidatedCentroids:
      return None

    centroids = self.perTextureConsolidatedCentroids[textureFile]

    # Apply reordering if available
    if textureFile in self.perTextureReorderMapping:
      reorderMapping = self.perTextureReorderMapping[textureFile]
      # Create reordered centroids
      reorderedCentroids = np.zeros_like(centroids)
      for oldIdx, newIdx in enumerate(reorderMapping):
        reorderedCentroids[newIdx] = centroids[oldIdx]
      return reorderedCentroids

    return centroids

#
# DeCALogic
#

class InterDeCALogic(ScriptedLoadableModuleLogic):
  """Logic class implementing InterDeCA analysis algorithms.

  This class contains the core computational methods for:
  - Dense correspondence analysis
  - Texture processing and color extraction
  - Landmark manipulation and subsampling
  - Atlas generation and alignment
  - Statistical analysis of shape and color patterns

  The interface is designed to be independent of the GUI widget,
  allowing batch processing and scripted automation.

  Uses ScriptedLoadableModuleLogic base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """
  def runSubsetLandmarks(self, baseNode, lmDirectory, lmDirectorySubset):
    """Creates subset of landmarks based on selected points.

    Args:
      baseNode: Markup node with selected/unselected points
      lmDirectory: Source directory containing landmark files
      lmDirectorySubset: Output directory for subset landmarks
    """
    deletionIndex = []  # Collects indices of unselected points
    for i in range(baseNode.GetNumberOfControlPoints()):
      if not baseNode.GetNthControlPointSelected(i):  # Checks selection status
        deletionIndex.append(i)  # Marks for deletion
    for lmFileName in os.listdir(lmDirectory):
      if not _is_visible_dataset_file(lmFileName):  # Skips hidden/resource-fork files
        continue
      currentLMNode = slicer.util.loadMarkups(os.path.join(lmDirectory, lmFileName))  # Loads landmark file
      for index in reversed(deletionIndex):  # Removes points in reverse order
        currentLMNode.RemoveNthControlPoint(index)  # Deletes unselected point
      slicer.util.saveNode(currentLMNode, os.path.join(lmDirectorySubset, lmFileName))  # Saves subset
      slicer.mrmlScene.RemoveNode(currentLMNode)  # Cleans up scene

  def runCheckPoints(self, atlasNode, spacingTolerance):
    spacingPercentage = spacingTolerance/100
    templateModel = self.downsampleModel(atlasNode, spacingPercentage)
    return templateModel, templateModel.GetNumberOfPoints()

  def runDeCAL(self, baseNode, baseLMPath, meshDirectory, landmarkDirectory, outputDirectory, spacingTolerance):
    spacingPercentage = spacingTolerance/100
    loadOption=False
    baseLandmarks=self.fiducialNodeToPolyData(baseLMPath, loadOption).GetPoints()
    landmarkNames, landmarks = self.importLandmarks(landmarkDirectory)
    self.modelNames, models = self.importMeshes(meshDirectory, ['ply','stl','vtp','vtk','obj'], restrict_to=landmarkNames)
    self.outputDirectory = outputDirectory
    denseCorrespondenceGroup = self.denseCorrespondenceBaseMesh(landmarks, models, baseNode.GetPolyData(), baseLandmarks)
    # get downsampled template with index array
    indexArrayName = "indexArray"
    self.addIndexArray(baseNode, indexArrayName)
    templateModel = self.downsampleModel(baseNode, spacingPercentage)
    templateIndex = templateModel.GetPointData().GetArray(indexArrayName)
    # saving point correspondences
    if(templateIndex):
      sampleNumber = denseCorrespondenceGroup.GetNumberOfBlocks()
      print("sample number:", sampleNumber)
      for i in range(sampleNumber):
        alignedMesh = denseCorrespondenceGroup.GetBlock(i)
        alignedPointNode= slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsFiducialNode',"alignedPoints")
        for j in range(templateIndex.GetNumberOfValues()):
          baseIndex = templateIndex.GetValue(j)
          alignedPoint = alignedMesh.GetPoint(baseIndex)
          alignedPointNode.AddControlPoint(alignedPoint, str(j))
        outputLMPath = os.path.join(outputDirectory, self.modelNames[i]+".mrk.json")
        slicer.util.saveNode(alignedPointNode, outputLMPath)
        slicer.mrmlScene.RemoveNode(alignedPointNode)
      # save base node correspondences
      basePointNode= slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsFiducialNode',"atlasLandmarks")
      for j in range(templateIndex.GetNumberOfValues()):
        baseIndex = templateIndex.GetValue(j)
        basePoint = baseNode.GetPolyData().GetPoint(baseIndex)
        basePointNode.AddControlPoint(basePoint, str(j))
      baseLMPath = os.path.join(outputDirectory, "atlas.mrk.json")
      slicer.util.saveNode(basePointNode, baseLMPath)
      #slicer.mrmlScene.RemoveNode(basePointNode)
      return basePointNode
    else:
      print("No index found")
      return None

  # Use downsampleModel from ATLAS shape bridge
  def downsampleModel(self, model, spacingPercentage):
    if atlasShapeBridge:
      return atlasShapeBridge.downsampleModel(model, spacingPercentage)
    # Fallback implementation if ATLAS is not available
    points=model.GetPolyData()
    cleanFilter=vtk.vtkCleanPolyData()
    cleanFilter.SetToleranceIsAbsolute(False)
    cleanFilter.SetTolerance(spacingPercentage)
    cleanFilter.SetInputData(points)
    cleanFilter.Update()
    return cleanFilter.GetOutput()

  # Use addIndexArray from ATLAS shape bridge
  def addIndexArray(self, mesh, arrayName):
    if atlasShapeBridge:
      return atlasShapeBridge.addIndexArray(mesh, arrayName)
    # Fallback implementation if ATLAS is not available
    indexArray = vtk.vtkIntArray()
    indexArray.SetNumberOfComponents(1)
    indexArray.SetName(arrayName)
    for i in range(mesh.GetPolyData().GetNumberOfPoints()):
      indexArray.InsertNextValue(i)
    mesh.GetPolyData().GetPointData().AddArray(indexArray)

  # Use computeNormals from ATLAS shape bridge
  def computeNormals(self, inputModel):
    if atlasShapeBridge:
      return atlasShapeBridge.computeNormals(inputModel)
    # Fallback implementation if ATLAS is not available
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(inputModel.GetPolyData())
    normals.SetAutoOrientNormals(True)
    normals.Update()
    inputModel.SetAndObservePolyData(normals.GetOutput())

  def runMirroring(self, meshDirectory, lmDirectory, mirrorMeshDirectory, mirrorLMDirectory, mirrorAxis, mirrorIndexText, slmDirectory=None, outputSLMDirectory=None, mirrorSLMIndexText=None):
    mirrorMatrix = vtk.vtkMatrix4x4()
    mirrorMatrix.SetElement(0, 0, mirrorAxis[0])
    mirrorMatrix.SetElement(1, 1, mirrorAxis[1])
    mirrorMatrix.SetElement(2, 2, mirrorAxis[2])
    lmFileList = os.listdir(lmDirectory)
    point=[0,0,0]
    #get order of mirrored sets
    if len(mirrorIndexText) != 0:
      mirrorIndexList=mirrorIndexText.split(",")
      mirrorIndexList=[int(x) for x in mirrorIndexList]
      mirrorIndex=np.asarray(mirrorIndexList)
    else:
      print("Error: no landmark index for mirrored mesh")
    semilandmarkOption = bool(slmDirectory and outputSLMDirectory and (len(mirrorSLMIndexText) != 0))
    if semilandmarkOption:
      mirrorSLMIndexList=mirrorSLMIndexText.split(",")
      mirrorSLMIndexList=[int(x) for x in mirrorSLMIndexList]
      mirrorSLMIndex=np.asarray(mirrorSLMIndexList)
    for meshFileName in os.listdir(meshDirectory):
      if(not meshFileName.startswith(".")):
        meshFilePath = os.path.join(meshDirectory, meshFileName)
        # RealityCapture meshes are written in LPS, matching the coordinate
        # system declared by their source .mrk.json files.  Keep both inputs
        # in the same space before mirroring/alignment.
        currentMeshNode = self._load_model_with_cs(meshFilePath, 'LPS')
        # Extract subject ID by removing only the first extension (e.g., .obj from .obj.rcInfo)
        name_parts = meshFileName.split('.', 1)
        subjectID = name_parts[0] if len(name_parts) > 1 else meshFileName
        currentLMNode = self.getLandmarkFileByID(lmDirectory, subjectID)
        if currentLMNode:
          lmFilePath = os.path.join(lmDirectory, subjectID)
          targetPoints = vtk.vtkPoints()
          for i in range(currentLMNode.GetNumberOfControlPoints()):
            point = currentLMNode.GetNthControlPointPosition(i)
            targetPoints.InsertNextPoint(point)
          mirrorTransform = vtk.vtkTransform()
          mirrorTransform.SetMatrix(mirrorMatrix)
          mirrorTransformNode=slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTransformNode","Mirror")
          mirrorTransformNode.SetAndObserveTransformToParent(mirrorTransform)
          # apply transform to the current surface mesh and landmarks
          currentMeshNode.SetAndObserveTransformNodeID(mirrorTransformNode.GetID())
          currentLMNode.SetAndObserveTransformNodeID(mirrorTransformNode.GetID())
          slicer.vtkSlicerTransformLogic().hardenTransform(currentMeshNode)
          slicer.vtkSlicerTransformLogic().hardenTransform(currentLMNode)
          # apply rigid transformation
          sourcePoints = vtk.vtkPoints()
          mirrorLMNode =slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode",subjectID)
          for i in range(currentLMNode.GetNumberOfControlPoints()):
            point = currentLMNode.GetNthControlPointPosition(mirrorIndex[i])
            mirrorLMNode.AddControlPoint(point, str(i))
            sourcePoints.InsertNextPoint(point)
          rigidTransform = vtk.vtkLandmarkTransform()
          rigidTransform.SetSourceLandmarks(sourcePoints)
          rigidTransform.SetTargetLandmarks(targetPoints)
          rigidTransform.SetModeToRigidBody()
          rigidTransformNode=slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTransformNode","Rigid")
          rigidTransformNode.SetAndObserveTransformToParent(rigidTransform)
          # compute normals
          self.computeNormals(currentMeshNode)
          currentMeshNode.SetAndObserveTransformNodeID(rigidTransformNode.GetID())
          mirrorLMNode.SetAndObserveTransformNodeID(rigidTransformNode.GetID())
          slicer.vtkSlicerTransformLogic().hardenTransform(currentMeshNode)
          slicer.vtkSlicerTransformLogic().hardenTransform(mirrorLMNode)
          # optional semi-landmark alignment
          if semilandmarkOption:
            currentSLMNode = self.getLandmarkFileByID(slmDirectory, subjectID)
            mirrorSLMNode =slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode",subjectID)
            for i in range(currentSLMNode.GetNumberOfControlPoints()):
              point = currentSLMNode.GetNthControlPointPosition(mirrorSLMIndex[i])
              mirrorSLMNode.AddControlPoint(point, str(i))
            if currentSLMNode :
              currentSLMNode.SetAndObserveTransformNodeID(mirrorTransformNode.GetID())
              slicer.vtkSlicerTransformLogic().hardenTransform(currentSLMNode)
              currentSLMNode.SetAndObserveTransformNodeID(rigidTransformNode.GetID())
              slicer.vtkSlicerTransformLogic().hardenTransform(currentSLMNode)
              outputSLMName = subjectID + '_mirror.mrk.json'
              outputSLMPath = os.path.join(outputSLMDirectory, outputSLMName)
              slicer.util.saveNode(mirrorSLMNode, outputSLMPath)
              slicer.mrmlScene.RemoveNode(currentSLMNode)
              slicer.mrmlScene.RemoveNode(mirrorSLMNode)
          # save output files
          outputMeshName = subjectID + '_mirror.ply'
          outputMeshPath = os.path.join(mirrorMeshDirectory, outputMeshName)
          # Mirrored meshes are pipeline intermediates.  Write them
          # explicitly as RAS because importMeshes() reads generated meshes
          # in RAS, while the original RealityCapture inputs above are LPS.
          self._save_model_with_cs(currentMeshNode, outputMeshPath, 'RAS')
          outputLMName = subjectID + '_mirror.mrk.json'
          outputLMPath = os.path.join(mirrorLMDirectory, outputLMName)
          slicer.util.saveNode(mirrorLMNode, outputLMPath)
          # clean up
          slicer.mrmlScene.RemoveNode(currentLMNode)
          slicer.mrmlScene.RemoveNode(currentMeshNode)
          slicer.mrmlScene.RemoveNode(mirrorTransformNode)
          slicer.mrmlScene.RemoveNode(rigidTransformNode)
          slicer.mrmlScene.RemoveNode(mirrorLMNode)

  def runDCAlign(self, baseMeshPath, baseLMPath, alignedMeshDir, landmarkDirectory, outputDirectory, optionErrorOutput,
                atlas_uv_template_obj=None):

    baseNode = self._load_model_with_cs(baseMeshPath, 'RAS')
    baseMesh = baseNode.GetPolyData()
    baseLandmarks = self.fiducialNodeToPolyData(baseLMPath).GetPoints()

    # --- IMPORTANT: load landmarks first, then meshes for exactly those subjects ---
    landmarkNames, landmarks = self.importLandmarks(landmarkDirectory)
    self.modelNames, models = self.importMeshes(alignedMeshDir, ['ply','stl','vtp','obj'], restrict_to=landmarkNames)

    # sanity check
    if len(self.modelNames) != len(landmarkNames):
      missing_mesh = [n for n in landmarkNames if n not in self.modelNames]
      extra_mesh   = [n for n in self.modelNames if n not in landmarkNames]
      raise ValueError(f"Mismatch between meshes and landmarks.\n"
                      f"Missing mesh for: {missing_mesh}\nExtra mesh: {extra_mesh}")

    denseCorrespondenceGroup = self.denseCorrespondenceBaseMesh(landmarks, models, baseMesh, baseLandmarks)

    # Save the result model before removing baseNode from scene
    self.addMagnitudeFeature(denseCorrespondenceGroup, self.modelNames, baseMesh)
    outputModelName = 'atlasResultModel.vtp'
    outputModelPath = os.path.join(outputDirectory, "ATLAS", outputModelName)

    # Save the result model with error handling
    try:
      if baseNode and slicer.mrmlScene.IsNodePresent(baseNode):
        slicer.util.saveNode(baseNode, outputModelPath)
        print(f"Successfully saved ATLAS result model to: {outputModelPath}")
      else:
        print(f"Warning: baseNode is not valid or not in scene, skipping save to {outputModelPath}")
    except Exception as e:
      print(f"Warning: Failed to save ATLAS result model to {outputModelPath}: {e}")

    # Now remove baseNode from scene
    slicer.mrmlScene.RemoveNode(baseNode)

    #  Save resampled models (VTK/PLY) and OBJ copies that reuse atlas UV (for Blender bake)
    resampledModelPath = os.path.join(outputDirectory, "ATLAS", "resampledModels")
    if os.path.exists(resampledModelPath):
      tempModelNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", "tempResampledModel")
      outOBJdir = os.path.join(outputDirectory, "colorAnalysis", "resampledOBJ_withUV")
      os.makedirs(outOBJdir, exist_ok=True)

      for i in range(denseCorrespondenceGroup.GetNumberOfBlocks()):
        resampledMesh = denseCorrespondenceGroup.GetBlock(i)
        subjectName = self.modelNames[i].replace('_align', '')

        # Save VTK/PLY for Slicer
        outputFileName = os.path.join(resampledModelPath, f"{subjectName}_resampled.ply")
        tempModelNode.SetAndObservePolyData(resampledMesh)
        self._save_model_with_cs(tempModelNode, outputFileName, 'RAS')

        # Also write an OBJ that reuses atlas UV (for Blender bake)
        if atlas_uv_template_obj and os.path.isfile(atlas_uv_template_obj):
          out_obj = os.path.join(outOBJdir, f"{subjectName}_resampled.obj")
          self.write_obj_with_uv_from_template(resampledMesh, atlas_uv_template_obj, out_obj)

      slicer.mrmlScene.RemoveNode(tempModelNode)



    # Clean up the temporary base node now that we're done with it
    try:
      if baseNode and slicer.mrmlScene.IsNodePresent(baseNode):
        slicer.mrmlScene.RemoveNode(baseNode)
    except Exception as e:
      print(f"Warning: Failed to remove baseNode: {e}")

  def runDCAlignSymmetric(self, baseMeshPath, baseLMPath, meshDir, landmarkDir, mirrorMeshDir, mirrorLandmarkDir, outputDir, optionErrorOutput):
    baseNode = self._load_model_with_cs(baseMeshPath, 'RAS')
    baseMesh = baseNode.GetPolyData()
    baseLandmarks=self.fiducialNodeToPolyData(baseLMPath).GetPoints()
    modelExt=['ply','stl','vtp', 'obj']
    # In the symmetry workflow meshDir may be the original RealityCapture
    # collection (LPS), whereas mirrorMeshDir contains generated RAS output.
    self.modelNames, models = self.importMeshes(meshDir, modelExt,
                                                coordinateSystem='LPS')
    landmarkNames, landmarks = self.importLandmarks(landmarkDir)
    modelMirrorNames, mirrorModels = self.importMeshes(mirrorMeshDir, modelExt,
                                                       coordinateSystem='RAS')
    mirrorLandmarkNames, mirrorLandmarks = self.importLandmarks(mirrorLandmarkDir)
    denseCorrespondenceGroup = self.denseCorrespondenceBaseMesh(landmarks, models, baseMesh, baseLandmarks)
    denseCorrespondenceGroupMirror = self.denseCorrespondenceBaseMesh(mirrorLandmarks, mirrorModels, baseMesh, baseLandmarks)
    self.addMagnitudeFeatureSymmetry(denseCorrespondenceGroup, denseCorrespondenceGroupMirror, self.modelNames, baseMesh)
    # save results to output directory
    outputModelName = 'atlasSymmetryResultModel.vtp'
    outputModelPath = os.path.join(outputDir, outputModelName)
    slicer.util.saveNode(baseNode, outputModelPath)

    # Clean up the temporary base node
    slicer.mrmlScene.RemoveNode(baseNode)

  def runMean(self, landmarkDirectory, meshDirectory):
    landmarkNames, landmarks = self.importLandmarks(landmarkDirectory)
    self.modelNames, models = self.importMeshes(meshDirectory, ['ply','stl','vtp','vtk','obj'], restrict_to=landmarkNames)
    [denseCorrespondenceGroup, closestToMeanIndex] = self.denseCorrespondence(landmarks, models)
    print("Sample closest to mean: ", closestToMeanIndex)
    # compute mean model
    averagePolyData = self.computeAverageModelFromGroup(denseCorrespondenceGroup, closestToMeanIndex)
    averageModelNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLModelNode', 'Atlas Model')
    averageModelNode.CreateDefaultDisplayNodes()
    averageModelNode.SetAndObservePolyData(averagePolyData)
     # compute mean landmarks
    averageLandmarkNode = self.computeAverageLM(landmarks)
    averageLandmarkNode.GetDisplayNode().SetPointLabelsVisibility(False)
    return averageModelNode, averageLandmarkNode

  # Use getLandmarkFileByID from ATLAS shape bridge
  def getLandmarkFileByID(self, directory, subjectID):
    if atlasShapeBridge:
      try:
        return atlasShapeBridge.getLandmarkFileByID(directory, subjectID)
      except Exception as e:
        print(f"Error using atlasShapeBridge.getLandmarkFileByID: {e}")
        # Fall back to local implementation
    # Fallback implementation if ATLAS is not available
    fileList = os.listdir(directory)
    for fileName in fileList:
      if not _is_visible_dataset_file(fileName):
        continue
      fileNameBase = Path(fileName)
      while fileNameBase.suffix in {'.fcsv', '.mrk', '.json'}:
        fileNameBase = fileNameBase.with_suffix('')
      if subjectID == str(fileNameBase):
        # if file with this subject id exists, load into scene
        filePath = os.path.join(directory, fileName)
        try:
          currentNode = slicer.util.loadMarkups(filePath)
          return currentNode
        except Exception as e:
          print(f"Error loading landmarks from {filePath}: {e}")
          return None
    print(f"No landmarks found for subject ID '{subjectID}' in {directory}")
    return None

  def getModelFileByID(self, directory, subjectID):
    fileList = os.listdir(directory)
    # Only process files with valid model extensions
    model_extensions = ['.ply', '.stl', '.obj', '.vtk', '.vtp']
    for fileName in fileList:
      if not _is_visible_dataset_file(fileName):
        continue
      # Get the first extension only (e.g., .obj from .obj.rcInfo)
      name_parts = fileName.split('.', 1)
      if len(name_parts) < 2:
        continue  # No extension found
      first_ext = '.' + name_parts[1].split('.')[0]
      file_ext = first_ext.lower()
      # Skip non-model files (e.g., .mtl)
      if file_ext not in model_extensions:
        continue
      # Get the base name by removing only the first extension
      fileNameBase = name_parts[0]
      # Check for partial match: filename base should start with the subject ID
      if str(fileNameBase).startswith(str(subjectID)):
        filePath = os.path.join(directory, fileName)
        try:
          # This lookup is used while building an atlas from the original
          # RealityCapture files, not for the RAS intermediates produced by
          # runAlign().
          currentNode = self._load_model_with_cs(filePath, 'LPS')
          return currentNode
        except Exception as e:
          print(f"Error loading model from {filePath}: {e}")
          return None
    print(f"No model found for subject ID '{subjectID}' in {directory}")
    return None

  def runAlign(self, baseMeshNode, baseLMNode, meshDirectory, lmDirectory, ouputMeshDirectory, outputLMDirectory, removeScaleOption, slmDirectory=False, outputSLMDirectory=False):
    semilandmarkOption = bool(slmDirectory and outputSLMDirectory)
    self._validate_mesh_landmarks(baseMeshNode, baseLMNode,
                                  context="atlas template mesh")
    targetPoints = vtk.vtkPoints()
    point=[0,0,0]
    # Set up base points for transform
    for i in range(baseLMNode.GetNumberOfControlPoints()):
      point = baseLMNode.GetNthControlPointPosition(i)
      targetPoints.InsertNextPoint(point)
    # Transform each subject to base
    for meshFileName in os.listdir(meshDirectory):
      if(not meshFileName.startswith(".")):
        lmFileList = os.listdir(lmDirectory)
        meshFilePath = os.path.join(meshDirectory, meshFileName)
        # Extract subject ID by removing only the first extension (e.g., .obj from .obj.rcInfo)
        name_parts = meshFileName.split('.', 1)
        subjectID = name_parts[0] if len(name_parts) > 1 else meshFileName
        currentLMNode = self.getLandmarkFileByID(lmDirectory, subjectID)
        if currentLMNode :
          try:
            # The source mesh and source markups are both in LPS.  Aligned
            # outputs are saved explicitly as RAS and are loaded as RAS by
            # importMeshes() later in the pipeline.
            currentMeshNode = self._load_model_with_cs(meshFilePath, 'LPS')
          except:
            slicer.mrmlScene.RemoveNode(currentLMNode)
            continue
          if currentLMNode.GetNumberOfControlPoints() != baseLMNode.GetNumberOfControlPoints():
            raise ValueError(f"Landmark points mismatch: subject has {currentLMNode.GetNumberOfControlPoints()} points, "
              f"atlas has {baseLMNode.GetNumberOfControlPoints()} points")
          self._validate_mesh_landmarks(currentMeshNode, currentLMNode,
                                        context=f"{subjectID} source mesh")
          # set up transform between base lms and current lms
          sourcePoints = vtk.vtkPoints()
          for i in range(currentLMNode.GetNumberOfControlPoints()):
            point = currentLMNode.GetNthControlPointPosition(i)
            sourcePoints.InsertNextPoint(point)
          transform = vtk.vtkLandmarkTransform()
          transform.SetSourceLandmarks(sourcePoints)
          transform.SetTargetLandmarks(targetPoints)
          transform.SetModeToRigidBody()

          transformNode=slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTransformNode","Alignment")
          transformNode.SetAndObserveTransformToParent(transform)
          # apply transform to the current surface mesh and landmarks
          currentMeshNode.SetAndObserveTransformNodeID(transformNode.GetID())
          currentLMNode.SetAndObserveTransformNodeID(transformNode.GetID())
          slicer.vtkSlicerTransformLogic().hardenTransform(currentMeshNode)
          slicer.vtkSlicerTransformLogic().hardenTransform(currentLMNode)
          # save output files
          outputMeshName = subjectID + '_align.ply'
          outputMeshPath = os.path.join(ouputMeshDirectory, outputMeshName)
          self._save_model_with_cs(currentMeshNode, outputMeshPath, 'RAS')

          # NEW: also write an OBJ copy so Blender can use the subject UVs
          try:
            tc = currentMeshNode.GetPolyData().GetPointData().GetTCoords()
            has_uv = bool(tc) and tc.GetNumberOfTuples() > 0
            if has_uv or os.path.splitext(meshFilePath)[1].lower() == '.obj':
              outputOBJPath = os.path.join(ouputMeshDirectory, subjectID + '_align.obj')
              self._save_model_with_cs(currentMeshNode, outputOBJPath, 'RAS')
          except Exception as e:
            logging.warning(f"Could not save aligned OBJ for {subjectID}: {e}")

          outputLMName = subjectID + '_align.mrk.json'
          outputLMPath = os.path.join(outputLMDirectory, outputLMName)
          slicer.util.saveNode(currentLMNode, outputLMPath)
          # optional semi-landmark alignment
          if semilandmarkOption :
            currentSLMNode = self.getLandmarkFileByID(slmDirectory, subjectID)
            if currentSLMNode :
              currentSLMNode.SetAndObserveTransformNodeID(transformNode.GetID())
              slicer.vtkSlicerTransformLogic().hardenTransform(currentSLMNode)
              outputSLMName = subjectID + '_align.mrk.json'
              outputSLMPath = os.path.join(outputSLMDirectory, outputSLMName)
              slicer.util.saveNode(currentSLMNode, outputSLMPath)
              slicer.mrmlScene.RemoveNode(currentSLMNode)
          # clean up
          try:
            slicer.mrmlScene.RemoveNode(currentLMNode)
            slicer.mrmlScene.RemoveNode(currentMeshNode)
            slicer.mrmlScene.RemoveNode(transformNode)
          except:
            print(f"could not find nodes to remove for {subjectID}")

  # Use distanceMatrix from ATLAS shape bridge
  def distanceMatrix(self, a):
    if atlasShapeBridge:
      return atlasShapeBridge.distanceMatrix(a)
    # Fallback implementation if ATLAS is not available
    """
    Computes the euclidean distance matrix for n points in a 3D space
    Returns a nXn matrix
     """
    id,jd=a.shape
    fnx = lambda q : q - np.reshape(q, (id, 1))
    dx=fnx(a[:,0])
    dy=fnx(a[:,1])
    dz=fnx(a[:,2])
    return (dx**2.0+dy**2.0+dz**2.0)**0.5

  # Use numpyToFiducialNode from ATLAS shape bridge
  def numpyToFiducialNode(self, numpyArray, nodeName):
    if atlasShapeBridge:
      return atlasShapeBridge.numpyToFiducialNode(numpyArray, nodeName)
    # Fallback implementation if ATLAS is not available
    fiducialNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsFiducialNode',nodeName)
    for index in range(len(numpyArray)):
      fiducialNode.AddControlPoint(numpyArray[index], str(index))
    return fiducialNode

  # Use computeAverageLM from ATLAS shape bridge
  def computeAverageLM(self, fiducialGroup):
    if atlasShapeBridge:
      return atlasShapeBridge.computeAverageLM(fiducialGroup)
    # Fallback implementation if ATLAS is not available
    sampleNumber = fiducialGroup.GetNumberOfBlocks()
    pointNumber = fiducialGroup.GetBlock(0).GetNumberOfPoints()
    groupArray_np = np.empty((pointNumber,3,sampleNumber))
    for i in range(sampleNumber):
      pointData = fiducialGroup.GetBlock(i).GetPoints().GetData()
      pointData_np = vtk_np.vtk_to_numpy(pointData)
      groupArray_np[:,:,i] = pointData_np
    #Calculate mean point positions of aligned group
    averagePoints_np = np.mean(groupArray_np, axis=2)
    averageLMNode = self.numpyToFiducialNode(averagePoints_np, "Atlas Landmarks")
    return averageLMNode

  # Use fiducialNodeToPolyData from ATLAS shape bridge
  def fiducialNodeToPolyData(self, nodeLocation, loadOption=True):
    if atlasShapeBridge:
      return atlasShapeBridge.fiducialNodeToPolyData(nodeLocation, loadOption)
    # Fallback implementation if ATLAS is not available
    point = [0,0,0]
    polydataPoints = vtk.vtkPolyData()
    points = vtk.vtkPoints()
    if not loadOption:
      fiducialNode = nodeLocation
    else:
      [success,fiducialNode] = slicer.util.loadMarkupsFiducialList(nodeLocation)
      if not success:
        print("Could not load landmarks: ", nodeLocation)
        return
    for i in range(fiducialNode.GetNumberOfControlPoints()):
      point = fiducialNode.GetNthControlPointPosition(i)
      points.InsertNextPoint(point)
    polydataPoints.SetPoints(points)
    slicer.mrmlScene.RemoveNode(fiducialNode)
    return polydataPoints

  def importLandmarks(self, topDir):
    # one landmark file per subject, returned in deterministic (sorted) order
    prefer = ['.mrk.json', '.json', '.fcsv']  # priority
    pick = {}  # base -> (rank, fullpath)

    for f in os.listdir(topDir):
      if not _is_visible_dataset_file(f):
        continue
      fl = f.lower()
      if fl.endswith(tuple(prefer)):
        p = Path(f)
        base = p
        # strip .mrk.json / .json / .fcsv
        while base.suffix.lower() in ('.mrk', '.json', '.fcsv'):
          base = base.with_suffix('')
        base = base.name  # e.g. 'Subject01_align'
        rank = 0 if fl.endswith('.mrk.json') else (1 if fl.endswith('.json') else 2)
        if base not in pick or rank < pick[base][0]:
          pick[base] = (rank, os.path.join(topDir, f))

    names = sorted(pick.keys())
    group = vtk.vtkMultiBlockDataGroupFilter()
    for name in names:
      polydataPoints = self.fiducialNodeToPolyData(pick[name][1])
      group.AddInputData(polydataPoints)
    group.Update()
    return names, group.GetOutput()


  def importMeshes(self, topDir, extensions, restrict_to=None,
                   coordinateSystem='RAS'):
    # choose exactly one mesh per subject, preferring OBJ over PLY/STL/VTP/VTK
    # Generated intermediates are RAS by default; raw source collections can
    # opt into LPS at the call site.
    priority = {'.obj':0, '.ply':1, '.stl':2, '.vtp':3, '.vtk':4}
    pick = {}  # base -> (rank, fullpath)

    for f in os.listdir(topDir):
      if not _is_visible_dataset_file(f):
        continue
      ext = os.path.splitext(f)[1].lower()
      if ext in priority:
        base = os.path.splitext(f)[0]  # e.g. 'Subject01_align'
        if (restrict_to is None) or (base in restrict_to):
          if base not in pick or priority[ext] < pick[base][0]:
            pick[base] = (priority[ext], os.path.join(topDir, f))

    if restrict_to is not None:
      names = [b for b in restrict_to if b in pick]  # keep same order as landmarks
    else:
      names = sorted(pick.keys())

    modelGroup = vtk.vtkMultiBlockDataGroupFilter()
    for b in names:
      inputFilePath = pick[b][1]
      modelNode = self._load_model_with_cs(inputFilePath, coordinateSystem)
      modelGroup.AddInputData(modelNode.GetPolyData())
      slicer.mrmlScene.RemoveNode(modelNode)
    modelGroup.Update()
    return names, modelGroup.GetOutput()

  # Use procrustesImposition from ATLAS shape bridge
  def procrustesImposition(self, originalLandmarks, sizeOption):
    if atlasShapeBridge:
      return atlasShapeBridge.procrustesImposition(originalLandmarks, sizeOption)
    # Fallback implementation if ATLAS is not available
    procrustesFilter = vtk.vtkProcrustesAlignmentFilter()
    if(sizeOption):
      procrustesFilter.GetLandmarkTransform().SetModeToRigidBody()

    procrustesFilter.SetInputData(originalLandmarks)
    procrustesFilter.Update()
    meanShape = procrustesFilter.GetMeanPoints()
    return [meanShape, procrustesFilter.GetOutput()]

  # Use getClosestToMeanIndex from ATLAS shape bridge
  def getClosestToMeanIndex(self, meanShape, alignedPoints):
    if atlasShapeBridge:
      return atlasShapeBridge.getClosestToMeanIndex(meanShape, alignedPoints)
    # Fallback implementation if ATLAS is not available
    import operator
    sampleNumber = alignedPoints.GetNumberOfBlocks()
    procrustesDistances = []
    for i in range(sampleNumber):
      alignedShape = alignedPoints.GetBlock(i)
      meanPoint = [0,0,0]
      alignedPoint = [0,0,0]
      distance = 0
      for j in range(meanShape.GetNumberOfPoints()):
        meanShape.GetPoint(j,meanPoint)
        alignedShape.GetPoint(j,alignedPoint)
        distance += np.sqrt(vtk.vtkMath.Distance2BetweenPoints(meanPoint,alignedPoint))
      procrustesDistances.append(distance)
    try:
      min_index, min_value = min(enumerate(procrustesDistances), key=operator.itemgetter(1))
      return min_index
    except:
      return 0

  # Use getClosestToMeanPath from ATLAS shape bridge
  def getClosestToMeanPath(self, landmarkDirectory):
    if atlasShapeBridge:
      try:
        return atlasShapeBridge.getClosestToMeanPath(landmarkDirectory)
      except Exception as e:
        print(f"Error using atlasShapeBridge.getClosestToMeanPath: {e}")
        # Fall back to local implementation
    # Fallback implementation if ATLAS is not available
    lmNames, landmarks = self.importLandmarks(landmarkDirectory)
    if not lmNames:
      print(f"No landmarks found in {landmarkDirectory}")
      return None
    meanShape, alignedLandmarks = self.procrustesImposition(landmarks, False)
    closestToMeanIndex = self.getClosestToMeanIndex(meanShape, alignedLandmarks)
    return lmNames[closestToMeanIndex]

  def denseCorrespondence(self, originalLandmarks, originalMeshes):
    meanShape, alignedPoints = self.procrustesImposition(originalLandmarks, False)
    sampleNumber = alignedPoints.GetNumberOfBlocks()
    denseCorrespondenceGroup = vtk.vtkMultiBlockDataGroupFilter()
    # get base mesh as the closest to the mean shape
    baseIndex = self.getClosestToMeanIndex(meanShape, alignedPoints)
    baseMesh = originalMeshes.GetBlock(baseIndex)
    baseLandmarks = originalLandmarks.GetBlock(baseIndex).GetPoints()
    for i in range(sampleNumber):
      correspondingMesh = self.denseSurfaceCorrespondencePair(originalMeshes.GetBlock(i),
      originalLandmarks.GetBlock(i).GetPoints(), alignedPoints.GetBlock(i).GetPoints(),
      baseMesh, baseLandmarks, meanShape, i)
      denseCorrespondenceGroup.AddInputData(correspondingMesh)

    denseCorrespondenceGroup.Update()
    return denseCorrespondenceGroup.GetOutput(), baseIndex

  def denseCorrespondenceCPD(self, originalLandmarks, originalMeshes, baseMesh, baseLandmarks):
    meanShape, alignedPoints = self.procrustesImposition(originalLandmarks, False)
    sampleNumber = alignedPoints.GetNumberOfBlocks()
    denseCorrespondenceGroup = vtk.vtkMultiBlockDataGroupFilter()

    # assign parameters for CPD
    parameters = {
      "SpacingTolerance": .04,
      "CPDIterations": 100,
      "CPDTolerence": 0.001,
      "alpha": 2,
      "beta": 2,
     }

    for i in range(sampleNumber):
      correspondingPoints = self.runCPDRegistration(originalMeshes.GetBlock(i), baseMesh, parameters)
      # convert to vtkPoints
      correspondingMesh = self.convertPointsToVTK(correspondingPoints)
      correspondingMesh.SetPolys(baseMesh.GetPolys())
      # convert to polydata
      denseCorrespondenceGroup.AddInputData(correspondingMesh)

    denseCorrespondenceGroup.Update()
    return denseCorrespondenceGroup.GetOutput()

  def denseCorrespondenceBaseMesh(self, originalLandmarks, originalMeshes, baseMesh, baseLandmarks):
    meanShape, alignedPoints = self.procrustesImposition(originalLandmarks, False)
    sampleNumber = alignedPoints.GetNumberOfBlocks()
    print("procrustes aligned samples: ", sampleNumber)
    denseCorrespondenceGroup = vtk.vtkMultiBlockDataGroupFilter()
    for i in range(sampleNumber):
      correspondingMesh = self.denseSurfaceCorrespondencePair(originalMeshes.GetBlock(i),
      originalLandmarks.GetBlock(i).GetPoints(), alignedPoints.GetBlock(i).GetPoints(),
      baseMesh, baseLandmarks, meanShape, i)
      denseCorrespondenceGroup.AddInputData(correspondingMesh)
    denseCorrespondenceGroup.Update()
    return denseCorrespondenceGroup.GetOutput()

  def denseSurfaceCorrespondencePair(self, originalMesh, originalLandmarks, alignedLandmarks, baseMesh, baseLandmarks, meanShape, iteration):
    # TPS warp target and base mesh to meanshape
    meanTransform = vtk.vtkThinPlateSplineTransform()
    meanTransform.SetSourceLandmarks(originalLandmarks)
    meanTransform.SetTargetLandmarks(meanShape)
    meanTransform.SetBasisToR() # for 3D transform

    meanTransformFilter = vtk.vtkTransformPolyDataFilter()
    if originalMesh and originalMesh.GetNumberOfPoints() > 0:
      meanTransformFilter.SetInputData(originalMesh)
      meanTransformFilter.SetTransform(meanTransform)
      meanTransformFilter.Update()
      meanWarpedMesh = meanTransformFilter.GetOutput()
    else:
      print(f"Warning: Empty or invalid originalMesh for iteration {iteration}")
      return baseMesh

    meanTransformBase = vtk.vtkThinPlateSplineTransform()
    meanTransformBase.SetSourceLandmarks(baseLandmarks)
    meanTransformBase.SetTargetLandmarks(meanShape)
    meanTransformBase.SetBasisToR() # for 3D transform

    meanTransformBaseFilter = vtk.vtkTransformPolyDataFilter()
    if baseMesh and baseMesh.GetNumberOfPoints() > 0:
      meanTransformBaseFilter.SetInputData(baseMesh)
      meanTransformBaseFilter.SetTransform(meanTransformBase)
      meanTransformBaseFilter.Update()
      meanWarpedBase = meanTransformBaseFilter.GetOutput() # Warped atlas
    else:
      print(f"Warning: Empty or invalid baseMesh for iteration {iteration}")
      return baseMesh

    # --- BEGIN INTEGRATED RESAMPLING AND UV TRANSFER ---

    # Check if the warped subject mesh has UVs to transfer
    warpedUVs = meanWarpedMesh.GetPointData().GetTCoords()
    if warpedUVs:
        print(f"UVs found for subject {self.modelNames[iteration]}, preparing for transfer.")
        newUVs = vtk.vtkFloatArray()
        newUVs.SetName("TransferredUVs")
        newUVs.SetNumberOfComponents(2)
        newUVs.SetNumberOfTuples(meanWarpedBase.GetNumberOfPoints())

    # Build search locator on the warped subject mesh
    cellLocator = vtk.vtkCellLocator()
    cellLocator.SetDataSet(meanWarpedMesh)
    cellLocator.BuildLocator()

    correspondingPoints = vtk.vtkPoints()
    for i in range(meanWarpedBase.GetNumberOfPoints()):
        point = meanWarpedBase.GetPoint(i)

        # Find the closest point on the warped SUBJECT's surface
        closestPoint, closestCellId, subId, dist2 = [0.0, 0.0, 0.0], vtk.reference(0), vtk.reference(0), vtk.reference(0.0)
        cellLocator.FindClosestPoint(point, closestPoint, closestCellId, subId, dist2)
        
        # This new point is the resampled POSITION
        correspondingPoints.InsertPoint(i, closestPoint)

        # If we have UVs, calculate the resampled UV as well
        if warpedUVs:
            actualCellId = closestCellId.get()
            cell = meanWarpedMesh.GetCell(actualCellId)
            
            if cell and cell.GetNumberOfPoints() == 3:
                weights = [0.0] * cell.GetNumberOfPoints()
                closestPointOutput = [0.0, 0.0, 0.0]
                pcoords_ignored = [0.0, 0.0, 0.0]
                dist2_ignored = vtk.reference(0.0)
                cell.EvaluatePosition(closestPoint, closestPointOutput, subId, pcoords_ignored, dist2_ignored, weights)
                
                cellPointIds = cell.GetPointIds()
                uv0, uv1, uv2 = warpedUVs.GetTuple2(cellPointIds.GetId(0)), warpedUVs.GetTuple2(cellPointIds.GetId(1)), warpedUVs.GetTuple2(cellPointIds.GetId(2))
                
                u_new = weights[0] * uv0[0] + weights[1] * uv1[0] + weights[2] * uv2[0]
                v_new = weights[0] * uv0[1] + weights[1] * uv1[1] + weights[2] * uv2[1]
                newUVs.SetTuple2(i, u_new, v_new)
            else:
                newUVs.SetTuple2(i, 0.0, 0.0) # Set a default UV if something goes wrong

    # --- END INTEGRATED RESAMPLING AND UV TRANSFER ---

    # Assemble the new mesh with the new points and (if available) new UVs
    correspondingMesh = vtk.vtkPolyData()
    correspondingMesh.SetPoints(correspondingPoints)
    correspondingMesh.SetPolys(meanWarpedBase.GetPolys()) # Use ATLAS connectivity
    if warpedUVs:
        correspondingMesh.GetPointData().SetTCoords(newUVs)

    # Apply inverse warping
    inverseTransform = vtk.vtkThinPlateSplineTransform()
    inverseTransform.SetSourceLandmarks(meanShape)
    inverseTransform.SetTargetLandmarks(originalLandmarks)
    inverseTransform.SetBasisToR() # for 3D transform

    inverseTransformFilter = vtk.vtkTransformPolyDataFilter()
    if correspondingMesh and correspondingMesh.GetNumberOfPoints() > 0:
      inverseTransformFilter.SetInputData(correspondingMesh)
      inverseTransformFilter.SetTransform(inverseTransform)
      inverseTransformFilter.Update()
      return inverseTransformFilter.GetOutput()
    else:
      print(f"Warning: Empty correspondingMesh for iteration {iteration}, returning original")
      return baseMesh

  # Use convertPointsToVTK from ATLAS shape bridge
  def convertPointsToVTK(self, points):
    if atlasShapeBridge:
      return atlasShapeBridge.convertPointsToVTK(points)
    # Fallback implementation if ATLAS is not available
    array_vtk = vtk_np.numpy_to_vtk(points, deep=True, array_type=vtk.VTK_FLOAT)
    points_vtk = vtk.vtkPoints()
    points_vtk.SetData(array_vtk)
    polydata_vtk = vtk.vtkPolyData()
    polydata_vtk.SetPoints(points_vtk)
    return polydata_vtk

  # Use computeAverageModelFromGroup from ATLAS shape bridge
  def computeAverageModelFromGroup(self, denseCorrespondenceGroup, baseIndex):
    if atlasShapeBridge:
      return atlasShapeBridge.computeAverageModelFromGroup(denseCorrespondenceGroup, baseIndex)
    # Fallback implementation if ATLAS is not available
    sampleNumber = denseCorrespondenceGroup.GetNumberOfBlocks()
    pointNumber = denseCorrespondenceGroup.GetBlock(0).GetNumberOfPoints()
    groupArray_np = np.empty((pointNumber,3,sampleNumber))
    # get base mesh as closest to the meanshape
    baseMesh = denseCorrespondenceGroup.GetBlock(baseIndex)
     # get points as array
    for i in range(sampleNumber):
      alignedMesh = denseCorrespondenceGroup.GetBlock(i)
      alignedMesh_np = vtk_np.vtk_to_numpy(alignedMesh.GetPoints().GetData())
      groupArray_np[:,:,i] = alignedMesh_np
    #Calculate mean point positions of aligned group
    averagePoints_np = np.mean(groupArray_np, axis=2)
    averagePointsPolydata = self.convertPointsToVTK(averagePoints_np)
    #Copy points into mesh with base connectivity
    averageModel = vtk.vtkPolyData()
    averageModel.SetPoints(averagePointsPolydata.GetPoints())
    averageModel.SetPolys(baseMesh.GetPolys())
    return averageModel

  # Use addMagnitudeFeature from ATLAS shape bridge
  def addMagnitudeFeature(self, denseCorrespondenceGroup, modelNameArray, model):
    if atlasShapeBridge:
      return atlasShapeBridge.addMagnitudeFeature(denseCorrespondenceGroup, modelNameArray, model)
    # Fallback implementation if ATLAS is not available
    sampleNumber = denseCorrespondenceGroup.GetNumberOfBlocks()
    pointNumber = denseCorrespondenceGroup.GetBlock(0).GetNumberOfPoints()
    statsArray = np.zeros((pointNumber, sampleNumber))
    magnitudeMean = vtk.vtkDoubleArray()
    magnitudeMean.SetNumberOfComponents(1)
    magnitudeMean.SetName("Magnitude Mean")
    magnitudeSD = vtk.vtkDoubleArray()
    magnitudeSD.SetNumberOfComponents(1)
    magnitudeSD.SetName("Magnitude SD")

     # get distance arrays
    for i in range(sampleNumber):
      alignedMesh = denseCorrespondenceGroup.GetBlock(i)
      magnitudes = vtk.vtkDoubleArray()
      magnitudes.SetNumberOfComponents(1)
      magnitudes.SetName(modelNameArray[i])
      for j in range(pointNumber):
        modelPoint = model.GetPoint(j)
        targetPoint = alignedMesh.GetPoint(j)
        distance = np.sqrt(vtk.vtkMath.Distance2BetweenPoints(modelPoint,targetPoint))
        magnitudes.InsertNextValue(distance)
        statsArray[j,i]=distance

      model.GetPointData().AddArray(magnitudes)

    for i in range(pointNumber):
      pointMean = statsArray[i,:].mean()
      magnitudeMean.InsertNextValue(pointMean)
      pointSD = statsArray[i,:].std()
      magnitudeSD.InsertNextValue(pointSD)

    model.GetPointData().AddArray(magnitudeMean)
    model.GetPointData().AddArray(magnitudeSD)

  # Use addMagnitudeFeatureSymmetry from ATLAS shape bridge
  def addMagnitudeFeatureSymmetry(self, denseCorrespondenceGroup, denseCorrespondenceGroupMirror, modelNameArray, model):
    if atlasShapeBridge:
      return atlasShapeBridge.addMagnitudeFeatureSymmetry(denseCorrespondenceGroup, denseCorrespondenceGroupMirror, modelNameArray, model)
    # Fallback implementation if ATLAS is not available
    sampleNumber = denseCorrespondenceGroup.GetNumberOfBlocks()
    pointNumber = denseCorrespondenceGroup.GetBlock(0).GetNumberOfPoints()
    statsArray = np.zeros((pointNumber, sampleNumber))
    magnitudeMean = vtk.vtkDoubleArray()
    magnitudeMean.SetNumberOfComponents(1)
    magnitudeMean.SetName("Magnitude Mean")
    magnitudeSD = vtk.vtkDoubleArray()
    magnitudeSD.SetNumberOfComponents(1)
    magnitudeSD.SetName("Magnitude SD")

     # get distance arrays
    for i in range(sampleNumber):
      alignedMesh = denseCorrespondenceGroup.GetBlock(i)
      mirrorMesh = denseCorrespondenceGroupMirror.GetBlock(i)
      magnitudes = vtk.vtkDoubleArray()
      magnitudes.SetNumberOfComponents(1)
      magnitudes.SetName(modelNameArray[i])
      for j in range(pointNumber):
        modelPoint = model.GetPoint(j)
        targetPoint1 = alignedMesh.GetPoint(j)
        targetPoint2 = mirrorMesh.GetPoint(j)
        distance = np.sqrt(vtk.vtkMath.Distance2BetweenPoints(targetPoint1,targetPoint2))
        magnitudes.InsertNextValue(distance)
        statsArray[j,i]=distance

      model.GetPointData().AddArray(magnitudes)

    for i in range(pointNumber):
      pointMean = statsArray[i,:].mean()
      magnitudeMean.InsertNextValue(pointMean)
      pointSD = statsArray[i,:].std()
      magnitudeSD.InsertNextValue(pointSD)

    model.GetPointData().AddArray(magnitudeMean)
    model.GetPointData().AddArray(magnitudeSD)

  # ---------- Coordinate system safe save ----------
  def _save_model_with_cs(self, modelNode, filePath, coordinateSystem='RAS'):
    if modelNode is None:
      raise ValueError(f"Model node is None, cannot save to {filePath}")

    storage = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLModelStorageNode')
    storage.SetFileName(filePath)
    cs = (coordinateSystem or 'RAS').upper()
    try:
      if cs == 'RAS': storage.SetCoordinateSystemToRAS()
      else:           storage.SetCoordinateSystemToLPS()
    except AttributeError:
      storage.SetCoordinateSystem(0 if cs == 'RAS' else 1)  # older API
    modelNode.SetAndObserveStorageNodeID(storage.GetID())
    ok = storage.WriteData(modelNode)
    slicer.mrmlScene.RemoveNode(storage)
    if not ok:
      raise RuntimeError(f"Failed to write model: {filePath}")

  # ---------- Blender: atlas cleanup + Smart UV ----------
  def blender_prepare_atlas(self, blender_exe, in_obj, out_obj,
                            merge_dist=0.0005, smart_angle=66.0, island_margin=0.002):
    import tempfile, textwrap, subprocess, sys, os
    tmp = tempfile.mkdtemp(prefix="InterDeCA_blUV_")
    script = os.path.join(tmp, "prep_uv.py")
    py = textwrap.dedent(f"""
    import bpy, sys
    argv = sys.argv
    argv = argv[argv.index("--")+1:] if "--" in argv else []
    in_path  = argv[0]
    out_path = argv[1]
    merge_d  = float(argv[2])
    ang      = float(argv[3])
    island_m = float(argv[4])

    bpy.ops.wm.read_homefile(use_empty=True)

    # Import
    try:
        bpy.ops.wm.obj_import(filepath=in_path, forward_axis='Y', up_axis='Z')
    except AttributeError:
        bpy.ops.import_scene.obj(filepath=in_path, use_split_objects=False, use_split_groups=False, axis_forward='Y', axis_up='Z')

    obj = [o for o in bpy.context.selected_objects if o.type=='MESH'][0]
    bpy.context.view_layer.objects.active = obj

    # Edit mode ops
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    # Merge by distance (robust across Blender versions)
    try:
        bpy.ops.mesh.remove_doubles(threshold=merge_d)
    except Exception:
        try:
            bpy.ops.mesh.merge_by_distance(distance=merge_d)
        except Exception:
            bpy.ops.mesh.merge(type='DISTANCE', distance=merge_d)

    # Smart UV Project
    bpy.ops.uv.smart_project(angle_limit=ang, island_margin=island_m, correct_aspect=True, scale_to_bounds=False)
    bpy.ops.object.mode_set(mode='OBJECT')

    # Export
    bpy.ops.wm.obj_export(
        filepath=out_path,
        export_selected_objects=True,
        export_triangulated_mesh=True,
        forward_axis='Y', up_axis='Z',
        export_materials=False,
    )
    """)
    with open(script, "w", encoding="utf-8") as f: f.write(py)
    args = [blender_exe, "--background", "--python", script, "--",
            in_obj, out_obj, str(merge_dist), str(smart_angle), str(island_margin)]
    subprocess.run(args, check=True)

  # ---------- Copy atlas UVs onto resampled meshes (write OBJ) ----------
  def _parse_obj_vt_and_faces(self, obj_path):
    vt = []
    faces = []   # list of tokens strings 'v/vt[/vn]' per tri (preserve exactly)
    with open(obj_path, "r", encoding="utf-8", errors="ignore") as f:
      for line in f:
        if line.startswith('vt '):
          _, u, v, *rest = line.strip().split()
          vt.append((float(u), float(v)))
        elif line.startswith('f '):
          parts = line.strip().split()[1:]
          if len(parts) == 3:
            faces.append(parts)
          else:
            # ensure triangulated export upstream
            pass
    if not vt or not faces:
      raise RuntimeError("Atlas OBJ missing vt or triangulated f lines.")
    return vt, faces

  def write_obj_with_uv_from_template(self, polydata, atlas_obj_path, out_obj_path):
    # Load atlas OBJ through VTK so we get the same VTK point order/TCoords
    atlas_node = self._load_model_with_cs(atlas_obj_path, 'RAS')
    atlas_pd = atlas_node.GetPolyData()

    # Basic checks
    if not atlas_pd or not atlas_pd.GetPointData() or not atlas_pd.GetPointData().GetTCoords():
      slicer.mrmlScene.RemoveNode(atlas_node)
      raise RuntimeError("Atlas OBJ has no per-vertex texture coordinates (TCoords).")

    if polydata.GetNumberOfPoints() != atlas_pd.GetNumberOfPoints():
      slicer.mrmlScene.RemoveNode(atlas_node)
      raise RuntimeError(
        f"Point-count mismatch between resampled ({polydata.GetNumberOfPoints()}) "
        f"and atlas ({atlas_pd.GetNumberOfPoints()})."
      )

    # Copy UVs (TCoords) from atlas to the resampled mesh (match by vertex index)
    tc_copy = vtk.vtkFloatArray()
    tc_copy.DeepCopy(atlas_pd.GetPointData().GetTCoords())
    tc_copy.SetName("TCoords")  # ensure standard name
    polydata.GetPointData().SetTCoords(tc_copy)

    # Write OBJ using VTK’s connectivity (avoids index mismatch)
    w = vtk.vtkOBJWriter()
    w.SetFileName(out_obj_path)
    w.SetInputData(polydata)
    w.Update()
    w.Write()

    slicer.mrmlScene.RemoveNode(atlas_node)

  # ---------- Blender bake for all subjects ----------
  def blender_bake_all(self, blender_exe, alignedDir, resampledUVDir, texturesDir, outDir,
                       bake_size=2048, bake_extrusion=0.005, bake_margin_px=2, merge_dist=0.0005):
    import subprocess, tempfile, textwrap, os

    def norm_id(name): return re.sub(r'(_align|_resampled)$', '', os.path.splitext(name)[0], flags=re.IGNORECASE)
    pr = {'.obj':0, '.ply':1, '.stl':2, '.vtp':3, '.vtk':4}
    best = {}
    for f in os.listdir(alignedDir):
      ext = os.path.splitext(f)[1].lower()
      if ext in pr:
        sid = norm_id(f)
        if sid not in best or pr[ext] < best[sid][0]:
          best[sid] = (pr[ext], os.path.join(alignedDir, f))
    aligned = {sid: path for sid, (_, path) in best.items()}
    targets = {norm_id(f): os.path.join(resampledUVDir, f)
               for f in os.listdir(resampledUVDir) if f.lower().endswith('.obj')}
    textures = {os.path.splitext(f)[0].lower(): os.path.join(texturesDir, f)
                for f in os.listdir(texturesDir) if f.lower().endswith(('.png', '.tiff', '.tif'))}

    def find_tex(sid):
      # exact, case-insensitive, or startswith
      k = sid.lower()
      if k in textures: return textures[k]
      for key in textures:
        if key.startswith(k): return textures[key]
      return None

    baked = {}
    os.makedirs(outDir, exist_ok=True)

    # blender script (runs once per subject for simplicity)
    py = textwrap.dedent("""
    import bpy, sys, os
    argv = sys.argv
    argv = argv[argv.index("--")+1:] if "--" in argv else []
    src_path, tgt_path, png_in, png_out, sz, extru, margin, merge_d = argv
    sz = int(sz); extru = float(extru); margin = int(margin); merge_d = float(merge_d)

    bpy.ops.wm.read_homefile(use_empty=True)
    # Import target (resampled with UV)
    try: bpy.ops.wm.obj_import(filepath=tgt_path, forward_axis='Y', up_axis='Z')
    except AttributeError: bpy.ops.import_scene.obj(filepath=tgt_path, use_split_objects=False, use_split_groups=False, axis_forward='Y', axis_up='Z')
    tgt = [o for o in bpy.context.selected_objects if o.type=='MESH'][0]

    imp_ok = False
    try:
        bpy.ops.wm.obj_import(filepath=src_path, forward_axis='Y', up_axis='Z'); imp_ok=True
    except Exception: pass
    if not imp_ok:
        raise RuntimeError("Cannot import aligned mesh: " + src_path)
    src = [o for o in bpy.context.selected_objects if o.type=='MESH'][-1]

    # Clean target just in case Slicer wrote disjoint faces
    bpy.context.view_layer.objects.active = tgt
    bpy.ops.object.mode_set(mode='EDIT'); bpy.ops.mesh.select_all(action='SELECT')
    try: bpy.ops.mesh.remove_doubles(threshold=merge_d)
    except Exception:
        try: bpy.ops.mesh.merge_by_distance(distance=merge_d)
        except Exception: bpy.ops.mesh.merge(type='DISTANCE', distance=merge_d)
    bpy.ops.object.mode_set(mode='OBJECT')

    # Materials: clear & rebuild
    tgt.data.materials.clear(); src.data.materials.clear()
    m_src = bpy.data.materials.new("MatSrc"); m_src.use_nodes=True
    nt = m_src.node_tree; nodes = nt.nodes
    img_node = nodes.new('ShaderNodeTexImage'); img_node.image = bpy.data.images.load(png_in)
    bsdf = next(n for n in nodes if n.type=='BSDF_PRINCIPLED')
    nt.links.new(img_node.outputs['Color'], bsdf.inputs['Base Color'])
    src.data.materials.append(m_src)

    m_tgt = bpy.data.materials.new("MatTgt"); m_tgt.use_nodes=True
    nt2 = m_tgt.node_tree; nodes2 = nt2.nodes
    imgT = bpy.data.images.new("BakeTarget", width=sz, height=sz, alpha=False)
    img_node_t = nodes2.new('ShaderNodeTexImage'); img_node_t.image = imgT
    tgt.data.materials.append(m_tgt)

    # Select order: src (selected), tgt (active)
    bpy.ops.object.select_all(action='DESELECT')
    src.select_set(True); tgt.select_set(True)
    bpy.context.view_layer.objects.active = tgt

    # Must ensure the target image node is selected/active
    for n in nodes2: n.select = False
    nodes2.active = img_node_t; img_node_t.select = True

    # Scene bake settings
    scn = bpy.context.scene
    scn.render.engine = 'CYCLES'
    scn.cycles.device = 'CPU'
    b = scn.render.bake
    b.use_selected_to_active = True
    b.cage_extrusion = extru
    b.margin = margin
    b.use_pass_direct = False
    b.use_pass_indirect = False
    b.use_pass_color = True

    # Bake (Diffuse Color)
    bpy.ops.object.bake(type='DIFFUSE')

    # Save image
    imgT.filepath_raw = png_out
    imgT.file_format = 'PNG'
    imgT.save()
    """)
    # write script once
    tmp_script = tempfile.NamedTemporaryFile(delete=False, suffix=".py"); tmp_script.write(py.encode("utf-8")); tmp_script.close()

    for sid, src_path in sorted(aligned.items()):
      tgt_path = targets.get(sid)
      tex_in   = find_tex(sid)
      if not (tgt_path and tex_in):  # skip without texture or target
        continue
      out_png = os.path.join(outDir, f"{sid}.png")
      args = [blender_exe, "--background", "--python", tmp_script.name, "--",
              src_path, tgt_path, tex_in, out_png,
              str(bake_size), str(bake_extrusion), str(bake_margin_px), str(merge_dist)]
      subprocess.run(args, check=True)
      baked[sid] = out_png

    return baked
  
  def _calculate_average_texture(self, outTexturesDir):
    atlas_texture = os.path.join(outTexturesDir, "average_texture.png")
    if os.path.exists(atlas_texture):
      return
    pngs = glob.glob(os.path.join(outTexturesDir, "*.png"))
    images = [imageio.imread(png) for png in pngs]
    average = np.mean(images, axis=0).astype(np.uint8)
    imageio.imwrite(atlas_texture, average)
    


  def applyTextureToModel(self, modelNode, pngPath):
    modelNode.CreateDefaultDisplayNodes()
    dn = modelNode.GetDisplayNode()
    dn.SetBackfaceCulling(0); dn.SetFrontfaceCulling(0)
    dn.SetScalarVisibility(False)
    try: dn.SetInterpolateTexture(1)
    except Exception: pass

    reader = vtk.vtkPNGReader()
    reader.SetFileName(pngPath)
    reader.Update()

    # No flipping – Blender/Slicer UVs now match
    dn.SetTextureImageDataConnection(reader.GetOutputPort())

  def _dataset_file_map(self, directory, kind):
    """Return one deterministic input path per specimen for preflight checks."""
    if kind == 'landmark':
      extensions = ('.mrk.json', '.json', '.fcsv')
    elif kind == 'model':
      extensions = ('.obj', '.ply', '.stl', '.vtp', '.vtk')
    else:
      raise ValueError(f"Unsupported dataset file kind: {kind}")

    selected = {}
    for file_name in os.listdir(directory):
      if not _is_visible_dataset_file(file_name):
        continue
      lower_name = file_name.lower()
      matched_extension = next((extension for extension in extensions
                                if lower_name.endswith(extension)), None)
      if matched_extension is None:
        continue
      subject_id = file_name[:-len(matched_extension)]
      rank = extensions.index(matched_extension)
      if subject_id not in selected or rank < selected[subject_id][0]:
        selected[subject_id] = (rank, os.path.join(directory, file_name))
    return {subject_id: value[1] for subject_id, value in selected.items()}

  def _landmark_to_surface_distances(self, modelNode, lmNode):
    locator = vtk.vtkStaticCellLocator()
    locator.SetDataSet(modelNode.GetPolyData())
    locator.BuildLocator()
    distances = []
    for index in range(lmNode.GetNumberOfControlPoints()):
      point = [0.0, 0.0, 0.0]
      lmNode.GetNthControlPointPosition(index, point)
      closest = [0.0, 0.0, 0.0]
      cell_id = vtk.mutable(0)
      sub_id = vtk.mutable(0)
      distance_squared = vtk.mutable(0.0)
      locator.FindClosestPoint(point, closest, cell_id, sub_id, distance_squared)
      distances.append(distance_squared.get() ** 0.5)
    return np.asarray(distances, dtype=float)

  def validateAtlasDataset(self, meshDirectory, landmarkDirectory,
                           coordinateSystem='LPS', progressCallback=None):
    """Run mesh/landmark, anchor-frame, and sequence QA before TPS.

    Severe failures indicate that dense correspondence is unsafe and should be
    stopped. Group-shape outliers are warnings because they may represent real
    biological variation rather than landmarking mistakes.
    """
    landmark_paths = self._dataset_file_map(landmarkDirectory, 'landmark')
    model_paths = self._dataset_file_map(meshDirectory, 'model')
    landmark_ids = set(landmark_paths)
    model_ids = set(model_paths)
    common_ids = sorted(landmark_ids & model_ids)
    report = {
      'checked': 0,
      'passed': False,
      'severe': [],
      'warnings': [],
      'subjects': {},
    }

    missing_models = sorted(landmark_ids - model_ids)
    missing_landmarks = sorted(model_ids - landmark_ids)
    if missing_models:
      report['severe'].append(f"Missing models for: {', '.join(missing_models)}")
    if missing_landmarks:
      report['severe'].append(f"Missing landmarks for: {', '.join(missing_landmarks)}")
    if not common_ids:
      report['severe'].append("No paired mesh and landmark files were found")
      return report

    reference_labels = None
    reference_point_count = None
    frame_records = []
    anchor_ratio_records = []
    side_records = []

    for specimen_index, subject_id in enumerate(common_ids):
      model_node = None
      landmark_node = None
      subject_metrics = {}
      subject_severe = []
      subject_warnings = []
      try:
        landmark_node = slicer.util.loadMarkups(landmark_paths[subject_id])
        model_node = self._load_model_with_cs(model_paths[subject_id], coordinateSystem)
        point_count = landmark_node.GetNumberOfControlPoints()
        labels = [landmark_node.GetNthControlPointLabel(i) for i in range(point_count)]
        points = np.empty((point_count, 3), dtype=float)
        for point_index in range(point_count):
          landmark_node.GetNthControlPointPosition(point_index, points[point_index])

        if reference_labels is None:
          reference_labels = labels
          reference_point_count = point_count
        elif point_count != reference_point_count:
          subject_severe.append(
            f"has {point_count} landmarks; expected {reference_point_count}"
          )
        elif labels != reference_labels:
          subject_severe.append("landmark labels or ordering differ from the dataset reference")

        polydata = model_node.GetPolyData()
        mesh_diagonal = float(polydata.GetLength()) if polydata else 0.0
        if mesh_diagonal <= 0 or not np.isfinite(mesh_diagonal):
          raise ValueError("mesh has an invalid or zero bounding-box diagonal")

        distances = self._landmark_to_surface_distances(model_node, landmark_node)
        relative_distances = distances / mesh_diagonal
        subject_metrics['median_surface_distance'] = float(np.median(relative_distances))
        subject_metrics['max_surface_distance'] = float(np.max(relative_distances))
        if subject_metrics['median_surface_distance'] > 0.02:
          subject_severe.append(
            f"median landmark-to-surface distance is "
            f"{subject_metrics['median_surface_distance']:.1%} of mesh size"
          )
        elif subject_metrics['max_surface_distance'] > 0.05:
          subject_warnings.append(
            f"one or more landmarks are as far as "
            f"{subject_metrics['max_surface_distance']:.1%} of mesh size from the surface"
          )

        label_to_index = {label: index for index, label in enumerate(labels)}
        missing_anchors = [label for label in ATLAS_ANCHOR_LABELS
                           if label not in label_to_index]
        if missing_anchors:
          subject_warnings.append(
            f"anchor-frame check skipped; missing: {', '.join(missing_anchors)}"
          )
        else:
          anchor_indices = [label_to_index[label] for label in ATLAS_ANCHOR_LABELS]
          anchor_surface_max = float(np.max(relative_distances[anchor_indices]))
          subject_metrics['anchor_surface_distance'] = anchor_surface_max
          if anchor_surface_max > 0.03:
            subject_severe.append(
              f"an anchor is {anchor_surface_max:.1%} of mesh size from the surface"
            )
          try:
            frame = _anatomical_anchor_frame(points, *anchor_indices)
            if labels == reference_labels:
              frame_records.append((subject_id, frame))
            beak = points[anchor_indices[0]]
            anterior = points[anchor_indices[1]]
            posterior = points[anchor_indices[2]]
            hinge_midpoint = 0.5 * (anterior + posterior)
            anchor_ratio_records.append((
              subject_id,
              np.array((np.linalg.norm(anterior - posterior) / mesh_diagonal,
                        np.linalg.norm(hinge_midpoint - beak) / mesh_diagonal))
            ))
            if ATLAS_SIDE_LABEL in label_to_index:
              side_value = frame[label_to_index[ATLAS_SIDE_LABEL], 2]
              if abs(side_value) > 1e-8:
                side_records.append((subject_id, int(np.sign(side_value))))
          except ValueError as error:
            subject_severe.append(f"invalid anatomical anchor frame: {error}")

        present_growth = [label for label in GROWTH_AXIS_LABELS
                          if label in label_to_index]
        if present_growth and len(present_growth) != len(GROWTH_AXIS_LABELS):
          missing_growth = [label for label in GROWTH_AXIS_LABELS
                            if label not in label_to_index]
          subject_severe.append(
            f"growth axis is incomplete; missing: {', '.join(missing_growth)}"
          )
        elif len(present_growth) == len(GROWTH_AXIS_LABELS) and 'beak' in label_to_index:
          growth_indices = [label_to_index[label] for label in GROWTH_AXIS_LABELS]
          quality = _growth_axis_quality(points, label_to_index['beak'], growth_indices)
          subject_metrics.update({f'growth_{key}': value
                                  for key, value in quality.items()})
          if quality['closest_to_beak'] != 0:
            subject_severe.append(
              f"growth axis is out of order: {GROWTH_AXIS_LABELS[quality['closest_to_beak']]} "
              "is closest to the beak, not max_growth_axis_001"
            )
          if quality['min_turn_cosine'] <= 0:
            maximum_turn = math.degrees(math.acos(
              float(np.clip(quality['min_turn_cosine'], -1.0, 1.0))))
            subject_severe.append(
              f"growth axis reverses direction ({maximum_turn:.1f} degree turn)"
            )
          if quality['spacing_cv'] > 0.35:
            subject_severe.append(
              f"growth-axis spacing variation is {quality['spacing_cv']:.0%}"
            )
          elif quality['spacing_cv'] > 0.15:
            subject_warnings.append(
              f"growth-axis spacing variation is {quality['spacing_cv']:.0%}"
            )
          if quality['path_to_direct'] > 1.75:
            subject_severe.append(
              f"growth-axis path is {quality['path_to_direct']:.2f}x its direct length"
            )
          elif quality['path_to_direct'] > 1.50:
            subject_warnings.append(
              f"growth-axis path is {quality['path_to_direct']:.2f}x its direct length"
            )

      except Exception as error:
        subject_severe.append(f"preflight could not read or validate the pair: {error}")
      finally:
        if landmark_node is not None:
          slicer.mrmlScene.RemoveNode(landmark_node)
        if model_node is not None:
          slicer.mrmlScene.RemoveNode(model_node)

      report['checked'] += 1
      report['subjects'][subject_id] = subject_metrics
      report['severe'].extend(f"{subject_id}: {message}" for message in subject_severe)
      report['warnings'].extend(f"{subject_id}: {message}" for message in subject_warnings)
      if progressCallback:
        progressCallback(specimen_index + 1, len(common_ids), subject_id)

    # Compare anchor-normalized shapes only after every specimen has its own
    # anatomical frame. These are warnings because genuine morphology can be
    # an outlier without being incorrectly landmarked.
    if len(frame_records) >= 5:
      frame_stack = np.stack([record[1] for record in frame_records])
      median_frame = np.median(frame_stack, axis=0)
      frame_deviation = np.sqrt(np.mean((frame_stack - median_frame) ** 2,
                                        axis=(1, 2)))
      frame_z = np.abs(_robust_modified_z(frame_deviation))
      for (subject_id, _), z_score in zip(frame_records, frame_z):
        report['subjects'][subject_id]['anchor_frame_outlier_z'] = float(z_score)
        if z_score > 6.0:
          report['warnings'].append(
            f"{subject_id}: anchor-normalized landmark shape is a robust group outlier "
            f"(z={z_score:.1f})"
          )

    if len(anchor_ratio_records) >= 5:
      ratios = np.stack([record[1] for record in anchor_ratio_records])
      ratio_z = np.abs(_robust_modified_z(ratios))
      for (subject_id, _), z_scores in zip(anchor_ratio_records, ratio_z):
        max_z = float(np.max(z_scores))
        report['subjects'][subject_id]['anchor_geometry_outlier_z'] = max_z
        if max_z > 6.0:
          report['warnings'].append(
            f"{subject_id}: anchor geometry is a robust group outlier (z={max_z:.1f})"
          )

    if len(side_records) >= 5:
      majority_sign = 1 if sum(record[1] for record in side_records) >= 0 else -1
      for subject_id, sign in side_records:
        if sign != majority_sign:
          report['warnings'].append(
            f"{subject_id}: anatomical frame handedness differs from the group; "
            "check for a mirrored valve or swapped hinge labels"
          )

    report['passed'] = not report['severe']
    return report

  def _median_landmark_to_surface_dist(self, modelNode, lmNode):
    distances = self._landmark_to_surface_distances(modelNode, lmNode)
    return np.median(distances) if len(distances) else float('inf')

  def _validate_mesh_landmarks(self, modelNode, lmNode, context="", threshold=0.02):
    """Warn when a mesh and its markups are not in the same coordinate space.

    Landmark points need not lie exactly on a reconstructed surface, so use
    the median point-to-surface distance.  Normalize by the mesh bounding-box
    diagonal rather than total edge length; this remains meaningful across
    meshes with different tessellation densities and scales.
    """
    if modelNode is None or lmNode is None or modelNode.GetPolyData() is None:
      return None
    polydata = modelNode.GetPolyData()
    bounds = [0.0] * 6
    polydata.GetBounds(bounds)
    diagonal = np.linalg.norm((bounds[1] - bounds[0],
                               bounds[3] - bounds[2],
                               bounds[5] - bounds[4]))
    if not np.isfinite(diagonal) or diagonal <= 0:
      return None
    median_dist = self._median_landmark_to_surface_dist(modelNode, lmNode)
    relative_dist = median_dist / diagonal
    if relative_dist > threshold:
      label = f" for {context}" if context else ""
      print("WARNING: median landmark-to-surface distance"
            f"{label} is {relative_dist:.2%} of mesh diagonal "
            f"({median_dist:.6g} / {diagonal:.6g}). "
            "Check mesh and landmark coordinate systems.")
    return relative_dist
  
  def _load_model_with_cs(self, filePath, coordinateSystem='RAS'):
    storage = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLModelStorageNode')
    cs = (coordinateSystem or 'RAS').upper()
    try:
      if cs == 'RAS': storage.SetCoordinateSystemToRAS()
      else:           storage.SetCoordinateSystemToLPS()
    except AttributeError:
      storage.SetCoordinateSystem(0 if cs == 'RAS' else 1)
    modelNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLModelNode')
    storage.SetFileName(filePath)
    ok = storage.ReadData(modelNode)
    slicer.mrmlScene.RemoveNode(storage)
    if not ok:
      slicer.mrmlScene.RemoveNode(modelNode)
      raise RuntimeError(f"Failed to read model: {filePath}")
    return modelNode

  def findBlenderExecutable(self):
    """
    Automatically find Blender executable on the system.
    Returns the path to Blender executable if found, None otherwise.
    """
    import platform
    import subprocess
    import shutil

    system = platform.system().lower()

    # First, try to find Blender in PATH
    blender_names = ['blender', 'blender.exe'] if system == 'windows' else ['blender']

    for name in blender_names:
      path = shutil.which(name)
      if path and os.path.isfile(path):
        print(f"Found Blender in PATH: {path}")
        return path

    # Try common installation locations based on OS
    common_paths = []

    if system == 'windows':
      # Windows common locations
      program_files = [
        os.environ.get('PROGRAMFILES', 'C:\\Program Files'),
        os.environ.get('PROGRAMFILES(X86)', 'C:\\Program Files (x86)')
      ]
      for pf in program_files:
        # Check various Blender versions
        blender_dirs = glob.glob(os.path.join(pf, 'Blender Foundation', 'Blender*'))
        for blender_dir in blender_dirs:
          common_paths.append(os.path.join(blender_dir, 'blender.exe'))

    elif system == 'darwin':  # macOS
      common_paths = [
        '/Applications/Blender.app/Contents/MacOS/Blender',
        '/opt/homebrew/bin/blender',
        '/usr/local/bin/blender'
      ]
      # Check for various Blender versions in Applications
      blender_apps = glob.glob('/Applications/Blender*.app/Contents/MacOS/Blender')
      common_paths.extend(blender_apps)

    else:  # Linux and other Unix-like systems
      common_paths = [
        '/usr/bin/blender',
        '/usr/local/bin/blender',
        '/opt/blender/blender',
        '/snap/bin/blender',
        os.path.expanduser('~/blender/blender'),
        os.path.expanduser('~/.local/bin/blender'),
        # Flatpak installation paths
        '/var/lib/flatpak/exports/bin/org.blender.Blender',
        os.path.expanduser('~/.local/share/flatpak/exports/bin/org.blender.Blender'),
        # AppImage installations
        os.path.expanduser('~/Applications/Blender.AppImage'),
        os.path.expanduser('~/Downloads/Blender.AppImage'),
        '/opt/Blender.AppImage',
        # Distribution-specific paths
        '/usr/share/blender/blender',  # Some distributions
        '/usr/games/blender',          # Debian games partition
        '/opt/blender-*/blender',      # Custom installations
      ]
      
      # Check for snap installations with version numbers
      snap_paths = glob.glob('/snap/blender/*/blender')
      common_paths.extend(snap_paths)
      
      # Check for versioned installations
      versioned_paths = glob.glob('/usr/bin/blender-*')
      versioned_paths.extend(glob.glob('/usr/local/bin/blender-*'))
      versioned_paths.extend(glob.glob('/opt/blender-*/blender'))
      common_paths.extend(versioned_paths)
      
      # Check for user-installed versions
      user_blender_dirs = glob.glob(os.path.expanduser('~/blender-*'))
      for blender_dir in user_blender_dirs:
        potential_path = os.path.join(blender_dir, 'blender')
        if os.path.isfile(potential_path):
          common_paths.append(potential_path)

    # Test each common path
    for path in common_paths:
      if os.path.isfile(path) and os.access(path, os.X_OK):
        print(f"Found Blender at: {path}")
        return path

    print("Blender executable not found in common locations")
    return None

  def _tryInstallBlenderMacOS(self, log_callback):
    """
    Try to install Blender on macOS using Homebrew.
    Returns True if successful, False otherwise.
    """
    import subprocess
    
    def log(message):
      if log_callback:
        log_callback(message)
      else:
        print(message)
    
    try:
      # Check if Homebrew is installed
      result = subprocess.run(['which', 'brew'], capture_output=True, text=True)
      if result.returncode != 0:
        log("Homebrew not found. Trying to install Homebrew first...")
        # Install Homebrew
        install_brew_cmd = '/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'
        result = subprocess.run(install_brew_cmd, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
          log("Failed to install Homebrew")
          return False
      
      log("Installing Blender via Homebrew...")
      result = subprocess.run(['brew', 'install', '--cask', 'blender'], capture_output=True, text=True, timeout=300)
      
      if result.returncode == 0:
        log("Blender installed successfully via Homebrew")
        return True
      else:
        log(f"Homebrew installation failed: {result.stderr}")
        return False
        
    except subprocess.TimeoutExpired:
      log("Homebrew installation timed out")
      return False
    except Exception as e:
      log(f"Error during Homebrew installation: {e}")
      return False

  def _tryInstallBlenderLinux(self, log_callback):
    """
    Try to install Blender on Linux using various package managers.
    Returns True if successful, False otherwise.
    """
    import subprocess
    import shutil
    
    def log(message):
      if log_callback:
        log_callback(message)
      else:
        print(message)
    
    # Detect distribution and get preferred package managers
    distro_name, preferred_managers = self._detectLinuxDistribution()
    log(f"Detected Linux distribution: {distro_name}")
    log(f"Preferred package managers: {', '.join(preferred_managers)}")
    
    # Define all package managers with their commands
    all_package_managers = {
      'snap': {
        'check': ['which', 'snap'],
        'install': ['sudo', 'snap', 'install', 'blender', '--classic'],
        'name': 'Snap'
      },
      'flatpak': {
        'check': ['which', 'flatpak'],
        'install': ['flatpak', 'install', '-y', 'flathub', 'org.blender.Blender'],
        'name': 'Flatpak',
        'setup': ['flatpak', 'remote-add', '--if-not-exists', 'flathub', 'https://flathub.org/repo/flathub.flatpakrepo']
      },
      'apt': {
        'check': ['which', 'apt'],
        'install': ['sudo', 'apt', 'update', '&&', 'sudo', 'apt', 'install', '-y', 'blender'],
        'name': 'APT'
      },
      'dnf': {
        'check': ['which', 'dnf'],
        'install': ['sudo', 'dnf', 'install', '-y', 'blender'],
        'name': 'DNF'
      },
      'yum': {
        'check': ['which', 'yum'],
        'install': ['sudo', 'yum', 'install', '-y', 'blender'],
        'name': 'YUM'
      },
      'pacman': {
        'check': ['which', 'pacman'],
        'install': ['sudo', 'pacman', '-S', '--noconfirm', 'blender'],
        'name': 'Pacman'
      },
      'zypper': {
        'check': ['which', 'zypper'],
        'install': ['sudo', 'zypper', 'install', '-y', 'blender'],
        'name': 'Zypper'
      }
    }
    
    # Try preferred package managers first, then fall back to others
    managers_to_try = []
    for pm_name in preferred_managers:
      if pm_name in all_package_managers:
        managers_to_try.append(all_package_managers[pm_name])
    
    # Add remaining package managers as fallbacks
    for pm_name, pm_info in all_package_managers.items():
      if pm_name not in preferred_managers:
        managers_to_try.append(pm_info)
    
    for pm in managers_to_try:
      try:
        # Check if package manager is available
        result = subprocess.run(pm['check'], capture_output=True, text=True)
        if result.returncode == 0:
          log(f"Found {pm['name']} package manager. Attempting installation...")
          
          # Run setup command if needed (e.g., for Flatpak)
          if 'setup' in pm:
            log(f"Setting up {pm['name']}...")
            setup_result = subprocess.run(pm['setup'], capture_output=True, text=True, timeout=60)
            if setup_result.returncode != 0:
              log(f"{pm['name']} setup failed, but continuing anyway: {setup_result.stderr}")
          
          # Handle shell commands with && properly
          if isinstance(pm['install'], list) and '&&' in ' '.join(pm['install']):
            cmd_str = ' '.join(pm['install'])
            result = subprocess.run(cmd_str, shell=True, capture_output=True, text=True, timeout=300)
          else:
            result = subprocess.run(pm['install'], capture_output=True, text=True, timeout=300)
          
          if result.returncode == 0:
            log(f"Blender installed successfully via {pm['name']}")
            return True
          else:
            log(f"{pm['name']} installation failed: {result.stderr}")
            continue
            
      except subprocess.TimeoutExpired:
        log(f"{pm['name']} installation timed out")
        continue
      except Exception as e:
        log(f"Error during {pm['name']} installation: {e}")
        continue
    
    log("All package manager installations failed")
    return False

  def _detectLinuxDistribution(self):
    """
    Detect Linux distribution to prioritize appropriate package managers.
    Returns a tuple of (distro_name, package_manager_priority)
    """
    import subprocess
    
    try:
      # Try to read /etc/os-release
      with open('/etc/os-release', 'r') as f:
        lines = f.readlines()
        distro_info = {}
        for line in lines:
          if '=' in line:
            key, value = line.strip().split('=', 1)
            distro_info[key] = value.strip('"')
        
        distro_id = distro_info.get('ID', '').lower()
        distro_like = distro_info.get('ID_LIKE', '').lower()
        
        # Return appropriate package manager priority based on distribution
        if 'ubuntu' in distro_id or 'debian' in distro_id or 'ubuntu' in distro_like:
          return ('debian', ['snap', 'apt', 'flatpak'])  # Snap is preferred on Ubuntu
        elif 'fedora' in distro_id or 'rhel' in distro_id or 'centos' in distro_id:
          return ('fedora', ['dnf', 'flatpak', 'snap'])
        elif 'arch' in distro_id or 'manjaro' in distro_id:
          return ('arch', ['pacman', 'flatpak', 'snap'])
        elif 'opensuse' in distro_id or 'suse' in distro_id:
          return ('suse', ['zypper', 'flatpak', 'snap'])
        elif 'alpine' in distro_id:
          return ('alpine', ['flatpak', 'snap'])  # Alpine doesn't have Blender in main repos
        else:
          return ('unknown', ['snap', 'flatpak', 'apt', 'dnf', 'pacman', 'zypper'])
          
    except FileNotFoundError:
      # Fallback to lsb_release or uname
      try:
        result = subprocess.run(['lsb_release', '-si'], capture_output=True, text=True)
        if result.returncode == 0:
          distro = result.stdout.strip().lower()
          if 'ubuntu' in distro or 'debian' in distro:
            return ('debian', ['snap', 'apt', 'flatpak'])
          elif 'fedora' in distro or 'red hat' in distro:
            return ('fedora', ['dnf', 'flatpak', 'snap'])
        
        # Final fallback
        return ('unknown', ['snap', 'flatpak', 'apt', 'dnf', 'pacman', 'zypper'])
        
      except:
        return ('unknown', ['snap', 'flatpak', 'apt', 'dnf', 'pacman', 'zypper'])

  def _installBlenderMacOSDMG(self, download_urls, filename, install_dir, log_callback):
    """
    Handle macOS DMG file download and installation.
    Returns the path to Blender executable if successful, None otherwise.
    """
    import subprocess
    import tempfile
    import urllib.request
    import ssl
    
    def log(message):
      if log_callback:
        log_callback(message)
      else:
        print(message)
    
    temp_dmg_path = None
    
    # Try each download URL until one works
    for download_url in download_urls:
      try:
        log(f"Downloading macOS DMG from: {download_url}")
        
        # Create request with proper headers
        request = urllib.request.Request(download_url)
        request.add_header('User-Agent', 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36')
        
        # Download with SSL context
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.dmg') as tmp_file:
          with urllib.request.urlopen(request, context=ssl_context) as response:
            content_type = response.headers.get('content-type', '').lower()
            
            if 'text/html' in content_type:
              log("Got HTML response (likely error page), trying next URL...")
              continue
            
            # Download in chunks
            while True:
              chunk = response.read(8192)
              if not chunk:
                break
              tmp_file.write(chunk)
          
          temp_dmg_path = tmp_file.name
          break
          
      except Exception as e:
        log(f"Failed to download from {download_url}: {e}")
        if temp_dmg_path and os.path.exists(temp_dmg_path):
          os.unlink(temp_dmg_path)
          temp_dmg_path = None
        continue
    
    if not temp_dmg_path:
      log("Failed to download DMG from any mirror")
      return None
    
    try:
      log("Mounting DMG file...")
      
      # Mount the DMG
      mount_result = subprocess.run(
        ['hdiutil', 'attach', temp_dmg_path, '-nobrowse', '-quiet'],
        capture_output=True, text=True
      )
      
      if mount_result.returncode != 0:
        log(f"Failed to mount DMG: {mount_result.stderr}")
        os.unlink(temp_dmg_path)
        return None
      
      # Parse mount output to find the volume path
      mount_point = None
      for line in mount_result.stdout.split('\n'):
        if '/Volumes/' in line:
          mount_point = line.split()[-1]
          break
      
      if not mount_point:
        log("Could not determine mount point")
        os.unlink(temp_dmg_path)
        return None
      
      log(f"DMG mounted at: {mount_point}")
      
      # Find Blender.app in the mounted volume
      blender_app_source = os.path.join(mount_point, 'Blender.app')
      if not os.path.exists(blender_app_source):
        # Try to find it with different names
        for item in os.listdir(mount_point):
          if item.endswith('.app') and 'blender' in item.lower():
            blender_app_source = os.path.join(mount_point, item)
            break
      
      if not os.path.exists(blender_app_source):
        log("Could not find Blender.app in mounted DMG")
        subprocess.run(['hdiutil', 'detach', mount_point, '-quiet'], capture_output=True)
        os.unlink(temp_dmg_path)
        return None
      
      # Copy Blender.app to Applications
      blender_app_dest = '/Applications/Blender.app'
      log(f"Installing Blender.app to {blender_app_dest}...")
      
      # Remove existing installation if present
      if os.path.exists(blender_app_dest):
        subprocess.run(['rm', '-rf', blender_app_dest], capture_output=True)
      
      # Copy the app bundle
      copy_result = subprocess.run(['cp', '-R', blender_app_source, blender_app_dest], capture_output=True, text=True)
      
      # Unmount the DMG
      subprocess.run(['hdiutil', 'detach', mount_point, '-quiet'], capture_output=True)
      os.unlink(temp_dmg_path)
      
      if copy_result.returncode != 0:
        log(f"Failed to copy Blender.app: {copy_result.stderr}")
        return None
      
      # Return path to the executable
      blender_executable = os.path.join(blender_app_dest, 'Contents', 'MacOS', 'Blender')
      if os.path.exists(blender_executable):
        log(f"Blender installed successfully at: {blender_executable}")
        return blender_executable
      else:
        log("Blender executable not found in installed app bundle")
        return None
        
    except Exception as e:
      log(f"Error during DMG installation: {e}")
      if temp_dmg_path and os.path.exists(temp_dmg_path):
        os.unlink(temp_dmg_path)
      return None

  def installBlender(self, log_callback=None):
    """
    Automatically download and install Blender with enhanced cross-platform support.
    
    Features:
    - macOS: Tries Homebrew first, falls back to DMG download and installation
    - Linux: Detects distribution and uses appropriate package manager (snap, flatpak, apt, dnf, pacman, zypper)
    - Windows: Direct download and extraction (existing functionality)
    - Supports both x64 and ARM64 architectures where available
    
    Returns the path to the installed Blender executable if successful, None otherwise.
    """
    import platform
    import subprocess
    import tempfile
    import zipfile
    import tarfile
    import urllib.request
    import urllib.parse
    import ssl

    def log(message):
      if log_callback:
        log_callback(message)
      else:
        print(message)

    system = platform.system().lower()
    architecture = platform.machine().lower()

    # Use current stable version URLs from blender.org
    blender_version = "4.5.3"

    if system == 'windows':
      if 'amd64' in architecture or 'x86_64' in architecture:
        filename = f"blender-{blender_version}-windows-x64.zip"
        blender_exe = "blender.exe"
      elif 'arm' in architecture or 'aarch64' in architecture:
        filename = f"blender-{blender_version}-windows-arm64.zip"
        blender_exe = "blender.exe"
      else:
        log("Unsupported Windows architecture")
        return None

    elif system == 'darwin':  # macOS
      # Try to install via Homebrew first, then fallback to direct download
      if self._tryInstallBlenderMacOS(log):
        return self.findBlenderExecutable()
      
      # If Homebrew fails, try direct download
      if 'amd64' in architecture or 'x86_64' in architecture:
        filename = f"blender-{blender_version}-macos-x64.dmg"
        blender_exe = "Blender"  # macOS app bundle executable
      elif 'arm' in architecture or 'aarch64' in architecture:
        filename = f"blender-{blender_version}-macos-arm64.dmg"
        blender_exe = "Blender"
      else:
        log("Unsupported macOS architecture")
        return None
    
    elif system == 'linux':
      # Try to install via package manager first
      if self._tryInstallBlenderLinux(log):
        return self.findBlenderExecutable()
      
      # If package manager fails, try direct download
      if 'amd64' in architecture or 'x86_64' in architecture:
        filename = f"blender-{blender_version}-linux-x64.tar.xz"
        blender_exe = "blender"
      elif 'arm' in architecture or 'aarch64' in architecture:
        filename = f"blender-{blender_version}-linux-arm64.tar.xz"
        blender_exe = "blender"
      else:
        log("Unsupported Linux architecture")
        return None

    else:
      log(f"Unsupported operating system: {system}")
      return None

    # Try multiple download URLs in order of preference
    download_urls = [
      f"https://www.blender.org/download/release/Blender4.5/{filename}",
      f"https://download.blender.org/release/Blender4.5/{filename}",
      f"https://mirror.clarkson.edu/blender/release/Blender4.5/{filename}",
      f"https://ftp.nluug.nl/pub/graphics/blender/release/Blender4.5/{filename}"
    ]
    
    # Special handling for macOS DMG files
    if system == 'darwin' and filename.endswith('.dmg'):
      return self._installBlenderMacOSDMG(download_urls, filename, install_dir, log)
    
    # Special handling for Linux ARM64
    if system == 'linux' and 'arm' in architecture:
      # ARM64 Linux support was added in Blender 3.0+
      if not any(url for url in download_urls if 'arm64' in filename):
        log("ARM64 Linux binaries may not be available for this Blender version. Trying x64 compatibility...")
        filename = f"blender-{blender_version}-linux-x64.tar.xz"
        download_urls = [url.replace('arm64', 'x64') for url in download_urls]
    
    # Create installation directory
    install_dir = os.path.join(os.path.expanduser('~'), '.slicer-blender')
    os.makedirs(install_dir, exist_ok=True)

    # Check if already installed
    arch_suffix = 'x64'
    if 'arm' in architecture or 'aarch64' in architecture:
      arch_suffix = 'arm64'
    
    expected_blender_dir = os.path.join(install_dir, f"blender-{blender_version}-{system}-{arch_suffix}")
    
    # For macOS, the executable is in a different location within the app bundle
    if system == 'darwin':
      expected_blender_path = '/Applications/Blender.app/Contents/MacOS/Blender'
      # Also check for manually downloaded version
      manual_blender_path = os.path.join(expected_blender_dir, 'Blender.app', 'Contents', 'MacOS', 'Blender')
      if os.path.isfile(manual_blender_path):
        log(f"Blender already installed at: {manual_blender_path}")
        return manual_blender_path
    else:
      expected_blender_path = os.path.join(expected_blender_dir, blender_exe)
    
    if os.path.isfile(expected_blender_path):
      log(f"Blender already installed at: {expected_blender_path}")
      return expected_blender_path

    log(f"Downloading Blender {blender_version}...")

    # Try each download URL until one works
    temp_path = None
    for download_url in download_urls:
      try:
        log(f"Trying URL: {download_url}")

        # Create request with proper headers
        request = urllib.request.Request(download_url)
        request.add_header('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')

        # Download with SSL context to handle certificate issues
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE

        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as tmp_file:
          with urllib.request.urlopen(request, context=ssl_context) as response:
            # Check if we got a valid response
            content_type = response.headers.get('content-type', '').lower()
            content_length = response.headers.get('content-length', '0')

            log(f"Content-Type: {content_type}")
            log(f"Content-Length: {content_length}")

            # Check if response looks like an error page
            if 'text/html' in content_type:
              log("Got HTML response (likely error page), trying next URL...")
              continue

            # Download in chunks to show progress for large files
            total_size = int(content_length) if content_length.isdigit() else 0
            downloaded = 0
            chunk_size = 8192

            while True:
              chunk = response.read(chunk_size)
              if not chunk:
                break
              tmp_file.write(chunk)
              downloaded += len(chunk)

              if total_size > 0:
                progress = (downloaded / total_size) * 100
                if downloaded % (chunk_size * 100) == 0:  # Log every 100 chunks
                  log(f"Download progress: {progress:.1f}%")

          temp_path = tmp_file.name
          log(f"Downloaded {downloaded} bytes to {temp_path}")
          break  # Success, exit the URL loop

      except Exception as e:
        log(f"Failed to download from {download_url}: {e}")
        if temp_path and os.path.exists(temp_path):
          os.unlink(temp_path)
          temp_path = None
        continue

    if not temp_path:
      log("Failed to download from any mirror")
      return None

    try:
      log("Download completed. Extracting...")

      # Verify file size before extraction
      file_size = os.path.getsize(temp_path)
      log(f"Downloaded file size: {file_size} bytes")

      if file_size < 1000000:  # Less than 1MB is suspicious
        log("Downloaded file is too small, likely an error page")
        with open(temp_path, 'r', encoding='utf-8', errors='ignore') as f:
          content = f.read(500)  # Read first 500 chars
          log(f"File content preview: {content}")
        os.unlink(temp_path)
        return None

      # Extract based on file type
      if filename.endswith('.zip'):
        try:
          with zipfile.ZipFile(temp_path, 'r') as zip_ref:
            zip_ref.extractall(install_dir)
        except zipfile.BadZipFile:
          log("Invalid zip file downloaded")
          os.unlink(temp_path)
          return None
      elif filename.endswith('.tar.xz'):
        try:
          with tarfile.open(temp_path, 'r:xz') as tar_ref:
            tar_ref.extractall(install_dir)
        except tarfile.TarError:
          log("Invalid tar.xz file downloaded")
          os.unlink(temp_path)
          return None

      # Clean up temporary file
      os.unlink(temp_path)

      # Find the extracted Blender executable
      blender_path = expected_blender_path
      if not os.path.isfile(blender_path):
        # Try to find it in any subdirectory
        log("Searching for blender executable in extracted files...")
        for root, dirs, files in os.walk(install_dir):
          if blender_exe in files:
            blender_path = os.path.join(root, blender_exe)
            log(f"Found blender at: {blender_path}")
            break

      if os.path.isfile(blender_path):
        # Make executable on Unix-like systems
        if system != 'windows':
          os.chmod(blender_path, 0o755)

        log(f"Blender installed successfully at: {blender_path}")
        return blender_path
      else:
        log("Failed to find Blender executable after extraction")
        log(f"Expected at: {expected_blender_path}")
        log("Extracted files:")
        for root, dirs, files in os.walk(install_dir):
          for file in files[:10]:  # Limit output
            log(f"  {os.path.join(root, file)}")
        return None

    except Exception as e:
      log(f"Failed to extract/install Blender: {e}")
      if temp_path and os.path.exists(temp_path):
        os.unlink(temp_path)
      return None

  def getBlenderExecutable(self, log_callback=None):
    """
    Get Blender executable path by trying auto-detection first, then auto-installation.
    Returns the path to Blender executable if found/installed, None otherwise.
    """
    def log(message):
      if log_callback:
        log_callback(message)
      else:
        print(message)

    # First try to find existing installation
    blender_path = self.findBlenderExecutable()
    if blender_path:
      return blender_path

    # If not found, try to install automatically
    log("Blender not found. Attempting automatic installation...")
    blender_path = self.installBlender(log_callback)
    if blender_path:
      return blender_path

    log("Failed to automatically install Blender. Please install manually.")
    return None
  def runColorsEDA(self, atlasModel, texturesDir, colorSpace, dimRedAlgo, progressCallback=None, logCallback=None, satCutoff=10.0, valueCutoff=10.0, enhanceColors=False):
    """
    Run color analysis with dimensionality reduction on face-averaged colors

    Args:
        atlasModel: VTK model node of the atlas
        texturesDir: Directory containing baked atlas-space PNG textures
        colorSpace: "RGB" or "HSV"
        dimRedAlgo: "PCA", "ICA", or "UMAP"
        progressCallback: Function to call with progress updates (0-100)
        logCallback: Function to call with log messages
        satCutoff: Minimum saturation threshold for HSV filtering (0-100)
        valueCutoff: Minimum value/brightness threshold for HSV filtering (0-100)
        enhanceColors: Whether to enhance colors for visibility in 2D plots

    Returns:
        bool: True if successful, False otherwise
    """
    try:
      if logCallback:
        logCallback("Initializing color analysis...")

      if progressCallback:
        progressCallback(5)

      # Check dependencies
      if dimRedAlgo in ["PCA", "ICA"] and not SKLEARN_AVAILABLE:
        if logCallback:
          logCallback("Error: sklearn not available for PCA/ICA")
        return False

      if dimRedAlgo == "UMAP" and not UMAP_AVAILABLE:
        if logCallback:
          logCallback("Error: umap-learn not available for UMAP")
        return False

      # Get atlas polydata
      atlasPolyData = atlasModel.GetPolyData()
      if not atlasPolyData:
        if logCallback:
          logCallback("Error: Atlas model has no polydata")
        return False

      if progressCallback:
        progressCallback(10)

      # Get list of texture files
      textureFiles = []
      if os.path.isdir(texturesDir):
        for f in os.listdir(texturesDir):
          if f.lower().endswith(('.png', '.tiff', '.tif')) and not f.lower().startswith('average_texture'):
            textureFiles.append(os.path.join(texturesDir, f))

      if not textureFiles:
        if logCallback:
          logCallback(f"Error: No texture files found in {texturesDir}")
        return False

      if logCallback:
        logCallback(f"Found {len(textureFiles)} texture files")

      if progressCallback:
        progressCallback(15)

      # Calculate face-averaged colors for each texture
      allFaceColors = []
      specimenNames = []

      for i, texturePath in enumerate(textureFiles):
        if logCallback:
          logCallback(f"Processing texture {i+1}/{len(textureFiles)}: {os.path.basename(texturePath)}")

        # Load texture image
        try:
          textureImage = imageio.imread(texturePath)
          if len(textureImage.shape) != 3 or textureImage.shape[2] < 3:
            if logCallback:
              logCallback(f"Warning: Skipping {os.path.basename(texturePath)} - invalid format")
            continue
        except Exception as e:
          if logCallback:
            logCallback(f"Warning: Could not load {os.path.basename(texturePath)}: {e}")
          continue

        # Calculate face-averaged colors
        faceColors = self._calculateFaceAverageColors(atlasPolyData, textureImage, colorSpace)
        if faceColors is not None:
          allFaceColors.append(faceColors)
          specimenNames.append(os.path.splitext(os.path.basename(texturePath))[0])

        if progressCallback:
          progressCallback(15 + int(60 * (i + 1) / len(textureFiles)))

      if not allFaceColors:
        if logCallback:
          logCallback("Error: No valid textures processed")
        return False

      if logCallback:
        logCallback(f"Successfully processed {len(allFaceColors)} textures")

      # Concatenate all face colors into a single array
      # Shape: (N_faces * N_specimens, 3 or 4) - 4 for HSV with hue as 2D vector
      allFaceColors = np.array(allFaceColors)  # Shape: (N_specimens, N_faces, 3 or 4)
      nSpecimens, nFaces, nChannels = allFaceColors.shape

      # Reshape to (N_faces * N_specimens, 3 or 4)
      colorData = allFaceColors.reshape(-1, nChannels)

      if logCallback:
        logCallback(f"Full color data shape: {colorData.shape}")

      if progressCallback:
        progressCallback(80)

      # Apply dimensionality reduction with optional HSV filtering
      if logCallback:
        logCallback(f"Applying {dimRedAlgo} dimensionality reduction...")

      # Filter data for dimensionality reduction if in HSV mode
      dimRedData = colorData
      if colorSpace == "HSV":
        # Extract saturation and value channels (indices 2 and 3)
        sat = colorData[:, 2]  # 0..100
        val = colorData[:, 3]  # 0..100

        # Create mask for saturation and value cutoffs
        mask = (sat >= satCutoff) & (val >= valueCutoff)
        dimRedData = colorData[mask]

        if logCallback:
          logCallback(f"HSV filtering: {np.sum(mask)}/{len(mask)} samples passed cutoffs (sat>={satCutoff}, val>={valueCutoff})")
          logCallback(f"Filtered data shape for dim reduction: {dimRedData.shape}")

      reducedData = self._applyDimensionalityReduction(dimRedData, dimRedAlgo)

      if reducedData is None:
        if logCallback:
          logCallback(f"Error: {dimRedAlgo} failed")
        return False

      if progressCallback:
        progressCallback(90)

      if progressCallback:
        progressCallback(100)

      # Return structured data for UI layer to create plots
      # For HSV with filtering, we need to adjust the specimen information
      if colorSpace == "HSV" and dimRedData.shape[0] != colorData.shape[0]:
        # Calculate how many faces per specimen passed the filter
        filteredNFaces = dimRedData.shape[0] // nSpecimens if nSpecimens > 0 else 0
        plotNFaces = filteredNFaces
      else:
        plotNFaces = nFaces

      return {
        "success": True,
        "reducedData": reducedData,
        "specimenNames": specimenNames,
        "nFaces": plotNFaces,
        "colorSpace": colorSpace,
        "algorithm": dimRedAlgo,
        "originalColorData": dimRedData,  # Filtered data for plotting
        "colorData": colorData,  # Full dataset for histograms
        "satCutoff": satCutoff,
        "valueCutoff": valueCutoff,
        "enhanceColors": enhanceColors
      }

    except Exception as e:
      if logCallback:
        logCallback(f"Error in runColorsEDA: {str(e)}")
      import traceback
      traceback.print_exc()
      return False

  def sampleColorData(self, atlasModel, texturesDir, randomSeed, samplePercent, progressCallback=None, logCallback=None):
    """
    Sample faces and calculate color averages from textures

    Args:
        atlasModel: VTK model node of the atlas
        texturesDir: Directory containing baked atlas-space PNG textures
        randomSeed: Random seed for reproducible sampling
        samplePercent: Percentage of faces to sample (1-100)
        progressCallback: Function to call with progress updates (0-100)
        logCallback: Function to call with log messages

    Returns:
        dict: Result with success flag, colorData, specimenNames, and faceIndices
    """
    try:
      if logCallback:
        logCallback("Starting face sampling...")

      if progressCallback:
        progressCallback(5)

      # Get atlas polydata
      atlasPolyData = atlasModel.GetPolyData()
      if not atlasPolyData:
        if logCallback:
          logCallback("Error: Atlas model has no polydata")
        return {"success": False}

      # Get total number of faces
      nTotalFaces = atlasPolyData.GetNumberOfCells()

      if logCallback:
        logCallback(f"Atlas model has {nTotalFaces} faces")

      # Calculate number of faces to sample
      nSampleFaces = max(1, int(nTotalFaces * samplePercent / 100.0))
      if logCallback:
        logCallback(f"Sampling {nSampleFaces} faces ({samplePercent:.1f}%)")

      # Set random seed and sample face indices
      np.random.seed(randomSeed)
      if samplePercent >= 100.0:
        # Use all faces
        sampledFaceIndices = np.arange(nTotalFaces)
      else:
        # Randomly sample faces
        sampledFaceIndices = np.random.choice(nTotalFaces, size=nSampleFaces, replace=False)
        sampledFaceIndices = np.sort(sampledFaceIndices)  # Sort for consistent processing

      if progressCallback:
        progressCallback(15)

      # Get list of texture files
      textureFiles = []
      if os.path.isdir(texturesDir):
        for f in os.listdir(texturesDir):
          if f.lower().endswith(('.png', '.tiff', '.tif')) and not f.lower().startswith('average_texture'):
            textureFiles.append(os.path.join(texturesDir, f))

      if not textureFiles:
        if logCallback:
          logCallback(f"Error: No texture files found in {texturesDir}")
        return {"success": False}

      if logCallback:
        logCallback(f"Found {len(textureFiles)} texture files")

      # Process each texture and calculate face colors for sampled faces only
      allFaceColors = []
      specimenNames = []

      for i, texturePath in enumerate(textureFiles):
        if logCallback:
          logCallback(f"Processing texture {i+1}/{len(textureFiles)}: {os.path.basename(texturePath)}")

        # Load texture image
        try:
          textureImage = imageio.imread(texturePath)
          if len(textureImage.shape) != 3 or textureImage.shape[2] < 3:
            if logCallback:
              logCallback(f"Warning: Skipping {os.path.basename(texturePath)} - invalid format")
            continue
        except Exception as e:
          if logCallback:
            logCallback(f"Warning: Could not load {os.path.basename(texturePath)}: {e}")
          continue

        # Calculate face-averaged colors for sampled faces only
        faceColors = self._calculateSampledFaceAverageColors(atlasPolyData, textureImage, sampledFaceIndices, "RGB")
        if faceColors is not None:
          allFaceColors.append(faceColors)
          specimenNames.append(os.path.splitext(os.path.basename(texturePath))[0])

        if progressCallback:
          progressCallback(15 + int(80 * (i + 1) / len(textureFiles)))

      if not allFaceColors:
        if logCallback:
          logCallback("Error: No valid textures processed")
        return {"success": False}

      # Convert to numpy array
      allFaceColors = np.array(allFaceColors)  # Shape: (N_specimens, N_sampled_faces, 3)

      if logCallback:
        logCallback(f"Successfully sampled data from {len(allFaceColors)} specimens")
        logCallback(f"Sampled color data shape: {allFaceColors.shape}")

      if progressCallback:
        progressCallback(100)

      return {
        "success": True,
        "colorData": allFaceColors,
        "specimenNames": specimenNames,
        "faceIndices": sampledFaceIndices
      }

    except Exception as e:
      if logCallback:
        logCallback(f"Error in sampleColorData: {str(e)}")
      import traceback
      traceback.print_exc()
      return {"success": False}

  def _applyNeighborAveraging(self, polyData, faceColors, adjacency=None, logCallback=None):
    """
    Apply neighbor averaging to smooth face colors based on mesh topology.
    Each face's color is replaced with the average of its own color and its neighbors' colors.

    Args:
        polyData: VTK polydata object
        faceColors: numpy array of shape (numFaces, 3) with RGB colors
        adjacency: Pre-computed face adjacency graph (optional, will build if None)
        logCallback: Function to call with log messages

    Returns:
        numpy array of smoothed face colors, or None if failed
    """
    try:
      numFaces = polyData.GetNumberOfCells()

      if len(faceColors) != numFaces:
        if logCallback:
          logCallback(f"  Error: Face color count mismatch")
        return None

      # Use provided adjacency graph or build a new one
      if adjacency is None:
        adjacency = self._buildFaceAdjacencyGraph(polyData, logCallback=None)
        if adjacency is None:
          if logCallback:
            logCallback(f"  Error: Failed to build face adjacency graph")
          return None

      # Build sparse averaging matrix for vectorized smoothing
      try:
        from scipy import sparse
        rows = []
        cols = []
        for faceId in range(numFaces):
          neighbors = adjacency.get(faceId, set())
          allIds = [faceId] + list(neighbors)
          for nId in allIds:
            rows.append(faceId)
            cols.append(nId)
        data = np.ones(len(rows), dtype=np.float64)
        avgMatrix = sparse.csr_matrix((data, (rows, cols)), shape=(numFaces, numFaces))
        # Normalize each row to compute the average
        rowSums = np.array(avgMatrix.sum(axis=1)).ravel()
        rowSums[rowSums == 0] = 1  # avoid division by zero
        diagInv = sparse.diags(1.0 / rowSums)
        avgMatrix = diagInv @ avgMatrix
        smoothedColors = avgMatrix @ faceColors
      except ImportError:
        # Fallback without scipy sparse
        smoothedColors = np.zeros_like(faceColors)
        for faceId in range(numFaces):
          neighbors = adjacency.get(faceId, set())
          allIds = [faceId] + list(neighbors)
          smoothedColors[faceId] = np.mean(faceColors[allIds], axis=0)

      if logCallback:
        logCallback(f"  Neighbor averaging complete: smoothed {numFaces} faces")

      return smoothedColors

    except Exception as e:
      if logCallback:
        logCallback(f"  Error in neighbor averaging: {str(e)}")
      import traceback
      traceback.print_exc()
      return None

  def _extractFaceConnectivity(self, polyData):
    """
    Extract face connectivity as a numpy array from VTK polydata.

    Returns:
        faces: (N_faces, 3) int array of vertex indices (triangles only),
               or None if non-triangular cells are present.
    """
    polys = polyData.GetPolys()
    if polys is None or polys.GetNumberOfCells() == 0:
      return None
    rawArray = vtk_np.vtk_to_numpy(polys.GetData())
    nFaces = polys.GetNumberOfCells()
    # For triangle meshes: rawArray is [3, v0, v1, v2, 3, v0, v1, v2, ...]
    # Check stride: total length should be nFaces * 4 for triangles
    if len(rawArray) == nFaces * 4:
      return rawArray.reshape(nFaces, 4)[:, 1:4]
    # For quad meshes: rawArray is [4, v0, v1, v2, v3, ...]
    if len(rawArray) == nFaces * 5:
      return rawArray.reshape(nFaces, 5)[:, 1:5]
    # Mixed cell types: fall back to None (caller should use per-cell iteration)
    return None

  def _calculateFaceAverageColors(self, polyData, textureImage, colorSpace):
    """
    Calculate average color for each face of the mesh using texture coordinates

    Args:
        polyData: VTK polydata of the atlas model
        textureImage: numpy array of the texture image (H, W, C)
        colorSpace: "RGB" or "HSV"

    Returns:
        numpy array of shape (N_faces, 3) for RGB or (N_faces, 4) for HSV
        HSV returns [hue_cos, hue_sin, saturation, value] to handle circular hue
    """
    try:
      # Get texture coordinates
      tcoords = polyData.GetPointData().GetTCoords()
      if not tcoords:
        return None

      tcoords_np = vtk_np.vtk_to_numpy(tcoords)

      # Get texture image dimensions
      height, width = textureImage.shape[:2]

      # Try vectorized path for triangle meshes
      faces = self._extractFaceConnectivity(polyData)
      if faces is not None and faces.shape[1] == 3:
        # Vectorized: sample texture at all vertex UV positions at once
        u = np.clip(tcoords_np[:, 0], 0, 1)
        v = np.clip(1.0 - tcoords_np[:, 1], 0, 1)  # Flip V
        px = np.clip((u * (width - 1)).astype(int), 0, width - 1)
        py = np.clip((v * (height - 1)).astype(int), 0, height - 1)
        vertexColors = textureImage[py, px, :3].astype(np.float64)

        # Average vertex colors per face: (nFaces, 3_vertices, 3_channels) -> (nFaces, 3_channels)
        faceColors = np.mean(vertexColors[faces], axis=1)

        if colorSpace == "HSV":
          # Vectorized RGB to HSV conversion
          rgb_norm = faceColors / 255.0
          r, g, b = rgb_norm[:, 0], rgb_norm[:, 1], rgb_norm[:, 2]
          maxc = np.maximum(np.maximum(r, g), b)
          minc = np.minimum(np.minimum(r, g), b)
          diff = maxc - minc

          # Hue calculation
          hue = np.zeros(len(faceColors))
          mask_r = (maxc == r) & (diff > 0)
          mask_g = (maxc == g) & (diff > 0)
          mask_b = (maxc == b) & (diff > 0)
          hue[mask_r] = ((g[mask_r] - b[mask_r]) / diff[mask_r]) % 6.0
          hue[mask_g] = ((b[mask_g] - r[mask_g]) / diff[mask_g]) + 2.0
          hue[mask_b] = ((r[mask_b] - g[mask_b]) / diff[mask_b]) + 4.0
          hue = hue / 6.0  # Normalize to [0, 1]

          # Saturation
          sat = np.where(maxc > 0, diff / maxc, 0.0)

          # Convert hue to cos/sin for circular representation
          hue_radians = hue * 2 * np.pi
          hue_cos = np.cos(hue_radians)
          hue_sin = np.sin(hue_radians)

          faceColors = np.column_stack([hue_cos, hue_sin, sat * 100, maxc * 100])

        return faceColors

      # Fallback: per-face loop for non-triangle meshes
      polys = polyData.GetPolys()
      nFaces = polys.GetNumberOfCells()
      faceColorsList = []

      for faceIdx in range(nFaces):
        cell = polyData.GetCell(faceIdx)
        nPoints = cell.GetNumberOfPoints()
        vertexIndices = [cell.GetPointId(ptIdx) for ptIdx in range(nPoints)]

        faceTexCoords = tcoords_np[vertexIndices].copy()
        faceTexCoords[:, 1] = 1.0 - faceTexCoords[:, 1]
        pixelCoords = np.clip(faceTexCoords, 0, 1) * [width - 1, height - 1]
        pixelCoords = pixelCoords.astype(int)
        facePixelColors = textureImage[pixelCoords[:, 1], pixelCoords[:, 0], :3]
        avgColor = np.mean(facePixelColors, axis=0)

        if colorSpace == "HSV":
          rgb_normalized = avgColor / 255.0
          hsv = colorsys.rgb_to_hsv(rgb_normalized[0], rgb_normalized[1], rgb_normalized[2])
          hue_radians = hsv[0] * 2 * np.pi
          hue_cos = np.cos(hue_radians)
          hue_sin = np.sin(hue_radians)
          avgColor = np.array([hue_cos, hue_sin, hsv[1] * 100, hsv[2] * 100])

        faceColorsList.append(avgColor)

      return np.array(faceColorsList)

    except Exception as e:
      print(f"Error calculating face colors: {e}")
      return None

  def _calculateSampledFaceAverageColors(self, polyData, textureImage, faceIndices, colorSpace):
    """
    Calculate average color for specific faces of the mesh using texture coordinates

    Args:
        polyData: VTK polydata of the atlas model
        textureImage: numpy array of the texture image (H, W, C)
        faceIndices: numpy array of face indices to process
        colorSpace: "RGB" or "HSV"

    Returns:
        numpy array of shape (N_sampled_faces, 3) for RGB or (N_sampled_faces, 4) for HSV
    """
    try:
      # Get texture coordinates
      tcoords = polyData.GetPointData().GetTCoords()
      if not tcoords:
        return None

      tcoords_np = vtk_np.vtk_to_numpy(tcoords)

      # Get texture image dimensions
      height, width = textureImage.shape[:2]

      # Try vectorized path for triangle meshes
      allFaces = self._extractFaceConnectivity(polyData)
      if allFaces is not None and allFaces.shape[1] == 3:
        # Get the subset of faces
        sampledFaces = allFaces[faceIndices]  # (N_sampled, 3)

        # Sample texture at all vertex UV positions at once
        u = np.clip(tcoords_np[:, 0], 0, 1)
        v = np.clip(1.0 - tcoords_np[:, 1], 0, 1)  # Flip V
        px = np.clip((u * (width - 1)).astype(int), 0, width - 1)
        py = np.clip((v * (height - 1)).astype(int), 0, height - 1)
        vertexColors = textureImage[py, px, :3].astype(np.float64)

        # Average vertex colors per sampled face
        faceColors = np.mean(vertexColors[sampledFaces], axis=1)

        if colorSpace == "HSV":
          # Vectorized RGB to HSV conversion
          rgb_norm = faceColors / 255.0
          r, g, b = rgb_norm[:, 0], rgb_norm[:, 1], rgb_norm[:, 2]
          maxc = np.maximum(np.maximum(r, g), b)
          minc = np.minimum(np.minimum(r, g), b)
          diff = maxc - minc

          hue = np.zeros(len(faceColors))
          mask_r = (maxc == r) & (diff > 0)
          mask_g = (maxc == g) & (diff > 0)
          mask_b = (maxc == b) & (diff > 0)
          hue[mask_r] = ((g[mask_r] - b[mask_r]) / diff[mask_r]) % 6.0
          hue[mask_g] = ((b[mask_g] - r[mask_g]) / diff[mask_g]) + 2.0
          hue[mask_b] = ((r[mask_b] - g[mask_b]) / diff[mask_b]) + 4.0
          hue = hue / 6.0

          sat = np.where(maxc > 0, diff / maxc, 0.0)
          hue_radians = hue * 2 * np.pi
          hue_cos = np.cos(hue_radians)
          hue_sin = np.sin(hue_radians)

          faceColors = np.column_stack([hue_cos, hue_sin, sat * 100, maxc * 100])

        return faceColors

      # Fallback: per-face loop for non-triangle meshes
      faceColorsList = []
      for faceIdx in faceIndices:
        cell = polyData.GetCell(int(faceIdx))
        nPoints = cell.GetNumberOfPoints()
        vertexIndices = [cell.GetPointId(ptIdx) for ptIdx in range(nPoints)]

        faceTexCoords = tcoords_np[vertexIndices].copy()
        faceTexCoords[:, 1] = 1.0 - faceTexCoords[:, 1]
        pixelCoords = np.clip(faceTexCoords, 0, 1) * [width - 1, height - 1]
        pixelCoords = pixelCoords.astype(int)
        facePixelColors = textureImage[pixelCoords[:, 1], pixelCoords[:, 0], :3]
        avgColor = np.mean(facePixelColors, axis=0)

        if colorSpace == "HSV":
          rgb_normalized = avgColor / 255.0
          hsv = colorsys.rgb_to_hsv(rgb_normalized[0], rgb_normalized[1], rgb_normalized[2])
          hue_radians = hsv[0] * 2 * np.pi
          avgColor = np.array([np.cos(hue_radians), np.sin(hue_radians), hsv[1] * 100, hsv[2] * 100])

        faceColorsList.append(avgColor)

      return np.array(faceColorsList)

    except Exception as e:
      print(f"Error calculating sampled face colors: {e}")
      return None

  def applyAverageFaceColorsFromTexture(self, modelNode, texturePath, progressCallback=None, logCallback=None):
    """
    Apply average face colors from a texture to a model

    Args:
        modelNode: VTK model node to apply colors to
        texturePath: Path to the texture image file
        progressCallback: Function to call with progress updates (0-100)
        logCallback: Function to call with log messages

    Returns:
        bool: True if successful, False otherwise
    """
    try:
      if logCallback:
        logCallback(f"Loading texture: {os.path.basename(texturePath)}")

      if progressCallback:
        progressCallback(10)

      # Load texture image
      try:
        textureImage = imageio.imread(texturePath)
        if len(textureImage.shape) != 3 or textureImage.shape[2] < 3:
          if logCallback:
            logCallback("Error: Invalid texture format")
          return False
      except Exception as e:
        if logCallback:
          logCallback(f"Error loading texture: {e}")
        return False

      if progressCallback:
        progressCallback(30)

      # Get model polydata
      polyData = modelNode.GetPolyData()
      if not polyData:
        if logCallback:
          logCallback("Error: No polydata in model")
        return False

      if logCallback:
        logCallback("Calculating face average colors...")

      if progressCallback:
        progressCallback(50)

      # Calculate face average colors
      faceColors = self._calculateFaceAverageColors(polyData, textureImage, "RGB")
      if faceColors is None:
        if logCallback:
          logCallback("Error: Failed to calculate face colors")
        return False

      if logCallback:
        logCallback(f"Calculated colors for {len(faceColors)} faces")
        logCallback(f"Sample colors: {faceColors[:3] if len(faceColors) > 0 else 'None'}")

      if progressCallback:
        progressCallback(70)

      # Apply colors to faces as cell data
      nFaces = polyData.GetNumberOfCells()
      if len(faceColors) != nFaces:
        if logCallback:
          logCallback(f"Error: Color count mismatch. Expected {nFaces}, got {len(faceColors)}")
        return False

      if logCallback:
        logCallback(f"Applying colors to {nFaces} faces")

      # Create VTK color array for RGB colors
      colorArray = vtk.vtkUnsignedCharArray()
      colorArray.SetNumberOfComponents(3)
      colorArray.SetName("FaceColors")
      colorArray.SetNumberOfTuples(nFaces)

      if logCallback:
        logCallback("Creating color array...")

      for i, color in enumerate(faceColors):
        # Ensure color values are in 0-255 range
        color_255 = np.clip(color, 0, 255).astype(np.uint8)
        colorArray.SetTuple3(i, int(color_255[0]), int(color_255[1]), int(color_255[2]))

      if logCallback:
        logCallback(f"Color array created with {colorArray.GetNumberOfTuples()} tuples")

      # Add color array to cell data
      polyData.GetCellData().SetScalars(colorArray)
      polyData.Modified()

      if logCallback:
        logCallback("Color array added to cell data")

      if progressCallback:
        progressCallback(90)

      # Update display to show colors
      displayNode = modelNode.GetDisplayNode()
      if displayNode:
        if logCallback:
          logCallback("Configuring display node...")

        # Turn off texture first
        displayNode.SetTextureImageDataConnection(None)

        # Enable scalar visibility and set to use RGB colors directly
        displayNode.SetScalarVisibility(True)
        displayNode.SetActiveScalarName("FaceColors")

        # Set to use cell data (not point data)
        displayNode.SetActiveAttributeLocation(vtk.vtkDataObject.CELL)

        # Use RGB color mode instead of lookup table
        displayNode.SetScalarRangeFlag(slicer.vtkMRMLDisplayNode.UseDirectMapping)

        # Clear any existing color node to use direct RGB values
        # displayNode.SetAndObserveColorNodeID(None)  # This causes errors, skip it

        # Force update
        displayNode.Modified()

        if logCallback:
          logCallback("Display node configured for face colors")
      else:
        if logCallback:
          logCallback("Warning: No display node found")

      if progressCallback:
        progressCallback(100)

      if logCallback:
        logCallback(f"Successfully applied average face colors from {len(faceColors)} faces")

      return True

    except Exception as e:
      if logCallback:
        logCallback(f"Error in applyAverageFaceColorsFromTexture: {str(e)}")
      import traceback
      traceback.print_exc()
      return False

  def applyAverageFaceColorsFromTextureAlternative(self, modelNode, texturePath, progressCallback=None, logCallback=None):
    """
    Alternative method for applying average face colors using point data interpolation
    This method converts face colors to point colors which might display better in Slicer
    """
    try:
      if logCallback:
        logCallback(f"Loading texture (alternative method): {os.path.basename(texturePath)}")

      # Load texture image
      try:
        textureImage = imageio.imread(texturePath)
        if len(textureImage.shape) != 3 or textureImage.shape[2] < 3:
          if logCallback:
            logCallback("Error: Invalid texture format")
          return False
      except Exception as e:
        if logCallback:
          logCallback(f"Error loading texture: {e}")
        return False

      # Get model polydata
      polyData = modelNode.GetPolyData()
      if not polyData:
        if logCallback:
          logCallback("Error: No polydata in model")
        return False

      # Calculate face average colors
      faceColors = self._calculateFaceAverageColors(polyData, textureImage, "RGB")
      if faceColors is None:
        if logCallback:
          logCallback("Error: Failed to calculate face colors")
        return False

      if logCallback:
        logCallback(f"Converting {len(faceColors)} face colors to point colors...")

      # Convert face colors to point colors by averaging adjacent face colors
      nPoints = polyData.GetNumberOfPoints()
      nFaces = polyData.GetNumberOfCells()

      pointColors = np.zeros((nPoints, 3))
      pointCounts = np.zeros(nPoints)

      # For each face, add its color to all its vertices
      for faceIdx in range(nFaces):
        cell = polyData.GetCell(faceIdx)
        nCellPoints = cell.GetNumberOfPoints()

        for ptIdx in range(nCellPoints):
          pointId = cell.GetPointId(ptIdx)
          pointColors[pointId] += faceColors[faceIdx]
          pointCounts[pointId] += 1

      # Average the colors for each point
      for ptIdx in range(nPoints):
        if pointCounts[ptIdx] > 0:
          pointColors[ptIdx] /= pointCounts[ptIdx]

      # Create VTK color array for point data
      colorArray = vtk.vtkUnsignedCharArray()
      colorArray.SetNumberOfComponents(3)
      colorArray.SetName("PointColors")
      colorArray.SetNumberOfTuples(nPoints)

      for i, color in enumerate(pointColors):
        color_255 = np.clip(color, 0, 255).astype(np.uint8)
        colorArray.SetTuple3(i, int(color_255[0]), int(color_255[1]), int(color_255[2]))

      # Add color array to point data
      polyData.GetPointData().SetScalars(colorArray)
      polyData.Modified()

      # Update display
      displayNode = modelNode.GetDisplayNode()
      if displayNode:
        displayNode.SetTextureImageDataConnection(None)
        displayNode.SetScalarVisibility(True)
        displayNode.SetActiveScalarName("PointColors")
        displayNode.SetActiveAttributeLocation(vtk.vtkDataObject.POINT)
        displayNode.SetScalarRangeFlag(slicer.vtkMRMLDisplayNode.UseDirectMapping)
        # displayNode.SetAndObserveColorNodeID(None)  # This causes errors, skip it
        displayNode.Modified()

      if logCallback:
        logCallback("Successfully applied average face colors using point data method")

      return True

    except Exception as e:
      if logCallback:
        logCallback(f"Error in alternative face coloring: {str(e)}")
      import traceback
      traceback.print_exc()
      return False

  def _applyDimensionalityReduction(self, colorData, algorithm):
    """
    Apply dimensionality reduction to color data

    Args:
        colorData: numpy array of shape (N_samples, 3) for RGB or (N_samples, 4) for HSV
        algorithm: "PCA", "ICA", or "UMAP"

    Returns:
        numpy array of shape (N_samples, 2) with reduced dimensions
    """
    try:
      if algorithm == "PCA":
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2)
        return reducer.fit_transform(colorData)

      elif algorithm == "ICA":
        from sklearn.decomposition import FastICA
        reducer = FastICA(n_components=2, random_state=42)
        return reducer.fit_transform(colorData)

      elif algorithm == "UMAP":
        import umap
        reducer = umap.UMAP(n_components=2, random_state=42)
        return reducer.fit_transform(colorData)

      else:
        print(f"Unknown algorithm: {algorithm}")
        return None

    except Exception as e:
      print(f"Error in dimensionality reduction: {e}")
      return None







  def runColorsEDAFromSampledData(self, sampledColorData, specimenNames, colorSpace, dimRedAlgo, progressCallback=None, logCallback=None, satCutoff=10.0, valueCutoff=10.0, enhanceColors=False):
    """
    Run color analysis with dimensionality reduction on pre-sampled color data

    Args:
        sampledColorData: numpy array of shape (N_specimens, N_sampled_faces, 3)
        specimenNames: list of specimen names
        colorSpace: "RGB" or "HSV"
        dimRedAlgo: "PCA", "ICA", or "UMAP"
        progressCallback: Function to call with progress updates (0-100)
        logCallback: Function to call with log messages
        satCutoff: Minimum saturation threshold for HSV filtering (0-100)
        valueCutoff: Minimum value/brightness threshold for HSV filtering (0-100)
        enhanceColors: Whether to enhance colors for visibility in 2D plots

    Returns:
        dict: Result with success flag, colorData, colorSpace, and chartNode
    """
    try:
      if logCallback:
        logCallback("Starting analysis on sampled data...")

      if progressCallback:
        progressCallback(10)

      # Check dependencies
      if dimRedAlgo in ["PCA", "ICA"] and not SKLEARN_AVAILABLE:
        if logCallback:
          logCallback("Error: sklearn not available for PCA/ICA")
        return {"success": False}

      if dimRedAlgo == "UMAP" and not UMAP_AVAILABLE:
        if logCallback:
          logCallback("Error: umap-learn not available for UMAP")
        return {"success": False}

      # Convert sampled data to the format expected by analysis
      nSpecimens, nSampledFaces, nChannels = sampledColorData.shape

      # Convert color space if needed
      if colorSpace == "HSV":
        if logCallback:
          logCallback("Converting RGB sampled data to HSV...")

        # Convert each specimen's data from RGB to HSV
        hsvColorData = []
        for specIdx in range(nSpecimens):
          specRgbData = sampledColorData[specIdx]  # Shape: (N_sampled_faces, 3)
          specHsvData = []

          for faceIdx in range(nSampledFaces):
            rgb = specRgbData[faceIdx] / 255.0  # Normalize to 0-1
            hsv = colorsys.rgb_to_hsv(rgb[0], rgb[1], rgb[2])
            # Convert hue to 2D vector (cos, sin) to handle circular nature
            hue_radians = hsv[0] * 2 * np.pi
            hue_cos = np.cos(hue_radians)
            hue_sin = np.sin(hue_radians)
            # Create 4D vector: [hue_cos, hue_sin, saturation, value]
            hsv_vec = np.array([hue_cos, hue_sin, hsv[1] * 100, hsv[2] * 100])
            specHsvData.append(hsv_vec)

          hsvColorData.append(np.array(specHsvData))

        colorData = np.array(hsvColorData)  # Shape: (N_specimens, N_sampled_faces, 4)
        nChannels = 4
      else:
        colorData = sampledColorData

      # Reshape to (N_sampled_faces * N_specimens, nChannels)
      colorDataFlat = colorData.reshape(-1, nChannels)

      if logCallback:
        logCallback(f"Color data shape: {colorDataFlat.shape}")

      if progressCallback:
        progressCallback(30)

      # Apply dimensionality reduction with optional HSV filtering
      if logCallback:
        logCallback(f"Applying {dimRedAlgo} dimensionality reduction...")

      # Filter data for dimensionality reduction if in HSV mode
      dimRedData = colorDataFlat
      if colorSpace == "HSV":
        # Extract saturation and value channels (indices 2 and 3)
        sat = colorDataFlat[:, 2]  # 0..100
        val = colorDataFlat[:, 3]  # 0..100

        # Create mask for saturation and value cutoffs
        mask = (sat >= satCutoff) & (val >= valueCutoff)
        dimRedData = colorDataFlat[mask]

        if logCallback:
          logCallback(f"HSV filtering: {np.sum(mask)}/{len(mask)} samples passed cutoffs (sat>={satCutoff}, val>={valueCutoff})")
          logCallback(f"Filtered data shape for dim reduction: {dimRedData.shape}")

      reducedData = self._applyDimensionalityReduction(dimRedData, dimRedAlgo)

      if reducedData is None:
        if logCallback:
          logCallback(f"Error: {dimRedAlgo} failed")
        return {"success": False}

      if progressCallback:
        progressCallback(70)

      if progressCallback:
        progressCallback(100)

      # Return structured data for UI layer to create plots
      # For HSV with filtering, we need to adjust the specimen information
      if colorSpace == "HSV" and dimRedData.shape[0] != colorDataFlat.shape[0]:
        # Calculate how many faces per specimen passed the filter
        filteredNFaces = dimRedData.shape[0] // nSpecimens if nSpecimens > 0 else 0
        plotNFaces = filteredNFaces
      else:
        plotNFaces = nSampledFaces

      return {
        "success": True,
        "reducedData": reducedData,
        "specimenNames": specimenNames,
        "nFaces": plotNFaces,
        "colorSpace": colorSpace,
        "algorithm": dimRedAlgo,
        "originalColorData": dimRedData,  # Filtered data for plotting
        "colorData": colorDataFlat,  # Full dataset for histograms
        "satCutoff": satCutoff,
        "valueCutoff": valueCutoff,
        "enhanceColors": enhanceColors
      }

    except Exception as e:
      if logCallback:
        logCallback(f"Error in runColorsEDAFromSampledData: {str(e)}")
      import traceback
      traceback.print_exc()
      return {"success": False}

  ################################### Color Quantization Functions ###################################

  def rgb_to_lab(self, rgb):
    """
    Convert RGB color to CIE Lab color space using scikit-image

    Args:
        rgb: numpy array of shape (..., 3) with RGB values in range [0, 255]

    Returns:
        numpy array of shape (..., 3) with Lab values
    """
    if not SKIMAGE_AVAILABLE:
      raise ImportError("scikit-image is required for color space conversion")

    # Normalize RGB to [0, 1] for scikit-image
    rgb_normalized = np.array(rgb, dtype=np.float64) / 255.0

    # Use scikit-image for accurate RGB to Lab conversion
    lab = skimage_color.rgb2lab(rgb_normalized)

    return lab

  def delta_e_2000(self, lab1, lab2):
    """
    Calculate ΔE2000 color difference between two Lab colors using scikit-image

    Args:
        lab1, lab2: numpy arrays of shape (..., 3) with Lab values

    Returns:
        numpy array of ΔE2000 values
    """
    if not SKIMAGE_AVAILABLE:
      raise ImportError("scikit-image is required for ΔE2000 calculation")

    # Use scikit-image's optimized ΔE2000 implementation
    return deltaE_ciede2000(lab1, lab2)

  def generate_high_contrast_palette(self, n_colors, logCallback=None):
    """
    Generate a high contrast color palette with up to 64 distinguishable colors

    Args:
        n_colors: Number of colors to generate (2-64)
        logCallback: Optional callback for logging messages

    Returns:
        numpy array of shape (n_colors, 3) with RGB values in range [0, 255]
    """
    if logCallback:
      logCallback(f"Generating high contrast palette with {n_colors} colors")

    # Clamp to valid range
    n_colors = max(2, min(64, n_colors))

    # Base high contrast colors (carefully chosen for maximum distinguishability)
    base_colors = [
      [255, 0, 0],     # Red
      [0, 255, 0],     # Green
      [0, 0, 255],     # Blue
      [255, 255, 0],   # Yellow
      [255, 0, 255],   # Magenta
      [0, 255, 255],   # Cyan
      [255, 128, 0],   # Orange
      [128, 0, 255],   # Purple
      [0, 128, 255],   # Light Blue
      [255, 0, 128],   # Pink
      [128, 255, 0],   # Lime
      [0, 255, 128],   # Spring Green
      [255, 255, 255], # White
      [0, 0, 0],       # Black
      [128, 128, 128], # Gray
      [192, 192, 192], # Light Gray
      [64, 64, 64],    # Dark Gray
      [128, 64, 0],    # Brown
      [64, 128, 0],    # Olive
      [0, 64, 128],    # Navy
      [128, 0, 64],    # Maroon
      [64, 0, 128],    # Indigo
      [0, 128, 64],    # Teal
      [255, 192, 128], # Peach
      [128, 255, 192], # Mint
      [192, 128, 255], # Lavender
      [255, 128, 192], # Rose
      [128, 192, 255], # Sky Blue
      [192, 255, 128], # Pale Green
      [255, 64, 64],   # Bright Red
      [64, 255, 64],   # Bright Green
      [64, 64, 255],   # Bright Blue
    ]

    if n_colors <= len(base_colors):
      # Use the first n_colors from our base palette
      return np.array(base_colors[:n_colors], dtype=np.uint8)

    # For more than 32 colors, generate additional colors using HSV space
    colors = base_colors.copy()

    # Generate additional colors by varying hue, saturation, and value systematically
    remaining = n_colors - len(colors)

    # Use HSV space to generate well-spaced colors
    for i in range(remaining):
      # Calculate hue with golden ratio spacing for good distribution
      golden_ratio = (1 + 5**0.5) / 2
      hue = (i * 360 / golden_ratio) % 360

      # Alternate between high and medium saturation/value for contrast
      if i % 4 == 0:
        sat, val = 1.0, 0.9  # Bright colors
      elif i % 4 == 1:
        sat, val = 0.7, 1.0  # Pastel colors
      elif i % 4 == 2:
        sat, val = 1.0, 0.6  # Dark colors
      else:
        sat, val = 0.5, 0.8  # Muted colors

      # Convert HSV to RGB
      rgb = colorsys.hsv_to_rgb(hue/360.0, sat, val)
      rgb_255 = [int(c * 255) for c in rgb]
      colors.append(rgb_255)

    if logCallback:
      logCallback(f"Generated {len(colors)} high contrast colors")

    return np.array(colors[:n_colors], dtype=np.uint8)

  def quantize_colors_lab_kmeans(self, rgb_colors, n_clusters, use_high_contrast=False, progressCallback=None, logCallback=None):
    """
    Quantize colors using k-means clustering in CIE Lab color space with ΔE2000 distance

    Args:
        rgb_colors: numpy array of shape (N, 3) with RGB values in range [0, 255]
        n_clusters: number of color clusters (2-64)
        use_high_contrast: if True, use high contrast palette instead of quantized colors
        progressCallback: Function to call with progress updates (0-100)
        logCallback: Function to call with log messages

    Returns:
        dict with 'success', 'quantized_colors', 'cluster_centers', 'labels'
    """
    try:
      if logCallback:
        logCallback(f"Starting color quantization with {n_clusters} clusters...")

      if progressCallback:
        progressCallback(10)

      # Check if required libraries are available
      if not SKLEARN_AVAILABLE:
        if logCallback:
          logCallback("Error: sklearn not available for k-means clustering")
        return {"success": False}

      if not SKIMAGE_AVAILABLE:
        if logCallback:
          logCallback("Error: scikit-image not available for color space conversion")
        return {"success": False}

      # Convert RGB to Lab
      if logCallback:
        logCallback("Converting RGB to CIE Lab color space...")

      lab_colors = self.rgb_to_lab(rgb_colors)

      if progressCallback:
        progressCallback(30)

      # Perform k-means clustering in Lab space
      if logCallback:
        logCallback(f"Performing k-means clustering with {n_clusters} clusters...")

      kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
      cluster_labels = kmeans.fit_predict(lab_colors)
      cluster_centers_lab = kmeans.cluster_centers_

      if progressCallback:
        progressCallback(70)

      # Convert cluster centers back to RGB or use high contrast palette
      if use_high_contrast:
        if logCallback:
          logCallback("Using high contrast palette instead of quantized colors...")

        # Generate high contrast palette
        cluster_centers_rgb = self.generate_high_contrast_palette(n_clusters, logCallback)

        # Create quantized colors by mapping each original color to its assigned high contrast color
        quantized_colors = cluster_centers_rgb[cluster_labels]
      else:
        if logCallback:
          logCallback("Converting cluster centers back to RGB...")

        cluster_centers_rgb = self.lab_to_rgb(cluster_centers_lab)

        # Create quantized colors by mapping each original color to its cluster center
        quantized_colors = cluster_centers_rgb[cluster_labels]

      if progressCallback:
        progressCallback(90)

      if logCallback:
        logCallback(f"Quantization complete. Reduced {len(rgb_colors)} colors to {n_clusters} clusters.")
        # Log some statistics
        unique_original = len(np.unique(rgb_colors.view(np.void), axis=0))
        logCallback(f"Original unique colors: {unique_original}, Quantized to: {n_clusters}")

      if progressCallback:
        progressCallback(100)

      return {
        "success": True,
        "quantized_colors": quantized_colors.astype(np.uint8),
        "cluster_centers": cluster_centers_rgb.astype(np.uint8),
        "labels": cluster_labels,
        "original_colors": rgb_colors
      }

    except Exception as e:
      if logCallback:
        logCallback(f"Error in color quantization: {str(e)}")
      import traceback
      traceback.print_exc()
      return {"success": False}

  def lab_to_rgb(self, lab):
    """
    Convert CIE Lab color to RGB color space using scikit-image

    Args:
        lab: numpy array of shape (..., 3) with Lab values

    Returns:
        numpy array of shape (..., 3) with RGB values in range [0, 255]
    """
    if not SKIMAGE_AVAILABLE:
      raise ImportError("scikit-image is required for color space conversion")

    # Use scikit-image for accurate Lab to RGB conversion
    rgb_normalized = skimage_color.lab2rgb(lab)

    # Convert from [0, 1] to [0, 255] and clamp
    rgb = np.clip(rgb_normalized * 255, 0, 255)

    return rgb

  def performMultiTextureClustering(self, modelNode, textureDir, textureFiles, initialClusters, consolidatedClusters,
                                     numSubsampledFaces=None, normalizeLuminosity=False, useNeighborAverage=False, faceAreas=None, progressCallback=None, logCallback=None):
    """
    Perform per-texture clustering with hierarchical consolidation and cross-texture reordering

    Args:
        modelNode: VTK model node to analyze
        textureDir: Directory containing texture files
        textureFiles: List of texture filenames
        initialClusters: Number of initial clusters per texture
        consolidatedClusters: Number of consolidated clusters (must be <= initialClusters)
        numSubsampledFaces: Number of faces to subsample for clustering (optional, uses all if None)
        normalizeLuminosity: Whether to normalize L* and C* across textures
        useNeighborAverage: Whether to use average color of neighboring faces for clustering
        faceAreas: Pre-computed face areas (optional, will calculate if None)
        progressCallback: Function to call with progress updates (0-100)
        logCallback: Function to call with log messages

    Returns:
        dict with 'success', 'pipeline' (ClusteringPipeline object), 'face_areas'
    """
    try:
      if logCallback:
        logCallback(f"Starting per-texture clustering pipeline...")
        logCallback(f"Initial clusters: {initialClusters}, Consolidated: {consolidatedClusters}")
        logCallback(f"Luminosity normalization: {'enabled' if normalizeLuminosity else 'disabled'}")

      if progressCallback:
        progressCallback(2)

      # Check if required libraries are available
      if not SKLEARN_AVAILABLE:
        if logCallback:
          logCallback("Error: sklearn not available for clustering")
        return {"success": False}

      if not SCIPY_AVAILABLE:
        if logCallback:
          logCallback("Error: scipy not available for hierarchical clustering")
        return {"success": False}

      if not SKIMAGE_AVAILABLE:
        if logCallback:
          logCallback("Error: scikit-image not available for color space conversion")
        return {"success": False}

      # Get model polydata
      polyData = modelNode.GetPolyData()
      if not polyData:
        if logCallback:
          logCallback("Error: No polydata in model")
        return {"success": False}

      # Use provided face areas or calculate them
      if faceAreas is None:
        if logCallback:
          logCallback("Calculating face areas...")

        faceAreas = self._calculateFaceAreas(polyData)
        if faceAreas is None:
          if logCallback:
            logCallback("Error: Failed to calculate face areas")
          return {"success": False}
      else:
        if logCallback:
          logCallback("Using provided face areas")

      if progressCallback:
        progressCallback(5)

      # Build face adjacency graph if needed (for subsampling or neighbor averaging)
      faceAdjacency = None
      if (numSubsampledFaces is not None and numSubsampledFaces > 0) or useNeighborAverage:
        if logCallback:
          logCallback("Building face adjacency graph...")

        # Clean mesh first
        polyData = self._cleanMesh(polyData, logCallback)

        # Build adjacency
        faceAdjacency = self._buildFaceAdjacencyGraph(polyData, logCallback)
        if faceAdjacency is None:
          if logCallback:
            logCallback("Error: Failed to build face adjacency graph")
          return {"success": False}

      # Subsample faces if requested
      subsampledFaceIndices = None
      nearestNeighborMapping = None
      if numSubsampledFaces is not None and numSubsampledFaces > 0:
        if logCallback:
          logCallback(f"Performing face subsampling...")

        # Use pre-built adjacency if available, otherwise build it
        if faceAdjacency is None:
          subsampledFaceIndices, nearestNeighborMapping, faceAdjacency = self._subsampleFacesUniformly(
            polyData, numSubsampledFaces, logCallback
          )
        else:
          # Reuse the pre-built adjacency graph
          subsampledFaceIndices, nearestNeighborMapping, _ = self._subsampleFacesUniformly(
            polyData, numSubsampledFaces, logCallback
          )

        if subsampledFaceIndices is None:
          if logCallback:
            logCallback("Error: Face subsampling failed")
          return {"success": False}

      # Create clustering pipeline
      pipeline = ClusteringPipeline(initialClusters, consolidatedClusters, normalizeLuminosity, useNeighborAverage)

      # Store subsampling information in pipeline
      pipeline.subsampledFaceIndices = subsampledFaceIndices
      pipeline.nearestNeighborMapping = nearestNeighborMapping
      pipeline.faceAdjacency = faceAdjacency

      # Step 1: Compute pooled LC statistics if normalization is enabled
      if normalizeLuminosity:
        if logCallback:
          logCallback("Computing pooled L*C* statistics for luminosity normalization...")

        pooledStats = self._computePooledLCStats(textureDir, textureFiles, logCallback)
        if pooledStats is None:
          if logCallback:
            logCallback("Error: Failed to compute pooled LC statistics")
          return {"success": False}

        pipeline.pooledLCStats = pooledStats
        if logCallback:
          mu_pool, sd_pool = pooledStats
          logCallback(f"Pooled stats - L: μ={mu_pool[0]:.2f} σ={sd_pool[0]:.2f}, C: μ={mu_pool[1]:.2f} σ={sd_pool[1]:.2f}")

      if progressCallback:
        progressCallback(10)

      # Step 2: Process each texture
      if logCallback:
        logCallback(f"Processing {len(textureFiles)} textures...")

      for i, textureFile in enumerate(textureFiles):
        texturePath = os.path.join(textureDir, textureFile)

        if logCallback:
          logCallback(f"\nTexture {i+1}/{len(textureFiles)}: {textureFile}")

        try:
          # Load texture image
          textureImage = imageio.imread(texturePath)
          if len(textureImage.shape) != 3 or textureImage.shape[2] < 3:
            if logCallback:
              logCallback(f"  Warning: Skipping invalid texture format")
            continue
        except Exception as e:
          if logCallback:
            logCallback(f"  Warning: Failed to load texture: {e}")
          continue

        # Calculate face average colors for this texture
        faceColors = self._calculateFaceAverageColors(polyData, textureImage, "RGB")
        if faceColors is None:
          if logCallback:
            logCallback(f"  Warning: Failed to calculate face colors")
          continue

        # Apply neighbor averaging if enabled
        if useNeighborAverage:
          if logCallback:
            logCallback(f"  Applying neighbor average smoothing...")

          faceColors = self._applyNeighborAveraging(polyData, faceColors, faceAdjacency, logCallback)
          if faceColors is None:
            if logCallback:
              logCallback(f"  Warning: Neighbor averaging failed")
            continue

        # Convert to Lab space
        faceColorsLab = self.rgb_to_lab(faceColors)

        # Subsample face colors if subsampling is enabled
        if subsampledFaceIndices is not None:
          faceColorsLabForClustering = faceColorsLab[subsampledFaceIndices]
          if logCallback:
            logCallback(f"  Using {len(subsampledFaceIndices)} subsampled faces for clustering")
        else:
          faceColorsLabForClustering = faceColorsLab

        # Apply luminosity normalization if enabled
        lcStats = None
        if normalizeLuminosity:
          if logCallback:
            logCallback(f"  Computing per-texture LC stats...")

          lcStats = self._computePerTextureLCStats(texturePath, logCallback)
          if lcStats is not None:
            mu_img, sd_img = lcStats
            mu_pool, sd_pool = pipeline.pooledLCStats

            if logCallback:
              logCallback(f"  Applying LC transform...")

            faceColorsLabForClustering = self._applyLCTransform(faceColorsLabForClustering, mu_img, sd_img, mu_pool, sd_pool)

        # Perform clustering and consolidation
        if logCallback:
          logCallback(f"  Clustering with {initialClusters} initial clusters...")

        clusteringResult = self._clusterAndConsolidateSingleTexture(
          faceColorsLabForClustering, initialClusters, consolidatedClusters, logCallback
        )

        if clusteringResult is None:
          if logCallback:
            logCallback(f"  Warning: Clustering failed for this texture")
          continue

        # Reorder clusters to match reference (if not first texture)
        reorderMapping = None
        if pipeline.referenceCentroids is not None:
          if logCallback:
            logCallback(f"  Reordering clusters to match reference...")

          reorderMapping = self._reorderClusterIndices(
            pipeline.referenceCentroids,
            clusteringResult["consolidatedCentroids"],
            logCallback
          )

          if reorderMapping is None:
            if logCallback:
              logCallback(f"  Warning: Cluster reordering failed")
            continue

        # Add results to pipeline
        pipeline.addTextureResults(textureFile, clusteringResult, lcStats, reorderMapping)

        # Update progress
        progress = 10 + (i + 1) * 85 / len(textureFiles)
        if progressCallback:
          progressCallback(progress)

      if len(pipeline.textureFiles) == 0:
        if logCallback:
          logCallback("Error: No textures were successfully processed")
        return {"success": False}

      # Compute shared palette by averaging all consolidated centroids
      if logCallback:
        logCallback(f"\nComputing shared palette from {len(pipeline.textureFiles)} textures...")

      pipeline.sharedPalette = pipeline.computeSharedPalette()

      if pipeline.sharedPalette is not None:
        if logCallback:
          logCallback(f"Shared palette computed with {len(pipeline.sharedPalette)} colors")
      else:
        if logCallback:
          logCallback("Warning: Failed to compute shared palette")

      if progressCallback:
        progressCallback(100)

      if logCallback:
        logCallback(f"\nMulti-texture clustering pipeline completed successfully!")
        logCallback(f"Processed {len(pipeline.textureFiles)} textures")
        logCallback(f"Reference texture: {pipeline.referenceTexture}")

      return {
        "success": True,
        "pipeline": pipeline,
        "face_areas": faceAreas,
        "num_textures_processed": len(pipeline.textureFiles)
      }

    except Exception as e:
      if logCallback:
        logCallback(f"Error in multi-texture clustering: {str(e)}")
      import traceback
      traceback.print_exc()
      return {"success": False}

  def performSubsampleOnly(self, modelNode, textureFiles, numSubsampledFaces=None, useNeighborAverage=False, faceAreas=None, progressCallback=None, logCallback=None):
    """
    Perform subsampling and face averaging without clustering

    Args:
        modelNode: VTK model node to analyze
        textureFiles: List of texture filenames (stored for later use in Steps 2-4)
        numSubsampledFaces: Number of faces to subsample (optional, uses all if None)
        useNeighborAverage: Whether to use average color of neighboring faces
        faceAreas: Pre-computed face areas (optional, will calculate if None)
        progressCallback: Function to call with progress updates (0-100)
        logCallback: Function to call with log messages

    Returns:
        dict with 'success', 'pipeline' (ClusteringPipeline object), 'face_areas'
    """
    try:
      if logCallback:
        logCallback(f"Starting subsample and average pipeline...")
        logCallback(f"Neighbor average: {'enabled' if useNeighborAverage else 'disabled'}")

      if progressCallback:
        progressCallback(2)

      if not SKIMAGE_AVAILABLE:
        if logCallback:
          logCallback("Error: scikit-image not available for color space conversion")
        return {"success": False}

      # Get model polydata
      polyData = modelNode.GetPolyData()
      if not polyData:
        if logCallback:
          logCallback("Error: No polydata in model")
        return {"success": False}

      # Use provided face areas or calculate them
      if faceAreas is None:
        if logCallback:
          logCallback("Calculating face areas...")

        faceAreas = self._calculateFaceAreas(polyData)
        if faceAreas is None:
          if logCallback:
            logCallback("Error: Failed to calculate face areas")
          return {"success": False}
      else:
        if logCallback:
          logCallback("Using provided face areas")

      if progressCallback:
        progressCallback(5)

      # Build face adjacency graph if needed (for subsampling or neighbor averaging)
      faceAdjacency = None
      if (numSubsampledFaces is not None and numSubsampledFaces > 0) or useNeighborAverage:
        if logCallback:
          logCallback("Building face adjacency graph...")

        # Clean mesh first
        polyData = self._cleanMesh(polyData, logCallback)

        # Build adjacency
        faceAdjacency = self._buildFaceAdjacencyGraph(polyData, logCallback)
        if faceAdjacency is None:
          if logCallback:
            logCallback("Error: Failed to build face adjacency graph")
          return {"success": False}

      # Subsample faces if requested
      subsampledFaceIndices = None
      nearestNeighborMapping = None
      if numSubsampledFaces is not None and numSubsampledFaces > 0:
        if logCallback:
          logCallback(f"Performing face subsampling...")

        # Use pre-built adjacency if available, otherwise build it
        if faceAdjacency is None:
          subsampledFaceIndices, nearestNeighborMapping, faceAdjacency = self._subsampleFacesUniformly(
            polyData, numSubsampledFaces, logCallback
          )
        else:
          # Reuse the pre-built adjacency graph
          subsampledFaceIndices, nearestNeighborMapping, _ = self._subsampleFacesUniformly(
            polyData, numSubsampledFaces, logCallback
          )

        if subsampledFaceIndices is None:
          if logCallback:
            logCallback("Error: Face subsampling failed")
          return {"success": False}

      # Create a simple pipeline object for subsample-only mode
      # We use ClusteringPipeline but with clustering disabled
      pipeline = ClusteringPipeline(1, 1, False, useNeighborAverage)

      # Store subsampling information in pipeline
      pipeline.subsampledFaceIndices = subsampledFaceIndices
      pipeline.nearestNeighborMapping = nearestNeighborMapping
      pipeline.faceAdjacency = faceAdjacency

      # Store texture file list for later use
      pipeline.textureFiles = textureFiles

      if progressCallback:
        progressCallback(100)

      if logCallback:
        logCallback(f"Subsample and average pipeline completed successfully!")
        if subsampledFaceIndices is not None:
          logCallback(f"Subsampled to {len(subsampledFaceIndices)} faces")
        logCallback(f"Ready to process {len(textureFiles)} textures in Steps 2-4")

      return {
        "success": True,
        "pipeline": pipeline,
        "face_areas": faceAreas,
        "num_textures_processed": len(textureFiles)
      }

    except Exception as e:
      if logCallback:
        logCallback(f"Error in subsample and average pipeline: {str(e)}")
      import traceback
      traceback.print_exc()
      return {"success": False}

  def _clusterAndConsolidateSingleTexture(self, faceColorsLab, initialClusters, consolidatedClusters, logCallback=None):
    """
    Perform initial clustering and hierarchical consolidation on a single texture

    Args:
        faceColorsLab: numpy array of face colors in Lab space (N, 3)
        initialClusters: number of initial clusters
        consolidatedClusters: number of consolidated clusters (must be <= initialClusters)
        logCallback: optional callback for logging

    Returns:
        dict with:
            - initialLabels: cluster labels from initial clustering (N,)
            - consolidatedLabels: cluster labels after consolidation (N,)
            - initialCentroids: initial cluster centroids (initialClusters, 3)
            - consolidatedCentroids: consolidated cluster centroids (consolidatedClusters, 3)
            - consolidationMapping: mapping from initial to consolidated indices (initialClusters,)
        Returns None if clustering fails
    """
    try:
      if not SKLEARN_AVAILABLE:
        if logCallback:
          logCallback("Error: sklearn not available for clustering")
        return None

      if not SCIPY_AVAILABLE:
        if logCallback:
          logCallback("Error: scipy not available for hierarchical clustering")
        return None

      # Step 1: Perform initial MiniBatchKMeans clustering
      if logCallback:
        logCallback(f"  Performing initial clustering with {initialClusters} clusters...")

      kmeans = MiniBatchKMeans(n_clusters=initialClusters, random_state=42, batch_size=1000)
      initialLabels = kmeans.fit_predict(faceColorsLab)
      initialCentroids = kmeans.cluster_centers_  # (initialClusters, 3)

      # Step 2: If initial == consolidated, no consolidation needed
      if initialClusters == consolidatedClusters:
        if logCallback:
          logCallback(f"  No consolidation needed (initial == consolidated)")

        consolidationMapping = np.arange(initialClusters)
        return {
          "initialLabels": initialLabels,
          "consolidatedLabels": initialLabels.copy(),
          "initialCentroids": initialCentroids,
          "consolidatedCentroids": initialCentroids.copy(),
          "consolidationMapping": consolidationMapping
        }

      # Step 3: Calculate pairwise distances between initial centroids
      if logCallback:
        logCallback(f"  Computing centroid distances for hierarchical clustering...")

      # Use Euclidean distance in Lab space
      centroidDistances = pdist(initialCentroids, metric='euclidean')

      # Step 4: Perform hierarchical clustering on centroids
      if logCallback:
        logCallback(f"  Performing hierarchical clustering...")

      linkageMatrix = hierarchy.linkage(centroidDistances, method='average')

      # Step 5: Cut dendrogram to get consolidated clusters
      if logCallback:
        logCallback(f"  Cutting dendrogram to {consolidatedClusters} clusters...")

      # fcluster returns cluster IDs starting from 1, we'll convert to 0-based
      consolidatedClusterIds = hierarchy.fcluster(linkageMatrix, consolidatedClusters, criterion='maxclust')
      consolidatedClusterIds = consolidatedClusterIds - 1  # Convert to 0-based

      # Step 6: Create consolidation mapping (initial index -> consolidated index)
      consolidationMapping = consolidatedClusterIds  # (initialClusters,)

      # Step 7: Apply consolidation mapping to initial labels
      consolidatedLabels = consolidationMapping[initialLabels]

      # Step 8: Compute consolidated centroids as mean of initial centroids in each group
      consolidatedCentroids = np.zeros((consolidatedClusters, 3))
      for consolidatedIdx in range(consolidatedClusters):
        # Find which initial clusters map to this consolidated cluster
        initialIndicesInGroup = np.where(consolidationMapping == consolidatedIdx)[0]
        if len(initialIndicesInGroup) > 0:
          # Average the centroids
          consolidatedCentroids[consolidatedIdx] = initialCentroids[initialIndicesInGroup].mean(axis=0)

      if logCallback:
        logCallback(f"  Consolidated {initialClusters} initial clusters into {consolidatedClusters} clusters")

      return {
        "initialLabels": initialLabels,
        "consolidatedLabels": consolidatedLabels,
        "initialCentroids": initialCentroids,
        "consolidatedCentroids": consolidatedCentroids,
        "consolidationMapping": consolidationMapping
      }

    except Exception as e:
      if logCallback:
        logCallback(f"Error in clustering and consolidation: {e}")
      import traceback
      traceback.print_exc()
      return None

  def _reorderClusterIndices(self, referenceCentroids, targetCentroids, logCallback=None):
    """
    Reorder target cluster indices to match reference clusters based on color similarity

    Uses Hungarian algorithm to find optimal matching between reference and target centroids

    Args:
        referenceCentroids: reference cluster centroids in Lab space (K, 3)
        targetCentroids: target cluster centroids in Lab space (K, 3)
        logCallback: optional callback for logging

    Returns:
        reorderMapping: array where reorderMapping[old_idx] = new_idx
        Returns None if reordering fails
    """
    try:
      if not SCIPY_AVAILABLE:
        if logCallback:
          logCallback("Error: scipy not available for cluster reordering")
        return None

      numClusters = len(referenceCentroids)

      if len(targetCentroids) != numClusters:
        if logCallback:
          logCallback(f"Error: Reference and target have different cluster counts ({numClusters} vs {len(targetCentroids)})")
        return None

      # Compute pairwise distances between reference and target centroids
      # Distance matrix: distMatrix[i, j] = distance from reference[i] to target[j]
      distMatrix = np.zeros((numClusters, numClusters))

      for i in range(numClusters):
        for j in range(numClusters):
          # Euclidean distance in Lab space
          distMatrix[i, j] = np.linalg.norm(referenceCentroids[i] - targetCentroids[j])

      # Use Hungarian algorithm to find optimal assignment
      # row_ind[i] is matched to col_ind[i]
      # We want: reference[i] matches target[col_ind[i]]
      row_ind, col_ind = linear_sum_assignment(distMatrix)

      # Create reordering mapping
      # reorderMapping[old_target_idx] = new_idx (to match reference order)
      reorderMapping = np.zeros(numClusters, dtype=int)

      for ref_idx, target_idx in zip(row_ind, col_ind):
        # Target cluster target_idx should be renamed to ref_idx
        reorderMapping[target_idx] = ref_idx

      if logCallback:
        totalDist = distMatrix[row_ind, col_ind].sum()
        logCallback(f"  Reordered clusters with total distance: {totalDist:.2f}")

      return reorderMapping

    except Exception as e:
      if logCallback:
        logCallback(f"Error in cluster reordering: {e}")
      import traceback
      traceback.print_exc()
      return None

  def _isBlackPixel(self, rgb):
    """
    Check if RGB color is pure black (all channels == 0)

    Args:
        rgb: numpy array of RGB values (0-255 or 0-1)

    Returns:
        boolean mask where True indicates black pixels
    """
    if rgb.max() <= 1.0:
      # Values are in [0, 1] range
      rgb_scaled = (rgb * 255).astype(np.uint8)
    else:
      rgb_scaled = rgb.astype(np.uint8)

    return (rgb_scaled.sum(axis=-1) == 0)

  def _computePooledLCStats(self, textureDir, textureFiles, logCallback=None):
    """
    Compute pooled L* and C* statistics across all textures

    Args:
        textureDir: Directory containing texture files
        textureFiles: List of texture filenames
        logCallback: Optional callback for logging

    Returns:
        tuple: (mu_pool, sd_pool) where each is [L_mean, C_mean] and [L_std, C_std]
               Returns None if computation fails
    """
    try:
      EPS = 1e-8
      sum_vec = np.zeros(2, dtype=np.float64)  # [L, C]
      sumsq_vec = np.zeros(2, dtype=np.float64)
      count = 0

      if logCallback:
        logCallback(f"Computing pooled L*C* statistics from {len(textureFiles)} textures...")

      for textureFile in textureFiles:
        texturePath = os.path.join(textureDir, textureFile)

        try:
          # Load texture
          textureImage = imageio.imread(texturePath)
          if len(textureImage.shape) != 3 or textureImage.shape[2] < 3:
            continue

          # Convert to float [0, 1]
          rgb = textureImage[:, :, :3].astype(np.float32) / 255.0

          # Identify black pixels
          mask_black = self._isBlackPixel(rgb)
          mask_nonblack = ~mask_black

          if not mask_nonblack.any():
            continue

          # Convert to Lab
          lab = self.rgb_to_lab((rgb * 255).astype(np.uint8))

          # Extract L, a, b for non-black pixels
          L = lab[..., 0][mask_nonblack]
          a = lab[..., 1][mask_nonblack]
          b = lab[..., 2][mask_nonblack]
          C = np.sqrt(a*a + b*b)

          # Accumulate statistics
          LC = np.stack([L, C], axis=1)
          sum_vec += LC.sum(axis=0)
          sumsq_vec += (LC ** 2).sum(axis=0)
          count += LC.shape[0]

        except Exception as e:
          if logCallback:
            logCallback(f"Warning: Failed to process {textureFile} for pooled stats: {e}")
          continue

      if count == 0:
        if logCallback:
          logCallback("Error: No valid pixels found for pooled statistics")
        return None

      # Compute mean and std
      mu_pool = sum_vec / count
      var = np.maximum(sumsq_vec / count - mu_pool**2, 0.0)
      sd_pool = np.sqrt(var) + EPS

      if logCallback:
        logCallback(f"Pooled L*C* stats - Mean: L={mu_pool[0]:.2f}, C={mu_pool[1]:.2f}; Std: L={sd_pool[0]:.2f}, C={sd_pool[1]:.2f}")

      return mu_pool, sd_pool

    except Exception as e:
      if logCallback:
        logCallback(f"Error computing pooled LC stats: {e}")
      import traceback
      traceback.print_exc()
      return None

  def _computePerTextureLCStats(self, texturePath, logCallback=None):
    """
    Compute per-texture L* and C* statistics

    Args:
        texturePath: Path to texture file
        logCallback: Optional callback for logging

    Returns:
        tuple: (mu_img, sd_img) where each is [L_mean, C_mean] and [L_std, C_std]
               Returns None if computation fails
    """
    try:
      EPS = 1e-8

      # Load texture
      textureImage = imageio.imread(texturePath)
      if len(textureImage.shape) != 3 or textureImage.shape[2] < 3:
        return None

      # Convert to float [0, 1]
      rgb = textureImage[:, :, :3].astype(np.float32) / 255.0

      # Identify black pixels
      mask_black = self._isBlackPixel(rgb)
      mask_nonblack = ~mask_black

      if not mask_nonblack.any():
        # All black - return neutral fallback
        mu_img = np.array([50.0, 20.0], dtype=np.float64)
        sd_img = np.array([1.0, 1.0], dtype=np.float64)
        return mu_img, sd_img

      # Convert to Lab
      lab = self.rgb_to_lab((rgb * 255).astype(np.uint8))

      # Extract L, a, b for non-black pixels
      L = lab[..., 0][mask_nonblack]
      a = lab[..., 1][mask_nonblack]
      b = lab[..., 2][mask_nonblack]
      C = np.sqrt(a*a + b*b)

      # Compute mean and std
      mu_img = np.array([L.mean(), C.mean()], dtype=np.float64)
      sd_img = np.array([L.std(), C.std()], dtype=np.float64) + EPS

      return mu_img, sd_img

    except Exception as e:
      if logCallback:
        logCallback(f"Error computing per-texture LC stats: {e}")
      return None

  def _applyLCTransform(self, labColors, mu_img, sd_img, mu_pool, sd_pool):
    """
    Apply L*C* transformation to Lab colors

    Args:
        labColors: numpy array of Lab colors (N, 3)
        mu_img: per-image mean [L, C]
        sd_img: per-image std [L, C]
        mu_pool: pooled mean [L, C]
        sd_pool: pooled std [L, C]

    Returns:
        numpy array of transformed Lab colors (N, 3)
    """
    try:
      # Extract L, a, b
      L = labColors[:, 0].copy()
      a = labColors[:, 1].copy()
      b = labColors[:, 2].copy()

      # Compute chroma and hue
      C = np.sqrt(a*a + b*b)
      h = np.arctan2(b, a)  # hue angle

      # Transform L and C
      L_transformed = (L - mu_img[0]) * (sd_pool[0] / sd_img[0]) + mu_pool[0]
      C_transformed = (C - mu_img[1]) * (sd_pool[1] / sd_img[1]) + mu_pool[1]

      # Clip to valid ranges
      L_transformed = np.clip(L_transformed, 0.0, 100.0)
      C_transformed = np.maximum(C_transformed, 0.0)

      # Reconstruct a, b with original hue
      a_transformed = C_transformed * np.cos(h)
      b_transformed = C_transformed * np.sin(h)

      # Clip a, b to valid ranges
      a_transformed = np.clip(a_transformed, -128.0, 127.0)
      b_transformed = np.clip(b_transformed, -128.0, 127.0)

      # Reconstruct Lab array
      labTransformed = np.stack([L_transformed, a_transformed, b_transformed], axis=1)

      return labTransformed

    except Exception as e:
      print(f"Error applying LC transform: {e}")
      return labColors  # Return original on error

  def _calculateFaceAreas(self, polyData):
    """
    Calculate the area of each face in the mesh

    Args:
        polyData: VTK polydata object

    Returns:
        numpy array of face areas, or None if failed
    """
    try:
      numFaces = polyData.GetNumberOfCells()

      # Try vectorized path for triangle meshes
      faces = self._extractFaceConnectivity(polyData)
      points_np = vtk_np.vtk_to_numpy(polyData.GetPoints().GetData())

      if faces is not None and faces.shape[1] == 3:
        # Vectorized triangle area: 0.5 * ||(p1-p0) x (p2-p0)||
        p0 = points_np[faces[:, 0]]
        p1 = points_np[faces[:, 1]]
        p2 = points_np[faces[:, 2]]
        cross = np.cross(p1 - p0, p2 - p0)
        faceAreas = 0.5 * np.linalg.norm(cross, axis=1)
        return faceAreas

      if faces is not None and faces.shape[1] == 4:
        # Vectorized quad area: two triangles (p0,p1,p2) + (p0,p2,p3)
        p0 = points_np[faces[:, 0]]
        p1 = points_np[faces[:, 1]]
        p2 = points_np[faces[:, 2]]
        p3 = points_np[faces[:, 3]]
        area1 = 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1)
        area2 = 0.5 * np.linalg.norm(np.cross(p2 - p0, p3 - p0), axis=1)
        return area1 + area2

      # Fallback: per-face loop for mixed cell types
      faceAreas = np.zeros(numFaces)
      for faceId in range(numFaces):
        cell = polyData.GetCell(faceId)
        if cell.GetNumberOfPoints() >= 3:
          pts = [np.array(polyData.GetPoint(cell.GetPointId(i))) for i in range(cell.GetNumberOfPoints())]
          v1 = pts[1] - pts[0]
          v2 = pts[2] - pts[0]
          area = 0.5 * np.linalg.norm(np.cross(v1, v2))
          if len(pts) == 4:
            v3 = pts[3] - pts[0]
            area += 0.5 * np.linalg.norm(np.cross(v2, v3))
          faceAreas[faceId] = area
      return faceAreas

    except Exception as e:
      print(f"Error calculating face areas: {e}")
      return None

  def _cleanMesh(self, polyData, logCallback=None):
    """
    Clean and merge the mesh to remove duplicate vertices and degenerate faces.

    This is necessary for meshes from 3D Slicer that may have:
    - Duplicate vertices at the same location
    - Degenerate faces
    - Non-manifold geometry

    Args:
        polyData: VTK polydata object
        logCallback: Function to call with log messages

    Returns:
        VTK polydata object (cleaned)
    """
    try:
      if logCallback:
        logCallback("Cleaning mesh...")

      # Step 1: Merge duplicate points
      cleanFilter = vtk.vtkCleanPolyData()
      cleanFilter.SetInputData(polyData)
      cleanFilter.SetTolerance(1e-10)  # Very small tolerance to catch exact duplicates
      cleanFilter.Update()

      cleaned = cleanFilter.GetOutput()

      if logCallback:
        origPoints = polyData.GetNumberOfPoints()
        newPoints = cleaned.GetNumberOfPoints()
        logCallback(f"  Merged duplicate vertices: {origPoints} -> {newPoints} points")

      return cleaned

    except Exception as e:
      if logCallback:
        logCallback(f"Error cleaning mesh: {str(e)}")
      import traceback
      traceback.print_exc()
      return polyData

  def _buildFaceAdjacencyGraph(self, polyData, logCallback=None):
    """
    Build an adjacency graph of faces based on shared edges or vertices.

    Handles both connected meshes and disconnected meshes with duplicated vertices.
    For disconnected meshes, faces sharing vertices are considered adjacent.

    Args:
        polyData: VTK polydata object
        logCallback: Function to call with log messages

    Returns:
        dict: adjacency[faceId] = set of adjacent face indices
    """
    try:
      # Clean the mesh first to merge duplicate vertices
      if logCallback:
        logCallback("Cleaning mesh before building adjacency graph...")

      polyData = self._cleanMesh(polyData, logCallback)

      numFaces = polyData.GetNumberOfCells()
      adjacency = {i: set() for i in range(numFaces)}

      # Try vectorized path for triangle meshes
      faces = self._extractFaceConnectivity(polyData)

      if faces is not None and faces.shape[1] == 3:
        if logCallback:
          logCallback(f"  Mesh structure: {{'vtkTriangle': {numFaces}}}")

        # Step 1: Build edge-based adjacency using vectorized numpy
        # Generate all 3 edges per triangle: (v0,v1), (v1,v2), (v2,v0)
        e0 = np.stack([faces[:, 0], faces[:, 1]], axis=1)  # edge 0
        e1 = np.stack([faces[:, 1], faces[:, 2]], axis=1)  # edge 1
        e2 = np.stack([faces[:, 2], faces[:, 0]], axis=1)  # edge 2
        allEdges = np.vstack([e0, e1, e2])  # (numFaces*3, 2)

        # Canonical form: smaller vertex id first
        allEdges = np.sort(allEdges, axis=1)

        # Face indices for each edge
        faceIds = np.tile(np.arange(numFaces), 3)  # [0..N-1, 0..N-1, 0..N-1]

        # Sort edges lexicographically to group identical edges together
        sortIdx = np.lexsort((allEdges[:, 1], allEdges[:, 0]))
        sortedEdges = allEdges[sortIdx]
        sortedFaceIds = faceIds[sortIdx]

        # Find where consecutive edges are equal (shared edges)
        sameAsNext = np.all(sortedEdges[:-1] == sortedEdges[1:], axis=1)

        # For shared edges, connect the two faces
        edgeCount = 0
        for idx in np.where(sameAsNext)[0]:
          f1 = int(sortedFaceIds[idx])
          f2 = int(sortedFaceIds[idx + 1])
          if f1 != f2:
            adjacency[f1].add(f2)
            adjacency[f2].add(f1)
            edgeCount += 1

        if logCallback:
          logCallback(f"  Edge analysis: {edgeCount} adjacencies from shared edges")

        # Step 2: Vertex-based adjacency using numpy
        # For each vertex, find all faces that use it, then connect them
        flatVertices = faces.ravel()  # all vertex indices
        flatFaceIds = np.repeat(np.arange(numFaces), 3)  # corresponding face for each vertex

        # Sort by vertex id to group faces sharing the same vertex
        sortIdx = np.argsort(flatVertices)
        sortedVerts = flatVertices[sortIdx]
        sortedFIds = flatFaceIds[sortIdx]

        # Find boundaries between vertex groups
        changes = np.where(np.diff(sortedVerts) != 0)[0] + 1
        groups = np.split(sortedFIds, changes)

        vertexCount = 0
        for group in groups:
          if len(group) > 1:
            uniqueFaces = np.unique(group)
            if len(uniqueFaces) > 1:
              for i in range(len(uniqueFaces)):
                for j in range(i + 1, len(uniqueFaces)):
                  f1, f2 = int(uniqueFaces[i]), int(uniqueFaces[j])
                  if f2 not in adjacency[f1]:
                    adjacency[f1].add(f2)
                    adjacency[f2].add(f1)
                    vertexCount += 1

        if logCallback:
          logCallback(f"  Vertex analysis: {vertexCount} new adjacencies from shared vertices")

        # Check for isolated faces
        connectedFaces = sum(1 for s in adjacency.values() if len(s) > 0)
        isolated = numFaces - connectedFaces

        if logCallback:
          logCallback(f"Face adjacency graph built:")
          logCallback(f"  - {connectedFaces} connected faces")
          logCallback(f"  - {isolated} isolated faces (no adjacencies)")

        # Fallback for completely disconnected meshes
        if isolated == numFaces:
          if logCallback:
            logCallback("  WARNING: Mesh is completely disconnected! Using spatial proximity fallback...")
          points_np = vtk_np.vtk_to_numpy(polyData.GetPoints().GetData())
          faceCenters = np.mean(points_np[faces], axis=1)

          # Estimate average edge length from first 1000 faces
          sampleN = min(1000, numFaces)
          sampleFaces = faces[:sampleN]
          p0 = points_np[sampleFaces[:, 0]]
          p1 = points_np[sampleFaces[:, 1]]
          p2 = points_np[sampleFaces[:, 2]]
          edgeLens = np.concatenate([
            np.linalg.norm(p1 - p0, axis=1),
            np.linalg.norm(p2 - p1, axis=1),
            np.linalg.norm(p0 - p2, axis=1),
          ])
          avgEdgeLength = np.mean(edgeLens)
          proximityThreshold = avgEdgeLength * 1.5

          if logCallback:
            logCallback(f"  Average edge length: {avgEdgeLength:.6f}, proximity threshold: {proximityThreshold:.6f}")

          try:
            from scipy.spatial import cKDTree
            tree = cKDTree(faceCenters)
            pairs = tree.query_pairs(proximityThreshold)
            for i, j in pairs:
              adjacency[i].add(j)
              adjacency[j].add(i)
            if logCallback:
              logCallback(f"  Added {len(pairs)} spatial proximity connections (KD-tree)")
          except ImportError:
            if logCallback:
              logCallback(f"  WARNING: scipy not available, using slower O(n²) spatial proximity")
            for i in range(numFaces):
              for j in range(i + 1, numFaces):
                if np.linalg.norm(faceCenters[i] - faceCenters[j]) < proximityThreshold:
                  adjacency[i].add(j)
                  adjacency[j].add(i)

        return adjacency

      # Fallback: per-face loop for non-triangle meshes
      edgeToFaces = {}
      cellTypes = {}

      for faceId in range(numFaces):
        cell = polyData.GetCell(faceId)
        numPoints = cell.GetNumberOfPoints()
        cellType = cell.GetClassName()
        cellTypes[cellType] = cellTypes.get(cellType, 0) + 1

        for i in range(numPoints):
          p1 = cell.GetPointId(i)
          p2 = cell.GetPointId((i + 1) % numPoints)
          edge = (min(p1, p2), max(p1, p2))
          if edge not in edgeToFaces:
            edgeToFaces[edge] = []
          edgeToFaces[edge].append(faceId)

      if logCallback:
        logCallback(f"  Mesh structure: {cellTypes}")

      for edge, faceList in edgeToFaces.items():
        if len(faceList) >= 2:
          for i in range(len(faceList)):
            for j in range(i + 1, len(faceList)):
              adjacency[faceList[i]].add(faceList[j])
              adjacency[faceList[j]].add(faceList[i])

      return adjacency

    except Exception as e:
      if logCallback:
        logCallback(f"Error building face adjacency graph: {str(e)}")
      import traceback
      traceback.print_exc()
      return None

  def _subsampleFacesUniformly(self, polyData, numSubsampledFaces, logCallback=None):
    """
    Subsample faces uniformly based on mesh topology using BFS with graph-based distance.

    This approach uses the face adjacency graph to ensure uniform distribution based on
    actual mesh connectivity rather than Euclidean distance.

    Algorithm:
    1. Build face adjacency graph (faces sharing edges)
    2. Random initial sampling
    3. Use BFS to compute graph distance from each face to nearest sampled face
    4. Iteratively swap poorly-placed samples with better candidates

    Args:
        polyData: VTK polydata object
        numSubsampledFaces: Target number of faces to subsample
        logCallback: Function to call with log messages

    Returns:
        tuple: (subsampledFaceIndices, nearestNeighborMapping, adjacency)
        - subsampledFaceIndices: numpy array of selected face indices
        - nearestNeighborMapping: numpy array where mapping[i] = j means face i's nearest sampled neighbor is j
        - adjacency: dict of face adjacency graph for reuse
    """
    try:
      numFaces = polyData.GetNumberOfCells()

      # Clamp the number of subsampled faces
      numSubsampledFaces = min(numSubsampledFaces, numFaces)

      if logCallback:
        logCallback(f"Subsampling {numSubsampledFaces} faces from {numFaces} total faces (graph-based)")

      # Step 1: Build face adjacency graph
      if logCallback:
        logCallback("Building face adjacency graph...")

      adjacency = self._buildFaceAdjacencyGraph(polyData, logCallback)
      if adjacency is None:
        return None, None, None

      # Step 2: Random initial sampling
      if logCallback:
        logCallback("Random initial sampling...")

      selectedIndices = np.random.choice(numFaces, size=numSubsampledFaces, replace=False)
      selectedSet = set(selectedIndices)

      # Step 3: Compute graph distances and nearest neighbors
      def compute_graph_distances_and_mapping(selected_set, adjacency, num_faces):
        """
        Compute graph distance from each face to nearest selected face using BFS.
        Returns: (distances, nearest_neighbor_mapping)
        """
        from collections import deque

        distances = np.full(num_faces, np.inf, dtype=np.float32)
        nearest_mapping = np.zeros(num_faces, dtype=np.int32)

        # BFS from all selected faces simultaneously
        queue = deque()
        for face_id in selected_set:
          distances[face_id] = 0
          nearest_mapping[face_id] = face_id
          queue.append(face_id)

        while queue:
          current_face = queue.popleft()
          current_dist = distances[current_face]

          # Explore neighbors
          for neighbor_face in adjacency[current_face]:
            new_dist = current_dist + 1
            if new_dist < distances[neighbor_face]:
              distances[neighbor_face] = new_dist
              nearest_mapping[neighbor_face] = nearest_mapping[current_face]
              queue.append(neighbor_face)

        return distances, nearest_mapping

      # Step 4: Iterative improvement via swapping (simplified for performance)
      if logCallback:
        logCallback("Optimizing sample distribution...")

      # Only do a few quick optimization passes for large meshes
      max_iterations = min(10, max(1, numSubsampledFaces // 1000))

      for iteration in range(max_iterations):
        if logCallback and iteration % 10 == 0:
          logCallback(f"  Iteration {iteration + 1}...")
        distances, _ = compute_graph_distances_and_mapping(selectedSet, adjacency, numFaces)

        # Find the unselected face with maximum distance to nearest selected face
        unselected_distances = distances.copy()
        unselected_distances[list(selectedSet)] = -np.inf

        worst_unselected_idx = np.argmax(unselected_distances)
        worst_unselected_dist = unselected_distances[worst_unselected_idx]

        if worst_unselected_dist <= 1:
          # All unselected faces are close to selected faces, optimization complete
          break

        # Simple heuristic: swap with a random selected face
        # (Much faster than finding the "best" one to swap)
        worst_selected_face = np.random.choice(list(selectedSet))

        # Swap if it improves the distribution
        selectedSet.remove(worst_selected_face)
        selectedSet.add(worst_unselected_idx)

      if logCallback:
        logCallback(f"Optimization complete ({iteration + 1} iterations)")

      subsampledFaceIndices = np.array(sorted(list(selectedSet)), dtype=np.int32)

      # Final nearest neighbor mapping
      if logCallback:
        logCallback("Computing final nearest neighbor mapping...")

      _, nearestNeighborMapping = compute_graph_distances_and_mapping(set(subsampledFaceIndices), adjacency, numFaces)

      if logCallback:
        logCallback(f"Subsampling complete: selected {len(subsampledFaceIndices)} faces")

      return subsampledFaceIndices, nearestNeighborMapping, adjacency

    except Exception as e:
      if logCallback:
        logCallback(f"Error in face subsampling: {str(e)}")
      import traceback
      traceback.print_exc()
      return None, None, None

    except Exception as e:
      print(f"Error calculating face areas: {e}")
      return None

  def applyIndividualTextureWithClusteredPalette(self, modelNode, texturePath, clusterCenters=None, clusteringPipeline=None,
                                                  faceAreas=None, progressCallback=None, logCallback=None):
    """
    Apply individual texture with pre-computed clustered palette (shared palette from Step 1)

    Args:
        modelNode: VTK model node to apply colors to
        texturePath: Path to the texture image file
        clusterCenters: Pre-computed cluster centers (RGB colors) - for backward compatibility
        clusteringPipeline: ClusteringPipeline object (preferred, overrides clusterCenters)
        faceAreas: Pre-computed face areas (optional, for caching)
        progressCallback: Function to call with progress updates (0-100)
        logCallback: Function to call with log messages

    Returns:
        bool: True if successful, False otherwise
    """
    try:
      # Determine which clustering data to use
      textureFilename = os.path.basename(texturePath)

      if clusteringPipeline is not None:
        # Use pipeline (preferred)
        if logCallback:
          logCallback(f"Using clustering pipeline for texture: {textureFilename}")

        # Use shared palette (averaged across all textures) instead of texture-specific centroids
        if clusteringPipeline.sharedPalette is not None:
          consolidatedCentroids = clusteringPipeline.sharedPalette
          if logCallback:
            logCallback(f"Using shared palette with {len(consolidatedCentroids)} colors")
        else:
          # Fallback to texture-specific centroids if shared palette not available
          consolidatedCentroids = clusteringPipeline.getConsolidatedCentroidsForTexture(textureFilename)
          if consolidatedCentroids is None:
            if logCallback:
              logCallback(f"Error: Texture {textureFilename} not found in pipeline")
            return False
          if logCallback:
            logCallback(f"Warning: Shared palette not available, using texture-specific centroids")

        # Convert centroids from Lab to RGB for palette
        clusterCentersRgb = self.lab_to_rgb(consolidatedCentroids).astype(np.uint8)

        # Check if luminosity normalization was used
        useLuminosityNorm = clusteringPipeline.normalizeLuminosity
        if useLuminosityNorm:
          lcStats = clusteringPipeline.perTextureLCStats.get(textureFilename)
          pooledStats = clusteringPipeline.pooledLCStats
        else:
          lcStats = None
          pooledStats = None

      elif clusterCenters is not None:
        # Backward compatibility: use clusterCenters directly
        if logCallback:
          logCallback(f"Using legacy cluster centers for texture: {textureFilename}")

        clusterCentersRgb = clusterCenters
        consolidatedCentroids = self.rgb_to_lab(clusterCenters)
        useLuminosityNorm = False
        lcStats = None
        pooledStats = None

      else:
        if logCallback:
          logCallback("Error: Neither clusteringPipeline nor clusterCenters provided")
        return False

      if logCallback:
        logCallback(f"Loading texture: {textureFilename}")

      if progressCallback:
        progressCallback(5)

      # Load texture image
      try:
        textureImage = imageio.imread(texturePath)
        if len(textureImage.shape) != 3 or textureImage.shape[2] < 3:
          if logCallback:
            logCallback("Error: Invalid texture format")
          return False
      except Exception as e:
        if logCallback:
          logCallback(f"Error loading texture: {e}")
        return False

      # Get model polydata
      polyData = modelNode.GetPolyData()
      if not polyData:
        if logCallback:
          logCallback("Error: No polydata in model")
        return False

      if progressCallback:
        progressCallback(15)

      # Calculate face average colors
      if logCallback:
        logCallback("Calculating average face colors...")

      faceColors = self._calculateFaceAverageColors(polyData, textureImage, "RGB")
      if faceColors is None:
        if logCallback:
          logCallback("Error: Failed to calculate face colors")
        return False

      if progressCallback:
        progressCallback(40)

      # Apply neighbor averaging if it was used during clustering
      if clusteringPipeline is not None and clusteringPipeline.useNeighborAverage:
        if logCallback:
          logCallback("Applying neighbor average smoothing to face colors...")

        faceColors = self._applyNeighborAveraging(polyData, faceColors, clusteringPipeline.faceAdjacency, logCallback)
        if faceColors is None:
          if logCallback:
            logCallback("Warning: Neighbor averaging failed, continuing with original colors")
          faceColors = self._calculateFaceAverageColors(polyData, textureImage, "RGB")

      if progressCallback:
        progressCallback(45)

      # Convert face colors to Lab space for clustering assignment
      if logCallback:
        logCallback("Converting colors to CIE Lab space...")

      faceColorsLab = self.rgb_to_lab(faceColors)

      # Apply luminosity normalization if it was used during clustering
      if useLuminosityNorm and lcStats is not None and pooledStats is not None:
        if logCallback:
          logCallback("Applying luminosity normalization transform...")

        mu_img, sd_img = lcStats
        mu_pool, sd_pool = pooledStats
        faceColorsLab = self._applyLCTransform(faceColorsLab, mu_img, sd_img, mu_pool, sd_pool)

        # Also apply normalization to raw face colors for display
        # Convert normalized Lab colors back to RGB for visualization
        faceColors = self.lab_to_rgb(faceColorsLab).astype(np.uint8)

      if progressCallback:
        progressCallback(50)

      # Use the shared palette (clustered colors)
      if logCallback:
        logCallback("Using shared palette from clustering...")
      paletteColors = clusterCentersRgb

      # Convert palette to Lab space for distance calculation
      paletteColorsLab = self.rgb_to_lab(paletteColors)

      if progressCallback:
        progressCallback(60)

      # Assign each face color to the nearest cluster center
      if logCallback:
        logCallback("Assigning face colors to nearest cluster centers...")

      quantizedColors = np.zeros_like(faceColors)

      # Use vectorized distance calculation for better performance
      # Calculate Euclidean distances in Lab space (faster than ΔE2000)
      numFaces = len(faceColorsLab)
      numClusters = len(paletteColorsLab)

      if logCallback:
        logCallback(f"Processing {numFaces} faces with {numClusters} clusters...")

      # Check if subsampling was used
      if clusteringPipeline is not None and clusteringPipeline.subsampledFaceIndices is not None:
        # Use subsampled faces for clustering assignment
        subsampledFaceIndices = clusteringPipeline.subsampledFaceIndices
        nearestNeighborMapping = clusteringPipeline.nearestNeighborMapping

        if logCallback:
          logCallback(f"Using subsampled faces ({len(subsampledFaceIndices)}) for color assignment...")

        # Get colors only for subsampled faces
        subsampledFaceColorsLab = faceColorsLab[subsampledFaceIndices]

        # Vectorized distance calculation for subsampled faces
        subsampledFaceColorsExpanded = subsampledFaceColorsLab[:, np.newaxis, :]  # (N_s, 1, 3)
        clusterColorsExpanded = paletteColorsLab[np.newaxis, :, :]  # (1, K, 3)

        # Calculate distances for subsampled faces
        distances = np.sqrt(np.sum((subsampledFaceColorsExpanded - clusterColorsExpanded) ** 2, axis=2))  # (N_s, K)

        # Find nearest cluster for each subsampled face
        subsampledNearestClusters = np.argmin(distances, axis=1)  # (N_s,)

        # Assign colors to subsampled faces
        subsampledQuantizedColors = paletteColors[subsampledNearestClusters]

        # Now propagate colors to all faces using nearest neighbor mapping
        for faceIdx in range(numFaces):
          nearestSampledIdx = nearestNeighborMapping[faceIdx]
          # Find which position this sampled face is in the subsampled array
          sampledPosition = np.where(subsampledFaceIndices == nearestSampledIdx)[0][0]
          quantizedColors[faceIdx] = subsampledQuantizedColors[sampledPosition]
      else:
        # Original behavior: assign all faces directly
        # Vectorized distance calculation
        # Reshape for broadcasting: faces (N,1,3) and clusters (1,K,3)
        faceColorsExpanded = faceColorsLab[:, np.newaxis, :]  # (N, 1, 3)
        clusterColorsExpanded = paletteColorsLab[np.newaxis, :, :]  # (1, K, 3)

        # Calculate Euclidean distances in Lab space
        distances = np.sqrt(np.sum((faceColorsExpanded - clusterColorsExpanded) ** 2, axis=2))  # (N, K)

        # Find nearest cluster for each face
        nearestClusters = np.argmin(distances, axis=1)  # (N,)

        # Assign colors
        quantizedColors = paletteColors[nearestClusters]

      if progressCallback:
        progressCallback(80)

      # Apply quantized colors to the model
      if logCallback:
        logCallback("Applying quantized colors to model...")

      # Create color array for VTK
      colorArray = vtk.vtkUnsignedCharArray()
      colorArray.SetNumberOfComponents(3)
      colorArray.SetName("Colors")
      colorArray.SetNumberOfTuples(polyData.GetNumberOfCells())

      for i in range(len(quantizedColors)):
        color = quantizedColors[i].astype(int)
        colorArray.SetTuple3(i, color[0], color[1], color[2])

      # Add colors to the polydata
      polyData.GetCellData().SetScalars(colorArray)
      polyData.Modified()
      modelNode.Modified()

      if progressCallback:
        progressCallback(90)

      # Update display to show colors
      displayNode = modelNode.GetDisplayNode()
      if displayNode:
        if logCallback:
          logCallback("Configuring display node...")

        # Turn off texture first
        displayNode.SetTextureImageDataConnection(None)

        # Enable scalar visibility and set to use RGB colors directly
        displayNode.SetScalarVisibility(True)
        displayNode.SetActiveScalarName("Colors")

        # Set to use cell data (not point data)
        displayNode.SetActiveAttributeLocation(vtk.vtkDataObject.CELL)

        # Set scalar range to use direct mapping (RGB values 0-255)
        displayNode.SetScalarRangeFlag(slicer.vtkMRMLDisplayNode.UseDirectMapping)

        if logCallback:
          logCallback("Display node configured for color visualization")
      else:
        if logCallback:
          logCallback("Warning: No display node found")

      if progressCallback:
        progressCallback(100)

      if logCallback:
        logCallback("Individual texture visualization applied successfully")

      return True

    except Exception as e:
      if logCallback:
        logCallback(f"Error in individual texture visualization: {str(e)}")
      import traceback
      traceback.print_exc()
      return False

  def applyTextureWithSubsamplingOnly(self, modelNode, texturePath, clusteringPipeline=None, faceAreas=None, progressCallback=None, logCallback=None):
    """
    Apply texture with subsampling and face averaging, but without color quantization

    Args:
        modelNode: VTK model node to apply colors to
        texturePath: Path to the texture image file
        clusteringPipeline: ClusteringPipeline object with subsampling info
        faceAreas: Pre-computed face areas (optional, for caching)
        progressCallback: Function to call with progress updates (0-100)
        logCallback: Function to call with log messages

    Returns:
        bool: True if successful, False otherwise
    """
    try:
      textureFilename = os.path.basename(texturePath)

      if logCallback:
        logCallback(f"Applying texture with subsampling only: {textureFilename}")

      if clusteringPipeline is None:
        if logCallback:
          logCallback("Error: No clustering pipeline provided")
        return False

      if progressCallback:
        progressCallback(5)

      # Load texture image
      try:
        textureImage = imageio.imread(texturePath)
        if len(textureImage.shape) != 3 or textureImage.shape[2] < 3:
          if logCallback:
            logCallback("Error: Invalid texture format")
          return False
      except Exception as e:
        if logCallback:
          logCallback(f"Error loading texture: {e}")
        return False

      # Get model polydata
      polyData = modelNode.GetPolyData()
      if not polyData:
        if logCallback:
          logCallback("Error: No polydata in model")
        return False

      if progressCallback:
        progressCallback(15)

      # Calculate face average colors
      if logCallback:
        logCallback("Calculating average face colors...")

      faceColors = self._calculateFaceAverageColors(polyData, textureImage, "RGB")
      if faceColors is None:
        if logCallback:
          logCallback("Error: Failed to calculate face colors")
        return False

      if progressCallback:
        progressCallback(40)

      # Apply neighbor averaging if it was used during subsampling
      if clusteringPipeline.useNeighborAverage:
        if logCallback:
          logCallback("Applying neighbor average smoothing to face colors...")

        faceColors = self._applyNeighborAveraging(polyData, faceColors, clusteringPipeline.faceAdjacency, logCallback)
        if faceColors is None:
          if logCallback:
            logCallback("Warning: Neighbor averaging failed, continuing with original colors")
          faceColors = self._calculateFaceAverageColors(polyData, textureImage, "RGB")

      if progressCallback:
        progressCallback(50)

      # Check if subsampling was used
      if clusteringPipeline.subsampledFaceIndices is not None:
        # Use subsampled faces only
        subsampledFaceIndices = clusteringPipeline.subsampledFaceIndices
        nearestNeighborMapping = clusteringPipeline.nearestNeighborMapping

        if logCallback:
          logCallback(f"Using subsampled faces ({len(subsampledFaceIndices)}) for color assignment...")

        # Get colors only for subsampled faces
        subsampledFaceColors = faceColors[subsampledFaceIndices]

        # Create color array for all faces, using nearest neighbor mapping
        numFaces = polyData.GetNumberOfCells()
        displayColors = np.zeros((numFaces, 3), dtype=np.uint8)

        # Assign colors to all faces based on nearest subsampled face
        for faceIdx in range(numFaces):
          nearestSampledIdx = nearestNeighborMapping[faceIdx]
          # Find which position this sampled face is in the subsampled array
          sampledPosition = np.where(subsampledFaceIndices == nearestSampledIdx)[0][0]
          displayColors[faceIdx] = subsampledFaceColors[sampledPosition]
      else:
        # No subsampling - use all face colors directly
        if logCallback:
          logCallback("No subsampling - using all face colors directly...")

        displayColors = faceColors

      if progressCallback:
        progressCallback(80)

      # Apply colors to the model
      if logCallback:
        logCallback("Applying colors to model...")

      # Create color array for VTK
      colorArray = vtk.vtkUnsignedCharArray()
      colorArray.SetNumberOfComponents(3)
      colorArray.SetName("Colors")
      colorArray.SetNumberOfTuples(polyData.GetNumberOfCells())

      for i in range(len(displayColors)):
        color = displayColors[i].astype(int)
        colorArray.SetTuple3(i, color[0], color[1], color[2])

      # Add colors to the polydata
      polyData.GetCellData().SetScalars(colorArray)
      polyData.Modified()
      modelNode.Modified()

      if progressCallback:
        progressCallback(90)

      # Update display to show colors
      displayNode = modelNode.GetDisplayNode()
      if displayNode:
        if logCallback:
          logCallback("Configuring display node...")

        # Turn off texture first
        displayNode.SetTextureImageDataConnection(None)

        # Enable scalar visibility and set to use RGB colors directly
        displayNode.SetScalarVisibility(True)
        displayNode.SetActiveScalarName("Colors")

        # Set to use cell data (not point data)
        displayNode.SetActiveAttributeLocation(vtk.vtkDataObject.CELL)

        # Set scalar range to use direct mapping (RGB values 0-255)
        displayNode.SetScalarRangeFlag(slicer.vtkMRMLDisplayNode.UseDirectMapping)

        if logCallback:
          logCallback("Display node configured for color visualization")
      else:
        if logCallback:
          logCallback("Warning: No display node found")

      if progressCallback:
        progressCallback(100)

      if logCallback:
        logCallback("Texture with subsampling applied successfully")

      return True

    except Exception as e:
      if logCallback:
        logCallback(f"Error in texture with subsampling: {str(e)}")
      import traceback
      traceback.print_exc()
      return False

  def applyMorphospaceColorsToModel(self, modelNode, texturePath, reconstructedColors, clusteringPipeline, faceAreas=None):
    """
    Apply morphospace-generated colors to the model

    This method takes reconstructed cluster colors from PCA inverse transform
    and applies them to the model by assigning faces to clusters based on the
    original texture.

    Args:
        modelNode: VTK model node to apply colors to
        texturePath: Path to the original texture (used for face-to-cluster assignment)
        reconstructedColors: Reconstructed cluster colors (n_clusters x 3) in RGB [0,1]
        clusteringPipeline: ClusteringPipeline object with cluster assignments
        faceAreas: Pre-computed face areas (optional)

    Returns:
        bool: True if successful, False otherwise
    """
    try:
      # Load texture image
      textureImage = imageio.imread(texturePath)
      if len(textureImage.shape) != 3 or textureImage.shape[2] < 3:
        return False

      # Get model polydata
      polyData = modelNode.GetPolyData()
      if not polyData:
        return False

      # Calculate face average colors from original texture
      faceColors = self._calculateFaceAverageColors(polyData, textureImage, "RGB")
      if faceColors is None:
        return False

      # Apply neighbor averaging if it was used during clustering
      if clusteringPipeline.useNeighborAverage:
        faceColors = self._applyNeighborAveraging(polyData, faceColors, clusteringPipeline.faceAdjacency, None)
        if faceColors is None:
          faceColors = self._calculateFaceAverageColors(polyData, textureImage, "RGB")

      # Convert face colors to Lab space for clustering assignment
      faceColorsLab = self.rgb_to_lab(faceColors)

      # Apply luminosity normalization if it was used during clustering
      textureFilename = os.path.basename(texturePath)
      if clusteringPipeline.normalizeLuminosity:
        lcStats = clusteringPipeline.perTextureLCStats.get(textureFilename)
        pooledStats = clusteringPipeline.pooledLCStats
        if lcStats is not None and pooledStats is not None:
          mu_img, sd_img = lcStats
          mu_pool, sd_pool = pooledStats
          faceColorsLab = self._applyLCTransform(faceColorsLab, mu_img, sd_img, mu_pool, sd_pool)

      # Convert reconstructed colors to Lab space for distance calculation
      reconstructedColorsRgb = (reconstructedColors * 255).astype(np.uint8)
      reconstructedColorsLab = self.rgb_to_lab(reconstructedColorsRgb)

      # Assign each face to the nearest reconstructed cluster color
      numFaces = len(faceColorsLab)
      numClusters = len(reconstructedColorsLab)

      quantizedColors = np.zeros_like(faceColors)

      # Check if subsampling was used
      if clusteringPipeline.subsampledFaceIndices is not None:
        subsampledFaceIndices = clusteringPipeline.subsampledFaceIndices
        nearestNeighborMapping = clusteringPipeline.nearestNeighborMapping

        # Assign subsampled faces to nearest cluster
        subsampledFaceColors = faceColorsLab[subsampledFaceIndices]
        distances = np.zeros((len(subsampledFaceIndices), numClusters))
        for j in range(numClusters):
          distances[:, j] = np.linalg.norm(subsampledFaceColors - reconstructedColorsLab[j], axis=1)

        clusterAssignments = np.argmin(distances, axis=1)

        # Apply colors to subsampled faces
        for i, faceIdx in enumerate(subsampledFaceIndices):
          clusterIdx = clusterAssignments[i]
          quantizedColors[faceIdx] = reconstructedColorsRgb[clusterIdx]

        # Propagate to non-subsampled faces
        for faceIdx in range(numFaces):
          if faceIdx not in subsampledFaceIndices:
            nearestSubsampledIdx = nearestNeighborMapping.get(faceIdx, subsampledFaceIndices[0])
            subsampledArrayIdx = np.where(subsampledFaceIndices == nearestSubsampledIdx)[0]
            if len(subsampledArrayIdx) > 0:
              clusterIdx = clusterAssignments[subsampledArrayIdx[0]]
              quantizedColors[faceIdx] = reconstructedColorsRgb[clusterIdx]
      else:
        # No subsampling - assign all faces
        distances = np.zeros((numFaces, numClusters))
        for j in range(numClusters):
          distances[:, j] = np.linalg.norm(faceColorsLab - reconstructedColorsLab[j], axis=1)

        clusterAssignments = np.argmin(distances, axis=1)

        for i in range(numFaces):
          clusterIdx = clusterAssignments[i]
          quantizedColors[i] = reconstructedColorsRgb[clusterIdx]

      # Apply colors to model
      cellData = polyData.GetCellData()
      colorArray = vtk.vtkUnsignedCharArray()
      colorArray.SetNumberOfComponents(3)
      colorArray.SetName("Colors")
      colorArray.SetNumberOfTuples(numFaces)

      for i in range(numFaces):
        colorArray.SetTuple3(i, int(quantizedColors[i, 0]), int(quantizedColors[i, 1]), int(quantizedColors[i, 2]))

      cellData.SetScalars(colorArray)
      polyData.Modified()
      modelNode.GetDisplayNode().SetScalarVisibility(True)

      return True

    except Exception as e:
      print(f"Error applying morphospace colors: {e}")
      import traceback
      traceback.print_exc()
      return False

  def applyMorphospaceColorsSubsampled(self, modelNode, subsampledColorsLab, clusteringPipeline):
    """
    Apply morphospace-generated colors to the model using subsampled face colors directly

    This method takes reconstructed subsampled face colors from PCA inverse transform
    and applies them to the model, propagating to non-subsampled faces via nearest neighbor.

    Args:
        modelNode: VTK model node to apply colors to
        subsampledColorsLab: Reconstructed subsampled face colors (n_subsampled x 3) in Lab space
        clusteringPipeline: ClusteringPipeline object with subsampling information

    Returns:
        bool: True if successful, False otherwise
    """
    try:
      # Get model polydata
      polyData = modelNode.GetPolyData()
      if not polyData:
        return False

      numFaces = polyData.GetNumberOfCells()

      # Get subsampling information
      subsampledFaceIndices = clusteringPipeline.subsampledFaceIndices
      nearestNeighborMapping = clusteringPipeline.nearestNeighborMapping

      if subsampledFaceIndices is None:
        print("Error: No subsampled face indices in clustering pipeline")
        return False

      # Convert Lab colors to RGB
      subsampledColorsRgb = self.lab_to_rgb(subsampledColorsLab)

      # Create color array for all faces
      allFaceColors = np.zeros((numFaces, 3), dtype=np.uint8)

      # Assign colors to subsampled faces
      for i, faceIdx in enumerate(subsampledFaceIndices):
        allFaceColors[faceIdx] = subsampledColorsRgb[i]

      # Propagate colors to non-subsampled faces using nearest neighbor mapping
      if nearestNeighborMapping is not None:
        # Create a mapping from subsampled face index to array index for fast lookup
        subsampledIndexToArrayIdx = {faceIdx: i for i, faceIdx in enumerate(subsampledFaceIndices)}

        for faceIdx in range(numFaces):
          if faceIdx not in subsampledIndexToArrayIdx:
            # Find the nearest subsampled face
            nearestSubsampledIdx = nearestNeighborMapping[faceIdx]
            # Get the array index for this subsampled face
            arrayIdx = subsampledIndexToArrayIdx.get(nearestSubsampledIdx)
            if arrayIdx is not None:
              allFaceColors[faceIdx] = subsampledColorsRgb[arrayIdx]

      # Apply colors to model
      cellData = polyData.GetCellData()
      colorArray = vtk.vtkUnsignedCharArray()
      colorArray.SetNumberOfComponents(3)
      colorArray.SetName("Colors")
      colorArray.SetNumberOfTuples(numFaces)

      for i in range(numFaces):
        colorArray.SetTuple3(i, int(allFaceColors[i, 0]), int(allFaceColors[i, 1]), int(allFaceColors[i, 2]))

      cellData.SetScalars(colorArray)
      polyData.Modified()
      modelNode.GetDisplayNode().SetScalarVisibility(True)

      return True

    except Exception as e:
      print(f"Error applying morphospace subsampled colors: {e}")
      import traceback
      traceback.print_exc()
      return False

  def performPopulationAnalysis(self, modelNode, textureDir, textureFiles, clusterCenters=None, clusteringPipeline=None,
                                 faceAreas=None, dimReductionMethod="PCA", n_components=3, progressCallback=None, logCallback=None):
    """
    Perform population analysis by creating area-weighted color vectors and dimensionality reduction

    Args:
        modelNode: VTK model node to analyze
        textureDir: Directory containing texture files
        textureFiles: List of texture filenames
        clusterCenters: Pre-computed cluster centers (RGB colors) - for backward compatibility
        clusteringPipeline: ClusteringPipeline object (preferred, overrides clusterCenters)
        faceAreas: Pre-computed face areas
        dimReductionMethod: "PCA", "UMAP", or "ICA"
        n_components: Number of components for dimensionality reduction (default: 3, used for PCA and ICA)
        progressCallback: Function to call with progress updates (0-100)
        logCallback: Function to call with log messages

    Returns:
        dict with 'success', 'reduced_data', 'texture_names', 'method', and 'model' (if PCA or ICA)
    """
    try:
      if logCallback:
        logCallback(f"Starting population analysis with {dimReductionMethod}...")

      if progressCallback:
        progressCallback(5)

      # Check if required libraries are available
      if not SKLEARN_AVAILABLE:
        if logCallback:
          logCallback("Error: sklearn not available for dimensionality reduction")
        return {"success": False}

      if dimReductionMethod == "UMAP" and not UMAP_AVAILABLE:
        if logCallback:
          logCallback("Error: UMAP not available. Please install umap-learn.")
        return {"success": False}

      if dimReductionMethod == "ICA" and not SKLEARN_AVAILABLE:
        if logCallback:
          logCallback("Error: sklearn not available for ICA")
        return {"success": False}

      # Get model polydata
      polyData = modelNode.GetPolyData()
      if not polyData:
        if logCallback:
          logCallback("Error: No polydata in model")
        return {"success": False}

      # Determine which clustering data to use
      if clusteringPipeline is not None:
        # Use pipeline (preferred)
        if logCallback:
          logCallback("Using clustering pipeline for population analysis")

        numClusters = clusteringPipeline.consolidatedClusters
        useLuminosityNorm = clusteringPipeline.normalizeLuminosity
        pooledStats = clusteringPipeline.pooledLCStats if useLuminosityNorm else None

        # Use reference centroids for consistent cluster ordering
        referenceCentroidsLab = clusteringPipeline.referenceCentroids

      elif clusterCenters is not None:
        # Backward compatibility
        if logCallback:
          logCallback("Using legacy cluster centers for population analysis")

        numClusters = len(clusterCenters)
        useLuminosityNorm = False
        pooledStats = None
        referenceCentroidsLab = self.rgb_to_lab(clusterCenters)

      else:
        if logCallback:
          logCallback("Error: Neither clusteringPipeline nor clusterCenters provided")
        return {"success": False}

      # Check if subsampling was used
      useSubsampling = (clusteringPipeline is not None and
                       clusteringPipeline.subsampledFaceIndices is not None)

      if useSubsampling:
        subsampledFaceIndices = clusteringPipeline.subsampledFaceIndices
        numSubsampledFaces = len(subsampledFaceIndices)
        if logCallback:
          logCallback(f"Creating subsampled face color vectors for {len(textureFiles)} textures...")
          logCallback(f"Using {numSubsampledFaces} subsampled faces (flattened to {numSubsampledFaces * 3} dimensions)")
      else:
        if logCallback:
          logCallback(f"Creating area-weighted color vectors for {len(textureFiles)} textures...")
          logCallback(f"Using {numClusters} consolidated clusters")

      # Create color vectors for each texture
      textureVectors = []
      textureNames = []

      for i, textureFile in enumerate(textureFiles):
        texturePath = os.path.join(textureDir, textureFile)

        if logCallback:
          logCallback(f"Processing texture {i+1}/{len(textureFiles)}: {textureFile}")

        try:
          # Load texture image
          textureImage = imageio.imread(texturePath)
          if len(textureImage.shape) != 3 or textureImage.shape[2] < 3:
            if logCallback:
              logCallback(f"Warning: Skipping invalid texture format: {textureFile}")
            continue
        except Exception as e:
          if logCallback:
            logCallback(f"Warning: Failed to load texture {textureFile}: {e}")
          continue

        # Calculate face average colors for this texture
        faceColors = self._calculateFaceAverageColors(polyData, textureImage, "RGB")
        if faceColors is None:
          if logCallback:
            logCallback(f"Warning: Failed to calculate face colors for {textureFile}")
          continue

        # Apply neighbor averaging if it was used during clustering
        if clusteringPipeline is not None and clusteringPipeline.useNeighborAverage:
          if logCallback:
            logCallback(f"  Applying neighbor average smoothing...")

          faceColors = self._applyNeighborAveraging(polyData, faceColors, clusteringPipeline.faceAdjacency, logCallback)
          if faceColors is None:
            if logCallback:
              logCallback(f"  Warning: Neighbor averaging failed, using original colors")
            faceColors = self._calculateFaceAverageColors(polyData, textureImage, "RGB")

        # Convert face colors to Lab space
        faceColorsLab = self.rgb_to_lab(faceColors)

        # Apply luminosity normalization if it was used during clustering
        if useLuminosityNorm and clusteringPipeline is not None:
          lcStats = clusteringPipeline.perTextureLCStats.get(textureFile)
          if lcStats is not None and pooledStats is not None:
            mu_img, sd_img = lcStats
            mu_pool, sd_pool = pooledStats
            faceColorsLab = self._applyLCTransform(faceColorsLab, mu_img, sd_img, mu_pool, sd_pool)

        # Create color vector based on whether subsampling was used
        if useSubsampling:
          # New approach: use subsampled face colors flattened into a single vector
          subsampledFaceColors = faceColorsLab[subsampledFaceIndices]  # (N_s, 3)
          colorVector = subsampledFaceColors.flatten()  # (N_s*3,)

          if logCallback:
            logCallback(f"  Created subsampled color vector with shape {colorVector.shape}")
        else:
          # Original approach: area-weighted color vector
          colorVector = np.zeros(numClusters)

          # Vectorized distance calculation for better performance
          numFaces = len(faceColorsLab)

          # Reshape for broadcasting: faces (N,1,3) and clusters (1,K,3)
          faceColorsExpanded = faceColorsLab[:, np.newaxis, :]  # (N, 1, 3)
          clusterColorsExpanded = referenceCentroidsLab[np.newaxis, :, :]  # (1, K, 3)

          # Calculate Euclidean distances in Lab space
          distances = np.sqrt(np.sum((faceColorsExpanded - clusterColorsExpanded) ** 2, axis=2))  # (N, K)

          # Find nearest cluster for each face
          nearestClusters = np.argmin(distances, axis=1)  # (N,)

          # Add face areas to corresponding clusters
          for faceIdx in range(min(numFaces, len(faceAreas))):
            nearestCluster = nearestClusters[faceIdx]
            colorVector[nearestCluster] += faceAreas[faceIdx]

          # Normalize vector to unit length
          vectorNorm = np.linalg.norm(colorVector)
          if vectorNorm > 0:
            colorVector = colorVector / vectorNorm

        textureVectors.append(colorVector)
        textureNames.append(os.path.splitext(textureFile)[0])  # Remove extension

        # Update progress
        progress = 10 + (i + 1) * 60 / len(textureFiles)
        if progressCallback:
          progressCallback(progress)

      if not textureVectors:
        if logCallback:
          logCallback("Error: No valid texture vectors created")
        return {"success": False}

      if logCallback:
        if useSubsampling:
          logCallback(f"Created {len(textureVectors)} subsampled face color vectors (dimension: {len(textureVectors[0])})")
        else:
          logCallback(f"Created {len(textureVectors)} area-weighted color vectors")
        logCallback(f"Performing {dimReductionMethod} dimensionality reduction...")

      if progressCallback:
        progressCallback(75)

      # Convert to numpy array
      textureVectors = np.array(textureVectors)

      # Perform dimensionality reduction
      if dimReductionMethod == "PCA":
        # Use n_components parameter for PCA
        reducer = PCA(n_components=n_components, random_state=42)
        reducedData = reducer.fit_transform(textureVectors)

        if logCallback:
          explained_variance = reducer.explained_variance_ratio_
          variance_str = ", ".join([f"PC{i+1}={explained_variance[i]:.3f}" for i in range(min(len(explained_variance), 3))])
          logCallback(f"PCA explained variance: {variance_str}")

      elif dimReductionMethod == "ICA":
        # Use n_components parameter for ICA
        reducer = FastICA(n_components=n_components, random_state=42, max_iter=500)
        reducedData = reducer.fit_transform(textureVectors)

        if logCallback:
          logCallback(f"ICA computed {n_components} independent components")
          logCallback(f"ICA convergence: {reducer.n_iter_} iterations")

      elif dimReductionMethod == "UMAP":
        # UMAP always uses 2 components for visualization
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=min(15, len(textureVectors)-1))
        reducedData = reducer.fit_transform(textureVectors)

      if progressCallback:
        progressCallback(90)

      if logCallback:
        logCallback(f"Population analysis data preparation completed successfully!")
        if dimReductionMethod in ["PCA", "ICA"]:
          logCallback(f"Prepared data for {len(textureNames)} textures in {n_components}D {dimReductionMethod} space")
        else:
          logCallback(f"Prepared data for {len(textureNames)} textures in 2D {dimReductionMethod} space")

      if progressCallback:
        progressCallback(100)

      # Return results with model if applicable
      result = {
        "success": True,
        "reduced_data": reducedData,
        "texture_names": textureNames,
        "method": dimReductionMethod,
        "use_subsampling": useSubsampling
      }

      # Include model for axis selection and morphospace
      if dimReductionMethod in ["PCA", "ICA"]:
        result["model"] = reducer  # Generic model key for both PCA and ICA
        result["pca_model"] = reducer  # Keep for backward compatibility with PCA
        result["n_components"] = n_components
      elif dimReductionMethod == "UMAP":
        result["model"] = reducer  # Generic model key for UMAP
        result["n_components"] = 2  # UMAP always uses 2 components

      return result

    except Exception as e:
      if logCallback:
        logCallback(f"Error in population analysis: {str(e)}")
      import traceback
      traceback.print_exc()
      return {"success": False}




  def applyQuantizedFaceColorsFromTexture(self, modelNode, texturePath, numClusters, useHighContrastPalette=False, progressCallback=None, logCallback=None):
    """
    Apply quantized average face colors from a texture to a model

    Args:
        modelNode: VTK model node to apply colors to
        texturePath: Path to the texture image file
        numClusters: Number of color clusters for quantization
        useHighContrastPalette: If True, use high contrast palette instead of quantized colors
        progressCallback: Function to call with progress updates (0-100)
        logCallback: Function to call with log messages

    Returns:
        bool: True if successful, False otherwise
    """
    try:
      if logCallback:
        logCallback(f"Loading texture for quantization: {os.path.basename(texturePath)}")

      if progressCallback:
        progressCallback(5)

      # Load texture image
      try:
        textureImage = imageio.imread(texturePath)
        if len(textureImage.shape) != 3 or textureImage.shape[2] < 3:
          if logCallback:
            logCallback("Error: Invalid texture format")
          return False
      except Exception as e:
        if logCallback:
          logCallback(f"Error loading texture: {e}")
        return False

      # Get model polydata
      polyData = modelNode.GetPolyData()
      if not polyData:
        if logCallback:
          logCallback("Error: No polydata in model")
        return False

      if progressCallback:
        progressCallback(15)

      # Calculate face average colors
      if logCallback:
        logCallback("Calculating average face colors...")

      faceColors = self._calculateFaceAverageColors(polyData, textureImage, "RGB")
      if faceColors is None:
        if logCallback:
          logCallback("Error: Failed to calculate face colors")
        return False

      if logCallback:
        logCallback(f"Calculated colors for {len(faceColors)} faces")

      if progressCallback:
        progressCallback(40)

      # Quantize the colors using k-means in Lab space
      if logCallback:
        logCallback(f"Quantizing colors to {numClusters} clusters...")

      quantResult = self.quantize_colors_lab_kmeans(
        faceColors, numClusters,
        use_high_contrast=useHighContrastPalette,
        progressCallback=lambda p: progressCallback(40 + p * 0.4) if progressCallback else None,
        logCallback=logCallback
      )

      if not quantResult.get("success", False):
        if logCallback:
          logCallback("Error: Color quantization failed")
        return False

      quantizedColors = quantResult["quantized_colors"]
      clusterCenters = quantResult["cluster_centers"]

      if logCallback:
        logCallback(f"Quantization successful. Cluster centers (RGB):")
        for i, center in enumerate(clusterCenters):
          logCallback(f"  Cluster {i+1}: [{center[0]}, {center[1]}, {center[2]}]")

      if progressCallback:
        progressCallback(85)

      # Apply quantized colors to faces as cell data
      nFaces = polyData.GetNumberOfCells()
      if len(quantizedColors) != nFaces:
        if logCallback:
          logCallback(f"Error: Color count mismatch. Expected {nFaces}, got {len(quantizedColors)}")
        return False

      if logCallback:
        logCallback(f"Applying quantized colors to {nFaces} faces")

      # Create VTK color arrays for RGB components
      colorArrayR = vtk.vtkUnsignedCharArray()
      colorArrayR.SetName("QuantizedColorR")
      colorArrayR.SetNumberOfComponents(1)
      colorArrayR.SetNumberOfTuples(nFaces)

      colorArrayG = vtk.vtkUnsignedCharArray()
      colorArrayG.SetName("QuantizedColorG")
      colorArrayG.SetNumberOfComponents(1)
      colorArrayG.SetNumberOfTuples(nFaces)

      colorArrayB = vtk.vtkUnsignedCharArray()
      colorArrayB.SetName("QuantizedColorB")
      colorArrayB.SetNumberOfComponents(1)
      colorArrayB.SetNumberOfTuples(nFaces)

      # Combined RGB array
      colorArrayRGB = vtk.vtkUnsignedCharArray()
      colorArrayRGB.SetName("QuantizedColors")
      colorArrayRGB.SetNumberOfComponents(3)
      colorArrayRGB.SetNumberOfTuples(nFaces)

      for i in range(nFaces):
        color = quantizedColors[i]
        colorArrayR.SetValue(i, int(color[0]))
        colorArrayG.SetValue(i, int(color[1]))
        colorArrayB.SetValue(i, int(color[2]))
        colorArrayRGB.SetTuple3(i, int(color[0]), int(color[1]), int(color[2]))

      # Add arrays to cell data
      polyData.GetCellData().AddArray(colorArrayR)
      polyData.GetCellData().AddArray(colorArrayG)
      polyData.GetCellData().AddArray(colorArrayB)
      polyData.GetCellData().AddArray(colorArrayRGB)
      polyData.GetCellData().SetActiveScalars("QuantizedColors")

      # Update display
      modelNode.CreateDefaultDisplayNodes()
      displayNode = modelNode.GetDisplayNode()
      if displayNode:
        # Turn off texture first
        displayNode.SetTextureImageDataConnection(None)

        # Enable scalar visibility and set to use RGB colors directly
        displayNode.SetScalarVisibility(True)
        displayNode.SetActiveScalarName("QuantizedColors")

        # Set to use cell data (not point data)
        displayNode.SetActiveAttributeLocation(vtk.vtkDataObject.CELL)

        # Use RGB color mode instead of lookup table
        displayNode.SetScalarRangeFlag(slicer.vtkMRMLDisplayNode.UseDirectMapping)

      if progressCallback:
        progressCallback(100)

      if logCallback:
        logCallback("Quantized face coloring applied successfully")

      return True

    except Exception as e:
      if logCallback:
        logCallback(f"Error in applyQuantizedFaceColorsFromTexture: {str(e)}")
      import traceback
      traceback.print_exc()
      return False

  def setMorphospaceLayout(self):
    """
    Set up a custom layout with 3D view on the left and plot view on the right.

    This layout is optimized for morphospace exploration, allowing simultaneous
    viewing of the 3D model and the PCA/ICA/UMAP scatter plot.

    Returns:
        tuple: (custom_layout_id, plot_view_node) or (None, None) on error
    """
    try:
      lm = slicer.app.layoutManager()
      layoutNode = lm.layoutLogic().GetLayoutNode()

      CUSTOM_LAYOUT_ID = 501  # Using 501 as custom layout ID (any unused integer >= 100 works)

      layoutXml = """<layout type="horizontal" split="true"><item stretch="1"><view class="vtkMRMLViewNode" singletontag="1"><property name="viewlabel" action="default">1</property></view></item><item stretch="1"><view class="vtkMRMLPlotViewNode" singletontag="PlotViewMorphospace"><property name="viewlabel" action="default">P</property></view></item></layout>"""

      # Register and apply the custom layout
      layoutNode.AddLayoutDescription(CUSTOM_LAYOUT_ID, layoutXml)
      lm.setLayout(CUSTOM_LAYOUT_ID)

      # Process events to ensure layout is fully applied
      for _ in range(3):
        slicer.app.processEvents()

      # Get the plot view node from the layout
      plotViewNode = None
      for attempt in range(10):
        plotWidget = lm.plotWidget(0)
        if plotWidget:
          plotViewNode = plotWidget.mrmlPlotViewNode()
          if plotViewNode:
            break
        slicer.app.processEvents()
        import time
        time.sleep(0.05)

      return CUSTOM_LAYOUT_ID, plotViewNode

    except Exception:
      return None, None

class ColorTheme:
    """Centralized color theme management for InterDeCA.

    Provides consistent styling across the module interface with
    support for both light and dark themes that adapt to Slicer's
    current appearance settings.
    """

    @staticmethod
    def getTheme():
        """Gets color theme based on Slicer's current theme.

        Returns:
            dict: Theme colors appropriate for current mode
        """
        # Checks if Slicer is in dark mode by examining window palette
        palette = qt.QApplication.palette()  # Gets application palette
        isDarkMode = palette.color(qt.QPalette.Window).lightness() < 128  # Determines if dark

        if isDarkMode:
            return ColorTheme.getDarkTheme()  # Returns dark theme colors
        else:
            return ColorTheme.getLightTheme()  # Returns light theme colors
    
    @staticmethod
    def getLightTheme():
        return {
            'primary': '#4CAF50',
            'primary_hover': '#45A049',
            'primary_pressed': '#3D8B40',
            'secondary': '#87CEEB',
            'secondary_hover': '#6BB6E8',
            'secondary_pressed': '#4FA8D8',
            'accent': '#9C27B0',
            'accent_hover': '#8E24AA',
            'accent_pressed': '#7B1FA2',
            'danger': '#FF6B6B',
            'danger_hover': '#FF5252',
            'danger_pressed': '#E53935',
            'neutral': '#607D8B',
            'neutral_hover': '#546E7A',
            'neutral_pressed': '#455A64',
            'text_primary': '#2C3E50',
            'text_secondary': 'palette(disabled-text)',
            'text_on_primary': 'white',
            'disabled_bg': '#CCCCCC',
            'disabled_text': '#666666'
        }
    
    @staticmethod
    def getDarkTheme():
        return {
            'primary': '#66BB6A',
            'primary_hover': '#5CB85C',
            'primary_pressed': '#4CAF50',
            'secondary': '#64B5F6',
            'secondary_hover': '#42A5F5',
            'secondary_pressed': '#2196F3',
            'accent': '#BA68C8',
            'accent_hover': '#AB47BC',
            'accent_pressed': '#9C27B0',
            'danger': '#EF5350',
            'danger_hover': '#E53935',
            'danger_pressed': '#D32F2F',
            'neutral': '#78909C',
            'neutral_hover': '#607D8B',
            'neutral_pressed': '#546E7A',
            'text_primary': '#FFFFFF',
            'text_secondary': 'palette(disabled-text)',
            'text_on_primary': 'white',
            'disabled_bg': '#424242',
            'disabled_text': '#9E9E9E'
        }
    
    @staticmethod
    def getButtonStyle(color_type='primary', disabled_style=True):
        """Generate button stylesheet with theme colors"""
        theme = ColorTheme.getTheme()
        
        style = f"""
        QPushButton {{
            background-color: {theme[color_type]};
            color: {theme['text_on_primary']};
            font-weight: bold;
            border: none;
            border-radius: 4px;
            padding: 6px 12px;
            min-height: 20px;
        }}
        QPushButton:hover {{
            background-color: {theme[color_type + '_hover']};
        }}
        QPushButton:pressed {{
            background-color: {theme[color_type + '_pressed']};
        }}"""
        
        if disabled_style:
            style += f"""
        QPushButton:disabled {{
            background-color: {theme['disabled_bg']};
            color: {theme['disabled_text']};
        }}"""
        
        return style
    
    @staticmethod
    def getLabelStyle(style_type='secondary'):
        """Generate label stylesheet with theme colors"""
        theme = ColorTheme.getTheme()
        return f"QLabel {{ color: {theme['text_secondary']}; font-style: italic; }}"
    
    @staticmethod
    def getProgressLabelStyle():
        """Get progress label style"""
        return """
        QLabel { 
            color: palette(link); 
            font-weight: bold; 
        }"""
    
    @staticmethod
    def getHeaderStyle():
        """Get header/collapsible button style with theme-aware colors"""
        theme = ColorTheme.getTheme()
        return f"""
        ctkCollapsibleButton {{ 
            border: 1px solid palette(mid);
            border-radius: 4px;
            background-color: palette(window);
        }}
        /* Style the QToolButton which is the actual header/title */
        ctkCollapsibleButton QToolButton {{
            font-weight: bold; 
            color: {theme['text_primary']}; 
            background-color: palette(alternate-base);
            padding: 4px;
            border: none;
            text-align: left;
        }}
        /* Hover effect only on the header button */
        ctkCollapsibleButton QToolButton:hover {{
            background-color: palette(highlight);
            color: palette(highlighted-text);
        }}
        """
    
    @staticmethod
    def getSectionLabelStyle():
        """Get section label style with theme-aware colors"""
        theme = ColorTheme.getTheme()
        return f"""
        QLabel {{ 
            font-weight: bold; 
            color: {theme['text_primary']}; 
            margin-top: 10px;
        }}
        """
    
    @staticmethod
    def getStatusLabelStyle(status_type='neutral'):
        """Get status label style with theme-aware colors"""
        theme = ColorTheme.getTheme()
        if status_type == 'success':
            color = theme['primary']
        elif status_type == 'error':
            color = theme['danger']
        elif status_type == 'warning':
            color = '#ff8c00'  # Orange that works in both themes
        elif status_type == 'progress':
            color = theme['secondary']
        else:  # neutral
            color = theme['text_secondary']
        
        return f"""
        QLabel {{ 
            color: {color}; 
            font-style: italic;
        }}
        """
    
    @staticmethod
    def getSeparatorStyle():
        """Get separator line style with theme-aware colors"""
        return """
        QFrame { 
            color: palette(mid); 
            margin: 10px 0px; 
        }
        """
    
    @staticmethod
    def getValidationLabelStyle(validation_type='neutral'):
        """Get validation label style with theme-aware colors"""
        if validation_type == 'success':
            color = 'palette(positive)'
        elif validation_type == 'error':
            color = 'palette(negative)'
        elif validation_type == 'warning':
            color = '#ff8c00'  # Orange that works in both themes
        else:  # neutral
            color = 'palette(disabled-text)'
        
        return f"""
        QLabel {{ 
            color: {color}; 
            font-style: italic;
        }}
        """
    
    @staticmethod
    def getComboBoxStyle():
        """Get combobox/dropdown style with theme-aware colors (for both QComboBox and qMRMLNodeComboBox)"""
        return """
        QComboBox, qMRMLNodeComboBox {
            color: palette(window-text);
            background-color: palette(base);
            border: 1px solid palette(mid);
            border-radius: 3px;
            padding: 3px 5px;
        }
        QComboBox:on, qMRMLNodeComboBox:on {
            color: palette(window-text);
        }
        QComboBox:hover, qMRMLNodeComboBox:hover {
            border: 1px solid palette(highlight);
        }
        QComboBox:disabled, qMRMLNodeComboBox:disabled {
            color: palette(disabled-text);
            background-color: palette(window);
        }
        QComboBox::drop-down, qMRMLNodeComboBox::drop-down {
            border: none;
        }
        QComboBox QAbstractItemView, qMRMLNodeComboBox QAbstractItemView {
            color: palette(window-text);
            background-color: palette(base);
            selection-background-color: palette(highlight);
            selection-color: palette(highlighted-text);
        }
        """
