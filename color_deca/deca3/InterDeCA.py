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
try:
    from sklearn.decomposition import PCA, FastICA
    from sklearn.manifold import TSNE
    from sklearn.cluster import KMeans, MiniBatchKMeans
    import umap
    SKLEARN_AVAILABLE = True
    UMAP_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    UMAP_AVAILABLE = False
    print("Warning: sklearn and/or umap not available. Colors EDA functionality will be limited.")

try:
    from skimage import color as skimage_color
    from skimage.color import deltaE_ciede2000
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    print("Warning: scikit-image not available. Color quantization functionality will be limited.")


# Import functions from the deca module to avoid duplication
import sys
import os

try:
    from deca.deca import decaLogic
    print('Successfully imported DeCA module!')
    print(f'decaLogic class: {decaLogic}')
except ImportError as e:
    # Handle case where deca module is not available
    print(f'Could not import DeCA module: {e}')
    decaLogic = None

print(f'Final decaLogic value: {decaLogic}')

#
# DeCA
#

class InterDeCA(ScriptedLoadableModule):
  """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """

  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "InterDeCA" # TODO make this more human readable by adding spaces
    self.parent.categories = ["SlicerMorph.DeCA Toolbox"]
    self.parent.dependencies = []
    self.parent.contributors = ["Sara Rolfe (SCRI)"] # replace with "Firstname Lastname (Organization)"
    self.parent.helpText = """
      This module provides several flexible workflows for finding and analyzing dense correspondence points between models.
      """
    self.parent.helpText += self.getDefaultModuleDocumentationLink()
    self.parent.acknowledgementText = """This extension was developed by funding from National Institutes of Health (OD032627 and HD104435) to A. Murat Maga (SCRI)
      """ # replace with organization, grant and thanks.

#
# DeCAWidget
#

class InterDeCAWidget(ScriptedLoadableModuleWidget):
  """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """

  def setup(self):
    ScriptedLoadableModuleWidget.setup(self)

    # This variable will hold our temporary interpolation model
    self.interpolatedModelNode = None
    self.selectedOriginalModelNode = None
    self.lastDeCAAlignedModelsPath = None
    
    # Progress tracking
    self.progressBar = None
    
    # Initialize persistent data storage
    self.settings = qt.QSettings()
    self.settings.beginGroup("InterDeCA")
    self.progressLabel = None
    self.cancelButton = None
    self.currentOperation = None

    # Set up tabs to split workflow
    tabsWidget = qt.QTabWidget()
    self.tabsWidget = tabsWidget
    DeCATab = qt.QWidget()
    DeCATabLayout = qt.QFormLayout(DeCATab)
    visualizeTab = qt.QWidget()
    visualizeTabLayout = qt.QFormLayout(visualizeTab)
    colorsEDATab = qt.QWidget()
    colorsEDATabLayout = qt.QFormLayout(colorsEDATab)
    recolorTab = qt.QWidget()
    recolorTabLayout = qt.QFormLayout(recolorTab)
    multiRecolorTab = qt.QWidget()
    multiRecolorTabLayout = qt.QFormLayout(multiRecolorTab)

    tabsWidget.addTab(DeCATab, "DeCA")
    tabsWidget.addTab(visualizeTab, "Visualize Results")
    tabsWidget.addTab(colorsEDATab, "Colors EDA")
    tabsWidget.addTab(recolorTab, "Recolor")
    tabsWidget.addTab(multiRecolorTab, "MultiRecolor")

    self.layout.addWidget(tabsWidget)

    ################################### DeCA Tab ###################################
    # ... (The DeCA Tab code remains unchanged) ...
    # Layout within the DeCA tab
    DeCAWidget=ctk.ctkCollapsibleButton()
    DeCAWidgetLayout = qt.QFormLayout(DeCAWidget)
    DeCAWidget.text = "Dense Correspondence I/0"
    DeCATabLayout.addRow(DeCAWidget)

    #
    # Select Atlas Type
    #
    self.calculateAtlasOptionDC=qt.QRadioButton()
    self.calculateAtlasOptionDC.setChecked(True)
    self.loadAtlasOptionDC=qt.QRadioButton()
    DCAtlasButtonGroup = qt.QButtonGroup(DeCAWidget)
    DCAtlasButtonGroup.addButton(self.calculateAtlasOptionDC)
    DCAtlasButtonGroup.addButton(self.loadAtlasOptionDC)
    DeCAWidgetLayout.addRow("Create atlas: ", self.calculateAtlasOptionDC)
    DeCAWidgetLayout.addRow("Load atlas: ", self.loadAtlasOptionDC)

    #
    # Hidden atlas options
    self.atlasCollapsibleButtonDC = ctk.ctkCollapsibleButton()
    self.atlasCollapsibleButtonDC.text = "Atlas Options"
    self.atlasCollapsibleButtonDC.collapsed = True
    self.atlasCollapsibleButtonDC.enabled = False
    DeCAWidgetLayout.addRow(self.atlasCollapsibleButtonDC)
    atlasOptionLayout = qt.QFormLayout(self.atlasCollapsibleButtonDC)

    #
    # Select base mesh
    #
    self.DCBaseModelSelector = ctk.ctkPathLineEdit()
    self.DCBaseModelSelector.filters  = ctk.ctkPathLineEdit().Files
    self.DCBaseModelSelector.nameFilters=["Model (*.ply *.stl *.obj *.vtk *.vtp)"]
    atlasOptionLayout.addRow("Atlas model: ", self.DCBaseModelSelector)

    #
    # Select base landmarks
    #
    self.DCBaseLMSelector = ctk.ctkPathLineEdit()
    self.DCBaseLMSelector.filters  = ctk.ctkPathLineEdit().Files
    self.DCBaseLMSelector.nameFilters=["Point set (*.fcsv *.json *.mrk.json)"]
    atlasOptionLayout.addRow("Atlas landmarks: ", self.DCBaseLMSelector)

    #
    # Select Analysis Type
    #
    self.analysisTypeShape=qt.QRadioButton()
    self.analysisTypeShape.setChecked(True)
    self.analysisTypeSymmetry=qt.QRadioButton()
    DCAnalysisButtonGroup = qt.QButtonGroup(DeCAWidget)
    DCAnalysisButtonGroup.addButton(self.analysisTypeShape)
    DCAnalysisButtonGroup.addButton(self.analysisTypeSymmetry)
    DeCAWidgetLayout.addRow("Shape analysis: ", self.analysisTypeShape)
    DeCAWidgetLayout.addRow("Symmetry analysis: ", self.analysisTypeSymmetry)

    #
    # Hidden symmetry options
    #
    self.symmetryCollapsibleButton = ctk.ctkCollapsibleButton()
    self.symmetryCollapsibleButton.text = "Symmetry Options"
    self.symmetryCollapsibleButton.collapsed = True
    self.symmetryCollapsibleButton.enabled = False
    DeCAWidgetLayout.addRow(self.symmetryCollapsibleButton)
    symmetryOptionLayout = qt.QFormLayout(self.symmetryCollapsibleButton)
    self.landmarkIndexText=qt.QLineEdit()
    self.landmarkIndexText.setToolTip("No spaces. Seperate numbers by commas.  Example:  2,1,3,5,4")
    symmetryOptionLayout.addRow('Mirror landmark index', self.landmarkIndexText)

    #
    # Input Directories Section
    #
    self.inputDirCollapsibleButton = ctk.ctkCollapsibleButton()
    self.inputDirCollapsibleButton.text = "Input Directories"
    self.inputDirCollapsibleButton.collapsed = False
    DeCAWidgetLayout.addRow(self.inputDirCollapsibleButton)
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
    self.textureValidationLabelDC.setStyleSheet("QLabel { color: palette(disabled-text); font-style: italic; }")
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
    DeCAWidgetLayout.addRow(" ", qt.QLabel())

    # --- Blender integration ---
    self.blenderGroup = ctk.ctkCollapsibleButton()
    self.blenderGroup.text = "Blender (cleanup, UV, bake)"
    self.blenderGroup.collapsed = True
    DeCAWidgetLayout.addRow(self.blenderGroup)
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

    # Add spacing
    DeCAWidgetLayout.addRow(" ", qt.QLabel())

    #
    # Analysis Options Section
    #
    self.analysisOptionsCollapsibleButton = ctk.ctkCollapsibleButton()
    self.analysisOptionsCollapsibleButton.text = "Analysis Options"
    self.analysisOptionsCollapsibleButton.collapsed = False
    DeCAWidgetLayout.addRow(self.analysisOptionsCollapsibleButton)
    analysisOptionsLayout = qt.QFormLayout(self.analysisOptionsCollapsibleButton)

    # Remove scale option
    self.removeScaleCheckBoxDC = qt.QCheckBox()
    self.removeScaleCheckBoxDC.checked = False
    self.removeScaleCheckBoxDC.setToolTip("If checked, DeCA alignment will include isotropic scaling.")
    analysisOptionsLayout.addRow("Remove scale: ", self.removeScaleCheckBoxDC)

    # Error checking directory option
    self.writeErrorCheckBox = qt.QCheckBox()
    self.writeErrorCheckBox.checked = False
    self.writeErrorCheckBox.setToolTip("If checked, DeCA will create a directory of results for use in estimating point correspondence error.")
    analysisOptionsLayout.addRow("Create output for error checking: ", self.writeErrorCheckBox)

    # Add spacing
    DeCAWidgetLayout.addRow(" ", qt.QLabel())

    #
    # Progress tracking widgets
    #
    self.progressWidgetDC = qt.QWidget()
    self.progressWidgetDC.setVisible(False)
    progressLayout = qt.QVBoxLayout(self.progressWidgetDC)
    progressLayout.setContentsMargins(0, 0, 0, 0)
    
    self.progressBarDC = qt.QProgressBar()
    self.progressBarDC.setRange(0, 100)
    self.progressBarDC.setValue(0)
    progressLayout.addWidget(self.progressBarDC)
    
    self.progressLabelDC = qt.QLabel("Ready")
    # Use theme-aware color that works in both light and dark modes
    # This will automatically adapt to the current Slicer theme
    self.progressLabelDC.setStyleSheet("""
      QLabel { 
        color: palette(link); 
        font-weight: bold; 
      }
    """)
    progressLayout.addWidget(self.progressLabelDC)
    
    self.cancelButtonDC = qt.QPushButton("Cancel Operation")
    self.cancelButtonDC.setMaximumWidth(120)
    self.cancelButtonDC.setVisible(False)
    progressLayout.addWidget(self.cancelButtonDC)
    
    DeCAWidgetLayout.addRow("Progress: ", self.progressWidgetDC)

    #
    # Run DeCA Button
    #
    self.applyButtonDC = qt.QPushButton("Run DeCA")
    self.applyButtonDC.toolTip = "Run non-rigid alignment"
    self.applyButtonDC.enabled = False
    self.applyButtonDC.setStyleSheet("""
      QPushButton {
        background-color: #87CEEB;
        color: #2C3E50;
        font-weight: bold;
        border: none;
        border-radius: 5px;
        padding: 8px 16px;
        min-height: 30px;
      }
      QPushButton:hover {
        background-color: #6BB6E8;
      }
      QPushButton:pressed {
        background-color: #4FA8D8;
      }
      QPushButton:disabled {
        background-color: #CCCCCC;
        color: #666666;
      }
    """)
    DeCAWidgetLayout.addRow(self.applyButtonDC)
    

    #
    # Log Information
    #
    self.logInfoDC = qt.QPlainTextEdit()
    self.logInfoDC.setPlaceholderText("DeCA log information")
    self.logInfoDC.setReadOnly(True)
    DeCAWidgetLayout.addRow(self.logInfoDC)

    # Connections
    self.analysisTypeShape.connect('toggled(bool)', self.onToggleAnalysis)
    self.analysisTypeSymmetry.connect('toggled(bool)', self.onToggleAnalysis)
    self.calculateAtlasOptionDC.connect('toggled(bool)', self.onToggleAtlasDC)
    self.loadAtlasOptionDC.connect('toggled(bool)', self.onToggleAtlasDC)
    self.DCBaseModelSelector.connect('validInputChanged(bool)', self.onParameterSelectDC)
    self.DCBaseLMSelector.connect('validInputChanged(bool)', self.onParameterSelectDC)
    self.meshDirectoryDC.connect('validInputChanged(bool)', self.onParameterSelectDC)
    self.meshDirectoryDC.connect('currentPathChanged(QString)', self.onMeshDirectoryChangedDC)
    self.landmarkDirectoryDC.connect('validInputChanged(bool)', self.onParameterSelectDC)
    self.landmarkDirectoryDC.connect('currentPathChanged(QString)', self.onLandmarkDirectoryChangedDC)
    self.outputDirectoryDC.connect('validInputChanged(bool)', self.onParameterSelectDC)
    self.applyButtonDC.connect('clicked(bool)', self.onDCApplyButton)
    self.textureDirectoryDC.connect('validInputChanged(bool)', self.onParameterSelectDC)
    self.textureDirectoryDC.connect('currentPathChanged(QString)', self.onTextureDirectoryChangedDC)
    self.blenderExeEdit.connect('validInputChanged(bool)', self.onParameterSelectDC)
    self.cancelButtonDC.connect('clicked(bool)', self.onCancelOperationDC)



    ################################### Visualize Tab ###################################
    # Layout within the tab
    visualizeWidget=ctk.ctkCollapsibleButton()
    visualizeWidgetLayout = qt.QFormLayout(visualizeWidget)
    visualizeWidget.text = "Visualize Results"
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
    self.subjectIDBox.enabled = False
    self.heatmapFrameLayout.addRow("Subject ID: ", self.subjectIDBox)

    # --- Frame for Interpolation Visualization (bring back) ---
    self.interpolationFrame = qt.QFrame()
    self.interpolationFrameLayout = qt.QFormLayout(self.interpolationFrame)
    self.interpolationFrame.setVisible(False)  # hidden by default
    visualizeWidgetLayout.addRow(self.interpolationFrame)

    # Atlas model (target of interpolation)
    self.atlasModelSelect = slicer.qMRMLNodeComboBox()
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
    self.previewTextureCombo.setToolTip("Preview a xbaked atlas-space PNG on the atlas model.")
    visualizeWidgetLayout.addRow("Preview baked texture:", self.previewTextureCombo)
    self.previewTextureCombo.connect("currentIndexChanged(int)", self.onPreviewTextureSelected)
    
    # Add spacing before the visualization button
    visualizeWidgetLayout.addRow(" ", qt.QLabel())

    #
    # Landmark Lock/Unlock Controls
    #
    landmarkControlWidget = qt.QWidget()
    landmarkControlLayout = qt.QHBoxLayout(landmarkControlWidget)
    landmarkControlLayout.setContentsMargins(0, 0, 0, 0)
    
    self.landmarkLockButton = qt.QPushButton("Lock Landmarks")
    self.landmarkLockButton.setToolTip("Lock/unlock all landmarks to prevent accidental movement")
    self.landmarkLockButton.setStyleSheet("""
      QPushButton {
        background-color: #ff6b6b;
        color: white;
        font-weight: bold;
        border: none;
        border-radius: 5px;
        padding: 6px 12px;
        min-height: 25px;
      }
      QPushButton:hover {
        background-color: #ff5252;
      }
      QPushButton:pressed {
        background-color: #e53935;
      }
    """)
    self.landmarkLockButton.connect('clicked(bool)', self.onToggleLandmarkLock)
    landmarkControlLayout.addWidget(self.landmarkLockButton)
    
    visualizeWidgetLayout.addRow("Landmark Control:", landmarkControlWidget)
    
    # Add spacing before the visualization button
    visualizeWidgetLayout.addRow(" ", qt.QLabel())

    #
    # --- Mesh Region Selection Section ---
    #
    self.regionSelectionWidget = ctk.ctkCollapsibleButton()
    self.regionSelectionWidget.text = "Mesh Region Selection"
    self.regionSelectionWidget.collapsed = True
    visualizeWidgetLayout.addRow(self.regionSelectionWidget)
    regionLayout = qt.QFormLayout(self.regionSelectionWidget)
    
    # Target mesh selector for region selection
    self.regionMeshSelector = slicer.qMRMLNodeComboBox()
    self.regionMeshSelector.nodeTypes = (("vtkMRMLModelNode"), "")
    self.regionMeshSelector.setToolTip("Select the mesh to perform region selection on")
    self.regionMeshSelector.selectNodeUponCreation = False
    self.regionMeshSelector.noneEnabled = True
    self.regionMeshSelector.addEnabled = False
    self.regionMeshSelector.removeEnabled = False
    self.regionMeshSelector.showHidden = False
    self.regionMeshSelector.setMRMLScene(slicer.mrmlScene)
    regionLayout.addRow("Target Mesh:", self.regionMeshSelector)
    
    # Selection method combo
    self.selectionMethodCombo = qt.QComboBox()
    self.selectionMethodCombo.addItems([
        "Segment Editor (Paint/Scissors)", 
        "Landmark + Radius",
        "Multiple Landmarks + Radius"
    ])
    self.selectionMethodCombo.setToolTip("Choose how to select regions on the mesh")
    regionLayout.addRow("Selection Method:", self.selectionMethodCombo)
    
    # --- Segment Editor Method Controls ---
    self.segmentEditorFrame = qt.QFrame()
    self.segmentEditorLayout = qt.QFormLayout()
    self.segmentEditorFrame.setLayout(self.segmentEditorLayout)
    
    # Existing segmentation selector for loading saved work
    self.existingSegmentationSelector = slicer.qMRMLNodeComboBox()
    self.existingSegmentationSelector.nodeTypes = (("vtkMRMLSegmentationNode"), "")
    self.existingSegmentationSelector.setToolTip("Load existing segmentation data to continue working")
    self.existingSegmentationSelector.selectNodeUponCreation = False
    self.existingSegmentationSelector.noneEnabled = True
    self.existingSegmentationSelector.addEnabled = False
    self.existingSegmentationSelector.removeEnabled = False
    self.existingSegmentationSelector.showHidden = False
    self.existingSegmentationSelector.setMRMLScene(slicer.mrmlScene)
    self.segmentEditorLayout.addRow("Load Existing Segmentation:", self.existingSegmentationSelector)
    
    self.loadSegmentationButton = qt.QPushButton("Load Segmentation")
    self.loadSegmentationButton.setToolTip("Load and configure existing segmentation for editing")
    self.loadSegmentationButton.enabled = False
    self.loadSegmentationButton.setStyleSheet(ColorTheme.getButtonStyle('secondary'))
    self.segmentEditorLayout.addRow(self.loadSegmentationButton)
    
    # Add a separator line
    separator1 = qt.QFrame()
    separator1.setFrameShape(qt.QFrame.HLine)
    separator1.setFrameShadow(qt.QFrame.Sunken)
    self.segmentEditorLayout.addRow(separator1)
    
    self.setupSegmentEditorButton = qt.QPushButton("Setup New Segmentation")
    self.setupSegmentEditorButton.setToolTip("Create new segmentation from model and open Segment Editor")
    self.setupSegmentEditorButton.setStyleSheet(ColorTheme.getButtonStyle('primary'))
    self.segmentEditorLayout.addRow(self.setupSegmentEditorButton)
    
    # Add another separator line
    separator2 = qt.QFrame()
    separator2.setFrameShape(qt.QFrame.HLine)
    separator2.setFrameShadow(qt.QFrame.Sunken)
    self.segmentEditorLayout.addRow(separator2)
    
    self.exportSelectionButton = qt.QPushButton("Export Selected Region")
    self.exportSelectionButton.setToolTip("Export painted region back to a model")
    self.exportSelectionButton.enabled = False
    self.exportSelectionButton.setStyleSheet(ColorTheme.getButtonStyle('secondary'))
    self.segmentEditorLayout.addRow(self.exportSelectionButton)
    
    regionLayout.addRow(self.segmentEditorFrame)
    
    # --- Landmark Method Controls ---
    self.landmarkFrame = qt.QFrame()
    self.landmarkLayout = qt.QFormLayout()
    self.landmarkFrame.setLayout(self.landmarkLayout)
    self.landmarkFrame.setVisible(False)  # Hidden by default
    
    # Markup selector for selection points
    self.selectionMarkupSelector = slicer.qMRMLNodeComboBox()
    self.selectionMarkupSelector.nodeTypes = (("vtkMRMLMarkupsFiducialNode"), "")
    self.selectionMarkupSelector.setToolTip("Select markup points to define region centers")
    self.selectionMarkupSelector.selectNodeUponCreation = False
    self.selectionMarkupSelector.noneEnabled = True
    self.selectionMarkupSelector.addEnabled = True
    self.selectionMarkupSelector.removeEnabled = False
    self.selectionMarkupSelector.showHidden = False
    self.selectionMarkupSelector.setMRMLScene(slicer.mrmlScene)
    self.landmarkLayout.addRow("Selection Points:", self.selectionMarkupSelector)
    
    # Radius control
    self.selectionRadiusSlider = ctk.ctkSliderWidget()
    self.selectionRadiusSlider.minimum = 0.01
    self.selectionRadiusSlider.maximum = 10.0
    self.selectionRadiusSlider.singleStep = 0.01  # Set after min/max to avoid bounds issues
    self.selectionRadiusSlider.value = 0.5
    try:
        self.selectionRadiusSlider.decimals = 2
    except AttributeError:
        pass  # Some versions might not have this property
    self.selectionRadiusSlider.setToolTip("Radius around each point to select mesh vertices")
    self.landmarkLayout.addRow("Selection Radius:", self.selectionRadiusSlider)
    
    # Selected points display (for single point mode)
    self.selectedPointsLabel = qt.QLabel("No points selected")
    self.selectedPointsLabel.setToolTip("Currently selected landmark points")
    self.selectedPointsLabel.setStyleSheet(ColorTheme.getLabelStyle())
    self.landmarkLayout.addRow("Selected Points:", self.selectedPointsLabel)
    
    # Click to select landmark button
    self.clickSelectLandmarkButton = qt.QPushButton("Click to Select Landmark")
    self.clickSelectLandmarkButton.setToolTip("Click on a landmark in the 3D view to select it for region selection")
    self.clickSelectLandmarkButton.enabled = False
    self.clickSelectLandmarkButton.setStyleSheet(ColorTheme.getButtonStyle('secondary'))
    self.landmarkLayout.addRow("", self.clickSelectLandmarkButton)
    
    # Apply landmark selection button
    self.applyLandmarkSelectionButton = qt.QPushButton("Apply Landmark Selection")
    self.applyLandmarkSelectionButton.setToolTip("Apply region selection using landmarks")
    self.applyLandmarkSelectionButton.enabled = False
    self.applyLandmarkSelectionButton.setStyleSheet(ColorTheme.getButtonStyle('primary'))
    self.landmarkLayout.addRow(self.applyLandmarkSelectionButton)
    
    # Export landmark selection button
    self.exportLandmarkSelectionButton = qt.QPushButton("Export Selected Region as Model")
    self.exportLandmarkSelectionButton.setToolTip("Export the selected region as a separate model")
    self.exportLandmarkSelectionButton.enabled = False
    self.exportLandmarkSelectionButton.setStyleSheet(ColorTheme.getButtonStyle('secondary'))
    self.landmarkLayout.addRow(self.exportLandmarkSelectionButton)
    
    regionLayout.addRow(self.landmarkFrame)
    
    # --- Common Controls ---
    # Clear selection button
    self.clearSelectionButton = qt.QPushButton("Clear Selection")
    self.clearSelectionButton.setToolTip("Clear the current region selection")
    self.clearSelectionButton.enabled = False
    self.clearSelectionButton.setStyleSheet(ColorTheme.getButtonStyle('neutral'))
    regionLayout.addRow(self.clearSelectionButton)
    
    # Selection info label
    self.selectionInfoLabel = qt.QLabel("No region selected")
    self.selectionInfoLabel.setStyleSheet(ColorTheme.getLabelStyle())
    regionLayout.addRow("Selection Info:", self.selectionInfoLabel)

    # Add spacing before the visualization button
    visualizeWidgetLayout.addRow(" ", qt.QLabel())

    #
    # Start Visualization Button (at bottom)
    #
    self.startVisualizationButton = qt.QPushButton("Start Visualization")
    self.startVisualizationButton.toolTip = "Prepare the 3D scene for visualization and show markups"
    self.startVisualizationButton.setStyleSheet("""
      QPushButton {
        background-color: #87CEEB;
        color: #2C3E50;
        font-weight: bold;
        border: none;
        border-radius: 5px;
        padding: 8px 16px;
        min-height: 30px;
      }
      QPushButton:hover {
        background-color: #6BB6E8;
      }
      QPushButton:pressed {
        background-color: #4FA8D8;
      }
      """)
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
    self.startVisualizationButton.connect('clicked(bool)', self.onStartVisualizationButton)
    
    # Region selection connections
    self.regionMeshSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onRegionSelectionInputChanged)
    self.selectionMethodCombo.connect("currentIndexChanged(int)", self.onSelectionMethodChanged)
    self.existingSegmentationSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onExistingSegmentationChanged)
    self.loadSegmentationButton.connect('clicked(bool)', self.onLoadSegmentation)
    self.setupSegmentEditorButton.connect('clicked(bool)', self.onSetupSegmentEditor)
    self.exportSelectionButton.connect('clicked(bool)', self.onExportSelection)
    self.selectionMarkupSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onRegionSelectionInputChanged)
    self.applyLandmarkSelectionButton.connect('clicked(bool)', self.onApplyLandmarkSelection)
    self.exportLandmarkSelectionButton.connect('clicked(bool)', self.onExportLandmarkSelection)
    self.clickSelectLandmarkButton.connect('clicked(bool)', self.onClickSelectLandmark)
    self.clearSelectionButton.connect('clicked(bool)', self.onClearSelection)
    

    # Auto-detect Blender executable on startup
    self.autoDetectBlender()

    ################################### Phase 1: Data Sampling ###################################
    # Data Sampling section
    dataSamplingWidget = ctk.ctkCollapsibleButton()
    dataSamplingWidget.setSizePolicy(qt.QSizePolicy.Preferred, qt.QSizePolicy.Maximum)
    dataSamplingWidget.setStyleSheet("ctkCollapsibleButton { font-weight: bold; background-color: #f0f8ff; }")
    dataSamplingWidgetLayout = qt.QFormLayout(dataSamplingWidget)
    dataSamplingWidgetLayout.setVerticalSpacing(6)
    dataSamplingWidgetLayout.setHorizontalSpacing(8)
    dataSamplingWidgetLayout.setFormAlignment(qt.Qt.AlignTop)
    dataSamplingWidgetLayout.setLabelAlignment(qt.Qt.AlignLeft | qt.Qt.AlignVCenter)
    dataSamplingWidget.text = "Phase 1: Data Sampling"
    colorsEDATabLayout.addRow(dataSamplingWidget)

    # Atlas model selector for Colors EDA
    self.colorsAtlasModelSelect = slicer.qMRMLNodeComboBox()
    self.colorsAtlasModelSelect.nodeTypes = (("vtkMRMLModelNode"), "")
    self.colorsAtlasModelSelect.setToolTip("Select the atlas model for color analysis")
    self.colorsAtlasModelSelect.selectNodeUponCreation = False
    self.colorsAtlasModelSelect.noneEnabled = True
    self.colorsAtlasModelSelect.addEnabled = False
    self.colorsAtlasModelSelect.removeEnabled = False
    self.colorsAtlasModelSelect.showHidden = False
    self.colorsAtlasModelSelect.setMRMLScene(slicer.mrmlScene)
    dataSamplingWidgetLayout.addRow("Atlas Model: ", self.colorsAtlasModelSelect)

    # Baked textures directory selector
    self.bakedTexturesDirectorySelector = ctk.ctkPathLineEdit()
    self.bakedTexturesDirectorySelector.filters = ctk.ctkPathLineEdit.Dirs
    self.bakedTexturesDirectorySelector.setToolTip("Select directory containing baked atlas-space textures")
    dataSamplingWidgetLayout.addRow("Textures Directory: ", self.bakedTexturesDirectorySelector)

    # Random seed input
    self.randomSeedSpin = qt.QSpinBox()
    self.randomSeedSpin.setRange(0, 999999)
    self.randomSeedSpin.setValue(42)  # Default seed
    self.randomSeedSpin.setToolTip("Random seed for reproducible face sampling")
    dataSamplingWidgetLayout.addRow("Random Seed: ", self.randomSeedSpin)

    # Percentage of faces to sample
    self.faceSamplePercentSpin = qt.QDoubleSpinBox()
    self.faceSamplePercentSpin.setRange(1.0, 100.0)
    self.faceSamplePercentSpin.setSingleStep(5.0)
    self.faceSamplePercentSpin.setSuffix(" %")
    self.faceSamplePercentSpin.setValue(100.0)  # Default to all faces
    self.faceSamplePercentSpin.setToolTip("Percentage of faces to randomly sample from the atlas model")
    dataSamplingWidgetLayout.addRow("Sample Percentage: ", self.faceSamplePercentSpin)

    # Sample Data button
    self.sampleDataButton = qt.QPushButton("Sample Data")
    self.sampleDataButton.toolTip = "Sample faces and calculate color averages from textures"
    self.sampleDataButton.enabled = False
    dataSamplingWidgetLayout.addRow(self.sampleDataButton)

    # Sampling progress and status
    self.samplingProgressBar = qt.QProgressBar()
    self.samplingProgressBar.setVisible(False)
    dataSamplingWidgetLayout.addRow("Sampling Progress: ", self.samplingProgressBar)

    self.samplingStatusLabel = qt.QLabel("No data sampled")
    self.samplingStatusLabel.setStyleSheet("color: #666; font-style: italic;")
    dataSamplingWidgetLayout.addRow("Status: ", self.samplingStatusLabel)

    # Add visual separator between phases
    separatorLine = qt.QFrame()
    separatorLine.setFrameShape(qt.QFrame.HLine)
    separatorLine.setFrameShadow(qt.QFrame.Sunken)
    separatorLine.setStyleSheet("QFrame { color: #cccccc; margin: 10px 0px; }")
    colorsEDATabLayout.addRow(separatorLine)

    ################################### Phase 2: Analysis & Plotting ###################################
    # Analysis section
    analysisWidget = ctk.ctkCollapsibleButton()
    analysisWidget.setSizePolicy(qt.QSizePolicy.Preferred, qt.QSizePolicy.Maximum)
    analysisWidget.setStyleSheet("ctkCollapsibleButton { font-weight: bold; background-color: #f8fff0; }")
    analysisWidgetLayout = qt.QFormLayout(analysisWidget)
    analysisWidgetLayout.setVerticalSpacing(6)
    analysisWidgetLayout.setHorizontalSpacing(8)
    analysisWidgetLayout.setFormAlignment(qt.Qt.AlignTop)
    analysisWidgetLayout.setLabelAlignment(qt.Qt.AlignLeft | qt.Qt.AlignVCenter)
    analysisWidget.text = "Phase 2: Analysis & Plotting"
    analysisWidget.enabled = False  # Disabled until data is sampled
    colorsEDATabLayout.addRow(analysisWidget)

    # Color space selection - compact layout
    self.colorSpaceWidget = qt.QWidget()
    self.colorSpaceWidget.setSizePolicy(qt.QSizePolicy.Preferred, qt.QSizePolicy.Fixed)
    self.colorSpaceWidget.setFixedHeight(25)  # Fixed height for consistency
    self.colorSpaceLayout = qt.QHBoxLayout(self.colorSpaceWidget)
    self.colorSpaceLayout.setContentsMargins(0, 2, 0, 2)  # Small vertical margins
    self.colorSpaceLayout.setSpacing(15)  # Slightly more space between buttons

    self.rgbRadio = qt.QRadioButton("RGB")
    self.hsvRadio = qt.QRadioButton("HSV")
    self.hsvRadio.setChecked(True)

    self.colorSpaceButtonGroup = qt.QButtonGroup()
    self.colorSpaceButtonGroup.addButton(self.rgbRadio)
    self.colorSpaceButtonGroup.addButton(self.hsvRadio)
    self.colorSpaceLayout.addWidget(self.rgbRadio)
    self.colorSpaceLayout.addWidget(self.hsvRadio)
    self.colorSpaceLayout.addStretch()  # Push buttons to the left

    analysisWidgetLayout.addRow("Color Space:", self.colorSpaceWidget)

    # Dimensionality reduction algorithm selection - compact layout
    self.dimRedWidget = qt.QWidget()
    self.dimRedWidget.setSizePolicy(qt.QSizePolicy.Preferred, qt.QSizePolicy.Fixed)
    self.dimRedWidget.setFixedHeight(25)  # Fixed height for consistency
    self.dimRedLayout = qt.QHBoxLayout(self.dimRedWidget)
    self.dimRedLayout.setContentsMargins(0, 2, 0, 2)  # Small vertical margins
    self.dimRedLayout.setSpacing(15)  # Slightly more space between buttons

    self.pcaRadio = qt.QRadioButton("PCA")
    self.pcaRadio.setChecked(True)
    self.icaRadio = qt.QRadioButton("ICA")
    self.umapRadio = qt.QRadioButton("UMAP")

    self.dimRedButtonGroup = qt.QButtonGroup()
    self.dimRedButtonGroup.addButton(self.pcaRadio)
    self.dimRedButtonGroup.addButton(self.icaRadio)
    self.dimRedButtonGroup.addButton(self.umapRadio)
    self.dimRedLayout.addWidget(self.pcaRadio)
    self.dimRedLayout.addWidget(self.icaRadio)
    self.dimRedLayout.addWidget(self.umapRadio)
    self.dimRedLayout.addStretch()  # Push buttons to the left

    analysisWidgetLayout.addRow("Dimensionality Reduction:", self.dimRedWidget)

    # Enable/disable based on availability
    if not SKLEARN_AVAILABLE:
        self.pcaRadio.setEnabled(False)
        self.icaRadio.setEnabled(False)
        self.pcaRadio.setToolTip("sklearn not available")
        self.icaRadio.setToolTip("sklearn not available")
    if not UMAP_AVAILABLE:
        self.umapRadio.setEnabled(False)
        self.umapRadio.setToolTip("umap-learn not available")

    # View mode selection - 2D vs Channel views
    self.viewModeWidget = qt.QWidget()
    self.viewModeWidget.setSizePolicy(qt.QSizePolicy.Preferred, qt.QSizePolicy.Fixed)
    self.viewModeWidget.setFixedHeight(25)
    self.viewModeLayout = qt.QHBoxLayout(self.viewModeWidget)
    self.viewModeLayout.setContentsMargins(0, 2, 0, 2)
    self.viewModeLayout.setSpacing(15)

    self.view2DRadio = qt.QRadioButton("2D (Dim Reduction)")
    self.view2DRadio.setChecked(True)
    self.viewChannelRadio = qt.QRadioButton("Channel Histograms")

    self.viewModeButtonGroup = qt.QButtonGroup()
    self.viewModeButtonGroup.addButton(self.view2DRadio)
    self.viewModeButtonGroup.addButton(self.viewChannelRadio)
    self.viewModeLayout.addWidget(self.view2DRadio)
    self.viewModeLayout.addWidget(self.viewChannelRadio)
    self.viewModeLayout.addStretch()

    analysisWidgetLayout.addRow("View Mode:", self.viewModeWidget)

    # Plot button (renamed from "Run Analysis")
    self.plotButton = qt.QPushButton("Plot")
    self.plotButton.toolTip = "Generate plots from sampled data using selected analysis settings"
    self.plotButton.enabled = False
    analysisWidgetLayout.addRow(self.plotButton)

    # Analysis progress and log information
    self.analysisProgressBar = qt.QProgressBar()
    self.analysisProgressBar.setVisible(False)
    analysisWidgetLayout.addRow("Analysis Progress: ", self.analysisProgressBar)

    self.colorsEDALogInfo = qt.QPlainTextEdit()
    self.colorsEDALogInfo.setPlaceholderText("Colors EDA log information")
    self.colorsEDALogInfo.setReadOnly(True)
    self.colorsEDALogInfo.setMaximumHeight(150)
    analysisWidgetLayout.addRow(self.colorsEDALogInfo)

    # Channel Histogram controls (rendered in the main plot viewer)
    self.histCollapsible = ctk.ctkCollapsibleButton()
    self.histCollapsible.text = "Channel Histogram Options"
    self.histCollapsible.collapsed = False
    self.histCollapsible.setSizePolicy(qt.QSizePolicy.Preferred, qt.QSizePolicy.Maximum)
    self.histLayout = qt.QFormLayout(self.histCollapsible)
    self.histLayout.setContentsMargins(0, 0, 0, 0)
    self.histLayout.setSpacing(4)

    self.histChannelSelector = qt.QComboBox()
    self.histChannelSelector.addItems(["Channel 1", "Channel 2", "Channel 3"])  # Labels updated after run
    self.histChannelSelector.setEnabled(False)
    self.histLayout.addRow("Channel:", self.histChannelSelector)

    # Color-by-bin-average toggle
    self.colorByBinAvgCheck = qt.QCheckBox("Color bars by bin average")
    self.colorByBinAvgCheck.setChecked(False)
    self.histLayout.addRow(self.colorByBinAvgCheck)

    # HSV Options section
    self.hsvOptionsLabel = qt.QLabel("HSV Options:")
    self.hsvOptionsLabel.setStyleSheet("font-weight: bold; color: #666;")
    self.histLayout.addRow(self.hsvOptionsLabel)

    # Color enhancement option for 2D plots
    self.enhanceColorsCheck = qt.QCheckBox("Enhance colors in 2D plot for visibility")
    self.enhanceColorsCheck.setChecked(False)  # Default to actual colors
    self.enhanceColorsCheck.setToolTip("When checked, boosts saturation/value for better visibility. When unchecked, uses actual colors from data.")
    self.histLayout.addRow(self.enhanceColorsCheck)

    # HSV Filtering section
    self.hsvFilterLabel = qt.QLabel("HSV Filtering (affects dim reduction & hue histogram):")
    self.hsvFilterLabel.setStyleSheet("font-weight: bold; color: #666; margin-top: 10px;")
    self.histLayout.addRow(self.hsvFilterLabel)

    # Saturation cutoff (percent)
    self.satCutoffSpin = qt.QDoubleSpinBox()
    self.satCutoffSpin.setRange(0.0, 100.0)
    self.satCutoffSpin.setSingleStep(5.0)
    self.satCutoffSpin.setSuffix(" %")
    self.satCutoffSpin.setValue(10.0)   # default from our earlier fix
    self.satCutoffSpin.setToolTip("Minimum saturation threshold for HSV filtering")
    self.histLayout.addRow("Saturation cutoff:", self.satCutoffSpin)

    self.valueCutoffSpin = qt.QDoubleSpinBox()
    self.valueCutoffSpin.setRange(0.0, 100.0)
    self.valueCutoffSpin.setSingleStep(5.0)
    self.valueCutoffSpin.setSuffix(" %")
    self.valueCutoffSpin.setValue(10.0)   # default from our earlier fix
    self.valueCutoffSpin.setToolTip("Minimum value/brightness threshold for HSV filtering")
    self.histLayout.addRow("Value cutoff:", self.valueCutoffSpin)

    # Note: Removed auto-refresh event hooks - now manual refresh via Plot button

    # Keep references to created MRML nodes (tables/series/charts) for cleanup
    self.histTableNodes = []
    self.histSeriesNodes = []
    self.histChartNodes = []

    analysisWidgetLayout.addRow(self.histCollapsible)

    # Store reference to analysis widget for enabling/disabling
    self.analysisWidget = analysisWidget

    # Connections for histogram controls
    self.histChannelSelector.connect("currentIndexChanged(int)", self.onHistChannelChanged)

    # Connections for Phase 1: Data Sampling
    self.colorsAtlasModelSelect.connect("currentNodeChanged(vtkMRMLNode*)", self.onSamplingParameterChanged)
    self.bakedTexturesDirectorySelector.connect("currentPathChanged(QString)", self.onSamplingParameterChanged)
    self.sampleDataButton.connect('clicked(bool)', self.onSampleDataButton)

    # Connections for Phase 2: Analysis & Plotting
    self.plotButton.connect('clicked(bool)', self.onPlotButton)
    self.viewModeButtonGroup.connect('buttonClicked(QAbstractButton*)', self.onViewModeChanged)

    # Initialize sampled data storage
    self.sampledColorData = None
    self.sampledSpecimenNames = None
    self.sampledFaceIndices = None

    # Add vertical spacer so extra space goes below content
    colorsEDATabLayout.addItem(qt.QSpacerItem(0, 0, qt.QSizePolicy.Minimum, qt.QSizePolicy.Expanding))

    ################################### Recolor Tab ###################################
    # Layout within the Recolor tab
    recolorWidget = ctk.ctkCollapsibleButton()
    recolorWidget.setSizePolicy(qt.QSizePolicy.Preferred, qt.QSizePolicy.Maximum)
    recolorWidgetLayout = qt.QFormLayout(recolorWidget)
    recolorWidgetLayout.setVerticalSpacing(4)
    recolorWidgetLayout.setHorizontalSpacing(8)
    recolorWidgetLayout.setFormAlignment(qt.Qt.AlignTop)
    recolorWidgetLayout.setLabelAlignment(qt.Qt.AlignLeft | qt.Qt.AlignVCenter)
    recolorWidget.text = "Recolor Settings"
    recolorTabLayout.addRow(recolorWidget)

    # Atlas model selector for Recolor
    self.recolorAtlasModelSelect = slicer.qMRMLNodeComboBox()
    self.recolorAtlasModelSelect.nodeTypes = (("vtkMRMLModelNode"), "")
    self.recolorAtlasModelSelect.setToolTip("Select the atlas model for recoloring")
    self.recolorAtlasModelSelect.selectNodeUponCreation = False
    self.recolorAtlasModelSelect.noneEnabled = True
    self.recolorAtlasModelSelect.addEnabled = False
    self.recolorAtlasModelSelect.removeEnabled = False
    self.recolorAtlasModelSelect.showHidden = False
    self.recolorAtlasModelSelect.setMRMLScene(slicer.mrmlScene)
    recolorWidgetLayout.addRow("Atlas Model: ", self.recolorAtlasModelSelect)

    # Textures directory selector
    self.recolorTexturesDirectorySelector = ctk.ctkPathLineEdit()
    self.recolorTexturesDirectorySelector.filters = ctk.ctkPathLineEdit.Dirs
    self.recolorTexturesDirectorySelector.setToolTip("Select directory containing texture images")
    recolorWidgetLayout.addRow("Textures Directory: ", self.recolorTexturesDirectorySelector)

    # Texture selection dropdown
    self.recolorTextureSelector = qt.QComboBox()
    self.recolorTextureSelector.setToolTip("Select a texture to apply to the model")
    self.recolorTextureSelector.enabled = False
    recolorWidgetLayout.addRow("Select Texture: ", self.recolorTextureSelector)

    # Average face color checkbox
    self.averageFaceColorCheckbox = qt.QCheckBox()
    self.averageFaceColorCheckbox.setChecked(False)
    self.averageFaceColorCheckbox.setToolTip("If checked, each face will be colored with the average color from the texture instead of using the texture directly")
    recolorWidgetLayout.addRow("Average Face Colors: ", self.averageFaceColorCheckbox)

    # Quantize colors checkbox
    self.quantizeColorsCheckbox = qt.QCheckBox()
    self.quantizeColorsCheckbox.setChecked(False)
    self.quantizeColorsCheckbox.setEnabled(False)  # Initially disabled
    self.quantizeColorsCheckbox.setToolTip("If checked, quantize the average face colors using k-means clustering in CIE Lab color space")
    recolorWidgetLayout.addRow("Quantize Colors: ", self.quantizeColorsCheckbox)

    # Number of color clusters
    self.numColorClustersSpin = qt.QSpinBox()
    self.numColorClustersSpin.setRange(2, 64)
    self.numColorClustersSpin.setValue(8)  # Default value
    self.numColorClustersSpin.setEnabled(False)  # Initially disabled
    self.numColorClustersSpin.setToolTip("Number of color clusters for quantization (2-64)")
    recolorWidgetLayout.addRow("Number of Colors: ", self.numColorClustersSpin)

    # High contrast palette checkbox
    self.useHighContrastPaletteCheckbox = qt.QCheckBox()
    self.useHighContrastPaletteCheckbox.setChecked(False)
    self.useHighContrastPaletteCheckbox.setEnabled(False)  # Initially disabled
    self.useHighContrastPaletteCheckbox.setToolTip("If checked, use a high contrast color palette instead of the quantized colors from the texture")
    recolorWidgetLayout.addRow("High Contrast Palette: ", self.useHighContrastPaletteCheckbox)

    # Apply texture button
    self.applyRecolorButton = qt.QPushButton("Apply Texture")
    self.applyRecolorButton.toolTip = "Apply the selected texture to the atlas model"
    self.applyRecolorButton.enabled = False
    recolorWidgetLayout.addRow(self.applyRecolorButton)

    # Progress and log information for Recolor
    self.recolorProgressBar = qt.QProgressBar()
    self.recolorProgressBar.setVisible(False)
    recolorWidgetLayout.addRow("Progress: ", self.recolorProgressBar)

    self.recolorLogInfo = qt.QPlainTextEdit()
    self.recolorLogInfo.setPlaceholderText("Recolor log information")
    self.recolorLogInfo.setReadOnly(True)
    self.recolorLogInfo.setMaximumHeight(150)
    recolorWidgetLayout.addRow(self.recolorLogInfo)

    # Connections for Recolor
    self.recolorAtlasModelSelect.connect("currentNodeChanged(vtkMRMLNode*)", self.onRecolorParameterChanged)
    self.recolorTexturesDirectorySelector.connect("currentPathChanged(QString)", self.onRecolorTexturesDirectoryChanged)
    self.recolorTextureSelector.connect("currentIndexChanged(int)", self.onRecolorParameterChanged)
    self.averageFaceColorCheckbox.connect("toggled(bool)", self.onAverageFaceColorToggled)
    self.quantizeColorsCheckbox.connect("toggled(bool)", self.onQuantizeColorsToggled)
    self.numColorClustersSpin.connect("valueChanged(int)", self.onRecolorParameterChanged)
    self.useHighContrastPaletteCheckbox.connect("toggled(bool)", self.onRecolorParameterChanged)
    self.applyRecolorButton.connect('clicked(bool)', self.onApplyRecolorButton)

    # Add vertical spacer so extra space goes below content
    recolorTabLayout.addItem(qt.QSpacerItem(0, 0, qt.QSizePolicy.Minimum, qt.QSizePolicy.Expanding))

    ################################### MultiRecolor Tab ###################################

    # Step 1: Multi-texture clustering section
    clusteringWidget = ctk.ctkCollapsibleButton()
    clusteringWidget.setSizePolicy(qt.QSizePolicy.Preferred, qt.QSizePolicy.Maximum)
    clusteringWidgetLayout = qt.QFormLayout(clusteringWidget)
    clusteringWidgetLayout.setVerticalSpacing(4)
    clusteringWidgetLayout.setHorizontalSpacing(8)
    clusteringWidgetLayout.setFormAlignment(qt.Qt.AlignTop)
    clusteringWidgetLayout.setLabelAlignment(qt.Qt.AlignLeft | qt.Qt.AlignVCenter)
    clusteringWidget.text = "Step 1: Multi-Texture Clustering"
    multiRecolorTabLayout.addRow(clusteringWidget)

    # Atlas model selector for MultiRecolor
    self.multiRecolorAtlasModelSelect = slicer.qMRMLNodeComboBox()
    self.multiRecolorAtlasModelSelect.nodeTypes = (("vtkMRMLModelNode"), "")
    self.multiRecolorAtlasModelSelect.setToolTip("Select the atlas model for multi-texture analysis")
    self.multiRecolorAtlasModelSelect.setMRMLScene(slicer.mrmlScene)
    clusteringWidgetLayout.addRow("Atlas Model: ", self.multiRecolorAtlasModelSelect)

    # Texture directory selector for MultiRecolor
    self.multiRecolorTextureDirectorySelector = ctk.ctkPathLineEdit()
    self.multiRecolorTextureDirectorySelector.filters = ctk.ctkPathLineEdit.Dirs
    self.multiRecolorTextureDirectorySelector.setToolTip("Select directory containing texture images")
    clusteringWidgetLayout.addRow("Texture Directory: ", self.multiRecolorTextureDirectorySelector)

    # Number of clusters for multi-texture analysis
    self.multiRecolorNumClustersSpin = qt.QSpinBox()
    self.multiRecolorNumClustersSpin.setRange(2, 64)
    self.multiRecolorNumClustersSpin.setValue(16)  # Default value for multi-texture analysis
    self.multiRecolorNumClustersSpin.setToolTip("Number of color clusters for multi-texture analysis (2-64)")
    clusteringWidgetLayout.addRow("Number of Clusters: ", self.multiRecolorNumClustersSpin)

    # Cluster button
    self.clusterButton = qt.QPushButton("Cluster")
    self.clusterButton.setToolTip("Process all textures and create color clusters")
    self.clusterButton.enabled = False
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
    individualWidgetLayout = qt.QFormLayout(individualWidget)
    individualWidgetLayout.setVerticalSpacing(4)
    individualWidgetLayout.setHorizontalSpacing(8)
    individualWidgetLayout.setFormAlignment(qt.Qt.AlignTop)
    individualWidgetLayout.setLabelAlignment(qt.Qt.AlignLeft | qt.Qt.AlignVCenter)
    individualWidget.text = "Step 2: Individual Visualization"
    multiRecolorTabLayout.addRow(individualWidget)

    # Texture selector dropdown
    self.individualTextureSelector = qt.QComboBox()
    self.individualTextureSelector.setToolTip("Select a texture to visualize with the clustered palette")
    self.individualTextureSelector.enabled = False
    individualWidgetLayout.addRow("Select Texture: ", self.individualTextureSelector)

    # High contrast palette option for individual visualization
    self.individualHighContrastCheckbox = qt.QCheckBox()
    self.individualHighContrastCheckbox.setChecked(False)
    self.individualHighContrastCheckbox.setEnabled(False)
    self.individualHighContrastCheckbox.setToolTip("Use high contrast palette instead of clustered colors")
    individualWidgetLayout.addRow("High Contrast Palette: ", self.individualHighContrastCheckbox)

    # Apply texture button for individual visualization
    self.applyIndividualTextureButton = qt.QPushButton("Apply Texture")
    self.applyIndividualTextureButton.setToolTip("Apply selected texture with clustered palette")
    self.applyIndividualTextureButton.enabled = False
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
    self.dimReductionMethodGroup.addButton(self.pcaRadioButton, 0)
    self.dimReductionMethodGroup.addButton(self.umapRadioButton, 1)

    dimReductionLayout = qt.QHBoxLayout()
    dimReductionLayout.addWidget(self.pcaRadioButton)
    dimReductionLayout.addWidget(self.umapRadioButton)
    dimReductionWidget = qt.QWidget()
    dimReductionWidget.setLayout(dimReductionLayout)
    populationWidgetLayout.addRow("Dimensionality Reduction: ", dimReductionWidget)

    # Compare textures button
    self.compareTexturesButton = qt.QPushButton("Compare Textures")
    self.compareTexturesButton.setToolTip("Analyze all textures and create population comparison plot")
    self.compareTexturesButton.enabled = False
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
    self.multiRecolorNumClustersSpin.connect("valueChanged(int)", self.onMultiRecolorParameterChanged)
    self.clusterButton.connect('clicked(bool)', self.onClusterButton)
    self.individualTextureSelector.connect("currentTextChanged(const QString &)", self.onIndividualTextureChanged)
    self.applyIndividualTextureButton.connect('clicked(bool)', self.onApplyIndividualTextureButton)
    self.compareTexturesButton.connect('clicked(bool)', self.onCompareTexturesButton)

    # Add vertical spacer so extra space goes below content
    multiRecolorTabLayout.addItem(qt.QSpacerItem(0, 0, qt.QSizePolicy.Minimum, qt.QSizePolicy.Expanding))

    # Initialize MultiRecolor state variables
    self.multiRecolorClusterCenters = None
    self.multiRecolorFaceAreas = None
    self.multiRecolorTextureFiles = []
    self.faceAreasCache = {}  # Cache face areas by model node ID

  def onExistingSegmentationChanged(self):
    """Handle selection of existing segmentation"""
    segmentationNode = self.existingSegmentationSelector.currentNode()
    self.loadSegmentationButton.enabled = (segmentationNode is not None)
    
  def onLoadSegmentation(self):
    """Load and configure existing segmentation for editing"""
    segmentationNode = self.existingSegmentationSelector.currentNode()
    if not segmentationNode:
      slicer.util.errorDisplay("Please select a segmentation to load.")
      return
    
    try:
      # Get the reference volume from the segmentation if it exists
      referenceVolumeNode = segmentationNode.GetNodeReference("referenceImageGeometryRef")
      
      # If no reference volume, try to find one with matching name
      if not referenceVolumeNode:
        # Look for reference volume with similar name
        segmentationName = segmentationNode.GetName()
        possibleRefVolumeName = f"{segmentationName.replace('_Segmentation', '')}_ReferenceVolume"
        referenceVolumeNode = slicer.util.getFirstNodeByName(possibleRefVolumeName)
        
        # If still no reference volume, create a minimal one from segmentation bounds
        if not referenceVolumeNode:
          bounds = [0, 0, 0, 0, 0, 0]
          segmentationNode.GetBounds(bounds)
          
          referenceVolumeNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
          referenceVolumeNode.SetName(f"{segmentationName}_ReferenceVolume")
          
          # Set up minimal volume geometry
          spacing = [0.5, 0.5, 0.5]
          imageSize = [
            max(20, int((bounds[1] - bounds[0]) / spacing[0]) + 1),
            max(20, int((bounds[3] - bounds[2]) / spacing[1]) + 1), 
            max(20, int((bounds[5] - bounds[4]) / spacing[2]) + 1)
          ]
          
          imageData = vtk.vtkImageData()
          imageData.SetDimensions(imageSize)
          imageData.SetSpacing(spacing)
          imageData.SetOrigin(bounds[0], bounds[2], bounds[4])
          imageData.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)
          imageData.GetPointData().GetScalars().Fill(100)
          
          referenceVolumeNode.SetAndObserveImageData(imageData)
          
          # Set reference geometry for the segmentation
          segmentationNode.SetReferenceImageGeometryParameterFromVolumeNode(referenceVolumeNode)
      
      # Configure segmentation display for 3D painting
      segmentationDisplayNode = segmentationNode.GetDisplayNode()
      if segmentationDisplayNode:
        # Enable 3D display with proper opacity
        segmentationDisplayNode.SetVisibility3D(True)
        segmentationDisplayNode.SetOpacity3D(0.8)  # More opaque for better visibility
        
        # Enable surface representation for 3D painting
        try:
          # Use the segmentation logic to get the correct representation name
          segmentationLogic = slicer.modules.segmentations.logic()
          closedSurfaceReprName = segmentationLogic.GetSegmentationClosedSurfaceRepresentationName()
          segmentationDisplayNode.SetPreferredDisplayRepresentationName3D(closedSurfaceReprName)
        except:
          # Fallback approach - try common representation names
          try:
            segmentationDisplayNode.SetPreferredDisplayRepresentationName3D("Closed surface")
          except:
            # Final fallback
            segmentationDisplayNode.SetPreferredDisplayRepresentationName3D("Binary labelmap")
        
        # Enable slice fill for 2D views
        segmentationDisplayNode.SetVisibility2DFill(True)
        segmentationDisplayNode.SetVisibility2DOutline(True)
        
        # Ensure segments are visible by default
        segmentationDisplayNode.SetAllSegmentsVisibility3D(True)
        segmentationDisplayNode.SetAllSegmentsVisibility2DFill(True)
        segmentationDisplayNode.SetAllSegmentsVisibility2DOutline(True)
      
      # Switch to Segment Editor module
      slicer.util.selectModule("SegmentEditor")
      
      # Set up segment editor widget
      segmentEditorWidget = slicer.modules.segmenteditor.widgetRepresentation().self().editor
      segmentEditorWidget.setSegmentationNode(segmentationNode)
      segmentEditorWidget.setSourceVolumeNode(referenceVolumeNode)
      
      # Select the first editable segment or create one if none exists
      segmentation = segmentationNode.GetSegmentation()
      segmentIDs = vtk.vtkStringArray()
      segmentation.GetSegmentIDs(segmentIDs)
      
      if segmentIDs.GetNumberOfValues() == 0:
        # No segments exist, create one
        segmentation.AddEmptySegment("LoadedRegion")
        segmentEditorWidget.setCurrentSegmentID("LoadedRegion")
      else:
        # Use the first segment
        firstSegmentID = segmentIDs.GetValue(0)
        segmentEditorWidget.setCurrentSegmentID(firstSegmentID)
      
      # Hide the reference volume completely to avoid conflicts
      if referenceVolumeNode.GetDisplayNode():
        referenceVolumeNode.GetDisplayNode().SetVisibility(False)
      
      # Configure 3D view for painting
      layoutManager = slicer.app.layoutManager()
      if layoutManager:
        threeDWidget = layoutManager.threeDWidget(0)
        if threeDWidget:
          threeDView = threeDWidget.threeDView()
          threeDViewNode = threeDView.mrmlViewNode()
          if threeDViewNode:
            # Set 3D view to perspective mode for better painting
            try:
              threeDViewNode.SetRenderMode(threeDViewNode.Perspective)
            except:
              # Fallback - just ensure the view is properly configured
              pass
      
      # Enable export button
      self.exportSelectionButton.enabled = True
      self.currentSegmentationNode = segmentationNode
      self.currentReferenceVolumeNode = referenceVolumeNode
      
      slicer.util.infoDisplay(f"Loaded segmentation '{segmentationNode.GetName()}' successfully!\n\n"
                            "You can now continue editing the loaded segmentation.\n"
                            "Use Paint, Erase, or other tools to modify your selection.\n\n"
                            "When done, click 'Export Selected Region' to create a new model.")
      
    except Exception as e:
      slicer.util.errorDisplay(f"Error loading segmentation: {str(e)}")
      print(f"Load segmentation error: {e}")

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
      label.setStyleSheet("QLabel { color: palette(disabled-text); font-style: italic; }")
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
        label.setStyleSheet("QLabel { color: palette(negative); }")
        return False, 0
      else:
        label.setText(f"{count} found")
        label.setStyleSheet("QLabel { color: palette(positive); }")
        return True, count
    except Exception as e:
      label.setText(f"Error reading directory")
      label.setStyleSheet("QLabel { color: palette(negative); }")
      return False, 0
  
  def onMeshDirectoryChangedDC(self, directory):
    """Validate mesh directory when changed"""
    model_extensions = ['.ply', '.stl', '.obj', '.vtk', '.vtp']
    self.validateDirectory(directory, model_extensions, self.meshValidationLabelDC, "models")
    # Update texture matching if textures are already loaded
    if self.textureDirectoryDC.currentPath:
      self.validateTextureMatching()
    self.onParameterSelectDC()
  
  def onLandmarkDirectoryChangedDC(self, directory):
    """Validate landmark directory when changed"""
    landmark_extensions = ['.fcsv', '.json', '.mrk.json']
    self.validateDirectory(directory, landmark_extensions, self.landmarkValidationLabelDC, "landmarks")
    # Update texture matching if textures are already loaded
    if self.textureDirectoryDC.currentPath:
      self.validateTextureMatching()
    self.onParameterSelectDC()
  
  def onTextureDirectoryChangedDC(self, directory):
    """Validate texture directory when changed"""
    texture_extensions = ['.png']
    valid, count = self.validateDirectory(directory, texture_extensions, self.textureValidationLabelDC, "textures")
    
    # If we have textures and models/landmarks, check for matches
    if valid and count > 0:
      self.validateTextureMatching()
    
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
      if f.lower().endswith('.png') and not f.startswith('.'):
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
        self.textureValidationLabelDC.setText(f"{len(texture_files)} found ({match_count}/{total_subjects} matched)")
        if match_count == total_subjects:
          self.textureValidationLabelDC.setStyleSheet("QLabel { color: palette(positive); }")
        else:
          self.textureValidationLabelDC.setStyleSheet("QLabel { color: #ff8c00; }")  # Orange color that works in both themes
      else:
        self.textureValidationLabelDC.setText(f"{len(texture_files)} found (no matches)")
        self.textureValidationLabelDC.setStyleSheet("QLabel { color: palette(negative); }")
  
  def updateProgressDC(self, value, text="", showCancel=False):
    """Update progress bar and label"""
    if not self.progressWidgetDC.isVisible():
      self.progressWidgetDC.setVisible(True)
    
    self.progressBarDC.setValue(value)
    if text:
      self.progressLabelDC.setText(text)
    
    self.cancelButtonDC.setVisible(showCancel)
    
    # Process events to update UI
    slicer.app.processEvents()
  
  def resetProgressDC(self):
    """Reset progress indicators"""
    self.progressWidgetDC.setVisible(False)
    self.progressBarDC.setValue(0)
    self.progressLabelDC.setText("Ready")
    self.cancelButtonDC.setVisible(False)
    self.currentOperation = None
  
  def onCancelOperationDC(self):
    """Handle operation cancellation"""
    if self.currentOperation:
      self.logInfoDC.appendPlainText("Operation cancelled by user")
      self.resetProgressDC()
      self.applyButtonDC.enabled = True
      # Note: Actual cancellation logic would depend on the specific operation

  # --- Region Selection Methods ---
  
  def onRegionSelectionInputChanged(self):
    """Update button states when region selection inputs change"""
    hasModel = bool(self.regionMeshSelector.currentNode())
    
    # Update segment editor button
    self.setupSegmentEditorButton.enabled = hasModel
    
    # Update landmark selection button
    hasMarkup = bool(self.selectionMarkupSelector.currentNode())
    self.applyLandmarkSelectionButton.enabled = hasModel and hasMarkup
    self.exportLandmarkSelectionButton.enabled = hasModel and hasMarkup
    self.clickSelectLandmarkButton.enabled = hasModel and hasMarkup
    
    # Update selected points display
    if hasMarkup:
      self._updateSelectedPointsDisplay()
    else:
      self.selectedPointsLabel.setText("No points selected")
  
  def onSelectionMethodChanged(self):
    """Handle selection method change"""
    method = self.selectionMethodCombo.currentText
    
    if method == "Segment Editor (Paint/Scissors)":
      self.segmentEditorFrame.setVisible(True)
      self.landmarkFrame.setVisible(False)
    else:  # Landmark methods
      self.segmentEditorFrame.setVisible(False)
      self.landmarkFrame.setVisible(True)
      
      # Show/hide point index selector based on method
      if method == "Landmark + Radius":
        self.pointIndexSpinBox.enabled = True
        self.pointIndexSpinBox.setToolTip("Index of the point to use for selection (0-based)")
      else:  # Multiple Landmarks + Radius
        self.pointIndexSpinBox.enabled = False
        self.pointIndexSpinBox.setToolTip("All points will be used for selection")
  
  def onSetupSegmentEditor(self):
    """Setup Segment Editor for region selection"""
    modelNode = self.regionMeshSelector.currentNode()
    if not modelNode:
      slicer.util.errorDisplay("Please select a mesh first.")
      return
    
    try:
      # Create a reference volume from the model bounds to enable paint tools
      bounds = [0, 0, 0, 0, 0, 0]
      modelNode.GetBounds(bounds)
      
      # Create a small reference volume that covers the model
      referenceVolumeNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
      referenceVolumeNode.SetName(f"{modelNode.GetName()}_ReferenceVolume")
      
      # Set up the volume geometry with tighter bounds around the model
      spacing = [0.2, 0.2, 0.2]  # Even finer spacing for better surface constraint
      # Add small margin around model bounds
      margin = 2.0  # 2mm margin
      imageSize = [
        max(20, int((bounds[1] - bounds[0] + 2*margin) / spacing[0]) + 1),
        max(20, int((bounds[3] - bounds[2] + 2*margin) / spacing[1]) + 1), 
        max(20, int((bounds[5] - bounds[4] + 2*margin) / spacing[2]) + 1)
      ]
      
      # Create the image data with margin
      imageData = vtk.vtkImageData()
      imageData.SetDimensions(imageSize)
      imageData.SetSpacing(spacing)
      imageData.SetOrigin(bounds[0] - margin, bounds[2] - margin, bounds[4] - margin)
      imageData.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)
      imageData.GetPointData().GetScalars().Fill(0)  # Start with zeros for better surface constraint
      
      referenceVolumeNode.SetAndObserveImageData(imageData)
      
      # Create segmentation from model
      segmentationNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
      segmentationNode.SetName(f"{modelNode.GetName()}_Segmentation")
      
      # Set the reference geometry from our volume
      segmentationNode.SetReferenceImageGeometryParameterFromVolumeNode(referenceVolumeNode)
      
      # Import model to segmentation as the base segment
      slicer.modules.segmentations.logic().ImportModelToSegmentationNode(modelNode, segmentationNode)
      
      # Create a new empty segment for selection
      segmentationNode.GetSegmentation().AddEmptySegment("SelectedRegion")
      
      # CRITICAL: Create a surface mask to constrain painting to model surface
      try:
        # Get the model's polydata
        modelPolyData = modelNode.GetPolyData()
        if modelPolyData:
          # Create a binary mask from the model surface
          segmentationLogic = slicer.modules.segmentations.logic()
          # This creates a proper surface constraint for painting
          segmentationLogic.CreateBinaryLabelmapRepresentation(segmentationNode)
          
          # Set the segmentation to use the model as a constraint
          segmentationNode.SetReferenceImageGeometryParameterFromVolumeNode(referenceVolumeNode)
      except Exception as e:
        print(f"Warning: Could not create surface constraint: {e}")
        # Continue anyway
      
      # Ensure the segmentation has both representations for 3D painting
      segmentationLogic = slicer.modules.segmentations.logic()
      try:
        # Create closed surface representation for 3D painting
        segmentationLogic.CreateClosedSurfaceRepresentation(segmentationNode)
        # Ensure binary labelmap representation exists for volume painting
        if not segmentationNode.GetSegmentation().ContainsRepresentation("Binary labelmap"):
          segmentationLogic.CreateBinaryLabelmapRepresentation(segmentationNode)
      except Exception as e:
        print(f"Warning: Could not create segmentation representations: {e}")
        # Continue anyway
      
      # Configure segmentation display for 3D painting
      segmentationDisplayNode = segmentationNode.GetDisplayNode()
      if segmentationDisplayNode:
        # Enable 3D display with proper opacity
        segmentationDisplayNode.SetVisibility3D(True)
        segmentationDisplayNode.SetOpacity3D(0.8)  # More opaque for better visibility
        
        # Enable surface representation for 3D painting
        try:
          # Use the segmentation logic to get the correct representation name
          segmentationLogic = slicer.modules.segmentations.logic()
          closedSurfaceReprName = segmentationLogic.GetSegmentationClosedSurfaceRepresentationName()
          segmentationDisplayNode.SetPreferredDisplayRepresentationName3D(closedSurfaceReprName)
        except:
          # Fallback approach - try common representation names
          try:
            segmentationDisplayNode.SetPreferredDisplayRepresentationName3D("Closed surface")
          except:
            # Final fallback
            segmentationDisplayNode.SetPreferredDisplayRepresentationName3D("Binary labelmap")
        
        # Enable slice fill for 2D views
        segmentationDisplayNode.SetVisibility2DFill(True)
        segmentationDisplayNode.SetVisibility2DOutline(True)
        
        # Ensure segments are visible by default
        segmentationDisplayNode.SetAllSegmentsVisibility3D(True)
        segmentationDisplayNode.SetAllSegmentsVisibility2DFill(True)
        segmentationDisplayNode.SetAllSegmentsVisibility2DOutline(True)
      
      # Switch to Segment Editor module
      slicer.util.selectModule("SegmentEditor")
      
      # Set up segment editor widget
      segmentEditorWidget = slicer.modules.segmenteditor.widgetRepresentation().self().editor
      segmentEditorWidget.setSegmentationNode(segmentationNode)
      segmentEditorWidget.setSourceVolumeNode(referenceVolumeNode)  # Set the reference volume
      
      # Select the new segment for editing
      segmentEditorWidget.setCurrentSegmentID("SelectedRegion")
      
      # Configure segment editor for 3D painting
      try:
        # Enable 3D painting in segment editor
        segmentEditorWidget.setActiveEffectByName("Paint")
        paintEffect = segmentEditorWidget.activeEffect()
        if paintEffect:
          # Set sphere brush for better 3D painting
          paintEffect.setParameter("BrushType", "Sphere")
          # Enable 3D painting mode
          paintEffect.setParameter("BrushSphere", "1")
          # Set a reasonable brush size
          paintEffect.setParameter("BrushAbsoluteDiameter", "3.0")  # Smaller brush for precision
          # CRITICAL: Enable surface constraint to prevent overflow
          paintEffect.setParameter("PaintOver", "0")  # Don't paint over existing segments
          paintEffect.setParameter("Threshold", "0.5")  # Surface threshold for better constraint
      except Exception as e:
        print(f"Warning: Could not configure paint effect: {e}")
        # Continue anyway
      
      # Configure the segment color and visibility
      segmentation = segmentationNode.GetSegmentation()
      selectedSegment = segmentation.GetSegment("SelectedRegion")
      if selectedSegment and segmentationDisplayNode:
        # Set a bright, visible color for the segment
        selectedSegment.SetColor(1.0, 0.0, 0.0)  # Red color
        # Ensure the segment is visible
        try:
          segmentationDisplayNode.SetSegmentVisibility3D("SelectedRegion", True)
          segmentationDisplayNode.SetSegmentVisibility2DFill("SelectedRegion", True)
          segmentationDisplayNode.SetSegmentVisibility2DOutline("SelectedRegion", True)
        except Exception as e:
          print(f"Warning: Could not set segment visibility: {e}")
          # Continue anyway - the segment should still work
      
      # Keep reference volume visible but very transparent for 3D painting to work
      if referenceVolumeNode.GetDisplayNode():
        referenceVolumeNode.GetDisplayNode().SetVisibility(True)
        referenceVolumeNode.GetDisplayNode().SetOpacity(0.01)  # Almost invisible but still there
        # This is important: the reference volume needs to be visible for 3D painting to work properly
      
      # Configure 3D view for painting
      layoutManager = slicer.app.layoutManager()
      if layoutManager:
        threeDWidget = layoutManager.threeDWidget(0)
        if threeDWidget:
          threeDView = threeDWidget.threeDView()
          threeDViewNode = threeDView.mrmlViewNode()
          if threeDViewNode:
            # Set 3D view to perspective mode for better painting
            try:
              threeDViewNode.SetRenderMode(threeDViewNode.Perspective)
            except:
              # Fallback - just ensure the view is properly configured
              pass
      
      # Force update the segmentation display
      try:
        segmentationNode.Modified()
        if segmentationDisplayNode:
          segmentationDisplayNode.Modified()
      except Exception as e:
        print(f"Warning: Could not update segmentation display: {e}")
        # Continue anyway
      
      # Enable export button
      self.exportSelectionButton.enabled = True
      self.currentSegmentationNode = segmentationNode
      self.currentReferenceVolumeNode = referenceVolumeNode
      
     
      
    except Exception as e:
      slicer.util.errorDisplay(f"Error setting up Segment Editor: {str(e)}")
      print(f"Segment Editor setup error: {e}")
  
  def onExportSelection(self):
    """Export the painted region as a new model"""
    if not hasattr(self, 'currentSegmentationNode') or not self.currentSegmentationNode:
      slicer.util.errorDisplay("No segmentation found. Please setup Segment Editor first.")
      return
    
    try:
      # Export selected segment to model
      shNode = slicer.mrmlScene.GetSubjectHierarchyNode()
      exportFolderItemId = shNode.CreateFolderItem(shNode.GetSceneItemID(), "Selected Regions")
      
      slicer.modules.segmentations.logic().ExportAllSegmentsToModels(
        self.currentSegmentationNode, exportFolderItemId)
      
      # Get the exported model
      exportedModels = []
      childIds = vtk.vtkIdList()
      shNode.GetItemChildren(exportFolderItemId, childIds)
      
      for i in range(childIds.GetNumberOfIds()):
        childId = childIds.GetId(i)
        modelNode = shNode.GetItemDataNode(childId)
        if modelNode and modelNode.IsA("vtkMRMLModelNode"):
          exportedModels.append(modelNode)
      
      if exportedModels:
        # Enable clear button
        self.clearSelectionButton.enabled = True
        
        # Update info
        modelNode = exportedModels[0]
        numVertices = modelNode.GetPolyData().GetNumberOfPoints()
        self.selectionInfoLabel.setText(f"Exported model: {numVertices} vertices")
        
        slicer.util.infoDisplay(f"Successfully exported selected region!\n"
                              f"New model: {modelNode.GetName()}\n"
                              f"Vertices: {numVertices}")
      else:
        slicer.util.warningDisplay("No segments were exported. Make sure you painted some regions.")
        
    except Exception as e:
      slicer.util.errorDisplay(f"Error exporting selection: {str(e)}")
      print(f"Export selection error: {e}")
  
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
      method = self.selectionMethodCombo.currentText
      radius = self.selectionRadiusSlider.value
      
      # Get selected landmark points
      selectedPoints = self._getSelectedPoints(markupNode)
      
      if not selectedPoints:
        slicer.util.errorDisplay("No landmark points are selected. Please select some points first.")
        return
      
      if method == "Landmark + Radius":
        if len(selectedPoints) > 1:
          slicer.util.warningDisplay(f"Multiple points selected ({len(selectedPoints)}), but using single point mode. Using first selected point: {selectedPoints[0]}")
        pointIndex = selectedPoints[0]
        selectedVertices = self.selectMeshRegionByRadius(modelNode, markupNode, pointIndex, radius)
      else:  # Multiple Landmarks + Radius
        # Use only the selected points
        selectedVertices = self.selectMeshRegionBySelectedPoints(modelNode, markupNode, selectedPoints, radius)
      
      # Visualize the selection
      self.visualizeRegionSelection(modelNode, selectedVertices)
      
      # Update info label
      numVertices = len(selectedVertices)
      totalVertices = modelNode.GetPolyData().GetNumberOfPoints()
      percentage = (numVertices / totalVertices) * 100 if totalVertices > 0 else 0
      self.selectionInfoLabel.setText(f"Selected: {numVertices}/{totalVertices} vertices ({percentage:.1f}%)")
      
      # Enable clear and export buttons
      self.clearSelectionButton.enabled = True
      self.exportLandmarkSelectionButton.enabled = True
      
      print(f"Landmark region selection completed: {numVertices} vertices selected")
      
    except Exception as e:
      slicer.util.errorDisplay(f"Error during landmark selection: {str(e)}")
      print(f"Landmark selection error: {e}")
  
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
      # Get the current selection parameters
      method = self.selectionMethodCombo.currentText
      radius = self.selectionRadiusSlider.value
      
      # Get selected landmark points
      selectedPoints = self._getSelectedPoints(markupNode)
      
      if not selectedPoints:
        slicer.util.errorDisplay("No landmark points are selected. Please select some points first.")
        return
      
      # Get selected vertices
      if method == "Landmark + Radius":
        if len(selectedPoints) > 1:
          slicer.util.warningDisplay(f"Multiple points selected ({len(selectedPoints)}), but using single point mode. Using first selected point: {selectedPoints[0]}")
        pointIndex = selectedPoints[0]
        selectedVertices = self.selectMeshRegionByRadius(modelNode, markupNode, pointIndex, radius)
      else:  # Multiple Landmarks + Radius
        # Use only the selected points
        selectedVertices = self.selectMeshRegionBySelectedPoints(modelNode, markupNode, selectedPoints, radius)
      
      if not selectedVertices:
        slicer.util.warningDisplay("No vertices were selected. Try adjusting the radius or landmark positions.")
        return
      
      # Create new model from selected vertices
      newModelNode = self.createModelFromSelectedVertices(modelNode, selectedVertices)
      
      if newModelNode:
        # Enable clear button
        self.clearSelectionButton.enabled = True
        
        # Update info
        numVertices = len(selectedVertices)
        self.selectionInfoLabel.setText(f"Exported model: {numVertices} vertices")
        
        slicer.util.infoDisplay(f"Successfully exported selected region!\n"
                              f"New model: {newModelNode.GetName()}\n"
                              f"Vertices: {numVertices}")
        
        print(f"Landmark selection exported: {numVertices} vertices to new model '{newModelNode.GetName()}'")
      else:
        slicer.util.errorDisplay("Failed to create model from selected vertices.")
        
    except Exception as e:
      slicer.util.errorDisplay(f"Error exporting landmark selection: {str(e)}")
      print(f"Export landmark selection error: {e}")
  
  def onClearSelection(self):
    """Clear the current region selection"""
    modelNode = self.regionMeshSelector.currentNode()
    if modelNode:
      self.clearRegionSelection(modelNode)
      self.selectionInfoLabel.setText("No region selected")
      self.clearSelectionButton.enabled = False
      self.exportLandmarkSelectionButton.enabled = False
      print("Region selection cleared")
  
  def onClickSelectLandmark(self):
    """Enable click-to-select mode for landmarks"""
    if not hasattr(self, '_clickSelectMode') or not self._clickSelectMode:
      # Enable click-to-select mode
      self._clickSelectMode = True
      self._setupClickSelectMode()
      self.clickSelectLandmarkButton.setText("Cancel Click Selection")
      self.clickSelectLandmarkButton.setToolTip("Click on a landmark in the 3D view to select it, or click this button to cancel")
      slicer.util.infoDisplay("Click-to-select mode enabled. Click on a landmark point in the 3D view to select it.")
    else:
      # Disable click-to-select mode
      self._disableClickSelectMode()
  
  def _setupClickSelectMode(self):
    """Setup the click-to-select interaction"""
    markupNode = self.selectionMarkupSelector.currentNode()
    if not markupNode:
      slicer.util.errorDisplay("Please select markup points first.")
      return
    
    # Store original interaction mode
    self._originalInteractionMode = slicer.app.applicationLogic().GetInteractionNode().GetCurrentInteractionMode()
    
    # Enable interaction mode for clicking
    slicer.app.applicationLogic().GetInteractionNode().SetCurrentInteractionMode(slicer.vtkMRMLInteractionNode.ViewTransform)
    
    # Connect to markup selection events using Slicer's built-in system
    self._setupMarkupSelectionObserver()
  
  def _disableClickSelectMode(self):
    """Disable click-to-select mode"""
    self._clickSelectMode = False
    self.clickSelectLandmarkButton.setText("Click to Select Landmark")
    self.clickSelectLandmarkButton.setToolTip("Click on a landmark in the 3D view to select it for region selection")
    
    # Restore original interaction mode
    if hasattr(self, '_originalInteractionMode'):
      slicer.app.applicationLogic().GetInteractionNode().SetCurrentInteractionMode(self._originalInteractionMode)
    
    # Disconnect from markup selection observers
    markupNode = self.selectionMarkupSelector.currentNode()
    if markupNode:
      if hasattr(self, '_markupObserver'):
        markupNode.RemoveObserver(self._markupObserver)
      if hasattr(self, '_markupSelectionObserver'):
        markupNode.RemoveObserver(self._markupSelectionObserver)
  
  def _setupMarkupSelectionObserver(self):
    """Setup observer for markup point selection events"""
    markupNode = self.selectionMarkupSelector.currentNode()
    if not markupNode:
      return
    
    # Add observer for markup point selection events
    # Use the correct event names for Slicer
    self._markupObserver = markupNode.AddObserver(slicer.vtkMRMLMarkupsNode.PointModifiedEvent, self._onMarkupPointInteraction)
    
    # Also observe for point selection events
    self._markupSelectionObserver = markupNode.AddObserver(slicer.vtkMRMLMarkupsNode.PointPositionDefinedEvent, self._onMarkupPointInteraction)
  
  
  def _onMarkupPointInteraction(self, caller, event):
    """Handle markup point interaction events"""
    if not self._clickSelectMode:
      return
    
    markupNode = self.selectionMarkupSelector.currentNode()
    if not markupNode:
      return
    
    # Update the selected points display
    self._updateSelectedPointsDisplay()
  
  def _updateSelectedPointsDisplay(self):
    """Update the display of selected landmark points"""
    markupNode = self.selectionMarkupSelector.currentNode()
    if not markupNode:
      self.selectedPointsLabel.setText("No points selected")
      return
    
    # Get list of selected points
    selectedPoints = self._getSelectedPoints(markupNode)
    
    if selectedPoints:
      pointsText = ", ".join(map(str, selectedPoints))
      self.selectedPointsLabel.setText(f"Points: {pointsText}")
    else:
      self.selectedPointsLabel.setText("No points selected")
  
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
  
  def selectMeshRegionByRadius(self, modelNode, markupNode, pointIndex, radius):
    """Select mesh vertices within radius of a specific markup point"""
    # Get the markup point position
    point = [0, 0, 0]
    markupNode.GetNthControlPointPosition(pointIndex, point)
    
    # Get mesh data
    polyData = modelNode.GetPolyData()
    points = polyData.GetPoints()
    totalPoints = points.GetNumberOfPoints()
    
    # Calculate mesh bounds for debugging
    bounds = polyData.GetBounds()
    meshSize = max(bounds[1]-bounds[0], bounds[3]-bounds[2], bounds[5]-bounds[4])
    
    # Find vertices within radius
    selectedVertices = []
    radiusSquared = radius * radius
    
    for i in range(totalPoints):
      vertex = points.GetPoint(i)
      distanceSquared = vtk.vtkMath.Distance2BetweenPoints(point, vertex)
      if distanceSquared <= radiusSquared:
        selectedVertices.append(i)
    
    # Debug information
    print(f"Landmark selection debug:")
    print(f"  Radius: {radius} units")
    print(f"  Mesh size: {meshSize:.3f} units (max dimension)")
    print(f"  Radius as % of mesh: {(radius/meshSize)*100:.2f}%")
    print(f"  Selected: {len(selectedVertices)}/{totalPoints} vertices ({(len(selectedVertices)/totalPoints)*100:.1f}%)")
    
    return selectedVertices
  
  def selectMeshRegionByMultiplePoints(self, modelNode, markupNode, radius):
    """Select mesh vertices within radius of any markup point"""
    # Get mesh data
    polyData = modelNode.GetPolyData()
    points = polyData.GetPoints()
    selectedVertices = set()  # Use set to avoid duplicates
    radiusSquared = radius * radius
    
    # Check each markup point
    for pointIndex in range(markupNode.GetNumberOfControlPoints()):
      # Get the markup point position
      point = [0, 0, 0]
      markupNode.GetNthControlPointPosition(pointIndex, point)
      
      # Find vertices within radius of this point
      for i in range(points.GetNumberOfPoints()):
        vertex = points.GetPoint(i)
        distanceSquared = vtk.vtkMath.Distance2BetweenPoints(point, vertex)
        if distanceSquared <= radiusSquared:
          selectedVertices.add(i)
    
    return list(selectedVertices)
  
  def selectMeshRegionBySelectedPoints(self, modelNode, markupNode, selectedPointIndices, radius):
    """Select mesh vertices within radius of selected markup points only"""
    # Get mesh data
    polyData = modelNode.GetPolyData()
    points = polyData.GetPoints()
    selectedVertices = set()  # Use set to avoid duplicates
    radiusSquared = radius * radius
    
    # Check only the selected markup points
    for pointIndex in selectedPointIndices:
      # Get the markup point position
      point = [0, 0, 0]
      markupNode.GetNthControlPointPosition(pointIndex, point)
      
      # Find vertices within radius of this point
      for i in range(points.GetNumberOfPoints()):
        vertex = points.GetPoint(i)
        distanceSquared = vtk.vtkMath.Distance2BetweenPoints(point, vertex)
        if distanceSquared <= radiusSquared:
          selectedVertices.add(i)
    
    return list(selectedVertices)
  
  def visualizeRegionSelection(self, modelNode, selectedVertices):
    """Visualize the selected region by coloring vertices"""
    polyData = modelNode.GetPolyData()
    
    # Create a scalar array for selection visualization
    selectionArray = vtk.vtkIntArray()
    selectionArray.SetName("RegionSelection")
    selectionArray.SetNumberOfTuples(polyData.GetNumberOfPoints())
    selectionArray.Fill(0)  # 0 = not selected
    
    # Mark selected vertices
    for vertexId in selectedVertices:
      selectionArray.SetValue(vertexId, 1)  # 1 = selected
    
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
    
    modelNode.Modified()
  
  def createModelFromSelectedVertices(self, modelNode, selectedVertices):
    """Create a new model from selected vertices"""
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
      for vertexId in selectedVertices:
        vertexMapping[vertexId] = newVertexId
        point = points.GetPoint(vertexId)
        newPoints.InsertNextPoint(point)
        newVertexId += 1
      
      # Add faces that only contain selected vertices
      polys.InitTraversal()
      cell = vtk.vtkIdList()
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
      
      # Set up the new polydata
      newPolyData.SetPoints(newPoints)
      newPolyData.SetPolys(newPolys)
      
      # Create new model node
      newModelNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode")
      newModelNode.SetAndObservePolyData(newPolyData)
      
      # Set name based on original model
      originalName = modelNode.GetName()
      newModelNode.SetName(f"{originalName}_SelectedRegion")
      
      # Copy display properties from original model
      originalDisplayNode = modelNode.GetDisplayNode()
      if originalDisplayNode:
        newDisplayNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelDisplayNode")
        newDisplayNode.SetColor(originalDisplayNode.GetColor())
        newDisplayNode.SetOpacity(originalDisplayNode.GetOpacity())
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
      
      # Ensure only relevant DeCA models are visible
      self._ensureModelsAreVisible()
      
      # Update the button text to indicate visualization is active
      self.startVisualizationButton.setText("Visualization Active")
      self.startVisualizationButton.setStyleSheet("""
        QPushButton {
          background-color: #90EE90;
          color: #2C3E50;
          font-weight: bold;
          border: none;
          border-radius: 5px;
          padding: 8px 16px;
          min-height: 30px;
        }
        QPushButton:hover {
          background-color: #7FDD7F;
        }
        QPushButton:pressed {
          background-color: #6ECC6E;
        }
      """)
      
      print("Visualization started - markups hidden and models made visible")
      
    except Exception as e:
      print(f"Error starting visualization: {e}")

  def setUpDeCADir(self, outDir, symmetryOption=False, errorDirectoryOption=False, DeCALOption=False, loadAtlasOption = False):
    dateTimeStamp = datetime.now().strftime('%Y_%m-%d_%H_%M_%S')
    outputFolderDC = os.path.join(outDir, dateTimeStamp)
    fileNameDictionary = {}
    try:
      os.makedirs(outputFolderDC)
      alignedLMFolderDC = os.path.join(outputFolderDC, "alignedLMs")
      os.makedirs(alignedLMFolderDC)
      alignedModelFolderDC = os.path.join(outputFolderDC, "alignedModels")
      os.makedirs(alignedModelFolderDC)
      resampledModelFolderDC = os.path.join(outputFolderDC, "resampledModels") # <-- ADD THIS LINE
      os.makedirs(resampledModelFolderDC)  
      # initialize the filename dictionary
      fileNameDictionary['output'] = str(outputFolderDC)
      fileNameDictionary['alignedLMs'] = str(alignedLMFolderDC)
      fileNameDictionary['alignedModels'] = str(alignedModelFolderDC)
      fileNameDictionary['resampledModels'] = str(resampledModelFolderDC) 
      if not loadAtlasOption:
        tempLMFolderDC = os.path.join(outputFolderDC, "tempAlignedLMs")
        os.makedirs(tempLMFolderDC)
        tempModelFolderDC = os.path.join(outputFolderDC, "tempAlignedModels")
        os.makedirs(tempModelFolderDC)
        fileNameDictionary['tempAlignedLMs'] = str(tempLMFolderDC)
        fileNameDictionary['tempAlignedModels'] = str(tempModelFolderDC)
      if symmetryOption:
        mirrorLMFolderDC = os.path.join(outputFolderDC, "mirrorLMs")
        os.makedirs(mirrorLMFolderDC)
        mirrorModelFolderDC = os.path.join(outputFolderDC, "mirrorModels")
        os.makedirs(mirrorModelFolderDC)
        fileNameDictionary['mirrorLMs'] = str(mirrorLMFolderDC)
        fileNameDictionary['mirrorModels'] = str(mirrorModelFolderDC)
      if errorDirectoryOption:
        errorCheckingFolderDC = os.path.join(outputFolderDC, "errorChecking")
        os.makedirs(errorCheckingFolderDC)
        fileNameDictionary['error'] = str(errorCheckingFolderDC)
      if DeCALOption:
        DeCALOutputFolder = os.path.join(outputFolderDC, "DeCALOutput")
        os.makedirs(DeCALOutputFolder)
        fileNameDictionary['DeCALOutput'] = str(DeCALOutputFolder)
    except:
      logging.debug('Result directory failed: Could not create output folder')
    return fileNameDictionary

  def onToggleAnalysis(self):
    if self.analysisTypeSymmetry.checked == True:
      self.symmetryCollapsibleButton.collapsed = False
      self.symmetryCollapsibleButton.enabled = True
    else:
      self.symmetryCollapsibleButton.collapsed = True
      self.symmetryCollapsibleButton.enabled = False

  def onToggleAtlasDC(self):
    if self.calculateAtlasOptionDC.checked == True:
      self.atlasCollapsibleButtonDC.collapsed = True
      self.atlasCollapsibleButtonDC.enabled = False
    else:
      self.atlasCollapsibleButtonDC.collapsed = False
      self.atlasCollapsibleButtonDC.enabled = True
    self.onParameterSelectDC()

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
    atlasPathSelected = bool(self.DCBaseModelSelector.currentPath and self.DCBaseLMSelector.currentPath) or self.calculateAtlasOptionDC.checked
    inputPathsSelected = bool(self.meshDirectoryDC.currentPath and self.landmarkDirectoryDC.currentPath and self.outputDirectoryDC.currentPath)
    self.applyButtonDC.enabled = bool(atlasPathSelected and inputPathsSelected)

  def onParameterSelectDCL(self):
    atlasPathSelected = bool(self.DCLBaseModelSelector.currentPath and self.DCLBaseLMSelector.currentPath) or self.calculateAtlasOptionDCL.checked
    inputPathsSelected = bool(self.meshDirectoryDCL.currentPath and self.landmarkDirectoryDCL.currentPath and self.OutputDirectoryDCL.currentPath)
    self.getAtlasButton.enabled = bool(atlasPathSelected and inputPathsSelected)

  def onPointSelectionSelect(self):
    self.subsetApplyButton.enabled = bool(self.DCLLandmarkDirectory.currentPath and self.pointSelection.currentNode())

  def onDCLLandmarkDirectorySelect(self):
    self.subsetApplyButton.enabled = bool(self.DCLLandmarkDirectory.currentPath and self.pointSelection.currentNode())

  def onGenerateAtlasButton(self):
    logic = InterDeCALogic()
    #set up output directory
    self.folderNames = self.setUpDeCADir(self.OutputDirectoryDCL.currentPath, False, False, True, self.loadAtlasOptionDCL.checked)
    if self.folderNames == {}:
      self.logInfoDCL.appendPlainText(f'Output folders could not be created in {self.OutputDirectoryDCL.currentPath}')
      return
    self.folderNames['originalLMs'] = self.landmarkDirectoryDCL.currentPath
    self.folderNames['originalModels'] = self.meshDirectoryDCL.currentPath
    if self.loadAtlasOptionDCL.checked:
      try:
        atlasModelPath = self.DCLBaseModelSelector.currentPath
        self.atlasModel = slicer.util.loadModel(atlasModelPath)
      except:
        self.logInfoDCL.appendPlainText(f"Can't load model from: {atlasModelPath}")
        return
      try:
        atlasLMPath = self.DCLBaseLMSelector.currentPath
        self.atlasLMs = slicer.util.loadMarkups(atlasLMPath)
      except:
        print("Can't load from: ", atlasLMPath)
        self.logInfoDCL.appendPlainText(f"Can't load landmarks from: {atlasLMPath}")
        return
    else:
      removeScale = True
      self.atlasModel, self.atlasLMs = self.generateNewAtlas(removeScale, self.logInfoDCL)
    atlasModelPath = os.path.join(self.folderNames['output'], 'decaAtlasModel.ply')
    self.logInfoDCL.appendPlainText(f"Saving atlas model to {atlasModelPath}")
    slicer.util.saveNode(self.atlasModel, atlasModelPath)
    atlasLMPath = os.path.join(self.folderNames['output'], 'decaAtlasLM.mrk.json')
    self.logInfoDCL.appendPlainText(f"Saving atlas landmarks to {atlasLMPath}")
    slicer.util.saveNode(self.atlasLMs, atlasLMPath)
    self.getPointNumberButton.enabled = True

  def onTabChanged(self, index):
    if self.tabsWidget.tabText(index) == "Visualize Results":
      # Only update the preview list, don't automatically change the scene
      self.updateBakedPreviewList()
      # Reset the visualization button state
      self.resetVisualizationButton()
    
  def resetVisualizationButton(self):
    """Reset the Start Visualization button to its initial state"""
    self.startVisualizationButton.setText("Start Visualization")
    self.startVisualizationButton.setStyleSheet("""
      QPushButton {
        background-color: #87CEEB;
        color: #2C3E50;
        font-weight: bold;
        border: none;
        border-radius: 5px;
        padding: 8px 16px;
        min-height: 30px;
      }
      QPushButton:hover {
        background-color: #6BB6E8;
      }
      QPushButton:pressed {
        background-color: #4FA8D8;
      }
    """)

  def onToggleLandmarkLock(self):
    """Toggle landmark lock state"""
    try:
      # Get all landmark/markup nodes
      markups = list(slicer.util.getNodesByClass('vtkMRMLMarkupsNode'))
      if not markups:  # fallback for older Slicer builds
        markups = list(slicer.util.getNodesByClass('vtkMRMLMarkupsFiducialNode'))
      
      if not markups:
        slicer.util.infoDisplay("No landmarks found in the scene.")
        return
      
      # Check current lock state (assume all landmarks have same lock state)
      current_locked = self._areLandmarksLocked(markups[0])
      new_locked = not current_locked
      
      # Apply lock state to all landmarks
      for markup in markups:
        self._setLandmarkLockState(markup, new_locked)
      
      # Update UI
      self._updateLandmarkLockUI(new_locked)
      
      status = "Locked" if new_locked else "Unlocked"
      print(f"Landmarks {status.lower()}")
      
    except Exception as e:
      slicer.util.errorDisplay(f"Error toggling landmark lock: {str(e)}")

  def _areLandmarksLocked(self, markup_node):
    """Check if landmarks are currently locked"""
    try:
      # Check if the markup node has locked property
      if hasattr(markup_node, 'GetLocked'):
        return markup_node.GetLocked()
      
      # Alternative: check display node properties
      display_node = markup_node.GetDisplayNode()
      if display_node and hasattr(display_node, 'GetLocked'):
        return display_node.GetLocked()
      
      # Default to unlocked if we can't determine
      return False
    except Exception:
      return False

  def _setLandmarkLockState(self, markup_node, locked):
    """Set the lock state for a landmark node"""
    try:
      # Try to set locked property on the markup node
      if hasattr(markup_node, 'SetLocked'):
        markup_node.SetLocked(locked)
      
      # Also try to set on display node
      display_node = markup_node.GetDisplayNode()
      if display_node and hasattr(display_node, 'SetLocked'):
        display_node.SetLocked(locked)
      
      # Alternative: disable interaction by setting visibility and interaction
      if display_node:
        if locked:
          # When locked, make landmarks visible but non-interactive
          display_node.SetVisibility(True)
          if hasattr(display_node, 'SetInteractive'):
            display_node.SetInteractive(False)
          # Change color to indicate locked state
          if hasattr(display_node, 'SetSelectedColor'):
            display_node.SetSelectedColor(1.0, 0.0, 0.0)  # Red for locked
        else:
          # When unlocked, restore normal interaction
          display_node.SetVisibility(True)
          if hasattr(display_node, 'SetInteractive'):
            display_node.SetInteractive(True)
          # Restore normal color
          if hasattr(display_node, 'SetSelectedColor'):
            display_node.SetSelectedColor(0.0, 1.0, 0.0)  # Green for unlocked
      
    except Exception as e:
      print(f"Warning: Could not set lock state for landmark: {e}")

  def _updateLandmarkLockUI(self, locked):
    """Update the UI to reflect the current lock state"""
    try:
      if locked:
        self.landmarkLockButton.setText("Unlock Landmarks")
        self.landmarkLockButton.setStyleSheet("""
          QPushButton {
            background-color: #4CAF50;
            color: white;
            font-weight: bold;
            border: none;
            border-radius: 5px;
            padding: 6px 12px;
            min-height: 25px;
          }
          QPushButton:hover {
            background-color: #45a049;
          }
          QPushButton:pressed {
            background-color: #3d8b40;
          }
        """)
      else:
        self.landmarkLockButton.setText("Lock Landmarks")
        self.landmarkLockButton.setStyleSheet("""
          QPushButton {
            background-color: #ff6b6b;
            color: white;
            font-weight: bold;
            border: none;
            border-radius: 5px;
            padding: 6px 12px;
            min-height: 25px;
          }
          QPushButton:hover {
            background-color: #ff5252;
          }
          QPushButton:pressed {
            background-color: #e53935;
          }
        """)
    except Exception as e:
      print(f"Warning: Could not update landmark lock UI: {e}")


  def generateNewAtlas(self, removeScale, log):
    """Generate a new atlas model and landmark set from data"""
    logic = InterDeCALogic()

    # getClosestToMeanPath returns a filename, we need to extract the base subject ID
    try:
      closestFileName = logic.getClosestToMeanPath(self.folderNames['originalLMs'])
      if closestFileName is None:
        log.appendPlainText("Error: Could not determine closest sample to mean")
        return None, None

      # Extract the base subject ID by removing landmark file extensions
      subjectID = closestFileName
      # Strip common landmark file extensions (.fcsv, .mrk, .json)
      fileNameBase = Path(subjectID)
      while fileNameBase.suffix in {'.fcsv', '.mrk', '.json'}:
        fileNameBase = fileNameBase.with_suffix('')
      subjectID = str(fileNameBase)

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
                    removeScale)
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
    logic = InterDeCALogic()
    subsampledTemplate, pointNumber = logic.runCheckPoints(self.atlasModel, self.spacingTolerance.value)
    self.logInfoDCL.appendPlainText(f'The subsampled template has a total of {pointNumber} points.')
    self.DCLApplyButton.enabled = True

  def onTabChanged(self, index):
    if self.tabsWidget.tabText(index) == "Visualize Results":
      # Only update the preview list, don't automatically change the scene
      self.updateBakedPreviewList()
      # Reset the visualization button state
      self.resetVisualizationButton()

  def updateBakedPreviewList(self):
    self.previewTextureCombo.blockSignals(True)
    self.previewTextureCombo.clear()
    d = self.lastBakedTexturesPath
    if d and os.path.isdir(d):
      items = [f for f in os.listdir(d) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
      items.sort()
      self.previewTextureCombo.addItems(items)
    self.previewTextureCombo.blockSignals(False)

  def onPreviewTextureSelected(self, idx):
    if idx < 0: return
    if not hasattr(self, 'atlasModel') or self.atlasModel is None:
      slicer.util.errorDisplay("Atlas model is not in the scene.")
      return
    sid = self.previewTextureCombo.currentText
    png = os.path.join(self.lastBakedTexturesPath or "", sid + ".png")
    if not os.path.isfile(png): return
    InterDeCALogic().applyTextureToModel(self.atlasModel, png)

  def onDCApplyButton(self):
    logic = InterDeCALogic()
    symmetryOption    = self.analysisTypeSymmetry.checked
    writeErrorOption  = self.writeErrorCheckBox.checked
    loadAtlasOption   = self.loadAtlasOptionDC.checked
    removeScaleOption = self.removeScaleCheckBoxDC.checked

    # Start progress tracking
    self.applyButtonDC.enabled = False
    self.updateProgressDC(0, "Initializing DeCA analysis...")

    # Folders
    self.folderNames = self.setUpDeCADir(self.outputDirectoryDC.currentPath, symmetryOption, writeErrorOption, False, loadAtlasOption)
    if not self.folderNames:
      self.logInfoDC.appendPlainText(f'Output folders could not be created in {self.outputDirectoryDC.currentPath}')
      self.resetProgressDC()
      self.applyButtonDC.enabled = True
      return
    self.folderNames['originalLMs']  = self.landmarkDirectoryDC.currentPath
    self.folderNames['originalModels'] = self.meshDirectoryDC.currentPath
    self.lastDeCAAlignedModelsPath = self.folderNames['resampledModels']  # for Visualize tab

    # ---- 1) Load or compute atlas (Slicer) ----
    self.updateProgressDC(10, "Loading or computing atlas...")
    if loadAtlasOption:
      try:
        self.atlasModel = slicer.util.loadModel(self.DCBaseModelSelector.currentPath)
      except Exception:
        self.logInfoDC.appendPlainText(f"Can't load model from: {self.DCBaseModelSelector.currentPath}")
        self.resetProgressDC()
        self.applyButtonDC.enabled = True
        return
      try:
        self.atlasLMs = slicer.util.loadMarkups(self.DCBaseLMSelector.currentPath)
      except Exception:
        self.logInfoDC.appendPlainText(f"Can't load landmarks from: {self.DCBaseLMSelector.currentPath}")
        self.resetProgressDC()
        self.applyButtonDC.enabled = True
        return
    else:
      self.atlasModel, self.atlasLMs = self.generateNewAtlas(removeScaleOption, self.logInfoDC)

    # Check if atlas generation was successful
    if self.atlasModel is None or self.atlasLMs is None:
      self.logInfoDC.appendPlainText("Failed to generate atlas. Please check the data and try again.")
      self.resetProgressDC()
      self.applyButtonDC.enabled = True
      return

    # Save an intermediate atlas file (RAS) so Blender can read it
    self.updateProgressDC(30, "Preparing atlas for UV mapping...")
    atlas_preuv_obj = os.path.join(self.folderNames['output'], 'decaAtlas_preUV.obj')
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
    atlas_uv_obj = os.path.join(self.folderNames['output'], 'decaAtlasUV.obj')
    try:
      logic.blender_prepare_atlas(blender_exe, atlas_preuv_obj, atlas_uv_obj,
                                  merge_dist=merge_dist, smart_angle=smart_angle, island_margin=island_margin)
      self.logInfoDC.appendPlainText(f"Atlas cleaned & UV’d in Blender → {atlas_uv_obj}")
    except Exception as e:
      self.logInfoDC.appendPlainText(f"Blender atlas UV step failed: {e}")
      return

    # Reload UV’d atlas back into Slicer (replace old atlas node)
    try:
      slicer.mrmlScene.RemoveNode(self.atlasModel)
    except Exception:
      pass
    self.atlasModel = logic._load_model_with_cs(atlas_uv_obj, 'RAS')

    median_dist = logic._median_landmark_to_surface_dist(self.atlasModel, self.atlasLMs)
    if median_dist > 5.0 * np.mean(self.atlasModel.GetPolyData().GetLength()):
      self.logInfoDC.appendPlainText(f"WARNING: Landmarks are far from the surface ({median_dist:.1f} mm)")

    # Save atlas landmarks & a copy of the atlas (PLY) for provenance
    atlasLMPath   = os.path.join(self.folderNames['output'], 'decaAtlasLM.mrk.json')
    slicer.util.saveNode(self.atlasLMs, atlasLMPath)
    atlasPlyPath  = os.path.join(self.folderNames['output'], 'decaAtlasModel.ply')
    logic._save_model_with_cs(self.atlasModel, atlasPlyPath, 'RAS')

    # ---- 3) Rigid alignment of subjects to atlas (Slicer) ----
    self.updateProgressDC(60, "Rigid alignment to atlas...")
    try:
      self.logInfoDC.appendPlainText("Rigid alignment to atlas")
      logic.runAlign(self.atlasModel, self.atlasLMs,
                     self.folderNames['originalModels'], self.folderNames['originalLMs'],
                     self.folderNames['alignedModels'], self.folderNames['alignedLMs'],
                     removeScaleOption)
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
        writeErrorOption,
        atlas_uv_template_obj=atlas_uv_obj  # NEW: used to stamp the same UVs onto resampled OBJ copies
      )
    except Exception as e:
      self.logInfoDC.appendPlainText(f"DeCA resampling failed: {e}")
      self.resetProgressDC()
      self.applyButtonDC.enabled = True
      return

    # ---- 5) Blender bake (selection→active) from aligned → resampled(OBJ with atlas UV) ----
    self.updateProgressDC(80, "Setting up texture baking...")
    self.lastBakedTexturesPath = os.path.join(self.folderNames['output'], "atlasTextures")
    os.makedirs(self.lastBakedTexturesPath, exist_ok=True)

    texturesDir = self.textureDirectoryDC.currentPath
    if os.path.isdir(texturesDir):
      self.updateProgressDC(85, "Baking textures with Blender...")
      try:
        made = logic.blender_bake_all(
          blender_exe=blender_exe,
          alignedDir=self.folderNames['alignedModels'],
          resampledUVDir=os.path.join(self.folderNames['output'], "resampledOBJ_withUV"),
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

    # ---- 6) Fill Visualize dropdown ----
    self.updateProgressDC(100, "Finalizing results...")
    self.updateBakedPreviewList()
    
    # Success - reset progress and re-enable button
    self.logInfoDC.appendPlainText("DeCA analysis completed successfully!")
    self.resetProgressDC()
    self.applyButtonDC.enabled = True


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

  ################################### Colors EDA Functions ###################################

  def onSamplingParameterChanged(self):
    """Enable/disable the sample data button based on parameter selection"""
    atlasSelected = bool(self.colorsAtlasModelSelect.currentNode())
    texturesSelected = bool(self.bakedTexturesDirectorySelector.currentPath and
                           os.path.isdir(self.bakedTexturesDirectorySelector.currentPath))
    self.sampleDataButton.enabled = atlasSelected and texturesSelected

  def onSampleDataButton(self):
    """Sample faces and calculate color averages from textures"""
    try:
      qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)
      self.samplingProgressBar.setVisible(True)
      self.samplingProgressBar.setValue(0)

      # Get parameters
      atlasModel = self.colorsAtlasModelSelect.currentNode()
      texturesDir = self.bakedTexturesDirectorySelector.currentPath
      randomSeed = int(self.randomSeedSpin.value)
      samplePercent = float(self.faceSamplePercentSpin.value)

      self.samplingStatusLabel.setText("Sampling in progress...")
      self.samplingStatusLabel.setStyleSheet("color: #0066cc; font-style: italic;")

      # Run the sampling
      logic = InterDeCALogic()
      result = logic.sampleColorData(
        atlasModel, texturesDir, randomSeed, samplePercent,
        progressCallback=self.updateSamplingProgress,
        logCallback=self.logSamplingMessage
      )

      if result and result.get('success'):
        # Store the sampled data
        self.sampledColorData = result['colorData']
        self.sampledSpecimenNames = result['specimenNames']
        self.sampledFaceIndices = result['faceIndices']

        # Update status
        nSpecimens = len(self.sampledSpecimenNames)
        nFaces = len(self.sampledFaceIndices)
        self.samplingStatusLabel.setText(f"Sampled {nFaces} faces from {nSpecimens} specimens")
        self.samplingStatusLabel.setStyleSheet("color: #006600; font-weight: bold;")

        # Enable analysis phase
        self.analysisWidget.enabled = True
        self.plotButton.enabled = True

        self.colorsEDALogInfo.appendPlainText(f"Data sampling completed successfully!")
        self.colorsEDALogInfo.appendPlainText(f"Sampled {nFaces} faces ({samplePercent:.1f}%) from {nSpecimens} specimens")
      else:
        self.samplingStatusLabel.setText("Sampling failed")
        self.samplingStatusLabel.setStyleSheet("color: #cc0000; font-weight: bold;")
        self.colorsEDALogInfo.appendPlainText("Data sampling failed - check log for details")

      self.samplingProgressBar.setVisible(False)
      qt.QApplication.restoreOverrideCursor()

    except Exception as e:
      self.samplingProgressBar.setVisible(False)
      qt.QApplication.restoreOverrideCursor()
      self.samplingStatusLabel.setText("Sampling failed")
      self.samplingStatusLabel.setStyleSheet("color: #cc0000; font-weight: bold;")
      self.colorsEDALogInfo.appendPlainText(f"Error during sampling: {str(e)}")
      slicer.util.errorDisplay(f"Data sampling failed: {str(e)}")
      import traceback
      traceback.print_exc()

  def updateSamplingProgress(self, value):
    """Update sampling progress bar"""
    self.samplingProgressBar.setValue(int(value))
    slicer.app.processEvents()

  def logSamplingMessage(self, message):
    """Log message during sampling"""
    self.colorsEDALogInfo.appendPlainText(message)
    slicer.app.processEvents()

  def _ensureMainPlotViewNode(self):
    """Get the main plot view node for displaying charts"""
    try:
      layoutManager = slicer.app.layoutManager()
      plotWidget = layoutManager.plotWidget(0)
      if plotWidget:
        return plotWidget.mrmlPlotViewNode()
      return None
    except Exception:
      return None

  def _refreshCurrentHistogram(self):
    """Re-render the histogram/scatter view using current UI options."""
    if not hasattr(self, '_lastColorData') or not hasattr(self, '_lastColorSpace'):
        return
    if not self.viewChannelRadio.isChecked():
        return  # only relevant to histogram view
    try:
        chan = int(self.histChannelSelector.currentIndex)
    except Exception:
        chan = 0
    try:
        self._plotHistogramInMainView(self._lastColorData, self._lastColorSpace, chan)
    except Exception as e:
        try: self.colorsEDALogInfo.appendPlainText(f"Refresh failed: {e}")
        except: pass


  def onPlotButton(self):
    """Generate plots from sampled data using selected analysis settings"""
    try:
      # Check if we have sampled data
      if self.sampledColorData is None:
        slicer.util.errorDisplay("No sampled data available. Please run 'Sample Data' first.")
        return

      qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)
      self.analysisProgressBar.setVisible(True)
      self.analysisProgressBar.setValue(0)

      # Get analysis parameters
      colorSpace = "HSV" if self.hsvRadio.isChecked() else "RGB"

      if self.pcaRadio.isChecked():
        dimRedAlgo = "PCA"
      elif self.icaRadio.isChecked():
        dimRedAlgo = "ICA"
      else:
        dimRedAlgo = "UMAP"

      # Get HSV cutoff parameters for dimensionality reduction
      satCutoff = float(self.satCutoffSpin.value)
      valueCutoff = float(self.valueCutoffSpin.value)
      enhanceColors = bool(self.enhanceColorsCheck.isChecked())

      self.colorsEDALogInfo.appendPlainText(f"Starting analysis on sampled data...")
      self.colorsEDALogInfo.appendPlainText(f"Color space: {colorSpace}")
      self.colorsEDALogInfo.appendPlainText(f"Dimensionality reduction: {dimRedAlgo}")

      # Run the analysis on pre-sampled data
      logic = InterDeCALogic()
      result = logic.runColorsEDAFromSampledData(
        self.sampledColorData, self.sampledSpecimenNames, colorSpace, dimRedAlgo,
        progressCallback=self.updateAnalysisProgress,
        logCallback=self.logAnalysisMessage,
        satCutoff=satCutoff,
        valueCutoff=valueCutoff,
        enhanceColors=enhanceColors
      )

      if result and isinstance(result, dict) and result.get('success'):
        self.colorsEDALogInfo.appendPlainText("Analysis completed successfully!")
        # Save color data and enable histogram selector
        if 'colorData' in result:
          self._lastColorData = result['colorData']  # Full dataset
          self._lastColorSpace = colorSpace
          self._refreshHistogramChannelOptions(colorSpace)
          # Default to first channel
          self.histChannelSelector.setCurrentIndex(0)

          # Store the cutoff values used for this analysis
          self._lastSatCutoff = satCutoff
          self._lastValueCutoff = valueCutoff

          # Store the 2D plot chart node for view switching
          if 'chartNode' in result and result['chartNode']:
            self._last2DPlotChartNode = result['chartNode']

          # Show appropriate view based on radio button selection
          if self.view2DRadio.isChecked():
            self.colorsEDALogInfo.appendPlainText("Results plotted in 2D viewer")
          else:
            self.colorsEDALogInfo.appendPlainText("Channel histogram view enabled")
            try:
              self._refreshCurrentHistogram()
            except Exception as e:
              self.colorsEDALogInfo.appendPlainText(f"Failed to show histogram: {e}")
      else:
        self.colorsEDALogInfo.appendPlainText("Analysis failed - check log for details")

      self.analysisProgressBar.setVisible(False)
      qt.QApplication.restoreOverrideCursor()

    except Exception as e:
      self.analysisProgressBar.setVisible(False)
      qt.QApplication.restoreOverrideCursor()
      self.colorsEDALogInfo.appendPlainText(f"Error: {str(e)}")
      slicer.util.errorDisplay(f"Analysis failed: {str(e)}")
      import traceback
      traceback.print_exc()

  def updateAnalysisProgress(self, value):
    """Update analysis progress bar"""
    self.analysisProgressBar.setValue(int(value))
    slicer.app.processEvents()

  def logAnalysisMessage(self, message):
    """Log message during analysis"""
    self.colorsEDALogInfo.appendPlainText(message)
    slicer.app.processEvents()

  # Legacy function names kept for compatibility with existing logic methods
  def updateColorsEDAProgress(self, value):
    """Update progress bar (legacy compatibility)"""
    self.analysisProgressBar.setValue(int(value))
    slicer.app.processEvents()

  def logColorsEDAMessage(self, message):
    """Log message to the Colors EDA log (legacy compatibility)"""
    self.colorsEDALogInfo.appendPlainText(message)
    slicer.app.processEvents()

  def _clearHistogramNodes(self):
    """Remove previously created MRML nodes for histograms to keep scene clean."""
    try:
      for n in getattr(self, 'histSeriesNodes', []):
        if n:
          slicer.mrmlScene.RemoveNode(n)
      for n in getattr(self, 'histTableNodes', []):
        if n:
          slicer.mrmlScene.RemoveNode(n)
      for n in getattr(self, 'histChartNodes', []):
        if n:
          slicer.mrmlScene.RemoveNode(n)
    except Exception as e:
      print(f"Histogram cleanup warning: {e}")
    self.histSeriesNodes = []
    self.histTableNodes = []
    self.histChartNodes = []

  def _createDensityHistogramPlot(self, data, bins, value_range, title, x_label, color=(0.4, 0.4, 0.4)):
    """Create a density histogram (bars) and return the chart node."""
    # Compute density histogram (ignore NaNs)
    data = np.asarray(data)
    data = data[np.isfinite(data)]
    if data.size == 0:
      raise ValueError("Empty data for density plot")
    density, edges = np.histogram(data, bins=bins, range=value_range, density=True)
    centers = (edges[:-1] + edges[1:]) / 2.0

    # Create table with X (bin centers) and Y (density)
    xArray = vtk.vtkFloatArray(); xArray.SetName("Value")
    xArray.SetNumberOfTuples(len(centers))
    yArray = vtk.vtkFloatArray(); yArray.SetName("Density")
    yArray.SetNumberOfTuples(len(density))
    for i in range(len(centers)):
      xArray.SetValue(i, float(centers[i]))
      yArray.SetValue(i, float(density[i]))

    tableNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTableNode")
    tableNode.AddColumn(xArray)
    tableNode.AddColumn(yArray)

    seriesNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLPlotSeriesNode")
    seriesNode.SetAndObserveTableNodeID(tableNode.GetID())
    seriesNode.SetXColumnName("Value")
    seriesNode.SetYColumnName("Density")
    seriesNode.SetPlotType(slicer.vtkMRMLPlotSeriesNode.PlotTypeBar)
    try:
      seriesNode.SetColor(*color)
    except Exception:
      pass

    chartNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLPlotChartNode")
    chartNode.AddAndObservePlotSeriesNodeID(seriesNode.GetID())
    chartNode.SetTitle(title)
    chartNode.SetXAxisTitle(x_label)
    chartNode.SetYAxisTitle("Density")
    chartNode.SetLegendVisibility(False)

    # Track nodes for cleanup
    self.histTableNodes.append(tableNode)
    self.histSeriesNodes.append(seriesNode)
    self.histChartNodes.append(chartNode)

    return chartNode

  def _ensureMainPlotViewNode(self):
    """Get the main plot view node used by the module (same pane as scatter)."""
    try:
      layoutManager = slicer.app.layoutManager()
      plotWidget = layoutManager.plotWidget(0)
      return plotWidget.mrmlPlotViewNode()
    except Exception:
      return None

  def _refreshHistogramChannelOptions(self, colorSpace):
    labels = ["R", "G", "B"] if colorSpace != "HSV" else ["Hue", "Saturation", "Value"]
    self.histChannelSelector.blockSignals(True)
    self.histChannelSelector.clear()
    self.histChannelSelector.addItems(labels)
    self.histChannelSelector.setEnabled(True)
    self.histChannelSelector.blockSignals(False)

  def onViewModeChanged(self, button):
    """Handle view mode radio button changes"""
    try:
      if hasattr(self, '_lastColorData') and hasattr(self, '_lastColorSpace'):
        if self.viewChannelRadio.isChecked():
          # Switch to channel histogram view
          self._refreshCurrentHistogram()
        elif self.view2DRadio.isChecked():
          # Switch to 2D dimensionality reduction view
          if hasattr(self, '_last2DPlotChartNode') and self._last2DPlotChartNode:
            plotViewNode = self._ensureMainPlotViewNode()
            if plotViewNode:
              plotViewNode.SetPlotChartNodeID(self._last2DPlotChartNode.GetID())
              try:
                self.colorsEDALogInfo.appendPlainText("Switched to 2D dimensionality reduction view")
              except Exception:
                pass
            else:
              try:
                self.colorsEDALogInfo.appendPlainText("Warning: Could not access plot view")
              except Exception:
                pass
          else:
            try:
              self.colorsEDALogInfo.appendPlainText("Warning: No 2D plot available. Run analysis first.")
            except Exception:
              pass
    except Exception as e:
      msg = f"Failed to switch view mode: {e}"
      print(msg)
      try:
        self.colorsEDALogInfo.appendPlainText(msg)
      except Exception:
        pass

  def onHistChannelChanged(self, index):
    try:
      if hasattr(self, '_lastColorData') and hasattr(self, '_lastColorSpace') and self.viewChannelRadio.isChecked():
        # self._plotHistogramInMainView(self._lastColorData, self._lastColorSpace, int(index))
        self._refreshCurrentHistogram()
    except Exception as e:
      msg = f"Failed to update histogram: {e}"
      print(msg)
      try:
        self.colorsEDALogInfo.appendPlainText(msg)
      except Exception:
        pass

  def _createHueHistogramDensityChart(self, hue_degrees, bins=72, sat=None, val=None,
                                    color_by_bin_average=False, line_width=12):
    """
    Build a vertical-segment hue histogram (density). When color_by_bin_average=True,
    each segment is colored by the mean RGB of samples that fell into that bin;
    otherwise colored by the bin center hue.

    Args:
        hue_degrees: array of hue values in degrees [0,360)
        bins: number of bins
        sat: optional array of saturation values (0..100) aligned with hue_degrees
        val: optional array of value/brightness (0..100) aligned with hue_degrees
        color_by_bin_average: bool
    """
    self._clearHistogramNodes()

    hue_degrees = np.asarray(hue_degrees)
    hue_degrees = hue_degrees[np.isfinite(hue_degrees)]
    if hue_degrees.size == 0:
        raise ValueError("Empty hue data for histogram")

    # Histogram (density) + edges
    density, edges = np.histogram(hue_degrees, bins=bins, range=(0.0, 360.0), density=True)
    centers = (edges[:-1] + edges[1:]) / 2.0

    # Precompute bin indices per sample for coloring-by-average if needed
    bin_colors = None
    if color_by_bin_average and sat is not None and val is not None:
        sat = np.asarray(sat) / 100.0
        val = np.asarray(val) / 100.0
        # map each sample to a bin index
        bin_idx = np.clip(np.digitize(hue_degrees, edges, right=False) - 1, 0, bins - 1)

        # accumulate RGB sums & counts per bin (reconstruct RGB from HSV)
        sum_rgb = np.zeros((bins, 3), dtype=np.float64)
        counts  = np.zeros(bins, dtype=np.int64)

        # hue in [0,1] for colorsys
        h01 = (hue_degrees % 360.0) / 360.0
        for i in range(h01.size):
            r, g, b = colorsys.hsv_to_rgb(float(h01[i]), float(sat[i]), float(val[i]))
            k = bin_idx[i]
            sum_rgb[k, 0] += r; sum_rgb[k, 1] += g; sum_rgb[k, 2] += b
            counts[k] += 1

        # mean RGB per bin; fallback to pure bin hue color if bin empty
        bin_colors = np.zeros((bins, 3), dtype=np.float32)
        for k in range(bins):
            if counts[k] > 0:
                bin_colors[k] = (sum_rgb[k] / counts[k]).astype(np.float32)
            else:
                # fallback: color by the bin center hue (full sat/value)
                r, g, b = colorsys.hsv_to_rgb(centers[k] / 360.0, 1.0, 1.0)
                bin_colors[k] = (r, g, b)

    # Build chart and vertical-segment “bars”
    chartNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLPlotChartNode")
    chartNode.SetTitle("Histogram: Hue (density)")
    chartNode.SetXAxisTitle("Hue (degrees)")
    chartNode.SetYAxisTitle("Density")
    chartNode.SetLegendVisibility(False)
    self.histChartNodes.append(chartNode)

    for i, (c, d) in enumerate(zip(centers, density)):
        x = vtk.vtkFloatArray(); x.SetName("Hue"); x.SetNumberOfTuples(2); x.SetValue(0, float(c)); x.SetValue(1, float(c))
        y = vtk.vtkFloatArray(); y.SetName("Density"); y.SetNumberOfTuples(2); y.SetValue(0, 0.0); y.SetValue(1, float(d))

        t = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTableNode")
        t.AddColumn(x); t.AddColumn(y)
        self.histTableNodes.append(t)

        s = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLPlotSeriesNode")
        s.SetAndObserveTableNodeID(t.GetID())
        s.SetXColumnName("Hue"); s.SetYColumnName("Density")
        s.SetPlotType(slicer.vtkMRMLPlotSeriesNode.PlotTypeScatter)
        s.SetLineStyle(slicer.vtkMRMLPlotSeriesNode.LineStyleSolid)
        s.SetMarkerStyle(slicer.vtkMRMLPlotSeriesNode.MarkerStyleNone)
        s.SetLineWidth(line_width)  # thicker “bar”; tweak in UI if you want

        if color_by_bin_average and bin_colors is not None:
            r, g, b = bin_colors[i]
        else:
            r, g, b = colorsys.hsv_to_rgb(float(c) / 360.0, 1.0, 1.0)
        try:
            s.SetColor(float(r), float(g), float(b))
        except Exception:
            pass

        chartNode.AddAndObservePlotSeriesNodeID(s.GetID())
        self.histSeriesNodes.append(s)

    return chartNode



  def _plotHistogramInMainView(self, colorData, colorSpace, channelIndex):
    """Render selected channel histogram in the same plotting viewer pane."""
    if colorData is None or len(colorData) == 0:
      raise ValueError("No color data available for histogram")

    plotViewNode = self._ensureMainPlotViewNode()
    if plotViewNode is None:
      raise RuntimeError("No plot view available to show histogram")

    # Branch: special handling for Hue to color bars
    if colorSpace == "HSV" and channelIndex == 0:
      hue_deg = np.mod(np.degrees(np.arctan2(colorData[:, 1], colorData[:, 0])), 360.0)
      sat = colorData[:, 2]  # 0..100
      val = colorData[:, 3]  # 0..100

      # read UI
      sat_thresh = float(self.satCutoffSpin.value) if hasattr(self, "satCutoffSpin") else 10.0
      val_thresh = float(self.valueCutoffSpin.value) if hasattr(self, "valueCutoffSpin") else 10.0
      by_bin_avg = bool(self.colorByBinAvgCheck.isChecked()) if hasattr(self, "colorByBinAvgCheck") else False

      # mask low saturation before binning
      mask = np.isfinite(hue_deg) & np.isfinite(sat) & np.isfinite(val) & (sat >= sat_thresh) & (val >= val_thresh)
      hue_deg, sat, val = hue_deg[mask], sat[mask], val[mask]

      chartNode = self._createHueHistogramDensityChart(
          hue_degrees=hue_deg,
          bins=72,
          sat=sat,
          val=val,
          color_by_bin_average=by_bin_avg,
          line_width=12
      )
      plotViewNode.SetPlotChartNodeID(chartNode.GetID())
      return

    # Default density histograms for other channels
    if colorSpace == "HSV":
      if channelIndex == 1:
        data = colorData[:, 2]; rng = (0.0, 100.0); title = "Histogram: Saturation (density)"; xlabel = "Saturation (0-100)"
      else:
        data = colorData[:, 3]; rng = (0.0, 100.0); title = "Histogram: Value (density)"; xlabel = "Value (0-100)"
    else:
      data = colorData[:, channelIndex]
      rng = (0.0, 255.0); names = ["R", "G", "B"]; xlabel = f"{names[channelIndex]} (0-255)"; title = f"Histogram: {names[channelIndex]} (density)"

    chartNode = self._createDensityHistogramPlot(data, bins=50, value_range=rng, title=title, x_label=xlabel)
    plotViewNode.SetPlotChartNodeID(chartNode.GetID())

  ################################### Recolor Functions ###################################

  def onRecolorParameterChanged(self):
    """Enable/disable the apply button based on parameter selection"""
    atlasSelected = bool(self.recolorAtlasModelSelect.currentNode())
    textureSelected = bool(self.recolorTextureSelector.currentText and
                          self.recolorTextureSelector.currentIndex >= 0)
    self.applyRecolorButton.enabled = atlasSelected and textureSelected

  def onAverageFaceColorToggled(self, checked):
    """Handle toggling of the average face color checkbox"""
    # Enable/disable quantization controls based on averaging checkbox
    self.quantizeColorsCheckbox.setEnabled(checked)
    quantizationEnabled = checked and self.quantizeColorsCheckbox.isChecked()
    self.numColorClustersSpin.setEnabled(quantizationEnabled)
    self.useHighContrastPaletteCheckbox.setEnabled(quantizationEnabled)

    # If averaging is disabled, also disable quantization and high contrast palette
    if not checked:
      self.quantizeColorsCheckbox.setChecked(False)
      self.useHighContrastPaletteCheckbox.setChecked(False)

    # Update the apply button state
    self.onRecolorParameterChanged()

  def onQuantizeColorsToggled(self, checked):
    """Handle toggling of the quantize colors checkbox"""
    # Enable/disable the number of clusters spin box and high contrast palette checkbox
    quantizationEnabled = checked and self.averageFaceColorCheckbox.isChecked()
    self.numColorClustersSpin.setEnabled(quantizationEnabled)
    self.useHighContrastPaletteCheckbox.setEnabled(quantizationEnabled)

    # If quantization is disabled, also disable high contrast palette
    if not checked:
      self.useHighContrastPaletteCheckbox.setChecked(False)

    # Update the apply button state
    self.onRecolorParameterChanged()

  def onRecolorTexturesDirectoryChanged(self, directory):
    """Update texture selector when directory changes"""
    self.recolorTextureSelector.clear()
    self.recolorTextureSelector.enabled = False

    if not os.path.isdir(directory):
      self.onRecolorParameterChanged()
      return

    # Find texture files in the directory
    textureExtensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
    textureFiles = []

    try:
      for filename in sorted(os.listdir(directory)):
        if any(filename.lower().endswith(ext) for ext in textureExtensions):
          textureFiles.append(filename)
    except Exception as e:
      self.recolorLogInfo.appendPlainText(f"Error reading directory: {e}")
      return

    if textureFiles:
      self.recolorTextureSelector.addItems(textureFiles)
      self.recolorTextureSelector.enabled = True
      self.recolorLogInfo.appendPlainText(f"Found {len(textureFiles)} texture files")
    else:
      self.recolorLogInfo.appendPlainText("No texture files found in directory")

    self.onRecolorParameterChanged()

  def onApplyRecolorButton(self):
    """Apply the selected texture to the atlas model"""
    try:
      qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)
      self.recolorProgressBar.setVisible(True)
      self.recolorProgressBar.setValue(0)

      # Get parameters
      atlasModel = self.recolorAtlasModelSelect.currentNode()
      texturesDir = self.recolorTexturesDirectorySelector.currentPath
      selectedTexture = self.recolorTextureSelector.currentText
      useAverageFaceColors = self.averageFaceColorCheckbox.isChecked()
      useQuantization = self.quantizeColorsCheckbox.isChecked()
      numClusters = self.numColorClustersSpin.value
      useHighContrastPalette = self.useHighContrastPaletteCheckbox.isChecked()

      if not atlasModel:
        self.recolorLogInfo.appendPlainText("Error: No atlas model selected")
        return

      if not selectedTexture:
        self.recolorLogInfo.appendPlainText("Error: No texture selected")
        return

      texturePath = os.path.join(texturesDir, selectedTexture)
      if not os.path.exists(texturePath):
        self.recolorLogInfo.appendPlainText(f"Error: Texture file not found: {texturePath}")
        return

      self.recolorLogInfo.appendPlainText(f"Applying texture: {selectedTexture}")
      self.recolorLogInfo.appendPlainText(f"Average face colors: {'Yes' if useAverageFaceColors else 'No'}")
      if useAverageFaceColors and useQuantization:
        paletteType = "High contrast palette" if useHighContrastPalette else "Quantized colors"
        self.recolorLogInfo.appendPlainText(f"Color quantization: Yes ({numClusters} clusters, {paletteType})")
      else:
        self.recolorLogInfo.appendPlainText("Color quantization: No")

      logic = InterDeCALogic()

      if useAverageFaceColors:
        if useQuantization:
          # Apply quantized face colors
          self.recolorLogInfo.appendPlainText("Applying quantized face colors...")
          success = logic.applyQuantizedFaceColorsFromTexture(
            atlasModel, texturePath, numClusters, useHighContrastPalette,
            progressCallback=self.updateRecolorProgress,
            logCallback=self.logRecolorMessage
          )
        else:
          # Try the main face coloring method first
          self.recolorLogInfo.appendPlainText("Trying face-based coloring method...")
          success = logic.applyAverageFaceColorsFromTexture(
            atlasModel, texturePath,
            progressCallback=self.updateRecolorProgress,
            logCallback=self.logRecolorMessage
          )

          # If that doesn't work, try the alternative point-based method
          if not success:
            self.recolorLogInfo.appendPlainText("Face-based method failed, trying point-based method...")
            success = logic.applyAverageFaceColorsFromTextureAlternative(
              atlasModel, texturePath,
              progressCallback=self.updateRecolorProgress,
              logCallback=self.logRecolorMessage
            )
      else:
        # Apply texture directly
        success = logic.applyTextureToModel(atlasModel, texturePath)
        self.recolorLogInfo.appendPlainText("Texture applied successfully")
        success = True

      if success:
        self.recolorLogInfo.appendPlainText("Recoloring completed successfully")
      else:
        self.recolorLogInfo.appendPlainText("Recoloring failed - check log for details")

      self.recolorProgressBar.setVisible(False)
      qt.QApplication.restoreOverrideCursor()

    except Exception as e:
      self.recolorProgressBar.setVisible(False)
      qt.QApplication.restoreOverrideCursor()
      self.recolorLogInfo.appendPlainText(f"Error: {str(e)}")
      slicer.util.errorDisplay(f"Recolor failed: {str(e)}")
      import traceback
      traceback.print_exc()

  def updateRecolorProgress(self, value):
    """Update progress bar for recolor operations"""
    self.recolorProgressBar.setValue(int(value))
    slicer.app.processEvents()

  def logRecolorMessage(self, message):
    """Log message to the Recolor log"""
    self.recolorLogInfo.appendPlainText(message)
    slicer.app.processEvents()

  ################################### MultiRecolor Event Handlers ###################################

  def onMultiRecolorParameterChanged(self):
    """Enable/disable buttons based on MultiRecolor parameter selection"""
    atlasSelected = bool(self.multiRecolorAtlasModelSelect.currentNode())
    textureDirectorySelected = bool(self.multiRecolorTextureDirectorySelector.currentPath and
                                   os.path.isdir(self.multiRecolorTextureDirectorySelector.currentPath))

    # Enable cluster button if atlas and texture directory are selected
    self.clusterButton.enabled = atlasSelected and textureDirectorySelected

    # Update texture file list when directory changes
    if textureDirectorySelected:
      self.updateMultiRecolorTextureList()

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
    """Handle Step 1: Multi-texture clustering"""
    try:
      qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)
      self.clusteringProgressBar.setVisible(True)
      self.clusteringProgressBar.setValue(0)
      self.clusteringLogInfo.clear()

      atlasModel = self.multiRecolorAtlasModelSelect.currentNode()
      textureDir = self.multiRecolorTextureDirectorySelector.currentPath
      numClusters = self.multiRecolorNumClustersSpin.value

      if not atlasModel:
        self.clusteringLogInfo.append("Error: No atlas model selected")
        return

      if not textureDir or not os.path.exists(textureDir):
        self.clusteringLogInfo.append("Error: Invalid texture directory")
        return

      if not self.multiRecolorTextureFiles:
        self.clusteringLogInfo.append("Error: No texture files found")
        return

      self.clusteringLogInfo.append(f"Starting multi-texture clustering with {numClusters} clusters...")
      self.clusteringLogInfo.append(f"Processing {len(self.multiRecolorTextureFiles)} textures...")

      logic = InterDeCALogic()

      # Get cached face areas
      cachedFaceAreas = self.getCachedFaceAreas(atlasModel)
      if cachedFaceAreas is None:
        self.clusteringLogInfo.append("Error: Failed to get face areas")
        return

      # Run the multi-texture clustering
      result = logic.performMultiTextureClustering(
        atlasModel, textureDir, self.multiRecolorTextureFiles, numClusters,
        cachedFaceAreas,  # Pass cached face areas
        progressCallback=self.updateClusteringProgress,
        logCallback=self.logClusteringMessage
      )

      if result.get("success", False):
        self.multiRecolorClusterCenters = result["cluster_centers"]
        self.multiRecolorFaceAreas = cachedFaceAreas  # Use cached areas

        self.clusteringLogInfo.append("Multi-texture clustering completed successfully!")
        self.clusteringLogInfo.append(f"Created {len(self.multiRecolorClusterCenters)} color clusters")

        # Enable Step 2 controls
        self.individualTextureSelector.setEnabled(True)
        self.onIndividualTextureChanged()  # Update button states

        # Enable Step 3 controls
        self.compareTexturesButton.enabled = True

      else:
        self.clusteringLogInfo.append("Multi-texture clustering failed - check log for details")

      self.clusteringProgressBar.setVisible(False)
      qt.QApplication.restoreOverrideCursor()

    except Exception as e:
      self.clusteringProgressBar.setVisible(False)
      qt.QApplication.restoreOverrideCursor()
      self.clusteringLogInfo.append(f"Error: {str(e)}")
      slicer.util.errorDisplay(f"Multi-texture clustering failed: {str(e)}")
      import traceback
      traceback.print_exc()

  def onIndividualTextureChanged(self):
    """Handle texture selection change in Step 2"""
    textureSelected = bool(self.individualTextureSelector.currentText)
    clustersAvailable = self.multiRecolorClusterCenters is not None

    # Enable apply button and high contrast option if texture is selected and clusters are available
    self.applyIndividualTextureButton.enabled = textureSelected and clustersAvailable
    self.individualHighContrastCheckbox.setEnabled(textureSelected and clustersAvailable)

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
      useHighContrast = self.individualHighContrastCheckbox.isChecked()

      if not atlasModel:
        self.individualLogInfo.append("Error: No atlas model selected")
        return

      if not selectedTexture:
        self.individualLogInfo.append("Error: No texture selected")
        return

      if self.multiRecolorClusterCenters is None:
        self.individualLogInfo.append("Error: No cluster centers available. Run clustering first.")
        return

      texturePath = os.path.join(textureDir, selectedTexture)
      if not os.path.exists(texturePath):
        self.individualLogInfo.append(f"Error: Texture file not found: {texturePath}")
        return

      self.individualLogInfo.append(f"Applying texture: {selectedTexture}")
      paletteType = "High contrast palette" if useHighContrast else "Clustered colors"
      self.individualLogInfo.append(f"Using: {paletteType}")

      logic = InterDeCALogic()

      # Apply the texture with the clustered palette
      success = logic.applyIndividualTextureWithClusteredPalette(
        atlasModel, texturePath, self.multiRecolorClusterCenters,
        useHighContrast, self.multiRecolorFaceAreas,
        progressCallback=self.updateIndividualProgress,
        logCallback=self.logIndividualMessage
      )

      if success:
        self.individualLogInfo.append("Individual texture visualization completed successfully!")
      else:
        self.individualLogInfo.append("Individual texture visualization failed - check log for details")

      self.individualProgressBar.setVisible(False)
      qt.QApplication.restoreOverrideCursor()

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

      if self.multiRecolorClusterCenters is None:
        self.populationLogInfo.append("Error: No cluster centers available. Run clustering first.")
        return

      if not self.multiRecolorTextureFiles:
        self.populationLogInfo.append("Error: No texture files found")
        return

      # Get selected dimensionality reduction method
      dimReductionMethod = "PCA" if self.pcaRadioButton.isChecked() else "UMAP"

      self.populationLogInfo.append(f"Starting population analysis with {dimReductionMethod}...")
      self.populationLogInfo.append(f"Analyzing {len(self.multiRecolorTextureFiles)} textures...")

      logic = InterDeCALogic()

      # Perform population analysis
      result = logic.performPopulationAnalysis(
        atlasModel, textureDir, self.multiRecolorTextureFiles,
        self.multiRecolorClusterCenters, self.multiRecolorFaceAreas,
        dimReductionMethod,
        progressCallback=self.updatePopulationProgress,
        logCallback=self.logPopulationMessage
      )

      if result.get("success", False):
        self.populationLogInfo.append("Population analysis completed successfully!")
        self.populationLogInfo.append(f"Created {dimReductionMethod} plot with {len(self.multiRecolorTextureFiles)} texture points")
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


#
# DeCALogic
#

class InterDeCALogic(ScriptedLoadableModuleLogic):
  """This class should implement all the actual
    computation done by your module.  The interface
    should be such that other python code can import
    this class and make use of the functionality without
    requiring an instance of the Widget.
    Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """
  def runSubsetLandmarks(self, baseNode, lmDirectory, lmDirectorySubset):
    deletionIndex = []
    for i in range(baseNode.GetNumberOfControlPoints()):
      if not baseNode.GetNthControlPointSelected(i):
        deletionIndex.append(i)
    for lmFileName in os.listdir(lmDirectory):
      if(not lmFileName.startswith(".")):
        currentLMNode = slicer.util.loadMarkups(os.path.join(lmDirectory, lmFileName))
        for index in reversed(deletionIndex):
          currentLMNode.RemoveNthControlPoint(index)
      slicer.util.saveNode(currentLMNode, os.path.join(lmDirectorySubset, lmFileName))
      slicer.mrmlScene.RemoveNode(currentLMNode)

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

  # Use downsampleModel from decaLogic to avoid duplication
  def downsampleModel(self, model, spacingPercentage):
    if decaLogic:
      return decaLogic().downsampleModel(model, spacingPercentage)
    # Fallback implementation if decaLogic is not available
    points=model.GetPolyData()
    cleanFilter=vtk.vtkCleanPolyData()
    cleanFilter.SetToleranceIsAbsolute(False)
    cleanFilter.SetTolerance(spacingPercentage)
    cleanFilter.SetInputData(points)
    cleanFilter.Update()
    return cleanFilter.GetOutput()

  # Use addIndexArray from decaLogic to avoid duplication
  def addIndexArray(self, mesh, arrayName):
    if decaLogic:
      return decaLogic().addIndexArray(mesh, arrayName)
    # Fallback implementation if decaLogic is not available
    indexArray = vtk.vtkIntArray()
    indexArray.SetNumberOfComponents(1)
    indexArray.SetName(arrayName)
    for i in range(mesh.GetPolyData().GetNumberOfPoints()):
      indexArray.InsertNextValue(i)
    mesh.GetPolyData().GetPointData().AddArray(indexArray)

  # Use computeNormals from decaLogic to avoid duplication
  def computeNormals(self, inputModel):
    if decaLogic:
      return decaLogic().computeNormals(inputModel)
    # Fallback implementation if decaLogic is not available
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
        currentMeshNode = slicer.util.loadModel(meshFilePath)
        subjectID = os.path.splitext(meshFileName)[0]
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
          slicer.util.saveNode(currentMeshNode, outputMeshPath)
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
    if optionErrorOutput:
      self.errorCheckPath = os.path.join(outputDirectory, "errorChecking")
      if not os.path.exists(self.errorCheckPath):
        os.mkdir(self.errorCheckPath)

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

    #  Save resampled models (VTK/PLY) and OBJ copies that reuse atlas UV (for Blender bake)
    resampledModelPath = os.path.join(outputDirectory, "resampledModels")
    if os.path.exists(resampledModelPath):
      tempModelNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", "tempResampledModel")
      outOBJdir = os.path.join(outputDirectory, "resampledOBJ_withUV")
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

    self.addMagnitudeFeature(denseCorrespondenceGroup, self.modelNames, baseMesh)
    outputModelName = 'decaResultModel.vtp'
    outputModelPath = os.path.join(outputDirectory, outputModelName)
    slicer.util.saveNode(baseNode, outputModelPath)

  def runDCAlignSymmetric(self, baseMeshPath, baseLMPath, meshDir, landmarkDir, mirrorMeshDir, mirrorLandmarkDir, outputDir, optionErrorOutput):
    if optionErrorOutput:
      self.errorCheckPath = os.path.join(outputDir, "errorChecking")
      if not os.path.exists(self.errorCheckPath):
        os.mkdir(self.errorCheckPath)
    baseNode = slicer.util.loadModel(baseMeshPath)
    baseMesh = baseNode.GetPolyData()
    baseLandmarks=self.fiducialNodeToPolyData(baseLMPath).GetPoints()
    modelExt=['ply','stl','vtp', 'obj']
    self.modelNames, models = self.importMeshes(meshDir, modelExt)
    landmarkNames, landmarks = self.importLandmarks(landmarkDir)
    modelMirrorNames, mirrorModels = self.importMeshes(mirrorMeshDir, modelExt)
    mirrorLandmarkNames, mirrorLandmarks = self.importLandmarks(mirrorLandmarkDir)
    denseCorrespondenceGroup = self.denseCorrespondenceBaseMesh(landmarks, models, baseMesh, baseLandmarks)
    denseCorrespondenceGroupMirror = self.denseCorrespondenceBaseMesh(mirrorLandmarks, mirrorModels, baseMesh, baseLandmarks)
    self.addMagnitudeFeatureSymmetry(denseCorrespondenceGroup, denseCorrespondenceGroupMirror, self.modelNames, baseMesh)
    # save results to output directory
    outputModelName = 'decaSymmetryResultModel.vtp'
    outputModelPath = os.path.join(outputDir, outputModelName)
    slicer.util.saveNode(baseNode, outputModelPath)

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

  # Use getLandmarkFileByID from decaLogic to avoid duplication
  def getLandmarkFileByID(self, directory, subjectID):
    if decaLogic:
      try:
        return decaLogic().getLandmarkFileByID(directory, subjectID)
      except Exception as e:
        print(f"Error using decaLogic.getLandmarkFileByID: {e}")
        # Fall back to local implementation
    # Fallback implementation if decaLogic is not available
    fileList = os.listdir(directory)
    for fileName in fileList:
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
    for fileName in fileList:
      fileNameBase = Path(fileName).stem
      if str(subjectID) == str(fileNameBase):
        filePath = os.path.join(directory, fileName)
        try:
          currentNode = self._load_model_with_cs(filePath, 'RAS')
          return currentNode
        except Exception as e:
          print(f"Error loading model from {filePath}: {e}")
          return None
    print(f"No model found for subject ID '{subjectID}' in {directory}")
    return None

  def runAlign(self, baseMeshNode, baseLMNode, meshDirectory, lmDirectory, ouputMeshDirectory, outputLMDirectory, removeScaleOption, slmDirectory=False, outputSLMDirectory=False):
    semilandmarkOption = bool(slmDirectory and outputSLMDirectory)
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
        subjectID = os.path.splitext(meshFileName)[0]
        currentLMNode = self.getLandmarkFileByID(lmDirectory, subjectID)
        if currentLMNode :
          try:
            currentMeshNode = slicer.util.loadModel(meshFilePath)
          except:
            slicer.mrmlScene.RemoveNode(currentLMNode)
            continue
          if currentLMNode.GetNumberOfControlPoints() != baseLMNode.GetNumberOfControlPoints():
            raise ValueError(f"Landmark points mismatch: subject has {currentLMNode.GetNumberOfControlPoints()} points, "
              f"atlas has {baseLMNode.GetNumberOfControlPoints()} points")
          # set up transform between base lms and current lms
          sourcePoints = vtk.vtkPoints()
          for i in range(currentLMNode.GetNumberOfControlPoints()):
            point = currentLMNode.GetNthControlPointPosition(i)
            sourcePoints.InsertNextPoint(point)
          transform = vtk.vtkLandmarkTransform()
          transform.SetSourceLandmarks(sourcePoints)
          transform.SetTargetLandmarks(targetPoints)
          if not removeScaleOption:
            transform.SetModeToRigidBody()
          else:
            transform.SetModeToSimilarity()

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

  # Use distanceMatrix from decaLogic to avoid duplication
  def distanceMatrix(self, a):
    if decaLogic:
      return decaLogic().distanceMatrix(a)
    # Fallback implementation if decaLogic is not available
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

  # Use numpyToFiducialNode from decaLogic to avoid duplication
  def numpyToFiducialNode(self, numpyArray, nodeName):
    if decaLogic:
      return decaLogic().numpyToFiducialNode(numpyArray, nodeName)
    # Fallback implementation if decaLogic is not available
    fiducialNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsFiducialNode',nodeName)
    for index in range(len(numpyArray)):
      fiducialNode.AddControlPoint(numpyArray[index], str(index))
    return fiducialNode

  # Use computeAverageLM from decaLogic to avoid duplication
  def computeAverageLM(self, fiducialGroup):
    if decaLogic:
      return decaLogic().computeAverageLM(fiducialGroup)
    # Fallback implementation if decaLogic is not available
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

  # Use fiducialNodeToPolyData from decaLogic to avoid duplication
  def fiducialNodeToPolyData(self, nodeLocation, loadOption=True):
    if decaLogic:
      return decaLogic().fiducialNodeToPolyData(nodeLocation, loadOption)
    # Fallback implementation if decaLogic is not available
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


  def importMeshes(self, topDir, extensions, restrict_to=None):
    # choose exactly one mesh per subject, preferring OBJ over PLY/STL/VTP/VTK
    priority = {'.obj':0, '.ply':1, '.stl':2, '.vtp':3, '.vtk':4}
    pick = {}  # base -> (rank, fullpath)

    for f in os.listdir(topDir):
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
      modelNode = self._load_model_with_cs(inputFilePath, 'RAS')
      modelGroup.AddInputData(modelNode.GetPolyData())
      slicer.mrmlScene.RemoveNode(modelNode)
    modelGroup.Update()
    return names, modelGroup.GetOutput()

  # Use procrustesImposition from decaLogic to avoid duplication
  def procrustesImposition(self, originalLandmarks, sizeOption):
    if decaLogic:
      return decaLogic().procrustesImposition(originalLandmarks, sizeOption)
    # Fallback implementation if decaLogic is not available
    procrustesFilter = vtk.vtkProcrustesAlignmentFilter()
    if(sizeOption):
      procrustesFilter.GetLandmarkTransform().SetModeToRigidBody()

    procrustesFilter.SetInputData(originalLandmarks)
    procrustesFilter.Update()
    meanShape = procrustesFilter.GetMeanPoints()
    return [meanShape, procrustesFilter.GetOutput()]

  # Use getClosestToMeanIndex from decaLogic to avoid duplication
  def getClosestToMeanIndex(self, meanShape, alignedPoints):
    if decaLogic:
      return decaLogic().getClosestToMeanIndex(meanShape, alignedPoints)
    # Fallback implementation if decaLogic is not available
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

  # Use getClosestToMeanPath from decaLogic to avoid duplication
  def getClosestToMeanPath(self, landmarkDirectory):
    if decaLogic:
      try:
        return decaLogic().getClosestToMeanPath(landmarkDirectory)
      except Exception as e:
        print(f"Error using decaLogic.getClosestToMeanPath: {e}")
        # Fall back to local implementation
    # Fallback implementation if decaLogic is not available
    lmNames, landmarks = self.importLandmarks(landmarkDirectory)
    if not lmNames:
      print(f"No landmarks found in {landmarkDirectory}")
      return None
    meanShape, alignedLandmarks = self.procrustesImposition(landmarks, False)
    closestToMeanIndex = self.getClosestToMeanIndex(meanShape, alignedLandmarks)
    return lmNames[closestToMeanIndex]

  def denseCorrespondence(self, originalLandmarks, originalMeshes, writeErrorOption=False):
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

  def denseCorrespondenceCPD(self, originalLandmarks, originalMeshes, baseMesh, baseLandmarks, writeErrorOption=False):
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
      # write ouput
      if writeErrorOption:
        plyWriterSubject = vtk.vtkPLYWriter()
        plyWriterSubject.SetFileName("/Users/sararolfe/Dropbox/SlicerWorkspace/SMwSML/Data/UBC/DECAOutCPD/" + str(i) + ".ply")
        plyWriterSubject.SetInputData(correspondingMesh)
        plyWriterSubject.Write()

        plyWriterBase = vtk.vtkPLYWriter()
        plyWriterBase.SetFileName("/Users/sararolfe/Dropbox/SlicerWorkspace/SMwSML/Data/UBC/DECAOutCPD/base.ply")
        plyWriterBase.SetInputData(baseMesh)
        plyWriterBase.Write()

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

  # Use convertPointsToVTK from decaLogic to avoid duplication
  def convertPointsToVTK(self, points):
    if decaLogic:
      return decaLogic().convertPointsToVTK(points)
    # Fallback implementation if decaLogic is not available
    array_vtk = vtk_np.numpy_to_vtk(points, deep=True, array_type=vtk.VTK_FLOAT)
    points_vtk = vtk.vtkPoints()
    points_vtk.SetData(array_vtk)
    polydata_vtk = vtk.vtkPolyData()
    polydata_vtk.SetPoints(points_vtk)
    return polydata_vtk

  # Use computeAverageModelFromGroup from decaLogic to avoid duplication
  def computeAverageModelFromGroup(self, denseCorrespondenceGroup, baseIndex):
    if decaLogic:
      return decaLogic().computeAverageModelFromGroup(denseCorrespondenceGroup, baseIndex)
    # Fallback implementation if decaLogic is not available
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

  # Use addMagnitudeFeature from decaLogic to avoid duplication
  def addMagnitudeFeature(self, denseCorrespondenceGroup, modelNameArray, model):
    if decaLogic:
      return decaLogic().addMagnitudeFeature(denseCorrespondenceGroup, modelNameArray, model)
    # Fallback implementation if decaLogic is not available
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

  # Use addMagnitudeFeatureSymmetry from decaLogic to avoid duplication
  def addMagnitudeFeatureSymmetry(self, denseCorrespondenceGroup, denseCorrespondenceGroupMirror, modelNameArray, model):
    if decaLogic:
      return decaLogic().addMagnitudeFeatureSymmetry(denseCorrespondenceGroup, denseCorrespondenceGroupMirror, modelNameArray, model)
    # Fallback implementation if decaLogic is not available
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
                for f in os.listdir(texturesDir) if f.lower().endswith('.png')}

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

  def _median_landmark_to_surface_dist(self, modelNode, lmNode):
    locator = vtk.vtkStaticCellLocator()
    locator.SetDataSet(modelNode.GetPolyData())
    locator.BuildLocator()
    dists = []
    for i in range(lmNode.GetNumberOfControlPoints()):
        p = [0.0,0.0,0.0]
        lmNode.GetNthControlPointPosition(i, p)
        cp = [0.0,0.0,0.0]
        cid = vtk.mutable(0); sid = vtk.mutable(0); d2 = vtk.mutable(0.0)
        locator.FindClosestPoint(p, cp, cid, sid, d2)
        dists.append(d2.get()**0.5)
    return np.median(dists) if dists else float('inf')
  
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
        os.path.expanduser('~/.local/bin/blender')
      ]
      # Check for snap installations
      snap_paths = glob.glob('/snap/blender/*/blender')
      common_paths.extend(snap_paths)

    # Test each common path
    for path in common_paths:
      if os.path.isfile(path) and os.access(path, os.X_OK):
        print(f"Found Blender at: {path}")
        return path

    print("Blender executable not found in common locations")
    return None

  def installBlender(self, log_callback=None):
    """
    Automatically download and install Blender.
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
      log("macOS installation not supported in automatic mode. Please install Blender manually from:")
      log("https://www.blender.org/download/")
      return None

    elif system == 'linux':
      if 'amd64' in architecture or 'x86_64' in architecture:
        filename = f"blender-{blender_version}-linux-x64.tar.xz"
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

    # Create installation directory
    install_dir = os.path.join(os.path.expanduser('~'), '.slicer-blender')
    os.makedirs(install_dir, exist_ok=True)

    # Check if already installed
    expected_blender_dir = os.path.join(install_dir, f"blender-{blender_version}-{system}-x64")
    if system == 'windows' and 'arm' in architecture:
      expected_blender_dir = os.path.join(install_dir, f"blender-{blender_version}-{system}-arm64")

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
          if f.lower().endswith('.png') and not f.lower().startswith('average_texture'):
            textureFiles.append(os.path.join(texturesDir, f))

      if not textureFiles:
        if logCallback:
          logCallback(f"Error: No PNG files found in {texturesDir}")
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

      # Plot results in 2D viewer
      # For HSV with filtering, we need to adjust the specimen information
      if colorSpace == "HSV" and dimRedData.shape[0] != colorData.shape[0]:
        # Calculate how many faces per specimen passed the filter
        filteredNFaces = dimRedData.shape[0] // nSpecimens if nSpecimens > 0 else 0
        plotResult = self._plotColorsEDAResults(reducedData, specimenNames, filteredNFaces, colorSpace, dimRedAlgo, dimRedData, enhanceColors)
      else:
        plotResult = self._plotColorsEDAResults(reducedData, specimenNames, nFaces, colorSpace, dimRedAlgo, dimRedData, enhanceColors)

      if progressCallback:
        progressCallback(100)

      # Return result details for downstream UI updates (e.g., histograms)
      # Always return the FULL color data for histogram use, not the filtered data
      if plotResult and isinstance(plotResult, dict):
        return {
          "success": True,
          "colorData": colorData,  # Full dataset for histograms
          "colorSpace": colorSpace,
          "chartNode": plotResult.get("chartNode"),
          "satCutoff": satCutoff,
          "valueCutoff": valueCutoff
        }
      else:
        return {
          "success": bool(plotResult),
          "colorData": colorData,  # Full dataset for histograms
          "colorSpace": colorSpace,
          "satCutoff": satCutoff,
          "valueCutoff": valueCutoff
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
          if f.lower().endswith('.png') and not f.lower().startswith('average_texture'):
            textureFiles.append(os.path.join(texturesDir, f))

      if not textureFiles:
        if logCallback:
          logCallback(f"Error: No PNG files found in {texturesDir}")
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

      # Get face connectivity
      polys = polyData.GetPolys()
      nFaces = polys.GetNumberOfCells()

      # Get texture image dimensions
      height, width = textureImage.shape[:2]

      faceColors = []

      # Process each face individually using VTK's cell iterator
      for faceIdx in range(nFaces):
        # Get the cell (face) points
        cell = polyData.GetCell(faceIdx)
        nPoints = cell.GetNumberOfPoints()

        # Get vertex indices for this face
        vertexIndices = []
        for ptIdx in range(nPoints):
          vertexIndices.append(cell.GetPointId(ptIdx))

        # Get texture coordinates for these vertices
        faceTexCoords = tcoords_np[vertexIndices]

        # Convert texture coordinates to pixel coordinates
        # Flip V coordinate (1 - v) to handle texture inversion
        faceTexCoords_flipped = faceTexCoords.copy()
        faceTexCoords_flipped[:, 1] = 1.0 - faceTexCoords_flipped[:, 1]

        pixelCoords = np.clip(faceTexCoords_flipped, 0, 1) * [width - 1, height - 1]
        pixelCoords = pixelCoords.astype(int)

        # Sample colors at these pixel locations
        facePixelColors = textureImage[pixelCoords[:, 1], pixelCoords[:, 0], :3]

        # Calculate average color for this face
        avgColor = np.mean(facePixelColors, axis=0)

        # Convert color space if needed
        if colorSpace == "HSV":
          # Convert RGB to HSV
          rgb_normalized = avgColor / 255.0
          hsv = colorsys.rgb_to_hsv(rgb_normalized[0], rgb_normalized[1], rgb_normalized[2])
          # Convert hue to 2D vector (cos, sin) to handle circular nature
          hue_radians = hsv[0] * 2 * np.pi  # Convert to radians
          hue_cos = np.cos(hue_radians)
          hue_sin = np.sin(hue_radians)
          # Create 4D vector: [hue_cos, hue_sin, saturation, value]
          avgColor = np.array([hue_cos, hue_sin, hsv[1] * 100, hsv[2] * 100])

        faceColors.append(avgColor)

      return np.array(faceColors)

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

      faceColors = []

      # Process only the specified face indices
      for faceIdx in faceIndices:
        # Get the cell (face) points
        cell = polyData.GetCell(int(faceIdx))
        nPoints = cell.GetNumberOfPoints()

        # Get vertex indices for this face
        vertexIndices = []
        for ptIdx in range(nPoints):
          vertexIndices.append(cell.GetPointId(ptIdx))

        # Get texture coordinates for these vertices
        faceTexCoords = tcoords_np[vertexIndices]

        # Convert texture coordinates to pixel coordinates
        # Flip V coordinate (1 - v) to handle texture inversion
        faceTexCoords_flipped = faceTexCoords.copy()
        faceTexCoords_flipped[:, 1] = 1.0 - faceTexCoords_flipped[:, 1]

        pixelCoords = np.clip(faceTexCoords_flipped, 0, 1) * [width - 1, height - 1]
        pixelCoords = pixelCoords.astype(int)

        # Sample colors at these pixel locations
        facePixelColors = textureImage[pixelCoords[:, 1], pixelCoords[:, 0], :3]

        # Calculate average color for this face
        avgColor = np.mean(facePixelColors, axis=0)

        # Convert color space if needed
        if colorSpace == "HSV":
          # Convert RGB to HSV
          rgb_normalized = avgColor / 255.0
          hsv = colorsys.rgb_to_hsv(rgb_normalized[0], rgb_normalized[1], rgb_normalized[2])
          # Convert hue to 2D vector (cos, sin) to handle circular nature
          hue_radians = hsv[0] * 2 * np.pi  # Convert to radians
          hue_cos = np.cos(hue_radians)
          hue_sin = np.sin(hue_radians)
          # Create 4D vector: [hue_cos, hue_sin, saturation, value]
          avgColor = np.array([hue_cos, hue_sin, hsv[1] * 100, hsv[2] * 100])

        faceColors.append(avgColor)

      return np.array(faceColors)

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

  def _plotColorsEDAResults(self, reducedData, specimenNames, nFaces, colorSpace, algorithm, originalColorData=None, enhanceColors=False):
    """
    Plot the dimensionality reduction results in a 2D viewer

    Args:
        reducedData: numpy array of shape (N_samples, 2)
        specimenNames: list of specimen names
        nFaces: number of faces per specimen
        colorSpace: "RGB" or "HSV"
        algorithm: "PCA", "ICA", or "UMAP"
        originalColorData: numpy array of original color data for hue-based coloring (optional)
        enhanceColors: whether to enhance colors for visibility

    Returns:
        dict: {"success": bool, "chartNode": node} if successful
    """
    try:
      # Create enhanced plot with color information when HSV data is available
      if colorSpace == "HSV" and originalColorData is not None and originalColorData.shape[1] >= 4:
        return self._createColoredScatterPlot(reducedData, originalColorData, algorithm, colorSpace, enhanceColors)

      # Standard single-series plot for RGB or when no color data available
      return self._createStandardPlot(reducedData, algorithm, colorSpace)

      # Create plot chart
      plotChartNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLPlotChartNode")
      plotChartNode.SetName(f"Colors_EDA_Chart_{algorithm}_{colorSpace}")
      plotChartNode.AddAndObservePlotSeriesNodeID(plotSeriesNode.GetID())

      # Set title based on whether hue coloring is used
      if colorArray is not None:
        plotChartNode.SetTitle(f"Color Analysis: {algorithm} on {colorSpace} Face Colors (Hue-Colored)")
      else:
        plotChartNode.SetTitle(f"Color Analysis: {algorithm} on {colorSpace} Face Colors")

      plotChartNode.SetXAxisTitle(f"{algorithm} Component 1")
      plotChartNode.SetYAxisTitle(f"{algorithm} Component 2")

      # Show in plot view
      layoutManager = slicer.app.layoutManager()
      plotWidget = layoutManager.plotWidget(0)
      plotViewNode = plotWidget.mrmlPlotViewNode()
      plotViewNode.SetPlotChartNodeID(plotChartNode.GetID())

      # Return both success status and chart node for UI to store
      return {"success": True, "chartNode": plotChartNode}

    except Exception as e:
      print(f"Error plotting results: {e}")
      return {"success": False, "chartNode": None}

  def _createStandardPlot(self, reducedData, algorithm, colorSpace):
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
      plotChartNode.SetTitle(f"Color Analysis: {algorithm} on {colorSpace} Face Colors")
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

  def _createColoredScatterPlot(self, reducedData, originalColorData, algorithm, colorSpace, enhanceColors=False):
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
        plotChartNode.SetTitle(f"Color Analysis: {algorithm} on {colorSpace} Face Colors"
                               + (" (Enhanced Colors)" if enhanceColors else " (Actual Colors)"))
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
        print(f"Error creating binned colored scatter: {e}")
        import traceback; traceback.print_exc()
        return {"success": False, "chartNode": None}

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

      # Plot results in 2D viewer
      # For HSV with filtering, we need to adjust the specimen information
      if colorSpace == "HSV" and dimRedData.shape[0] != colorDataFlat.shape[0]:
        # Calculate how many faces per specimen passed the filter
        filteredNFaces = dimRedData.shape[0] // nSpecimens if nSpecimens > 0 else 0
        plotResult = self._plotColorsEDAResults(reducedData, specimenNames, filteredNFaces, colorSpace, dimRedAlgo, dimRedData, enhanceColors)
      else:
        plotResult = self._plotColorsEDAResults(reducedData, specimenNames, nSampledFaces, colorSpace, dimRedAlgo, dimRedData, enhanceColors)

      if progressCallback:
        progressCallback(100)

      # Return result details for downstream UI updates (e.g., histograms)
      # Always return the FULL color data for histogram use, not the filtered data
      if plotResult and isinstance(plotResult, dict):
        return {
          "success": True,
          "colorData": colorDataFlat,  # Full dataset for histograms
          "colorSpace": colorSpace,
          "chartNode": plotResult.get("chartNode"),
          "satCutoff": satCutoff,
          "valueCutoff": valueCutoff
        }
      else:
        return {
          "success": bool(plotResult),
          "colorData": colorDataFlat,  # Full dataset for histograms
          "colorSpace": colorSpace,
          "satCutoff": satCutoff,
          "valueCutoff": valueCutoff
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

  def performMultiTextureClustering(self, modelNode, textureDir, textureFiles, numClusters, faceAreas=None, progressCallback=None, logCallback=None):
    """
    Perform multi-texture clustering using MiniBatchKMeans on face average colors

    Args:
        modelNode: VTK model node to analyze
        textureDir: Directory containing texture files
        textureFiles: List of texture filenames
        numClusters: Number of color clusters
        faceAreas: Pre-computed face areas (optional, will calculate if None)
        progressCallback: Function to call with progress updates (0-100)
        logCallback: Function to call with log messages

    Returns:
        dict with 'success', 'cluster_centers', 'face_areas'
    """
    try:
      if logCallback:
        logCallback(f"Starting multi-texture clustering with {numClusters} clusters...")

      if progressCallback:
        progressCallback(5)

      # Check if required libraries are available
      if not SKLEARN_AVAILABLE:
        if logCallback:
          logCallback("Error: sklearn not available for MiniBatchKMeans clustering")
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
        progressCallback(10)

      # Collect all face colors from all textures
      allFaceColors = []

      if logCallback:
        logCallback(f"Processing {len(textureFiles)} texture files...")

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
        if faceColors is not None:
          allFaceColors.append(faceColors)
        else:
          if logCallback:
            logCallback(f"Warning: Failed to calculate face colors for {textureFile}")

        # Update progress
        progress = 10 + (i + 1) * 60 / len(textureFiles)
        if progressCallback:
          progressCallback(progress)

      if not allFaceColors:
        if logCallback:
          logCallback("Error: No valid face colors extracted from any texture")
        return {"success": False}

      if logCallback:
        logCallback(f"Successfully processed {len(allFaceColors)} textures")
        logCallback("Combining all face colors for clustering...")

      # Combine all face colors into a single array
      combinedColors = np.vstack(allFaceColors)

      if logCallback:
        logCallback(f"Total face colors for clustering: {len(combinedColors)}")

      if progressCallback:
        progressCallback(75)

      # Convert to Lab color space for clustering
      if logCallback:
        logCallback("Converting colors to CIE Lab space...")

      labColors = self.rgb_to_lab(combinedColors)

      # Perform MiniBatchKMeans clustering
      if logCallback:
        logCallback(f"Performing MiniBatchKMeans clustering with {numClusters} clusters...")

      kmeans = MiniBatchKMeans(n_clusters=numClusters, random_state=42, batch_size=1000)
      kmeans.fit(labColors)
      clusterCentersLab = kmeans.cluster_centers_

      if progressCallback:
        progressCallback(90)

      # Convert cluster centers back to RGB
      if logCallback:
        logCallback("Converting cluster centers back to RGB...")

      clusterCentersRgb = self.lab_to_rgb(clusterCentersLab)

      if progressCallback:
        progressCallback(100)

      if logCallback:
        logCallback(f"Multi-texture clustering completed successfully!")
        logCallback(f"Created {numClusters} color clusters from {len(allFaceColors)} textures")

      return {
        "success": True,
        "cluster_centers": clusterCentersRgb.astype(np.uint8),
        "face_areas": faceAreas,
        "num_textures_processed": len(allFaceColors)
      }

    except Exception as e:
      if logCallback:
        logCallback(f"Error in multi-texture clustering: {str(e)}")
      import traceback
      traceback.print_exc()
      return {"success": False}

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
      faceAreas = np.zeros(numFaces)

      for faceId in range(numFaces):
        cell = polyData.GetCell(faceId)
        if cell.GetNumberOfPoints() >= 3:
          # Get the points of the face
          points = []
          for i in range(cell.GetNumberOfPoints()):
            pointId = cell.GetPointId(i)
            point = polyData.GetPoint(pointId)
            points.append(point)

          # Calculate area using cross product for triangular faces
          if len(points) >= 3:
            # For triangular faces
            p0, p1, p2 = np.array(points[0]), np.array(points[1]), np.array(points[2])
            v1 = p1 - p0
            v2 = p2 - p0
            area = 0.5 * np.linalg.norm(np.cross(v1, v2))

            # For quad faces, add the second triangle
            if len(points) == 4:
              p3 = np.array(points[3])
              v3 = p3 - p0
              area += 0.5 * np.linalg.norm(np.cross(v2, v3))

            faceAreas[faceId] = area

      return faceAreas

    except Exception as e:
      print(f"Error calculating face areas: {e}")
      return None

  def applyIndividualTextureWithClusteredPalette(self, modelNode, texturePath, clusterCenters, useHighContrast=False, faceAreas=None, progressCallback=None, logCallback=None):
    """
    Apply individual texture with pre-computed clustered palette

    Args:
        modelNode: VTK model node to apply colors to
        texturePath: Path to the texture image file
        clusterCenters: Pre-computed cluster centers (RGB colors)
        useHighContrast: If True, use high contrast palette instead of cluster centers
        faceAreas: Pre-computed face areas (optional, for caching)
        progressCallback: Function to call with progress updates (0-100)
        logCallback: Function to call with log messages

    Returns:
        bool: True if successful, False otherwise
    """
    try:
      if logCallback:
        logCallback(f"Loading texture: {os.path.basename(texturePath)}")

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

      # Convert face colors to Lab space for clustering assignment
      if logCallback:
        logCallback("Converting colors to CIE Lab space...")

      faceColorsLab = self.rgb_to_lab(faceColors)

      if progressCallback:
        progressCallback(50)

      # Determine which palette to use
      if useHighContrast:
        if logCallback:
          logCallback("Using high contrast palette...")
        paletteColors = self.generate_high_contrast_palette(len(clusterCenters))
      else:
        if logCallback:
          logCallback("Using clustered palette...")
        paletteColors = clusterCenters

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

  def performPopulationAnalysis(self, modelNode, textureDir, textureFiles, clusterCenters, faceAreas, dimReductionMethod="PCA", progressCallback=None, logCallback=None):
    """
    Perform population analysis by creating area-weighted color vectors and dimensionality reduction

    Args:
        modelNode: VTK model node to analyze
        textureDir: Directory containing texture files
        textureFiles: List of texture filenames
        clusterCenters: Pre-computed cluster centers (RGB colors)
        faceAreas: Pre-computed face areas
        dimReductionMethod: "PCA" or "UMAP"
        progressCallback: Function to call with progress updates (0-100)
        logCallback: Function to call with log messages

    Returns:
        dict with 'success' and plot information
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

      # Get model polydata
      polyData = modelNode.GetPolyData()
      if not polyData:
        if logCallback:
          logCallback("Error: No polydata in model")
        return {"success": False}

      # Convert cluster centers to Lab space for distance calculations
      clusterCentersLab = self.rgb_to_lab(clusterCenters)
      numClusters = len(clusterCenters)

      if logCallback:
        logCallback(f"Creating area-weighted color vectors for {len(textureFiles)} textures...")

      # Create area-weighted color vectors for each texture
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

        # Convert face colors to Lab space
        faceColorsLab = self.rgb_to_lab(faceColors)

        # Create area-weighted color vector
        colorVector = np.zeros(numClusters)

        # Vectorized distance calculation for better performance
        numFaces = len(faceColorsLab)

        # Reshape for broadcasting: faces (N,1,3) and clusters (1,K,3)
        faceColorsExpanded = faceColorsLab[:, np.newaxis, :]  # (N, 1, 3)
        clusterColorsExpanded = clusterCentersLab[np.newaxis, :, :]  # (1, K, 3)

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
        logCallback(f"Created {len(textureVectors)} area-weighted color vectors")
        logCallback(f"Performing {dimReductionMethod} dimensionality reduction...")

      if progressCallback:
        progressCallback(75)

      # Convert to numpy array
      textureVectors = np.array(textureVectors)

      # Perform dimensionality reduction
      if dimReductionMethod == "PCA":
        reducer = PCA(n_components=2, random_state=42)
        reducedData = reducer.fit_transform(textureVectors)

        if logCallback:
          explained_variance = reducer.explained_variance_ratio_
          logCallback(f"PCA explained variance: PC1={explained_variance[0]:.3f}, PC2={explained_variance[1]:.3f}")

      elif dimReductionMethod == "UMAP":
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=min(15, len(textureVectors)-1))
        reducedData = reducer.fit_transform(textureVectors)

      if progressCallback:
        progressCallback(90)

      # Create plot
      if logCallback:
        logCallback("Creating population analysis plot...")

      plotResult = self._createPopulationPlot(reducedData, textureNames, dimReductionMethod)

      if progressCallback:
        progressCallback(100)

      if logCallback:
        logCallback(f"Population analysis completed successfully!")
        logCallback(f"Plotted {len(textureNames)} textures in 2D {dimReductionMethod} space")

      return {
        "success": True,
        "reduced_data": reducedData,
        "texture_names": textureNames,
        "method": dimReductionMethod,
        "plot_result": plotResult
      }

    except Exception as e:
      if logCallback:
        logCallback(f"Error in population analysis: {str(e)}")
      import traceback
      traceback.print_exc()
      return {"success": False}

  def _createPopulationPlot(self, reducedData, textureNames, method):
    """
    Create a population analysis plot using Slicer's plotting functionality
    with equal X/Y numeric ranges.
    """
    try:
      # --- series ---
      plotSeriesNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLPlotSeriesNode")
      plotSeriesNode.SetName(f"MultiRecolor_Population_{method}")

      xArray = vtk.vtkFloatArray(); xArray.SetName(f"{method} Component 1")
      yArray = vtk.vtkFloatArray(); yArray.SetName(f"{method} Component 2")
      xArray.SetNumberOfTuples(len(reducedData))
      yArray.SetNumberOfTuples(len(reducedData))
      labelsArray = vtk.vtkStringArray(); labelsArray.SetName("Texture Names")
      labelsArray.SetNumberOfTuples(len(reducedData))

      for i, (point, name) in enumerate(zip(reducedData, textureNames)):
        xArray.SetValue(i, float(point[0]))
        yArray.SetValue(i, float(point[1]))
        labelsArray.SetValue(i, name)

      tableNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTableNode")
      tableNode.SetName(f"MultiRecolor_Population_Data_{method}")
      tableNode.AddColumn(xArray); tableNode.AddColumn(yArray); tableNode.AddColumn(labelsArray)

      plotSeriesNode.SetAndObserveTableNodeID(tableNode.GetID())
      plotSeriesNode.SetXColumnName(xArray.GetName())
      plotSeriesNode.SetYColumnName(yArray.GetName())
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
      plotChartNode.SetXAxisTitle(f"{method} Component 1")
      plotChartNode.SetYAxisTitle(f"{method} Component 2")

      # Disable auto-range BEFORE setting manual ranges
      if hasattr(plotChartNode, "SetXAxisRangeAuto"):
        plotChartNode.SetXAxisRangeAuto(False)
      if hasattr(plotChartNode, "SetYAxisRangeAuto"):
        plotChartNode.SetYAxisRangeAuto(False)

      # Equal numeric span on both axes
      x_min, x_max = float(np.min(reducedData[:,0])), float(np.max(reducedData[:,0]))
      y_min, y_max = float(np.min(reducedData[:,1])), float(np.max(reducedData[:,1]))
      x_center = (x_min + x_max) / 2.0
      y_center = (y_min + y_max) / 2.0
      span = max(x_max - x_min, y_max - y_min)
      span = max(span, 1e-6)  # avoid zero span
      span *= 1.2  # 20% padding

      plotChartNode.SetXAxisRange(x_center - span/2.0, x_center + span/2.0)
      plotChartNode.SetYAxisRange(y_center - span/2.0, y_center + span/2.0)
      plotChartNode.Modified()

      # show
      plotViewNode = slicer.app.layoutManager().plotWidget(0).mrmlPlotViewNode()
      plotViewNode.SetPlotChartNodeID(plotChartNode.GetID())

      return {"success": True, "chart_node": plotChartNode, "series_node": plotSeriesNode, "table_node": tableNode}

    except Exception as e:
      print(f"Error creating population plot: {e}")
      import traceback; traceback.print_exc()
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

class ColorTheme:
    """Centralized color theme management for InterDeCA"""
    
    @staticmethod
    def getTheme():
        """Get color theme based on Slicer's current theme"""
        # Check if Slicer is in dark mode
        palette = qt.QApplication.palette()
        isDarkMode = palette.color(qt.QPalette.Window).lightness() < 128
        
        if isDarkMode:
            return ColorTheme.getDarkTheme()
        else:
            return ColorTheme.getLightTheme()
    
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
