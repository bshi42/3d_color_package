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
from sklearn.cluster import KMeans # slicer.util.pip_install('scikit-learn')
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt # slicer.util.pip_install('matplotlib')
import seaborn as sns # slicer.util.pip_install('seaborn')
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist
import plotly.graph_objects as go # slicer.util.pip_install('plotly')
import plotly.express as px

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
    colorAnalysisTab = qt.QWidget()
    colorAnalysisTabLayout = qt.QFormLayout(colorAnalysisTab)

    tabsWidget.addTab(DeCATab, "DeCA")
    tabsWidget.addTab(visualizeTab, "Visualize Results")
    tabsWidget.addTab(colorAnalysisTab, "Color Analysis")

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
    self.DCBaseLMSelector.nameFilters=["Point set (*.fcsv *.json *.mrk.json"]
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
    self.meshValidationLabelDC.setStyleSheet("QLabel { color: palette(disabled-text); font-style: italic; }")
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
    self.landmarkValidationLabelDC.setStyleSheet("QLabel { color: palette(disabled-text); font-style: italic; }")
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
    self.interpolationSlider.singleStep = 0.01
    self.interpolationSlider.minimum = 0.0
    self.interpolationSlider.maximum = 1.0
    self.interpolationSlider.value = 0.0
    self.interpolationSlider.setToolTip("Interpolate between original model (0.0) and atlas model (1.0)")
    self.interpolationSlider.enabled = False
    self.interpolationFrameLayout.addRow("Interpolation (Original to Atlas):", self.interpolationSlider)


    self.previewTextureCombo = qt.QComboBox()
    self.previewTextureCombo.setToolTip("Preview a baked atlas-space PNG on the atlas model.")
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
    
    self.landmarkLockButton = qt.QPushButton("🔒 Lock Landmarks")
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
    
    self.landmarkLockStatusLabel = qt.QLabel("Landmarks: Unlocked")
    self.landmarkLockStatusLabel.setStyleSheet("QLabel { color: palette(disabled-text); font-style: italic; }")
    landmarkControlLayout.addWidget(self.landmarkLockStatusLabel)
    
    visualizeWidgetLayout.addRow("Landmark Control:", landmarkControlWidget)
    
    # Add spacing before the visualization button
    visualizeWidgetLayout.addRow(" ", qt.QLabel())

    #
    # Texture Analysis Section
    #
    self.textureAnalysisCollapsible = ctk.ctkCollapsibleButton()
    self.textureAnalysisCollapsible.text = "Texture Analysis"
    self.textureAnalysisCollapsible.collapsed = True
    visualizeWidgetLayout.addRow(self.textureAnalysisCollapsible)
    
    textureAnalysisLayout = qt.QFormLayout(self.textureAnalysisCollapsible)
    
    # Model selector for texture analysis
    self.textureAnalysisModelSelector = slicer.qMRMLNodeComboBox()
    self.textureAnalysisModelSelector.nodeTypes = (("vtkMRMLModelNode"), "")
    self.textureAnalysisModelSelector.setToolTip("Select model for texture analysis")
    self.textureAnalysisModelSelector.setCurrentNode(None)
    textureAnalysisLayout.addRow("Model:", self.textureAnalysisModelSelector)
    
    # Texture analysis type
    self.textureAnalysisTypeCombo = qt.QComboBox()
    self.textureAnalysisTypeCombo.addItems([
      "Surface Roughness Analysis",
      "Texture Pattern Recognition", 
      "Regional Texture Comparison",
      "Texture Histogram Analysis"
    ])
    self.textureAnalysisTypeCombo.setToolTip("Select type of texture analysis to perform")
    textureAnalysisLayout.addRow("Analysis Type:", self.textureAnalysisTypeCombo)
    
    # Analysis parameters
    self.textureAnalysisParamsWidget = qt.QWidget()
    self.textureAnalysisParamsLayout = qt.QFormLayout(self.textureAnalysisParamsWidget)
    
    # Roughness analysis parameters
    self.roughnessKernelSizeSpin = qt.QSpinBox()
    self.roughnessKernelSizeSpin.setRange(3, 21)
    self.roughnessKernelSizeSpin.setValue(5)
    self.roughnessKernelSizeSpin.setToolTip("Kernel size for roughness calculation (odd numbers only)")
    self.roughnessKernelSizeSpin.setSingleStep(2)
    self.textureAnalysisParamsLayout.addRow("Kernel Size:", self.roughnessKernelSizeSpin)
    
    # Pattern recognition parameters
    self.patternScaleSpin = qt.QDoubleSpinBox()
    self.patternScaleSpin.setRange(0.1, 10.0)
    self.patternScaleSpin.setValue(1.0)
    self.patternScaleSpin.setDecimals(1)
    self.patternScaleSpin.setToolTip("Scale factor for pattern detection")
    self.textureAnalysisParamsLayout.addRow("Pattern Scale:", self.patternScaleSpin)
    
    # Regional comparison parameters
    self.regionCountSpin = qt.QSpinBox()
    self.regionCountSpin.setRange(2, 20)
    self.regionCountSpin.setValue(4)
    self.regionCountSpin.setToolTip("Number of regions to compare")
    self.textureAnalysisParamsLayout.addRow("Region Count:", self.regionCountSpin)
    
    # Histogram parameters
    self.histogramBinsSpin = qt.QSpinBox()
    self.histogramBinsSpin.setRange(10, 256)
    self.histogramBinsSpin.setValue(64)
    self.histogramBinsSpin.setToolTip("Number of bins for histogram analysis")
    self.textureAnalysisParamsLayout.addRow("Histogram Bins:", self.histogramBinsSpin)
    
    textureAnalysisLayout.addRow("Parameters:", self.textureAnalysisParamsWidget)
    
    # Analysis buttons
    self.runTextureAnalysisButton = qt.QPushButton("Run Texture Analysis")
    self.runTextureAnalysisButton.setStyleSheet("""
      QPushButton {
        background-color: #9C27B0;
        color: white;
        font-weight: bold;
        border: none;
        border-radius: 5px;
        padding: 8px 16px;
        min-height: 30px;
      }
      QPushButton:hover {
        background-color: #8E24AA;
      }
      QPushButton:pressed {
        background-color: #7B1FA2;
      }
    """)
    textureAnalysisLayout.addRow(self.runTextureAnalysisButton)
    
    # Results display
    self.textureAnalysisResultsText = qt.QTextEdit()
    self.textureAnalysisResultsText.setMaximumHeight(150)
    self.textureAnalysisResultsText.setReadOnly(True)
    self.textureAnalysisResultsText.setToolTip("Texture analysis results will be displayed here")
    textureAnalysisLayout.addRow("Results:", self.textureAnalysisResultsText)
    
    # Export results button
    self.exportTextureResultsButton = qt.QPushButton("Export Results")
    self.exportTextureResultsButton.setEnabled(False)
    self.exportTextureResultsButton.setStyleSheet("""
      QPushButton {
        background-color: #607D8B;
        color: white;
        font-weight: bold;
        border: none;
        border-radius: 5px;
        padding: 6px 12px;
        min-height: 25px;
      }
      QPushButton:hover {
        background-color: #546E7A;
      }
      QPushButton:pressed {
        background-color: #455A64;
      }
      QPushButton:disabled {
        background-color: #BDBDBD;
        color: #757575;
      }
    """)
    textureAnalysisLayout.addRow(self.exportTextureResultsButton)

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
    
    # Texture analysis connections
    self.runTextureAnalysisButton.connect('clicked(bool)', self.onRunTextureAnalysis)
    self.exportTextureResultsButton.connect('clicked(bool)', self.onExportTextureResults)
    self.textureAnalysisTypeCombo.connect('currentIndexChanged(int)', self.onTextureAnalysisTypeChanged)

    ################################### Color Analysis Tab ###################################
    
    # Load DeCA Models Section
    self.loadDecaModelsWidget = ctk.ctkCollapsibleButton()
    self.loadDecaModelsWidget.text = "Load DeCA Models"
    self.loadDecaModelsWidget.collapsed = False
    colorAnalysisTabLayout.addRow(self.loadDecaModelsWidget)
    loadDecaModelsLayout = qt.QFormLayout(self.loadDecaModelsWidget)
    
    # Atlas model selector
    self.atlasModelPathSelector = ctk.ctkPathLineEdit()
    self.atlasModelPathSelector.filters = ctk.ctkPathLineEdit.Files
    self.atlasModelPathSelector.nameFilters = ["Model (*.ply *.stl *.obj *.vtk *.vtp)"]
    self.atlasModelPathSelector.setToolTip("Select the DeCA atlas model file")
    loadDecaModelsLayout.addRow("Atlas Model:", self.atlasModelPathSelector)
    
    # DeCA output directory selector
    self.decaOutputDirSelector = ctk.ctkPathLineEdit()
    self.decaOutputDirSelector.filters = ctk.ctkPathLineEdit.Dirs
    self.decaOutputDirSelector.setToolTip("Select the DeCA output directory containing aligned models and textures")
    loadDecaModelsLayout.addRow("DeCA Output Directory:", self.decaOutputDirSelector)
    
    # Load DeCA models button
    self.loadDecaModelsButton = qt.QPushButton("Load DeCA Models")
    self.loadDecaModelsButton.toolTip = "Load atlas model and aligned specimens from DeCA output"
    self.loadDecaModelsButton.setStyleSheet("""
      QPushButton {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border: none;
        border-radius: 5px;
        padding: 8px 16px;
        min-height: 30px;
      }
      QPushButton:hover {
        background-color: #45a049;
      }
      QPushButton:pressed {
        background-color: #3d8b40;
      }
    """)
    loadDecaModelsLayout.addRow(self.loadDecaModelsButton)
    
    # Status label for loaded models
    self.loadedModelsStatusLabel = qt.QLabel("No DeCA models loaded")
    self.loadedModelsStatusLabel.setStyleSheet("QLabel { color: palette(disabled-text); font-style: italic; }")
    loadDecaModelsLayout.addRow("Status:", self.loadedModelsStatusLabel)
    
    # Model selector for color extraction
    self.colorExtractionModelSelector = slicer.qMRMLNodeComboBox()
    self.colorExtractionModelSelector.nodeTypes = (("vtkMRMLModelNode"), "")
    self.colorExtractionModelSelector.setToolTip("Select model for color analysis")
    self.colorExtractionModelSelector.selectNodeUponCreation = False
    self.colorExtractionModelSelector.addEnabled = False
    self.colorExtractionModelSelector.removeEnabled = False
    self.colorExtractionModelSelector.noneEnabled = True
    self.colorExtractionModelSelector.showHidden = False
    self.colorExtractionModelSelector.showChildNodeTypes = False
    self.colorExtractionModelSelector.setMRMLScene(slicer.mrmlScene)
    loadDecaModelsLayout.addRow("Select model for analysis:", self.colorExtractionModelSelector)

    # Shared Coordinate System Visualization Section
    self.coordinateSystemWidget = ctk.ctkCollapsibleButton()
    self.coordinateSystemWidget.text = "Shared Coordinate System Visualization"
    self.coordinateSystemWidget.collapsed = False
    colorAnalysisTabLayout.addRow(self.coordinateSystemWidget)
    coordinateSystemLayout = qt.QFormLayout(self.coordinateSystemWidget)
    
    # Side-by-side comparison controls
    self.beforeAfterComparisonButton = qt.QPushButton("Show Before/After Alignment")
    self.beforeAfterComparisonButton.toolTip = "Display specimens before and after DeCA alignment side-by-side"
    self.beforeAfterComparisonButton.setStyleSheet("""
      QPushButton {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border: none;
        border-radius: 5px;
        padding: 8px 16px;
        min-height: 30px;
      }
      QPushButton:hover {
        background-color: #45A049;
      }
      QPushButton:pressed {
        background-color: #3D8B40;
      }
    """)
    coordinateSystemLayout.addRow(self.beforeAfterComparisonButton)
    
    # Layout options for comparison
    self.comparisonLayoutCombo = qt.QComboBox()
    self.comparisonLayoutCombo.addItem("Single View (All models)")
    self.comparisonLayoutCombo.addItem("Four-Up View")
    self.comparisonLayoutCombo.addItem("Side-by-Side View")
    self.comparisonLayoutCombo.addItem("Conventional View")
    self.comparisonLayoutCombo.setToolTip("Choose how to display the before/after comparison")
    coordinateSystemLayout.addRow("Comparison Layout:", self.comparisonLayoutCombo)
    
    # Spacing control
    self.modelSpacingSlider = ctk.ctkSliderWidget()
    self.modelSpacingSlider.singleStep = 10
    self.modelSpacingSlider.minimum = 50
    self.modelSpacingSlider.maximum = 500
    self.modelSpacingSlider.value = 200
    self.modelSpacingSlider.setToolTip("Adjust spacing between models")
    coordinateSystemLayout.addRow("Model Spacing:", self.modelSpacingSlider)
    
    # Color Clustering & Analysis Section
    self.colorClusteringWidget = ctk.ctkCollapsibleButton()
    self.colorClusteringWidget.text = "Color Clustering and Analysis Tools"
    self.colorClusteringWidget.collapsed = False
    colorAnalysisTabLayout.addRow(self.colorClusteringWidget)
    colorClusteringLayout = qt.QFormLayout(self.colorClusteringWidget)
    
    # Clustering controls in horizontal layout
    clusteringControlsFrame = qt.QFrame()
    clusteringControlsLayout = qt.QHBoxLayout(clusteringControlsFrame)
    clusteringControlsLayout.setContentsMargins(0, 0, 0, 0)
    clusteringControlsLayout.setSpacing(10)

    # Number of clusters
    clusteringControlsLayout.addWidget(qt.QLabel("Clusters:"))
    self.numClustersSpinBox = qt.QSpinBox()
    self.numClustersSpinBox.setMinimum(2)
    self.numClustersSpinBox.setMaximum(20)
    self.numClustersSpinBox.setValue(3)
    self.numClustersSpinBox.setToolTip("Number of color clusters to create")
    clusteringControlsLayout.addWidget(self.numClustersSpinBox)

    # Clustering method selection
    clusteringControlsLayout.addWidget(qt.QLabel("Method:"))
    self.clusteringMethodCombo = qt.QComboBox()
    self.clusteringMethodCombo.addItem("K-means")
    self.clusteringMethodCombo.addItem("K-means++")
    self.clusteringMethodCombo.addItem("Hierarchical")
    self.clusteringMethodCombo.setToolTip("Clustering algorithm to use")
    clusteringControlsLayout.addWidget(self.clusteringMethodCombo)

    clusteringControlsLayout.addStretch()
    colorClusteringLayout.addRow(clusteringControlsFrame)
    
    # Run clustering button
    self.runClusteringButton = qt.QPushButton("Run Color Clustering")
    self.runClusteringButton.toolTip = "Perform color clustering analysis on selected model"
    self.runClusteringButton.setStyleSheet("""
      QPushButton {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border: none;
        border-radius: 5px;
        padding: 8px 16px;
        min-height: 30px;
      }
      QPushButton:hover {
        background-color: #45a049;
      }
      QPushButton:pressed {
        background-color: #3d8b40;
      }
    """)
    colorClusteringLayout.addRow(self.runClusteringButton)
    
    # Cluster statistics display
    self.clusterStatsText = qt.QPlainTextEdit()
    self.clusterStatsText.setMaximumHeight(80)
    self.clusterStatsText.setPlaceholderText("Cluster statistics will appear here...")
    self.clusterStatsText.setReadOnly(True)
    colorClusteringLayout.addRow("Results:", self.clusterStatsText)

    # Hierarchical Color Space Merging Section
    self.hierarchicalMergingWidget = ctk.ctkCollapsibleButton()
    self.hierarchicalMergingWidget.text = "Hierarchical Color Space Merging"
    self.hierarchicalMergingWidget.collapsed = True
    colorAnalysisTabLayout.addRow(self.hierarchicalMergingWidget)
    hierarchicalMergingLayout = qt.QFormLayout(self.hierarchicalMergingWidget)
    
    # Color similarity threshold
    self.colorSimilarityThreshold = ctk.ctkSliderWidget()
    self.colorSimilarityThreshold.singleStep = 0.01
    self.colorSimilarityThreshold.minimum = 0.0
    self.colorSimilarityThreshold.maximum = 1.0
    self.colorSimilarityThreshold.value = 0.1
    self.colorSimilarityThreshold.setToolTip("Threshold for merging similar colors")
    hierarchicalMergingLayout.addRow("Similarity threshold:", self.colorSimilarityThreshold)
    
    # Generate dendrogram button
    self.generateDendrogramButton = qt.QPushButton("Generate Color Dendrogram")
    self.generateDendrogramButton.toolTip = "Create hierarchical clustering dendrogram of colors"
    self.generateDendrogramButton.setStyleSheet("""
      QPushButton {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border: none;
        border-radius: 5px;
        padding: 8px 16px;
        min-height: 30px;
      }
      QPushButton:hover {
        background-color: #45a049;
      }
      QPushButton:pressed {
        background-color: #3d8b40;
      }
    """)
    hierarchicalMergingLayout.addRow(self.generateDendrogramButton)


    # Color Analysis Connections
    self.loadDecaModelsButton.connect('clicked(bool)', self.onLoadDecaModels)
    self.beforeAfterComparisonButton.connect('clicked(bool)', self.onBeforeAfterComparison)
    self.runClusteringButton.connect('clicked(bool)', self.onRunColorClustering)
    self.generateDendrogramButton.connect('clicked(bool)', self.onGenerateDendrogram)
    

    # Auto-detect Blender executable on startup
    self.autoDetectBlender()
    
    # Load persistent data after UI is set up
    self.loadPersistentData()

  def savePersistentData(self):
    """Save important data to persistent storage"""
    try:
      # Save directory paths
      if hasattr(self, 'meshDirectoryDC') and self.meshDirectoryDC.currentPath:
        self.settings.setValue("meshDirectory", self.meshDirectoryDC.currentPath)
      if hasattr(self, 'landmarkDirectoryDC') and self.landmarkDirectoryDC.currentPath:
        self.settings.setValue("landmarkDirectory", self.landmarkDirectoryDC.currentPath)
      if hasattr(self, 'textureDirectoryDC') and self.textureDirectoryDC.currentPath:
        self.settings.setValue("textureDirectory", self.textureDirectoryDC.currentPath)
      if hasattr(self, 'outputDirectoryDC') and self.outputDirectoryDC.currentPath:
        self.settings.setValue("outputDirectory", self.outputDirectoryDC.currentPath)
      if hasattr(self, 'blenderExeEdit') and self.blenderExeEdit.currentPath:
        self.settings.setValue("blenderExecutable", self.blenderExeEdit.currentPath)
      
      # Save atlas model reference (by name, since node objects can't be serialized)
      if hasattr(self, 'atlasModel') and self.atlasModel:
        self.settings.setValue("atlasModelName", self.atlasModel.GetName())
      
      # Save other important settings
      if hasattr(self, 'bakeSizeSpin'):
        self.settings.setValue("bakeSize", self.bakeSizeSpin.value)
      if hasattr(self, 'bakeExtrusionSpin'):
        self.settings.setValue("bakeExtrusion", self.bakeExtrusionSpin.value)
      if hasattr(self, 'bakeMarginPxSpin'):
        self.settings.setValue("bakeMarginPx", self.bakeMarginPxSpin.value)
      
      # Save landmark lock state
      if hasattr(self, 'landmarkLockButton'):
        # Check if landmarks are currently locked
        markups = list(slicer.util.getNodesByClass('vtkMRMLMarkupsNode'))
        if markups and hasattr(self, '_areLandmarksLocked'):
          is_locked = self._areLandmarksLocked(markups[0])
          self.settings.setValue("landmarksLocked", is_locked)
        
      self.settings.sync()
    except Exception as e:
      print(f"Warning: Could not save persistent data: {e}")

  def loadPersistentData(self):
    """Load persistent data from storage"""
    try:
      # Load directory paths
      if hasattr(self, 'meshDirectoryDC'):
        mesh_dir = self.settings.value("meshDirectory", "")
        if mesh_dir and os.path.exists(mesh_dir):
          self.meshDirectoryDC.setCurrentPath(mesh_dir)
          
      if hasattr(self, 'landmarkDirectoryDC'):
        lm_dir = self.settings.value("landmarkDirectory", "")
        if lm_dir and os.path.exists(lm_dir):
          self.landmarkDirectoryDC.setCurrentPath(lm_dir)
          
      if hasattr(self, 'textureDirectoryDC'):
        tex_dir = self.settings.value("textureDirectory", "")
        if tex_dir and os.path.exists(tex_dir):
          self.textureDirectoryDC.setCurrentPath(tex_dir)
          
      if hasattr(self, 'outputDirectoryDC'):
        out_dir = self.settings.value("outputDirectory", "")
        if out_dir and os.path.exists(out_dir):
          self.outputDirectoryDC.setCurrentPath(out_dir)
          
      if hasattr(self, 'blenderExeEdit'):
        blender_exe = self.settings.value("blenderExecutable", "")
        if blender_exe and os.path.exists(blender_exe):
          self.blenderExeEdit.setCurrentPath(blender_exe)
      
      # Try to restore atlas model by name
      atlas_name = self.settings.value("atlasModelName", "")
      if atlas_name:
        try:
          atlas_node = slicer.util.getNode(atlas_name)
          if atlas_node and atlas_node.IsA("vtkMRMLModelNode"):
            self.atlasModel = atlas_node
            print(f"Restored atlas model: {atlas_name}")
        except Exception:
          pass  # Atlas model not found, that's okay
      
      # Load other settings
      if hasattr(self, 'bakeSizeSpin'):
        bake_size = self.settings.value("bakeSize", 2048)
        self.bakeSizeSpin.setValue(int(bake_size))
      if hasattr(self, 'bakeExtrusionSpin'):
        bake_extrusion = self.settings.value("bakeExtrusion", 0.005)
        self.bakeExtrusionSpin.setValue(float(bake_extrusion))
      if hasattr(self, 'bakeMarginPxSpin'):
        bake_margin = self.settings.value("bakeMarginPx", 2)
        self.bakeMarginPxSpin.setValue(int(bake_margin))
      
      # Load landmark lock state
      if hasattr(self, 'landmarkLockButton'):
        landmarks_locked = self.settings.value("landmarksLocked", False)
        if landmarks_locked:
          # Apply the saved lock state
          markups = list(slicer.util.getNodesByClass('vtkMRMLMarkupsNode'))
          if markups and hasattr(self, '_setLandmarkLockState'):
            for markup in markups:
              self._setLandmarkLockState(markup, True)
            self._updateLandmarkLockUI(True)
        
    except Exception as e:
      print(f"Warning: Could not load persistent data: {e}")

  def cleanup(self):
    """Called when module is about to be unloaded - save persistent data"""
    try:
      self.savePersistentData()
    except Exception as e:
      print(f"Warning: Could not save persistent data during cleanup: {e}")

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
    self.savePersistentData()  # Save when directory changes
    # Update texture matching if textures are already loaded
    if self.textureDirectoryDC.currentPath:
      self.validateTextureMatching()
    self.onParameterSelectDC()
  
  def onLandmarkDirectoryChangedDC(self, directory):
    """Validate landmark directory when changed"""
    landmark_extensions = ['.fcsv', '.json', '.mrk.json']
    self.validateDirectory(directory, landmark_extensions, self.landmarkValidationLabelDC, "landmarks")
    self.savePersistentData()  # Save when directory changes
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
    
    self.savePersistentData()  # Save when directory changes
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
        self.landmarkLockButton.setText("🔓 Unlock Landmarks")
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
        self.landmarkLockStatusLabel.setText("Landmarks: Locked")
        self.landmarkLockStatusLabel.setStyleSheet("QLabel { color: palette(negative); font-weight: bold; }")
      else:
        self.landmarkLockButton.setText("🔒 Lock Landmarks")
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
        self.landmarkLockStatusLabel.setText("Landmarks: Unlocked")
        self.landmarkLockStatusLabel.setStyleSheet("QLabel { color: palette(positive); font-weight: bold; }")
    except Exception as e:
      print(f"Warning: Could not update landmark lock UI: {e}")

  ################################### Texture Analysis Methods ###################################

  def onTextureAnalysisTypeChanged(self, index):
    """Update parameter visibility based on analysis type"""
    try:
      # Show/hide relevant parameters based on analysis type
      analysis_type = self.textureAnalysisTypeCombo.currentText
      
      # Reset all parameter visibility
      self.roughnessKernelSizeSpin.setVisible(False)
      self.patternScaleSpin.setVisible(False)
      self.regionCountSpin.setVisible(False)
      self.histogramBinsSpin.setVisible(False)
      
      # Show relevant parameters
      if "Roughness" in analysis_type:
        self.roughnessKernelSizeSpin.setVisible(True)
      elif "Pattern" in analysis_type:
        self.patternScaleSpin.setVisible(True)
      elif "Regional" in analysis_type:
        self.regionCountSpin.setVisible(True)
      elif "Histogram" in analysis_type:
        self.histogramBinsSpin.setVisible(True)
        
    except Exception as e:
      print(f"Warning: Could not update texture analysis parameters: {e}")

  def onRunTextureAnalysis(self):
    """Run the selected texture analysis"""
    try:
      model_node = self.textureAnalysisModelSelector.currentNode()
      if not model_node:
        slicer.util.errorDisplay("Please select a model for texture analysis.")
        return
      
      analysis_type = self.textureAnalysisTypeCombo.currentText
      self.textureAnalysisResultsText.clear()
      self.textureAnalysisResultsText.append(f"Running {analysis_type}...")
      
      # Get model data
      poly_data = model_node.GetPolyData()
      if not poly_data:
        slicer.util.errorDisplay("Selected model has no geometry data.")
        return
      
      # Run the appropriate analysis
      if "Roughness" in analysis_type:
        results = self._analyzeSurfaceRoughness(poly_data)
      elif "Pattern" in analysis_type:
        results = self._analyzeTexturePatterns(poly_data)
      elif "Regional" in analysis_type:
        results = self._analyzeRegionalTexture(poly_data)
      elif "Histogram" in analysis_type:
        results = self._analyzeTextureHistogram(poly_data)
      else:
        slicer.util.errorDisplay("Unknown analysis type selected.")
        return
      
      # Display results
      self._displayTextureAnalysisResults(results, analysis_type)
      self.exportTextureResultsButton.setEnabled(True)
      
    except Exception as e:
      error_msg = f"Error running texture analysis: {str(e)}"
      slicer.util.errorDisplay(error_msg)
      self.textureAnalysisResultsText.append(f"ERROR: {error_msg}")

  def _analyzeSurfaceRoughness(self, poly_data):
    """Analyze surface roughness using local variance"""
    try:
      import numpy as np
      from scipy import ndimage
      
      # Get points and normals
      points = slicer.util.arrayFromModelPoints(poly_data)
      normals = slicer.util.arrayFromModelPointNormals(poly_data)
      
      if points is None or normals is None:
        return {"error": "Could not extract point data from model"}
      
      # Calculate local surface roughness using normal variance
      kernel_size = self.roughnessKernelSizeSpin.value
      
      # For each point, calculate local normal variance
      roughness_values = []
      for i in range(len(points)):
        # Find nearby points (simplified - in practice would use spatial indexing)
        distances = np.linalg.norm(points - points[i], axis=1)
        nearby_indices = np.where(distances < kernel_size * 0.1)[0]  # Scale factor
        
        if len(nearby_indices) > 1:
          nearby_normals = normals[nearby_indices]
          # Calculate variance of normal directions
          normal_variance = np.var(np.linalg.norm(nearby_normals, axis=1))
          roughness_values.append(normal_variance)
        else:
          roughness_values.append(0.0)
      
      roughness_values = np.array(roughness_values)
      
      # Calculate statistics
      mean_roughness = np.mean(roughness_values)
      std_roughness = np.std(roughness_values)
      min_roughness = np.min(roughness_values)
      max_roughness = np.max(roughness_values)
      
      # Create roughness map on model
      self._applyTextureToModel(poly_data, roughness_values, "Surface Roughness")
      
      return {
        "type": "Surface Roughness Analysis",
        "mean": mean_roughness,
        "std": std_roughness,
        "min": min_roughness,
        "max": max_roughness,
        "values": roughness_values,
        "kernel_size": kernel_size
      }
      
    except Exception as e:
      return {"error": f"Surface roughness analysis failed: {str(e)}"}

  def _analyzeTexturePatterns(self, poly_data):
    """Analyze texture patterns using frequency domain analysis"""
    try:
      import numpy as np
      from scipy import fft
      
      # Get texture coordinates if available
      texture_coords = poly_data.GetPointData().GetTCoords()
      if texture_coords is None:
        return {"error": "Model has no texture coordinates for pattern analysis"}
      
      # Convert to numpy array
      tex_array = slicer.util.arrayFromVTKMatrix(texture_coords)
      
      # Analyze patterns in texture space
      scale = self.patternScaleSpin.value
      
      # Simple pattern detection using FFT
      pattern_strength = []
      for i in range(len(tex_array)):
        # Sample local texture region (simplified)
        local_tex = tex_array[max(0, i-10):min(len(tex_array), i+10)]
        if len(local_tex) > 5:
          # Calculate FFT magnitude as pattern indicator
          fft_result = np.abs(fft.fft(local_tex[:, 0]))  # Use U coordinate
          pattern_strength.append(np.mean(fft_result[1:len(fft_result)//2]))  # Exclude DC component
        else:
          pattern_strength.append(0.0)
      
      pattern_strength = np.array(pattern_strength) * scale
      
      # Calculate pattern statistics
      mean_pattern = np.mean(pattern_strength)
      std_pattern = np.std(pattern_strength)
      
      # Detect repeating patterns
      pattern_regions = np.where(pattern_strength > mean_pattern + std_pattern)[0]
      
      return {
        "type": "Texture Pattern Recognition",
        "mean_pattern_strength": mean_pattern,
        "std_pattern_strength": std_pattern,
        "pattern_regions_count": len(pattern_regions),
        "pattern_regions": pattern_regions,
        "scale_factor": scale
      }
      
    except Exception as e:
      return {"error": f"Pattern recognition analysis failed: {str(e)}"}

  def _analyzeRegionalTexture(self, poly_data):
    """Compare texture properties between different anatomical regions"""
    try:
      import numpy as np
      from sklearn.cluster import KMeans
      
      # Get points and normals
      points = slicer.util.arrayFromModelPoints(poly_data)
      normals = slicer.util.arrayFromModelPointNormals(poly_data)
      
      if points is None or normals is None:
        return {"error": "Could not extract point data from model"}
      
      # Combine position and normal information for clustering
      features = np.column_stack([points, normals])
      
      # Cluster points into regions
      n_regions = self.regionCountSpin.value
      kmeans = KMeans(n_clusters=n_regions, random_state=42)
      region_labels = kmeans.fit_predict(features)
      
      # Analyze texture properties for each region
      region_stats = {}
      for region_id in range(n_regions):
        region_mask = region_labels == region_id
        region_points = points[region_mask]
        region_normals = normals[region_mask]
        
        if len(region_points) > 0:
          # Calculate region statistics
          region_center = np.mean(region_points, axis=0)
          region_size = len(region_points)
          normal_variance = np.var(np.linalg.norm(region_normals, axis=1))
          
          region_stats[region_id] = {
            "center": region_center,
            "size": region_size,
            "normal_variance": normal_variance,
            "points": region_points
          }
      
      # Create regional comparison visualization
      self._visualizeRegionalTexture(poly_data, region_labels, region_stats)
      
      return {
        "type": "Regional Texture Comparison",
        "n_regions": n_regions,
        "region_stats": region_stats,
        "region_labels": region_labels
      }
      
    except Exception as e:
      return {"error": f"Regional texture analysis failed: {str(e)}"}

  def _analyzeTextureHistogram(self, poly_data):
    """Analyze texture value distributions using histograms"""
    try:
      import numpy as np
      import matplotlib.pyplot as plt
      
      # Get texture coordinates
      texture_coords = poly_data.GetPointData().GetTCoords()
      if texture_coords is None:
        return {"error": "Model has no texture coordinates for histogram analysis"}
      
      # Convert to numpy array
      tex_array = slicer.util.arrayFromVTKMatrix(texture_coords)
      
      # Calculate texture intensity (simplified - would use actual texture data)
      # For now, use texture coordinate values as proxy
      u_values = tex_array[:, 0]
      v_values = tex_array[:, 1]
      
      # Create histograms
      n_bins = self.histogramBinsSpin.value
      
      u_hist, u_bins = np.histogram(u_values, bins=n_bins, range=(0, 1))
      v_hist, v_bins = np.histogram(v_values, bins=n_bins, range=(0, 1))
      
      # Calculate histogram statistics
      u_mean = np.mean(u_values)
      u_std = np.std(u_values)
      v_mean = np.mean(v_values)
      v_std = np.std(v_values)
      
      # Detect peaks in histograms
      u_peaks = self._findHistogramPeaks(u_hist)
      v_peaks = self._findHistogramPeaks(v_hist)
      
      # Create histogram visualization
      self._createHistogramVisualization(u_hist, u_bins, v_hist, v_bins)
      
      return {
        "type": "Texture Histogram Analysis",
        "u_stats": {"mean": u_mean, "std": u_std, "peaks": u_peaks},
        "v_stats": {"mean": v_mean, "std": v_std, "peaks": v_peaks},
        "u_histogram": u_hist,
        "v_histogram": v_hist,
        "u_bins": u_bins,
        "v_bins": v_bins,
        "n_bins": n_bins
      }
      
    except Exception as e:
      return {"error": f"Histogram analysis failed: {str(e)}"}

  def _findHistogramPeaks(self, histogram, min_height=0.1):
    """Find peaks in histogram data"""
    try:
      from scipy.signal import find_peaks
      peaks, properties = find_peaks(histogram, height=min_height * np.max(histogram))
      return peaks.tolist()
    except Exception:
      # Fallback: simple peak detection
      peaks = []
      for i in range(1, len(histogram) - 1):
        if histogram[i] > histogram[i-1] and histogram[i] > histogram[i+1]:
          if histogram[i] > min_height * np.max(histogram):
            peaks.append(i)
      return peaks

  def _applyTextureToModel(self, poly_data, values, name):
    """Apply texture values to model for visualization"""
    try:
      # Create a new model node with the texture values
      model_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode")
      model_node.SetAndObservePolyData(poly_data)
      model_node.SetName(f"{name}_Model")
      
      # Add texture values as point data
      texture_array = vtk.util.numpy_support.numpy_to_vtk(values)
      texture_array.SetName(name)
      poly_data.GetPointData().AddArray(texture_array)
      
      # Set as active scalar
      poly_data.GetPointData().SetActiveScalars(name)
      
      # Create display node
      display_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelDisplayNode")
      display_node.SetAndObserveColorNodeID("vtkMRMLColorTableNodeFilePlasma.txt")
      display_node.SetScalarVisibility(True)
      model_node.SetAndObserveDisplayNodeID(display_node.GetID())
      
      return model_node
      
    except Exception as e:
      print(f"Warning: Could not apply texture to model: {e}")

  def _visualizeRegionalTexture(self, poly_data, region_labels, region_stats):
    """Create visualization for regional texture analysis"""
    try:
      # Create model with regional coloring
      model_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode")
      model_node.SetAndObservePolyData(poly_data)
      model_node.SetName("Regional_Texture_Analysis")
      
      # Add region labels as point data
      region_array = vtk.util.numpy_support.numpy_to_vtk(region_labels)
      region_array.SetName("Region_Labels")
      poly_data.GetPointData().AddArray(region_array)
      poly_data.GetPointData().SetActiveScalars("Region_Labels")
      
      # Create display node
      display_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelDisplayNode")
      display_node.SetAndObserveColorNodeID("vtkMRMLColorTableNodeFileRainbow.txt")
      display_node.SetScalarVisibility(True)
      model_node.SetAndObserveDisplayNodeID(display_node.GetID())
      
      return model_node
      
    except Exception as e:
      print(f"Warning: Could not create regional visualization: {e}")

  def _createHistogramVisualization(self, u_hist, u_bins, v_hist, v_bins):
    """Create histogram visualization plots"""
    try:
      import matplotlib.pyplot as plt
      
      # Create figure with subplots
      fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
      
      # U coordinate histogram
      ax1.bar(u_bins[:-1], u_hist, width=np.diff(u_bins), alpha=0.7, color='blue')
      ax1.set_title('U Coordinate Histogram')
      ax1.set_xlabel('U Value')
      ax1.set_ylabel('Frequency')
      ax1.grid(True, alpha=0.3)
      
      # V coordinate histogram
      ax2.bar(v_bins[:-1], v_hist, width=np.diff(v_bins), alpha=0.7, color='red')
      ax2.set_title('V Coordinate Histogram')
      ax2.set_xlabel('V Value')
      ax2.set_ylabel('Frequency')
      ax2.grid(True, alpha=0.3)
      
      plt.tight_layout()
      plt.show()
      
    except Exception as e:
      print(f"Warning: Could not create histogram visualization: {e}")

  def _displayTextureAnalysisResults(self, results, analysis_type):
    """Display texture analysis results in the text widget"""
    try:
      self.textureAnalysisResultsText.clear()
      self.textureAnalysisResultsText.append(f"=== {analysis_type} Results ===\n")
      
      if "error" in results:
        self.textureAnalysisResultsText.append(f"ERROR: {results['error']}")
        return
      
      # Display results based on analysis type
      if "Roughness" in analysis_type:
        self.textureAnalysisResultsText.append(f"Mean Roughness: {results['mean']:.4f}")
        self.textureAnalysisResultsText.append(f"Std Deviation: {results['std']:.4f}")
        self.textureAnalysisResultsText.append(f"Min Roughness: {results['min']:.4f}")
        self.textureAnalysisResultsText.append(f"Max Roughness: {results['max']:.4f}")
        self.textureAnalysisResultsText.append(f"Kernel Size: {results['kernel_size']}")
        
      elif "Pattern" in analysis_type:
        self.textureAnalysisResultsText.append(f"Mean Pattern Strength: {results['mean_pattern_strength']:.4f}")
        self.textureAnalysisResultsText.append(f"Std Pattern Strength: {results['std_pattern_strength']:.4f}")
        self.textureAnalysisResultsText.append(f"Pattern Regions Found: {results['pattern_regions_count']}")
        self.textureAnalysisResultsText.append(f"Scale Factor: {results['scale_factor']}")
        
      elif "Regional" in analysis_type:
        self.textureAnalysisResultsText.append(f"Number of Regions: {results['n_regions']}")
        for region_id, stats in results['region_stats'].items():
          self.textureAnalysisResultsText.append(f"\nRegion {region_id}:")
          self.textureAnalysisResultsText.append(f"  Size: {stats['size']} points")
          self.textureAnalysisResultsText.append(f"  Center: ({stats['center'][0]:.3f}, {stats['center'][1]:.3f}, {stats['center'][2]:.3f})")
          self.textureAnalysisResultsText.append(f"  Normal Variance: {stats['normal_variance']:.4f}")
          
      elif "Histogram" in analysis_type:
        u_stats = results['u_stats']
        v_stats = results['v_stats']
        self.textureAnalysisResultsText.append(f"U Coordinate Statistics:")
        self.textureAnalysisResultsText.append(f"  Mean: {u_stats['mean']:.4f}")
        self.textureAnalysisResultsText.append(f"  Std: {u_stats['std']:.4f}")
        self.textureAnalysisResultsText.append(f"  Peaks: {u_stats['peaks']}")
        self.textureAnalysisResultsText.append(f"\nV Coordinate Statistics:")
        self.textureAnalysisResultsText.append(f"  Mean: {v_stats['mean']:.4f}")
        self.textureAnalysisResultsText.append(f"  Std: {v_stats['std']:.4f}")
        self.textureAnalysisResultsText.append(f"  Peaks: {v_stats['peaks']}")
        self.textureAnalysisResultsText.append(f"\nHistogram Bins: {results['n_bins']}")
      
      self.textureAnalysisResultsText.append(f"\nAnalysis completed successfully!")
      
    except Exception as e:
      self.textureAnalysisResultsText.append(f"Error displaying results: {str(e)}")

  def onExportTextureResults(self):
    """Export texture analysis results to file"""
    try:
      if not self.textureAnalysisResultsText.toPlainText().strip():
        slicer.util.warningDisplay("No results to export.")
        return
      
      # Get output directory
      output_dir = self.outputDirectoryDC.currentPath if hasattr(self, 'outputDirectoryDC') else ""
      if not output_dir:
        output_dir = slicer.app.temporaryPath
      
      # Create filename
      analysis_type = self.textureAnalysisTypeCombo.currentText.replace(" ", "_")
      filename = f"texture_analysis_{analysis_type}_{slicer.util.getDate()}.txt"
      filepath = os.path.join(output_dir, filename)
      
      # Write results to file
      with open(filepath, 'w') as f:
        f.write(self.textureAnalysisResultsText.toPlainText())
      
      slicer.util.infoDisplay(f"Results exported to: {filepath}")
      
    except Exception as e:
      slicer.util.errorDisplay(f"Error exporting results: {str(e)}")

  ################################### Color Analysis Methods ###################################

    
  def onLoadDecaModels(self):
    """Load DeCA atlas model and aligned specimens for color analysis"""
    try:
      atlas_path = self.atlasModelPathSelector.currentPath
      output_dir = self.decaOutputDirSelector.currentPath
      
      if not atlas_path or not os.path.exists(atlas_path):
        slicer.util.warningDisplay("Please select a valid atlas model file.")
        return
        
      if not output_dir or not os.path.isdir(output_dir):
        slicer.util.warningDisplay("Please select a valid DeCA output directory.")
        return
      
      # Clear existing DeCA models
      modelNodes = slicer.util.getNodesByClass('vtkMRMLModelNode')
      for node in modelNodes:
        if 'deca_' in node.GetName().lower():
          slicer.mrmlScene.RemoveNode(node)
      
      loaded_models = []
      
      # Load atlas model
      try:
        atlas_model = slicer.util.loadModel(atlas_path)
        atlas_model.SetName("DeCA_Atlas")
        
        # Ensure display node exists
        if not atlas_model.GetDisplayNode():
          atlas_model.CreateDefaultDisplayNodes()
        
        display_node = atlas_model.GetDisplayNode()
        if display_node:
          display_node.SetColor(1.0, 0.8, 0.2)  # Gold color for atlas
          display_node.SetVisibility(True)
        
        loaded_models.append("Atlas")
        print(f"Loaded atlas model: {os.path.basename(atlas_path)}")
      except Exception as e:
        slicer.util.errorDisplay(f"Failed to load atlas model: {str(e)}")
        return
      
      # Look for aligned models directory
      aligned_models_dir = None
      possible_dirs = ['alignedModels', 'aligned', 'models', 'output']
      for dir_name in possible_dirs:
        test_path = os.path.join(output_dir, dir_name)
        if os.path.isdir(test_path):
          aligned_models_dir = test_path
          break
      
      if not aligned_models_dir:
        # Use output directory directly
        aligned_models_dir = output_dir
      
      # Load aligned models
      model_files = [f for f in os.listdir(aligned_models_dir) 
                    if f.lower().endswith(('.ply', '.obj', '.stl', '.vtk', '.vtp'))]
      
      loaded_count = 0
      for i, model_file in enumerate(model_files[:10]):  # Limit to 10 models for performance
        try:
          model_path = os.path.join(aligned_models_dir, model_file)
          model = slicer.util.loadModel(model_path)
          specimen_name = os.path.splitext(model_file)[0]
          model.SetName(f"DeCA_Specimen_{specimen_name}")
          
          # Ensure display node exists
          if not model.GetDisplayNode():
            model.CreateDefaultDisplayNodes()
          
          display_node = model.GetDisplayNode()
          if display_node:
            # Set different colors for each specimen
            color_index = i % 10
            colors = [
              [0.8, 0.2, 0.2],  # Red
              [0.2, 0.8, 0.2],  # Green  
              [0.2, 0.2, 0.8],  # Blue
              [0.8, 0.8, 0.2],  # Yellow
              [0.8, 0.2, 0.8],  # Magenta
              [0.2, 0.8, 0.8],  # Cyan
              [0.8, 0.5, 0.2],  # Orange
              [0.5, 0.2, 0.8],  # Purple
              [0.2, 0.8, 0.5],  # Teal
              [0.8, 0.5, 0.5]   # Pink
            ]
            
            display_node.SetColor(*colors[color_index])
            display_node.SetVisibility(True)
          
          loaded_count += 1
          print(f"Loaded specimen {loaded_count}: {specimen_name}")
          
        except Exception as e:
          print(f"Failed to load model {model_file}: {str(e)}")
          continue
      
      loaded_models.append(f"{loaded_count} specimens")
      
      # Look for textures directory
      textures_dir = None
      possible_texture_dirs = ['atlasTextures', 'textures', 'baked', 'atlas_textures']
      for dir_name in possible_texture_dirs:
        test_path = os.path.join(output_dir, dir_name)
        if os.path.isdir(test_path):
          textures_dir = test_path
          break
      
      texture_count = 0
      if textures_dir:
        texture_files = [f for f in os.listdir(textures_dir) 
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        texture_count = len(texture_files)
        loaded_models.append(f"{texture_count} textures")
        
        # Store texture directory for later use
        self.lastBakedTexturesPath = textures_dir
        
        # Try to apply texture to atlas if available
        if texture_count > 0:
          avg_texture = os.path.join(textures_dir, "average_texture.png")
          if os.path.exists(avg_texture):
            logic = InterDeCALogic()
            logic.applyTextureToModel(atlas_model, avg_texture)
            loaded_models.append("atlas textured")
      
      # Update status
      status_text = f"Loaded: {', '.join(loaded_models)}"
      self.loadedModelsStatusLabel.setText(status_text)
      self.loadedModelsStatusLabel.setStyleSheet("QLabel { color: palette(positive); font-weight: bold; }")
      
      # Update color extraction model selector to include loaded models
      self.colorExtractionModelSelector.setCurrentNode(atlas_model)
      
      # Reset view to show all models
      slicer.app.layoutManager().resetThreeDViews()
      
      print(f"DeCA models loaded successfully: {status_text}")
      slicer.util.infoDisplay(f"Successfully loaded DeCA models:\n{status_text}")
      
    except Exception as e:
      error_msg = f"Error loading DeCA models: {str(e)}"
      print(error_msg)
      self.loadedModelsStatusLabel.setText("Failed to load models")
      self.loadedModelsStatusLabel.setStyleSheet("QLabel { color: palette(negative); font-weight: bold; }")
      slicer.util.errorDisplay(error_msg)
    
  def onBeforeAfterComparison(self):
    """Show side-by-side comparison of specimens before and after DeCA alignment"""
    try:
      # Debug: List all current models
      modelNodes = slicer.util.getNodesByClass('vtkMRMLModelNode')
      print(f"DEBUG: Found {len(modelNodes)} total models in scene:")
      for node in modelNodes:
        visibility = "visible" if node.GetDisplayNode() and node.GetDisplayNode().GetVisibility() else "hidden"
        print(f"  - {node.GetName()} ({visibility})")
      
      # Check if we have loaded DeCA models
      deca_models = [node for node in modelNodes if 'DeCA_' in node.GetName()]
      print(f"DEBUG: Found {len(deca_models)} DeCA models")
      
      if not deca_models:
        slicer.util.warningDisplay("No DeCA models found in scene.\nPlease load DeCA models first using the 'Load DeCA Models' button.")
        return
      
      if deca_models:
        # Use loaded DeCA models for comparison
        print("Using loaded DeCA models for comparison")
        logic = InterDeCALogic()
        success = logic.createComparisonFromLoadedModels(self.modelSpacingSlider.value, 
                                                        self.comparisonLayoutCombo.currentText)
        if success:
          print("Before/After comparison created from loaded models")
          
          # Ensure all comparison models are visible
          comparison_models = [node for node in slicer.util.getNodesByClass('vtkMRMLModelNode') 
                             if 'comparison_' in node.GetName()]
          print(f"DEBUG: Making {len(comparison_models)} comparison models visible")
          for node in comparison_models:
            if node.GetDisplayNode():
              node.GetDisplayNode().SetVisibility(True)
              print(f"  - Made {node.GetName()} visible")
          
          # Force view update
          slicer.app.layoutManager().resetThreeDViews()
          
          slicer.util.infoDisplay("Comparison created successfully!\nBlue = Individual specimens\nGreen = Atlas reference")
        else:
          slicer.util.warningDisplay("Failed to create comparison from loaded models.")
        return
      
      # Fallback to original workflow if no DeCA models are loaded
      originalDir = getattr(self, 'meshDirectoryDC', None)
      alignedDir = getattr(self, 'lastDeCAAlignedModelsPath', None)
      
      if not originalDir or not originalDir.currentPath:
        slicer.util.warningDisplay("Please either:\n1. Load DeCA models using 'Load DeCA Models' button above, OR\n2. Run DeCA analysis first to set original models directory.")
        return
        
      if not alignedDir or not os.path.exists(alignedDir):
        slicer.util.warningDisplay("No aligned models found. Please run DeCA analysis first.")
        return
        
      # Create comparison visualization using original workflow
      logic = InterDeCALogic()
      logic.createBeforeAfterComparison(originalDir.currentPath, alignedDir)
      
      print("Before/After comparison created successfully")
      
    except Exception as e:
      print(f"ERROR in onBeforeAfterComparison: {str(e)}")
      import traceback
      traceback.print_exc()
      slicer.util.errorDisplay(f"Error creating before/after comparison: {str(e)}")


  def onRunColorClustering(self):
    """Perform color clustering analysis on selected model"""
    try:
      # Get the atlas model from loaded DeCA models
      modelNodes = slicer.util.getNodesByClass('vtkMRMLModelNode')
      atlas_models = [node for node in modelNodes if 'DeCA_Atlas' in node.GetName()]
      
      if not atlas_models:
        slicer.util.warningDisplay("Please load DeCA models first using 'Load DeCA Models' button.")
        return
        
      model = atlas_models[0]
        
      numClusters = self.numClustersSpinBox.value
      method = self.clusteringMethodCombo.currentText
      
      logic = InterDeCALogic()
      clusterResults = logic.performColorClustering(model, numClusters, method)
      
      if clusterResults:
        # Display cluster statistics
        self.clusterStatsText.clear()
        self.clusterStatsText.appendPlainText(f"Clustering Method: {method}")
        self.clusterStatsText.appendPlainText(f"Number of Clusters: {numClusters}")
        self.clusterStatsText.appendPlainText(f"Total Points: {clusterResults['total_points']}")
        
        for i, (centroid, count, percentage) in enumerate(zip(
            clusterResults['centroids'], 
            clusterResults['cluster_counts'], 
            clusterResults['percentages'])):
          rgb = [int(c*255) for c in centroid[:3]]
          self.clusterStatsText.appendPlainText(
            f"Cluster {i+1}: RGB({rgb[0]}, {rgb[1]}, {rgb[2]}) - {count} points ({percentage:.1f}%)")
        
        print(f"Color clustering completed with {numClusters} clusters")
      else:
        slicer.util.warningDisplay("Color clustering failed. Please ensure the model has texture data.")
        
    except Exception as e:
      slicer.util.errorDisplay(f"Error performing color clustering: {str(e)}")

  def onGenerateDendrogram(self):
    """Generate hierarchical clustering dendrogram of colors"""
    try:
      # Get the atlas model from loaded DeCA models
      modelNodes = slicer.util.getNodesByClass('vtkMRMLModelNode')
      atlas_models = [node for node in modelNodes if 'DeCA_Atlas' in node.GetName()]
      
      if not atlas_models:
        slicer.util.warningDisplay("Please load DeCA models first using 'Load DeCA Models' button.")
        return
        
      model = atlas_models[0]
        
      threshold = self.colorSimilarityThreshold.value
      
      logic = InterDeCALogic()
      dendrogram_path = logic.generateColorDendrogram(model, threshold)
      
      if dendrogram_path and os.path.exists(dendrogram_path):
        print(f"Color dendrogram saved to: {dendrogram_path}")
        slicer.util.infoDisplay(f"Color dendrogram generated and saved to:\n{dendrogram_path}")
      else:
        slicer.util.warningDisplay("Failed to generate color dendrogram.")
        
    except Exception as e:
      slicer.util.errorDisplay(f"Error generating dendrogram: {str(e)}")


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

  def updateBakedPreviewList(self):
    self.previewTextureCombo.blockSignals(True)
    self.previewTextureCombo.clear()
    d = self.lastBakedTexturesPath
    if d and os.path.isdir(d):
      items = [os.path.splitext(f)[0] for f in sorted(os.listdir(d)) if f.lower().endswith('.png')]
      for it in items: self.previewTextureCombo.addItem(it)
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

    # Initialize progress tracking
    self.currentOperation = "DeCA Analysis"
    self.applyButtonDC.enabled = False
    self.updateProgressDC(0, "Initializing DeCA analysis...", showCancel=True)

    try:
      # Folders
      self.updateProgressDC(5, "Creating output directories...")
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
      if loadAtlasOption:
        self.updateProgressDC(10, "Loading atlas model and landmarks...")
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
        self.updateProgressDC(10, "Generating new atlas from data...")
        self.atlasModel, self.atlasLMs = self.generateNewAtlas(removeScaleOption, self.logInfoDC)

      # Check if atlas generation was successful
      if self.atlasModel is None or self.atlasLMs is None:
        self.logInfoDC.appendPlainText("Failed to generate atlas. Please check the data and try again.")
        return

      # Save an intermediate atlas file (RAS) so Blender can read it
      self.updateProgressDC(20, "Saving atlas for Blender processing...")
      atlas_preuv_obj = os.path.join(self.folderNames['output'], 'decaAtlas_preUV.obj')
      logic._save_model_with_cs(self.atlasModel, atlas_preuv_obj, 'RAS')

      # ---- 2) Blender cleanup + Smart UV ----
      self.updateProgressDC(25, "Preparing atlas with Blender (cleanup + UV)...")
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
          self.resetProgressDC()
          self.applyButtonDC.enabled = True
          return
      atlas_uv_obj = os.path.join(self.folderNames['output'], 'decaAtlasUV.obj')
      try:
        logic.blender_prepare_atlas(blender_exe, atlas_preuv_obj, atlas_uv_obj,
                                    merge_dist=merge_dist, smart_angle=smart_angle, island_margin=island_margin)
        self.logInfoDC.appendPlainText(f"Atlas cleaned & UV'd in Blender → {atlas_uv_obj}")
      except Exception as e:
        self.logInfoDC.appendPlainText(f"Blender atlas UV step failed: {e}")
        self.resetProgressDC()
        self.applyButtonDC.enabled = True
        return

      # Reload UV'd atlas back into Slicer (replace old atlas node)
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
      self.updateProgressDC(40, "Performing rigid alignment to atlas...")
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
      self.updateProgressDC(60, "Calculating point correspondences (resampling)...")
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
      
    except Exception as e:
      # Handle any unexpected errors
      self.logInfoDC.appendPlainText(f"Unexpected error during DeCA analysis: {str(e)}")
      self.resetProgressDC()
      self.applyButtonDC.enabled = True



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
    """Make only relevant DeCA models visible, not all models in the scene"""
    # Only show models that are likely to be DeCA-related
    relevant_models = []
    
    # Check for atlas models
    if hasattr(self, 'atlasModel') and self.atlasModel:
      relevant_models.append(self.atlasModel)
    
    # Check for loaded DeCA models from the model selector
    if hasattr(self, 'meshSelect') and self.meshSelect.currentNode():
      relevant_models.append(self.meshSelect.currentNode())
    
    # Check for models in the color extraction selector
    if hasattr(self, 'colorExtractionModelSelector') and self.colorExtractionModelSelector.currentNode():
      relevant_models.append(self.colorExtractionModelSelector.currentNode())
    
    # Check for models that might be DeCA results (look for common naming patterns)
    for m in slicer.util.getNodesByClass('vtkMRMLModelNode'):
      try:
        node_name = m.GetName().lower()
        # Look for DeCA-related naming patterns
        if any(keyword in node_name for keyword in ['deca', 'atlas', 'aligned', 'resampled', 'correspondence']):
          relevant_models.append(m)
      except Exception:
        pass
    
    # Make only relevant models visible
    for model in relevant_models:
      try:
        dn = model.GetDisplayNode()
        if dn: 
          dn.SetVisibility(True)
      except Exception:
        pass
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
    meanTransformFilter.SetInputData(originalMesh)
    meanTransformFilter.SetTransform(meanTransform)
    meanTransformFilter.Update()
    meanWarpedMesh = meanTransformFilter.GetOutput()

    meanTransformBase = vtk.vtkThinPlateSplineTransform()
    meanTransformBase.SetSourceLandmarks(baseLandmarks)
    meanTransformBase.SetTargetLandmarks(meanShape)
    meanTransformBase.SetBasisToR() # for 3D transform

    meanTransformBaseFilter = vtk.vtkTransformPolyDataFilter()
    meanTransformBaseFilter.SetInputData(baseMesh)
    meanTransformBaseFilter.SetTransform(meanTransformBase)
    meanTransformBaseFilter.Update()
    meanWarpedBase = meanTransformBaseFilter.GetOutput() # Warped atlas

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
    inverseTransformFilter.SetInputData(correspondingMesh)
    inverseTransformFilter.SetTransform(inverseTransform)
    inverseTransformFilter.Update()

    return inverseTransformFilter.GetOutput()

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
    """Apply a texture image to a model node"""
    if not modelNode or not os.path.exists(pngPath):
      return False

    # Ensure display node exists
    if not modelNode.GetDisplayNode():
      modelNode.CreateDefaultDisplayNodes()
    
    dn = modelNode.GetDisplayNode()
    if not dn:
      print(f"Error: Could not create display node for {modelNode.GetName()}")
      return False
    
    try:
      dn.SetBackfaceCulling(0)
      dn.SetFrontfaceCulling(0)
      dn.SetScalarVisibility(False)
      try: 
        dn.SetInterpolateTexture(1)
      except Exception: 
        pass

      reader = vtk.vtkPNGReader()
      reader.SetFileName(pngPath)
      reader.Update()

      # No flipping – Blender/Slicer UVs now match
      dn.SetTextureImageDataConnection(reader.GetOutputPort())
      print(f"Applied texture {os.path.basename(pngPath)} to {modelNode.GetName()}")
      return True
      
    except Exception as e:
      print(f"Error applying texture to {modelNode.GetName()}: {str(e)}")
      return False

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

  ################################### Color Analysis Logic Methods ###################################

  def createBeforeAfterComparison(self, originalDir, alignedDir):
    """Create side-by-side comparison of specimens before and after DeCA alignment"""
    try:
      # Clear existing comparison models
      modelNodes = slicer.util.getNodesByClass('vtkMRMLModelNode')
      for node in modelNodes:
        if 'comparison_' in node.GetName():
          slicer.mrmlScene.RemoveNode(node)
      
      # Get model files
      original_files = [f for f in os.listdir(originalDir) if f.lower().endswith(('.ply', '.obj', '.stl', '.vtk'))]
      aligned_files = [f for f in os.listdir(alignedDir) if f.lower().endswith(('.ply', '.obj', '.stl', '.vtk'))]
      
      # Load up to 3 models for comparison
      comparison_count = min(3, len(original_files), len(aligned_files))
      
      # Get spacing from UI control
      spacing_factor = self.modelSpacingSlider.value
      
      for i in range(comparison_count):
        # Load original model
        original_path = os.path.join(originalDir, original_files[i])
        original_model = slicer.util.loadModel(original_path)
        original_model.SetName(f"comparison_original_{i+1}")
        
        # Load aligned model
        aligned_path = os.path.join(alignedDir, aligned_files[i])
        aligned_model = slicer.util.loadModel(aligned_path)
        aligned_model.SetName(f"comparison_aligned_{i+1}")
        
        # Apply transforms to separate models spatially
        # Original models on the left, aligned on the right
        # Each pair separated vertically
        
        # Create transforms
        original_transform = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLLinearTransformNode')
        original_transform.SetName(f"OriginalTransform_{i+1}")
        aligned_transform = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLLinearTransformNode')
        aligned_transform.SetName(f"AlignedTransform_{i+1}")
        
        # Set transform matrices for positioning
        original_matrix = vtk.vtkMatrix4x4()
        aligned_matrix = vtk.vtkMatrix4x4()
        
        # Position original models on the left, aligned on the right
        # Separate each pair vertically
        y_offset = i * spacing_factor  # Vertical separation between pairs
        x_offset_original = -spacing_factor * 0.7  # Left side for originals
        x_offset_aligned = spacing_factor * 0.7    # Right side for aligned
        
        original_matrix.SetElement(0, 3, x_offset_original)  # X translation
        original_matrix.SetElement(1, 3, y_offset)           # Y translation
        original_matrix.SetElement(2, 3, 0)                  # Z translation
        
        aligned_matrix.SetElement(0, 3, x_offset_aligned)    # X translation
        aligned_matrix.SetElement(1, 3, y_offset)            # Y translation
        aligned_matrix.SetElement(2, 3, 0)                   # Z translation
        
        original_transform.SetMatrixTransformToParent(original_matrix)
        aligned_transform.SetMatrixTransformToParent(aligned_matrix)
        
        # Apply transforms to models
        original_model.SetAndObserveTransformNodeID(original_transform.GetID())
        aligned_model.SetAndObserveTransformNodeID(aligned_transform.GetID())
        
        # Set colors to distinguish original (blue) vs aligned (green)
        original_model.GetDisplayNode().SetColor(0.2, 0.4, 1.0)  # Blue for original
        aligned_model.GetDisplayNode().SetColor(0.2, 1.0, 0.4)   # Green for aligned
        
        # Add labels to distinguish the models
        original_model.GetDisplayNode().SetVisibility(True)
        aligned_model.GetDisplayNode().SetVisibility(True)
        
        print(f"Positioned pair {i+1}: Original (blue) at ({x_offset_original:.0f}, {y_offset:.0f}, 0), "
              f"Aligned (green) at ({x_offset_aligned:.0f}, {y_offset:.0f}, 0)")
      
      # Set up view layout based on user selection
      layout_option = self.comparisonLayoutCombo.currentText
      self.setupComparisonViewLayout(layout_option)
      
      # Reset view and fit all models
      slicer.app.layoutManager().resetThreeDViews()
      
      # Center the view on all models
      threeDWidget = slicer.app.layoutManager().threeDWidget(0)
      threeDView = threeDWidget.threeDView()
      threeDView.resetFocalPoint()
      
      print(f"Before/After comparison created with {comparison_count} specimen pairs")
      print("Blue models = Original, Green models = Aligned")
      print("Models are arranged left-to-right (original vs aligned) and top-to-bottom (different specimens)")
      
    except Exception as e:
      print(f"Error in createBeforeAfterComparison: {str(e)}")
      raise

  def setupComparisonViewLayout(self, layout_option):
    """Setup view layout based on user selection"""
    try:
      layoutManager = slicer.app.layoutManager()
      
      # Get available layout constants
      layoutNode = slicer.vtkMRMLLayoutNode
      
      if layout_option == "Single View (All models)":
        layoutManager.setLayout(layoutNode.SlicerLayoutOneUp3DView)
        print("Using Single 3D View - all models in one view")
        
      elif layout_option == "Four-Up View":
        layoutManager.setLayout(layoutNode.SlicerLayoutFourUpView)
        print("Using Four-Up View layout")
        
      elif layout_option == "Side-by-Side View":
        # Try different side-by-side layout options
        try:
          layoutManager.setLayout(layoutNode.SlicerLayoutSideBySideView)
          print("Using Side-by-Side View layout")
        except AttributeError:
          try:
            layoutManager.setLayout(layoutNode.SlicerLayoutDual3DView)
            print("Using Dual 3D View layout")
          except AttributeError:
            # Fallback to Four-Up if side-by-side not available
            layoutManager.setLayout(layoutNode.SlicerLayoutFourUpView)
            print("Using Four-Up View layout (side-by-side not available)")
        
      elif layout_option == "Conventional View":
        try:
          layoutManager.setLayout(layoutNode.SlicerLayoutConventionalView)
          print("Using Conventional View layout")
        except AttributeError:
          # Fallback to Four-Up if conventional not available
          layoutManager.setLayout(layoutNode.SlicerLayoutFourUpView)
          print("Using Four-Up View layout (conventional not available)")
        
      else:
        # Default to single view
        layoutManager.setLayout(layoutNode.SlicerLayoutOneUp3DView)
        print("Using default Single 3D View")
      
    except Exception as e:
      print(f"Error setting up comparison view layout: {str(e)}")
      # Fallback to single view
      try:
        layoutManager.setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutOneUp3DView)
        print("Fallback: Using Single 3D View")
      except Exception as fallback_error:
        print(f"Even fallback failed: {fallback_error}")
        pass

  def createComparisonFromLoadedModels(self, spacing_factor, layout_option):
    """Create comparison visualization from loaded DeCA models"""
    try:
      print(f"DEBUG: Starting comparison with spacing={spacing_factor}, layout={layout_option}")
      
      # Get loaded DeCA models
      modelNodes = slicer.util.getNodesByClass('vtkMRMLModelNode')
      atlas_models = [node for node in modelNodes if 'DeCA_Atlas' in node.GetName()]
      specimen_models = [node for node in modelNodes if 'DeCA_Specimen_' in node.GetName()]
      
      print(f"DEBUG: Found {len(atlas_models)} atlas models, {len(specimen_models)} specimen models")
      
      if not atlas_models:
        print("ERROR: No DeCA atlas model found")
        return False
        
      if not specimen_models:
        print("ERROR: No DeCA specimen models found")
        return False
      
      # Clear existing comparison models
      comparison_models = [node for node in modelNodes if 'comparison_' in node.GetName()]
      print(f"DEBUG: Removing {len(comparison_models)} existing comparison models")
      for node in comparison_models:
        slicer.mrmlScene.RemoveNode(node)
      
      atlas_model = atlas_models[0]
      print(f"DEBUG: Using atlas model: {atlas_model.GetName()}")
      
      # Create comparison by duplicating models with different positions and colors
      comparison_count = min(3, len(specimen_models))  # Limit to 3 for clarity
      print(f"DEBUG: Creating {comparison_count} comparison pairs")
      
      created_models = []
      
      for i in range(comparison_count):
        specimen_model = specimen_models[i]
        print(f"DEBUG: Processing specimen {i+1}: {specimen_model.GetName()}")
        
        # Create "original" version (specimen as if it were original)
        original_copy = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLModelNode')
        original_copy.SetName(f"comparison_original_{i+1}")
        original_copy.SetAndObservePolyData(specimen_model.GetPolyData())
        original_copy.CreateDefaultDisplayNodes()
        
        # Create "aligned" version (atlas positioned as aligned result)
        aligned_copy = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLModelNode')
        aligned_copy.SetName(f"comparison_aligned_{i+1}")
        aligned_copy.SetAndObservePolyData(atlas_model.GetPolyData())
        aligned_copy.CreateDefaultDisplayNodes()
        
        # Create transforms for positioning
        original_transform = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLLinearTransformNode')
        original_transform.SetName(f"OriginalTransform_{i+1}")
        aligned_transform = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLLinearTransformNode')
        aligned_transform.SetName(f"AlignedTransform_{i+1}")
        
        # Get model bounds to calculate appropriate spacing
        bounds = specimen_model.GetPolyData().GetBounds()
        model_width = bounds[1] - bounds[0]  # x range
        model_height = bounds[3] - bounds[2]  # y range
        model_depth = bounds[5] - bounds[4]  # z range
        max_dimension = max(model_width, model_height, model_depth)
        
        print(f"DEBUG: Model {specimen_model.GetName()} bounds: {bounds}")
        print(f"DEBUG: Model dimensions: {model_width:.2f} x {model_height:.2f} x {model_depth:.2f}, max: {max_dimension:.2f}")
        
        # Scale the spacing to be proportional to model size
        # If models are very small (< 1 unit), use much smaller spacing
        if max_dimension < 1.0:
          # For small models, use very compact spacing
          base_spacing = max(max_dimension * 3, 0.5)  # Only 3x model size, minimum 0.5 units
          actual_spacing = max(base_spacing, spacing_factor * max_dimension * 0.01)  # Scale down user spacing significantly
        else:
          # For larger models, use the user-specified spacing
          actual_spacing = max(spacing_factor, max_dimension * 1.5)
        
        # Position models
        original_matrix = vtk.vtkMatrix4x4()
        aligned_matrix = vtk.vtkMatrix4x4()
        
        y_offset = i * actual_spacing
        # Make x-axis spacing even smaller for very small models
        if max_dimension < 1.0:
          x_offset_original = -actual_spacing * 0.8  # Closer together for small models
          x_offset_aligned = actual_spacing * 0.8
        else:
          x_offset_original = -actual_spacing * 0.6
          x_offset_aligned = actual_spacing * 0.6
        
        print(f"DEBUG: Positioning pair {i+1} - Original at ({x_offset_original:.1f}, {y_offset:.1f}, 0), Aligned at ({x_offset_aligned:.1f}, {y_offset:.1f}, 0)")
        print(f"DEBUG: Using spacing: {actual_spacing:.1f}")
        
        original_matrix.SetElement(0, 3, x_offset_original)
        original_matrix.SetElement(1, 3, y_offset)
        original_matrix.SetElement(2, 3, 0)
        
        aligned_matrix.SetElement(0, 3, x_offset_aligned)
        aligned_matrix.SetElement(1, 3, y_offset)
        aligned_matrix.SetElement(2, 3, 0)
        
        original_transform.SetMatrixTransformToParent(original_matrix)
        aligned_transform.SetMatrixTransformToParent(aligned_matrix)
        
        # Apply transforms
        original_copy.SetAndObserveTransformNodeID(original_transform.GetID())
        aligned_copy.SetAndObserveTransformNodeID(aligned_transform.GetID())
        
        # Set colors and visibility
        original_display = original_copy.GetDisplayNode()
        aligned_display = aligned_copy.GetDisplayNode()
        
        if original_display:
          original_display.SetColor(0.2, 0.4, 1.0)  # Blue for "original"
          original_display.SetVisibility(True)
          original_display.SetOpacity(1.0)
          original_display.SetBackfaceCulling(False)
          print(f"DEBUG: Set original display properties for {original_copy.GetName()}")
        else:
          print(f"WARNING: No display node for {original_copy.GetName()}")
          
        if aligned_display:
          aligned_display.SetColor(0.2, 1.0, 0.4)   # Green for "aligned"
          aligned_display.SetVisibility(True)
          aligned_display.SetOpacity(1.0)
          aligned_display.SetBackfaceCulling(False)
          print(f"DEBUG: Set aligned display properties for {aligned_copy.GetName()}")
        else:
          print(f"WARNING: No display node for {aligned_copy.GetName()}")
        
        created_models.extend([original_copy, aligned_copy])
        print(f"DEBUG: Created comparison pair {i+1}: {specimen_model.GetName()} vs Atlas")
      
      print(f"DEBUG: Created {len(created_models)} comparison models total")
      
      # Set up view layout
      print(f"DEBUG: Setting up view layout: {layout_option}")
      self.setupComparisonViewLayout(layout_option)
      
      # Reset view and center on models
      print("DEBUG: Resetting 3D views")
      slicer.app.layoutManager().resetThreeDViews()
      
      # Calculate overall bounds of all comparison models for camera positioning
      overall_bounds = [float('inf'), float('-inf'), float('inf'), float('-inf'), float('inf'), float('-inf')]
      for model in created_models:
        if model.GetPolyData() and model.GetPolyData().GetNumberOfPoints() > 0:
          bounds = model.GetPolyData().GetBounds()
          # Update overall bounds
          overall_bounds[0] = min(overall_bounds[0], bounds[0])  # min x
          overall_bounds[1] = max(overall_bounds[1], bounds[1])  # max x
          overall_bounds[2] = min(overall_bounds[2], bounds[2])  # min y
          overall_bounds[3] = max(overall_bounds[3], bounds[3])  # max y
          overall_bounds[4] = min(overall_bounds[4], bounds[4])  # min z
          overall_bounds[5] = max(overall_bounds[5], bounds[5])  # max z
      
      print(f"DEBUG: Overall scene bounds: {overall_bounds}")
      
      # Try to center the view on all models and fit to view
      try:
        # Get all 3D widgets
        layoutManager = slicer.app.layoutManager()
        threeDWidgetCount = layoutManager.threeDViewCount
        print(f"DEBUG: Found {threeDWidgetCount} 3D view widgets")
        
        for widgetIndex in range(threeDWidgetCount):
          threeDWidget = layoutManager.threeDWidget(widgetIndex)
          if threeDWidget:
            threeDView = threeDWidget.threeDView()
            
            # Calculate center point of all models
            center_x = (overall_bounds[0] + overall_bounds[1]) / 2
            center_y = (overall_bounds[2] + overall_bounds[3]) / 2
            center_z = (overall_bounds[4] + overall_bounds[5]) / 2
            
            # Calculate appropriate camera distance
            scene_width = overall_bounds[1] - overall_bounds[0]
            scene_height = overall_bounds[3] - overall_bounds[2] 
            scene_depth = overall_bounds[5] - overall_bounds[4]
            scene_size = max(scene_width, scene_height, scene_depth)
            
            # For very small models with wide spacing, ensure camera is far enough to see everything
            min_camera_distance = max(scene_width, scene_height) * 2  # At least 2x the widest dimension
            camera_distance = max(scene_size * 3, min_camera_distance, 2.0)  # Minimum 2 units away
            
            print(f"DEBUG: Scene center: ({center_x:.1f}, {center_y:.1f}, {center_z:.1f})")
            print(f"DEBUG: Scene dimensions: {scene_width:.1f} x {scene_height:.1f} x {scene_depth:.1f}")
            print(f"DEBUG: Scene size: {scene_size:.1f}, Min distance: {min_camera_distance:.1f}, Camera distance: {camera_distance:.1f}")
            
            # Set camera position
            camera = threeDView.renderWindow().GetRenderers().GetFirstRenderer().GetActiveCamera()
            camera.SetPosition(center_x, center_y - camera_distance, center_z + camera_distance * 0.5)
            camera.SetFocalPoint(center_x, center_y, center_z)
            camera.SetViewUp(0, 0, 1)
            
            # Reset and fit view
            threeDView.resetFocalPoint()
            threeDView.resetCamera()
            print(f"DEBUG: Configured camera for view {widgetIndex}")
            
      except Exception as view_error:
        print(f"DEBUG: Error configuring view: {view_error}")
        import traceback
        traceback.print_exc()
      
      # Force a scene update
      slicer.app.processEvents()
      
      # Final fallback: if models still not visible, try simple fit all
      try:
        import time
        time.sleep(0.5)  # Brief pause to let scene update
        layoutManager = slicer.app.layoutManager()
        for widgetIndex in range(layoutManager.threeDViewCount):
          threeDWidget = layoutManager.threeDWidget(widgetIndex)
          if threeDWidget:
            threeDView = threeDWidget.threeDView()
            # Try VTK's fit all functionality
            renderer = threeDView.renderWindow().GetRenderers().GetFirstRenderer()
            renderer.ResetCamera()
            print(f"DEBUG: Applied fallback camera reset for view {widgetIndex}")
      except Exception as fallback_error:
        print(f"DEBUG: Fallback camera reset failed: {fallback_error}")
      
      print(f"SUCCESS: Comparison created with {comparison_count} pairs from loaded DeCA models")
      print("Blue = Individual specimens, Green = Atlas (aligned reference)")
      return True
      
    except Exception as e:
      print(f"ERROR in createComparisonFromLoadedModels: {str(e)}")
      import traceback
      traceback.print_exc()
      return False


  def performColorClustering(self, modelNode, numClusters, method):
    """Perform color clustering analysis on model"""
    try:
      polydata = modelNode.GetPolyData()
      colorArray = polydata.GetPointData().GetScalars()
      
      if not colorArray:
        print("Model has no color data for clustering")
        return None
      
      # Convert colors to numpy array
      colors = vtk_np.vtk_to_numpy(colorArray)
      if colors.shape[1] < 3:
        print("Color data must have at least 3 components (RGB)")
        return None
      
      # Normalize colors to [0,1] if needed
      if colors.max() > 1.0:
        colors = colors / 255.0
      
      # Perform clustering
      if method.startswith("K-means"):
        init = 'k-means++' if method == "K-means++" else 'random'
        kmeans = KMeans(n_clusters=numClusters, init=init, random_state=42)
        labels = kmeans.fit_predict(colors[:, :3])
        centroids = kmeans.cluster_centers_
      elif method == "Hierarchical":
        from sklearn.cluster import AgglomerativeClustering
        clustering = AgglomerativeClustering(n_clusters=numClusters)
        labels = clustering.fit_predict(colors[:, :3])
        
        # Calculate centroids manually for hierarchical clustering
        centroids = []
        for i in range(numClusters):
          cluster_colors = colors[labels == i, :3]
          if len(cluster_colors) > 0:
            centroids.append(np.mean(cluster_colors, axis=0))
          else:
            centroids.append([0, 0, 0])
        centroids = np.array(centroids)
      
      # Calculate cluster statistics
      cluster_counts = np.bincount(labels, minlength=numClusters)
      total_points = len(labels)
      percentages = (cluster_counts / total_points) * 100
      
      # Create color-coded visualization
      clusteredColorArray = vtk.vtkUnsignedCharArray()
      clusteredColorArray.SetNumberOfComponents(3)
      clusteredColorArray.SetNumberOfTuples(total_points)
      clusteredColorArray.SetName("ClusterColors")
      
      # Assign cluster colors using a simple color scheme
      cluster_colors_rgb = [
        [255, 0, 0],    # Red
        [0, 255, 0],    # Green
        [0, 0, 255],    # Blue
        [255, 255, 0],  # Yellow
        [255, 0, 255],  # Magenta
        [0, 255, 255],  # Cyan
        [255, 128, 0],  # Orange
        [128, 0, 255],  # Purple
        [255, 192, 203], # Pink
        [128, 128, 128]  # Gray
      ]
      
      for i in range(total_points):
        cluster_id = labels[i]
        color = cluster_colors_rgb[cluster_id % len(cluster_colors_rgb)]
        clusteredColorArray.SetTuple3(i, color[0], color[1], color[2])
      
      # Apply clustered colors to model
      polydata.GetPointData().SetScalars(clusteredColorArray)
      modelNode.Modified()
      
      results = {
        'labels': labels,
        'centroids': centroids,
        'cluster_counts': cluster_counts,
        'percentages': percentages,
        'total_points': total_points,
        'method': method
      }
      
      print(f"Color clustering completed: {numClusters} clusters, {total_points} points")
      return results
      
    except Exception as e:
      print(f"Error in performColorClustering: {str(e)}")
      return None

  def generateColorDendrogram(self, modelNode, threshold):
    """Generate hierarchical clustering dendrogram of colors"""
    try:
      polydata = modelNode.GetPolyData()
      colorArray = polydata.GetPointData().GetScalars()
      
      if not colorArray:
        print("Model has no color data for dendrogram")
        return None
      
      # Convert colors to numpy and sample for performance
      colors = vtk_np.vtk_to_numpy(colorArray)
      if colors.shape[1] < 3:
        print("Color data must have at least 3 components (RGB)")
        return None
      
      # Normalize colors
      if colors.max() > 1.0:
        colors = colors / 255.0
      
      # Sample colors for dendrogram (max 1000 points for performance)
      max_samples = 1000
      if len(colors) > max_samples:
        indices = np.random.choice(len(colors), max_samples, replace=False)
        sampled_colors = colors[indices, :3]
      else:
        sampled_colors = colors[:, :3]
      
      # Compute pairwise distances
      distances = pdist(sampled_colors, metric='euclidean')
      
      # Perform hierarchical clustering
      linkage_matrix = linkage(distances, method='ward')
      
      # Create dendrogram plot
      plt.figure(figsize=(12, 8))
      dendrogram(linkage_matrix, truncate_mode='level', p=10)
      plt.title('Color Hierarchy Dendrogram')
      plt.xlabel('Sample Index')
      plt.ylabel('Distance')
      
      # Save dendrogram
      output_dir = os.path.expanduser("~/Desktop")
      dendrogram_path = os.path.join(output_dir, "color_dendrogram.png")
      plt.savefig(dendrogram_path, dpi=300, bbox_inches='tight')
      plt.close()
      
      print(f"Generated color dendrogram saved to: {dendrogram_path}")
      return dendrogram_path
      
    except Exception as e:
      print(f"Error in generateColorDendrogram: {str(e)}")
      return None


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