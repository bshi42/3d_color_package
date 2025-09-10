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

    # Set up tabs to split workflow
    tabsWidget = qt.QTabWidget()
    self.tabsWidget = tabsWidget
    DeCATab = qt.QWidget()
    DeCATabLayout = qt.QFormLayout(DeCATab)
    DeCALTab = qt.QWidget()
    DeCALTabLayout = qt.QFormLayout(DeCALTab)
    visualizeTab = qt.QWidget()
    visualizeTabLayout = qt.QFormLayout(visualizeTab)

    tabsWidget.addTab(DeCATab, "DeCA")
    tabsWidget.addTab(DeCALTab, "DeCAL")
    tabsWidget.addTab(visualizeTab, "Visualize Results")

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
    # Select model directory
    #
    self.meshDirectoryDC=ctk.ctkPathLineEdit()
    self.meshDirectoryDC.filters = ctk.ctkPathLineEdit.Dirs
    self.meshDirectoryDC.setToolTip("Select directory containing models")
    DeCAWidgetLayout.addRow("Model directory: ", self.meshDirectoryDC)

    #
    # Select landmark directory
    #
    self.landmarkDirectoryDC=ctk.ctkPathLineEdit()
    self.landmarkDirectoryDC.filters = ctk.ctkPathLineEdit.Dirs
    self.landmarkDirectoryDC.setToolTip("Select directory containing landmarks")
    DeCAWidgetLayout.addRow("Landmark directory: ", self.landmarkDirectoryDC)

    # --- Textures directory (subject PNGs) ---
    self.textureDirectoryDC = ctk.ctkPathLineEdit()
    self.textureDirectoryDC.filters = ctk.ctkPathLineEdit.Dirs
    self.textureDirectoryDC.setToolTip("Directory with subject PNG textures (file name must match subject ID).")
    DeCAWidgetLayout.addRow("Textures directory (png): ", self.textureDirectoryDC)

    # --- Blender integration ---
    self.blenderGroup = ctk.ctkCollapsibleButton()
    self.blenderGroup.text = "Blender (cleanup, UV, bake)"
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

    #
    # Select DeCA output directory
    #
    self.outputDirectoryDC=ctk.ctkPathLineEdit()
    self.outputDirectoryDC.filters = ctk.ctkPathLineEdit.Dirs
    self.outputDirectoryDC.setToolTip("Select directory for DeCA output")
    DeCAWidgetLayout.addRow("DeCA output directory: ", self.outputDirectoryDC)

    #
    # Remove scale option
    #
    self.removeScaleCheckBoxDC = qt.QCheckBox()
    self.removeScaleCheckBoxDC.checked = False
    self.removeScaleCheckBoxDC.setToolTip("If checked, DeCA alignment will include isotropic scaling.")
    DeCAWidgetLayout.addRow("Remove scale: ", self.removeScaleCheckBoxDC)

    #
    # Error checking directory option
    #
    self.writeErrorCheckBox = qt.QCheckBox()
    self.writeErrorCheckBox.checked = False
    self.writeErrorCheckBox.setToolTip("If checked, DeCA will create a directory of results for use in estimating point correspondence error.")
    DeCAWidgetLayout.addRow("Create output for error checking: ", self.writeErrorCheckBox)

    #
    # Run DeCA Button
    #
    self.applyButtonDC = qt.QPushButton("Run DeCA")
    self.applyButtonDC.toolTip = "Run non-rigid alignment"
    self.applyButtonDC.enabled = False
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
    self.landmarkDirectoryDC.connect('validInputChanged(bool)', self.onParameterSelectDC)
    self.outputDirectoryDC.connect('validInputChanged(bool)', self.onParameterSelectDC)
    self.applyButtonDC.connect('clicked(bool)', self.onDCApplyButton)
    self.textureDirectoryDC.connect('validInputChanged(bool)', self.onParameterSelectDC)
    self.blenderExeEdit.connect('validInputChanged(bool)', self.onParameterSelectDC)


    ################################### DeCAL Tab ###################################
    # ... (The DeCAL Tab code remains unchanged) ...
    # Layout within the DeCA tab
    DeCALWidget=ctk.ctkCollapsibleButton()
    DeCALWidgetLayout = qt.QFormLayout(DeCALWidget)
    DeCALWidget.text = "Dense Correspondence Landmarking"
    DeCALTabLayout.addRow(DeCALWidget)

    #
    # Select Atlas Type
    #
    DCLAtlasButtonGroup = qt.QButtonGroup(DeCALWidget)
    self.calculateAtlasOptionDCL=qt.QRadioButton()
    self.calculateAtlasOptionDCL.setChecked(True)
    DCLAtlasButtonGroup.addButton(self.calculateAtlasOptionDCL)
    self.loadAtlasOptionDCL=qt.QRadioButton()
    self.loadAtlasOptionDCL.setChecked(False)
    DCLAtlasButtonGroup.addButton(self.loadAtlasOptionDCL)
    DeCALWidgetLayout.addRow("Create atlas: ", self.calculateAtlasOptionDCL)
    DeCALWidgetLayout.addRow("Load atlas: ", self.loadAtlasOptionDCL)

    #
    # Hidden atlas options
    #
    self.atlasCollapsibleButtonDCL = ctk.ctkCollapsibleButton()
    self.atlasCollapsibleButtonDCL.text = "Load Atlas"
    self.atlasCollapsibleButtonDCL.collapsed = True
    self.atlasCollapsibleButtonDCL.enabled = False
    DeCALWidgetLayout.addRow(self.atlasCollapsibleButtonDCL)
    DeCALAtlasOptionLayout = qt.QFormLayout(self.atlasCollapsibleButtonDCL)

    #
    # Select base mesh
    #
    self.DCLBaseModelSelector = ctk.ctkPathLineEdit()
    self.DCLBaseModelSelector.filters  = ctk.ctkPathLineEdit().Files
    self.DCLBaseModelSelector.nameFilters=["Model (*.ply *.stl *.obj *.vtk *.vtp)"]
    DeCALAtlasOptionLayout.addRow("Atlas model: ", self.DCLBaseModelSelector)

    #
    # Select base landmarks
    #
    self.DCLBaseLMSelector = ctk.ctkPathLineEdit()
    self.DCLBaseLMSelector.filters  = ctk.ctkPathLineEdit().Files
    self.DCLBaseLMSelector.nameFilters=["Point set (*.fcsv *.json *.mrk.json"]
    DeCALAtlasOptionLayout.addRow("Atlas landmarks: ", self.DCLBaseLMSelector)

    #
    # Select meshes directory
    #
    self.meshDirectoryDCL=ctk.ctkPathLineEdit()
    self.meshDirectoryDCL.filters = ctk.ctkPathLineEdit.Dirs
    self.meshDirectoryDCL.setToolTip("Select directory containing models")
    DeCALWidgetLayout.addRow("Model directory: ", self.meshDirectoryDCL)

    #
    # Select landmarks directory
    #
    self.landmarkDirectoryDCL=ctk.ctkPathLineEdit()
    self.landmarkDirectoryDCL.filters = ctk.ctkPathLineEdit.Dirs
    self.landmarkDirectoryDCL.setToolTip("Select directory containing landmarks")
    DeCALWidgetLayout.addRow("Landmark directory: ", self.landmarkDirectoryDCL)

    #
    # Select DeCA output directory
    #
    self.OutputDirectoryDCL=ctk.ctkPathLineEdit()
    self.OutputDirectoryDCL.filters = ctk.ctkPathLineEdit.Dirs
    self.OutputDirectoryDCL.setToolTip("Select directory for DeCAL output")
    DeCALWidgetLayout.addRow("DeCAL output directory: ", self.OutputDirectoryDCL)

    #
    # Set spacing tolerance
    #
    self.spacingTolerance = ctk.ctkSliderWidget()
    self.spacingTolerance.singleStep = .1
    self.spacingTolerance.minimum = 0
    self.spacingTolerance.maximum = 10
    self.spacingTolerance.value = 4
    self.spacingTolerance.setToolTip("Set tolerance of spacing as a percentage of the image diagonal")
    DeCALWidgetLayout.addRow("Point density adjustment: ", self.spacingTolerance)

    #
    # Generate Atlas Button
    #
    self.getAtlasButton = qt.QPushButton("Create\\Load atlas")
    self.getAtlasButton.toolTip = "Generate a new atlas model and landmark set from data"
    self.getAtlasButton.enabled = False
    DeCALWidgetLayout.addRow(self.getAtlasButton)

    #
    # Get Subsample Rate Button
    #
    self.getPointNumberButton = qt.QPushButton("Run subsampling")
    self.getPointNumberButton.toolTip = "Get the number of output points that will be generated"
    self.getPointNumberButton.enabled = False
    DeCALWidgetLayout.addRow(self.getPointNumberButton)

    #
    # Apply Button
    #
    self.DCLApplyButton = qt.QPushButton("Run DeCAL")
    self.DCLApplyButton.toolTip = "Generate a set of corresponding landmarks"
    self.DCLApplyButton.enabled = False
    DeCALWidgetLayout.addRow(self.DCLApplyButton)

    #
    # Log Information
    #
    self.logInfoDCL = qt.QPlainTextEdit()
    self.logInfoDCL.setPlaceholderText("DeCAL log information")
    self.logInfoDCL.setReadOnly(True)
    DeCALWidgetLayout.addRow(self.logInfoDCL)

    #
    # Subsetting menu
    #
    self.subsetCollapsibleButton = ctk.ctkCollapsibleButton()
    self.subsetCollapsibleButton.text = "Subset output points"
    self.subsetCollapsibleButton.collapsed = True
    self.subsetCollapsibleButton.enabled = True
    DeCALWidgetLayout.addRow(self.subsetCollapsibleButton)
    DeCALSubsetLayout = qt.QFormLayout(self.subsetCollapsibleButton)

    #
    # Select landmark node
    #
    self.pointSelection = slicer.qMRMLNodeComboBox()
    self.pointSelection.nodeTypes = (("vtkMRMLMarkupsFiducialNode"), "")
    self.pointSelection.setToolTip("Atlas landmarks with subset points selected")
    self.pointSelection.selectNodeUponCreation = False
    self.pointSelection.noneEnabled = True
    self.pointSelection.addEnabled = False
    self.pointSelection.removeEnabled = False
    self.pointSelection.showHidden = False
    self.pointSelection.setMRMLScene(slicer.mrmlScene)
    DeCALSubsetLayout.addRow("Atlas landmarks: ", self.pointSelection)

    #
    # Select DeCAL output directory
    #
    self.DCLLandmarkDirectory=ctk.ctkPathLineEdit()
    self.DCLLandmarkDirectory.filters = ctk.ctkPathLineEdit.Dirs
    self.DCLLandmarkDirectory.setToolTip("Select directory for DeCAL sampled landmarks to subset")
    DeCALSubsetLayout.addRow("DeCAL landmark directory: ", self.DCLLandmarkDirectory)

    #
    # Apply Subsetting Button
    #
    self.subsetApplyButton = qt.QPushButton("Run subsetting")
    self.subsetApplyButton.toolTip = "Generate a subset of corresponding landmarks"
    self.subsetApplyButton.enabled = False
    DeCALSubsetLayout.addRow(self.subsetApplyButton)

    # connections
    self.calculateAtlasOptionDCL.connect('toggled(bool)', self.onToggleAtlasDCL)
    self.loadAtlasOptionDCL.connect('toggled(bool)', self.onToggleAtlasDCL)
    self.DCLBaseModelSelector.connect('validInputChanged(bool)', self.onParameterSelectDCL)
    self.DCLBaseLMSelector.connect('validInputChanged(bool)', self.onParameterSelectDCL)
    self.meshDirectoryDCL.connect('validInputChanged(bool)', self.onParameterSelectDCL)
    self.landmarkDirectoryDCL.connect('validInputChanged(bool)', self.onParameterSelectDCL)
    self.OutputDirectoryDCL.connect('validInputChanged(bool)', self.onParameterSelectDCL)
    self.getAtlasButton.connect('clicked(bool)', self.onGenerateAtlasButton)
    self.getPointNumberButton.connect('clicked(bool)', self.onGetPointNumberButton)
    self.DCLApplyButton.connect('clicked(bool)', self.onDCLApplyButton)
    self.subsetApplyButton.connect('clicked(bool)', self.onSubsetApplyButton)
    self.pointSelection.connect('currentNodeChanged(vtkMRMLNode*)', self.onPointSelectionSelect)
    self.DCLLandmarkDirectory.connect('validInputChanged(bool)', self.onDCLLandmarkDirectorySelect)

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

  ################################### GUI SUpport Functions
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
      subjectID = self.subjectIDBox.currentText
      self.resultNode.GetDisplayNode().SetActiveScalarName(subjectID)
      self.resultNode.GetDisplayNode().SetAndObserveColorNodeID('vtkMRMLColorTableNodeFilePlasma.txt')
      print(subjectID)
    except:
      print("Error: No array found")

  def onVisualizeMeshSelect(self):
    # This function is part of the original Heatmap mode and is unchanged
    if bool(self.meshSelect.currentNode()):
      self.resultNode = self.meshSelect.currentNode()
      self.resultNode.GetDisplayNode().SetVisibility(True)
      self.resultNode.GetDisplayNode().SetScalarVisibility(True)
      resultData = self.resultNode.GetPolyData().GetPointData()
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

  def generateNewAtlas(self, removeScale, log):
    logic = InterDeCALogic()

    # getClosestToMeanPath now returns a SUBJECT BASENAME (no extension)
    subjectID = logic.getClosestToMeanPath(self.folderNames['originalLMs'])
    log.appendPlainText(f"Closest sample to mean: {subjectID}")

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
      self._hideMarkupsForVisualization(remove=False) 
      self.updateBakedPreviewList()
      self._ensureModelsAreVisible()

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

    # Folders
    self.folderNames = self.setUpDeCADir(self.outputDirectoryDC.currentPath, symmetryOption, writeErrorOption, False, loadAtlasOption)
    if not self.folderNames:
      self.logInfoDC.appendPlainText(f'Output folders could not be created in {self.outputDirectoryDC.currentPath}')
      return
    self.folderNames['originalLMs']  = self.landmarkDirectoryDC.currentPath
    self.folderNames['originalModels'] = self.meshDirectoryDC.currentPath
    self.lastDeCAAlignedModelsPath = self.folderNames['resampledModels']  # for Visualize tab

    # ---- 1) Load or compute atlas (Slicer) ----
    if loadAtlasOption:
      try:
        self.atlasModel = slicer.util.loadModel(self.DCBaseModelSelector.currentPath)
      except Exception:
        self.logInfoDC.appendPlainText(f"Can't load model from: {self.DCBaseModelSelector.currentPath}")
        return
      try:
        self.atlasLMs = slicer.util.loadMarkups(self.DCBaseLMSelector.currentPath)
      except Exception:
        self.logInfoDC.appendPlainText(f"Can't load landmarks from: {self.DCBaseLMSelector.currentPath}")
        return
    else:
      self.atlasModel, self.atlasLMs = self.generateNewAtlas(removeScaleOption, self.logInfoDC)

    # Save an intermediate atlas file (RAS) so Blender can read it
    atlas_preuv_obj = os.path.join(self.folderNames['output'], 'decaAtlas_preUV.obj')
    logic._save_model_with_cs(self.atlasModel, atlas_preuv_obj, 'RAS')

    # ---- 2) Blender cleanup + Smart UV ----
    blender_exe    = self.blenderExeEdit.currentPath
    merge_dist     = float(self.blMergeDistSpin.value)
    smart_angle    = float(self.blSmartAngleSpin.value)
    island_margin  = float(self.blIslandMarginSpin.value)
    if not (os.path.isfile(blender_exe) or os.access(blender_exe, os.X_OK)):
      self.logInfoDC.appendPlainText("Blender path not set or invalid; cannot run cleanup/UV/bake.")
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
    try:
      self.logInfoDC.appendPlainText("Rigid alignment to atlas")
      logic.runAlign(self.atlasModel, self.atlasLMs,
                     self.folderNames['originalModels'], self.folderNames['originalLMs'],
                     self.folderNames['alignedModels'], self.folderNames['alignedLMs'],
                     removeScaleOption)
    except ValueError as errorText:
      self.logInfoDC.appendPlainText(str(errorText))
      return

    # ---- 4) DeCA resampling (Slicer). Also create OBJ copies that reuse atlas UV (for Blender bake) ----
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
      return

    # ---- 5) Blender bake (selection→active) from aligned → resampled(OBJ with atlas UV) ----
    self.lastBakedTexturesPath = os.path.join(self.folderNames['output'], "atlasTextures")
    os.makedirs(self.lastBakedTexturesPath, exist_ok=True)

    texturesDir = self.textureDirectoryDC.currentPath
    if os.path.isdir(texturesDir):
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
        logic._calculate_average_texture(self.lastBakedTexturesPath)
        self.logInfoDC.appendPlainText(f"Baked {len(made)} textures to {self.lastBakedTexturesPath}")
      except Exception as e:
        self.logInfoDC.appendPlainText(f"Blender baking failed: {e}")
    else:
      self.logInfoDC.appendPlainText("No textures directory set → skipping bake.")

    # ---- 6) Fill Visualize dropdown ----
    self.updateBakedPreviewList()


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

  def _ensureModelsAreVisible(self):
    for m in slicer.util.getNodesByClass('vtkMRMLModelNode'):
      try:
        dn = m.GetDisplayNode()
        if dn: dn.SetVisibility(True)
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

  def downsampleModel(self, model, spacingPercentage):
    points=model.GetPolyData()
    cleanFilter=vtk.vtkCleanPolyData()
    cleanFilter.SetToleranceIsAbsolute(False)
    cleanFilter.SetTolerance(spacingPercentage)
    cleanFilter.SetInputData(points)
    cleanFilter.Update()
    return cleanFilter.GetOutput()

  def addIndexArray(self, mesh, arrayName):
    # Array of original index values
    indexArray = vtk.vtkIntArray()
    indexArray.SetNumberOfComponents(1)
    indexArray.SetName(arrayName)
    for i in range(mesh.GetPolyData().GetNumberOfPoints()):
      indexArray.InsertNextValue(i)
    mesh.GetPolyData().GetPointData().AddArray(indexArray)

  def computeNormals(self, inputModel):
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

  def getLandmarkFileByID(self, directory, subjectID):
    fileList = os.listdir(directory)
    for fileName in fileList:
      fileNameBase = Path(fileName)
      while fileNameBase.suffix in {'.fcsv', '.mrk', '.json'}:
        fileNameBase = fileNameBase.with_suffix('')
      if subjectID == str(fileNameBase):
        # if file with this subject id exists, load into scene
        filePath = os.path.join(directory, fileName)
        currentNode = slicer.util.loadMarkups(filePath)
        return currentNode

  def getModelFileByID(self, directory, subjectID):
    fileList = os.listdir(directory)
    for fileName in fileList:
      fileNameBase = Path(fileName).stem
      if str(subjectID) == str(fileNameBase):
        filePath = os.path.join(directory, fileName)
        currentNode = self._load_model_with_cs(filePath, 'RAS')
        return currentNode

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

  def distanceMatrix(self, a):
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

  def numpyToFiducialNode(self, numpyArray, nodeName):
    fiducialNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsFiducialNode',nodeName)
    for index in range(len(numpyArray)):
      fiducialNode.AddControlPoint(numpyArray[index], str(index))
    return fiducialNode

  def computeAverageLM(self, fiducialGroup):
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

  def fiducialNodeToPolyData(self, nodeLocation, loadOption=True):
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

  def procrustesImposition(self, originalLandmarks, sizeOption):
    procrustesFilter = vtk.vtkProcrustesAlignmentFilter()
    if(sizeOption):
      procrustesFilter.GetLandmarkTransform().SetModeToRigidBody()

    procrustesFilter.SetInputData(originalLandmarks)
    procrustesFilter.Update()
    meanShape = procrustesFilter.GetMeanPoints()
    return [meanShape, procrustesFilter.GetOutput()]

  def getClosestToMeanIndex(self, meanShape, alignedPoints):
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

  def getClosestToMeanPath(self, landmarkDirectory):
    lmNames, landmarks = self.importLandmarks(landmarkDirectory)
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

  def convertPointsToVTK(self, points):
    array_vtk = vtk_np.numpy_to_vtk(points, deep=True, array_type=vtk.VTK_FLOAT)
    points_vtk = vtk.vtkPoints()
    points_vtk.SetData(array_vtk)
    polydata_vtk = vtk.vtkPolyData()
    polydata_vtk.SetPoints(points_vtk)
    return polydata_vtk

  def computeAverageModelFromGroup(self, denseCorrespondenceGroup, baseIndex):
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

  def addMagnitudeFeature(self, denseCorrespondenceGroup, modelNameArray, model):
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

  def addMagnitudeFeatureSymmetry(self, denseCorrespondenceGroup, denseCorrespondenceGroupMirror, modelNameArray, model):
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