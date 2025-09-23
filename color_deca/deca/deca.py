# -*- coding: utf-8 -*-
"""
DeCA (Dense Correspondence Analysis) module for 3D Slicer.

Provides workflows for establishing dense point correspondences between 3D models
using landmark-based registration and non-rigid alignment. Supports shape analysis,
symmetry analysis, and texture-based segmentation for biological specimens.

Key features:
- Atlas-based correspondence mapping
- Dense landmarking with subsampling control
- Symmetric structure analysis
- Texture integration for color-based analysis

Original development in Colab:
    https://colab.research.google.com/drive/1I5HVk7KOmgTSesAYsK4UDC3BTmlvACkx
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

#
# DeCA
#

class deca(ScriptedLoadableModule):
  """
  Module class for Dense Correspondence Analysis (DeCA) in 3D Slicer.

  Implements morphometric analysis workflows for biological specimens,
  providing tools for establishing point correspondences across samples
  and performing shape/symmetry analyses.

  Base class documentation:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, parent):
    """
    Initializes the DeCA module with metadata and configuration.

    Args:
      parent: Parent object from 3D Slicer framework
    """
    ScriptedLoadableModule.__init__(self, parent)

    # Sets module metadata for 3D Slicer's module browser
    self.parent.title = "DeCA"  # Dense Correspondence Analysis
    self.parent.categories = ["SlicerMorph.DeCA Toolbox"]
    self.parent.dependencies = []  # No external module dependencies
    self.parent.contributors = ["Sara Rolfe (SCRI)"]

    # Provides user-facing documentation
    self.parent.helpText = """
      This module provides several flexible workflows for finding and analyzing dense correspondence points between models.
      """
    self.parent.helpText += self.getDefaultModuleDocumentationLink()

    # Acknowledges funding sources
    self.parent.acknowledgementText = """This extension was developed by funding from National Institutes of Health (OD032627 and HD104435) to A. Murat Maga (SCRI)
      """

#
# DeCAWidget
#

class decaWidget(ScriptedLoadableModuleWidget):
  """
  GUI widget for the DeCA module.

  Creates and manages the user interface for dense correspondence analysis,
  including tabs for DeCA, DeCAL (Dense Correspondence Landmarking), and
  visualization of results. Handles user interactions and workflow management.

  Base class documentation:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def setup(self):
    """
    Builds the module's user interface.

    Creates a tabbed interface with three main workflows:
    - DeCA: Dense correspondence analysis with shape/symmetry options
    - DeCAL: Dense correspondence landmarking with point sampling
    - Visualize: Result visualization with heat maps
    """
    ScriptedLoadableModuleWidget.setup(self)

    # Sets up tabs to split workflow into logical sections
    tabsWidget = qt.QTabWidget()
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
    # Layout within the DeCA tab
    # Creates main collapsible section for DeCA inputs/outputs
    decaWidget=ctk.ctkCollapsibleButton()
    DeCAWidgetLayout = qt.QFormLayout(decaWidget)
    decaWidget.text = "Dense Correspondence I/O"  # Input/Output configuration
    DeCATabLayout.addRow(decaWidget)

    #
    # Select Atlas Type
    #
    # Creates radio buttons for atlas selection mode
    self.calculateAtlasOptionDC=qt.QRadioButton()
    self.calculateAtlasOptionDC.setChecked(True)  # Default: generate new atlas
    self.loadAtlasOptionDC=qt.QRadioButton()

    # Groups radio buttons for exclusive selection
    DCAtlasButtonGroup = qt.QButtonGroup(decaWidget)
    DCAtlasButtonGroup.addButton(self.calculateAtlasOptionDC)
    DCAtlasButtonGroup.addButton(self.loadAtlasOptionDC)

    DeCAWidgetLayout.addRow("Create atlas: ", self.calculateAtlasOptionDC)
    DeCAWidgetLayout.addRow("Load atlas: ", self.loadAtlasOptionDC)

    #
    # Hidden atlas options
    # Creates collapsible section for atlas loading parameters
    self.atlasCollapsibleButtonDC = ctk.ctkCollapsibleButton()
    self.atlasCollapsibleButtonDC.text = "Atlas Options"
    self.atlasCollapsibleButtonDC.collapsed = True  # Initially hidden
    self.atlasCollapsibleButtonDC.enabled = False  # Disabled until load option selected
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
    # Creates radio buttons for analysis mode selection
    self.analysisTypeShape=qt.QRadioButton()
    self.analysisTypeShape.setChecked(True)  # Default: shape analysis
    self.analysisTypeSymmetry=qt.QRadioButton()

    # Groups for exclusive selection
    DCAnalysisButtonGroup = qt.QButtonGroup(decaWidget)
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
    # NEW: Texture directory selection
    # Enables texture-based analysis for colored specimens
    self.textureDirectoryDC=ctk.ctkPathLineEdit()
    self.textureDirectoryDC.filters = ctk.ctkPathLineEdit.Dirs
    self.textureDirectoryDC.setToolTip("Select directory containing texture images")
    DeCAWidgetLayout.addRow("Texture directory: ", self.textureDirectoryDC)
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
    # Controls whether to normalize for size differences
    self.removeScaleCheckBoxDC = qt.QCheckBox()
    self.removeScaleCheckBoxDC.checked = False  # Default: preserve scale
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

    ################################### DeCAL Tab ###################################
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
    
    #NEW
    self.textureDirectoryDCL=ctk.ctkPathLineEdit()
    self.textureDirectoryDCL.filters = ctk.ctkPathLineEdit.Dirs
    self.textureDirectoryDCL.setToolTip("Select directory containing landmarks")
    DeCALWidgetLayout.addRow("Landmark directory: ", self.textureDirectoryDCL)
    #NEW
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
    self.getAtlasButton = qt.QPushButton("Create\Load atlas")
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
    visualizeWidget.text = "Visualize the output feature heat maps"
    visualizeTabLayout.addRow(visualizeWidget)

    #
    # Select output model
    #
    self.meshSelect = slicer.qMRMLNodeComboBox()
    self.meshSelect.nodeTypes = (("vtkMRMLModelNode"), "")
    self.meshSelect.setToolTip("Select model node with result arrays")
    self.meshSelect.selectNodeUponCreation = False
    self.meshSelect.noneEnabled = True
    self.meshSelect.addEnabled = False
    self.meshSelect.removeEnabled = False
    self.meshSelect.showHidden = False
    self.meshSelect.setMRMLScene(slicer.mrmlScene)
    visualizeWidgetLayout.addRow("Result Model: ", self.meshSelect)

    #
    # Select Subject ID
    #
    self.subjectIDBox=qt.QComboBox()
    self.subjectIDBox.enabled = False
    visualizeWidgetLayout.addRow("Subject ID: ", self.subjectIDBox)

    # Connections
    self.meshSelect.connect("currentNodeChanged(vtkMRMLNode*)", self.onVisualizeMeshSelect)
    self.subjectIDBox.connect("currentIndexChanged(int)", self.onSubjectIDSelect)

  ################################### GUI SUpport Functions
  def setUpDeCADir(self, outDir, symmetryOption=False, errorDirectoryOption=False, DeCALOption=False, loadAtlasOption = False):
    """
    Creates directory structure for DeCA output.

    Generates timestamped output folders organized by data type,
    ensuring clean separation of results and intermediate files.

    Args:
      outDir: Base output directory path
      symmetryOption: Whether to create mirror data folders
      errorDirectoryOption: Whether to create error checking folder
      DeCALOption: Whether to create DeCAL output folder
      loadAtlasOption: Whether atlas is pre-loaded (skips temp folders)

    Returns:
      Dictionary mapping folder types to their paths
    """
    # Creates unique timestamped folder for this analysis
    dateTimeStamp = datetime.now().strftime('%Y_%m-%d_%H_%M_%S')
    outputFolderDC = os.path.join(outDir, dateTimeStamp)
    fileNameDictionary = {}

    try:
      # Creates main output structure
      os.makedirs(outputFolderDC)

      # Creates folders for aligned data
      alignedLMFolderDC = os.path.join(outputFolderDC, "alignedLMs")
      os.makedirs(alignedLMFolderDC)
      alignedModelFolderDC = os.path.join(outputFolderDC, "alignedModels")
      os.makedirs(alignedModelFolderDC)
      # initialize the filename dictionary
      fileNameDictionary['output'] = str(outputFolderDC)
      fileNameDictionary['alignedLMs'] = str(alignedLMFolderDC)
      fileNameDictionary['alignedModels'] = str(alignedModelFolderDC)
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
    """
    Handles analysis type toggle between shape and symmetry.

    Shows or hides symmetry-specific options based on selection.
    """
    if self.analysisTypeSymmetry.checked == True:
      # Shows symmetry options when symmetry analysis selected
      self.symmetryCollapsibleButton.collapsed = False
      self.symmetryCollapsibleButton.enabled = True
    else:
      # Hides symmetry options for shape analysis
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

  def onSubjectIDSelect(self):
    try:
      subjectID = self.subjectIDBox.currentText
      self.resultNode.GetDisplayNode().SetActiveScalarName(subjectID)
      self.resultNode.GetDisplayNode().SetAndObserveColorNodeID('vtkMRMLColorTableNodeFilePlasma.txt')
      print(subjectID)
    except:
      print("Error: No array found")

  def onVisualizeMeshSelect(self):
    if bool(self.meshSelect.currentNode()):
      self.resultNode = self.meshSelect.currentNode()
      self.resultNode.GetDisplayNode().SetVisibility(True)
      self.resultNode.GetDisplayNode().SetScalarVisibility(True)
      resultData = self.resultNode.GetPolyData().GetPointData()
      self.subjectIDBox.enabled = True
      arrayNumber = resultData.GetNumberOfArrays()
      if arrayNumber > 0:
        for i in range(resultData.GetNumberOfArrays()):
          arrayName = resultData.GetArrayName(i)
          self.subjectIDBox.addItem(arrayName)
      else:
        self.subjectIDBox.clear()
        self.subjectIDBox.enabled = False
  #NEW
  def onParameterSelectDC(self):
    """
    Updates DeCA button state based on parameter selection.

    Enables the run button only when all required inputs are provided.
    Validates atlas source and directory paths.
    """
    # Checks if atlas is available (either loaded or will be generated)
    atlasPathSelected = bool(self.DCBaseModelSelector.currentPath and self.DCBaseLMSelector.currentPath) or self.calculateAtlasOptionDC.checked
    # Verifies all required directories are selected
    inputPathsSelected = bool(self.meshDirectoryDC.currentPath and self.landmarkDirectoryDC.currentPath and self.outputDirectoryDC.currentPath and self.textureDirectoryDC.currentPath)
    # Enables button only when all requirements met
    self.applyButtonDC.enabled = bool(atlasPathSelected and inputPathsSelected)
  #NEW
  def onParameterSelectDCL(self):
    atlasPathSelected = bool(self.DCLBaseModelSelector.currentPath and self.DCLBaseLMSelector.currentPath) or self.calculateAtlasOptionDCL.checked
    inputPathsSelected = bool(self.meshDirectoryDCL.currentPath and self.landmarkDirectoryDCL.currentPath and self.OutputDirectoryDCL.currentPath)
    self.getAtlasButton.enabled = bool(atlasPathSelected and inputPathsSelected)

  def onPointSelectionSelect(self):
    self.subsetApplyButton.enabled = bool(self.DCLLandmarkDirectory.currentPath and self.pointSelection.currentNode())

  def onDCLLandmarkDirectorySelect(self):
    self.subsetApplyButton.enabled = bool(self.DCLLandmarkDirectory.currentPath and self.pointSelection.currentNode())

  def onGenerateAtlasButton(self):
    # Instantiates DeCA logic processor for atlas generation algorithms
    logic = decaLogic()

    # Creates timestamped output directory structure for DeCAL analysis results
    # Parameters control symmetry (False), error checking (False), DeCAL mode (True), and atlas loading
    self.folderNames = self.setUpDeCADir(self.OutputDirectoryDCL.currentPath, False, False, True, self.loadAtlasOptionDCL.checked)
    # Validates directory creation succeeded before continuing
    if self.folderNames == {}:
      # Logs error message to GUI when folder creation fails
      self.logInfoDCL.appendPlainText(f'Output folders could not be created in {self.OutputDirectoryDCL.currentPath}')
      return  # Exits early if setup failed
    # Stores source directory paths in the folder name dictionary
    self.folderNames['originalLMs'] = self.landmarkDirectoryDCL.currentPath  # Landmark files location
    self.folderNames['originalModels'] = self.meshDirectoryDCL.currentPath  # 3D model files location
    self.folderNames['originalTextures']=self.textureDirectoryDCL.currentPath  # Texture image files location
    # Handles pre-existing atlas loading workflow
    if self.loadAtlasOptionDCL.checked:
      try:
        # Retrieves user-specified atlas model path from GUI
        atlasModelPath = self.DCLBaseModelSelector.currentPath
        # Loads atlas model into 3D Slicer scene
        self.atlasModel = slicer.util.loadModel(atlasModelPath)
      except:
        # Reports model loading failure to user
        self.logInfoDCL.appendPlainText(f"Can't load model from: {atlasModelPath}")
        return  # Aborts if atlas model can't be loaded
      try:
        # Retrieves user-specified atlas landmark path from GUI
        atlasLMPath = self.DCLBaseLMSelector.currentPath
        # Loads atlas landmarks into 3D Slicer scene
        self.atlasLMs = slicer.util.loadMarkups(atlasLMPath)
      except:
        # Prints debug message to console
        print("Can't load from: ", atlasLMPath)
        # Reports landmark loading failure to user
        self.logInfoDCL.appendPlainText(f"Can't load landmarks from: {atlasLMPath}")
        return  # Aborts if atlas landmarks can't be loaded
    else:
      # Generates new atlas from input specimens
      removeScale = True  # Normalizes for size differences across specimens
      # Creates unbiased atlas from all specimens in the dataset
      self.atlasModel, self.atlasLMs = self.generateNewAtlas(removeScale, self.logInfoDCL)
    # Constructs output path for atlas model file
    atlasModelPath = os.path.join(self.folderNames['output'], 'decaAtlasModel.ply')
    # Notifies user of save operation
    self.logInfoDCL.appendPlainText(f"Saving atlas model to {atlasModelPath}")
    # Writes atlas model to disk in PLY format
    slicer.util.saveNode(self.atlasModel, atlasModelPath)

    # Constructs output path for atlas landmark file
    atlasLMPath = os.path.join(self.folderNames['output'], 'decaAtlasLM.mrk.json')
    # Notifies user of save operation
    self.logInfoDCL.appendPlainText(f"Saving atlas landmarks to {atlasLMPath}")
    # Writes atlas landmarks to disk in Slicer's JSON format
    slicer.util.saveNode(self.atlasLMs, atlasLMPath)

    # Enables point number calculation button now that atlas is ready
    self.getPointNumberButton.enabled = True

  def generateNewAtlas(self, removeScale, log):
    """
    Generates a new atlas from input specimens.

    Finds the specimen closest to the mean shape, aligns all specimens
    to it, then computes the average to create an unbiased atlas.

    Args:
      removeScale: Whether to normalize for size differences
      log: Text widget for progress logging

    Returns:
      Tuple of (atlas model node, atlas landmark node)
    """
    logic = decaLogic()

    # Identifies specimen closest to mean configuration
    closestToMeanLandmarkPath = logic.getClosestToMeanPath(self.folderNames['originalLMs'])
    tempBaseLMs = slicer.util.loadMarkups(os.path.join(self.folderNames['originalLMs'],closestToMeanLandmarkPath))

    # Extracts subject ID from filename
    subjectID = Path(closestToMeanLandmarkPath)
    while subjectID.suffix in {'.fcsv', '.mrk', '.json'}:
      subjectID = subjectID.with_suffix('')
    log.appendPlainText(f"Closest sample to mean: {subjectID}")

    # Loads corresponding model
    tempBaseModel = logic.getModelFileByID(self.folderNames['originalModels'], subjectID)
    log.appendPlainText(f"Rigid Alignment to: {subjectID}")
    logic.runAlign(tempBaseModel, tempBaseLMs, self.folderNames['originalModels'], self.folderNames['originalLMs'], self.folderNames['tempAlignedModels'], self.folderNames['tempAlignedLMs'], removeScale)
    self.logInfoDCL.appendPlainText(f"Generating the average template")
    atlasModel, atlasLMs = logic.runMean(self.folderNames['tempAlignedLMs'], self.folderNames['tempAlignedModels'],self.folderNames['originalTextures'])
    slicer.mrmlScene.RemoveNode(tempBaseModel)
    slicer.mrmlScene.RemoveNode(tempBaseLMs)
    shutil.rmtree(self.folderNames['tempAlignedModels'])
    shutil.rmtree(self.folderNames['tempAlignedLMs'])
    return atlasModel, atlasLMs
#need to set up indexing correctly for the code to run follow down from runMean
  def onGetPointNumberButton(self):
    # Instantiates logic processor for point calculations
    logic = decaLogic()

    # Performs downsampling on atlas model based on user-specified spacing
    # Returns both the downsampled mesh and the resulting point count
    subsampledTemplate, pointNumber = logic.runCheckPoints(self.atlasModel, self.spacingTolerance.value)

    # Reports point count to user for planning computational requirements
    self.logInfoDCL.appendPlainText(f'The subsampled template has a total of {pointNumber} points.')

    # Enables DeCAL execution now that point density is confirmed
    self.DCLApplyButton.enabled = True

  def onDCApplyButton(self):
    """
    Handles DeCA execution button click.

    Orchestrates the full DeCA workflow including atlas generation/loading,
    specimen alignment, and correspondence computation based on selected
    analysis type (shape or symmetry).
    """
    logic = decaLogic()

    # Gathers user-selected options
    symmetryOption = self.analysisTypeSymmetry.checked
    writeErrorOption = self.writeErrorCheckBox.checked
    loadAtlasOption = self.loadAtlasOptionDC.checked
    removeScaleOption = self.removeScaleCheckBoxDC.checked

    # Sets up output directory structure
    self.folderNames = self.setUpDeCADir(self.outputDirectoryDC.currentPath, symmetryOption, writeErrorOption, False, loadAtlasOption)
    if self.folderNames == {}:
      self.logInfoDC.appendPlainText(f'Output folders could not be created in {self.outputDirectoryDC.currentPath}')
      return
    self.folderNames['originalLMs'] = self.landmarkDirectoryDC.currentPath
    self.folderNames['originalModels'] = self.meshDirectoryDC.currentPath
    #generate or load atlas
    if loadAtlasOption:
      try:
        atlasModelPath = self.DCBaseModelSelector.currentPath
        self.atlasModel = slicer.util.loadModel(atlasModelPath)
      except:
        self.logInfoDC.appendPlainText(f"Can't load model from: {atlasModelPath}")
        return
      try:
        atlasLMPath = self.DCBaseLMSelector.currentPath
        self.atlasLMs = slicer.util.loadMarkups(atlasLMPath)
      except:
        print("Can't load from: ", atlasLMPath)
        self.logInfoDC.appendPlainText(f"Can't load landmarks from: {atlasLMPath}")
        return
    else:
      atlasModel, atlasLMs = self.generateNewAtlas(removeScaleOption, self.logInfoDC)
    # save atlas model and landmarks to output file
    atlasModelPath = os.path.join(self.folderNames['output'], 'decaAtlasModel.ply')
    self.logInfoDC.appendPlainText(f"Saving atlas model to {atlasModelPath}")
    slicer.util.saveNode(atlasModel, atlasModelPath)
    atlasLMPath = os.path.join(self.folderNames['output'], 'decaAtlasLM.mrk.json')
    self.logInfoDC.appendPlainText(f"Saving atlas landmarks to {atlasLMPath}")
    slicer.util.saveNode(atlasLMs, atlasLMPath)
    # rigid alignment to atlas
    logic.runAlign(atlasModel, atlasLMs, self.folderNames['originalModels'], self.folderNames['originalLMs'],
    self.folderNames['alignedModels'], self.folderNames['alignedLMs'], removeScaleOption)
    # run DeCA shape analysis
    #NEW
    if self.analysisTypeShape.checked:
      self.logInfoDCL.appendPlainText(f"Calculating point correspondences to atlas")
      logic.runDCAlign(atlasModelPath, atlasLMPath, self.folderNames['alignedModels'],
      self.folderNames['alignedLMs'], self.folderNames['output'], self.writeErrorCheckBox.checked, self.textureDirectoryDC.currentPath)
    #NEW
    # run DeCA symmetry analysis
    else:
      # generate mirrored landmarks and models
      axis = [-1,1,1] #set symmetry to x-axis
      self.logInfoDC.appendPlainText(f"Generating mirrored models and landmarks")
      logic.runMirroring(self.folderNames['alignedModels'], self.folderNames['alignedLMs'], self.folderNames['mirrorModels'],
      self.folderNames['mirrorLMs'], axis, self.landmarkIndexText.text)
      self.logInfoDCL.appendPlainText(f"Calculating point correspondences to atlas")
      logic.runDCAlignSymmetric(atlasModelPath, atlasLMPath, self.folderNames['alignedModels'],
      self.folderNames['alignedLMs'], self.folderNames['mirrorModels'], self.folderNames['mirrorLMs'], self.folderNames['output'],
      self.writeErrorCheckBox.checked)
    slicer.mrmlScene.RemoveNode(atlasModel)
    slicer.mrmlScene.RemoveNode(atlasLMs)

  def onDCLApplyButton(self):
    logic = decaLogic()
    # rigidly align to template
    self.logInfoDCL.appendPlainText(f"Rigid alignment to the atlas")
    removeScale = True
    logic.runAlign(self.atlasModel, self.atlasLMs, self.folderNames['originalModels'], self.folderNames['originalLMs'],
    self.folderNames['alignedModels'], self.folderNames['alignedLMs'], removeScale)
    # generate point correspondences
    self.logInfoDCL.appendPlainText(f"Calculating point correspondences")
    atlasDenseLandmarks = logic.runDeCAL(self.atlasModel, self.atlasLMs, self.folderNames['alignedModels'],
    self.folderNames['alignedLMs'], self.folderNames['DeCALOutput'], self.spacingTolerance.value)
    # setup for optional subsetting
    self.pointSelection.setCurrentNode(atlasDenseLandmarks)
    self.DCLLandmarkDirectory.setCurrentPath(self.folderNames['DeCALOutput'])

  def onSubsetApplyButton(self):
    logic = decaLogic()
    topDir = os.path.dirname(self.DCLLandmarkDirectory.currentPath)
    lmDirectorySubset = os.path.join(topDir, "DeCALSubset")
    os.makedirs(lmDirectorySubset)
    atlasNode = self.pointSelection.currentNode()
    lmDirectorySubset = logic.runSubsetLandmarks(atlasNode, self.DCLLandmarkDirectory.currentPath, lmDirectorySubset)

#
# DeCALogic
#

class decaLogic(ScriptedLoadableModuleLogic):
  """
  Processing logic for dense correspondence analysis.

  Implements core algorithms for atlas generation, specimen alignment,
  point correspondence computation, and morphometric analysis. Designed
  to be callable independently of the GUI for batch processing.

  Key methods:
  - Atlas generation and alignment
  - Dense correspondence mapping
  - Symmetry analysis
  - Texture-based feature extraction

  Base class documentation:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """
  def runSubsetLandmarks(self, baseNode, lmDirectory, lmDirectorySubset):
    """
    Subsets landmark sets based on selected points.

    Removes unselected landmarks from all specimens to create a
    reduced landmark set for focused analysis.

    Args:
      baseNode: Atlas landmarks with points selected/unselected
      lmDirectory: Source directory with full landmark sets
      lmDirectorySubset: Output directory for subset landmarks
    """
    # Identifies points to remove (unselected ones)
    deletionIndex = []
    for i in range(baseNode.GetNumberOfControlPoints()):
      if not baseNode.GetNthControlPointSelected(i):
        deletionIndex.append(i)

    # Processes each landmark file
    for lmFileName in os.listdir(lmDirectory):
      if(not lmFileName.startswith(".")):
        currentLMNode = slicer.util.loadMarkups(os.path.join(lmDirectory, lmFileName))
        # Removes points in reverse order to preserve indices
        for index in reversed(deletionIndex):
          currentLMNode.RemoveNthControlPoint(index)
      # Saves subset landmarks
      slicer.util.saveNode(currentLMNode, os.path.join(lmDirectorySubset, lmFileName))
      slicer.mrmlScene.RemoveNode(currentLMNode)

  def runCheckPoints(self, atlasNode, spacingTolerance):
    # Converts user-specified tolerance value to percentage for VTK filter
    spacingPercentage = spacingTolerance/100

    # Applies downsampling filter to reduce point density
    templateModel = self.downsampleModel(atlasNode, spacingPercentage)

    # Returns both the downsampled model and its point count for user feedback
    return templateModel, templateModel.GetNumberOfPoints()

  def runDeCAL(self, baseNode, baseLMPath, meshDirectory, landmarkDirectory, outputDirectory, spacingTolerance):
    """
    Performs Dense Correspondence Landmarking (DeCAL).

    Generates densely sampled corresponding landmarks across specimens
    using non-rigid registration guided by sparse anatomical landmarks.

    Args:
      baseNode: Atlas model node
      baseLMPath: Atlas landmark node
      meshDirectory: Directory containing aligned models
      landmarkDirectory: Directory containing aligned landmarks
      outputDirectory: Output directory for dense landmarks
      spacingTolerance: Point spacing control (percentage)

    Returns:
      Atlas landmark node with dense sampling
    """
    # Converts tolerance to percentage for downsampling
    spacingPercentage = spacingTolerance/100
    loadOption=False

    # Loads atlas landmarks as VTK points
    baseLandmarks=self.fiducialNodeToPolyData(baseLMPath, loadOption).GetPoints()

    # Imports all specimen data
    modelExt=['ply','stl','vtp', 'vtk']
    self.modelNames, models = self.importMeshes(meshDirectory, modelExt)
    landmarkNames, landmarks = self.importLandmarks(landmarkDirectory)
    self.outputDirectory = outputDirectory

    # Computes dense correspondences
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
    # Extracts polydata from the input model node
    points=model.GetPolyData()

    # Creates VTK clean filter to merge nearby points
    cleanFilter=vtk.vtkCleanPolyData()

    # Uses relative tolerance based on mesh bounding box
    cleanFilter.SetToleranceIsAbsolute(False)

    # Sets merge tolerance as percentage of diagonal
    cleanFilter.SetTolerance(spacingPercentage)

    # Connects input polydata to filter
    cleanFilter.SetInputData(points)

    # Executes the downsampling operation
    cleanFilter.Update()

    # Returns the decimated polydata
    return cleanFilter.GetOutput()

  def addIndexArray(self, mesh, arrayName):
    # Creates integer array to store original point indices
    # This preserves point identity through downsampling operations
    indexArray = vtk.vtkIntArray()

    # Sets array as single-component (one value per point)
    indexArray.SetNumberOfComponents(1)

    # Assigns identifying name for later retrieval
    indexArray.SetName(arrayName)

    # Iterates through all mesh points
    for i in range(mesh.GetPolyData().GetNumberOfPoints()):
      # Stores original index value for each point
      indexArray.InsertNextValue(i)

    # Attaches index array to mesh point data
    mesh.GetPolyData().GetPointData().AddArray(indexArray)

  def computeNormals(self, inputModel):
    # Creates VTK filter for normal vector computation
    normals = vtk.vtkPolyDataNormals()

    # Connects input model's polydata to filter
    normals.SetInputData(inputModel.GetPolyData())

    # Ensures consistent normal orientation across surface
    normals.SetAutoOrientNormals(True)

    # Executes normal computation
    normals.Update()

    # Replaces model's polydata with normal-enhanced version
    inputModel.SetAndObservePolyData(normals.GetOutput())

  def runMirroring(self, meshDirectory, lmDirectory, mirrorMeshDirectory, mirrorLMDirectory, mirrorAxis, mirrorIndexText, slmDirectory=None, outputSLMDirectoryy=None, mirrorSLMIndexText=None):
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
    semilandmarkOption = bool(slmDirectory and outputSLMDirectoryy and (len(mirrorSLMIndexText) != 0))
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
              outputSLMPath = os.path.join(outputSLMDirectoryy, outputSLMName)
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



  #NEW
  def runDCAlign(self, baseMeshPath, baseLMPath, meshDirectory, landmarkDirectory, outputDirectory, optionErrorOutput, textureDirectory):
    """
    Performs DeCA shape analysis with optional texture features.

    Computes dense correspondences and extracts morphometric features
    including shape deformation magnitudes and color-based comparisons.

    Args:
      baseMeshPath: Path to atlas model
      baseLMPath: Path to atlas landmarks
      meshDirectory: Directory with aligned models
      landmarkDirectory: Directory with aligned landmarks
      outputDirectory: Output directory for results
      optionErrorOutput: Whether to generate error checking data
      textureDirectory: Directory with texture images for color analysis
    """
    # Sets up error checking directory if requested
    if optionErrorOutput:
      self.errorCheckPath = os.path.join(outputDirectory, "errorChecking")
      if not os.path.exists(self.errorCheckPath):
        os.mkdir(self.errorCheckPath)

    # Loads atlas data
    baseNode = slicer.util.loadModel(baseMeshPath)
    baseMesh = baseNode.GetPolyData()
    baseLandmarks=self.fiducialNodeToPolyData(baseLMPath).GetPoints()

    # Imports specimen data
    modelExt=['ply','stl','vtp']
    self.modelNames, models = self.importMeshes(meshDirectory, modelExt)
    landmarkNames,landmarks = self.importLandmarks(landmarkDirectory)

    # Computes correspondences with texture support
    denseCorrespondenceGroup = self.denseCorrespondenceBaseMesh(landmarks, models, baseMesh, baseLandmarks, textureDirectory)

    # Extracts morphometric features
    self.addMagnitudeFeature(denseCorrespondenceGroup, self.modelNames, baseMesh)
    # Adds color-based features for texture analysis
    self.addColorComparisonFeatures(denseCorrespondenceGroup, self.modelNames, baseMesh)
    # save results to output directory
    outputModelName = 'decaResultModel.vtp'
    outputModelPath = os.path.join(outputDirectory, outputModelName)
    slicer.util.saveNode(baseNode, outputModelPath)
  #NEW
  def runDCAlignSymmetric(self, baseMeshPath, baseLMPath, meshDir, landmarkDir, mirrorMeshDir, mirrorLandmarkDir, outputDir, optionErrorOutput):
    if optionErrorOutput:
      self.errorCheckPath = os.path.join(outputDir, "errorChecking")
      if not os.path.exists(self.errorCheckPath):
        os.mkdir(self.errorCheckPath)
    baseNode = slicer.util.loadModel(baseMeshPath)
    baseMesh = baseNode.GetPolyData()
    baseLandmarks=self.fiducialNodeToPolyData(baseLMPath).GetPoints()
    modelExt=['ply','stl','vtp']
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
    modelExt=['ply','stl','vtp','vtk']
    self.modelNames, models = self.importMeshes(meshDirectory, modelExt)
    landmarkNames, landmarks = self.importLandmarks(landmarkDirectory)
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
        currentNode = slicer.util.loadModel(filePath)
        return currentNode

  def runAlign(self, baseMeshNode, baseLMNode, meshDirectory, lmDirectory, outputMeshDirectoryDirectory, outputLMDirectory, removeScaleOption, slmDirectory=False, outputSLMDirectory=False):
    """
    Aligns all specimens to atlas using landmark-based registration.

    Performs rigid or similarity transformation to align specimens
    to a common reference frame defined by the atlas landmarks.

    Args:
      baseMeshNode: Atlas model node
      baseLMNode: Atlas landmark node
      meshDirectory: Directory with original models
      lmDirectory: Directory with original landmarks
      outputMeshDirectoryDirectory: Output for aligned models
      outputLMDirectory: Output for aligned landmarks
      removeScaleOption: Whether to normalize for size
      slmDirectory: Optional semi-landmark directory
      outputSLMDirectory: Optional output for aligned semi-landmarks
    """
    semilandmarkOption = bool(slmDirectory and outputSLMDirectory)
    targetPoints = vtk.vtkPoints()
    point=[0,0,0]

    # Extracts atlas landmark positions as target
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
          outputMeshPath = os.path.join(outputMeshDirectoryDirectory, outputMeshName)
          slicer.util.saveNode(currentMeshNode, outputMeshPath)
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
              outputSLMPath = os.path.join(outputSLMDirectoryy, outputSLMName)
              slicer.util.saveNode(currentSLMNode, outputSLMPath)
              slicer.mrmlScene.RemoveNode(currentSLMNode)
          # clean up
          try:
            slicer.mrmlScene.RemoveNode(currentLMNode)
            slicer.mrmlScene.RemoveNode(currentMeshNode)
            slicer.mrmlScene.RemoveNode(transformNode)
            #slicer.mrmlScene.RemoveNode(baseMeshNode)
            #slicer.mrmlScene.RemoveNode(baseLMNode)
          except:
            print(f"could not find nodes to remove for {subjectID}")

  def distanceMatrix(self, a):
    """
    Computes the euclidean distance matrix for n points in a 3D space
    Returns a nXn matrix
     """
    # Extracts shape dimensions from input array
    id,jd=a.shape

    # Defines lambda for creating difference matrices
    # Subtracts each element from reshaped column vector
    fnx = lambda q : q - np.reshape(q, (id, 1))

    # Computes pairwise differences for x coordinates
    dx=fnx(a[:,0])

    # Computes pairwise differences for y coordinates
    dy=fnx(a[:,1])

    # Computes pairwise differences for z coordinates
    dz=fnx(a[:,2])

    # Calculates Euclidean distances from coordinate differences
    return (dx**2.0+dy**2.0+dz**2.0)**0.5

  def numpyToFiducialNode(self, numpyArray, nodeName):
    # Creates new fiducial node in 3D Slicer scene
    fiducialNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsFiducialNode',nodeName)

    # Iterates through numpy array points
    for index in range(len(numpyArray)):
      # Adds each point as control point with string index label
      fiducialNode.AddControlPoint(numpyArray[index], str(index))

    # Returns populated fiducial node
    return fiducialNode

  def computeAverageLM(self, fiducialGroup):
    # Counts number of specimens in the group
    sampleNumber = fiducialGroup.GetNumberOfBlocks()

    # Gets landmark count from first specimen
    pointNumber = fiducialGroup.GetBlock(0).GetNumberOfPoints()

    # Initializes 3D array: points x coordinates x samples
    groupArray_np = np.empty((pointNumber,3,sampleNumber))

    # Extracts landmark coordinates from each specimen
    for i in range(sampleNumber):
      # Gets VTK point data from current specimen
      pointData = fiducialGroup.GetBlock(i).GetPoints().GetData()

      # Converts VTK data to numpy array
      pointData_np = vtk_np.vtk_to_numpy(pointData)

      # Stores in 3D array structure
      groupArray_np[:,:,i] = pointData_np

    # Calculates mean position across all specimens for each landmark
    averagePoints_np = np.mean(groupArray_np, axis=2)

    # Converts mean positions to fiducial node
    averageLMNode = self.numpyToFiducialNode(averagePoints_np, "Atlas Landmarks")

    return averageLMNode

  def fiducialNodeToPolyData(self, nodeLocation, loadOption=True):
    # Initializes point coordinate array
    point = [0,0,0]

    # Creates empty polydata object
    polydataPoints = vtk.vtkPolyData()

    # Creates VTK points container
    points = vtk.vtkPoints()

    # Determines whether to load from file or use existing node
    if not loadOption:
      # Uses passed node directly
      fiducialNode = nodeLocation
    else:
      # Loads fiducial node from file path
      [success,fiducialNode] = slicer.util.loadMarkupsFiducialList(nodeLocation)

      # Validates successful loading
      if not success:
        print("Could not load landmarks: ", nodeLocation)
        return

    # Converts fiducial points to VTK points
    for i in range(fiducialNode.GetNumberOfControlPoints()):
      # Gets position of current control point
      point = fiducialNode.GetNthControlPointPosition(i)

      # Adds point to VTK points collection
      points.InsertNextPoint(point)

    # Assigns points to polydata structure
    polydataPoints.SetPoints(points)

    # Removes temporary node from scene
    slicer.mrmlScene.RemoveNode(fiducialNode)

    return polydataPoints

  def importLandmarks(self, topDir):
    # Creates multi-block filter to group multiple landmark sets
    fiducialGroup = vtk.vtkMultiBlockDataGroupFilter()

    # Initializes list to track processed filenames
    fileNameList = []

    # Processes all landmark files in directory
    for file in sorted(os.listdir(topDir)):
      # Filters for supported landmark formats
      if file.endswith(".fcsv") or file.endswith(".json"):
        # Stores filename for reference
        fileNameList.append(file)

        # Constructs full file path
        inputFilePath = os.path.join(topDir, file)

        # Converts landmark file to polydata format
        # Note: may want to replace with vtk reader for efficiency
        polydataPoints = self.fiducialNodeToPolyData(inputFilePath)

        # Adds to multi-block group
        fiducialGroup.AddInputData(polydataPoints)

    # Executes grouping operation
    fiducialGroup.Update()

    # Returns filenames and grouped landmark data
    return fileNameList, fiducialGroup.GetOutput()

  def importMeshes(self, topDir, extensions):
      # Creates multi-block filter to group multiple mesh models
      modelGroup = vtk.vtkMultiBlockDataGroupFilter()

      # Initializes list to track processed model names
      fileNameList = []

      # Processes all mesh files in directory
      for file in sorted(os.listdir(topDir)):
        # Checks if file has supported mesh extension
        if file.endswith(tuple(extensions)):
          # Extracts base filename without extension
          base, ext = os.path.splitext(file)

          # Stores base name for specimen identification
          fileNameList.append(base)

          # Constructs full file path
          inputFilePath = os.path.join(topDir, file)

          # Loads mesh model into Slicer scene
          # Note: may want to replace with vtk reader for efficiency
          modelNode = slicer.util.loadModel(inputFilePath)

          # Extracts polydata and adds to group
          modelGroup.AddInputData(modelNode.GetPolyData())

          # Removes temporary node from scene to free memory
          slicer.mrmlScene.RemoveNode(modelNode)

      # Executes grouping operation
      modelGroup.Update()

      # Returns model names and grouped mesh data
      return fileNameList, modelGroup.GetOutput()

  def procrustesImposition(self, originalLandmarks, sizeOption):
    # Creates Procrustes alignment filter for shape analysis
    procrustesFilter = vtk.vtkProcrustesAlignmentFilter()

    # Configures alignment mode based on size normalization preference
    if(sizeOption):
      # Uses rigid body transformation (preserves size)
      procrustesFilter.GetLandmarkTransform().SetModeToRigidBody()
    # Default mode is similarity transform (removes size)

    # Connects landmark data to filter
    procrustesFilter.SetInputData(originalLandmarks)

    # Executes Procrustes alignment
    procrustesFilter.Update()

    # Extracts computed mean shape
    meanShape = procrustesFilter.GetMeanPoints()

    # Returns mean shape and aligned landmark sets
    return [meanShape, procrustesFilter.GetOutput()]

  def getClosestToMeanIndex(self, meanShape, alignedPoints):
    # Imports operator for min function key extraction
    import operator

    # Gets number of specimens in aligned group
    sampleNumber = alignedPoints.GetNumberOfBlocks()

    # Initializes list to store Procrustes distances
    procrustesDistances = []

    # Calculates distance from each specimen to mean shape
    for i in range(sampleNumber):
      # Gets current specimen's aligned landmarks
      alignedShape = alignedPoints.GetBlock(i)

      # Initializes point coordinate arrays
      meanPoint = [0,0,0]
      alignedPoint = [0,0,0]

      # Accumulates total distance for this specimen
      distance = 0

      # Sums distances across all landmark points
      for j in range(meanShape.GetNumberOfPoints()):
        # Gets coordinates from mean shape
        meanShape.GetPoint(j,meanPoint)

        # Gets corresponding coordinates from specimen
        alignedShape.GetPoint(j,alignedPoint)

        # Adds Euclidean distance to total
        distance += np.sqrt(vtk.vtkMath.Distance2BetweenPoints(meanPoint,alignedPoint))

      # Stores total distance for this specimen
      procrustesDistances.append(distance)

    try:
      # Finds specimen with minimum distance to mean
      min_index, min_value = min(enumerate(procrustesDistances), key=operator.itemgetter(1))
      return min_index
    except:
      # Returns first specimen if error occurs
      return 0

  def getClosestToMeanPath(self, landmarkDirectory):
    # Imports all landmark files from directory
    lmNames, landmarks = self.importLandmarks(landmarkDirectory)

    # Performs Procrustes alignment without size preservation
    meanShape, alignedLandmarks = self.procrustesImposition(landmarks, False)

    # Identifies specimen closest to mean configuration
    closestToMeanIndex = self.getClosestToMeanIndex(meanShape, alignedLandmarks)

    # Returns filename of most representative specimen
    return lmNames[closestToMeanIndex]

  def denseCorrespondence(self, originalLandmarks, originalMeshes, textureImageNode, writeErrorOption=False):
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
      baseMesh, baseLandmarks, meanShape, i, textureImageNode)
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
#NEW
  def denseCorrespondenceBaseMesh(self, originalLandmarks, originalMeshes, baseMesh, baseLandmarks, textureDirectory):
    meanShape, alignedPoints = self.procrustesImposition(originalLandmarks, False)
    sampleNumber = alignedPoints.GetNumberOfBlocks()
    print("procrustes aligned samples: ", sampleNumber)
    denseCorrespondenceGroup = vtk.vtkMultiBlockDataGroupFilter()

    # Load all texture images
    textureImages = {}
    for texFile in os.listdir(textureDirectory):
        if texFile.lower().endswith(('.png', '.jpg', '.jpeg')):
            texPath = os.path.join(textureDirectory, texFile)
            textureNode = slicer.util.loadVolume(texPath)
            textureImages[texFile.split('.')[0]] = textureNode

    for i in range(sampleNumber):
      subjectID = self.modelNames[i]
      textureImageNode = textureImages.get(subjectID, None)
      correspondingMesh = self.denseSurfaceCorrespondencePair(originalMeshes.GetBlock(i),
      originalLandmarks.GetBlock(i).GetPoints(), alignedPoints.GetBlock(i).GetPoints(),
      baseMesh, baseLandmarks, meanShape, i, textureImageNode)
      denseCorrespondenceGroup.AddInputData(correspondingMesh)

    denseCorrespondenceGroup.Update()
    return denseCorrespondenceGroup.GetOutput()
  #NEW

  def denseSurfaceCorrespondencePair(self, originalMesh, originalLandmarks, alignedLandmarks,baseMesh, baseLandmarks, meanShape, iteration,textureImageNode, saveAsPointData='uchar-vector'):
    # Creates thin-plate spline transform for warping to mean space
    meanTransform = vtk.vtkThinPlateSplineTransform()
    # Sets source landmarks from original specimen
    meanTransform.SetSourceLandmarks(originalLandmarks)
    # Sets target as mean shape coordinates
    meanTransform.SetTargetLandmarks(meanShape)
    # Uses radial basis function for 3D deformation
    meanTransform.SetBasisToR()

    # Creates filter to apply TPS transform to mesh
    meanTransformFilter = vtk.vtkTransformPolyDataFilter()
    # Connects original mesh as input
    meanTransformFilter.SetInputData(originalMesh)
    # Applies TPS warping transform
    meanTransformFilter.SetTransform(meanTransform)
    # Executes transformation
    meanTransformFilter.Update()
    # Retrieves warped mesh in mean space
    meanWarpedMesh = meanTransformFilter.GetOutput()

    # Creates TPS transform for warping atlas to mean space
    meanTransformBase = vtk.vtkThinPlateSplineTransform()
    # Sets source landmarks from atlas
    meanTransformBase.SetSourceLandmarks(baseLandmarks)
    # Sets target as mean shape coordinates
    meanTransformBase.SetTargetLandmarks(meanShape)
    # Uses radial basis function for 3D deformation
    meanTransformBase.SetBasisToR()

    # Creates filter to apply TPS transform to atlas
    meanTransformBaseFilter = vtk.vtkTransformPolyDataFilter()
    # Connects atlas mesh as input
    meanTransformBaseFilter.SetInputData(baseMesh)
    # Applies TPS warping transform
    meanTransformBaseFilter.SetTransform(meanTransformBase)
    # Executes transformation
    meanTransformBaseFilter.Update()
    # Retrieves warped atlas in mean space
    meanWarpedBase = meanTransformBaseFilter.GetOutput()

    # Writes intermediate results for error checking if requested
    if hasattr(self,"errorCheckPath"):
      # Creates PLY writer for subject mesh
      plyWriterSubject = vtk.vtkPLYWriter()
      # Constructs filename with specimen ID
      plyName = "subject_" + self.modelNames[iteration] + ".ply"
      plyPath = os.path.join(self.errorCheckPath, plyName)
      # Configures writer with output path
      plyWriterSubject.SetFileName(plyPath)
      # Sets warped subject mesh as data source
      plyWriterSubject.SetInputData(meanWarpedMesh)
      # Writes subject mesh to file
      plyWriterSubject.Write()

      # Creates PLY writer for atlas mesh
      plyWriterBase = vtk.vtkPLYWriter()
      # Uses fixed name for atlas reference
      plyName = "base.ply"
      plyPath = os.path.join(self.errorCheckPath, plyName)
      # Configures writer with output path
      plyWriterBase.SetFileName(plyPath)
      # Sets warped atlas mesh as data source
      plyWriterBase.SetInputData(meanWarpedBase)
      # Writes atlas mesh to file
      plyWriterBase.Write()

    # Creates cell locator for finding closest points
    cellLocator = vtk.vtkCellLocator()
    # Sets warped subject mesh as search target
    cellLocator.SetDataSet(meanWarpedMesh)
    # Builds spatial search structure
    cellLocator.BuildLocator()

    # Initializes coordinate arrays
    point = [0,0,0]
    correspondingPoint = [0,0,0]
    # Creates container for correspondence points
    correspondingPoints = vtk.vtkPoints()
    # Creates reference variables for VTK output
    cellId = vtk.reference(0)  # Cell containing closest point
    subId = vtk.reference(0)  # Subcell ID
    distance = vtk.reference(0.0)  # Distance to closest point

    # Finds correspondence for each atlas point
    for i in range(meanWarpedBase.GetNumberOfPoints()):
      # Gets current atlas point position
      meanWarpedBase.GetPoint(i,point)
      # Finds closest point on subject mesh
      cellLocator.FindClosestPoint(point,correspondingPoint,cellId, subId, distance)
      # Stores corresponding point with same index
      correspondingPoints.InsertPoint(i,correspondingPoint)

    # Creates new mesh with correspondence points and atlas topology
    correspondingMesh = vtk.vtkPolyData()
    # Assigns correspondence points
    correspondingMesh.SetPoints(correspondingPoints)
    # Copies polygon connectivity from atlas
    correspondingMesh.SetPolys(meanWarpedBase.GetPolys())

    # Creates inverse TPS transform to return from mean space
    inverseTransform = vtk.vtkThinPlateSplineTransform()
    # Sets mean shape as source (reverse of forward transform)
    inverseTransform.SetSourceLandmarks(meanShape)
    # Sets original landmarks as target
    inverseTransform.SetTargetLandmarks(originalLandmarks)
    # Uses radial basis function for 3D deformation
    inverseTransform.SetBasisToR()

    # Creates filter to apply inverse transform
    inverseTransformFilter = vtk.vtkTransformPolyDataFilter()
    # Connects correspondence mesh as input
    inverseTransformFilter.SetInputData(correspondingMesh)
    # Applies inverse TPS transform
    inverseTransformFilter.SetTransform(inverseTransform)
    # Executes inverse transformation
    inverseTransformFilter.Update()
    # Retrieves final mesh in original space
    finalMesh = inverseTransformFilter.GetOutput()

    # Transfers texture colors from original to correspondence mesh
    colorWrappedMesh = self.wrapTextureFromImage(originalMesh, textureImageNode, finalMesh, saveAsPointData)

    return colorWrappedMesh

  def convertPointsToVTK(self, points):
    # Converts numpy array to VTK array format
    # deep=True creates independent copy, VTK_FLOAT specifies data type
    array_vtk = vtk_np.numpy_to_vtk(points, deep=True, array_type=vtk.VTK_FLOAT)

    # Creates VTK points object
    points_vtk = vtk.vtkPoints()

    # Assigns converted array as point data
    points_vtk.SetData(array_vtk)

    # Creates polydata structure to hold points
    polydata_vtk = vtk.vtkPolyData()

    # Attaches points to polydata
    polydata_vtk.SetPoints(points_vtk)

    return polydata_vtk
#NEW BEGIN
  def wrapTextureFromImage(self, sourceMesh, textureImageNode, targetMesh, saveAsPointData='uchar-vector'):
    """
    Transfer texture from an image to the target mesh using the source mesh as intermediary.

    1. Applies texture image as color data to the source mesh points.
    2. Transfers those point colors to the target mesh via spatial correspondence.

    :param sourceMesh: VTK model node with texture UVs.
    :param textureImageNode: VTK image node representing the texture.
    :param targetMesh: VTK model node to which the texture color will be transferred.
    :param saveAsPointData: How to save the point data ('uchar-vector', 'float-vector', 'float-components').
    :return: Modified target mesh with texture-based color.
    """

    # Step 1: Convert texture to color on sourceMesh using UVs
    self.convertTextureToPointAttribute(sourceMesh, textureImageNode, saveAsPointData)

    # Step 2: Wrap those colors from sourceMesh to targetMesh
    sourceColors = sourceMesh.GetPointData().GetScalars()
    if not sourceColors:
        raise RuntimeError("Source mesh has no scalar color data.")

    # Create new color array for the target
    targetColors = vtk.vtkUnsignedCharArray()
    targetColors.SetNumberOfComponents(sourceColors.GetNumberOfComponents())
    targetColors.SetName("WrappedColors")

    # Build locator for spatial correspondence
    cellLocator = vtk.vtkCellLocator()
    cellLocator.SetDataSet(sourceMesh)
    cellLocator.BuildLocator()

    # Transfer color via nearest point
    for i in range(targetMesh.GetNumberOfPoints()):
        targetPoint = targetMesh.GetPoint(i)
        closestPoint = [0.0, 0.0, 0.0]
        cellId = vtk.reference(0)
        subId = vtk.reference(0)
        dist = vtk.reference(0.0)
        cellLocator.FindClosestPoint(targetPoint, closestPoint, cellId, subId, dist)
        closestPointId = cellLocator.FindPoint(closestPoint)
        color = sourceColors.GetTuple(closestPointId)
        targetColors.InsertNextTuple(color)

    targetMesh.GetPointData().SetScalars(targetColors)
    targetMesh.Modified()
    return targetMesh
  def convertTextureToPointAttribute(self, modelNode, textureImageNode, saveAsPointData):
    """
    Map texture image colors to vertex points of a mesh.

    :param modelNode: VTK model node containing the mesh.
    :param textureImageNode: VTK image node containing the texture.
    :param saveAsPointData: How to save the point data ('uchar-vector', 'float-vector', 'float-components').
    """
    # Retrieve polydata from the model node
    polyData = modelNode.GetPolyData()

    # Retrieve texture coordinates from the polydata
    tcoords_vtk = polyData.GetPointData().GetTCoords()
    tcoords_np = nps.vtk_to_numpy(tcoords_vtk)

    # Flip texture image vertically
    textureImageFlipVert = vtk.vtkImageFlip()
    textureImageFlipVert.SetFilteredAxis(1)
    textureImageFlipVert.SetInputConnection(textureImageNode.GetImageDataConnection())
    textureImageFlipVert.Update()
    textureImageData = textureImageFlipVert.GetOutput()

    # Convert texture image to NumPy array
    textureImage_np = nps.vtk_to_numpy(textureImageData.GetPointData().GetScalars()).reshape(
        textureImageData.GetDimensions()[1], textureImageData.GetDimensions()[0], -1)

    # Normalize and scale texture coordinates
    print("Original Texture Coordinates: ", tcoords_np)

    # Scale the texture coordinates
    # Ensure that you're using the correct image dimensions
    width, height = textureImage_np.shape[1], textureImage_np.shape[0]
    uv_scaled = np.clip(tcoords_np, 0, 1) * [width - 1, height - 1]

    print("Scaled UV Coordinates: ", uv_scaled)

    # Map texture coordinates to texture colors (vectorized operation)
    colors = textureImage_np[uv_scaled[:, 1].astype(int), uv_scaled[:, 0].astype(int)]

    # Convert results back to VTK and save as point data
    if saveAsPointData == 'uchar-vector':
        colorArray_vtk = nps.numpy_to_vtk(colors, deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
        colorArray_vtk.SetName('RGB')
        polyData.GetPointData().SetScalars(colorArray_vtk)
    elif saveAsPointData == 'float-vector':
        colorArray_vtk = nps.numpy_to_vtk(colors.astype(float) / 255.0, deep=True, array_type=vtk.VTK_FLOAT)
        colorArray_vtk.SetName('Color')
        polyData.GetPointData().SetScalars(colorArray_vtk)
    elif saveAsPointData == 'float-components':
        for i, name in enumerate(['ColorRed', 'ColorGreen', 'ColorBlue']):
            componentArray_vtk = nps.numpy_to_vtk(colors[:, i].astype(float), deep=True, array_type=vtk.VTK_FLOAT)
            componentArray_vtk.SetName(name)
            polyData.GetPointData().AddArray(componentArray_vtk)
    else:
        raise ValueError(f"Invalid saveAsPointData: {saveAsPointData}")

    # Mark the polydata as modified
    polyData.Modified()
  def addColorComparisonFeatures(self, denseCorrespondenceGroup, modelNameArray, baseMesh):
    """Compare colors across corresponding points"""
    sampleNumber = denseCorrespondenceGroup.GetNumberOfBlocks()
    pointNumber = denseCorrespondenceGroup.GetBlock(0).GetNumberOfPoints()

    # Color variance arrays
    colorMean = vtk.vtkDoubleArray()
    colorMean.SetNumberOfComponents(3)  # RGB
    colorMean.SetName("Color Mean")

    colorVariance = vtk.vtkDoubleArray()
    colorVariance.SetNumberOfComponents(1)
    colorVariance.SetName("Color Variance")

    for i in range(pointNumber):
        colorValues = []
        for j in range(sampleNumber):
            mesh = denseCorrespondenceGroup.GetBlock(j)
            colors = mesh.GetPointData().GetScalars("WrappedColors")
            if colors:
                color = colors.GetTuple3(i)
                colorValues.append(color)

        if colorValues:
            # Calculate mean and variance
            meanColor = np.mean(colorValues, axis=0)
            colorVar = np.var(colorValues)

            colorMean.InsertNextTuple3(meanColor[0], meanColor[1], meanColor[2])
            colorVariance.InsertNextValue(colorVar)

    baseMesh.GetPointData().AddArray(colorMean)
    baseMesh.GetPointData().AddArray(colorVariance)
  #NEW END
  def computeAverageModelFromGroup(self, denseCorrespondenceGroup, baseIndex):
    # Counts number of specimens in correspondence group
    sampleNumber = denseCorrespondenceGroup.GetNumberOfBlocks()

    # Gets point count from first specimen
    pointNumber = denseCorrespondenceGroup.GetBlock(0).GetNumberOfPoints()

    # Initializes 3D array: points x coordinates x samples
    groupArray_np = np.empty((pointNumber,3,sampleNumber))

    # Retrieves base mesh for topology reference
    # Base mesh is specimen closest to mean shape
    baseMesh = denseCorrespondenceGroup.GetBlock(baseIndex)

    # Extracts point coordinates from all specimens
    for i in range(sampleNumber):
      # Gets current specimen's aligned mesh
      alignedMesh = denseCorrespondenceGroup.GetBlock(i)

      # Converts VTK points to numpy array
      alignedMesh_np = vtk_np.vtk_to_numpy(alignedMesh.GetPoints().GetData())

      # Stores in 3D array structure
      groupArray_np[:,:,i] = alignedMesh_np

    # Calculates mean position for each point across all specimens
    averagePoints_np = np.mean(groupArray_np, axis=2)

    # Converts mean points to VTK format
    averagePointsPolydata = self.convertPointsToVTK(averagePoints_np)

    # Creates new polydata with mean points and base topology
    averageModel = vtk.vtkPolyData()

    # Assigns averaged point positions
    averageModel.SetPoints(averagePointsPolydata.GetPoints())

    # Copies connectivity from base mesh
    averageModel.SetPolys(baseMesh.GetPolys())

    return averageModel

  def addMagnitudeFeature(self, denseCorrespondenceGroup, modelNameArray, model):
    # Counts specimens in correspondence group
    sampleNumber = denseCorrespondenceGroup.GetNumberOfBlocks()

    # Gets number of corresponding points
    pointNumber = denseCorrespondenceGroup.GetBlock(0).GetNumberOfPoints()

    # Initializes array to store deformation magnitudes
    statsArray = np.zeros((pointNumber, sampleNumber))

    # Creates VTK array for mean deformation values
    magnitudeMean = vtk.vtkDoubleArray()
    magnitudeMean.SetNumberOfComponents(1)
    magnitudeMean.SetName("Magnitude Mean")

    # Creates VTK array for deformation standard deviations
    magnitudeSD = vtk.vtkDoubleArray()
    magnitudeSD.SetNumberOfComponents(1)
    magnitudeSD.SetName("Magnitude SD")

    # Calculates deformation distances for each specimen
    for i in range(sampleNumber):
      # Gets current specimen's aligned mesh
      alignedMesh = denseCorrespondenceGroup.GetBlock(i)

      # Creates array for this specimen's deformations
      magnitudes = vtk.vtkDoubleArray()
      magnitudes.SetNumberOfComponents(1)
      magnitudes.SetName(modelNameArray[i])  # Names array by specimen ID

      # Computes point-wise distances from atlas
      for j in range(pointNumber):
        # Gets atlas point position
        modelPoint = model.GetPoint(j)

        # Gets corresponding point on specimen
        targetPoint = alignedMesh.GetPoint(j)

        # Calculates Euclidean distance
        distance = np.sqrt(vtk.vtkMath.Distance2BetweenPoints(modelPoint,targetPoint))

        # Stores distance in VTK array
        magnitudes.InsertNextValue(distance)

        # Stores distance in statistics array
        statsArray[j,i]=distance

      # Attaches specimen's magnitude array to model
      model.GetPointData().AddArray(magnitudes)

    # Computes statistics across specimens for each point
    for i in range(pointNumber):
      # Calculates mean deformation at this point
      pointMean = statsArray[i,:].mean()
      magnitudeMean.InsertNextValue(pointMean)

      # Calculates standard deviation at this point
      pointSD = statsArray[i,:].std()
      magnitudeSD.InsertNextValue(pointSD)

    # Attaches statistical arrays to model for visualization
    model.GetPointData().AddArray(magnitudeMean)
    model.GetPointData().AddArray(magnitudeSD)

  def addMagnitudeFeatureSymmetry(self, denseCorrespondenceGroup, denseCorrespondenceGroupMirror, modelNameArray, model):
    # Counts specimens in correspondence groups
    sampleNumber = denseCorrespondenceGroup.GetNumberOfBlocks()

    # Gets number of corresponding points
    pointNumber = denseCorrespondenceGroup.GetBlock(0).GetNumberOfPoints()

    # Initializes array for asymmetry measurements
    statsArray = np.zeros((pointNumber, sampleNumber))

    # Creates VTK array for mean asymmetry values
    magnitudeMean = vtk.vtkDoubleArray()
    magnitudeMean.SetNumberOfComponents(1)
    magnitudeMean.SetName("Magnitude Mean")

    # Creates VTK array for asymmetry standard deviations
    magnitudeSD = vtk.vtkDoubleArray()
    magnitudeSD.SetNumberOfComponents(1)
    magnitudeSD.SetName("Magnitude SD")

    # Calculates asymmetry distances for each specimen
    for i in range(sampleNumber):
      # Gets original aligned mesh
      alignedMesh = denseCorrespondenceGroup.GetBlock(i)

      # Gets mirrored aligned mesh
      mirrorMesh = denseCorrespondenceGroupMirror.GetBlock(i)

      # Creates array for this specimen's asymmetry
      magnitudes = vtk.vtkDoubleArray()
      magnitudes.SetNumberOfComponents(1)
      magnitudes.SetName(modelNameArray[i])  # Names array by specimen ID

      # Computes point-wise asymmetry
      for j in range(pointNumber):
        # Gets atlas point (unused but kept for consistency)
        modelPoint = model.GetPoint(j)

        # Gets corresponding point on original
        targetPoint1 = alignedMesh.GetPoint(j)

        # Gets corresponding point on mirror
        targetPoint2 = mirrorMesh.GetPoint(j)

        # Calculates distance between original and mirrored points
        distance = np.sqrt(vtk.vtkMath.Distance2BetweenPoints(targetPoint1,targetPoint2))

        # Stores asymmetry distance
        magnitudes.InsertNextValue(distance)

        # Stores in statistics array
        statsArray[j,i]=distance

      # Attaches specimen's asymmetry array to model
      model.GetPointData().AddArray(magnitudes)

    # Computes asymmetry statistics for each point
    for i in range(pointNumber):
      # Calculates mean asymmetry at this point
      pointMean = statsArray[i,:].mean()
      magnitudeMean.InsertNextValue(pointMean)

      # Calculates asymmetry variability at this point
      pointSD = statsArray[i,:].std()
      magnitudeSD.InsertNextValue(pointSD)

    # Attaches statistical arrays for visualization
    model.GetPointData().AddArray(magnitudeMean)
    model.GetPointData().AddArray(magnitudeSD)
