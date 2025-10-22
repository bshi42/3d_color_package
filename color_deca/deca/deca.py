# -*- coding: utf-8 -*-
"""DeCA (Dense Correspondence Analysis) Module for 3D Slicer.

Provides tools for establishing dense point correspondences between
biological specimens for morphometric analysis. Originally developed
for analyzing 3D shape variations in biological structures.

Key Features:
- Atlas generation from specimen sets
- Dense correspondence mapping
- Procrustes alignment and shape analysis
- Landmark-based registration
- Statistical shape modeling

Originally generated from Colab notebook:
    https://colab.research.google.com/drive/1I5HVk7KOmgTSesAYsK4UDC3BTmlvACkx
"""

import os  # Provides file system operations
import unittest  # Enables unit testing framework
import vtk, qt, ctk, slicer  # Imports core 3D Slicer dependencies
from slicer.ScriptedLoadableModule import *  # Imports base module class
import logging  # Enables logging capabilities
import fnmatch  # Provides filename pattern matching
import numpy as np  # Imports numerical computing library
import random  # Provides random number generation
import math  # Imports mathematical functions
from datetime import datetime  # Enables date/time operations
import re  # Provides regular expression support
import csv  # Enables CSV file operations
import vtk.util.numpy_support as vtk_np  # Converts between VTK and NumPy arrays
from pathlib import Path  # Provides object-oriented filesystem paths
import shutil  # Enables high-level file operations

#
# DeCA
#

class deca(ScriptedLoadableModule):
  """Module class for Dense Correspondence Analysis (DeCA).

  Registers the module with 3D Slicer and defines metadata.
  Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, parent):
    """Initializes the DeCA module with metadata and configuration.

    Args:
      parent: Parent object from 3D Slicer framework
    """
    ScriptedLoadableModule.__init__(self, parent)  # Calls parent constructor
    self.parent.title = "DeCA"  # Sets module name in Slicer interface
    self.parent.categories = ["SlicerMorph.DeCA Toolbox"]  # Categorizes module in menu
    self.parent.dependencies = []  # Declares no dependencies
    self.parent.contributors = ["Sara Rolfe (SCRI)"]  # Lists module authors
    self.parent.helpText = """
      This module provides several flexible workflows for finding and analyzing dense correspondence points between models.
      """  # Provides user-facing help documentation
    self.parent.helpText += self.getDefaultModuleDocumentationLink()  # Adds link to online docs
    self.parent.acknowledgementText = """This extension was developed by funding from National Institutes of Health (OD032627 and HD104435) to A. Murat Maga (SCRI)
      """  # Acknowledges funding sources

#
# DeCAWidget
#

class decaWidget(ScriptedLoadableModuleWidget):
  """GUI widget for the DeCA module.

  Creates and manages the user interface with three main tabs:
  - DeCA: Main dense correspondence workflow
  - DeCAL: Dense landmarking functionality
  - Visualize Results: Output visualization tools

  Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def setup(self):
    """Builds the module's user interface with tabbed layout."""
    ScriptedLoadableModuleWidget.setup(self)  # Initializes base widget

    # Sets up tabs to organize workflow into logical sections
    tabsWidget = qt.QTabWidget()  # Creates main tab container
    DeCATab = qt.QWidget()  # Creates widget for DeCA workflow
    DeCATabLayout = qt.QFormLayout(DeCATab)  # Sets form layout for DeCA tab
    DeCALTab = qt.QWidget()  # Creates widget for DeCAL workflow
    DeCALTabLayout = qt.QFormLayout(DeCALTab)  # Sets form layout for DeCAL tab
    visualizeTab = qt.QWidget()  # Creates widget for visualization
    visualizeTabLayout = qt.QFormLayout(visualizeTab)  # Sets form layout for visualization

    tabsWidget.addTab(DeCATab, "DeCA")  # Adds main DeCA tab
    tabsWidget.addTab(DeCALTab, "DeCAL")  # Adds landmarking tab
    tabsWidget.addTab(visualizeTab, "Visualize Results")  # Adds visualization tab

    self.layout.addWidget(tabsWidget)  # Adds tab widget to main layout

    ################################### DeCA Tab ###################################
    # Creates layout within the DeCA tab for main workflow
    decaWidget=ctk.ctkCollapsibleButton()  # Creates collapsible section
    DeCAWidgetLayout = qt.QFormLayout(decaWidget)  # Sets form layout
    decaWidget.text = "Dense Correspondence I/O"  # Sets section title
    DeCATabLayout.addRow(decaWidget)  # Adds to tab layout

    #
    # Select Atlas Type
    #
    self.calculateAtlasOptionDC=qt.QRadioButton()  # Creates option to generate atlas
    self.calculateAtlasOptionDC.setChecked(True)  # Sets as default option
    self.loadAtlasOptionDC=qt.QRadioButton()  # Creates option to load existing atlas
    DCAtlasButtonGroup = qt.QButtonGroup(decaWidget)  # Groups radio buttons
    DCAtlasButtonGroup.addButton(self.calculateAtlasOptionDC)  # Adds calculate option to group
    DCAtlasButtonGroup.addButton(self.loadAtlasOptionDC)  # Adds load option to group
    DeCAWidgetLayout.addRow("Create atlas: ", self.calculateAtlasOptionDC)  # Adds to form
    DeCAWidgetLayout.addRow("Load atlas: ", self.loadAtlasOptionDC)  # Adds to form

    #
    # Hidden atlas options - Shows additional settings when atlas loading is selected
    self.atlasCollapsibleButtonDC = ctk.ctkCollapsibleButton()  # Creates collapsible section
    self.atlasCollapsibleButtonDC.text = "Atlas Options"  # Sets section label
    self.atlasCollapsibleButtonDC.collapsed = True  # Starts collapsed
    self.atlasCollapsibleButtonDC.enabled = False  # Starts disabled until atlas option selected
    DeCAWidgetLayout.addRow(self.atlasCollapsibleButtonDC)  # Adds to layout
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
    #NEW
    #select texture directory
    #
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
    self.removeScaleCheckBoxDC = qt.QCheckBox()  # Creates checkbox for scale option
    self.removeScaleCheckBoxDC.checked = False  # Defaults to keeping scale
    self.removeScaleCheckBoxDC.setToolTip("If checked, DeCA alignment will include isotropic scaling.")  # Provides helpful tooltip
    DeCAWidgetLayout.addRow("Remove scale: ", self.removeScaleCheckBoxDC)  # Adds to form

    #
    # Error checking directory option
    #
    self.writeErrorCheckBox = qt.QCheckBox()  # Creates checkbox for error checking
    self.writeErrorCheckBox.checked = False  # Defaults to no error checking output
    self.writeErrorCheckBox.setToolTip("If checked, DeCA will create a directory of results for use in estimating point correspondence error.")  # Explains purpose
    DeCAWidgetLayout.addRow("Create output for error checking: ", self.writeErrorCheckBox)  # Adds to form

    #
    # Run DeCA Button
    #
    self.applyButtonDC = qt.QPushButton("Run DeCA")  # Creates main execution button
    self.applyButtonDC.toolTip = "Run non-rigid alignment"  # Provides button description
    self.applyButtonDC.enabled = False  # Starts disabled until inputs are valid
    DeCAWidgetLayout.addRow(self.applyButtonDC)  # Adds button to layout

    #
    # Log Information
    #
    self.logInfoDC = qt.QPlainTextEdit()  # Creates text area for log output
    self.logInfoDC.setPlaceholderText("DeCA log information")  # Shows placeholder text
    self.logInfoDC.setReadOnly(True)  # Makes log read-only
    DeCAWidgetLayout.addRow(self.logInfoDC)  # Adds log widget to form

    # Connects UI signals to handler methods
    self.analysisTypeShape.connect('toggled(bool)', self.onToggleAnalysis)  # Handles analysis type toggle
    self.analysisTypeSymmetry.connect('toggled(bool)', self.onToggleAnalysis)  # Handles symmetry toggle
    self.calculateAtlasOptionDC.connect('toggled(bool)', self.onToggleAtlasDC)  # Handles atlas creation option
    self.loadAtlasOptionDC.connect('toggled(bool)', self.onToggleAtlasDC)  # Handles atlas loading option
    self.DCBaseModelSelector.connect('validInputChanged(bool)', self.onParameterSelectDC)  # Validates base model
    self.DCBaseLMSelector.connect('validInputChanged(bool)', self.onParameterSelectDC)  # Validates base landmarks
    self.meshDirectoryDC.connect('validInputChanged(bool)', self.onParameterSelectDC)  # Validates mesh directory
    self.landmarkDirectoryDC.connect('validInputChanged(bool)', self.onParameterSelectDC)  # Validates landmark directory
    self.outputDirectoryDC.connect('validInputChanged(bool)', self.onParameterSelectDC)  # Validates output directory
    self.applyButtonDC.connect('clicked(bool)', self.onDCApplyButton)  # Connects run button to handler

    ################################### DeCAL Tab ###################################
    # Creates layout within the DeCAL tab for landmarking workflow
    DeCALWidget=ctk.ctkCollapsibleButton()  # Creates collapsible section for DeCAL
    DeCALWidgetLayout = qt.QFormLayout(DeCALWidget)  # Sets form layout for controls
    DeCALWidget.text = "Dense Correspondence Landmarking"  # Sets section title
    DeCALTabLayout.addRow(DeCALWidget)  # Adds widget to tab

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
    # Set spacing tolerance for point density
    #
    self.spacingTolerance = ctk.ctkSliderWidget()  # Creates slider for spacing control
    self.spacingTolerance.singleStep = .1  # Sets increment step
    self.spacingTolerance.minimum = 0  # Sets minimum value
    self.spacingTolerance.maximum = 10  # Sets maximum value
    self.spacingTolerance.value = 4  # Sets default value to 4%
    self.spacingTolerance.setToolTip("Set tolerance of spacing as a percentage of the image diagonal")  # Explains parameter
    DeCALWidgetLayout.addRow("Point density adjustment: ", self.spacingTolerance)  # Adds to form

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
    dateTimeStamp = datetime.now().strftime('%Y_%m-%d_%H_%M_%S')
    outputFolderDC = os.path.join(outDir, dateTimeStamp)
    fileNameDictionary = {}
    try:
      os.makedirs(outputFolderDC)
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
    atlasPathSelected = bool(self.DCBaseModelSelector.currentPath and self.DCBaseLMSelector.currentPath) or self.calculateAtlasOptionDC.checked
    inputPathsSelected = bool(self.meshDirectoryDC.currentPath and self.landmarkDirectoryDC.currentPath and self.outputDirectoryDC.currentPath and self.textureDirectoryDC.currentPath)
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
    logic = decaLogic()
    #set up output directory
    self.folderNames = self.setUpDeCADir(self.OutputDirectoryDCL.currentPath, False, False, True, self.loadAtlasOptionDCL.checked)
    if self.folderNames == {}:
      self.logInfoDCL.appendPlainText(f'Output folders could not be created in {self.OutputDirectoryDCL.currentPath}')
      return
    self.folderNames['originalLMs'] = self.landmarkDirectoryDCL.currentPath
    self.folderNames['originalModels'] = self.meshDirectoryDCL.currentPath
    self.folderNames['originalTextures']=self.textureDirectoryDCL.currentPath
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
    logic = decaLogic()
    closestToMeanLandmarkPath = logic.getClosestToMeanPath(self.folderNames['originalLMs'])
    tempBaseLMs = slicer.util.loadMarkups(os.path.join(self.folderNames['originalLMs'],closestToMeanLandmarkPath))
    subjectID = Path(closestToMeanLandmarkPath)
    while subjectID.suffix in {'.fcsv', '.mrk', '.json'}:
      subjectID = subjectID.with_suffix('')
    log.appendPlainText(f"Closest sample to mean: {subjectID}")
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
    logic = decaLogic()
    subsampledTemplate, pointNumber = logic.runCheckPoints(self.atlasModel, self.spacingTolerance.value)
    self.logInfoDCL.appendPlainText(f'The subsampled template has a total of {pointNumber} points.')
    self.DCLApplyButton.enabled = True

  def onDCApplyButton(self):
    logic = decaLogic()
    #set up output directory
    symmetryOption = self.analysisTypeSymmetry.checked
    writeErrorOption = self.writeErrorCheckBox.checked
    loadAtlasOption = self.loadAtlasOptionDC.checked
    removeScaleOption = self.removeScaleCheckBoxDC.checked
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
  """Logic class implementing DeCA analysis algorithms.

  Contains the core computational methods for:
  - Atlas generation from specimen sets
  - Dense correspondence mapping
  - Procrustes alignment and registration
  - Landmark manipulation and subsampling
  - Mesh processing and transformation
  - Statistical shape analysis

  The interface is designed to be independent of the GUI,
  allowing batch processing and scripted automation.

  Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """
  def runSubsetLandmarks(self, baseNode, lmDirectory, lmDirectorySubset):
    """Creates subset of landmarks based on selected points.

    Processes all landmark files in a directory to keep only the
    points that are selected in the base node.

    Args:
      baseNode: Reference markup node with selected/unselected points
      lmDirectory: Source directory containing landmark files
      lmDirectorySubset: Output directory for subset landmarks
    """
    deletionIndex = []  # Collects indices of points to remove
    for i in range(baseNode.GetNumberOfControlPoints()):
      if not baseNode.GetNthControlPointSelected(i):  # Checks if point is unselected
        deletionIndex.append(i)  # Marks index for deletion
    for lmFileName in os.listdir(lmDirectory):
      if(not lmFileName.startswith(".")):  # Skips hidden files
        currentLMNode = slicer.util.loadMarkups(os.path.join(lmDirectory, lmFileName))  # Loads landmark file
        for index in reversed(deletionIndex):  # Processes in reverse to maintain indices
          currentLMNode.RemoveNthControlPoint(index)  # Removes unselected point
      slicer.util.saveNode(currentLMNode, os.path.join(lmDirectorySubset, lmFileName))  # Saves subset file
      slicer.mrmlScene.RemoveNode(currentLMNode)  # Cleans up scene

  def runCheckPoints(self, atlasNode, spacingTolerance):
    spacingPercentage = spacingTolerance/100
    templateModel = self.downsampleModel(atlasNode, spacingPercentage)
    return templateModel, templateModel.GetNumberOfPoints()

  def runDeCAL(self, baseNode, baseLMPath, meshDirectory, landmarkDirectory, outputDirectory, spacingTolerance):
    spacingPercentage = spacingTolerance/100
    loadOption=False
    baseLandmarks=self.fiducialNodeToPolyData(baseLMPath, loadOption).GetPoints()
    modelExt=['ply','stl','vtp', 'vtk']
    self.modelNames, models = self.importMeshes(meshDirectory, modelExt)
    landmarkNames, landmarks = self.importLandmarks(landmarkDirectory)
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
    if optionErrorOutput:
      self.errorCheckPath = os.path.join(outputDirectory, "errorChecking")
      if not os.path.exists(self.errorCheckPath):
        os.mkdir(self.errorCheckPath)
    baseNode = slicer.util.loadModel(baseMeshPath)
    baseMesh = baseNode.GetPolyData()
    baseLandmarks=self.fiducialNodeToPolyData(baseLMPath).GetPoints()
    modelExt=['ply','stl','vtp']
    self.modelNames, models = self.importMeshes(meshDirectory, modelExt)
    landmarkNames,landmarks = self.importLandmarks(landmarkDirectory)
    denseCorrespondenceGroup = self.denseCorrespondenceBaseMesh(landmarks, models, baseMesh, baseLandmarks, textureDirectory)
    self.addMagnitudeFeature(denseCorrespondenceGroup, self.modelNames, baseMesh)
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
    semilandmarkOption = bool(slmDirectory and outputSLMDirectoryy)
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
    fiducialGroup = vtk.vtkMultiBlockDataGroupFilter()
    fileNameList = []
    for file in sorted(os.listdir(topDir)):
      if file.endswith(".fcsv") or file.endswith(".json"):
        fileNameList.append(file)
        inputFilePath = os.path.join(topDir, file)
        # may want to replace with vtk reader
        polydataPoints = self.fiducialNodeToPolyData(inputFilePath)
        fiducialGroup.AddInputData(polydataPoints)
    fiducialGroup.Update()
    return fileNameList, fiducialGroup.GetOutput()

  def importMeshes(self, topDir, extensions):
      modelGroup = vtk.vtkMultiBlockDataGroupFilter()
      fileNameList = []
      for file in sorted(os.listdir(topDir)):
        if file.endswith(tuple(extensions)):
          base, ext = os.path.splitext(file)
          fileNameList.append(base)
          inputFilePath = os.path.join(topDir, file)
          # may want to replace with vtk reader
          modelNode = slicer.util.loadModel(inputFilePath)
          modelGroup.AddInputData(modelNode.GetPolyData())
          slicer.mrmlScene.RemoveNode(modelNode)
      modelGroup.Update()
      return fileNameList, modelGroup.GetOutput()

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
    meanWarpedBase = meanTransformBaseFilter.GetOutput()

    # write ouput
    if hasattr(self,"errorCheckPath"):
      plyWriterSubject = vtk.vtkPLYWriter()
      plyName = "subject_" + self.modelNames[iteration] + ".ply"
      plyPath = os.path.join(self.errorCheckPath, plyName)
      plyWriterSubject.SetFileName(plyPath)
      plyWriterSubject.SetInputData(meanWarpedMesh)
      plyWriterSubject.Write()

      plyWriterBase = vtk.vtkPLYWriter()
      plyName = "base.ply"
      plyPath = os.path.join(self.errorCheckPath, plyName)
      plyWriterBase.SetFileName(plyPath)
      plyWriterBase.SetInputData(meanWarpedBase)
      plyWriterBase.Write()

    # Dense correspondence
    cellLocator = vtk.vtkCellLocator()
    cellLocator.SetDataSet(meanWarpedMesh)
    cellLocator.BuildLocator()

    point = [0,0,0]
    correspondingPoint = [0,0,0]
    correspondingPoints = vtk.vtkPoints()
    cellId = vtk.reference(0)
    subId = vtk.reference(0)
    distance = vtk.reference(0.0)
    for i in range(meanWarpedBase.GetNumberOfPoints()):
      meanWarpedBase.GetPoint(i,point)
      cellLocator.FindClosestPoint(point,correspondingPoint,cellId, subId, distance)
      correspondingPoints.InsertPoint(i,correspondingPoint)

    #Copy points into mesh with base connectivity
    correspondingMesh = vtk.vtkPolyData()
    correspondingMesh.SetPoints(correspondingPoints)
    correspondingMesh.SetPolys(meanWarpedBase.GetPolys())

    # Apply inverse warping
    inverseTransform = vtk.vtkThinPlateSplineTransform()
    inverseTransform.SetSourceLandmarks(meanShape)
    inverseTransform.SetTargetLandmarks(originalLandmarks)
    inverseTransform.SetBasisToR() # for 3D transform

    inverseTransformFilter = vtk.vtkTransformPolyDataFilter()
    inverseTransformFilter.SetInputData(correspondingMesh)
    inverseTransformFilter.SetTransform(inverseTransform)
    inverseTransformFilter.Update()
    finalMesh = inverseTransformFilter.GetOutput()
    #NEW LINE FOR COLOR
    colorWrappedMesh = self.wrapTextureFromImage(originalMesh, textureImageNode, finalMesh, saveAsPointData)

    return colorWrappedMesh

  def convertPointsToVTK(self, points):
    array_vtk = vtk_np.numpy_to_vtk(points, deep=True, array_type=vtk.VTK_FLOAT)
    points_vtk = vtk.vtkPoints()
    points_vtk.SetData(array_vtk)
    polydata_vtk = vtk.vtkPolyData()
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
