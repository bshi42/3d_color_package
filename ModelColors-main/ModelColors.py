"""
3D Slicer module for texture-based manipulation and segmentation of 3D models.

Provides functionality for applying textures to 3D models, scaling geometries,
and performing color-based segmentation using k-means clustering. Integrates
with 3D Slicer's visualization and model management capabilities.
"""

import random
import os
import unittest
from __main__ import vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *
import logging

import numpy as np
import vtk.util.numpy_support as nps
#from sklearn.cluster import KMeans  # Requires scikit-learn installation

#
# ModelColors
#

class ModelColors(ScriptedLoadableModule):
  """
  Module class for texture-based model manipulation in 3D Slicer.

  Extends ScriptedLoadableModule to provide a plugin interface for
  applying textures, scaling models, and performing color-based segmentation.
  Integrates with the 3D Slicer application framework.

  Base class documentation:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, parent):
    """
    Initializes the ModelColors module with metadata and configuration.

    Args:
      parent: Parent object from 3D Slicer framework
    """
    ScriptedLoadableModule.__init__(self, parent)

    # Sets module metadata for 3D Slicer's module browser
    self.parent.title = "Model Colors"
    self.parent.categories = ["Surface Models"]
    self.parent.dependencies = []  # No external module dependencies
    self.parent.contributors = ["Arthur Porto (UF)" ]

    # Provides user-facing documentation
    self.parent.helpText = """This module applies a texture (stored in a volume node) to a model node and optionally scales or segments it.
It is typically used to display colored surfaces, provided by surface scanners, exported in OBJ format.
The model must contain texture coordinates. Only a single texture file per model is supported. This module is largely based on SlicerIGT's Texture Model module.
For more information, visit <a href='https://github.com/SlicerIGT/SlicerIGT/#user-documentation'>SlicerIGT project website</a>.
"""
    self.parent.acknowledgementText = """ """ # replace with organization, grant and thanks.

#
# ModelColorsWidget
#

class ModelColorsWidget(ScriptedLoadableModuleWidget):
  """
  GUI widget for the ModelColors module.

  Creates and manages the user interface for texture application,
  model scaling, and color-based segmentation. Handles user interactions
  and connects UI elements to processing logic.

  Base class documentation:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def setup(self):
    """
    Builds the module's user interface.

    Creates all UI elements including input selectors, parameter controls,
    and action buttons. Establishes connections between UI elements and
    their corresponding event handlers.
    """
    ScriptedLoadableModuleWidget.setup(self)

    # Instantiate and connect widgets ...

    #
    # Parameters Area
    #
    # Creates collapsible section for main parameters
    parametersCollapsibleButton = ctk.ctkCollapsibleButton()
    parametersCollapsibleButton.text = "Parameters"
    self.layout.addWidget(parametersCollapsibleButton)

    # Layout within the dummy collapsible button
    parametersFormLayout = qt.QFormLayout(parametersCollapsibleButton)

    #
    # input volume selector
    #
    # Creates selector for 3D model input
    self.inputModelSelector = slicer.qMRMLNodeComboBox()
    self.inputModelSelector.nodeTypes = [ "vtkMRMLModelNode" ]  # Restricts to model nodes only
    self.inputModelSelector.addEnabled = False  # Prevents creating new nodes from this selector
    self.inputModelSelector.removeEnabled = True  # Allows removing nodes
    self.inputModelSelector.renameEnabled = True  # Allows renaming nodes
    self.inputModelSelector.noneEnabled = False  # Requires selection
    self.inputModelSelector.showHidden = False  # Hides hidden nodes
    self.inputModelSelector.showChildNodeTypes = False  # Shows only parent types
    self.inputModelSelector.setMRMLScene( slicer.mrmlScene )  # Connects to scene
    self.inputModelSelector.setToolTip( "Model node containing geometry and texture coordinates." )
    parametersFormLayout.addRow("Model: ", self.inputModelSelector)

    # input texture selector
    # Creates selector for texture image input
    self.inputTextureSelector = slicer.qMRMLNodeComboBox()
    self.inputTextureSelector.nodeTypes = [ "vtkMRMLVectorVolumeNode" ]  # RGB/color images
    self.inputTextureSelector.addEnabled = False
    self.inputTextureSelector.removeEnabled = True
    self.inputTextureSelector.renameEnabled = True
    self.inputTextureSelector.noneEnabled = False
    self.inputTextureSelector.showHidden = False
    self.inputTextureSelector.showChildNodeTypes = False
    self.inputTextureSelector.setMRMLScene( slicer.mrmlScene )
    self.inputTextureSelector.setToolTip( "Color image containing texture image." )
    parametersFormLayout.addRow("Texture: ", self.inputTextureSelector)

    # Creates dropdown for color data storage format
    self.addColorAsPointAttributeComboBox = qt.QComboBox()
    self.addColorAsPointAttributeComboBox.addItem("disabled")  # No color storage
    self.addColorAsPointAttributeComboBox.addItem("RGB vector", "uchar-vector")  # 0-255 range
    self.addColorAsPointAttributeComboBox.addItem("RGB float vector", "float-vector")  # 0-1 range
    self.addColorAsPointAttributeComboBox.addItem("RGB float components", "float-components")  # Separate channels
    self.addColorAsPointAttributeComboBox.setCurrentIndex(0)
    self.addColorAsPointAttributeComboBox.setToolTip('Save color in point data.'
      ' "RGB vector" is recommended for compatibility with most software.'
      ' The point data may be used for thresholding or color-based processing.')
    parametersFormLayout.addRow("Save color information as point data: ", self.addColorAsPointAttributeComboBox)
        
    # Add a field for the scaling factor
    # Creates numeric input for model scaling
    self.scaleFactorSpinBox = qt.QDoubleSpinBox()
    self.scaleFactorSpinBox.setDecimals(3)  # Allows 3 decimal precision
    self.scaleFactorSpinBox.setMinimum(0.001)  # Prevents zero or negative scaling
    self.scaleFactorSpinBox.setMaximum(1000)  # Reasonable upper limit
    self.scaleFactorSpinBox.setValue(1.0)  # Default is no scaling
    parametersFormLayout.addRow("Scale Factor (e.g., 0.001 for mm to m):", self.scaleFactorSpinBox)

    # Add a button for applying scale
    # Creates button to trigger scaling operation
    self.applyScaleButton = qt.QPushButton("Apply Scale")
    self.applyScaleButton.toolTip = "Apply scale to selected model."
    self.applyScaleButton.enabled = False  # Initially disabled until model selected
    parametersFormLayout.addRow(self.applyScaleButton)

    #
    # Apply Button
    #
    # Creates button to trigger texture application
    self.applyButton = qt.QPushButton("Apply Texture")
    self.applyButton.toolTip = "Apply texture to selected model."
    self.applyButton.enabled = False  # Initially disabled until inputs selected
    parametersFormLayout.addRow(self.applyButton)

    #
    # Segmentation Area
    #
    # Creates collapsible section for color-based segmentation
    segmentationCollapsibleButton = ctk.ctkCollapsibleButton()
    segmentationCollapsibleButton.text = "Color Segmentation"
    self.layout.addWidget(segmentationCollapsibleButton)

    # Layout within the dummy collapsible button
    segmentationFormLayout = qt.QFormLayout(segmentationCollapsibleButton)
    
    # Creates selector for model to segment
    self.segModelSelector = slicer.qMRMLNodeComboBox()
    self.segModelSelector.nodeTypes = [ "vtkMRMLModelNode" ]
    self.segModelSelector.addEnabled = False
    self.segModelSelector.removeEnabled = True
    self.segModelSelector.renameEnabled = True
    self.segModelSelector.noneEnabled = False
    self.segModelSelector.showHidden = False
    self.segModelSelector.showChildNodeTypes = False
    self.segModelSelector.setMRMLScene( slicer.mrmlScene )
    self.segModelSelector.setToolTip( "Model node containing geometry and texture coordinates." )
    segmentationFormLayout.addRow("Model: ", self.segModelSelector)

    # Create a QSpinBox for the number of clusters
    # Creates numeric input for k-means cluster count
    self.clusterNumberSpinBox = qt.QSpinBox()
    self.clusterNumberSpinBox.setMinimum(1)  # At least one cluster required
    self.clusterNumberSpinBox.setMaximum(15)  # Practical upper limit for visualization
    self.clusterNumberSpinBox.setValue(5)  # Common default for segmentation tasks
    self.clusterNumberSpinBox.setToolTip("Enter the number of clusters for color-based segmentation.")

    # Add the spin box to your form layout
    segmentationFormLayout.addRow("Number of clusters: ", self.clusterNumberSpinBox)

    #
    # Segment Button
    #
    # Creates button to trigger segmentation
    self.segmentButton = qt.QPushButton("Segment")
    self.segmentButton.toolTip = "Segment texture and split model."
    self.segmentButton.enabled = False  # Initially disabled until model selected
    segmentationFormLayout.addRow(self.segmentButton)

    #
    # Models Area
    #
    # Creates collapsible section for model management
    modelsCollapsibleButton = ctk.ctkCollapsibleButton()
    modelsCollapsibleButton.text = "Models"
    self.layout.addWidget(modelsCollapsibleButton)
    modelsFormLayout = qt.QFormLayout(modelsCollapsibleButton)

    #
    # Models module
    #
    # Embeds Slicer's built-in models module for model visualization control
    modelsModule = slicer.modules.models.createNewWidgetRepresentation()
    displayCollapsibleButton = modelsModule.findChild('ctkCollapsibleButton', 'DisplayButton')
    displayCollapsibleButton.collapsed = True  # Collapses display options by default

    # Set the scene in the models module
    modelsModule.setMRMLScene(slicer.app.mrmlScene())
    modelsFormLayout.addRow(modelsModule)

    # connections
    # Establishes signal-slot connections for UI interactions
    self.applyButton.connect('clicked(bool)', self.onApplyButton)
    self.applyScaleButton.connect('clicked(bool)', self.onApplyScaleButton)
    self.segmentButton.connect('clicked(bool)', self.onSegmentButton)
    self.segModelSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onSelect)
    self.inputModelSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onSelect)
    self.inputTextureSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onSelect)

    # Add vertical spacer
    self.layout.addStretch(1)  # Pushes content to top of widget

    # Refresh Apply button state
    self.onSelect()  # Initializes button states based on current selections

  def cleanup(self):
    """
    Performs cleanup when module is closed.

    Called when the module widget is destroyed. Currently no cleanup
    needed as connections are automatically removed.
    """
    pass

  def onSelect(self):
    """
    Updates button states based on current node selections.

    Enables or disables action buttons depending on whether required
    inputs have been selected. Called whenever node selections change.
    """
    # Enables texture button only when both model and texture are selected
    self.applyButton.enabled = self.inputTextureSelector.currentNode() and self.inputModelSelector.currentNode()
    # Enables segment button only when a model is selected for segmentation
    self.segmentButton.enabled = self.segModelSelector.currentNode()
    # Enables scale button only when a model is selected
    self.applyScaleButton.enabled = self.inputModelSelector.currentNode()

  def onApplyButton(self):
    """
    Handles texture application button click.

    Applies the selected texture to the selected model, optionally
    saving color data as point attributes. Shows wait cursor during
    processing and displays errors if operation fails.
    """
    try:
      # Shows wait cursor during processing
      qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)

      # Creates logic instance and applies texture
      logic = ModelColorsLogic()
      logic.applyTexture(self.inputModelSelector.currentNode(), self.inputTextureSelector.currentNode(),
        self.addColorAsPointAttributeComboBox.currentData)

      # Restores normal cursor
      qt.QApplication.restoreOverrideCursor()
    except Exception as e:
      # Ensures cursor is restored even if error occurs
      qt.QApplication.restoreOverrideCursor()
      # Displays error message to user
      slicer.util.errorDisplay("Failed to compute results: "+str(e))
      import traceback
      traceback.print_exc()  # Logs full traceback for debugging 

  def onSegmentButton(self):
    """
    Handles segmentation button click.

    Performs k-means clustering on model vertex colors to segment
    the model into color-based regions. Creates separate child models
    for each cluster with appropriate coloring.
    """
    try:
      # Shows wait cursor during processing
      qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)
      logic = ModelColorsLogic()

      # Gets selected model and cluster count
      modelNode = self.segModelSelector.currentNode()
      num_clusters = self.clusterNumberSpinBox.value

      # Verifies that model has color data
      if not modelNode.GetPolyData().GetPointData().HasArray('RGB'):
          slicer.util.errorDisplay('Model does not have RGB array')
          return

      # Accesses the vertex colors (RGB)
      colorArray = nps.vtk_to_numpy(modelNode.GetPolyData().GetPointData().GetArray('RGB'))

      # Performs k-means clustering on color data
      kmeans = KMeans(n_clusters=num_clusters, n_init=15)  # n_init=15 for stable results
      clusters = kmeans.fit_predict(colorArray[:, :3])  # Uses RGB, excludes alpha if present
      centroid_colors = kmeans.cluster_centers_  # Average color for each cluster

      # Creates child meshes for each cluster
      for cluster_id in range(num_clusters):
          # Creates boolean mask for vertices in this cluster
          mask = clusters == cluster_id
          centroid_color = centroid_colors[cluster_id]
          # Generates new model for this cluster
          logic.create_child_model(modelNode, mask, cluster_id, centroid_color)

      qt.QApplication.restoreOverrideCursor()

    except Exception as e:
      # Ensures cursor is restored even if error occurs
      qt.QApplication.restoreOverrideCursor()
      # Displays error message to user
      slicer.util.errorDisplay("Failed to compute results: "+str(e))
      import traceback
      traceback.print_exc()  # Logs full traceback for debugging 
    
      
  def onApplyScaleButton(self):
      """
      Handles scale button click.

      Applies uniform scaling to the selected model based on the
      scale factor specified in the UI. Useful for unit conversions
      (e.g., mm to m).
      """
      # Gets scale factor from UI
      scaleFactor = self.scaleFactorSpinBox.value
      # Gets selected model
      modelNode = self.inputModelSelector.currentNode()

      if modelNode:
          try:
              # Shows wait cursor during processing
              qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)

              # Creates logic instance and applies scaling
              logic = ModelColorsLogic()
              logic.applyScaling(modelNode, scaleFactor)

              # Restores normal cursor
              qt.QApplication.restoreOverrideCursor()
          except Exception as e:
              # Ensures cursor is restored even if error occurs
              qt.QApplication.restoreOverrideCursor()
              # Displays error message to user
              slicer.util.errorDisplay("Failed to apply scaling: "+str(e))
              import traceback
              traceback.print_exc()  # Logs full traceback for debugging

#
# ModelColorsLogic
#
class ModelColorsLogic(ScriptedLoadableModuleLogic):
  """
  Processing logic for texture application and model manipulation.

  Implements the core algorithms for applying textures to 3D models,
  performing color-based segmentation, and scaling geometries. Separated
  from UI code to enable testing and reuse.

  Base class documentation:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def applyTexture(self, modelNode, textureImageNode, saveAsPointData=None):
    """
    Applies texture to model node with optional point data storage.

    Args:
      modelNode: VTK model node to apply texture to
      textureImageNode: VTK image node containing the texture
      saveAsPointData: Storage format for color data:
        - None: Don't save as point data (texture only)
        - 'uchar-vector': RGB values as unsigned char (0-255)
        - 'float-vector': RGB values as float (0-1)
        - 'float-components': Separate R, G, B float arrays
    """
    # Applies texture for visualization
    self.showTextureOnModel(modelNode, textureImageNode)

    # Optionally saves color data as vertex attributes
    if saveAsPointData:
      print(f"saveAsPointData: {saveAsPointData}")
      self.convertTextureToPointAttribute(modelNode, textureImageNode, saveAsPointData)

  # Show texture
  def showTextureOnModel(self, modelNode, textureImageNode):
    """
    Displays texture on 3D model surface.

    Flips texture vertically to match UV coordinate convention and
    applies it to the model's display node for visualization.

    Args:
      modelNode: Target model for texture display
      textureImageNode: Source texture image
    """
    # Gets display node for model
    modelDisplayNode = modelNode.GetDisplayNode()
    # Disables backface culling to show both sides of surfaces
    modelDisplayNode.SetBackfaceCulling(0)

    # Creates vertical flip filter (textures often need Y-axis flip)
    textureImageFlipVert = vtk.vtkImageFlip()
    textureImageFlipVert.SetFilteredAxis(1)  # Flips along Y axis
    textureImageFlipVert.SetInputConnection(textureImageNode.GetImageDataConnection())

    # Applies flipped texture to model
    modelDisplayNode.SetTextureImageDataConnection(textureImageFlipVert.GetOutputPort())

  # Add texture data to scalars
  def convertTextureToPointAttribute(self, modelNode, textureImageNode, saveAsPointData):
    """
    Maps texture image colors to vertex points of a mesh.

    Samples texture colors at UV coordinates and stores them as
    point data attributes for further processing (e.g., segmentation).

    Args:
      modelNode: VTK model node containing the mesh
      textureImageNode: VTK image node containing the texture
      saveAsPointData: Storage format ('uchar-vector', 'float-vector', 'float-components')
    """
    # Retrieves polydata from the model node
    polyData = modelNode.GetPolyData()

    # Retrieves texture coordinates from the polydata
    tcoords_vtk = polyData.GetPointData().GetTCoords()  # UV coordinates
    tcoords_np = nps.vtk_to_numpy(tcoords_vtk)  # Converts to numpy for processing

    # Flips texture image vertically to match UV conventions
    textureImageFlipVert = vtk.vtkImageFlip()
    textureImageFlipVert.SetFilteredAxis(1)  # Y-axis flip
    textureImageFlipVert.SetInputConnection(textureImageNode.GetImageDataConnection())
    textureImageFlipVert.Update()  # Forces pipeline execution
    textureImageData = textureImageFlipVert.GetOutput()
    
    # Converts texture image to NumPy array
    # Reshapes flat array into 2D image with color channels
    textureImage_np = nps.vtk_to_numpy(textureImageData.GetPointData().GetScalars()).reshape(
        textureImageData.GetDimensions()[1], textureImageData.GetDimensions()[0], -1)

    # Normalizes and scales texture coordinates
    print("Original Texture Coordinates: ", tcoords_np)

    # Scales the texture coordinates from [0,1] to pixel indices
    # Ensures coordinates are within valid range before scaling
    width, height = textureImage_np.shape[1], textureImage_np.shape[0]
    uv_scaled = np.clip(tcoords_np, 0, 1) * [width - 1, height - 1]  # -1 for 0-based indexing

    print("Scaled UV Coordinates: ", uv_scaled)

    # Maps texture coordinates to texture colors (vectorized operation)
    # Samples texture at each vertex's UV coordinate
    colors = textureImage_np[uv_scaled[:, 1].astype(int), uv_scaled[:, 0].astype(int)]

    # Converts results back to VTK and saves as point data
    if saveAsPointData == 'uchar-vector':
        # Stores as unsigned char RGB vector (0-255 range)
        colorArray_vtk = nps.numpy_to_vtk(colors, deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
        colorArray_vtk.SetName('RGB')
        polyData.GetPointData().SetScalars(colorArray_vtk)
    elif saveAsPointData == 'float-vector':
        # Stores as normalized float RGB vector (0-1 range)
        colorArray_vtk = nps.numpy_to_vtk(colors.astype(float) / 255.0, deep=True, array_type=vtk.VTK_FLOAT)
        colorArray_vtk.SetName('Color')
        polyData.GetPointData().SetScalars(colorArray_vtk)
    elif saveAsPointData == 'float-components':
        # Stores as separate float arrays for R, G, B channels
        for i, name in enumerate(['ColorRed', 'ColorGreen', 'ColorBlue']):
            componentArray_vtk = nps.numpy_to_vtk(colors[:, i].astype(float), deep=True, array_type=vtk.VTK_FLOAT)
            componentArray_vtk.SetName(name)
            polyData.GetPointData().AddArray(componentArray_vtk)
    else:
        raise ValueError(f"Invalid saveAsPointData: {saveAsPointData}")

    # Marks the polydata as modified to trigger rendering update
    polyData.Modified()

  # Function to create a child model from a mask
  def create_child_model(self, modelNode, mask, cluster_id, centroid_color):
      """
      Creates a new model containing only vertices belonging to a specific cluster.

      Extracts a subset of the original mesh based on a boolean mask,
      creating a new independent model with its own display properties.

      Args:
        modelNode: Original model to extract from
        mask: Boolean array indicating which vertices belong to this cluster
        cluster_id: Numeric identifier for the cluster
        centroid_color: RGB color representing the cluster's average color
      """

      # Retrieves mesh geometry from original model
      polyData = modelNode.GetPolyData()
      points = polyData.GetPoints()  # Vertex positions
      polys = polyData.GetPolys()  # Triangle connectivity

      # Converts VTK Points to NumPy array for manipulation
      np_points = nps.vtk_to_numpy(points.GetData())

      # Converts VTK Polys to NumPy array
      np_polys = nps.vtk_to_numpy(polys.GetData())

      # Reshapes and filters polygons based on the mask
      # VTK stores triangles as [3, v0, v1, v2] where 3 is vertex count
      tri_polys = np_polys.reshape(-1, 4)[:, 1:]  # Removes count, keeps indices
      # Keeps only triangles where all vertices are in the cluster
      is_in_cluster = np.all(mask[tri_polys], axis=1)
      cluster_polys = tri_polys[is_in_cluster]

      # Creates new points and polys for the cluster
      # Finds unique vertex indices and remaps polygon connectivity
      unique_points, new_poly_indices = np.unique(cluster_polys, return_inverse=True)
      new_poly_indices = new_poly_indices.reshape(-1, 3)  # Restores triangle structure

      # Converts back to VTK data structures
      new_vtk_points = vtk.vtkPoints()
      new_vtk_points.SetData(nps.numpy_to_vtk(np_points[unique_points]))  # Uses only needed vertices

      # Rebuilds polygon connectivity
      new_vtk_polys = vtk.vtkCellArray()
      for poly in new_poly_indices:
          new_vtk_polys.InsertNextCell(len(poly), poly)  # Adds each triangle

      # Creates new vtkPolyData for child model
      newPolyData = vtk.vtkPolyData()
      newPolyData.SetPoints(new_vtk_points)  # Sets vertex positions
      newPolyData.SetPolys(new_vtk_polys)  # Sets triangle connectivity

      # Adds new model node to the scene
      newModelNode = slicer.vtkMRMLModelNode()
      newModelNode.SetName(f'Cluster_{cluster_id}')  # Descriptive name for identification
      newModelNode.SetAndObservePolyData(newPolyData)  # Associates geometry with node
      slicer.mrmlScene.AddNode(newModelNode)  # Registers with scene

      # Sets display properties
      # Converts cluster color to display format
      display_color = (centroid_color).astype(int).tolist()

      # Creates display node with cluster's average color
      displayNode = slicer.vtkMRMLModelDisplayNode()
      displayNode.SetColor(display_color[0] / 255.0, display_color[1] / 255.0, display_color[2] / 255.0)  # Normalizes to 0-1
      slicer.mrmlScene.AddNode(displayNode)  # Registers display node
      newModelNode.SetAndObserveDisplayNodeID(displayNode.GetID())  # Links to model
    
  def applyScaling(self, modelNode, scaleFactor):
      """
      Applies uniform scaling to model node.

      Transforms all vertex positions by a uniform scale factor,
      useful for unit conversions (e.g., millimeters to meters).

      Args:
        modelNode: The model to scale
        scaleFactor: The scaling factor (e.g., 0.001 for mm to m)
      """
      # Retrieves polydata from the model node
      polyData = modelNode.GetPolyData()

      # Creates a transformation matrix for scaling
      transformMatrix = vtk.vtkMatrix4x4()
      transformMatrix.Identity()  # Starts with identity matrix
      # Sets diagonal elements for uniform scaling
      for i in range(3):  # Scales x, y, and z coordinates equally
          transformMatrix.SetElement(i, i, scaleFactor)

      # Applies the transformation
      transform = vtk.vtkTransform()
      transform.SetMatrix(transformMatrix)

      # Creates filter to apply transformation to geometry
      transformFilter = vtk.vtkTransformPolyDataFilter()
      transformFilter.SetTransform(transform)
      transformFilter.SetInputData(polyData)
      transformFilter.Update()  # Executes transformation

      # Updates the model with the transformed data
      modelNode.SetAndObservePolyData(transformFilter.GetOutput())

class ModelColorsTest(ScriptedLoadableModuleTest):
  """
  Test case for the ModelColors module.

  Provides automated testing to verify module functionality,
  including texture application and model processing. Helps ensure
  stability across updates and platform changes.

  Base class documentation:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def setUp(self):
    """
    Resets the testing environment.

    Clears the scene to ensure tests run in a clean state,
    preventing interference from previous test data.
    """
    slicer.mrmlScene.Clear(0)  # 0 = clear all nodes

  def runTest(self):
    """
    Executes all test methods.

    Entry point for test execution. Runs setup and then
    executes individual test cases.
    """
    self.setUp()
    self.test_ModelColors1()

  def test_ModelColors1(self):
    """
    Tests basic texture application functionality.

    Downloads sample data, loads a textured model, and verifies
    that texture can be successfully applied. This test ensures
    the core functionality remains intact across updates.

    Test data: Femur head surface scan with texture
    Expected result: Texture successfully applied without errors
    """

    slicer.util.delayDisplay("Starting the test")

    # Downloads test data from Slicer's official repository
    import urllib
    url = 'https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/752ce9afe8b708fcd4f8448612170f8e730670d845f65177860edc0e08004ecf'
    zipFilePath = slicer.app.temporaryPath + '/' + 'FemurHeadSurfaceScan.zip'
    extractPath = slicer.app.temporaryPath + '/' + 'FemurHeadSurfaceScan'

    # Downloads only if not already cached
    if not os.path.exists(zipFilePath) or os.stat(zipFilePath).st_size == 0:
      logging.info('Requesting download from %s...\n' % url)
      urllib.request.urlretrieve(url, zipFilePath)
      slicer.util.delayDisplay('Finished with download\n')

    # Unzips test data
    slicer.util.delayDisplay("Unzipping to %s" % (extractPath))
    qt.QDir().mkpath(extractPath)  # Creates extraction directory
    applicationLogic = slicer.app.applicationLogic()
    applicationLogic.Unzip(zipFilePath, extractPath)  # Extracts archive

    # Loads test model and texture
    slicer.util.loadModel(extractPath+"/head_obj.obj")  # 3D model with UV coordinates
    slicer.util.loadVolume(extractPath+"/head_obj_0.png")  # Texture image

    slicer.util.delayDisplay('Finished with download and loading')

    # Tests texture application
    modelNode = slicer.util.getNode("head_obj")  # Retrieves loaded model
    textureNode = slicer.util.getNode("head_obj_0")  # Retrieves loaded texture

    # Applies texture using module logic
    logic = ModelColorsLogic()
    logic.applyTexture(modelNode, textureNode)

    # Confirms successful execution
    slicer.util.delayDisplay('Test passed!')
