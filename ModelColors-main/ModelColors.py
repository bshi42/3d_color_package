import random
import os
import unittest
from __main__ import vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *
import logging

import numpy as np
import vtk.util.numpy_support as nps
from sklearn.cluster import KMeans

#
# ModelColors
#

class ModelColors(ScriptedLoadableModule):
  """Uses ScriptedLoadableModule base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "Model Colors"
    self.parent.categories = ["Surface Models"]
    self.parent.dependencies = []
    self.parent.contributors = ["Arthur Porto (UF)" ]
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
  """Uses ScriptedLoadableModuleWidget base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def setup(self):
    ScriptedLoadableModuleWidget.setup(self)

    # Instantiate and connect widgets ...

    #
    # Parameters Area
    #
    parametersCollapsibleButton = ctk.ctkCollapsibleButton()
    parametersCollapsibleButton.text = "Parameters"
    self.layout.addWidget(parametersCollapsibleButton)

    # Layout within the dummy collapsible button
    parametersFormLayout = qt.QFormLayout(parametersCollapsibleButton)

    #
    # input volume selector
    #
    self.inputModelSelector = slicer.qMRMLNodeComboBox()
    self.inputModelSelector.nodeTypes = [ "vtkMRMLModelNode" ]
    self.inputModelSelector.addEnabled = False
    self.inputModelSelector.removeEnabled = True
    self.inputModelSelector.renameEnabled = True
    self.inputModelSelector.noneEnabled = False
    self.inputModelSelector.showHidden = False
    self.inputModelSelector.showChildNodeTypes = False
    self.inputModelSelector.setMRMLScene( slicer.mrmlScene )
    self.inputModelSelector.setToolTip( "Model node containing geometry and texture coordinates." )
    parametersFormLayout.addRow("Model: ", self.inputModelSelector)

    #input texture selector
    self.inputTextureSelector = slicer.qMRMLNodeComboBox()
    self.inputTextureSelector.nodeTypes = [ "vtkMRMLVectorVolumeNode" ]
    self.inputTextureSelector.addEnabled = False
    self.inputTextureSelector.removeEnabled = True
    self.inputTextureSelector.renameEnabled = True
    self.inputTextureSelector.noneEnabled = False
    self.inputTextureSelector.showHidden = False
    self.inputTextureSelector.showChildNodeTypes = False
    self.inputTextureSelector.setMRMLScene( slicer.mrmlScene )
    self.inputTextureSelector.setToolTip( "Color image containing texture image." )
    parametersFormLayout.addRow("Texture: ", self.inputTextureSelector)

    self.addColorAsPointAttributeComboBox = qt.QComboBox()
    self.addColorAsPointAttributeComboBox.addItem("disabled")
    self.addColorAsPointAttributeComboBox.addItem("RGB vector", "uchar-vector")
    self.addColorAsPointAttributeComboBox.addItem("RGB float vector", "float-vector")
    self.addColorAsPointAttributeComboBox.addItem("RGB float components", "float-components")
    self.addColorAsPointAttributeComboBox.setCurrentIndex(0)
    self.addColorAsPointAttributeComboBox.setToolTip('Save color in point data.'
      ' "RGB vector" is recommended for compatibility with most software.'
      ' The point data may be used for thresholding or color-based processing.')
    parametersFormLayout.addRow("Save color information as point data: ", self.addColorAsPointAttributeComboBox)
        
    # Add a field for the scaling factor
    self.scaleFactorSpinBox = qt.QDoubleSpinBox()
    self.scaleFactorSpinBox.setDecimals(3)
    self.scaleFactorSpinBox.setMinimum(0.001)  # min value
    self.scaleFactorSpinBox.setMaximum(1000)  # max value
    self.scaleFactorSpinBox.setValue(1.0)  # default value
    parametersFormLayout.addRow("Scale Factor (e.g., 0.001 for mm to m):", self.scaleFactorSpinBox)

    # Add a button for applying scale
    self.applyScaleButton = qt.QPushButton("Apply Scale")
    self.applyScaleButton.toolTip = "Apply scale to selected model."
    self.applyScaleButton.enabled = False
    parametersFormLayout.addRow(self.applyScaleButton)
    #
    # Apply Button
    #
    self.applyButton = qt.QPushButton("Apply Texture")
    self.applyButton.toolTip = "Apply texture to selected model."
    self.applyButton.enabled = False
    parametersFormLayout.addRow(self.applyButton)

    #
    # Segmentation Area
    #
    segmentationCollapsibleButton = ctk.ctkCollapsibleButton()
    segmentationCollapsibleButton.text = "Color Segmentation"
    self.layout.addWidget(segmentationCollapsibleButton)

    # Layout within the dummy collapsible button
    segmentationFormLayout = qt.QFormLayout(segmentationCollapsibleButton)
    
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
    self.clusterNumberSpinBox = qt.QSpinBox()
    self.clusterNumberSpinBox.setMinimum(1)  # Set a minimum value (e.g., 1)
    self.clusterNumberSpinBox.setMaximum(15)  # Set a reasonable maximum value
    self.clusterNumberSpinBox.setValue(5)  # Set a default value
    self.clusterNumberSpinBox.setToolTip("Enter the number of clusters for color-based segmentation.")

    # Add the spin box to your form layout
    segmentationFormLayout.addRow("Number of clusters: ", self.clusterNumberSpinBox)

    #
    # Segment Button
    #
    self.segmentButton = qt.QPushButton("Segment")
    self.segmentButton.toolTip = "Segment texture and split model."
    self.segmentButton.enabled = False
    segmentationFormLayout.addRow(self.segmentButton)

    #
    # Models Area
    #
    modelsCollapsibleButton = ctk.ctkCollapsibleButton()
    modelsCollapsibleButton.text = "Models"
    self.layout.addWidget(modelsCollapsibleButton)
    modelsFormLayout = qt.QFormLayout(modelsCollapsibleButton)

    #
    # Models module
    #
    modelsModule = slicer.modules.models.createNewWidgetRepresentation()
    displayCollapsibleButton = modelsModule.findChild('ctkCollapsibleButton', 'DisplayButton')
    displayCollapsibleButton.collapsed = True

    # Set the scene in the models module
    modelsModule.setMRMLScene(slicer.app.mrmlScene())
    modelsFormLayout.addRow(modelsModule)

    # connections
    self.applyButton.connect('clicked(bool)', self.onApplyButton)
    self.applyScaleButton.connect('clicked(bool)', self.onApplyScaleButton)
    self.segmentButton.connect('clicked(bool)', self.onSegmentButton)
    self.segModelSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onSelect)
    self.inputModelSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onSelect)
    self.inputTextureSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onSelect)

    # Add vertical spacer
    self.layout.addStretch(1)

    # Refresh Apply button state
    self.onSelect()

  def cleanup(self):
    pass

  def onSelect(self):
    self.applyButton.enabled = self.inputTextureSelector.currentNode() and self.inputModelSelector.currentNode()
    self.segmentButton.enabled = self.segModelSelector.currentNode()
    self.applyScaleButton.enabled = self.inputModelSelector.currentNode()

  def onApplyButton(self):
    try:
      qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)
      logic = ModelColorsLogic()
      logic.applyTexture(self.inputModelSelector.currentNode(), self.inputTextureSelector.currentNode(),
        self.addColorAsPointAttributeComboBox.currentData)
      qt.QApplication.restoreOverrideCursor()
    except Exception as e:
      qt.QApplication.restoreOverrideCursor()
      slicer.util.errorDisplay("Failed to compute results: "+str(e))
      import traceback
      traceback.print_exc() 

  def onSegmentButton(self):
    try:
      qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)
      logic = ModelColorsLogic()
      
      modelNode = self.segModelSelector.currentNode()
      num_clusters = self.clusterNumberSpinBox.value

      #Assert that the model has array 'RGBA'
      if not modelNode.GetPolyData().GetPointData().HasArray('RGB'):
          slicer.util.errorDisplay('Model does not have RGB array')
          return

      # Access the vertex colors (RGBA)
      colorArray = nps.vtk_to_numpy(modelNode.GetPolyData().GetPointData().GetArray('RGB'))  # Replace 'Colors'
      # Perform k-means clustering on color data
      kmeans = KMeans(n_clusters=num_clusters, n_init=15)
      clusters = kmeans.fit_predict(colorArray[:, :3])  # Exclude alpha channel
      centroid_colors = kmeans.cluster_centers_

      # Creating child meshes for each cluster
      for cluster_id in range(num_clusters):
          mask = clusters == cluster_id
          centroid_color = centroid_colors[cluster_id]
          logic.create_child_model(modelNode, mask, cluster_id, centroid_color)
      qt.QApplication.restoreOverrideCursor()

    except Exception as e:
      qt.QApplication.restoreOverrideCursor()
      slicer.util.errorDisplay("Failed to compute results: "+str(e))
      import traceback
      traceback.print_exc() 
    
      
  def onApplyScaleButton(self):
      scaleFactor = self.scaleFactorSpinBox.value
      modelNode = self.inputModelSelector.currentNode()
      if modelNode:
          try:
              qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)
              logic = ModelColorsLogic()
              logic.applyScaling(modelNode, scaleFactor)
              qt.QApplication.restoreOverrideCursor()
          except Exception as e:
              qt.QApplication.restoreOverrideCursor()
              slicer.util.errorDisplay("Failed to apply scaling: "+str(e))
              import traceback
              traceback.print_exc()

#
# ModelColorsLogic
#
class ModelColorsLogic(ScriptedLoadableModuleLogic):
  """This class implements all the actual computations.
  Uses ScriptedLoadableModuleLogic base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def applyTexture(self, modelNode, textureImageNode, saveAsPointData=None):
    """
    Apply texture to model node
    :param saveAsPointData: None (not saved), `vector`, `float-vector`, `float-components`
    """
    self.showTextureOnModel(modelNode, textureImageNode)
    if saveAsPointData:
      print(f"saveAsPointData: {saveAsPointData}")
      self.convertTextureToPointAttribute(modelNode, textureImageNode, saveAsPointData)

  # Show texture
  def showTextureOnModel(self, modelNode, textureImageNode):
    modelDisplayNode = modelNode.GetDisplayNode()
    modelDisplayNode.SetBackfaceCulling(0)
    textureImageFlipVert = vtk.vtkImageFlip()
    textureImageFlipVert.SetFilteredAxis(1)
    textureImageFlipVert.SetInputConnection(textureImageNode.GetImageDataConnection())
    modelDisplayNode.SetTextureImageDataConnection(textureImageFlipVert.GetOutputPort())

  # Add texture data to scalars
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

  # Function to create a child model from a mask
  def create_child_model(self, modelNode, mask, cluster_id, centroid_color):

      polyData = modelNode.GetPolyData()
      points = polyData.GetPoints()
      polys = polyData.GetPolys()

      # Convert VTK Points to NumPy array
      np_points = nps.vtk_to_numpy(points.GetData())

      # Convert VTK Polys to NumPy array
      np_polys = nps.vtk_to_numpy(polys.GetData())

      # Reshape and filter polygons based on the mask
      tri_polys = np_polys.reshape(-1, 4)[:, 1:]  # Assuming 3 points per polygon + size info
      is_in_cluster = np.all(mask[tri_polys], axis=1)
      cluster_polys = tri_polys[is_in_cluster]

      # Create new points and polys for the cluster
      unique_points, new_poly_indices = np.unique(cluster_polys, return_inverse=True)
      new_poly_indices = new_poly_indices.reshape(-1, 3)  # Reshape back to polygon structure

      # Convert back to VTK data structures
      new_vtk_points = vtk.vtkPoints()
      new_vtk_points.SetData(nps.numpy_to_vtk(np_points[unique_points]))
      new_vtk_polys = vtk.vtkCellArray()
      for poly in new_poly_indices:
          new_vtk_polys.InsertNextCell(len(poly), poly)

      # Create new vtkPolyData for child model
      newPolyData = vtk.vtkPolyData()
      newPolyData.SetPoints(new_vtk_points)
      newPolyData.SetPolys(new_vtk_polys)

      # Add new model node to the scene
      newModelNode = slicer.vtkMRMLModelNode()
      newModelNode.SetName(f'Cluster_{cluster_id}')
      newModelNode.SetAndObservePolyData(newPolyData)
      slicer.mrmlScene.AddNode(newModelNode)

      # Set display properties
      display_color = (centroid_color).astype(int).tolist()
      displayNode = slicer.vtkMRMLModelDisplayNode()
      displayNode.SetColor(display_color[0] / 255.0, display_color[1] / 255.0, display_color[2] / 255.0)
      slicer.mrmlScene.AddNode(displayNode)
      newModelNode.SetAndObserveDisplayNodeID(displayNode.GetID())
    
  def applyScaling(self, modelNode, scaleFactor):
      """
      Apply scaling to model node.
      :param modelNode: The model to scale.
      :param scaleFactor: The scaling factor (float).
      """
      # Retrieve polydata from the model node
      polyData = modelNode.GetPolyData()

      # Create a transformation matrix for scaling
      transformMatrix = vtk.vtkMatrix4x4()
      transformMatrix.Identity()
      for i in range(3):  # Scale x, y, and z coordinates
          transformMatrix.SetElement(i, i, scaleFactor)

      # Apply the transformation
      transform = vtk.vtkTransform()
      transform.SetMatrix(transformMatrix)
      transformFilter = vtk.vtkTransformPolyDataFilter()
      transformFilter.SetTransform(transform)
      transformFilter.SetInputData(polyData)
      transformFilter.Update()

      # Update the model with the transformed data
      modelNode.SetAndObservePolyData(transformFilter.GetOutput())

class ModelColorsTest(ScriptedLoadableModuleTest):
  """
  This is the test case for your scripted module.
  Uses ScriptedLoadableModuleTest base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def setUp(self):
    """ Do whatever is needed to reset the state - typically a scene clear will be enough.
    """
    slicer.mrmlScene.Clear(0)

  def runTest(self):
    """Run as few or as many tests as needed here.
    """
    self.setUp()
    self.test_ModelColors1()

  def test_ModelColors1(self):
    """ Ideally you should have several levels of tests.  At the lowest level
    tests should exercise the functionality of the logic with different inputs
    (both valid and invalid).  At higher levels your tests should emulate the
    way the user would interact with your code and confirm that it still works
    the way you intended.
    One of the most important features of the tests is that it should alert other
    developers when their changes will have an impact on the behavior of your
    module.  For example, if a developer removes a feature that you depend on,
    your test should break so they know that the feature is needed.
    """

    slicer.util.delayDisplay("Starting the test")

    # Download
    import urllib
    url = 'https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/752ce9afe8b708fcd4f8448612170f8e730670d845f65177860edc0e08004ecf'
    zipFilePath = slicer.app.temporaryPath + '/' + 'FemurHeadSurfaceScan.zip'
    extractPath = slicer.app.temporaryPath + '/' + 'FemurHeadSurfaceScan'
    if not os.path.exists(zipFilePath) or os.stat(zipFilePath).st_size == 0:
      logging.info('Requesting download from %s...\n' % url)
      urllib.request.urlretrieve(url, zipFilePath)
      slicer.util.delayDisplay('Finished with download\n')

    # Unzip
    slicer.util.delayDisplay("Unzipping to %s" % (extractPath))
    qt.QDir().mkpath(extractPath)
    applicationLogic = slicer.app.applicationLogic()
    applicationLogic.Unzip(zipFilePath, extractPath)

    # Load
    slicer.util.loadModel(extractPath+"/head_obj.obj")
    slicer.util.loadVolume(extractPath+"/head_obj_0.png")

    slicer.util.delayDisplay('Finished with download and loading')

    # Test
    modelNode = slicer.util.getNode("head_obj")
    textureNode = slicer.util.getNode("head_obj_0")
    logic = ModelColorsLogic()
    logic.applyTexture(modelNode, textureNode)
    slicer.util.delayDisplay('Test passed!')
