#!/usr/bin/env python3
"""
Debug script for face color issues in the Recolor tab.
Run this in 3D Slicer's Python console to diagnose face coloring problems.
"""

import vtk
import numpy as np

def debug_model_face_colors(modelNode):
    """Debug face color application on a model"""
    print("=== Debugging Face Colors ===")
    
    if not modelNode:
        print("ERROR: No model node provided")
        return
    
    polyData = modelNode.GetPolyData()
    if not polyData:
        print("ERROR: No polydata in model")
        return
    
    print(f"Model: {modelNode.GetName()}")
    print(f"Number of points: {polyData.GetNumberOfPoints()}")
    print(f"Number of cells: {polyData.GetNumberOfCells()}")
    
    # Check texture coordinates
    tcoords = polyData.GetPointData().GetTCoords()
    if tcoords:
        print(f"Texture coordinates: {tcoords.GetNumberOfTuples()} tuples")
    else:
        print("WARNING: No texture coordinates found")
    
    # Check cell data
    cellData = polyData.GetCellData()
    print(f"Cell data arrays: {cellData.GetNumberOfArrays()}")
    
    for i in range(cellData.GetNumberOfArrays()):
        array = cellData.GetArray(i)
        print(f"  Array {i}: {array.GetName()}, {array.GetNumberOfTuples()} tuples, {array.GetNumberOfComponents()} components")
    
    # Check display node
    displayNode = modelNode.GetDisplayNode()
    if displayNode:
        print(f"Display node exists: {displayNode.GetClassName()}")
        print(f"Scalar visibility: {displayNode.GetScalarVisibility()}")
        print(f"Active scalar name: {displayNode.GetActiveScalarName()}")
        print(f"Active attribute location: {displayNode.GetActiveAttributeLocation()}")
        
        colorNode = displayNode.GetColorNode()
        if colorNode:
            print(f"Color node: {colorNode.GetName()}")
        else:
            print("Color node: None (using direct RGB)")
    else:
        print("ERROR: No display node found")

def create_test_face_colors(modelNode, logCallback=None):
    """Create simple test face colors to verify the coloring system works"""
    print("=== Creating Test Face Colors ===")
    
    if not modelNode:
        print("ERROR: No model node provided")
        return False
    
    polyData = modelNode.GetPolyData()
    if not polyData:
        print("ERROR: No polydata in model")
        return False
    
    nFaces = polyData.GetNumberOfCells()
    print(f"Creating test colors for {nFaces} faces")
    
    # Create simple test colors (red, green, blue pattern)
    colorArray = vtk.vtkUnsignedCharArray()
    colorArray.SetNumberOfComponents(3)
    colorArray.SetName("TestFaceColors")
    colorArray.SetNumberOfTuples(nFaces)
    
    for i in range(nFaces):
        if i % 3 == 0:
            colorArray.SetTuple3(i, 255, 0, 0)  # Red
        elif i % 3 == 1:
            colorArray.SetTuple3(i, 0, 255, 0)  # Green
        else:
            colorArray.SetTuple3(i, 0, 0, 255)  # Blue
    
    # Add color array to cell data
    polyData.GetCellData().SetScalars(colorArray)
    polyData.Modified()
    
    # Update display
    displayNode = modelNode.GetDisplayNode()
    if displayNode:
        # Turn off texture
        displayNode.SetTextureImageDataConnection(None)
        
        # Enable scalar visibility
        displayNode.SetScalarVisibility(True)
        displayNode.SetActiveScalarName("TestFaceColors")
        
        # Set to use cell data
        displayNode.SetActiveAttributeLocation(vtk.vtkDataObject.CELL)
        
        # Use direct RGB mapping
        displayNode.SetScalarRangeFlag(slicer.vtkMRMLDisplayNode.UseDirectMapping)
        displayNode.SetAndObserveColorNodeID(None)
        
        displayNode.Modified()
        
        print("Test face colors applied successfully")
        return True
    else:
        print("ERROR: No display node found")
        return False

def run_face_color_debug():
    """Main debug function to run in Slicer"""
    print("Face Color Debug Tool")
    print("=" * 50)
    
    # Get the currently selected model
    try:
        import slicer
        
        # Try to get a model from the scene
        modelNodes = slicer.util.getNodesByClass('vtkMRMLModelNode')
        
        if not modelNodes:
            print("No model nodes found in scene. Please load a model first.")
            return
        
        # Use the first model node
        modelNode = modelNodes[0]
        print(f"Using model: {modelNode.GetName()}")
        
        # Debug the model
        debug_model_face_colors(modelNode)
        
        # Ask user if they want to apply test colors
        print("\nApplying test face colors...")
        success = create_test_face_colors(modelNode)
        
        if success:
            print("✓ Test colors applied. Check the 3D viewer.")
            print("If you see red/green/blue faces, the face coloring system works.")
            print("If not, there may be an issue with the display configuration.")
        else:
            print("✗ Failed to apply test colors.")
            
    except Exception as e:
        print(f"Error during debug: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # This would be run in 3D Slicer's Python console
    run_face_color_debug()
