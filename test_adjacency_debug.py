#!/usr/bin/env python
"""Debug script to test face adjacency graph building"""

import sys
import os
sys.path.insert(0, '/home/alek/projects/3d_color_package')

# Create a simple test mesh with VTK
import vtk
import numpy as np

def create_simple_mesh():
    """Create a simple 2-triangle mesh for testing"""
    points = vtk.vtkPoints()
    points.InsertNextPoint(0, 0, 0)  # 0
    points.InsertNextPoint(1, 0, 0)  # 1
    points.InsertNextPoint(0, 1, 0)  # 2
    points.InsertNextPoint(1, 1, 0)  # 3
    
    cells = vtk.vtkCellArray()
    
    # Triangle 1: (0, 1, 2)
    triangle1 = vtk.vtkTriangle()
    triangle1.GetPointIds().SetId(0, 0)
    triangle1.GetPointIds().SetId(1, 1)
    triangle1.GetPointIds().SetId(2, 2)
    cells.InsertNextCell(triangle1)
    
    # Triangle 2: (1, 3, 2) - shares edge (1, 2) with triangle 1
    triangle2 = vtk.vtkTriangle()
    triangle2.GetPointIds().SetId(0, 1)
    triangle2.GetPointIds().SetId(1, 3)
    triangle2.GetPointIds().SetId(2, 2)
    cells.InsertNextCell(triangle2)
    
    polyData = vtk.vtkPolyData()
    polyData.SetPoints(points)
    polyData.SetPolys(cells)
    
    return polyData

def test_adjacency():
    """Test the adjacency graph building"""
    from color_deca.deca3.InterDeCA import InterDeCALogic
    
    # Create test mesh
    polyData = create_simple_mesh()
    print(f"Created test mesh with {polyData.GetNumberOfCells()} faces")
    
    # Create logic instance
    logic = InterDeCALogic()
    
    # Test adjacency graph building
    def log_callback(msg):
        print(f"  {msg}")
    
    adjacency = logic._buildFaceAdjacencyGraph(polyData, log_callback)
    
    print(f"\nAdjacency graph:")
    for faceId, neighbors in adjacency.items():
        print(f"  Face {faceId}: neighbors = {neighbors}")
    
    # Check if faces are connected
    if 0 in adjacency[1] and 1 in adjacency[0]:
        print("\n✓ Faces 0 and 1 are correctly connected!")
    else:
        print("\n✗ ERROR: Faces 0 and 1 should be connected!")

if __name__ == '__main__':
    test_adjacency()

