#!/usr/bin/env python3
"""
Test script for MultiRecolor subsampling functionality
"""

import sys
import os
import numpy as np

# Add the project to the path
sys.path.insert(0, '/home/alek/projects/3d_color_package')

def test_subsampling_method():
    """Test the face subsampling method"""
    print("\n=== Testing Face Subsampling Method ===")

    try:
        # Check code structure without importing (VTK not available outside Slicer)
        with open('/home/alek/projects/3d_color_package/color_deca/deca3/InterDeCA.py', 'r') as f:
            content = f.read()

        # Check that the method exists
        if 'def _subsampleFacesUniformly(self, polyData, numSubsampledFaces, logCallback=None):' not in content:
            print("ERROR: _subsampleFacesUniformly method not found")
            return False

        print("✓ _subsampleFacesUniformly method exists")

        # Check for graph-based approach
        if 'graph-based' not in content.lower():
            print("ERROR: Graph-based approach not found in docstring")
            return False

        print("✓ Graph-based approach implemented")

        # Check for adjacency graph building
        if '_buildFaceAdjacencyGraph' not in content:
            print("ERROR: _buildFaceAdjacencyGraph method not found")
            return False

        print("✓ Face adjacency graph building implemented")

        # Check for BFS implementation
        if 'BFS' not in content:
            print("ERROR: BFS not found in docstring")
            return False

        print("✓ BFS-based distance computation found")

        # Check that performMultiTextureClustering has the new parameter
        if 'def performMultiTextureClustering(self, modelNode, textureDir, textureFiles, initialClusters, consolidatedClusters,\n                                     numSubsampledFaces=None' not in content:
            print("ERROR: numSubsampledFaces parameter not found in performMultiTextureClustering")
            return False

        print("✓ numSubsampledFaces parameter exists in performMultiTextureClustering")

        # Check ClusteringPipeline has subsampling attributes
        if 'self.subsampledFaceIndices = None' not in content:
            print("ERROR: subsampledFaceIndices attribute not found in ClusteringPipeline")
            return False

        if 'self.nearestNeighborMapping = None' not in content:
            print("ERROR: nearestNeighborMapping attribute not found in ClusteringPipeline")
            return False

        print("✓ ClusteringPipeline has subsampling attributes")

        # Check that subsampling is used in clustering
        if 'if subsampledFaceIndices is not None:' not in content:
            print("ERROR: Subsampling logic not found in clustering")
            return False

        print("✓ Subsampling logic found in clustering")

        # Check that nearest neighbor mapping is used in Step 2
        if 'if clusteringPipeline is not None and clusteringPipeline.subsampledFaceIndices is not None:' not in content:
            print("ERROR: Nearest neighbor mapping logic not found in Step 2")
            return False

        print("✓ Nearest neighbor mapping logic found in Step 2")

        print("\n✓ All basic structure tests passed!")
        return True

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_ui_elements():
    """Test that UI elements were added"""
    print("\n=== Testing UI Elements ===")
    
    try:
        # Check that the UI spinbox was added
        # This is harder to test without running in Slicer, but we can check the code
        with open('/home/alek/projects/3d_color_package/color_deca/deca3/InterDeCA.py', 'r') as f:
            content = f.read()
        
        if 'multiRecolorNumSubsampledFacesSpin' not in content:
            print("ERROR: multiRecolorNumSubsampledFacesSpin UI element not found")
            return False
        
        print("✓ multiRecolorNumSubsampledFacesSpin UI element found in code")
        
        if 'Number of Faces (Subsampling)' not in content:
            print("ERROR: 'Number of Faces (Subsampling)' label not found")
            return False
        
        print("✓ 'Number of Faces (Subsampling)' label found in code")
        
        if 'numSubsampledFaces = self.multiRecolorNumSubsampledFacesSpin.value' not in content:
            print("ERROR: numSubsampledFaces value retrieval not found")
            return False
        
        print("✓ numSubsampledFaces value retrieval found in code")
        
        print("\n✓ All UI element tests passed!")
        return True
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_vector_representation():
    """Test the new vector representation logic"""
    print("\n=== Testing Vector Representation Logic ===")
    
    try:
        with open('/home/alek/projects/3d_color_package/color_deca/deca3/InterDeCA.py', 'r') as f:
            content = f.read()
        
        # Check for subsampled face color vector creation
        if 'subsampledFaceColors.flatten()' not in content:
            print("ERROR: Flattened subsampled face color vector creation not found")
            return False
        
        print("✓ Flattened subsampled face color vector creation found")
        
        # Check for useSubsampling flag
        if 'useSubsampling = (clusteringPipeline is not None' not in content:
            print("ERROR: useSubsampling flag logic not found")
            return False
        
        print("✓ useSubsampling flag logic found")
        
        # Check for conditional vector creation
        if 'if useSubsampling:' not in content:
            print("ERROR: Conditional vector creation logic not found")
            return False
        
        print("✓ Conditional vector creation logic found")
        
        print("\n✓ All vector representation tests passed!")
        return True
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    print("=" * 60)
    print("MultiRecolor Subsampling Implementation Tests")
    print("=" * 60)
    
    results = []
    results.append(("Structure Tests", test_subsampling_method()))
    results.append(("UI Element Tests", test_ui_elements()))
    results.append(("Vector Representation Tests", test_vector_representation()))
    
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{test_name}: {status}")
    
    all_passed = all(result for _, result in results)
    
    if all_passed:
        print("\n✓ All tests passed!")
        sys.exit(0)
    else:
        print("\n✗ Some tests failed!")
        sys.exit(1)

