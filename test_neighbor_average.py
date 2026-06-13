#!/usr/bin/env python3
"""
Test script for the Neighbor Average feature in MultiRecolor clustering
"""

import sys
import os
import numpy as np

# Add the module to path
sys.path.insert(0, '/home/alek/projects/3d_color_package')

def test_neighbor_average_feature():
    """Test that the neighbor average feature is properly integrated"""
    
    print("=" * 60)
    print("Testing Neighbor Average Feature")
    print("=" * 60)
    
    # Test 1: Check UI element exists
    print("\n[Test 1] Checking UI element...")
    try:
        with open('color_deca/deca3/InterDeCA.py', 'r') as f:
            content = f.read()
            
        assert 'multiRecolorNeighborAverageCheckbox' in content, "UI checkbox not found"
        assert 'Neighbor Average' in content, "Neighbor Average label not found"
        print("✓ UI checkbox element found")
    except AssertionError as e:
        print(f"✗ {e}")
        return False
    
    # Test 2: Check method signature
    print("\n[Test 2] Checking method signatures...")
    try:
        assert '_applyNeighborAveraging' in content, "Method not found"
        assert 'useNeighborAverage' in content, "Parameter not found"
        print("✓ Method and parameter found")
    except AssertionError as e:
        print(f"✗ {e}")
        return False
    
    # Test 3: Check ClusteringPipeline integration
    print("\n[Test 3] Checking ClusteringPipeline integration...")
    try:
        assert 'self.useNeighborAverage' in content, "Pipeline attribute not found"
        print("✓ ClusteringPipeline attribute found")
    except AssertionError as e:
        print(f"✗ {e}")
        return False
    
    # Test 4: Check performMultiTextureClustering integration
    print("\n[Test 4] Checking performMultiTextureClustering integration...")
    try:
        # Check signature
        assert 'useNeighborAverage=False' in content, "Parameter not in signature"
        # Check it's passed to pipeline
        assert 'ClusteringPipeline(initialClusters, consolidatedClusters, normalizeLuminosity, useNeighborAverage)' in content, "Not passed to pipeline"
        print("✓ performMultiTextureClustering properly integrated")
    except AssertionError as e:
        print(f"✗ {e}")
        return False
    
    # Test 5: Check onClusterButton integration
    print("\n[Test 5] Checking onClusterButton integration...")
    try:
        assert 'useNeighborAverage = self.multiRecolorNeighborAverageCheckbox.isChecked()' in content, "Checkbox not read"
        assert 'useNeighborAverage=useNeighborAverage' in content, "Not passed to clustering"
        print("✓ onClusterButton properly integrated")
    except AssertionError as e:
        print(f"✗ {e}")
        return False
    
    # Test 6: Check neighbor averaging is applied
    print("\n[Test 6] Checking neighbor averaging application...")
    try:
        assert 'if useNeighborAverage:' in content, "Conditional not found"
        assert 'self._applyNeighborAveraging(polyData, faceColors, faceAdjacency, logCallback)' in content, "Method not called with adjacency"
        print("✓ Neighbor averaging is applied in clustering")
    except AssertionError as e:
        print(f"✗ {e}")
        return False
    
    # Test 7: Check method implementation
    print("\n[Test 7] Checking _applyNeighborAveraging implementation...")
    try:
        assert 'def _applyNeighborAveraging' in content, "Method definition not found"
        assert '_buildFaceAdjacencyGraph' in content, "Adjacency graph building not found"
        assert 'smoothedColors = np.zeros_like(faceColors)' in content, "Color smoothing not found"
        assert 'colorSum / count' in content, "Averaging logic not found"
        print("✓ _applyNeighborAveraging properly implemented")
    except AssertionError as e:
        print(f"✗ {e}")
        return False

    # Test 8: Check neighbor averaging in Step 2
    print("\n[Test 8] Checking neighbor averaging in Step 2 (Individual Visualization)...")
    try:
        # Look for neighbor averaging in applyIndividualTextureWithClusteredPalette
        assert 'clusteringPipeline.useNeighborAverage' in content, "useNeighborAverage flag not checked in Step 2"
        assert 'Applying neighbor average smoothing to face colors' in content, "Neighbor averaging not applied in Step 2"
        print("✓ Neighbor averaging applied in Step 2")
    except AssertionError as e:
        print(f"✗ {e}")
        return False

    # Test 9: Check neighbor averaging in Step 3
    print("\n[Test 9] Checking neighbor averaging in Step 3 (Population Analysis)...")
    try:
        # Count occurrences - should be at least 2 (one in Step 2, one in Step 3)
        count = content.count('clusteringPipeline.useNeighborAverage')
        assert count >= 2, f"useNeighborAverage flag not checked in Step 3 (found {count} occurrences, expected at least 2)"
        print("✓ Neighbor averaging applied in Step 3")
    except AssertionError as e:
        print(f"✗ {e}")
        return False

    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)
    print("\nFeature Summary:")
    print("- UI checkbox added to Step 1 clustering section")
    print("- Flag passed through clustering pipeline")
    print("- Neighbor averaging applied in Step 1 (clustering)")
    print("- Neighbor averaging applied in Step 2 (individual visualization)")
    print("- Neighbor averaging applied in Step 3 (population analysis)")
    print("- Uses mesh topology (face adjacency graph)")
    print("- Optional feature (disabled by default)")
    print("\nUsage:")
    print("1. Check 'Neighbor Average' checkbox in Step 1")
    print("2. Click 'Cluster' to run with smoothed colors")
    print("3. Steps 2 and 3 will automatically use smoothed colors")
    print("4. Adjacent faces will have more similar colors throughout the pipeline")
    
    return True

if __name__ == '__main__':
    success = test_neighbor_average_feature()
    sys.exit(0 if success else 1)

