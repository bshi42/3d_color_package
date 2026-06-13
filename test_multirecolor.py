#!/usr/bin/env python3
"""
Test script for MultiRecolor functionality in InterDeCA module
Run this in 3D Slicer's Python console
"""

import numpy as np
import os

def test_multirecolor_ui():
    """Test the MultiRecolor UI elements"""
    print("\n=== Testing MultiRecolor UI Elements ===")
    
    try:
        import slicer
        
        # Get the InterDeCA module widget
        moduleWidget = slicer.modules.interdeca.widgetRepresentation().self()
        
        # Check if the new MultiRecolor UI elements exist
        required_elements = [
            'multiRecolorAtlasModelSelect',
            'multiRecolorTextureDirectorySelector',
            'multiRecolorNumClustersSpin',
            'clusterButton',
            'individualTextureSelector',
            'individualHighContrastCheckbox',
            'applyIndividualTextureButton',
            'compareTexturesButton',
            'pcaRadioButton',
            'umapRadioButton',
            'faceAreasCache'
        ]
        
        for element in required_elements:
            if not hasattr(moduleWidget, element):
                print(f"ERROR: {element} not found")
                return False
            else:
                print(f"✓ {element} found")
        
        # Test initial states
        print("\nTesting initial UI states:")
        print(f"  Cluster button enabled: {moduleWidget.clusterButton.enabled}")
        print(f"  Individual texture selector enabled: {moduleWidget.individualTextureSelector.enabled}")
        print(f"  Apply individual texture button enabled: {moduleWidget.applyIndividualTextureButton.enabled}")
        print(f"  Compare textures button enabled: {moduleWidget.compareTexturesButton.enabled}")
        print(f"  PCA radio button checked: {moduleWidget.pcaRadioButton.isChecked()}")
        print(f"  UMAP radio button checked: {moduleWidget.umapRadioButton.isChecked()}")

        # Test directory selector type (should match Recolor tab)
        print(f"  Directory selector type: {type(moduleWidget.multiRecolorTextureDirectorySelector).__name__}")
        print(f"  Directory selector has currentPath: {hasattr(moduleWidget.multiRecolorTextureDirectorySelector, 'currentPath')}")
        print(f"  Directory selector has filters: {hasattr(moduleWidget.multiRecolorTextureDirectorySelector, 'filters')}")

        # Test log widget types (should be QTextEdit for MultiRecolor)
        print(f"  Clustering log type: {type(moduleWidget.clusteringLogInfo).__name__}")
        print(f"  Individual log type: {type(moduleWidget.individualLogInfo).__name__}")
        print(f"  Population log type: {type(moduleWidget.populationLogInfo).__name__}")
        print(f"  Clustering log has append method: {hasattr(moduleWidget.clusteringLogInfo, 'append')}")
        print(f"  Individual log has append method: {hasattr(moduleWidget.individualLogInfo, 'append')}")
        print(f"  Population log has append method: {hasattr(moduleWidget.populationLogInfo, 'append')}")
        
        # Test that buttons are initially disabled (as expected)
        if (not moduleWidget.clusterButton.enabled and 
            not moduleWidget.individualTextureSelector.enabled and
            not moduleWidget.applyIndividualTextureButton.enabled and
            not moduleWidget.compareTexturesButton.enabled):
            print("✓ Initial button states are correct (disabled)")
        else:
            print("WARNING: Some buttons should be disabled initially")
        
        print("✓ MultiRecolor UI tests completed")
        return True
        
    except Exception as e:
        print(f"ERROR in MultiRecolor UI test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_face_area_caching():
    """Test the face area caching system"""
    print("\n=== Testing Face Area Caching ===")
    
    try:
        import slicer
        from color_deca.deca3.InterDeCA import InterDeCALogic
        
        # Get the InterDeCA module widget
        moduleWidget = slicer.modules.interdeca.widgetRepresentation().self()
        
        # Test that cache is initialized
        if hasattr(moduleWidget, 'faceAreasCache'):
            print(f"✓ Face areas cache initialized: {type(moduleWidget.faceAreasCache)}")
            print(f"  Cache size: {len(moduleWidget.faceAreasCache)}")
        else:
            print("ERROR: Face areas cache not found")
            return False
        
        # Test face area calculation method
        logic = InterDeCALogic()
        
        # Create a simple test mesh (triangle)
        points = vtk.vtkPoints()
        points.InsertNextPoint(0, 0, 0)
        points.InsertNextPoint(1, 0, 0)
        points.InsertNextPoint(0, 1, 0)
        
        triangle = vtk.vtkTriangle()
        triangle.GetPointIds().SetId(0, 0)
        triangle.GetPointIds().SetId(1, 1)
        triangle.GetPointIds().SetId(2, 2)
        
        cells = vtk.vtkCellArray()
        cells.InsertNextCell(triangle)
        
        polyData = vtk.vtkPolyData()
        polyData.SetPoints(points)
        polyData.SetPolys(cells)
        
        # Test face area calculation
        faceAreas = logic._calculateFaceAreas(polyData)
        
        if faceAreas is not None:
            print(f"✓ Face area calculation works: {len(faceAreas)} faces")
            print(f"  Triangle area: {faceAreas[0]:.3f} (expected: 0.5)")
            
            # Check if area is approximately correct for a right triangle with legs 1,1
            expected_area = 0.5
            if abs(faceAreas[0] - expected_area) < 0.01:
                print("✓ Face area calculation is accurate")
            else:
                print(f"WARNING: Face area calculation may be inaccurate")
        else:
            print("ERROR: Face area calculation failed")
            return False
        
        print("✓ Face area caching tests completed")
        return True
        
    except Exception as e:
        print(f"ERROR in face area caching test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_multirecolor_methods():
    """Test MultiRecolor logic methods"""
    print("\n=== Testing MultiRecolor Logic Methods ===")
    
    try:
        import slicer
        from color_deca.deca3.InterDeCA import InterDeCALogic
        
        logic = InterDeCALogic()
        
        # Test that required methods exist
        required_methods = [
            'performMultiTextureClustering',
            'applyIndividualTextureWithClusteredPalette',
            'performPopulationAnalysis',
            '_calculateFaceAreas',
            '_createPopulationPlot'
        ]
        
        for method in required_methods:
            if not hasattr(logic, method):
                print(f"ERROR: {method} not found")
                return False
            else:
                print(f"✓ {method} found")
        
        # Test method signatures (basic check)
        import inspect
        
        # Check performMultiTextureClustering signature
        sig = inspect.signature(logic.performMultiTextureClustering)
        expected_params = ['modelNode', 'textureDir', 'textureFiles', 'numClusters', 'faceAreas']
        for param in expected_params:
            if param not in sig.parameters:
                print(f"ERROR: {param} parameter missing from performMultiTextureClustering")
                return False
        print("✓ performMultiTextureClustering signature correct")
        
        # Check applyIndividualTextureWithClusteredPalette signature
        sig = inspect.signature(logic.applyIndividualTextureWithClusteredPalette)
        expected_params = ['modelNode', 'texturePath', 'clusterCenters', 'useHighContrast', 'faceAreas']
        for param in expected_params:
            if param not in sig.parameters:
                print(f"ERROR: {param} parameter missing from applyIndividualTextureWithClusteredPalette")
                return False
        print("✓ applyIndividualTextureWithClusteredPalette signature correct")
        
        # Check performPopulationAnalysis signature
        sig = inspect.signature(logic.performPopulationAnalysis)
        expected_params = ['modelNode', 'textureDir', 'textureFiles', 'clusterCenters', 'faceAreas', 'dimReductionMethod']
        for param in expected_params:
            if param not in sig.parameters:
                print(f"ERROR: {param} parameter missing from performPopulationAnalysis")
                return False
        print("✓ performPopulationAnalysis signature correct")
        
        print("✓ MultiRecolor logic method tests completed")
        return True
        
    except Exception as e:
        print(f"ERROR in MultiRecolor logic methods test: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_dependencies():
    """Check if required dependencies are available for MultiRecolor"""
    print("=== MultiRecolor Dependency Check ===")
    
    dependencies = {
        'numpy': True,  # Always available in Slicer
        'sklearn': False,
        'scikit-image': False,
        'umap': False,
        'slicer': False,
        'vtk': False,
        'imageio': False
    }
    
    try:
        import sklearn
        dependencies['sklearn'] = True
    except ImportError:
        pass
    
    try:
        import skimage
        dependencies['scikit-image'] = True
    except ImportError:
        pass
    
    try:
        import umap
        dependencies['umap'] = True
    except ImportError:
        pass
    
    try:
        import slicer
        dependencies['slicer'] = True
    except ImportError:
        pass
    
    try:
        import vtk
        dependencies['vtk'] = True
    except ImportError:
        pass
    
    try:
        import imageio
        dependencies['imageio'] = True
    except ImportError:
        pass
    
    for dep, available in dependencies.items():
        status = "✓" if available else "✗"
        print(f"  {dep}: {status}")
    
    missing = [dep for dep, available in dependencies.items() if not available]
    if missing:
        print(f"\nMissing dependencies: {', '.join(missing)}")
        print("MultiRecolor requires sklearn, scikit-image, and imageio.")
        print("UMAP is optional but recommended for population analysis.")
        return False
    else:
        print("\n✓ All dependencies available")
        return True

def run_multirecolor_tests():
    """Run all MultiRecolor tests"""
    print("MultiRecolor Test Suite")
    print("=" * 50)
    
    # Check dependencies first
    if not check_dependencies():
        print("\nSkipping tests due to missing dependencies.")
        return False
    
    tests = [
        ("MultiRecolor UI Elements", test_multirecolor_ui),
        ("Face Area Caching", test_face_area_caching),
        ("MultiRecolor Logic Methods", test_multirecolor_methods),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\nRunning {test_name} test...")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"FAILED: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 50)
    print("MULTIRECOLOR TEST RESULTS:")
    all_passed = True
    for test_name, success in results:
        status = "PASS" if success else "FAIL"
        print(f"  {test_name}: {status}")
        if not success:
            all_passed = False
    
    print(f"\nOverall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    return all_passed

# Run tests if executed directly
if __name__ == "__main__":
    run_multirecolor_tests()
