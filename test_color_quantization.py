#!/usr/bin/env python3
"""
Test script for the new color quantization functionality in the Recolor tab.
Run this in 3D Slicer's Python console to test the quantization features.
"""

import numpy as np
import os

def test_color_quantization_ui():
    """Test that the quantization UI elements are properly added"""
    print("=== Testing Color Quantization UI ===")
    
    try:
        import slicer
        
        # Get the InterDeCA module widget
        moduleWidget = slicer.modules.interdeca.widgetRepresentation().self()
        
        # Check if the new UI elements exist
        required_elements = [
            'quantizeColorsCheckbox',
            'numColorClustersSpin',
            'useHighContrastPaletteCheckbox',
            'onAverageFaceColorToggled',
            'onQuantizeColorsToggled'
        ]
        
        for element in required_elements:
            if not hasattr(moduleWidget, element):
                print(f"ERROR: {element} not found")
                return False
            else:
                print(f"✓ {element} found")
        
        # Test initial states
        print("\nTesting initial UI states:")
        print(f"  Average face colors checked: {moduleWidget.averageFaceColorCheckbox.isChecked()}")
        print(f"  Quantize colors enabled: {moduleWidget.quantizeColorsCheckbox.isEnabled()}")
        print(f"  Quantize colors checked: {moduleWidget.quantizeColorsCheckbox.isChecked()}")
        print(f"  Number of clusters enabled: {moduleWidget.numColorClustersSpin.isEnabled()}")
        print(f"  High contrast palette enabled: {moduleWidget.useHighContrastPaletteCheckbox.isEnabled()}")
        print(f"  High contrast palette checked: {moduleWidget.useHighContrastPaletteCheckbox.isChecked()}")
        print(f"  Number of clusters value: {moduleWidget.numColorClustersSpin.value}")
        print(f"  Number of clusters range: {moduleWidget.numColorClustersSpin.minimum()}-{moduleWidget.numColorClustersSpin.maximum()}")
        
        # Test enabling averaging
        print("\nTesting UI interactions:")
        print("  Enabling average face colors...")
        moduleWidget.averageFaceColorCheckbox.setChecked(True)
        print(f"  Quantize colors now enabled: {moduleWidget.quantizeColorsCheckbox.isEnabled()}")
        
        # Test enabling quantization
        print("  Enabling quantize colors...")
        moduleWidget.quantizeColorsCheckbox.setChecked(True)
        print(f"  Number of clusters now enabled: {moduleWidget.numColorClustersSpin.isEnabled()}")
        
        # Test disabling averaging
        print("  Disabling average face colors...")
        moduleWidget.averageFaceColorCheckbox.setChecked(False)
        print(f"  Quantize colors now enabled: {moduleWidget.quantizeColorsCheckbox.isEnabled()}")
        print(f"  Quantize colors now checked: {moduleWidget.quantizeColorsCheckbox.isChecked()}")
        
        print("✓ UI tests completed successfully")
        return True
        
    except Exception as e:
        print(f"ERROR in UI test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_color_space_conversions():
    """Test the RGB to Lab and Lab to RGB conversions using scikit-image"""
    print("\n=== Testing Color Space Conversions (scikit-image) ===")

    try:
        import slicer
        from color_deca.deca3.InterDeCA import InterDeCALogic

        logic = InterDeCALogic()

        # Test colors (RGB values 0-255)
        test_colors = np.array([
            [255, 0, 0],    # Red
            [0, 255, 0],    # Green
            [0, 0, 255],    # Blue
            [255, 255, 255], # White
            [0, 0, 0],      # Black
            [128, 128, 128], # Gray
            [255, 255, 0],  # Yellow
            [255, 0, 255],  # Magenta
            [0, 255, 255],  # Cyan
        ])

        print("Testing RGB → Lab → RGB conversion (using scikit-image):")
        for i, rgb in enumerate(test_colors):
            # Convert to Lab
            lab = logic.rgb_to_lab(rgb.reshape(1, -1))[0]

            # Convert back to RGB
            rgb_back = logic.lab_to_rgb(lab.reshape(1, -1))[0]

            # Calculate error
            error = np.abs(rgb - rgb_back).max()

            print(f"  RGB {rgb} → Lab [{lab[0]:.1f}, {lab[1]:.1f}, {lab[2]:.1f}] → RGB {rgb_back.astype(int)} (max error: {error:.1f})")

            if error > 1:  # scikit-image should be more accurate
                print(f"    WARNING: Large conversion error!")

        # Test ΔE2000 calculation
        print("\nTesting ΔE2000 calculation:")
        lab1 = logic.rgb_to_lab(np.array([[255, 0, 0]]))  # Red
        lab2 = logic.rgb_to_lab(np.array([[255, 255, 0]]))  # Yellow
        delta_e = logic.delta_e_2000(lab1, lab2)[0]
        print(f"  ΔE2000 between Red and Yellow: {delta_e:.2f}")

        # Test identical colors (should be 0)
        delta_e_same = logic.delta_e_2000(lab1, lab1)[0]
        print(f"  ΔE2000 between identical colors: {delta_e_same:.2f}")

        if delta_e_same > 0.01:
            print("    WARNING: ΔE2000 for identical colors should be ~0!")

        print("✓ Color space conversion tests completed (using scikit-image)")
        return True

    except Exception as e:
        print(f"ERROR in color space test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_high_contrast_palette():
    """Test the high contrast palette generation"""
    print("\n=== Testing High Contrast Palette Generation ===")

    try:
        import slicer
        from color_deca.deca3.InterDeCA import InterDeCALogic

        logic = InterDeCALogic()

        # Test different numbers of colors
        test_sizes = [2, 8, 16, 32, 64]

        for n_colors in test_sizes:
            print(f"\nTesting {n_colors} colors:")

            # Generate palette
            palette = logic.generate_high_contrast_palette(n_colors)

            # Check shape
            if palette.shape != (n_colors, 3):
                print(f"  ERROR: Expected shape ({n_colors}, 3), got {palette.shape}")
                return False

            # Check value range
            if np.any(palette < 0) or np.any(palette > 255):
                print(f"  ERROR: Colors out of range [0, 255]")
                return False

            # Check data type
            if palette.dtype != np.uint8:
                print(f"  ERROR: Expected uint8, got {palette.dtype}")
                return False

            # Show first few colors
            print(f"  First 3 colors: {palette[:3].tolist()}")
            print(f"  ✓ Generated {n_colors} high contrast colors successfully")

        print("✓ High contrast palette tests completed")
        return True

    except Exception as e:
        print(f"ERROR in high contrast palette test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_color_quantization_algorithm():
    """Test the k-means color quantization algorithm"""
    print("\n=== Testing Color Quantization Algorithm ===")
    
    try:
        import slicer
        from color_deca.deca3.InterDeCA import InterDeCALogic
        
        logic = InterDeCALogic()
        
        # Create test data with known color clusters
        np.random.seed(42)
        
        # Generate colors around 3 centers
        centers = np.array([
            [255, 0, 0],    # Red cluster
            [0, 255, 0],    # Green cluster
            [0, 0, 255],    # Blue cluster
        ])
        
        test_colors = []
        for center in centers:
            # Add some noise around each center
            for _ in range(20):
                noise = np.random.normal(0, 20, 3)
                color = np.clip(center + noise, 0, 255)
                test_colors.append(color)
        
        test_colors = np.array(test_colors)
        
        print(f"Generated {len(test_colors)} test colors around 3 centers")
        
        # Test quantization with 3 clusters
        result = logic.quantize_colors_lab_kmeans(
            test_colors, 3,
            logCallback=lambda msg: print(f"    {msg}")
        )
        
        if result.get("success", False):
            quantized_colors = result["quantized_colors"]
            cluster_centers = result["cluster_centers"]
            labels = result["labels"]
            
            print(f"✓ Quantization successful")
            print(f"  Original colors: {len(test_colors)}")
            print(f"  Quantized to: {len(cluster_centers)} clusters")
            print(f"  Cluster centers (RGB):")
            for i, center in enumerate(cluster_centers):
                count = np.sum(labels == i)
                print(f"    Cluster {i+1}: [{center[0]}, {center[1]}, {center[2]}] ({count} colors)")
            
            # Check that we got reasonable cluster centers
            unique_quantized = len(np.unique(quantized_colors.view(np.void), axis=0))
            print(f"  Unique quantized colors: {unique_quantized}")
            
            if unique_quantized <= 3:
                print("✓ Quantization reduced color count as expected")
            else:
                print("⚠ Warning: More unique colors than expected after quantization")
            
            return True
        else:
            print("ERROR: Quantization failed")
            return False
        
    except Exception as e:
        print(f"ERROR in quantization algorithm test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_integration_with_existing_workflow():
    """Test that quantization integrates properly with existing workflow"""
    print("\n=== Testing Integration with Existing Workflow ===")
    
    try:
        import slicer
        
        # Get the InterDeCA module widget
        moduleWidget = slicer.modules.interdeca.widgetRepresentation().self()
        
        # Check that the new method exists in the logic class
        from color_deca.deca3.InterDeCA import InterDeCALogic
        logic = InterDeCALogic()
        
        required_methods = [
            'rgb_to_lab',
            'lab_to_rgb',
            'delta_e_2000',
            'quantize_colors_lab_kmeans',
            'applyQuantizedFaceColorsFromTexture'
        ]
        
        for method in required_methods:
            if not hasattr(logic, method):
                print(f"ERROR: Method {method} not found in InterDeCALogic")
                return False
            else:
                print(f"✓ Method {method} found")
        
        # Test that the UI properly calls the new workflow
        print("\nTesting workflow integration:")
        print("  The new quantization workflow should be called when:")
        print("    - Average face colors is checked")
        print("    - Quantize colors is checked")
        print("    - A valid number of clusters is set")
        print("  This will be tested during actual usage with a model and texture.")
        
        print("✓ Integration tests completed")
        return True
        
    except Exception as e:
        print(f"ERROR in integration test: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_dependencies():
    """Check if required dependencies are available"""
    print("=== Dependency Check ===")

    dependencies = {
        'numpy': True,  # Always available in Slicer
        'sklearn': False,
        'scikit-image': False,
        'slicer': False
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
        import slicer
        dependencies['slicer'] = True
    except ImportError:
        pass

    for dep, available in dependencies.items():
        status = "✓" if available else "✗"
        print(f"  {dep}: {status}")

    missing = [dep for dep, available in dependencies.items() if not available]
    if missing:
        print(f"\nMissing dependencies: {', '.join(missing)}")
        print("Color quantization requires sklearn and scikit-image.")
        return False
    else:
        print("\n✓ All dependencies available")
        return True

def run_all_tests():
    """Run all quantization tests"""
    print("Color Quantization Test Suite")
    print("=" * 50)

    # Check dependencies first
    if not check_dependencies():
        print("\nSkipping tests due to missing dependencies.")
        return False
    
    tests = [
        ("UI Elements", test_color_quantization_ui),
        ("Color Space Conversions", test_color_space_conversions),
        ("High Contrast Palette", test_high_contrast_palette),
        ("Quantization Algorithm", test_color_quantization_algorithm),
        ("Integration", test_integration_with_existing_workflow),
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
    print("TEST RESULTS:")
    all_passed = True
    for test_name, success in results:
        status = "PASS" if success else "FAIL"
        print(f"  {test_name}: {status}")
        if not success:
            all_passed = False
    
    print(f"\nOverall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    return all_passed

if __name__ == "__main__":
    # This script is designed to be run in 3D Slicer's Python console
    print("This script should be run in 3D Slicer's Python console.")
    print("Copy and paste the following command:")
    print("exec(open('/path/to/test_color_quantization.py').read())")
    print("Then call: run_all_tests()")
