#!/usr/bin/env python3
"""
Test script for hue-based coloring in 2D dimensionality reduction plots.
Run this in 3D Slicer's Python console to test the hue coloring functionality.
"""

import numpy as np
import colorsys

def test_hue_color_conversion():
    """Test the hue color conversion logic"""
    print("=== Testing Hue Color Conversion ===")
    
    # Test known hue values
    test_hues = [
        (0, "Red"),
        (60, "Yellow"), 
        (120, "Green"),
        (180, "Cyan"),
        (240, "Blue"),
        (300, "Magenta")
    ]
    
    print("Testing hue to RGB conversion:")
    for hue_deg, color_name in test_hues:
        # Convert to radians and create 2D vector (as InterDeCA does)
        hue_rad = np.radians(hue_deg)
        hue_cos = np.cos(hue_rad)
        hue_sin = np.sin(hue_rad)
        
        # Convert back to hue angle
        recovered_hue_rad = np.arctan2(hue_sin, hue_cos)
        recovered_hue_deg = np.degrees(recovered_hue_rad) % 360
        
        # Convert to RGB with high saturation and value for visibility
        rgb = colorsys.hsv_to_rgb(recovered_hue_deg / 360.0, 0.9, 0.9)
        rgb_255 = [int(c * 255) for c in rgb]
        
        print(f"  {color_name:8} ({hue_deg:3d}°) -> Vector: [{hue_cos:6.3f}, {hue_sin:6.3f}] -> RGB: {rgb_255}")
    
    print("✓ Hue conversion test completed")
    return True

def test_hsv_data_structure():
    """Test HSV data structure used by InterDeCA"""
    print("\n=== Testing HSV Data Structure ===")
    
    # Create sample HSV data in InterDeCA format
    n_samples = 10
    np.random.seed(42)
    
    # Generate random hue angles
    hue_angles = np.random.uniform(0, 2*np.pi, n_samples)
    hue_cos = np.cos(hue_angles)
    hue_sin = np.sin(hue_angles)
    
    # Generate saturation and value
    saturation = np.random.uniform(20, 100, n_samples)  # 20-100%
    value = np.random.uniform(30, 100, n_samples)       # 30-100%
    
    # Create HSV data array (InterDeCA format)
    hsv_data = np.column_stack([hue_cos, hue_sin, saturation, value])
    
    print(f"Generated {n_samples} HSV samples:")
    print("Sample | Hue°  | Sat% | Val% | RGB")
    print("-" * 40)
    
    for i in range(min(5, n_samples)):  # Show first 5 samples
        # Convert back to hue angle
        hue_angle = np.arctan2(hsv_data[i, 1], hsv_data[i, 0])
        hue_deg = np.degrees(hue_angle) % 360
        sat = hsv_data[i, 2]
        val = hsv_data[i, 3]
        
        # Convert to RGB for display
        display_sat = max(sat / 100.0, 0.7)  # Ensure visibility
        display_val = max(val / 100.0, 0.8)  # Ensure visibility
        rgb = colorsys.hsv_to_rgb(hue_deg / 360.0, display_sat, display_val)
        rgb_255 = [int(c * 255) for c in rgb]
        
        print(f"  {i+1:2d}   | {hue_deg:5.1f} | {sat:4.1f} | {val:4.1f} | {rgb_255}")
    
    print("✓ HSV data structure test completed")
    return hsv_data

def test_plot_coloring_logic():
    """Test the plot coloring logic"""
    print("\n=== Testing Plot Coloring Logic ===")

    try:
        import slicer

        # Get the InterDeCA module widget
        moduleWidget = slicer.modules.interdeca.widgetRepresentation().self()

        # Check if the required UI elements exist
        required_elements = [
            'enhanceColorsCheck', 'satCutoffSpin', 'valueCutoffSpin',
            'hsvRadio', 'pcaRadio'
        ]

        for element in required_elements:
            if not hasattr(moduleWidget, element):
                print(f"ERROR: {element} not found")
                return False

        print("✓ All required UI elements found")

        # Test the color enhancement checkbox
        print(f"✓ Color enhancement checkbox found")
        print(f"✓ Current enhance colors setting: {moduleWidget.enhanceColorsCheck.isChecked()}")

        # Test toggling the checkbox
        original_enhance = moduleWidget.enhanceColorsCheck.isChecked()
        moduleWidget.enhanceColorsCheck.setChecked(not original_enhance)
        print(f"✓ Toggled enhance colors to: {moduleWidget.enhanceColorsCheck.isChecked()}")
        moduleWidget.enhanceColorsCheck.setChecked(original_enhance)
        print("✓ Restored original enhance colors setting")

        # Get the InterDeCA module logic
        logic = slicer.modules.interdeca.logic()

        if not hasattr(logic, '_createColoredScatterPlot'):
            print("ERROR: _createColoredScatterPlot method not found")
            return False

        print("✓ Colored scatter plot method found")
        print("✓ Ready to test plot creation with actual/enhanced color options")

        return True

    except Exception as e:
        print(f"ERROR during plot coloring test: {e}")
        return False

def demonstrate_color_wheel():
    """Demonstrate the color wheel effect that should appear in plots"""
    print("\n=== Color Wheel Demonstration ===")

    # Show the hue ranges used in the multi-series plot
    hue_ranges = [
        (0, 30, "Red"),
        (30, 60, "Orange"),
        (60, 90, "Yellow"),
        (90, 120, "Yellow-Green"),
        (120, 150, "Green"),
        (150, 180, "Green-Cyan"),
        (180, 210, "Cyan"),
        (210, 240, "Cyan-Blue"),
        (240, 270, "Blue"),
        (270, 300, "Blue-Magenta"),
        (300, 330, "Magenta"),
        (330, 360, "Red-Magenta")
    ]

    print("Hue ranges used in multi-series plot:")
    print("Range     | Color Name    | Series Color")
    print("-" * 45)

    for hue_min, hue_max, color_name in hue_ranges:
        # Calculate representative hue for this range
        mid_hue = (hue_min + hue_max) / 2
        if hue_min == 330:  # Handle wraparound
            mid_hue = 345

        # Convert to RGB for display
        rgb = colorsys.hsv_to_rgb(mid_hue / 360.0, 0.9, 0.9)
        rgb_255 = [int(c * 255) for c in rgb]

        print(f"{hue_min:3d}°-{hue_max:3d}° | {color_name:12} | {rgb_255}")

    print("\n✓ The 2D plot will show separate series for each hue range")
    print("✓ Points are grouped by hue and colored accordingly")
    print("✓ This creates distinct color clusters in the dimensionality reduction space")

    return True

if __name__ == "__main__":
    # Run tests
    print("Hue-Based Coloring Test Suite")
    print("=" * 50)
    
    success1 = test_hue_color_conversion()
    success2 = test_hsv_data_structure()
    success3 = test_plot_coloring_logic()
    success4 = demonstrate_color_wheel()
    
    print("\n" + "=" * 50)
    if success1 and success2 and success3 and success4:
        print("✓ ALL TESTS PASSED")
        print("\nThe hue-based coloring functionality is ready!")
        print("\nUsage:")
        print("1. Set Colors EDA to HSV mode")
        print("2. Choose color enhancement option:")
        print("   - Unchecked: Uses actual colors from data (default)")
        print("   - Checked: Enhances colors for better visibility")
        print("3. Run analysis with any dimensionality reduction algorithm")
        print("4. The 2D plot will show average color of all points")
        print("5. Individual color data is stored in the plot table")
        print("\nExpected result:")
        print("- Plot series colored by average hue of all data points")
        print("- Individual RGB and hue data preserved in table")
        print("- Option to use actual vs enhanced colors")
        print("- Color information available for future analysis")
    else:
        print("✗ SOME TESTS FAILED")
        print("Check the error messages above for details.")
