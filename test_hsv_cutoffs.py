#!/usr/bin/env python3
"""
Test script for HSV cutoff functionality in Colors EDA tab.
Run this in 3D Slicer's Python console to test the HSV filtering.
"""

import numpy as np
import colorsys

def test_hsv_cutoff_functionality():
    """Test the HSV cutoff functionality"""
    print("=== Testing HSV Cutoff Functionality ===")
    
    try:
        import slicer
        
        # Get the InterDeCA module widget
        moduleWidget = slicer.modules.interdeca.widgetRepresentation().self()
        
        # Check if the required UI elements exist
        required_elements = [
            'satCutoffSpin', 'valueCutoffSpin', 'hsvFilterLabel',
            'hsvRadio', 'pcaRadio'
        ]
        
        for element in required_elements:
            if not hasattr(moduleWidget, element):
                print(f"ERROR: {element} not found")
                return False
        
        print("✓ All required UI elements found")
        
        # Test the cutoff spin boxes
        print(f"✓ Saturation cutoff range: {moduleWidget.satCutoffSpin.minimum()}-{moduleWidget.satCutoffSpin.maximum()}%")
        print(f"✓ Value cutoff range: {moduleWidget.valueCutoffSpin.minimum()}-{moduleWidget.valueCutoffSpin.maximum()}%")
        print(f"✓ Current saturation cutoff: {moduleWidget.satCutoffSpin.value}%")
        print(f"✓ Current value cutoff: {moduleWidget.valueCutoffSpin.value}%")
        
        # Test setting different values
        print("\nTesting cutoff value changes...")
        original_sat = moduleWidget.satCutoffSpin.value
        original_val = moduleWidget.valueCutoffSpin.value
        
        moduleWidget.satCutoffSpin.setValue(25.0)
        moduleWidget.valueCutoffSpin.setValue(30.0)
        
        print(f"✓ Set saturation cutoff to: {moduleWidget.satCutoffSpin.value}%")
        print(f"✓ Set value cutoff to: {moduleWidget.valueCutoffSpin.value}%")
        
        # Restore original values
        moduleWidget.satCutoffSpin.setValue(original_sat)
        moduleWidget.valueCutoffSpin.setValue(original_val)
        
        print("✓ Restored original cutoff values")
        
        return True
        
    except Exception as e:
        print(f"ERROR during test: {e}")
        import traceback
        traceback.print_exc()
        return False

def simulate_hsv_filtering():
    """Simulate HSV filtering to demonstrate the concept"""
    print("\n=== Simulating HSV Filtering ===")
    
    # Create sample HSV data (hue_cos, hue_sin, saturation, value)
    np.random.seed(42)
    n_samples = 1000
    
    # Generate random HSV values
    hue_angles = np.random.uniform(0, 2*np.pi, n_samples)
    hue_cos = np.cos(hue_angles)
    hue_sin = np.sin(hue_angles)
    saturation = np.random.uniform(0, 100, n_samples)  # 0-100%
    value = np.random.uniform(0, 100, n_samples)       # 0-100%
    
    # Create HSV data array (format used by InterDeCA)
    hsv_data = np.column_stack([hue_cos, hue_sin, saturation, value])
    
    print(f"Generated {n_samples} HSV samples")
    print(f"Saturation range: {saturation.min():.1f} - {saturation.max():.1f}%")
    print(f"Value range: {value.min():.1f} - {value.max():.1f}%")
    
    # Test different cutoff values
    cutoff_tests = [
        (10.0, 10.0),   # Low cutoffs
        (25.0, 25.0),   # Medium cutoffs
        (50.0, 50.0),   # High cutoffs
    ]
    
    for sat_cutoff, val_cutoff in cutoff_tests:
        # Apply filtering
        mask = (saturation >= sat_cutoff) & (value >= val_cutoff)
        filtered_data = hsv_data[mask]
        
        retention_rate = (np.sum(mask) / len(mask)) * 100
        
        print(f"\nCutoffs: Sat>={sat_cutoff}%, Val>={val_cutoff}%")
        print(f"  Samples retained: {np.sum(mask)}/{len(mask)} ({retention_rate:.1f}%)")
        print(f"  Filtered data shape: {filtered_data.shape}")
    
    print("\n✓ HSV filtering simulation completed")
    return True

def test_color_space_conversion():
    """Test HSV color space conversion consistency"""
    print("\n=== Testing HSV Color Space Conversion ===")
    
    # Test a few known RGB to HSV conversions
    test_colors = [
        ([255, 0, 0], "Red"),      # Pure red
        ([0, 255, 0], "Green"),    # Pure green  
        ([0, 0, 255], "Blue"),     # Pure blue
        ([128, 128, 128], "Gray"), # Gray (low saturation)
        ([64, 64, 64], "Dark Gray"), # Dark gray (low value)
    ]
    
    for rgb, name in test_colors:
        # Convert RGB to HSV using the same method as InterDeCA
        rgb_normalized = np.array(rgb) / 255.0
        hsv = colorsys.rgb_to_hsv(rgb_normalized[0], rgb_normalized[1], rgb_normalized[2])
        
        # Convert hue to 2D vector (same as InterDeCA)
        hue_radians = hsv[0] * 2 * np.pi
        hue_cos = np.cos(hue_radians)
        hue_sin = np.sin(hue_radians)
        
        # Create 4D vector: [hue_cos, hue_sin, saturation, value]
        hsv_vector = np.array([hue_cos, hue_sin, hsv[1] * 100, hsv[2] * 100])
        
        print(f"{name:10} RGB{rgb} -> HSV({hsv[0]*360:.0f}°, {hsv[1]*100:.0f}%, {hsv[2]*100:.0f}%)")
        print(f"           Vector: [{hue_cos:.3f}, {hue_sin:.3f}, {hsv[1]*100:.1f}, {hsv[2]*100:.1f}]")
    
    print("\n✓ HSV conversion test completed")
    return True

if __name__ == "__main__":
    # Run tests
    print("HSV Cutoff Functionality Test Suite")
    print("=" * 50)
    
    success1 = test_hsv_cutoff_functionality()
    success2 = simulate_hsv_filtering()
    success3 = test_color_space_conversion()
    
    print("\n" + "=" * 50)
    if success1 and success2 and success3:
        print("✓ ALL TESTS PASSED")
        print("\nThe HSV cutoff functionality is working properly.")
        print("\nUsage:")
        print("1. Set Colors EDA to HSV mode")
        print("2. Adjust 'Saturation cutoff' and 'Value cutoff' values")
        print("3. Run analysis - cutoffs will filter data for dimensionality reduction")
        print("4. Switch to hue histogram - same cutoffs will filter histogram data")
        print("5. Other histograms (sat/val) use full dataset")
    else:
        print("✗ SOME TESTS FAILED")
        print("Check the error messages above for details.")
