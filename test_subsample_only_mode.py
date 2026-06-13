#!/usr/bin/env python3
"""
Test script for subsample-only mode in MultiRecolor functionality
Run this in 3D Slicer's Python console
"""

import numpy as np
import os

def test_subsample_only_mode():
    """Test the subsample-only mode workflow"""
    print("\n=== Testing Subsample-Only Mode ===")
    
    try:
        import slicer
        
        # Get the InterDeCA module widget
        moduleWidget = slicer.modules.interdeca.widgetRepresentation().self()
        
        # Check if the new UI elements exist
        print("\n1. Checking UI elements...")
        required_elements = [
            'multiRecolorClusteringRadio',
            'multiRecolorSubsampleOnlyRadio',
            'multiRecolorModeGroup'
        ]
        
        for element in required_elements:
            if not hasattr(moduleWidget, element):
                print(f"  ✗ {element} not found")
                return False
            else:
                print(f"  ✓ {element} found")
        
        # Test mode selection
        print("\n2. Testing mode selection...")
        
        # Check initial state (should be clustering mode)
        if moduleWidget.multiRecolorClusteringRadio.isChecked():
            print("  ✓ Clustering mode is default")
        else:
            print("  ✗ Clustering mode should be default")
            return False
        
        # Switch to subsample-only mode
        moduleWidget.multiRecolorSubsampleOnlyRadio.setChecked(True)
        if moduleWidget.multiRecolorSubsampleOnlyRadio.isChecked():
            print("  ✓ Successfully switched to subsample-only mode")
        else:
            print("  ✗ Failed to switch to subsample-only mode")
            return False
        
        # Check that clustering controls are disabled
        print("\n3. Checking clustering controls state...")
        if not moduleWidget.multiRecolorInitialClustersSpin.enabled:
            print("  ✓ Initial clusters spin disabled in subsample-only mode")
        else:
            print("  ✗ Initial clusters spin should be disabled")
            return False
        
        if not moduleWidget.multiRecolorConsolidatedClustersSpin.enabled:
            print("  ✓ Consolidated clusters spin disabled in subsample-only mode")
        else:
            print("  ✗ Consolidated clusters spin should be disabled")
            return False
        
        # Switch back to clustering mode
        print("\n4. Testing mode switch back to clustering...")
        moduleWidget.multiRecolorClusteringRadio.setChecked(True)
        if moduleWidget.multiRecolorClusteringRadio.isChecked():
            print("  ✓ Successfully switched back to clustering mode")
        else:
            print("  ✗ Failed to switch back to clustering mode")
            return False
        
        # Check that clustering controls are enabled
        if moduleWidget.multiRecolorInitialClustersSpin.enabled:
            print("  ✓ Initial clusters spin enabled in clustering mode")
        else:
            print("  ✗ Initial clusters spin should be enabled")
            return False
        
        if moduleWidget.multiRecolorConsolidatedClustersSpin.enabled:
            print("  ✓ Consolidated clusters spin enabled in clustering mode")
        else:
            print("  ✗ Consolidated clusters spin should be enabled")
            return False
        
        print("\n✓ All subsample-only mode tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_subsample_only_logic():
    """Test the subsample-only logic methods"""
    print("\n=== Testing Subsample-Only Logic Methods ===")
    
    try:
        # Import the logic class
        import sys
        sys.path.insert(0, '/home/alek/projects/3d_color_package/color_deca/deca3')
        from InterDeCA import InterDeCALogic
        
        logic = InterDeCALogic()
        
        # Check if the new methods exist
        print("\n1. Checking logic methods...")
        required_methods = [
            'performSubsampleOnly',
            'applyTextureWithSubsamplingOnly'
        ]
        
        for method in required_methods:
            if hasattr(logic, method):
                print(f"  ✓ {method} found")
            else:
                print(f"  ✗ {method} not found")
                return False
        
        print("\n✓ All logic method tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("Subsample-Only Mode Test Suite")
    print("=" * 60)
    
    # Run tests
    ui_test_passed = test_subsample_only_mode()
    logic_test_passed = test_subsample_only_logic()
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"UI Tests: {'PASSED' if ui_test_passed else 'FAILED'}")
    print(f"Logic Tests: {'PASSED' if logic_test_passed else 'FAILED'}")
    
    if ui_test_passed and logic_test_passed:
        print("\n✓ All tests passed!")
    else:
        print("\n✗ Some tests failed")

