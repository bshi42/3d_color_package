#!/usr/bin/env python3
"""
Test script to verify the Recolor tab functionality in InterDeCA module.
This script can be run in 3D Slicer to test the new Recolor tab.
"""

import os
import sys

def test_recolor_tab():
    """Test the Recolor tab functionality"""
    print("Testing Recolor tab functionality...")
    
    try:
        # Import the InterDeCA module
        import slicer
        from slicer.ScriptedLoadableModule import *
        
        # Check if InterDeCA module is available
        moduleManager = slicer.app.moduleManager()
        if not moduleManager.isLoaded('InterDeCA'):
            print("InterDeCA module is not loaded. Please load it first.")
            return False
        
        # Get the InterDeCA widget
        interDeCAModule = moduleManager.module('InterDeCA')
        widget = interDeCAModule.widgetRepresentation()
        
        # Check if the Recolor tab exists
        tabsWidget = widget.tabsWidget
        tabCount = tabsWidget.count
        tabNames = [tabsWidget.tabText(i) for i in range(tabCount)]
        
        print(f"Available tabs: {tabNames}")
        
        if "Recolor" not in tabNames:
            print("ERROR: Recolor tab not found!")
            return False
        
        print("SUCCESS: Recolor tab found!")
        
        # Test accessing Recolor tab components
        recolorTabIndex = tabNames.index("Recolor")
        tabsWidget.setCurrentIndex(recolorTabIndex)
        
        # Check if the Recolor tab components exist
        components_to_check = [
            'recolorAtlasModelSelect',
            'recolorTexturesDirectorySelector', 
            'recolorTextureSelector',
            'averageFaceColorCheckbox',
            'applyRecolorButton',
            'recolorLogInfo'
        ]
        
        missing_components = []
        for component in components_to_check:
            if not hasattr(widget, component):
                missing_components.append(component)
        
        if missing_components:
            print(f"ERROR: Missing components: {missing_components}")
            return False
        
        print("SUCCESS: All Recolor tab components found!")
        
        # Test the logic class has the new method
        from color_deca.deca3.InterDeCA import InterDeCALogic
        logic = InterDeCALogic()
        
        if not hasattr(logic, 'applyAverageFaceColorsFromTexture'):
            print("ERROR: applyAverageFaceColorsFromTexture method not found in logic class!")
            return False
        
        print("SUCCESS: applyAverageFaceColorsFromTexture method found in logic class!")
        
        # Test callback functions exist
        callback_functions = [
            'onRecolorParameterChanged',
            'onRecolorTexturesDirectoryChanged',
            'onApplyRecolorButton',
            'updateRecolorProgress',
            'logRecolorMessage'
        ]
        
        missing_callbacks = []
        for callback in callback_functions:
            if not hasattr(widget, callback):
                missing_callbacks.append(callback)
        
        if missing_callbacks:
            print(f"ERROR: Missing callback functions: {missing_callbacks}")
            return False
        
        print("SUCCESS: All callback functions found!")
        
        print("\n=== Recolor Tab Test Summary ===")
        print("✓ Recolor tab exists")
        print("✓ All UI components present")
        print("✓ Logic method implemented")
        print("✓ All callback functions present")
        print("✓ All tests passed!")
        
        return True
        
    except Exception as e:
        print(f"ERROR during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

def demo_usage():
    """Demonstrate how to use the Recolor tab"""
    print("\n=== How to use the Recolor tab ===")
    print("1. Load an atlas model into 3D Slicer")
    print("2. Go to the InterDeCA module")
    print("3. Click on the 'Recolor' tab")
    print("4. Select your atlas model from the dropdown")
    print("5. Browse and select a directory containing texture images")
    print("6. Choose a texture from the dropdown list")
    print("7. Optionally check 'Average Face Colors' to color each face with")
    print("   the average color from that region of the texture")
    print("8. Click 'Apply Texture' to apply the texture to the model")
    print("\nThe model will be rendered in the 3D viewer with the selected texture!")

if __name__ == "__main__":
    # This would be run in 3D Slicer's Python console
    print("Recolor Tab Test Script")
    print("=" * 50)
    
    success = test_recolor_tab()
    
    if success:
        demo_usage()
    else:
        print("Tests failed. Please check the implementation.")
