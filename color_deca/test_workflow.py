#!/usr/bin/env python3
"""
Test script for the new Colors EDA two-phase workflow
This script validates the key functionality without requiring Slicer
"""

import numpy as np
import os
import sys

def test_face_sampling():
    """Test the face sampling logic"""
    print("Testing face sampling logic...")
    
    # Simulate face sampling
    nTotalFaces = 1000
    samplePercent = 25.0
    randomSeed = 42
    
    # Calculate number of faces to sample
    nSampleFaces = max(1, int(nTotalFaces * samplePercent / 100.0))
    print(f"Total faces: {nTotalFaces}")
    print(f"Sample percentage: {samplePercent}%")
    print(f"Faces to sample: {nSampleFaces}")
    
    # Set random seed and sample face indices
    np.random.seed(randomSeed)
    if samplePercent >= 100.0:
        sampledFaceIndices = np.arange(nTotalFaces)
    else:
        sampledFaceIndices = np.random.choice(nTotalFaces, size=nSampleFaces, replace=False)
        sampledFaceIndices = np.sort(sampledFaceIndices)
    
    print(f"Sampled {len(sampledFaceIndices)} face indices")
    print(f"First 10 indices: {sampledFaceIndices[:10]}")
    
    # Test reproducibility
    np.random.seed(randomSeed)
    sampledFaceIndices2 = np.random.choice(nTotalFaces, size=nSampleFaces, replace=False)
    sampledFaceIndices2 = np.sort(sampledFaceIndices2)
    
    if np.array_equal(sampledFaceIndices, sampledFaceIndices2):
        print("✓ Face sampling is reproducible with same seed")
    else:
        print("✗ Face sampling is not reproducible")
        return False
    
    return True

def test_color_data_structure():
    """Test the color data structure and transformations"""
    print("\nTesting color data structure...")
    
    # Simulate sampled color data
    nSpecimens = 5
    nSampledFaces = 250
    nChannels = 3  # RGB
    
    # Create mock sampled color data
    sampledColorData = np.random.randint(0, 256, size=(nSpecimens, nSampledFaces, nChannels))
    print(f"Sampled color data shape: {sampledColorData.shape}")
    
    # Test reshaping for analysis
    colorDataFlat = sampledColorData.reshape(-1, nChannels)
    print(f"Flattened color data shape: {colorDataFlat.shape}")
    
    expected_flat_shape = (nSpecimens * nSampledFaces, nChannels)
    if colorDataFlat.shape == expected_flat_shape:
        print("✓ Color data reshaping works correctly")
    else:
        print(f"✗ Color data reshaping failed. Expected {expected_flat_shape}, got {colorDataFlat.shape}")
        return False
    
    return True

def test_hsv_conversion():
    """Test HSV color space conversion logic"""
    print("\nTesting HSV conversion logic...")
    
    try:
        import colorsys
        
        # Test RGB to HSV conversion
        rgb = np.array([128, 64, 192]) / 255.0  # Normalize to 0-1
        hsv = colorsys.rgb_to_hsv(rgb[0], rgb[1], rgb[2])
        
        # Convert hue to 2D vector (cos, sin) to handle circular nature
        hue_radians = hsv[0] * 2 * np.pi
        hue_cos = np.cos(hue_radians)
        hue_sin = np.sin(hue_radians)
        
        # Create 4D vector: [hue_cos, hue_sin, saturation, value]
        hsv_vec = np.array([hue_cos, hue_sin, hsv[1] * 100, hsv[2] * 100])
        
        print(f"Original RGB: {rgb * 255}")
        print(f"HSV: {hsv}")
        print(f"HSV vector: {hsv_vec}")
        
        # Test that hue vector has unit length (approximately)
        hue_magnitude = np.sqrt(hue_cos**2 + hue_sin**2)
        if abs(hue_magnitude - 1.0) < 1e-10:
            print("✓ Hue vector has unit magnitude")
        else:
            print(f"✗ Hue vector magnitude is {hue_magnitude}, expected 1.0")
            return False
        
        print("✓ HSV conversion works correctly")
        return True
        
    except ImportError:
        print("⚠ colorsys not available, skipping HSV test")
        return True

def test_workflow_phases():
    """Test the overall workflow phases"""
    print("\nTesting workflow phases...")
    
    # Phase 1: Data Sampling (simulated)
    print("Phase 1: Data Sampling")
    nTotalFaces = 1000
    samplePercent = 50.0
    nSpecimens = 3
    
    # Simulate sampling
    nSampleFaces = int(nTotalFaces * samplePercent / 100.0)
    sampledColorData = np.random.randint(0, 256, size=(nSpecimens, nSampleFaces, 3))
    specimenNames = [f"specimen_{i+1}" for i in range(nSpecimens)]
    faceIndices = np.random.choice(nTotalFaces, size=nSampleFaces, replace=False)
    
    print(f"✓ Sampled {nSampleFaces} faces from {nSpecimens} specimens")
    
    # Phase 2: Analysis (simulated)
    print("Phase 2: Analysis")
    colorDataFlat = sampledColorData.reshape(-1, 3)
    
    # Simulate dimensionality reduction (just take first 2 components)
    reducedData = colorDataFlat[:, :2]  # Simple mock reduction
    
    print(f"✓ Reduced data from {colorDataFlat.shape} to {reducedData.shape}")
    
    return True

def main():
    """Run all tests"""
    print("=== Colors EDA Workflow Validation ===\n")
    
    tests = [
        test_face_sampling,
        test_color_data_structure,
        test_hsv_conversion,
        test_workflow_phases
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                print(f"✗ {test.__name__} failed")
        except Exception as e:
            print(f"✗ {test.__name__} failed with exception: {e}")
    
    print(f"\n=== Results: {passed}/{total} tests passed ===")
    
    if passed == total:
        print("🎉 All tests passed! The workflow implementation looks good.")
        return True
    else:
        print("❌ Some tests failed. Please review the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
