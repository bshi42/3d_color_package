#!/usr/bin/env python3
"""
Demo script showing how the Colors EDA functionality would work
This demonstrates the core algorithm without requiring Slicer
"""

import numpy as np
import os
import colorsys
try:
    import imageio
    IMAGEIO_AVAILABLE = True
except ImportError:
    IMAGEIO_AVAILABLE = False
    print("Warning: imageio not available")

try:
    from sklearn.decomposition import PCA, FastICA
    from sklearn.manifold import TSNE
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: sklearn not available")

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("Warning: umap-learn not available")

def rgb_to_hsv_array(rgb_array):
    """Convert RGB array to HSV array with hue as 2D vector"""
    hsv_array = np.zeros((rgb_array.shape[0], 4))  # 4D: [hue_cos, hue_sin, sat, val]
    for i in range(rgb_array.shape[0]):
        rgb_normalized = rgb_array[i] / 255.0
        hsv = colorsys.rgb_to_hsv(rgb_normalized[0], rgb_normalized[1], rgb_normalized[2])
        # Convert hue to 2D vector to handle circular nature
        hue_radians = hsv[0] * 2 * np.pi
        hue_cos = np.cos(hue_radians)
        hue_sin = np.sin(hue_radians)
        hsv_array[i] = [hue_cos, hue_sin, hsv[1] * 100, hsv[2] * 100]
    return hsv_array

def simulate_face_colors(n_faces=1000, n_specimens=20, color_space="RGB"):
    """
    Simulate face-averaged colors for demonstration
    In the real implementation, this would come from texture sampling
    """
    print(f"Simulating {n_faces} faces across {n_specimens} specimens in {color_space} space")
    
    # Create some realistic color variations
    # Simulate different color patterns for different specimens
    all_face_colors = []
    
    for specimen_idx in range(n_specimens):
        # Each specimen has a base color with variations
        base_hue = (specimen_idx / n_specimens) * 360  # Spread hues across specimens
        base_saturation = 50 + np.random.normal(0, 10)  # Some variation in saturation
        base_value = 70 + np.random.normal(0, 15)  # Some variation in brightness
        
        specimen_colors = []
        for face_idx in range(n_faces):
            # Add some noise to create realistic face-to-face variation
            hue = (base_hue + np.random.normal(0, 20)) % 360
            sat = np.clip(base_saturation + np.random.normal(0, 15), 0, 100)
            val = np.clip(base_value + np.random.normal(0, 20), 0, 100)
            
            # Convert HSV to RGB
            rgb = colorsys.hsv_to_rgb(hue/360, sat/100, val/100)
            rgb_255 = [int(c * 255) for c in rgb]
            
            if color_space == "HSV":
                # Convert hue to 2D vector for circular representation
                hue_radians = (hue / 360.0) * 2 * np.pi
                hue_cos = np.cos(hue_radians)
                hue_sin = np.sin(hue_radians)
                specimen_colors.append([hue_cos, hue_sin, sat, val])
            else:  # RGB
                specimen_colors.append(rgb_255)
        
        all_face_colors.append(specimen_colors)
    
    return np.array(all_face_colors)

def apply_dimensionality_reduction(color_data, algorithm="PCA"):
    """Apply dimensionality reduction to color data"""
    print(f"Applying {algorithm} to data shape: {color_data.shape}")
    
    if algorithm == "PCA" and SKLEARN_AVAILABLE:
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2)
        return reducer.fit_transform(color_data)
    
    elif algorithm == "ICA" and SKLEARN_AVAILABLE:
        from sklearn.decomposition import FastICA
        reducer = FastICA(n_components=2, random_state=42)
        return reducer.fit_transform(color_data)
    
    elif algorithm == "UMAP" and UMAP_AVAILABLE:
        import umap
        reducer = umap.UMAP(n_components=2, random_state=42)
        return reducer.fit_transform(color_data)
    
    else:
        print(f"Algorithm {algorithm} not available, using simple projection")
        # Fallback: just use first two principal components manually
        centered_data = color_data - np.mean(color_data, axis=0)
        cov_matrix = np.cov(centered_data.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        # Sort by eigenvalues (descending)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, idx]
        # Project onto first two components
        return centered_data @ eigenvectors[:, :2]

def demo_colors_eda():
    """Demonstrate the Colors EDA workflow"""
    print("=== Colors EDA Demo ===")
    print()
    
    # Parameters
    n_faces = 500  # Number of faces in atlas
    n_specimens = 15  # Number of specimens
    
    for color_space in ["RGB", "HSV"]:
        print(f"\n--- Testing {color_space} color space ---")
        
        # Step 1: Simulate face-averaged colors
        face_colors = simulate_face_colors(n_faces, n_specimens, color_space)
        print(f"Generated face colors shape: {face_colors.shape}")
        
        # Step 2: Reshape for dimensionality reduction
        # From (n_specimens, n_faces, 3) to (n_faces * n_specimens, 3)
        color_data = face_colors.reshape(-1, 3)
        print(f"Reshaped for analysis: {color_data.shape}")
        
        # Step 3: Apply dimensionality reduction
        for algorithm in ["PCA", "ICA", "UMAP"]:
            print(f"\n  Testing {algorithm}:")
            
            reduced_data = apply_dimensionality_reduction(color_data, algorithm)
            print(f"  Reduced data shape: {reduced_data.shape}")
            print(f"  Component 1 range: [{reduced_data[:, 0].min():.2f}, {reduced_data[:, 0].max():.2f}]")
            print(f"  Component 2 range: [{reduced_data[:, 1].min():.2f}, {reduced_data[:, 1].max():.2f}]")
            
            # In the real implementation, this would be plotted in Slicer's 2D viewer
            print(f"  -> Would plot {reduced_data.shape[0]} points in 2D viewer")

def main():
    print("Colors EDA Demo - Testing the core algorithm")
    print("=" * 50)
    
    # Check dependencies
    print("Dependency check:")
    print(f"  numpy: ✓")
    print(f"  colorsys: ✓")
    print(f"  imageio: {'✓' if IMAGEIO_AVAILABLE else '✗'}")
    print(f"  sklearn: {'✓' if SKLEARN_AVAILABLE else '✗'}")
    print(f"  umap-learn: {'✓' if UMAP_AVAILABLE else '✗'}")
    print()
    
    # Run demo
    demo_colors_eda()
    
    print("\n" + "=" * 50)
    print("Demo completed!")
    print("\nIn the actual Slicer module:")
    print("1. User selects atlas model and baked textures directory")
    print("2. User chooses RGB or HSV color space")
    print("3. User selects PCA, ICA, or UMAP algorithm")
    print("4. System calculates face-averaged colors from textures")
    print("5. Dimensionality reduction is applied")
    print("6. Results are plotted in Slicer's 2D viewer")

if __name__ == "__main__":
    main()
