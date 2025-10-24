"""
Test script for PCA Morphospace functionality

This script demonstrates the PCA color morphospace analysis
without requiring 3D Slicer to be running.

Usage:
    python test_pca_morphospace.py

Author: PCA Morphospace Implementation Team
Date: October 2024
"""

import sys
import os
import numpy as np
from pathlib import Path
import tempfile

# Adds parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from . import PCAMorphospaceVisualization


def create_sample_pca_data():
    """Creates sample PCA analysis data for testing

    Returns:
        dict: Sample PCA results
    """

    # Simulates 15 fish specimens
    n_specimens = 15
    n_vertices = 1000  # Number of vertices per model
    n_components = 5

    # Generates random PC scores for specimens
    pc_scores = np.random.randn(n_specimens, n_components) * 10

    # Creates specimen names
    specimen_names = [
        "N_brichardi_001", "N_brichardi_002", "N_brichardi_003",
        "N_brichardi_004", "N_brichardi_005", "N_brichardi_006",
        "N_pulcher_001", "N_pulcher_002", "N_pulcher_003",
        "N_pulcher_004", "N_pulcher_005", "N_pulcher_006",
        "Hybrid_001", "Hybrid_002", "Hybrid_003"
    ]

    # Generates variance explained (typical PCA pattern)
    variance_explained = [45.2, 23.1, 12.4, 8.7, 5.1]

    # Simulates mean color vector (flattened vertex colors)
    mean_colors = np.random.rand(n_vertices * 3) * 0.5 + 0.25  # Mid-range colors

    # Generates PC loadings (eigenvectors)
    pc_loadings = np.random.randn(n_components, n_vertices * 3) * 0.1

    # Creates vertex color dictionary for each specimen
    vertex_colors = {}
    for i, name in enumerate(specimen_names):
        # Reconstructs colors from PCA
        # color = mean + sum(score_i * loading_i)
        colors = mean_colors.copy()
        for j in range(n_components):
            colors += pc_scores[i, j] * pc_loadings[j] * 0.01

        # Clips to valid range
        colors = np.clip(colors, 0, 1)
        vertex_colors[name] = colors.reshape(n_vertices, 3)

    return {
        'specimen_names': specimen_names,
        'pc_scores': pc_scores,
        'variance_explained': variance_explained,
        'mean_colors': mean_colors,
        'pc_loadings': pc_loadings,
        'vertex_colors': vertex_colors,
        'n_vertices': n_vertices
    }


def test_color_interpolation(pca_data):
    """Tests color interpolation along PC axes

    Args:
        pca_data: PCA analysis results

    Returns:
        dict: Interpolated colors at different PC positions
    """

    print("\nTesting color interpolation along PC1...")

    # Gets PC1 statistics
    pc1_scores = pca_data['pc_scores'][:, 0]
    pc1_std = np.std(pc1_scores)
    pc1_mean = np.mean(pc1_scores)

    print(f"PC1 Statistics:")
    print(f"  Mean: {pc1_mean:.2f}")
    print(f"  Std Dev: {pc1_std:.2f}")
    print(f"  Range: [{pc1_scores.min():.2f}, {pc1_scores.max():.2f}]")

    # Tests interpolation at different positions
    test_positions = [-2, -1, 0, 1, 2]  # Standard deviations
    interpolated_colors = {}

    for sd_pos in test_positions:
        # Calculates interpolated colors
        # color = mean + sd_position * sqrt(eigenvalue) * eigenvector
        shift = sd_pos * pc1_std * pca_data['pc_loadings'][0]
        colors = pca_data['mean_colors'] + shift

        # Clips to valid range
        colors = np.clip(colors, 0, 1)

        # Reshapes to vertex colors
        n_vertices = pca_data['n_vertices']
        colors_reshaped = colors.reshape(n_vertices, 3)

        interpolated_colors[f"{sd_pos:+.0f}SD"] = colors_reshaped

        # Calculates average color at this position
        avg_color = colors_reshaped.mean(axis=0)
        print(f"\nPosition {sd_pos:+.0f} SD:")
        print(f"  Average RGB: [{avg_color[0]:.3f}, {avg_color[1]:.3f}, {avg_color[2]:.3f}]")

    return interpolated_colors


def main():
    """Main test function"""

    print("=" * 60)
    print("PCA COLOR MORPHOSPACE TEST")
    print("=" * 60)

    # Creates sample PCA data
    print("\nGenerating sample PCA analysis data...")
    pca_data = create_sample_pca_data()

    print(f"\nDataset Summary:")
    print(f"  Specimens: {len(pca_data['specimen_names'])}")
    print(f"  Vertices per model: {pca_data['n_vertices']}")
    print(f"  Principal Components: {len(pca_data['variance_explained'])}")
    print(f"  Total variance explained: {sum(pca_data['variance_explained']):.1f}%")

    # Tests color interpolation
    interpolated = test_color_interpolation(pca_data)

    # Creates output directory
    output_dir = Path(tempfile.mkdtemp(prefix="pca_morphospace_test_"))
    print(f"\nOutput directory: {output_dir}")

    # Generates HTML visualization
    print("\nGenerating interactive HTML visualization...")
    html_path = PCAMorphospaceVisualization.generate_interactive_html(
        pca_data,
        output_dir / "pca_morphospace.html"
    )

    print(f"\nHTML visualization created: {html_path}")

    # Exports additional data
    import json

    # Saves PCA scores
    scores_file = output_dir / "pca_scores.json"
    scores_dict = {
        name: pca_data['pc_scores'][i].tolist()
        for i, name in enumerate(pca_data['specimen_names'])
    }
    with open(scores_file, 'w') as f:
        json.dump(scores_dict, f, indent=2)
    print(f"PCA scores saved: {scores_file}")

    # Saves variance explained
    variance_file = output_dir / "variance_explained.json"
    variance_dict = {
        f"PC{i+1}": var for i, var in enumerate(pca_data['variance_explained'])
    }
    with open(variance_file, 'w') as f:
        json.dump(variance_dict, f, indent=2)
    print(f"Variance explained saved: {variance_file}")

    # Generates full report
    print("\nGenerating complete morphospace report...")
    report_path = PCAMorphospaceVisualization.generate_color_morphospace_report(
        str(output_dir),
        str(output_dir / "report")
    )

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
    print("\nTo view the interactive visualization:")
    print(f"1. Open your web browser")
    print(f"2. Navigate to: file://{html_path}")
    print("\nFiles generated:")
    for file in output_dir.rglob("*"):
        if file.is_file():
            print(f"  - {file.relative_to(output_dir)}")

    return output_dir


if __name__ == "__main__":
    output_dir = main()
    print(f"\nAll test files saved to: {output_dir}")