"""
Example usage script for ResultsReporter module.

Demonstrates how to generate comprehensive HTML reports from
DeCA/InterDeCA analysis results.
"""

import sys
from pathlib import Path

# Adds the color_deca directory to Python path
module_path = Path(__file__).parent.parent.parent
if str(module_path) not in sys.path:
    sys.path.insert(0, str(module_path))

from color_deca.reporting import ResultsReporter


def generate_example_report(output_directory):
    """Generates an example report with sample parameters.

    Args:
        output_directory: Path to the analysis output directory

    Returns:
        str: Path to the generated HTML report
    """
    # Creates reporter instance
    reporter = ResultsReporter(output_directory)

    # Defines sample analysis parameters
    parameters = {
        "Analysis Type": "Shape and Color Correspondence",
        "Number of Specimens": 5,
        "Template Method": "Auto-generated (mean shape)",
        "Landmark Count": "75 per specimen",
        "Point Density": "4% tolerance",
        "Processing Time": "12.3 minutes",
        "Alignment Method": "Generalized Procrustes Analysis",
        "Color Space": "RGB"
    }

    # Defines sample analysis statistics
    analysis_stats = {
        "Mean Shape Error": "0.95 mm",
        "Landmark Correspondence": "100% successful",
        "Texture Resolution": "2048x2048 pixels",
        "Atlas Quality Score": "0.95",
        "Color Variance": "73%",
        "Data Completeness": "100%"
    }

    # Generates the comprehensive report
    report_path = reporter.generate_comprehensive_report(
        parameters=parameters,
        analysis_stats=analysis_stats
    )

    print(f"Report generated successfully!")
    print(f"Location: {report_path}")
    print(f"\nTo view the report:")
    print(f"1. Open your web browser")
    print(f"2. Navigate to: file://{report_path}")

    return report_path


def main():
    """Main function to run the example."""
    import tempfile

    # Creates a temporary directory for the example
    temp_dir = Path(tempfile.mkdtemp(prefix="deca_report_example_"))

    print("=" * 60)
    print("RESULTS REPORTER EXAMPLE")
    print("=" * 60)
    print(f"\nOutput directory: {temp_dir}")

    # Creates some sample output files
    print("\nCreating sample files...")

    # Creates sample model files
    models_dir = temp_dir / "models"
    models_dir.mkdir(exist_ok=True)
    for i in range(3):
        model_file = models_dir / f"specimen_{i:03d}.ply"
        model_file.write_text(f"PLY sample content for specimen {i}")

    # Creates sample texture files
    textures_dir = temp_dir / "textures"
    textures_dir.mkdir(exist_ok=True)
    for i in range(3):
        texture_file = textures_dir / f"specimen_{i:03d}_texture.png"
        texture_file.write_text(f"PNG sample content for specimen {i}")

    # Creates sample landmark files
    landmarks_dir = temp_dir / "landmarks"
    landmarks_dir.mkdir(exist_ok=True)
    for i in range(3):
        landmark_file = landmarks_dir / f"specimen_{i:03d}_landmarks.json"
        landmark_file.write_text(f'{{"specimen": {i}, "points": 75}}')

    print("Sample files created successfully!")

    # Generates the report
    print("\nGenerating report...")
    report_path = generate_example_report(temp_dir)

    print("\n" + "=" * 60)
    print("EXAMPLE COMPLETE")
    print("=" * 60)

    return report_path


if __name__ == "__main__":
    main()