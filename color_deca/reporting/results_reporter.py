"""
Results Reporter Module for InterDeCA/DeCA Analysis

Generates comprehensive HTML reports with statistics, visualizations, and detailed analysis summaries.
Perfect for team presentations and documenting research results.
"""

import os
import json
from datetime import datetime
from pathlib import Path
import hashlib

class ResultsReporter:
    """
    Creates comprehensive HTML reports for morphometric analysis results.
    Includes statistics, file metrics, validation checks, and professional formatting.
    """

    def __init__(self, output_directory):
        """
        Initializes the reporter with the output directory path.

        Args:
            output_directory: Path to the analysis output folder
        """
        self.output_dir = Path(output_directory)
        self.report_path = self.output_dir / "analysis_report.html"
        self.timestamp = datetime.now()

    def generate_comprehensive_report(self, parameters=None, analysis_stats=None):
        """
        Generates a comprehensive HTML report with all analysis details.

        Args:
            parameters: Dictionary of analysis parameters used
            analysis_stats: Optional dictionary of statistical results

        Returns:
            Path to the generated report file
        """
        # Collects comprehensive information
        output_files = self._scan_output_files_detailed()
        validation_results = self._validate_outputs(output_files)
        file_statistics = self._calculate_statistics(output_files)

        # Creates HTML content
        html_content = self._create_comprehensive_html(
            output_files,
            parameters,
            validation_results,
            file_statistics,
            analysis_stats
        )

        # Writes the report with UTF-8 encoding
        with open(self.report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"Report generated: {self.report_path}")
        return str(self.report_path)

    def _scan_output_files_detailed(self):
        """
        Performs detailed scan of output directory with file metrics.

        Returns:
            Dictionary with detailed file information
        """
        files = {
            'models': [],
            'textures': [],
            'landmarks': [],
            'plots': [],
            'other': [],
            'summary': {
                'total_files': 0,
                'total_size_mb': 0,
                'file_types': {}
            }
        }

        if not self.output_dir.exists():
            return files

        total_size = 0

        for file_path in self.output_dir.rglob('*'):
            if file_path.is_file():
                # Gets file information
                file_size = file_path.stat().st_size
                total_size += file_size
                ext = file_path.suffix.lower()

                file_info = {
                    'name': file_path.name,
                    'size_kb': round(file_size / 1024, 2),
                    'modified': datetime.fromtimestamp(file_path.stat().st_mtime).strftime('%Y-%m-%d %H:%M'),
                    'path': str(file_path.relative_to(self.output_dir))
                }

                # Categorizes by extension with detailed info
                if ext in ['.ply', '.obj', '.stl', '.vtp', '.vtk']:
                    files['models'].append(file_info)
                elif ext in ['.png', '.jpg', '.jpeg', '.bmp']:
                    files['textures'].append(file_info)
                elif ext in ['.json', '.fcsv', '.mrk']:
                    files['landmarks'].append(file_info)
                elif ext in ['.pdf', '.svg']:
                    files['plots'].append(file_info)
                else:
                    files['other'].append(file_info)

                # Tracks file type distribution
                files['summary']['file_types'][ext] = files['summary']['file_types'].get(ext, 0) + 1

        files['summary']['total_files'] = sum(len(f) for f in [files['models'], files['textures'], files['landmarks'], files['plots'], files['other']])
        files['summary']['total_size_mb'] = round(total_size / (1024 * 1024), 2)

        return files

    def _validate_outputs(self, files):
        """
        Validates output completeness and consistency.

        Returns:
            Dictionary with validation results and warnings
        """
        validation = {
            'status': 'success',
            'warnings': [],
            'checks': []
        }

        # Checks for expected outputs
        if len(files['models']) == 0:
            validation['warnings'].append("No 3D models found in output")
            validation['status'] = 'warning'
        else:
            validation['checks'].append(f"Found {len(files['models'])} 3D models")

        if len(files['landmarks']) == 0:
            validation['warnings'].append("No landmark files found")
            validation['status'] = 'warning'
        else:
            validation['checks'].append(f"Found {len(files['landmarks'])} landmark files")

        # Checks for atlas files
        model_names = [f['name'] for f in files['models']]
        has_atlas = any('atlas' in name.lower() for name in model_names)
        if has_atlas:
            validation['checks'].append("Atlas/template model generated")
        else:
            validation['warnings'].append("No atlas/template model found")

        # Checks for alignment consistency
        aligned_models = [f for f in files['models'] if 'align' in f['name'].lower()]
        if aligned_models:
            validation['checks'].append(f"{len(aligned_models)} aligned models found")

        return validation

    def _calculate_statistics(self, files):
        """
        Calculates statistical summaries of the outputs.

        Returns:
            Dictionary with statistical information
        """
        stats = {
            'file_counts': {
                '3D Models': len(files['models']),
                'Textures': len(files['textures']),
                'Landmarks': len(files['landmarks']),
                'Plots': len(files['plots']),
                'Other': len(files['other'])
            },
            'size_distribution': {},
            'processing_info': {
                'timestamp': self.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                'output_path': str(self.output_dir),
                'total_size_mb': files['summary']['total_size_mb']
            }
        }

        # Calculates size distribution by category
        for category in ['models', 'textures', 'landmarks']:
            if files[category]:
                sizes = [f['size_kb'] for f in files[category]]
                stats['size_distribution'][category] = {
                    'total_kb': round(sum(sizes), 2),
                    'average_kb': round(sum(sizes) / len(sizes), 2),
                    'min_kb': round(min(sizes), 2),
                    'max_kb': round(max(sizes), 2)
                }

        return stats

    def _create_comprehensive_html(self, files, parameters, validation, statistics, analysis_stats):
        """
        Creates comprehensive HTML report with professional styling.
        """
        # Builds parameter section
        param_html = self._create_parameter_section(parameters)

        # Builds validation section
        validation_html = self._create_validation_section(validation)

        # Builds statistics section
        stats_html = self._create_statistics_section(statistics)

        # Builds file details section
        files_html = self._create_file_details_section(files)

        # Builds analysis results section if provided
        analysis_html = self._create_analysis_section(analysis_stats) if analysis_stats else ""

        # Creates complete HTML
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>DeCA/InterDeCA Analysis Report - {self.timestamp.strftime("%Y-%m-%d")}</title>
            <meta charset="UTF-8">
            <style>
                * {{
                    margin: 0;
                    padding: 0;
                    box-sizing: border-box;
                }}
                body {{
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    min-height: 100vh;
                    padding: 20px;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    background: white;
                    border-radius: 20px;
                    box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                    overflow: hidden;
                }}
                .header {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 40px;
                    text-align: center;
                }}
                .header h1 {{
                    font-size: 2.5em;
                    margin-bottom: 10px;
                    font-weight: 700;
                }}
                .header .subtitle {{
                    font-size: 1.2em;
                    opacity: 0.95;
                }}
                .header .timestamp {{
                    margin-top: 20px;
                    font-size: 0.9em;
                    opacity: 0.9;
                }}
                .content {{
                    padding: 40px;
                }}
                .section {{
                    margin-bottom: 40px;
                    animation: fadeIn 0.5s ease-in;
                }}
                @keyframes fadeIn {{
                    from {{ opacity: 0; transform: translateY(10px); }}
                    to {{ opacity: 1; transform: translateY(0); }}
                }}
                h2 {{
                    color: #667eea;
                    border-bottom: 3px solid #667eea;
                    padding-bottom: 10px;
                    margin-bottom: 20px;
                    font-size: 1.8em;
                }}
                h3 {{
                    color: #764ba2;
                    margin: 20px 0 15px 0;
                    font-size: 1.3em;
                }}
                .status-card {{
                    background: #f8f9fa;
                    border-radius: 10px;
                    padding: 20px;
                    margin: 20px 0;
                    border-left: 5px solid #667eea;
                }}
                .success {{ border-left-color: #28a745; }}
                .warning {{ border-left-color: #ffc107; }}
                .error {{ border-left-color: #dc3545; }}
                .stats-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 20px;
                    margin: 20px 0;
                }}
                .stat-card {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 20px;
                    border-radius: 10px;
                    text-align: center;
                    box-shadow: 0 5px 15px rgba(0,0,0,0.1);
                    transition: transform 0.3s ease;
                }}
                .stat-card:hover {{
                    transform: translateY(-5px);
                }}
                .stat-value {{
                    font-size: 2em;
                    font-weight: bold;
                    margin: 10px 0;
                }}
                .stat-label {{
                    font-size: 0.9em;
                    opacity: 0.9;
                    text-transform: uppercase;
                    letter-spacing: 1px;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                }}
                th {{
                    background: #667eea;
                    color: white;
                    padding: 12px;
                    text-align: left;
                    font-weight: 600;
                }}
                td {{
                    padding: 10px 12px;
                    border-bottom: 1px solid #e0e0e0;
                }}
                tr:hover {{
                    background: #f5f5f5;
                }}
                tr:last-child td {{
                    border-bottom: none;
                }}
                .file-list {{
                    max-height: 400px;
                    overflow-y: auto;
                    border: 1px solid #e0e0e0;
                    border-radius: 5px;
                    padding: 10px;
                    background: #f9f9f9;
                }}
                .file-item {{
                    padding: 8px;
                    margin: 5px 0;
                    background: white;
                    border-radius: 5px;
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                }}
                .file-name {{
                    font-weight: 500;
                    color: #333;
                }}
                .file-size {{
                    font-size: 0.9em;
                    color: #666;
                    background: #f0f0f0;
                    padding: 2px 8px;
                    border-radius: 12px;
                }}
                .check-item {{
                    padding: 5px 0;
                    color: #28a745;
                }}
                .warning-item {{
                    padding: 5px 0;
                    color: #ffc107;
                }}
                .footer {{
                    background: #f8f9fa;
                    padding: 30px;
                    text-align: center;
                    color: #666;
                    border-top: 1px solid #e0e0e0;
                }}
                .badge {{
                    display: inline-block;
                    padding: 4px 12px;
                    border-radius: 20px;
                    font-size: 0.85em;
                    font-weight: 600;
                    margin: 0 5px;
                }}
                .badge-success {{ background: #d4edda; color: #155724; }}
                .badge-warning {{ background: #fff3cd; color: #856404; }}
                .badge-info {{ background: #d1ecf1; color: #0c5460; }}
                .progress-bar {{
                    width: 100%;
                    height: 30px;
                    background: #f0f0f0;
                    border-radius: 15px;
                    overflow: hidden;
                    margin: 20px 0;
                }}
                .progress-fill {{
                    height: 100%;
                    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    color: white;
                    font-weight: bold;
                    transition: width 1s ease;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>3D Fish Color Modeling Analysis Report</h1>
                    <div class="subtitle">Dense Correspondence Analysis Results</div>
                    <div class="timestamp">Generated: {self.timestamp.strftime("%B %d, %Y at %I:%M %p")}</div>
                </div>

                <div class="content">
                    {validation_html}
                    {stats_html}
                    {param_html}
                    {analysis_html}
                    {files_html}
                </div>

                <div class="footer">
                    <p><strong>3D Fish Color Modeling Pipeline</strong></p>
                    <p>DeCA/InterDeCA Analysis System v1.0</p>
                    <p style="margin-top: 10px; font-size: 0.9em;">
                        Report generated automatically •
                        <span class="badge badge-info">Research Mode</span>
                    </p>
                </div>
            </div>
        </body>
        </html>
        """

        return html

    def _create_parameter_section(self, parameters):
        """Creates the parameters section HTML."""
        if not parameters:
            return ""

        param_rows = ""
        for key, value in parameters.items():
            param_rows += f"""
                <tr>
                    <td><strong>{key}</strong></td>
                    <td>{value}</td>
                </tr>
            """

        return f"""
            <div class="section">
                <h2>Analysis Parameters</h2>
                <table>
                    {param_rows}
                </table>
            </div>
        """

    def _create_validation_section(self, validation):
        """Creates the validation section HTML."""
        status_class = validation['status']
        status_text = "Analysis Status" if status_class == "success" else "Analysis Status - Warnings"

        checks_html = "".join([f'<div class="check-item">{check}</div>' for check in validation['checks']])
        warnings_html = "".join([f'<div class="warning-item">{warning}</div>' for warning in validation['warnings']])

        return f"""
            <div class="section">
                <h2>{status_text}</h2>
                <div class="status-card {status_class}">
                    <h3>Validation Results</h3>
                    {checks_html}
                    {warnings_html if warnings_html else '<div class="check-item">All validation checks passed</div>'}
                </div>
            </div>
        """

    def _create_statistics_section(self, statistics):
        """Creates the statistics section HTML."""
        stats_cards = ""
        for label, value in statistics['file_counts'].items():
            if value > 0:
                stats_cards += f"""
                    <div class="stat-card">
                        <div class="stat-label">{label}</div>
                        <div class="stat-value">{value}</div>
                    </div>
                """

        # Creates size distribution table
        size_table = ""
        if statistics['size_distribution']:
            size_rows = ""
            for category, sizes in statistics['size_distribution'].items():
                size_rows += f"""
                    <tr>
                        <td>{category.capitalize()}</td>
                        <td>{sizes['total_kb']} KB</td>
                        <td>{sizes['average_kb']} KB</td>
                        <td>{sizes['min_kb']} - {sizes['max_kb']} KB</td>
                    </tr>
                """

            size_table = f"""
                <h3>File Size Distribution</h3>
                <table>
                    <tr>
                        <th>Category</th>
                        <th>Total Size</th>
                        <th>Average Size</th>
                        <th>Range</th>
                    </tr>
                    {size_rows}
                </table>
            """

        return f"""
            <div class="section">
                <h2>Output Statistics</h2>
                <div class="stats-grid">
                    {stats_cards}
                </div>
                {size_table}
                <div class="status-card">
                    <strong>Total Output Size:</strong> {statistics['processing_info']['total_size_mb']} MB<br>
                    <strong>Output Location:</strong> <code>{statistics['processing_info']['output_path']}</code>
                </div>
            </div>
        """

    def _create_file_details_section(self, files):
        """Creates the file details section HTML."""
        sections_html = ""

        category_info = {
            'models': ('', '3D Models', 'Mesh files containing vertex and face data'),
            'textures': ('', 'Textures', 'Color and pattern information'),
            'landmarks': ('', 'Landmarks', 'Anatomical point correspondences'),
            'plots': ('', 'Plots', 'Statistical visualizations'),
            'other': ('', 'Other Files', 'Additional analysis outputs')
        }

        for category, (emoji, title, description) in category_info.items():
            if files[category]:
                file_items = ""
                for f in sorted(files[category], key=lambda x: x['name']):
                    file_items += f"""
                        <div class="file-item">
                            <span class="file-name">{f['name']}</span>
                            <span class="file-size">{f['size_kb']} KB</span>
                        </div>
                    """

                sections_html += f"""
                    <h3>{title} ({len(files[category])} files)</h3>
                    <p style="color: #666; margin-bottom: 10px;">{description}</p>
                    <div class="file-list">
                        {file_items}
                    </div>
                """

        return f"""
            <div class="section">
                <h2>Output Files</h2>
                {sections_html}
            </div>
        """

    def _create_analysis_section(self, analysis_stats):
        """Creates optional analysis results section."""
        if not analysis_stats:
            return ""

        stats_html = ""
        for key, value in analysis_stats.items():
            stats_html += f"""
                <tr>
                    <td><strong>{key}</strong></td>
                    <td>{value}</td>
                </tr>
            """

        return f"""
            <div class="section">
                <h2>Analysis Results</h2>
                <table>
                    {stats_html}
                </table>
            </div>
        """


def test_reporter():
    """
    Comprehensive test function for team presentation.
    Creates a realistic report with sample data.
    """
    import tempfile
    from pathlib import Path

    print("="*60)
    print("RESULTS REPORTER - TEAM PRESENTATION TEST")
    print("="*60)

    # Creates realistic temp directory structure
    temp_dir = Path(tempfile.mkdtemp(prefix="deca_analysis_"))
    print(f"\nCreating test analysis output at:\n   {temp_dir}")

    # Creates realistic file structure
    print("\nGenerating sample analysis outputs...")

    # Models
    models = ['atlas_model.ply', 'fish_001_aligned.ply', 'fish_002_aligned.ply',
              'fish_003_aligned.ply', 'template_mesh.obj']
    for model in models:
        file = temp_dir / model
        file.write_text("dummy" * 1000)  # Creates files with some size

    # Textures
    textures = ['fish_001_texture.png', 'fish_002_texture.png', 'fish_003_texture.png',
                'atlas_texture.jpg']
    for texture in textures:
        file = temp_dir / texture
        file.write_text("dummy" * 500)

    # Landmarks
    landmarks = ['fish_001_landmarks.json', 'fish_002_landmarks.json',
                 'fish_003_landmarks.json', 'atlas_landmarks.mrk.json']
    for landmark in landmarks:
        file = temp_dir / landmark
        file.write_text("dummy" * 200)

    # Additional outputs
    (temp_dir / 'analysis_log.txt').write_text("Analysis completed successfully")
    (temp_dir / 'parameters.json').write_text('{"specimens": 3}')

    print("Created sample files:")
    print(f"   - {len(models)} 3D models")
    print(f"   - {len(textures)} texture files")
    print(f"   - {len(landmarks)} landmark files")

    # Creates the reporter
    print("\nGenerating comprehensive report...")
    reporter = ResultsReporter(temp_dir)

    # Defines realistic parameters
    test_params = {
        "Analysis Type": "Shape and Color Correspondence",
        "Number of Specimens": 3,
        "Template Generation": "Automatic (mean shape)",
        "Model Directory": "/data/fish_models/batch_2024",
        "Landmark Directory": "/data/fish_landmarks/batch_2024",
        "Texture Directory": "/data/fish_textures/batch_2024",
        "Point Density": "4% tolerance",
        "Color Space": "RGB",
        "Remove Scale": "Yes",
        "Error Checking": "Enabled",
        "Processing Time": "12.3 minutes"
    }

    # Defines sample analysis results
    analysis_stats = {
        "Mean Shape Variance": "0.0234 ± 0.0045",
        "Color Pattern Similarity": "87.3%",
        "Landmark Correspondence Error": "0.98 mm",
        "Successfully Processed": "3/3 specimens",
        "Atlas Quality Score": "0.95",
        "Texture Resolution": "2048x2048 pixels"
    }

    # Generates the comprehensive report
    report_path = reporter.generate_comprehensive_report(test_params, analysis_stats)

    print(f"\nREPORT GENERATED SUCCESSFULLY!")
    print(f"\nReport location:")
    print(f"   {report_path}")
    print("\n" + "="*60)
    print("TO VIEW THE REPORT:")
    print("="*60)
    print(f"1. Copy this path: {report_path}")
    print("2. Open in your web browser")
    print("\nThis report demonstrates:")
    print("   - Professional presentation format")
    print("   - Comprehensive file analysis")
    print("   - Statistical summaries")
    print("   - Validation checks")
    print("   - Clear data organization")
    print("\nPerfect for team presentations and documentation!")
    print("="*60)

    return report_path


if __name__ == "__main__":
    # Runs test when executed directly
    test_reporter()