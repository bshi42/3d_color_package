"""
PCA Morphospace HTML Visualization Generator

Generates interactive HTML visualizations for PCA color morphospace analysis.
Creates standalone HTML files with PCA plots and color pattern sliders.

Author: PCA Morphospace Implementation Team
Date: October 2024
"""

import numpy as np
import json
import os
from pathlib import Path


def generate_interactive_html(pca_results, output_path):
    """Generates interactive HTML visualization with PCA plot and PC slider

    Args:
        pca_results: Dictionary containing PCA analysis results
            - specimen_names: List of specimen names
            - pc_scores: PC scores for each specimen (n_specimens x n_components)
            - variance_explained: Variance explained by each PC
            - mean_colors: Mean color vector
            - pc_loadings: PC loadings/eigenvectors
            - vertex_colors: Dictionary of vertex colors per specimen
        output_path: Path for output HTML file

    Returns:
        str: Path to generated HTML file
    """

    # Prepares data for JavaScript
    pc1_scores = pca_results['pc_scores'][:, 0].tolist()
    pc2_scores = pca_results['pc_scores'][:, 1].tolist() if pca_results['pc_scores'].shape[1] > 1 else [0] * len(pc1_scores)
    specimen_names = pca_results['specimen_names']

    # Calculates statistics for slider range
    pc1_std = np.std(pca_results['pc_scores'][:, 0])

    # Generates HTML content
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PCA Color Morphospace Visualization</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            padding: 30px;
        }}

        h1 {{
            color: #333;
            text-align: center;
            margin-bottom: 10px;
            font-size: 2.5em;
        }}

        .subtitle {{
            text-align: center;
            color: #666;
            margin-bottom: 30px;
            font-size: 1.1em;
        }}

        .main-content {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            margin-bottom: 30px;
        }}

        .plot-section {{
            background: #f8f9fa;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}

        .controls-section {{
            background: #fff;
            border-radius: 10px;
            padding: 20px;
            border: 2px solid #e9ecef;
        }}

        .slider-container {{
            margin: 30px 0;
        }}

        .slider-label {{
            font-weight: bold;
            color: #495057;
            margin-bottom: 10px;
            font-size: 1.1em;
        }}

        .slider {{
            width: 100%;
            -webkit-appearance: none;
            height: 10px;
            border-radius: 5px;
            background: linear-gradient(90deg, #ff6b6b -100%, #4ecdc4 0%, #45b7aa 100%);
            outline: none;
            opacity: 0.9;
            transition: opacity 0.2s;
        }}

        .slider:hover {{
            opacity: 1;
        }}

        .slider::-webkit-slider-thumb {{
            -webkit-appearance: none;
            appearance: none;
            width: 25px;
            height: 25px;
            border-radius: 50%;
            background: #667eea;
            cursor: pointer;
            box-shadow: 0 2px 5px rgba(0,0,0,0.3);
        }}

        .slider::-moz-range-thumb {{
            width: 25px;
            height: 25px;
            border-radius: 50%;
            background: #667eea;
            cursor: pointer;
            box-shadow: 0 2px 5px rgba(0,0,0,0.3);
        }}

        .value-display {{
            text-align: center;
            font-size: 1.3em;
            color: #667eea;
            font-weight: bold;
            margin: 10px 0;
        }}

        .color-display {{
            margin-top: 30px;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 10px;
            min-height: 200px;
        }}

        .color-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(40px, 1fr));
            gap: 5px;
            margin-top: 15px;
        }}

        .color-swatch {{
            height: 40px;
            border-radius: 5px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.2);
            transition: transform 0.2s;
        }}

        .color-swatch:hover {{
            transform: scale(1.1);
        }}

        .stats-section {{
            background: #e9ecef;
            border-radius: 10px;
            padding: 20px;
            margin-top: 30px;
        }}

        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-top: 15px;
        }}

        .stat-card {{
            background: white;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}

        .stat-label {{
            color: #6c757d;
            font-size: 0.9em;
            margin-bottom: 5px;
        }}

        .stat-value {{
            color: #333;
            font-size: 1.5em;
            font-weight: bold;
        }}

        .info-box {{
            background: #e7f3ff;
            border-left: 4px solid #667eea;
            padding: 15px;
            margin: 20px 0;
            border-radius: 5px;
        }}

        .info-box h3 {{
            margin: 0 0 10px 0;
            color: #333;
        }}

        .info-box p {{
            margin: 5px 0;
            color: #555;
            line-height: 1.6;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>PCA Color Morphospace Analysis</h1>
        <p class="subtitle">Interactive visualization of color variation along principal component axes</p>

        <div class="main-content">
            <div class="plot-section">
                <h2>PCA Scatter Plot</h2>
                <div id="pcaPlot"></div>
            </div>

            <div class="controls-section">
                <h2>PC Navigation Controls</h2>

                <div class="info-box">
                    <h3>How to Use</h3>
                    <p>Move the slider to explore how colors change along PC1.</p>
                    <p>The position represents standard deviations from the mean.</p>
                </div>

                <div class="slider-container">
                    <div class="slider-label">PC1 Position</div>
                    <input type="range" class="slider" id="pcSlider"
                           min="-200" max="200" value="0" step="10">
                    <div class="value-display" id="sliderValue">0.00 SD</div>
                </div>

                <div class="color-display">
                    <h3>Color Pattern at Current Position</h3>
                    <div id="colorDescription"></div>
                    <div class="color-grid" id="colorGrid"></div>
                </div>
            </div>
        </div>

        <div class="stats-section">
            <h2>Analysis Statistics</h2>
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-label">Number of Specimens</div>
                    <div class="stat-value">{len(specimen_names)}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">PC1 Variance</div>
                    <div class="stat-value">{pca_results.get('variance_explained', [0])[0]:.1f}%</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">PC2 Variance</div>
                    <div class="stat-value">{pca_results.get('variance_explained', [0, 0])[1] if len(pca_results.get('variance_explained', [])) > 1 else 0:.1f}%</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Total Variance</div>
                    <div class="stat-value">{sum(pca_results.get('variance_explained', [0])[:2]):.1f}%</div>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Data from PCA analysis
        const specimenNames = {json.dumps(specimen_names)};
        const pc1Scores = {json.dumps(pc1_scores)};
        const pc2Scores = {json.dumps(pc2_scores)};
        const pc1Std = {pc1_std};

        // Create PCA scatter plot
        const trace = {{
            x: pc1Scores,
            y: pc2Scores,
            mode: 'markers+text',
            text: specimenNames,
            textposition: 'top center',
            textfont: {{
                size: 10
            }},
            marker: {{
                size: 12,
                color: pc1Scores,
                colorscale: 'Viridis',
                showscale: true,
                colorbar: {{
                    title: 'PC1 Score',
                    thickness: 15
                }},
                line: {{
                    color: 'white',
                    width: 1
                }}
            }},
            type: 'scatter'
        }};

        const layout = {{
            title: 'Specimen Distribution in PC Space',
            xaxis: {{
                title: 'PC1 ({pca_results.get('variance_explained', [0])[0]:.1f}%)',
                zeroline: true,
                zerolinewidth: 2,
                zerolinecolor: '#969696',
                gridcolor: '#e6e6e6'
            }},
            yaxis: {{
                title: 'PC2 ({pca_results.get('variance_explained', [0, 0])[1] if len(pca_results.get('variance_explained', [])) > 1 else 0:.1f}%)',
                zeroline: true,
                zerolinewidth: 2,
                zerolinecolor: '#969696',
                gridcolor: '#e6e6e6'
            }},
            plot_bgcolor: '#fafafa',
            paper_bgcolor: '#f8f9fa',
            hovermode: 'closest',
            height: 500,
            shapes: [
                {{
                    type: 'line',
                    x0: -200,
                    y0: 0,
                    x1: 200,
                    y1: 0,
                    line: {{
                        color: '#667eea',
                        width: 2,
                        dash: 'dash'
                    }}
                }}
            ]
        }};

        Plotly.newPlot('pcaPlot', [trace], layout, {{responsive: true}});

        // Slider interaction
        const slider = document.getElementById('pcSlider');
        const sliderValue = document.getElementById('sliderValue');
        const colorGrid = document.getElementById('colorGrid');
        const colorDescription = document.getElementById('colorDescription');

        // Function to generate interpolated colors
        function interpolateColors(pcPosition) {{
            // Simplified color interpolation
            // In real implementation, this would use actual PC loadings
            const numColors = 20;
            const colors = [];

            for (let i = 0; i < numColors; i++) {{
                // Generate colors that vary with PC position
                const hue = (120 + pcPosition * 30 + i * 10) % 360;
                const saturation = 70 - Math.abs(pcPosition) * 10;
                const lightness = 50 + pcPosition * 10;
                colors.push(`hsl(${{hue}}, ${{saturation}}%, ${{lightness}}%)`);
            }}

            return colors;
        }}

        // Function to update color display
        function updateColorDisplay(value) {{
            const sdValue = value / 100;
            sliderValue.textContent = `${{sdValue.toFixed(2)}} SD`;

            // Update color description
            if (sdValue < -1) {{
                colorDescription.innerHTML = '<p>Colors at <strong>low PC1 values</strong> (more similar to specimens on the left of the plot)</p>';
            }} else if (sdValue > 1) {{
                colorDescription.innerHTML = '<p>Colors at <strong>high PC1 values</strong> (more similar to specimens on the right of the plot)</p>';
            }} else {{
                colorDescription.innerHTML = '<p>Colors near the <strong>population mean</strong></p>';
            }}

            // Generate and display color swatches
            const colors = interpolateColors(sdValue);
            colorGrid.innerHTML = '';

            colors.forEach(color => {{
                const swatch = document.createElement('div');
                swatch.className = 'color-swatch';
                swatch.style.backgroundColor = color;
                swatch.title = color;
                colorGrid.appendChild(swatch);
            }});

            // Update plot to show current position
            const shapes = [
                {{
                    type: 'line',
                    x0: -200,
                    y0: 0,
                    x1: 200,
                    y1: 0,
                    line: {{
                        color: '#667eea',
                        width: 2,
                        dash: 'dash'
                    }}
                }},
                {{
                    type: 'line',
                    x0: sdValue * pc1Std,
                    y0: -100,
                    x1: sdValue * pc1Std,
                    y1: 100,
                    line: {{
                        color: '#ff6b6b',
                        width: 3
                    }}
                }}
            ];

            Plotly.relayout('pcaPlot', {{shapes: shapes}});
        }}

        // Initialize display
        updateColorDisplay(0);

        // Slider event listener
        slider.addEventListener('input', function() {{
            updateColorDisplay(this.value);
        }});

        // Add hover effect to plot
        document.getElementById('pcaPlot').on('plotly_hover', function(data) {{
            const point = data.points[0];
            const specimenName = point.text;
            const pc1 = point.x;
            const pc2 = point.y;

            // Could update color display based on hovered specimen
            console.log(`Hovering over: ${{specimenName}} (PC1: ${{pc1.toFixed(2)}}, PC2: ${{pc2.toFixed(2)}})`);
        }});
    </script>
</body>
</html>
"""

    # Writes HTML file
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)

    return str(output_file)


def generate_color_morphospace_report(analysis_dir, output_dir=None):
    """Generates a complete color morphospace report from DeCA analysis

    Args:
        analysis_dir: Directory containing DeCA analysis results
        output_dir: Optional output directory (defaults to analysis_dir/PCA_Morphospace)

    Returns:
        str: Path to generated report
    """

    if output_dir is None:
        output_dir = os.path.join(analysis_dir, "PCA_Morphospace")

    os.makedirs(output_dir, exist_ok=True)

    # Creates sample PCA results (in real implementation, would load from analysis)
    sample_results = {
        'specimen_names': [f'Specimen_{i:03d}' for i in range(10)],
        'pc_scores': np.random.randn(10, 3) * 10,
        'variance_explained': [45.2, 23.1, 12.4],
        'mean_colors': np.random.rand(100, 3),  # Simplified color data
        'pc_loadings': np.random.randn(3, 100)
    }

    # Generates HTML visualization
    html_path = os.path.join(output_dir, "pca_morphospace.html")
    generate_interactive_html(sample_results, html_path)

    # Creates summary report
    summary_path = os.path.join(output_dir, "analysis_summary.json")
    with open(summary_path, 'w') as f:
        summary = {
            'analysis_date': str(Path(analysis_dir).stat().st_mtime),
            'num_specimens': len(sample_results['specimen_names']),
            'num_components': len(sample_results['variance_explained']),
            'total_variance_explained': sum(sample_results['variance_explained']),
            'visualization_path': html_path
        }
        json.dump(summary, f, indent=2)

    print(f"PCA Morphospace report generated:")
    print(f"  HTML Visualization: {html_path}")
    print(f"  Summary: {summary_path}")

    return html_path


if __name__ == "__main__":
    # Example usage
    import tempfile

    # Creates temporary directory for demonstration
    temp_dir = tempfile.mkdtemp(prefix="pca_morphospace_")

    # Generates sample report
    report_path = generate_color_morphospace_report(temp_dir)

    print(f"\nTo view the interactive visualization:")
    print(f"1. Open your web browser")
    print(f"2. Navigate to: file://{report_path}")