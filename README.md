# 3D Fish Color Modeling - Demo Branch

## Project Overview

This demo branch contains the **InterDeCA (Interactive Dense Correspondence Analysis)** module for 3D Slicer, which extends the original DeCA module with advanced color analysis capabilities for biological specimens. This toolkit is specifically designed for morphometric analysis of fish specimens, combining shape and color pattern analysis in a unified framework.

## Table of Contents

- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Module Components](#module-components)
- [Core Functionality](#core-functionality)
- [Workflow Guide](#workflow-guide)
- [API Documentation](#api-documentation)
- [Results Reporting](#results-reporting)
- [Technical Details](#technical-details)
- [Dependencies](#dependencies)
- [Contributing](#contributing)

## Key Features

### 1. **Enhanced Dense Correspondence Analysis**
   - Automatic atlas generation from specimen sets
   - Procrustes alignment with scale normalization
   - Dense point correspondence mapping
   - Texture-aware mesh processing

### 2. **Color Pattern Analysis**
   - RGB/HSV color space analysis
   - Dimensionality reduction (PCA, t-SNE, UMAP)
   - Color clustering with K-means
   - Face-level color quantization
   - Texture baking through Blender integration

### 3. **Interactive Visualization**
   - Real-time shape interpolation
   - Multi-specimen comparison
   - Color pattern overlays
   - Statistical visualization plots

### 4. **Comprehensive Reporting**
   - Automated HTML report generation
   - Statistical summaries and validation
   - File organization tracking
   - Analysis parameter documentation

## System Architecture

```
3D Fish Color Modeling Package
├── color_deca/
│   ├── deca/                    # Original DeCA module
│   │   ├── deca.py              # Core DeCA implementation
│   │   ├── Resources/           # Module resources
│   │   └── CMakeLists.txt       # Build configuration
│   ├── deca3/                   # Enhanced InterDeCA module
│   │   ├── InterDeCA.py         # Extended DeCA with color analysis
│   │   ├── PCAMorphospace.py    # PCA color morphospace analysis (NEW)
│   │   ├── PCAMorphospaceVisualization.py  # Interactive HTML generation (NEW)
│   │   ├── Resources/           # Module resources
│   │   └── CMakeLists.txt       # Build configuration
│   └── reporting/               # Reporting utilities
│       ├── __init__.py          # Package initialization
│       ├── results_reporter.py  # HTML report generation
│       └── example_usage.py     # Working example/tutorial
└── ModelColors-main/            # Color analysis utilities
```

### Module Structure

```
InterDeCA System
├── Core Analysis Engine
│   ├── Dense Correspondence (DeCA)
│   ├── Landmarking (DeCAL)
│   └── Atlas Generation
├── Color Analysis Pipeline
│   ├── Texture Processing
│   ├── UV Mapping (via Blender)
│   ├── Color Space Transformation
│   └── Pattern Clustering
├── Visualization Components
│   ├── 3D Model Rendering
│   ├── Interpolation Controls
│   └── Statistical Plots
└── Reporting System
    ├── Results Reporter
    └── Plot Generation
```

## Installation

### Prerequisites

1. **3D Slicer** (version 5.0 or higher)
2. **Python 3.9+** (included with Slicer)
3. **Optional: Blender** (for texture baking functionality)

### Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone git@github.com:bshi42/3d_color_package.git
   cd 3d_color_package/color_deca
   git checkout demo
   ```

2. **Install in 3D Slicer**
   - Open 3D Slicer
   - Go to Edit → Application Settings → Modules
   - Add the path to `color_deca/deca3` directory
   - Restart Slicer

3. **Install Required Packages**
   
   The module will automatically detect missing packages and offer to install them.
   Required packages include:
   - `numpy` - Numerical computing (usually pre-installed)
   - `imageio` - Image I/O operations
   - `scikit-learn` - Machine learning algorithms
   - `umap-learn` - UMAP dimensionality reduction
   - `scikit-image` - Advanced image processing

   Or install manually in Slicer's Python console:
   ```python
   import slicer
   slicer.util.pip_install('imageio')
   slicer.util.pip_install('scikit-learn')
   slicer.util.pip_install('umap-learn')
   slicer.util.pip_install('scikit-image')
   ```

## Module Components

### deca.py (Original DeCA Module)

The core DeCA module (`deca/deca.py`) contains approximately 1,700 lines implementing the original Dense Correspondence Analysis functionality:

#### Main Classes

1. **`deca`** - Module definition class
   - Registers DeCA module with Slicer
   - Defines metadata and dependencies
   - Sets up help documentation

2. **`decaWidget`** - GUI widget class
   - Creates three-tab interface:
     - DeCA: Main correspondence workflow
     - DeCAL: Dense landmarking
     - Visualize Results: Output visualization
   - Manages user interactions
   - Handles file I/O operations

3. **`decaLogic`** - Core analysis logic
   - Implements atlas generation algorithms
   - Handles Procrustes alignment
   - Manages landmark subsampling
   - Performs mesh transformations

#### Key Features
- Atlas creation from specimen sets
- Dense point correspondence mapping
- Procrustes superimposition
- Landmark-based registration
- Scale normalization options
- Error checking outputs

### InterDeCA.py (Enhanced Module)

The core module file (`deca3/InterDeCA.py`) contains approximately 9,000 lines of code implementing:

#### Package Management System
- **`checkAndOfferPackageInstallation()`** - Detects missing packages
- **`installMissingPackages()`** - Automated package installation with progress tracking

#### Main Classes

1. **`InterDeCA`** - Module definition class
   - Registers module with Slicer
   - Defines metadata and dependencies
   - Sets up help documentation

2. **`InterDeCAWidget`** - GUI widget class
   - Creates multi-tab interface
   - Manages user interactions
   - Coordinates analysis workflows

3. **`InterDeCALogic`** - Analysis logic class
   - Implements core algorithms
   - Handles data processing
   - Manages file I/O operations

### User Interface Tabs

#### 1. **DeCA Tab**
- Model and landmark directory selection
- Template mesh generation
- Texture processing options
- Point density tolerance settings
- Color mode selection (RGB/HSV)
- Advanced options for scale removal and error checking

#### 2. **DeCAL Tab**
- Dense landmarking functionality
- Model selection and alignment
- Landmark propagation
- Quality control metrics

#### 3. **Visualize Tab**
- Results directory browser
- Shape interpolation controls
- Multi-model comparison
- Statistical visualization

#### 4. **Colors EDA Tab**
- Exploratory Data Analysis for colors
- Dimensionality reduction methods:
  - PCA (Principal Component Analysis)
  - t-SNE (t-distributed Stochastic Neighbor Embedding)
  - UMAP (Uniform Manifold Approximation and Projection)
  - ICA (Independent Component Analysis)
- Clustering options (K-means)
- Scatter plot visualization

#### 5. **Recolor Tab**
- Single texture application
- Color quantization (16-1024 colors)
- UV coordinate handling
- Texture baking integration

#### 6. **MultiRecolor Tab**
- Multi-texture clustering analysis
- Batch processing capabilities
- Comparative color analysis
- Pattern extraction

#### 7. **PCA Morphospace Tab** (NEW)
- Interactive PCA color morphospace visualization
- Slider-based exploration of color variation along PC axes
- Real-time color pattern interpolation from -2SD to +2SD
- HTML export with interactive plots
- Area-weighted PCA on vertex colors
- Support for RGB/HSV/LAB color spaces

## Core Functionality

### Dense Correspondence Analysis

```python
# Key functions for correspondence analysis
def generateAtlasButton():
    """Creates atlas from multiple specimens"""
    # Aligns all models
    # Generates mean shape
    # Creates dense correspondence

def alignModels(models, landmarks):
    """Procrustes alignment of specimen set"""
    # Centers models
    # Removes scale (optional)
    # Minimizes rotation differences
```

### Texture Processing Pipeline

```python
def processTextures(model, texture, uv_coords):
    """Handles texture mapping and color extraction"""
    # Maps texture to 3D surface
    # Extracts per-vertex colors
    # Transforms color spaces
```

### Blender Integration

```python
def callBlenderTextureBaking(ply_file, texture_file, output_path):
    """Executes Blender for UV mapping and texture baking"""
    # Generates Blender script
    # Calls Blender subprocess
    # Processes output textures
```

### Color Analysis Methods

```python
def performColorEDA(colors, method='pca', n_components=3):
    """Dimensionality reduction on color data"""
    # Preprocesses color values
    # Applies selected method
    # Returns transformed coordinates

def clusterColors(colors, n_clusters=5):
    """K-means clustering of color patterns"""
    # Normalizes color space
    # Performs clustering
    # Assigns cluster labels
```

## Workflow Guide

### Basic Analysis Workflow

1. **Data Preparation**
   - Organize 3D models (.ply, .obj, .stl, .vtp)
   - Prepare landmark files (.fcsv, .json)
   - Collect texture images (.png, .jpg)

2. **Atlas Generation**
   - Select model directory
   - Choose landmark directory
   - Set texture directory (optional)
   - Configure parameters:
     - Point density tolerance (0.01-1.0)
     - Color mode (RGB/HSV)
     - Scale removal option
   - Click "Generate Atlas"

3. **Results Analysis**
   - Navigate to output directory
   - Review generated files:
     - Atlas model
     - Aligned specimens
     - Correspondence maps
     - Texture outputs

4. **Visualization**
   - Load results in Visualize tab
   - Use interpolation slider
   - Compare specimens
   - Export visualizations

5. **Report Generation**
   - Automatic HTML reports created
   - Statistical summaries included
   - File organization documented

### Advanced Color Analysis Workflow

1. **Color Space Exploration**
   - Load textured models
   - Select Colors EDA tab
   - Choose analysis method
   - Configure parameters
   - Generate scatter plots

2. **Pattern Clustering**
   - Apply K-means clustering
   - Visualize cluster assignments
   - Export cluster statistics

3. **Multi-Texture Analysis**
   - Load multiple textures
   - Use MultiRecolor tab
   - Perform comparative analysis
   - Generate clustering results
   - Visualize population structure
   - - PCA finds linear relationships between colors
   - - UMAP finds non-linear relationships between colors
   - - ICA finds independent components of color variation


### PCA Color Morphospace Workflow (NEW)

The PCA Morphospace module provides interactive visualization of color variation across specimens along principal component axes, as described in morphospace analysis papers like the recolorize methodology.

1. **Data Preparation**
   - Run DeCA analysis first to generate resampled models with vertex correspondence
   - Ensure texture baking has been completed (atlasTextures folder exists)
   - All specimens must have consistent vertex topology (handled by DeCA)

2. **PCA Analysis**
   - Navigate to the PCA Morphospace tab
   - Select DeCA results directory
   - Click "Load Data for PCA Analysis"
   - Configure settings:
     - Number of principal components (2-10)
     - Color space (RGB, HSV, or LAB)
     - Area-weighted PCA option (recommended)
   - Click "Run PCA Analysis"

3. **Interactive Exploration**
   - View PCA scatter plot showing specimen distribution
   - Use PC slider to explore color variation:
     - Range: -2 SD to +2 SD along selected PC
     - Real-time color pattern updates
     - Shows interpolated colors at current position
   - Observe variance explained by each PC

4. **Export Results**
   - Generate interactive HTML visualization
   - Export PCA scores and loadings
   - Save color interpolation data
   - Create shareable reports

**Technical Implementation:**
- Extracts vertex colors from UV-mapped textures
- Performs PCA on flattened color vectors (n_vertices × 3 per specimen)
- Interpolates along PC axes: `color = mean + (sd_position × sqrt(eigenvalue) × eigenvector)`
- Generates color swatches showing variation patterns

## Results Reporting

The reporting module is located in `color_deca/reporting/` and provides comprehensive HTML report generation for analysis results.

### Quick Start

A complete working example is provided in `color_deca/reporting/example_usage.py`:

```bash
# Run the example to see how it works
python color_deca/reporting/example_usage.py
```

This example script:
- Creates sample DeCA output files
- Generates a comprehensive HTML report
- Shows proper usage of all parameters
- Demonstrates best practices

### Basic Usage

```python
# Import the reporting module
# Note: The file is results_reporter.py but we import the ResultsReporter class
from color_deca.reporting import ResultsReporter

# Alternative import methods:
# from color_deca.reporting.results_reporter import ResultsReporter
# import color_deca.reporting.results_reporter as reporter

# Create reporter instance
reporter = ResultsReporter(output_directory)

# Generate comprehensive report
report_path = reporter.generate_comprehensive_report(
    parameters=analysis_params,
    analysis_stats=results_stats
)
```

### results_reporter.py

Located in `color_deca/reporting/results_reporter.py`, this module provides:

#### Dependencies
The results_reporter module uses only Python standard library modules:
- `os` - File system operations
- `json` - JSON data handling
- `datetime` - Timestamp generation
- `pathlib` - Path manipulation
- `hashlib` - File hashing (if needed)

No external packages required - works with base Python installation!

#### Class Methods
```python
class ResultsReporter:
    def generate_comprehensive_report(parameters=None, analysis_stats=None)
    def _scan_output_files_detailed()  # File metrics
    def _validate_outputs()             # Quality checks
    def _calculate_statistics()         # Statistical summaries
    def _create_comprehensive_html()    # Professional formatting
```

**Features:**
- Professional gradient design
- File size metrics and organization
- Validation checks and warnings
- Statistical distributions
- Interactive hover effects
- UTF-8 encoding support

### Report Sections

1. **Analysis Status**
   - Validation results
   - Warning messages
   - Success indicators

2. **Output Statistics**
   - File counts by category
   - Size distributions
   - Processing metrics

3. **Analysis Parameters**
   - Input configurations
   - Method selections
   - Processing options

4. **File Details**
   - 3D Models inventory
   - Texture listings
   - Landmark files
   - Generated plots

## Technical Details

### File Formats Supported

**3D Models:**
- PLY (Polygon File Format)
- OBJ (Wavefront)
- STL (Stereolithography)
- VTP (VTK Polydata)

**Landmarks:**
- FCSV (Fiducial CSV)
- JSON (JavaScript Object Notation)
- MRK.JSON (Markups JSON)

**Textures:**
- PNG (Portable Network Graphics)
- JPG/JPEG (Joint Photographic Experts Group)
- BMP (Bitmap)

**Reports:**
- HTML (HyperText Markup Language)
- JSON (Metadata)
- CSV (Statistical exports)

### Color Space Transformations

```python
# RGB to HSV conversion
def rgb_to_hsv(r, g, b):
    return colorsys.rgb_to_hsv(r/255, g/255, b/255)

# Color quantization
def quantize_colors(colors, n_colors=256):
    kmeans = KMeans(n_clusters=n_colors)
    labels = kmeans.fit_predict(colors)
    return kmeans.cluster_centers_[labels]
```

### Performance Considerations

- **Memory Management**: Large datasets handled through chunking
- **Parallel Processing**: Multi-threaded operations where possible
- **Caching**: Results cached to avoid recomputation
- **Progress Tracking**: Visual feedback for long operations

## Dependencies

### Core Dependencies
- **3D Slicer** - Main application framework
- **VTK** - Visualization Toolkit (included with Slicer)
- **Qt** - GUI framework (included with Slicer)
- **NumPy** - Numerical computing

### Optional Dependencies
- **scikit-learn** - Machine learning algorithms
- **scikit-image** - Image processing
- **umap-learn** - UMAP algorithm
- **imageio** - Image I/O
- **Blender** - 3D modeling software (external)

### Python Version
- Requires Python 3.9+ (included with Slicer 5.0+)

## Testing

### Running Tests

```python
# In Slicer Python console
exec(open("/path/to/test_reporter.py").read())
```

### Simulation Scripts

```python
# Generate simulated results
exec(open("/path/to/simulate_results_clean.py").read())

# Create analysis plots
exec(open("/path/to/generate_analysis_plots.py").read())
```

## Troubleshooting

### Common Issues

1. **Module not appearing in Slicer**
   - Ensure path is added to module settings
   - Restart Slicer after adding path
   - Check Python console for import errors

2. **Package installation failures**
   - Run Slicer as administrator (Windows)
   - Check internet connection
   - Install packages manually via pip

3. **Blender integration issues**
   - Verify Blender installation path
   - Check Blender version compatibility
   - Review generated Blender scripts

4. **Memory errors with large datasets**
   - Reduce point density tolerance
   - Process smaller batches
   - Increase system RAM allocation

## Contributing

### Development Setup

1. Fork the repository
2. Create feature branch
3. Make changes with clear commits
4. Add tests for new functionality
5. Update documentation
6. Submit pull request

### Code Style

- Follow PEP 8 guidelines
- Use descriptive variable names
- Add comprehensive docstrings
- Include inline comments for complex logic
- Maintain consistent indentation (4 spaces)

### Documentation Style

- Use detailed descriptions for inline comments
- Write thorough docstrings for all functions
- Include parameter and return type annotations
- Provide usage examples where appropriate

## License

This project is developed with funding from:
- Georgia Institute of Technology

## Contact

**Module Authors:** Breeana Shi, Alek Spiridonov, Le Yang Loh, Charlie Clark, Alan Nadelsticher

**Repository:** https://github.com/bshi42/3d_color_package

**Branch:** demo

## Acknowledgments

- SlicerMorph team for the original DeCA implementation
- 3D Slicer community for the platform
- Contributors to scientific Python ecosystem
- Research participants and specimen providers

---

*Last Updated: October 2025*
*Version: Demo Branch 1.0*