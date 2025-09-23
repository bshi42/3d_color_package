# 3D Fish Color Modeling Package

A comprehensive toolkit for texture-based analysis and morphometric study of 3D biological specimens, with specialized support for color pattern analysis in fish and other vertebrates.

## Overview

This package provides two main modules for 3D Slicer that enable researchers to:
- Apply and analyze textures on 3D models from photogrammetry or surface scanning
- Perform color-based segmentation and morphometric analysis
- Establish dense point correspondences across specimens
- Conduct shape and symmetry analyses with texture integration

## Components

### 1. ModelColors Module

**Purpose**: Texture application, visualization, and color-based segmentation of 3D models.

**Key Features**:
- **Texture Mapping**: Applies texture images to 3D models using UV coordinates
- **Color Data Storage**: Saves vertex colors in multiple formats for compatibility
- **Model Scaling**: Performs unit conversions (e.g., mm to meters)
- **Color Segmentation**: Uses k-means clustering to segment models by color patterns
- **Batch Processing**: Handles multiple models efficiently

**Use Cases**:
- Visualizing colored surface scans from photogrammetry
- Analyzing color patterns on biological specimens
- Segmenting anatomical regions based on coloration
- Preparing models for morphometric analysis

### 2. DeCA (Dense Correspondence Analysis) Module

**Purpose**: Establishes dense point correspondences across specimens for morphometric analysis.

**Key Features**:
- **Atlas Generation**: Creates unbiased template from specimen collection
- **Dense Landmarking (DeCAL)**: Generates thousands of corresponding points
- **Shape Analysis**: Quantifies shape variation across specimens
- **Symmetry Analysis**: Evaluates bilateral symmetry in organisms
- **Texture Integration**: Incorporates color information into correspondence analysis

**Use Cases**:
- Comparative morphology studies
- Shape variation analysis in populations
- Symmetry assessment in biological structures
- Integration of color and shape data

## Installation

### Prerequisites

- **3D Slicer** (version 5.0 or later)
- **Operating Systems**: Windows (x64), macOS (x64/ARM), Linux (x64)
- **Python packages** (installed within Slicer):
  - numpy
  - vtk (included with Slicer)
  - scikit-learn (for color segmentation)

### Installation Steps

1. **Download the repository**:
   ```bash
   git clone git@github.com:bshi42/3d_color_package.git
   ```

2. **Install ModelColors module**:
   - Open 3D Slicer
   - Go to Edit → Application Settings → Modules
   - Add path to `ModelColors-main` folder to "Additional Module Paths"
   - Restart 3D Slicer

3. **Install DeCA module**:
   - Add path to `color_deca/deca` folder to "Additional Module Paths"
   - Restart 3D Slicer

4. **Verify installation**:
   - Modules should appear under "Surface Models" (ModelColors) and "SlicerMorph.DeCA Toolbox" (DeCA)

## Quick Start Guide

### ModelColors Workflow

1. **Load your data**:
   - Import 3D model (OBJ, PLY, STL, VTK)
   - Import texture image (PNG, JPG)

2. **Apply texture**:
   - Select model and texture in the interface
   - Choose color data storage format (RGB vector recommended)
   - Click "Apply Texture"

3. **Optional - Scale model**:
   - Enter scale factor (e.g., 0.001 for mm to m)
   - Click "Apply Scale"

4. **Optional - Segment by color**:
   - Select textured model
   - Set number of color clusters (2-15)
   - Click "Segment"

### DeCA Workflow

1. **Prepare data**:
   - Organize models in one directory
   - Organize landmarks in another directory
   - Organize textures in a third directory (optional)

2. **Choose analysis type**:
   - Shape analysis: Standard morphometric comparison
   - Symmetry analysis: Bilateral symmetry assessment

3. **Run DeCA**:
   - Select directories
   - Choose atlas option (create new or load existing)
   - Configure parameters (scale removal, error checking)
   - Click "Run DeCA"

4. **Visualize results**:
   - Load result model in Visualize tab
   - Select subject to display heat maps
   - Examine shape differences or symmetry patterns

## Data Organization

### Recommended Directory Structure

```
project_folder/
├── models/           # 3D mesh files
│   ├── specimen1.ply
│   ├── specimen2.obj
│   └── ...
├── landmarks/        # Anatomical landmarks
│   ├── specimen1.mrk.json
│   ├── specimen2.fcsv
│   └── ...
├── textures/         # Texture images (optional)
│   ├── specimen1.png
│   ├── specimen2.jpg
│   └── ...
└── output/          # Analysis results
```

### File Naming Conventions

- **Corresponding files must share the same base name**:
  - Model: `fish_001.ply`
  - Landmarks: `fish_001.mrk.json`
  - Texture: `fish_001.png`

### Supported File Formats

- **3D Models**: PLY, OBJ, STL, VTK, VTP
- **Landmarks**: FCSV, MRK.JSON, JSON
- **Textures**: PNG, JPG, JPEG, BMP

## Advanced Features

### Texture-Based Analysis

The package integrates texture information into morphometric analysis:
- **Color Features**: Extracts RGB values at each vertex
- **Pattern Analysis**: Identifies color-based anatomical regions
- **Comparative Coloration**: Quantifies color differences between specimens

### Symmetry Analysis

For bilateral organisms:
- **Mirror Alignment**: Automatically mirrors and aligns specimens
- **Asymmetry Detection**: Identifies deviations from perfect symmetry
- **Landmark Reordering**: Handles bilateral landmark correspondence

### Error Checking

Optional validation outputs:
- **Correspondence Accuracy**: Evaluates point matching quality
- **Alignment Diagnostics**: Checks registration success
- **Outlier Detection**: Identifies problematic specimens

## Tutorial Resources

- **3D Slicer Extension Tutorial**: [YouTube](https://www.youtube.com/watch?v=QsxzjQb05D4)
- **DeCA Tutorial**: [GitHub](https://github.com/SlicerMorph/Tutorials/blob/main/DeCA/README.md)
- **Related Paper**: [Springer](https://link.springer.com/chapter/10.1007/978-3-031-46914-5_21)
- **2D Pattern Analysis Background**: [Ecology Letters](https://onlinelibrary.wiley.com/doi/10.1111/ele.14378)
- **Example Data**: [Dropbox](https://www.dropbox.com/scl/fo/4ck4dt6fsmd2z0ayl27oe/AMjEaTHBQGpV-zDNRRp0QhI?rlkey=69vaf4mi7y7t9kfn4c13o5fta&st=hpjox7ts&dl=0)

## Common Issues and Solutions

### Issue: Texture not displaying correctly
**Solution**: Ensure model has proper UV coordinates. Check texture orientation (may need vertical flip).

### Issue: Segmentation produces unexpected results
**Solution**: Verify model has RGB point data. Adjust cluster count based on color complexity.

### Issue: DeCA alignment fails
**Solution**: Check that landmark counts match across specimens. Ensure landmarks are in consistent order.

### Issue: Memory errors with large datasets
**Solution**: Reduce point density using spacingTolerance parameter. Process in smaller batches.

## Citation

If you use this package in your research, please cite:

```bibtex
@software{3d_color_package,
  title = {3D Fish Color Modeling Package},
  author = {[Author names]},
  year = {2025},
  url = {https://github.com/bshi42/3d_color_package}
}
```

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes with clear comments
4. Submit a pull request with description

## License

[Specify license - e.g., MIT, GPL, etc.]

## Acknowledgments

- **ModelColors** module based on SlicerIGT's Texture Model module
- **DeCA** module developed by Sara Rolfe (SCRI) with NIH funding (OD032627, HD104435)
- Principal Investigator: A. Murat Maga (Seattle Children's Research Institute)

## Contact

For questions, issues, or collaborations:
- GitHub Issues: [https://github.com/bshi42/3d_color_package/issues](https://github.com/bshi42/3d_color_package/issues)
- Email: [contact information]

## Version History

- **v1.0.0** - Initial release with basic texture and DeCA functionality
- **v1.1.0** - Added texture directory support for color-based analysis
- **[Current]** - Enhanced documentation and code comments