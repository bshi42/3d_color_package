# MultiRecolor Feature Guide

## Overview

The **MultiRecolor** tab provides advanced multi-texture analysis capabilities for population studies and comparative visualization. Unlike the single-texture Recolor tab, MultiRecolor processes multiple textures simultaneously to create unified color palettes and perform population-level analysis.

## Key Features

- **Multi-texture clustering**: Process all textures together to create a unified color palette
- **Individual visualization**: Apply the unified palette to individual textures
- **Population analysis**: Compare textures using area-weighted color vectors and dimensionality reduction
- **High contrast option**: Use scientifically optimized color palettes for maximum distinguishability
- **Face area caching**: Efficient processing by caching geometric calculations

## Three-Step Workflow

### Step 1: Multi-Texture Clustering

**Purpose**: Create a unified color palette from all textures in your dataset.

**Process**:
1. Select your atlas model
2. Choose the directory containing all texture images
3. Set the number of color clusters (2-64)
4. Click "Cluster" to process all textures

**What happens**:
- Each texture is loaded and face average colors are calculated
- All face colors from all textures are combined
- MiniBatchKMeans clustering is performed in CIE Lab color space
- A unified palette of cluster centers is created

**Output**: A set of cluster centers that represent the most common colors across all textures.

### Step 2: Individual Visualization

**Purpose**: Visualize individual textures using the unified color palette.

**Process**:
1. Select a texture from the dropdown (populated after clustering)
2. Optionally enable "High Contrast Palette" for maximum color distinguishability
3. Click "Apply Texture" to visualize

**What happens**:
- The selected texture's face colors are calculated
- Each face is assigned to the nearest cluster center using ΔE2000 color distance
- The model is colored using either the cluster colors or high contrast palette

**Benefits**:
- Consistent color mapping across all textures in your dataset
- Easy comparison between different specimens
- Scientific visualization with high contrast colors

### Step 3: Population Analysis

**Purpose**: Compare all textures in a 2D visualization space.

**Process**:
1. Choose dimensionality reduction method (PCA or UMAP)
2. Click "Compare Textures" to analyze the population

**What happens**:
- Each texture is processed to create an area-weighted color vector
- Vector elements represent the proportion of each cluster color on the model
- Face areas are used as weights (larger faces contribute more)
- Vectors are normalized to unit length
- Dimensionality reduction maps vectors to 2D space
- Results are plotted with texture names as labels

**Output**: A 2D scatter plot with individual points (no connecting lines) and equal axis scaling showing relationships between textures based on their color distributions.

## UI Elements

### Step 1: Multi-Texture Clustering
- **Atlas Model**: Select the 3D model for analysis
- **Texture Directory**: Directory containing all texture image files
- **Number of Clusters**: Color clusters to create (2-64, default: 16)
- **Cluster Button**: Start the multi-texture clustering process
- **Progress Bar**: Shows clustering progress
- **Log**: Detailed feedback about the clustering process

### Step 2: Individual Visualization
- **Select Texture**: Dropdown of available textures (populated after clustering)
- **High Contrast Palette**: Use optimized colors instead of cluster colors
- **Apply Texture Button**: Apply selected texture with unified palette
- **Progress Bar**: Shows visualization progress
- **Log**: Detailed feedback about the visualization process

### Step 3: Population Analysis
- **PCA/UMAP Radio Buttons**: Choose dimensionality reduction method
- **Compare Textures Button**: Start population analysis
- **Progress Bar**: Shows analysis progress
- **Log**: Detailed feedback about the analysis process

## Technical Details

### Color Processing Pipeline
1. **Face Color Calculation**: Average RGB color computed for each face using texture coordinates
2. **Color Space Conversion**: RGB colors converted to CIE Lab for perceptually uniform processing
3. **Clustering**: MiniBatchKMeans applied to combined color data from all textures
4. **Distance Calculation**: Vectorized Euclidean distance in CIE Lab space for efficient cluster assignment
5. **Palette Application**: Cluster centers or high contrast colors applied to faces

### Area-Weighted Color Vectors
- Each texture is represented as a vector with dimensions equal to the number of clusters
- Vector elements represent the total face area assigned to each cluster color
- Vectors are normalized to unit length for fair comparison
- Face areas are cached for efficiency across multiple operations

### Dimensionality Reduction
- **PCA**: Principal Component Analysis for linear dimensionality reduction
- **UMAP**: Uniform Manifold Approximation and Projection for non-linear reduction
- Both methods reduce high-dimensional color vectors to 2D for visualization

### Performance Optimizations
- **Face Area Caching**: Areas calculated once per model and reused
- **MiniBatchKMeans**: Efficient clustering for large datasets
- **Vectorized Distance Calculations**: Fast numpy operations for cluster assignment
- **Batch Processing**: Textures processed in batches with progress reporting

## Use Cases

### Biological Studies
- **Species Comparison**: Compare coloration patterns across different species
- **Sexual Dimorphism**: Analyze color differences between males and females
- **Geographic Variation**: Study color variation across populations
- **Developmental Studies**: Track color changes over time

### Material Analysis
- **Surface Characterization**: Analyze material surface properties
- **Quality Control**: Compare manufactured items to standards
- **Wear Analysis**: Study surface changes over time
- **Texture Classification**: Group materials by surface characteristics

### Art and Cultural Studies
- **Artwork Analysis**: Compare color palettes across artworks
- **Cultural Patterns**: Study color usage in different cultures
- **Historical Analysis**: Track color trends over time periods
- **Style Classification**: Group artworks by color characteristics

## Dependencies

### Required
- **sklearn**: MiniBatchKMeans clustering, PCA dimensionality reduction
- **scikit-image**: Accurate color space conversions (RGB ↔ CIE Lab)
- **numpy**: Numerical computations and array operations
- **imageio**: Texture image loading
- **VTK**: 3D mesh processing (available in Slicer)

### Optional
- **umap-learn**: UMAP dimensionality reduction (recommended for non-linear analysis)

## Workflow Tips

### Data Preparation
- Ensure all textures are in the same format (PNG, JPG, TIFF)
- Use consistent image resolution for best results
- Verify texture coordinates are properly mapped to your atlas model

### Parameter Selection
- **Number of Clusters**: Start with 8-16 for most applications
- **Higher cluster counts** (32-64) for detailed analysis of complex textures
- **Lower cluster counts** (2-8) for broad categorization

### Analysis Strategy
1. Start with clustering to establish the unified palette
2. Use individual visualization to verify the palette works well
3. Perform population analysis to identify patterns and outliers
4. Use high contrast palette for presentations and publications

### Troubleshooting
- **Clustering fails**: Check that texture directory contains valid image files
- **No texture files found**: Verify image file extensions are supported
- **Individual visualization fails**: Ensure clustering was completed successfully
- **Population analysis fails**: Check that sklearn and required libraries are installed
- **UMAP not available**: Install umap-learn or use PCA instead

## Output Interpretation

### Individual Visualization
- Colors represent cluster assignments based on the unified palette
- Consistent coloring allows direct comparison between textures
- High contrast mode enhances visual distinction between clusters

### Population Analysis Plot
- **Proximity**: Textures close together have similar color distributions
- **Clusters**: Groups of textures with similar coloration patterns
- **Outliers**: Textures with unique color characteristics
- **Axes**: Principal components (PCA) or UMAP dimensions representing color variation
- **Equal Scaling**: Both axes use the same scale for accurate distance interpretation

### Statistical Insights
- **PCA Explained Variance**: Shows how much color variation is captured by each component
- **Cluster Separation**: Well-separated clusters indicate distinct color groups
- **Vector Magnitudes**: Normalized vectors ensure fair comparison regardless of model size

## Best Practices

1. **Consistent Atlas**: Use the same atlas model for all textures in a study
2. **Quality Control**: Verify texture mapping quality before analysis
3. **Parameter Documentation**: Record cluster numbers and methods for reproducibility
4. **Validation**: Use individual visualization to validate clustering results
5. **Multiple Methods**: Compare PCA and UMAP results for comprehensive analysis
6. **High Contrast**: Use high contrast palette for scientific presentations
7. **Caching**: Take advantage of face area caching for large datasets
