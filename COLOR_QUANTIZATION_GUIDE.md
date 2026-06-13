# Color Quantization Feature Guide

## Overview

The InterDeCA module now includes color quantization functionality in the Recolor tab. This feature allows you to reduce the number of colors in your textured models using k-means clustering in CIE Lab color space with ΔE2000 distance calculations for perceptually accurate color grouping.

## New UI Elements

### Quantize Colors Checkbox
- **Location**: Recolor tab, below "Average Face Colors" checkbox
- **Function**: Enables/disables color quantization
- **Behavior**: Only enabled when "Average Face Colors" is checked
- **Default**: Unchecked (disabled)

### Number of Colors Spinner
- **Location**: Recolor tab, below "Quantize Colors" checkbox
- **Function**: Sets the number of color clusters for quantization
- **Range**: 2-64 colors
- **Default**: 8 colors
- **Behavior**: Only enabled when both "Average Face Colors" and "Quantize Colors" are checked

### High Contrast Palette Checkbox
- **Location**: Recolor tab, below "Number of Colors" spinner
- **Function**: Uses a high contrast color palette instead of quantized colors from the texture
- **Purpose**: Provides maximum color distinguishability for analysis and visualization
- **Default**: Unchecked (uses quantized colors)
- **Behavior**: Only enabled when both "Average Face Colors" and "Quantize Colors" are checked

## How It Works

### Technical Implementation

1. **Face Color Calculation**: First calculates the average color for each face of the mesh using the texture coordinates
2. **Color Space Conversion**: Converts RGB colors to CIE Lab color space for perceptually uniform clustering
3. **K-means Clustering**: Groups similar colors using k-means clustering in Lab space
4. **ΔE2000 Distance**: Uses the industry-standard ΔE2000 formula for accurate color difference calculations
5. **Color Application**: Applies the quantized colors back to the mesh faces

### Color Spaces Used

- **Input**: RGB color space (0-255 range)
- **Processing**: CIE Lab color space (perceptually uniform)
- **Distance Metric**: ΔE2000 (most accurate color difference formula)
- **Output**: RGB color space applied to mesh faces

## Usage Instructions

### Basic Workflow

1. **Load Model**: Import your atlas model into 3D Slicer
2. **Open InterDeCA**: Navigate to the InterDeCA module
3. **Select Recolor Tab**: Click on the "Recolor" tab
4. **Choose Atlas Model**: Select your model from the dropdown
5. **Select Texture Directory**: Browse to folder containing texture images
6. **Pick Texture**: Choose a specific texture from the dropdown
7. **Enable Averaging**: Check "Average Face Colors" checkbox
8. **Enable Quantization**: Check "Quantize Colors" checkbox (now enabled)
9. **Set Color Count**: Choose number of colors (2-64) using the spinner
10. **(Optional) High Contrast**: Check "High Contrast Palette" for maximum color distinguishability
11. **Apply**: Click "Apply Texture" to render the quantized model

### UI Behavior

- **Quantization controls are disabled** until "Average Face Colors" is checked
- **Number of colors spinner is disabled** until both averaging and quantization are enabled
- **High contrast palette checkbox is disabled** until both averaging and quantization are enabled
- **Disabling averaging** automatically disables and unchecks quantization and high contrast palette
- **Progress bar** shows quantization progress during processing
- **Log messages** provide detailed feedback about the quantization process

### Color Palette Options

#### Quantized Colors (Default)
- Uses the actual colors from the texture after k-means clustering
- Preserves the original color relationships and aesthetic
- Best for artistic and realistic applications
- Colors are derived from the texture content

#### High Contrast Palette
- Uses a predefined set of maximally distinguishable colors
- Optimized for visual analysis and data interpretation
- Best for scientific visualization and color-based analysis
- Colors are chosen for maximum perceptual difference
- Supports up to 64 distinct colors with optimal spacing in color space

## Use Cases

### Artistic Stylization
- Create posterized or cartoon-like effects
- Reduce color complexity for artistic purposes
- Generate consistent color palettes across models

### Data Analysis
- Identify dominant colors in textured models
- Simplify color data for statistical analysis
- Create color-based model classifications

### Performance Optimization
- Reduce color complexity for faster rendering
- Simplify textures for lower-end hardware
- Create level-of-detail (LOD) versions

## Technical Details

### Color Quantization Algorithm

```python
# Simplified workflow:
1. Calculate average RGB color for each face
2. Convert RGB → CIE Lab color space (using scikit-image)
3. Apply k-means clustering in Lab space
4. Convert cluster centers Lab → RGB (using scikit-image)
5. Map each face to its cluster center color
6. Apply quantized colors to mesh
```

### Supported Features

- **Color Spaces**: RGB input/output, CIE Lab processing
- **Clustering**: K-means with configurable cluster count (2-64)
- **Distance Metric**: ΔE2000 for perceptually accurate clustering
- **Progress Tracking**: Real-time progress updates
- **Error Handling**: Comprehensive error checking and reporting
- **Integration**: Seamless integration with existing recolor workflow

### Dependencies

- **sklearn**: Required for k-means clustering
- **scikit-image**: Required for accurate color space conversions and ΔE2000 calculations
- **numpy**: Required for numerical computations
- **imageio**: Required for texture loading
- **VTK**: Required for mesh processing (already available in Slicer)

## Troubleshooting

### Common Issues

1. **"sklearn not available" error**
   - Solution: Ensure sklearn is installed in the Slicer Python environment

2. **"scikit-image not available" error**
   - Solution: Ensure scikit-image is installed in the Slicer Python environment

3. **Quantization controls disabled**
   - Solution: First check "Average Face Colors" checkbox

4. **No visible color change**
   - Solution: Try a lower number of clusters or check that the texture has sufficient color variation

5. **Performance issues with large models**
   - Solution: Use fewer clusters or consider model simplification

### Performance Tips

- **Start with fewer clusters** (4-8) for faster processing
- **Use smaller textures** for initial testing
- **Monitor progress bar** for processing status
- **Check log messages** for detailed feedback

## Examples

### Example 1: Simple Quantization
- Model: Fish atlas with colorful texture
- Settings: Average Face Colors ✓, Quantize Colors ✓, 6 colors
- Result: Fish rendered with 6 dominant colors

### Example 2: High Contrast Posterization  
- Model: Human face with skin texture
- Settings: Average Face Colors ✓, Quantize Colors ✓, 3 colors
- Result: High-contrast, poster-like appearance

### Example 3: Color Palette Extraction
- Model: Landscape model with varied textures
- Settings: Average Face Colors ✓, Quantize Colors ✓, 12 colors
- Result: Model showing 12 most representative colors

## Future Enhancements

Potential future improvements could include:
- Additional clustering algorithms (hierarchical, DBSCAN)
- Custom color palette specification
- Color harmony analysis
- Export of extracted color palettes
- Real-time preview of quantization results

## Testing

Use the provided test script (`test_color_quantization.py`) to verify functionality:

```python
# In 3D Slicer Python console:
exec(open('test_color_quantization.py').read())
run_all_tests()
```

This will test UI elements, color space conversions, quantization algorithms, and integration with the existing workflow.
