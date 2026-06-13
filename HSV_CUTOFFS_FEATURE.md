# HSV Cutoffs Feature Documentation

## Overview

The HSV Cutoffs feature allows users to filter low-saturation and low-value (brightness) colors when working in HSV color space mode in the Colors EDA tab. This filtering helps focus analysis on more vibrant and visible colors while maintaining the complete dataset for other analyses.

## Key Features

### 🎯 **Selective Filtering**
- **Dimensionality Reduction**: Cutoffs filter data going into PCA/ICA/UMAP analysis
- **Hue Histograms**: Cutoffs filter data for hue histogram display
- **Other Histograms**: Saturation and Value histograms use the full dataset (no filtering)
- **Full Dataset Preservation**: Complete color data is always maintained internally

### 🎛️ **User Controls**
- **Saturation Cutoff**: Minimum saturation threshold (0-100%)
- **Value Cutoff**: Minimum value/brightness threshold (0-100%)
- **Real-time Updates**: Changes to cutoffs immediately update histograms
- **Tooltips**: Helpful descriptions for each control

## How It Works

### Data Flow Architecture

```
Texture Analysis
       ↓
Full Face Color Dataset (preserved)
       ↓
   HSV Mode?
       ↓
    ┌─ YES ─┐                    ┌─ NO ──┐
    ↓       ↓                    ↓       ↓
Apply       Use Full         Use Full   Use Full
Cutoffs     Dataset          Dataset    Dataset
    ↓       ↓                    ↓       ↓
Filtered    Sat/Val          Dim Red    Histograms
Dataset     Histograms
    ↓
┌───────────────┐
│ Dim Reduction │
│ Hue Histogram │
└───────────────┘
```

### HSV Color Space Representation

The system uses a 4D representation for HSV colors to handle the circular nature of hue:
- **Hue**: Represented as 2D vector `[cos(hue), sin(hue)]` to handle 0°/360° wraparound
- **Saturation**: Linear scale 0-100%
- **Value**: Linear scale 0-100%

**Example**: Red color (RGB: 255,0,0)
- Hue: 0° → Vector: [1.0, 0.0]
- Saturation: 100%
- Value: 100%
- Final vector: [1.0, 0.0, 100.0, 100.0]

## User Interface

### Location
Colors EDA Tab → Histogram Options Section

### Controls
1. **HSV Filtering Label**: Indicates which analyses are affected
2. **Saturation Cutoff**: Spinner (0-100%, default: 10%)
3. **Value Cutoff**: Spinner (0-100%, default: 10%)
4. **Auto-refresh**: Changes immediately update current histogram view

### Visual Feedback
- Log messages show filtering statistics during analysis
- Example: "HSV filtering: 8,234/10,000 samples passed cutoffs (sat>=25, val>=30)"

## Usage Examples

### Example 1: Standard Analysis
```
Settings:
- Color Space: HSV
- Saturation Cutoff: 10%
- Value Cutoff: 10%
- Algorithm: PCA

Result:
- Filters out very gray/dark colors
- Focuses analysis on more vibrant colors
- Retains ~85-90% of typical data
```

### Example 2: High-Contrast Analysis
```
Settings:
- Color Space: HSV
- Saturation Cutoff: 50%
- Value Cutoff: 50%
- Algorithm: UMAP

Result:
- Only analyzes highly saturated, bright colors
- Extreme filtering for specific research questions
- May retain only ~30-50% of data
```

### Example 3: Conservative Filtering
```
Settings:
- Color Space: HSV
- Saturation Cutoff: 5%
- Value Cutoff: 5%
- Algorithm: ICA

Result:
- Minimal filtering, removes only very dull colors
- Retains ~95-98% of data
- Good for comprehensive analysis
```

## Technical Implementation

### Filtering Logic
```python
# Extract HSV channels
sat = colorData[:, 2]  # Saturation (0-100)
val = colorData[:, 3]  # Value (0-100)

# Create filter mask
mask = (sat >= satCutoff) & (val >= valueCutoff)

# Apply to dimensionality reduction
filteredData = colorData[mask]
reducedData = applyDimensionalityReduction(filteredData)
```

### Data Preservation
- **Full Dataset**: Always stored in `_lastColorData`
- **Filtered Dataset**: Used only for specific analyses
- **Histogram Data**: Uses appropriate dataset based on histogram type

## Benefits

### 🔬 **Scientific Advantages**
- **Noise Reduction**: Eliminates low-quality color samples
- **Focus on Biological Variation**: Emphasizes meaningful color differences
- **Improved Clustering**: Better separation in dimensionality reduction
- **Reduced Computational Load**: Fewer samples for complex algorithms

### 👥 **User Experience**
- **Intuitive Controls**: Simple percentage-based cutoffs
- **Real-time Feedback**: Immediate visual updates
- **Flexible Analysis**: Easy to adjust filtering strength
- **Preserved Data**: Can always return to full dataset

## Best Practices

### Recommended Cutoff Values
- **Conservative**: Sat ≥ 5%, Val ≥ 5% (minimal filtering)
- **Standard**: Sat ≥ 10%, Val ≥ 10% (default, good balance)
- **Moderate**: Sat ≥ 25%, Val ≥ 25% (focus on vibrant colors)
- **Aggressive**: Sat ≥ 50%, Val ≥ 50% (only highly saturated colors)

### When to Use Different Settings
- **Biological Specimens**: Standard settings work well
- **Artificial/Painted Surfaces**: May need higher cutoffs
- **Low-Light Images**: Lower cutoffs to retain data
- **High-Contrast Studies**: Higher cutoffs for specificity

### Workflow Recommendations
1. Start with default cutoffs (10%, 10%)
2. Run initial analysis to see data distribution
3. Adjust cutoffs based on research questions
4. Compare results with different cutoff values
5. Document cutoff values used in analysis

## Troubleshooting

### Common Issues
- **Too Few Samples**: Reduce cutoff values
- **No Filtering Effect**: Increase cutoff values
- **Unexpected Results**: Check color space mode (RGB vs HSV)
- **Performance Issues**: Higher cutoffs reduce computational load

### Validation
- Check log messages for filtering statistics
- Compare histogram distributions with/without filtering
- Verify dimensionality reduction plots show expected patterns
- Test with known color samples

This feature provides powerful tools for focused color analysis while maintaining the flexibility to work with complete datasets when needed.
