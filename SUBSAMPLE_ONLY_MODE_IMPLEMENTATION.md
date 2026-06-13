# Subsample-Only Mode Implementation

## Overview
This document describes the implementation of the "Subsample and Average Only" mode for the MultiRecolor tab in InterDeCA. This mode allows users to perform face subsampling and averaging without color clustering/quantization.

## Changes Made

### 1. UI Changes (Step 1)
**File**: `color_deca/deca3/InterDeCA.py` (Lines 1286-1361)

Added radio button selection for two modes:
- **Clustering** (default): Full clustering pipeline with color quantization
- **Subsample and Average Only**: Subsampling and face averaging without clustering

Added mode change handler `onMultiRecolorModeChanged()` that:
- Enables/disables clustering-specific controls based on selected mode
- Disables initial/consolidated cluster spinners in subsample-only mode
- Disables luminosity normalization checkbox in subsample-only mode

### 2. Step 1 Handler Modification
**File**: `color_deca/deca3/InterDeCA.py` (Lines 5305-5349)

Modified `onClusterButton()` to:
- Check which mode is selected
- Dispatch to appropriate helper method:
  - `_performClusteringMode()` for clustering mode
  - `_performSubsampleOnlyMode()` for subsample-only mode

### 3. New Helper Methods
**File**: `color_deca/deca3/InterDeCA.py` (Lines 5351-5459)

#### `_performClusteringMode()`
- Performs full clustering pipeline
- Calls `logic.performMultiTextureClustering()`
- Stores cluster centers and pipeline
- Enables Step 2/3 controls

#### `_performSubsampleOnlyMode()`
- Performs subsample-only pipeline
- Calls `logic.performSubsampleOnly()`
- Sets `multiRecolorClusterCenters = None` (no clusters in this mode)
- Stores pipeline with subsampling info
- Enables Step 2/3 controls

### 4. Logic Layer - Subsample-Only Pipeline
**File**: `color_deca/deca3/InterDeCA.py` (Lines 9809-10013)

Added `performSubsampleOnly()` method that:
- Performs face subsampling (if requested)
- Calculates face average colors
- Applies neighbor averaging (if enabled)
- Stores results in ClusteringPipeline object
- Does NOT perform clustering or color quantization
- Returns pipeline with `subsampledFaceIndices` set

### 5. Step 2 Modifications
**File**: `color_deca/deca3/InterDeCA.py` (Lines 5461-5471, 5497-5536)

#### `onIndividualTextureChanged()`
- Updated to enable apply button in subsample-only mode
- Checks: `textureSelected and (clustersAvailable or isSubsampleOnlyMode)`

#### `onApplyIndividualTextureButton()`
- Added check for subsample-only mode
- Calls `logic.applyTextureWithSubsamplingOnly()` in subsample-only mode
- Falls back to clustering mode for other cases

### 6. New Step 2 Logic Method
**File**: `color_deca/deca3/InterDeCA.py` (Lines 11134-11296)

Added `applyTextureWithSubsamplingOnly()` method that:
- Loads texture and calculates face average colors
- Applies neighbor averaging if enabled
- Uses subsampled face colors directly (no quantization)
- Propagates colors to non-subsampled faces via nearest neighbor mapping
- Applies colors to model display

### 7. Step 3 Modifications
**File**: `color_deca/deca3/InterDeCA.py` (Lines 5568-5584)

Modified `onCompareTexturesButton()` to:
- Allow population analysis in subsample-only mode
- Check for pipeline instead of requiring cluster centers
- Existing `performPopulationAnalysis()` already handles subsampling correctly

### 8. Step 4 (Morphospace)
**File**: `color_deca/deca3/InterDeCA.py` (Lines 6001-6092)

No changes needed - existing code already handles subsampling:
- `applyMorphospaceColors()` checks `use_subsampling` flag
- Calls `applyMorphospaceColorsSubsampled()` for subsampled data
- Works automatically with subsample-only mode

## Workflow

### Subsample-Only Mode Workflow

1. **Step 1**: Select "Subsample and Average Only" mode
   - Set number of subsampled faces
   - Optionally enable neighbor averaging
   - Click "Cluster" button
   - Pipeline performs subsampling and face averaging (no clustering)

2. **Step 2**: Individual Texture Visualization
   - Select texture from dropdown
   - Click "Apply Individual Texture"
   - Colors are applied using subsampled face averaging
   - No color quantization is performed

3. **Step 3**: Population Analysis
   - Select PCA or UMAP
   - Click "Compare Textures"
   - Creates color vectors from subsampled face colors
   - Performs dimensionality reduction

4. **Step 4**: Morphospace Visualization
   - Select starting texture
   - Click "Visualize Morphospace"
   - Explore PCA space using sliders
   - Colors are reconstructed from PCA inverse transform

## Key Design Decisions

1. **Pipeline Reuse**: Uses existing `ClusteringPipeline` class with clustering disabled
2. **Subsampling Info**: Stores `subsampledFaceIndices` and `nearestNeighborMapping` in pipeline
3. **No Cluster Centers**: Sets `multiRecolorClusterCenters = None` in subsample-only mode
4. **Automatic Detection**: Steps 3-4 automatically detect subsampling via pipeline properties
5. **Backward Compatibility**: Clustering mode remains unchanged and default

## Testing

Run the test script to verify implementation:
```bash
python test_subsample_only_mode.py
```

This tests:
- UI elements exist and are properly connected
- Mode switching works correctly
- Clustering controls are enabled/disabled appropriately
- Logic methods are available

## Future Enhancements

1. Add preset configurations for common subsampling levels
2. Add visualization of subsampled faces in Step 1
3. Add comparison metrics between clustering and subsample-only modes
4. Add export functionality for subsampled color data

