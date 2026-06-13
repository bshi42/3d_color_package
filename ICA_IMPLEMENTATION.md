# ICA (Independent Component Analysis) Implementation

## Overview
Added ICA as a dimensionality reduction option for Step 3 (Population Analysis) and Step 4 (Morphospace Visualization) in the MultiRecolor workflow. ICA is now available alongside PCA and UMAP.

## Changes Made

### 1. UI Changes - Step 3 Radio Button Selection
**File**: `color_deca/deca3/InterDeCA.py` (Lines 1437-1453)

Added ICA radio button to the dimensionality reduction method selection:
```python
self.icaRadioButton = qt.QRadioButton("ICA")
self.dimReductionMethodGroup.addButton(self.icaRadioButton, 2)
```

Now users can select between:
- PCA (default)
- UMAP
- ICA (new)

### 2. Step 3 Handler Updates
**File**: `color_deca/deca3/InterDeCA.py` (Lines 5591-5607)

Updated `onCompareTexturesButton()` to detect ICA selection:
```python
if self.pcaRadioButton.isChecked():
    dimReductionMethod = "PCA"
elif self.umapRadioButton.isChecked():
    dimReductionMethod = "UMAP"
elif self.icaRadioButton.isChecked():
    dimReductionMethod = "ICA"
```

### 3. Axis Selector Updates
**File**: `color_deca/deca3/InterDeCA.py` (Lines 5623-5661)

Updated axis selector logic to handle ICA:
- Uses "IC" labels for ICA components (IC1, IC2, etc.)
- Uses "PC" labels for PCA components (PC1, PC2, etc.)
- Enables axis selectors for both PCA and ICA

### 4. Logic Layer - performPopulationAnalysis
**File**: `color_deca/deca3/InterDeCA.py` (Lines 11450-11707)

#### Updated method signature and documentation
- Added "ICA" to supported methods
- Updated n_components parameter description

#### Added ICA support in dimensionality reduction
```python
elif dimReductionMethod == "ICA":
    reducer = FastICA(n_components=n_components, random_state=42, max_iter=500)
    reducedData = reducer.fit_transform(textureVectors)
```

#### Updated return values
- Returns both `"model"` (generic key) and `"pca_model"` (backward compatibility)
- Stores n_components for both PCA and ICA

### 5. Plot Creation - createPopulationPlot
**File**: `color_deca/deca3/InterDeCA.py` (Lines 4050-4080)

Updated axis label generation:
```python
if method == "PCA":
    x_label = f"PC{x_axis_idx + 1}"
    y_label = f"PC{y_axis_idx + 1}"
elif method == "ICA":
    x_label = f"IC{x_axis_idx + 1}"
    y_label = f"IC{y_axis_idx + 1}"
```

### 6. Step 4 Morphospace Updates
**File**: `color_deca/deca3/InterDeCA.py` (Lines 5958-5995, 6020-6053)

#### Updated createMorphospacePlot
- Uses appropriate component labels (IC1, IC2, etc. for ICA)
- Handles variance explained (PCA only)

#### Updated applyMorphospaceColors
- Uses generic `model` key instead of `pca_model`
- Works with both PCA and ICA inverse transforms
- Maintains backward compatibility

## Key Features

1. **Consistent Labeling**: ICA components labeled as IC1, IC2, etc. (vs PC1, PC2 for PCA)
2. **Automatic Morphospace**: Step 4 automatically works with ICA results
3. **Backward Compatible**: Existing PCA code continues to work unchanged
4. **Flexible n_components**: Users can set number of components for both PCA and ICA
5. **Inverse Transform**: Both PCA and ICA support inverse transform for morphospace visualization

## Workflow

### Using ICA in Step 3
1. Select "ICA" radio button
2. Set number of components (default: 3)
3. Click "Compare Textures"
4. Axis selectors populate with IC1, IC2, IC3, etc.

### Using ICA in Step 4
1. After Step 3 with ICA, morphospace controls are automatically enabled
2. Select starting texture
3. Click "Visualize Morphospace"
4. Sliders control IC1 and IC2 coordinates
5. Colors are reconstructed using ICA inverse transform

## Technical Details

- **ICA Algorithm**: FastICA from scikit-learn
- **Max Iterations**: 500 (configurable)
- **Random State**: 42 (for reproducibility)
- **Inverse Transform**: Uses model.inverse_transform() for both PCA and ICA
- **Component Naming**: IC for ICA, PC for PCA, Component for UMAP

## Dependencies

- scikit-learn (already required for PCA)
- FastICA is part of sklearn.decomposition

## Bug Fixes

### Fixed 1: explained_variance_ratio_ AttributeError
**Issue**: ICA doesn't have `explained_variance_ratio_` attribute like PCA does, causing AttributeError when trying to access it.

**Solution**: Added method checks to only access `explained_variance_ratio_` for PCA:
- Line 5271: Check `if method == "PCA"` before accessing explained_variance_ratio_
- Line 5673: Check `if method == "PCA"` before accessing explained_variance_ratio_
- Line 5937: Check `if method == "PCA"` before accessing explained_variance_ratio_
- Line 11680: Removed from ICA branch, only logs n_iter_ for ICA

**Key Difference**:
- **PCA**: Has `explained_variance_ratio_` (variance explained by each component)
- **ICA**: Has `n_iter_` (number of iterations to convergence)
- **Both**: Have `inverse_transform()` method for morphospace visualization

### Fixed 2: Axis Selection Plot Not Updating for ICA
**Issue**: When changing IC axis selection (X/Y), the plot didn't update because the code only updated for PCA.

**Solution**: Updated `onMultiRecolorAxisChanged()` method (Line 5261):
- Changed from: `if result.get("method") != "PCA": return`
- Changed to: `if result.get("method") not in ["PCA", "ICA"]: return`

Now plot updates when changing axes for both PCA and ICA.

### Fixed 3: Morphospace Visualize Button Disabled for ICA
**Issue**: The "Visualize Morphospace" button in Step 4 was disabled when using ICA because it only checked for PCA.

**Solution**: Updated `onMorphospaceTextureChanged()` method (Line 5711):
- Changed from: `hasPCA = ... and self.multiRecolorPopulationResult.get("method") == "PCA"`
- Changed to: `hasReducer = ... and self.multiRecolorPopulationResult.get("method") in ["PCA", "ICA"]`
- Updated button enable logic to use `hasReducer` instead of `hasPCA`

Now the Visualize button is enabled for both PCA and ICA results.

### Fixed 4: Morphospace Visualization Rejected ICA Results
**Issue**: When clicking "Visualize Morphospace" with ICA results, got error: "Error: Run PCA population analysis first (Step 3)"

**Solution**: Updated `onVisualizeMorphospace()` method (Line 5800):
- Changed from: `if not self.multiRecolorPopulationResult or self.multiRecolorPopulationResult.get("method") != "PCA":`
- Changed to: `if not self.multiRecolorPopulationResult or self.multiRecolorPopulationResult.get("method") not in ["PCA", "ICA"]:`
- Updated error message to: "Error: Run PCA or ICA population analysis first (Step 3)"

Now morphospace visualization works with both PCA and ICA results.

## Testing

To test ICA functionality:
1. Load textures in MultiRecolor
2. Run Step 1 (clustering or subsample-only)
3. In Step 3, select ICA radio button
4. Set n_components to 3
5. Click "Compare Textures"
6. Verify axis labels show IC1, IC2, IC3
7. **Change axis selection** - verify plot updates with new axes
8. In Step 4, select a starting texture
9. **Click "Visualize Morphospace"** - button should be enabled and visualization should start
10. Verify morphospace works with ICA coordinates
11. Check log output shows "ICA computed 3 independent components" and convergence info

## Summary of All Fixes

| Issue | Location | Status |
|-------|----------|--------|
| AttributeError on `explained_variance_ratio_` | Lines 5271, 5673, 5937, 11680 | ✅ Fixed |
| Plot not updating on axis change | Line 5261 | ✅ Fixed |
| Visualize button disabled | Line 5711 | ✅ Fixed |
| Morphospace rejected ICA results | Line 5800 | ✅ Fixed |
| UMAP not supported in morphospace | Lines 5261, 5632, 5711, 5800, 11728 | ✅ Fixed |

## UMAP Support Added

UMAP is now fully supported for morphospace visualization:

**Changes Made:**
1. **Line 5261**: Updated axis change handler to include UMAP
2. **Line 5632**: Updated axis selector population to include UMAP with "UMAP1", "UMAP2" labels
3. **Line 5711**: Updated morphospace button enable check to include UMAP
4. **Line 5800**: Updated morphospace visualization check to include UMAP
5. **Line 11728**: Added UMAP model storage in population analysis results

**UMAP Morphospace Workflow:**
1. Step 3: Select UMAP radio button → Compare Textures
2. Plot displays with UMAP1, UMAP2 labels (UMAP always uses 2 components)
3. Step 4: Select texture → Visualize button enabled
4. Click Visualize → Morphospace visualization with UMAP coordinates
5. Sliders control UMAP1 and UMAP2 → Colors reconstruct via UMAP inverse transform

**All dimensionality reduction methods now fully operational!** 🎉
- ✅ PCA with morphospace
- ✅ ICA with morphospace
- ✅ UMAP with morphospace

