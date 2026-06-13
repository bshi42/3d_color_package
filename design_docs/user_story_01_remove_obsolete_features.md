# InterDeCA Tab Refactoring Story

## Overview
Streamline the InterDeCA module by removing redundant and experimental tabs (Colors EDA, Recolor, PCA Morphospace, Visualize Results) and consolidating functionality into a focused, cohesive workflow. This refactoring improves user experience by eliminating duplicate features and creating a cleaner interface.

## Reasoning

### User Story
As a **researcher using InterDeCA**, I want to **remove the experimental and redundant tabs (Colors EDA, Recolor, PCA Morphospace, Visualize Results)** so that **I have a focused, production-ready workflow without redundant features that were only used during development**.

### Background
During development, we experimented with multiple approaches to color analysis and texture visualization:
- Colors EDA tab: Early exploratory analysis approach
- Recolor tab: Initial single-texture application method
- PCA Morphospace tab: Standalone morphospace visualization
- Visualize Results tab: Interpolation visualization (redundant with MultiRecolor Step 4)

These experimental features have been superseded by the more comprehensive MultiRecolor workflow (Steps 1-4), which consolidates all functionality into a unified pipeline. Keeping these obsolete tabs creates:
- **Code Maintenance Burden**: Multiple code paths doing similar things
- **User Confusion**: 7 tabs with overlapping functionality
- **Testing Complexity**: More features to test and maintain
- **Unclear Workflow**: Users unsure which tab to use for their analysis

### Benefits of Consolidation
- **Single Source of Truth**: MultiRecolor Steps 1-4 provide complete analysis pipeline
- **Reduced Maintenance**: Fewer code paths, clearer responsibility
- **Improved UX**: Linear workflow progression (Cluster → Visualize → Analyze → Explore)
- **Production Ready**: Remove experimental code, keep proven functionality
- **Preserved Capabilities**: All features retained in MultiRecolor; nothing lost

## Acceptance Criteria

### AC1: Colors EDA Tab Removal

**Scenario 1: Colors EDA tab is no longer visible in the UI**
- **Given** the InterDeCA module is loaded
- **When** I look at the tab widget
- **Then** the "Colors EDA" tab should not exist
- **And** the tab count should be 4 (ATLAS, Mesh Selection, Population Analysis, Clustering View)

**Scenario 2: Colors EDA UI components are removed**
- **Given** the InterDeCA module is loaded
- **When** I inspect the widget attributes
- **Then** `self.colorsEDATab` should not exist
- **And** `self.sampleDataButton` should not exist
- **And** `self.plotButton` should not exist
- **And** `self.bakedTexturesDirectorySelector` should not exist

**Scenario 3: Colors EDA event handlers are removed**
- **Given** the InterDeCA module is loaded
- **When** I search for Colors EDA methods
- **Then** `onSampleDataButton()` should not exist
- **And** `onPlotButton()` should not exist
- **And** `onViewModeChanged()` should not exist
- **And** `onHistChannelChanged()` should not exist

### AC2: Recolor Tab Removal

**Scenario 1: Recolor tab is no longer visible in the UI**
- **Given** the InterDeCA module is loaded
- **When** I look at the tab widget
- **Then** the "Recolor" tab should not exist

**Scenario 2: Recolor UI components are removed**
- **Given** the InterDeCA module is loaded
- **When** I inspect the widget attributes
- **Then** `self.recolorTab` should not exist
- **And** `self.recolorAtlasModelSelect` should not exist
- **And** `self.applyRecolorButton` should not exist
- **And** `self.quantizeColorsCheckbox` should not exist

**Scenario 3: Recolor event handlers are removed**
- **Given** the InterDeCA module is loaded
- **When** I search for Recolor methods
- **Then** `onRecolorParameterChanged()` should not exist
- **And** `onApplyRecolorButton()` should not exist
- **And** `onAverageFaceColorToggled()` should not exist

### AC3: PCA Morphospace Tab Removal

**Scenario 1: PCA Morphospace tab is no longer instantiated**
- **Given** the InterDeCA module is loaded
- **When** I look at the tab widget
- **Then** the "PCA Morphospace" tab should not exist

**Scenario 2: PCA Morphospace instantiation code is removed**
- **Given** I inspect the `setup()` method
- **When** I search for PCAMorphospace instantiation
- **Then** the `if PCA_MORPHOSPACE_AVAILABLE:` block should not instantiate a tab
- **And** `pcaMorphospaceWidget` should not be added to `tabsWidget`

### AC4: Visualize Results Tab Removal

**Scenario 1: Visualize Results tab is no longer visible in the UI**
- **Given** the InterDeCA module is loaded
- **When** I look at the tab widget
- **Then** the "Visualize Results" tab should not exist

**Scenario 2: Visualize Results UI components are removed**
- **Given** the InterDeCA module is loaded
- **When** I inspect the widget attributes
- **Then** `self.visualizeResultsTab` should not exist
- **And** `self.interpolationSliders` should not exist
- **And** `self.interpolationButton` should not exist
- **And** all interpolation-related controls should not exist

**Scenario 3: Visualize Results event handlers are removed**
- **Given** the InterDeCA module is loaded
- **When** I search for Visualize Results methods
- **Then** `onInterpolationSliderChanged()` should not exist
- **And** `onInterpolateButton()` should not exist
- **And** `onVisualizationParameterChanged()` should not exist

### AC5: MultiRecolor Tab Renamed to Population Analysis

**Scenario 1: MultiRecolor tab is renamed to Population Analysis**
- **Given** the InterDeCA module is loaded
- **When** I look at the tab widget
- **Then** a tab labeled "Population Analysis" should exist
- **And** the "MultiRecolor" tab should not exist

**Scenario 2: All MultiRecolor functionality remains intact**
- **Given** the "Population Analysis" tab is active
- **When** I inspect the tab contents
- **Then** Step 1 (Multi-Texture Clustering) section should exist
- **And** Step 2 (Individual Visualization) section should exist
- **And** Step 3 (Population Analysis) section should exist
- **And** Step 4 (Morphospace) section should exist
- **And** all controls should be functional

### AC6: New Clustering View Tab Created

**Scenario 1: Clustering View tab appears as the last tab**
- **Given** the InterDeCA module is loaded
- **When** I look at the tab widget
- **Then** a tab labeled "Clustering View" should exist
- **And** it should be the last tab (after Population Analysis)

**Scenario 2: Clustering View code is in separate module**
- **Given** I inspect the codebase
- **When** I look for the Clustering View implementation
- **Then** `color_deca/deca3/clustering_view/ClusteringView.py` should exist
- **And** it should contain a `ClusteringViewWidget` class
- **And** it should contain a `ClusteringViewLogic` class
- **And** it should follow the same pattern as PCAMorphospace module

**Scenario 3: Clustering View tab is empty and ready for future implementation**
- **Given** I switch to the Clustering View tab
- **When** the tab is displayed
- **Then** the tab should be empty (placeholder for future functionality)
- **And** the tab should not cause any errors

### AC7: Code Cleanup and Simplification

**Scenario 1: Colors EDA state variables are removed**
- **Given** I inspect the `InterDeCAWidget` class
- **When** I search for Colors EDA state variables
- **Then** `self.sampledColorData` should not exist
- **And** `self.sampledSpecimenNames` should not exist
- **And** `self.sampledFaceIndices` should not exist

**Scenario 2: Visualize Results state variables are removed**
- **Given** I inspect the `InterDeCAWidget` class
- **When** I search for Visualize Results state variables
- **Then** `self.interpolationData` should not exist
- **And** `self.interpolationWeights` should not exist
- **And** all interpolation-related state variables should not exist

**Scenario 3: Texture directory syncing is simplified**
- **Given** I inspect the texture directory handling
- **When** I look for `syncTextureDirectories()` method
- **Then** `syncTextureDirectories()` should not exist
- **And** texture directory syncing should only apply to MultiRecolor
- **And** Colors EDA, Recolor, and Visualize Results directory syncing should be removed

**Scenario 4: Obsolete imports are removed**
- **Given** I inspect the imports in `InterDeCA.py`
- **When** I search for unused imports
- **Then** imports only used by removed tabs should be removed
- **And** core functionality imports should remain

### AC8: Module Loads and Functions Correctly

**Scenario 1: Module loads without errors**
- **Given** I load the InterDeCA module in 3D Slicer
- **When** the module initializes
- **Then** no errors should appear in the console
- **And** no warnings about missing tabs should appear

**Scenario 2: All remaining tabs render correctly**
- **Given** the InterDeCA module is loaded
- **When** I click through each tab
- **Then** ATLAS tab should render without errors
- **And** Mesh Selection tab should render without errors
- **And** Population Analysis tab should render without errors
- **And** Clustering View tab should render without errors

**Scenario 3: No orphaned references to removed tabs**
- **Given** I inspect the codebase
- **When** I search for references to removed tabs
- **Then** no code should reference "Colors EDA" tab
- **And** no code should reference "Recolor" tab
- **And** no code should reference "PCA Morphospace" tab
- **And** no code should reference "Visualize Results" tab
- **And** no event handlers should try to access removed widgets

### AC9: MultiRecolor Workflow Remains Functional

**Scenario 1: MultiRecolor Step 1 (Clustering) works end-to-end**
- **Given** I have test data with multiple textures
- **When** I run Step 1 (Clustering)
- **Then** clustering should complete without errors
- **And** cluster centers should be computed
- **And** results should be available for Step 2

**Scenario 2: MultiRecolor Step 2 (Individual Visualization) works**
- **Given** Step 1 (Clustering) has completed
- **When** I select a texture and apply it in Step 2
- **Then** the texture should be applied to the model
- **And** colors should be quantized using cluster centers
- **And** the visualization should display correctly

**Scenario 3: MultiRecolor Step 3 (Population Analysis) works**
- **Given** Step 1 (Clustering) has completed
- **When** I run Step 3 (Population Analysis)
- **Then** dimensionality reduction should execute
- **And** a population comparison plot should be generated
- **And** results should feed into Step 4

**Scenario 4: MultiRecolor Step 4 (Morphospace) works**
- **Given** Step 3 (Population Analysis) has completed
- **When** I interact with morphospace sliders
- **Then** the model should update with interpolated colors
- **And** the visualization should respond to slider changes

## Implementation Details

### Phase 1: Remove Colors EDA Tab
**File**: `color_deca/deca3/InterDeCA.py`

1. **UI Removal** (lines ~399-1170):
   - Delete `colorsEDATab` widget creation
   - Delete `colorsEDATabLayout` and all child widgets
   - Delete `tabsWidget.addTab(colorsEDATab, "Colors EDA")`
   - Remove Phase 1 (Data Sampling) section
   - Remove Phase 2 (Analysis & Plotting) section
   - Remove histogram controls section

2. **Event Handlers** (search for `onSample`, `onPlot`, `onViewMode`, `onHist`):
   - Delete `onSampleDataButton()`
   - Delete `onPlotButton()`
   - Delete `onViewModeChanged()`
   - Delete `onHistChannelChanged()`
   - Delete `onSamplingParameterChanged()`

3. **State Variables**:
   - Delete `self.sampledColorData`, `self.sampledSpecimenNames`, `self.sampledFaceIndices`
   - Delete histogram-related state variables

### Phase 2: Remove Recolor Tab
**File**: `color_deca/deca3/InterDeCA.py`

1. **UI Removal** (lines ~1171-1268):
   - Delete `recolorTab` widget creation
   - Delete `recolorTabLayout` and all child widgets
   - Delete `tabsWidget.addTab(recolorTab, "Recolor")`

2. **Event Handlers** (search for `onRecolor`, `onAverage`, `onQuantize`):
   - Delete `onRecolorParameterChanged()`
   - Delete `onRecolorTexturesDirectoryChanged()`
   - Delete `onAverageFaceColorToggled()`
   - Delete `onQuantizeColorsToggled()`
   - Delete `onApplyRecolorButton()`

3. **Logic Methods**:
   - Delete `applyAverageFaceColorsFromTexture()` (move to MultiRecolor if needed)
   - Delete `applyAverageFaceColorsFromTextureAlternative()`

### Phase 3: Remove PCA Morphospace Tab
**File**: `color_deca/deca3/InterDeCA.py`

1. **Tab Instantiation** (lines ~414-423):
   - Delete PCAMorphospace import and availability check
   - Delete `if PCA_MORPHOSPACE_AVAILABLE:` block
   - Delete `pcaMorphospaceWidget` creation and tab addition

2. **Imports**:
   - Keep PCAMorphospace import for reference but don't instantiate

### Phase 4: Remove Visualize Results Tab
**File**: `color_deca/deca3/InterDeCA.py`

1. **UI Removal**:
   - Delete `visualizeResultsTab` widget creation
   - Delete `visualizeResultsTabLayout` and all child widgets
   - Delete `tabsWidget.addTab(visualizeResultsTab, "Visualize Results")`
   - Remove all interpolation slider controls
   - Remove interpolation button and related controls

2. **Event Handlers** (search for `onInterpolation`, `onVisualization`):
   - Delete `onInterpolationSliderChanged()`
   - Delete `onInterpolateButton()`
   - Delete `onVisualizationParameterChanged()`
   - Delete `onInterpolationModeChanged()`

3. **Logic Methods**:
   - Delete `interpolateColors()` (if not used elsewhere)
   - Delete `updateInterpolationVisualization()`
   - Delete `computeInterpolationWeights()`

4. **State Variables**:
   - Delete `self.interpolationData`, `self.interpolationWeights`
   - Delete all interpolation-related state variables

### Phase 5: Rename MultiRecolor to Population Analysis
**File**: `color_deca/deca3/InterDeCA.py`

1. **Tab Label** (line ~412):
   - Change `tabsWidget.addTab(multiRecolorTab, "MultiRecolor")` to `"Population Analysis"`

2. **Documentation**:
   - Update module docstring to reflect new tab structure
   - Update class docstring for InterDeCAWidget

### Phase 6: Create Clustering View Tab (Empty Placeholder)
**File**: `color_deca/deca3/clustering_view/ClusteringView.py` (NEW)

1. **Module Structure**:
   - Create `clustering_view/` directory
   - Create `ClusteringView.py` with `ClusteringViewWidget` class
   - Create `ClusteringViewLogic` class (empty for now)
   - Create `__init__.py` for module imports

2. **Widget Implementation** (Minimal):
   - Create empty `ClusteringViewWidget` that extends `qt.QWidget`
   - Add a placeholder label: "Clustering View - Coming Soon"
   - No functionality implemented yet (reserved for future development)

3. **Integration**:
   - Instantiate in InterDeCA.setup() after MultiRecolor tab
   - Add as last tab: `tabsWidget.addTab(clusteringViewTab, "Clustering View")`
   - Tab should render without errors but have no interactive features

### Phase 7: Simplify Texture Directory Syncing
**File**: `color_deca/deca3/InterDeCA.py`

1. **Remove Sync Logic** (lines ~1700-1735):
   - Delete `syncTextureDirectories()` method
   - Delete cross-tab syncing in `saveTextureDirectory()`
   - Keep only MultiRecolor texture directory persistence

2. **Update Restore Logic** (lines ~1671-1691):
   - Simplify `restoreTextureDirectories()` to only handle MultiRecolor

## Testing Plan

### Unit Tests
1. **Tab Structure**:
   - Verify 4 tabs exist: ATLAS, Mesh Selection, Population Analysis, Clustering View
   - Verify Colors EDA, Recolor, PCA Morphospace tabs don't exist

2. **MultiRecolor Workflow**:
   - Run Step 1 (Clustering) with test data
   - Verify Step 2 (Individual Visualization) works
   - Verify Step 3 (Population Analysis) works
   - Verify Step 4 (Morphospace) works

3. **Clustering View**:
   - Verify tab renders without errors
   - Verify placeholder content is displayed
   - Verify tab is empty and ready for future implementation

### Integration Tests
1. **Module Loading**: Module loads without errors or warnings
2. **Tab Navigation**: All tabs render and respond to user interaction
3. **Data Flow**: MultiRecolor output feeds into Clustering View
4. **Persistence**: Texture directories persist across sessions

### Regression Tests
1. **Existing Features**: All MultiRecolor functionality unchanged
2. **ATLAS Tab**: Dense correspondence workflow unaffected
3. **Mesh Selection**: Region selection workflow unaffected

## Files Modified
- `color_deca/deca3/InterDeCA.py` (major refactoring)
- `color_deca/deca3/clustering_view/ClusteringView.py` (NEW)
- `color_deca/deca3/clustering_view/__init__.py` (NEW)

## Estimated Effort
- **Phase 1-5**: 3-4 hours (tab removal and renaming)
- **Phase 6**: 0.5-1 hour (Clustering View empty placeholder)
- **Phase 7**: 1 hour (simplify syncing)
- **Testing**: 2-3 hours
- **Total**: ~7-9 hours

## Rollback Plan
- Git branch for safe development
- Keep removed code in git history for reference
- Test thoroughly before merging to main

