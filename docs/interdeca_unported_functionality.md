# InterDeCA Unported Functionality Audit

This document tracks functionality that exists in
`color_deca/deca3/InterDeCA.py` but is not yet fully replicated in the
clean-room package under `color_deca/interdeca/`.

The current refactor has enough implementation for the full small-clams E2E
workflow to pass through `color_deca.interdeca.facade`. That means atlas
generation, rigid alignment, dense correspondence, UV transfer, Blender bake,
average texture generation, and SHA comparison against saved legacy output are
covered. Most interactive UI, color-analysis, plotting, package-management,
and optional workflow behavior is still unported or only scaffolded.

## Need Classification

- **Needed for current E2E**: Required by `tests/end_to_end/test_interdeca_e2e.py`.
- **Needed for full legacy parity**: Required if the refactored package is to
  replace `InterDeCA.py` without removing user-visible behavior.
- **Likely optional/legacy**: Behavior that may be useful but should be
  confirmed before spending refactor time.

Because the migration assumption is that all existing code is used, anything
marked "No" for current E2E should not be interpreted as safe to delete.

## Currently Ported Baseline

The following functionality has been copied or implemented enough for parity
with the current full small-clams E2E fixture:

- Dataset discovery and subject matching.
- Standard workflow output paths.
- Dependency availability probing.
- Slicer-side facade methods used by the E2E driver:
  - `getClosestToMeanPath`
  - `getLandmarkFileByID`
  - `getModelFileByID`
  - `runAlign`
  - `runMean`
  - `runDCAlign`
  - dense correspondence helpers used by `runMean` and `runDCAlign`
  - `_load_model_with_cs`
  - `_save_model_with_cs`
  - `blender_prepare_atlas`
  - `write_obj_with_uv_from_template`
  - `blender_bake_all`
  - `_calculate_average_texture`
- VTK face connectivity and face area calculation.
- UV-based face color sampling.
- `ColorTheme`.
- SHA folder comparison test utility.

## Summary Table

| Legacy area | Refactor status | Needed for current E2E | Needed for full legacy parity |
| --- | --- | --- | --- |
| Slicer module registration | Not ported | No | Yes |
| Main Slicer UI construction | Controller shells only | No | Yes |
| Directory validation and saved UI state | Not ported | No | Yes |
| Package install UI and Slicer pip flow | Not ported | No | Yes |
| Full ATLAS tab UI workflow | Not ported as UI | No | Yes |
| E2E atlas workflow logic | Partially ported in facade | Yes | Yes |
| Shape service decomposition | Not cleanly ported | No | Yes |
| Mirroring and symmetric workflows | Not ported | No | Yes, if supported workflows remain |
| Mesh selection algorithms and UI | Mostly not ported | No | Yes |
| Visualization tab | Not ported | No | Yes |
| Plot generation and Slicer layouts | Not ported | No | Yes |
| Colors EDA | Not ported | No | Yes |
| MultiRecolor clustering | Placeholder only | No | Yes |
| Population analysis | Placeholder only | No | Yes |
| Morphospace visualization | Not ported | No | Yes |
| Color quantization | Placeholder only | No | Yes |
| Texture application display helpers | Not ported | No | Yes |
| Blender discovery/install | Partially ported discovery only | No | Yes for UI parity |
| Alternative/error/cancel paths | Not ported | No | Yes |

## 1. Slicer Module Registration

### Legacy functionality

`InterDeCA.py` defines the actual Slicer module classes:

- `InterDeCA(ScriptedLoadableModule)`
- `InterDeCAWidget(ScriptedLoadableModuleWidget)`
- `InterDeCALogic(ScriptedLoadableModuleLogic)`

These classes provide Slicer metadata, category placement, help text,
acknowledgements, widget setup, and logic construction.

### Refactor status

Not ported. The new package has `InterDeCALogicFacade`, but there is no new
Slicer module wrapper that replaces the legacy module entrypoint.

### Needed?

- Current E2E: No. The E2E driver imports the facade directly.
- Full parity: Yes. A real replacement module still needs a Slicer entrypoint.

### Notes

This should be one of the last migration steps. Until package behavior is
stable, keeping `InterDeCA.py` untouched as the behavior oracle is safer.

## 2. Package Checking And Installation

### Legacy functionality

Top-level functions and widget handlers inspect optional dependencies and can
offer installation through Slicer's Python environment:

- `checkAndOfferPackageInstallation`
- `installMissingPackages`
- `updatePackageStatus`
- `onInstallPackagesClicked`
- `onInstallAllPackagesClicked`

The code checks packages such as `numpy`, `sklearn`, `umap`, `skimage`,
`imageio`, and `imagecodecs`, displays status in the UI, and attempts
installation when requested.

### Refactor status

Only dependency probing is ported in `core/dependencies.py`. Installation,
status labels, UI buttons, and Slicer pip behavior are not ported.

### Needed?

- Current E2E: No.
- Full parity: Yes, if the Slicer UI should continue managing optional
  dependency setup for users.

### Notes

The new dependency checks correctly keep sklearn independent from UMAP. The
installation flow should remain Slicer-specific and probably belongs in
`slicer_adapter` plus `ui/package_management.py`.

## 3. Main UI Construction

### Legacy functionality

`InterDeCAWidget.setup()` builds the full Slicer UI. This includes:

- ATLAS input/output controls.
- Blender parameter controls.
- Atlas override controls.
- Run buttons.
- Progress bars and log widgets.
- Mesh Selection tab.
- MultiRecolor tab.
- Visualization controls.
- Package management panel.
- Node selectors, directory buttons, sliders, checkboxes, combo boxes, and
  collapsible sections.

### Refactor status

Only controller shells exist:

- `DeCATabController`
- `MeshSelectionController`
- `VisualizationController`
- `MultiRecolorController`
- `PackageManagementController`
- `InterDeCAUiComposition`

These currently hold references but do not build the full UI.

### Needed?

- Current E2E: No.
- Full parity: Yes.

### Notes

This is a large missing area. The first UI migration should copy existing
widget names and signal connections before any cleanup, because many handlers
depend on top-level widget attributes.

## 4. Directory Validation, UI State, And Progress Handling

### Legacy functionality

The widget manages directory state and validation:

- `setupKeyboardShortcuts`
- `onEscapeKey`
- `cleanup`
- `restoreTextureDirectories`
- `saveTextureDirectory`
- `autoDetectBlender`
- `validateDirectory`
- `onMeshDirectoryChangedDC`
- `onLandmarkDirectoryChangedDC`
- `onTextureDirectoryChangedDC`
- `restoreSavedDirectories`
- `onOutputDirectoryChangedDC`
- `validateTextureMatching`
- `updateProgressDC`
- `resetProgressDC`
- `onCancelOperationDC`

This is the user-facing glue that validates model, landmark, texture, and
output directories, restores recent paths, updates progress, and supports
cancellation.

### Refactor status

Not ported.

### Needed?

- Current E2E: No.
- Full parity: Yes.

### Notes

Some validation overlaps with the new `DatasetDiscovery`, but the UI behavior,
status labels, persistence, and cancellation hooks still need to be copied.

## 5. Full ATLAS Tab UI Workflow

### Legacy functionality

The ATLAS tab drives the full user workflow:

- `setUpDeCADir`
- `onGenerateAtlasButton`
- `generateNewAtlas`
- `onDCApplyButton`
- `updateUIAfterDeCACompletion`
- `onGetPointNumberButton`
- atlas display toggles
- baked texture preview list updates
- preview texture selection

This code combines UI validation, output directory creation, atlas generation,
alignment, UV preparation, dense correspondence, baking, average texture
generation, loading final outputs, and UI state updates.

### Refactor status

The non-UI core workflow used by E2E is partially copied into the facade and
the driver. It is not yet extracted into `AtlasTextureWorkflow`, and the UI
flow is not ported.

### Needed?

- Current E2E: Partly. The computational path is needed and currently passes.
- Full parity: Yes. The UI-driven workflow is not yet equivalent.

### Notes

The next architectural cleanup should move the copied facade workflow into
`workflow.py`, `shape.py`, `slicer_adapter/model_io.py`, and Blender services.

## 6. Shape And ATLAS Operations Outside Current E2E

### Legacy functionality

`InterDeCALogic` contains shape and ATLAS operations beyond the current E2E
path:

- `runSubsetLandmarks`
- `runCheckPoints`
- `runDeCAL`
- `downsampleModel`
- `addIndexArray`
- `computeNormals`
- `runMirroring`
- `runDCAlignSymmetric`
- `denseCorrespondenceCPD`
- `addMagnitudeFeatureSymmetry`

These cover landmark subsets, point checks, older DeCA-compatible workflows,
mirroring, symmetric correspondence, CPD correspondence, normals, and indexing.

### Refactor status

Not ported, except for shared helper ideas in the E2E facade. `shape.py` is
currently a placeholder.

### Needed?

- Current E2E: No.
- Full parity: Yes, if all legacy workflow options remain supported.

### Notes

This should be extracted into `AtlasShapeService`. Methods that delegate to
`atlas_integration.shape_bridge` should preserve the same fallback behavior.

## 7. Mesh Region Selection

### Legacy functionality

The Mesh Selection tab supports interactive region selection and export:

- markup observers
- creating and placing markup/curve nodes
- fast surface paint mode
- brush-based selection
- landmark-based polygon selection
- polygon-area flood fill
- landmark-side filtering
- spatial fallback selection
- selecting from existing closed curves
- mirrored selection
- mirrored markup generation
- mirrored vertex detection
- connected region filtering
- region growth
- selected region visualization
- selected model export
- clearing selection display

Representative methods:

- `onRegionSelectionInputChanged`
- `onMarkupNodeChanged`
- `onFastSurfacePaint`
- `_setupSurfacePainting`
- `_updateSurfaceSelection`
- `onApplyLandmarkSelection`
- `onExportLandmarkSelection`
- `onCreateSelectionCurve`
- `_findClickedLandmark`
- `_selectMeshRegionByPolygonAreaFloodFill`
- `_detectLandmarkSide`
- `selectMeshRegionByPolygonArea`
- `_fallbackSpatialSelection`
- `mapCutModelToOriginalVertices`
- `selectMeshRegionByExistingCurve`
- `selectMeshRegionBySelectedPoints`
- `selectMirroredRegion`
- `_landmarksFormCompleteArea`
- `_calculatePolygonArea`
- `_createMirroredMarkup`
- `findMirroredVertices`
- `_filterToConnectedRegion`
- `_growRegionFromVertices`
- `filterVerticesToSameSide`
- `visualizeRegionSelection`
- `createModelFromSelectedVertices`
- `clearRegionSelection`

### Refactor status

Only a very small VTK-only XY polygon selector exists in
`vtk_ops/mesh_selection.py`. The Slicer UI and almost all selection algorithms
are not ported.

### Needed?

- Current E2E: No.
- Full parity: Yes.

### Notes

This is a major feature area. It should be split into:

- pure/VTK selection algorithms
- Slicer node creation and display adapters
- UI controller event handling

## 8. Visualization Tab And Display State

### Legacy functionality

The visualization code supports:

- starting visualization
- atlas/model display toggles
- subject ID selection
- visualized mesh selection
- visualization mode switching
- original model directory browsing
- model file selection
- interpolation controls
- parameter and point selection
- display cleanup
- hiding/showing landmarks and models
- maximizing 3D view
- switching to optimal Slicer layouts
- updating baked texture previews

Representative methods:

- `onStartVisualizationButton`
- `onToggleAtlasDCL`
- `onSubjectIDSelect`
- `onVisualizeMeshSelect`
- `onVisualizationModeChanged`
- `onVisOriginalModelDirChanged`
- `onVisOriginalModelFileSelected`
- `onInterpolationInputChanged`
- `onInterpolationSliderChanged`
- `_hideMarkupsForVisualization`
- `_hideUnwantedModels`
- `_ensureModelsAreVisible`
- `_hideAllLandmarks`
- `_hideOtherModels`
- `maximize3DViewer`
- `switchToOptimalViewLayout`

### Refactor status

Not ported. `ui/visualization.py` is only a shell.

### Needed?

- Current E2E: No.
- Full parity: Yes.

### Notes

This should mostly live in `VisualizationController` plus Slicer scene/layout
adapters.

## 9. Slicer Plotting

### Legacy functionality

The module creates and manages plots inside Slicer:

- population PCA/ICA/UMAP scatter plots
- colored scatter plots
- Colors EDA plots
- morphospace plots
- moving morphospace point updates
- variance labels
- plot/table/chart nodes
- layout switching for plot views

Representative methods:

- `createPopulationPlot`
- `_createStandardPlot`
- `_createColoredScatterPlot`
- `createColorsEDAPlot`
- `createMorphospacePlot`
- `updateMorphospacePoint`
- `maximizePlotViewer`

### Refactor status

Not ported. `slicer_adapter/plots.py` only has a minimal table-node helper.

### Needed?

- Current E2E: No.
- Full parity: Yes.

### Notes

Plot computation and plot display should be separated. Population or EDA data
should be created in core services; Slicer plot-node construction should remain
in `slicer_adapter`.

## 10. Colors EDA

### Legacy functionality

Colors EDA samples face colors across textures and reduces them for plotting:

- `runColorsEDA`
- `sampleColorData`
- `_applyNeighborAveraging`
- `_applyDimensionalityReduction`
- `runColorsEDAFromSampledData`
- `applyAverageFaceColorsFromTexture`
- `applyAverageFaceColorsFromTextureAlternative`

It supports RGB/HSV feature extraction, saturation/value cutoffs, enhanced
colors, random sampling, progress callbacks, and plotting integration.

### Refactor status

Partially ported only at the lowest level:

- UV face average colors are copied into `vtk_ops/color_sampling.py`.
- Face areas and connectivity are copied into `vtk_ops/mesh_geometry.py`.

The full EDA pipeline is not ported.

### Needed?

- Current E2E: No.
- Full parity: Yes.

### Notes

This should become a `ColorEdaService` or be split between sampling,
dimensionality reduction, and plot preparation services.

## 11. Mesh Cleanup, Adjacency, And Subsampling

### Legacy functionality

The logic includes performance-oriented mesh utilities:

- `_extractFaceConnectivity`
- `_calculateFaceAreas`
- `_cleanMesh`
- `_buildFaceAdjacencyGraph`
- `_subsampleFacesUniformly`

These support clustering, neighbor averaging, area-weighted sampling, and
subsampled display.

### Refactor status

Partially ported:

- `_extractFaceConnectivity` is copied as `extract_face_connectivity`.
- `_calculateFaceAreas` is copied as `calculate_face_areas`.
- A smaller adjacency helper exists.

Not ported:

- legacy mesh cleaning
- full adjacency graph behavior
- uniform face subsampling
- related progress/log callbacks

### Needed?

- Current E2E: No.
- Full parity: Yes, especially for MultiRecolor and Colors EDA.

### Notes

This belongs in `vtk_ops/mesh_geometry.py`, with exact numerical and indexing
tests.

## 12. Color Quantization

### Legacy functionality

The module supports perceptual color quantization:

- `rgb_to_lab`
- `delta_e_2000`
- `generate_high_contrast_palette`
- `quantize_colors_lab_kmeans`
- `lab_to_rgb`
- `applyQuantizedFaceColorsFromTexture`

It uses Lab color space, DeltaE2000, optional high-contrast palette generation,
and VTK color-array application.

### Refactor status

Not ported. `core/color_quantization.py` is a placeholder.

### Needed?

- Current E2E: No.
- Full parity: Yes.

### Notes

This should be copied into `ColorQuantizationService`. Keep scikit-image
conversion paths unchanged to avoid numerical drift.

## 13. MultiRecolor Clustering

### Legacy functionality

MultiRecolor clusters color patterns across texture sets:

- `performMultiTextureClustering`
- `performSubsampleOnly`
- `_clusterAndConsolidateSingleTexture`
- `_reorderClusterIndices`
- `_isBlackPixel`
- `_computePooledLCStats`
- `_computePerTextureLCStats`
- `_applyLCTransform`
- `applyIndividualTextureWithClusteredPalette`
- `applyTextureWithSubsamplingOnly`

It handles initial clusters, consolidated clusters, Lab-space clustering,
luminosity normalization, black-pixel filtering, cluster reordering, shared
palettes, individual texture application, subsampling, and area weighting.

### Refactor status

Not ported. `core/clustering.py` currently contains only a placeholder service
and a small `NamedTuple` that does not match the legacy `ClusteringPipeline`
behavior.

### Needed?

- Current E2E: No.
- Full parity: Yes.

### Notes

This is one of the most important unported analysis features. It needs golden
tests for palette shape, cluster ordering, random seeds, Lab-space math, and
no-NaN guarantees.

## 14. Legacy `ClusteringPipeline`

### Legacy functionality

`ClusteringPipeline` stores shared clustering state:

- initial cluster count
- consolidated cluster count
- luminosity normalization flag
- neighbor averaging flag
- per-texture clustering results
- per-texture LC stats
- reorder mappings
- shared palette calculation
- per-texture consolidated centroid lookup

Methods:

- `__init__`
- `addTextureResults`
- `computeSharedPalette`
- `getConsolidatedCentroidsForTexture`

### Refactor status

Not ported. The new `ClusteringPipeline` is only a minimal placeholder
`NamedTuple`.

### Needed?

- Current E2E: No.
- Full parity: Yes.

### Notes

This should be copied before porting `performMultiTextureClustering`, because
many downstream methods depend on its state shape.

## 15. Population Analysis

### Legacy functionality

Population analysis constructs texture-level feature vectors and applies
dimensionality reduction:

- `performPopulationAnalysis`
- PCA, ICA, and UMAP modes
- texture vector construction
- clustered and subsampled vector modes
- morphospace reconstruction data
- texture name ordering
- variance explained metadata

### Refactor status

Not ported. `core/population.py` is a placeholder.

### Needed?

- Current E2E: No.
- Full parity: Yes.

### Notes

This should be extracted after clustering, because it depends on clustering
pipeline state and sampled color vectors.

## 16. Morphospace Visualization And Color Application

### Legacy functionality

The module can visualize reconstructed color patterns from morphospace
coordinates:

- `onMorphospaceTextureChanged`
- `onMorphospaceXSliderChanged`
- `onMorphospaceYSliderChanged`
- `onVisualizeMorphospace`
- `updateMorphospaceVisualizationXY`
- `createMorphospacePlot`
- `updateMorphospacePoint`
- `applyMorphospaceColors`
- `applyMorphospaceColorsToModel`
- `applyMorphospaceColorsSubsampled`
- `setMorphospaceLayout`

This connects population PCA space, slider controls, plot markers, reconstructed
colors, and Slicer model display.

### Refactor status

Not ported.

### Needed?

- Current E2E: No.
- Full parity: Yes.

### Notes

This should be split into:

- core morphospace reconstruction
- Slicer scalar/color array application
- UI slider and plot controller behavior

## 17. Texture Application And VTK Display

### Legacy functionality

The legacy module applies texture and color data to Slicer models:

- `applyTextureToModel`
- `applyAverageFaceColorsFromTexture`
- `applyAverageFaceColorsFromTextureAlternative`
- `applyIndividualTextureWithClusteredPalette`
- `applyTextureWithSubsamplingOnly`
- `applyMorphospaceColorsToModel`
- `applyMorphospaceColorsSubsampled`
- `applyQuantizedFaceColorsFromTexture`

These methods create or update VTK arrays, set active scalars, configure model
display nodes, and update visibility/color mapping.

### Refactor status

Not ported except for low-level color sampling. The current E2E does not apply
cell-color arrays to models.

### Needed?

- Current E2E: No.
- Full parity: Yes.

### Notes

This should be a Slicer adapter plus pure color-array preparation code. VTK
array names and dtypes must match legacy behavior.

## 18. Blender Discovery And Installation

### Legacy functionality

The legacy code can locate or install Blender:

- `findBlenderExecutable`
- `_tryInstallBlenderMacOS`
- `_tryInstallBlenderLinux`
- `_detectLinuxDistribution`
- `_installBlenderMacOSDMG`
- `installBlender`
- `getBlenderExecutable`

It searches common paths, checks platform-specific locations, optionally
downloads installers, and logs progress to the UI.

### Refactor status

Partially ported:

- `BlenderExecutableService.find`
- `BlenderExecutableService.require`

Not ported:

- install flows
- DMG handling
- Linux package-manager handling
- UI prompts and log callbacks

### Needed?

- Current E2E: No. The E2E receives a Blender executable path.
- Full parity: Yes, if the UI should retain Blender setup assistance.

### Notes

Installation should not live in core numerical modules. It belongs in an
adapter/service with explicit callbacks and tests that mock downloads and
subprocesses.

## 19. Alternative And Edge-Case Workflows

### Legacy functionality

The legacy file contains paths that are not exercised by current tests:

- symmetric dense correspondence
- mirrored mesh and landmark workflows
- semi-landmark alignment
- CPD registration path
- alternative average color application
- workflow cancellation
- optional error output paths
- fallback handling when ATLAS bridge calls fail

### Refactor status

Mostly not ported.

### Needed?

- Current E2E: No.
- Full parity: Probably yes. Each path should be confirmed with a fixture or
  UI workflow before deciding it is obsolete.

### Notes

These should be ported after the primary UI and analysis workflows, unless a
user workflow depends on them immediately.

## 20. Error Handling, Logging, And User Feedback

### Legacy functionality

Many methods accept callbacks or write to Slicer UI log widgets:

- progress callbacks
- log callbacks
- warning messages
- UI status labels
- exception messages
- installation logs
- clustering logs
- population-analysis logs

### Refactor status

Only a small amount is preserved in the E2E facade and tests.

### Needed?

- Current E2E: Partly.
- Full parity: Yes.

### Notes

When services are extracted, callbacks should remain explicit inputs. Core
services should not directly know about Qt widgets.

## 21. Current Technical Debt In The Refactor

Some functionality is copied into the new package but not yet located in its
final architecture:

- The E2E shape/Blender/Slicer workflow is in `facade.py`.
- `shape.py` is still a placeholder.
- `workflow.py` is still a placeholder.
- `slicer_adapter/model_io.py` has model IO helpers, but the E2E facade still
  contains its own copies.
- `core/blender_scripts.py` and `core/blender.py` exist, but the E2E facade
  still uses inline Blender script generation copied from legacy code.

This is acceptable for the first parity step, but the next cleanup pass should
move copied behavior out of `facade.py` and into focused services while keeping
the same E2E SHA comparison green.

## Recommended Port Order

1. Move the already-copied E2E workflow out of `facade.py` into:
   - `AtlasTextureWorkflow`
   - `AtlasShapeService`
   - `SlicerModelIO`
   - `BlenderTextureBakeService`
   - `TextureTransferService`
2. Port `ClusteringPipeline` exactly.
3. Port MultiRecolor clustering and individual texture application.
4. Port population analysis and morphospace reconstruction.
5. Port quantization.
6. Port mesh selection algorithms.
7. Port plotting and Slicer layout display.
8. Port UI controllers and signal wiring.
9. Port package installation and Blender installation flows.
10. Add a thin new Slicer module wrapper only after package-level parity is
    established.

## Test Gaps To Add Before Calling The Refactor Complete

- New Slicer UI smoke test for controller construction.
- Slicer-side facade method coverage beyond full E2E.
- Mesh selection fixture test.
- MultiRecolor analysis fixture test.
- Population PCA/ICA/UMAP fixture test.
- Quantization golden tests.
- Plot-node creation tests inside Slicer.
- Blender script parity tests against legacy inline script behavior.
- Full fixture parity test for `data/All_Clams`.

