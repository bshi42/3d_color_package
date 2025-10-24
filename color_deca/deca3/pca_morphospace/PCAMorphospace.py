"""
PCA Morphospace Module - Extension for InterDeCA
Implements PCA-based color morphospace visualization with interactive slider

This module adds a new tab to InterDeCA for visualizing how specimen colors
vary along principal component axes. It uses the resampled models and baked
textures from DeCA output.

Author: PCA Morphospace Implementation Team
Date: October 2024
"""

import qt
import ctk
import slicer
import vtk
import numpy as np
import os
import json
import colorsys
from pathlib import Path
import sys

# Import ColorTheme from parent module
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
try:
    from InterDeCA import ColorTheme
except ImportError:
    # Fallback if ColorTheme not available
    class ColorTheme:
        @staticmethod
        def getButtonStyle(style_type='primary'):
            if style_type == 'primary':
                return "QPushButton { background-color: #4CAF50; color: white; font-weight: bold; border: none; border-radius: 4px; padding: 6px 12px; min-height: 20px; } QPushButton:hover { background-color: #45A049; } QPushButton:disabled { background-color: #CCCCCC; color: #666666; }"
            elif style_type == 'secondary':
                return "QPushButton { background-color: #87CEEB; color: white; font-weight: bold; border: none; border-radius: 4px; padding: 6px 12px; min-height: 20px; } QPushButton:hover { background-color: #6BB6E8; } QPushButton:disabled { background-color: #CCCCCC; color: #666666; }"
            return ""
        @staticmethod
        def getStatusLabelStyle(style_type='neutral'):
            if style_type == 'success':
                return "QLabel { color: green; font-weight: bold; }"
            elif style_type == 'error':
                return "QLabel { color: red; font-weight: bold; }"
            elif style_type == 'progress':
                return "QLabel { color: orange; font-weight: bold; }"
            else:
                return "QLabel { color: palette(link); font-weight: bold; }"
        @staticmethod
        def getProgressLabelStyle():
            return "QLabel { color: palette(link); font-weight: bold; }"
        @staticmethod
        def getComboBoxStyle():
            return ""

class PCAMorphospaceWidget:
    """Widget for PCA Morphospace visualization tab"""

    def __init__(self, parent=None):
        """Initializes the PCA Morphospace widget

        Args:
            parent: Parent widget (InterDeCAWidget)
        """
        self.parent = parent
        self.logic = PCAMorphospaceLogic()

        # Stores PCA results
        self.pca_data = None
        self.specimen_colors = {}
        self.mean_colors = None
        self.pc_loadings = None
        self.pc_variance = None

    def setup(self):
        """Creates the PCA Morphospace tab interface"""

        # Creates main collapsible area
        pcaCollapsibleButton = ctk.ctkCollapsibleButton()
        pcaCollapsibleButton.text = "PCA Color Morphospace Analysis"
        pcaCollapsibleButton.collapsed = False

        # Creates form layout
        pcaFormLayout = qt.QFormLayout(pcaCollapsibleButton)

        # ======= Input Section =======
        inputGroupBox = qt.QGroupBox("Input Data")
        inputLayout = qt.QFormLayout(inputGroupBox)

        # Directory selector for DeCA results
        self.decaResultsSelector = ctk.ctkPathLineEdit()
        self.decaResultsSelector.filters = ctk.ctkPathLineEdit.Dirs
        self.decaResultsSelector.setToolTip("Select the DeCA output directory containing resampled models and baked textures")
        inputLayout.addRow("DeCA Results Directory:", self.decaResultsSelector)

        # Load button
        self.loadDataButton = qt.QPushButton("Load Data for PCA Analysis")
        self.loadDataButton.toolTip = "Load resampled models and textures for PCA"
        self.loadDataButton.enabled = True
        self.loadDataButton.setStyleSheet(ColorTheme.getButtonStyle('primary'))
        self.loadDataButton.connect('clicked(bool)', self.onLoadData)
        inputLayout.addRow(self.loadDataButton)

        # Status label with centered alignment
        self.statusLabel = qt.QLabel("Status: Ready")
        self.statusLabel.setAlignment(qt.Qt.AlignCenter)
        self.statusLabel.setStyleSheet(ColorTheme.getStatusLabelStyle('neutral'))
        inputLayout.addRow(self.statusLabel)

        pcaFormLayout.addWidget(inputGroupBox)

        # ======= PCA Settings Section =======
        settingsGroupBox = qt.QGroupBox("PCA Settings")
        settingsLayout = qt.QFormLayout(settingsGroupBox)

        # Number of components
        self.numComponentsSpin = qt.QSpinBox()
        self.numComponentsSpin.setMinimum(2)
        self.numComponentsSpin.setMaximum(10)
        self.numComponentsSpin.setValue(3)
        self.numComponentsSpin.setToolTip("Number of principal components to compute")
        settingsLayout.addRow("Number of PCs:", self.numComponentsSpin)

        # Color space selection
        self.colorSpaceCombo = qt.QComboBox()
        self.colorSpaceCombo.addItems(["RGB", "HSV", "LAB"])
        self.colorSpaceCombo.setToolTip("Color space for PCA analysis")
        self.colorSpaceCombo.setStyleSheet(ColorTheme.getComboBoxStyle())
        settingsLayout.addRow("Color Space:", self.colorSpaceCombo)

        # Area weighting checkbox
        self.areaWeightingCheck = qt.QCheckBox()
        self.areaWeightingCheck.setChecked(True)
        self.areaWeightingCheck.setToolTip("Weight colors by face area in PCA calculation")
        settingsLayout.addRow("Area-weighted PCA:", self.areaWeightingCheck)

        # Run PCA button
        self.runPCAButton = qt.QPushButton("Run PCA Analysis")
        self.runPCAButton.toolTip = "Perform PCA on specimen colors"
        self.runPCAButton.enabled = False
        self.runPCAButton.setStyleSheet(ColorTheme.getButtonStyle('primary'))
        self.runPCAButton.connect('clicked(bool)', self.onRunPCA)
        settingsLayout.addRow(self.runPCAButton)

        # Add separator
        settingsLayout.addRow(" ", qt.QLabel())

        # Progress tracking widgets - matching InterDeCA style exactly
        self.progressWidget = qt.QWidget()
        self.progressWidget.setVisible(False)
        self.progressWidget.setMinimumHeight(30)  # Ensure minimum height
        progressLayout = qt.QVBoxLayout(self.progressWidget)
        progressLayout.setContentsMargins(0, 0, 0, 0)

        self.progressBar = qt.QProgressBar()
        self.progressBar.setRange(0, 100)
        self.progressBar.setValue(0)
        self.progressBar.setMinimumHeight(25)  # Ensure progress bar is visible
        self.progressBar.setTextVisible(True)  # Show percentage text
        # Style the progress bar to match green theme with bold text
        self.progressBar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #4CAF50;
                border-radius: 5px;
                text-align: center;
                background-color: #f0f0f0;
                font-weight: bold;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                border-radius: 3px;
            }
        """)
        progressLayout.addWidget(self.progressBar)

        settingsLayout.addRow("Progress: ", self.progressWidget)

        pcaFormLayout.addWidget(settingsGroupBox)

        # ======= Visualization Section =======
        visGroupBox = qt.QGroupBox("PCA Morphospace Visualization")
        visLayout = qt.QVBoxLayout(visGroupBox)

        # Variance explained label
        self.varianceLabel = qt.QLabel("")
        self.varianceLabel.setStyleSheet("font-weight: bold; padding: 10px;")
        visLayout.addWidget(self.varianceLabel)

        # Plot axis controls
        axisLayout = qt.QHBoxLayout()

        axisLayout.addWidget(qt.QLabel("X-axis:"))
        self.xAxisCombo = qt.QComboBox()
        self.xAxisCombo.setToolTip("Select PC for X-axis")
        self.xAxisCombo.enabled = False
        self.xAxisCombo.setStyleSheet(ColorTheme.getComboBoxStyle())
        self.xAxisCombo.connect('currentIndexChanged(int)', self.onAxisChanged)
        axisLayout.addWidget(self.xAxisCombo)

        axisLayout.addWidget(qt.QLabel("Y-axis:"))
        self.yAxisCombo = qt.QComboBox()
        self.yAxisCombo.setToolTip("Select PC for Y-axis")
        self.yAxisCombo.enabled = False
        self.yAxisCombo.setStyleSheet(ColorTheme.getComboBoxStyle())
        self.yAxisCombo.connect('currentIndexChanged(int)', self.onAxisChanged)
        axisLayout.addWidget(self.yAxisCombo)

        visLayout.addLayout(axisLayout)

        # Plot control button
        self.showPlotButton = qt.QPushButton("Update Plot")
        self.showPlotButton.toolTip = "Update PCA plot with selected axes"
        self.showPlotButton.enabled = False
        self.showPlotButton.setStyleSheet(ColorTheme.getButtonStyle('primary'))
        self.showPlotButton.connect('clicked(bool)', self.onShowPlot)
        visLayout.addWidget(self.showPlotButton)

        # ======= PC Navigation Section =======
        pcNavFrame = qt.QFrame()
        pcNavFrame.setFrameStyle(qt.QFrame.StyledPanel)
        pcNavLayout = qt.QFormLayout(pcNavFrame)

        # PC selection
        self.pcSelectCombo = qt.QComboBox()
        self.pcSelectCombo.setToolTip("Select which PC to explore")
        self.pcSelectCombo.enabled = False
        self.pcSelectCombo.setStyleSheet(ColorTheme.getComboBoxStyle())
        self.pcSelectCombo.connect('currentIndexChanged(int)', self.onPCSelectionChanged)
        pcNavLayout.addRow("Principal Component:", self.pcSelectCombo)

        # PC position slider
        self.pcSlider = qt.QSlider(qt.Qt.Horizontal)
        self.pcSlider.setMinimum(-200)  # -2 SD * 100
        self.pcSlider.setMaximum(200)   # +2 SD * 100
        self.pcSlider.setValue(0)       # Mean
        self.pcSlider.setTickInterval(50)
        self.pcSlider.setTickPosition(qt.QSlider.TicksBelow)
        self.pcSlider.enabled = False
        self.pcSlider.setToolTip("Slide to explore color variation along selected PC")
        self.pcSlider.connect('valueChanged(int)', self.onSliderChanged)

        # Slider value label
        self.sliderValueLabel = qt.QLabel("Position: 0.00 SD")

        pcNavLayout.addRow("PC Position:", self.pcSlider)
        pcNavLayout.addRow("", self.sliderValueLabel)

        visLayout.addWidget(pcNavFrame)

        # ======= Color Display Section =======
        colorFrame = qt.QFrame()
        colorFrame.setFrameStyle(qt.QFrame.Box)
        colorFrame.setMinimumHeight(300)  # Increased to accommodate info display
        colorLayout = qt.QVBoxLayout(colorFrame)

        # Color display info
        self.colorDisplayLabel = qt.QLabel("Color patterns at current PC position will appear here")
        self.colorDisplayLabel.setAlignment(qt.Qt.AlignCenter)
        colorLayout.addWidget(self.colorDisplayLabel)

        # Create horizontal layout for colors and info
        colorContentLayout = qt.QHBoxLayout()

        # Color swatches grid (left side)
        self.colorGrid = qt.QWidget()
        self.colorGridLayout = qt.QGridLayout(self.colorGrid)
        colorContentLayout.addWidget(self.colorGrid)

        # Color info display (right side)
        self.colorInfoFrame = qt.QFrame()
        self.colorInfoFrame.setFrameStyle(qt.QFrame.StyledPanel)
        self.colorInfoFrame.setMaximumWidth(200)
        self.colorInfoFrame.setMinimumWidth(200)
        colorInfoLayout = qt.QVBoxLayout(self.colorInfoFrame)

        self.colorInfoTitle = qt.QLabel("Color Information")
        self.colorInfoTitle.setStyleSheet("font-weight: bold; padding: 5px;")
        self.colorInfoTitle.setAlignment(qt.Qt.AlignCenter)
        colorInfoLayout.addWidget(self.colorInfoTitle)

        self.colorInfoText = qt.QLabel("Click on a color to see details")
        self.colorInfoText.setWordWrap(True)
        self.colorInfoText.setStyleSheet("padding: 5px;")
        self.colorInfoText.setAlignment(qt.Qt.AlignTop | qt.Qt.AlignLeft)
        colorInfoLayout.addWidget(self.colorInfoText)
        colorInfoLayout.addStretch()  # Push content to top

        colorContentLayout.addWidget(self.colorInfoFrame)
        colorLayout.addLayout(colorContentLayout)

        visLayout.addWidget(colorFrame)

        pcaFormLayout.addWidget(visGroupBox)

        # ======= Export Section =======
        exportGroupBox = qt.QGroupBox("Export Options")
        exportLayout = qt.QFormLayout(exportGroupBox)

        # Export results button
        self.exportButton = qt.QPushButton("Export PCA Results")
        self.exportButton.toolTip = "Export PCA analysis results and visualizations"
        self.exportButton.enabled = False
        self.exportButton.setStyleSheet(ColorTheme.getButtonStyle('primary'))
        self.exportButton.connect('clicked(bool)', self.onExportResults)
        exportLayout.addRow(self.exportButton)

        # Export path
        self.exportPathEdit = ctk.ctkPathLineEdit()
        self.exportPathEdit.filters = ctk.ctkPathLineEdit.Dirs
        self.exportPathEdit.setToolTip("Directory for exported results")
        exportLayout.addRow("Export Directory:", self.exportPathEdit)

        pcaFormLayout.addWidget(exportGroupBox)

        return pcaCollapsibleButton

    def onLoadData(self):
        """Handles loading of DeCA results for PCA analysis"""

        resultsDir = self.decaResultsSelector.currentPath
        if not resultsDir or not os.path.exists(resultsDir):
            slicer.util.errorDisplay("Please select a valid DeCA results directory")
            return

        self.statusLabel.text = "Status: Loading data..."
        self.statusLabel.setStyleSheet(ColorTheme.getStatusLabelStyle('progress'))
        slicer.app.processEvents()

        try:
            # Loads resampled models and textures
            success = self.logic.loadSpecimenData(resultsDir)

            if success:
                self.statusLabel.text = f"Status: Loaded {len(self.logic.specimens)} specimens"
                self.statusLabel.setStyleSheet(ColorTheme.getStatusLabelStyle('success'))
                self.runPCAButton.enabled = True
            else:
                self.statusLabel.text = "Status: Failed to load data"
                self.statusLabel.setStyleSheet(ColorTheme.getStatusLabelStyle('error'))

        except Exception as e:
            self.statusLabel.text = f"Status: Error - {str(e)}"
            self.statusLabel.setStyleSheet(ColorTheme.getStatusLabelStyle('error'))
            print(f"Error loading data: {str(e)}")

    def onRunPCA(self):
        """Handles PCA computation on specimen colors"""

        # Show progress widget
        self.progressWidget.setVisible(True)
        self.progressBar.setValue(0)
        self.statusLabel.text = "Status: Running PCA..."
        self.statusLabel.setStyleSheet(ColorTheme.getStatusLabelStyle('progress'))
        slicer.app.processEvents()

        try:
            # Update progress
            self.progressBar.setValue(20)
            self.statusLabel.text = "Status: Preparing color data..."
            slicer.app.processEvents()

            # Runs PCA analysis
            n_components = self.numComponentsSpin.value
            color_space = self.colorSpaceCombo.currentText
            area_weighted = self.areaWeightingCheck.isChecked()

            # Update progress
            self.progressBar.setValue(50)
            self.statusLabel.text = "Status: Computing principal components..."
            slicer.app.processEvents()

            results = self.logic.performPCA(
                n_components=n_components,
                color_space=color_space,
                area_weighted=area_weighted
            )

            # Update progress
            self.progressBar.setValue(80)
            self.statusLabel.text = "Status: Finalizing results..."
            slicer.app.processEvents()

            if results:
                self.pca_data = results
                self.updateVisualization()

                # Enables controls
                self.pcSelectCombo.enabled = True
                self.pcSlider.enabled = True
                self.exportButton.enabled = True
                self.showPlotButton.enabled = True
                self.xAxisCombo.enabled = True
                self.yAxisCombo.enabled = True

                # Populates PC selectors
                self.pcSelectCombo.clear()
                self.xAxisCombo.clear()
                self.yAxisCombo.clear()
                for i in range(n_components):
                    pc_label = f"PC{i+1}"
                    self.pcSelectCombo.addItem(pc_label)
                    self.xAxisCombo.addItem(pc_label)
                    self.yAxisCombo.addItem(pc_label)

                # Set default axes (PC1 vs PC2)
                self.xAxisCombo.setCurrentIndex(0)  # PC1
                if n_components > 1:
                    self.yAxisCombo.setCurrentIndex(1)  # PC2

                # Complete progress
                self.progressBar.setValue(100)
                slicer.app.processEvents()

                self.statusLabel.text = "Status: PCA complete"
                self.statusLabel.setStyleSheet(ColorTheme.getStatusLabelStyle('success'))

                # Updates variance label
                variance_text = "Variance explained: "
                for i, var in enumerate(results['variance_explained'][:3]):
                    variance_text += f"PC{i+1}: {var:.1f}% "
                self.varianceLabel.text = variance_text

            else:
                self.statusLabel.text = "Status: PCA failed"
                self.statusLabel.setStyleSheet(ColorTheme.getStatusLabelStyle('error'))

        except Exception as e:
            self.statusLabel.text = f"Status: Error - {str(e)}"
            self.statusLabel.setStyleSheet(ColorTheme.getStatusLabelStyle('error'))
            print(f"Error running PCA: {str(e)}")

        finally:
            # Hide progress widget
            self.progressWidget.setVisible(False)
            self.progressBar.setValue(0)

    def onShowPlot(self):
        """Recreates and shows PCA plot with currently selected axes"""

        if not self.pca_data:
            return

        try:
            # Remove old plot nodes if they exist
            if hasattr(self, 'currentPlotChartNode') and self.currentPlotChartNode:
                slicer.mrmlScene.RemoveNode(self.currentPlotChartNode)

            # Recreate plot with new axes
            self.createPCAPlot()

            if hasattr(self, 'currentPlotChartNode') and self.currentPlotChartNode:
                # Sets layout to show plot
                layoutManager = slicer.app.layoutManager()
                layoutManager.setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutOneUpPlotView)

                # Gets plot widget and sets chart
                plotWidget = layoutManager.plotWidget(0)
                if plotWidget:
                    plotViewNode = plotWidget.mrmlPlotViewNode()
                    if plotViewNode:
                        plotViewNode.SetPlotChartNodeID(self.currentPlotChartNode.GetID())
                        self.statusLabel.text = "Status: Plot updated with new axes"
                        self.statusLabel.setStyleSheet("color: green")
        except Exception as e:
            slicer.util.errorDisplay(f"Error updating plot: {str(e)}")

    def onAxisChanged(self, index):
        """Handles axis selection changes - just enables the update button"""
        # User needs to click Update Plot to refresh with new axes
        pass

    def onPCSelectionChanged(self, index):
        """Handles PC selection changes"""

        if not self.pca_data or index < 0:
            return

        # Updates the display for the new PC
        sd_value = self.pcSlider.value / 100.0
        self.updateColorDisplay(sd_value)

        # Updates plot title
        if hasattr(self, 'currentPlotChartNode') and self.currentPlotChartNode:
            pc_num = index + 1
            self.currentPlotChartNode.SetTitle(f"PCA Color Morphospace (Slider at PC{pc_num} = {sd_value:+.2f} SD)")

    def onSliderChanged(self, value):
        """Handles slider value changes"""

        if not self.pca_data:
            return

        # Converts to standard deviations
        sd_value = value / 100.0
        self.sliderValueLabel.text = f"Position: {sd_value:+.2f} SD"

        # Updates color display
        self.updateColorDisplay(sd_value)

        # Updates plot title to show current position
        if hasattr(self, 'currentPlotChartNode') and self.currentPlotChartNode:
            pc_index = self.pcSelectCombo.currentIndex
            if pc_index >= 0:
                pc_num = pc_index + 1
                self.currentPlotChartNode.SetTitle(f"PCA Color Morphospace (Slider at PC{pc_num} = {sd_value:+.2f} SD)")

    def createSliderIndicatorSeries(self, chartNode):
        """Creates series for vertical line and shaded region to show slider position

        Args:
            chartNode: The chart node to add series to
        """

        # Creates table for vertical line (2 points)
        self.sliderLineTable = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTableNode")
        self.sliderLineTable.SetName("PCA_SliderLine_Data")

        # Creates arrays for line position
        lineX = vtk.vtkFloatArray()
        lineX.SetName("LineX")
        lineY = vtk.vtkFloatArray()
        lineY.SetName("LineY")

        # Initializes at position 0 (mean)
        lineX.InsertNextValue(0)
        lineX.InsertNextValue(0)

        # Gets Y range from PC2 data
        pc2_min = float(min(self.pca_data['transformed'][:, 1])) if self.pca_data['transformed'].shape[1] > 1 else -10
        pc2_max = float(max(self.pca_data['transformed'][:, 1])) if self.pca_data['transformed'].shape[1] > 1 else 10
        y_range = pc2_max - pc2_min
        y_padding = y_range * 0.1

        lineY.InsertNextValue(pc2_min - y_padding)
        lineY.InsertNextValue(pc2_max + y_padding)

        self.sliderLineTable.AddColumn(lineX)
        self.sliderLineTable.AddColumn(lineY)

        # Creates series for vertical line
        self.sliderLineSeries = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLPlotSeriesNode")
        self.sliderLineSeries.SetName("SliderPosition")
        self.sliderLineSeries.SetAndObserveTableNodeID(self.sliderLineTable.GetID())
        self.sliderLineSeries.SetXColumnName("LineX")
        self.sliderLineSeries.SetYColumnName("LineY")
        self.sliderLineSeries.SetPlotType(slicer.vtkMRMLPlotSeriesNode.PlotTypeLine)
        self.sliderLineSeries.SetLineStyle(slicer.vtkMRMLPlotSeriesNode.LineStyleSolid)
        self.sliderLineSeries.SetLineWidth(3.0)
        self.sliderLineSeries.SetColor(1.0, 0.2, 0.2)  # Red line

        # Adds to chart
        chartNode.AddAndObservePlotSeriesNodeID(self.sliderLineSeries.GetID())

        # Creates shaded region series (optional - rectangle around position)
        self.regionTable = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTableNode")
        self.regionTable.SetName("PCA_Region_Data")

        regionX = vtk.vtkFloatArray()
        regionX.SetName("RegionX")
        regionY = vtk.vtkFloatArray()
        regionY.SetName("RegionY")

        # Creates rectangle points (will be updated by slider)
        for i in range(5):  # 5 points to close rectangle
            regionX.InsertNextValue(0)
            regionY.InsertNextValue(0)

        self.regionTable.AddColumn(regionX)
        self.regionTable.AddColumn(regionY)

        # Creates series for shaded region
        self.regionSeries = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLPlotSeriesNode")
        self.regionSeries.SetName("SliderRegion")
        self.regionSeries.SetAndObserveTableNodeID(self.regionTable.GetID())
        self.regionSeries.SetXColumnName("RegionX")
        self.regionSeries.SetYColumnName("RegionY")
        self.regionSeries.SetPlotType(slicer.vtkMRMLPlotSeriesNode.PlotTypeLine)
        self.regionSeries.SetLineStyle(slicer.vtkMRMLPlotSeriesNode.LineStyleSolid)
        self.regionSeries.SetLineWidth(1.0)
        self.regionSeries.SetColor(0.8, 0.8, 0.2)  # Yellow region
        self.regionSeries.SetOpacity(0.3)  # Semi-transparent

        # Adds to chart
        chartNode.AddAndObservePlotSeriesNodeID(self.regionSeries.GetID())

        # Stores Y range for updates
        self.pc2_min = pc2_min - y_padding
        self.pc2_max = pc2_max + y_padding

    def updateSliderIndicator(self, sd_position):
        """Updates the position of the vertical line and shaded region

        Args:
            sd_position: Position in standard deviations from mean
        """

        if not hasattr(self, 'currentPlotChartNode') or not self.pca_data:
            return

        # Calculates actual position on PC1 axis
        pc1_std = np.std(self.pca_data['transformed'][:, 0])
        pc1_position = sd_position * pc1_std

        # Remove old line and region if they exist
        if hasattr(self, 'sliderLineSeries'):
            self.currentPlotChartNode.RemovePlotSeriesNodeID(self.sliderLineSeries.GetID())
            slicer.mrmlScene.RemoveNode(self.sliderLineSeries)
            slicer.mrmlScene.RemoveNode(self.sliderLineTable)

        if hasattr(self, 'regionSeries'):
            self.currentPlotChartNode.RemovePlotSeriesNodeID(self.regionSeries.GetID())
            slicer.mrmlScene.RemoveNode(self.regionSeries)
            slicer.mrmlScene.RemoveNode(self.regionTable)

        # Create new vertical line
        self.sliderLineTable = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTableNode")
        self.sliderLineTable.SetName("PCA_SliderLine_Data_Updated")

        lineX = vtk.vtkFloatArray()
        lineX.SetName("LineX")
        lineX.InsertNextValue(pc1_position)
        lineX.InsertNextValue(pc1_position)

        lineY = vtk.vtkFloatArray()
        lineY.SetName("LineY")
        lineY.InsertNextValue(self.pc2_min)
        lineY.InsertNextValue(self.pc2_max)

        self.sliderLineTable.AddColumn(lineX)
        self.sliderLineTable.AddColumn(lineY)

        self.sliderLineSeries = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLPlotSeriesNode")
        self.sliderLineSeries.SetName("SliderPosition_Updated")
        self.sliderLineSeries.SetAndObserveTableNodeID(self.sliderLineTable.GetID())
        self.sliderLineSeries.SetXColumnName("LineX")
        self.sliderLineSeries.SetYColumnName("LineY")
        self.sliderLineSeries.SetPlotType(slicer.vtkMRMLPlotSeriesNode.PlotTypeLine)
        self.sliderLineSeries.SetLineStyle(slicer.vtkMRMLPlotSeriesNode.LineStyleSolid)
        self.sliderLineSeries.SetLineWidth(3.0)
        self.sliderLineSeries.SetColor(1.0, 0.2, 0.2)  # Red line

        # Add to chart
        self.currentPlotChartNode.AddAndObservePlotSeriesNodeID(self.sliderLineSeries.GetID())

        # Create shaded region
        region_width = 0.5 * pc1_std
        left_x = pc1_position - region_width
        right_x = pc1_position + region_width

        self.regionTable = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTableNode")
        self.regionTable.SetName("PCA_Region_Data_Updated")

        regionX = vtk.vtkFloatArray()
        regionX.SetName("RegionX")
        regionY = vtk.vtkFloatArray()
        regionY.SetName("RegionY")

        # Rectangle coordinates
        regionX.InsertNextValue(left_x)
        regionY.InsertNextValue(self.pc2_min)
        regionX.InsertNextValue(right_x)
        regionY.InsertNextValue(self.pc2_min)
        regionX.InsertNextValue(right_x)
        regionY.InsertNextValue(self.pc2_max)
        regionX.InsertNextValue(left_x)
        regionY.InsertNextValue(self.pc2_max)
        regionX.InsertNextValue(left_x)
        regionY.InsertNextValue(self.pc2_min)

        self.regionTable.AddColumn(regionX)
        self.regionTable.AddColumn(regionY)

        self.regionSeries = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLPlotSeriesNode")
        self.regionSeries.SetName("SliderRegion_Updated")
        self.regionSeries.SetAndObserveTableNodeID(self.regionTable.GetID())
        self.regionSeries.SetXColumnName("RegionX")
        self.regionSeries.SetYColumnName("RegionY")
        self.regionSeries.SetPlotType(slicer.vtkMRMLPlotSeriesNode.PlotTypeLine)
        self.regionSeries.SetLineStyle(slicer.vtkMRMLPlotSeriesNode.LineStyleSolid)
        self.regionSeries.SetLineWidth(1.0)
        self.regionSeries.SetColor(0.8, 0.8, 0.2)  # Yellow
        self.regionSeries.SetOpacity(0.3)

        # Add to chart
        self.currentPlotChartNode.AddAndObservePlotSeriesNodeID(self.regionSeries.GetID())

    def updateVisualization(self):
        """Updates the PCA plot visualization"""

        if not self.pca_data:
            return

        # Creates native Slicer plot
        self.createPCAPlot()

    def onColorClicked(self, colorWidget):
        """Handles click on color swatch to display information"""

        # Get stored properties
        rgb = colorWidget.property("rgb")
        vertex_index = colorWidget.property("vertex_index")
        sd_position = colorWidget.property("sd_position")
        pc_index = colorWidget.property("pc_index")

        if rgb:
            r, g, b = rgb

            # Calculate HSV values
            h, s, v = colorsys.rgb_to_hsv(r/255.0, g/255.0, b/255.0)
            h_degrees = int(h * 360)
            s_percent = int(s * 100)
            v_percent = int(v * 100)

            # Format detailed information
            info_text = f"""
<b>Selected Color Details:</b><br/>
<br/>
<b>Location:</b> Vertex #{vertex_index}<br/>
<br/>
<b>RGB Values:</b><br/>
R: {r} | G: {g} | B: {b}<br/>
<br/>
<b>HSV Values:</b><br/>
H: {h_degrees}° | S: {s_percent}% | V: {v_percent}%<br/>
<br/>
<b>Hex Code:</b> #{r:02X}{g:02X}{b:02X}<br/>
<br/>
<b>PC Position:</b> {sd_position:+.2f} SD<br/>
<b>Current PC:</b> PC{pc_index + 1}
"""

            # Update the info display
            self.colorInfoText.setText(info_text)

            # Calculate luminance to determine text color
            # Using relative luminance formula
            luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0
            text_color = "white" if luminance < 0.5 else "black"

            # Create a color preview with appropriate text color
            preview_style = f"background-color: rgb({r}, {g}, {b}); color: {text_color}; border: 2px solid black; padding: 5px; font-weight: bold;"
            self.colorInfoTitle.setStyleSheet(preview_style)

    def updateColorDisplay(self, sd_position):
        """Updates color swatch display based on PC position

        Args:
            sd_position: Position along PC in standard deviations
        """

        if not self.pca_data:
            return

        # Gets selected PC
        pc_index = self.pcSelectCombo.currentIndex
        if pc_index < 0:
            return

        # Calculates colors at this position
        colors = self.logic.interpolateColors(pc_index, sd_position)

        # Updates color display label
        self.colorDisplayLabel.text = f"Colors at PC{pc_index+1} = {sd_position:+.2f} SD"

        # Reset info display
        self.colorInfoTitle.setText("Color Information")
        self.colorInfoTitle.setStyleSheet("font-weight: bold; padding: 5px;")
        self.colorInfoText.setText("Click on a color to see details")

        # Clears existing color swatches
        while self.colorGridLayout.count():
            item = self.colorGridLayout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Creates color swatches
        if colors is not None:
            # Sample colors for display (show subset for performance)
            n_display = min(100, len(colors))  # Show up to 100 color samples
            step = max(1, len(colors) // n_display)

            count = 0
            for i in range(0, len(colors), step):
                if i >= len(colors) or count >= n_display:
                    break

                # Creates color swatch as QPushButton for better interaction
                colorWidget = qt.QPushButton()
                colorWidget.setFixedSize(25, 25)  # Medium size
                colorWidget.setFlat(True)  # Flat button appearance
                colorWidget.setEnabled(True)  # Keep enabled for interaction

                # Converts color to RGB 0-255
                r = int(colors[i, 0] * 255)
                g = int(colors[i, 1] * 255)
                b = int(colors[i, 2] * 255)

                # Sets background color with hover effect
                colorWidget.setStyleSheet(f"""
                    QPushButton {{
                        background-color: rgb({r}, {g}, {b});
                        border: 1px solid gray;
                    }}
                    QPushButton:hover {{
                        border: 2px solid black;
                    }}
                    QPushButton:pressed {{
                        border: 3px solid #4CAF50;
                    }}
                """)

                # Store color info as properties for click handler
                colorWidget.setProperty("rgb", (r, g, b))
                colorWidget.setProperty("vertex_index", i)
                colorWidget.setProperty("sd_position", sd_position)
                colorWidget.setProperty("pc_index", pc_index)

                # Connect click event
                colorWidget.clicked.connect(lambda checked, widget=colorWidget: self.onColorClicked(widget))

                # Adds to grid with proper positioning
                row = count // 10
                col = count % 10
                self.colorGridLayout.addWidget(colorWidget, row, col)
                count += 1

        # Updates slider position indicator on plot if exists
        if hasattr(self, 'currentPlotChartNode') and self.currentPlotChartNode:
            # Could add a vertical line at current PC position
            # This would require modifying the plot data
            pass

    def onExportResults(self):
        """Handles export of PCA results"""

        exportDir = self.exportPathEdit.currentPath
        if not exportDir:
            exportDir = os.path.join(self.decaResultsSelector.currentPath, "PCA_Morphospace")

        os.makedirs(exportDir, exist_ok=True)

        try:
            # Exports PCA results
            self.logic.exportResults(exportDir)

            self.statusLabel.text = f"Status: Successfully exported to {os.path.basename(exportDir)}"
            self.statusLabel.setStyleSheet(ColorTheme.getStatusLabelStyle('success'))
            print(f"PCA results exported to: {exportDir}")

        except Exception as e:
            self.statusLabel.text = f"Status: Export failed - {str(e)}"
            self.statusLabel.setStyleSheet(ColorTheme.getStatusLabelStyle('error'))
            print(f"Error exporting results: {str(e)}")

    def createPCAPlot(self):
        """Creates PCA scatter plot using Slicer's native plotting"""

        if not self.pca_data:
            return None

        try:
            # Get selected axes
            x_axis_idx = self.xAxisCombo.currentIndex if hasattr(self, 'xAxisCombo') and self.xAxisCombo.enabled else 0
            y_axis_idx = self.yAxisCombo.currentIndex if hasattr(self, 'yAxisCombo') and self.yAxisCombo.enabled else 1

            # Make sure indices are valid
            max_idx = self.pca_data['transformed'].shape[1] - 1
            x_axis_idx = min(x_axis_idx, max_idx)
            y_axis_idx = min(y_axis_idx, max_idx)

            # Creates table node for data
            tableNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTableNode")
            tableNode.SetName("PCA_Morphospace_Data")

            # Adds X-axis PC data
            pcXArray = vtk.vtkFloatArray()
            pcXArray.SetName(f"PC{x_axis_idx + 1}")
            for score in self.pca_data['transformed'][:, x_axis_idx]:
                pcXArray.InsertNextValue(score)
            tableNode.AddColumn(pcXArray)

            # Adds Y-axis PC data
            pcYArray = vtk.vtkFloatArray()
            pcYArray.SetName(f"PC{y_axis_idx + 1}")
            if self.pca_data['transformed'].shape[1] > y_axis_idx:
                for score in self.pca_data['transformed'][:, y_axis_idx]:
                    pcYArray.InsertNextValue(score)
            else:
                for _ in range(len(self.pca_data['transformed'])):
                    pcYArray.InsertNextValue(0)
            tableNode.AddColumn(pcYArray)

            # Adds specimen labels
            labelsArray = vtk.vtkStringArray()
            labelsArray.SetName("Specimen")
            for name in self.pca_data['specimen_names']:
                labelsArray.InsertNextValue(name)
            tableNode.AddColumn(labelsArray)

            # Creates plot series node
            plotSeriesNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLPlotSeriesNode")
            plotSeriesNode.SetName("PCA_Morphospace_Series")
            plotSeriesNode.SetAndObserveTableNodeID(tableNode.GetID())
            plotSeriesNode.SetXColumnName(f"PC{x_axis_idx + 1}")
            plotSeriesNode.SetYColumnName(f"PC{y_axis_idx + 1}")
            plotSeriesNode.SetLabelColumnName("Specimen")
            plotSeriesNode.SetPlotType(slicer.vtkMRMLPlotSeriesNode.PlotTypeScatter)
            plotSeriesNode.SetMarkerStyle(slicer.vtkMRMLPlotSeriesNode.MarkerStyleCircle)
            plotSeriesNode.SetMarkerSize(8)
            plotSeriesNode.SetLineStyle(slicer.vtkMRMLPlotSeriesNode.LineStyleNone)
            plotSeriesNode.SetColor(0.2, 0.6, 0.8)

            # Creates chart node
            plotChartNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLPlotChartNode")
            plotChartNode.SetName("PCA_Morphospace_Chart")
            plotChartNode.AddAndObservePlotSeriesNodeID(plotSeriesNode.GetID())

            # Sets titles with variance explained for selected axes
            var_x = self.pca_data['variance_explained'][x_axis_idx] if len(self.pca_data['variance_explained']) > x_axis_idx else 0
            var_y = self.pca_data['variance_explained'][y_axis_idx] if len(self.pca_data['variance_explained']) > y_axis_idx else 0
            plotChartNode.SetTitle("PCA Color Morphospace")
            plotChartNode.SetXAxisTitle(f"PC{x_axis_idx + 1} ({var_x:.1f}%)")
            plotChartNode.SetYAxisTitle(f"PC{y_axis_idx + 1} ({var_y:.1f}%)")

            # Store PC data range for reference
            self.pc2_min = float(min(self.pca_data['transformed'][:, 1])) if self.pca_data['transformed'].shape[1] > 1 else -10
            self.pc2_max = float(max(self.pca_data['transformed'][:, 1])) if self.pca_data['transformed'].shape[1] > 1 else 10

            # Shows in plot viewer
            layoutManager = slicer.app.layoutManager()
            layoutManager.setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutOneUpPlotView)
            plotWidget = layoutManager.plotWidget(0)
            if plotWidget:
                plotViewNode = plotWidget.mrmlPlotViewNode()
                if plotViewNode:
                    plotViewNode.SetPlotChartNodeID(plotChartNode.GetID())

            # Stores references
            self.currentPlotChartNode = plotChartNode
            self.pcaDataTable = tableNode  # Store table for later use

            return plotChartNode

        except Exception as e:
            slicer.util.errorDisplay(f"Error creating plot: {str(e)}")
            return None


class PCAMorphospaceLogic:
    """Logic class for PCA Morphospace analysis"""

    def __init__(self):
        """Initializes the PCA logic"""
        self.specimens = {}
        self.vertex_colors = None
        self.face_areas = None
        self.pca_model = None
        self.mean_colors = None
        self.transformed_data = None

    def loadSpecimenData(self, resultsDir):
        """Loads resampled models and textures from DeCA results

        Args:
            resultsDir: Path to DeCA results directory

        Returns:
            bool: Success status
        """

        # Finds resampled models directory
        resampledDir = os.path.join(resultsDir, "DeCA", "resampledModels")
        if not os.path.exists(resampledDir):
            raise ValueError(f"Resampled models directory not found: {resampledDir}")

        # Finds texture directory
        textureDir = os.path.join(resultsDir, "colorAnalysis", "atlasTextures")
        if not os.path.exists(textureDir):
            raise ValueError(f"Texture directory not found: {textureDir}")

        # Loads each specimen
        self.specimens = {}
        model_files = [f for f in os.listdir(resampledDir) if f.endswith(('.ply', '.vtk', '.vtp'))]

        for model_file in model_files:
            specimen_name = os.path.splitext(model_file)[0].replace('_resampled', '')

            # Loads model
            model_path = os.path.join(resampledDir, model_file)

            # Finds corresponding texture
            texture_path = None
            for tex_file in os.listdir(textureDir):
                if tex_file.startswith(specimen_name):
                    texture_path = os.path.join(textureDir, tex_file)
                    break

            if texture_path and os.path.exists(texture_path):
                self.specimens[specimen_name] = {
                    'model': model_path,
                    'texture': texture_path
                }

        return len(self.specimens) > 0

    def extractVertexColors(self, model_path, texture_path):
        """Extracts vertex colors from model and texture

        Args:
            model_path: Path to model file
            texture_path: Path to texture file

        Returns:
            numpy array: Vertex colors (N x 3)
        """

        # Loads model
        reader = vtk.vtkPLYReader() if model_path.endswith('.ply') else vtk.vtkPolyDataReader()
        reader.SetFileName(model_path)
        reader.Update()
        polydata = reader.GetOutput()

        # Gets UV coordinates
        uvs = polydata.GetPointData().GetTCoords()
        if not uvs:
            raise ValueError(f"No UV coordinates found in {model_path}")

        # Loads texture image
        import imageio
        texture_img = imageio.imread(texture_path)
        height, width = texture_img.shape[:2]

        # Extracts colors at UV coordinates
        n_vertices = polydata.GetNumberOfPoints()
        vertex_colors = np.zeros((n_vertices, 3))

        for i in range(n_vertices):
            u, v = uvs.GetTuple2(i)
            # Converts UV to pixel coordinates
            x = int(u * (width - 1))
            y = int((1 - v) * (height - 1))  # Flip V coordinate

            # Gets color at this position
            if 0 <= x < width and 0 <= y < height:
                vertex_colors[i] = texture_img[y, x, :3] / 255.0  # Normalize to [0,1]

        return vertex_colors

    def performPCA(self, n_components=3, color_space='RGB', area_weighted=True):
        """Performs PCA on specimen colors

        Args:
            n_components: Number of principal components
            color_space: Color space for analysis
            area_weighted: Whether to weight by face area

        Returns:
            dict: PCA results
        """

        from sklearn.decomposition import PCA

        # Collects vertex colors from all specimens
        all_colors = []
        specimen_names = []

        for name, paths in self.specimens.items():
            colors = self.extractVertexColors(paths['model'], paths['texture'])

            # Converts color space if needed
            if color_space == 'HSV':
                colors = self.rgbToHsv(colors)
            elif color_space == 'LAB':
                colors = self.rgbToLab(colors)

            all_colors.append(colors.flatten())  # Flattens to 1D vector
            specimen_names.append(name)

        # Stacks into matrix (specimens x features)
        color_matrix = np.vstack(all_colors)

        # Performs PCA
        pca = PCA(n_components=n_components)
        transformed = pca.fit_transform(color_matrix)

        # Stores results
        self.pca_model = pca
        self.mean_colors = pca.mean_
        self.transformed_data = transformed

        return {
            'specimen_names': specimen_names,
            'transformed': transformed,
            'variance_explained': pca.explained_variance_ratio_ * 100,
            'components': pca.components_,
            'mean': pca.mean_
        }

    def interpolateColors(self, pc_index, sd_position):
        """Interpolates colors at a position along a PC axis

        Args:
            pc_index: Which PC to move along (0-based)
            sd_position: Position in standard deviations from mean

        Returns:
            numpy array: Interpolated colors
        """

        if not self.pca_model:
            return None

        # Calculates position along PC
        # mean + sd_position * sqrt(eigenvalue) * eigenvector
        std_dev = np.sqrt(self.pca_model.explained_variance_[pc_index])
        shift = sd_position * std_dev * self.pca_model.components_[pc_index]

        # Applies shift to mean colors
        interpolated = self.mean_colors + shift

        # Clips to valid color range
        interpolated = np.clip(interpolated, 0, 1)

        # Reshapes to vertex colors
        n_vertices = len(interpolated) // 3
        return interpolated.reshape(n_vertices, 3)

    def rgbToHsv(self, rgb):
        """Converts RGB colors to HSV

        Args:
            rgb: RGB colors (N x 3)

        Returns:
            numpy array: HSV colors
        """
        import colorsys
        hsv = np.zeros_like(rgb)
        for i in range(len(rgb)):
            hsv[i] = colorsys.rgb_to_hsv(*rgb[i])
        return hsv

    def rgbToLab(self, rgb):
        """Converts RGB colors to LAB

        Args:
            rgb: RGB colors (N x 3)

        Returns:
            numpy array: LAB colors
        """
        # Simplified - would use proper color space conversion
        # For now, returns RGB unchanged
        return rgb

    def exportResults(self, exportDir):
        """Exports PCA results to files

        Args:
            exportDir: Directory for exports
        """

        if self.pca_model is None or self.transformed_data is None:
            raise ValueError("No PCA results to export")

        # Exports PCA scores
        scores_file = os.path.join(exportDir, "pca_scores.csv")
        with open(scores_file, 'w') as f:
            f.write("Specimen,PC1,PC2,PC3\n")
            for i, name in enumerate(self.specimens.keys()):
                scores = self.transformed_data[i]
                f.write(f"{name},{scores[0]},{scores[1]},{scores[2] if len(scores) > 2 else 0}\n")

        # Exports variance explained
        variance_file = os.path.join(exportDir, "variance_explained.txt")
        with open(variance_file, 'w') as f:
            for i, var in enumerate(self.pca_model.explained_variance_ratio_):
                f.write(f"PC{i+1}: {var*100:.2f}%\n")

        # Exports mean colors
        mean_file = os.path.join(exportDir, "mean_colors.npy")
        np.save(mean_file, self.mean_colors)

        # Exports loadings
        loadings_file = os.path.join(exportDir, "pca_loadings.npy")
        np.save(loadings_file, self.pca_model.components_)
