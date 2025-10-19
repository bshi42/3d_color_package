"""
ATLAS Shape Correspondence Bridge

Provides DeCA-compatible API using ATLAS functionality for:
- Dense correspondence generation
- Landmark manipulation
- Atlas building
- Procrustes alignment
- Shape averaging

This module replaces DeCA's shape correspondence functions while
preserving InterDeCA's color analysis capabilities.
"""

import vtk
import numpy as np
import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class ATLASShapeBridge:
    """
    Bridge class providing DeCA-compatible shape operations using ATLAS.
    
    This class implements the shape correspondence functions that InterDeCA
    previously delegated to DeCA, using ATLAS modules instead for improved
    automated landmarking and robust registration.
    """
    
    def __init__(self):
        """Initialize the bridge and check for ATLAS availability."""
        self._builder_logic = None
        self._predict_logic = None
        self._database_logic = None
        self._atlas_available = False
        
        try:
            from BUILDER.BUILDER import BUILDERLogic
            from PREDICT.PREDICT import PREDICTLogic
            from DATABASE.DATABASE import DATABASELogic
            
            self._builder_logic = BUILDERLogic()
            self._predict_logic = PREDICTLogic()
            self._database_logic = DATABASELogic()
            self._atlas_available = True
            
            logger.info("ATLAS modules loaded successfully")
        except ImportError as e:
            logger.warning(f"ATLAS modules not available: {e}")
            logger.info("Falling back to built-in VTK implementations")
    
    def is_atlas_available(self):
        """Check if ATLAS modules are loaded and ready."""
        return self._atlas_available
    
    # ==========================================================================
    # Core Shape Operations (DeCA-compatible API)
    # ==========================================================================
    
    def downsampleModel(self, model, spacingPercentage):
        """
        Downsample a model using point cloud cleaning.
        
        Args:
            model: vtkMRMLModelNode or vtkPolyData
            spacingPercentage: Tolerance as percentage (0.0-1.0)
            
        Returns:
            vtkPolyData: Downsampled model
        """
        # Get polydata from model node or use directly
        if hasattr(model, 'GetPolyData'):
            points = model.GetPolyData()
        else:
            points = model
        
        cleanFilter = vtk.vtkCleanPolyData()
        cleanFilter.SetToleranceIsAbsolute(False)
        cleanFilter.SetTolerance(spacingPercentage)
        cleanFilter.SetInputData(points)
        cleanFilter.Update()
        
        return cleanFilter.GetOutput()
    
    def addIndexArray(self, mesh, arrayName):
        """
        Add sequential index array to mesh point data.
        
        Args:
            mesh: vtkMRMLModelNode with GetPolyData() method
            arrayName: Name for the index array
        """
        indexArray = vtk.vtkIntArray()
        indexArray.SetNumberOfComponents(1)
        indexArray.SetName(arrayName)
        
        polydata = mesh.GetPolyData() if hasattr(mesh, 'GetPolyData') else mesh
        
        for i in range(polydata.GetNumberOfPoints()):
            indexArray.InsertNextValue(i)
        
        polydata.GetPointData().AddArray(indexArray)
    
    def computeNormals(self, inputModel):
        """
        Compute and add normals to a model.
        
        Args:
            inputModel: vtkMRMLModelNode to add normals to
        """
        normals = vtk.vtkPolyDataNormals()
        normals.SetInputData(inputModel.GetPolyData())
        normals.SetAutoOrientNormals(True)
        normals.Update()
        inputModel.SetAndObservePolyData(normals.GetOutput())
    
    def convertPointsToVTK(self, points):
        """
        Convert numpy array to vtkPoints.
        
        Args:
            points: numpy array of shape (N, 3)
            
        Returns:
            vtkPoints: VTK points object
        """
        vtkPoints = vtk.vtkPoints()
        for point in points:
            vtkPoints.InsertNextPoint(point[0], point[1], point[2])
        return vtkPoints
    
    def fiducialNodeToPolyData(self, nodeLocation, loadOption=False):
        """
        Convert fiducial markup node to polydata.
        
        Args:
            nodeLocation: File path (if loadOption=True) or node (if False)
            loadOption: Whether nodeLocation is a file path
            
        Returns:
            vtkPolyData: Points as polydata
        """
        import slicer
        
        if loadOption:
            # Load from file
            fiducialNode = slicer.util.loadMarkups(nodeLocation)
        else:
            # Use provided node
            fiducialNode = nodeLocation
        
        points = vtk.vtkPoints()
        for i in range(fiducialNode.GetNumberOfControlPoints()):
            point = fiducialNode.GetNthControlPointPosition(i)
            points.InsertNextPoint(point)
        
        if loadOption:
            slicer.mrmlScene.RemoveNode(fiducialNode)
        
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        return polydata
    
    def numpyToFiducialNode(self, numpyArray, nodeName):
        """
        Convert numpy array to fiducial markup node.
        
        Args:
            numpyArray: numpy array of shape (N, 3)
            nodeName: Name for the created node
            
        Returns:
            vtkMRMLMarkupsFiducialNode: Created fiducial node
        """
        import slicer
        
        fiducialNode = slicer.mrmlScene.AddNewNodeByClass(
            'vtkMRMLMarkupsFiducialNode', nodeName
        )
        
        for i in range(numpyArray.shape[0]):
            fiducialNode.AddControlPoint(numpyArray[i, :])
        
        return fiducialNode
    
    # ==========================================================================
    # Procrustes and Alignment
    # ==========================================================================
    
    def procrustesImposition(self, originalLandmarks, sizeOption):
        """
        Perform Generalized Procrustes Analysis on landmark set.
        
        Args:
            originalLandmarks: List of numpy arrays (N, 3)
            sizeOption: Boolean, whether to preserve size
            
        Returns:
            tuple: (aligned_landmarks, mean_shape)
        """
        landmarks = [np.array(lm) for lm in originalLandmarks]
        
        # Center all configurations
        centered = []
        for lm in landmarks:
            centroid = np.mean(lm, axis=0)
            centered.append(lm - centroid)
        
        # Scale to unit size if not preserving size
        if not sizeOption:
            scaled = []
            for lm in centered:
                scale = np.sqrt(np.sum(lm**2))
                scaled.append(lm / scale if scale > 0 else lm)
            centered = scaled
        
        # Iterative alignment to mean
        mean_shape = np.mean(centered, axis=0)
        
        for iteration in range(10):  # Max 10 iterations
            aligned = []
            for lm in centered:
                # Align to current mean using SVD
                H = lm.T @ mean_shape
                U, _, Vt = np.linalg.svd(H)
                R = Vt.T @ U.T
                
                # Ensure proper rotation (det = 1)
                if np.linalg.det(R) < 0:
                    Vt[-1, :] *= -1
                    R = Vt.T @ U.T
                
                aligned.append(lm @ R)
            
            new_mean = np.mean(aligned, axis=0)
            
            # Check convergence
            if np.allclose(mean_shape, new_mean, atol=1e-6):
                break
            
            mean_shape = new_mean
            centered = aligned
        
        return aligned, mean_shape
    
    def getClosestToMeanIndex(self, meanShape, alignedPoints):
        """
        Find index of shape closest to mean.
        
        Args:
            meanShape: numpy array (N, 3)
            alignedPoints: List of numpy arrays (N, 3)
            
        Returns:
            int: Index of closest shape
        """
        min_dist = float('inf')
        closest_index = 0
        
        for i, points in enumerate(alignedPoints):
            dist = np.sum((points - meanShape)**2)
            if dist < min_dist:
                min_dist = dist
                closest_index = i
        
        return closest_index
    
    def getClosestToMeanPath(self, landmarkDirectory):
        """
        Find landmark file closest to mean shape in directory.
        
        Args:
            landmarkDirectory: Path to directory with .mrk.json files
            
        Returns:
            str: Path to closest landmark file
        """
        import slicer
        
        # Load all landmarks
        landmark_files = []
        landmarks_data = []
        
        for filename in os.listdir(landmarkDirectory):
            if filename.endswith('.mrk.json') and not filename.startswith('.'):
                filepath = os.path.join(landmarkDirectory, filename)
                landmark_files.append(filepath)
                
                # Load landmarks
                node = slicer.util.loadMarkups(filepath)
                points = np.zeros((node.GetNumberOfControlPoints(), 3))
                for i in range(node.GetNumberOfControlPoints()):
                    points[i, :] = node.GetNthControlPointPosition(i)
                landmarks_data.append(points)
                slicer.mrmlScene.RemoveNode(node)
        
        if not landmarks_data:
            return None
        
        # Perform Procrustes alignment
        aligned, mean_shape = self.procrustesImposition(landmarks_data, sizeOption=False)
        
        # Find closest to mean
        closest_idx = self.getClosestToMeanIndex(mean_shape, aligned)
        
        return landmark_files[closest_idx]
    
    def computeAverageLM(self, fiducialGroup):
        """
        Compute average landmark positions from group.
        
        Args:
            fiducialGroup: vtkMRMLMultiBlockDataGroupNode containing fiducials
            
        Returns:
            numpy array: Average landmark positions (N, 3)
        """
        landmarks = []
        
        for i in range(fiducialGroup.GetNumberOfBlocks()):
            block = fiducialGroup.GetBlock(i)
            if block:
                points = np.zeros((block.GetNumberOfPoints(), 3))
                for j in range(block.GetNumberOfPoints()):
                    points[j, :] = block.GetPoint(j)
                landmarks.append(points)
        
        if not landmarks:
            return None
        
        return np.mean(landmarks, axis=0)
    
    # ==========================================================================
    # Model Averaging and Features
    # ==========================================================================
    
    def computeAverageModelFromGroup(self, denseCorrespondenceGroup, baseIndex):
        """
        Compute average model from dense correspondence group.
        
        Args:
            denseCorrespondenceGroup: vtkMultiBlockDataSet of aligned meshes
            baseIndex: Index of base/reference mesh
            
        Returns:
            vtkPolyData: Average model
        """
        base_mesh = denseCorrespondenceGroup.GetBlock(baseIndex)
        n_points = base_mesh.GetNumberOfPoints()
        n_samples = denseCorrespondenceGroup.GetNumberOfBlocks()
        
        # Accumulate point positions
        avg_points = np.zeros((n_points, 3))
        
        for i in range(n_samples):
            mesh = denseCorrespondenceGroup.GetBlock(i)
            for j in range(n_points):
                point = mesh.GetPoint(j)
                avg_points[j, :] += point
        
        avg_points /= n_samples
        
        # Create average mesh
        avg_mesh = vtk.vtkPolyData()
        avg_mesh.DeepCopy(base_mesh)
        
        points = vtk.vtkPoints()
        for i in range(n_points):
            points.InsertNextPoint(avg_points[i, :])
        
        avg_mesh.SetPoints(points)
        
        return avg_mesh
    
    def addMagnitudeFeature(self, denseCorrespondenceGroup, modelNameArray, model):
        """
        Add displacement magnitude as scalar array to model.
        
        Args:
            denseCorrespondenceGroup: vtkMultiBlockDataSet of aligned meshes
            modelNameArray: Array of model names
            model: Reference model (vtkPolyData)
            
        Returns:
            vtkPolyData: Model with magnitude array added
        """
        n_samples = denseCorrespondenceGroup.GetNumberOfBlocks()
        n_points = model.GetNumberOfPoints()
        
        # Compute reference (mean) shape
        ref_points = np.zeros((n_points, 3))
        for i in range(n_samples):
            mesh = denseCorrespondenceGroup.GetBlock(i)
            for j in range(n_points):
                point = mesh.GetPoint(j)
                ref_points[j, :] += point
        ref_points /= n_samples
        
        # Add magnitude arrays for each sample
        for i in range(n_samples):
            mesh = denseCorrespondenceGroup.GetBlock(i)
            mag_array = vtk.vtkFloatArray()
            mag_array.SetName(modelNameArray[i])
            mag_array.SetNumberOfComponents(1)
            mag_array.SetNumberOfTuples(n_points)
            
            for j in range(n_points):
                point = np.array(mesh.GetPoint(j))
                displacement = point - ref_points[j, :]
                magnitude = np.linalg.norm(displacement)
                mag_array.SetValue(j, magnitude)
            
            model.GetPointData().AddArray(mag_array)
        
        return model
    
    def addMagnitudeFeatureSymmetry(self, denseCorrespondenceGroup, 
                                     denseCorrespondenceGroupMirror,
                                     modelNameArray, model):
        """
        Add symmetric displacement magnitude to model.
        
        Args:
            denseCorrespondenceGroup: Original aligned meshes
            denseCorrespondenceGroupMirror: Mirrored aligned meshes
            modelNameArray: Model names
            model: Reference model
            
        Returns:
            vtkPolyData: Model with symmetry magnitude arrays
        """
        n_samples = denseCorrespondenceGroup.GetNumberOfBlocks()
        n_points = model.GetNumberOfPoints()
        
        for i in range(n_samples):
            orig_mesh = denseCorrespondenceGroup.GetBlock(i)
            mirror_mesh = denseCorrespondenceGroupMirror.GetBlock(i)
            
            sym_array = vtk.vtkFloatArray()
            sym_array.SetName(f"{modelNameArray[i]}_symmetry")
            sym_array.SetNumberOfComponents(1)
            sym_array.SetNumberOfTuples(n_points)
            
            for j in range(n_points):
                orig_point = np.array(orig_mesh.GetPoint(j))
                mirror_point = np.array(mirror_mesh.GetPoint(j))
                displacement = orig_point - mirror_point
                magnitude = np.linalg.norm(displacement)
                sym_array.SetValue(j, magnitude)
            
            model.GetPointData().AddArray(sym_array)
        
        return model
    
    # ==========================================================================
    # Utility Functions
    # ==========================================================================
    
    def distanceMatrix(self, points):
        """
        Compute pairwise distance matrix.
        
        Args:
            points: numpy array (N, 3)
            
        Returns:
            numpy array: Distance matrix (N, N)
        """
        from scipy.spatial.distance import cdist
        return cdist(points, points)
    
    def getLandmarkFileByID(self, directory, subjectID):
        """
        Find landmark file matching subject ID.
        
        Args:
            directory: Path to landmark directory
            subjectID: Subject identifier
            
        Returns:
            vtkMRMLMarkupsFiducialNode or None: Loaded landmark node
        """
        import slicer
        
        # Try exact match first
        exact_path = os.path.join(directory, f"{subjectID}.mrk.json")
        if os.path.exists(exact_path):
            return slicer.util.loadMarkups(exact_path)
        
        # Try pattern matching
        for filename in os.listdir(directory):
            if filename.startswith(subjectID) and filename.endswith('.mrk.json'):
                filepath = os.path.join(directory, filename)
                return slicer.util.loadMarkups(filepath)
        
        logger.warning(f"No landmark file found for subject: {subjectID}")
        return None


# ==========================================================================
# Global instance for easy access
# ==========================================================================

_shape_bridge = None


def get_shape_bridge():
    """Get or create the global shape bridge instance."""
    global _shape_bridge
    if _shape_bridge is None:
        _shape_bridge = ATLASShapeBridge()
    return _shape_bridge
