"""Slicer-side compatibility facade for the clean-room InterDeCA package."""

import glob
import logging
import operator
import os
import re
import subprocess
import tempfile
import textwrap
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import vtk
from vtk.util import numpy_support as vtk_np

from color_deca.interdeca.core.dependencies import detect_dependencies

try:
    import slicer
except ModuleNotFoundError:
    slicer = None

try:
    from color_deca.atlas_integration.shape_bridge import get_shape_bridge
except ImportError:
    try:
        from atlas_integration.shape_bridge import get_shape_bridge
    except ImportError:
        get_shape_bridge = None


dependency_status = detect_dependencies()
SKLEARN_AVAILABLE = dependency_status.sklearn_available
UMAP_AVAILABLE = dependency_status.umap_available
SCIPY_AVAILABLE = dependency_status.scipy_available
SKIMAGE_AVAILABLE = dependency_status.skimage_available

atlasShapeBridge = get_shape_bridge() if get_shape_bridge else None


def _require_slicer():
    if slicer is None:
        raise RuntimeError("InterDeCA Slicer facade must be used inside 3D Slicer")


class InterDeCALogicFacade:
    """Compatibility facade exposing the legacy InterDeCALogic E2E surface."""

    def __init__(self, **services):
        self.services = services
        self.modelNames = []

    def get_service(self, name: str):
        try:
            return self.services[name]
        except KeyError as exc:
            raise KeyError(f"Service is not configured: {name}") from exc

    def runDCAlign(
        self,
        baseMeshPath,
        baseLMPath,
        alignedMeshDir,
        landmarkDirectory,
        outputDirectory,
        optionErrorOutput,
        atlas_uv_template_obj=None,
    ):
        _require_slicer()
        baseNode = self._load_model_with_cs(baseMeshPath, "RAS")
        baseMesh = baseNode.GetPolyData()
        baseLandmarks = self.fiducialNodeToPolyData(baseLMPath).GetPoints()

        landmarkNames, landmarks = self.importLandmarks(landmarkDirectory)
        self.modelNames, models = self.importMeshes(alignedMeshDir, ["ply", "stl", "vtp", "obj"], restrict_to=landmarkNames)

        if len(self.modelNames) != len(landmarkNames):
            missing_mesh = [name for name in landmarkNames if name not in self.modelNames]
            extra_mesh = [name for name in self.modelNames if name not in landmarkNames]
            raise ValueError(
                "Mismatch between meshes and landmarks.\n"
                f"Missing mesh for: {missing_mesh}\nExtra mesh: {extra_mesh}"
            )

        denseCorrespondenceGroup = self.denseCorrespondenceBaseMesh(landmarks, models, baseMesh, baseLandmarks)
        self.addMagnitudeFeature(denseCorrespondenceGroup, self.modelNames, baseMesh)
        outputModelPath = os.path.join(outputDirectory, "ATLAS", "atlasResultModel.vtp")

        try:
            if baseNode and slicer.mrmlScene.IsNodePresent(baseNode):
                slicer.util.saveNode(baseNode, outputModelPath)
                print(f"Successfully saved ATLAS result model to: {outputModelPath}")
            else:
                print(f"Warning: baseNode is not valid or not in scene, skipping save to {outputModelPath}")
        except Exception as exc:
            print(f"Warning: Failed to save ATLAS result model to {outputModelPath}: {exc}")

        slicer.mrmlScene.RemoveNode(baseNode)

        resampledModelPath = os.path.join(outputDirectory, "ATLAS", "resampledModels")
        if os.path.exists(resampledModelPath):
            tempModelNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", "tempResampledModel")
            outOBJdir = os.path.join(outputDirectory, "colorAnalysis", "resampledOBJ_withUV")
            os.makedirs(outOBJdir, exist_ok=True)

            for index in range(denseCorrespondenceGroup.GetNumberOfBlocks()):
                resampledMesh = denseCorrespondenceGroup.GetBlock(index)
                subjectName = self.modelNames[index].replace("_align", "")

                outputFileName = os.path.join(resampledModelPath, f"{subjectName}_resampled.ply")
                tempModelNode.SetAndObservePolyData(resampledMesh)
                self._save_model_with_cs(tempModelNode, outputFileName, "RAS")

                if atlas_uv_template_obj and os.path.isfile(atlas_uv_template_obj):
                    out_obj = os.path.join(outOBJdir, f"{subjectName}_resampled.obj")
                    self.write_obj_with_uv_from_template(resampledMesh, atlas_uv_template_obj, out_obj)

            slicer.mrmlScene.RemoveNode(tempModelNode)

        try:
            if baseNode and slicer.mrmlScene.IsNodePresent(baseNode):
                slicer.mrmlScene.RemoveNode(baseNode)
        except Exception as exc:
            print(f"Warning: Failed to remove baseNode: {exc}")

    def runMean(self, landmarkDirectory, meshDirectory):
        _require_slicer()
        landmarkNames, landmarks = self.importLandmarks(landmarkDirectory)
        self.modelNames, models = self.importMeshes(meshDirectory, ["ply", "stl", "vtp", "vtk", "obj"], restrict_to=landmarkNames)
        denseCorrespondenceGroup, closestToMeanIndex = self.denseCorrespondence(landmarks, models)
        print("Sample closest to mean: ", closestToMeanIndex)
        averagePolyData = self.computeAverageModelFromGroup(denseCorrespondenceGroup, closestToMeanIndex)
        averageModelNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", "Atlas Model")
        averageModelNode.CreateDefaultDisplayNodes()
        averageModelNode.SetAndObservePolyData(averagePolyData)
        averageLandmarkNode = self.computeAverageLM(landmarks)
        averageLandmarkNode.GetDisplayNode().SetPointLabelsVisibility(False)
        return averageModelNode, averageLandmarkNode

    def getLandmarkFileByID(self, directory, subjectID):
        _require_slicer()
        if atlasShapeBridge:
            try:
                return atlasShapeBridge.getLandmarkFileByID(directory, subjectID)
            except Exception as exc:
                print(f"Error using atlasShapeBridge.getLandmarkFileByID: {exc}")

        for fileName in os.listdir(directory):
            fileNameBase = Path(fileName)
            while fileNameBase.suffix in {".fcsv", ".mrk", ".json"}:
                fileNameBase = fileNameBase.with_suffix("")
            if subjectID == str(fileNameBase):
                filePath = os.path.join(directory, fileName)
                try:
                    return slicer.util.loadMarkups(filePath)
                except Exception as exc:
                    print(f"Error loading landmarks from {filePath}: {exc}")
                    return None
        print(f"No landmarks found for subject ID '{subjectID}' in {directory}")
        return None

    def getModelFileByID(self, directory, subjectID):
        _require_slicer()
        model_extensions = [".ply", ".stl", ".obj", ".vtk", ".vtp"]
        for fileName in os.listdir(directory):
            name_parts = fileName.split(".", 1)
            if len(name_parts) < 2:
                continue
            first_ext = "." + name_parts[1].split(".")[0]
            file_ext = first_ext.lower()
            if file_ext not in model_extensions:
                continue
            fileNameBase = name_parts[0]
            if str(fileNameBase).startswith(str(subjectID)):
                filePath = os.path.join(directory, fileName)
                try:
                    return self._load_model_with_cs(filePath, "RAS")
                except Exception as exc:
                    print(f"Error loading model from {filePath}: {exc}")
                    return None
        print(f"No model found for subject ID '{subjectID}' in {directory}")
        return None

    def runAlign(
        self,
        baseMeshNode,
        baseLMNode,
        meshDirectory,
        lmDirectory,
        ouputMeshDirectory,
        outputLMDirectory,
        removeScaleOption,
        slmDirectory=False,
        outputSLMDirectory=False,
    ):
        _require_slicer()
        semilandmarkOption = bool(slmDirectory and outputSLMDirectory)
        targetPoints = vtk.vtkPoints()
        for index in range(baseLMNode.GetNumberOfControlPoints()):
            point = baseLMNode.GetNthControlPointPosition(index)
            targetPoints.InsertNextPoint(point)

        for meshFileName in os.listdir(meshDirectory):
            if meshFileName.startswith("."):
                continue
            meshFilePath = os.path.join(meshDirectory, meshFileName)
            name_parts = meshFileName.split(".", 1)
            subjectID = name_parts[0] if len(name_parts) > 1 else meshFileName
            currentLMNode = self.getLandmarkFileByID(lmDirectory, subjectID)
            if currentLMNode:
                try:
                    currentMeshNode = slicer.util.loadModel(meshFilePath)
                except Exception:
                    slicer.mrmlScene.RemoveNode(currentLMNode)
                    continue
                if currentLMNode.GetNumberOfControlPoints() != baseLMNode.GetNumberOfControlPoints():
                    raise ValueError(
                        f"Landmark points mismatch: subject has {currentLMNode.GetNumberOfControlPoints()} points, "
                        f"atlas has {baseLMNode.GetNumberOfControlPoints()} points"
                    )
                sourcePoints = vtk.vtkPoints()
                for index in range(currentLMNode.GetNumberOfControlPoints()):
                    point = currentLMNode.GetNthControlPointPosition(index)
                    sourcePoints.InsertNextPoint(point)
                transform = vtk.vtkLandmarkTransform()
                transform.SetSourceLandmarks(sourcePoints)
                transform.SetTargetLandmarks(targetPoints)
                transform.SetModeToRigidBody()

                transformNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTransformNode", "Alignment")
                transformNode.SetAndObserveTransformToParent(transform)
                currentMeshNode.SetAndObserveTransformNodeID(transformNode.GetID())
                currentLMNode.SetAndObserveTransformNodeID(transformNode.GetID())
                slicer.vtkSlicerTransformLogic().hardenTransform(currentMeshNode)
                slicer.vtkSlicerTransformLogic().hardenTransform(currentLMNode)

                outputMeshPath = os.path.join(ouputMeshDirectory, subjectID + "_align.ply")
                self._save_model_with_cs(currentMeshNode, outputMeshPath, "RAS")

                try:
                    tcoords = currentMeshNode.GetPolyData().GetPointData().GetTCoords()
                    has_uv = bool(tcoords) and tcoords.GetNumberOfTuples() > 0
                    if has_uv or os.path.splitext(meshFilePath)[1].lower() == ".obj":
                        outputOBJPath = os.path.join(ouputMeshDirectory, subjectID + "_align.obj")
                        self._save_model_with_cs(currentMeshNode, outputOBJPath, "RAS")
                except Exception as exc:
                    logging.warning(f"Could not save aligned OBJ for {subjectID}: {exc}")

                outputLMPath = os.path.join(outputLMDirectory, subjectID + "_align.mrk.json")
                slicer.util.saveNode(currentLMNode, outputLMPath)

                if semilandmarkOption:
                    currentSLMNode = self.getLandmarkFileByID(slmDirectory, subjectID)
                    if currentSLMNode:
                        currentSLMNode.SetAndObserveTransformNodeID(transformNode.GetID())
                        slicer.vtkSlicerTransformLogic().hardenTransform(currentSLMNode)
                        outputSLMPath = os.path.join(outputSLMDirectory, subjectID + "_align.mrk.json")
                        slicer.util.saveNode(currentSLMNode, outputSLMPath)
                        slicer.mrmlScene.RemoveNode(currentSLMNode)

                try:
                    slicer.mrmlScene.RemoveNode(currentLMNode)
                    slicer.mrmlScene.RemoveNode(currentMeshNode)
                    slicer.mrmlScene.RemoveNode(transformNode)
                except Exception:
                    print(f"could not find nodes to remove for {subjectID}")

    def distanceMatrix(self, array):
        if atlasShapeBridge:
            return atlasShapeBridge.distanceMatrix(array)
        rows, _ = array.shape
        fnx = lambda q: q - np.reshape(q, (rows, 1))
        dx = fnx(array[:, 0])
        dy = fnx(array[:, 1])
        dz = fnx(array[:, 2])
        return (dx**2.0 + dy**2.0 + dz**2.0) ** 0.5

    def numpyToFiducialNode(self, numpyArray, nodeName):
        _require_slicer()
        if atlasShapeBridge:
            return atlasShapeBridge.numpyToFiducialNode(numpyArray, nodeName)
        fiducialNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", nodeName)
        for index in range(len(numpyArray)):
            fiducialNode.AddControlPoint(numpyArray[index], str(index))
        return fiducialNode

    def computeAverageLM(self, fiducialGroup):
        if atlasShapeBridge:
            return atlasShapeBridge.computeAverageLM(fiducialGroup)
        sampleNumber = fiducialGroup.GetNumberOfBlocks()
        pointNumber = fiducialGroup.GetBlock(0).GetNumberOfPoints()
        groupArray_np = np.empty((pointNumber, 3, sampleNumber))
        for index in range(sampleNumber):
            pointData = fiducialGroup.GetBlock(index).GetPoints().GetData()
            groupArray_np[:, :, index] = vtk_np.vtk_to_numpy(pointData)
        averagePoints_np = np.mean(groupArray_np, axis=2)
        return self.numpyToFiducialNode(averagePoints_np, "Atlas Landmarks")

    def fiducialNodeToPolyData(self, nodeLocation, loadOption=True):
        _require_slicer()
        if atlasShapeBridge:
            return atlasShapeBridge.fiducialNodeToPolyData(nodeLocation, loadOption)
        polydataPoints = vtk.vtkPolyData()
        points = vtk.vtkPoints()
        if not loadOption:
            fiducialNode = nodeLocation
        else:
            success, fiducialNode = slicer.util.loadMarkupsFiducialList(nodeLocation)
            if not success:
                print("Could not load landmarks: ", nodeLocation)
                return None
        for index in range(fiducialNode.GetNumberOfControlPoints()):
            point = fiducialNode.GetNthControlPointPosition(index)
            points.InsertNextPoint(point)
        polydataPoints.SetPoints(points)
        slicer.mrmlScene.RemoveNode(fiducialNode)
        return polydataPoints

    def importLandmarks(self, topDir):
        prefer = [".mrk.json", ".json", ".fcsv"]
        picked = {}
        for fileName in os.listdir(topDir):
            lower_name = fileName.lower()
            if lower_name.endswith(tuple(prefer)):
                base = Path(fileName)
                while base.suffix.lower() in (".mrk", ".json", ".fcsv"):
                    base = base.with_suffix("")
                base_name = base.name
                rank = 0 if lower_name.endswith(".mrk.json") else (1 if lower_name.endswith(".json") else 2)
                if base_name not in picked or rank < picked[base_name][0]:
                    picked[base_name] = (rank, os.path.join(topDir, fileName))

        names = sorted(picked.keys())
        group = vtk.vtkMultiBlockDataGroupFilter()
        for name in names:
            polydataPoints = self.fiducialNodeToPolyData(picked[name][1])
            group.AddInputData(polydataPoints)
        group.Update()
        return names, group.GetOutput()

    def importMeshes(self, topDir, extensions, restrict_to=None):
        priority = {".obj": 0, ".ply": 1, ".stl": 2, ".vtp": 3, ".vtk": 4}
        picked = {}
        for fileName in os.listdir(topDir):
            extension = os.path.splitext(fileName)[1].lower()
            if extension in priority:
                base = os.path.splitext(fileName)[0]
                if (restrict_to is None) or (base in restrict_to):
                    if base not in picked or priority[extension] < picked[base][0]:
                        picked[base] = (priority[extension], os.path.join(topDir, fileName))

        if restrict_to is not None:
            names = [base for base in restrict_to if base in picked]
        else:
            names = sorted(picked.keys())

        modelGroup = vtk.vtkMultiBlockDataGroupFilter()
        for base in names:
            inputFilePath = picked[base][1]
            modelNode = self._load_model_with_cs(inputFilePath, "RAS")
            modelGroup.AddInputData(modelNode.GetPolyData())
            slicer.mrmlScene.RemoveNode(modelNode)
        modelGroup.Update()
        return names, modelGroup.GetOutput()

    def procrustesImposition(self, originalLandmarks, sizeOption):
        if atlasShapeBridge:
            return atlasShapeBridge.procrustesImposition(originalLandmarks, sizeOption)
        procrustesFilter = vtk.vtkProcrustesAlignmentFilter()
        if sizeOption:
            procrustesFilter.GetLandmarkTransform().SetModeToRigidBody()
        procrustesFilter.SetInputData(originalLandmarks)
        procrustesFilter.Update()
        meanShape = procrustesFilter.GetMeanPoints()
        return [meanShape, procrustesFilter.GetOutput()]

    def getClosestToMeanIndex(self, meanShape, alignedPoints):
        if atlasShapeBridge:
            return atlasShapeBridge.getClosestToMeanIndex(meanShape, alignedPoints)
        sampleNumber = alignedPoints.GetNumberOfBlocks()
        procrustesDistances = []
        for sample_index in range(sampleNumber):
            alignedShape = alignedPoints.GetBlock(sample_index)
            meanPoint = [0, 0, 0]
            alignedPoint = [0, 0, 0]
            distance = 0
            for point_index in range(meanShape.GetNumberOfPoints()):
                meanShape.GetPoint(point_index, meanPoint)
                alignedShape.GetPoint(point_index, alignedPoint)
                distance += np.sqrt(vtk.vtkMath.Distance2BetweenPoints(meanPoint, alignedPoint))
            procrustesDistances.append(distance)
        try:
            min_index, _ = min(enumerate(procrustesDistances), key=operator.itemgetter(1))
            return min_index
        except Exception:
            return 0

    def getClosestToMeanPath(self, landmarkDirectory):
        if atlasShapeBridge:
            try:
                return atlasShapeBridge.getClosestToMeanPath(landmarkDirectory)
            except Exception as exc:
                print(f"Error using atlasShapeBridge.getClosestToMeanPath: {exc}")
        lmNames, landmarks = self.importLandmarks(landmarkDirectory)
        if not lmNames:
            print(f"No landmarks found in {landmarkDirectory}")
            return None
        meanShape, alignedLandmarks = self.procrustesImposition(landmarks, False)
        closestToMeanIndex = self.getClosestToMeanIndex(meanShape, alignedLandmarks)
        return lmNames[closestToMeanIndex]

    def denseCorrespondence(self, originalLandmarks, originalMeshes):
        meanShape, alignedPoints = self.procrustesImposition(originalLandmarks, False)
        sampleNumber = alignedPoints.GetNumberOfBlocks()
        denseCorrespondenceGroup = vtk.vtkMultiBlockDataGroupFilter()
        baseIndex = self.getClosestToMeanIndex(meanShape, alignedPoints)
        baseMesh = originalMeshes.GetBlock(baseIndex)
        baseLandmarks = originalLandmarks.GetBlock(baseIndex).GetPoints()
        for index in range(sampleNumber):
            correspondingMesh = self.denseSurfaceCorrespondencePair(
                originalMeshes.GetBlock(index),
                originalLandmarks.GetBlock(index).GetPoints(),
                alignedPoints.GetBlock(index).GetPoints(),
                baseMesh,
                baseLandmarks,
                meanShape,
                index,
            )
            denseCorrespondenceGroup.AddInputData(correspondingMesh)
        denseCorrespondenceGroup.Update()
        return denseCorrespondenceGroup.GetOutput(), baseIndex

    def denseCorrespondenceBaseMesh(self, originalLandmarks, originalMeshes, baseMesh, baseLandmarks):
        meanShape, alignedPoints = self.procrustesImposition(originalLandmarks, False)
        sampleNumber = alignedPoints.GetNumberOfBlocks()
        print("procrustes aligned samples: ", sampleNumber)
        denseCorrespondenceGroup = vtk.vtkMultiBlockDataGroupFilter()
        for index in range(sampleNumber):
            correspondingMesh = self.denseSurfaceCorrespondencePair(
                originalMeshes.GetBlock(index),
                originalLandmarks.GetBlock(index).GetPoints(),
                alignedPoints.GetBlock(index).GetPoints(),
                baseMesh,
                baseLandmarks,
                meanShape,
                index,
            )
            denseCorrespondenceGroup.AddInputData(correspondingMesh)
        denseCorrespondenceGroup.Update()
        return denseCorrespondenceGroup.GetOutput()

    def denseSurfaceCorrespondencePair(
        self,
        originalMesh,
        originalLandmarks,
        alignedLandmarks,
        baseMesh,
        baseLandmarks,
        meanShape,
        iteration,
    ):
        meanTransform = vtk.vtkThinPlateSplineTransform()
        meanTransform.SetSourceLandmarks(originalLandmarks)
        meanTransform.SetTargetLandmarks(meanShape)
        meanTransform.SetBasisToR()

        meanTransformFilter = vtk.vtkTransformPolyDataFilter()
        if originalMesh and originalMesh.GetNumberOfPoints() > 0:
            meanTransformFilter.SetInputData(originalMesh)
            meanTransformFilter.SetTransform(meanTransform)
            meanTransformFilter.Update()
            meanWarpedMesh = meanTransformFilter.GetOutput()
        else:
            print(f"Warning: Empty or invalid originalMesh for iteration {iteration}")
            return baseMesh

        meanTransformBase = vtk.vtkThinPlateSplineTransform()
        meanTransformBase.SetSourceLandmarks(baseLandmarks)
        meanTransformBase.SetTargetLandmarks(meanShape)
        meanTransformBase.SetBasisToR()

        meanTransformBaseFilter = vtk.vtkTransformPolyDataFilter()
        if baseMesh and baseMesh.GetNumberOfPoints() > 0:
            meanTransformBaseFilter.SetInputData(baseMesh)
            meanTransformBaseFilter.SetTransform(meanTransformBase)
            meanTransformBaseFilter.Update()
            meanWarpedBase = meanTransformBaseFilter.GetOutput()
        else:
            print(f"Warning: Empty or invalid baseMesh for iteration {iteration}")
            return baseMesh

        warpedUVs = meanWarpedMesh.GetPointData().GetTCoords()
        if warpedUVs:
            print(f"UVs found for subject {self.modelNames[iteration]}, preparing for transfer.")
            newUVs = vtk.vtkFloatArray()
            newUVs.SetName("TransferredUVs")
            newUVs.SetNumberOfComponents(2)
            newUVs.SetNumberOfTuples(meanWarpedBase.GetNumberOfPoints())

        cellLocator = vtk.vtkCellLocator()
        cellLocator.SetDataSet(meanWarpedMesh)
        cellLocator.BuildLocator()

        correspondingPoints = vtk.vtkPoints()
        for point_index in range(meanWarpedBase.GetNumberOfPoints()):
            point = meanWarpedBase.GetPoint(point_index)
            closestPoint = [0.0, 0.0, 0.0]
            closestCellId = vtk.reference(0)
            subId = vtk.reference(0)
            dist2 = vtk.reference(0.0)
            cellLocator.FindClosestPoint(point, closestPoint, closestCellId, subId, dist2)
            correspondingPoints.InsertPoint(point_index, closestPoint)

            if warpedUVs:
                actualCellId = closestCellId.get()
                cell = meanWarpedMesh.GetCell(actualCellId)
                if cell and cell.GetNumberOfPoints() == 3:
                    weights = [0.0] * cell.GetNumberOfPoints()
                    closestPointOutput = [0.0, 0.0, 0.0]
                    pcoords_ignored = [0.0, 0.0, 0.0]
                    dist2_ignored = vtk.reference(0.0)
                    cell.EvaluatePosition(closestPoint, closestPointOutput, subId, pcoords_ignored, dist2_ignored, weights)
                    cellPointIds = cell.GetPointIds()
                    uv0 = warpedUVs.GetTuple2(cellPointIds.GetId(0))
                    uv1 = warpedUVs.GetTuple2(cellPointIds.GetId(1))
                    uv2 = warpedUVs.GetTuple2(cellPointIds.GetId(2))
                    u_new = weights[0] * uv0[0] + weights[1] * uv1[0] + weights[2] * uv2[0]
                    v_new = weights[0] * uv0[1] + weights[1] * uv1[1] + weights[2] * uv2[1]
                    newUVs.SetTuple2(point_index, u_new, v_new)
                else:
                    newUVs.SetTuple2(point_index, 0.0, 0.0)

        correspondingMesh = vtk.vtkPolyData()
        correspondingMesh.SetPoints(correspondingPoints)
        correspondingMesh.SetPolys(meanWarpedBase.GetPolys())
        if warpedUVs:
            correspondingMesh.GetPointData().SetTCoords(newUVs)

        inverseTransform = vtk.vtkThinPlateSplineTransform()
        inverseTransform.SetSourceLandmarks(meanShape)
        inverseTransform.SetTargetLandmarks(originalLandmarks)
        inverseTransform.SetBasisToR()

        inverseTransformFilter = vtk.vtkTransformPolyDataFilter()
        if correspondingMesh and correspondingMesh.GetNumberOfPoints() > 0:
            inverseTransformFilter.SetInputData(correspondingMesh)
            inverseTransformFilter.SetTransform(inverseTransform)
            inverseTransformFilter.Update()
            return inverseTransformFilter.GetOutput()
        print(f"Warning: Empty correspondingMesh for iteration {iteration}, returning original")
        return baseMesh

    def convertPointsToVTK(self, points):
        if atlasShapeBridge:
            return atlasShapeBridge.convertPointsToVTK(points)
        array_vtk = vtk_np.numpy_to_vtk(points, deep=True, array_type=vtk.VTK_FLOAT)
        points_vtk = vtk.vtkPoints()
        points_vtk.SetData(array_vtk)
        polydata_vtk = vtk.vtkPolyData()
        polydata_vtk.SetPoints(points_vtk)
        return polydata_vtk

    def computeAverageModelFromGroup(self, denseCorrespondenceGroup, baseIndex):
        if atlasShapeBridge:
            return atlasShapeBridge.computeAverageModelFromGroup(denseCorrespondenceGroup, baseIndex)
        sampleNumber = denseCorrespondenceGroup.GetNumberOfBlocks()
        pointNumber = denseCorrespondenceGroup.GetBlock(0).GetNumberOfPoints()
        groupArray_np = np.empty((pointNumber, 3, sampleNumber))
        baseMesh = denseCorrespondenceGroup.GetBlock(baseIndex)
        for index in range(sampleNumber):
            alignedMesh = denseCorrespondenceGroup.GetBlock(index)
            alignedMesh_np = vtk_np.vtk_to_numpy(alignedMesh.GetPoints().GetData())
            groupArray_np[:, :, index] = alignedMesh_np
        averagePoints_np = np.mean(groupArray_np, axis=2)
        averagePointsPolydata = self.convertPointsToVTK(averagePoints_np)
        averageModel = vtk.vtkPolyData()
        averageModel.SetPoints(averagePointsPolydata.GetPoints())
        averageModel.SetPolys(baseMesh.GetPolys())
        return averageModel

    def addMagnitudeFeature(self, denseCorrespondenceGroup, modelNameArray, model):
        if atlasShapeBridge:
            return atlasShapeBridge.addMagnitudeFeature(denseCorrespondenceGroup, modelNameArray, model)
        sampleNumber = denseCorrespondenceGroup.GetNumberOfBlocks()
        pointNumber = denseCorrespondenceGroup.GetBlock(0).GetNumberOfPoints()
        statsArray = np.zeros((pointNumber, sampleNumber))
        magnitudeMean = vtk.vtkDoubleArray()
        magnitudeMean.SetNumberOfComponents(1)
        magnitudeMean.SetName("Magnitude Mean")
        magnitudeSD = vtk.vtkDoubleArray()
        magnitudeSD.SetNumberOfComponents(1)
        magnitudeSD.SetName("Magnitude SD")

        for sample_index in range(sampleNumber):
            alignedMesh = denseCorrespondenceGroup.GetBlock(sample_index)
            magnitudes = vtk.vtkDoubleArray()
            magnitudes.SetNumberOfComponents(1)
            magnitudes.SetName(modelNameArray[sample_index])
            for point_index in range(pointNumber):
                modelPoint = model.GetPoint(point_index)
                targetPoint = alignedMesh.GetPoint(point_index)
                distance = np.sqrt(vtk.vtkMath.Distance2BetweenPoints(modelPoint, targetPoint))
                magnitudes.InsertNextValue(distance)
                statsArray[point_index, sample_index] = distance
            model.GetPointData().AddArray(magnitudes)

        for point_index in range(pointNumber):
            magnitudeMean.InsertNextValue(statsArray[point_index, :].mean())
            magnitudeSD.InsertNextValue(statsArray[point_index, :].std())

        model.GetPointData().AddArray(magnitudeMean)
        model.GetPointData().AddArray(magnitudeSD)

    def _save_model_with_cs(self, modelNode, filePath, coordinateSystem="RAS"):
        _require_slicer()
        if modelNode is None:
            raise ValueError(f"Model node is None, cannot save to {filePath}")

        storage = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelStorageNode")
        storage.SetFileName(filePath)
        coordinateSystem = (coordinateSystem or "RAS").upper()
        try:
            if coordinateSystem == "RAS":
                storage.SetCoordinateSystemToRAS()
            else:
                storage.SetCoordinateSystemToLPS()
        except AttributeError:
            storage.SetCoordinateSystem(0 if coordinateSystem == "RAS" else 1)
        modelNode.SetAndObserveStorageNodeID(storage.GetID())
        ok = storage.WriteData(modelNode)
        slicer.mrmlScene.RemoveNode(storage)
        if not ok:
            raise RuntimeError(f"Failed to write model: {filePath}")

    def blender_prepare_atlas(self, blender_exe, in_obj, out_obj, merge_dist=0.0005, smart_angle=66.0, island_margin=0.002):
        tmp = tempfile.mkdtemp(prefix="InterDeCA_blUV_")
        script = os.path.join(tmp, "prep_uv.py")
        py = textwrap.dedent(
            """
            import bpy, sys
            argv = sys.argv
            argv = argv[argv.index("--")+1:] if "--" in argv else []
            in_path  = argv[0]
            out_path = argv[1]
            merge_d  = float(argv[2])
            ang      = float(argv[3])
            island_m = float(argv[4])

            bpy.ops.wm.read_factory_settings(use_empty=True)

            try:
                bpy.ops.wm.obj_import(filepath=in_path, forward_axis='Y', up_axis='Z')
            except AttributeError:
                bpy.ops.import_scene.obj(filepath=in_path, use_split_objects=False, use_split_groups=False, axis_forward='Y', axis_up='Z')

            obj = [o for o in bpy.context.selected_objects if o.type=='MESH'][0]
            bpy.context.view_layer.objects.active = obj

            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.mesh.select_all(action='SELECT')
            try:
                bpy.ops.mesh.remove_doubles(threshold=merge_d)
            except Exception:
                try:
                    bpy.ops.mesh.merge_by_distance(distance=merge_d)
                except Exception:
                    bpy.ops.mesh.merge(type='DISTANCE', distance=merge_d)

            bpy.ops.uv.smart_project(angle_limit=ang, island_margin=island_m, correct_aspect=True, scale_to_bounds=False)
            bpy.ops.object.mode_set(mode='OBJECT')

            bpy.ops.wm.obj_export(
                filepath=out_path,
                export_selected_objects=True,
                export_triangulated_mesh=True,
                forward_axis='Y', up_axis='Z',
                export_materials=False,
            )
            """
        )
        with open(script, "w", encoding="utf-8") as script_file:
            script_file.write(py)
        args = [
            blender_exe,
            "--background",
            "--factory-startup",
            "--python",
            script,
            "--",
            in_obj,
            out_obj,
            str(merge_dist),
            str(smart_angle),
            str(island_margin),
        ]
        subprocess.run(args, check=True)

    def write_obj_with_uv_from_template(self, polydata, atlas_obj_path, out_obj_path):
        _require_slicer()
        atlas_node = self._load_model_with_cs(atlas_obj_path, "RAS")
        atlas_pd = atlas_node.GetPolyData()

        if not atlas_pd or not atlas_pd.GetPointData() or not atlas_pd.GetPointData().GetTCoords():
            slicer.mrmlScene.RemoveNode(atlas_node)
            raise RuntimeError("Atlas OBJ has no per-vertex texture coordinates (TCoords).")

        if polydata.GetNumberOfPoints() != atlas_pd.GetNumberOfPoints():
            slicer.mrmlScene.RemoveNode(atlas_node)
            raise RuntimeError(
                f"Point-count mismatch between resampled ({polydata.GetNumberOfPoints()}) "
                f"and atlas ({atlas_pd.GetNumberOfPoints()})."
            )

        tc_copy = vtk.vtkFloatArray()
        tc_copy.DeepCopy(atlas_pd.GetPointData().GetTCoords())
        tc_copy.SetName("TCoords")
        polydata.GetPointData().SetTCoords(tc_copy)

        writer = vtk.vtkOBJWriter()
        writer.SetFileName(out_obj_path)
        writer.SetInputData(polydata)
        writer.Update()
        writer.Write()

        slicer.mrmlScene.RemoveNode(atlas_node)

    def blender_bake_all(
        self,
        blender_exe,
        alignedDir,
        resampledUVDir,
        texturesDir,
        outDir,
        bake_size=2048,
        bake_extrusion=0.005,
        bake_margin_px=2,
        merge_dist=0.0005,
    ):
        def norm_id(name):
            return re.sub(r"(_align|_resampled)$", "", os.path.splitext(name)[0], flags=re.IGNORECASE)

        priority = {".obj": 0, ".ply": 1, ".stl": 2, ".vtp": 3, ".vtk": 4}
        best = {}
        for fileName in os.listdir(alignedDir):
            extension = os.path.splitext(fileName)[1].lower()
            if extension in priority:
                subject_id = norm_id(fileName)
                if subject_id not in best or priority[extension] < best[subject_id][0]:
                    best[subject_id] = (priority[extension], os.path.join(alignedDir, fileName))
        aligned = {subject_id: path for subject_id, (_, path) in best.items()}
        targets = {
            norm_id(fileName): os.path.join(resampledUVDir, fileName)
            for fileName in os.listdir(resampledUVDir)
            if fileName.lower().endswith(".obj")
        }
        textures = {
            os.path.splitext(fileName)[0].lower(): os.path.join(texturesDir, fileName)
            for fileName in os.listdir(texturesDir)
            if fileName.lower().endswith((".png", ".tiff", ".tif"))
        }

        def find_tex(subject_id):
            key = subject_id.lower()
            if key in textures:
                return textures[key]
            for texture_key in textures:
                if texture_key.startswith(key):
                    return textures[texture_key]
            return None

        baked = {}
        os.makedirs(outDir, exist_ok=True)

        py = textwrap.dedent(
            """
            import bpy, sys, os
            argv = sys.argv
            argv = argv[argv.index("--")+1:] if "--" in argv else []
            src_path, tgt_path, png_in, png_out, sz, extru, margin, merge_d = argv
            sz = int(sz); extru = float(extru); margin = int(margin); merge_d = float(merge_d)

            bpy.ops.wm.read_factory_settings(use_empty=True)
            try: bpy.ops.wm.obj_import(filepath=tgt_path, forward_axis='Y', up_axis='Z')
            except AttributeError: bpy.ops.import_scene.obj(filepath=tgt_path, use_split_objects=False, use_split_groups=False, axis_forward='Y', axis_up='Z')
            tgt = [o for o in bpy.context.selected_objects if o.type=='MESH'][0]

            imp_ok = False
            try:
                bpy.ops.wm.obj_import(filepath=src_path, forward_axis='Y', up_axis='Z'); imp_ok=True
            except Exception: pass
            if not imp_ok:
                raise RuntimeError("Cannot import aligned mesh: " + src_path)
            src = [o for o in bpy.context.selected_objects if o.type=='MESH'][-1]

            bpy.context.view_layer.objects.active = tgt
            bpy.ops.object.mode_set(mode='EDIT'); bpy.ops.mesh.select_all(action='SELECT')
            try: bpy.ops.mesh.remove_doubles(threshold=merge_d)
            except Exception:
                try: bpy.ops.mesh.merge_by_distance(distance=merge_d)
                except Exception: bpy.ops.mesh.merge(type='DISTANCE', distance=merge_d)
            bpy.ops.object.mode_set(mode='OBJECT')

            tgt.data.materials.clear(); src.data.materials.clear()
            m_src = bpy.data.materials.new("MatSrc"); m_src.use_nodes=True
            nt = m_src.node_tree; nodes = nt.nodes
            img_node = nodes.new('ShaderNodeTexImage'); img_node.image = bpy.data.images.load(png_in)
            bsdf = next(n for n in nodes if n.type=='BSDF_PRINCIPLED')
            nt.links.new(img_node.outputs['Color'], bsdf.inputs['Base Color'])
            src.data.materials.append(m_src)

            m_tgt = bpy.data.materials.new("MatTgt"); m_tgt.use_nodes=True
            nt2 = m_tgt.node_tree; nodes2 = nt2.nodes
            imgT = bpy.data.images.new("BakeTarget", width=sz, height=sz, alpha=False)
            img_node_t = nodes2.new('ShaderNodeTexImage'); img_node_t.image = imgT
            tgt.data.materials.append(m_tgt)

            bpy.ops.object.select_all(action='DESELECT')
            src.select_set(True); tgt.select_set(True)
            bpy.context.view_layer.objects.active = tgt

            for n in nodes2: n.select = False
            nodes2.active = img_node_t; img_node_t.select = True

            scn = bpy.context.scene
            scn.render.engine = 'CYCLES'
            scn.cycles.device = 'CPU'
            b = scn.render.bake
            b.use_selected_to_active = True
            b.cage_extrusion = extru
            b.margin = margin
            b.use_pass_direct = False
            b.use_pass_indirect = False
            b.use_pass_color = True

            bpy.ops.object.bake(type='DIFFUSE')

            imgT.filepath_raw = png_out
            imgT.file_format = 'PNG'
            imgT.save()
            """
        )
        tmp_script = tempfile.NamedTemporaryFile(delete=False, suffix=".py")
        tmp_script.write(py.encode("utf-8"))
        tmp_script.close()

        for subject_id, src_path in sorted(aligned.items()):
            tgt_path = targets.get(subject_id)
            tex_in = find_tex(subject_id)
            if not (tgt_path and tex_in):
                continue
            out_png = os.path.join(outDir, f"{subject_id}.png")
            args = [
                blender_exe,
                "--background",
                "--factory-startup",
                "--python",
                tmp_script.name,
                "--",
                src_path,
                tgt_path,
                tex_in,
                out_png,
                str(bake_size),
                str(bake_extrusion),
                str(bake_margin_px),
                str(merge_dist),
            ]
            subprocess.run(args, check=True)
            baked[subject_id] = out_png

        return baked

    def _calculate_average_texture(self, outTexturesDir):
        atlas_texture = os.path.join(outTexturesDir, "average_texture.png")
        if os.path.exists(atlas_texture):
            return
        pngs = glob.glob(os.path.join(outTexturesDir, "*.png"))
        images = [imageio.imread(png) for png in pngs]
        average = np.mean(images, axis=0).astype(np.uint8)
        imageio.imwrite(atlas_texture, average)

    def _load_model_with_cs(self, filePath, coordinateSystem="RAS"):
        _require_slicer()
        storage = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelStorageNode")
        coordinateSystem = (coordinateSystem or "RAS").upper()
        try:
            if coordinateSystem == "RAS":
                storage.SetCoordinateSystemToRAS()
            else:
                storage.SetCoordinateSystemToLPS()
        except AttributeError:
            storage.SetCoordinateSystem(0 if coordinateSystem == "RAS" else 1)
        modelNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode")
        storage.SetFileName(filePath)
        ok = storage.ReadData(modelNode)
        slicer.mrmlScene.RemoveNode(storage)
        if not ok:
            slicer.mrmlScene.RemoveNode(modelNode)
            raise RuntimeError(f"Failed to read model: {filePath}")
        return modelNode


InterDeCALogic = InterDeCALogicFacade
