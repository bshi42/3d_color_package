import pyvista as pv, pyacvd, numpy as np
import vtk

# ----- helpers -----
def _cell_normals_numpy(poly: pv.PolyData) -> np.ndarray:
    p = poly.points; f = poly.faces.reshape(-1, 4)[:, 1:]
    v1, v2 = p[f[:,1]] - p[f[:,0]], p[f[:,2]] - p[f[:,0]]
    n = np.cross(v1, v2); L = np.linalg.norm(n, axis=1); m = L > 0
    n[m] /= L[m][:, None]
    return n

def _ensure_cell_normals(poly: pv.PolyData, auto_orient: bool) -> tuple[pv.PolyData, np.ndarray]:
    q = poly.triangulate().clean(tolerance=0.0).copy()
    q = q.compute_normals(point_normals=False, cell_normals=True,
                          auto_orient_normals=auto_orient, feature_angle=180.0, inplace=False)
    if 'Normals' in q.cell_data:
        return q, np.asarray(q.cell_data['Normals'])
    return q, _cell_normals_numpy(q)

def orient_by_regions(remesh: pv.PolyData,
                      reference: pv.PolyData = None,
                      method: str = "reference") -> pv.PolyData:
    """Flip entire patches bounded by seams until all normals are outward."""
    m0 = remesh.triangulate().clean(tolerance=0.0)

    # 1) consistent winding first
    m_cons, m_norm = _ensure_cell_normals(m0, auto_orient=True)
    centers = m_cons.cell_centers().points

    # 2) face-wise test for “outside”
    if method == "reference" and reference is not None:
        ref = reference.triangulate().clean(tolerance=0.0)
        refN, ref_norm = _ensure_cell_normals(ref, auto_orient=True)
        loc = vtk.vtkStaticCellLocator(); loc.SetDataSet(refN); loc.BuildLocator()
        cp = [0,0,0]; cid = vtk.mutable(0); subid = vtk.mutable(0); dist2 = vtk.mutable(0.0)
        dots = np.empty(m_cons.n_cells, float)
        for i in range(m_cons.n_cells):
            loc.FindClosestPoint(centers[i], cp, cid, subid, dist2)
            dots[i] = float(np.dot(m_norm[i], ref_norm[int(cid)]))
    else:
        areas = m_cons.compute_cell_sizes(length=False, area=True, volume=False).cell_data['Area']
        gc = (centers * areas[:, None]).sum(0) / areas.sum()
        vout = centers - gc; vout /= (np.linalg.norm(vout, axis=1)[:, None] + 1e-12)
        dots = np.einsum('ij,ij->i', m_norm, vout)

    bad_ids = np.flatnonzero(dots < 0.0)
    if bad_ids.size == 0:
        return m_cons.compute_normals(point_normals=True, cell_normals=False,
                                      auto_orient_normals=False, feature_angle=180.0, inplace=False)

    # 3) carry original cell ids, subset to “bad” faces
    m_cons.cell_data['origCellId'] = np.arange(m_cons.n_cells, dtype=np.int64)
    bad_subset = m_cons.extract_cells(bad_ids)   # keeps origCellId

    # 4) split bad faces into patches with vtkConnectivityFilter (robust RegionId)
    cf = vtk.vtkConnectivityFilter()
    cf.SetInputData(bad_subset)
    cf.SetExtractionModeToAllRegions()
    cf.ColorRegionsOn()
    cf.Update()
    ug = pv.wrap(cf.GetOutput())                 # UnstructuredGrid with RegionId + origCellId
    region = ug.cell_data.get('RegionId', None)
    if region is None:                           # fallback: single region
        region = np.zeros(ug.n_cells, dtype=np.int32)
    origids = ug.cell_data['origCellId'].astype(np.int64)

    # 5) flip each region in the ORIGINAL faces array
    faces = m_cons.faces.reshape(-1, 4).copy()   # [3, v0, v1, v2]
    for rid in np.unique(region):
        sel = origids[region == rid]
        tmp = faces[sel, 1].copy()
        faces[sel, 1] = faces[sel, 2]
        faces[sel, 2] = tmp
    m_cons.faces = faces.reshape(-1)

    # 6) final smooth, unsplit point normals
    out = m_cons.compute_normals(point_normals=True, cell_normals=False,
                                 auto_orient_normals=False, feature_angle=180.0, inplace=False)
    return out

mesh = pv.read("/media/alek/e6852e67-f061-4723-a0d3-c6271961077a/ml_data/color-modeling-pub/3D_Fish/3D_Fish/deca_out/2025_09-16_00_35_40/decaAtlasUV.obj")
# mesh = mesh.triangulate().clean()
mesh.save('intermediate.ply')
clus = pyacvd.Clustering(mesh)
clus.subdivide(3)
clus.cluster(30000)
remesh = clus.create_mesh().triangulate().clean()

# Option A (best): compare to original surface
remesh = orient_by_regions(remesh, reference=mesh, method="centroid")

# Option B (no reference): centroid heuristic
# remesh = orient_by_regions(remesh, method="centroid")

remesh = remesh.connectivity(largest=True)   # optional
remesh.save("out_uniform.ply")

# print("seam before:", remesh.extract_feature_edges(True, True, False, False).n_cells)

# # stitch only the thin slit; uses estimated gap with a 3× multiplier:
# remesh = weld_seam_only_across(remesh, reference=mesh, gap_mult=3.0, radius_pad=1.3, max_iters=4)

# print("seam after :", remesh.extract_feature_edges(True, True, False, False).n_cells)