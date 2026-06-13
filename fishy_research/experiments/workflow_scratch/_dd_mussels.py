"""Deep-dive: segmentation + SPATIAL descriptor on REAL mussels (no ground truth).

Pipeline:
  - build per-face colors for the real mussel population
  - segment into 8 palette colors (shared palette, 1 smoothing pass)
  - compute SPATIAL descriptor + Endler transitions
  - PCA(StandardScaler) -> 2D
  - characterize PC1/PC2 by correlation with interpretable per-shell scalars
  - dip-test gate on the top PCs (real clusters vs continuous structure)
  - flag outlier shells by name
"""
import numpy as np
import scipy.sparse as sp
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from diptest import diptest

from fishpipe import dataset, segment, spatial, blobs, structure
from fishpipe.features import rgb_to_lab

rng = np.random.default_rng(0)

# ---- 0. mussel-specific mesh adjacency + principal axis ------------------
# The default face_adjacency()/principal_axis() are hardcoded to the FISH mesh.
# Build a mussel-mesh face adjacency (faces sharing an edge) and a mussel body
# axis, then inject them so segment/spatial/blobs operate on the right mesh.
def build_adjacency(mesh):
    fv = mesh.face_v                                  # (Nf,3)
    nf = fv.shape[0]
    e = np.vstack([fv[:, [0, 1]], fv[:, [1, 2]], fv[:, [2, 0]]])  # (3Nf,2)
    e.sort(axis=1)
    fid = np.tile(np.arange(nf), 3)
    order = np.lexsort((e[:, 1], e[:, 0]))
    e = e[order]; fid = fid[order]
    same = np.all(e[1:] == e[:-1], axis=1)            # consecutive identical edges
    idx = np.where(same)[0]
    r = np.concatenate([fid[idx], fid[idx + 1]])
    c = np.concatenate([fid[idx + 1], fid[idx]])
    A = sp.coo_matrix((np.ones(len(r)), (r, c)), shape=(nf, nf)).tocsr()
    A.data[:] = 1.0
    return A

mesh = dataset.MUSSELS.load_mesh()
A_mussel = build_adjacency(mesh)

def mussel_principal_axis():
    cents = mesh.face_centroids()
    c = cents - cents.mean(0)
    _, _, vt = np.linalg.svd(c - c.mean(0), full_matrices=False)
    return vt[0]
# patch the axis used inside spatial_descriptor (spatial imports it by name)
spatial.principal_axis = mussel_principal_axis
structure.principal_axis = mussel_principal_axis

# ---- 1. build data -------------------------------------------------------
fcd, _ = dataset.build_face_colors(dataset.MUSSELS, verbose=False)
seg = segment.segment(fcd, n_colors=8, smooth_iters=1, mesh_adjacency=A_mussel)
spat = spatial.spatial_descriptor(fcd, seg, mesh_adjacency=A_mussel)   # (N, K*10)
endl = blobs.endler_transitions(fcd, seg, mesh_adjacency=A_mussel)     # (N, K*(K+1)/2)
names = np.array(fcd.names)
N = fcd.colors.shape[0]
print(f"N shells={N}  spat={spat.shape}  endl={endl.shape}")

# ---- 2. interpretable per-shell scalars ----------------------------------
lab = rgb_to_lab(fcd.colors)                          # (N, Nf, 3) L*,a*,b*
# area-weighted moments per shell
w = fcd.areas / fcd.areas.sum()                       # (Nf,)
L = lab[..., 0]; a = lab[..., 1]
meanL = (L * w[None]).sum(axis=1)
# weighted std of L*
varL = ((L - meanL[:, None]) ** 2 * w[None]).sum(axis=1)
stdL = np.sqrt(varL)
meana = (a * w[None]).sum(axis=1)

# number of dark blobs: count blobs in the darkest segmentation color
n_dark = 1
dark_blobs = blobs.dark_blob_count(fcd, seg, n_dark=n_dark, mesh_adjacency=A_mussel)  # (N,)
dark_blobs = np.asarray(dark_blobs).reshape(N, -1).sum(axis=1).astype(float)

scalars = {
    "darkness(-meanL*)":     -meanL,
    "ring-contrast(stdL*)":   stdL,
    "greenness(-mean a*)":   -meana,
    "n_dark_blobs":           dark_blobs,
}

# ---- 3. PCA of spatial+endler --------------------------------------------
X = np.hstack([spat, endl])
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
Xs = StandardScaler().fit_transform(X)
pca = PCA(n_components=min(6, Xs.shape[1])).fit(Xs)
Z = pca.transform(Xs)
evr = pca.explained_variance_ratio_
print("explained var ratio (PC1..):", np.round(evr[:6], 3))

# ---- 4. characterize PC1/PC2 by correlation with scalars -----------------
def corr_table(pc_idx):
    out = {}
    for k, v in scalars.items():
        r, p = stats.spearmanr(Z[:, pc_idx], v)
        out[k] = (r, p)
    return out

print("\n=== PC correlations (Spearman r, p) ===")
for pc in range(2):
    print(f"\nPC{pc+1} (evr={evr[pc]:.3f}):")
    for k, (r, p) in corr_table(pc).items():
        star = "*" if p < 0.05 else " "
        print(f"   {k:22s} r={r:+.3f}  p={p:.3g} {star}")

# ---- 5. dip-test gate ----------------------------------------------------
# Honest gate: a PC has 'real clusters' only if
#   (a) dip-test p < 0.05  AND
#   (b) its dip statistic exceeds the MAX dip from column-shuffled nulls
# Column-shuffled null: independently permute each feature column, re-PCA,
# take that PC's dip. This destroys cross-feature structure (the thing that
# would create genuine clusters) while preserving each marginal distribution.
n_null = 200
def dip_of_pc(data_Z, pc):
    d, p = diptest(np.ascontiguousarray(data_Z[:, pc]))
    return d, p

print("\n=== Dip-test cluster gate ===")
real_dip = {pc: dip_of_pc(Z, pc) for pc in range(2)}

null_max = {0: -np.inf, 1: -np.inf}
null_dips = {0: [], 1: []}
for _ in range(n_null):
    Xsh = Xs.copy()
    for j in range(Xsh.shape[1]):
        Xsh[:, j] = Xsh[rng.permutation(N), j]
    Zsh = PCA(n_components=2).fit_transform(Xsh)
    for pc in range(2):
        d, _ = diptest(np.ascontiguousarray(Zsh[:, pc]))
        null_dips[pc].append(d)
        null_max[pc] = max(null_max[pc], d)

verdicts = {}
for pc in range(2):
    d, p = real_dip[pc]
    nmax = null_max[pc]
    n95 = float(np.percentile(null_dips[pc], 95))
    clusters = (p < 0.05) and (d > nmax)
    verdicts[pc] = clusters
    print(f"PC{pc+1}: dip={d:.4f} p={p:.3g} | null_max={nmax:.4f} null95={n95:.4f} "
          f"-> {'CLUSTERS' if clusters else 'continuous'}")

# ---- 6. outlier shells ---------------------------------------------------
# Mahalanobis-ish: distance in the 2D PCA space (robust z on each PC)
def robust_z(x):
    med = np.median(x); mad = stats.median_abs_deviation(x) + 1e-9
    return (x - med) / (1.4826 * mad)
z1 = robust_z(Z[:, 0]); z2 = robust_z(Z[:, 1])
dist = np.sqrt(z1**2 + z2**2)
order = np.argsort(dist)[::-1]
print("\n=== Top outlier shells (2D PCA robust distance) ===")
for j in order[:6]:
    print(f"   {names[j]:18s} d={dist[j]:.2f}  PC1z={z1[j]:+.2f} PC2z={z2[j]:+.2f}"
          f"  | darkL*={-scalars['darkness(-meanL*)'][j]:.1f}"
          f" stdL*={stdL[j]:.1f} a*={meana[j]:+.1f} ndark={dark_blobs[j]:.0f}")

# machine-readable summary line
import json
summary = {
    "N": int(N),
    "evr": [round(float(x), 4) for x in evr[:4]],
    "pc1_corr": {k: [round(float(r), 3), float(f"{p:.2g}")] for k, (r, p) in corr_table(0).items()},
    "pc2_corr": {k: [round(float(r), 3), float(f"{p:.2g}")] for k, (r, p) in corr_table(1).items()},
    "dip": {f"PC{pc+1}": {"d": round(float(real_dip[pc][0]), 4),
                           "p": float(f"{real_dip[pc][1]:.2g}"),
                           "null_max": round(float(null_max[pc]), 4),
                           "clusters": bool(verdicts[pc])} for pc in range(2)},
    "top_outliers": [names[j] for j in order[:5]],
}
print("\nSUMMARY_JSON " + json.dumps(summary))
