"""Graph-spectral pattern descriptors (mesh Laplacian / manifold-harmonics).

Idea: treat each specimen's surface color as a *signal on the mesh graph*. The eigenvectors
of the mesh Laplacian are a natural "Fourier basis on the surface", ordered low→high spatial
frequency. Projecting a color channel onto this basis and summarizing the **energy in
eigenvalue bands** yields a compact, **shift-invariant** descriptor of pattern STRUCTURE
(coarse blobs vs fine bands vs tiny spots) that is blind to *where* on the body the pattern
sits — exactly what we need so that stripe count etc. survive cross-specimen position jitter.

CPU-only: the eigenbasis is computed ONCE for the shared mesh (cached); per-specimen
projection is a cheap dense matmul.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from . import config
from .data import FaceColorData, load_mesh
from .features import rgb_to_lab


# ---------------------------------------------------------------------------
# Per-vertex color signal (area-weighted average of incident faces) — cached
# ---------------------------------------------------------------------------
def vertex_lab(fcd: FaceColorData, mesh=None, cache_dir=None) -> np.ndarray:
    """(N, Nv, 3) per-vertex Lab, area-weighted from per-face colors. Cached float32.

    `mesh`/`cache_dir` default to the fishy mesh/cache for backward compat; pass a
    `dataset.Dataset`-derived mesh + its cache_dir to run on any shared-atlas dataset.
    """
    cache = (cache_dir or config.CACHE_DIR) / "vertex_lab_f32.npy"
    if cache.exists():
        return np.load(cache)
    mesh = mesh if mesh is not None else load_mesh()
    fv = mesh.face_v                      # (Nf, 3)
    areas = fcd.areas                     # (Nf,)
    nv = mesh.vertices.shape[0]
    # per-vertex incident area (denominator), shared across specimens
    varea = np.zeros(nv)
    np.add.at(varea, fv.ravel(), np.repeat(areas, 3))
    varea[varea == 0] = 1.0

    lab_faces = rgb_to_lab(fcd.colors)    # (N, Nf, 3)
    N = lab_faces.shape[0]
    out = np.zeros((N, nv, 3), dtype=np.float32)
    for i in range(N):
        for c in range(3):
            acc = np.zeros(nv)
            np.add.at(acc, fv.ravel(), np.repeat(areas * lab_faces[i, :, c], 3))
            out[i, :, c] = (acc / varea).astype(np.float32)
    np.save(cache, out)
    return out


# ---------------------------------------------------------------------------
# Mesh graph Laplacian eigenbasis — cached
# ---------------------------------------------------------------------------
def laplacian_eigenbasis(k: int = 300, mesh=None, cache_dir=None) -> tuple[np.ndarray, np.ndarray]:
    """Smallest-k eigenpairs of the symmetric-normalized mesh graph Laplacian.

    Returns (eigvals (k,), U (Nv, k)). Cached. Uses shift-invert for the low end.
    `mesh`/`cache_dir` default to the fishy mesh/cache; inject for other datasets.
    """
    cdir = cache_dir or config.CACHE_DIR
    cache_u = cdir / f"lap_U_{k}.npy"
    cache_w = cdir / f"lap_w_{k}.npy"
    if cache_u.exists() and cache_w.exists():
        return np.load(cache_w), np.load(cache_u)

    mesh = mesh if mesh is not None else load_mesh()
    fv = mesh.face_v
    nv = mesh.vertices.shape[0]
    # undirected edges from triangle sides
    e = np.vstack([fv[:, [0, 1]], fv[:, [1, 2]], fv[:, [2, 0]]])
    e = np.vstack([e, e[:, ::-1]])
    data = np.ones(len(e))
    A = sp.coo_matrix((data, (e[:, 0], e[:, 1])), shape=(nv, nv)).tocsr()
    A.data[:] = 1.0                       # binary adjacency
    A = ((A + A.T) > 0).astype(np.float64)
    deg = np.asarray(A.sum(1)).ravel()
    dinv = 1.0 / np.sqrt(np.maximum(deg, 1e-12))
    D = sp.diags(dinv)
    L = sp.identity(nv) - D @ A @ D       # symmetric normalized Laplacian
    L = L.tocsc()
    # smallest eigenvalues via shift-invert near 0
    w, U = spla.eigsh(L, k=k, sigma=-1e-6, which="LM")
    order = np.argsort(w)
    w, U = w[order], U[:, order]
    np.save(cache_w, w)
    np.save(cache_u, U.astype(np.float32))
    return w, U.astype(np.float32)


# ---------------------------------------------------------------------------
# Spectral band-energy descriptor
# ---------------------------------------------------------------------------
def spectral_coeffs(
    fcd: FaceColorData, k: int = 40, channels=("L", "a", "b"), n_basis: int = 300,
    standardize_modes: bool = False, mesh=None, cache_dir=None,
) -> np.ndarray:
    """Per-specimen **signed, compressed GFT coefficients** → (N, k*len(channels)).

    EXP-19: keeping the signed low-frequency coefficients (phase preserved) retains more
    structure than band *energy* (magnitude), and truncating to k≈40 modes denoises + compacts.
    For each channel, project the mean-removed per-vertex signal onto the k lowest Laplacian
    eigenvectors. This is the recommended replacement for `spectral_descriptor` (band energy).
    """
    w, U = laplacian_eigenbasis(max(n_basis, k), mesh=mesh, cache_dir=cache_dir)
    Uk = U[:, :k]                                   # k lowest-frequency modes (compression)
    vlab = vertex_lab(fcd, mesh=mesh, cache_dir=cache_dir)
    sig = {
        "L": vlab[..., 0], "a": vlab[..., 1], "b": vlab[..., 2],
        "chroma": np.sqrt(vlab[..., 1] ** 2 + vlab[..., 2] ** 2),
    }
    blocks = []
    for ch in channels:
        s = sig[ch].astype(np.float64)
        s = s - s.mean(axis=1, keepdims=True)
        c = s @ Uk                                  # (N, k) signed coefficients
        if standardize_modes:                       # equalize modes so low-energy ones compete
            c = c / (np.abs(c).std(axis=0, keepdims=True) + 1e-9)
        blocks.append(c)
    return np.concatenate(blocks, axis=1)


def _band_edges(eigvals: np.ndarray, n_bands: int, log: bool = True) -> np.ndarray:
    lo = max(eigvals[1], 1e-6)            # skip the DC (constant) eigenvector
    hi = eigvals[-1]
    if log:
        return np.geomspace(lo, hi, n_bands + 1)
    return np.linspace(lo, hi, n_bands + 1)


def spectral_descriptor(
    fcd: FaceColorData, k: int = 300, n_bands: int = 12,
    channels=("L", "a", "b", "chroma"), normalize: bool = True, log_bands: bool = True,
    mesh=None, cache_dir=None,
) -> np.ndarray:
    """Per-specimen spectral band-energy descriptor → (N, n_bands*len(channels)).

    For each requested color channel, project the (mean-removed) per-vertex signal onto the
    Laplacian eigenbasis and sum squared coefficients within each eigenvalue band. The
    result encodes *how much pattern energy at each spatial scale* — coarse→fine.
    `mesh`/`cache_dir` default to fishy; inject for any shared-atlas dataset.
    """
    w, U = laplacian_eigenbasis(k, mesh=mesh, cache_dir=cache_dir)
    vlab = vertex_lab(fcd, mesh=mesh, cache_dir=cache_dir)   # (N, Nv, 3)
    N = vlab.shape[0]

    sig = {
        "L": vlab[..., 0],
        "a": vlab[..., 1],
        "b": vlab[..., 2],
        "chroma": np.sqrt(vlab[..., 1] ** 2 + vlab[..., 2] ** 2),
    }
    edges = _band_edges(w, n_bands, log=log_bands)
    # assign each eigenvector (skip index 0 = DC) to a band
    band_of = np.clip(np.digitize(w, edges) - 1, 0, n_bands - 1)
    band_of[0] = -1                      # exclude DC

    blocks = []
    for ch in channels:
        s = sig[ch].astype(np.float64)               # (N, Nv)
        s = s - s.mean(axis=1, keepdims=True)         # remove DC
        coeff = s @ U                                 # (N, k) spectral coefficients
        energy = coeff ** 2
        be = np.zeros((N, n_bands))
        for band in range(n_bands):
            cols = np.where(band_of == band)[0]
            if cols.size:
                be[:, band] = energy[:, cols].sum(axis=1)
        if normalize:                                 # per-channel relative spectral shape
            tot = be.sum(axis=1, keepdims=True)
            tot[tot == 0] = 1.0
            be = be / tot
        blocks.append(be)
    return np.concatenate(blocks, axis=1)
