"""Engine for the interactive feedback-morphing morphospace demo.

Pure compute — no web framework. One `Session` per (dataset) holds:
  * the combined COLOR+PATTERN per-region block descriptor (EXP-40 style, so every
    descriptor column belongs to exactly one body region -> exact PC->region attribution),
  * a PCA whose z-scored scores Z are the morphospace coordinates,
  * the region-saliency matrix C[r,k] (where PC k lives on the body), and
  * the live morph state: per-PC accumulated ranking triplets that re-space ONE PC axis at a
    time via the `semisup.spring_embed` 1-D ordinal energy model.

The feedback loop: show an anchor + a panel of specimens spread along the ACTIVE PC; the expert
ranks them most->least similar (or says "none are more/less similar" -> a tie, which adds no
constraint); the active PC coordinate is re-morphed from its ORIGINAL position using ALL
accumulated triplets (deterministic + resettable). Morphospace + dendrogram redraw from Z.

Datasets:
  * fishy  (n=250, synthetic, HAS ground truth)  — render_side_view lateral view, R=128 regions
  * mussel (n=31,  real,      NO ground truth)    — render_textured_rgba valve view, R=48 regions
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from fishpipe import config, data, features, render, semisup, textons  # noqa: E402
from fishpipe.dataset import MUSSELS, build_face_colors as ds_build_face_colors  # noqa: E402

PDIM = 11           # per-region descriptor width: 5 color + 6 texture
COLOR_CH = 5        # first 5 channels of each block are color, last 6 are pattern
CHANNELS = ["meanL", "mean_a*", "mean_b*", "hiChroma_a*", "hiChroma_b*",
            "bandpassL_mean", "bandpassL_std", "bandpassL_p90",
            "localLstd_mean", "multiscaleBP_mean", "darkFraction"]


# --------------------------------------------------------------------------- descriptor
def region_labels_for(fcd, n_regions: int, cache_dir: Path, seed: int = 0) -> np.ndarray:
    """Per-face region id (Nf,) int32 from THIS dataset's 3D centroids; cached per dataset.

    Built directly from `fcd.centroids` (NOT features._region_labels, which is hardcoded to the
    fishy cache) so the same code path serves fishy and mussel.
    """
    p = Path(cache_dir) / f"region_labels_{n_regions}.npy"
    if p.exists():
        return np.load(p)
    lab = KMeans(n_clusters=n_regions, n_init=4, random_state=seed).fit_predict(
        fcd.centroids).astype(np.int32)
    np.save(p, lab)
    return lab


def _region_desc(lab, chroma, jet, faces) -> np.ndarray:
    """(N, 11) color+texture summary of one body region across all specimens."""
    Lf, af, bf = lab[:, faces, 0], lab[:, faces, 1], lab[:, faces, 2]
    am = np.argmax(chroma[:, faces], axis=1)                     # highest-chroma face / specimen
    color = np.column_stack([Lf.mean(1), af.mean(1), bf.mean(1),
                             af[np.arange(len(am)), am], bf[np.arange(len(am)), am]])
    bp, ms, ls = jet[:, faces, 6], jet[:, faces, 7], jet[:, faces, 8]
    tex = np.column_stack([np.abs(bp).mean(1), bp.std(1), np.percentile(np.abs(bp), 90, 1),
                           ls.mean(1), np.abs(ms).mean(1), (Lf < 20).mean(1)])
    return np.hstack([color, tex])


def build_descriptor(fcd, reg, n_regions, cache_dir):
    """Concatenate per-region 11-dim blocks in a FIXED region order -> (N, len(rids)*11).

    Returns (X, rids) where rids is the ordered list of region ids that have >=20 faces, so
    column c of X belongs to region rids[c // 11], channel c % 11.
    """
    lab = features.rgb_to_lab(fcd.colors)                        # (N, Nf, 3)
    chroma = np.sqrt(lab[..., 1] ** 2 + lab[..., 2] ** 2)
    jet = textons.local_jet(fcd, scales=(2, 4), cache_dir=cache_dir)   # (N, Nf, 9)
    rids = [r for r in range(n_regions) if (reg == r).sum() >= 20]
    blocks = [_region_desc(lab, chroma, jet, np.where(reg == r)[0]) for r in rids]
    return np.hstack(blocks), rids


# --------------------------------------------------------------------------- dataset registry
@dataclass
class DatasetSpec:
    name: str
    title: str
    n_regions: int
    has_gt: bool
    renderer: str               # "side" (fishy lateral) or "valve" (mussel textured)


DATASETS = {
    "fishy": DatasetSpec("fishy", "Fishy (synthetic, n=250)", n_regions=128, has_gt=True, renderer="side"),
    "mussel": DatasetSpec("mussel", "Mussel (real, n=31)", n_regions=48, has_gt=False, renderer="valve"),
}


def list_datasets():
    return [{"name": s.name, "title": s.title, "has_gt": s.has_gt} for s in DATASETS.values()]


@dataclass
class LoadedDataset:
    spec: DatasetSpec
    mesh: object
    fcd: object
    names: list
    cache_dir: Path
    reg: np.ndarray
    rids: list
    Xs: np.ndarray
    bounds: tuple
    gt: object = None                       # GroundTruth or None
    gt_factors: dict = field(default_factory=dict)   # factor name -> (N,) int (fishy only)


_CACHE: dict[str, LoadedDataset] = {}


def load_dataset(name: str) -> LoadedDataset:
    if name in _CACHE:
        return _CACHE[name]
    spec = DATASETS[name]
    if name == "fishy":
        fcd = data.build_face_colors(verbose=False)
        mesh = data.load_mesh()
        cache_dir = config.CACHE_DIR
        gt = data.load_ground_truth()
        gt_factors = {f: np.asarray(gt.labels[f]) for f in gt.labels}
    elif name == "mussel":
        out = ds_build_face_colors(MUSSELS, force=False, verbose=False)
        fcd = out[0] if isinstance(out, tuple) else out
        mesh = MUSSELS.load_mesh()
        cache_dir = MUSSELS.cache_dir
        gt = None
        gt_factors = {}
    else:
        raise KeyError(name)

    reg = region_labels_for(fcd, spec.n_regions, cache_dir)
    X, rids = build_descriptor(fcd, reg, spec.n_regions, cache_dir)
    assert X.shape[1] == len(rids) * PDIM, "descriptor block-width mismatch"
    Xs = StandardScaler().fit_transform(X)
    bounds = (float(mesh.vertices[:, 2].min()), float(mesh.vertices[:, 2].max()),
              float(mesh.vertices[:, 1].min()), float(mesh.vertices[:, 1].max()))
    ld = LoadedDataset(spec=spec, mesh=mesh, fcd=fcd, names=list(fcd.names), cache_dir=Path(cache_dir),
                       reg=reg, rids=rids, Xs=Xs, bounds=bounds, gt=gt, gt_factors=gt_factors)
    _CACHE[name] = ld
    return ld


# --------------------------------------------------------------------------- helpers
def _zscore_cols(Z):
    return (Z - Z.mean(0)) / (Z.std(0) + 1e-9)


def _fps(coords, k, seed=0, start=None):
    """Farthest-point sample k DISTINCT indices from `coords` (n, d). Diverse, not kNN.

    Selected indices are masked to -1 before each argmax so degenerate/duplicate coordinates
    (e.g. many specimens sharing one active-PC value) still yield exactly min(k, n) distinct
    picks instead of breaking early on a repeated argmax.
    """
    n = len(coords)
    k = min(k, n)
    rng = np.random.default_rng(seed)
    first = int(start) if start is not None else int(rng.integers(n))
    idx = [first]
    d = np.linalg.norm(coords - coords[first], axis=1)
    while len(idx) < k:
        d[idx] = -1.0                                   # exclude already-selected
        j = int(np.argmax(d))                           # unselected pt of max distance (>=0 > -1)
        if j in idx:                                    # safety: nothing left to pick
            break
        idx.append(j)
        d = np.minimum(d, np.linalg.norm(coords - coords[j], axis=1))
    return idx


# --------------------------------------------------------------------------- learned expert metric
def _delta2(Z, pairs):
    """pairs = [(i,j,amp)] -> (per-PC squared differences (n,K), amplitudes (n,))."""
    if not pairs:
        return np.zeros((0, Z.shape[1])), np.zeros(0)
    I = np.array([p[0] for p in pairs]); J = np.array([p[1] for p in pairs])
    return (Z[I] - Z[J]) ** 2, np.array([p[2] for p in pairs], dtype=float)


def _fit_diag_metric(DS, aS, DD, aD, w0, lam, mu, iters=500, lr=0.4):
    """Diagonal per-PC metric w>=0 for d^2_w(i,j)=sum_k w_k (z_ik-z_jk)^2 that pulls SIMILAR pairs
    close and pushes DISSIMILAR pairs past margin mu, shrunk toward w0, with sum(w)=K fixed.
    Convex; projected sub-gradient. Each w_k is "how much PC k drives the expert's distinctions"."""
    K = len(w0); w = w0.astype(float).copy()
    nc = max(1, len(DS) + len(DD))
    pull = (aS[:, None] * DS).sum(0) / nc if len(DS) else np.zeros(K)   # constant (linear term)
    for _ in range(iters):
        if len(DD):
            viol = (DD @ w) < mu
            push = (aD[viol, None] * DD[viol]).sum(0) / nc if viol.any() else np.zeros(K)
        else:
            push = np.zeros(K)
        w = w - lr * (pull - push + 2.0 * lam * (w - w0))
        np.maximum(w, 0.0, out=w)
        s = w.sum()
        w = (w * K / s) if s > 1e-9 else w0.copy()
    return w


# --------------------------------------------------------------------------- 8-cluster recovery
def _lowrank_metric(Z, pairs, t, rank=4, iters=600, lr=0.05, lam=1e-3, seed=0):
    """Low-rank metric L (rank x K) s.t. ||L(z_i-z_j)||^2 ~ target t_ij (graded). Returns the projected
    coords Z @ L.T. Rank ~ #latent factors; far fewer params than a full metric -> works from few
    labels in a high-K space (EXP-43/44: diagonal too weak, full Mahalanobis overfits)."""
    rng = np.random.default_rng(seed)
    K = Z.shape[1]
    L = rng.standard_normal((rank, K)) * 0.3
    Dz = np.array([Z[i] - Z[j] for (i, j) in pairs], float)
    t = np.asarray(t, float)
    mL = vL = 0.0
    b1, b2, eps = 0.9, 0.999, 1e-8
    for it in range(1, iters + 1):
        Y = Dz @ L.T
        e = (Y ** 2).sum(1) - t
        g = 4.0 * ((e[:, None] * Y).T @ Dz) / len(pairs) + 2.0 * lam * L
        mL = b1 * mL + (1 - b1) * g
        vL = b2 * vL + (1 - b2) * g * g
        L = L - lr * (mL / (1 - b1 ** it)) / (np.sqrt(vL / (1 - b2 ** it)) + eps)
    return Z @ L.T


def _balanced_code(cen):
    """3-bit code per centroid via recursive balanced median bisection on the local top PC -> a clean
    2x2x2 factor grid (only used when n_clusters == 8)."""
    code = np.zeros((len(cen), 3), int)

    def split(idx, level):
        if level == 3 or len(idx) <= 1:
            return
        X = cen[idx]
        Xc = X - X.mean(0)
        u = np.linalg.svd(Xc, full_matrices=False)[2][0]
        order = np.argsort(Xc @ u)
        hi = set(np.asarray(idx)[order[len(idx) // 2:]])
        lo_i, hi_i = [], []
        for k in idx:
            (hi_i if k in hi else lo_i).append(k)
            if k in hi:
                code[k, level] = 1
        split(lo_i, level + 1)
        split(hi_i, level + 1)

    split(list(range(len(cen))), 0)
    return code


def _grid_layout(Y, labels, n_clusters):
    """Per-specimen (x, y) in [0,1]^2: clusters placed on a structured grid (factor grid for k=8, else a
    hierarchical row order), members jittered around their cell. Lets the demo SHOW the recovered groups
    even though a 2-D force layout can't settle into them."""
    cen = np.array([Y[labels == c].mean(0) if (labels == c).any() else Y.mean(0)
                    for c in range(n_clusters)])
    if n_clusters == 8:
        code = _balanced_code(cen)
        col = code[:, 0] * 2 + code[:, 1]                       # 4 quadrants (strong factors)
        cell = np.stack([col % 2 + (code[:, 2]) * 0.0, col // 2], 1).astype(float)
        cell[:, 0] = (col % 2) + 0.0                            # x: 0/1 (two columns of quadrants)
        cx = (col % 2).astype(float)
        cy = (col // 2).astype(float) * 2 + code[:, 2]          # y: quadrant-row*2 + weak-factor row
        gx, gy = cx, cy
        nx, ny = 2.0, 4.0
    else:
        order = np.argsort(cen @ (np.linalg.svd(cen - cen.mean(0), full_matrices=False)[2][0]))
        rank_of = {c: r for r, c in enumerate(order)}
        ncol = int(np.ceil(np.sqrt(n_clusters)))
        gx = np.array([rank_of[c] % ncol for c in range(n_clusters)], float)
        gy = np.array([rank_of[c] // ncol for c in range(n_clusters)], float)
        nx = ny = float(ncol)
    rng = np.random.default_rng(0)
    pos = np.zeros((len(labels), 2))
    for i, c in enumerate(labels):
        pos[i] = [(gx[c] + 0.5) / nx + rng.normal(0, 0.05),
                  1.0 - (gy[c] + 0.5) / ny + rng.normal(0, 0.05 / max(1, ny / nx))]
    return pos


# --------------------------------------------------------------------------- session
class Session:
    """One interactive session over one dataset. Holds live morph state."""

    def __init__(self, dataset: str, n_pcs: int = 24, seed: int = 0):
        ld = load_dataset(dataset)
        self.ld = ld
        self.dataset = dataset
        self.seed = seed
        self.N = len(ld.names)
        self.K = int(min(n_pcs, self.N - 1))
        if self.K < 2:
            raise ValueError(f"need >=3 specimens for a 2-D morphospace (N={self.N}, K={self.K})")

        self.pca = PCA(self.K, random_state=seed).fit(ld.Xs)
        self.W = self.pca.components_                       # (K, F)
        self.evr = self.pca.explained_variance_ratio_
        self.Z0 = _zscore_cols(ld.Xs @ self.W.T)            # (N, K) baseline morphospace coords
        self.Z = self.Z0.copy()                            # live (morphed) coords

        # region saliency C[r,k] = share of PC k's squared loadings living in region r
        self.C = np.zeros((len(ld.rids), self.K))
        for i in range(len(ld.rids)):
            blk = slice(i * PDIM, (i + 1) * PDIM)
            self.C[i] = (self.W[:, blk] ** 2).sum(1)

        # per-PC color vs pattern share
        self.color_share = np.zeros(self.K)
        for k in range(self.K):
            cs = sum((self.W[k, i * PDIM:i * PDIM + COLOR_CH] ** 2).sum() for i in range(len(ld.rids)))
            self.color_share[k] = cs    # total loading norm is 1, so pattern share = 1 - cs

        self.active_pc = 0
        self.axes = (0, 1)
        self.trips_by_pc: dict[int, list] = {}             # pc -> accumulated (a, near, far) triplets
        self.round = 0
        self.history: list[dict] = []                      # feedback log
        self.shown_anchors: list[int] = []                 # anchors shown so far (for "New anchor")

    # ---- morph -----------------------------------------------------------
    def _remorph_pc(self, pc: int):
        """Re-space PC `pc` from its ORIGINAL coordinate using ALL accumulated triplets."""
        trips = self.trips_by_pc.get(pc, [])
        y0 = self.Z0[:, pc:pc + 1].copy()
        if not trips:
            self.Z[:, pc] = self.Z0[:, pc]
            return
        y = semisup.spring_embed(y0, trips, n_iter=500, lr=0.008, margin=0.4, lam=0.06, seed=self.seed)
        y = (y[:, 0] - y[:, 0].mean()) / (y[:, 0].std() + 1e-9)
        self.Z[:, pc] = y

    def feedback(self, anchor: int, ranked: list[int], tied: bool, remote: int | None = None):
        """Apply one feedback round to the ACTIVE PC.

        ranked = neighbor indices ordered most->least similar to anchor. tied=True ("none are
        more or less similar") asserts NO ordering -> adds no triplet (correct: a tie is not a
        constraint). `remote` (optional far-check specimen) is treated as farther than all.
        """
        pc = self.active_pc
        added = 0
        if not tied and len(ranked) >= 2:
            trips = semisup.triplets_from_ranking(anchor, list(ranked), remote=remote)
            self.trips_by_pc.setdefault(pc, []).extend(trips)
            added = len(trips)
        self._remorph_pc(pc)
        self.round += 1
        self.history.append({"round": self.round, "pc": pc, "anchor": int(anchor),
                             "ranked": [int(r) for r in ranked], "tied": bool(tied),
                             "remote": (int(remote) if remote is not None else None),
                             "triplets_added": added})
        return {"round": self.round, "pc": pc, "triplets_added": added,
                "total_triplets_pc": len(self.trips_by_pc.get(pc, []))}

    def reset(self, pc: int | None = None):
        """Forget feedback: one PC if given, else all."""
        if pc is None:
            self.trips_by_pc.clear()
            self.Z = self.Z0.copy()
            self.shown_anchors.clear()
        else:
            self.trips_by_pc.pop(pc, None)
            self.Z[:, pc] = self.Z0[:, pc]
        self.round += 1

    # ---- selectors -------------------------------------------------------
    def set_active_pc(self, pc: int):
        self.active_pc = int(np.clip(pc, 0, self.K - 1))
        # keep the active PC on the morphospace x-axis for a coherent "morph this axis" story
        y = self.axes[1] if self.axes[1] != self.active_pc else (self.active_pc + 1) % self.K
        self.axes = (self.active_pc, y)

    def set_axes(self, x: int, y: int):
        self.axes = (int(np.clip(x, 0, self.K - 1)), int(np.clip(y, 0, self.K - 1)))

    # ---- panel sampling --------------------------------------------------
    def pick_anchor(self) -> int:
        """Return a FRESH, diverse anchor each call: the specimen farthest (in the PCA similarity
        space) from every anchor shown so far, so repeated 'New anchor' clicks sweep the space
        instead of repeating the same one. Cycles once all specimens have been anchors."""
        if len(self.shown_anchors) >= self.N:
            self.shown_anchors = []
        if not self.shown_anchors:
            return int(_fps(self.Z0, 1, seed=self.seed)[0])
        d = np.full(self.N, np.inf)
        for s in self.shown_anchors:
            d = np.minimum(d, np.linalg.norm(self.Z0 - self.Z0[s], axis=1))
        d[self.shown_anchors] = -1.0
        return int(np.argmax(d))

    def panel(self, anchor: int | None = None, k: int = 5):
        """anchor + k DIVERSE neighbors (overall color+pattern similarity) + one remote far-check.

        Candidates are the anchor's overall-nearest specimens (broadly comparable); we then
        farthest-point sample them in the FULL similarity space so the panel is diverse rather
        than near-duplicates. `remote` is the specimen globally farthest from the anchor.
        """
        if anchor is None:
            anchor = self.pick_anchor()
        anchor = int(anchor)
        if anchor not in self.shown_anchors:           # remember it so the next pick differs
            self.shown_anchors.append(anchor)
        d_full = np.linalg.norm(self.Z0 - self.Z0[anchor], axis=1)
        pool = [int(j) for j in np.argsort(d_full) if j != anchor][:min(self.N - 1, 40)]
        sel = _fps(self.Z0[pool], k, seed=self.seed + len(self.shown_anchors), start=0)
        neighbors = [pool[s] for s in sel]
        remote = int(np.argmax(d_full))
        if remote in neighbors or remote == anchor:
            remote = None
        return {"anchor": anchor, "neighbors": neighbors, "remote": remote, "active_pc": self.active_pc}

    # ---- live similarity graph (client-side force layout) -----------------
    def graph_edges(self, k_nn: int = 8):
        """kNN similarity edges (in the standardized color+pattern descriptor) + an MST so the
        graph is CONNECTED (a tug anywhere can ripple everywhere). Rest lengths are the high-D
        distances rescaled to the PCA-2D seed scale, so the seed layout starts near equilibrium.
        Returns list of [i, j, rest_length]."""
        Xs = self.ld.Xs
        D = squareform(pdist(Xs))
        kk = int(min(k_nn, self.N - 1))
        nbr = np.argsort(D, axis=1)[:, 1:kk + 1]
        edge_d: dict[tuple, float] = {}
        for i in range(self.N):
            for j in nbr[i]:
                e = (i, int(j)) if i < j else (int(j), i)
                edge_d[e] = float(D[i, j])
        mst = minimum_spanning_tree(D).tocoo()                 # guarantee connectivity
        for i, j, w in zip(mst.row, mst.col, mst.data):
            e = (int(i), int(j)) if i < j else (int(j), int(i))
            edge_d.setdefault(e, float(w))
        seed = self.Z0[:, list(self.axes)]
        Dh = np.median(list(edge_d.values())) or 1.0
        Ds = np.median([np.linalg.norm(seed[a] - seed[b]) for a, b in edge_d]) or 1.0
        scale = Ds / Dh
        return [[a, b, round(edge_d[(a, b)] * scale, 4)] for (a, b) in edge_d]

    def graph(self, k_nn: int = 8):
        """Everything the client force-layout needs: seed node positions (current PC axes),
        similarity edges, per-PC color values, and ground-truth labels (fishy)."""
        x, y = self.axes
        seed = self.Z0
        nodes = [{"i": i, "name": n, "x": round(float(seed[i, x]), 4), "y": round(float(seed[i, y]), 4)}
                 for i, n in enumerate(self.ld.names)]
        pc_values = []
        for pc in range(self.K):
            a = self.Z0[:, pc]
            pc_values.append([round(float(v), 4) for v in (a - a.min()) / (np.ptp(a) + 1e-9)])
        out = {"nodes": nodes, "edges": self.graph_edges(k_nn), "axes": [x, y],
               "active_pc": self.active_pc, "pc_values": pc_values, "has_gt": self.ld.spec.has_gt}
        if self.ld.spec.has_gt:
            out["gt_factors"] = {f: [int(v) for v in self.ld.gt_factors[f]] for f in self.ld.gt_factors}
            try:
                joint = data.joint_label(self.ld.gt, factors=("belly", "tail", "stripe"))
            except Exception:
                joint = np.zeros(self.N, dtype=int)
            out["gt_joint"] = [int(v) for v in joint]
        return out

    def dendro_leaf_indices(self, k: int = 28):
        """Deterministic representative leaf set (all for small n, else a seeded random sample).
        Independent of the live layout, so hi-res leaf thumbnails can be prerendered up front."""
        if self.N <= k + 6:
            return list(range(self.N))
        return sorted(np.random.default_rng(self.seed).choice(self.N, k, replace=False).tolist())

    def dendro_from_positions(self, positions, k: int = 28):
        """Subset of the CLIENT's live 2-D layout for WYSIWYG clustering.

        A REPRESENTATIVE subset (random / all for small n) rather than farthest-point sampling
        (which over-picks the convex hull and lopsides the tree), then per-axis standardized so x
        and y count equally. Outlier-distance dominance is handled at draw time by plotting the
        Ward topology at UNIFORM merge heights (see render_dendrogram), so no clipping here.
        """
        P = np.asarray(positions, dtype=float)
        if P.ndim != 2 or P.shape[0] != self.N or P.shape[1] != 2:
            raise ValueError(f"positions must be ({self.N}, 2), got {P.shape}")
        idx = self.dendro_leaf_indices(k)
        Q = P[idx]
        Q = (Q - Q.mean(0)) / (Q.std(0) + 1e-9)
        return idx, Q

    # ---- views -----------------------------------------------------------
    def morphospace(self):
        x, y = self.axes
        Zx, Zy = self.Z[:, x], self.Z[:, y]
        pts = []
        for i, name in enumerate(self.ld.names):
            p = {"i": i, "name": name, "x": float(Zx[i]), "y": float(Zy[i])}
            pts.append(p)
        out = {"axes": [x, y], "active_pc": self.active_pc, "round": self.round, "points": pts,
               "has_gt": self.ld.spec.has_gt}
        if self.ld.spec.has_gt:
            out["gt_factors"] = {f: [int(v) for v in self.ld.gt_factors[f]] for f in self.ld.gt_factors}
            try:
                joint = data.joint_label(self.ld.gt, factors=("belly", "tail", "stripe"))
            except Exception:
                joint = np.zeros(self.N, dtype=int)
            out["gt_joint"] = [int(v) for v in joint]   # always present when has_gt
        # active-PC value (for continuous coloring when no GT)
        a = self.Z[:, self.active_pc]
        out["active_value"] = [float(v) for v in (a - a.min()) / (np.ptp(a) + 1e-9)]
        return out

    def pc_info(self, pc: int | None = None):
        pc = self.active_pc if pc is None else int(pc)
        order = np.argsort(self.C[:, pc])[::-1]
        top = [{"region": int(self.ld.rids[i]), "share": float(self.C[i, pc])} for i in order[:6]]
        return {"pc": pc, "K": self.K,
                "evr": float(self.evr[pc]),
                "color_share": float(self.color_share[pc]),
                "pattern_share": float(1.0 - self.color_share[pc]),
                "top_regions": top,
                "morphed": bool(len(self.trips_by_pc.get(pc, [])) > 0),
                "triplets": len(self.trips_by_pc.get(pc, []))}

    def region_face_rgb(self, pc: int | None = None):
        """(Nf, 3) uint8 inferno heatmap of where PC `pc` lives on the body."""
        import matplotlib.cm as cm
        pc = self.active_pc if pc is None else int(pc)
        score_r = np.zeros(int(self.ld.reg.max()) + 1)
        for i, r in enumerate(self.ld.rids):
            score_r[r] = self.C[i, pc]
        fs = score_r[self.ld.reg]
        s = (fs - fs.min()) / (np.ptp(fs) + 1e-9)
        return (cm.inferno(s)[:, :3] * 255).astype(np.uint8)

    def dendro_subset(self, k: int = 28):
        """A diverse FPS subset (legible leaves) + their current full-Z coords for linkage."""
        idx = sorted(_fps(self.Z, k, seed=self.seed))
        return idx, self.Z[idx]

    # ---- explain the expert's grouping in the PCA basis -------------------
    def region_face_rgb_weighted(self, weights):
        """(Nf,3) inferno heatmap of region importance = sum_k weights_k * C[r,k] (where the
        expert's learned metric lives on the body)."""
        import matplotlib.cm as cm
        w = np.asarray(weights, dtype=float)
        ireg = (self.C * w[None, :]).sum(1)                 # (len(rids),)
        score_r = np.zeros(int(self.ld.reg.max()) + 1)
        for i, r in enumerate(self.ld.rids):
            score_r[r] = ireg[i]
        fs = score_r[self.ld.reg]
        s = (fs - fs.min()) / (np.ptp(fs) + 1e-9)
        return (cm.inferno(s)[:, :3] * 255).astype(np.uint8)

    def explain(self, pairs, positions=None, perm=300, boot=200, seed=0):
        """Learn which PCs explain the expert's similar/dissimilar feedback, attribute to
        color/pattern + body regions, and validate. `pairs` = [{a,b,kind('near'/'far'),amp}].
        Stores the region weights on self._explain_w for the heatmap endpoint."""
        Z, K = self.Z0, self.K
        S = [(int(p["a"]), int(p["b"]), float(p.get("amp", 1.0))) for p in pairs if p.get("kind") == "near"]
        D = [(int(p["a"]), int(p["b"]), float(p.get("amp", 1.0))) for p in pairs if p.get("kind") == "far"]
        if not (S or D):
            return {"ok": False, "reason": "no similar/dissimilar feedback yet"}
        DS, aS = _delta2(Z, S); DD, aD = _delta2(Z, D)
        alld2 = np.vstack([x for x in (DS, DD) if len(x)])
        mu = float(np.median(alld2.sum(1)))                 # scale-matched margin
        w0 = np.ones(K)
        lam = 1.0
        w = _fit_diag_metric(DS, aS, DD, aD, w0, lam, mu)
        dw = w - w0                                         # deviation from uniform = expert contribution
        dwp = np.maximum(dw, 0.0)
        self._explain_w = dwp
        swp = float(dwp.sum()) + 1e-9

        # attribution
        color_share = float((dwp * self.color_share).sum() / swp)
        ireg = (self.C * dwp[None, :]).sum(1)
        rorder = np.argsort(ireg)[::-1]
        top_regions = [{"region": int(self.ld.rids[i]), "imp": round(float(ireg[i]), 4)} for i in rorder[:6]]
        feat_imp = (dwp[:, None] * (self.W ** 2)).sum(0)    # (F,)
        forder = np.argsort(feat_imp)[::-1]
        top_features = [{"region": int(self.ld.rids[int(c) // PDIM]), "channel": CHANNELS[int(c) % PDIM],
                         "imp": round(float(feat_imp[c]), 4)} for c in forder[:6]]
        pcs = [{"pc": k, "dw": round(float(dw[k]), 4), "w": round(float(w[k]), 4),
                "evr": round(float(self.evr[k]), 4), "color_share": round(float(self.color_share[k]), 3)}
               for k in range(K)]
        kstar = int(np.argmax(dw))
        zk = Z[:, kstar]; o = np.argsort(zk)
        ne = min(3, self.N)
        exemplars = {"pc": kstar,
                     "low": [{"i": int(o[i]), "name": self.ld.names[int(o[i])]} for i in range(ne)],
                     "high": [{"i": int(o[-1 - i]), "name": self.ld.names[int(o[-1 - i])]} for i in range(ne)]}

        # ---- validation ----
        def _acc(wv, teS, teD):
            ok = sum(1 for (i, j, _) in teS if (Z[i] - Z[j]) ** 2 @ wv < mu)
            ok += sum(1 for (i, j, _) in teD if (Z[i] - Z[j]) ** 2 @ wv >= mu)
            return ok / max(1, len(teS) + len(teD))

        allc = [("S", i) for i in range(len(S))] + [("D", i) for i in range(len(D))]
        ho = base = 0.0
        for kind, hi in allc:                               # leave-one-constraint-out
            trS = [S[i] for i in range(len(S)) if not (kind == "S" and i == hi)]
            trD = [D[i] for i in range(len(D)) if not (kind == "D" and i == hi)]
            ds, as_ = _delta2(Z, trS); dd, ad = _delta2(Z, trD)
            wt = _fit_diag_metric(ds, as_, dd, ad, w0, lam, mu)
            teS, teD = ([S[hi]], []) if kind == "S" else ([], [D[hi]])
            ho += _acc(wt, teS, teD); base += _acc(w0, teS, teD)
        heldout = ho / len(allc); baseline = base / len(allc)

        rng = np.random.default_rng(seed)
        allp = [(i, j, a, 1) for (i, j, a) in S] + [(i, j, a, 0) for (i, j, a) in D]
        nS = len(S)
        permw = np.empty((perm, K))
        for b in range(perm):                               # permute similar/dissimilar labels
            idx = rng.permutation(len(allp))
            ps = [allp[t][:3] for t in idx[:nS]]; pd = [allp[t][:3] for t in idx[nS:]]
            ds, as_ = _delta2(Z, ps); dd, ad = _delta2(Z, pd)
            permw[b] = _fit_diag_metric(ds, as_, dd, ad, w0, lam, mu)
        pval = [round(float((permw[:, k] >= w[k]).mean()), 3) for k in range(K)]

        bw = np.empty((boot, K))
        for b in range(boot):                               # bootstrap stability
            sb = [S[t] for t in rng.integers(0, len(S), len(S))] if S else []
            db = [D[t] for t in rng.integers(0, len(D), len(D))] if D else []
            ds, as_ = _delta2(Z, sb); dd, ad = _delta2(Z, db)
            bw[b] = _fit_diag_metric(ds, as_, dd, ad, w0, lam, mu)
        ci = [[round(float(np.percentile(bw[:, k], 5)), 3), round(float(np.percentile(bw[:, k], 95)), 3)] for k in range(K)]
        for p, c in zip(pcs, range(K)):
            p["p"] = pval[c]; p["ci"] = ci[c]

        # leave-one-out PLACEMENT: predict each held-out specimen's 2-D position from the others
        # under the learned metric (kNN-barycentric) -> how well the metric predicts location.
        place_err = None
        if positions is not None:
            P = np.asarray(positions, float)
            if P.shape == (self.N, 2):
                errs = []
                for j in range(self.N):
                    d2 = ((Z - Z[j]) ** 2 * w[None, :]).sum(1); d2[j] = np.inf
                    nn = np.argsort(d2)[:min(8, self.N - 1)]
                    al = np.exp(-d2[nn] / (np.median(d2[np.isfinite(d2)]) + 1e-9)); al /= al.sum()
                    errs.append(float(np.linalg.norm((al[:, None] * P[nn]).sum(0) - P[j])))
                scale = float(np.median(pdist(P))) if self.N > 1 else 1.0
                place_err = round(float(np.median(errs)) / (scale + 1e-9), 3)

        if heldout >= baseline + 0.08 and heldout > 0.6:
            coverage = "good"
        elif heldout > baseline + 0.02:
            coverage = "weak"
        else:
            coverage = "none"

        return {"ok": True, "K": K, "n_similar": len(S), "n_dissimilar": len(D),
                "pcs": pcs, "color_share": round(color_share, 3), "pattern_share": round(1 - color_share, 3),
                "top_regions": top_regions, "top_features": top_features, "exemplars": exemplars,
                "heldout_acc": round(heldout, 3), "baseline_acc": round(baseline, 3),
                "place_err": place_err, "coverage": coverage}

    # ---- recover the latent cluster structure from graded feedback (EXP-42..45 recipe) -----
    def recover(self, pairs, n_clusters: int = 8, rank: int = 4, seed: int = 0):
        """Learn a low-rank metric from the graded similar/dissimilar feedback, cluster in it (GMM), and
        lay the clusters on a structured factor grid. `pairs` = [{a,b,kind('near'/'far'),amp}].

        The graded amplitude IS the signal: each pair's TARGET squared distance is small for very-similar
        and large for very-dissimilar, so partial-overlap pairs still separate proportionally (no need for
        the rare all-different pair). Returns per-specimen recovered labels + a [0,1]^2 grid layout (+ ARI
        vs the joint GT when available)."""
        Z = self.Z0
        P, t = [], []
        for p in pairs:
            a, b = int(p["a"]), int(p["b"])
            amp = float(p.get("amp", 1.0))
            P.append((a, b))
            t.append((1.0 - amp) if p.get("kind") == "near" else (1.0 + amp))   # near->0, far->2
        if len(P) < max(3, n_clusters - 1):
            return {"ok": False, "reason": "need more similar/dissimilar feedback to recover clusters"}
        # scale graded targets to the data's distance spread so the metric fit is well-conditioned
        d2 = pdist(Z) ** 2
        t = np.asarray(t) * (np.percentile(d2, 95) / 2.0)
        Y = _lowrank_metric(Z, P, t, rank=rank, seed=seed)
        labels = GaussianMixture(int(n_clusters), covariance_type="full", n_init=3,
                                 random_state=seed).fit_predict(Y)
        pos = _grid_layout(Y, labels, int(n_clusters))
        out = {"ok": True, "n_clusters": int(n_clusters), "rank": int(rank),
               "labels": [int(v) for v in labels],
               "grid": [[round(float(x), 4), round(float(y), 4)] for x, y in pos],
               "sizes": [int(v) for v in np.bincount(labels, minlength=int(n_clusters))]}
        if self.ld.spec.has_gt:
            try:
                joint = data.joint_label(self.ld.gt, factors=("belly", "tail", "stripe"))
                out["ari"] = round(float(adjusted_rand_score(joint, labels)), 3)
                out["gt_joint"] = [int(v) for v in joint]
            except Exception:
                pass
        return out
