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


# --------------------------------------------------------------------------- session
class Session:
    """One interactive session over one dataset. Holds live morph state."""

    def __init__(self, dataset: str, n_pcs: int = 10, seed: int = 0):
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
