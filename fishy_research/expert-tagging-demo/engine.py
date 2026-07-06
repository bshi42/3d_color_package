"""Engine for the multi-label TAGGING morphospace demo.

The expert creates tags ("green tail", "4 stripes", …) and applies them to specimens (multi-label). After
each batch we retrain a per-tag linear classifier, project its tag-score latent to 2-D with a WARM-STARTED
UMAP (so the map reacts immediately yet evolves smoothly), and cluster the latent into a tree. No pairwise
similarity / anchor machinery — that lived in the old similarity demo.

Validated offline as EXP-46..51 in the fishy_research repo. Reuses that repo's dataset + per-region
color+texture descriptor (set FISHPIPE_ROOT if it lives elsewhere).
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import umap
from scipy.cluster.hierarchy import linkage
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# this demo now lives at <fishy_research>/expert-tagging-demo/, so the repo root is one level up
FISHPIPE_ROOT = Path(os.environ.get("FISHPIPE_ROOT", str(Path(__file__).resolve().parent.parent)))
if str(FISHPIPE_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(FISHPIPE_ROOT / "src"))

from fishpipe import config, data, features, textons  # noqa: E402
from fishpipe.dataset import MUSSELS, build_face_colors as ds_build_face_colors  # noqa: E402

PDIM = 11
COLOR_CH = 5
UMAP_KW = dict(n_neighbors=15, min_dist=0.12)
WARM_EPOCHS = 60


# --------------------------------------------------------------------------- descriptor (from EXP pipeline)
def region_labels_for(fcd, n_regions, cache_dir, seed=0):
    p = Path(cache_dir) / f"region_labels_{n_regions}.npy"
    if p.exists():
        return np.load(p)
    lab = KMeans(n_clusters=n_regions, n_init=4, random_state=seed).fit_predict(fcd.centroids).astype(np.int32)
    np.save(p, lab)
    return lab


def _region_desc(lab, chroma, jet, faces):
    Lf, af, bf = lab[:, faces, 0], lab[:, faces, 1], lab[:, faces, 2]
    am = np.argmax(chroma[:, faces], axis=1)
    color = np.column_stack([Lf.mean(1), af.mean(1), bf.mean(1),
                             af[np.arange(len(am)), am], bf[np.arange(len(am)), am]])
    bp, ms, ls = jet[:, faces, 6], jet[:, faces, 7], jet[:, faces, 8]
    tex = np.column_stack([np.abs(bp).mean(1), bp.std(1), np.percentile(np.abs(bp), 90, 1),
                           ls.mean(1), np.abs(ms).mean(1), (Lf < 20).mean(1)])
    return np.hstack([color, tex])


def build_descriptor(fcd, reg, n_regions, cache_dir):
    lab = features.rgb_to_lab(fcd.colors)
    chroma = np.sqrt(lab[..., 1] ** 2 + lab[..., 2] ** 2)
    jet = textons.local_jet(fcd, scales=(2, 4), cache_dir=cache_dir)
    rids = [r for r in range(n_regions) if (reg == r).sum() >= 20]
    blocks = [_region_desc(lab, chroma, jet, np.where(reg == r)[0]) for r in rids]
    return np.hstack(blocks), rids


@dataclass
class DatasetSpec:
    name: str
    title: str
    n_regions: int
    has_gt: bool
    renderer: str


DATASETS = {
    "fishy": DatasetSpec("fishy", "Fishy (synthetic, n=250)", 128, True, "side"),
    "mussel": DatasetSpec("mussel", "Mussel (real, n=31)", 48, False, "valve"),
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
    gt_factors: dict = field(default_factory=dict)


_CACHE: dict[str, LoadedDataset] = {}


def load_dataset(name):
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
        gt_factors = {}
    else:
        raise KeyError(name)
    reg = region_labels_for(fcd, spec.n_regions, cache_dir)
    X, rids = build_descriptor(fcd, reg, spec.n_regions, cache_dir)
    Xs = StandardScaler().fit_transform(X)
    bounds = (float(mesh.vertices[:, 2].min()), float(mesh.vertices[:, 2].max()),
              float(mesh.vertices[:, 1].min()), float(mesh.vertices[:, 1].max()))
    ld = LoadedDataset(spec, mesh, fcd, list(fcd.names), Path(cache_dir), reg, rids, Xs, bounds, gt_factors)
    _CACHE[name] = ld
    return ld


def _zscore_cols(Z):
    return (Z - Z.mean(0)) / (Z.std(0) + 1e-9)


# --------------------------------------------------------------------------- tagging session
class TagSession:
    def __init__(self, dataset, n_pcs=24, seed=0):
        ld = load_dataset(dataset)
        self.ld = ld
        self.dataset = dataset
        self.seed = seed
        self.N = len(ld.names)
        self.K = int(min(n_pcs, self.N - 1))
        pca = PCA(self.K, random_state=seed).fit(ld.Xs)
        self.Z = _zscore_cols(ld.Xs @ pca.components_.T).astype("float32")
        # cold-start NOVELTY score: mean distance to the k nearest neighbours in Z (higher = more distinctive).
        # Used to seed the first batch so rare/atypical specimens surface immediately (EXP-58/60: ~10-16x faster
        # than random at finding a rare class; works in the compressed PCA space, not the raw descriptor).
        from sklearn.neighbors import NearestNeighbors
        kk = int(min(9, max(2, self.N)))
        nd, _ = NearestNeighbors(n_neighbors=kk).fit(self.Z).kneighbors(self.Z)
        self.novelty = nd[:, 1:].mean(1).astype("float32")
        self.tags: list[str] = []
        self.labels: dict[int, set] = {}
        self.prev_emb = None
        self.latent = self.Z                                   # latent the tree clusters on
        self.coords = self._unsupervised_layout()              # 2-D morphospace before any tags
        self.pred = None                                       # (N, T) per-tag probability
        self.batch: list[int] = []

    # ---- layout / classifier -------------------------------------------------
    def _unsupervised_layout(self):
        e = umap.UMAP(random_state=self.seed, **UMAP_KW).fit_transform(self.Z)
        self.prev_emb = np.asarray(e, "float32")
        return self.prev_emb.copy()

    def _tag_scores(self):
        T = len(self.tags)
        S = np.zeros((self.N, T), "float32")
        labeled = sorted(self.labels)
        for t in range(T):
            y = np.array([1 if t in self.labels[i] else 0 for i in labeled])
            if len(labeled) < 2 or len(np.unique(y)) < 2:
                S[:, t] = float(y.mean()) if len(y) else 0.0
                continue
            # C=0.5 robust at tiny n; class_weight='balanced' helps RARE tags in the early incremental regime
            # (EXP-56: ~+0.09 balanced acc for a 21% factor at ~40 labels, matches SMOTE without the dependency)
            clf = LogisticRegression(C=0.5, class_weight="balanced", max_iter=2000).fit(self.Z[labeled], y)
            S[:, t] = clf.decision_function(self.Z)
        return S

    def recompute(self):
        """Retrain the per-tag classifier + re-project (warm-started UMAP). Reacts from the first batch."""
        if not self.tags or not self.labels:
            self.latent = self.Z
            self.pred = None
            self.coords = self._unsupervised_layout()
            return
        S = self._tag_scores()
        self.latent = S
        init = None
        if self.prev_emb is not None:
            p = self.prev_emb
            init = ((p - p.mean(0)) / (p.std() + 1e-9)).astype("float32")
        kw = dict(UMAP_KW, random_state=self.seed)
        e = (umap.UMAP(init=init, n_epochs=WARM_EPOCHS, **kw) if init is not None
             else umap.UMAP(**kw)).fit_transform(S)
        self.coords = np.asarray(e, "float32")
        self.prev_emb = self.coords.copy()
        self.pred = (1.0 / (1.0 + np.exp(-S))).astype("float32")

    # ---- tag ops -------------------------------------------------------------
    def add_tag(self, name):
        name = str(name).strip()
        if name and name not in self.tags:
            self.tags.append(name)
        return self.tags.index(name) if name in self.tags else -1

    def remove_tag(self, idx):
        if not (0 <= idx < len(self.tags)):
            return
        self.tags.pop(idx)
        new = {}
        for i, ts in self.labels.items():
            nt = {(t if t < idx else t - 1) for t in ts if t != idx}
            if nt:
                new[i] = nt
        self.labels = new
        self.recompute()

    def apply_labels(self, updates):
        """updates = {specimen_idx: [tag_idxs]} — REPLACE each specimen's tag set; then retrain + re-project."""
        for k, v in updates.items():
            i = int(k)
            ts = {int(t) for t in v if 0 <= int(t) < len(self.tags)}
            if ts:
                self.labels[i] = ts
            else:
                self.labels.pop(i, None)
        self.recompute()

    def roll_batch(self, n, only_unlabeled=True, active=True):
        """Pick the next batch. With `active`:
          * once a classifier exists -> MOST INFORMATIVE unlabeled = highest mean per-tag uncertainty
            (prob nearest 0.5) — EXP-52: ~0.96 vs ~0.81 ARI at 80 labels vs random;
          * COLD START (no classifier yet) -> seed by NOVELTY so distinctive/rare specimens surface
            immediately instead of after ~16 random labels (EXP-58/60).
        `active=False` -> plain random."""
        n = int(max(1, n))
        unlabeled = [i for i in range(self.N) if not self.labels.get(i)]
        if active and unlabeled:
            pool = np.array(unlabeled)
            if self.pred is not None:
                u = (1.0 - np.abs(2.0 * self.pred - 1.0)).mean(1)    # mean per-tag uncertainty
                self.batch = [int(x) for x in pool[np.argsort(u[pool])[::-1]][:n]]
            else:
                ranked = pool[np.argsort(self.novelty[pool])[::-1]]  # most-novel first
                band = ranked[:max(n, min(2 * n, len(ranked)))]      # sample within the top band so re-rolls vary
                sel = np.random.default_rng().choice(band, min(n, len(band)), replace=False)
                self.batch = [int(x) for x in sel[np.argsort(self.novelty[sel])[::-1]]]
            return self.batch
        rng = np.random.default_rng()
        pool = unlabeled if (only_unlabeled and unlabeled) else list(range(self.N))
        if not pool:
            pool = list(range(self.N))
        self.batch = sorted(int(x) for x in rng.choice(pool, min(n, len(pool)), replace=False))
        return self.batch

    # ---- serialisation -------------------------------------------------------
    def export_state(self):
        return {"dataset": self.dataset, "tags": list(self.tags),
                "labels": {str(i): sorted(ts) for i, ts in self.labels.items()}}

    def import_state(self, d):
        if d.get("dataset") and d["dataset"] != self.dataset:
            raise ValueError(f"labels are for '{d['dataset']}', current dataset is '{self.dataset}'")
        self.tags = [str(t) for t in d.get("tags", [])]
        self.labels = {int(k): {int(t) for t in v} for k, v in d.get("labels", {}).items() if v}
        self.recompute()

    # ---- views ---------------------------------------------------------------
    def state(self):
        # NOTE: ground-truth factors are deliberately NOT sent — the demo visualises only the classifier's
        # own output, never planted labels (no prior-informed colouring, even for fishy).
        return {"dataset": self.dataset, "title": self.ld.spec.title, "has_gt": self.ld.spec.has_gt,
                "names": list(self.ld.names), "N": self.N, "tags": list(self.tags),
                "labels": {str(i): sorted(ts) for i, ts in self.labels.items()},
                "coords": [[round(float(x), 4), round(float(y), 4)] for x, y in self.coords],
                "pred": None if self.pred is None else [[round(float(v), 3) for v in row] for row in self.pred],
                "batch": list(self.batch)}

    def dendrogram(self, k=28):
        """Ward tree of the CURRENT LATENT (tag-score space if tagged, else the descriptor). Depth-aligned,
        thickness = merge distance — the payload the client SVG renderer expects."""
        from matplotlib.colors import to_hex
        from scipy.cluster.hierarchy import dendrogram as scd
        idx = list(range(self.N)) if self.N <= k + 6 else \
            sorted(np.random.default_rng(self.seed).choice(self.N, k, replace=False).tolist())
        Q = self.latent[idx]
        Q = (Q - Q.mean(0)) / (Q.std(0) + 1e-9)
        Z = linkage(Q, method="ward")
        true_d = Z[:, 2].astype(float).copy()
        m = len(Z); Nl = m + 1
        level = np.zeros(m)
        for kk in range(m):
            a, b = int(Z[kk, 0]), int(Z[kk, 1])
            la = 0.0 if a < Nl else level[a - Nl]
            lb = 0.0 if b < Nl else level[b - Nl]
            level[kk] = 1.0 + max(la, lb)
        Z[:, 2] = np.arange(1.0, m + 1.0)
        kc = int(np.clip(6, 2, max(2, len(idx) - 1)))
        h = Z[:, 2]
        ct = float(0.5 * (h[-kc] + h[-(kc - 1)])) if len(h) >= kc else 0.0
        dd = scd(Z, no_plot=True, color_threshold=ct, above_threshold_color="#b8b8b8")
        dmin = float(true_d.min()); dhi = float(np.percentile(true_d, 95))

        def lvl_of(uh):
            return 0.0 if uh < 0.5 else float(level[int(round(uh)) - 1])
        desc = [None] * len(Z)

        def leaves(cid):
            return [cid] if cid < len(idx) else desc[cid - len(idx)]
        for kk in range(len(Z)):
            desc[kk] = leaves(int(Z[kk, 0])) + leaves(int(Z[kk, 1]))
        pos_of = {lf: p for p, lf in enumerate(dd["leaves"])}
        links = []
        for xs, ys, col in zip(dd["icoord"], dd["dcoord"], dd["color_list"]):
            mi = max(0, min(len(true_d) - 1, int(round(ys[1])) - 1))
            w = float(np.clip((true_d[mi] - dmin) / (dhi - dmin + 1e-9), 0.0, 1.0))
            ly = [lvl_of(ys[0]), level[mi], level[mi], lvl_of(ys[3])]
            ps = [pos_of[lf] for lf in desc[mi]]
            links.append({"x": [round(float(v), 2) for v in xs], "y": [round(float(v), 2) for v in ly],
                          "color": to_hex(col), "w": round(w, 3), "span": [int(min(ps)), int(max(ps))]})
        lc = dd.get("leaves_color_list", ["#444"] * len(idx))
        leaves_out = [{"pos": p, "i": int(idx[lf]), "name": str(self.ld.names[idx[lf]]),
                       "color": to_hex(lc[p]) if p < len(lc) else "#444"}
                      for p, lf in enumerate(dd["leaves"])]
        return {"n": len(idx), "ymax": float(level.max()), "step": 10.0,
                "links": links, "leaves": leaves_out, "dataset": self.dataset}
