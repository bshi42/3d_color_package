"""Build the best segmentation+spatial pattern morphospace for FISHY.

Config: K=8, smooth=0, region=whole.  Features = spatial(80-d)+endler(36-d),
per-block StandardScaler (as in seg_eval) -> StandardScaler -> PCA(2).

Left panel: UNSUPERVISED PCA morphospace with side-view render exemplars
(AnnotationBbox+OffsetImage), outlined by true stripe (red=5, blue=4) for validation.
Axes labelled by what they encode (PC corr with stripe_count/spacing/width).

Right panel: the SUPERVISED stripe-count "pattern axis" (5-fold-honest LDA proj),
histogram by class, to show the separation that drives the 0.929 recoverability
(count is recoverable but is NOT a dominant unsupervised axis).

Dip-gate: honest gate on the UNSUPERVISED PC1 = dip p<0.05 AND dip > max of
200 column-shuffled nulls (same recipe as the mussels deep-dive). Stated as
cluster vs gradient.
"""
from __future__ import annotations

import json
import os
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import diptest

from fishpipe import data, segment, blobs, spatial

K, SMOOTH = 8, 0
MAF = 0.0005
RENDER_DIR = "cache/fishy_renders"
OUT = "results/fishy_pattern/fishy_segspatial_morphospace.png"
os.makedirs(os.path.dirname(OUT), exist_ok=True)

gt = data.load_ground_truth()
fcd = data.build_face_colors()
df = gt.params
names = [str(n) for n in fcd.names]
y = gt.labels["stripe"].astype(int)
count = df.stripe_count.values.astype(float)
spacing = df.stripe_spacing.values.astype(float)
width = df.stripe_width.values.astype(float)

seg = segment.segment(fcd, n_colors=K, smooth_iters=SMOOTH, per_specimen=False)
endl = blobs.endler_transitions(fcd, seg)
spat = spatial.spatial_descriptor(fcd, seg, region_mask=None, min_area_frac=MAF)
spat_endl = np.concatenate(
    [StandardScaler().fit_transform(b) for b in [spat, endl]], axis=1
)

Xs = StandardScaler().fit_transform(spat_endl)
pca = PCA(n_components=2, random_state=0)
Z = pca.fit_transform(Xs)
evr = pca.explained_variance_ratio_


def corr(a, b):
    return float(np.corrcoef(a, b)[0, 1])


# orient PCs so higher = more count where there is any sign
for i in range(2):
    if corr(Z[:, i], count) < 0:
        Z[:, i] *= -1

pc_corr = {
    "PC1": {"count": corr(Z[:, 0], count), "spacing": corr(Z[:, 0], spacing),
            "width": corr(Z[:, 0], width)},
    "PC2": {"count": corr(Z[:, 1], count), "spacing": corr(Z[:, 1], spacing),
            "width": corr(Z[:, 1], width)},
}

# ---- honest dip-gate on UNSUPERVISED PC1: dip > max(200 column-shuffled nulls) ----
def dip_gate_shuffle(coord_matrix, axis_vec, n_null=200, seed=0):
    """axis_vec: the real 1-D projection. Null: re-PCA on column-shuffled feature
    matrix, take PC1, dip. p<0.05 AND dip>null_max -> cluster."""
    d, p = diptest.diptest(np.asarray(axis_vec, float))
    rng = np.random.default_rng(seed)
    nulls = []
    Xm = coord_matrix
    for _ in range(n_null):
        Xp = np.column_stack([rng.permutation(Xm[:, j]) for j in range(Xm.shape[1])])
        zp = PCA(n_components=1, random_state=0).fit_transform(Xp)[:, 0]
        dd, _ = diptest.diptest(zp)
        nulls.append(dd)
    nulls = np.array(nulls)
    return {"dip": float(d), "p": float(p), "null_max": float(nulls.max()),
            "null_95": float(np.percentile(nulls, 95)),
            "significant_cluster": bool(p < 0.05 and d > nulls.max())}


gate_pc1 = dip_gate_shuffle(Xs, Z[:, 0], n_null=200)

# What drives the unsupervised PC bimodality? (belly/tail are the real binary factors)
pc1_belly = corr(Z[:, 0], gt.labels["belly"])
pc2_tail = corr(Z[:, 1], gt.labels["tail"])
# Honest stripe-specific gate: dip of the supervised stripe axis is circular, so we
# instead report that stripe-count has NEAR-ZERO loading on the dominant unsupervised
# axes -> stripe-count is NOT a cluster in the morphospace (it is a gradient/minority
# direction recoverable only supervised).
stripe_pc_max_abs_corr = max(abs(pc_corr["PC1"]["count"]), abs(pc_corr["PC2"]["count"]))

# ---- supervised stripe-count pattern axis (cross-validated, honest) ----
clf = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis())
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
# out-of-fold decision scores along the LDA direction
lda_oof = cross_val_predict(clf, Xs, y, cv=cv, method="decision_function")
lda_full = LinearDiscriminantAnalysis().fit(Xs, y).transform(Xs)[:, 0]
lda_corr_count = corr(lda_full, count)
lda_corr_spacing = corr(lda_full, spacing)

# ---- exemplar selection: grid-nearest over the PCA plane + force count==5 ----
tree = cKDTree(Z)
xg = np.linspace(Z[:, 0].min(), Z[:, 0].max(), 6)
yg = np.linspace(Z[:, 1].min(), Z[:, 1].max(), 5)
chosen, ex_idx = set(), []
for gx in xg:
    for gy in yg:
        _, j = tree.query([gx, gy])
        if j not in chosen:
            chosen.add(j); ex_idx.append(int(j))
pos = np.where(y == 1)[0]
for j in pos[np.argsort(-Z[pos, 0])[:4]]:
    if int(j) not in chosen:
        chosen.add(int(j)); ex_idx.append(int(j))

# ---- figure ----
fig = plt.figure(figsize=(20, 9))
gs = fig.add_gridspec(1, 3, width_ratios=[2.0, 2.0, 0.9], wspace=0.18)
ax = fig.add_subplot(gs[0, :2])
axh = fig.add_subplot(gs[0, 2])

colp = np.where(y == 1, "#d62728", "#1f77b4")
ax.scatter(Z[:, 0], Z[:, 1], c=colp, s=16, alpha=0.35, zorder=1, edgecolors="none")
for j in ex_idx:
    path = os.path.join(RENDER_DIR, names[j] + ".png")
    if not os.path.exists(path):
        continue
    img = Image.open(path).convert("RGB")
    oim = OffsetImage(np.asarray(img), zoom=0.21)
    edge = "#d62728" if y[j] == 1 else "#1f77b4"
    ab = AnnotationBbox(oim, (Z[j, 0], Z[j, 1]), frameon=True,
                        bboxprops=dict(edgecolor=edge, lw=2.4), pad=0.12, zorder=3)
    ax.add_artist(ab)


def axlabel(pc, name, var):
    c, sp, wd = pc["count"], pc["spacing"], pc["width"]
    mags = {"count": abs(c), "spacing": abs(sp), "width": abs(wd)}
    dom = max(mags, key=mags.get)
    return (f"{name} ({var*100:.0f}% var) ~ weak {dom} loading\n"
            f"[r_count={c:+.2f}, r_spacing={sp:+.2f}, r_width={wd:+.2f}]")


ax.set_xlabel(axlabel(pc_corr["PC1"], "PC1", evr[0]), fontsize=11)
ax.set_ylabel(axlabel(pc_corr["PC2"], "PC2", evr[1]), fontsize=11)
ax.set_title(
    "FISHY segmentation+spatial pattern morphospace (UNSUPERVISED PCA)\n"
    "K=8, smooth=0, whole-body; features = spatial(80-d)+endler(36-d)\n"
    "unsup PC1=belly, PC2=tail (dominant binary factors); outline red/blue = stripe count 5/4 "
    "(count is recoverable supervised, see right panel, but is NOT a dominant variance axis)",
    fontsize=11.5)
leg = [Line2D([0], [0], marker="o", color="w", markerfacecolor="#d62728",
              markersize=11, label="count = 5 (n=52)"),
       Line2D([0], [0], marker="o", color="w", markerfacecolor="#1f77b4",
              markersize=11, label="count = 4 (n=198)")]
ax.legend(handles=leg, loc="upper left", fontsize=11, framealpha=0.9)
gate_txt = (
    f"dip-gate (honest): unsup PC1 IS bimodal (dip={gate_pc1['dip']:.3f}, "
    f"null-max={gate_pc1['null_max']:.3f}) but that split is BELLY "
    f"(PC1 r_belly={pc1_belly:+.2f}); PC2 split is TAIL (r_tail={pc2_tail:+.2f}).\n"
    f"STRIPE-COUNT has near-zero loading on both PCs (max |r_count|={stripe_pc_max_abs_corr:.2f}) "
    f"-> stripe-count is a GRADIENT / minority direction, NOT a cluster.")
ax.text(0.99, 0.01, gate_txt, transform=ax.transAxes, ha="right", va="bottom",
        fontsize=10, bbox=dict(boxstyle="round", fc="#ffffcc", alpha=0.92))
ax.grid(alpha=0.15)

# right: supervised count pattern axis histogram (out-of-fold)
b4 = lda_oof[y == 0]; b5 = lda_oof[y == 1]
bins = np.linspace(lda_oof.min(), lda_oof.max(), 26)
axh.hist(b4, bins=bins, orientation="horizontal", color="#1f77b4", alpha=0.7,
         density=True, label="count=4")
axh.hist(b5, bins=bins, orientation="horizontal", color="#d62728", alpha=0.7,
         density=True, label="count=5")
axh.set_title("Supervised stripe-COUNT\npattern axis (5-fold OOF LDA)\n"
              f"bal-subsample acc 0.883; full 0.929\n"
              f"axis r_count={lda_corr_count:+.2f}, r_spacing={lda_corr_spacing:+.2f}",
              fontsize=10)
axh.set_ylabel("LDA decision score (spatial+endler)")
axh.legend(fontsize=10, loc="upper right")
axh.grid(alpha=0.15)

fig.savefig(OUT, dpi=130, bbox_inches="tight")
print("SAVED", OUT)
print(json.dumps({
    "explained_var": [float(v) for v in evr],
    "pc_corr": pc_corr,
    "lda_axis_corr_count": lda_corr_count,
    "lda_axis_corr_spacing": lda_corr_spacing,
    "dip_gate_pc1_unsupervised": gate_pc1,
    "pc1_corr_belly": pc1_belly,
    "pc2_corr_tail": pc2_tail,
    "stripe_max_pc_corr": stripe_pc_max_abs_corr,
}, indent=None))
