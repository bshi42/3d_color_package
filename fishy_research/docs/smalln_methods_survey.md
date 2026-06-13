# Methods survey — finding pattern structure with UNSUPERVISED methods on SMALL samples

Compiled 2026-06-12 (Fable). Companion to `methods_survey.md` (descriptors) and
`RECOMMENDATIONS.md`. Focus: the real biology datasets are **small** (cichlid-synth n=25,
mussels n=31), which is the regime where the team's "discoverability wall" (RESEARCH_LOG C8)
actually bites. Every method below was scored for: does it genuinely help at n≪p, is it
CPU-only, is it interpretable/publication-grade, and is it NEW vs what the team already tried.

> **The one-paragraph thesis.** The team's wall is not a bug in their pipeline — it is a
> *theorem* about high-dimension-low-sample-size (HDLSS) geometry. The right response is a
> three-part program: **(1)** push what unsupervised discovery *can* reach with a better
> **representation** (shared-codebook textons/VLAD) and **multi-view fusion**; **(2)** make
> every claimed cluster **trustworthy** with HDLSS-aware significance + resampling stability
> so the method finds real structure and *refuses to invent it*; **(3)** for structure that is
> real but provably *not* density-discoverable (subtle/overlapping/rare), provide a
> **minimal-supervision escape** — a handful of expert pairwise judgments that warp the space
> so the team's existing unsupervised gates can fire. (1) and (3) are prototyped and validated
> on fishy in `RESEARCH_LOG.md` EXP-29..33.

---

## 0. Why the wall exists — HDLSS geometry (the diagnostic, state it in the paper)

- **Distance concentration / simplex geometry.** As p grows with n fixed, pairwise distances
  concentrate and points sit near the vertices of a regular simplex (pairwise angles → 90°).
  Cluster structure is detectable only if the signal eigenvalue grows fast enough vs noise.
  *Hall, Marron & Neeman 2005 JRSS-B 67(3):427-444; Ahn, Marron, Müller & Chi 2007 Biometrika.*
- **Spiked-covariance PCA consistency (the team's wall as a theorem).** The leading PCA
  direction is a consistent estimate of the signal axis **iff** the spike eigenvalue
  λ₁ ∼ p^α with α>1 (equivalently n ∼ p^β, β>1−α). When subtle pattern-signal variance < per-
  face noise variance, the top eigenvector rotates ~90° off the true axis ⇒ unsupervised
  reduction **cannot** surface it — *exactly* "supervised-recoverable but not unsupervised-
  discoverable." *Jung & Marron 2009 Annals of Statistics 37(6B):4104-4130.*
- **Use it as an a-priori go/no-go.** Estimate the descriptor's eigen-spectrum; if the
  candidate signal eigenvalue doesn't clear the noise floor by the Jung–Marron margin, predict
  "un-discoverable" and abstain *before* running anything. This turns C8 from an empirical
  observation into a quantitative, reportable prediction.

---

## 1. Trustworthy discovery at n≪p — significance + stability gates (UPGRADE the dip gate)

The team's gate (1-D Hartigan dip + column-shuffled null) is too easy to beat (the shuffle
destroys feature covariance) and is blind to tilted/rare structure. Replace/augment with:

- **SigClust** — "one Gaussian blob vs two clusters?" Test statistic = 2-means cluster index;
  null = single Gaussian whose **eigen-spectrum** is estimated (never the full p×p covariance)
  with small eigenvalues **soft-thresholded** to a background-noise level so noise dims don't
  inflate Type-I error. Multivariate, direction-agnostic. *Liu, Hayes, Nobel & Marron 2008
  JASA 103:1281-1293; Huang et al. 2015 JCGS 24:975-993 (soft-threshold).* **CPU: yes (~30 lines numpy).**
  ⚠️ caveat we found empirically (EXP-32): SigClust rejects "one Gaussian" for *any* non-
  Gaussianity including a continuous gradient, so it is necessary but not sufficient for
  *discrete* clusters — pair it with the stability/PAC gate below.
- **Unbalanced SigClust** — the on-target tool for the unsolved **rare-minority** case
  (cheeks, 5%/n=14): replaces the balanced 2-means index (which refuses to put few points in
  one cluster) with a generalized index sensitive to unbalanced splits. *Keefe & Marron 2023
  arXiv:2308.13079 → JCGS 2025; authors ship Python.*
- **DiProPerm** — exact two-sample permutation test that a *hypothesized* split (a known
  covariate, a measured trait axis) separates groups: fit a DWD/SVM normal direction, project,
  permute labels. Use to *validate* an axis, not the split a clustering just maximized. *Wei,
  Lee, Wichers & Marron 2016 JCGS 25:549-569; Python port idc9/diproperm.*
- **Consensus clustering + PAC + M3C null** — resample specimens, recluster, score the
  **Proportion of Ambiguous Clustering**, compare to a **PCA-covariance-preserving** null →
  calibrated p-value **for each k including k=1 (abstain)**. Fixes the over-calling of plain
  consensus/BIC. *Monti 2003 ML 52:91-118; Şenbabaoğlu 2014 Sci Rep 4:6207; John et al. 2020
  Sci Rep 10:1816 (M3C).* **CPU: yes.**
- **Per-cluster & per-specimen confidence** (what a biologist acts on): bootstrap **Jaccard**
  per cluster (Hennig 2007 clusterboot; ≥0.85 highly stable, 0.6-0.75 questionable, ≤0.5
  dissolved) and **prediction strength** per specimen (Tibshirani & Walther 2005 JCGS 14:511-528,
  validated at exactly n=50/100 incl. a d=1000 80/20 case).
- **Selective inference** (post-clustering) for fully calibrated p-values that survive the
  double-dipping of inventing-and-testing a cluster on the same data — the rigorous anti-
  fabrication layer. *Gao, Bien & Witten 2024 JASA 119:332-345; Chen & Gao 2024 (CADET);
  Hivert et al. 2024 (VALIDICLUST).* R-only; port if needed.

**Pick:** soft-threshold **SigClust** (+ unbalanced variant for the rare class) as the
significance gate, wrapped with **consensus-PAC + M3C null** for the k/abstain decision and
**clusterboot Jaccard + prediction strength** for per-cluster/per-specimen trust.

---

## 2. Better representation — escape n≪p by learning the vocabulary from patches, not specimens

The decisive small-n lever: there are only ~25-250 specimens, but each carries tens of
thousands of faces, so a texture **vocabulary** can be learned from *millions of pooled local
patches* (huge effective n) and each specimen summarized as a denoised histogram/VLAD over
that **shared** codebook.

- **Patch-exemplar textons / BoW** — cluster small local Lab patches pooled across all
  specimens into a shared dictionary; per-specimen = histogram of nearest textons. *Varma &
  Zisserman 2005 IJCV 62:61-81; 2009 TPAMI 31:2032-2047 (raw patches beat filter banks).*
- **VLAD / Fisher vectors** — residual (1st-order) / gradient (1st+2nd-order) pooling over the
  shared codebook: informative from far fewer tokens than a histogram, and **power+L2
  normalization** counters "burstiness" (a large uniform base color swamping a rare motif —
  exactly the fish/shell situation). *Jégou et al. 2010 CVPR; Perronnin et al. 2010 ECCV;
  Sánchez et al. 2013 IJCV 105:222-245.*
- **Spatial pyramid** — encode within coarse anatomical cells (head/mid/tail × dorsal/ventral)
  and concatenate, restoring the layout an orderless histogram throws away — the very axis
  where the wall lives. *Lazebnik, Schmid & Ponce 2006 CVPR.*
- **Mesh-native codebook** — define the patch on the mesh (local color stats over a geodesic
  ring ± a Laplace-Beltrami spectral patch), so it is the *local, vocabulary-pooled* successor
  to the team's *global* GFT vector — UV/seam-proof. *Johnson & Hebert 1999 (spin images);
  Litman et al. 2014 (BoW + spectral descriptors).*

**This generalizes the team's KMeans-segmentation + Endler adjacency**: a shared, soft, multi-
scale codebook instead of a per-image hard color KMeans (Endler adjacency can ride on top:
codebook = nodes, adjacency = edges). **Prototyped (EXP-30/31):** a mesh-native texton BoW
(K=128) **beats** the team's best GFT descriptor on stripe and provably measures *count* not
the spacing confound. **Pick:** VLAD/Fisher over a shared mesh-native codebook + coarse spatial
pyramid; every codeword repaints onto the mesh for interpretation.

---

## 3. Multi-view fusion — combine color + pattern + adjacency without concatenating

At n≪p, concatenating a 40-d GFT block, a high-d Gabor block and a low-d adjacency block lets
scale/dimension mismatch dominate. Fuse **similarity networks** instead:

- **Similarity Network Fusion (SNF)** — one kNN similarity net per descriptor block, cross-
  diffused so edges supported across views are reinforced and view-specific noise erased;
  spectral-cluster the fused net. Built for the n≪p multi-omics regime; the cross-diffusion
  *amplifies structure that is weak-but-consistent across views* — a lever concatenation can't
  pull. *Wang et al. 2014 Nature Methods 11:333-337; Python snfpy.* Honest limit: it cannot
  manufacture a gap absent from **all** views, so gate the fused eigengap with a view-permuted
  null (NEMO's relative-eigengap; *Rappoport & Shamir 2019 Bioinformatics 35:3348-3356*).
- **MOFA+** — interpretable shared latent factors with a per-block **variance decomposition**
  ("Factor 2 is 90% adjacency-driven, separates 4-vs-5 stripe"). *Argelaguet et al. 2018 MSB
  14:e8124; 2020 Genome Biology 21:111.*
- **Cluster ensembles over VIEWS** (not random subspaces) — co-association is scale-free.
  *Strehl & Ghosh 2002 JMLR 3:583-617.*

**Pick:** SNF (snfpy) as the fusion engine + a view-permutation abstain gate + MOFA+ for the
"which block carries the signal" readout.

---

## 4. The minimal-supervision escape — the highest-leverage move for n≪p (prototyped)

When structure is supervised-recoverable but has no density gap (the team's own finding for
stripe: 0.93 supervised, ~chance unsupervised), the answer is *not* a cleverer unsupervised
method — it is to inject a trickle of expert judgment and let the **existing** unsupervised
gates fire in a warped space.

- **Constraint metric learning (ITML / RCA / DCA)** — learn a Mahalanobis warp from must-link /
  cannot-link pairs (regularized toward Euclidean so it doesn't overfit at n=25), then run the
  team's ordinary PCA/dip/HDBSCAN/GMM in the warped space. The learned projection is
  inspectable (which Lab/GFT coords got up-weighted = a publishable "what distinguishes these
  morphs" figure). *Xing et al. 2002 NIPS; Davis et al. 2007 ICML (ITML); Bar-Hillel et al.
  2005 (RCA).* **CPU: trivial.**
- **Active constraint selection** — choose *which* pairs to ask so the budget is tiny:
  Explore-Consolidate / Min-Max / NPU reach a given accuracy in roughly **half** the
  constraints of random. *Basu et al. 2004 SDM; Mallapragada et al. 2008 ICPR; Xiong, Azimi &
  Fern 2014 TKDE.* Crucial for the **rare class**: rank specimens by the team's label-free
  novelty score and annotate the top-K → catch most of the 14 cheeks with a handful of queries.
  ⚠️ **Empirical caveat (EXP-33):** naive uncertainty-sampling *underperformed random* on stripe,
  because the dominant ambiguity in the raw space is the *color* boundary, so the budget chased
  the wrong factor. When multiple factors compete, use **random coverage** as the default, and
  make active selection target-aware (label by the trait of interest) or seed the warp first.
- **Constraint-based clustering SELECTION (COBS)** — most conservative: don't fit anything new;
  use ~20-30 constraints to *select* among the team's existing clusterings. *Van Craenendonck
  & Blockeel 2017 ML journal; COBRAS 2018 (active).*
- **Few-exemplar label propagation** — lightest touch; sklearn `LabelSpreading`. Propagation
  confidence doubles as an "is this ambiguous" map. *Zhou et al. 2004 NIPS.* (Caveat we found:
  on the *raw* descriptor the kNN graph follows color/spacing nuisance, so propagation helps
  little for the subtle factor — warp first, then propagate.)
- **Query-complexity theory + weak oracle ("not-sure")** — turns C8 into a quotable budget:
  ~O(k log n) same-cluster queries recover k clusters under a margin; "not-sure" answers are
  the formal analogue of the gate abstaining on continuous variation. *Ashtiani et al. 2016
  NeurIPS; Kim & Ghosh 2017.*

**Pick:** active pairwise constraints → ITML warp → existing gated discovery; quote the
ARI-vs-#constraints curve as the headline "how few suffice" figure. **Prototyped (EXP-33).**

---

## 5. Visualization & topology — a trustworthy morphospace for n=25-40

- **Mapper** — the one method that sidesteps the density-gap wall: it needs a **filter (lens)**
  along which structure varies, not a density valley. A 4-vs-5 split shows as a **fork/flare**
  off a trunk even with zero density gap, if the lens is pattern-aware (e.g. a SW1PerS stripe-
  periodicity score, or a GFT band ratio). The graph **is** the biologist-facing morphospace.
  *Singh, Mémoli & Carlsson 2007; Lum et al. 2013 Sci Rep 3:1236.* Gate flares with the
  bootstrap confidence of **Statistical Mapper** (*Carrière, Michel & Oudot 2018 JMLR 19:1-39*).
- **SW1PerS** — a genuinely new **per-specimen** periodicity/stripe-count score from the
  roundness of a time-delay embedding of a 1-D body-axis profile; shape-agnostic, jitter-robust,
  *n-independent* (measured within one specimen). *Perea et al. 2015 BMC Bioinformatics 16:257.*
- **PHATE / diffusion maps** — deterministic, denoising small-n layouts that preserve
  continuous progressions PCA flattens; **demote UMAP/t-SNE** at small n (they manufacture
  clusters from noise). *Moon et al. 2019 Nat Biotech 37:1482-1492; Coifman & Lafon 2006;
  Kobak & Linderman 2021; Chari & Pachter 2023.*
- **ToMATo / persistence confidence bands** — readable persistence-diagram criterion for "how
  many real clusters" + loop detection. *Chazal et al. 2013 J.ACM; Fasy et al. 2014 Ann.Stat.*

---

## 6. Match the field (publication-grade, interpretable, small-n inference)

- **Spatial homology — the lever the team hasn't pulled.** patternize-style per-pixel/per-
  vertex PCA on **registered** rasters makes each descriptor dimension an *anatomical location*
  shared across specimens, so a stripe present in some and absent in others becomes a localized
  high-loading cluster (and the n×n Gram PCA is exact at n=25-31). *Van Belleghem et al. 2018
  MEE 9:390-398; Colormesh: Valvo et al. 2021 Ecol Evol 11:12468.* The team has graph-Fourier
  but never anatomically-homologous sampling.
- **RRPP / D-PGLS** — the confirmatory engine the field expects: residual-randomization
  permutation tests with **effect sizes** valid at p≫n, phylogeny-ready ("stripe-count explains
  18% of pattern variance, Z=3.1, p=0.004"). *Collyer & Adams 2018 MEE 9:1772-1779.*
- **Granularity / pattern-energy spectra** — reduce a pattern to ~5-7 *named* traits (dominant
  marking size, contrast, complexity); low-dim ⇒ trustworthy effect sizes at small n. *Stoddard
  & Stevens 2010 Proc R Soc B 277:1387; micaToolbox, Troscianko & Stevens 2015 MEE 6:1320.*
- **Spherical-harmonic texture** — rotation-invariant **3D-native** pattern-scale spectrum on
  the mesh (no lossy UV flatten); stays accurate with little training data. *Gros et al. 2025
  PLoS Comput Biol 21(1):e1012349.*
- **Frozen foundation embeddings (benchmark / upper-bound).** DINOv2-with-registers over
  masked renders (the mesh/UV gives a perfect silhouette mask → kills the #1 small-n confound),
  PCA-whiten → existing gate. Best treated as the **discoverability ceiling** + a confound-
  robust second opinion; interpretability via nearest-exemplars / patch-PCA heatmaps / a sparse
  dictionary. *Oquab et al. 2023 (DINOv2); Darcet et al. 2024 (registers); Stevens et al. 2024
  (BioCLIP).* CPU forward pass over ~250 renders = minutes.

---

## Bottom line (the recommended small-n program)

1. **Predict first** (HDLSS spike diagnostic): will this descriptor's structure be discoverable?
2. **Represent better**: shared-codebook texton/VLAD (+ spatial pyramid), fuse views with SNF.
3. **Gate hard**: SigClust (+ unbalanced) + consensus-PAC/M3C + per-cluster Jaccard → find real
   structure, abstain honestly, never fabricate.
4. **Escape with minimal supervision** what the gate (correctly) rejects: active pairwise
   constraints → metric warp → existing gated discovery; quote the ARI-vs-#constraints curve.
5. **Visualize** with Mapper (pattern lens) / PHATE, not UMAP; **confirm** with RRPP effect
   sizes; keep every axis back-projectable onto the mesh.
