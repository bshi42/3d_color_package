# The "fishy" Dataset — Slack Record & Summary

**Purpose:** This document collects the verbatim Slack conversations about the **synthetic "fishy" dataset** and summarizes what was done, so the work can be handed off to new collaborators wrapping up the 3D color-segmentation project / 3D Slicer module.

- **Channel:** `#color-modeling-pub` (private, Slack workspace `humanaugmente-e7j6563`)
- **Primary author of the fishy work:** Alek Spiridonov (`@Alek`, Slack ID `U08R6D26YA2`, `aspiridonov3@gatech.edu`)
- **Other participants quoted:** Bree Shi (`@Bree Shi`, project lead / thesis author), Arthur Porto (`@Arthur Porto`, advisor/PI), Alex Ro (`@Alex Ro`, new collaborator)
- **Trigger for this document:** Bree's 2026‑05‑26 request below — *"can you send or forward conversations on your fishy stuff for the new ppl"*
- **Compiled:** 2026-06-01

> **Note on scope.** The verbatim section below contains every message in `#color-modeling-pub` that concerns the synthetic fishy work, in chronological order (2025‑11‑07 → 2026‑02‑10), followed by the 2026‑05 hand-off messages. Earlier September 2025 messages in the same channel about DeCA / UV transfer / texture baking are *background* (the fishy pipeline reuses that machinery) and are not reproduced here; they are referenced in the summary.

---

## 1. Verbatim Slack Messages

> Timestamps are as shown in Slack (US Eastern, EST/EDT). `•` marks original bullet points. Attachments and threads are noted in italics. User mentions are rendered as `@Name`.

### 2025-11-07 — weekly update (first mention of running the full fish set)
**Alek Spiridonov, 14:21:14 EST**
> stuck in a meeting at work, so can't attend today.
> This week I haven't really been able to work on this project due to some stuff at work. The only thing I really got done this week was running some results for Bree over the weekend on the full fish dataset.

---

### 2025-11-09 — the "fishy" concept (foundational message)
**Alek Spiridonov, 19:00:14 EST** — *2 image attachments; reaction: ✅*
> I wanted a dataset with more controlled set of textures with known distributions. So I created a 3D model `fishy` and am working on automating generating the coloring. It will have a few visual features:
> • base color
> • black stripes
> • belly color
> • tail color
> • "rosy cheeks"
> Various parameters have simple normal distributions (for example, base color shift, stripe width, stripe spacing, stripe offset, etc), while others (like stripe count, belly color hue shift, rosy cheecks, tail color shift) will have clusters. For example, 80% of fish will have 4 stripes and 20% 5 stripes. 95% will not have rosy cheeks, 5% will have it (to create a minority class with a unique palette). There may be some other post-processing steps added to add some noise to the dataset (maybe white color balance, hue shifts, etc.) The idea is that we would have the raw parameters used to generate the textures, which can be PCA'd on their own and then compared to 3d color based population analysis/PCA.

---

### 2025-11-10 — first analysis results
**Alek Spiridonov, 00:33:51 EST**
> color PCA didn't provide a very good result given the ground truth

**Alek Spiridonov, 00:38:18 EST**
> For reference, the input data has clustering on 4 dimensions:
> • there are two belly colors
> • there are two tail colors
> • there are either 4 or 5 stripes
> • there is a 5% probability of the fish having rosy cheeks
> all of this shows up very well in a 4 component PCA when looking at the raw parameters that are used to generate the textures
> ```
>         base_color_hue,
>         base_color_sat,
>         base_color_val,
>         stripe_count,
>         stripe_spacing,
>         stripe_width,
>         stripe_longitudinal_offset,
>         belly_hue,
>         belly_strength,
>         belly_translation,
>         tail_hue,
>         tail_strength,
>         rosy_cheeks_present,
>         rosy_cheeks_hue,
>         rosy_cheeks_strength,
>         rosy_cheeks_translation_y,
>         rosy_cheeks_translation_z,
> ```
> PC1 and PC2 are probably the belly and tail hues (with 4 easily observed clusters on PC1, 4 clusters on PC2)
> PC3 and PC4 are probably presence of cheeks and stripe count (with each having 2 easily observed clusters along each axis)
> (though I haven't colored the plots to check this)

**Alek Spiridonov, 00:49:20 EST**
> PCA on the color data only really detected 2 clusters (belly hues) on PC1, and along all other PCs (2-5) all the data is just a single blob.
> Using the morphospace slider, PC2 turned out to be the belly patch translation (the patch can have a forward and rearward bias)
> PC3-5 seem to be driven by different aspects of the striping

**Alek Spiridonov, 00:51:33 EST**
> I may add an option for color PCA to run it without color quantization/clustering to see if that improves things

**Alek Spiridonov, 00:54:17 EST** — *(thread parent)*
> I don't expect the stripe count to be an observable cluster in color PCA (because of the noise in stripe width/spacing/offset), but I would expect to see clusters for belly color, tail color, and the rosy cheeks

> &nbsp;&nbsp;↳ **Arthur Porto, 2025-11-13 17:43:55 EST** *(thread reply)*
> &nbsp;&nbsp;> Did you try non linear approaches? I wonder if tsne and umap would recover it

> &nbsp;&nbsp;↳ **Alek Spiridonov, 2025-11-13 18:29:01 EST** *(thread reply)*
> &nbsp;&nbsp;> 1. We do have umap in the module, but I haven't run it yet (also, we don't have the morphospace implemented for umap yet so visualizing the results is not as good).
> &nbsp;&nbsp;> 2. I also implemented ICA for morphospace. It had better qualitative results when it comes to alignment of an IC axis and some aspect of stripes. (With PCA morphospace visualization often had the stripes appear as a blurry blob, where as ICA morphospace visualization recreated the stripes moire faithfully - which kind of aligns with what I remember from the ML course regarding ICA being a better at "feature discovery")
> &nbsp;&nbsp;> 3. While I was analyzing the morphospace with PCA I realized my two "main clusters" had the tail and belly hues correlated. Which made me take another look at data synthesis script and it turned out I had a bug there that made them correlated 1:1. This explains why I got 2 clusters and not 4 (at least for belly and tail hues)
> &nbsp;&nbsp;> 4. The biggest issue with recovering a minority class feature (in this case, "rosy cheeks") is statistics. The cheeks are present on a small fraction of the total area (so they make up few of the dimensions of each vector) and they are present in few specimens (5%). I have one idea about how to normalize for area to some extent, but I don't have anything yet for the issue of low probability

---

### 2026-01-23 — Bree assigns "fishy validation" as a weekly task
**Bree Shi, 14:13:19 EST**
> this week tasks:

**Bree Shi, 14:14:08 EST**
> @Alek fishy validation

---

### 2026-01-30 — adapting the fishy generator to the real fish (cichlid) dataset
**Alek Spiridonov, 12:11:09 EST**
> can't attend the meeting today...
> my update is that I've been playing around with ways to adapt my fishy texture generation approach to the fish dataset, the approach that I've kind of settled on is to
> • use one specimen as the reference
> • using the specimen's landmarks, rotate and scale reference specimen into a reasonable orientation with unit scale for painting convenience
> • define the painting operations in the reference landmark space
> • when painting different models, map the paint operations from the reference landmark space to specimen landmark space (use nearest landmarks to triangulate the paint operation's coordinates)
> I'm still working out the mapping part of this approach

**Alek Spiridonov, 12:15:43 EST**
> one fish is used to form a reference coordinate system (based on its landmarks)
> all fish geometries are used for painting on. Their coordinate systems are effectively deformed to match the reference coordinate system. This way the painted features have the same position relative to the landmarks (more or less)

**Alek Spiridonov, 12:23:16 EST** — *(image attachment)*

**Alek Spiridonov, 12:26:35 EST**
> even though LM0,LM1,LM2,LM3 are in a different location in a particular specimen vs the reference specimen, the F0 point is more or less in the same location relative to LM0, LM1,LM3 and F1 is more or less in the same location relative to LM1,LM2,LM3

**Alek Spiridonov, 12:28:33 EST**
> as a result, I would expect these paint features top show up in the same location on the atlas model (even though they were painted on different models)

**Alek Spiridonov, 12:30:54 EST**
> I'll check with chatgpt/similar if there is a formal name for what I'm doing

---

### 2026-02-06 — synthetic textures on the real cichlids (n=25) + run through the pipeline
**Alek Spiridonov, 13:10:32 EST** — *(thread parent; 4 image attachments: `image.png`, `broken-fish.png`, `image.png`, `image.png`)*
> I've been working on getting synthetic textures on the cichlid and I think I got it to a good point now (see screenshot).
> I've generated a small (n=25) synthetic dataset and ran it through our code. Some observations:
> • the atlas model seems to have been strongly influenced by repeated model meshes (see the "bend/jog" in the screenshot)... I think one of the cichlids with the fin that stretches out to the side may have caused that (screenshot), so maybe I can remove it from the dataset and rerun it
> • despite the weird bend/jog in the atlas model, the textures seem to have transferred mostly OK, especially away from the anomaly (screenshot)
> • it would be nice to decimate these meshes to make them smaller to have the analysis run faster
> I did a little bit of writing regarding the generated textures... I still need to describe the coloring algorithm used across different specimens... though it's not that complicated since it's just translation, rotation, scaling, and TPS warping based on landmarks... it's very similar to what deca does, so I would expect deca to be able to reverse the transforms onto an atlas model without too many issues (and I think the results agree)

> &nbsp;&nbsp;↳ **Arthur Porto, 13:12:11 EST** *(thread reply)*
> &nbsp;&nbsp;> @Alek that bending is likely a landmarking issue..e.g. swapping right and left landmarks when landmarking

> &nbsp;&nbsp;↳ **Arthur Porto, 13:12:27 EST** *(thread reply)*
> &nbsp;&nbsp;> if you do a simple GPA, it will tell you the problematic sample

> &nbsp;&nbsp;↳ **Alek Spiridonov, 14:28:02 EST** *(thread reply; image attachment)*
> &nbsp;&nbsp;> what's GPA in this context?
> &nbsp;&nbsp;> I removed the model that I thought was causing the issue (the one with the fin stuck out to the side) and now it produces a duck billed fish 🙂
> &nbsp;&nbsp;> not sure why we're having any issues at all, the landmark files are just copied from their matching originals, which worked ok when the models+landmarks were not repeated

> &nbsp;&nbsp;↳ **Alek Spiridonov, 14:29:47 EST** *(thread reply)*
> &nbsp;&nbsp;> I might try to do a run on the original files while duplicating some of them with new names and seeing if that poisons deca

> &nbsp;&nbsp;↳ **Arthur Porto, 2026-02-10 22:58:55 EST** *(thread reply)*
> &nbsp;&nbsp;> Sorry, just catching up. GPA is generalized Procrustes analysis

> &nbsp;&nbsp;↳ **Arthur Porto, 2026-02-10 23:00:14 EST** *(thread reply)*
> &nbsp;&nbsp;> depending on how you copied the landmarks, you might not have taken into account the coordinate system..the easiest thing is to load them in slicer and see if they still are where they should be..or send me and I can check

---

### 2026-05-26 / 2026-05-27 — hand-off to new collaborators
**Bree Shi, 2026-05-26 16:38:57 EDT** — *reaction: 🐟*
> @Alek can you send or forward conversations on your fishy stuff for the new ppl we are going to loop them in on wrapping this up

**Alex Ro, 2026-05-27 10:56:15 EDT**
> Hi everyone, I watched the recording. It sounds like the fishy dataset is the main area where I may be able to contribute right away, especially since it is being considered for inclusion in the paper but is not yet in the draft.
>
> I assume the above zip files are the datasets. I can start by reviewing and exploring what exists. then take on any larger code or data-processing fixes that come up from collaborator feedback. Let me know if there's any specifics I should look into.
>
> CC: @Charlie Clark, @Bree Shi

---

## 2. Summary

### What the "fishy" dataset is and why it exists
The fishy dataset is a **synthetic, fully-parameterized texture dataset** Alek built to serve as a **ground-truth validation harness** for the project's 3D color-analysis pipeline (the DeCA-based texture transfer → atlas → color population analysis / morphospace used on the real mussel and cichlid datasets).

The motivation (2025‑11‑09): the real biological data has *unknown* underlying structure, so it's hard to tell whether the color PCA / morphospace / clustering is recovering real signal. A synthetic fish ("fishy") whose coloration is generated from **known parameter distributions** lets you check: *does the color-based population analysis recover the clusters we know are there?*

**The fish's coloration** has five features — **base color** (yellow), **black stripes**, **belly color** (blue), **tail color** (green), and a minority **"rosy cheeks"** (red). Some parameters are continuous (normal noise on hue/sat/value, stripe width/spacing/offset, patch translations/strengths); others are deliberately **clustered/categorical** to plant a known cluster structure:
- **belly hue** — 2 clusters (1‑D `make_blobs`)
- **tail hue** — 2 clusters
- **stripe count** — 4 (80%) or 5 (20%)
- **rosy cheeks present** — 5% minority class (a deliberate rare/unique palette)

The generator records the raw generating parameters per sample (`samples.csv`), so the **ground-truth parameter PCA** can be compared against the **pipeline's color-based PCA/ICA/UMAP**.

### Timeline of what Alek did
1. **Synthetic single-model generator (Nov 2025).** Built the `fishy` 3D model and an automated, Blender-based texture generator. Coloration is produced by **volumetric "brushes"** — spheres and capsules with a hard core + Gaussian falloff — evaluated against mesh vertices, written as a vertex-color attribute, then **baked to a UV texture (Cycles, EMIT pass)**. Each sample's parameters are written to a CSV.
2. **First validation results (Nov 10).** A **4‑component PCA on the raw generating parameters cleanly recovered all 4 planted clusters**. But **PCA on the actual color data did poorly** — it only found ~2 clusters (belly hue) on PC1; other PCs were a single blob, driven by belly-patch translation and striping.
3. **Method improvements + a bug (Nov 13).** Added **ICA** as a morphospace dimensionality-reduction option (qualitatively better at isolating stripe structure than PCA, which smeared stripes into a "blurry blob"); **UMAP** existed in the module but lacked morphospace visualization. Crucially, found a **bug in the synthesis script that correlated belly hue and tail hue 1:1**, which explained why only 2 (not 4) hue clusters appeared. Flagged that recovering the **minority "rosy cheeks" class is statistically hard** (tiny surface area → few vector dimensions; only 5% of specimens).
4. **"fishy validation" formally tasked (Jan 23, 2026)** by Bree as a weekly deliverable.
5. **Generalizing from the toy fish to the real cichlids (Jan 30).** Settled on a **landmark-based painting approach**: pick one specimen as a **reference**, use its landmarks to build a canonical (rotated/scaled/unit) coordinate frame, **define paint operations in reference space**, then map them onto each other specimen by deforming coordinate systems via the landmarks (so painted features land in the same anatomical position on every fish).
6. **Synthetic textures on real cichlid meshes, run through the full pipeline (Feb 6).** Generated a small **n=25** synthetic cichlid dataset and ran it through the DeCA code. The coloring algorithm is **translation, rotation, scaling, and Thin-Plate-Spline (TPS) warping based on landmarks** — deliberately "very similar to what DeCA does," so DeCA should be able to invert the transforms onto an atlas. Findings: textures transferred mostly OK, but the **atlas model showed a "bend/jog"** attributed to repeated meshes / a specimen with a fin sticking out to the side. Porto diagnosed the bending as a **landmarking / coordinate-system issue** (e.g. left/right swap, or landmark coordinate frame not preserved when copying landmark files) and recommended **GPA (generalized Procrustes analysis)** to find the offending sample. Open follow-ups: remove/fix the anomalous specimen, decimate meshes for speed, and finish the methods write-up of the coloring algorithm.
7. **Hand-off (May 2026).** The fishy dataset is **being considered for the paper but is not yet in the draft**. Bree asked Alek to forward these conversations to loop in new contributors for the final stretch; **Alex Ro volunteered to pick up the fishy dataset**.

### Where things stand / open items for the new owner
- **Decimate** the synthetic cichlid meshes so the analysis runs faster.
- **Resolve the atlas "bend/jog"** — run GPA to identify the bad sample; verify copied landmark files preserve the Slicer coordinate system (load them in Slicer and confirm placement); consider dropping the fin-out specimen.
- **Finish the methods write-up** of the cross-specimen coloring algorithm (the translation/rotation/scaling/TPS-warp procedure).
- **Minority-class recovery** ("rosy cheeks") remains unsolved — area-normalization was an idea; low prevalence (5%) is the harder problem.
- Consider running **color PCA without color quantization/clustering**, and exercising **UMAP/ICA** (UMAP still needs morphospace visualization).
- Decide whether/how the synthetic dataset enters the manuscript.

---

## 3. Relevant code & files in this repo

> The fishy scripts live at the repo root and are currently **untracked** (working files); the Slicer-module pipeline they feed lives under `color_deca/deca3/`.

| File | What it is |
|---|---|
| **`color_fishy_tps.py`** *(mtime 2026‑02‑06, the current/main script)* | **Synthetic cichlid texture generator** implementing the Jan 30 / Feb 6 approach. Loads 3D Slicer markup landmark JSON; computes a **canonical transform** per specimen (`spring_pull_rotation` + scale-to-unit + translate); fits a **regularized 3‑D Thin‑Plate Spline** (`class TPS3D`) mapping reference-canonical → specimen-canonical; generates the brush coloration in reference space; **warps brushes through the TPS into each specimen's world space** (`warp_brushes`), scaling brush radii by the local TPS Jacobian; applies brushes to mesh vertices and **bakes to a 2048px texture** via Blender Cycles. Writes per-sample PNG, copies of the mesh + landmarks, and a `samples.csv` of ground-truth parameters. `N_SAMPLES=25`; output dir `…/cichlid-synth`. This is the file referenced by the recent commit *"added synthetic texture generation for cichlid meshes."* |
| **`color_fishy.py`** *(mtime 2025‑11‑21)* | The **original toy-`fishy` generator** (the Nov 9 concept). Colors the single `fishy` model, samples `N_SAMPLES=250` parameter sets from the documented distributions, bakes each variation to a texture, writes `samples.csv`. Its module docstring is the canonical description of the 5 features and which parameters are "cluster features." Shares the brush/bake helpers with the TPS version. |
| **`color_fishy_old.py`** *(mtime 2025‑11‑09)* | Earlier prototype: hardcoded `BRUSHES` + a dorsal/ventral **countershading** base gradient; single-texture output. Precursor to `color_fishy.py`. |
| **`fishy_pca.py`** *(mtime 2025‑11‑21)* | **Ground-truth parameter analysis.** Reads `samples.csv` and runs a 4‑component **PCA** (imports `FastICA` too), plots PC1 vs PC2/PC3/PC4 and belly/tail-hue histograms. This is the "raw parameters PCA" that recovered the 4 clusters (Nov 10). |
| **`preview_fishy_dist.py`** *(mtime 2025‑11‑14)* | Sanity-check of the **sampling distributions** (belly/tail hue histograms) without baking. |
| **`fishy1.blend` / `fishy1.blend1`** | The Blender model of the toy `fishy` itself (`.blend1` is a backup). |
| **`color_deca/deca3/InterDeCA.py`** | The Slicer **module the synthetic dataset is run through** ("InterDeCA"/DeCA3). Hosts the **dimensionality-reduction choice (PCA / UMAP / ICA)** for Step 3 population analysis and Step 4 morphospace (radio buttons at lines ~1053‑1064; `PCA, FastICA` + `umap` imports at lines ~43‑46). |
| **`color_deca/deca3/pca_morphospace/PCAMorphospace.py`** + `PCAMorphospaceVisualization.py` | The **morphospace** logic/UI: `performPCA(n_components, color_space, area_weighted=…)` and the **morphospace slider visualization** used to interpret what each PC corresponds to (referenced in the Nov 10 messages — "Using the morphospace slider, PC2 turned out to be the belly patch translation"). |
| **`color_deca/deca3/clustering_view/ClusteringView.py`** | Color **quantization / clustering** view (the "color quantization/clustering" Alek considered disabling to improve color PCA). |
| **`ICA_IMPLEMENTATION.md`** *(mtime 2025‑11‑14)* | Design note documenting the **addition of ICA** alongside PCA/UMAP for Step 3 / Step 4 (the Nov 13 "I also implemented ICA for morphospace"). |
| Supporting docs | `COLOR_QUANTIZATION_GUIDE.md`, `MULTIRECOLOR_GUIDE.md`, `HUE_COLORING_FEATURE.md`, `HSV_CUTOFFS_FEATURE.md`, `NEIGHBOR_AVERAGE_*`, `LUMINOSITY_NORMALIZATION_*`, `GRAPH_BASED_SUBSAMPLING.md`, etc. — feature/algorithm notes for the color-analysis module the fishy data validates. |

### Conceptual data flow
```
color_fishy.py / color_fishy_tps.py
   (sample params  ──► brushes ──► vertex colors ──► bake to UV texture)
        │                                   ▲
        ├── samples.csv (ground-truth) ──► fishy_pca.py  (parameter-space PCA/ICA: the "truth")
        │
        └── synthetic textures on meshes ──► color_deca/deca3 (DeCA atlas + texture transfer)
                                                  └─► PCA / ICA / UMAP morphospace + clustering
                                                         (compare recovered structure vs. ground truth)
```
