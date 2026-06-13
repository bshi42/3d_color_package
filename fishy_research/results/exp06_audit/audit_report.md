# EXP-06 — Overfitting / Robustness / Leakage Audit

**Date:** 2026-06-01 · **Scope:** `fishpipe.recommended` pipeline (stripe `pattern_vector`, cheeks `rare_variant_descriptor`, color morphospace + `auto_cluster`). Dataset N=250. Metric: stratified 5-fold balanced CV accuracy, standardized + `LogisticRegression(class_weight='balanced')`. Chance ≈ 0.5.

Headline being audited: **stripe 0.64 → ~0.88** (measured banding), **cheeks → ~0.78**, **color auto-clusters ARI ~0.99**.

---

## Task 1 — Seed / stability

Re-ran with 5 StratifiedKFold seeds (`cross_val_predict` + `balanced_accuracy_score`, balanced LogReg). "Metric own CV" = the package metric (seed=0 internally).

| descriptor | metric own CV | seeds {0,1,2,7,123} mean | min | max | spread |
|---|---|---|---|---|---|
| stripe `pattern_vector` | 0.887 | **0.874** | 0.863 | 0.887 | ±0.012 |
| cheeks `rare_variant_descriptor` | 0.775 | **0.787** | 0.775 | 0.794 | ±0.010 |

**Stable.** Both vary < ±0.02 across seeds; the headline 0.887 / 0.775 are representative, not lucky-seed peaks (cheeks is actually marginally *higher* on other seeds).

## Task 2 — Hyperparameter sensitivity (stripe `pattern_traits`)

Replica of `pattern_traits` (matches package exactly: 0.8868 at q=0.80, n_bins=64). Swept salient-mask `quantile` × `n_bins`:

| quantile \ n_bins | 48 | 64 | 96 |
|---|---|---|---|
| **0.70** | 0.721 | 0.743 | 0.757 |
| **0.80** | 0.901 | **0.887** | 0.907 |
| **0.90** | 0.887 | 0.887 | 0.887 |

Grid mean 0.842, min 0.721, max 0.907.

**Mostly robust, with one real caveat.** For quantile ∈ {0.80, 0.90} the result is rock-stable at 0.887–0.907 across all n_bins — **not cherry-picked**, and n_bins is essentially irrelevant there. **But at quantile=0.70 it collapses to ~0.72–0.76** (back near the unsupervised baseline). Loosening the salient mask to the top 30% of faces lets non-pattern (belly/tail/base) region into the profile and dilutes the banding signal. The default 0.80 sits just above this cliff; 0.90 is safer. Concrete weakness: the salient-mask quantile is a real knob and the chosen 0.80 is near the edge of the good regime.

## Task 3 — Leakage check

Traced every feature code path: `color_morphospace`, `salient_pattern_mask`, `pattern_traits`, `rare_variant_descriptor`/`spectral_descriptor`, `auto_cluster`.

- `grep` for `load_ground_truth | gt.labels | gt.params | samples.csv | stripe_count | rosy_cheeks | belly_hue | tail_hue` across `recommended.py`, `features.py`, `structure.py`, `spectral.py`, `embed.py` → **NO MATCHES**.
- All feature functions take only `fcd` (face colors + shared mesh geometry) or `scores`. Ground truth is imported/used **only** in `experiments/*` (scoring), `plotting.py` (point colors), and `data.joint_label` — never inside a feature builder.
- The metric CV uses a `make_pipeline(StandardScaler, LogReg)` inside `cross_val_predict`, so scaling/fitting is fold-internal (no scaler leakage).

**No label leakage.** One honest *transductive* note (not leakage): `salient_pattern_mask` (cross-specimen lightness **variance**) and `novelty_score` (cross-specimen per-face **median**) are built once on the full population that is then scored. They use **no labels**, and the effect is negligible — the salient mask computed on a random **half** of the specimens overlaps the full-population mask **97.5%** (q=0.80). So the feature definition is effectively population-independent; the headline numbers would not move under a strict train-only feature build.

## Task 4 — Negative control (shuffled stripe labels)

Shuffled `y_stripe` (10 permutations), re-scored the real `pattern_vector`:

| | balacc |
|---|---|
| shuffled (10 perms) mean | **0.504** (range 0.411–0.570) |
| real | **0.887** |

**Collapses to chance.** The 0.887 is genuine signal in the feature ↔ stripe relationship, not an artifact of dimensionality (8-D vector) or the CV procedure.

## Task 5 — Figure review (`results/exp05/`)

- **fig1_color_morphospace.png** — *Honest and strong.* Four well-separated blobs; auto-clusters (k=4, left) align cell-for-cell with true belly×tail quadrants (right), ARI=0.99. Auto-cluster is reproducible: k=4 and ARI=0.989 for all `random_state` 0–5. No complaints.
- **fig2_pattern_view.png** — *Left panel honest, right panel weak/misleading.* Left ("measured banding count" vs strength): 4-stripe fish cluster at count≈4, 5-stripe at count≈5 — the measured axis genuinely separates the groups (with expected overlap + a few count≈6 outliers). **Right panel ("pattern-feature PCA", balacc=0.89) is the weak spot**: the 2D PCA of the pattern vector shows the two classes heavily *intermixed* — it does **not** visually support "0.89". The 0.89 comes from a supervised linear classifier on the full 8-D vector, not from this 2D view. The panel arguably undercuts the headline if read as a separation plot; it actually re-proves "discovery ≠ measurement" (unsupervised PCA of even the pattern features doesn't separate stripe).
- **fig3_before_after.png** — *Honest contrast.* BEFORE (color PCA, balacc 0.64): 4/5-stripe fully intermixed. AFTER (measured banding count, dashed split at 4.5): clear left/right grouping. Correctly uses the *interpretable measured axis* (not the PCA panel from fig2) for the "after", so the before/after contrast is fair and compelling.

---

## VERDICT

**The headline result is robust and leakage-free.**

- **stripe 0.64 → ~0.88:** real (negative control → 0.50), seed-stable (±0.01), and robust to n_bins. Reproduced exactly by an independent replica.
- **cheeks → ~0.78:** seed-stable (0.775–0.794), no leakage.
- **color auto-clusters ARI ~0.99:** deterministic across 6 seeds (k=4, ARI=0.989); figure honestly shows it.
- **No ground-truth leakage** anywhere in the feature paths; population statistics are label-free and ~97.5% stable to the sample (transduction is benign).

**Concrete weaknesses / caveats (none fatal):**
1. **Salient-mask quantile is a real knob near an edge.** Default 0.80 → 0.89, but 0.70 → ~0.74 (signal dilution). The good regime is q ≥ 0.80; 0.90 is the safest default. Worth documenting that 0.80 sits just above a cliff.
2. **fig2 right panel ("pattern-feature PCA") is a weak/borderline-misleading visual** — it pairs a "balacc=0.89" caption with a plot where the classes are visually intermixed. The 0.89 is a supervised-classifier number, not a property of that 2D embedding. Prefer the measured banding-count axis (as fig3 does) for the public-facing figure.
3. **Banding descriptor is taxon-specific** (assumes banding along the auto principal body axis). Already flagged in RESEARCH_LOG C6; not a robustness bug but a generality limit.
4. **cheeks 0.78 rests on 14/250 minority specimens** — small absolute n; CI is inherently wide despite seed-stability. Treat as "promising," not locked.
