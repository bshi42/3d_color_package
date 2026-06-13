# fishy_research

Offline research harness for improving the **color-based population cluster analysis**
produced by the 3D Slicer color-DeCA module (`color_deca/deca3/InterDeCA.py`,
`pca_morphospace/PCAMorphospace.py`).

The goal: make it easy for non-technical users (biologists) to **see the clusters /
segments of a population** in a morphospace plot, while keeping the module UI simple.

We iterate on the **synthetic "fishy" dataset**, which has *known* ground-truth cluster
structure (see `FISHY_DATASET_SUMMARY.md` in the repo root), so we can objectively measure
whether an analysis recovers the planted clusters.

## Data
- Textures: `/mnt/data/ml_data/fishy_all/fishy_000000.png … fishy_000249.png` (250 baked 2048² UV textures, all on the **same** mesh).
- Mesh: `/mnt/data/ml_data/fishy_all/fishy1.obj` (shared mesh + UV for every specimen).
- Ground-truth generating parameters: `/mnt/data/ml_data/fishy/samples.csv` (250 rows).

Because all 250 textures share one mesh + UV, the DeCA atlas-transfer step is the identity
here; the analysis reduces to *sample per-face colors → build a feature vector → reduce →
visualize*. That is exactly what this harness reproduces.

## Setup
```bash
cd fishy_research
uv sync          # creates .venv with all deps and installs `fishpipe` (editable)
uv run python experiments/exp01_baseline.py
```

## Layout
- `src/fishpipe/` — importable pipeline package (mesh, feature extraction, metrics, embeddings, plotting).
- `experiments/` — numbered, self-contained experiment scripts. Each writes plots + a scorecard to `results/`.
- `cache/` — cached per-face color tensors (`.npy`), gitignored. Built once, reused by every experiment.
- `results/` — generated plots and scorecards.
- `RESEARCH_LOG.md` — the running scientific log: hypotheses, experiments, results, conclusions, next steps.

Start by reading `RESEARCH_LOG.md`.
