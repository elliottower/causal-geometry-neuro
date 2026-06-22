# Causal Geometric Structure in Neural Populations

This project applies differential geometry and causal inference to Neuropixels recordings from mouse visual decision-making. The core question: when standard analysis tools disagree about neural population structure, can geometry tell you why?

Standard similarity metrics (CKA, Procrustes) give opposite answers about which brain regions are "similar" --- the anti-correlation is strong and replicates across datasets. Dimensionality mediates it: CKA measures kernel alignment (sensitive to variance), Procrustes measures subspace orientation (sensitive to geometry), and they diverge predictably in high-dimensional regions. Linear subspace methods systematically fail in the highest-dimensional brain regions where causal signals are strongest.

## Slides

**[Differential geometry of neural population codes](docs/slides/causal_geometric_structure_slides_diffgeo.pdf)** ([source](docs/slides/causal_geometric_structure_slides_diffgeo.tex)) --- 47 pages covering geometric dissociation, spectral universality, evidence-choice alignment, dimensionality as computational strategy, optogenetic validation, and IBL cross-dataset replication.

**[IIA vacuity investigation](docs/slides/causal_geometric_structure_slides_iia.pdf)** ([source](docs/slides/causal_geometric_structure_slides_iia.tex)) --- IIA interchange interventions, structured VAE subspaces, Sutter et al. vacuity replication, engagement separation, and external validation.

**[Full slide deck](docs/slides/causal_geometric_structure_slides_v5.pdf)** ([source](docs/slides/causal_geometric_structure_slides_v5.tex)) --- 94 pages, all results combined across 4 parts.

To compile:
```bash
cd docs/slides
pdflatex causal_geometric_structure_slides_diffgeo.tex
```

## Paper

Draft paper in `paper/main_v5_3.tex` (working draft, not yet submitted).

## Key findings

1. CKA and Procrustes anti-correlate across brain regions (Steinmetz: rho = -0.90, n = 50 regions; IBL replication: rho = -0.94, n = 11 regions, p < 0.0001)
2. Dimensionality mediates the dissociation: partial correlation controlling for power-law exponent reverses the sign (partial r = +0.44)
3. A structured VAE finds causal subspaces 3.4x stronger than LDA (73/73 regions, p = 5.7e-14)
4. LDA is anti-correlated with optogenetic causal importance (rho = -0.73, p = 0.01)
5. IIA is vacuous for nonlinear methods (Sutter et al. 2025) --- external validation is essential

## Data

| Source | Loader | Description |
|--------|--------|-------------|
| [Steinmetz et al. 2019](https://doi.org/10.1038/s41586-019-1787-x) | `data/steinmetz.py` | Neuropixels recordings (73 regions, 39 sessions, 10 mice) |
| [IBL Brain-Wide Map](https://int-brain-lab.github.io/) | `data/ibl.py` | Cross-dataset replication (139 mice, 12 labs) |
| [Allen Institute](https://connectivity.brain-map.org/) | `data/allen.py` | Mouse Brain Atlas structural connectivity |
| Modal GPU | `results/exp*/` | All experiment results (JSON, JSONL) |

## Experiments

72 experiments in `experiments/`, run on Modal GPUs. Key ones:

| Experiment | What it tests |
|-----------|---------------|
| `exp22` | Geometric type predicts optimal decoder |
| `exp42` | Core CKA-Procrustes anti-correlation across 73 regions |
| `exp46` | IBL cross-dataset replication (11 matched regions) |
| `exp47b` | Optogenetic silencing validation (n=16 regions) |
| `exp51` | Confound control (neuron count, firing rate, temporal shuffle) |
| `exp53` | Dose-response: progressive dimension ablation |
| `exp57` | Structured VAE for nonlinear causal subspaces |
| `exp59` | IIA vacuity test (Sutter et al. 2025) |
| `exp67` | Potent/null space decomposition |
| `exp70` | Cross-region activation patching (1,438 directed pairs) |

## Library

| Module | Description |
|--------|-------------|
| `geometry/distances.py` | Grassmannian distance, CKA, Procrustes, subspace angles |
| `geometry/subspace.py` | Subspace extraction (PCA, LDA, VAE latent) |
| `geometry/holonomy.py` | Parallel transport and holonomy on the Grassmannian |
| `geometry/sheaf.py` | Sheaf cohomology for inter-region consistency |

## Scripts

| Script | What it does |
|--------|-------------|
| `modal_run.py` | Run experiments on Modal GPUs |
| `docs/slides/generate_figures.py` | Generate slide figures from exp42 results |
| `docs/slides/generate_exp46_figures_v2.py` | Generate IBL cross-dataset figures |
| `paper/generate_figures.py` | Generate paper figures |

## Setup

```bash
uv sync

# Configure GCS credentials (for data caching)
cp .env.example .env
# Edit .env with your GCS_BUCKET and GCS_SA_KEY_PATH
```

## Reproducing

```bash
# Run an experiment locally (CPU, small subset)
uv run python -m experiments.exp42_real_iia

# Run on Modal GPU
uv run modal run --detach modal_run.py --experiment exp42
```

Pre-computed results for all experiments are in `results/`.

## License

MIT

## Citation

<!-- TODO: replace with actual DOI after Zenodo upload -->
<!-- [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX) -->

Tower, E. (2026). *Causal Geometric Structure in Neural Populations.* Zenodo.
