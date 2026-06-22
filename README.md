# Causal Geometric Structure in Neural Populations

Resolving the linear-nonlinear divide in neural analysis tools. Different similarity metrics (CKA, Procrustes) give opposite answers about which brain regions are "similar" --- we show dimensionality explains why, and use causal interventions (interchange intervention analysis) to determine which tool to trust.

## Slides

**[Slides](docs/slides/causal_geometric_structure_slides_v5.pdf)** ([LaTeX source](docs/slides/causal_geometric_structure_slides_v5.tex)) --- 78 slides in 3 parts: core results (geometric dissociation, causal interventions, learned subspaces), stress-testing (validity checks, IIA vacuity, external validation), and deeper analysis (potent/null decomposition, activation patching, choice information flow).

To compile:
```bash
cd docs/slides
pdflatex causal_geometric_structure_slides_v5.tex
```

## Paper

Draft paper in progress (TBD). Current working version in `paper/`.

## Key findings

1. CKA and Procrustes anti-correlate across brain regions ($\rho = -0.85$); dimensionality explains the disagreement
2. A structured VAE finds causal subspaces 3.4x stronger than LDA (73/73 regions, $p = 5.7 \times 10^{-14}$)
3. LDA is anti-correlated with optogenetic causal importance ($\rho = -0.82$, $p = 0.023$)
4. Cross-region activation patching reveals directed choice information flow with sender/receiver hub structure
5. IIA is vacuous for nonlinear methods (Sutter et al. 2025) --- external validation is essential

## Data

| Directory | Description | Source |
|-----------|-------------|--------|
| `data/steinmetz.py` | Loader for Steinmetz et al. 2019 Neuropixels data (73 regions, 39 sessions, 10 mice) | [Steinmetz et al., Nature 2019](https://doi.org/10.1038/s41586-019-1787-x) |
| `data/ibl.py` | Loader for IBL repeated-site data (cross-dataset replication) | [IBL, 2022](https://int-brain-lab.github.io/) |
| `data/allen.py` | Allen Mouse Brain Atlas structural connectivity | [Allen Institute](https://connectivity.brain-map.org/) |
| `results/exp*/` | All experiment results (JSON, JSONL) — downloaded from Modal volume | Computed via Modal GPU |

## Experiments

72 experiments in `experiments/`, run on Modal GPUs. Key ones:

| Experiment | What it tests |
|-----------|---------------|
| `exp42_real_iia.py` | Core IIA interchange interventions across 73 regions |
| `exp47b_silencing_real_data.py` | Optogenetic silencing validation (n=16 regions) |
| `exp51_confound_control.py` | Confound control (neuron count, firing rate, temporal shuffle) |
| `exp53_graded_response.py` | Dose-response: progressive dimension ablation |
| `exp57_structured_vae.py` | Structured VAE for nonlinear causal subspaces |
| `exp59_sutter_dilemma.py` | IIA vacuity test (Sutter et al. 2025 replication) |
| `exp61_engagement_subspace.py` | Engagement vs choice disentanglement |
| `exp67_potent_null_space.py` | Potent/null space decomposition |
| `exp70_cross_region_patching.py` | Cross-region activation patching (1,438 directed pairs) |
| `exp72_sutter_continuous.py` | Continuous metrics for IIA vacuity diagnosis |

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
| `modal_download_opto.py` | Download optogenetic silencing data |
| `docs/slides/generate_figures.py` | Generate slide figures |
| `paper/generate_figures.py` | Generate paper figures |

## Setup

```bash
# Install
uv sync

# Configure GCS credentials (for data caching)
cp .env.example .env
# Edit .env with your GCS_BUCKET and GCS_SA_KEY_PATH
```

The `.env` file must set `GCS_BUCKET` and `GCS_SA_KEY_PATH`. Local data cache goes to `cache/` in the repo root.

## Reproducing

```bash
# Run an experiment locally (CPU, small subset)
uv run python -m experiments.exp42_real_iia

# Run on Modal GPU
uv run modal run modal_run.py --detach
```

Pre-computed results for all experiments are in `results/`. Most experiments were run on Modal A10G GPUs.

## Mechanistic Validity

All findings are self-audited using the [Mechanistic Validity](https://mechanistic-validity.github.io/mechanistic-validity/) framework (Tower 2026). This was designed for mechanistic interpretability research in machine learning, but we are testing it out on biological neural networks. Many of the criteria are not possible or do not apply, but we believe it still serves as a useful frame of reference.

The full audit is in [`experiments/VALIDITY_AUDIT_MECHVAL_V1.md`](experiments/VALIDITY_AUDIT_MECHVAL_V1.md) --- 9 findings scored across 27 criteria and 5 validity types. Current verdicts: 2 Causally Suggestive (IIA interventions), 2 Proposed-strong (CKA anti-correlation, parcellation), 5 Proposed, 1 Disconfirmed (IIA does not predict silencing effects). Bottleneck is internal validity --- only one causal method (IIA swap) tested so far.

## License

MIT
