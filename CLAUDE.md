# CLAUDE.md

## Project overview

**neuro-causal-geometry** implements the Grassmannian Structural Causal Model (G-SCM) framework
for testing mechanism identity across neural populations. The central question: when do two
neural populations instantiate the *same* mechanism vs merely producing similar behavior?

Three computable invariants:
1. **Gauge-normalized Grassmannian distance** — subspace angle after removing gauge symmetries
2. **Holonomy fingerprints** — parallel transport around trial loops reveals circuit curvature
3. **Sheaf cohomology (H⁰, H¹)** — topological localizability of distributed circuits

Nine experiments across three datasets:
- **IBL Brain-Wide Map** (Experiments 1-3): cross-animal/lab choice subspace stability
- **Allen Visual Behavior Neuropixels** (Experiments 4-6): stimulus subspace transportability
- **Steinmetz et al. 2019** (Experiments 7-9): multi-region sheaf cohomology + holonomy

Paper: `docs/neuro_causal_geometry_paper.md`

## Commands

```bash
# Setup
uv sync

# Tests
uv run python -m pytest tests/ -xvs

# Run an experiment
uv run python -m experiments.exp1_cross_animal_stability --config configs/exp1.yaml

# Modal (GPU experiments)
modal run experiments/modal_exp1.py --detach
```

## Architecture

```
data/          — Dataset loaders (IBL, Allen, Steinmetz). Pure Python, no Modal.
geometry/      — Grassmannian math: distances, transport, holonomy, sheaf cohomology.
experiments/   — One file per experiment. Each has a modal_ wrapper for GPU runs.
notebooks/     — EDA and visualization.
tests/         — Correctness invariants, not shape checks.
artifacts/     — Results (gitignored except summaries).
configs/       — YAML configs per experiment.
```

## Guidelines

- `uv run` for all local execution (not python3, not pixi)
- Modal with `--detach` for GPU experiments. NEVER run Modal without --detach
- Pure Python modules + thin Modal wrappers. Never mix Modal into core logic
- Write results to artifacts/ as JSON/npy, never rely on stdout
- All scripts must have tqdm progress bars and timestamp logs
- Tests focus on mathematical invariants (e.g., geodesic distance is a metric,
  parallel transport preserves inner products, boundary operator squares to zero)
