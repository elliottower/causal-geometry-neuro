# Batch 0: Steinmetz Validation (5 sessions)

Date: 2026-06-19
Platform: Modal (CPU, 4 cores, 16GB)
Duration: ~15 min total

## What ran

| Experiment | Status | Key result |
|-----------|--------|------------|
| exp7 (sheaf cohomology) | Done | H1 = 75-495 across 5 sessions |
| exp8 (holonomy) | Done | All zeros (PCA limitation) |
| exp9 (multiple realization) | Done | CKA same ≈ CKA cross (0.123 vs 0.139) |

## Interpretation

- exp7: H1 >> 0 in all sessions. BUT — see batch0.5 controls. This may be trivially
  true due to noisy cross-region restriction maps. Different brain regions have
  different neurons (no shared coordinate system like transformers' d_model), so
  cross-correlation matrices are noisy and subspaces can't "glue" regardless of
  whether the mechanism is truly distributed.

- exp8: Holonomy = 0 everywhere. PCA gives stable subspaces across time bins
  (PCA captures variance, not causation). Need DAS (torch optimization on GPU)
  to see time-varying causal structure.

- exp9: CKA can't distinguish same-region from cross-region mechanism comparisons.
  Preliminary signal for CKA-d_G dissociation, but only 1 same-dimension pair
  for Grassmannian distance (different regions have different neuron counts).

## Results location

- GCS: gs://neuro-causal-geometry-data/results/
- Local: artifacts/results/exp7.json, exp8.json, exp9.json
- Modal volume: neuro-causal-geometry-results
