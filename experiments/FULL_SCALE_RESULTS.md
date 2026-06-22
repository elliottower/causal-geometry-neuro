# Full-Scale Results — All Sessions (10 mice, 73 regions, 1316 pairs)

## Scale Comparison

| | Small (5 sessions) | Full (all sessions) |
|---|---|---|
| Regions | 24 | 73 |
| Cross-session pairs | 29 | 1316 |
| Region-sessions | 42 | ~730 |

## Key Results at Scale

### exp11: Linear vs Nonlinear Dissociation
- **CKA vs UMAP Procrustes: Spearman rho = -0.850 (p ≈ 0)**
- 1316 pairs across 73 regions
- Held up perfectly from small batch (was -0.864)
- This is the headline number: linear and nonlinear metrics anti-correlate

### exp12: Topology vs Geometry
- 73 regions, 1316 pairs
- **beta_1 > 0 in only 9/73 regions** (12%)
- Wasserstein H1: 2.05 ± 0.88
- Grassmannian vs topology: rho = 0.25 (p = 0.52, n = 9) — no correlation
- Topology is trivial for choice encoding; geometry and topology dissociate

### exp13: Static vs Dynamic
- 1316 pairs
- Trajectory distance: 0.482 ± 0.160
- Grassmannian only computable for 9 same-dimension pairs (2.80 ± 0.25)
- Static geometry more conserved than dynamics (confirmed at scale)

### exp17: Spectral Universality
- 1316 pairs, 50 regions with summaries
- **SV correlation: 0.936 ± 0.062**
- **Trial mode similarity: 0.089 ± 0.070**
- The spectrum is universal (0.94); the content is not (0.09)
- This is the "same shape, different content" dissociation

### exp18: CKA Parcellation
- 73 total regions, 50 parcellated (up from 9)
- k=3 clusters:
  - Cluster 1 (thalamic/subcortical): MRN, TH, VPL, CP, PO, GPe, ZI, BLA, VPM, SCig, SSp, MOp, LSr, VISa
  - Cluster 2 (cortical/hippocampal): ACA, CA3, DG, MOs, SUB, VISp, CA1, VISl, VISpm, VISam, LGd, OLF, ORB, PL, ILA, LP, MB, POL, SCm, MD, PAG, RSP, RT, ACB, SNr, APN, LS, root, MG, LD
  - Cluster 3 (posterior/collicular): POST, SPF, TT, SCsg, VISrl, SCs
- At scale: thalamic vs cortical vs posterior split is the primary axis

### exp20: Rotational Dynamics
- **72/73 regions show rotation above null** (up from 21/24)
- Mean strength: 0.053 ± 0.060
- Rotation is essentially universal during task engagement
- High variability in strength (CV > 1) — region-specific scaling

## What Changed at Scale

1. **exp11 held perfectly** — rho went from -0.864 to -0.850, still extremely strong
2. **exp12 topology became more informative** — 9/73 regions have beta_1 > 0, suggesting a minority of regions DO have topological structure (was 2/24)
3. **exp17 barely moved** — SV correlation 0.936 vs 0.96, trial similarity 0.089 vs 0.08. Rock solid.
4. **exp18 parcellation much richer** — 50 regions (vs 9) gives meaningful clusters
5. **exp20 rotation went from 21/24 to 72/73** — nearly universal

## Still Running
- exp21 (baselines: sliceTCA, power-law, capacity, DSA) — sliceTCA is slow on CPU
- exp4, exp6 (Allen VBN) — session string parsing fix deployed
