# Experiment Ideas

## Core thesis

Different brain regions' mechanisms live on different geometric strata.
The novel contribution is the DISSOCIATION between geometric types,
not applying any single method.

## Active experiments (running or ready)

| Exp | Name | Geometric type | Status |
|-----|------|---------------|--------|
| 1 | Cross-animal Grassmannian hierarchy | Type 2 (subspace) | Running on Modal (IBL) |
| 2 | CKA vs d_G dissociation | Type 0 vs Type 2 | Running on Modal (IBL) |
| 4 | Stimulus transportability | Type 2 (subspace) | Running on Modal (Allen) |
| 5 | Behavioral state modulation | Type 4 (fiber bundle) | Running on Modal (Allen) |
| 6 | Gauge correction | Type 3 (orbit) | Running on Modal (Allen) |
| 9 | Multiple realization (LDA) | Type 0 vs Type 2 | Running on Modal (Steinmetz) |
| 10 | Direction vs subspace | Type 1 vs Type 2 | Running on Modal (Steinmetz) |

## Dissociation experiments (THE NOVEL CONTRIBUTION)

| Exp | Name | What it tests | Data |
|-----|------|--------------|------|
| 11 | Linear vs nonlinear | Grassmannian vs UMAP Procrustes — flat or curved? | Steinmetz |
| 12 | Topology vs geometry | Grassmannian vs persistence diagram Wasserstein | Steinmetz |
| 13 | Static vs dynamic | Grassmannian vs trajectory Procrustes | Steinmetz |
| 14 | Geometric type classifier | ALL metrics per region, cluster by conservation pattern | Steinmetz |

## Perplexity experiments (domains 1-5)

| Exp | Name | What it tests | Data |
|-----|------|--------------|------|
| 15 | Communication subspace sheaf | RRR-based restriction maps → retry H¹ properly | Steinmetz |
| 16 | SAE on spike trains | Superposition test + LDA alignment | Steinmetz |
| 17 | Neural factor bank | SVD factorization, cross-animal basis stability | Steinmetz |
| 18 | Grassmannian parcellation | Cluster regions by subspace similarity | Steinmetz |
| 19 | Latent causal discovery | ICA/LiNGAM → do latent nodes match LDA? | Steinmetz |
| 20 | jPCA rotation frequency | Cross-animal rotation frequency conservation | Steinmetz |

## Dead experiments

| Exp | Name | Why dead |
|-----|------|----------|
| 3 | IBL sheaf cohomology | Same artifact as exp7 |
| 7 | Steinmetz sheaf cohomology | H1 is pure artifact (controls proved it) |
| 8 | Holonomy | Needs causal intervention (optogenetics) |

## Future (needs new datasets or heavy infra)

- MICrONS EM + sheaf with actual synaptic weights as restriction maps
- FlyWire/C. elegans MechVal evaluation of connectome claims
- MechVal tier audit of published neuroscience circuit claims (writing, no code)

## Rejected ideas (not novel standalone)

- Persistent homology alone (Gardner 2022, Rybakken 2019)
- Communication subspaces alone (Semedo 2019, 2022)
- jPCA alone (Churchland 2012)
- UMAP/diffusion maps alone (everyone)
- Procrustes alignment alone (Haxby hyperalignment)
- Representational drift alone (Driscoll 2022)

These are TOOLS inside the dissociation experiments, not standalone results.

## Research agent findings (potential additions)

- sliceTCA — tensor decomposition already tested on IBL data (Pellegrino, Nat Neuro 2024)
- DSA (Dynamical Similarity Analysis) — pip install dsa-metric, Ostrow NeurIPS 2023
- MFTMA (Manifold Capacity) — binary choice separability per region
- Shesha geometric stability — already validated on Steinmetz 42 regions
- Power-law eigenvalue spectrum — Stringer/Steinmetz Nature 2019
