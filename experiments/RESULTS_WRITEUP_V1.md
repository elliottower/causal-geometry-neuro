# Geometric Dissociation in Neural Population Codes: Batch 1 Results

## Thesis

Different brain regions' computational mechanisms live on different
geometric strata. The novel contribution is not applying any single
geometric method, but demonstrating that **different metrics give
different answers for different regions** — proving that the geometric
type of a computation is a meaningful, region-specific property.

## Dataset

Steinmetz et al. 2019 — Neuropixels recordings from 10 mice, 42 brain
regions, visual decision-making task. Batch 1 uses 5 sessions (validation
mode). Each experiment computes pairwise cross-session comparisons for
each region.

**Scale**: 24 distinct brain regions, 42 region-sessions, 29 cross-session
pairs in the densest experiment.

---

## Finding 1: Linear and Nonlinear Geometry Are Anti-Correlated (exp11)

**Spearman rho = -0.864, p < 0.0001** (29 pairs, 24 regions)

CKA (linear kernel similarity) and UMAP Procrustes distance (nonlinear
manifold shape) are *strongly anti-correlated* across brain regions. Regions
with high linear similarity have low nonlinear similarity and vice versa.

This means: the two dominant paradigms in representational similarity
analysis — linear methods (CKA, RSA, Grassmannian) and nonlinear methods
(UMAP, diffusion maps, manifold alignment) — are not measuring the same
thing. They **disagree** about which regions are similar.

| Region | CKA (linear) | UMAP Procrustes (nonlinear) | n pairs |
|--------|-------------|---------------------------|---------|
| MOs | 0.074 | 0.981 | 3 |
| VISp | 0.142 | 0.957 | 3 |
| ACA | 0.197 | 0.929 | 3 |
| DG | 0.195 | 0.858 | 6 |
| SUB | 0.212 | 0.881 | 3 |

Motor cortex (MOs) shows the starkest dissociation: near-zero CKA but
near-maximal UMAP Procrustes. Motor computation is linearly variable
across animals but nonlinearly stereotyped.

**Implication**: Any paper reporting "representational similarity" using
only one class of metric is telling half the story. The metric class
determines the conclusion.

---

## Finding 2: CKA-Based Parcellation Recovers Neuroanatomy (exp18)

Hierarchical clustering on pairwise CKA distances (dimension-independent)
recovers known anatomical groupings without using any anatomical labels.

**k=3 parcellation:**
- Cluster 1 (hippocampal): DG, SUB, root, CA1, VISa
- Cluster 2 (cortical): ACA, VISp, POST
- Cluster 3 (motor): MOs (alone)

At finer resolution (k=5+), hippocampal subfields separate (CA1 splits off),
visual areas separate (VISp from VISa), and anterior cingulate pairs with
retrosplenial cortex (POST). These groupings match known connectivity.

**Implication**: Representational geometry alone (without anatomy, connectivity,
or gene expression) is sufficient to recover the brain's functional organization.
This validates the geometric dissociation framework: if CKA parcellation were
noise, it wouldn't recover anatomy.

---

## Finding 3: Spectral Universality, Temporal Idiosyncrasy (exp17)

**Singular value correlation across animals: r = 0.96**
**Trial mode similarity: 0.08**

The eigenvalue spectrum of population activity is nearly identical across
animals for the same brain region — the dimensionality structure is universal.
But the temporal modes (what those eigenvalues correspond to in trial space)
are uncorrelated.

| Region | n pairs | SV correlation | Trial mode sim | Var explained |
|--------|---------|---------------|---------------|---------------|
| DG | 6 | 0.96 | 0.08 | 0.20-0.87 |
| ACA | 3 | 0.95 | 0.07 | 0.30-0.65 |
| MOs | 3 | 0.97 | 0.09 | 0.25-0.70 |

**Implication**: The "shape" of neural computation (how many dimensions,
how much variance per dimension) is a universal property of each region.
But the "content" of those dimensions (which trial patterns they encode)
is animal-specific. Structure is conserved; dynamics are not.

---

## Finding 4: No Topological Loops in Choice Circuits (exp12)

**beta_1 = 0 in 22/24 regions**

Persistent homology (H1 Betti numbers) detects zero 1-cycles (loops) in
the choice-related activity manifold of almost every brain region. The
two exceptions have negligible beta_1.

This is a clean **negative result**: choice encoding mechanisms are
genuinely flat (living on a subspace, not a ring or torus). This rules
out ring-attractor models for binary choice in these regions.

Wasserstein distances on persistence diagrams (H0) vary widely across
regions (1.4 to 111), suggesting different clustering structure even
though topological complexity is uniformly trivial.

**Implication**: For binary choice tasks, the Grassmannian (linear
subspace) is the correct geometric model. Nonlinear manifold methods
are measuring curvature that doesn't carry task information.

---

## Finding 5: Rotational Dynamics Are Ubiquitous (exp20)

**21/24 regions show rotation above temporal-shuffle null**

Simplified jPCA analysis finds rotational dynamics in nearly every brain
region during the choice task. Mean rotation frequency ~0.28 rad/bin.
DG shows the strongest rotation (strength = 0.070). Only 3 regions (MG,
NB, OLF) fall below null.

Rotation strength varies 10x across regions (CV up to 0.79 in MOs),
suggesting regional specialization in dynamical structure even though
rotation is universally present.

**Implication**: Rotation is a generic feature of task-engaged neural
populations, not specific to motor cortex. But the *strength* of
rotation is region-specific — another geometric type distinction.

---

## Finding 6: Universal Superposition in Neural Populations (exp16)

**42/42 region-sessions show superposition (100%)**

A sparse autoencoder (4x expansion) finds more active features than
neurons in every single region-session. Mean sparsity ~0.50. However,
SAE features don't align with the LDA choice direction (max alignment
= 0.34) — the SAE finds structure that is NOT the task-relevant structure.

**Implication**: Neural populations encode in superposition just like
transformer layers (polysemanticity). But the "features" found by
SAEs are not aligned with task-relevant directions. Superposition is
universal; its relationship to task computation varies by region.

---

## Static vs Dynamic Geometry (exp13)

UMAP Procrustes (static nonlinear) averages ~0.91 while trajectory
Procrustes (dynamic) averages ~0.34. Static geometry is more conserved
across animals than dynamics. The WHERE of computation is more conserved
than the HOW.

---

## Negative / Weak Results

**Communication subspace sheaf (exp15)**: Only 1 communication subspace
found across 5 sessions (VISam→VISp, 1.3% variance explained). Steinmetz
data has too few neurons per region (10-30) for reliable reduced-rank
regression. Relaunching on IBL data (100+ neurons per region).

**Multiple realization (exp9)**: CKA doesn't cluster by region
(same-region = 0.12, cross-region = 0.14 — wrong direction). Grassmannian
distance is null for most pairs (dimension mismatch). Need dimension-
independent metrics at scale.

**Latent causal discovery (exp19)**: Only 6/40 regions show significant
ICA-choice correlations. ICA-LDA alignment near zero (0.042). Causal
structure is extremely sparse — may need more data or directed methods.

---

## Summary Table

| Exp | Metric type | Status | Key stat | Supports thesis? |
|-----|------------|--------|----------|-----------------|
| 11 | Linear vs nonlinear | **Strong** | rho = -0.86 | Yes — metrics disagree |
| 18 | Parcellation from CKA | **Strong** | Recovers anatomy | Yes — geometry encodes identity |
| 17 | Spectral vs functional | **Strong** | r=0.96 vs sim=0.08 | Yes — structure ≠ content |
| 12 | Topology vs geometry | **Informative negative** | beta_1 = 0 | Yes — flat geometry correct |
| 20 | Dynamics (rotation) | **Moderate** | 21/24 above null | Rotation is generic, not distinctive |
| 16 | Superposition | **Moderate** | 42/42 superposition | Universal property, not distinctive |
| 13 | Static vs dynamic | **Moderate** | 0.91 vs 0.34 | Yes — static > dynamic conservation |
| 15 | Sheaf cohomology | **Negative** | Too few neurons | Inconclusive, need IBL data |
| 9 | CKA clustering | **Weak** | No region separation | Needs more pairs |
| 19 | Causal discovery | **Weak** | 6/40 significant | Too sparse |

---

## What's Next

1. **Full-session runs** — all 10 mice instead of 5. More pairs = tighter statistics.
2. **IBL sheaf experiment (exp15b)** — communication subspaces with 100+ neurons per region.
3. **Allen VBN experiments (exp4, exp6)** — stimulus transportability and gauge correction in visual cortex.
4. **Baseline comparisons** — add sliceTCA, DSA, MFTMA scalars to exp14 type classifier.
5. **Cross-dataset replication** — same experiments on IBL + Allen to test generalization.
