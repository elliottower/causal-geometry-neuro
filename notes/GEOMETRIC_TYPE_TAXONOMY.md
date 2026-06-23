# Geometric Type Taxonomy for Neural Mechanisms

Adapted from the Mechanistic Views stratified view and Perplexity iteration 1 analysis.
Maps the full hierarchy of "what kind of thing is a mechanism" to both transformers and real neural data.

## The hierarchy

| Type | Name | Math | Transformers | Neural data | Our experiment |
|------|------|------|-------------|-------------|----------------|
| 0 | Scalar | R | "Head L7H3 accounts for 40% of IOI" | "MOs decodes choice at 0.82 accuracy" | Baseline |
| 1 | Direction | RP^{d-1} | Steering vectors, linear probes | "Choice axis" in a region | Not yet (should add) |
| 2 | Subspace | Gr(k, d) | DAS rotation, factor bank | k-dim choice subspace | Exp 1, 2, 4, 6, 9 |
| 3 | Orbit | Gr(k,d)/G | Bank column span mod O(k) gauge | Subspace mod electrode placement | Exp 6 (gauge correction) |
| 4 | Fiber bundle | E -> B with fibers in Gr(k,d) | Task-varying selectors | State-dependent subspace | Exp 5, 8 |
| 5 | Sheaf | Cech cohomology over region graph | Cross-site DAS coherence | Regional subspaces + compatibility | Exp 3, 7 |
| 6 | Flag | Fl(k1,...,km; d) | Nested IOI sub-circuits | Motor hierarchy (output-null inside prep) | Not yet |
| 7 | Stratified | Whitney stratification | Different circuits on different strata | Unknown — this is the meta-question | All experiments together |

## What our results say so far (5-session Steinmetz)

### Exp 7 (Sheaf): H1 >> 0 in 5/5 sessions
The choice mechanism is NOT Type 2 (flat subspace). It's at least Type 5 (sheaf).
Different regions have local choice subspaces that don't glue into a global one.

### Exp 8 (Holonomy): = 0
Inconclusive. PCA tracks variance, not causation. Need DAS (Type 4 test).

### Exp 9 (CKA vs d_G): CKA same-region ~ CKA cross-region
Type 0 (scalar) similarity metrics are useless for mechanism identity.
CKA can't tell same-mechanism from different-mechanism.

## Decision tree (where are we?)

```
H1 = 0?
├── YES → Type 2 (subspace) or lower
│         → Test holonomy: != 0?
│           ├── NO  → Type 2 (flat subspace)
│           └── YES → Type 4 (fiber bundle)
└── NO  → Type 5 (sheaf) or higher  ← WE ARE HERE
          → Next: compute obstruction cocycles
          ├── Defects at specific edges → Type 5 with known loci
          └── Defects everywhere → Type 6 (flag) or Type 7 (stratified)
```

## Next computations (priority order)

### 1. Obstruction cocycles (highest priority)
Decompose H1 into representative cocycles mapped onto the anatomical region graph.
For each edge (region_A, region_B), compute how much their local choice subspaces
fail to agree. The magnitude = "inconsistency" at that connection.
This turns "choice circuit is distributed" into "choice circuit is distributed
BECAUSE the MOs-ACA connection is geometrically incoherent."

### 2. Direction vs subspace test (new experiment)
Compute choice direction (top eigenvector) per session per region.
Pairwise cosine similarity across animals.
Prediction: cosine similarity is nearly uniform (directions vary)
but Grassmannian distance is structured (subspaces are stable).
This proves Type 1 is wrong and Type 2+ is necessary.

### 3. Fix holonomy with DAS (needs GPU)
Use DAS rotation optimization instead of PCA for time-varying subspaces.
Natural loop in IBL: high-prior-left -> neutral -> high-prior-right -> neutral.
Holonomy of this loop = how much top-down context modulates circuit geometry.

### 4. Flag structure (future)
Compute nested subspace hierarchy per region: 2-d core inside 5-d extension
inside full k-d subspace. If the hierarchy is consistent across animals,
the mechanism is Type 6 (flag).

## Connection to Mechanistic Views

| MechView | Geometric type | Evidence |
|----------|---------------|----------|
| Instrumental | Type 0 (scalar) | Decoding accuracy |
| Object | Type 1 (direction) | Ablation of single neurons/directions |
| Role | Type 2 (subspace) | Cross-animal subspace stability |
| Subspace | Type 2-3 (subspace/orbit) | DAS, Grassmannian distance |
| Structural | Type 3 (orbit) | Gauge-normalized comparisons |
| Process | Type 4 (fiber bundle) | Training/learning trajectory |
| Stratified | Type 7 (all of the above) | All experiments simultaneously |

The sheaf (Type 5) is new — it doesn't map cleanly onto any single mechview
because it's about the RELATIONSHIP between local descriptions across regions.
It's closest to the structural view (equivalence class under symmetry) but
the symmetry is spatial (across brain regions) rather than algebraic (gauge freedom).
