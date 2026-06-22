# Pre-Registered Hypotheses

Filed before first data run. Each experiment has a prediction and
a falsification criterion. If the falsification criterion is met,
the hypothesis is rejected.

## Steinmetz (exp7, exp8, exp9) — simultaneous multi-region

### Exp 7: Sheaf cohomology of the choice circuit
- **H7a**: Full choice circuit (frontal + sensorimotor + basal ganglia + midbrain) has H¹ > 0 (distributed, not localizable to any single region)
- **H7b**: Frontal-motor sub-circuit (MOs, ACA, PL, MOp) has H⁰ > 0 (locally coherent)
- **Falsification**: H¹ = 0 for full circuit → choice is fully localizable, no distributed mechanism
- **Comparisons**: 2 (full circuit H¹, frontal H⁰)

### Exp 8: Holonomy fingerprints
- **H8a**: Within-region holonomy distance < cross-region holonomy distance (mechanism identity is stable within a region across animals)
- **H8b**: Cross-region holonomy distance is large even when decoding accuracy is matched (holonomy discriminates mechanisms that CKA doesn't)
- **Falsification**: Within ≥ cross → holonomy is not diagnostic of mechanism identity
- **Comparisons**: 1 (within vs cross, one-sided Mann-Whitney)

### Exp 9: Multiple realization (CKA vs d_G dissociation)
- **H9a**: For accuracy-matched region pairs, Spearman(CKA, d_G) < 0.7 (CKA and Grassmannian distance measure different things)
- **H9b**: Cross-region pairs have higher d_G but similar CKA compared to within-region pairs (CKA conflates representation with mechanism)
- **Falsification**: Spearman ≥ 0.7 → CKA and d_G are empirically equivalent, no dissociation
- **Comparisons**: 2 (correlation test, within vs cross d_G)

## IBL (exp1, exp2, exp3) — cross-animal reproducibility

### Exp 1: Cross-animal choice subspace stability
- **H1**: d_G ordering: within-animal < within-lab < across-lab (choice subspace is conserved but with lab-specific variation)
- **Falsification**: No significant ordering → subspace is not conserved across animals
- **Comparisons**: 2 (pairwise ordering tests)

### Exp 2: CKA vs Grassmannian dissociation (IBL)
- **H2**: There exist region pairs where CKA > 0.8 but d_G > median(d_G) — high representational similarity does NOT imply mechanism identity
- **Falsification**: CKA > 0.8 always implies d_G < median → CKA is sufficient for mechanism comparison
- **Comparisons**: 1 (existence test)

### Exp 3: IBL sheaf cohomology
- **H3a**: Full IBL choice circuit across 15+ regions has H¹ > 0
- **H3b**: H¹ dimension correlates with number of choice-encoding regions (more regions → more distributed)
- **Falsification**: H¹ = 0 → choice circuit is localizable even at whole-brain scale
- **Comparisons**: 2

## Allen VBN (exp4, exp5, exp6) — visual processing

### Exp 4: Stimulus subspace transportability
- **H4**: d_G(same-region, cross-animal) < d_G(cross-region, same-animal) — the stimulus mechanism is more conserved within a brain area across animals than across areas within the same animal
- **Falsification**: Reverse ordering → mechanism is animal-specific, not region-specific
- **Comparisons**: 1 (ordering test)

### Exp 5: Behavioral state modulation
- **H5**: Mean principal angle between active and passive stimulus subspaces < 15° — the causal subspace is circuit-determined, not state-dependent
- **Falsification**: Mean angle > 30° → behavioral state fundamentally changes the mechanism
- **Comparisons**: 1 (mean angle vs threshold)

### Exp 6: Gauge correction effect
- **H6**: Gauge normalization reduces variance of within-animal cross-session d_G by > 30%
- **Falsification**: Variance reduction < 10% → gauge freedom is not a practical concern
- **Comparisons**: 1 (variance ratio)

---

## Multiple Comparisons Summary

| Experiment | Comparisons | Dataset |
|-----------|------------|---------|
| Exp 1 | 2 | IBL |
| Exp 2 | 1 | IBL |
| Exp 3 | 2 | IBL |
| Exp 4 | 1 | Allen |
| Exp 5 | 1 | Allen |
| Exp 6 | 1 | Allen |
| Exp 7 | 2 | Steinmetz |
| Exp 8 | 1 | Steinmetz |
| Exp 9 | 2 | Steinmetz |
| **Total** | **13** | |

Bonferroni-corrected α = 0.05/13 = 0.00385

mc_iayn FWER at m=13, α=0.05: 1 - (1-0.05)^13 = 0.487 uncorrected.
After Bonferroni: each test at α=0.00385, FWER ≤ 0.05.
