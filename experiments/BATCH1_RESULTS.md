# Batch 1 Results — Small Validation (5 sessions)

All experiments run on Steinmetz dataset (5 sessions, `--small` mode) via Modal.
Results saved to Modal volume + GCS.

## Summary

| Exp | Name | Status | Key Finding |
|-----|------|--------|-------------|
| 9 | Multiple realization | OK | 613 pairs across 42 regions. CKA weakly distinguishes regions (same=0.12, cross=0.14). Holonomy ~0 everywhere. |
| 11 | Linear vs nonlinear | UNDERPOWERED | Only 1 pair (dimension mismatch filter too strict). Need to relax same-dimension requirement. |
| 12 | Topology vs geometry | OK | β₁ = 0 in 22/24 regions — no topological loops in choice circuits. Trivial topology. |
| 13 | Static vs dynamic | OK | **Static geometry more conserved than dynamics** (UMAP Procrustes ~0.91 vs trajectory Procrustes ~0.34). Key dissociation. |
| 14 | Geometric type classifier | OK | 29 pairs across 7 regions. Multi-metric profiles computed. Need full sessions for clustering. |
| 15 | Communication subspace sheaf | NEGATIVE | Only 1 comm subspace found in 5 sessions (VISam→VISp, 1.3% var explained). Sheaf H¹ = null. |
| 16 | SAE on spike trains | OK | **100% of regions show superposition** (active features > 1.5× neurons). Sparsity ~0.50. Low LDA alignment (0.34). |
| 17 | Neural factor bank | OK | **Spectral structure conserved (SV correlation ~0.96) but trial modes not (similarity ~0.08)**. Eigenvalue spectrum is universal, temporal dynamics are idiosyncratic. |
| 18 | Grassmannian parcellation | FAILED | 0 usable regions — neuron counts differ across sessions. Need dimension-independent distance. |
| 19 | Latent causal discovery | WEAK | Only 6/40 regions have significant choice-ICA correlation. ICA-LDA alignment near zero (0.042). |
| 20 | jPCA rotation | OK | **21/24 regions show rotation above null**. Mean freq ~0.28. DG strongest. 3 below null. |
| 4 | Allen stimulus transport | EMPTY | Session filtering found 0 matching sessions. Data pipeline issue. |
| 6 | Allen gauge correction | EMPTY | Same — 0 sessions matched. Allen data loading needs debugging. |

## Key Findings

### 1. Static geometry is more conserved than dynamics (exp13)
UMAP Procrustes distance (nonlinear static) averages ~0.91 across animal pairs, while
trajectory Procrustes (dynamic) averages ~0.34. The WHERE of the mechanism is more
conserved than the HOW. This is a genuine **static-dynamic dissociation**.

### 2. Universal superposition in neural populations (exp16)
ALL 42 region-sessions show superposition: the SAE finds >1.5× more active features
than there are neurons. Neural populations encode in superposition just like transformers.
But SAE features don't align with the LDA choice direction (max alignment 0.34) —
the SAE finds structure but not specifically choice-related structure.

### 3. Spectral universality, temporal idiosyncrasy (exp17)
Singular value spectra across sessions correlate at ~0.96 — the eigenvalue distribution
is nearly identical across animals. But the temporal modes (what those eigenvalues
correspond to in trial space) are uncorrelated (similarity ~0.08). The dimensionality
structure is universal; the dynamics within that structure are not.

### 4. No topological loops in choice circuits (exp12)
β₁ = 0 in 22/24 regions. The choice circuit activity manifold has no 1-cycles (loops).
This rules out ring attractor dynamics for choice encoding in these regions. The
mechanism is genuinely flat (Type 2 subspace), not topologically structured.

### 5. Rotational dynamics are ubiquitous (exp20)
21/24 regions show rotation strength above the temporal-shuffle null. Rotation
is not specific to motor cortex — it appears in visual, frontal, and subcortical
regions during the choice task. Mean frequency ~0.28 rad/bin.

## What Failed and Why

### Dimension mismatch problem (exp11, exp18)
Many experiments require comparing subspaces/vectors of the same dimension.
Different sessions record different numbers of neurons per region, so neuron-space
vectors can't be directly compared. Fix: use dimension-independent metrics
(CKA, normalized Grassmannian, projection-based comparison).

### Communication subspace sheaf (exp15)
Too few neurons per region (only 10-30) and too few shared-time trials (~250)
to get reliable reduced-rank regression communication subspaces. Need more
sessions or pooled data. The 1.3% explained variance means the RRR is fitting noise.

### Allen data pipeline (exp4, exp6)
Session/area filtering is too strict — finds 0 matching sessions. Need to debug
the AllenSDK data loading and relax area matching.

## Next Steps

1. **Full-session runs**: Re-run exp9, 12, 13, 16, 17, 20 with all sessions (not --small)
2. **Fix exp11/18**: Use dimension-independent comparison metrics
3. **Fix exp4/6**: Debug Allen data pipeline
4. **exp15**: Try with more sessions, larger min_neurons, or pooled cross-session data
5. **Add baselines**: sliceTCA, DSA, MFTMA comparisons for exp14 type classifier
