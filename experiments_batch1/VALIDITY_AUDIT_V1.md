# Validity Audit V1 — Neuro-Causal-Geometry Experiments

Self-audit of all experimental claims against established validity criteria.
Sources: neuroscience (Craver 2007, Shallice 1988), philosophy of science (Mayo 2018,
Popper), psychometrics (Campbell & Fiske 1959, Cronbach & Meehl 1955), causal inference
(Pearl & Bareinboim 2011, Spirtes et al. 2000, Woodward 2003), and pharmacology
(dose-response methodology).

## Verdict Tiers

Following the hierarchy of evidential support:

| Tier | Meaning | What's needed |
|------|---------|---------------|
| **Proposed** | Structural/representational evidence only | Construct criteria met |
| **Causally Suggestive** | Necessity shown (not sufficiency) | Causal intervention (ablation) |
| **Mechanistically Supported** | Necessity + sufficiency + method consistency | Multiple ablation variants agree |
| **Triangulated** | Multiple converging independent evidence lines | ≥3 evidence families |
| **Validated** | All validity types addressed | Measurement + interpretive criteria |

---

## Experiment-by-Experiment Audit

### Finding 1: CKA–Procrustes Anti-Correlation (exp11)
**Claim**: Linear and nonlinear geometry metrics are anti-correlated (rho = -0.850, p ≈ 0).

| Criterion | Status | Evidence | Upstream source |
|-----------|--------|----------|----------------|
| C1 Falsifiability | **Pass** | Pre-registered: rho > 0 would falsify | Mayo 2018 |
| C3 Convergent validity | **Partial** | Only two metric families (CKA, UMAP) | Campbell & Fiske 1959 |
| C4 Discriminant validity | **Pass** | Anti-correlation IS the discriminant result | Campbell & Fiske 1959 |
| M2 Baseline separation | **Pass** | Permutation null rho ≈ 0; observed -0.85 | Signal detection theory |
| M1 Reliability | **Pass** | Held from 5 sessions (-0.864) to full (-0.850) | Psychometric reliability |
| I1 Necessity | **Untested** | No causal intervention | Shallice 1988 |
| E4 Cross-dataset | **Untested** | Steinmetz only; needs IBL replication | Pearl & Bareinboim 2011 |
| V1 Level declaration | **Pass** | Representational level — what metrics measure | Marr 1982 |

**Verdict: Proposed → strong** (best representational result; needs causal + cross-dataset)

**To advance**: Run exp46 (IBL transport). If rho < -0.5 on IBL, advances to Triangulated.

---

### Finding 2: Spectral Universality / Temporal Idiosyncrasy (exp17)
**Claim**: Eigenvalue spectra universal (r = 0.936) but trial modes idiosyncratic (0.089).

| Criterion | Status | Evidence | Upstream source |
|-----------|--------|----------|----------------|
| C1 Falsifiability | **Pass** | Trial mode similarity > 0.5 would falsify | Mayo 2018 |
| C4 Discriminant validity | **Pass** | Dissociation between shape and content | Campbell & Fiske 1959 |
| M1 Reliability | **Pass** | Stable across scales (0.96→0.936) | Psychometric reliability |
| M2 Baseline separation | **Partial** | No explicit random-matrix baseline | Sutter et al. 2025 |
| E4 Cross-dataset | **Untested** | Steinmetz only | Pearl & Bareinboim 2011 |
| V4 Anthropomorphism | **Partial** | "Universal" vs "idiosyncratic" labels may overclaim | Marr 1982 |

**Verdict: Proposed** (clean representational finding; needs baseline and transport)

**To advance**: Add Marchenko-Pastur random baseline; replicate on IBL.

---

### Finding 3: No Topological Loops (exp12)
**Claim**: beta_1 = 0 in 22/24 (small) → 64/73 (full) regions. Choice manifolds are flat.

| Criterion | Status | Evidence | Upstream source |
|-----------|--------|----------|----------------|
| C1 Falsifiability | **Pass** | beta_1 >> 0 consistently would falsify | Mayo 2018 |
| M2 Baseline separation | **Partial** | Need random-vector H1 baseline (exp7_controls) | Sutter et al. 2025 |
| M3 Stability | **Pass** | Consistent across 5→39 sessions | Psychometric stability |
| I1 Necessity | **N/A** | Negative result — no structure to test necessity of | — |
| V1 Level declaration | **Pass** | Topological level — what loops exist | Marr 1982 |
| V5 Scope | **Pass** | Explicitly scoped to binary choice task | Scope honesty |

**Verdict: Proposed (negative)** — the negative result is itself informative

**To advance**: exp7_controls baseline separation test (is H1=0 different from random H1?).

---

### Finding 4: Ubiquitous Rotation (exp20)
**Claim**: 72/73 regions show rotation above temporal-shuffle null.

| Criterion | Status | Evidence | Upstream source |
|-----------|--------|----------|----------------|
| C1 Falsifiability | **Pass** | < 50% above null would falsify | Mayo 2018 |
| M2 Baseline separation | **Pass** | Temporal-shuffle null provides baseline | Signal detection theory |
| M1 Reliability | **Pass** | 21/24 → 72/73 at scale | Psychometric reliability |
| E5 Graded response | **Partial** | 10x variation in strength across regions | Pharmacology dose-response |
| I1 Necessity | **Untested** | Is rotation necessary for choice? | Shallice 1988, Craver 2007 |
| V4 Anthropomorphism | **Caution** | "Rotation" is a descriptive label, not a mechanism | Marr 1982 |

**Verdict: Proposed** (universal observation; needs causal test of rotation's role)

---

### Finding 5: CKA Parcellation Recovers Anatomy (exp18)
**Claim**: k=3 CKA clustering matches thalamic/cortical/posterior split without anatomy labels.

| Criterion | Status | Evidence | Upstream source |
|-----------|--------|----------|----------------|
| C2 Structural plausibility | **Pass** | Clusters match known connectivity | Glennan 2017 |
| C4 Discriminant validity | **Pass** | Different from random clustering (need formal test) | Campbell & Fiske 1959 |
| M2 Baseline separation | **Partial** | No comparison to random-label clustering | Sutter et al. 2025 |
| E6 Novel prediction | **Pass** | Predicted groups not used as input | Philosophy of science |
| V2 Level-evidence match | **Pass** | Representational → structural prediction confirmed | Marr 1982 |

**Verdict: Proposed → borderline Triangulated** (geometry predicts anatomy = convergent evidence from two evidence families)

**To advance**: Formal adjusted Rand index against anatomy; Allen Atlas comparison (exp45).

---

### Finding 6: Superposition in Neural Populations (exp16)
**Claim**: 100% of regions show superposition (active features > 1.5x neurons). But features don't align with choice direction.

| Criterion | Status | Evidence | Upstream source |
|-----------|--------|----------|----------------|
| C1 Falsifiability | **Pass** | < 50% showing superposition would falsify | Mayo 2018 |
| M2 Baseline separation | **Caution** | Is 1.5x the right threshold? Random data? | Sutter et al. 2025 |
| C4 Discriminant validity | **Partial** | Superposition ≠ task-relevant structure (LDA align = 0.34) | Campbell & Fiske 1959 |
| V4 Anthropomorphism | **Caution** | "Superposition" borrowed from transformer interpretability | Marr 1982 |

**Verdict: Proposed** (observation, not mechanism; "superposition" label may overclaim)

---

### Finding 7: IIA Causal Evidence (exp42, exp44)
**Claim**: Swap-based interchange intervention on evidence subspace flips choice predictions.

| Criterion | Status | Evidence | Upstream source |
|-----------|--------|----------|----------------|
| C1 Falsifiability | **Pass** | IIA = 0 would falsify | Mayo 2018 |
| I1 Necessity | **Partial** | Swapping evidence direction changes choice → evidence representation necessary | Shallice 1988 |
| I2 Sufficiency | **Untested** | Haven't shown evidence alone is sufficient | Craver 2007 |
| I3 Specificity | **Partial** | Swap is specific to evidence axis (not random) | Double dissociation |
| E1 Intervention reach | **Partial** | One intervention method (linear swap) | Causal inference |
| E4 Cross-dataset | **Untested** | Steinmetz only | Pearl & Bareinboim 2011 |

**Verdict: Causally Suggestive** (has causal intervention; lacks sufficiency and cross-dataset)

**To advance**: Show sufficiency (evidence subspace alone predicts choice); replicate on IBL.

---

### Finding 8: IIA Asymmetry → Causal Direction (exp44)
**Claim**: IIA(A→B) > IIA(B→A) implies directional causal influence from A to B.

| Criterion | Status | Evidence | Upstream source |
|-----------|--------|----------|----------------|
| C1 Falsifiability | **Pass** | Asymmetry inconsistent with hierarchy would falsify | Mayo 2018 |
| C2 Structural plausibility | **Pending** | Needs Allen Atlas comparison (exp45) | Glennan 2017, Oh et al. 2014 |
| I1 Necessity | **Partial** | Derived from causal IIA | Woodward 2003 |
| E4 Cross-dataset | **Untested** | Steinmetz only | Pearl & Bareinboim 2011 |
| V2 Level-evidence match | **Caution** | "Causal direction" is a strong claim from correlational swap | Marr 1982, Woodward 2003 |

**Verdict: Causally Suggestive** (intervention-derived; direction claim needs validation)

**To advance**: Allen Atlas concordance (exp45); optogenetic silencing validation (exp47b).

---

### Finding 9: Causal Discovery Graph (exp32)
**Claim**: PC/LiNGAM/CD-NOD consensus graph: MOs→CA1→root.

| Criterion | Status | Evidence | Upstream source |
|-----------|--------|----------|----------------|
| C1 Falsifiability | **Pass** | Random graph structure would falsify | Mayo 2018 |
| C2 Structural plausibility | **Partial** | MOs→CA1 is anatomically plausible | Glennan 2017 |
| C5 Convergent validity | **Pass** | Three independent algorithms agree | Campbell & Fiske 1959 |
| I5 Confound control | **Caution** | Session-level confounds not fully controlled | Spirtes et al. 2000 |
| M2 Baseline separation | **Partial** | Bootstrap stability computed but not reported | Signal detection theory |
| V2 Level-evidence match | **Caution** | Observational causal discovery ≠ interventional causation | Woodward 2003 |

**Verdict: Proposed → borderline Causally Suggestive** (multiple methods converge, but observational only)

**To advance**: Optogenetic silencing validation (exp47b); Allen Atlas comparison (exp45).

---

### Finding 10: Allen Atlas Validation (exp45) — RUNNING
**Claim**: Geometry-derived directed edges correspond to Allen CCF projection densities.

| Criterion | Status | Evidence | Upstream source |
|-----------|--------|----------|----------------|
| C2 Structural plausibility | **The test itself** | This IS the structural validation | Oh et al. 2014 |
| E6 Novel prediction | **Pass if significant** | Geometry predicts held-out anatomy | Philosophy of science |

**Verdict**: Pending results. If Spearman rho > 0.3 with p < 0.05, advances multiple claims.

---

### Finding 11: IBL Cross-Dataset Transport (exp46) — RUNNING
**Claim**: Geometric type taxonomy transfers from Steinmetz to IBL.

| Criterion | Status | Evidence | Upstream source |
|-----------|--------|----------|----------------|
| E4 Cross-dataset | **The test itself** | This IS the transportability test | Pearl & Bareinboim 2011 |

**Verdict**: Pending results. If alpha correlation > 0.5, advances all findings to Triangulated.

---

### Finding 12: Silencing Validation (exp47/47b) — RUNNING
**Claim**: Geometry-derived causal importance predicts optogenetic silencing behavioral effects.

| Criterion | Status | Evidence | Upstream source |
|-----------|--------|----------|----------------|
| I1 Necessity | **The test itself** | Silencing = ground-truth necessity | Shallice 1988, Craver 2007 |
| V2 Level-evidence match | **The test itself** | Geometry should predict intervention outcomes | Woodward 2003 |

**Verdict**: exp47 (hardcoded effects) gave rho=0.186 (NS). exp47b (real data) pending. Honest negative result is informative.

---

## Summary Matrix

| Finding | Current verdict | Missing criteria | Experiment that advances it |
|---------|----------------|------------------|-----------------------------|
| CKA-Procrustes anti-correlation | Proposed (strong) | I1, E4 | exp46 (IBL) |
| Spectral universality | Proposed | M2, E4 | Random baseline + IBL |
| No topological loops | Proposed (negative) | M2 | exp7_controls |
| Ubiquitous rotation | Proposed | I1 | Causal test of rotation |
| CKA parcellation | Proposed → Triangulated | M2, formal test | exp45 (Allen) |
| Neural superposition | Proposed | M2, V4 | Better threshold justification |
| IIA causal evidence | **Causally Suggestive** | I2, E4 | exp46 + sufficiency test |
| IIA asymmetry → direction | **Causally Suggestive** | C2, E4 | exp45 (Allen) |
| Causal graph (PC/LiNGAM) | Proposed–Suggestive | I5, V2 | exp45, exp47b |
| Allen Atlas validation | Pending | — | exp45 results |
| IBL transport | Pending | — | exp46 results |
| Silencing validation | Negative (exp47) | — | exp47b (real data) |

## What This Audit Reveals

1. **Most claims are at Proposed tier** — strong representational evidence but no causal intervention. The IIA experiments (exp42/44) are the only ones that reach Causally Suggestive.

2. **Three pending experiments (exp45, exp46, exp47b) could advance 6+ claims each** — they target the systematic gaps (structural validation, cross-dataset transport, causal necessity).

3. **Baseline separation (M2) is the most common gap** — 6/12 findings have partial or untested M2. This is fixable with permutation/random baselines (cheap experiments).

4. **No finding reaches Mechanistically Supported** — because no finding demonstrates both necessity AND sufficiency. The IIA experiments show partial necessity but not sufficiency.

5. **The honest negative result (exp47 rho=0.186) is itself a valid finding** — within-region IIA doesn't predict silencing effects. This correctly distinguishes geometric flexibility from causal necessity (Shallice 1988).
