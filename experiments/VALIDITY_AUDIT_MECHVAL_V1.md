# Validity Audit (MECHVAL Framework) V1 — Neuro-Causal-Geometry Experiments

Self-audit using the MECHVAL framework criteria (Tower 2026). This version uses
MECHVAL criterion IDs directly. See VALIDITY_AUDIT_V1.md for the venue-agnostic
version that cites upstream sources.

## MECHVAL Reference

- 27 criteria across 5 validity types: Construct (C1-C5), Measurement (M1-M6),
  Internal (I1-I5), External (E1-E6), Interpretive (V1-V5)
- 5 verdict tiers: Proposed → Causally Suggestive → Mechanistically Supported →
  Triangulated → Validated
- 6 evidence families: Causal (A), Structural (B), Information-theoretic (C),
  Behavioral (D), Representational (E), Measurement (F)
- Dependency chain: Construct → Measurement → Internal → External → Interpretive

## Cross-Domain Note

MECHVAL was developed for transformer circuit claims but explicitly imports its
criteria from neuroscience (I1-I5 from lesion/stimulation methodology), philosophy
of science (C1 from Popper/Mayo), psychometrics (M1-M6 from test theory), pharmacology
(E5 dose-response), and causal inference (E4 from Pearl/Bareinboim transportability).
This audit applies those same criteria back to biological neural circuits — the domain
they were originally drawn from.

---

## Full Criterion Audit

### Finding 1: CKA–Procrustes Anti-Correlation (exp11)
**Claim**: rho = -0.850 between CKA (linear) and UMAP Procrustes (nonlinear) across 1316 region pairs.
**Evidence family**: Representational (E)

| MECHVAL ID | Criterion | Status | Notes |
|------------|-----------|--------|-------|
| C1 | Falsifiability | ✅ | rho > 0 would falsify |
| C2 | Structural plausibility | ✅ | CKA and Procrustes measure different geometric properties by construction |
| C3 | Task specificity | ⚠️ Partial | Anti-correlation holds for choice encoding; not tested off-task |
| C4 | Minimality | N/A | Observational, not circuit |
| C5 | Convergent validity | ⚠️ Partial | Only 2 metric families tested (should add Grassmannian, RSA) |
| M1 | Reliability | ✅ | Stable 5-session→full (rho: -0.864→-0.850) |
| M2 | Baseline separation | ✅ | Permutation null rho ≈ 0 |
| M3 | Stability | ✅ | Robust to session count |
| M4 | Calibration | ⚠️ Partial | rho magnitude interpretable but no published baseline |
| M5 | Sensitivity | N/A | |
| M6 | Invariance | ⚠️ Partial | One dataset only |
| I1 | Necessity | ❌ Untested | No causal intervention |
| I2 | Sufficiency | ❌ Untested | |
| I3 | Specificity | ❌ Untested | |
| I4 | Consistency | ✅ | Cross-session, cross-animal |
| I5 | Confound control | ⚠️ Partial | Neuron count could confound (controlled in exp23) |
| E1 | Intervention reach | N/A | Observational |
| E2 | Prompt generalization | N/A | Not applicable to neural data |
| E3 | Cross-task | ❌ Untested | Only choice task |
| E4 | Cross-dataset | ❌ Untested | Needs IBL (exp46) |
| E5 | Graded response | ❌ Untested | |
| E6 | Novel prediction | ✅ | Predicts which metric class is appropriate per region |
| V1 | Level declaration | ✅ | Representational level |
| V2 | Level-evidence match | ✅ | Evidence supports representational claim |
| V3 | Alternative level | ⚠️ | Could be explained by neuron count artifact (partially controlled) |
| V4 | Anthropomorphism | ✅ | No functional attribution |
| V5 | Scope declaration | ⚠️ Partial | Need to state it's choice-task-specific |

**Verdict: Proposed** | Pass: 9/27 | Partial: 7/27 | Untested: 7/27 | N/A: 4/27
**Evidence families**: Representational (E), Measurement (F) — needs Causal (A) for advancement

---

### Finding 2: Spectral Universality (exp17)
**Claim**: SV correlation 0.936 across animals; trial mode similarity 0.089.

| MECHVAL ID | Status | Notes |
|------------|--------|-------|
| C1 | ✅ | Trial mode sim > 0.5 would falsify the dissociation |
| C4 | ✅ | Dissociation IS discriminant validity |
| M1 | ✅ | Stable across scales |
| M2 | ⚠️ | No Marchenko-Pastur random-matrix baseline |
| M6 | ❌ | One dataset |
| I1 | ❌ | No intervention |
| E4 | ❌ | Needs IBL |
| V1 | ✅ | Representational |
| V4 | ⚠️ | "Universal" may overclaim |

**Verdict: Proposed** | 4 pass, 2 partial, 3 untested

---

### Finding 3: No Topological Loops (exp12)
**Claim**: beta_1 = 0 in 64/73 regions.

| MECHVAL ID | Status | Notes |
|------------|--------|-------|
| C1 | ✅ | Consistent beta_1 >> 0 would falsify |
| M2 | ⚠️ | Need random-vector H1 baseline (exp7_controls) |
| M3 | ✅ | Stable across session counts |
| I1 | N/A | Negative result |
| V1 | ✅ | Topological level |
| V5 | ✅ | Scoped to binary choice |

**Verdict: Proposed (negative result)** — informative absence, not a failure

---

### Finding 4: Ubiquitous Rotation (exp20)
**Claim**: 72/73 regions rotate above temporal-shuffle null.

| MECHVAL ID | Status | Notes |
|------------|--------|-------|
| C1 | ✅ | < 50% above null would falsify |
| M1 | ✅ | 21/24 → 72/73 at scale |
| M2 | ✅ | Temporal-shuffle null |
| E5 | ⚠️ | 10x variation in strength is graded response |
| I1 | ❌ | Is rotation necessary for choice? |
| V4 | ⚠️ | "Rotation" is descriptive, not mechanism |

**Verdict: Proposed** | Reliable observation; causal role unknown

---

### Finding 5: CKA Parcellation Recovers Anatomy (exp18)
**Claim**: k=3 CKA clustering matches thalamic/cortical/posterior anatomy.

| MECHVAL ID | Status | Notes |
|------------|--------|-------|
| C2 | ✅ | Clusters match known connectivity |
| C5 | ✅ | Geometry → anatomy = convergent evidence |
| E6 | ✅ | Predicts anatomy without anatomy input |
| M2 | ⚠️ | No random-label clustering baseline |
| V2 | ✅ | Representational → structural is valid level transition |

**Verdict: Proposed → borderline Triangulated** (2 evidence families: representational + structural)

---

### Finding 6: IIA Causal Evidence (exp42, exp44)
**Claim**: Swapping evidence subspace projections flips choice predictions.

| MECHVAL ID | Status | Notes |
|------------|--------|-------|
| C1 | ✅ | IIA = 0 falsifies |
| I1 | ⚠️ | Partial necessity — swap changes prediction |
| I2 | ❌ | Sufficiency not tested |
| I3 | ⚠️ | Specific to evidence axis (not random) but one axis only |
| I4 | ❌ | No double dissociation with other variables |
| E1 | ⚠️ | One intervention method (linear projection swap) |
| E4 | ❌ | Steinmetz only |
| V2 | ⚠️ | Evidence for representational causal claim, not mechanistic |

**Verdict: Causally Suggestive** (has causal intervention A; lacks sufficiency)

---

### Finding 7: IIA Asymmetry → Causal Direction (exp44)
**Claim**: IIA(A→B) > IIA(B→A) implies A causes B.

| MECHVAL ID | Status | Notes |
|------------|--------|-------|
| C1 | ✅ | Asymmetry inconsistent with hierarchy falsifies |
| C2 | ⏳ | Pending Allen Atlas validation (exp45) |
| I1 | ⚠️ | Derived from causal IIA intervention |
| E4 | ❌ | One dataset |
| V2 | ⚠️ | "Causal direction" is strong for the evidence |

**Verdict: Causally Suggestive** | Strongest claim needs most validation

---

### Finding 8: Consensus Causal Graph (exp32)
**Claim**: MOs→CA1→root from PC/LiNGAM/CD-NOD consensus.

| MECHVAL ID | Status | Notes |
|------------|--------|-------|
| C5 | ✅ | Three algorithms agree |
| I5 | ⚠️ | Session-level confounds not fully controlled |
| M2 | ⚠️ | Bootstrap stability computed, not formally tested |
| V2 | ⚠️ | Observational causal discovery ≠ intervention |

**Verdict: Proposed → borderline Causally Suggestive** (convergent methods but observational)

---

### Finding 9: Silencing Validation (exp47)
**Claim**: Within-region IIA predicts silencing behavioral effect.
**Result**: rho = 0.186, NOT significant.

| MECHVAL ID | Status | Notes |
|------------|--------|-------|
| I1 | ❌ | The test FAILED — IIA doesn't predict necessity |
| M2 | ✅ | Clear baseline (permutation null) |
| V5 | ✅ | Honest negative result reported |

**Verdict: Disconfirmed (for this specific claim)** — within-region IIA ≠ causal necessity. This is a valid MECHVAL negative result: the criterion was tested and failed. The claim is properly scoped to "geometric flexibility" rather than "causal importance."

---

## Aggregate Score Card

### By Verdict Tier

| Tier | Findings | Count |
|------|----------|-------|
| **Validated** | — | 0 |
| **Triangulated** | — | 0 |
| **Mechanistically Supported** | — | 0 |
| **Causally Suggestive** | IIA (exp42/44), IIA asymmetry (exp44) | 2 |
| **Proposed (strong)** | CKA anti-correlation, parcellation | 2 |
| **Proposed** | Spectral, topology, rotation, superposition, causal graph | 5 |
| **Disconfirmed** | IIA→silencing (exp47) | 1 |
| **Pending** | Allen (exp45), IBL (exp46), real silencing (exp47b) | 3 |

### By Evidence Family Coverage

| Family | Experiments | Coverage |
|--------|-------------|----------|
| **Causal (A)** | exp42, exp44 | ⚠️ IIA only — one method |
| **Structural (B)** | exp18 (parcellation), exp45 (Allen) pending | ⚠️ Partial |
| **Information (C)** | — | ❌ None |
| **Behavioral (D)** | exp47 (silencing effect sizes) | ⚠️ One test, negative |
| **Representational (E)** | exp11, 12, 13, 17, 18, 20 | ✅ Strong |
| **Measurement (F)** | exp23 (subsampling), exp24 (robustness) | ⚠️ Partial |

**Critical gap**: Evidence family coverage is heavily representational. No information-theoretic evidence. Causal evidence from one method only (IIA swap). Structural validation pending.

### By Validity Type

| Type | Met | Partial | Untested |
|------|-----|---------|----------|
| **Construct** | C1, C2, C4 | C3, C5 | — |
| **Measurement** | M1, M3 | M2, M4, M6 | M5 |
| **Internal** | — | I1 (partial, IIA only) | I2, I3, I4, I5 |
| **External** | — | E5, E6 | E1, E2, E3, E4 |
| **Interpretive** | V1, V5 | V2, V4 | V3 |

**Bottleneck**: Internal validity. Only I1 (necessity) is partially addressed, and only
through one method (IIA swap). No sufficiency (I2), no double dissociation (I4), no
confound control (I5). This is the systematic gap that prevents advancement to
Mechanistically Supported.

### Three Experiments That Would Change Everything

1. **exp45 (Allen Atlas)**: If Spearman rho > 0.3, advances Finding 5 (parcellation) and
   Finding 7 (IIA direction) by adding Structural (B) evidence family. Moves parcellation
   toward Triangulated.

2. **exp46 (IBL transport)**: If alpha correlation > 0.5, advances ALL findings by
   satisfying E4 (cross-dataset). This is the single highest-impact experiment.

3. **exp47b (real silencing data)**: Even if negative, the per-coordinate behavioral
   effects from Zatka-Haas provide ground-truth behavioral (D) evidence. If positive,
   advances IIA claims from Suggestive toward Mechanistically Supported.

## Comparison to Published MI Circuits (MECHVAL Case Studies)

| Claim | MECHVAL verdict | Our closest analog | Our verdict |
|-------|----------------|--------------------|----|
| IOI Circuit (Wang et al. 2023) | Mechanistically Supported | IIA causal evidence (exp42/44) | Causally Suggestive |
| Induction Heads (Olsson et al. 2022) | Triangulated | CKA anti-correlation (exp11) | Proposed |
| Knowledge Neurons | Proposed | Neural superposition (exp16) | Proposed |
| Grokking Circuit | Causally Suggestive | IIA asymmetry (exp44) | Causally Suggestive |

Our strongest claims (IIA-based) are at the same tier as Grokking circuits in MI.
To reach IOI-level (Mechanistically Supported), we need sufficiency evidence (I2)
and multi-method intervention (E1).
