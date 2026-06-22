# Mechanism Identity Across Neural Populations: A Causal Geometry Approach

**Working Draft — Elliott Tower & [Neuro Co-author]**
*June 2026*

***

## Abstract

Neuroscience increasingly describes cognition through neural population geometry: low-dimensional manifolds encoding task variables in the coordinated activity of many neurons. Yet the field lacks a principled answer to a prior question: when do two neural populations — across animals, labs, brain regions, or recording sessions — instantiate the *same* mechanism, as opposed to merely producing similar behavior? We propose that this question is geometric. A mechanism is a causal subspace — a point on the Grassmannian manifold — equipped with transport maps induced by the circuit's connectivity. Two populations instantiate the same mechanism if and only if their causal subspaces are related by a transport-respecting alignment: a map that respects the underlying connectivity structure rather than merely matching behavioral outputs. We formalize this using the Grassmannian Structural Causal Model (G-SCM), introduce three computable invariants for mechanism identity (geodesic distance under gauge normalization, holonomy fingerprints, and sheaf-cohomological localizability), and propose a No-Free-Lunch Triangulation principle: no single evidence domain is sufficient to identify a mechanism, but convergent evidence from connectivity-space, activation-space, and dynamics-space is jointly sufficient. We apply this framework to three open datasets — IBL Brain-Wide Map, Allen Visual Behavior Neuropixels, and Steinmetz et al. 2019 — to ask whether population subspaces encoding decision variables are stable across animals, labs, and brain regions. The central empirical question: is cross-animal subspace transportability a reliable signature of shared mechanistic implementation, or merely of shared behavioral function?[^1]

***

## 1. Introduction

### 1.1 The Problem: What Does It Mean for Two Neural Populations to Compute the Same Thing?

Neuroscience is entering a geometrical era. The dominant explanatory framework for many cognitive computations — decision-making, working memory, motor planning — is no longer "which neurons fire" but "what low-dimensional subspace organizes population activity". Decision variables are encoded in low-dimensional manifolds embedded in high-dimensional population activity. Motor intent occupies orthogonal subspaces that can be selectively silenced without disrupting other computations. Neural manifolds reconfigure during learning. This is genuine progress.

But it generates a new foundational problem. The field now routinely asks whether "the same" manifold structure appears across animals, labs, brain regions, or species. The IBL Brain-Wide Map project — recording 621,733 neurons across 139 mice in 12 laboratories — was explicitly designed to ask whether neural correlates of decision-making are reproducible across animals and labs. The Allen Visual Behavior Neuropixels dataset allows comparison across 153 recording sessions from 81 mice performing a standardized behavioral task. The Steinmetz et al. (2019) dataset maps choice-related activity across 42 brain regions simultaneously.

In all of these cases, similarity is measured using representational similarity analysis (RSA), centered kernel alignment (CKA), or cross-temporal generalization of linear decoders. These measures quantify whether two populations encode *similar information*, not whether they implement the *same mechanism*. A population that encodes choice via an attractor mechanism and one that encodes it via a feed-forward ramp can have high CKA if both produce similar choice-correlated population trajectories, while implementing fundamentally different computations.

This distinction — between representational similarity and mechanistic identity — is the gap this paper addresses.

### 1.2 The Gap: Representational Similarity ≠ Mechanism Identity

The distinction can be made precise. CKA measures the alignment between kernel matrices of two populations' activity: it is invariant to orthogonal transformation and uniform scaling, but not to arbitrary linear transformation. Two populations with high linear CKA have similar *representational geometry*, but may differ in: (a) which dimensions are causally active for behavior, (b) how information flows between populations (transport structure), and (c) whether the subspace is a stable attractor or a transient trajectory.

Critically, recent work formalizes this failure: without structural constraints on alignment maps, any two populations with equivalent behavioral readouts can be aligned with arbitrarily high fidelity to a given causal model. The field's implicit solution — restrict to linear alignment maps — is an empirical regularity rather than a principled solution. What is needed is a framework where the *connectivity structure* itself constrains which alignments are valid. This is the Grassmannian SCM (G-SCM): a formalization of mechanism identity in terms of subspaces equipped with transport maps induced by actual circuit connectivity.

### 1.3 Contributions

This paper makes four contributions:

1. **Conceptual:** A taxonomy of six ways two neural populations can appear similar while implementing different mechanisms, motivating why geometric invariants stronger than CKA are needed.
2. **Formal:** The G-SCM applied to neural circuits, with three computable invariants for cross-population mechanism identity: gauge-normalized Grassmannian distance, holonomy group isomorphism, and sheaf cohomology class.
3. **Empirical:** Analysis of IBL, Allen VBN, and Steinmetz datasets using these invariants to ask which brain regions and animal groups share mechanism identity (not just representational similarity) for decision-relevant variables.
4. **Methodological:** A practical decision procedure for when cross-population comparisons constitute evidence for mechanism identity vs. mere functional convergence.

***

## 2. Background

### 2.1 Neural Population Geometry

The neural population geometry framework treats the activity of N neurons as a point in an N-dimensional space, and characterizes cognition through the geometry of the trajectories and manifolds this activity traces. Key measures include: intrinsic dimensionality of the manifold, principal angles between task-variable subspaces, and decodability of behavioral variables from linear projections. The framework has successfully characterized decision-making circuits, motor planning, and cortical reorganization during learning.

Recent work demonstrates that neural manifold dimensionality is stable across motor perturbations — manifolds persist even when behavioral strategy changes, suggesting that the underlying geometry reflects circuit organization rather than just task statistics. This is consistent with the core claim of this paper: causal subspaces are latent in circuit connectivity, not merely in behavioral outputs.

### 2.2 Representational Similarity Measures and Their Limitations

Representational Similarity Analysis (RSA) compares the geometry of pairwise stimulus distances across two populations. CKA extends this by aligning kernel matrices, yielding a measure invariant to orthogonal transformation. These measures are powerful for asking whether two populations encode similar information but do not address causal structure.

The fundamental limitation: CKA can be high between two populations implementing the same behavior via different mechanisms, and can be low between populations implementing the same mechanism on different tasks (because the task-relevant subspace is a small fraction of total variance). For mechanism identity, we need a measure that specifically compares the *causally active* subspace — the subspace whose disruption changes behavior — not the total representational geometry.

### 2.3 Causal Models of Neural Population Activity

The interventional state-space model (iSSM) is a recent framework for building causal models of neural dynamics that predict responses to novel perturbations. Unlike standard SSMs, iSSMs can distinguish statistical association from causal structure by fitting models to data collected during optogenetic or electrical stimulation. Identifiability theory for iSSMs establishes that pairs of distinct perfect interventions on each latent node suffice for identifiability. This constrains what kind of evidence is needed to make mechanism identity claims that go beyond representational similarity.

### 2.4 Multiple Realizability and Mechanism Identity

The problem of cross-animal mechanism identity is a variant of the classical multiple realizability problem in philosophy of mind and neuroscience. Multiple realizability holds that the same cognitive function can be implemented by different neural substrates. Recent work argues mechanistic explanation survives multiple realizability once we identify the correct level of description. Our framework operationalizes this: populations can be multiply realizing the same mechanism (same causal subspace under different connectivity configurations) or implementing genuinely different mechanisms (different causal subspaces that produce similar behavioral outputs). The framework provides criteria to distinguish these cases.

***

## 3. The Grassmannian Structural Causal Model for Neural Circuits

### 3.1 Causal Subspaces in Neural Populations

Let a neural population be described by its firing rate matrix \(X \in \mathbb{R}^{N \times T}\) across N neurons and T time points. A causal subspace \(S \in \text{Gr}(k, N)\) for a task variable V is a k-dimensional linear subspace of the neural activity space such that swapping the k-dimensional projection of X between a source and base trial causes the population to behave as if V took its source value. This is the neural analogue of the DAS definition from mechanistic interpretability.

The Grassmannian \(\text{Gr}(k, N)\) is the smooth manifold of all k-dimensional subspaces of \(\mathbb{R}^N\). It carries a natural metric: the geodesic distance between two subspaces \(S_1\) and \(S_2\) is

\[d(S_1, S_2) = \left(\sum_{i=1}^k \theta_i^2\right)^{1/2}\]

where \(\theta_1 \geq \cdots \geq \theta_k \geq 0\) are the principal angles between the subspaces.

### 3.2 The G-SCM for Neural Circuits

A Grassmannian Structural Causal Model for a neural circuit consists of:

- A set of sites V (neural populations, indexed by region and time)
- A directed acyclic graph E over sites
- For each site v, a causal subspace \(S_v \in \text{Gr}(k_v, N_v)\)
- For each edge \(e = (u, v)\), a transport map \(\phi_e: \text{Gr}(k_u, N_u) \to \text{Gr}(k_v, N_v)\) defined by the effective connectivity matrix \(W_{uv}\) as \(\phi_e(S) = \text{span}(W_{uv} S)\)

A high-level causal model H (e.g., a three-variable model of stimulus → accumulation → choice) is faithfully implemented by a G-SCM if there exists a transport-respecting alignment — an assignment of abstract variables to causal subspaces such that the transport maps of H are respected by the circuit's connectivity-induced maps.

**Critical property (non-vacuity):** Transport-respecting alignments are non-vacuous. To align to the wrong algorithm, the alignment must map to a subspace that the circuit's actual connectivity cannot reach from the upstream site. This blocks the vacuity problem that afflicts unconstrained alignment approaches.

### 3.3 Cross-Population Mechanism Identity

Two neural populations \(P_1\) and \(P_2\) implement the same mechanism for variable V if and only if there exists a transport-respecting alignment \(\phi\) such that \(\phi(S_1) = S_2\) and \(\phi\) commutes with the respective circuit transport maps.

In practice, without access to true connectivity W, this is estimated via three evidence domains:

1. **Activation-space:** Grassmannian distance from DAS-recovered causal subspaces
2. **Connectivity-space:** Transport map consistency from estimated effective connectivity
3. **Dynamics-space:** Developmental trajectory of the subspace over learning or trial time

The No-Free-Lunch Triangulation principle holds: none of these three alone is sufficient, but together they are injective on the space of natural neural mechanisms.

***

## 4. Three Computable Invariants

### 4.1 Gauge-Normalized Grassmannian Distance

Neural population recordings are subject to gauge symmetries: permutations of neuron labels across sessions, rotations of the latent space within a recording, and overall scaling. Two subspace estimates from different animals may appear different on the raw Grassmannian while being gauge-equivalent.

We define gauge normalization by projecting to the orbit space of the relevant symmetry group: (a) arbitrary rotation within the causal subspace, (b) permutation of neurons across animals (handled by fitting subspaces from trial-averaged responses to matched stimuli), and (c) overall amplitude scaling (normalized by subspace effective rank).

After gauge normalization, the Grassmannian distance \(d^G(S_1, S_2)\) provides a principled measure of geometric dissimilarity not confounded by representation-irrelevant symmetries. **The key prediction:** same-mechanism populations will have significantly smaller \(d^G\) than different-mechanism populations, even when CKA is similar.

### 4.2 Holonomy Fingerprints

When a causal subspace is parallel-transported around a closed loop — a sequence of circuit activation states that returns to the starting condition — the result may differ from the original subspace. This discrepancy is holonomy. Holonomy is automatically gauge-invariant and provides a fingerprint characterizing the curvature of the subspace's embedding.

For neural circuits, holonomy can be estimated by measuring the subspace at baseline, after a stimulus sequence, and at return to baseline. Matching holonomy groups across animals is a stronger criterion for mechanism identity than matching subspace geometry, because holonomy reflects the circuit's global transport structure, not just its instantaneous geometry.

**Testable prediction:** populations implementing the same mechanism (e.g., attractor-based choice computation) will have isomorphic holonomy groups. Populations implementing different mechanisms (ramp vs. attractor) will have structurally distinct holonomy groups even when encoding the same variable.

### 4.3 Sheaf-Cohomological Localizability

The classical question of whether a neural mechanism is "localized" to one region or "distributed" is currently answered descriptively — by reporting which regions show above-threshold decoding accuracy. This has no principled stopping rule.

Sheaf cohomology formalizes this. Define a circuit sheaf F over the brain's region graph: to each region r, assign the causal subspace F(r) recovered by DAS or iSSM; to each anatomical connection (r, s), assign the restriction map from the effective connectivity matrix. The zeroth cohomology group H⁰(F) is nonzero if and only if a globally consistent single-region localization exists. The first cohomology group H¹(F) measures the obstruction to global consistency: if H¹(F) ≠ 0, the mechanism is provably distributed — no single-site localization can be correct, regardless of which region is chosen.

This converts "is this mechanism distributed?" from an empirical question into a computable topological invariant.

***

## 5. Datasets, Access, and Experimental Plan

### 5.1 IBL Brain-Wide Map

**Dataset:** 621,733 neurons across 139 mice, 12 laboratories, 279 brain areas during a standardized visual decision-making task. The task has sensory (visual contrast), motor (wheel movement), and cognitive (prior probability) components, with known neural correlates across frontal, motor, and subcortical regions.

**Access:** ONE Python API — `pip install ONE-api`, data at `https://openalyx.internationalbrainlab.org`. Full documentation and tutorial notebooks available at the IBL code library.

**Why IBL is ideal:** IBL was designed to test cross-lab and cross-animal reproducibility — precisely your mechanism identity question. The standardized task means behavioral demands are matched; the multi-lab design means we can ask whether Grassmannian distances are structured by biology vs. methodology.

***

**Experiment 1: Cross-animal choice subspace stability.**
For each region with >20 recorded neurons per session, fit a causal subspace for the choice variable using linear DAS (rotation matrix optimization). Compute pairwise Grassmannian distances across animals and labs. Test whether within-animal cross-session distance < within-lab cross-animal distance < across-lab distance. This establishes the biological vs. methodological noise floor.

*Prediction:* \(d^G\)(within-region, same animal, different session) < \(d^G\)(within-region, different animal, same lab) < \(d^G\)(within-region, different lab).

*Falsification:* Cross-lab distance ≈ cross-animal distance, implying methodology drives subspace variation more than biology.

***

**Experiment 2: CKA vs. Grassmannian distance dissociation.**
For pairs of populations with matched choice-decoding accuracy (behavioral equivalence), compare their linear CKA and gauge-normalized Grassmannian distance. The framework predicts high-CKA, high-Grassmannian-distance pairs exist — populations that encode the same variable with similar geometry but via different causal subspaces.

*Prediction:* The correlation between CKA and \(d^G\) is < 1.0 and significantly less than 1.0 for cross-region pairs.

*Falsification:* CKA and \(d^G\) are perfectly correlated across all region-pair types, implying representational similarity and mechanism identity are equivalent.

***

**Experiment 3: Sheaf cohomology of the IBL choice circuit.**
Define the circuit sheaf over the 15–20 regions most reliably encoding choice across animals. Compute H⁰ and H¹ using Čech cohomology over the region graph with estimated effective connectivity as restriction maps. Ask: is the IBL choice circuit topologically localizable?

*Prediction:* H¹ ≠ 0 for the full choice circuit; frontal-motor sub-circuits have H⁰ ≠ 0 (locally localizable).

*Falsification:* H¹ = 0, implying the circuit is fully localizable to a single region.

*Connection to dark matter:* If H¹ ≠ 0, predict that the dark matter ratio (full-model logit difference / circuit logit difference, analogous to the MI dark matter problem) is proportional to the dimension of H¹.

***

### 5.2 Allen Visual Behavior Neuropixels

**Dataset:** 153 sessions, 81 mice, recordings across visual cortex (V1, LM, AL, PM, AM) and subcortical regions (LGd, LP) during a change-detection task.

**Access:** AllenSDK Python package — `pip install allensdk`, data via S3 (`s3://visual-behavior-neuropixels-data/`). Full tutorial: `allenswdb.github.io/physiology/ephys/visual-behavior`.

***

**Experiment 4: Stimulus subspace transportability across visual areas.**
Fit causal subspaces for stimulus identity (grating orientation, contrast) in V1, LM, and AL across sessions and animals. Compare: (a) V1 across animals vs. (b) V1-to-LM within an animal. Is the visual stimulus mechanism more similar within-region-across-animals or within-animal-across-regions?

*Prediction:* Within-V1-cross-animal distance < within-animal V1-to-LM distance (same circuit, different instance is more similar than different circuit, same animal).

*Falsification:* Reverse ordering — circuit is more animal-specific than region-specific.

***

**Experiment 5: Behavioral state modulation of sensory subspaces.**
The VBN dataset includes passive viewing and active change-detection epochs for the same stimuli. Test whether the sensory causal subspace is stable across behavioral states.

*Prediction:* Small Grassmannian distance (< 15 degrees principal angle) between passive and active subspaces — subspace is circuit-determined, not state-dependent.

*Falsification:* Large distance — subspace shifts with behavioral context, consistent with top-down modulation changing the mechanistic implementation.

***

**Experiment 6: Gauge correction effect.**
For cross-session comparisons within animals (same neural population across days), compute raw and gauge-normalized Grassmannian distances. Quantify the variance reduction from gauge normalization.

*Prediction:* Gauge normalization reduces variance in cross-session comparisons by >30%.

*Falsification:* No significant reduction, implying the raw Grassmannian distance is already gauge-invariant in practice.

***

### 5.3 Steinmetz et al. 2019

**Dataset:** ~30,000 neurons, 42 brain regions, 39 sessions, 10 mice performing visual discrimination (2AFC). Multi-region simultaneous recording: all 42 regions recorded from the same animal in the same session, enabling within-animal cross-region comparisons without subject-level confounds.

**Access:** Direct download at `steinmetzlab.net/shared/` or via ONE API interface. NWB format with existing analysis toolkits.

***

**Experiment 7: Multi-region sheaf cohomology of the choice circuit.**
With 42 simultaneously recorded regions, compute the full circuit sheaf for choice encoding. Estimate restriction maps from spike-count cross-correlations at short lags (5–10ms). Compute H⁰ and H¹ over the full 42-region graph and sub-graphs.

*Prediction:* Frontal-motor sub-sheaf is localizable (H⁰ ≠ 0, H¹ = 0); the full 42-region circuit is not (H¹ ≠ 0).

*Falsification:* Full circuit is localizable (H¹ = 0 everywhere).

***

**Experiment 8: Holonomy estimation on choice subspaces.**
Using the trial structure of the 2AFC task, estimate holonomy by comparing the choice subspace at trial onset, mid-deliberation, and post-choice. If the task variable returns to baseline after choice, the deliberation period provides a natural "loop" for holonomy measurement.

*Prediction:* Holonomy is stable within a region across animals (mechanism identity signature) and variable across regions performing the same encoding (mechanistic pluralism signature).

*Falsification:* No significant cross-region differences in holonomy, implying the holonomy fingerprint is not diagnostic.

***

**Experiment 9: Multiple realization test.**
Steinmetz et al. records 42 regions, many encoding choice with similar decoding accuracy. For region pairs with matched decoding accuracy but different anatomical connectivity, test whether Grassmannian distance and holonomy similarity are smaller or larger than predicted by CKA.

*Prediction:* CKA is uniform (behaviorally equivalent), but Grassmannian distance and holonomy discriminate mechanism identity — demonstrating that CKA and mechanism identity are genuinely dissociable.

*Falsification:* CKA and \(d^G\) are equally discriminative, suggesting no empirical difference between representational similarity and mechanism identity.

***

## 6. Theoretical Results

### 6.1 No-Free-Lunch Triangulation for Neural Circuits

**Theorem (NFL Neural):** Define three evidence maps from the space of neural mechanisms to observables:

- \(E_W\): connectivity-space evidence (effective connectivity matrix, weight-induced transport)
- \(E_A\): activation-space evidence (DAS-recovered causal subspace, CKA, RSA)
- \(E_D\): dynamics-space evidence (subspace trajectory over trial time or learning)

For any class of neural mechanisms satisfying a general position condition, each \(E_W\), \(E_A\), \(E_D\) is individually non-injective on the mechanism class, but the joint map \((E_W, E_A, E_D)\) is injective.

**Proof sketch via three defeating pairs:**

*Pair defeating \(E_A\):* An attractor mechanism and a feed-forward ramp both encode the same choice variable with similar DAS-recovered subspaces (behavioral-fit subspace reflects the variable, not the dynamics implementing it). \(E_W\) distinguishes them via recurrent connectivity structure.

*Pair defeating \(E_W\):* Two populations with identical estimated effective connectivity (from cross-correlations) but different detailed microcircuit structure. \(E_A\) distinguishes them by precise causal subspace geometry.

*Pair defeating \(E_D\):* Two mechanisms that form via different developmental trajectories but converge to the same fixed-point subspace. \(E_A\) and \(E_W\) distinguish them; \(E_D\) alone cannot.

**Corollary:** Representational similarity measures alone (CKA, RSA, raw Grassmannian distance) are provably insufficient for mechanism identity. A complete argument requires evidence from at least two of the three domains.

### 6.2 Cohomological Localizability

**Theorem:** A mechanism described by circuit sheaf F admits a consistent single-region localization if and only if H⁰(F) ≠ 0. The mechanism is provably distributed if and only if H¹(F) ≠ 0.

This converts the localization question from "which regions pass a decoding threshold?" (a continuous quantity depending on chosen threshold) into a binary topological invariant that is threshold-independent and method-invariant.

***

## 7. Methodological Comparison

| Method | What it measures | Mechanism-sensitive? | Computable from spikes? |
|--------|-----------------|---------------------|-------------------------|
| CKA | Representational geometry similarity | No | Yes |
| RSA | Pairwise stimulus distance correlation | No | Yes |
| Linear decoding accuracy | Variable encoding fidelity | No | Yes |
| Grassmannian distance (raw) | Subspace angle difference | Partially | Yes |
| Gauge-normalized Grassmannian | Mechanism-relevant subspace difference | Yes (with connectivity) | With connectivity proxy |
| Holonomy | Circuit transport structure | Yes | Requires trial loops |
| Sheaf cohomology H⁰, H¹ | Localizability class | Yes | Requires connectivity estimate |
| iSSM | Causal dynamics under perturbation | Yes | Requires stimulation data |

The predicted **CKA / \(d^G\) dissociation** is the key empirical signature. Two populations implementing the same *behavior* via different *mechanisms* (attractor vs. ramp) will have high CKA but large \(d^G\). The framework predicts such dissociations are common across the IBL dataset because: (a) different brain regions are known to use different dynamical motifs for the same variable, and (b) across animals, circuit variability can produce mechanistically distinct implementations of identical behavioral functions.

***

## 8. Connection to the Mechanistic Views Framework

This paper imports the Mechanistic Views taxonomy into computational neuroscience. The five-view structure maps cleanly onto the neural manifold literature:

| Mechanistic Views dimension | Neural manifold analogue |
|----------------------------|--------------------------|
| Object view (which components) | Which neurons/regions contribute |
| Role view (functional contribution) | What computational role the subspace plays |
| Subspace view (geometry) | The causal subspace on the Grassmannian |
| Structural view (connectivity) | Effective connectivity matrix inducing transport |
| Process view (temporal dynamics) | Trial-time trajectory of the subspace |

A key claim from Mechanistic Views: these dimensions are not automatically aligned. Evidence for object-view identity (same neurons) does not imply structural-view identity (same connectivity-induced transport). The empirical prediction: region pairs with matched object-view profiles (similar neuron types, similar stimulus selectivity) will not always have matched structural-view profiles (similar causal subspace geometry), confirming the need for the full taxonomy.

The multiple realization test (Experiment 9) directly addresses a longstanding debate in philosophy of neuroscience: if different realizers of the same behavioral function implement the same causal subspace, the subspace is the correct level of mechanistic description. If they implement different causal subspaces, mechanisms are genuinely multiply realizable at the subspace level — implying that no single level of description is canonical.

***

## 9. Related Work

### Neural Manifold Stability

Recent work shows neural manifold dimensionality is stable across motor perturbations despite behavioral changes. The "method of analogous cycles" framework tracks topological features of manifolds across populations using dissimilarity matrices — topological rather than geometric, complementary to our sheaf cohomology approach. Multiplexed subspace networks show that different dimensions of population activity in the same region connect functionally to different cortex-wide networks, consistent with our subspace-level mechanism claims.

### Causal Neural Models

iSSMs provide a causal framework for predicting neural responses to novel perturbations. Our framework is complementary: iSSMs ask "what causal model of dynamics fits the data," while our framework asks "which causal subspace within that model constitutes the mechanism." iSSM identifiability results inform our Experiments 7–9.

### Cross-Population Alignment

Procrustes alignment and optimal transport methods align neural populations at the level of individual neurons or latent dimensions. Our Grassmannian approach operates at the subspace level — more invariant to within-subspace rotations and more directly connected to causal claims about which dimensions drive behavior.

### Population Subspace Decomposition

Population subspace analyses have demonstrated orthogonal encoding of multiple variables in motor cortex. Neural population geometry for optimal coding of tasks with shared latent structure and broad surveys of neural population geometry provide the empirical context for the invariants introduced here.

***

## 10. Significance

### For Neuroscience

This paper provides a principled, computable framework for the question "do these two populations compute the same thing?" — currently answered by CKA and linear decoding, which do not distinguish mechanism from behavior. The sheaf cohomology criterion makes the localization question into a topological computation rather than a threshold-dependent empirical judgment. The holonomy fingerprint provides a global circuit invariant not confounded by gauge symmetries.

**Practical workflow:** (1) fit causal subspaces per session using DAS or iSSM, (2) compute gauge-normalized Grassmannian distances, (3) for region comparisons with close geometric distances, check holonomy consistency, (4) for distributed mechanisms, compute the sheaf cohomology class to determine whether distribution is principled or an artifact of incomplete localization.

### For Mechanistic Interpretability

The neural setting generalizes the transformer case: transformers have known weights (exact connectivity), neural circuits have estimated effective connectivity. The framework developed here for neural circuits extends the Mechanism Geometry program to settings where connectivity is partial and estimated from data — relevant for large production models where weight access is limited.

### For Philosophy of Science

The multiple realization test (Experiment 9) makes the mechanistic pluralism argument from Mechanistic Views empirically testable in neuroscience. The result would have direct implications for whether mechanistic explanation survives multiple realizability in neural systems — a longstanding philosophical debate now made tractable with public data.

***

## 11. Five-Year Research Agenda

**Year 1 (this paper):** Establish the three-invariant framework, validate on IBL/Allen/Steinmetz with nine experiments, demonstrate CKA/Grassmannian dissociation empirically.

**Year 2:** Apply iSSM to datasets with photostimulation (ALM + opto, macaque dlPFC + microstimulation) for interventional evidence — moving from purely observational causal subspace estimates to perturbation-validated ones. This would confirm or disconfirm the transport-respecting alignment predictions using actual causal interventions.

**Year 3:** Cross-species comparison. If the holonomy fingerprint and sheaf cohomology class are genuinely mechanism-level invariants, decision-making circuits in mice, rats, and primates should share cohomology classes despite anatomical differences. This is a strong test of the framework's generality.

**Year 4:** Scale to large-scale connectome data (C. elegans connectome is fully known; Drosophila connectome is increasingly complete). True connectivity W is available, allowing exact G-SCM transport map computation rather than estimation from cross-correlations.

**Year 5:** Formal theory — proving that holonomy group isomorphism is a necessary and sufficient condition for mechanism identity under the general position condition. The current paper states this as a conjecture; the five-year target is a proof, potentially with AI-assisted formalization in Lean 4.

***

## 12. Conclusion

The question of mechanism identity across neural populations is neither purely empirical nor purely philosophical — it is geometric. Two populations implement the same mechanism when their causal subspaces are related by a transport-respecting alignment that respects actual circuit connectivity. Three computable invariants — gauge-normalized Grassmannian distance, holonomy fingerprint, and sheaf cohomology class — operationalize this definition. Nine experiments across three fully open datasets test the framework's core predictions.

The empirical stakes are concrete. If cross-animal Grassmannian distances are smaller than cross-region distances for the same variable (Experiments 1, 4), the field has a justified operationalization of "same mechanism across animals" for the first time. If the IBL choice circuit has nonzero H¹ (Experiment 3), the localization debate has a topological answer immune to threshold choice. If CKA and \(d^G\) dissociate for matched-accuracy region pairs (Experiments 2, 9), the distinction between mechanism identity and representational similarity becomes empirically demonstrated rather than merely conceptually motivated.

***

## References

*(Core references — full bibliography to be compiled)*

- IBL Brain-Wide Map, *Nature* 2025
- IBL documentation and API
- Allen Visual Behavior Neuropixels
- Steinmetz et al. 2019, *Nature*
- Geiger et al. 2024, DAS, CLeaR
- Interventional SSM, ICML 2025
- Kornblith et al. 2019, CKA
- Neural population geometry review
- Manifold stability under perturbation, *bioRxiv* 2026
- Multiplexed subspaces
- Population subspaces for motor cortex
- Method of analogous cycles
- Neural population geometry and optimal coding
- Choice dynamics in premotor cortex
- Multiple realizability and mechanistic explanation
- Nonparametric identifiability
- Tower 2026a, Causal Geometry / Mechanism Geometry
- Tower 2026b, Mechanistic Views

---

## References

1. [Neural population geometry: An approach for understanding ... - PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10695674/) - We review examples of geometrical approaches providing insight into the function of biological and a...

