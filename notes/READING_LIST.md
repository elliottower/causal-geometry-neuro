# Reading List — Prioritized

## MUST READ (core/) — Read these before writing anything

These are the papers this project directly builds on or responds to. If you
haven't read these, you won't understand the claims.

### Datasets (know what we're analyzing)
1. **IBL Brain-Wide Map** — International Brain Lab, Nature 2025
   - 621K neurons, 139 mice, 12 labs. THE dataset for cross-animal reproducibility.
   - Paper: https://doi.org/10.1038/s41586-023-06031-6
   - Data: https://docs.internationalbrainlab.org

2. **Steinmetz et al. 2019** — "Distributed coding of choice, action and engagement across the mouse brain", Nature
   - 30K neurons, 42 regions simultaneously. Our sheaf cohomology testbed.
   - Paper: https://doi.org/10.1038/s41586-019-1787-x
   - Data: https://osf.io/agvxh/

3. **Allen Visual Behavior Neuropixels** — Allen Institute
   - 153 sessions, 81 mice, V1/LM/AL/PM/AM + subcortical.
   - Portal: https://portal.brain-map.org/circuits-behavior/visual-behavior-neuropixels

### Representational geometry (the baseline we're improving on)
4. **Kornblith et al. 2019** — "Similarity of Neural Network Representations Revisited", ICML
   - Defines CKA. This is what everyone uses. We argue it's insufficient.
   - Paper: https://arxiv.org/abs/1905.00414

5. **Chung & Abbott 2021** — "Neural population geometry: An approach for understanding biological and artificial neural networks"
   - The review paper for neural population geometry. Sets the vocabulary.
   - Paper: https://doi.org/10.1016/j.conb.2021.10.005

### Causal subspaces (the framework we're extending)
6. **Geiger et al. 2024** — "Finding Alignments Between Interpretable Causal Variables and Distributed Neural Representations", CLeaR
   - Defines DAS (Distributed Alignment Search). The rotation optimization we adapt for neural data.
   - Paper: https://arxiv.org/abs/2305.08809

7. **Geiger et al. 2024** — "Causal Abstraction: A Theoretical Foundation for Mechanistic Interpretability"
   - The formal theory behind causal subspaces. Transport-respecting alignment is from here.
   - Paper: https://arxiv.org/abs/2301.04709

### Mechanism identity (what we're formalizing)
8. **Tower 2026a** — "Causal Geometry / Mechanism Geometry" (your own work)
9. **Tower 2026b** — "Mechanistic Views" (your own work)

---

## SHOULD READ (extended/) — Important context, read selectively

### Neural manifold dynamics
10. **Sadtler et al. 2014** — "Neural constraints on learning", Nature
    - Manifold stability under learning. Relevant to our claim that subspaces are circuit-determined.

11. **Gallego et al. 2020** — "Long neural manifolds emerge from motor cortex", Neuron
    - Manifold dimensionality is stable across perturbations. Our gauge normalization builds on this.

12. **Humphries 2021** — "Strong and weak principles of neural dimension reduction"
    - When is low-dimensional structure real vs artifact? Important methodological caution.

### Causal models of neural dynamics
13. **Varley et al. 2025** — "Interventional state-space models for causal discovery"
    - iSSMs: causal models that predict perturbation responses. Our Experiment 7-9 design is informed by their identifiability results.
    - Paper: https://arxiv.org/abs/2311.01445

14. **Safaai et al. 2023** — "Stimulus/choice decoding across cortex"
    - Cross-area decoding comparison in the Steinmetz dataset. Direct predecessor to Exp 9.

### Cross-population alignment
15. **Williams et al. 2021** — "Generalized Shape Metrics on Neural Representations", NeurIPS
    - Procrustes + shape distance. The alignment method we compare against.

16. **Barannikov et al. 2022** — "Representation Topology Divergence"
    - Topological comparison of representations. Complementary to our sheaf approach.

---

## METHODS REFERENCE (methods/) — Look up as needed

### Grassmannian geometry
17. **Edelman et al. 1998** — "The Geometry of Algorithms with Orthogonality Constraints", SIMAX
    - The Grassmannian metric, geodesics, optimization. Math reference for geometry/.

18. **Absil et al. 2004** — "Riemannian Geometry of Grassmann Manifolds"
    - Parallel transport, curvature. Reference for holonomy estimation.

### Sheaf theory
19. **Curry 2014** — "Sheaves, Cosheaves and Applications"
    - Tutorial on cellular sheaves. Reference for sheaf.py.

20. **Hansen & Ghrist 2019** — "Toward a spectral theory of cellular sheaves", J. Appl. & Comp. Topology
    - Sheaf Laplacians and cohomology computation. Reference for our H⁰/H¹ computation.

### Tools
21. **Miolane et al. 2020** — "Geomstats: A Python Package for Riemannian Geometry in Machine Learning"
    - The geomstats library paper. Reference for the geometry-ext optional dep.

22. **Wu et al. 2024** — "pyvene: A Library for Understanding and Improving PyTorch Models via Interventions"
    - The pyvene library. Reference for DAS implementation.

23. **Vieira et al. 2024** — "pynapple: A toolbox for data analysis in neuroscience"
    - The pynapple library. Spike train analysis, manifold projection.
