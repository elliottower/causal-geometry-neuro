# Novelty Argument and Key Citations

## The Allen Atlas Validation -- Most Important Citation

The Allen Mouse Brain Connectivity Atlas is a publicly downloadable mesoscale connectome with AAV-traced axonal projection densities between every region pair in the mouse brain, registered to a common 3D space (CCF). It's queryable via the AllenSDK with a few lines of Python. The validation experiment: for every directed edge (region A -> region B) in the cross-region IIA matrix, look up the Allen Atlas projection density from A to B. Test whether high-IIA edges (geometry predicts A causally influences B) correspond to high anatomical projection density in the Atlas. Null model: shuffle the region labels on the IIA matrix. If Spearman rho between IIA and Allen projection density beats the shuffle 95th percentile, you have a genuine prediction of anatomy from geometry. This is exactly the kind of "predict held-out ground truth" validation that Neuron reviewers look for.

A December 2025 biorXiv paper on frontal motor circuits is the most directly relevant precedent -- it used cortex-wide optogenetic perturbations to show a specific frontal circuit (what maps onto MOs region) was causally required for abstract economic decisions independently of sensorimotor contingencies. This is the paper to cite when saying "our geometry-derived causal graph predicts the circuit identified by direct optogenetic perturbation."

### AllenSDK References
- Mouse Connectivity AllenSDK docs: https://allensdk.readthedocs.io/en/latest/connectivity.html
- Example notebook: https://alleninstitute.github.io/AllenSDK/_static/examples/nb/mouse_connectivity.html
- Oh et al. 2014 "A mesoscale connectome of the mouse brain" (Nature)
- Neuroinformatics of the Allen Mouse Brain Connectivity Atlas (PubMed 25536338)

## The Novelty Statement

Prior work has compared neural representations across animals using CKA (Kornblith 2019), RSA (Kriegeskorte 2008), and Procrustes alignment (Deitch 2021), but always with a single metric -- implicitly assuming that all metrics agree. Williams et al. (2021) proved that many similarity measures are special cases of a generalized shape metric, but exclusively within the linear/kernel family. The theoretical prediction that dimensionality mediates metric informativeness was stated by Harvey et al. (2024) but never tested empirically across a brain-wide hierarchy. Effective dimensionality was shown to scale unboundedly with neuron number (Stringer 2024) and to vary along the cortical hierarchy (Huntenburg 2024), but no prior work asked whether this variation predicts which metric family is appropriate. Causal abstraction (DAS, Geiger 2024) has been applied exclusively to artificial neural networks; we provide the first implementation on biological electrophysiology data using naturally-occurring experimental interventions as interchange interventions. The February 2026 paper on representation geometry and generalization showed that effective dimension predicts neural network performance across 52 architectures -- we show the same relationship holds for biological circuits, where dimensionality predicts not performance but *which causal claims are identifiable*.

## The Cross-Dataset Generalization Citation

The strongest empirical novelty move: train the geometric type classifier on Steinmetz (10 mice), predict region types in IBL (139 mice, 12 labs), report Spearman rho between predicted and observed dimensionality regime. The IBL Brain-Wide Map paper is the cite; it covers 241 regions with enough sessions per region for reliable estimation. The February 2026 geometry-generalization paper is the DNN analog -- they replicate across ImageNet and CIFAR-10. That's the methodological precedent for why cross-dataset replication counts as a discovery claim.

### References
- IBL 2025 Brain-Wide Map: https://docs.internationalbrainlab.org/notebooks_external/2025_data_release_brainwidemap.html
- Feb 2026 geometry paper: https://huggingface.co/papers/2602.00130

## The Optogenetic Silencing Validation -- Strongest Causal Claim

Steinmetz et al. 2019 included optogenetic silencing of cortical regions via light-gated inhibition (halorhodopsins) on a subset of trials -- this is in the public dataset. The prediction: regions the IIA matrix identifies as causally downstream of evidence encoding should show altered choice subspace geometry when their putative upstream region is silenced. Regions not in the causal path should be unaffected. This uses existing data already in the dataset and produces the claim "subspace geometry predicts which regions are causally necessary for choice, validated against optogenetic silencing." That single sentence is a Neuron abstract opener.

### References
- Principles of designing interpretable optogenetic behavior (PMC4371169)
- Gauld et al. 2024 "A latent pool of neurons silenced by sensory-evoked inhibition" (UCL Discovery 10192307)
- Frontal motor circuits paper: https://www.biorxiv.org/content/10.64898/2025.12.11.693624v1

## The Geometric Deep Learning Framing

Bronstein et al.'s Geometric Deep Learning blueprint gives the principled framing for why Grassmannian geometry is the right tool: different symmetry groups imply different invariants, and the Grassmannian is the natural space for subspaces that are invariant to within-subspace rotation (gauge freedom). Citing this connects the paper to the geometric ML community, not just systems neuroscience -- which matters for TMLR.

### Reference
- Bronstein et al. 2021 "Geometric Deep Learning: Grids, Groups, Graphs, Geodesics" arXiv:2104.13478

## The One Paper Most Directly Competing With

The February 2026 paper on representation geometry predicting neural network performance -- effective dimension, partial rho = 0.75 across 52 models -- is the closest thing in AI. Our result is structurally identical (effective dimensionality predicts which metric is informative), but in biological circuits rather than pretrained models, and with a causal interpretation (DAS/IIA) rather than just a correlation. Cite it as showing the principle extends from AI to biology, and be explicit that our contribution is (a) the biological validation and (b) the causal, not just predictive, interpretation. That framing positions us as extending a principle rather than competing with a result.

### Reference
- "On the Relationship Between Representation Geometry and Generalization in Deep Neural Networks" https://huggingface.co/papers/2602.00130
