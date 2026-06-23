# Metric Dissociation v2 — Notes

## Peer Review Issues to Address

### CRITICAL: Biased CKA
- Murphy et al. 2024 (arXiv:2405.01012): biased CKA hits max on random data in low-trial/high-neuron regime — exactly our setting (250 trials, 15-40 neurons, wildly different neuron counts across 73 regions)
- Chun et al. 2025 (arXiv:2502.15104): feature-sampling biases in CKA depend on representation geometry
- FIX: Switch to debiased CKA (Kornblith Appendix or Murphy correction), OR add control showing CKA not dominated by sample-feature ratio

### CRITICAL: Power-law alpha has finite-sample bias
- Chun et al. 2025 (arXiv:2509.26560): participation ratio and power-law fits to finite PCA spectra are highly biased with small samples
- Fitting power law to ranks 10-50 of spectrum with only 15-40 ranks total in smallest regions
- FIX: bias-corrected dimensionality estimator, OR robustness table matching neuron counts, OR standalone partial correlation controlling for n_neurons separately

### MODERATE: CKA-Procrustes anti-correlation predicted by Harvey 2024
- Harvey et al. 2024 (arXiv:2411.08197) PROVES this mathematically: Procrustes upper-bounds decoding distance, CKA ~ average linear readout alignment, equivalent only at low participation ratio
- Current framing oversells as "discovery" — actually an empirical verification of a known theorem
- REFRAME: "first large-scale empirical validation of Harvey's theorem in biological neural data"
- Quote Harvey's theorem briefly — makes our result look like testing a precise prediction (paradoxically stronger)

### MODERATE: UMAP stochasticity unaddressed
- No description of: fixed seed? Multiple runs averaged? Variance?
- Robustness table covers neuron subsampling and mouse leave-one-out but NOT UMAP hyperparameter sensitivity

### MODERATE: Missing citations
- Sadtler et al. 2014 — definitive experimental test of manifold constraints, directly relevant
- "Decoding Alignment Without Encoding Alignment" 2024 preprint — structural critique of entire RSA/CKA/Procrustes literature
- Optimal transport comparison (arXiv:2412.14421) — acknowledge static vs temporal limitation
- Stringer et al. 2019/2025 PNAS — 1/n power law is optimal for population coding, contextualizes alpha

### HONEST ASSESSMENT
- Core finding is real but novelty window closing (Harvey 2024 proves it theoretically)
- With debiased CKA + Harvey reframing, this is a clean NeurIPS paper
- Novel contributions: (1) first large-scale empirical test, (2) dimensionality mediation mechanism (partial r=+0.44), (3) IBL replication, (4) potent/null decomposition (rho=-0.86)
- Estimated odds: ~35% NeurIPS with fixes, ~10% without

### TODO for v2
- [ ] Implement debiased CKA (code change + rerun)
- [ ] Reframe contribution around Harvey 2024
- [ ] UMAP stochasticity analysis (multiple seeds, hyperparameter sweep)
- [ ] Add missing citations
- [ ] Add figures (currently none wired up)
- [ ] Make self-contained (remove "[Paper B]" forward refs or make brief inline descriptions)
