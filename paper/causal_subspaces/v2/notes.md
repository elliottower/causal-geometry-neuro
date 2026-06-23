# Causal Subspaces v2 — Notes

## Peer Review Issues to Address

### CRITICAL: IIA vacuity logic gap
- VAE achieves IIA=0.70 on random noise, then claim external validation "rescues" the claim
- Shuffled-label control (Exp62) is MORE important than optogenetic validation for closing the gap
- Without Exp62: "VAE fits noise AND real data equally well"
- Three pending experiments (Exp62/63/65/66/70) cited as supporting evidence — structural problem

### CRITICAL: n=12 optogenetic validation too small
- VAE individually: rho=+0.33, p=0.30 (NOT significant)
- Delta-rho=+1.06, CI [0.31, 1.51] is valid but fragile at n=12
- Hierarchical matching n=16 only reaches p=0.11 for VAE alone
- Honest framing: "LDA anti-correlation significant; VAE shows positive trend; contrast is significant"
- FIX: expand matching via Allen CCF more aggressively (parent/sibling/proximity), target n=25+
- ADD: power analysis showing n needed for VAE alone to reach p<0.05

### CRITICAL: SAE "outperforms" on vacuous metric
- Table 5 SAE IIA=0.962 vs VAE 0.939 — but you just argued IIA is vacuous for nonlinear methods
- Logical contradiction: ablation ranking uses IIA among methods all shown to be vacuous
- FIX OPTIONS: (1) run SAE optogenetic validation, (2) caveat IIA as relative ordering only, (3) reframe around classification loss not IIA

### MODERATE: pi-VAE identifiability conditions
- Binary choice (2 values) barely meets k=1, m_k+1=2 requirement — state this explicitly
- Missing: NeurIPS 2024 "Disentangling Interpretable Factors with Supervised methods"

### MODERATE: Makelov "Interpretability Illusion" relevant
- Makelov et al. 2023 + reply (arXiv:2401.12631): DAS can achieve high IIA for trivial reasons
- You're using DAS as "honest" baseline — need to explain why (provably rejects noise)
- Triangle: Sutter (nonlinear vacuous) + Makelov (even linear manipulable) + your DAS baseline

### MODERATE: Pereira et al. 2025 needs more discussion
- Closest prior work: residual dimensions (not main PCA) causally drive choice via transient amplification in ALM
- Direct challenge: if residual dimensions are causal, why does structured VAE find the right subspace?

### MINOR: Missing technique citations
- Bootstrap BCa: cite Efron & Tibshirani
- Allen CCF ontology: cite Wang et al. 2020 or original 2011
- Wilcoxon usage: one sentence on why nonparametric (non-normal IIA distributions)

### HONEST ASSESSMENT
- Actually the more original paper — nobody has validated nonlinear subspaces against optogenetic ground truth
- LDA anti-correlation with causal importance (rho=-0.73, p=0.01) is striking and clean
- IIA vacuity extending to structured VAEs is novel empirical contribution
- Estimated odds: ~45% NeurIPS if Exp62 runs + optogenetic n expands, ~20% as-is

### PRIORITY TODO for v2
- [ ] Run Exp62 (shuffled-label control) — HIGHEST PRIORITY, closes logic gap
- [ ] Expand optogenetic matching to n=25+ via Allen CCF
- [ ] Run SAE optogenetic validation (one extra row in Table 3)
- [ ] Run Exp63 (linear VAE ablation), Exp66 (per-mouse optogenetic)
- [ ] Reframe SAE ablation around classification loss not IIA
- [ ] Add Makelov citation and DAS baseline justification
- [ ] Add missing citations (BCa, Allen CCF, pi-VAE conditions)
- [ ] Add figures
- [ ] Make self-contained (remove "[Paper A]" refs)
