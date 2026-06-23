# Paper TODO

Everything from the Perplexity peer review, organized by priority.

---

## Experiments to Run

### HIGHEST PRIORITY
- [x] **Direction-vs-subspace experiment (exp10)** — DONE. Directions indistinguishable from random (Mann-Whitney U p~1.0), confirming subspaces are needed.
- [x] **Exp62: shuffled-label control** — DONE. Real IIA=0.625 vs shuffled=0.430, Wilcoxon p=6e-8. All 24/24 regions real > shuffled. Closes IIA vacuity gap.
- [x] **Debiased CKA rerun (exp74)** — DONE. Biased rho=0.850, debiased rho=0.445, partial rho (ctrl n_neurons)=0.342. Debiasing WEAKENS CKA-Procrustes correlation by delta=-0.40. Anti-correlation with Grassmannian is STRENGTHENED. Murphy et al. concern is real and substantive.

### HIGH PRIORITY
- [ ] **Expand optogenetic matching to n=25+ via coordinate matching (exp75)** — RUNNING on Modal. Uses spatial nearest-neighbor in stereotaxic (ML, AP) coordinates instead of string matching. Multiple distance cutoffs (0.5-2.0mm).
- [ ] **SAE optogenetic validation (exp79)** — STUB. Needs exp75 results + GPU. Deferred.
- [x] **Exp63: linear VAE ablation** — DONE. Nonlinearity effect is near-zero (mean diff=0.0008, p=0.65). Nonlinear VAE provides virtually no benefit over linear — supports "IIA is vacuous for nonlinear methods."
- [x] **Exp66: per-mouse optogenetic** — DONE (ran on Modal, results on volume).

### MODERATE PRIORITY
- [x] **Exp65: temporal sliding window IIA** — DONE. 100% of regions show pre-response emergence for both LDA and VAE. Mean onset bin=35.9.
- [ ] **Exp70: cross-region activation patching** — cited in discussion, not yet run.
- [ ] **UMAP stochasticity analysis (exp76)** — RUNNING on Modal. 20 seeds x 9 hyperparameter configs.
- [x] **Alpha bias robustness (exp77)** — DONE. alpha vs n_neurons: rho=-0.916 (p=1e-113). Strong confound! Must control for n_neurons in all alpha-based claims. alpha vs n_trials: rho=0.076 (n.s.).
- [x] **Power analysis for optogenetic (exp78)** — DONE. LDA (rho=-0.73): 80% power at n=14 (well-powered at n=12). VAE (rho=+0.33): never reaches 80% power even at n=60. Need n>60 or stronger signal (i.e., exp75 coordinate matching).

---

## Framing Changes

### Paper A (metric_dissociation)
- [ ] **Reframe around Harvey 2024** — currently positioned as discovery, but Harvey et al. (arXiv:2411.08197) proves the CKA-Procrustes relationship mathematically. Reframe as: "first large-scale empirical validation of Harvey's theorem in biological neural data." Quote the theorem briefly.
- [ ] **Novel contributions to emphasize**: (1) first large-scale empirical test across 73 regions, (2) dimensionality mediation mechanism (partial r=+0.44), (3) IBL replication (139 mice, 12 labs), (4) potent/null decomposition (rho=-0.86).

### Paper B (causal_subspaces)
- [ ] **Reframe SAE ablation** — currently uses IIA to rank methods you just argued are all vacuous. Either validate SAE against optogenetic, caveat IIA as relative ordering only, or reframe around classification loss.
- [ ] **Soften optogenetic "rescue" language** — honest framing: "LDA anti-correlation significant; VAE shows positive trend; contrast is significant." Not "rescuing."
- [ ] **State pi-VAE identifiability conditions explicitly** — binary choice (2 values) barely meets k=1, m_k+1=2 requirement.
- [ ] **Expand Pereira et al. 2025 discussion** — closest prior work, deserves more than one sentence. Residual dimensions causally driving choice is a direct challenge to reconcile.
- [ ] **Add DAS baseline justification** — cite Makelov et al. 2023 + reply (arXiv:2401.12631). Explain triangle: Sutter (nonlinear vacuous) + Makelov (linear manipulable) + why linear DAS still provably rejects noise.

### Cross-cutting
- [ ] **Decide: split or combine** — combined ~55% odds vs split ~35%/~45%. Combined eliminates circular cross-refs, tells one story.
- [ ] **Make each paper self-contained** — if splitting, remove "[Paper A]"/[Paper B]" forward refs, replace with brief inline descriptions. Reviewer getting one paper can't see the other.

---

## Missing Citations to Add

### Paper A
- [ ] Murphy, Zylberberg & Fyshe 2024 (arXiv:2405.01012) — biased CKA correction. MUST cite.
- [ ] Chun et al. 2025 (arXiv:2502.15104) — feature-sampling biases in CKA depend on geometry.
- [ ] Chun, Canatar, Chung & Lee 2025 (arXiv:2509.26560) — dimensionality estimation bias from finite samples.
- [ ] Sadtler et al. 2014 — manifold constraints on behavior. In Paper B discussion but missing from Paper A related work.
- [ ] "Decoding Alignment Without Encoding Alignment" 2024 preprint — structural critique of RSA/CKA/Procrustes literature.
- [ ] Optimal transport comparison (arXiv:2412.14421) — acknowledge static vs temporal limitation.
- [ ] Stringer et al. 2019/2025 PNAS — 1/n power law optimal for population coding, contextualizes alpha.
- [ ] "Geometry and Dimensionality of Brain-wide Activity" 2025 eLife Reviews — covers similar terrain, must differentiate.

### Paper B
- [ ] Makelov et al. 2023 + reply (arXiv:2401.12631) — DAS interpretability illusion.
- [ ] NeurIPS 2024 "Disentangling Interpretable Factors with Supervised methods" — structured VAEs for disentanglement.
- [ ] Allen Mouse Brain Atlas (Wang et al. 2020 or original 2011) — for hierarchical region matching.
- [ ] Efron & Tibshirani — bootstrap BCa method.

### Both
- [ ] Harvey et al. 2024 (arXiv:2411.08197) — quote theorem explicitly, not just cite.

---

## Writing / Presentation

- [ ] Add figures to both papers (currently zero wired up)
- [ ] One sentence on why Wilcoxon throughout (non-normal IIA distributions)
- [ ] Describe UMAP procedure (seed, runs, variance)
- [ ] v2 should "actually explain what's going on, have figures, be easy to understand and not just super dense bullshit"

---

## Strategic Decisions

### DECISION: Do the combined paper
Perplexity verdict: combined is the stronger play (~55% vs ~35%/~45% split).
- Paper A alone = "empirical confirmation of Harvey 2024's theorem" — fine for TMLR but weak standalone
- Paper B alone has optogenetic validation (genuinely novel) but its "why nonlinear methods?" motivation leans on Paper A results the reader doesn't have
- Combined story is airtight: metrics disagree because of dimensionality → linear causal methods fail for same reason → optogenetic proof + better method
- Also fixes the pending experiments problem — TMLR has no hard deadline, submit when ready
- Combined structure: (I) metric dissociation + dimensionality mediation, (II) causal subspace failure + VAE fix, (III) validation + ablation. ~20 pages, within TMLR limits (12pp for 2-week review, or go over for 4-week — either fine)

### VENUE: TMLR
- Scope explicitly includes "computational models of natural learning systems at the behavioral or neural level"
- Acceptance criteria: "are the claims sound and would the audience be interested?" — no novelty gatekeeping
- Best for this work because: strongest on methodological rigor (IIA vacuity, optogenetic validation), weakest on novelty (Paper A partially confirming Harvey). TMLR rewards the first, doesn't penalize the second.
- J2C track: TMLR papers get invited to present at NeurIPS/ICML/ICLR if "Featured" cert. Journal permanence + revision rights + potentially conference presentation.
- Median turnaround ~91 days. Acceptance rate ~46-62%. Can revise after reviews instead of flat reject.
- For independent researcher without institutional backing, best risk-adjusted path.
- Submit with arXiv preprint simultaneously (TMLR allows). Timestamp before anyone scoops optogenetic validation.

### Execution order (from Perplexity)
1. Run Exp62 (shuffled-label control, CPU, ~2h) — single highest-leverage experiment
2. Fix debiased CKA — code change, rerun. Must know if result holds before writing anything
3. Run direction-vs-subspace (CPU, fast) — becomes Figure 1
4. Outline combined paper as one document with three sections
5. Submit to TMLR + arXiv simultaneously

---

## Data & Resources

### Zatka-Haas 2021 optogenetic data — FULLY PUBLIC
- Paper: elifesciences.org/articles/63163 (figures and data tab has direct downloads)
- 52 cortical coordinates, 47,002 laser trials across 10 mice
- Widefield calcium imaging + optogenetic inactivation + behavioral data + code all downloadable
- No need to email anyone for the data itself

### Steinmetz 2019 data
- steinmetzlab.net/shared + Figshare (already have this)

### The n=12 → n=30+ fix is a CODE problem, not a data problem
- Current matching does string matching on region names → only 12 direct matches
- Should do spatial nearest-neighbor matching in Allen CCF coordinate space
- Each Steinmetz Neuropixels recording has an Allen CCF coordinate
- Each Zatka-Haas optogenetic site has a widefield coordinate
- Nearest-neighbor matching with distance cutoff in CCF space should yield n=30+
- This is the single highest-leverage code fix for Paper B's main vulnerability
- CPU only, ~2h to implement and rerun

### Contacts (if needed for warm intros, not for data)
- Peter Zatka-Haas (first author 2021) — check current affiliation via Google Scholar
- Nick Steinmetz at UW Seattle (steinmetzlab.net) — lab actively engages with data users
- Jazayeri lab at MIT BCS — does geometric analysis of neural population dynamics, direct match for this work, highest-yield cold email in Boston
