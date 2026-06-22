# Geometric Structure in Neural Population Codes

[One sentence about who you are and why you did this work. Either: "I'm a [role] interested in [thing]" or "This started as [origin story]."]

[One sentence about what this post is. Either: "This is an informal summary of exploratory analysis on neural population recordings" or "These are findings I wanted to share before they get stale, with code attached."]

## The question

[One sentence framing the question in your own words. Something like: "When hundreds of neurons fire together during a decision, does the *shape* of that activity matter --- or just the firing rates?"]

[One sentence about why existing work doesn't answer this. Either: "Most neural geometry papers describe structure but don't test whether it's causal" or "Geometric methods from ML interpretability haven't been applied to real neural data with proper baselines."]

## The data

We used the Steinmetz et al. (2019) dataset: 39 recording sessions from 10 mice, 73 brain regions, Neuropixels probes. The task is visual discrimination --- which side has higher contrast?

[Optional: one sentence about why this dataset. Either: "It's the largest publicly available multi-region Neuropixels dataset" or "It has enough regions and trials to do geometry properly."]

## What we found

### 1. Linear and nonlinear geometry are anti-correlated

Regions that look similar under linear metrics (CKA) look *different* under nonlinear metrics (Procrustes on UMAP embeddings), with $\rho = -0.85$. This held from 5 sessions to all 39.

[One sentence about what you think this means. Either: "You can't characterize neural geometry with a single metric" or "This was the most surprising finding --- I expected them to agree."]

### 2. Eigenvalue spectra are universal; trial content is not

Power-law eigenvalue decay looks the same across all 73 regions ($r = 0.936$). But the trial-by-trial modes are region-specific ($r = 0.089$). Same scaffolding, different content.

### 3. Choice manifolds are flat

Persistent homology ($\beta_1$) shows no topological loops in 64/73 regions. For a binary choice task, the simplest possible topology is the actual topology. This rules out ring attractor models for this task.

### 4. Every region rotates

Rotational dynamics (jPCA) appear in 72/73 regions, but strength varies 10x. Motor/frontal regions rotate fastest, hippocampal regions slowest.

### 5. Geometry recovers anatomy

$k=3$ clustering on the CKA similarity matrix recovers cortical / thalamic / hippocampal groups --- without using any anatomy labels as input.

[One sentence reaction. Either: "This was the result that convinced me the geometry is real" or "Structure predicts function, even when you don't tell the algorithm about structure."]

### 6. Evidence subspaces causally mediate choice (IIA)

This is the key causal result. We used interchange intervention analysis (IIA): take two trials with opposite evidence, swap their evidence-subspace projections, and check if the choice classifier flips. It does --- and swapping reaction time, feedback, or random subspaces doesn't produce the same effect (specificity ratio 2.3x).

### 7. Four intervention methods agree

We tested projection swap, Gaussian noise injection, mean-shift, and subspace zeroing. All four produce correlated effect profiles across regions ($\bar{\rho} = 0.72$). The effect isn't an artifact of any one method.

## How we validated it

[One sentence about why you think validation matters. Either: "Most geometry papers skip this part" or "I wanted to know if I was fooling myself."]

We applied validity criteria from philosophy of science:

| Criterion | What we tested | Result |
|-----------|---------------|--------|
| **Sufficiency** (Craver 2007) | Evidence subspace alone decodes choice | Recovery = 1.16 (20/24 regions) |
| **Specificity** (Shallice 1988) | Evidence IIA >> random/RT/feedback IIA | Ratio = 2.3x, $p < 0.001$ |
| **Confound control** (Mayo 2018) | Neuron count, firing rate, trial count | No strong confound ($|\rho| < 0.5$) |
| **Multi-method** (causal inference) | 4 intervention types agree | $\bar{\rho} = 0.72$ |
| **Graded response** (pharmacology) | Progressive ablation → progressive degradation | IIA monotonic in 80% of regions |
| **Baseline separation** (Campbell 1959) | Real vs random-matrix baselines | Spectral $\alpha$ passes; IIA mixed |

[One sentence framing this honestly. Either: "5 of 7 pass cleanly, which is more validation than most published work" or "Not everything passes, and I'm reporting the failures too."]

### 8. Novel predictions

**Geometric type predicts decoder choice (exp22):** Regions where linear geometry dominates (high CKA) favor linear decoders (LDA); regions where nonlinear geometry dominates (high Procrustes) favor kNN. Spearman $\rho = -0.509$, $p = 0.00016$ across 50 regions. This is a genuine out-of-sample prediction: knowing a region's geometric type tells you which decoder will work better, without training on any decoding data.

**Causal abstraction (exp33):** Power-law exponent $\alpha$ correlates moderately with evidence-choice subspace routing ($\rho = 0.288$, $p = 0.014$). Regions with stronger low-dimensional structure route evidence to choice more efficiently. Not as strong as exp22 but consistent with the picture.

**Allen Atlas: an honest null (exp45):** We tested whether structural connectivity (from the Allen Mouse Brain Atlas) predicts functional IIA. It doesn't: $\rho = 0.031$, $p = 0.65$. Anatomical wiring does not predict which regions causally mediate evidence. This is worth reporting because it means the functional geometry we found is not just recapitulating known anatomy.

[One sentence about what the null result means to you. Either: "I was hoping this would work but I'm glad it didn't --- it means the geometry is telling us something new" or "The null is as important as the positives."]

## Limitations

- One dataset (Steinmetz 2019)
- Binary task only --- unclear if this generalizes
- IIA effect sizes are modest (~10% flip rate)
- Double dissociation (evidence vs RT subspaces) is partial
- All analysis is post-hoc, no preregistration
- [Anything else you want to flag]

## What's next

- IBL cross-dataset replication (exp46, relaunched --- pending results)
- Zatka-Haas optogenetic validation with real silencing data (exp47b, data downloaded)
- Multi-task generalization (beyond binary choice)
- [Anything you'd do with more time]

## Intellectual roots

[One or two sentences about how you came to this work. Something grounding it in real intellectual connections rather than sounding like it came from nowhere.]

This work draws on ideas from three specific research traditions that I've encountered through [advisors / collaborators / reading]:

- **Structured representation learning** (Siddharth Mishra-Sharma) --- specifically the structured disentanglement framework (Esmaeili et al., AISTATS 2019) where factors aren't fully independent but have structured overlap, StrAE (Opper et al., EMNLP 2023) using explicit graph structure to constrain latent spaces, and Banyan (ICML 2025) on explicit structural inductive biases improving identifiability. Our dimensionality-based metric taxonomy is exactly this kind of structural prior. His work on RL-derived vs. biological spatial representations (Vision Research 2020) is the direct bridge between AI representation methods and biological neural recordings.
- **Relational causal models** (Jensen and Neville) --- brain region interactions are inherently relational, not i.i.d. Their framework for extending causal inference to multi-level systems shaped how we think about inter-region evidence routing.
- **Dynamical systems** (Hava Siegelmann) --- the connection between network topology and computational capacity. Our finding that rotational dynamics are universal but vary 10x in strength connects to this tradition.

[One sentence about what grounding means to you. Either: "These aren't name-drops --- these are the specific ideas that made me think the geometry might be real" or "I wanted to be honest about where the intuitions came from."]

## Code and data

[Link to Zenodo DOI once created]

[Link to GitHub repo]

All experiments are runnable via `modal run modal_run.py --experiment expNN`. Results are saved to GCS and Modal volumes. 54 experiments total, 39 recording sessions, 73 brain regions.

---

[One closing sentence in your voice. Either: "This is exploratory. Feedback welcome." or "I think geometric validity matters. This is my attempt to take it seriously." or something else entirely.]
