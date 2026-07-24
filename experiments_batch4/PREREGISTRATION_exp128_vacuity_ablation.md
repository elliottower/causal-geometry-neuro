# Pre-registration v3: exp128 — Vacuity controls for all 6 inductive-bias variants

## Motivation

exp73 showed that sparse overcomplete variants achieve the highest IIA on real
neural data (Sparse Structured AE = 0.962, best of 6 variants). exp59 showed
that the Structured VAE achieves comparable IIA on random Gaussian noise as on
real data (0.70 vs 0.69), confirming IIA is vacuous for nonlinear methods.

The missing experiment: **do the sparse variants pass the noise control?** If
sparsity constrains the encoder enough to reject noise, that is a genuine
contribution. If not, the sparse IIA improvement is uninterpretable.

## Bug fixes from exp73 (applied in exp128)

The following issues in exp73's code are corrected in this experiment:

1. **In-sample post-hoc classifier (CRITICAL)**: exp73 `_compute_iia` fits a
   LogisticRegression on the same z_choice vectors used for swap evaluation.
   On noise, any encoder that creates two clusters will appear "causal."
   Moreover, exp59 correctly uses each model's own classifier head — the
   causal question is "does the model's own prediction change when we swap
   z_choice?" Using a separate post-hoc classifier conflates encoder
   capacity with IIA.
   **Fix**: exp128 uses each model's built-in `choice_head` or `classifier`
   for IIA evaluation. For PiVAE (model B, no classifier head), posterior-prior
   log-likelihood scoring is used: predict label by comparing
   log N(z | prior_mu[y=0]) vs log N(z | prior_mu[y=1]) under PiVAE's own
   learned conditional priors. This is PiVAE's native discrimination
   mechanism — no external classifier is needed.

2. **Pairing variable mismatch (CRITICAL)**: exp73 pairs trials by
   `choice_labels` (behavioral response); exp59 pairs by `evidence_labels`
   (stimulus direction). These differ because animals sometimes choose wrong.
   Pairing by the same variable the classifier was trained on makes IIA
   trivially inflatable.
   **Fix**: exp128 pairs by `evidence_labels` (stimulus contrast direction),
   consistent with exp59 and with the causal interpretation (we intervene on
   the stimulus-driven representation, not the behavioral output).

3. **No z-scoring / data leakage (MODERATE)**: exp73 feeds raw activity to
   models; exp59 z-scores first. Different brain regions have wildly different
   firing rates. Whole-data z-scoring leaks test-set statistics into training.
   **Fix**: exp128 computes z-score parameters (mu, std) on the train split
   only, then applies those parameters to both train and test data.

4. **Sequential pair sampling (MINOR)**: exp73 uses `i % len(left_idx)`,
   wrapping around and reusing pairs. exp59 uses random sampling.
   **Fix**: exp128 uses `np.random.choice` without replacement.

5. **No-go trial handling**: `get_choice_labels` maps `response > 0` to 1
   and `response <= 0` (including no-go, `response == 0`) to 0.
   **Fix**: exp128 filters to `response != 0` for pure left-vs-right choice.
   Original inclusive behavior reported as sensitivity analysis.

**Consequence**: exp73's IIA numbers (0.939-0.962) were measured with a
fundamentally different procedure than exp59 (0.69-0.70). They are not
directly comparable. exp128 establishes a single consistent IIA procedure
applied to all models and conditions.

## Design

### Required preconditions (must pass before primary analysis)

1. **Untrained-encoder baseline**: for each model architecture, run the full
   IIA pipeline on real data with UNTRAINED weights (random initialization,
   no training). This gives the true empirical null IIA for each architecture
   under the same pairing/classifier procedure.

   **Pass criterion**: untrained IIA must be < 0.6 for all architectures.
   If any architecture exceeds 0.6 untrained, that architecture's results
   are uninterpretable and excluded from primary analysis.

2. **Reconstruction-only baseline**: for each architecture, train with
   reconstruction loss only (no classification loss, alpha_choice=0).
   A model that achieves high IIA without any classification signal is
   evidence of information leakage through reconstruction.

   **Pass criterion**: reconstruction-only IIA must be < 0.65. If exceeded,
   the reconstruction pathway leaks label information and the split-based
   IIA interpretation is compromised.

### Data conditions (2)

1. **Real neural data** — Steinmetz Neuropixels, 73 regions, binary choice
   labels (left/right). Post-stimulus 150-350ms, trial-averaged, min 15
   neurons per region. Z-scored per region-session. No-go trials excluded
   (response != 0).

2. **Random Gaussian noise** — Same shape as real data per region
   (n_trials x n_neurons), drawn from N(0,1). Real evidence labels preserved
   (same evidence direction, random features).

### Label conditions (2)

For each data condition:

- **Real labels** — actual left/right evidence direction from stimulus contrast.
- **Shuffled labels** — random permutation of evidence labels within each
  region-session. Same neural data, broken stimulus-activity relationship.

### Models (6 primary + per-architecture controls)

| ID | Name | Split | Label prior | Overcomplete+L1 | IIA classifier |
|----|------|-------|-------------|------------------|----------------|
| A  | Structured VAE | yes | no | no | model.choice_head |
| B  | pi-VAE | no | yes | no | posterior-prior scoring (no external classifier) |
| C  | LC Structured VAE | yes | yes | no | model.choice_head |
| D  | Sparse Structured VAE | yes | no | yes | model.choice_head |
| E  | LC Sparse Plain VAE | no | yes | yes | model.classifier |
| F  | LCS-VAE (full) | yes | yes | yes | model.choice_head |

For each of A-F, two control conditions are also run:
- **Untrained**: random init, no training, full IIA pipeline
- **Reconstruction-only**: alpha_choice=0, full training, full IIA pipeline

Hyperparameters (frozen from exp73):
- z_choice_dim = 3, z_other_dim = 15
- hidden_dim = 128, n_epochs = 300, batch_size = 64
- lr = 1e-3, beta_kl = 1.0, alpha_choice = 10.0
- L1_coeff = 1e-3, expansion_factor = 8 (for sparse variants)
- n_iia_pairs = 100 per region-session

### IIA computation (corrected)

For each region-session, repeat N_REPLICATES=3 times with different seeds:

1. Generate shared splits: stratified 50/50 train/test by evidence label,
   plus N_IIA_PAIRS=100 intervention pairs from test set. All 6 models
   within a replicate use the same split and pairs.
2. Z-score: compute mu/std from train split only, apply to both train and test.
3. Prepare data condition (noise: replace real activity with N(0,1) same shape;
   shuffled: permute evidence labels). Noise and shuffle use per-condition
   deterministic seeds.
4. Train model on TRAIN split only (or skip for untrained baseline; or set
   alpha_choice=0 for reconstruction-only baseline). Model never sees test
   data during training.
5. Encode ALL trials (train + test) to get z_choice (using mu, not sampled z).
   IIA is evaluated on test-set trials only.
6. Predict labels:
   - Models with classifier head (A, C, D, E, F): model's own
     `choice_head(z_choice)` or `classifier(z_choice)`.
   - PiVAE (B): posterior-prior scoring — predict by comparing
     log N(z | prior_mu[y=0], prior_logvar[y=0]) vs
     log N(z | prior_mu[y=1], prior_logvar[y=1]).
7. For each pre-selected intervention pair (opposite evidence labels):
   swap z_choice. Re-predict with same method. Count bidirectional flips
   (both A->B and B->A directions per pair).
8. IIA = flip_count / (2 * n_pairs).
9. Average across N_REPLICATES for final per-region estimate.

### Evaluation metrics

For each (model x data_condition x label_condition x region), averaged across
N_REPLICATES:
- **IIA** (bidirectional flip rate after z_choice swap)
- **Classification accuracy** on z_choice (posterior-prior for B; model head
  for rest)
- **Reconstruction MSE**
- **Sparsity** (sparse variants only):
  - L0 (scale-relative): # dimensions where |z_j| > 0.1 * std_j (std from
    training set), avoids arbitrary fixed threshold
  - Hoyer sparsity: (sqrt(n) - L1/L2) / (sqrt(n) - 1), threshold-free
  - Top-3 concentration: fraction of L1 norm in top 3 dimensions
  - Feature activation frequency: fraction of samples where each dim is active
- **Latent cosine spread**: 1 - mean pairwise cosine similarity of z_choice
  vectors within each label class (computed in latent space, not decoded space).
  0 = degenerate encoder, 1 = maximally diverse.

### Primary analysis

**Vacuity test per model**: Wilcoxon signed-rank across 73 regions comparing
IIA(real data, real labels) vs IIA(noise, real labels).

A model is **non-vacuous** if ALL of:
- IIA(real) > IIA(noise) with p < 0.001 (Wilcoxon signed-rank)
- Effect size: mean IIA gap (real - noise) >= 0.10 OR Cohen's d >= 0.5
- IIA(noise) is within 0.05 of the untrained-encoder baseline for that
  architecture (noise IIA does not exceed what random weights achieve)

**Shuffled-label test per model**: Wilcoxon signed-rank across 73 regions
comparing IIA(real data, real labels) vs IIA(real data, shuffled labels).

A model **learns real structure** if:
- IIA(real labels) > IIA(shuffled labels) with p < 0.001
- Effect size: mean IIA gap >= 0.10 OR Cohen's d >= 0.5

**Non-independence sensitivity**: brain regions co-occur within the same
mouse/session. We report:
- Primary: naive Wilcoxon across all 73 regions
- Sensitivity: hierarchical bootstrap clustered by mouse (resample mice with
  replacement, then sessions within mouse). 10,000 iterations. Report 95% CI
  on the IIA gap. If bootstrap CI includes zero while naive test rejects,
  flag as potentially inflated by within-cluster correlation.

**Equivalence testing (TOST)**: "not significantly above chance" does not mean
"equivalent to chance." For models where we claim noise IIA is near chance,
we apply two one-sided tests (TOST) with equivalence margin delta=0.05
around the untrained baseline. A model's noise IIA is "equivalent to chance"
only if TOST rejects at alpha=0.05.

**Replication**: each region-session runs N_REPLICATES=3 matched replicates
with different seeds for model initialization, noise draws, and label
shuffles. Per-region IIA is averaged across replicates; within-replicate
variance (IIA std) is reported as a measure of estimator stability.

### Predictions

| Model | Noise vacuous? | Shuffled-label control? | Reasoning |
|-------|---------------|------------------------|-----------|
| A. Structured VAE | YES (vacuous) | Passes | Known from exp59 |
| B. pi-VAE | YES (vacuous) | Unclear | No structural constraint on encoder |
| C. LC Structured VAE | YES (vacuous) | Passes | Label prior doesn't constrain encoder capacity |
| D. Sparse Structured VAE | **UNCERTAIN** | Passes | L1 may constrain encoder — the key test |
| E. LC Sparse Plain VAE | **UNCERTAIN** | Unclear | Sparsity helps but no split |
| F. LCS-VAE (full) | **UNCERTAIN** | Passes | Full stack — strongest candidate |

The headline question: **does D (or F) pass the noise control where A fails?**

If yes -> sparsity is the inductive bias that makes nonlinear causal discovery
non-vacuous. Write up for NeurReps.

If no -> all nonlinear methods are vacuous on IIA. The paper leans entirely on
optogenetic external validation. Still submittable but weaker headline.

### Secondary analyses

1. **L1 coefficient sweep**: run sparse variants (D, E, F) at three L1
   values: {1e-4, 1e-3, 1e-2}. Primary uses 1e-3. If the vacuity verdict
   changes across L1 values, report the full curve. If stable across all
   three, the L1=1e-3 result is robust.

2. **Optogenetic correlation per model**: Spearman rho between each model's
   IIA and optogenetic silencing effect across 12 matched regions.

3. **Sparsity-vacuity relationship**: scatter L0 vs IIA(noise) across models
   and regions. If sparser codes -> lower noise IIA, sparsity is the mechanism.

4. **Reconstruction quality on noise**: if MSE(noise) ~ MSE(real), the encoder
   memorizes. If MSE(noise) >> MSE(real), reconstruction loss provides some
   constraint.

5. **No-go sensitivity**: rerun primary analysis including no-go trials
   (response=0 mapped to label=0, matching exp73). If results differ
   meaningfully, report both.

## Implementation

Script: `experiments_batch4/exp128_vacuity_ablation.py`
- Self-contained model classes (copied from exp73 to avoid import issues)
- Reuse data loading from `data/steinmetz.py`
- `_make_shared_splits()`: shared stratified train/test + intervention pairs
- `_zscore_fit()` / `_zscore_apply()`: train-only z-score parameters
- `_make_predict_fn()`: model's own classifier or PiVAE posterior-prior scoring
- `_compute_iia_corrected()`: evidence-based pairing, random sampling,
  bidirectional flips, all on pre-selected test-set pairs
- `_compute_sparsity_metrics()`: Hoyer, scale-relative L0, top-k concentration
- `_run_single_condition()`: runs ALL models on shared splits for one condition
- `_hierarchical_bootstrap()`: mouse-clustered bootstrap for non-independence
- Audit trail saved per region (splits, pairs, seeds) in `results_batch4/exp128/audit/`
- Results saved incrementally per region to `.jsonl`

## Execution

- GPU: Modal A10G (same as exp73)
- Estimated runtime: ~6-8 hours (6 models x 4 data conditions x 73 regions x
  300 epochs x 3 replicates, plus untrained/recon-only controls)
- Results saved incrementally per region to `.jsonl`
- Final summary to `exp128_summary_<timestamp>.json`

## Changelog

- v1: Initial pre-registration
- v2: Bug fixes documented (classifier, pairing, z-scoring, sampling, no-go)
- v3: PiVAE posterior-prior scoring (not held-out LR), train-only z-scoring,
  shared splits across models, N_REPLICATES=3, scale-relative sparsity
  metrics (Hoyer + L0 + top-k + feature frequency), equivalence testing
  (TOST), audit trail
