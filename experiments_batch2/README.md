# Batch 2: Reviewer Fixes

All experiments motivated by the Perplexity peer review (June 2026).
These address critical reviewer concerns before TMLR submission.

## Results Summary (as of 2026-06-22)

### Completed
| Exp | Result | Key number |
|-----|--------|-----------|
| 10 | Directions indistinguishable from random (p~1.0) | Mann-Whitney U confirms subspaces needed |
| 62 | Real IIA=0.625 vs shuffled=0.430 | Wilcoxon p=6e-8, 24/24 regions |
| 63 | Nonlinearity effect near zero | mean diff=0.0008, p=0.65 |
| 65 | 100% pre-response emergence | LDA and VAE both show onset before behavior |
| 66 | Per-mouse optogenetic | Results on Modal volume |
| 74 | Debiased CKA rho=0.445 vs biased rho=0.850 | delta=-0.40; partial rho=0.342. Murphy et al. concern confirmed |
| 75 | Coordinate matching: 10-15 matches at 0.5-2.0mm | Many-to-one issue; decoding proxy (not IIA) shows no correlation |
| 77 | alpha vs n_neurons rho=-0.916 | Chun et al. bias confirmed — must control for n_neurons |
| 78 | LDA 80% power at n=14; VAE never reaches 80% | Need n>60 or stronger signal for VAE alone |

### Still Running
| Exp | Status |
|-----|--------|
| 76 | Running on Modal (UMAP stochasticity, 20 seeds x 9 configs) |

### Not Yet Implemented
| Exp | Why |
|-----|-----|
| 79 | Needs GPU + exp75 coordinate matching results + trained exp73 models |

## Experiments

### Already scripted (in experiments/, need to run)
| Exp | Script | CPU/GPU | Priority | What |
|-----|--------|---------|----------|------|
| 10 | exp10_direction_vs_subspace.py | CPU | HIGHEST | Figure 1: cosine sim random, Grassmannian structured |
| 62 | exp62_shuffled_label_control.py | CPU | HIGHEST | VAE IIA on shuffled labels — closes vacuity logic gap |
| 63 | exp63_linear_vae_ablation.py | CPU | HIGH | Linear vs nonlinear encoder ablation |
| 65 | exp65_temporal_iia.py | CPU | MODERATE | When does choice subspace IIA emerge during trial? |
| 66 | exp66_per_mouse_silencing.py | CPU | HIGH | Per-mouse optogenetic validation |

### New experiments (in this folder)
| Exp | Script | CPU/GPU | Priority | What |
|-----|--------|---------|----------|------|
| 74 | exp74_debiased_cka.py | CPU | HIGHEST | Murphy et al. 2024 debiased CKA — rerun main result |
| 75 | exp75_ccf_coordinate_matching.py | CPU | HIGH | Spatial nearest-neighbor optogenetic matching |
| 76 | exp76_umap_stochasticity.py | CPU | MODERATE | Multi-seed UMAP Procrustes robustness |
| 77 | exp77_alpha_bias_robustness.py | CPU | MODERATE | Neuron-count-matched controls for power-law alpha |
| 78 | exp78_optogenetic_power_analysis.py | CPU | MODERATE | Power analysis: n needed for VAE correlation p<0.05 |
| 79 | exp79_sae_optogenetic.py | GPU | HIGH | SAE variants vs optogenetic silencing correlation |
