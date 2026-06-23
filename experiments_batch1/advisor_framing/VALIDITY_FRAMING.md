# Validity Framing for Neuro Paper — Upstream Citations

Use these ideas and cite the original sources directly (NOT MECHVAL itself).

## The Key Mapping: Neuro Claims → Validity Criteria → Upstream Sources

| Neuro claim | Validity concern | Original source to cite | What to say |
|---|---|---|---|
| Choice subspace localized in region X | Necessity not shown without silencing | Shallice 1988 (From Neuropsychology to Mental Structure), Craver 2007 (Explaining the Brain) | "Following the necessity–sufficiency distinction standard in neuropsychology (Shallice 1988), we note that subspace localization demonstrates correlation, not causal necessity" |
| H¹ >> 0 means distributed circuit | Baseline separation — metric may measure its own flexibility | Sutter et al. 2025 (Is Causal Abstraction Enough for MI?), Mayo 2018 (Statistical Inference as Severe Testing) | "A metric can produce a confident high number while measuring its own flexibility rather than the system (cf. Sutter et al. 2025). Following Mayo's severe testing (2018), we report Δ_shuffle rather than raw H¹" |
| CKA/Grassmannian dissociation | Discriminant validity — the two metrics measure different things | Campbell & Fiske 1959 (Multitrait-multimethod matrix) | "The dissociation between CKA and Grassmannian distance establishes discriminant validity (Campbell & Fiske 1959) — different instruments sensitive to different geometric properties" |
| Geometric type taxonomy generalizes | Cross-dataset transportability | Pearl & Bareinboim 2011 (Transportability of Causal and Statistical Relations) | "Formal transportability (Pearl & Bareinboim 2011) requires that the finding transfer to an independent dataset with different experimenters and subjects — the IBL Brain-Wide Map provides exactly this test" |
| IIA asymmetry → causal direction | Interventionist causation | Woodward 2003 (Making Things Happen) | "Under the interventionist framework (Woodward 2003), asymmetric IIA — where swapping evidence in A changes choice in B more than vice versa — implies directional influence from A to B" |
| "Routing vs holding" region labels | Anthropomorphism risk | Marr 1982 (Vision), Craver 2007 | "Description-level must match evidence level (Marr 1982). We constrain claims to the representational level — what information is encoded and in what geometry — rather than imputing computational-level function to regions" |
| Causal graph from PC/LiNGAM | Causal discovery from observational data | Spirtes, Glymour & Scheines 2000 (Causation, Prediction, and Search) | "Constraint-based causal discovery (Spirtes et al. 2000) from cross-session covariance requires the faithfulness assumption, which we test via bootstrap stability" |
| Allen Atlas correlation | Structural plausibility | Oh et al. 2014 (A mesoscale connectome of the mouse brain) | "Structural plausibility: geometry-derived directed edges should correspond to anatomical projection densities (Oh et al. 2014)" |

## The Self-Audit Table (use in paper)

Score each experiment against the relevant criteria, citing the original sources:

| Experiment | Claim tier | What it demonstrates | What would advance it | Source tradition |
|---|---|---|---|---|
| exp1–5 (representational) | Proposed | Convergent geometric evidence | Causal intervention → Suggestive | Philosophy of science (Mayo 2018) |
| exp7 (sheaf H¹) | Proposed → Suggestive or Disconfirmed | Non-trivial topology | Baseline separation (exp7_controls) | Signal detection theory |
| exp14 (type classifier) | Proposed | Predictive classification | Cross-dataset transport (exp46) | Pearl/Bareinboim transportability |
| exp42/44 (IIA) | Causally Suggestive | Swap-based causal evidence | Sufficiency (restoration) | Neuroscience (Craver 2007) |
| exp45 (Allen Atlas) | Structural validation | Anatomy–geometry correspondence | — | Connectomics (Oh et al. 2014) |
| exp47 (silencing) | Cross-modal validation | Geometry predicts silencing effects | Direct opto data (exp47b) | Neuropsychology (Shallice 1988) |

## Complete Citation List (for bibliography)

```bibtex
@book{shallice1988neuropsychology,
  author = {Shallice, Tim},
  title = {From Neuropsychology to Mental Structure},
  publisher = {Cambridge University Press},
  year = {1988}
}

@book{craver2007explaining,
  author = {Craver, Carl F.},
  title = {Explaining the Brain: Mechanisms and the Mosaic Unity of Neuroscience},
  publisher = {Oxford University Press},
  year = {2007}
}

@book{mayo2018statistical,
  author = {Mayo, Deborah G.},
  title = {Statistical Inference as Severe Testing: How to Get Beyond the Statistics Wars},
  publisher = {Cambridge University Press},
  year = {2018}
}

@inproceedings{pearl2011transportability,
  author = {Pearl, Judea and Bareinboim, Elias},
  title = {Transportability of Causal and Statistical Relations: A Formal Approach},
  booktitle = {AAAI},
  year = {2011}
}

@book{woodward2003making,
  author = {Woodward, James},
  title = {Making Things Happen: A Theory of Causal Explanation},
  publisher = {Oxford University Press},
  year = {2003}
}

@book{marr1982vision,
  author = {Marr, David},
  title = {Vision: A Computational Investigation into the Human Representation and Processing of Visual Information},
  publisher = {MIT Press},
  year = {1982}
}

@book{spirtes2000causation,
  author = {Spirtes, Peter and Glymour, Clark and Scheines, Richard},
  title = {Causation, Prediction, and Search},
  publisher = {MIT Press},
  edition = {2nd},
  year = {2000}
}

@inproceedings{sutter2025nonlinear,
  author = {Sutter, Thomas and Minder, Julian and Casanova, Pablo and Geiger, Atticus and Roth, Volker},
  title = {Is Causal Abstraction Enough for Mechanistic Interpretability?},
  booktitle = {NeurIPS},
  year = {2025}
}

@article{campbell1959convergent,
  author = {Campbell, Donald T. and Fiske, Donald W.},
  title = {Convergent and Discriminant Validation by the Multitrait-Multimethod Matrix},
  journal = {Psychological Bulletin},
  volume = {56},
  number = {2},
  pages = {81--105},
  year = {1959}
}

@book{glennan2017new,
  author = {Glennan, Stuart},
  title = {The New Mechanical Philosophy},
  publisher = {Oxford University Press},
  year = {2017}
}

@article{oh2014mesoscale,
  author = {Oh, Seung Wook and Harris, Julie A. and Ng, Lydia and others},
  title = {A Mesoscale Connectome of the Mouse Brain},
  journal = {Nature},
  volume = {508},
  pages = {207--214},
  year = {2014}
}

@article{geiger2023causal,
  author = {Geiger, Atticus and others},
  title = {Causal Abstraction: A Theoretical Foundation for Mechanistic Interpretability},
  journal = {JMLR},
  volume = {26},
  year = {2025}
}
```

## Three Strategic Uses (from MECHVAL ideas, not citing MECHVAL)

1. **Self-audit table in the paper.** Score each experiment against the relevant criterion, cite the original source, mark Confirmed/Partial/Untested. This pre-empts "is this real?" reviewer objections.

2. **exp7 controls as baseline-separation test.** Cite Sutter et al. 2025 and Mayo 2018 directly. "Not H¹=495 but Δ_shuffle=X, Δ_random=Y." A modest number with large delta is a finding; a large number with tiny delta is not.

3. **E4/transportability as formal backbone of cross-dataset generalization.** Cite Pearl & Bareinboim 2011 directly. Steinmetz→IBL transport becomes a formal transportability test, not ad-hoc replication.

## The Unifying Pitch

The paper is honest about what each experiment demonstrates and what it doesn't. Each claim is tagged with its evidence level using terminology from the fields that invented validity testing (neuroscience for necessity/sufficiency, philosophy of science for severe testing, causal inference for transportability). This turns honesty about limitations into a methodological strength.
