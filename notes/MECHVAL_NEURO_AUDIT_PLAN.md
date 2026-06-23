# MechVal Validity Audit of Neural Circuit Claims

## Core Idea

Apply the MechVal validity framework (developed for AI interpretability) to canonical neuroscience circuit claims. Re-analyze public datasets to upgrade OR downgrade validity levels for each claim.

## Why This Works

1. Neuroscience has 30+ years of circuit claims with no formal validity framework
2. All target datasets are public (Steinmetz, IBL, Allen, Churchland motor)
3. Our existing geometric pipeline gives us tools no prior audit has used
4. The MechVal framework is already built — we just apply it to a new domain

## Target Papers (8-10 case studies)

| Paper | Claim | Public Data | Expected MechVal Gap |
|---|---|---|---|
| Steinmetz 2019 | MOs necessary for distributed choice | Yes (our dataset) | I2 (no stimulation), C4 (arousal confound) |
| Churchland 2012 | Motor cortex implements rotational dynamics | Yes | E4 (replication contested), C4 (behavior confound) |
| Stringer 2019 | Visual cortex has power-law spectrum alpha~1 | Yes | C3 (convergent validity) — does spectrum survive shuffling? |
| Gardner 2022 | Hippocampus encodes environment on a torus | IBL/public | I2 (no stimulation along torus) |
| IBL 2023 | Brain-wide choice signal is reproducible | Yes (IBL BWM) | C4 (movement confounds), I3 (specificity) |
| Gallego 2020 | Stable manifold across perturbations | Yes | E4 (limited replication), I2 (sufficiency) |

## For Each Case Study

1. Extract claim formally: [region/circuit] is [necessary/sufficient] for [behavior] via [mechanism]
2. Run MechVal 6-layer checklist on their published evidence
3. Re-analyze their public data with our tools:
   - Shuffle labels → does the "effect" survive? (C4 test)
   - Compute geometric type profile → is their metric appropriate for this region?
   - Held-out mice → does the claim replicate within-dataset? (E4 partial)
   - Power-law spectrum → does alpha match their claims? (convergent validity)
4. Issue verdict: current tier + specific gap + what experiment would close it

## Deliverable

Paper: "Mechanistic Validity in Systems Neuroscience: A Framework for Evaluating Circuit Claims"

## Venue

eLife or PLOS Computational Biology (neuroscience audience)
NeurIPS workshop track (ML crossover)

## Timeline

~3-4 weeks focused work (2-3 days per case study, already have tools)
