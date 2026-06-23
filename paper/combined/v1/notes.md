# Combined Paper v1 — Notes

## Why combine?

Perplexity review verdict: "The combined paper angle is actually worth considering."
- Paper A establishes the geometric framework (metrics disagree, dimensionality is why)
- Paper B establishes the causal consequence (LDA is wrong, VAE corrects it)
- Together that's one story; split, both feel like they're missing half their motivation

## Cross-cutting issues that go away if combined
- Circular cross-citation ("[Paper A]"/"[Paper B]") eliminated
- Paper B's intro assumes Paper A's rho=-0.85 — currently a forward ref to unpublished work
- Reviewer getting only Paper B has 8 forward refs to results they can't see
- Harvey 2024 framing works better as opening motivation for the whole arc

## Combined story arc
1. Opening figure: direction-vs-subspace experiment (cosine similarity random, Grassmannian structured) — motivates why subspaces not directions
2. CKA vs Procrustes anti-correlation (Paper A core) — metrics disagree, dimensionality mediates
3. Why this matters: LDA (linear) identifies WRONG regions as causal (Paper B optogenetic)
4. Structured VAE corrects this, but IIA is vacuous — need external validation
5. SAE ablation: overcomplete sparse representations best for causal subspaces
6. The whole thing is one story about why geometry matters for causal inference in neural data

## Venue estimate
~55% NeurIPS if direction-vs-subspace experiment + debiased CKA + Exp62 all done

## What needs to happen first
- [ ] Direction-vs-subspace experiment (CPU only, ~2h, would be Figure 1)
- [ ] Debiased CKA rerun
- [ ] Exp62 shuffled-label control
- [ ] Expand optogenetic matching to n=25+
- [ ] Then merge tex files into one coherent narrative
