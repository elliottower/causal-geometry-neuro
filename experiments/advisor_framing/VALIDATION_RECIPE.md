# From Correlation to Discovery: Validation Recipe

## What Separates a Discovery From a Correlation

A biological discovery makes a **specific, falsifiable prediction about something you didn't use to derive it.** Right now the rho = -0.59 is descriptive -- it summarizes the data you fit on. To become a discovery, the method has to predict a fact that exists independently in the world and that you can check against a separate source. The test neuroscientists apply is: "If I told you the answer was X, and your method said X before you looked, that's a discovery. If your method only describes the data it was fit on, that's a measurement."

## The Three Validation Strategies, Ranked by Feasibility

### 1. Predict known anatomy from geometry (strongest, no new data)

The cleanest play -- the CKA clustering already recovers thalamic/cortical/posterior groupings. Push it to a real prediction: the cross-region IIA matrix predicts a *directed causal graph* over regions. Compare it against the **Allen Mouse Brain Connectivity Atlas**, which has ground-truth anatomical projection strengths between every region pair. If geometry-derived causal edges (e.g. VISp->MOs high IIA) match the Allen Atlas's axonal projection densities better than chance, that's a discovery: "subspace-level causal influence predicts anatomical connectivity." The Allen Atlas is fully public, so this is a held-out validation against an independent biological source -- exactly the gold standard.

### 2. Out-of-distribution prediction across datasets (very strong)

Derive the geometric type taxonomy on Steinmetz, then *predict* the geometric types of the same regions in IBL -- a completely separate dataset, different labs, different mice. If MOs is Procrustes-type in Steinmetz and the method predicts (without refitting) that it's Procrustes-type in IBL, and it holds, that's cross-dataset generalization. This is leave-one-dataset-out validation and it's the most convincing thing you can do with existing data.

### 3. Predict a known perturbation result (strongest if it exists)

Steinmetz has optogenetic silencing on some trials -- a real do(region) intervention. The causal abstraction method predicts which downstream regions should change when a given region is silenced. Check whether the regions the geometry flags as causally downstream are the ones that actually show altered choice activity under silencing. If they match, you've validated a causal claim against a real causal manipulation that already happened in the data.

## The "Predicts It" Move

The key is to make the prediction *prospective relative to the validation data*:

- Lock the method on dataset/region set A.
- State the prediction explicitly and in writing *before* touching dataset B (or the Allen Atlas, or the silencing trials).
- Don't tune anything on B. No peeking, no refitting.
- Report the prediction accuracy with proper cross-validation and a null model (shuffle the region labels, show the prediction beats shuffle).

The failure mode reviewers hunt for is circularity -- fitting and validating on the same data, or tuning hyperparameters on the validation set. The paper already has the machinery to avoid this (leave-one-mouse-out and bootstrap CIs). Apply the same rigor to the cross-dataset prediction.

## The Concrete Recipe to Get to a Discovery

1. **Run cross-region IIA on Steinmetz** to get the directed causal influence matrix over regions.
2. **Validate against the Allen Connectivity Atlas** -- does subspace-level causal influence predict anatomical projection density above a shuffled null? (Discovery claim #1.)
3. **Validate the silencing prediction** -- do regions flagged causally-downstream actually change under Steinmetz's optogenetic silencing? (Discovery claim #2, the strongest because it's a real causal manipulation.)
4. **Cross-dataset transport to IBL** -- does the geometric type taxonomy predict region types in a held-out dataset? (Discovery claim #3, generalization.)

If even one of these lands cleanly -- especially #3 (silencing), since it's a real causal intervention -- the claim becomes "subspace geometry predicts the causal architecture of the choice circuit, validated against anatomical connectivity and optogenetic perturbation." That's a finding about the brain, not about metrics.

Honest caveat: these are predictions, not interventions *we* performed, so the claim is "geometry predicts causal structure," not "geometry proves causation." That's still a real discovery and reviewers accept it -- but state the limitation plainly.

## Sources

1. Beyond Brain Mapping: Using Neural Measures to Predict (PMC3903296)
2. A cross-validated cytoarchitectonic atlas of the human (Goebel 2018)
3. Data mining opens the door to predictive neuroscience (Science Daily 2012)
4. arXiv:1606.05201v2 [stat.ML]
5. How Do We Validate Computational Neural Circuit Models? (YouTube)
6. Multiplexed Subspaces Route Neural Activity Across Brain-Wide Networks (PMC9934668)
7. Behavioral Studies Using Large-Scale Brain Networks (Frontiers 2022)
