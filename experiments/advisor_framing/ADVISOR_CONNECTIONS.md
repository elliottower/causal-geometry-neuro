# Advisor Connections to Paper

## Sid — Identifiability & Disentanglement

### Direct connections
1. **ICLR 2026 "Mechanistic Independence for Identifiable Disentanglement"**: The core theorem says factors are identifiable (up to permutation + within-block invertible transforms) when the Jacobian is sparser in factor-aligned coordinates. Our claim that different brain regions implement different geometric types IS a claim that the choice subspace is mechanistically independent of other variance sources. Sid's theorem gives the conditions under which that claim is identifiable from data.

2. **Concrete use**: "The mechanistic independence condition is approximately satisfied in low-dimensional regions (steep eigenvalue spectrum → sparse Jacobian structure) but not in high-dimensional regions (flat spectrum → dense Jacobian), which explains why CKA works in one regime but not the other."

3. **2017 NeurIPS Semi-Supervised DGMs**: Cross-animal comparison is the semi-supervised disentanglement setting — labeled trials (choice direction) + unlabeled variance (everything else), want to encode only the labeled factor into a separable subspace. We implement it geometrically (Grassmannian) rather than generatively (VAE).

### What to ask Sid
- Formalize identifiability conditions for cross-animal subspace comparison
- Which 2025/2026 identifiability results tighten conditions for Grassmannian comparison recovering true causal subspace vs aligned artifact?

### Paper changes needed
- Methods: add identifiability framing ("Under what conditions is the evidence subspace identifiable from finite-neuron recordings?")
- Cite ICLR 2026 paper directly with theorem statement
- Discussion: connect dimensionality regime to Jacobian sparsity

---

## David Jensen — Relational Causal Discovery

### Direct connections
1. **Relational Causal Discovery (RCD)**: Brain regions interacting across animals is a relational domain — entities are (animal, region) pairs with attributes (subspace geometry), learning causal structure between them. His relational d-separation and abstract ground graph framework applies to constructing the cross-region causal graph from the IIA matrix.

2. **Asymmetric dependence → causal direction**: He proved that in relational domains, statistical dependence is inherently asymmetric. If region A's subspace shift causally precedes B's, then IIA(A→B) > IIA(B→A). Apply his relational causal direction test to the IIA matrix → partially oriented causal graph over brain regions from observational data.

3. **HSIC for relational dependence**: He proved HSIC is a consistent test for relational statistical dependence. We already use HSIC (Section 3.7 — confirming CKA and Procrustes are nonlinearly dependent). Cite and extend.

### What to ask Jensen
- Does the relational d-separation framework apply to multi-region, multi-animal structure?
- Help running the causal direction asymmetry test on the IIA matrix → partially-oriented causal DAG

### Paper changes needed
- Results: add IIA asymmetry analysis (new experiment — exp44)
- Methods: frame cross-region IIA in relational causal discovery terms
- Cite RCD papers (Maier et al. UAI 2013, etc.)

### New experiment needed: exp44_iia_asymmetry.py
- Compute IIA(A→B) and IIA(B→A) for all region pairs
- Test asymmetry significance
- Apply PC algorithm or RCD to orient edges
- Compare resulting DAG to known neuroanatomical connectivity

---

## Hava Siegelmann — Dynamical Systems & Cortical Hierarchy

### Direct connections
1. **Cognitive abstraction hierarchy**: Her Dartmouth 2015 talk and ongoing work — cognitive-behavioral hierarchy in cortex derived from fMRI. Our dimensionality taxonomy (low-dim = linear/abstract, high-dim = distributed/geometric) is the same principle, now from Neuropixels at single-neuron resolution.

2. **Bio → AI design principle**: Geometric type taxonomy as architecture design: sensory cortex style (low-dim, linearly stable, fast readout → CKA-type) vs association cortex style (high-dim, geometry-stable, flexible routing → Procrustes-type).

3. **DARPA L2M lifelong learning**: Spectral universality (SV spectra r=0.94 across animals, temporal modes vary) = the architectural scaffold is stable across biological instances, specific memories are not. Direct lifelong learning evidence.

4. **jPCA rotation results**: Ubiquitous but 50x variable in strength — directly in her dynamical systems wheelhouse.

### What to ask Hava
- Frame the biological grounding of geometric type taxonomy in cortical hierarchy terms
- Interpret jPCA rotation results (why 50x variation? what predicts rotation strength?)
- Connect to BINDS lab ICLR 2025 / ANT June 2026 papers

### Paper changes needed
- Discussion: frame dimensionality taxonomy as instantiation of her cognitive abstraction hierarchy
- Add "Implications for artificial systems" paragraph
- Cite L2M work for spectral universality interpretation

---

## Unified Pitch (TMLR)

The paper's core claim pitched three ways:

- **Sid**: "We show empirically that the mechanistic independence condition determines which representational similarity metric is identifiable — first large-scale test of identifiability theory on biological data."
- **Jensen**: "We apply relational causal discovery to multi-region neural data and show asymmetric IIA recovers a partially-oriented causal graph consistent with known neuroanatomy."
- **Hava**: "We derive a computational hierarchy of brain regions from Neuropixels geometry alone — recovering the cognitive abstraction hierarchy at single-neuron resolution."

These are the same paper pitched at three parts of the contribution. TMLR rewards methodological depth across communities.

## References to add to bib

```bibtex
@inproceedings{sid2026mechanistic,
  title={Mechanistic Independence for Identifiable Disentanglement},
  author={...},  % Get exact authors from Sid
  booktitle={International Conference on Learning Representations},
  year={2026}
}

@inproceedings{narayanaswamy2017learning,
  title={Learning Disentangled Representations with Semi-Supervised Deep Generative Models},
  author={Narayanaswamy, Siddharth and Paige, T Brooks and van de Meent, Jan-Willem and Desmaison, Alban and Goodman, Noah and Kohli, Pushmeet and Wood, Frank and Torr, Philip},
  booktitle={Advances in Neural Information Processing Systems},
  year={2017}
}

@inproceedings{maier2013sound,
  title={A Sound and Complete Algorithm for Learning Causal Models from Relational Data},
  author={Maier, Marc and Marazopoulou, Katerina and Arbour, David and Jensen, David},
  booktitle={Uncertainty in Artificial Intelligence},
  year={2013}
}

@phdthesis{maier2014causal,
  title={Causal Discovery for Relational Domains},
  author={Maier, Marc},
  school={University of Massachusetts Amherst},
  year={2014}
}
```
