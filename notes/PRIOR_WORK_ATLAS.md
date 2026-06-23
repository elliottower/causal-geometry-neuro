# Prior Work Atlas

Paths to all related prior work from the research program.
Relevance rated for this project (applying causal geometry to real neural data).

## VERY HIGH relevance — direct foundations

| Document | Path | What |
|----------|------|------|
| Causal Geometry v2 | `/Users/elliottower/Downloads/CAUSAL_GEOMETRY/_PAPER/causal_geometry_v2_ORIGINAL_BODY.tex` | Core proposal: causal variables as points on Gr(k,n), six mathematical frameworks. Direct template for neural subspace analysis. |
| Mechanism Geometry Program | `/Users/elliottower/Documents/GitHub/weight-circuit-discovery/paper/UPDATES_6_1/mechanism_geometry_program.tex` | G-SCM formalization, transport maps, three-domain evidence. Weight matrices → connectivity; residual stream → population activity; DAS → dimensionality reduction. |
| Mechanistic Validity (paper) | `/Users/elliottower/Documents/GitHub/mechanistic-validity/docs/paper/UPDATED_PAPER/FINAL/main.tex` | 27-criteria validation framework, six-layer pipeline. Directly replicable for neural circuit claims. |

## HIGH relevance — frameworks that transfer

| Document | Path | What |
|----------|------|------|
| Mechanistic Views v4 | `/Users/elliottower/Documents/GitHub/mechanistic-views/paper/mechviews_v4.tex` | Five-axis ontology for "what counts as a mechanism." The subspace view is exactly what we're testing in neural data. |
| MechVal Interface | `/Users/elliottower/Documents/GitHub/mechanistic-views/src/content/docs/mechval-interface.md` | How validity criteria map to view axes. Grassmannian convergence conditions directly operationalize neural validation. |
| Methods Classification | `/Users/elliottower/Documents/GitHub/mechanistic-views/src/content/docs/methods.md` | Which MI techniques carry which ontological commitments. Maps to neural dimensionality reduction choices. |
| Validity Lenses | `/Users/elliottower/Documents/GitHub/mechanistic-validity/docs/src/content/docs/framework/lenses/index.md` | 11 intellectual lenses grounding 27 criteria. Neuroscience lens (double dissociation), geometry lens (Fisher-Rao, sheaf consistency) directly applicable. |
| Decisive Upgrade | `/Users/elliottower/Downloads/decisive_upgrade.md` | Shift from hypothesis-testing to precision measurement. Template: "which subspace, how large, how stable?" |

## MEDIUM relevance — program structure

| Document | Path | What |
|----------|------|------|
| Research Program Upgrade | `/Users/elliottower/Downloads/research_program_upgrade.md` | Roadmap addressing 27 criteria. Two-tier structure (core + extensions) maps to neural study design. |

## Tooling

| Tool | Path | Usage |
|------|------|-------|
| MechVal CLI | `/Users/elliottower/Documents/GitHub/mechanistic-validity/` | `mechval verify spec.json` — run causal model testing on claim specs |
| MechVal Skills | `/Users/elliottower/Documents/GitHub/mechanistic-validity/skills/` | Claude Code skills for mechanistic validation |
| MC-IAYN Auditor | `/Users/elliottower/Downloads/mc_iayn/src/auditor/mciayn_audit.py` | `uv run mciayn_audit.py --manual --baselines N ...` — FWER calculation |
| MechViews Docs | `/Users/elliottower/Documents/GitHub/mechanistic-views/src/content/docs/` | Full documentation site for mechanistic views framework |
| MechVal Docs | `/Users/elliottower/Documents/GitHub/mechanistic-validity/docs/` | Framework docs, lenses, criteria |

## Key conceptual bridges

**Transformer → Neural data translations:**
- Weight matrices → Connectivity/synaptic matrices
- Residual stream → Population activity vector
- Attention heads → Functional cell assemblies
- DAS rotation → Dimensionality reduction of neural recordings
- Factor bank → Shared neural basis set across regions
- Selector matrix → Region-specific readout weights
- Gauge freedom → Arbitrary rotation of neural coordinates (electrode placement)

**Shared mathematical objects:**
- Gr(k, d) — same manifold for both transformer subspaces and neural subspaces
- Principal angles — mechanism identity in both domains
- Transport maps — information flow via weights (transformers) or connectivity (neural)
- Sheaf cohomology — circuit localizability in both domains
- Holonomy — circuit curvature under temporal evolution

**Where the analogy breaks:**
- Transformers have discrete layers; neural circuits have continuous recurrence
- Transformer weights are fixed at inference; neural connectivity is plastic
- We can compute transformer gradients; neural "gradients" require optogenetics
- Transformer dimensions are cleanly defined; neural dimensions depend on recording technology
