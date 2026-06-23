"""Experiment 79: SAE variant optogenetic validation.

Table 5 (exp73) shows SAE Structured achieves IIA=0.962 vs VAE's 0.939. But
the paper argues IIA is vacuous for nonlinear methods (Sutter dilemma). The
ablation conclusion — "SAE-style sparse representations are the dominant inductive
bias" — is based on IIA differences among methods already shown to be vacuous.
This is a logical contradiction a reviewer will catch.

Fix: run all 6 SAE/VAE variants through the optogenetic correlation pipeline.
If SAE Structured also has a higher optogenetic correlation than plain VAE,
the ablation is validated by external ground truth, not just vacuous IIA.

This experiment:
1. For each of the 6 models from exp73 (structured_vae, pi_vae, pi_structured_vae,
   sae_structured, pi_sae_plain, pi_sae_structured):
   - Train on each of the matched optogenetic regions
   - Compute IIA per region
   - Correlate IIA profile with Zatka-Haas silencing effect sizes
2. Report Spearman rho and p-value for each model variant
3. Key question: does SAE Structured outperform plain VAE on the optogenetic
   correlation too, or only on the vacuous IIA metric?

GPU recommended (6 model variants x n_regions x 300 epochs). ~4h on A100.
"""
import json
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results" / "exp79"


def run(max_sessions=None, model_filter=None):
    """Main entry point.

    Args:
        model_filter: if set, only run this model variant (for parallelization)
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    # TODO: implement — reuse exp73 model classes + exp75 coordinate matching
    raise NotImplementedError


if __name__ == "__main__":
    run()
