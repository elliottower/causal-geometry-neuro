"""Experiment 19: Latent causal discovery (Steinmetz).

Apply score-based latent causal discovery to neural population data.
The observed variables are neuron activities; the latent variables are
task constructs (choice, stimulus, arousal, movement).

The method recovers a causal DAG over latent variables from the observed
data, without knowing the latent variables a priori. The key question:
do the recovered latent nodes correspond to the LDA choice subspace?

If yes: the Grassmannian subspace IS the causal variable, bridging our
geometric framework with formal causal discovery.

Uses linear score-based approach (most appropriate for ~250 trials).
"""
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr
from sklearn.decomposition import FastICA
from tqdm import tqdm

from data.steinmetz import get_choice_labels, get_region_activity, list_regions, load_all
from geometry.subspace import fit_lda_subspace

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent.parent / "artifacts" / "exp19"
MIN_NEURONS = 15
TIME_WINDOW = slice(15, 35)
N_LATENTS = 5


def _fit_latent_model(activity, n_latents=5):
    """Recover latent sources via ICA (observational causal discovery proxy).

    ICA recovers independent components — under linear non-Gaussian assumptions,
    these correspond to latent causal variables (Shimizu et al. LiNGAM).
    """
    activity_c = activity - activity.mean(axis=0, keepdims=True)
    n_components = min(n_latents, activity.shape[1], activity.shape[0] - 1)

    ica = FastICA(n_components=n_components, random_state=42, max_iter=1000)
    sources = ica.fit_transform(activity_c)
    mixing = ica.mixing_

    return sources, mixing, ica


def _estimate_causal_order(sources):
    """Estimate causal ordering via pairwise independence tests.

    Simple heuristic: if source i predicts source j (high |correlation|)
    but not vice versa after conditioning, i→j. This is a proxy for
    proper PC algorithm — sufficient for a first pass.
    """
    n = sources.shape[1]
    adj = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            r, p = pearsonr(sources[:, i], sources[:, j])
            if p < 0.05:
                adj[i, j] = abs(r)

    return adj


def run(max_sessions: int | None = None):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    sessions = load_all()
    if max_sessions:
        sessions = sessions[:max_sessions]

    all_results = []

    for sess_idx, sess in enumerate(tqdm(sessions, desc="Sessions")):
        labels = get_choice_labels(sess)
        if len(np.unique(labels)) < 2:
            continue

        mouse = str(sess.get("mouse_name", f"mouse_{sess_idx}"))
        regions = list_regions(sess, min_neurons=MIN_NEURONS)

        for region in regions:
            act = get_region_activity(sess, region)
            if act is None or act.shape[1] < MIN_NEURONS:
                continue

            n = min(act.shape[0], len(labels))
            activity = act[:n, :, TIME_WINDOW].mean(axis=2)

            try:
                sources, mixing, ica = _fit_latent_model(activity, n_latents=N_LATENTS)

                source_choice_corr = []
                for c in range(sources.shape[1]):
                    r, p = pearsonr(sources[:n, c], labels[:n])
                    source_choice_corr.append({"component": c, "r": float(r), "p": float(p)})

                best_choice_component = max(source_choice_corr, key=lambda x: abs(x["r"]))

                k = min(5, activity.shape[1] - 1)
                U = fit_lda_subspace(activity, labels[:n], k=k)
                lda_direction = U[:, 0]

                mixing_norms = np.linalg.norm(mixing, axis=0, keepdims=True)
                mixing_normalized = mixing / np.maximum(mixing_norms, 1e-8)
                ica_lda_cosines = np.abs(mixing_normalized.T @ lda_direction)

                adj = _estimate_causal_order(sources[:n])

                all_results.append({
                    "session_idx": sess_idx,
                    "mouse": mouse,
                    "region": region,
                    "n_neurons": activity.shape[1],
                    "n_latents": sources.shape[1],
                    "source_choice_correlations": source_choice_corr,
                    "best_choice_component": best_choice_component,
                    "ica_lda_alignment": [float(c) for c in ica_lda_cosines],
                    "max_ica_lda_alignment": float(ica_lda_cosines.max()),
                    "adjacency_density": float((adj > 0).mean()),
                    "n_edges": int((adj > 0).sum()),
                })
            except Exception as e:
                logger.warning(f"Failed {mouse}/{region}: {e}")

    results = {
        "timestamp": datetime.now().isoformat(),
        "n_regions_analyzed": len(all_results),
        "n_latents": N_LATENTS,
        "regions": all_results,
    }

    if all_results:
        alignments = [r["max_ica_lda_alignment"] for r in all_results]
        choice_corrs = [abs(r["best_choice_component"]["r"]) for r in all_results]
        results["summary"] = {
            "mean_ica_lda_alignment": float(np.mean(alignments)),
            "mean_best_choice_correlation": float(np.mean(choice_corrs)),
            "n_with_significant_choice": sum(1 for r in all_results if r["best_choice_component"]["p"] < 0.05),
        }

    out_path = RESULTS_DIR / "latent_causal_discovery.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Saved to {out_path}")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-sessions", type=int, default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    run(max_sessions=args.max_sessions)
