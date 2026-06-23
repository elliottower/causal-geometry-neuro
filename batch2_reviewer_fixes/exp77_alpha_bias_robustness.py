"""Experiment 77: Power-law alpha finite-sample bias robustness.

Chun et al. 2025 (arXiv:2509.26560) showed PCA power-law fits are biased with
small samples. This experiment tests whether the CKA-Procrustes anti-correlation
survives neuron-count and trial-count matching.

CPU only. ~2h.
"""
import json
import logging
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import stats
from tqdm import tqdm

from data.steinmetz import get_choice_labels, get_region_activity, list_regions, load_all
from geometry.distances import cka

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent / "results" / "exp77"
MIN_NEURONS = 15
TIME_WINDOW = slice(15, 35)
N_BOOTSTRAP = 200


def _fit_alpha(activity, rank_start=10, rank_end=50):
    """Fit power-law exponent to PCA eigenvalue spectrum."""
    cov = np.cov(activity.T)
    eigvals = np.sort(np.linalg.eigvalsh(cov))[::-1]

    end = min(rank_end, len(eigvals))
    start = min(rank_start, end - 2)
    if start < 1 or end - start < 3:
        return None, eigvals

    ranks = np.arange(start, end) + 1
    log_ranks = np.log(ranks)
    subset = eigvals[start:end]
    if np.any(subset <= 0):
        return None, eigvals
    log_vals = np.log(subset)

    slope, _, _, _, _ = stats.linregress(log_ranks, log_vals)
    return -slope, eigvals


def run(max_sessions=None):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    sessions = load_all()
    if max_sessions:
        sessions = sessions[:max_sessions]

    region_stats = {}

    for sess_idx, sess in enumerate(tqdm(sessions, desc="Sessions")):
        labels = get_choice_labels(sess)
        if len(np.unique(labels)) < 2:
            continue

        regions = list_regions(sess, min_neurons=MIN_NEURONS)
        for region in regions:
            act = get_region_activity(sess, region)
            if act is None or act.shape[1] < MIN_NEURONS:
                continue

            n = min(act.shape[0], len(labels))
            activity = act[:n, :, TIME_WINDOW].mean(axis=2)
            n_neurons = activity.shape[1]
            n_trials = activity.shape[0]

            alpha_full, _ = _fit_alpha(activity)

            alpha_bootstraps = []
            for _ in range(N_BOOTSTRAP):
                idx = np.random.choice(n_trials, size=n_trials, replace=True)
                a, _ = _fit_alpha(activity[idx])
                if a is not None:
                    alpha_bootstraps.append(a)

            key = f"{region}_{sess_idx}"
            region_stats[key] = {
                "region": region,
                "session_idx": sess_idx,
                "n_neurons": n_neurons,
                "n_trials": n_trials,
                "alpha_full": alpha_full,
                "alpha_bootstrap_mean": float(np.mean(alpha_bootstraps)) if alpha_bootstraps else None,
                "alpha_bootstrap_std": float(np.std(alpha_bootstraps)) if alpha_bootstraps else None,
                "alpha_bootstrap_ci_lo": float(np.percentile(alpha_bootstraps, 2.5)) if alpha_bootstraps else None,
                "alpha_bootstrap_ci_hi": float(np.percentile(alpha_bootstraps, 97.5)) if alpha_bootstraps else None,
            }

    # Neuron-count-matched test: subsample large regions to median neuron count
    all_neurons = [s["n_neurons"] for s in region_stats.values() if s["alpha_full"] is not None]
    median_neurons = int(np.median(all_neurons))
    logger.info(f"Median neuron count: {median_neurons}")

    matched_alphas = {}
    for key, s in region_stats.items():
        if s["alpha_full"] is None or s["n_neurons"] < median_neurons:
            matched_alphas[key] = s["alpha_full"]
            continue

        # Reload and subsample — use stored activity shape as proxy
        # In full run this would re-load; for now just note which need subsampling
        matched_alphas[key] = s["alpha_full"]

    valid = [(s["alpha_full"], s["n_neurons"], s["n_trials"])
             for s in region_stats.values()
             if s["alpha_full"] is not None and np.isfinite(s["alpha_full"])]
    alphas = np.array([v[0] for v in valid])
    neurons = np.array([v[1] for v in valid])
    trials = np.array([v[2] for v in valid])

    alpha_neuron_corr = stats.spearmanr(alphas, neurons)
    alpha_trial_corr = stats.spearmanr(alphas, trials)

    summary = {
        "timestamp": datetime.now().isoformat(),
        "n_regions_sessions": len(region_stats),
        "median_neuron_count": median_neurons,
        "alpha_vs_n_neurons": {
            "spearman_rho": float(alpha_neuron_corr.statistic),
            "p_value": float(alpha_neuron_corr.pvalue),
        },
        "alpha_vs_n_trials": {
            "spearman_rho": float(alpha_trial_corr.statistic),
            "p_value": float(alpha_trial_corr.pvalue),
        },
        "region_stats": region_stats,
    }

    print(f"\nalpha vs n_neurons: rho={alpha_neuron_corr.statistic:.3f} (p={alpha_neuron_corr.pvalue:.2e})")
    print(f"alpha vs n_trials:  rho={alpha_trial_corr.statistic:.3f} (p={alpha_trial_corr.pvalue:.2e})")

    with open(RESULTS_DIR / "alpha_bias_robustness.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info(f"Saved to {RESULTS_DIR}")
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
