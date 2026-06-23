"""Experiment 7: Multi-region sheaf cohomology of the choice circuit (Steinmetz).

With 42 simultaneously recorded regions, compute the full circuit sheaf
for choice encoding. Estimate restriction maps from spike-count
cross-correlations at short lags (5-10ms).

Prediction: frontal-motor sub-sheaf is localizable (H⁰ ≠ 0, H¹ = 0);
full 42-region circuit is not (H¹ ≠ 0).
"""
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
from tqdm import tqdm

from data.steinmetz import get_choice_labels, get_region_activity, list_regions, load_all
from geometry.sheaf import CircuitSheaf
from geometry.subspace import fit_pca_subspace

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent.parent / "artifacts" / "exp7"
SUBSPACE_K = 5
MIN_NEURONS = 15
FRONTAL_MOTOR = {"MOs", "MOp", "ACA", "PL", "ILA", "ORBm"}
CC_LAG_BINS = 1  # cross-correlation lag in 10ms bins


def estimate_connectivity(
    activity_source: np.ndarray,
    activity_target: np.ndarray,
    lag: int = 1,
) -> np.ndarray:
    """Estimate effective connectivity from lagged cross-correlation.

    Args:
        activity_source: (n_trials, n_source, n_bins)
        activity_target: (n_trials, n_target, n_bins)
        lag: number of bins for lagged correlation

    Returns:
        (n_target, n_source) estimated connectivity matrix
    """
    n_trials, n_source, n_bins = activity_source.shape
    n_target = activity_target.shape[1]

    source_flat = activity_source[:, :, : n_bins - lag].reshape(-1, n_source).T
    target_flat = activity_target[:, :, lag:].reshape(-1, n_target).T

    source_centered = source_flat - source_flat.mean(axis=1, keepdims=True)
    target_centered = target_flat - target_flat.mean(axis=1, keepdims=True)

    source_std = source_centered.std(axis=1, keepdims=True) + 1e-10
    target_std = target_centered.std(axis=1, keepdims=True) + 1e-10

    corr = (target_centered / target_std) @ (source_centered / source_std).T / source_flat.shape[1]
    return corr


def run(max_sessions: int | None = None):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    sessions = load_all()
    if max_sessions:
        sessions = sessions[:max_sessions]

    all_results = []

    for sess_idx, sess in enumerate(tqdm(sessions, desc="Sessions")):
        regions = list_regions(sess, min_neurons=MIN_NEURONS)
        if len(regions) < 3:
            continue

        labels = get_choice_labels(sess)
        if len(np.unique(labels)) < 2:
            continue

        activities = {}
        subspaces = {}

        for region in regions:
            act = get_region_activity(sess, region)
            if act is None or act.shape[1] < MIN_NEURONS:
                continue

            n = min(act.shape[0], len(labels))
            act = act[:n]
            trial_avg = act.mean(axis=2)

            try:
                U = fit_pca_subspace(trial_avg, labels[:n], k=min(SUBSPACE_K, act.shape[1] - 1))
                subspaces[region] = U
                activities[region] = act
            except Exception as e:
                logger.warning(f"PCA failed for {region}: {e}")

        if len(subspaces) < 3:
            continue

        sheaf = CircuitSheaf()
        for region, U in subspaces.items():
            sheaf.add_region(region, U)

        region_names = sorted(subspaces.keys())
        for i, r1 in enumerate(region_names):
            for r2 in region_names[i + 1 :]:
                try:
                    W = estimate_connectivity(activities[r1], activities[r2], lag=CC_LAG_BINS)
                    sheaf.add_connection(r1, r2, W)
                    W_rev = estimate_connectivity(activities[r2], activities[r1], lag=CC_LAG_BINS)
                    sheaf.add_connection(r2, r1, W_rev)
                except Exception as e:
                    logger.warning(f"Connectivity {r1}-{r2} failed: {e}")

        h0_full, h1_full = sheaf.compute_cohomology()

        frontal_regions = [r for r in subspaces if r in FRONTAL_MOTOR]
        h0_frontal, h1_frontal = 0, 0
        if len(frontal_regions) >= 2:
            frontal_sheaf = CircuitSheaf()
            for r in frontal_regions:
                frontal_sheaf.add_region(r, subspaces[r])
            for (s, t), W in sheaf.connections.items():
                if s in frontal_regions and t in frontal_regions:
                    frontal_sheaf.add_connection(s, t, W)
            h0_frontal, h1_frontal = frontal_sheaf.compute_cohomology()

        result = {
            "session_idx": sess_idx,
            "mouse": str(sess.get("mouse_name", "unknown")),
            "n_regions": len(subspaces),
            "regions": list(subspaces.keys()),
            "full_circuit": {"h0": h0_full, "h1": h1_full},
            "frontal_motor": {
                "h0": h0_frontal,
                "h1": h1_frontal,
                "regions": frontal_regions,
            },
        }
        all_results.append(result)
        logger.info(
            f"Session {sess_idx}: {len(subspaces)} regions, "
            f"H⁰={h0_full} H¹={h1_full} (full), "
            f"H⁰={h0_frontal} H¹={h1_frontal} (frontal)"
        )

    out = {
        "timestamp": datetime.now().isoformat(),
        "k": SUBSPACE_K,
        "sessions": all_results,
    }
    out_path = RESULTS_DIR / "sheaf_cohomology.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    logger.info(f"Saved to {out_path}")
    return out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--max-sessions", type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    run(max_sessions=args.max_sessions)
