"""Experiment 13: Static vs dynamic dissociation (Steinmetz).

For each region, compare:
  (a) Subspace similarity — Grassmannian distance on LDA subspaces (static geometry)
  (b) Trajectory similarity — Procrustes on time-resolved activity paths (dynamics)

Regions where subspace is conserved but trajectory isn't → static mechanism
(same computational subspace, different temporal dynamics).
Regions where trajectory is conserved but subspace isn't → dynamic mechanism
(same temporal pattern, different embedding).

This separates WHERE the mechanism lives (geometry) from HOW it moves (dynamics).
"""
import json
import logging
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.spatial import procrustes
from tqdm import tqdm

from data.steinmetz import get_choice_labels, get_region_activity, list_regions, load_all
from geometry.distances import grassmannian_distance
from geometry.subspace import fit_lda_subspace

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent.parent / "artifacts" / "exp13"
SUBSPACE_K = 5
MIN_NEURONS = 15
N_TIME_BINS = 5
TIME_SLICES = [slice(i * 10, (i + 1) * 10) for i in range(N_TIME_BINS)]


def _extract_trajectory(activity_3d, labels, k=5):
    """Extract time-resolved trajectory in LDA subspace.

    Returns (n_time_bins, k) trajectory — the class-difference centroid
    at each time bin projected into the LDA subspace.
    """
    n = min(activity_3d.shape[0], len(labels))
    activity_3d = activity_3d[:n]
    labels = labels[:n]

    overall_activity = activity_3d.mean(axis=2)
    k = min(k, overall_activity.shape[1] - 1)
    U = fit_lda_subspace(overall_activity, labels, k=k)

    trajectory = np.zeros((len(TIME_SLICES), k))
    for t, ts in enumerate(TIME_SLICES):
        act_t = activity_3d[:, :, ts].mean(axis=2)
        mean_0 = act_t[labels == 0].mean(axis=0)
        mean_1 = act_t[labels == 1].mean(axis=0)
        diff = mean_1 - mean_0
        trajectory[t] = U.T @ diff

    return trajectory, U


def _trajectory_distance(traj1, traj2):
    """Procrustes distance between two trajectories."""
    _, _, disparity = procrustes(traj1, traj2)
    return float(disparity)


def run(max_sessions: int | None = None):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    sessions = load_all()
    if max_sessions:
        sessions = sessions[:max_sessions]

    region_data = {}

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

            try:
                trajectory, U = _extract_trajectory(act, labels[:n], k=SUBSPACE_K)

                if region not in region_data:
                    region_data[region] = []
                region_data[region].append({
                    "session_idx": sess_idx,
                    "mouse": mouse,
                    "subspace": U,
                    "trajectory": trajectory,
                    "n_neurons": act.shape[1],
                })
            except Exception as e:
                logger.warning(f"Failed {mouse}/{region}: {e}")

    pairs = []
    for region, measurements in tqdm(region_data.items(), desc="Regions"):
        if len(measurements) < 2:
            continue

        for (i, j) in combinations(range(len(measurements)), 2):
            m1, m2 = measurements[i], measurements[j]

            if m1["n_neurons"] != m2["n_neurons"]:
                d_grass = None
            else:
                d_grass = grassmannian_distance(m1["subspace"], m2["subspace"])

            d_traj = _trajectory_distance(m1["trajectory"], m2["trajectory"])
            same_mouse = m1["mouse"] == m2["mouse"]

            pairs.append({
                "region": region,
                "mouse_1": m1["mouse"],
                "mouse_2": m2["mouse"],
                "same_mouse": same_mouse,
                "grassmannian_distance": d_grass,
                "trajectory_distance": d_traj,
            })

    results = {
        "timestamp": datetime.now().isoformat(),
        "n_regions": len(region_data),
        "n_pairs": len(pairs),
        "n_time_bins": N_TIME_BINS,
        "pairs": pairs,
    }

    pairs_with_both = [p for p in pairs if p["grassmannian_distance"] is not None]
    if len(pairs_with_both) >= 5:
        from scipy.stats import spearmanr
        d_g = [p["grassmannian_distance"] for p in pairs_with_both]
        d_t = [p["trajectory_distance"] for p in pairs_with_both]
        rho, p_val = spearmanr(d_g, d_t)
        results["geometry_vs_dynamics_correlation"] = {
            "spearman_rho": float(rho),
            "p_value": float(p_val),
            "n": len(pairs_with_both),
            "interpretation": "rho < 1 means static geometry and dynamics dissociate",
        }

        per_region = {}
        for region in region_data:
            rp = [p for p in pairs_with_both if p["region"] == region]
            if len(rp) >= 3:
                rg = [p["grassmannian_distance"] for p in rp]
                rt = [p["trajectory_distance"] for p in rp]
                r, pv = spearmanr(rg, rt)
                per_region[region] = {
                    "n_pairs": len(rp),
                    "spearman_rho": float(r),
                    "grass_mean": float(np.mean(rg)),
                    "traj_mean": float(np.mean(rt)),
                }
        results["per_region"] = per_region

    out_path = RESULTS_DIR / "static_vs_dynamic.json"
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
