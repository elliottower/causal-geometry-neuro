"""Experiment 8: Holonomy estimation on choice subspaces (Steinmetz).

Using the trial structure of the 2AFC task, estimate holonomy by comparing
the choice subspace at trial onset, mid-deliberation, and post-choice.
The deliberation period provides a natural "loop" for holonomy measurement.

Prediction: Holonomy is stable within a region across animals (mechanism
identity signature) and variable across regions performing the same encoding
(mechanistic pluralism signature).

Falsification: No significant cross-region differences in holonomy, implying
the holonomy fingerprint is not diagnostic.
"""
import json
import logging
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np
from tqdm import tqdm

from data.steinmetz import get_choice_labels, get_region_activity, list_regions, load_all
from geometry.holonomy import estimate_holonomy, holonomy_angle, holonomy_distance

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent.parent / "artifacts" / "exp8"
SUBSPACE_K = 4
MIN_NEURONS = 15
# Time bins in 10ms: pre-stim (0-10), early deliberation (15-25),
# late deliberation (25-35), post-choice (35-45), ITI (45-50)
TIME_POINTS = [
    slice(0, 10),    # pre-stimulus baseline
    slice(15, 25),   # early deliberation
    slice(25, 35),   # late deliberation
    slice(35, 45),   # post-choice
    slice(45, 50),   # return to baseline (ITI)
]
TIME_LABELS = ["pre_stim", "early_delib", "late_delib", "post_choice", "iti"]


def run(max_sessions: int | None = None):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    sessions = load_all()
    if max_sessions:
        sessions = sessions[:max_sessions]

    all_holonomies = []

    for sess_idx, sess in enumerate(tqdm(sessions, desc="Sessions")):
        mouse = str(sess.get("mouse_name", f"mouse_{sess_idx}"))
        labels = get_choice_labels(sess)
        if len(np.unique(labels)) < 2:
            continue

        regions = list_regions(sess, min_neurons=MIN_NEURONS)

        for region in regions:
            act = get_region_activity(sess, region)
            if act is None or act.shape[1] < MIN_NEURONS:
                continue

            n = min(act.shape[0], len(labels))
            act = act[:n]
            time_activities = []
            for t_slice in TIME_POINTS:
                t_act = act[:, :, t_slice].mean(axis=2)  # (n_trials, n_neurons)
                time_activities.append(t_act)

            try:
                H = estimate_holonomy(time_activities, labels[:n], k=min(SUBSPACE_K, act.shape[1] - 1))
                angle = holonomy_angle(H)

                all_holonomies.append({
                    "session_idx": sess_idx,
                    "mouse": mouse,
                    "region": region,
                    "holonomy_angle": float(angle),
                    "holonomy_matrix": H.tolist(),
                    "n_neurons": int(act.shape[1]),
                    "n_trials": int(act.shape[0]),
                })
            except Exception as e:
                logger.warning(f"Holonomy failed for {mouse}/{region}: {e}")

    per_region = {}
    for entry in all_holonomies:
        r = entry["region"]
        if r not in per_region:
            per_region[r] = []
        per_region[r].append(entry)

    region_summary = {}
    for r, entries in per_region.items():
        angles = [e["holonomy_angle"] for e in entries]
        region_summary[r] = {
            "mean_holonomy_angle": float(np.mean(angles)),
            "std_holonomy_angle": float(np.std(angles)),
            "n_sessions": len(entries),
        }

    within_region_dists = []
    cross_region_dists = []

    for (e1, e2) in combinations(all_holonomies, 2):
        H1 = np.array(e1["holonomy_matrix"])
        H2 = np.array(e2["holonomy_matrix"])
        if H1.shape != H2.shape:
            continue
        d = holonomy_distance(H1, H2)

        if e1["region"] == e2["region"]:
            within_region_dists.append({"distance": d, "region": e1["region"]})
        else:
            cross_region_dists.append({
                "distance": d,
                "region_1": e1["region"],
                "region_2": e2["region"],
            })

    def _stats(vals):
        ds = [v["distance"] for v in vals]
        if not ds:
            return {"mean": None, "std": None, "n": 0}
        return {"mean": float(np.mean(ds)), "std": float(np.std(ds)), "n": len(ds)}

    results = {
        "timestamp": datetime.now().isoformat(),
        "k": SUBSPACE_K,
        "time_points": TIME_LABELS,
        "n_total_measurements": len(all_holonomies),
        "region_summary": region_summary,
        "within_region_holonomy_distance": _stats(within_region_dists),
        "cross_region_holonomy_distance": _stats(cross_region_dists),
        "prediction": "within_region < cross_region (mechanism stability within region)",
        "all_holonomies": all_holonomies,
    }

    logger.info(f"\nWithin-region holonomy distance: {_stats(within_region_dists)}")
    logger.info(f"Cross-region holonomy distance: {_stats(cross_region_dists)}")

    out_path = RESULTS_DIR / "holonomy_estimation.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved to {out_path}")
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--max-sessions", type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    run(max_sessions=args.max_sessions)
