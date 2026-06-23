"""Experiment 10: Direction vs subspace diagnostic (Steinmetz).

For each region with enough neurons, compute:
  (a) The choice DIRECTION (top-1 LDA axis) — a point on RP^{n-1}
  (b) The choice SUBSPACE (top-k LDA) — a point on Gr(k, n)

Then for all cross-session pairs of the same region:
  - Cosine similarity of directions (Type 1 metric)
  - Grassmannian distance of subspaces (Type 2 metric)

Prediction: Cosine similarity is near-uniform (directions vary across animals
due to electrode placement / network initialization), but Grassmannian distance
is structured (subspaces are stable within-region across animals).

This proves Type 1 (direction) is the wrong geometric type and Type 2+
(subspace) is necessary.
"""
import json
import logging
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np
from tqdm import tqdm

from data.steinmetz import get_choice_labels, get_region_activity, list_regions, load_all
from geometry.distances import grassmannian_distance
from geometry.subspace import fit_lda_subspace

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent.parent / "artifacts" / "exp10"
SUBSPACE_K = 5
MIN_NEURONS = 15
TIME_WINDOW = slice(15, 35)


def run(max_sessions: int | None = None):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    sessions = load_all()
    if max_sessions:
        sessions = sessions[:max_sessions]

    region_measurements = {}

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
            act = act[:n]
            activity = act[:, :, TIME_WINDOW].mean(axis=2)

            try:
                k = min(SUBSPACE_K, activity.shape[1] - 1)
                U_k = fit_lda_subspace(activity, labels[:n], k=k)
                direction = U_k[:, 0]

                if region not in region_measurements:
                    region_measurements[region] = []
                region_measurements[region].append({
                    "session_idx": sess_idx,
                    "mouse": mouse,
                    "direction": direction,
                    "subspace": U_k,
                    "n_neurons": activity.shape[1],
                })
            except Exception as e:
                logger.warning(f"Failed {mouse}/{region}: {e}")

    direction_cosines = {"same_region": [], "labels": []}
    subspace_distances = {"same_region": [], "labels": []}

    for region, measurements in region_measurements.items():
        if len(measurements) < 2:
            continue

        for (i, j) in combinations(range(len(measurements)), 2):
            m1, m2 = measurements[i], measurements[j]

            if m1["n_neurons"] != m2["n_neurons"]:
                continue

            cos_sim = abs(float(m1["direction"] @ m2["direction"]))
            d_g = grassmannian_distance(m1["subspace"], m2["subspace"])
            same_mouse = m1["mouse"] == m2["mouse"]

            direction_cosines["same_region"].append(cos_sim)
            subspace_distances["same_region"].append(d_g)
            direction_cosines["labels"].append(
                f"{region}|{'same' if same_mouse else 'diff'}_mouse"
            )
            subspace_distances["labels"].append(
                f"{region}|{'same' if same_mouse else 'diff'}_mouse"
            )

    n_random = min(500, len(direction_cosines["same_region"]) * 5)
    random_cosines = []
    all_dirs = []
    for measurements in region_measurements.values():
        for m in measurements:
            all_dirs.append(m["direction"])
    if len(all_dirs) > 1:
        for _ in range(n_random):
            d = all_dirs[0].shape[0]
            v1 = np.random.randn(d)
            v2 = np.random.randn(d)
            v1 /= np.linalg.norm(v1)
            v2 /= np.linalg.norm(v2)
            random_cosines.append(abs(float(v1 @ v2)))

    cos_vals = direction_cosines["same_region"]
    dg_vals = subspace_distances["same_region"]

    results = {
        "timestamp": datetime.now().isoformat(),
        "k": SUBSPACE_K,
        "n_regions": len(region_measurements),
        "n_pairs": len(cos_vals),
        "direction_cosine": {
            "mean": float(np.mean(cos_vals)) if cos_vals else None,
            "std": float(np.std(cos_vals)) if cos_vals else None,
            "n": len(cos_vals),
        },
        "random_cosine": {
            "mean": float(np.mean(random_cosines)) if random_cosines else None,
            "std": float(np.std(random_cosines)) if random_cosines else None,
        },
        "grassmannian_distance": {
            "mean": float(np.mean(dg_vals)) if dg_vals else None,
            "std": float(np.std(dg_vals)) if dg_vals else None,
            "n": len(dg_vals),
        },
        "prediction": "direction cosines near random baseline; Grassmannian distances structured",
        "pairs": [
            {
                "label": direction_cosines["labels"][i],
                "cosine": cos_vals[i],
                "grassmannian": dg_vals[i],
            }
            for i in range(len(cos_vals))
        ],
    }

    if cos_vals and random_cosines:
        from scipy.stats import mannwhitneyu
        stat, p = mannwhitneyu(cos_vals, random_cosines, alternative="greater")
        results["direction_vs_random_test"] = {
            "test": "Mann-Whitney U (greater)",
            "statistic": float(stat),
            "p_value": float(p),
            "interpretation": "p > 0.05 means directions are indistinguishable from random",
        }

    logger.info(f"Direction cosine: {results['direction_cosine']}")
    logger.info(f"Random cosine: {results['random_cosine']}")
    logger.info(f"Grassmannian distance: {results['grassmannian_distance']}")

    out_path = RESULTS_DIR / "direction_vs_subspace.json"
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
