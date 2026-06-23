"""Experiment 18: Grassmannian parcellation (Steinmetz).

Redefine brain regions by their representational similarity rather than
anatomical labels. Two regions with similar choice representations belong
to the same "causal parcel" even if anatomically distant.

Uses CKA (dimension-independent) to compute pairwise similarity between
all region pairs, then hierarchical clusters. Compares resulting
parcellation to anatomical atlas.

Novel claim: CKA-based parcellation reveals functional groupings
that anatomical boundaries miss.
"""
import json
import logging
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from tqdm import tqdm

from data.steinmetz import get_choice_labels, get_region_activity, list_regions, load_all
from geometry.distances import cka

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent.parent / "artifacts" / "exp18"
MIN_NEURONS = 10
TIME_WINDOW = slice(15, 35)


def run(max_sessions: int | None = None):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    sessions = load_all()
    if max_sessions:
        sessions = sessions[:max_sessions]

    region_activities = {}

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

            if region not in region_activities:
                region_activities[region] = []
            region_activities[region].append({
                "session_idx": sess_idx,
                "activity": activity,
                "n_neurons": activity.shape[1],
                "n_trials": n,
            })

    regions_with_multiple = [r for r, v in region_activities.items() if len(v) >= 2]
    logger.info(f"{len(regions_with_multiple)} regions with >= 2 sessions")

    if len(regions_with_multiple) < 3:
        results = {
            "timestamp": datetime.now().isoformat(),
            "n_regions": len(region_activities),
            "n_usable_regions": len(regions_with_multiple),
            "error": "Not enough regions with multiple sessions",
        }
        out_path = RESULTS_DIR / "grassmannian_parcellation.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        return results

    mean_cka = {}
    for r1, r2 in combinations(regions_with_multiple, 2):
        cka_vals = []
        for m1 in region_activities[r1]:
            for m2 in region_activities[r2]:
                n_shared = min(m1["n_trials"], m2["n_trials"])
                c = cka(m1["activity"][:n_shared], m2["activity"][:n_shared])
                cka_vals.append(c)
        if cka_vals:
            mean_cka[(r1, r2)] = float(np.mean(cka_vals))

    n = len(regions_with_multiple)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            key = (regions_with_multiple[i], regions_with_multiple[j])
            key_rev = (regions_with_multiple[j], regions_with_multiple[i])
            c = mean_cka.get(key, mean_cka.get(key_rev, 0.0))
            d = 1.0 - c
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d

    condensed = squareform(dist_matrix)
    Z = linkage(condensed, method="ward")

    parcellations = {}
    for n_clusters in [3, 5, 7, 10]:
        if n_clusters >= n:
            continue
        cluster_labels = fcluster(Z, t=n_clusters, criterion="maxclust")
        parcellation = {}
        for idx, region in enumerate(regions_with_multiple):
            cluster_id = int(cluster_labels[idx])
            if cluster_id not in parcellation:
                parcellation[cluster_id] = []
            parcellation[cluster_id].append(region)
        parcellations[f"k={n_clusters}"] = parcellation

    results = {
        "timestamp": datetime.now().isoformat(),
        "n_total_regions": len(region_activities),
        "n_parcellated_regions": n,
        "regions": regions_with_multiple,
        "cka_matrix": {f"{r1}→{r2}": c for (r1, r2), c in mean_cka.items()},
        "parcellations": parcellations,
    }

    out_path = RESULTS_DIR / "grassmannian_parcellation.json"
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
