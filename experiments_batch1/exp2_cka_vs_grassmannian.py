"""Experiment 2: CKA vs Grassmannian distance dissociation (IBL).

For pairs of populations with matched choice-decoding accuracy,
compare linear CKA and gauge-normalized Grassmannian distance.

The framework predicts: high-CKA, high-d_G pairs exist — populations
that encode the same variable with similar geometry but via different
causal subspaces.

Key test: correlation(CKA, d_G) < 1.0 for cross-region pairs.
"""
import json
import logging
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from tqdm import tqdm

from data.ibl import bin_spikes, filter_by_region, find_sessions_for_region, load_session
from geometry.distances import cka, gauge_normalized_distance, grassmannian_distance
from geometry.subspace import fit_lda_subspace

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent.parent / "artifacts" / "exp2"
REGIONS = ["MOs", "ACA", "PL", "MOp", "CP"]
SUBSPACE_K = 5
TIME_WINDOW = slice(50, 100)


def decode_accuracy(activity: np.ndarray, labels: np.ndarray) -> float:
    """5-fold cross-validated logistic regression accuracy."""
    clf = LogisticRegression(max_iter=1000, solver="lbfgs")
    scores = cross_val_score(clf, activity, labels, cv=5, scoring="accuracy")
    return float(scores.mean())


def run(max_sessions: int | None = None):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = {
        "timestamp": datetime.now().isoformat(),
        "pairs": [],
    }

    region_data = {}
    for region in REGIONS:
        logger.info(f"Loading {region}...")
        sessions_meta = find_sessions_for_region(region, min_neurons=20)
        if max_sessions:
            sessions_meta = sessions_meta[:max_sessions]

        for sess in tqdm(sessions_meta, desc=region):
            try:
                data = load_session(sess["eid"])
                counts = bin_spikes(
                    data["spike_times"],
                    data["spike_clusters"],
                    data["trial_intervals"],
                )
                region_counts = filter_by_region(counts, data["cluster_regions"], region)
                if region_counts.shape[1] < 20:
                    continue

                activity = region_counts[:, :, TIME_WINDOW].mean(axis=2)
                labels = (data["trial_choice"] > 0).astype(int)
                if len(np.unique(labels)) < 2:
                    continue

                acc = decode_accuracy(activity, labels)
                U = fit_lda_subspace(activity, labels, k=SUBSPACE_K)

                key = (region, sess["eid"])
                region_data[key] = {
                    "activity": activity,
                    "labels": labels,
                    "subspace": U,
                    "accuracy": acc,
                    "region": region,
                    "session": sess,
                }
            except Exception as e:
                logger.warning(f"Failed: {e}")

    logger.info(f"Total region-sessions loaded: {len(region_data)}")

    keys = list(region_data.keys())
    for (k1, k2) in tqdm(list(combinations(keys, 2)), desc="Computing distances"):
        d1, d2 = region_data[k1], region_data[k2]

        acc_diff = abs(d1["accuracy"] - d2["accuracy"])
        if acc_diff > 0.1:
            continue

        d_g = grassmannian_distance(d1["subspace"], d2["subspace"])

        n_shared = min(d1["activity"].shape[0], d2["activity"].shape[0])
        cka_val = cka(d1["activity"][:n_shared], d2["activity"][:n_shared])

        pair_type = "within_region" if d1["region"] == d2["region"] else "cross_region"

        results["pairs"].append({
            "region_1": d1["region"],
            "region_2": d2["region"],
            "accuracy_1": d1["accuracy"],
            "accuracy_2": d2["accuracy"],
            "grassmannian_distance": d_g,
            "cka": cka_val,
            "pair_type": pair_type,
        })

    out_path = RESULTS_DIR / "cka_vs_grassmannian.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved {len(results['pairs'])} pairs to {out_path}")
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--max-sessions", type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    run(max_sessions=args.max_sessions)
