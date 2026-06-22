"""Experiment 1: Cross-animal choice subspace stability (IBL).

For each region with >20 neurons per session, fit causal subspaces for the
choice variable. Compute pairwise Grassmannian distances across animals and labs.

Test: d_G(same animal, diff session) < d_G(diff animal, same lab) < d_G(diff lab)

This establishes the biological vs methodological noise floor.
"""
import json
import logging
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np
from tqdm import tqdm

from data.ibl import bin_spikes, filter_by_region, find_sessions_for_region, load_session
from geometry.distances import cka, grassmannian_distance
from geometry.subspace import fit_das_subspace, fit_lda_subspace, fit_pca_subspace

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent.parent / "artifacts" / "exp1"
TARGET_REGIONS = ["MOs", "ACA", "PL", "MOp", "SSp-bfd", "CP", "GPe"]
MIN_NEURONS = 20
SUBSPACE_K = 5
TIME_WINDOW = slice(50, 100)  # 500-1000ms post-stimulus onset (10ms bins)


def run(
    regions: list[str] | None = None,
    max_sessions: int | None = None,
    method: str = "lda",
    device: str = "cpu",
):
    regions = regions or TARGET_REGIONS
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = {
        "timestamp": datetime.now().isoformat(),
        "method": method,
        "k": SUBSPACE_K,
        "regions": {},
    }

    for region in regions:
        logger.info(f"\n=== Region: {region} ===")
        sessions_meta = find_sessions_for_region(region, min_neurons=MIN_NEURONS)
        if max_sessions:
            sessions_meta = sessions_meta[:max_sessions]

        subspaces = []
        metadata = []

        for sess in tqdm(sessions_meta, desc=f"Loading {region}"):
            try:
                data = load_session(sess["eid"])
                counts = bin_spikes(
                    data["spike_times"],
                    data["spike_clusters"],
                    data["trial_intervals"],
                )
                region_counts = filter_by_region(counts, data["cluster_regions"], region)

                if region_counts.shape[1] < MIN_NEURONS:
                    continue

                activity = region_counts[:, :, TIME_WINDOW].mean(axis=2)
                labels = (data["trial_choice"] > 0).astype(int)

                if method == "das":
                    U = fit_das_subspace(activity, labels, k=SUBSPACE_K, device=device)
                elif method == "lda":
                    U = fit_lda_subspace(activity, labels, k=SUBSPACE_K)
                else:
                    U = fit_pca_subspace(activity, labels, k=SUBSPACE_K)

                subspaces.append(U)
                metadata.append(sess)
            except Exception as e:
                logger.warning(f"Failed session {sess['eid']}: {e}")
                continue

        if len(subspaces) < 2:
            logger.warning(f"Not enough sessions for {region}")
            continue

        distances = {"within_animal": [], "within_lab": [], "across_lab": []}
        cka_values = {"within_animal": [], "within_lab": [], "across_lab": []}

        for (i, j) in combinations(range(len(subspaces)), 2):
            d_g = grassmannian_distance(subspaces[i], subspaces[j])
            mi, mj = metadata[i], metadata[j]

            if mi["subject"] == mj["subject"]:
                cat = "within_animal"
            elif mi["lab"] == mj["lab"]:
                cat = "within_lab"
            else:
                cat = "across_lab"

            distances[cat].append(d_g)

        region_result = {
            "n_sessions": len(subspaces),
            "distances": {k: {"mean": np.mean(v), "std": np.std(v), "n": len(v)}
                          for k, v in distances.items() if len(v) > 0},
        }
        results["regions"][region] = region_result
        logger.info(f"{region}: {len(subspaces)} sessions")
        for cat, vals in distances.items():
            if vals:
                logger.info(f"  {cat}: d_G = {np.mean(vals):.3f} ± {np.std(vals):.3f} (n={len(vals)})")

    out_path = RESULTS_DIR / "cross_animal_stability.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"\nSaved results to {out_path}")
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--regions", nargs="+", default=None)
    parser.add_argument("--max-sessions", type=int, default=None)
    parser.add_argument("--method", choices=["pca", "lda", "das"], default="lda")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    run(regions=args.regions, max_sessions=args.max_sessions, method=args.method, device=args.device)
