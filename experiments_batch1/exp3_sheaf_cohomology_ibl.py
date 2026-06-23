"""Experiment 3: Sheaf cohomology of the IBL choice circuit.

Define the circuit sheaf over the 15-20 regions most reliably encoding choice
across animals. Compute H⁰ and H¹ using Čech cohomology over the region graph
with estimated effective connectivity as restriction maps.

Prediction: H¹ ≠ 0 for the full choice circuit; frontal-motor sub-circuits
have H⁰ ≠ 0 (locally localizable).

Falsification: H¹ = 0, implying the circuit is fully localizable to a single region.

Connection to dark matter: If H¹ ≠ 0, predict that the dark matter ratio
is proportional to dim(H¹).
"""
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from tqdm import tqdm

from data.ibl import bin_spikes, filter_by_region, find_sessions_for_region, load_session
from geometry.sheaf import CircuitSheaf
from geometry.subspace import fit_pca_subspace

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent.parent / "artifacts" / "exp3"

CANDIDATE_REGIONS = [
    "MOs", "ACA", "PL", "ILA", "ORBm",  # frontal
    "MOp", "SSp-bfd", "SSp-tr",          # sensorimotor
    "CP", "GPe", "SNr",                   # basal ganglia
    "SCm", "MRN",                         # midbrain
    "VISp", "VISl",                        # visual
]
FRONTAL_MOTOR = {"MOs", "ACA", "PL", "ILA", "ORBm", "MOp"}
MIN_NEURONS = 15
SUBSPACE_K = 5
TIME_WINDOW = slice(50, 100)
DECODE_THRESHOLD = 0.55


def _decode_accuracy(activity: np.ndarray, labels: np.ndarray) -> float:
    clf = LogisticRegression(max_iter=1000, solver="lbfgs")
    scores = cross_val_score(clf, activity, labels, cv=5, scoring="accuracy")
    return float(scores.mean())


def _estimate_connectivity_from_sessions(
    region_activities: dict[str, list[np.ndarray]],
    region_a: str,
    region_b: str,
) -> np.ndarray | None:
    """Estimate effective connectivity between two regions from cross-session data.

    Uses trial-averaged activity correlations as a proxy for effective connectivity.
    In the IBL case, we don't have simultaneous multi-region recordings per session,
    so we use across-session trial-averaged covariance as a proxy.
    """
    acts_a = region_activities.get(region_a, [])
    acts_b = region_activities.get(region_b, [])
    if not acts_a or not acts_b:
        return None

    mean_a = np.mean([a.mean(axis=0) for a in acts_a], axis=0)
    mean_b = np.mean([a.mean(axis=0) for a in acts_b], axis=0)

    n_a, n_b = len(mean_a), len(mean_b)
    W = np.outer(mean_b, mean_a)
    W /= max(np.linalg.norm(W), 1e-10)
    return W


def run(max_sessions_per_region: int = 20):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    region_subspaces = {}
    region_activities = {}
    region_decode_acc = {}

    for region in tqdm(CANDIDATE_REGIONS, desc="Regions"):
        sessions_meta = find_sessions_for_region(region, min_neurons=MIN_NEURONS)
        sessions_meta = sessions_meta[:max_sessions_per_region]

        subspaces = []
        activities = []
        accs = []

        for sess in tqdm(sessions_meta, desc=f"  {region}", leave=False):
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
                if len(np.unique(labels)) < 2:
                    continue

                acc = _decode_accuracy(activity, labels)
                accs.append(acc)

                if acc >= DECODE_THRESHOLD:
                    U = fit_pca_subspace(activity, labels, k=min(SUBSPACE_K, activity.shape[1] - 1))
                    subspaces.append(U)
                    activities.append(activity)
            except Exception as e:
                logger.warning(f"Failed {region}/{sess['eid']}: {e}")

        if subspaces:
            mean_k = min(s.shape[1] for s in subspaces)
            avg_subspace = np.mean([s[:, :mean_k] for s in subspaces], axis=0)
            Q, _ = np.linalg.qr(avg_subspace)
            region_subspaces[region] = Q[:, :mean_k]
            region_activities[region] = activities
            region_decode_acc[region] = float(np.mean(accs))
            logger.info(f"{region}: {len(subspaces)} sessions, decode={np.mean(accs):.3f}")

    choice_regions = [r for r in region_subspaces if region_decode_acc.get(r, 0) >= DECODE_THRESHOLD]
    logger.info(f"\nChoice-encoding regions (acc >= {DECODE_THRESHOLD}): {choice_regions}")

    if len(choice_regions) < 3:
        logger.error("Too few choice-encoding regions for sheaf cohomology")
        return None

    sheaf_full = CircuitSheaf()
    for r in choice_regions:
        sheaf_full.add_region(r, region_subspaces[r])

    for i, r1 in enumerate(choice_regions):
        for r2 in choice_regions[i + 1:]:
            W = _estimate_connectivity_from_sessions(region_activities, r1, r2)
            if W is not None:
                sheaf_full.add_connection(r1, r2, W)
                W_rev = _estimate_connectivity_from_sessions(region_activities, r2, r1)
                if W_rev is not None:
                    sheaf_full.add_connection(r2, r1, W_rev)

    h0_full, h1_full = sheaf_full.compute_cohomology()

    frontal_regions = [r for r in choice_regions if r in FRONTAL_MOTOR]
    h0_frontal, h1_frontal = 0, 0
    if len(frontal_regions) >= 2:
        sheaf_frontal = CircuitSheaf()
        for r in frontal_regions:
            sheaf_frontal.add_region(r, region_subspaces[r])
        for (s, t), W in sheaf_full.connections.items():
            if s in frontal_regions and t in frontal_regions:
                sheaf_frontal.add_connection(s, t, W)
        h0_frontal, h1_frontal = sheaf_frontal.compute_cohomology()

    results = {
        "timestamp": datetime.now().isoformat(),
        "k": SUBSPACE_K,
        "decode_threshold": DECODE_THRESHOLD,
        "choice_regions": choice_regions,
        "region_decode_accuracy": region_decode_acc,
        "full_circuit": {
            "n_regions": len(choice_regions),
            "h0": h0_full,
            "h1": h1_full,
            "is_localizable": h0_full > 0,
            "is_distributed": h1_full > 0,
        },
        "frontal_motor": {
            "regions": frontal_regions,
            "h0": h0_frontal,
            "h1": h1_frontal,
            "is_localizable": h0_frontal > 0,
            "is_distributed": h1_frontal > 0,
        },
    }

    out_path = RESULTS_DIR / "ibl_sheaf_cohomology.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nFull circuit: H⁰={h0_full}, H¹={h1_full}")
    logger.info(f"Frontal-motor: H⁰={h0_frontal}, H¹={h1_frontal}")
    logger.info(f"Saved to {out_path}")
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--max-sessions", type=int, default=20)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    run(max_sessions_per_region=args.max_sessions)
