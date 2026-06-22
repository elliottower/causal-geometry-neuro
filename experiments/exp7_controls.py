"""Controls for Exp 7: Is H¹ >> 0 trivially true or meaningful?

Three null models:
1. SHUFFLE: Randomly permute trial labels, recompute H¹.
   If H¹_shuffle ≈ H¹_real → result is trivial (restriction maps are noise).
2. SPLIT: Take ONE region, split neurons randomly into two "pseudo-regions,"
   compute H¹. Should be ≈ 0 if the method works on a known-localizable case.
3. RANDOM: Replace all subspaces with random orthonormal bases, keep same
   restriction maps. If H¹_random ≈ H¹_real → the restriction maps are the
   problem, not the subspaces.

If H¹_real >> H¹_shuffle, the result is meaningful.
If H¹_real ≈ H¹_shuffle, the result is an artifact of noisy restriction maps.
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

RESULTS_DIR = Path(__file__).parent.parent / "artifacts" / "exp7_controls"
SUBSPACE_K = 5
MIN_NEURONS = 15
N_SHUFFLES = 20


def _build_sheaf(region_subspaces, region_activities, regions):
    """Build sheaf from pre-computed subspaces and activities."""
    sheaf = CircuitSheaf()
    for r in regions:
        sheaf.add_region(r, region_subspaces[r])

    for i, r1 in enumerate(regions):
        for r2 in regions[i + 1:]:
            act1 = region_activities[r1]
            act2 = region_activities[r2]
            n_trials = min(act1.shape[0], act2.shape[0])
            mean1 = act1[:n_trials].mean(axis=2)
            mean2 = act2[:n_trials].mean(axis=2)
            W = (mean2.T @ mean1) / n_trials
            W /= max(np.linalg.norm(W), 1e-10)
            sheaf.add_connection(r1, r2, W)
            W_rev = (mean1.T @ mean2) / n_trials
            W_rev /= max(np.linalg.norm(W_rev), 1e-10)
            sheaf.add_connection(r2, r1, W_rev)

    return sheaf.compute_cohomology()


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

        regions_list = list_regions(sess, min_neurons=MIN_NEURONS)
        if len(regions_list) < 3:
            continue

        region_subspaces = {}
        region_activities = {}

        for region in regions_list:
            act = get_region_activity(sess, region)
            if act is None or act.shape[1] < MIN_NEURONS:
                continue
            n = min(act.shape[0], len(labels))
            act = act[:n]
            trial_avg = act.mean(axis=2)
            try:
                U = fit_pca_subspace(trial_avg, labels[:n], k=min(SUBSPACE_K, act.shape[1] - 1))
                region_subspaces[region] = U
                region_activities[region] = act
            except Exception:
                continue

        valid_regions = sorted(region_subspaces.keys())
        if len(valid_regions) < 3:
            continue

        # REAL
        h0_real, h1_real = _build_sheaf(region_subspaces, region_activities, valid_regions)

        # CONTROL 1: Shuffle labels
        h1_shuffles = []
        for _ in range(N_SHUFFLES):
            shuffled_labels = labels.copy()
            np.random.shuffle(shuffled_labels)
            shuffled_subspaces = {}
            for region in valid_regions:
                act = region_activities[region]
                n = min(act.shape[0], len(shuffled_labels))
                trial_avg = act[:n].mean(axis=2)
                try:
                    U = fit_pca_subspace(trial_avg, shuffled_labels[:n],
                                         k=min(SUBSPACE_K, act.shape[1] - 1))
                    shuffled_subspaces[region] = U
                except Exception:
                    shuffled_subspaces[region] = region_subspaces[region]
            _, h1_shuf = _build_sheaf(shuffled_subspaces, region_activities, valid_regions)
            h1_shuffles.append(h1_shuf)

        # CONTROL 2: Random subspaces, same restriction maps
        random_subspaces = {}
        for region in valid_regions:
            n_neurons = region_subspaces[region].shape[0]
            k = region_subspaces[region].shape[1]
            Q, _ = np.linalg.qr(np.random.randn(n_neurons, k))
            random_subspaces[region] = Q
        _, h1_random = _build_sheaf(random_subspaces, region_activities, valid_regions)

        # CONTROL 3: Split one large region into two pseudo-regions
        h1_split = None
        largest_region = max(valid_regions, key=lambda r: region_activities[r].shape[1])
        act_largest = region_activities[largest_region]
        n_neurons_largest = act_largest.shape[1]
        if n_neurons_largest >= 2 * MIN_NEURONS:
            perm = np.random.permutation(n_neurons_largest)
            mid = n_neurons_largest // 2
            act_a = act_largest[:, perm[:mid], :]
            act_b = act_largest[:, perm[mid:], :]
            n = min(act_a.shape[0], len(labels))

            trial_avg_a = act_a[:n].mean(axis=2)
            trial_avg_b = act_b[:n].mean(axis=2)
            try:
                U_a = fit_pca_subspace(trial_avg_a, labels[:n], k=min(SUBSPACE_K, mid - 1))
                U_b = fit_pca_subspace(trial_avg_b, labels[:n], k=min(SUBSPACE_K, n_neurons_largest - mid - 1))

                split_sheaf = CircuitSheaf()
                split_sheaf.add_region("pseudo_A", U_a)
                split_sheaf.add_region("pseudo_B", U_b)
                W_ab = (trial_avg_b[:n].T @ trial_avg_a[:n]) / n
                W_ab /= max(np.linalg.norm(W_ab), 1e-10)
                split_sheaf.add_connection("pseudo_A", "pseudo_B", W_ab)
                W_ba = (trial_avg_a[:n].T @ trial_avg_b[:n]) / n
                W_ba /= max(np.linalg.norm(W_ba), 1e-10)
                split_sheaf.add_connection("pseudo_B", "pseudo_A", W_ba)
                _, h1_split = split_sheaf.compute_cohomology()
            except Exception as e:
                logger.warning(f"Split control failed: {e}")

        result = {
            "session_idx": sess_idx,
            "mouse": str(sess.get("mouse_name", f"mouse_{sess_idx}")),
            "n_regions": len(valid_regions),
            "h1_real": h1_real,
            "h1_shuffle_mean": float(np.mean(h1_shuffles)),
            "h1_shuffle_std": float(np.std(h1_shuffles)),
            "h1_shuffle_all": h1_shuffles,
            "h1_random": h1_random,
            "h1_split": h1_split,
            "real_gt_shuffle": h1_real > np.mean(h1_shuffles) + 2 * np.std(h1_shuffles),
        }
        all_results.append(result)

        logger.info(
            f"Session {sess_idx} ({result['mouse']}): "
            f"H1_real={h1_real}, "
            f"H1_shuffle={np.mean(h1_shuffles):.0f}±{np.std(h1_shuffles):.0f}, "
            f"H1_random={h1_random}, "
            f"H1_split={h1_split}, "
            f"MEANINGFUL={result['real_gt_shuffle']}"
        )

    out = {
        "timestamp": datetime.now().isoformat(),
        "n_shuffles": N_SHUFFLES,
        "sessions": all_results,
    }

    out_path = RESULTS_DIR / "exp7_controls.json"
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
