"""Experiment 15b: Communication subspace sheaf (IBL).

Same analysis as exp15 but on IBL Brain-Wide Map data which has
100+ neurons per region (vs 10-30 in Steinmetz). This gives RRR
enough degrees of freedom to find real communication subspaces.

Uses IBL sessions with regions: MOs, ACA, MOp, PL, ILA (frontal motor/decision).
These form natural loops for sheaf cohomology testing.
"""
import json
import logging
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.linalg import svd
from tqdm import tqdm

from data.ibl import bin_spikes, filter_by_region, get_one, load_session

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent.parent / "artifacts" / "exp15b"
COMM_RANK = 5
MIN_NEURONS = 30
TARGET_REGIONS = ["MOs", "ACA", "MOp", "PL", "ILA", "VISp", "VISa", "CA1", "DG"]


def _reduced_rank_regression(X, Y, rank=5):
    """Reduced-rank regression: find the rank-r subspace of X that best predicts Y."""
    n = min(X.shape[0], Y.shape[0])
    X, Y = X[:n], Y[:n]

    X_mean = X.mean(axis=0, keepdims=True)
    Y_mean = Y.mean(axis=0, keepdims=True)
    X_c = X - X_mean
    Y_c = Y - Y_mean

    B_ols = np.linalg.lstsq(X_c, Y_c, rcond=None)[0]

    U, S, Vt = svd(Y_c @ B_ols.T, full_matrices=False)
    rank = min(rank, len(S), X.shape[1], Y.shape[1])

    B_rrr = B_ols @ Vt[:rank].T

    Y_pred = X_c @ B_rrr @ Vt[:rank]
    ss_res = np.sum((Y_c - Y_pred) ** 2)
    ss_tot = np.sum(Y_c ** 2)
    explained_var = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    Q, _ = np.linalg.qr(B_rrr)
    return Q[:, :rank], float(explained_var)


def _compute_h1(restriction_maps, regions):
    """Compute H1 of the communication subspace sheaf on triangles."""
    n_regions = len(regions)
    if n_regions < 3:
        return None

    loop_inconsistencies = []
    for i, j, k in combinations(range(n_regions), 3):
        key_ij = (regions[i], regions[j])
        key_jk = (regions[j], regions[k])
        key_ik = (regions[i], regions[k])

        if key_ij in restriction_maps and key_jk in restriction_maps and key_ik in restriction_maps:
            R_ij = restriction_maps[key_ij]
            R_jk = restriction_maps[key_jk]
            R_ik = restriction_maps[key_ik]

            composed = R_ij @ R_jk
            direct = R_ik[:composed.shape[0], :composed.shape[1]] if R_ik.shape != composed.shape else R_ik
            try:
                inconsistency = np.linalg.norm(composed - direct, 'fro')
                loop_inconsistencies.append(float(inconsistency))
            except Exception:
                pass

    if not loop_inconsistencies:
        return None
    return float(np.mean(loop_inconsistencies))


def run(max_sessions: int | None = None):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    one = get_one()

    session_regions = {}
    for region in tqdm(TARGET_REGIONS, desc="Finding sessions per region"):
        try:
            pids = one.search_insertions(
                atlas_acronym=region,
                datasets="spikes.times.npy",
                project="brainwide",
            )
            for pid in pids:
                try:
                    info = one.alyx.rest("insertions", "read", id=pid)
                    eid = info["session"]
                    subject = info.get("session_info", {}).get("subject", "unknown")
                    if eid not in session_regions:
                        session_regions[eid] = {"regions": set(), "subject": subject}
                    session_regions[eid]["regions"].add(region)
                except Exception:
                    pass
            logger.info(f"  {region}: {len(pids)} insertions")
        except Exception as e:
            logger.warning(f"Failed searching {region}: {e}")

    multi_region_sessions = {eid: info for eid, info in session_regions.items() if len(info["regions"]) >= 3}
    logger.info(f"{len(multi_region_sessions)} sessions with >= 3 target regions (from {len(session_regions)} total)")

    if max_sessions:
        eids = sorted(multi_region_sessions.keys(), key=lambda e: len(multi_region_sessions[e]["regions"]), reverse=True)[:max_sessions]
    else:
        eids = list(multi_region_sessions.keys())

    session_results = []

    for eid in tqdm(eids, desc="Sessions"):
        sinfo = multi_region_sessions[eid]
        subject = sinfo["subject"]
        target = sorted(sinfo["regions"])

        try:
            data = load_session(eid)
        except Exception as e:
            logger.warning(f"Failed loading session {eid}: {e}")
            continue

        counts = bin_spikes(
            data["spike_times"], data["spike_clusters"],
            data["trial_intervals"],
            bin_size=0.025, pre_time=0.2, post_time=0.6,
        )

        region_activities = {}
        for region in target:
            region_counts = filter_by_region(counts, data["cluster_regions"], region)
            n_neurons = region_counts.shape[1]
            if n_neurons >= MIN_NEURONS:
                activity = region_counts.mean(axis=2)
                region_activities[region] = activity
                logger.info(f"  {region}: {n_neurons} neurons, {activity.shape[0]} trials")

        if len(region_activities) < 3:
            logger.info(f"  Only {len(region_activities)} regions with >= {MIN_NEURONS} neurons, skipping")
            continue

        restriction_maps = {}
        comm_subspaces = {}
        region_list = sorted(region_activities.keys())

        for r1, r2 in combinations(region_list, 2):
            X = region_activities[r1]
            Y = region_activities[r2]
            n = min(X.shape[0], Y.shape[0])

            try:
                basis, ev = _reduced_rank_regression(X[:n], Y[:n], rank=COMM_RANK)
                restriction_maps[(r1, r2)] = basis
                comm_subspaces[(r1, r2)] = {"explained_var": ev, "rank": basis.shape[1]}
                logger.info(f"  RRR {r1}→{r2}: explained_var={ev:.3f}, rank={basis.shape[1]}")
            except Exception as e:
                logger.warning(f"  RRR failed {r1}→{r2}: {e}")

        h1_real = _compute_h1(restriction_maps, region_list)

        n_shuffles = 50
        h1_shuffles = []
        for _ in range(n_shuffles):
            shuffled_maps = {}
            for (r1, r2), basis in restriction_maps.items():
                random_basis = np.linalg.qr(np.random.randn(*basis.shape))[0]
                shuffled_maps[(r1, r2)] = random_basis[:, :basis.shape[1]]
            h1_shuf = _compute_h1(shuffled_maps, region_list)
            if h1_shuf is not None:
                h1_shuffles.append(h1_shuf)

        meaningful = (
            h1_real is not None
            and h1_shuffles
            and h1_real < np.mean(h1_shuffles) - 2 * np.std(h1_shuffles)
        )

        session_results.append({
            "eid": eid,
            "subject": subject,
            "regions": region_list,
            "n_regions": len(region_list),
            "n_comm_subspaces": len(comm_subspaces),
            "n_neurons_per_region": {r: a.shape[1] for r, a in region_activities.items()},
            "h1_real": h1_real,
            "h1_shuffle_mean": float(np.mean(h1_shuffles)) if h1_shuffles else None,
            "h1_shuffle_std": float(np.std(h1_shuffles)) if h1_shuffles else None,
            "meaningful": meaningful,
            "comm_subspace_stats": {
                f"{r1}→{r2}": v for (r1, r2), v in comm_subspaces.items()
            },
        })

    n_meaningful = sum(1 for s in session_results if s["meaningful"])
    results = {
        "timestamp": datetime.now().isoformat(),
        "dataset": "IBL Brain-Wide Map",
        "comm_rank": COMM_RANK,
        "min_neurons": MIN_NEURONS,
        "target_regions": TARGET_REGIONS,
        "n_sessions": len(session_results),
        "n_meaningful": n_meaningful,
        "sessions": session_results,
    }

    out_path = RESULTS_DIR / "communication_subspace_sheaf_ibl.json"
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
