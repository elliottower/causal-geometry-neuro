"""Experiment 15: Communication subspace sheaf (Steinmetz).

Replace the failed cross-correlation restriction maps (exp7) with
reduced-rank regression (RRR) communication subspaces (Semedo et al. 2019).

The communication subspace from region A→B is the subspace of A's activity
that best predicts B's activity. Using these as sheaf restriction maps
makes H¹ well-posed: it measures whether information routing around
a circuit loop (e.g., MOs→ACA→MOp→MOs) is globally coherent.

If H¹ ≈ 0: communication is coherent around loops.
If H¹ >> 0: there's an information bottleneck or routing inconsistency.

Key difference from exp7: restriction maps are predictive (RRR), not
correlative (noisy cross-region projections between different-dimensional
spaces).
"""
import json
import logging
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.linalg import svd
from tqdm import tqdm

from data.steinmetz import get_choice_labels, get_region_activity, list_regions, load_all

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent.parent / "artifacts" / "exp15"
COMM_RANK = 3
MIN_NEURONS = 10
TIME_WINDOW = slice(15, 35)


def _reduced_rank_regression(X, Y, rank=3):
    """Reduced-rank regression: find the rank-r subspace of X that best predicts Y.

    Returns:
        B_rrr: (n_x, rank) — the communication subspace basis in X's neuron space
        explained_var: float — fraction of Y variance explained
    """
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
    """Compute H¹ of the communication subspace sheaf on a complete graph.

    Each edge (i,j) has a restriction map R_{ij}: subspace of region i → region j.
    H¹ measures global inconsistency of the sheaf.
    """
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
            rank_c = min(composed.shape[0], composed.shape[1], R_ik.shape[0], R_ik.shape[1])
            if rank_c > 0:
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
    sessions = load_all()
    if max_sessions:
        sessions = sessions[:max_sessions]

    session_results = []

    for sess_idx, sess in enumerate(tqdm(sessions, desc="Sessions")):
        labels = get_choice_labels(sess)
        if len(np.unique(labels)) < 2:
            continue

        mouse = str(sess.get("mouse_name", f"mouse_{sess_idx}"))
        regions = list_regions(sess, min_neurons=MIN_NEURONS)

        region_activities = {}
        for region in regions:
            act = get_region_activity(sess, region)
            if act is None or act.shape[1] < MIN_NEURONS:
                continue
            n = min(act.shape[0], len(labels))
            activity = act[:n, :, TIME_WINDOW].mean(axis=2)
            region_activities[region] = activity

        if len(region_activities) < 3:
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
            except Exception as e:
                logger.warning(f"RRR failed {r1}→{r2}: {e}")

        h1_real = _compute_h1(restriction_maps, region_list)

        n_shuffles = 20
        h1_shuffles = []
        for _ in range(n_shuffles):
            shuffled_maps = {}
            for (r1, r2), basis in restriction_maps.items():
                random_basis = np.linalg.qr(np.random.randn(*basis.shape))[0]
                shuffled_maps[(r1, r2)] = random_basis[:, :basis.shape[1]]
            h1_shuf = _compute_h1(shuffled_maps, region_list)
            if h1_shuf is not None:
                h1_shuffles.append(h1_shuf)

        session_results.append({
            "session_idx": sess_idx,
            "mouse": mouse,
            "n_regions": len(region_list),
            "n_comm_subspaces": len(comm_subspaces),
            "h1_real": h1_real,
            "h1_shuffle_mean": float(np.mean(h1_shuffles)) if h1_shuffles else None,
            "h1_shuffle_std": float(np.std(h1_shuffles)) if h1_shuffles else None,
            "meaningful": h1_real is not None and h1_shuffles and h1_real < np.mean(h1_shuffles) - 2 * np.std(h1_shuffles),
            "comm_subspace_stats": {
                f"{r1}→{r2}": v for (r1, r2), v in comm_subspaces.items()
            },
        })

    results = {
        "timestamp": datetime.now().isoformat(),
        "n_sessions": len(session_results),
        "comm_rank": COMM_RANK,
        "sessions": session_results,
    }

    out_path = RESULTS_DIR / "communication_subspace_sheaf.json"
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
