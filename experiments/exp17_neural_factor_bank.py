"""Experiment 17: Neural factor bank (Steinmetz).

Direct neural analog of transformer factorization: does population activity
factorize into a shared basis of reusable temporal modes?

Fit NMF on the population activity matrix (neurons x time), then test:
  - Are temporal basis vectors conserved across animals for the same region?
  - Do different regions use different subsets of the basis?

The mechanism identity claim: two regions sharing basis vectors implement
the same computation; two regions with non-overlapping basis vectors don't.

This is strictly richer than Grassmannian because it decomposes the subspace
into interpretable factors and tests which are universal vs region-specific.
"""
import json
import logging
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.linalg import svd
from scipy.spatial.distance import cosine
from tqdm import tqdm

from data.steinmetz import get_choice_labels, get_region_activity, list_regions, load_all

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent.parent / "artifacts" / "exp17"
N_FACTORS = 5
MIN_NEURONS = 15
TIME_WINDOW = slice(15, 35)


def _fit_svd_factors(activity, n_factors=5):
    """SVD factorization: activity ≈ U @ diag(S) @ V.T

    Returns:
        U: (n_neurons, n_factors) — neural modes (how each neuron loads onto factors)
        V: (n_time_or_trials, n_factors) — temporal/trial modes
        S: (n_factors,) — singular values
    """
    activity_c = activity - activity.mean(axis=0, keepdims=True)
    U, S, Vt = svd(activity_c, full_matrices=False)
    n = min(n_factors, len(S))
    return U[:, :n], S[:n], Vt[:n].T


def _factor_similarity(V1, V2):
    """Compare two sets of temporal modes via best-match cosine similarity.

    V1: (n_trials_1, k), V2: (n_trials_2, k) — may have different trial counts.
    Normalize columns to unit vectors first so dot product = cosine similarity
    regardless of trial count.
    """
    n1, n2 = V1.shape[1], V2.shape[1]
    n = min(n1, n2)

    v1_norm = V1[:, :n] / (np.linalg.norm(V1[:, :n], axis=0, keepdims=True) + 1e-8)
    v2_norm = V2[:, :n] / (np.linalg.norm(V2[:, :n], axis=0, keepdims=True) + 1e-8)

    n_shared = min(v1_norm.shape[0], v2_norm.shape[0])
    sim_matrix = np.abs(v1_norm[:n_shared, :].T @ v2_norm[:n_shared, :])
    matched_sims = []
    used = set()
    for _ in range(n):
        best = np.unravel_index(sim_matrix.argmax(), sim_matrix.shape)
        if best[0] in used:
            break
        matched_sims.append(float(sim_matrix[best]))
        sim_matrix[best[0], :] = -1
        sim_matrix[:, best[1]] = -1
        used.add(best[0])

    return float(np.mean(matched_sims)) if matched_sims else 0.0


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
            activity = act[:n, :, TIME_WINDOW].mean(axis=2)

            try:
                U, S, V = _fit_svd_factors(activity, n_factors=N_FACTORS)
                variance_explained = float(np.sum(S ** 2) / np.sum((activity - activity.mean(axis=0)) ** 2))

                if region not in region_data:
                    region_data[region] = []
                region_data[region].append({
                    "session_idx": sess_idx,
                    "mouse": mouse,
                    "neural_modes": U,
                    "trial_modes": V,
                    "singular_values": S,
                    "variance_explained": variance_explained,
                    "n_neurons": activity.shape[1],
                })
            except Exception as e:
                logger.warning(f"Failed {mouse}/{region}: {e}")

    pairs = []
    for region, measurements in tqdm(region_data.items(), desc="Regions"):
        if len(measurements) < 2:
            continue

        for (i, j) in combinations(range(len(measurements)), 2):
            m1, m2 = measurements[i], measurements[j]

            trial_sim = _factor_similarity(m1["trial_modes"], m2["trial_modes"])

            sv_corr = float(np.corrcoef(
                m1["singular_values"][:min(len(m1["singular_values"]), len(m2["singular_values"]))],
                m2["singular_values"][:min(len(m1["singular_values"]), len(m2["singular_values"]))]
            )[0, 1]) if len(m1["singular_values"]) > 1 and len(m2["singular_values"]) > 1 else None

            pairs.append({
                "region": region,
                "mouse_1": m1["mouse"],
                "mouse_2": m2["mouse"],
                "same_mouse": m1["mouse"] == m2["mouse"],
                "trial_mode_similarity": trial_sim,
                "singular_value_correlation": sv_corr,
                "var_explained_1": m1["variance_explained"],
                "var_explained_2": m2["variance_explained"],
            })

    region_summaries = {}
    for region in region_data:
        rp = [p for p in pairs if p["region"] == region]
        if rp:
            region_summaries[region] = {
                "n_sessions": len(region_data[region]),
                "n_pairs": len(rp),
                "mean_trial_mode_similarity": float(np.mean([p["trial_mode_similarity"] for p in rp])),
                "mean_var_explained": float(np.mean([m["variance_explained"] for m in region_data[region]])),
            }

    results = {
        "timestamp": datetime.now().isoformat(),
        "n_factors": N_FACTORS,
        "n_regions": len(region_data),
        "n_pairs": len(pairs),
        "pairs": pairs,
        "region_summaries": region_summaries,
    }

    out_path = RESULTS_DIR / "neural_factor_bank.json"
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
