"""Experiment 14: Geometric type classifier (Steinmetz).

For each region, compute ALL invariants:
  - LDA direction cosine (Type 1 — direction)
  - Grassmannian distance (Type 2 — subspace)
  - Procrustes on UMAP embedding (nonlinear manifold)
  - Wasserstein on persistence diagrams (topology)
  - Procrustes on trajectories (dynamics)

Then for each region, measure which invariants are cross-animal CONSERVED
(low variance across animal pairs) vs NOT conserved (high variance).

The pattern of conserved invariants determines the geometric type:
  - Only direction conserved → Type 1
  - Subspace conserved, direction not → Type 2
  - Nonlinear conserved, linear not → curved manifold
  - Topology conserved, geometry not → topological mechanism
  - Dynamics conserved, statics not → dynamical mechanism

The clustering of regions by their conservation pattern IS the stratified
view applied to real neural data.
"""
import json
import logging
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.spatial import procrustes
from tqdm import tqdm

from data.steinmetz import get_choice_labels, get_region_activity, list_regions, load_all
from geometry.distances import grassmannian_distance
from geometry.subspace import fit_lda_subspace

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent.parent / "artifacts" / "exp14"
SUBSPACE_K = 5
MIN_NEURONS = 15
TIME_WINDOW = slice(15, 35)
N_TIME_BINS = 5
TIME_SLICES = [slice(i * 10, (i + 1) * 10) for i in range(N_TIME_BINS)]


def _compute_all_invariants(activity_3d, labels, n_neurons):
    """Compute all geometric invariants for one session/region."""
    n = min(activity_3d.shape[0], len(labels))
    activity_3d = activity_3d[:n]
    labels = labels[:n]

    activity = activity_3d[:, :, TIME_WINDOW].mean(axis=2)
    k = min(SUBSPACE_K, n_neurons - 1)

    U = fit_lda_subspace(activity, labels, k=k)
    direction = U[:, 0]

    trajectory = np.zeros((N_TIME_BINS, k))
    for t, ts in enumerate(TIME_SLICES):
        act_t = activity_3d[:, :, ts].mean(axis=2)
        mean_0 = act_t[labels == 0].mean(axis=0)
        mean_1 = act_t[labels == 1].mean(axis=0)
        trajectory[t] = U.T @ (mean_1 - mean_0)

    try:
        from umap import UMAP
        reducer = UMAP(n_components=min(k, n_neurons - 1), n_neighbors=15, min_dist=0.1, random_state=42)
        embedding = reducer.fit_transform(activity)
    except Exception:
        embedding = None

    try:
        from ripser import ripser
        dgm = ripser(activity, maxdim=1, n_perm=100)["dgms"]
    except Exception:
        dgm = None

    return {
        "direction": direction,
        "subspace": U,
        "trajectory": trajectory,
        "embedding": embedding,
        "persistence": dgm,
    }


def _compare_pair(inv1, inv2, same_dim):
    """Compute all pairwise distances between two sets of invariants."""
    result = {}

    d1, d2 = inv1["direction"], inv2["direction"]
    if d1.shape[0] == d2.shape[0]:
        result["direction_cosine"] = abs(float(d1 @ d2))
    else:
        result["direction_cosine"] = None

    if same_dim:
        result["grassmannian"] = grassmannian_distance(inv1["subspace"], inv2["subspace"])
    else:
        result["grassmannian"] = None

    try:
        _, _, d = procrustes(inv1["trajectory"], inv2["trajectory"])
        result["trajectory_procrustes"] = float(d)
    except Exception:
        result["trajectory_procrustes"] = None

    if inv1["embedding"] is not None and inv2["embedding"] is not None:
        try:
            n = min(inv1["embedding"].shape[0], inv2["embedding"].shape[0])
            _, _, d = procrustes(inv1["embedding"][:n], inv2["embedding"][:n])
            result["umap_procrustes"] = float(d)
        except Exception:
            result["umap_procrustes"] = None
    else:
        result["umap_procrustes"] = None

    if inv1["persistence"] is not None and inv2["persistence"] is not None:
        try:
            from persim import wasserstein
            d1 = inv1["persistence"][1]
            d2 = inv2["persistence"][1]
            d1 = d1[np.isfinite(d1).all(axis=1)]
            d2 = d2[np.isfinite(d2).all(axis=1)]
            if len(d1) > 0 or len(d2) > 0:
                result["wasserstein_h1"] = float(wasserstein(d1, d2))
            else:
                result["wasserstein_h1"] = 0.0
        except Exception:
            result["wasserstein_h1"] = None
    else:
        result["wasserstein_h1"] = None

    return result


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

            try:
                invariants = _compute_all_invariants(act, labels[:n], act.shape[1])
                if region not in region_data:
                    region_data[region] = []
                region_data[region].append({
                    "session_idx": sess_idx,
                    "mouse": mouse,
                    "n_neurons": act.shape[1],
                    "invariants": invariants,
                })
            except Exception as e:
                logger.warning(f"Failed {mouse}/{region}: {e}")

    all_pairs = []
    for region, measurements in tqdm(region_data.items(), desc="Comparing"):
        if len(measurements) < 2:
            continue

        for (i, j) in combinations(range(len(measurements)), 2):
            m1, m2 = measurements[i], measurements[j]
            same_dim = m1["n_neurons"] == m2["n_neurons"]

            distances = _compare_pair(m1["invariants"], m2["invariants"], same_dim)
            distances["region"] = region
            distances["same_mouse"] = m1["mouse"] == m2["mouse"]
            all_pairs.append(distances)

    region_profiles = {}
    metrics = ["direction_cosine", "grassmannian", "trajectory_procrustes",
               "umap_procrustes", "wasserstein_h1"]

    for region in region_data:
        rp = [p for p in all_pairs if p["region"] == region]
        if len(rp) < 2:
            continue

        profile = {}
        for m in metrics:
            vals = [p[m] for p in rp if p[m] is not None]
            if vals:
                profile[m] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                    "cv": float(np.std(vals) / np.mean(vals)) if np.mean(vals) > 0 else None,
                    "n": len(vals),
                }
        region_profiles[region] = profile

    results = {
        "timestamp": datetime.now().isoformat(),
        "n_regions": len(region_data),
        "n_pairs": len(all_pairs),
        "region_profiles": region_profiles,
        "pairs": all_pairs,
    }

    out_path = RESULTS_DIR / "geometric_type_classifier.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Saved {len(all_pairs)} pairs across {len(region_profiles)} regions")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-sessions", type=int, default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    run(max_sessions=args.max_sessions)
