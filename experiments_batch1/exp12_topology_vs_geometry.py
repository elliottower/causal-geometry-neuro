"""Experiment 12: Topology vs geometry dissociation (Steinmetz).

For each region, compute:
  (a) Grassmannian distance between animals' choice subspaces (geometry)
  (b) Wasserstein distance between persistence diagrams (topology)

Find region pairs where geometry says "different" but topology says "same"
(conserved topological structure despite different linear embedding) or
vice versa.

If such dissociations exist, the mechanism has topological invariants
beyond its subspace — it lives on a higher geometric stratum.
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

RESULTS_DIR = Path(__file__).parent.parent / "artifacts" / "exp12"
SUBSPACE_K = 5
MIN_NEURONS = 15
TIME_WINDOW = slice(15, 35)


def _persistence_diagram(activity, max_dim=1):
    """Compute persistence diagram of the activity point cloud."""
    from ripser import ripser
    result = ripser(activity, maxdim=max_dim, n_perm=100)
    return result["dgms"]


def _wasserstein_distance(dgm1, dgm2, dim=1):
    """Wasserstein distance between persistence diagrams at given homology dimension."""
    from persim import wasserstein

    d1 = dgm1[dim] if dim < len(dgm1) else np.empty((0, 2))
    d2 = dgm2[dim] if dim < len(dgm2) else np.empty((0, 2))

    d1 = d1[np.isfinite(d1).all(axis=1)]
    d2 = d2[np.isfinite(d2).all(axis=1)]

    if len(d1) == 0 and len(d2) == 0:
        return 0.0
    return float(wasserstein(d1, d2))


def _betti_numbers(dgm, threshold=0.1):
    """Count features alive at a given threshold."""
    bettis = []
    for dim_dgm in dgm:
        finite = dim_dgm[np.isfinite(dim_dgm).all(axis=1)]
        alive = ((finite[:, 0] <= threshold) & (finite[:, 1] > threshold)).sum()
        bettis.append(int(alive))
    return bettis


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
                k = min(SUBSPACE_K, activity.shape[1] - 1)
                U = fit_lda_subspace(activity, labels[:n], k=k)
                dgm = _persistence_diagram(activity)
                bettis = _betti_numbers(dgm)

                if region not in region_data:
                    region_data[region] = []
                region_data[region].append({
                    "session_idx": sess_idx,
                    "mouse": mouse,
                    "subspace": U,
                    "persistence_diagram": dgm,
                    "betti_numbers": bettis,
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

            if m1["n_neurons"] != m2["n_neurons"]:
                d_grass = None
            else:
                d_grass = grassmannian_distance(m1["subspace"], m2["subspace"])

            d_wass_0 = _wasserstein_distance(m1["persistence_diagram"], m2["persistence_diagram"], dim=0)
            d_wass_1 = _wasserstein_distance(m1["persistence_diagram"], m2["persistence_diagram"], dim=1)

            same_mouse = m1["mouse"] == m2["mouse"]

            pairs.append({
                "region": region,
                "mouse_1": m1["mouse"],
                "mouse_2": m2["mouse"],
                "same_mouse": same_mouse,
                "grassmannian_distance": d_grass,
                "wasserstein_h0": d_wass_0,
                "wasserstein_h1": d_wass_1,
                "betti_1": m1["betti_numbers"],
                "betti_2": m2["betti_numbers"],
            })

    results = {
        "timestamp": datetime.now().isoformat(),
        "n_regions": len(region_data),
        "n_pairs": len(pairs),
        "pairs": pairs,
    }

    pairs_with_both = [p for p in pairs if p["grassmannian_distance"] is not None]
    if len(pairs_with_both) >= 5:
        from scipy.stats import spearmanr
        d_g = [p["grassmannian_distance"] for p in pairs_with_both]
        d_w = [p["wasserstein_h1"] for p in pairs_with_both]
        rho, p_val = spearmanr(d_g, d_w)
        results["grass_vs_topology_correlation"] = {
            "spearman_rho": float(rho),
            "p_value": float(p_val),
            "n": len(pairs_with_both),
            "interpretation": "rho < 1 means geometry and topology dissociate",
        }

    betti_summary = {}
    for region, measurements in region_data.items():
        bettis = [m["betti_numbers"] for m in measurements]
        if bettis:
            betti_summary[region] = {
                "n_sessions": len(bettis),
                "beta_0_mean": float(np.mean([b[0] for b in bettis if len(b) > 0])),
                "beta_1_mean": float(np.mean([b[1] for b in bettis if len(b) > 1])),
            }
    results["betti_summary"] = betti_summary

    out_path = RESULTS_DIR / "topology_vs_geometry.json"
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
