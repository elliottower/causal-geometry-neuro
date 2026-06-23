"""Experiment 11: Linear vs nonlinear dissociation (Steinmetz).

For each region, compare cross-animal alignment using:
  (a) LDA subspace → Grassmannian distance (linear)
  (b) UMAP embedding → Procrustes distance after alignment (nonlinear)

If nonlinear gives BETTER cross-animal similarity than linear for some regions
but not others, those regions have curved mechanisms that live on a higher
geometric stratum than Gr(k, n).

The dissociation between linear and nonlinear metrics is the result.
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
from geometry.distances import cka, grassmannian_distance
from geometry.subspace import fit_lda_subspace

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent.parent / "artifacts" / "exp11"
SUBSPACE_K = 5
UMAP_DIM = 5
MIN_NEURONS = 15
TIME_WINDOW = slice(15, 35)


def _umap_embed(activity, n_components=5):
    """UMAP embedding of trial activity."""
    from umap import UMAP
    reducer = UMAP(n_components=n_components, n_neighbors=15, min_dist=0.1, random_state=42)
    return reducer.fit_transform(activity)


def _procrustes_distance(X, Y):
    """Procrustes distance between two point clouds (after optimal alignment)."""
    n = min(X.shape[0], Y.shape[0])
    _, _, disparity = procrustes(X[:n], Y[:n])
    return float(disparity)


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

                embedding = _umap_embed(activity, n_components=min(UMAP_DIM, activity.shape[1] - 1))

                if region not in region_data:
                    region_data[region] = []
                region_data[region].append({
                    "session_idx": sess_idx,
                    "mouse": mouse,
                    "subspace": U,
                    "embedding": embedding,
                    "activity": activity,
                    "n_neurons": activity.shape[1],
                    "n_trials": n,
                })
            except Exception as e:
                logger.warning(f"Failed {mouse}/{region}: {e}")

    pairs = []
    for region, measurements in tqdm(region_data.items(), desc="Regions"):
        if len(measurements) < 2:
            continue

        for (i, j) in combinations(range(len(measurements)), 2):
            m1, m2 = measurements[i], measurements[j]

            n_shared = min(m1["n_trials"], m2["n_trials"])
            cka_linear = cka(m1["activity"][:n_shared], m2["activity"][:n_shared])
            d_proc = _procrustes_distance(m1["embedding"], m2["embedding"])

            if m1["n_neurons"] == m2["n_neurons"]:
                d_grass = grassmannian_distance(m1["subspace"], m2["subspace"])
            else:
                d_grass = None

            same_mouse = m1["mouse"] == m2["mouse"]

            pairs.append({
                "region": region,
                "mouse_1": m1["mouse"],
                "mouse_2": m2["mouse"],
                "same_mouse": same_mouse,
                "cka_linear": cka_linear,
                "grassmannian_distance": d_grass,
                "procrustes_distance": d_proc,
            })

    results = {
        "timestamp": datetime.now().isoformat(),
        "n_regions": len(region_data),
        "n_pairs": len(pairs),
        "pairs": pairs,
    }

    if pairs:
        from scipy.stats import spearmanr
        d_cka = [p["cka_linear"] for p in pairs]
        d_p = [p["procrustes_distance"] for p in pairs]
        rho, p_val = spearmanr(d_cka, d_p)
        results["cka_vs_procrustes"] = {
            "spearman_rho": float(rho),
            "p_value": float(p_val),
            "interpretation": "rho < 1 means linear (CKA) and nonlinear (UMAP) dissociate",
        }

        per_region = {}
        for region in region_data:
            rp = [p for p in pairs if p["region"] == region]
            if len(rp) >= 2:
                rc = [p["cka_linear"] for p in rp]
                rpr = [p["procrustes_distance"] for p in rp]
                per_region[region] = {
                    "n_pairs": len(rp),
                    "cka_mean": float(np.mean(rc)),
                    "proc_mean": float(np.mean(rpr)),
                }
        results["per_region"] = per_region

    out_path = RESULTS_DIR / "linear_vs_nonlinear.json"
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
