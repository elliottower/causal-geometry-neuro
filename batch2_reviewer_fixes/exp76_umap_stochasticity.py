"""Experiment 76: UMAP stochasticity robustness analysis.

Run UMAP Procrustes N times with different seeds per cross-session pair.
Report whether the CKA-Procrustes anti-correlation is stable across seeds
and UMAP hyperparameters.

CPU only. ~4h.
"""
import json
import logging
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.spatial import procrustes
from tqdm import tqdm

from data.steinmetz import get_choice_labels, get_region_activity, list_regions, load_all
from geometry.distances import cka

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent / "results" / "exp76"
MIN_NEURONS = 15
TIME_WINDOW = slice(15, 35)
UMAP_DIM = 5
N_SEEDS = 20
NEIGHBOR_VALUES = [5, 15, 30]
MIN_DIST_VALUES = [0.0, 0.1, 0.5]


def _umap_embed(activity, n_components, n_neighbors, min_dist, seed):
    from umap import UMAP
    reducer = UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=seed,
    )
    return reducer.fit_transform(activity)


def _procrustes_distance(X, Y):
    n = min(X.shape[0], Y.shape[0])
    _, _, disparity = procrustes(X[:n], Y[:n])
    return float(disparity)


def run(max_sessions=None):
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

            if region not in region_data:
                region_data[region] = []
            region_data[region].append({
                "session_idx": sess_idx,
                "mouse": mouse,
                "activity": activity,
                "n_neurons": activity.shape[1],
                "n_trials": n,
            })

    sweep_results = []

    for n_neighbors in NEIGHBOR_VALUES:
        for min_dist in MIN_DIST_VALUES:
            config_label = f"nn={n_neighbors}_md={min_dist}"
            logger.info(f"Sweeping {config_label}...")

            seed_correlations = []

            for seed in range(N_SEEDS):
                pairs_cka = []
                pairs_proc = []

                for region, measurements in region_data.items():
                    if len(measurements) < 2:
                        continue

                    embeddings = {}
                    for idx, m in enumerate(measurements):
                        try:
                            nc = min(UMAP_DIM, m["activity"].shape[1] - 1)
                            emb = _umap_embed(m["activity"], nc, n_neighbors, min_dist, seed)
                            embeddings[idx] = emb
                        except Exception:
                            pass

                    for (i, j) in combinations(embeddings.keys(), 2):
                        m1, m2 = measurements[i], measurements[j]
                        n_shared = min(m1["n_trials"], m2["n_trials"])
                        c = cka(m1["activity"][:n_shared], m2["activity"][:n_shared])
                        d = _procrustes_distance(embeddings[i], embeddings[j])
                        pairs_cka.append(c)
                        pairs_proc.append(1.0 - d)

                if len(pairs_cka) > 5:
                    rho, p = stats.spearmanr(pairs_cka, pairs_proc)
                    seed_correlations.append({"seed": seed, "rho": float(rho), "p": float(p), "n_pairs": len(pairs_cka)})

            rhos = [s["rho"] for s in seed_correlations]
            sweep_results.append({
                "n_neighbors": n_neighbors,
                "min_dist": min_dist,
                "n_seeds": len(seed_correlations),
                "mean_rho": float(np.mean(rhos)) if rhos else None,
                "std_rho": float(np.std(rhos)) if rhos else None,
                "min_rho": float(np.min(rhos)) if rhos else None,
                "max_rho": float(np.max(rhos)) if rhos else None,
                "all_negative": all(r < 0 for r in rhos) if rhos else None,
                "per_seed": seed_correlations,
            })

            if rhos:
                print(f"  {config_label}: mean_rho={np.mean(rhos):.3f} +/- {np.std(rhos):.3f} (range [{np.min(rhos):.3f}, {np.max(rhos):.3f}])")

    summary = {
        "timestamp": datetime.now().isoformat(),
        "n_regions": len(region_data),
        "sweep": sweep_results,
        "conclusion": "stable" if all(s.get("all_negative") for s in sweep_results) else "unstable",
    }

    with open(RESULTS_DIR / "umap_stochasticity.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Saved to {RESULTS_DIR}")
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
