"""Experiment 74: Debiased CKA — rerun CKA-Procrustes anti-correlation with bias correction.

Murphy, Zylberberg & Fyshe 2024 (arXiv:2405.01012) showed that standard (biased) CKA
produces inflated similarity scores for random matrices in the low-data, high-dimensionality
regime — exactly our setting (250 trials, 15-40 neurons, unequal region sizes).

This experiment reruns the exp11 CKA-vs-Procrustes analysis with both biased and
debiased CKA, reports whether the rho=-0.85 anti-correlation holds, and adds
partial correlation controls for n_neurons.

CPU only. ~1-2h.
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
from geometry.distances import cka, debiased_cka
from geometry.subspace import fit_lda_subspace

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent / "results" / "exp74"
UMAP_DIM = 5
MIN_NEURONS = 15
TIME_WINDOW = slice(15, 35)


def _umap_embed(activity, n_components=5):
    from umap import UMAP
    reducer = UMAP(n_components=n_components, n_neighbors=15, min_dist=0.1, random_state=42)
    return reducer.fit_transform(activity)


def _procrustes_distance(X, Y):
    n = min(X.shape[0], Y.shape[0])
    _, _, disparity = procrustes(X[:n], Y[:n])
    return float(disparity)


def _partial_correlation(x, y, z):
    """Partial Spearman correlation of x and y controlling for z."""
    rx = stats.spearmanr(x, z).statistic
    ry = stats.spearmanr(y, z).statistic
    rxy = stats.spearmanr(x, y).statistic
    denom = np.sqrt((1 - rx**2) * (1 - ry**2))
    if denom < 1e-10:
        return 0.0
    return (rxy - rx * ry) / denom


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

            try:
                embedding = _umap_embed(activity, n_components=min(UMAP_DIM, activity.shape[1] - 1))

                if region not in region_data:
                    region_data[region] = []
                region_data[region].append({
                    "session_idx": sess_idx,
                    "mouse": mouse,
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
            act1 = m1["activity"][:n_shared]
            act2 = m2["activity"][:n_shared]

            cka_biased = cka(act1, act2)
            cka_debiased = debiased_cka(act1, act2)
            d_proc = _procrustes_distance(m1["embedding"], m2["embedding"])
            proc_sim = 1.0 - d_proc

            pairs.append({
                "region": region,
                "mouse_1": m1["mouse"],
                "mouse_2": m2["mouse"],
                "cka_biased": cka_biased,
                "cka_debiased": cka_debiased,
                "procrustes_similarity": proc_sim,
                "n_neurons_1": m1["n_neurons"],
                "n_neurons_2": m2["n_neurons"],
                "n_neurons_mean": (m1["n_neurons"] + m2["n_neurons"]) / 2,
                "n_trials_shared": n_shared,
            })

    cka_b = np.array([p["cka_biased"] for p in pairs])
    cka_d = np.array([p["cka_debiased"] for p in pairs])
    proc = np.array([p["procrustes_similarity"] for p in pairs])
    nn = np.array([p["n_neurons_mean"] for p in pairs])

    rho_biased, p_biased = stats.spearmanr(cka_b, proc)
    rho_debiased, p_debiased = stats.spearmanr(cka_d, proc)

    partial_biased = _partial_correlation(cka_b, proc, nn)
    partial_debiased = _partial_correlation(cka_d, proc, nn)

    summary = {
        "timestamp": datetime.now().isoformat(),
        "n_regions": len(region_data),
        "n_pairs": len(pairs),
        "biased_cka": {
            "spearman_rho": float(rho_biased),
            "p_value": float(p_biased),
            "partial_rho_controlling_n_neurons": float(partial_biased),
        },
        "debiased_cka": {
            "spearman_rho": float(rho_debiased),
            "p_value": float(p_debiased),
            "partial_rho_controlling_n_neurons": float(partial_debiased),
        },
        "comparison": {
            "debiasing_strengthens": bool(abs(rho_debiased) > abs(rho_biased)),
            "delta_rho": float(abs(rho_debiased) - abs(rho_biased)),
        },
    }

    print(f"\n{'='*60}")
    print(f"Biased CKA vs Procrustes:   rho={rho_biased:.3f} (p={p_biased:.2e})")
    print(f"Debiased CKA vs Procrustes: rho={rho_debiased:.3f} (p={p_debiased:.2e})")
    print(f"Partial (biased, ctrl n_neurons):   {partial_biased:.3f}")
    print(f"Partial (debiased, ctrl n_neurons): {partial_debiased:.3f}")
    print(f"Debiasing {'STRENGTHENS' if summary['comparison']['debiasing_strengthens'] else 'WEAKENS'} the result (delta={summary['comparison']['delta_rho']:.3f})")
    print(f"{'='*60}\n")

    with open(RESULTS_DIR / "debiased_cka_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    with open(RESULTS_DIR / "pairs.json", "w") as f:
        json.dump(pairs, f, indent=2)

    logger.info(f"Saved to {RESULTS_DIR}")
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
