"""Experiment 23: Sub-sampling sensitivity control.

Addresses the Kriegeskorte & Wei (2021) concern: CKA estimates may be biased
by different neuron counts across sessions. We re-compute CKA and UMAP
Procrustes after sub-sampling each session to the minimum neuron count in
each pair. If the anti-correlation holds, it's not an artifact of
neuron-count asymmetry.
"""
import json
import logging
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.spatial import procrustes
from scipy.stats import spearmanr
from tqdm import tqdm

from data.steinmetz import get_choice_labels, get_region_activity, list_regions, load_all
from geometry.distances import cka

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent.parent / "artifacts" / "exp23"
UMAP_DIM = 5
MIN_NEURONS = 15
TIME_WINDOW = slice(15, 35)
N_SUBSAMPLES = 10


def _umap_embed(activity, n_components=5):
    from umap import UMAP
    reducer = UMAP(n_components=n_components, n_neighbors=15, min_dist=0.1, random_state=42)
    return reducer.fit_transform(activity)


def _procrustes_distance(X, Y):
    n = min(X.shape[0], Y.shape[0])
    _, _, disparity = procrustes(X[:n], Y[:n])
    return float(disparity)


def _subsample_neurons(activity, target_n, rng):
    if activity.shape[1] <= target_n:
        return activity
    idx = rng.choice(activity.shape[1], size=target_n, replace=False)
    return activity[:, idx]


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

            if region not in region_data:
                region_data[region] = []
            region_data[region].append({
                "session_idx": sess_idx,
                "mouse": mouse,
                "activity": activity,
                "n_neurons": activity.shape[1],
                "n_trials": n,
            })

    original_pairs = []
    subsampled_pairs = []
    rng = np.random.default_rng(42)

    for region, measurements in tqdm(region_data.items(), desc="Regions"):
        if len(measurements) < 2:
            continue

        for (i, j) in combinations(range(len(measurements)), 2):
            m1, m2 = measurements[i], measurements[j]
            a1, a2 = m1["activity"], m2["activity"]
            n_shared = min(m1["n_trials"], m2["n_trials"])
            n_min_neurons = min(m1["n_neurons"], m2["n_neurons"])

            cka_orig = cka(a1[:n_shared], a2[:n_shared])

            try:
                e1 = _umap_embed(a1, n_components=min(UMAP_DIM, a1.shape[1] - 1))
                e2 = _umap_embed(a2, n_components=min(UMAP_DIM, a2.shape[1] - 1))
                proc_orig = _procrustes_distance(e1, e2)
            except Exception:
                proc_orig = None

            original_pairs.append({
                "region": region,
                "cka": float(cka_orig),
                "procrustes": float(proc_orig) if proc_orig is not None else None,
                "n_neurons_1": m1["n_neurons"],
                "n_neurons_2": m2["n_neurons"],
                "neuron_ratio": max(m1["n_neurons"], m2["n_neurons"]) / min(m1["n_neurons"], m2["n_neurons"]),
            })

            sub_ckas = []
            sub_procs = []
            for s in range(N_SUBSAMPLES):
                a1_sub = _subsample_neurons(a1, n_min_neurons, rng)
                a2_sub = _subsample_neurons(a2, n_min_neurons, rng)

                sub_cka = cka(a1_sub[:n_shared], a2_sub[:n_shared])
                sub_ckas.append(float(sub_cka))

                try:
                    e1s = _umap_embed(a1_sub, n_components=min(UMAP_DIM, n_min_neurons - 1))
                    e2s = _umap_embed(a2_sub, n_components=min(UMAP_DIM, n_min_neurons - 1))
                    sub_proc = _procrustes_distance(e1s, e2s)
                    sub_procs.append(float(sub_proc))
                except Exception:
                    pass

            subsampled_pairs.append({
                "region": region,
                "cka_mean": float(np.mean(sub_ckas)) if sub_ckas else None,
                "cka_std": float(np.std(sub_ckas)) if sub_ckas else None,
                "procrustes_mean": float(np.mean(sub_procs)) if sub_procs else None,
                "procrustes_std": float(np.std(sub_procs)) if sub_procs else None,
                "n_min_neurons": n_min_neurons,
            })

    original_corr = {}
    if original_pairs:
        valid = [p for p in original_pairs if p["procrustes"] is not None]
        if len(valid) >= 4:
            rho, pv = spearmanr(
                [p["cka"] for p in valid],
                [p["procrustes"] for p in valid],
            )
            original_corr = {"spearman_rho": float(rho), "p_value": float(pv), "n": len(valid)}

    subsampled_corr = {}
    if subsampled_pairs:
        valid_s = [p for p in subsampled_pairs if p["cka_mean"] is not None and p["procrustes_mean"] is not None]
        if len(valid_s) >= 4:
            rho_s, pv_s = spearmanr(
                [p["cka_mean"] for p in valid_s],
                [p["procrustes_mean"] for p in valid_s],
            )
            subsampled_corr = {"spearman_rho": float(rho_s), "p_value": float(pv_s), "n": len(valid_s)}

    neuron_ratio_effect = {}
    if original_pairs:
        ratios = [p["neuron_ratio"] for p in original_pairs if p["procrustes"] is not None]
        ckas = [p["cka"] for p in original_pairs if p["procrustes"] is not None]
        if len(ratios) >= 4:
            rho_r, pv_r = spearmanr(ratios, ckas)
            neuron_ratio_effect = {
                "ratio_vs_cka_rho": float(rho_r),
                "ratio_vs_cka_p": float(pv_r),
                "interpretation": "If rho ~ 0, neuron count asymmetry does not drive CKA values",
            }

    results = {
        "timestamp": datetime.now().isoformat(),
        "n_regions": len(region_data),
        "n_pairs_original": len(original_pairs),
        "n_pairs_subsampled": len(subsampled_pairs),
        "original_correlation": original_corr,
        "subsampled_correlation": subsampled_corr,
        "neuron_ratio_effect": neuron_ratio_effect,
        "conclusion": (
            "If original and subsampled correlations are similar, the "
            "CKA-UMAP anti-correlation is not an artifact of neuron count differences."
        ),
    }

    out_path = RESULTS_DIR / "subsample_control.json"
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
