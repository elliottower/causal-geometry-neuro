"""Experiment 26: Finite-sampling CKA denoising control.

A NeurIPS 2025 paper showed that CKA is systematically underestimated under
finite neuron sampling due to eigenvector delocalization, with bias depending
on spectral properties. Flat-spectrum (high-dimensional) regions are most
biased toward low CKA under finite sampling — exactly our pattern.

We test whether the anti-correlation survives after applying a spectral
denoising correction: shrink eigenvalues toward the Marchenko-Pastur bulk
before computing CKA. If the anti-correlation persists (or strengthens)
after denoising, it is not a finite-sampling artifact.
"""
import json
import logging
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.spatial import procrustes
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from tqdm import tqdm

from data.steinmetz import get_choice_labels, get_region_activity, list_regions, load_all
from geometry.distances import cka

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent.parent / "artifacts" / "exp26"
UMAP_DIM = 5
MIN_NEURONS = 15
TIME_WINDOW = slice(15, 35)


def _marchenko_pastur_threshold(n_samples, n_features, noise_var=1.0):
    gamma = n_features / n_samples
    return noise_var * (1 + np.sqrt(gamma)) ** 2


def _denoise_activity(activity):
    n_samples, n_features = activity.shape
    pca = PCA(n_components=min(n_samples - 1, n_features))
    scores = pca.fit_transform(activity)
    eigenvalues = pca.explained_variance_

    noise_var = np.median(eigenvalues)
    threshold = _marchenko_pastur_threshold(n_samples, n_features, noise_var)

    signal_mask = eigenvalues > threshold
    n_signal = signal_mask.sum()

    if n_signal == 0:
        return activity

    denoised_scores = scores.copy()
    denoised_scores[:, ~signal_mask] = 0
    return pca.inverse_transform(denoised_scores)


def _umap_embed(activity, n_components=5):
    from umap import UMAP
    reducer = UMAP(n_components=n_components, n_neighbors=15, min_dist=0.1, random_state=42)
    return reducer.fit_transform(activity)


def _procrustes_distance(X, Y):
    n = min(X.shape[0], Y.shape[0])
    _, _, disparity = procrustes(X[:n], Y[:n])
    return float(disparity)


def _power_law_exponent(activity):
    n_components = min(50, activity.shape[1], activity.shape[0])
    pca = PCA(n_components=n_components)
    pca.fit(activity)
    eigenvalues = pca.explained_variance_
    eigenvalues = eigenvalues[eigenvalues > 0]
    if len(eigenvalues) < 10:
        return None
    start, end = 9, min(49, len(eigenvalues) - 1)
    log_rank = np.log10(np.arange(start + 1, end + 2))
    log_eig = np.log10(eigenvalues[start:end + 1])
    coeffs = np.polyfit(log_rank, log_eig, 1)
    return float(-coeffs[0])


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
            alpha = _power_law_exponent(activity)

            try:
                denoised = _denoise_activity(activity)
            except Exception:
                denoised = activity

            if region not in region_data:
                region_data[region] = []
            region_data[region].append({
                "session_idx": sess_idx,
                "mouse": mouse,
                "activity": activity,
                "denoised": denoised,
                "n_neurons": activity.shape[1],
                "n_trials": n,
                "alpha": alpha,
            })

    pairs_raw = []
    pairs_denoised = []

    for region, measurements in tqdm(region_data.items(), desc="Regions"):
        if len(measurements) < 2:
            continue

        for (i, j) in combinations(range(len(measurements)), 2):
            m1, m2 = measurements[i], measurements[j]
            n_shared = min(m1["n_trials"], m2["n_trials"])

            cka_raw = cka(m1["activity"][:n_shared], m2["activity"][:n_shared])
            cka_den = cka(m1["denoised"][:n_shared], m2["denoised"][:n_shared])

            try:
                e1 = _umap_embed(m1["activity"], n_components=min(UMAP_DIM, m1["n_neurons"] - 1))
                e2 = _umap_embed(m2["activity"], n_components=min(UMAP_DIM, m2["n_neurons"] - 1))
                proc = _procrustes_distance(e1, e2)
            except Exception:
                proc = None

            pairs_raw.append({
                "region": region,
                "cka": float(cka_raw),
                "procrustes": float(proc) if proc is not None else None,
            })
            pairs_denoised.append({
                "region": region,
                "cka": float(cka_den),
                "procrustes": float(proc) if proc is not None else None,
            })

    raw_corr = {}
    denoised_corr = {}
    valid_raw = [p for p in pairs_raw if p["procrustes"] is not None]
    valid_den = [p for p in pairs_denoised if p["procrustes"] is not None]

    if len(valid_raw) >= 4:
        rho_r, pv_r = spearmanr([p["cka"] for p in valid_raw], [p["procrustes"] for p in valid_raw])
        raw_corr = {"spearman_rho": float(rho_r), "p_value": float(pv_r), "n": len(valid_raw)}

    if len(valid_den) >= 4:
        rho_d, pv_d = spearmanr([p["cka"] for p in valid_den], [p["procrustes"] for p in valid_den])
        denoised_corr = {"spearman_rho": float(rho_d), "p_value": float(pv_d), "n": len(valid_den)}

    cka_shift = {}
    if valid_raw and valid_den:
        raw_ckas = [p["cka"] for p in valid_raw]
        den_ckas = [p["cka"] for p in valid_den]
        cka_shift = {
            "mean_raw": float(np.mean(raw_ckas)),
            "mean_denoised": float(np.mean(den_ckas)),
            "mean_increase": float(np.mean(den_ckas) - np.mean(raw_ckas)),
            "interpretation": (
                "Positive increase means denoising raised CKA (correcting "
                "finite-sampling underestimation). If the anti-correlation "
                "survives, it is not a sampling artifact."
            ),
        }

    results = {
        "timestamp": datetime.now().isoformat(),
        "n_regions": len(region_data),
        "n_pairs": len(pairs_raw),
        "raw_correlation": raw_corr,
        "denoised_correlation": denoised_corr,
        "cka_shift": cka_shift,
    }

    out_path = RESULTS_DIR / "cka_denoising.json"
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
