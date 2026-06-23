"""Experiment 24: Robustness controls — bootstrap, leave-one-out, partial correlation.

Tests whether the CKA-UMAP anti-correlation is robust to:
  1. Bootstrap resampling (confidence intervals)
  2. Leave-one-mouse-out (no single animal drives it)
  3. Leave-one-region-out (no single region drives it)
  4. Partial correlation controlling for neuron count
  5. Partial correlation controlling for power-law exponent
  6. HSIC-based conditional independence test
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

RESULTS_DIR = Path(__file__).parent.parent / "artifacts" / "exp24"
UMAP_DIM = 5
MIN_NEURONS = 15
TIME_WINDOW = slice(15, 35)
N_BOOTSTRAP = 10000


def _umap_embed(activity, n_components=5):
    from umap import UMAP
    reducer = UMAP(n_components=n_components, n_neighbors=15, min_dist=0.1, random_state=42)
    return reducer.fit_transform(activity)


def _procrustes_distance(X, Y):
    n = min(X.shape[0], Y.shape[0])
    _, _, disparity = procrustes(X[:n], Y[:n])
    return float(disparity)


def _power_law_exponent(activity):
    from sklearn.decomposition import PCA
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


def _partial_correlation(x, y, z):
    from numpy.linalg import lstsq
    x, y = np.array(x, dtype=float), np.array(y, dtype=float)
    z = np.array(z, dtype=float)
    if z.ndim == 1:
        z = z.reshape(-1, 1)
    rx = x - z @ lstsq(z, x, rcond=None)[0]
    ry = y - z @ lstsq(z, y, rcond=None)[0]
    return float(np.corrcoef(rx, ry)[0, 1])


def _hsic(x, y, sigma=None):
    x = np.array(x).reshape(-1, 1)
    y = np.array(y).reshape(-1, 1)
    n = len(x)
    if sigma is None:
        sigma_x = np.median(np.abs(x - x.T))
        sigma_y = np.median(np.abs(y - y.T))
        sigma_x = max(sigma_x, 1e-8)
        sigma_y = max(sigma_y, 1e-8)
    else:
        sigma_x = sigma_y = sigma

    Kx = np.exp(-0.5 * (x - x.T) ** 2 / sigma_x ** 2)
    Ky = np.exp(-0.5 * (y - y.T) ** 2 / sigma_y ** 2)
    H = np.eye(n) - np.ones((n, n)) / n
    return float(np.trace(Kx @ H @ Ky @ H) / (n - 1) ** 2)


def _hsic_permutation_test(x, y, n_perm=1000):
    rng = np.random.default_rng(42)
    observed = _hsic(x, y)
    count = 0
    for _ in range(n_perm):
        y_perm = rng.permutation(y)
        if _hsic(x, y_perm) >= observed:
            count += 1
    return observed, float(count / n_perm)


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

            if region not in region_data:
                region_data[region] = []
            region_data[region].append({
                "session_idx": sess_idx,
                "mouse": mouse,
                "activity": activity,
                "n_neurons": activity.shape[1],
                "n_trials": n,
                "power_law_alpha": alpha,
            })

    pairs = []
    for region, measurements in tqdm(region_data.items(), desc="Computing pairs"):
        if len(measurements) < 2:
            continue

        for (i, j) in combinations(range(len(measurements)), 2):
            m1, m2 = measurements[i], measurements[j]
            n_shared = min(m1["n_trials"], m2["n_trials"])

            cka_val = cka(m1["activity"][:n_shared], m2["activity"][:n_shared])

            try:
                e1 = _umap_embed(m1["activity"], n_components=min(UMAP_DIM, m1["n_neurons"] - 1))
                e2 = _umap_embed(m2["activity"], n_components=min(UMAP_DIM, m2["n_neurons"] - 1))
                proc_val = _procrustes_distance(e1, e2)
            except Exception:
                proc_val = None

            pairs.append({
                "region": region,
                "mouse_1": m1["mouse"],
                "mouse_2": m2["mouse"],
                "cka": float(cka_val),
                "procrustes": float(proc_val) if proc_val is not None else None,
                "n_neurons_mean": (m1["n_neurons"] + m2["n_neurons"]) / 2,
                "alpha_mean": (
                    (m1["power_law_alpha"] + m2["power_law_alpha"]) / 2
                    if m1["power_law_alpha"] is not None and m2["power_law_alpha"] is not None
                    else None
                ),
            })

    valid = [p for p in pairs if p["procrustes"] is not None]
    cka_arr = np.array([p["cka"] for p in valid])
    proc_arr = np.array([p["procrustes"] for p in valid])

    rho_orig, p_orig = spearmanr(cka_arr, proc_arr)

    rng = np.random.default_rng(42)
    bootstrap_rhos = []
    for _ in tqdm(range(N_BOOTSTRAP), desc="Bootstrap"):
        idx = rng.integers(0, len(valid), size=len(valid))
        r, _ = spearmanr(cka_arr[idx], proc_arr[idx])
        bootstrap_rhos.append(float(r))
    bootstrap_rhos = np.array(bootstrap_rhos)
    ci_lower = float(np.percentile(bootstrap_rhos, 2.5))
    ci_upper = float(np.percentile(bootstrap_rhos, 97.5))

    mice = sorted(set(p["mouse_1"] for p in valid) | set(p["mouse_2"] for p in valid))
    loo_mouse = {}
    for mouse in mice:
        subset = [p for p in valid if p["mouse_1"] != mouse and p["mouse_2"] != mouse]
        if len(subset) >= 4:
            r, pv = spearmanr(
                [p["cka"] for p in subset],
                [p["procrustes"] for p in subset],
            )
            loo_mouse[mouse] = {"rho": float(r), "p_value": float(pv), "n": len(subset)}

    regions_in_pairs = sorted(set(p["region"] for p in valid))
    loo_region = {}
    for region in regions_in_pairs:
        subset = [p for p in valid if p["region"] != region]
        if len(subset) >= 4:
            r, pv = spearmanr(
                [p["cka"] for p in subset],
                [p["procrustes"] for p in subset],
            )
            loo_region[region] = {"rho": float(r), "p_value": float(pv), "n": len(subset)}

    n_neurons_arr = np.array([p["n_neurons_mean"] for p in valid])
    partial_corr_neurons = _partial_correlation(
        [p["cka"] for p in valid],
        [p["procrustes"] for p in valid],
        [p["n_neurons_mean"] for p in valid],
    )

    valid_alpha = [p for p in valid if p["alpha_mean"] is not None]
    partial_corr_alpha = None
    if len(valid_alpha) >= 4:
        partial_corr_alpha = _partial_correlation(
            [p["cka"] for p in valid_alpha],
            [p["procrustes"] for p in valid_alpha],
            [p["alpha_mean"] for p in valid_alpha],
        )

    partial_corr_both = None
    if len(valid_alpha) >= 5:
        partial_corr_both = _partial_correlation(
            [p["cka"] for p in valid_alpha],
            [p["procrustes"] for p in valid_alpha],
            [[p["n_neurons_mean"], p["alpha_mean"]] for p in valid_alpha],
        )

    hsic_stat, hsic_pval = _hsic_permutation_test(cka_arr, proc_arr, n_perm=1000)

    results = {
        "timestamp": datetime.now().isoformat(),
        "n_pairs": len(valid),
        "original": {
            "spearman_rho": float(rho_orig),
            "p_value": float(p_orig),
        },
        "bootstrap": {
            "n_resamples": N_BOOTSTRAP,
            "ci_95_lower": ci_lower,
            "ci_95_upper": ci_upper,
            "mean_rho": float(np.mean(bootstrap_rhos)),
            "std_rho": float(np.std(bootstrap_rhos)),
        },
        "leave_one_mouse_out": loo_mouse,
        "leave_one_region_out": {
            "n_regions": len(loo_region),
            "rho_min": float(min(v["rho"] for v in loo_region.values())) if loo_region else None,
            "rho_max": float(max(v["rho"] for v in loo_region.values())) if loo_region else None,
            "all_significant": all(v["p_value"] < 0.05 for v in loo_region.values()) if loo_region else None,
            "details": loo_region,
        },
        "partial_correlations": {
            "controlling_neuron_count": partial_corr_neurons,
            "controlling_power_law_alpha": partial_corr_alpha,
            "controlling_both": partial_corr_both,
            "interpretation": (
                "If partial correlations remain strongly negative, the "
                "dissociation is not explained by neuron count or effective "
                "dimensionality differences."
            ),
        },
        "hsic_independence_test": {
            "hsic_statistic": hsic_stat,
            "permutation_p_value": hsic_pval,
            "interpretation": (
                "HSIC tests nonlinear dependence. p < 0.05 means CKA and "
                "Procrustes are not independent even under nonlinear mappings."
            ),
        },
    }

    out_path = RESULTS_DIR / "robustness_controls.json"
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
