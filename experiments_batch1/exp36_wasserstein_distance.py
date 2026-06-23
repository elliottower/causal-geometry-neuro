"""Experiment 36: Wasserstein (optimal transport) distance between neural activity distributions.

Computes W2 distance between neural activity distributions across sessions for the
same brain region, and compares against CKA and UMAP Procrustes.

Key idea: W2 distance compares activity *distributions* without requiring matched
neurons or trials. Each session's activity for a region is a cloud of points in
trial-space (after PCA projection to shared dimensionality). W2 measures the cost
of transporting one distribution to another.

For each region pair (same region, different sessions):
1. PCA-project both sessions to min(10, shared_dims) dimensions
2. Compute sliced Wasserstein distance (50 random projections)
3. Compute CKA between the two sessions
4. Store per-pair: wasserstein, cka, source alpha, n_neurons

Prediction tests (Spearman):
- wasserstein_vs_cka: do they agree or disagree?
- alpha_vs_wasserstein: does geometric type predict OT distance?
- wasserstein_vs_procrustes: correlation with UMAP Procrustes (if available)
"""
import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr, wasserstein_distance
from sklearn.decomposition import PCA
from tqdm import tqdm

from data.steinmetz import get_choice_labels, get_region_activity, list_regions, load_all
from geometry.distances import cka

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent.parent / "artifacts" / "exp36"
MIN_NEURONS = 10
TIME_WINDOW = slice(15, 35)
N_PROJECTIONS = 50
N_PCA = 10
N_BOOTSTRAP = 1000
N_PERMUTATIONS = 500


def _bootstrap_spearman(x, y, n_boot=N_BOOTSTRAP, ci=0.95):
    x, y = np.array(x), np.array(y)
    n = len(x)
    rho_obs, p_obs = spearmanr(x, y)
    rhos = np.empty(n_boot)
    for i in range(n_boot):
        idx = np.random.choice(n, n, replace=True)
        rhos[i] = spearmanr(x[idx], y[idx])[0]
    alpha = (1 - ci) / 2
    lo, hi = np.nanpercentile(rhos, [100 * alpha, 100 * (1 - alpha)])
    return {
        "rho": float(rho_obs),
        "p": float(p_obs),
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "n": n,
        "bootstrap_se": float(np.nanstd(rhos)),
    }


def _permutation_test_spearman(x, y, n_perm=N_PERMUTATIONS):
    x, y = np.array(x), np.array(y)
    rho_obs = spearmanr(x, y)[0]
    count = 0
    for _ in range(n_perm):
        perm = np.random.permutation(len(y))
        if abs(spearmanr(x, y[perm])[0]) >= abs(rho_obs):
            count += 1
    return float((count + 1) / (n_perm + 1))


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
    log_eig = np.log10(eigenvalues[start : end + 1])
    coeffs = np.polyfit(log_rank, log_eig, 1)
    return float(-coeffs[0])


def _sliced_wasserstein(X, Y, n_projections=N_PROJECTIONS):
    """Sliced Wasserstein distance between two point clouds in R^d.

    Projects both clouds onto n_projections random unit directions,
    computes the 1D Wasserstein distance for each, and averages.

    Args:
        X: (n1, d) point cloud 1
        Y: (n2, d) point cloud 2
        n_projections: number of random projection directions

    Returns:
        Mean 1D Wasserstein distance across projections
    """
    d = X.shape[1]
    rng = np.random.default_rng()
    directions = rng.standard_normal((n_projections, d))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)

    distances = np.empty(n_projections)
    for i in range(n_projections):
        proj_x = X @ directions[i]
        proj_y = Y @ directions[i]
        distances[i] = wasserstein_distance(proj_x, proj_y)

    return float(np.mean(distances))


def _umap_procrustes(X, Y, n_components=2):
    """UMAP Procrustes distance between two point clouds.

    Returns None if umap is not installed.
    """
    try:
        import umap
        from scipy.spatial import procrustes
    except ImportError:
        return None

    n = min(X.shape[0], Y.shape[0])
    if n < 10:
        return None

    X_sub, Y_sub = X[:n], Y[:n]
    reducer = umap.UMAP(n_components=n_components, random_state=42)
    emb_x = reducer.fit_transform(X_sub)
    emb_y = reducer.fit_transform(Y_sub)

    _, _, disparity = procrustes(emb_x, emb_y)
    return float(disparity)


def run(max_sessions: int | None = None):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"[{datetime.now().isoformat()}] Starting exp36_wasserstein_distance")

    sessions = load_all()
    if max_sessions:
        sessions = sessions[:max_sessions]
    logger.info(f"[{datetime.now().isoformat()}] Loaded {len(sessions)} sessions")

    # Collect per-region, per-session activity matrices
    region_sessions = {}  # region -> list of {activity, session_idx, n_neurons, alpha}

    for sess_idx, sess in enumerate(tqdm(sessions, desc="Loading sessions")):
        labels = get_choice_labels(sess)
        if len(np.unique(labels)) < 2:
            continue

        regions = list_regions(sess, min_neurons=MIN_NEURONS)
        for region in regions:
            act = get_region_activity(sess, region)
            if act is None or act.shape[1] < MIN_NEURONS:
                continue

            n = min(act.shape[0], len(labels))
            activity = act[:n, :, TIME_WINDOW].mean(axis=2)  # (trials, neurons)

            alpha = _power_law_exponent(activity)

            if region not in region_sessions:
                region_sessions[region] = []
            region_sessions[region].append({
                "activity": activity,
                "session_idx": sess_idx,
                "n_neurons": activity.shape[1],
                "alpha": alpha,
            })

    logger.info(
        f"[{datetime.now().isoformat()}] Collected {len(region_sessions)} regions, "
        f"building cross-session pairs"
    )

    # Compare all pairs of sessions for the same region
    region_pair_results = []
    w_list, cka_list, alpha_list, proc_list = [], [], [], []

    for region, sess_list in tqdm(region_sessions.items(), desc="Region pairs"):
        if len(sess_list) < 2:
            continue

        for i in range(len(sess_list)):
            for j in range(i + 1, len(sess_list)):
                s1, s2 = sess_list[i], sess_list[j]
                a1, a2 = s1["activity"], s2["activity"]

                # PCA project to shared dimensionality
                shared_dim = min(N_PCA, a1.shape[1], a2.shape[1])
                if shared_dim < 1:
                    continue

                pca1 = PCA(n_components=shared_dim)
                pca2 = PCA(n_components=shared_dim)
                z1 = pca1.fit_transform(a1)
                z2 = pca2.fit_transform(a2)

                # Sliced Wasserstein
                w_dist = _sliced_wasserstein(z1, z2)

                # CKA (needs matched trial count)
                n_shared = min(z1.shape[0], z2.shape[0])
                cka_val = cka(z1[:n_shared], z2[:n_shared])

                # UMAP Procrustes (optional)
                proc_val = _umap_procrustes(z1, z2)

                # Mean alpha across the pair
                alphas = [a for a in [s1["alpha"], s2["alpha"]] if a is not None]
                mean_alpha = float(np.mean(alphas)) if alphas else None

                pair = {
                    "region": region,
                    "session_i": s1["session_idx"],
                    "session_j": s2["session_idx"],
                    "wasserstein": w_dist,
                    "cka": cka_val,
                    "procrustes": proc_val,
                    "mean_alpha": mean_alpha,
                    "n_neurons_i": s1["n_neurons"],
                    "n_neurons_j": s2["n_neurons"],
                    "shared_pca_dim": shared_dim,
                }
                region_pair_results.append(pair)

                w_list.append(w_dist)
                cka_list.append(cka_val)
                if mean_alpha is not None:
                    alpha_list.append(mean_alpha)
                if proc_val is not None:
                    proc_list.append(proc_val)

    logger.info(
        f"[{datetime.now().isoformat()}] Computed {len(region_pair_results)} pairs, "
        f"running prediction tests"
    )

    # Prediction tests
    prediction_tests = {}

    # wasserstein vs cka
    if len(w_list) >= 4:
        boot = _bootstrap_spearman(w_list, cka_list)
        perm_p = _permutation_test_spearman(w_list, cka_list)
        prediction_tests["wasserstein_vs_cka"] = {
            **boot,
            "permutation_p": perm_p,
            "interpretation": (
                "Negative rho means high Wasserstein (dissimilar distributions) "
                "corresponds to low CKA (dissimilar representations) — agreement. "
                "Near-zero rho means the two metrics capture different aspects."
            ),
        }

    # alpha vs wasserstein (use only pairs with valid alpha)
    if len(alpha_list) >= 4:
        # Align: alpha_list was appended only when mean_alpha is not None,
        # so we need to rebuild aligned lists
        w_with_alpha = [
            p["wasserstein"]
            for p in region_pair_results
            if p["mean_alpha"] is not None
        ]
        a_with_alpha = [
            p["mean_alpha"]
            for p in region_pair_results
            if p["mean_alpha"] is not None
        ]
        if len(a_with_alpha) >= 4:
            boot = _bootstrap_spearman(a_with_alpha, w_with_alpha)
            perm_p = _permutation_test_spearman(a_with_alpha, w_with_alpha)
            prediction_tests["alpha_vs_wasserstein"] = {
                **boot,
                "permutation_p": perm_p,
                "interpretation": (
                    "Positive rho means high-alpha (steep spectrum, CKA-type) regions "
                    "have larger Wasserstein distances across sessions — less distributional "
                    "stability. Negative rho means CKA-type regions are more OT-stable."
                ),
            }

    # wasserstein vs procrustes
    if len(proc_list) >= 4:
        w_with_proc = [
            p["wasserstein"]
            for p in region_pair_results
            if p["procrustes"] is not None
        ]
        p_with_proc = [
            p["procrustes"]
            for p in region_pair_results
            if p["procrustes"] is not None
        ]
        if len(p_with_proc) >= 4:
            boot = _bootstrap_spearman(w_with_proc, p_with_proc)
            perm_p = _permutation_test_spearman(w_with_proc, p_with_proc)
            prediction_tests["wasserstein_vs_procrustes"] = {
                **boot,
                "permutation_p": perm_p,
                "interpretation": (
                    "Positive rho means Wasserstein and UMAP Procrustes agree on which "
                    "session pairs are more dissimilar. High correlation suggests both "
                    "capture similar distributional structure."
                ),
            }

    # Summary statistics
    summary = {
        "n_pairs": len(region_pair_results),
        "n_regions": len(set(p["region"] for p in region_pair_results)),
        "wasserstein_mean": float(np.mean(w_list)) if w_list else None,
        "wasserstein_std": float(np.std(w_list)) if w_list else None,
        "cka_mean": float(np.mean(cka_list)) if cka_list else None,
        "cka_std": float(np.std(cka_list)) if cka_list else None,
    }

    results = {
        "timestamp": datetime.now().isoformat(),
        "n_sessions": len(sessions),
        "region_pair_results": region_pair_results,
        "prediction_tests": prediction_tests,
        "summary": summary,
    }

    out_path = RESULTS_DIR / "wasserstein_distance.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"[{datetime.now().isoformat()}] Saved results to {out_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exp36: Wasserstein distance between neural activity distributions")
    parser.add_argument("--max-sessions", type=int, default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(max_sessions=args.max_sessions)
