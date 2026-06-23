"""Experiment 37: Riemannian distance on the SPD manifold between neural covariance matrices.

Each session's activity for a region defines a covariance matrix. The space of
covariance matrices is a Riemannian manifold (symmetric positive definite, SPD).
The affine-invariant Riemannian distance captures geometric structure that
Euclidean distance on covariances misses.

For each region appearing in multiple sessions:
1. Project to shared PCA space (min(15, n_neurons) dims) to handle varying neuron counts
2. Compute shrinkage-regularized covariance in that shared space
3. Compute pairwise affine-invariant and log-Euclidean distances across sessions
4. Test whether dimensionality (power-law exponent) predicts Riemannian distance,
   whether the two SPD metrics agree, and whether within-mouse distances are smaller
"""
import json
import logging
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np
import scipy.linalg
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from tqdm import tqdm

from data.steinmetz import get_choice_labels, get_region_activity, list_regions, load_all

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent.parent / "artifacts" / "exp37"
MIN_NEURONS = 10
TIME_WINDOW = slice(15, 35)
PCA_SHARED_DIMS = 15
SHRINKAGE = 0.1
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


def _spd_riemannian_distance(C1, C2):
    """Affine-invariant Riemannian distance: d(C1, C2) = ||log(C1^{-1/2} C2 C1^{-1/2})||_F."""
    eigvals_C1, eigvecs_C1 = np.linalg.eigh(C1)
    eigvals_C1 = np.maximum(eigvals_C1, 1e-10)
    C1_sqrt_inv = eigvecs_C1 @ np.diag(eigvals_C1 ** -0.5) @ eigvecs_C1.T
    M = C1_sqrt_inv @ C2 @ C1_sqrt_inv
    eigvals_M = np.linalg.eigvalsh(M)
    eigvals_M = np.maximum(eigvals_M, 1e-10)
    return float(np.sqrt(np.sum(np.log(eigvals_M) ** 2)))


def _log_euclidean_distance(C1, C2):
    """Log-Euclidean distance: ||logm(C1) - logm(C2)||_F."""
    log_C1 = scipy.linalg.logm(C1)
    log_C2 = scipy.linalg.logm(C2)
    return float(np.linalg.norm(log_C1 - log_C2, "fro"))


def _shrinkage_covariance(X, shrink=SHRINKAGE):
    """Compute shrinkage-regularized covariance: (1-s)*cov + s*trace(cov)/d * I."""
    cov = np.cov(X, rowvar=False)
    d = cov.shape[0]
    trace_over_d = np.trace(cov) / d
    return (1 - shrink) * cov + shrink * trace_over_d * np.eye(d)


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

    logger.info(f"Loaded {len(sessions)} sessions")

    # Collect per-region, per-session data
    # region -> list of {activity, mouse_name, session_idx, alpha}
    region_sessions = {}
    for sess_idx, sess in enumerate(tqdm(sessions, desc="Extracting regions")):
        mouse = str(sess.get("mouse_name", f"mouse_{sess_idx}"))
        labels = get_choice_labels(sess)
        for region in list_regions(sess, min_neurons=MIN_NEURONS):
            act = get_region_activity(sess, region)
            if act is None:
                continue
            n = min(act.shape[0], len(labels))
            activity = act[:n, :, TIME_WINDOW].mean(axis=2)  # (trials, neurons)
            if activity.shape[1] < MIN_NEURONS:
                continue

            alpha = _power_law_exponent(activity)

            if region not in region_sessions:
                region_sessions[region] = []
            region_sessions[region].append({
                "activity": activity,
                "mouse": mouse,
                "session_idx": sess_idx,
                "alpha": alpha,
            })

    logger.info(f"Found {len(region_sessions)} regions with >= {MIN_NEURONS} neurons")

    # For each region with multiple sessions, compute pairwise SPD distances
    region_results = {}
    all_alphas = []
    all_affine_dists = []
    all_log_euclidean_dists = []
    all_within_mouse = []

    for region, entries in tqdm(sorted(region_sessions.items()), desc="Computing SPD distances"):
        if len(entries) < 2:
            continue

        # Determine shared PCA dimensionality
        min_neurons = min(e["activity"].shape[1] for e in entries)
        shared_dims = min(PCA_SHARED_DIMS, min_neurons)
        if shared_dims < 2:
            continue

        # Project each session to shared PCA space, then compute covariance
        covariances = []
        mice = []
        alphas = []
        for entry in entries:
            activity = entry["activity"]
            n_pca = min(shared_dims, activity.shape[0] - 1)
            if n_pca < 2:
                continue
            pca = PCA(n_components=n_pca)
            projected = pca.fit_transform(activity)  # (trials, shared_dims)
            cov = _shrinkage_covariance(projected)
            covariances.append(cov)
            mice.append(entry["mouse"])
            alphas.append(entry["alpha"])

        if len(covariances) < 2:
            continue

        affine_dists = []
        log_euclidean_dists = []
        within_mouse_affine = []
        across_mouse_affine = []

        for (i, j) in combinations(range(len(covariances)), 2):
            try:
                d_affine = _spd_riemannian_distance(covariances[i], covariances[j])
                d_log_euc = _log_euclidean_distance(covariances[i], covariances[j])
            except Exception as e:
                logger.warning(f"Distance computation failed for {region} ({i},{j}): {e}")
                continue

            affine_dists.append(d_affine)
            log_euclidean_dists.append(d_log_euc)

            same_mouse = mice[i] == mice[j]
            if same_mouse:
                within_mouse_affine.append(d_affine)
            else:
                across_mouse_affine.append(d_affine)

            # Collect for global correlation tests
            all_affine_dists.append(d_affine)
            all_log_euclidean_dists.append(d_log_euc)
            all_within_mouse.append(same_mouse)
            mean_alpha = np.mean([a for a in [alphas[i], alphas[j]] if a is not None])
            if not np.isnan(mean_alpha):
                all_alphas.append(mean_alpha)
            else:
                all_alphas.append(None)

        region_results[region] = {
            "n_sessions": len(covariances),
            "shared_dims": int(shared_dims),
            "mean_alpha": float(np.mean([a for a in alphas if a is not None])) if any(a is not None for a in alphas) else None,
            "affine_invariant": {
                "mean": float(np.mean(affine_dists)),
                "std": float(np.std(affine_dists)),
                "n_pairs": len(affine_dists),
            },
            "log_euclidean": {
                "mean": float(np.mean(log_euclidean_dists)),
                "std": float(np.std(log_euclidean_dists)),
                "n_pairs": len(log_euclidean_dists),
            },
            "within_mouse_affine": {
                "mean": float(np.mean(within_mouse_affine)) if within_mouse_affine else None,
                "n": len(within_mouse_affine),
            },
            "across_mouse_affine": {
                "mean": float(np.mean(across_mouse_affine)) if across_mouse_affine else None,
                "n": len(across_mouse_affine),
            },
        }

        logger.info(
            f"{region}: {len(covariances)} sessions, "
            f"affine={np.mean(affine_dists):.3f}+/-{np.std(affine_dists):.3f}, "
            f"log_euc={np.mean(log_euclidean_dists):.3f}+/-{np.std(log_euclidean_dists):.3f}"
        )

    # Prediction tests
    prediction_tests = {}

    # 1. alpha vs SPD distance
    valid_alpha_mask = [a is not None for a in all_alphas]
    if sum(valid_alpha_mask) >= 5:
        valid_alphas = [a for a, v in zip(all_alphas, valid_alpha_mask) if v]
        valid_dists = [d for d, v in zip(all_affine_dists, valid_alpha_mask) if v]
        boot = _bootstrap_spearman(valid_alphas, valid_dists)
        perm_p = _permutation_test_spearman(valid_alphas, valid_dists)
        prediction_tests["alpha_vs_spd_distance"] = {
            **boot,
            "permutation_p": perm_p,
            "interpretation": (
                "Tests whether dimensionality (power-law exponent) predicts "
                "Riemannian distance between covariance matrices."
            ),
        }

    # 2. Affine-invariant vs log-Euclidean agreement
    if len(all_affine_dists) >= 5:
        boot = _bootstrap_spearman(all_affine_dists, all_log_euclidean_dists)
        perm_p = _permutation_test_spearman(all_affine_dists, all_log_euclidean_dists)
        prediction_tests["spd_affine_vs_log_euclidean"] = {
            **boot,
            "permutation_p": perm_p,
            "interpretation": (
                "Tests whether affine-invariant and log-Euclidean SPD metrics agree. "
                "High correlation means both capture similar structure."
            ),
        }

    # 3. Within-mouse vs across-mouse Riemannian distance
    within_dists = [d for d, w in zip(all_affine_dists, all_within_mouse) if w]
    across_dists = [d for d, w in zip(all_affine_dists, all_within_mouse) if not w]
    if len(within_dists) >= 3 and len(across_dists) >= 3:
        # Encode within=0, across=1 and correlate with distance
        labels_binary = [0] * len(within_dists) + [1] * len(across_dists)
        dists_combined = within_dists + across_dists
        boot = _bootstrap_spearman(labels_binary, dists_combined)
        perm_p = _permutation_test_spearman(labels_binary, dists_combined)
        prediction_tests["alpha_vs_spd_within_mouse"] = {
            **boot,
            "permutation_p": perm_p,
            "n_within": len(within_dists),
            "n_across": len(across_dists),
            "mean_within": float(np.mean(within_dists)),
            "mean_across": float(np.mean(across_dists)),
            "interpretation": (
                "Positive rho means within-mouse Riemannian distance is smaller "
                "than across-mouse, indicating mouse-specific covariance structure."
            ),
        }

    results = {
        "timestamp": datetime.now().isoformat(),
        "n_sessions": len(sessions),
        "n_regions_analyzed": len(region_results),
        "params": {
            "min_neurons": MIN_NEURONS,
            "time_window": [TIME_WINDOW.start, TIME_WINDOW.stop],
            "pca_shared_dims": PCA_SHARED_DIMS,
            "shrinkage": SHRINKAGE,
        },
        "regions": region_results,
        "prediction_tests": prediction_tests,
    }

    out_path = RESULTS_DIR / "spd_riemannian.json"
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
