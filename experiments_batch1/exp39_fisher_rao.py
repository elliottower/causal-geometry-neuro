"""Experiment 39: Fisher-Rao information-geometric distance between sessions.

Each region's activity in a session defines a probability distribution over neural
states. The Fisher-Rao metric measures distance on the statistical manifold of these
distributions. For Gaussian-approximated distributions (reasonable for trial-averaged
firing rates), the Bhattacharyya distance provides a closed-form lower bound on
Fisher-Rao distance (Rao 1945, Skovgaard 1984).

For each region observed in multiple sessions:
1. Project both sessions to a shared PCA space to handle different neuron counts
2. Estimate mean and shrinkage-regularized covariance in PCA space
3. Compute Bhattacharyya distance and symmetrized KL divergence

Prediction tests:
- alpha_vs_bhattacharyya: does spectral dimensionality predict info-geometric distance?
- alpha_vs_kl: same for KL divergence
- bhattacharyya_vs_kl: do the two divergence measures agree?
- within_mouse_vs_across: is within-mouse distance smaller than across-mouse?
"""
import argparse
import json
import logging
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.decomposition import PCA
from tqdm import tqdm

from data.steinmetz import get_choice_labels, get_region_activity, list_regions, load_all

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent.parent / "artifacts" / "exp39"
MIN_NEURONS = 10
TIME_WINDOW = slice(15, 35)
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
    """Estimate spectral power-law exponent (alpha) from PCA eigenspectrum."""
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


def _shrinkage_covariance(X, shrinkage=0.1):
    """Ledoit-Wolf-style shrinkage covariance estimate.

    Blends the sample covariance with a scaled identity matrix to ensure
    positive definiteness and reduce estimation noise.
    """
    n, p = X.shape
    sample_cov = np.cov(X, rowvar=False)
    if sample_cov.ndim == 0:
        return np.array([[float(sample_cov)]])
    target = np.trace(sample_cov) / p * np.eye(p)
    return (1 - shrinkage) * sample_cov + shrinkage * target


def _bhattacharyya_distance(mu1, cov1, mu2, cov2):
    """Bhattacharyya distance between two Gaussians."""
    cov_avg = (cov1 + cov2) / 2
    cov_avg += 1e-6 * np.eye(cov_avg.shape[0])
    sign, logdet_avg = np.linalg.slogdet(cov_avg)
    _, logdet1 = np.linalg.slogdet(cov1 + 1e-6 * np.eye(cov1.shape[0]))
    _, logdet2 = np.linalg.slogdet(cov2 + 1e-6 * np.eye(cov2.shape[0]))

    diff = mu1 - mu2
    term1 = 0.125 * diff @ np.linalg.solve(cov_avg, diff)
    term2 = 0.5 * (logdet_avg - 0.5 * (logdet1 + logdet2))
    return float(term1 + term2)


def _kl_divergence_gaussian(mu1, cov1, mu2, cov2):
    """KL divergence KL(N(mu1,cov1) || N(mu2,cov2))."""
    k = len(mu1)
    cov2_reg = cov2 + 1e-6 * np.eye(k)
    cov1_reg = cov1 + 1e-6 * np.eye(k)
    cov2_inv = np.linalg.inv(cov2_reg)
    diff = mu2 - mu1
    return float(0.5 * (
        np.trace(cov2_inv @ cov1_reg)
        + diff @ cov2_inv @ diff
        - k
        + np.linalg.slogdet(cov2_reg)[1]
        - np.linalg.slogdet(cov1_reg)[1]
    ))


def _symmetrized_kl(mu1, cov1, mu2, cov2):
    """Symmetrized KL divergence (Jensen-Shannon-like)."""
    return 0.5 * (
        _kl_divergence_gaussian(mu1, cov1, mu2, cov2)
        + _kl_divergence_gaussian(mu2, cov2, mu1, cov1)
    )


def _project_to_shared_dims(activity1, activity2, n_dims):
    """Project two activity matrices to the same number of PCA dimensions independently."""
    n1 = min(n_dims, activity1.shape[1], activity1.shape[0] - 1)
    n2 = min(n_dims, activity2.shape[1], activity2.shape[0] - 1)
    k = min(n1, n2)
    if k < 1:
        return None, None
    pca1 = PCA(n_components=k)
    pca2 = PCA(n_components=k)
    proj1 = pca1.fit_transform(activity1)
    proj2 = pca2.fit_transform(activity2)
    return proj1, proj2


def run(max_sessions: int | None = None):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    sessions = load_all()
    if max_sessions:
        sessions = sessions[:max_sessions]

    logger.info(f"Loaded {len(sessions)} sessions")

    # Collect per-session, per-region data
    region_sessions = {}  # region -> list of {activity, alpha, mouse, sess_idx}

    for sess_idx, sess in enumerate(tqdm(sessions, desc="Loading sessions")):
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

            if region not in region_sessions:
                region_sessions[region] = []
            region_sessions[region].append({
                "activity": activity,
                "alpha": alpha,
                "mouse": mouse,
                "sess_idx": sess_idx,
            })

    logger.info(f"Found {len(region_sessions)} regions with data")

    # Compute pairwise distances for each region
    pair_records = []
    jsonl_path = RESULTS_DIR / "fisher_rao_incremental.jsonl"
    computed_pairs = set()
    if jsonl_path.exists():
        with open(jsonl_path) as jf:
            for line in jf:
                r = json.loads(line)
                computed_pairs.add((r["region"], r["sess_i"], r["sess_j"]))
                pair_records.append(r)
        logger.info(f"Resuming: loaded {len(computed_pairs)} pre-computed pairs")

    region_summaries = {}

    for region, sess_list in tqdm(region_sessions.items(), desc="Computing distances"):
        if len(sess_list) < 2:
            continue

        n_pca = min(10, min(s["activity"].shape[1] for s in sess_list))
        region_bhatt = []
        region_kl = []
        region_pairs = []

        for i, j in combinations(range(len(sess_list)), 2):
            s1, s2 = sess_list[i], sess_list[j]
            pair_key = (region, s1["sess_idx"], s2["sess_idx"])
            if pair_key in computed_pairs:
                continue
            proj1, proj2 = _project_to_shared_dims(s1["activity"], s2["activity"], n_pca)
            if proj1 is None:
                continue

            mu1, mu2 = proj1.mean(axis=0), proj2.mean(axis=0)
            cov1 = _shrinkage_covariance(proj1)
            cov2 = _shrinkage_covariance(proj2)

            bhatt = _bhattacharyya_distance(mu1, cov1, mu2, cov2)
            skl = _symmetrized_kl(mu1, cov1, mu2, cov2)
            same_mouse = s1["mouse"] == s2["mouse"]
            alpha_avg = None
            if s1["alpha"] is not None and s2["alpha"] is not None:
                alpha_avg = (s1["alpha"] + s2["alpha"]) / 2

            record = {
                "region": region,
                "sess_i": s1["sess_idx"],
                "sess_j": s2["sess_idx"],
                "mouse_i": s1["mouse"],
                "mouse_j": s2["mouse"],
                "same_mouse": same_mouse,
                "bhattacharyya": bhatt,
                "symmetrized_kl": skl,
                "alpha_avg": alpha_avg,
            }
            pair_records.append(record)
            with open(jsonl_path, "a") as jf:
                jf.write(json.dumps(record, default=str) + "\n")
            region_bhatt.append(bhatt)
            region_kl.append(skl)
            region_pairs.append(record)

        if region_bhatt:
            alphas = [s["alpha"] for s in sess_list if s["alpha"] is not None]
            region_summaries[region] = {
                "n_sessions": len(sess_list),
                "n_pairs": len(region_bhatt),
                "mean_bhattacharyya": float(np.mean(region_bhatt)),
                "std_bhattacharyya": float(np.std(region_bhatt)),
                "mean_symmetrized_kl": float(np.mean(region_kl)),
                "std_symmetrized_kl": float(np.std(region_kl)),
                "mean_alpha": float(np.mean(alphas)) if alphas else None,
            }

    logger.info(f"Computed {len(pair_records)} pairwise distances across {len(region_summaries)} regions")

    # Prediction tests
    prediction_tests = {}

    # 1. alpha vs bhattacharyya (across pairs)
    alphas_for_test = [r["alpha_avg"] for r in pair_records if r["alpha_avg"] is not None]
    bhatts_for_test = [r["bhattacharyya"] for r in pair_records if r["alpha_avg"] is not None]
    if len(alphas_for_test) >= 5:
        boot = _bootstrap_spearman(alphas_for_test, bhatts_for_test)
        perm_p = _permutation_test_spearman(alphas_for_test, bhatts_for_test)
        prediction_tests["alpha_vs_bhattacharyya"] = {
            **boot,
            "permutation_p": perm_p,
            "interpretation": (
                "Tests whether spectral dimensionality (alpha) predicts "
                "information-geometric distance between sessions. Positive rho "
                "means steeper spectra (lower effective dim) associate with "
                "larger distributional shifts."
            ),
        }

    # 2. alpha vs symmetrized KL
    kls_for_test = [r["symmetrized_kl"] for r in pair_records if r["alpha_avg"] is not None]
    if len(alphas_for_test) >= 5:
        boot = _bootstrap_spearman(alphas_for_test, kls_for_test)
        perm_p = _permutation_test_spearman(alphas_for_test, kls_for_test)
        prediction_tests["alpha_vs_kl"] = {
            **boot,
            "permutation_p": perm_p,
            "interpretation": (
                "Same as alpha_vs_bhattacharyya but using symmetrized KL divergence. "
                "Should correlate with Bhattacharyya if both capture similar structure."
            ),
        }

    # 3. bhattacharyya vs KL agreement
    all_bhatts = [r["bhattacharyya"] for r in pair_records]
    all_kls = [r["symmetrized_kl"] for r in pair_records]
    if len(all_bhatts) >= 5:
        boot = _bootstrap_spearman(all_bhatts, all_kls)
        perm_p = _permutation_test_spearman(all_bhatts, all_kls)
        prediction_tests["bhattacharyya_vs_kl"] = {
            **boot,
            "permutation_p": perm_p,
            "interpretation": (
                "Measures agreement between the two divergence measures. "
                "High positive rho indicates they capture similar structure."
            ),
        }

    # 4. within-mouse vs across-mouse distances
    within_bhatts = [r["bhattacharyya"] for r in pair_records if r["same_mouse"]]
    across_bhatts = [r["bhattacharyya"] for r in pair_records if not r["same_mouse"]]
    if len(within_bhatts) >= 3 and len(across_bhatts) >= 3:
        stat, p = mannwhitneyu(within_bhatts, across_bhatts, alternative="less")
        prediction_tests["within_mouse_vs_across"] = {
            "within_mean": float(np.mean(within_bhatts)),
            "across_mean": float(np.mean(across_bhatts)),
            "within_n": len(within_bhatts),
            "across_n": len(across_bhatts),
            "mann_whitney_U": float(stat),
            "p": float(p),
            "interpretation": (
                "Tests whether within-mouse Fisher-Rao distance is smaller than "
                "across-mouse distance (one-sided). Small p means within-mouse "
                "distributions are more similar, as expected if mouse identity "
                "shapes neural geometry."
            ),
        }

    # Sort regions by mean Bhattacharyya distance
    sorted_regions = sorted(
        region_summaries.items(),
        key=lambda x: x[1]["mean_bhattacharyya"],
    )

    results = {
        "timestamp": datetime.now().isoformat(),
        "n_sessions": len(sessions),
        "n_regions_analyzed": len(region_summaries),
        "n_pairs_total": len(pair_records),
        "region_summaries": region_summaries,
        "prediction_tests": prediction_tests,
        "most_stable_regions": [
            {"region": r, "mean_bhattacharyya": v["mean_bhattacharyya"],
             "mean_alpha": v["mean_alpha"], "n_pairs": v["n_pairs"]}
            for r, v in sorted_regions[:5]
        ],
        "least_stable_regions": [
            {"region": r, "mean_bhattacharyya": v["mean_bhattacharyya"],
             "mean_alpha": v["mean_alpha"], "n_pairs": v["n_pairs"]}
            for r, v in sorted_regions[-5:]
        ],
        "pair_records": pair_records,
    }

    out_path = RESULTS_DIR / "fisher_rao_distances.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Saved results to {out_path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exp39: Fisher-Rao information-geometric distance")
    parser.add_argument("--max-sessions", type=int, default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    run(max_sessions=args.max_sessions)
