"""Experiment 46: Cross-dataset generalization of geometric type taxonomy.

Tests whether the power-law exponent (alpha) taxonomy derived from
Steinmetz 2019 data generalizes to the IBL Brain-Wide Map dataset.

Core finding validated on Steinmetz (39 sessions, 10 mice, 73 regions):
  - Low alpha (flat eigenvalue spectrum) = high-dimensional = "Procrustes-type"
    (CKA fails, manifold metrics work)
  - High alpha (steep eigenvalue spectrum) = low-dimensional = "CKA-type"
    (CKA works well)

This experiment:
  1. Compute alpha per region from Steinmetz ("training" set).
  2. Compute alpha per region from IBL ("test" set) for overlapping regions.
  3. Test: does Steinmetz alpha predict IBL alpha? (Spearman + bootstrap CI)
  4. Test: does the CKA-type vs Procrustes-type classification transfer?
  5. Secondary: compute CKA and UMAP Procrustes on IBL, check anti-correlation.
"""
import argparse
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

from data.ibl import bin_spikes, filter_by_region, find_sessions_for_region, load_session
from data.steinmetz import get_choice_labels, get_region_activity, list_regions, load_all
from geometry.distances import cka

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent.parent / "artifacts" / "exp46"

# Regions likely to overlap between Steinmetz and IBL.
# Prioritized by expected neuron counts in both datasets.
TARGET_REGIONS = [
    "CA1", "VISp", "MOs", "DG", "ACA", "MOp", "VISl", "PL",
    "VISam", "VISpm", "SCm", "CP", "LS", "GPe",
]

MIN_NEURONS_STEINMETZ = 15
MIN_NEURONS_IBL = 20
TIME_WINDOW_STEINMETZ = slice(15, 35)  # ~150-350ms post-stim in 10ms bins
N_BOOTSTRAP = 1000
ALPHA_THRESHOLD = 1.5  # boundary between CKA-type (high) and Procrustes-type (low)
MAX_IBL_SESSIONS_PER_REGION = 5


def _power_law_exponent(activity: np.ndarray) -> float | None:
    """Compute power-law exponent alpha from the eigenvalue spectrum.

    Fits a line to log(eigenvalue) vs log(rank) for components 10-50.
    Higher alpha = steeper decay = more low-dimensional.

    Args:
        activity: (n_samples, n_features) matrix.

    Returns:
        Alpha (positive float) or None if insufficient components.
    """
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


def _bootstrap_spearman(x: np.ndarray, y: np.ndarray, n_boot: int = N_BOOTSTRAP) -> dict:
    """Spearman correlation with bootstrap confidence interval."""
    rho, p = spearmanr(x, y)
    boot_rhos = np.empty(n_boot)
    n = len(x)
    rng = np.random.default_rng(42)
    for i in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        r, _ = spearmanr(x[idx], y[idx])
        boot_rhos[i] = r
    ci_lo, ci_hi = np.percentile(boot_rhos, [2.5, 97.5])
    return {
        "rho": float(rho),
        "p_value": float(p),
        "n": int(n),
        "bootstrap_ci_95": [float(ci_lo), float(ci_hi)],
        "bootstrap_mean": float(np.mean(boot_rhos)),
        "bootstrap_std": float(np.std(boot_rhos)),
    }


def _umap_embed(activity: np.ndarray, n_components: int = 5) -> np.ndarray:
    from umap import UMAP

    reducer = UMAP(
        n_components=min(n_components, activity.shape[1] - 1),
        n_neighbors=15,
        min_dist=0.1,
        random_state=42,
    )
    return reducer.fit_transform(activity)


def compute_steinmetz_alphas(
    sessions: list[dict], max_sessions: int | None = None
) -> dict[str, list[float]]:
    """Compute alpha for each region across Steinmetz sessions.

    Returns {region: [alpha_per_session...]}.
    """
    if max_sessions:
        sessions = sessions[:max_sessions]

    region_alphas: dict[str, list[float]] = {}

    for sess_idx, sess in enumerate(tqdm(sessions, desc="Steinmetz alpha")):
        regions = list_regions(sess, min_neurons=MIN_NEURONS_STEINMETZ)
        for region in regions:
            act = get_region_activity(sess, region)
            if act is None or act.shape[1] < MIN_NEURONS_STEINMETZ:
                continue
            # Average across time window -> (n_trials, n_neurons)
            activity = act[:, :, TIME_WINDOW_STEINMETZ].mean(axis=2)
            alpha = _power_law_exponent(activity)
            if alpha is not None:
                region_alphas.setdefault(region, []).append(alpha)

    return region_alphas


def compute_ibl_alphas(
    target_regions: list[str],
    max_sessions_per_region: int = MAX_IBL_SESSIONS_PER_REGION,
) -> dict[str, list[float]]:
    """Compute alpha for each region across IBL sessions.

    Returns {region: [alpha_per_session...]}.
    Writes incremental results to JSONL so nothing is wasted on failure.
    """
    region_alphas: dict[str, list[float]] = {}
    jsonl_path = RESULTS_DIR / "ibl_alphas_incremental.jsonl"

    for region in tqdm(target_regions, desc="IBL regions"):
        logger.info(f"[{datetime.now().strftime('%H:%M:%S')}] Searching IBL sessions for {region}")
        try:
            sessions_meta = find_sessions_for_region(region, min_neurons=MIN_NEURONS_IBL)
        except Exception as e:
            logger.warning(f"Failed to find sessions for {region}: {e}")
            entry = {"ts": datetime.now().isoformat(), "region": region, "event": "search_failed", "error": str(e)}
            with open(jsonl_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
            continue

        sessions_meta = sessions_meta[:max_sessions_per_region]
        logger.info(f"  Found {len(sessions_meta)} insertions, loading up to {max_sessions_per_region}")
        entry = {"ts": datetime.now().isoformat(), "region": region, "event": "search_ok", "n_insertions": len(sessions_meta)}
        with open(jsonl_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

        for sess in tqdm(sessions_meta, desc=f"  {region}", leave=False):
            try:
                data = load_session(sess["eid"])
                counts = bin_spikes(
                    data["spike_times"],
                    data["spike_clusters"],
                    data["trial_intervals"],
                    bin_size=0.01,
                    pre_time=0.05,
                    post_time=0.45,
                )
                region_counts = filter_by_region(counts, data["cluster_regions"], region)

                if region_counts.shape[1] < MIN_NEURONS_IBL:
                    logger.info(f"    {sess['eid']}: only {region_counts.shape[1]} neurons, skipping")
                    continue

                activity = region_counts.mean(axis=2)
                alpha = _power_law_exponent(activity)

                if alpha is not None:
                    region_alphas.setdefault(region, []).append(alpha)
                    result = {
                        "ts": datetime.now().isoformat(), "region": region,
                        "event": "alpha_computed", "eid": sess["eid"],
                        "n_neurons": int(region_counts.shape[1]),
                        "n_trials": int(activity.shape[0]),
                        "alpha": float(alpha),
                    }
                    with open(jsonl_path, "a") as f:
                        f.write(json.dumps(result) + "\n")
                    logger.info(
                        f"    {sess['eid']}: {region_counts.shape[1]} neurons, "
                        f"{activity.shape[0]} trials, alpha={alpha:.3f}"
                    )
            except Exception as e:
                logger.warning(f"    Failed loading {sess['eid']}: {e}")

    return region_alphas


def compute_ibl_cka_procrustes(
    target_regions: list[str],
    max_sessions_per_region: int = MAX_IBL_SESSIONS_PER_REGION,
) -> dict:
    """Compute CKA and UMAP Procrustes for IBL region pairs.

    Returns dict with per-region CKA/Procrustes means and the anti-correlation test.
    """
    region_session_data: dict[str, list[dict]] = {}

    for region in tqdm(target_regions, desc="IBL CKA/Procrustes"):
        try:
            sessions_meta = find_sessions_for_region(region, min_neurons=MIN_NEURONS_IBL)
        except Exception:
            continue

        sessions_meta = sessions_meta[:max_sessions_per_region]

        for sess in sessions_meta:
            try:
                data = load_session(sess["eid"])
                counts = bin_spikes(
                    data["spike_times"],
                    data["spike_clusters"],
                    data["trial_intervals"],
                    bin_size=0.01,
                    pre_time=0.05,
                    post_time=0.45,
                )
                region_counts = filter_by_region(counts, data["cluster_regions"], region)
                if region_counts.shape[1] < MIN_NEURONS_IBL:
                    continue

                activity = region_counts.mean(axis=2)
                embedding = _umap_embed(activity)

                region_session_data.setdefault(region, []).append({
                    "activity": activity,
                    "embedding": embedding,
                    "n_neurons": region_counts.shape[1],
                    "n_trials": activity.shape[0],
                })
            except Exception as e:
                logger.warning(f"CKA/Proc failed for {region}/{sess['eid']}: {e}")

    region_cka_means = {}
    region_proc_means = {}
    pairs = []

    for region, measurements in region_session_data.items():
        if len(measurements) < 2:
            continue

        cka_vals = []
        proc_vals = []
        for i, j in combinations(range(len(measurements)), 2):
            m1, m2 = measurements[i], measurements[j]
            n_shared = min(m1["n_trials"], m2["n_trials"])

            cka_val = cka(m1["activity"][:n_shared], m2["activity"][:n_shared])
            cka_vals.append(float(cka_val))

            try:
                n_emb = min(m1["embedding"].shape[0], m2["embedding"].shape[0])
                _, _, d = procrustes(m1["embedding"][:n_emb], m2["embedding"][:n_emb])
                proc_vals.append(float(d))
            except Exception:
                pass

        if cka_vals:
            region_cka_means[region] = float(np.mean(cka_vals))
        if proc_vals:
            region_proc_means[region] = float(np.mean(proc_vals))

        pairs.append({
            "region": region,
            "n_sessions": len(measurements),
            "cka_mean": float(np.mean(cka_vals)) if cka_vals else None,
            "cka_std": float(np.std(cka_vals)) if len(cka_vals) > 1 else None,
            "procrustes_mean": float(np.mean(proc_vals)) if proc_vals else None,
            "procrustes_std": float(np.std(proc_vals)) if len(proc_vals) > 1 else None,
        })

    # Anti-correlation test: does CKA anti-correlate with Procrustes across regions?
    anti_correlation = None
    shared_regions = sorted(set(region_cka_means) & set(region_proc_means))
    if len(shared_regions) >= 4:
        cka_arr = np.array([region_cka_means[r] for r in shared_regions])
        proc_arr = np.array([region_proc_means[r] for r in shared_regions])
        rho, p = spearmanr(cka_arr, proc_arr)
        anti_correlation = {
            "regions": shared_regions,
            "spearman_rho": float(rho),
            "p_value": float(p),
            "n_regions": len(shared_regions),
            "interpretation": (
                "Negative rho confirms the CKA-Procrustes anti-correlation "
                "found in Steinmetz also holds in IBL."
            ),
        }

    return {
        "pairs": pairs,
        "anti_correlation": anti_correlation,
    }


def run(max_sessions: int | None = None):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().isoformat()
    logger.info(f"[{timestamp}] Starting exp46: cross-dataset alpha generalization")

    # --- Step 1: Steinmetz alphas ---
    logger.info("Step 1: Computing Steinmetz alphas")
    sessions = load_all()
    steinmetz_raw = compute_steinmetz_alphas(sessions, max_sessions=max_sessions)
    steinmetz_alphas = {r: float(np.mean(vs)) for r, vs in steinmetz_raw.items() if vs}
    logger.info(f"  Steinmetz: alpha computed for {len(steinmetz_alphas)} regions")

    # --- Step 2: IBL alphas for overlapping regions ---
    # Focus on regions present in Steinmetz that are also likely in IBL.
    overlapping_targets = [r for r in TARGET_REGIONS if r in steinmetz_alphas]
    # Also include any Steinmetz regions not in TARGET_REGIONS but worth trying.
    extra = [r for r in steinmetz_alphas if r not in overlapping_targets][:5]
    ibl_target_regions = overlapping_targets + extra
    logger.info(f"Step 2: Computing IBL alphas for {len(ibl_target_regions)} target regions")

    ibl_raw = compute_ibl_alphas(
        ibl_target_regions,
        max_sessions_per_region=MAX_IBL_SESSIONS_PER_REGION,
    )
    ibl_alphas = {r: float(np.mean(vs)) for r, vs in ibl_raw.items() if vs}
    logger.info(f"  IBL: alpha computed for {len(ibl_alphas)} regions")

    # --- Step 3: Match regions and test correlation ---
    matched_regions = sorted(set(steinmetz_alphas) & set(ibl_alphas))
    logger.info(f"Step 3: {len(matched_regions)} matched regions")

    alpha_correlation = None
    per_region_comparison = []
    for region in matched_regions:
        per_region_comparison.append({
            "region": region,
            "steinmetz_alpha": steinmetz_alphas[region],
            "ibl_alpha": ibl_alphas[region],
            "steinmetz_n_sessions": len(steinmetz_raw.get(region, [])),
            "ibl_n_sessions": len(ibl_raw.get(region, [])),
        })

    if len(matched_regions) >= 4:
        s_arr = np.array([steinmetz_alphas[r] for r in matched_regions])
        i_arr = np.array([ibl_alphas[r] for r in matched_regions])
        alpha_correlation = _bootstrap_spearman(s_arr, i_arr)
        logger.info(
            f"  Alpha correlation: rho={alpha_correlation['rho']:.3f}, "
            f"p={alpha_correlation['p_value']:.4f}, "
            f"CI={alpha_correlation['bootstrap_ci_95']}"
        )

    # --- Step 4: Geometric type classification transfer ---
    classification_transfer = None
    if len(matched_regions) >= 4:
        steinmetz_types = {r: "CKA-type" if steinmetz_alphas[r] > ALPHA_THRESHOLD else "Procrustes-type" for r in matched_regions}
        ibl_types = {r: "CKA-type" if ibl_alphas[r] > ALPHA_THRESHOLD else "Procrustes-type" for r in matched_regions}

        n_agree = sum(1 for r in matched_regions if steinmetz_types[r] == ibl_types[r])
        n_total = len(matched_regions)
        accuracy = n_agree / n_total

        # Permutation test for classification agreement
        rng = np.random.default_rng(42)
        n_perm = 10000
        perm_agrees = np.empty(n_perm)
        ibl_type_list = [ibl_types[r] for r in matched_regions]
        steinmetz_type_list = [steinmetz_types[r] for r in matched_regions]
        for p_idx in range(n_perm):
            shuffled = rng.permutation(ibl_type_list)
            perm_agrees[p_idx] = np.mean([s == sh for s, sh in zip(steinmetz_type_list, shuffled)])
        p_val_perm = float(np.mean(perm_agrees >= accuracy))

        classification_transfer = {
            "threshold": ALPHA_THRESHOLD,
            "n_regions": n_total,
            "n_agree": n_agree,
            "accuracy": accuracy,
            "permutation_p_value": p_val_perm,
            "steinmetz_types": steinmetz_types,
            "ibl_types": ibl_types,
            "per_region": [
                {
                    "region": r,
                    "steinmetz_type": steinmetz_types[r],
                    "ibl_type": ibl_types[r],
                    "match": steinmetz_types[r] == ibl_types[r],
                }
                for r in matched_regions
            ],
        }
        logger.info(
            f"  Classification transfer: {n_agree}/{n_total} agree "
            f"(accuracy={accuracy:.2f}, perm p={p_val_perm:.4f})"
        )

    # --- Step 5: CKA/Procrustes anti-correlation on IBL ---
    logger.info("Step 5: CKA/Procrustes anti-correlation on IBL")
    cka_proc_results = compute_ibl_cka_procrustes(
        [r for r in matched_regions if r in ibl_alphas],
        max_sessions_per_region=MAX_IBL_SESSIONS_PER_REGION,
    )

    # --- Assemble results ---
    results = {
        "timestamp": timestamp,
        "config": {
            "min_neurons_steinmetz": MIN_NEURONS_STEINMETZ,
            "min_neurons_ibl": MIN_NEURONS_IBL,
            "time_window_steinmetz": str(TIME_WINDOW_STEINMETZ),
            "alpha_threshold": ALPHA_THRESHOLD,
            "n_bootstrap": N_BOOTSTRAP,
            "max_ibl_sessions_per_region": MAX_IBL_SESSIONS_PER_REGION,
        },
        "steinmetz_alphas": steinmetz_alphas,
        "steinmetz_alpha_per_session": {r: vs for r, vs in steinmetz_raw.items()},
        "ibl_alphas": ibl_alphas,
        "ibl_alpha_per_session": {r: vs for r, vs in ibl_raw.items()},
        "matched_regions": matched_regions,
        "n_matched": len(matched_regions),
        "alpha_correlation": alpha_correlation,
        "per_region_comparison": per_region_comparison,
        "classification_transfer": classification_transfer,
        "cka_procrustes_anticorrelation": cka_proc_results,
    }

    out_path = RESULTS_DIR / "cross_dataset_alpha.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {out_path}")

    # Print summary
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info(f"  Steinmetz regions with alpha: {len(steinmetz_alphas)}")
    logger.info(f"  IBL regions with alpha: {len(ibl_alphas)}")
    logger.info(f"  Matched regions: {len(matched_regions)}")
    if alpha_correlation:
        logger.info(
            f"  Alpha correlation: rho={alpha_correlation['rho']:.3f} "
            f"[{alpha_correlation['bootstrap_ci_95'][0]:.3f}, "
            f"{alpha_correlation['bootstrap_ci_95'][1]:.3f}]"
        )
    if classification_transfer:
        logger.info(
            f"  Type classification transfer: "
            f"{classification_transfer['n_agree']}/{classification_transfer['n_regions']} "
            f"(p={classification_transfer['permutation_p_value']:.4f})"
        )
    if cka_proc_results.get("anti_correlation"):
        ac = cka_proc_results["anti_correlation"]
        logger.info(f"  CKA-Procrustes anti-correlation: rho={ac['spearman_rho']:.3f} (p={ac['p_value']:.4f})")
    logger.info("=" * 60)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Exp46: Cross-dataset generalization of geometric type taxonomy"
    )
    parser.add_argument(
        "--max-sessions", type=int, default=None,
        help="Limit Steinmetz sessions (for faster testing)",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    run(max_sessions=args.max_sessions)
