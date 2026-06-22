"""Experiment 27: Tucker decomposition of population tensors.

Tucker decomposition decomposes (trials, neurons, time) tensors into
core tensor + factor matrices along each mode. This gives:
  - Trial factors: shared vs unique trial structure
  - Neuron factors: population subspace structure
  - Time factors: temporal dynamics structure

Key tests:
  1. Do the neuron factors from Tucker correlate with the CKA/Procrustes
     geometric type? (Tucker neuron rank ~ effective dimensionality)
  2. Are Tucker core tensors more conserved across sessions than raw activity?
  3. Does Tucker R^2 (reconstruction quality) vary with geometric type?

Tucker is a natural fit because it separates the three modes that CKA and
Procrustes conflate: CKA compares across neurons (collapsing time), UMAP
embeds across time (collapsing trial structure), Tucker separates all three.
"""
import json
import logging
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from tqdm import tqdm

from data.steinmetz import get_choice_labels, get_region_activity, list_regions, load_all
from geometry.distances import cka

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent.parent / "artifacts" / "exp27"
MIN_NEURONS = 15
TIME_WINDOW = slice(15, 35)
TUCKER_RANKS = (5, 5, 5)


def _tucker_decompose(tensor_3d, ranks=(5, 5, 5)):
    n_trials, n_neurons, n_time = tensor_3d.shape
    r0 = min(ranks[0], n_trials - 1)
    r1 = min(ranks[1], n_neurons - 1)
    r2 = min(ranks[2], n_time - 1)
    if r0 < 1 or r1 < 1 or r2 < 1:
        raise ValueError(f"Too small for Tucker: {tensor_3d.shape}")

    def _left_svd(mat, k):
        U, _, _ = np.linalg.svd(mat - mat.mean(axis=0), full_matrices=False)
        return U[:, :k]

    unf0 = tensor_3d.reshape(n_trials, -1)
    unf1 = tensor_3d.transpose(1, 0, 2).reshape(n_neurons, -1)
    unf2 = tensor_3d.transpose(2, 0, 1).reshape(n_time, -1)

    U0 = _left_svd(unf0, r0)
    U1 = _left_svd(unf1, r1)
    U2 = _left_svd(unf2, r2)

    projected = np.einsum("ijk,ia->ajk", tensor_3d, U0)
    projected = np.einsum("ajk,jb->abk", projected, U1)
    core = np.einsum("abk,kc->abc", projected, U2)

    recon = np.einsum("abc,ia->ibc", core, U0)
    recon = np.einsum("ibc,jb->ijc", recon, U1)
    recon = np.einsum("ijc,kc->ijk", recon, U2)

    ss_total = np.sum((tensor_3d - tensor_3d.mean()) ** 2)
    ss_resid = np.sum((tensor_3d - recon) ** 2)
    r_squared = 1 - ss_resid / ss_total if ss_total > 0 else 0.0

    return {
        "trial_factors": U0,
        "neuron_factors": U1,
        "time_factors": U2,
        "core": core,
        "r_squared": float(r_squared),
        "ranks": (r0, r1, r2),
    }


def _factor_similarity(U1, U2):
    n_shared = min(U1.shape[0], U2.shape[0])
    k = min(U1.shape[1], U2.shape[1])
    u1 = U1[:n_shared, :k]
    u2 = U2[:n_shared, :k]
    u1 = u1 / (np.linalg.norm(u1, axis=0, keepdims=True) + 1e-8)
    u2 = u2 / (np.linalg.norm(u2, axis=0, keepdims=True) + 1e-8)
    return float(np.mean(np.abs(np.diag(u1.T @ u2))))


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
        logger.info(f"Session {sess_idx} ({mouse}): {len(regions)} regions")

        for region in regions:
            act = get_region_activity(sess, region)
            if act is None:
                continue
            logger.info(f"  {region}: act.shape={act.shape}, ndim={act.ndim}")
            if act.ndim == 2:
                continue
            if act.shape[1] < MIN_NEURONS:
                continue

            n = min(act.shape[0], len(labels))
            activity_3d = act[:n, :, TIME_WINDOW]
            activity_2d = activity_3d.mean(axis=2)

            alpha = _power_law_exponent(activity_2d)

            try:
                tucker = _tucker_decompose(activity_3d, TUCKER_RANKS)
            except Exception as e:
                logger.error(f"Tucker failed {mouse}/{region} shape={activity_3d.shape}: {e}")
                continue

            if region not in region_data:
                region_data[region] = []
            region_data[region].append({
                "session_idx": sess_idx,
                "mouse": mouse,
                "activity_2d": activity_2d,
                "tucker": tucker,
                "n_neurons": activity_2d.shape[1],
                "n_trials": n,
                "alpha": alpha,
            })

    region_profiles = {}
    pairs = []

    for region, measurements in tqdm(region_data.items(), desc="Regions"):
        r2s = [m["tucker"]["r_squared"] for m in measurements]
        alphas = [m["alpha"] for m in measurements if m["alpha"] is not None]

        region_profiles[region] = {
            "n_sessions": len(measurements),
            "tucker_r2_mean": float(np.mean(r2s)),
            "alpha_mean": float(np.mean(alphas)) if alphas else None,
        }

        if len(measurements) < 2:
            continue

        for (i, j) in combinations(range(len(measurements)), 2):
            m1, m2 = measurements[i], measurements[j]
            n_shared = min(m1["n_trials"], m2["n_trials"])

            cka_val = cka(m1["activity_2d"][:n_shared], m2["activity_2d"][:n_shared])

            neuron_sim = _factor_similarity(
                m1["tucker"]["neuron_factors"],
                m2["tucker"]["neuron_factors"],
            )
            time_sim = _factor_similarity(
                m1["tucker"]["time_factors"],
                m2["tucker"]["time_factors"],
            )
            trial_sim = _factor_similarity(
                m1["tucker"]["trial_factors"],
                m2["tucker"]["trial_factors"],
            )

            n_core = min(m1["tucker"]["core"].shape[0], m2["tucker"]["core"].shape[0])
            c1 = m1["tucker"]["core"][:n_core, :n_core, :n_core].ravel()
            c2 = m2["tucker"]["core"][:n_core, :n_core, :n_core].ravel()
            core_corr = float(np.corrcoef(c1, c2)[0, 1]) if len(c1) > 1 else None

            pairs.append({
                "region": region,
                "cka": float(cka_val),
                "neuron_factor_similarity": neuron_sim,
                "time_factor_similarity": time_sim,
                "trial_factor_similarity": trial_sim,
                "core_correlation": core_corr,
            })

    tucker_vs_cka = {}
    if len(pairs) >= 4:
        rho_n, pv_n = spearmanr([p["cka"] for p in pairs], [p["neuron_factor_similarity"] for p in pairs])
        rho_t, pv_t = spearmanr([p["cka"] for p in pairs], [p["time_factor_similarity"] for p in pairs])
        valid_core = [p for p in pairs if p["core_correlation"] is not None]
        rho_c, pv_c = (None, None)
        if len(valid_core) >= 4:
            rho_c, pv_c = spearmanr([p["cka"] for p in valid_core], [p["core_correlation"] for p in valid_core])

        tucker_vs_cka = {
            "cka_vs_neuron_factors": {"rho": float(rho_n), "p": float(pv_n)},
            "cka_vs_time_factors": {"rho": float(rho_t), "p": float(pv_t)},
            "cka_vs_core": {"rho": float(rho_c) if rho_c is not None else None,
                            "p": float(pv_c) if pv_c is not None else None},
        }

    r2_vs_alpha = {}
    regions_with_both = [(r, p) for r, p in region_profiles.items() if p["alpha_mean"] is not None]
    if len(regions_with_both) >= 4:
        rho_ra, pv_ra = spearmanr(
            [p["tucker_r2_mean"] for _, p in regions_with_both],
            [p["alpha_mean"] for _, p in regions_with_both],
        )
        r2_vs_alpha = {
            "rho": float(rho_ra),
            "p_value": float(pv_ra),
            "interpretation": (
                "Positive rho means steep-spectrum (low-dim) regions are better "
                "captured by Tucker — consistent with Tucker as a linear method "
                "that favors low-dimensional structure."
            ),
        }

    results = {
        "timestamp": datetime.now().isoformat(),
        "n_regions": len(region_data),
        "n_pairs": len(pairs),
        "region_profiles": region_profiles,
        "tucker_vs_cka": tucker_vs_cka,
        "r2_vs_alpha": r2_vs_alpha,
        "pairs_summary": {
            "neuron_sim_mean": float(np.mean([p["neuron_factor_similarity"] for p in pairs])) if pairs else None,
            "time_sim_mean": float(np.mean([p["time_factor_similarity"] for p in pairs])) if pairs else None,
            "trial_sim_mean": float(np.mean([p["trial_factor_similarity"] for p in pairs])) if pairs else None,
        },
    }

    out_path = RESULTS_DIR / "tucker_decomposition.json"
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
