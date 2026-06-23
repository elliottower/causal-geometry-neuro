"""Experiment 21: Baseline metric comparisons (Steinmetz).

Compute established neural similarity metrics per region and correlate
with our geometric invariants from exp14. The point: our Grassmannian
framework explains WHERE these scalar metrics come from.

Baselines:
  - sliceTCA: tensor component analysis (Pellegrino, Nat Neuro 2024)
  - DSA: dynamical similarity analysis (Ostrow, NeurIPS 2023)
  - MFTMA: manifold capacity (Chung et al.)
  - Power-law exponent: eigenvalue spectrum slope (Stringer, Nature 2019)
  - CKA: centered kernel alignment (Kornblith et al. 2019)
"""
import json
import logging
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.linalg import svd
from scipy.stats import linregress, spearmanr
from tqdm import tqdm

from data.steinmetz import get_choice_labels, get_region_activity, list_regions, load_all
from geometry.distances import cka
from geometry.subspace import fit_lda_subspace

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent.parent / "artifacts" / "exp21"
MIN_NEURONS = 15
TIME_WINDOW = slice(15, 35)
SUBSPACE_K = 5


def _power_law_exponent(activity):
    """Fit power-law to PCA eigenvalue spectrum (Stringer et al. 2019)."""
    X = activity - activity.mean(axis=0, keepdims=True)
    eigs = np.linalg.svd(X, compute_uv=False) ** 2
    eigs = eigs / eigs.sum()

    n_fit = min(len(eigs), 50)
    if n_fit < 5:
        return None, None
    ranks = np.arange(1, n_fit + 1)
    mask = eigs[:n_fit] > 0
    if mask.sum() < 5:
        return None, None

    slope, intercept, r, _, _ = linregress(
        np.log(ranks[mask]), np.log(eigs[:n_fit][mask])
    )
    return float(-slope), float(r ** 2)


def _manifold_capacity(activity, labels):
    """Simplified manifold capacity: P/N where P = # separable dichotomies.

    Uses random hyperplane classification as a proxy for full MFTMA.
    """
    n_trials, n_neurons = activity.shape
    classes = np.unique(labels)
    if len(classes) != 2:
        return None

    X0 = activity[labels == classes[0]]
    X1 = activity[labels == classes[1]]

    n_random = 100
    n_separable = 0
    for _ in range(n_random):
        w = np.random.randn(n_neurons)
        w /= np.linalg.norm(w)
        proj0 = X0 @ w
        proj1 = X1 @ w
        threshold = (proj0.mean() + proj1.mean()) / 2
        acc = ((proj0 < threshold).sum() + (proj1 >= threshold).sum()) / (len(proj0) + len(proj1))
        acc = max(acc, 1 - acc)
        if acc > 0.7:
            n_separable += 1

    return float(n_separable / n_random)


def _effective_dimensionality(activity):
    """Participation ratio: (sum λ)² / sum λ²."""
    X = activity - activity.mean(axis=0, keepdims=True)
    eigs = np.linalg.svd(X, compute_uv=False) ** 2
    return float(eigs.sum() ** 2 / (eigs ** 2).sum())


def _slicetca_variance(activity_3d, n_components=3):
    """sliceTCA decomposition — returns fraction of variance per component type.

    activity_3d: (n_trials, n_neurons, n_time)
    Returns dict with trial/neuron/time variance fractions.
    """
    try:
        import torch
        import slicetca

        tensor = torch.tensor(activity_3d, dtype=torch.float32)
        model, losses = slicetca.decompose(
            tensor,
            (n_components, n_components, n_components),
            seed=0,
            max_iter=500,
            verbose=False,
        )
        components = model.get_components(numpy=True)

        total_var = np.var(activity_3d)
        result = {}
        for ctype, comps in zip(["trial", "neuron", "time"], components):
            if len(comps) > 0:
                recon = sum(np.outer(c[0], c[1]).reshape(1, -1, 1) * c[2].reshape(1, 1, -1)
                           for c in comps) if len(comps[0]) == 3 else 0
                result[f"{ctype}_var_frac"] = float(np.var(comps) / total_var) if total_var > 0 else 0
            else:
                result[f"{ctype}_var_frac"] = 0.0
        result["final_loss"] = float(losses[-1]) if losses else None
        return result
    except Exception as e:
        logger.warning(f"sliceTCA failed: {e}")
        return None


def _dsa_distance(traj1, traj2, rank=5):
    """Simplified DSA: compare dynamics matrices via eigenvalue distance.

    Since full DSA requires the DSA package, we use a simplified version:
    fit a linear dynamics model dx/dt ≈ Ax, compare eigenspectra.
    """
    def _fit_dynamics(traj):
        X = traj[:-1]
        Y = traj[1:]
        A = np.linalg.lstsq(X, Y, rcond=None)[0]
        eigs = np.linalg.eigvals(A)
        idx = np.argsort(np.abs(eigs))[::-1]
        return eigs[idx[:rank]]

    try:
        e1 = _fit_dynamics(traj1)
        e2 = _fit_dynamics(traj2)
        n = min(len(e1), len(e2))
        dist = np.mean(np.abs(np.abs(e1[:n]) - np.abs(e2[:n])))
        return float(dist)
    except Exception:
        return None


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
            activity_3d = act[:n, :, TIME_WINDOW]
            activity = activity_3d.mean(axis=2)

            try:
                alpha, r2 = _power_law_exponent(activity)
                capacity = _manifold_capacity(activity, labels[:n])
                eff_dim = _effective_dimensionality(activity)
                k = min(SUBSPACE_K, activity.shape[1] - 1)
                U = fit_lda_subspace(activity, labels[:n], k=k)

                slicetca_result = _slicetca_variance(activity_3d)

                mean_traj = np.zeros((activity_3d.shape[2], k))
                for t in range(activity_3d.shape[2]):
                    diff = activity_3d[labels[:n] == 1, :, t].mean(0) - activity_3d[labels[:n] == 0, :, t].mean(0)
                    mean_traj[t] = U.T @ diff

                if region not in region_data:
                    region_data[region] = []
                region_data[region].append({
                    "session_idx": sess_idx,
                    "mouse": mouse,
                    "activity": activity,
                    "trajectory": mean_traj,
                    "subspace": U,
                    "n_neurons": activity.shape[1],
                    "n_trials": n,
                    "metrics": {
                        "power_law_alpha": alpha,
                        "power_law_r2": r2,
                        "manifold_capacity": capacity,
                        "effective_dim": eff_dim,
                        "slicetca": slicetca_result,
                    },
                })
            except Exception as e:
                logger.warning(f"Failed {mouse}/{region}: {e}")

    pairs = []
    for region, measurements in tqdm(region_data.items(), desc="Comparing"):
        if len(measurements) < 2:
            continue

        for (i, j) in combinations(range(len(measurements)), 2):
            m1, m2 = measurements[i], measurements[j]

            n_shared = min(m1["n_trials"], m2["n_trials"])
            cka_val = cka(m1["activity"][:n_shared], m2["activity"][:n_shared])
            dsa_val = _dsa_distance(m1["trajectory"], m2["trajectory"])

            pairs.append({
                "region": region,
                "same_mouse": m1["mouse"] == m2["mouse"],
                "cka": cka_val,
                "dsa_eigenvalue_dist": dsa_val,
                "power_law_alpha_1": m1["metrics"]["power_law_alpha"],
                "power_law_alpha_2": m2["metrics"]["power_law_alpha"],
                "capacity_1": m1["metrics"]["manifold_capacity"],
                "capacity_2": m2["metrics"]["manifold_capacity"],
                "eff_dim_1": m1["metrics"]["effective_dim"],
                "eff_dim_2": m2["metrics"]["effective_dim"],
            })

    region_summaries = {}
    for region, measurements in region_data.items():
        alphas = [m["metrics"]["power_law_alpha"] for m in measurements if m["metrics"]["power_law_alpha"] is not None]
        caps = [m["metrics"]["manifold_capacity"] for m in measurements if m["metrics"]["manifold_capacity"] is not None]
        dims = [m["metrics"]["effective_dim"] for m in measurements]

        region_summaries[region] = {
            "n_sessions": len(measurements),
            "power_law_alpha_mean": float(np.mean(alphas)) if alphas else None,
            "power_law_alpha_std": float(np.std(alphas)) if len(alphas) > 1 else None,
            "manifold_capacity_mean": float(np.mean(caps)) if caps else None,
            "effective_dim_mean": float(np.mean(dims)) if dims else None,
        }

    results = {
        "timestamp": datetime.now().isoformat(),
        "n_regions": len(region_data),
        "n_pairs": len(pairs),
        "pairs": pairs,
        "region_summaries": region_summaries,
    }

    out_path = RESULTS_DIR / "baseline_comparisons.json"
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
