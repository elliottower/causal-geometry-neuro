"""Experiment 31: Causal representation learning across brain regions.

Each brain region is treated as a "modality." We identify shared causal
subspaces between region pairs — the part of region A's activity that is
causally shared with region B, versus region-A-specific activity.

Based on multi-modal causal representation learning identifiability theory
(PMC 2025): under structural sparsity, latent causal subspaces can be
identified from multi-modal data.

This provides a principled way to estimate inter-regional causal structure
and tests whether the shared causal subspace dimensionality correlates
with geometric type (alpha).

Key predictions:
1. The shared causal subspace between two regions is lower-dimensional
   than either region's full activity subspace.
2. The ratio (shared_dim / full_dim) varies systematically with alpha:
   high-alpha (flat spectrum) regions share proportionally less of their
   activity with other regions.
3. This gives a principled replacement for ad-hoc restriction maps.
"""
import json
import logging
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from tqdm import tqdm

from data.steinmetz import get_choice_labels, get_region_activity, list_regions, load_all
from geometry.distances import cka

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent.parent / "artifacts" / "exp31"
MIN_NEURONS = 15
TIME_WINDOW = slice(15, 35)
MAX_CCA_DIMS = 10


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


def _effective_dim(activity):
    pca = PCA(n_components=min(50, activity.shape[1], activity.shape[0]))
    pca.fit(activity)
    ev = pca.explained_variance_
    ev = ev[ev > 0]
    return float((ev.sum() ** 2) / (ev ** 2).sum())


def _shared_causal_subspace(X_a, X_b, max_dims=MAX_CCA_DIMS):
    """Identify shared causal subspace via CCA + significance testing.

    The shared subspace is the set of CCA dimensions where the canonical
    correlation significantly exceeds shuffled baseline. This is the
    multi-modal identifiability approach: dimensions with high canonical
    correlation across "modalities" (regions) capture shared causal content.

    Returns: (shared_dim, canonical_correlations, total_cca_dims)
    """
    n = min(X_a.shape[0], X_b.shape[0])
    X_a, X_b = X_a[:n], X_b[:n]

    n_dims = min(max_dims, X_a.shape[1] - 1, X_b.shape[1] - 1, n - 1)
    if n_dims < 1:
        return None, None, None

    try:
        cca = CCA(n_components=n_dims, max_iter=500)
        X_a_c, X_b_c = cca.fit_transform(X_a, X_b)
    except Exception:
        return None, None, None

    canonical_corrs = np.array([
        np.corrcoef(X_a_c[:, i], X_b_c[:, i])[0, 1]
        for i in range(n_dims)
    ])

    rng = np.random.default_rng(42)
    n_shuffles = 20
    shuffle_corrs = np.zeros((n_shuffles, n_dims))
    for s in range(n_shuffles):
        X_b_shuf = X_b[rng.permutation(n)]
        try:
            cca_s = CCA(n_components=n_dims, max_iter=500)
            X_a_s, X_b_s = cca_s.fit_transform(X_a, X_b_shuf)
            for i in range(n_dims):
                shuffle_corrs[s, i] = np.corrcoef(X_a_s[:, i], X_b_s[:, i])[0, 1]
        except Exception:
            pass

    shuffle_95 = np.percentile(shuffle_corrs, 95, axis=0)
    shared_dim = int(np.sum(canonical_corrs > shuffle_95))

    return shared_dim, canonical_corrs.tolist(), n_dims


def _region_specific_dim(activity, shared_dim):
    """Estimate region-specific dimensionality = effective_dim - shared_dim."""
    eff = _effective_dim(activity)
    return max(0, eff - shared_dim)


def run(max_sessions: int | None = None):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    sessions = load_all()
    if max_sessions:
        sessions = sessions[:max_sessions]

    session_regions = {}

    for sess_idx, sess in enumerate(tqdm(sessions, desc="Loading sessions")):
        labels = get_choice_labels(sess)
        if len(np.unique(labels)) < 2:
            continue

        mouse = str(sess.get("mouse_name", f"mouse_{sess_idx}"))
        regions = list_regions(sess, min_neurons=MIN_NEURONS)
        region_activities = {}

        for region in regions:
            act = get_region_activity(sess, region)
            if act is None or act.shape[1] < MIN_NEURONS:
                continue
            n = min(act.shape[0], len(labels))
            activity = act[:n, :, TIME_WINDOW].mean(axis=2)
            region_activities[region] = activity

        if len(region_activities) >= 2:
            session_regions[sess_idx] = {
                "mouse": mouse,
                "activities": region_activities,
            }

    logger.info(f"Loaded {len(session_regions)} sessions")

    region_pair_shared_dims = {}
    region_alphas = {}
    region_eff_dims = {}

    for sess_idx, sr in tqdm(session_regions.items(), desc="CCA analysis"):
        regions = sorted(sr["activities"].keys())

        for region in regions:
            act = sr["activities"][region]
            if region not in region_alphas:
                alpha = _power_law_exponent(act)
                if alpha is not None:
                    region_alphas[region] = alpha
                    region_eff_dims[region] = _effective_dim(act)

        for r1, r2 in combinations(regions, 2):
            act1 = sr["activities"][r1]
            act2 = sr["activities"][r2]

            shared_dim, canon_corrs, total_dims = _shared_causal_subspace(act1, act2)
            if shared_dim is None:
                continue

            pair_key = f"{r1}_{r2}"
            if pair_key not in region_pair_shared_dims:
                region_pair_shared_dims[pair_key] = []

            region_pair_shared_dims[pair_key].append({
                "sess_idx": sess_idx,
                "mouse": sr["mouse"],
                "shared_dim": shared_dim,
                "canonical_correlations": canon_corrs,
                "total_cca_dims": total_dims,
                "r1_neurons": act1.shape[1],
                "r2_neurons": act2.shape[1],
            })

    pair_summaries = {}
    for pair_key, measurements in region_pair_shared_dims.items():
        r1, r2 = pair_key.split("_", 1)
        if "_" in r2:
            parts = pair_key.split("_")
            r1, r2 = parts[0], parts[1]

        shared_dims = [m["shared_dim"] for m in measurements]
        pair_summaries[pair_key] = {
            "regions": [r1, r2],
            "n_sessions": len(measurements),
            "mean_shared_dim": float(np.mean(shared_dims)),
            "std_shared_dim": float(np.std(shared_dims)),
            "r1_alpha": region_alphas.get(r1),
            "r2_alpha": region_alphas.get(r2),
            "r1_eff_dim": region_eff_dims.get(r1),
            "r2_eff_dim": region_eff_dims.get(r2),
            "mean_top_canonical_corr": float(np.mean([
                m["canonical_correlations"][0] for m in measurements
                if m["canonical_correlations"]
            ])) if measurements else None,
        }

        if region_eff_dims.get(r1) and region_eff_dims.get(r2):
            min_eff = min(region_eff_dims[r1], region_eff_dims[r2])
            pair_summaries[pair_key]["shared_fraction"] = (
                float(np.mean(shared_dims) / min_eff) if min_eff > 0 else None
            )

    region_shared_profile = {}
    for pair_key, ps in pair_summaries.items():
        for r in ps["regions"]:
            if r not in region_shared_profile:
                region_shared_profile[r] = {
                    "alpha": region_alphas.get(r),
                    "eff_dim": region_eff_dims.get(r),
                    "shared_dims_with_partners": [],
                    "shared_fractions": [],
                }
            if ps.get("shared_fraction") is not None:
                region_shared_profile[r]["shared_fractions"].append(ps["shared_fraction"])
            region_shared_profile[r]["shared_dims_with_partners"].append(ps["mean_shared_dim"])

    for r, profile in region_shared_profile.items():
        profile["mean_shared_dim_across_partners"] = float(np.mean(profile["shared_dims_with_partners"]))
        profile["mean_shared_fraction"] = (
            float(np.mean(profile["shared_fractions"])) if profile["shared_fractions"] else None
        )

    prediction_tests = {}

    valid_regions = [
        (r, p) for r, p in region_shared_profile.items()
        if p["alpha"] is not None and p.get("mean_shared_fraction") is not None
    ]

    if len(valid_regions) >= 4:
        alphas = [p["alpha"] for _, p in valid_regions]
        fractions = [p["mean_shared_fraction"] for _, p in valid_regions]
        rho, p = spearmanr(alphas, fractions)
        prediction_tests["alpha_vs_shared_fraction"] = {
            "rho": float(rho), "p": float(p), "n": len(valid_regions),
            "interpretation": (
                "Negative rho means high-alpha (flat spectrum) regions share "
                "proportionally less of their activity with other regions — "
                "more of their dimensionality is region-specific."
            )
        }

    if len(valid_regions) >= 4:
        alphas = [p["alpha"] for _, p in valid_regions]
        shared_dims = [p["mean_shared_dim_across_partners"] for _, p in valid_regions]
        rho, p = spearmanr(alphas, shared_dims)
        prediction_tests["alpha_vs_absolute_shared_dim"] = {
            "rho": float(rho), "p": float(p), "n": len(valid_regions),
        }

    summary = {
        "n_region_pairs": len(pair_summaries),
        "n_regions_profiled": len(region_shared_profile),
        "overall_mean_shared_dim": float(np.mean([
            ps["mean_shared_dim"] for ps in pair_summaries.values()
        ])) if pair_summaries else None,
        "overall_mean_shared_fraction": float(np.mean([
            ps["shared_fraction"] for ps in pair_summaries.values()
            if ps.get("shared_fraction") is not None
        ])) if pair_summaries else None,
    }

    results = {
        "timestamp": datetime.now().isoformat(),
        "n_sessions": len(sessions),
        "n_pairs": len(pair_summaries),
        "pair_summaries": pair_summaries,
        "region_profiles": {
            r: {k: v for k, v in p.items() if k != "shared_dims_with_partners"}
            for r, p in region_shared_profile.items()
        },
        "prediction_tests": prediction_tests,
        "summary": summary,
    }

    out_path = RESULTS_DIR / "causal_representation_learning.json"
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
