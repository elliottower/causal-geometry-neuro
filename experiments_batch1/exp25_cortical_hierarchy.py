"""Experiment 25: Cortical hierarchy alignment.

Tests whether geometric type (dimensionality regime) tracks the canonical
sensory-to-association hierarchy. If high-dimensional flat-spectrum regions
cluster in association cortex and low-dimensional steep-spectrum regions
cluster in primary sensory cortex, the paper becomes a claim about cortical
organization, not just metric comparison.

Also includes task-specificity control: re-run with shuffled trial labels
to test whether the anti-correlation is about task structure or resting
noise covariance.
"""
import json
import logging
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.spatial import procrustes
from scipy.stats import f_oneway, kruskal, spearmanr
from sklearn.decomposition import PCA
from tqdm import tqdm

from data.steinmetz import get_choice_labels, get_region_activity, list_regions, load_all
from geometry.distances import cka

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent.parent / "artifacts" / "exp25"
UMAP_DIM = 5
MIN_NEURONS = 15
TIME_WINDOW = slice(15, 35)

HIERARCHY_TIERS = {
    "primary_sensory": ["VISp", "VISl", "VISrl", "VISpm", "VISam", "VISa", "SSp", "AUDp"],
    "secondary_sensory_motor": ["MOs", "MOp", "SSs", "AUDs", "VISal", "RSP"],
    "association": ["ACA", "PL", "ILA", "ORB", "FRP"],
    "hippocampal": ["CA1", "CA3", "DG", "SUB", "POST"],
    "thalamic": ["TH", "VPL", "VPM", "LP", "LD", "MD", "PO", "RT", "LGd", "MG", "SPF", "POL"],
    "subcortical": ["CP", "GPe", "SNr", "ACB", "LS", "LSr", "BLA", "ZI"],
    "midbrain_collicular": ["SCig", "SCsg", "SCs", "SCm", "MRN", "PAG", "APN", "MB"],
}

SENSORY_ASSOCIATION_GRADIENT = {
    "VISp": 1, "VISl": 1, "VISrl": 1, "VISpm": 1, "VISam": 1.5, "VISa": 1.5,
    "SSp": 1, "AUDp": 1,
    "MOs": 2, "MOp": 2, "SSs": 2, "RSP": 2,
    "ACA": 3, "PL": 3, "ILA": 3, "ORB": 3, "FRP": 3,
}


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


def _effective_dim(activity):
    pca = PCA(n_components=min(50, activity.shape[1], activity.shape[0]))
    pca.fit(activity)
    ev = pca.explained_variance_
    ev = ev[ev > 0]
    return float((ev.sum() ** 2) / (ev ** 2).sum())


def _classify_tier(region):
    for tier, regions in HIERARCHY_TIERS.items():
        if region in regions:
            return tier
    return "other"


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
            eff_dim = _effective_dim(activity)

            if region not in region_data:
                region_data[region] = []
            region_data[region].append({
                "session_idx": sess_idx,
                "mouse": mouse,
                "activity": activity,
                "labels": labels[:n],
                "n_neurons": activity.shape[1],
                "n_trials": n,
                "power_law_alpha": alpha,
                "effective_dim": eff_dim,
            })

    region_profiles = {}
    pairs_real = []
    pairs_shuffled = []
    rng = np.random.default_rng(42)

    for region, measurements in tqdm(region_data.items(), desc="Regions"):
        alphas = [m["power_law_alpha"] for m in measurements if m["power_law_alpha"] is not None]
        eff_dims = [m["effective_dim"] for m in measurements]
        tier = _classify_tier(region)
        gradient_pos = SENSORY_ASSOCIATION_GRADIENT.get(region, None)

        region_profiles[region] = {
            "n_sessions": len(measurements),
            "tier": tier,
            "gradient_position": gradient_pos,
            "alpha_mean": float(np.mean(alphas)) if alphas else None,
            "effective_dim_mean": float(np.mean(eff_dims)),
            "n_neurons_mean": float(np.mean([m["n_neurons"] for m in measurements])),
        }

        if len(measurements) < 2:
            continue

        for (i, j) in combinations(range(len(measurements)), 2):
            m1, m2 = measurements[i], measurements[j]
            n_shared = min(m1["n_trials"], m2["n_trials"])

            cka_real = cka(m1["activity"][:n_shared], m2["activity"][:n_shared])
            try:
                e1 = _umap_embed(m1["activity"], n_components=min(UMAP_DIM, m1["n_neurons"] - 1))
                e2 = _umap_embed(m2["activity"], n_components=min(UMAP_DIM, m2["n_neurons"] - 1))
                proc_real = _procrustes_distance(e1, e2)
            except Exception:
                proc_real = None

            pairs_real.append({
                "region": region,
                "tier": tier,
                "cka": float(cka_real),
                "procrustes": float(proc_real) if proc_real is not None else None,
            })

            a1_shuf = m1["activity"][rng.permutation(n_shared)]
            a2_shuf = m2["activity"][rng.permutation(n_shared)]
            cka_shuf = cka(a1_shuf[:n_shared], a2_shuf[:n_shared])
            try:
                e1s = _umap_embed(a1_shuf, n_components=min(UMAP_DIM, m1["n_neurons"] - 1))
                e2s = _umap_embed(a2_shuf, n_components=min(UMAP_DIM, m2["n_neurons"] - 1))
                proc_shuf = _procrustes_distance(e1s, e2s)
            except Exception:
                proc_shuf = None

            pairs_shuffled.append({
                "region": region,
                "cka": float(cka_shuf),
                "procrustes": float(proc_shuf) if proc_shuf is not None else None,
            })

    hierarchy_test = {}
    tier_regions = {}
    for r, p in region_profiles.items():
        t = p["tier"]
        if t not in tier_regions:
            tier_regions[t] = []
        if p["alpha_mean"] is not None:
            tier_regions[t].append({"region": r, "alpha": p["alpha_mean"], "eff_dim": p["effective_dim_mean"]})

    tier_alphas = {t: [x["alpha"] for x in rs] for t, rs in tier_regions.items() if len(rs) >= 2}
    if len(tier_alphas) >= 2:
        groups = list(tier_alphas.values())
        if all(len(g) >= 2 for g in groups):
            stat, p_val = kruskal(*groups)
            hierarchy_test["kruskal_alpha_by_tier"] = {
                "statistic": float(stat),
                "p_value": float(p_val),
                "tier_means": {t: float(np.mean(v)) for t, v in tier_alphas.items()},
                "tier_counts": {t: len(v) for t, v in tier_alphas.items()},
            }

    gradient_test = {}
    gradient_regions = [(r, p) for r, p in region_profiles.items()
                        if p["gradient_position"] is not None and p["alpha_mean"] is not None]
    if len(gradient_regions) >= 4:
        positions = [p["gradient_position"] for _, p in gradient_regions]
        alphas_g = [p["alpha_mean"] for _, p in gradient_regions]
        eff_dims_g = [p["effective_dim_mean"] for _, p in gradient_regions]
        rho_a, pv_a = spearmanr(positions, alphas_g)
        rho_d, pv_d = spearmanr(positions, eff_dims_g)
        gradient_test = {
            "n_regions": len(gradient_regions),
            "gradient_vs_alpha_rho": float(rho_a),
            "gradient_vs_alpha_p": float(pv_a),
            "gradient_vs_eff_dim_rho": float(rho_d),
            "gradient_vs_eff_dim_p": float(pv_d),
            "regions": {r: {"pos": p["gradient_position"], "alpha": p["alpha_mean"],
                            "eff_dim": p["effective_dim_mean"]}
                        for r, p in gradient_regions},
        }

    shuffle_test = {}
    valid_real = [p for p in pairs_real if p["procrustes"] is not None]
    valid_shuf = [p for p in pairs_shuffled if p["procrustes"] is not None]
    if len(valid_real) >= 4 and len(valid_shuf) >= 4:
        rho_r, pv_r = spearmanr([p["cka"] for p in valid_real], [p["procrustes"] for p in valid_real])
        rho_s, pv_s = spearmanr([p["cka"] for p in valid_shuf], [p["procrustes"] for p in valid_shuf])
        shuffle_test = {
            "real_rho": float(rho_r),
            "real_p": float(pv_r),
            "shuffled_rho": float(rho_s),
            "shuffled_p": float(pv_s),
            "interpretation": (
                "If shuffled rho is weaker, the anti-correlation depends on "
                "task structure, not just noise covariance."
            ),
        }

    results = {
        "timestamp": datetime.now().isoformat(),
        "n_regions": len(region_data),
        "n_pairs": len(pairs_real),
        "region_profiles": region_profiles,
        "hierarchy_test": hierarchy_test,
        "gradient_test": gradient_test,
        "shuffle_test": shuffle_test,
    }

    out_path = RESULTS_DIR / "cortical_hierarchy.json"
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
