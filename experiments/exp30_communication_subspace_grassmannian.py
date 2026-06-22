"""Experiment 30: Communication subspace analysis.

Estimates communication subspaces between brain region pairs (the low-dimensional
subspace of region A's activity that predicts region B's activity) via CCA, then
tests whether communication subspace properties correlate with geometric type.

V2: Uses within-session metrics (canonical correlations, subspace overlap) instead
of cross-session Grassmannian distance, which was severely underpowered (only 6/127
pairs had enough matching sessions).

Key predictions:
1. Communication subspaces occupy a smaller fraction of the source region's
   activity subspace in high-alpha (CKA-type) regions — low-dim regions have
   more structured, selective communication.
2. Canonical correlation strength correlates with geometric type — CKA-type
   regions should have stronger canonical correlations (more linearly predictable
   communication).
3. Communication dimensionality (number of significant canonical correlations)
   scales with source region effective dimensionality.
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

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent.parent / "artifacts" / "exp30"
MIN_NEURONS = 10
TIME_WINDOW = slice(15, 35)
COMM_SUBSPACE_DIM = 5
MIN_SESSIONS_PER_PAIR = 2
N_SHUFFLES = 20


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


def _communication_analysis(X_source, X_target, n_dims=COMM_SUBSPACE_DIM, n_shuffles=N_SHUFFLES):
    """Analyze communication subspace between source and target regions.

    Returns within-session metrics that don't require cross-session matching.
    """
    n_samples = min(X_source.shape[0], X_target.shape[0])
    X_s = X_source[:n_samples]
    X_t = X_target[:n_samples]

    n_dims_use = min(n_dims, X_s.shape[1] - 1, X_t.shape[1] - 1, n_samples - 1)
    if n_dims_use < 1:
        return None

    X_s_c = X_s - X_s.mean(axis=0)
    X_t_c = X_t - X_t.mean(axis=0)

    try:
        cca = CCA(n_components=n_dims_use, max_iter=500)
        X_s_proj, X_t_proj = cca.fit_transform(X_s_c, X_t_c)
    except Exception:
        return None

    canon_corrs = []
    for i in range(n_dims_use):
        r = np.corrcoef(X_s_proj[:, i], X_t_proj[:, i])[0, 1]
        if not np.isnan(r):
            canon_corrs.append(float(r))

    if not canon_corrs:
        return None

    shuffle_corrs = []
    for _ in range(n_shuffles):
        perm = np.random.permutation(n_samples)
        try:
            cca_shuf = CCA(n_components=n_dims_use, max_iter=500)
            _, X_t_shuf = cca_shuf.fit_transform(X_s_c, X_t_c[perm])
            r_shuf = np.corrcoef(X_s_proj[:, 0], X_t_shuf[:, 0])[0, 1]
            shuffle_corrs.append(float(r_shuf) if not np.isnan(r_shuf) else 0.0)
        except Exception:
            pass

    shuffle_threshold = np.percentile(shuffle_corrs, 95) if shuffle_corrs else 0.3
    n_significant = sum(1 for r in canon_corrs if r > shuffle_threshold)

    source_pca = PCA(n_components=min(20, X_s.shape[1] - 1))
    source_pca.fit(X_s_c)
    source_var_explained = source_pca.explained_variance_ratio_

    comm_weights = cca.x_weights_
    comm_var_explained = np.var(X_s_c @ comm_weights, axis=0)
    total_var = np.var(X_s_c)
    comm_fraction = float(comm_var_explained.sum() / (total_var + 1e-10))

    return {
        "canonical_correlations": canon_corrs,
        "mean_canon_corr": float(np.mean(canon_corrs)),
        "max_canon_corr": float(max(canon_corrs)),
        "n_significant_dims": n_significant,
        "comm_fraction": comm_fraction,
        "shuffle_threshold": float(shuffle_threshold),
        "n_cca_dims": n_dims_use,
    }


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

    logger.info(f"Loaded {len(session_regions)} sessions with 2+ regions")

    pair_data = {}
    for sess_idx, sr in session_regions.items():
        for r1, r2 in combinations(sorted(sr["activities"].keys()), 2):
            key = f"{r1}_to_{r2}"
            if key not in pair_data:
                pair_data[key] = []
            pair_data[key].append({
                "sess_idx": sess_idx,
                "mouse": sr["mouse"],
                "source": sr["activities"][r1],
                "target": sr["activities"][r2],
            })

    pair_results = {}
    region_alpha_cache = {}
    region_dim_cache = {}

    for pair_key, sessions_data in tqdm(pair_data.items(), desc="Communication analysis"):
        if len(sessions_data) < MIN_SESSIONS_PER_PAIR:
            continue

        r1 = pair_key.split("_to_")[0]
        r2 = pair_key.split("_to_")[1]

        comm_results = []
        for sd in sessions_data:
            result = _communication_analysis(sd["source"], sd["target"])
            if result is not None:
                comm_results.append(result)

            if r1 not in region_alpha_cache:
                alpha = _power_law_exponent(sd["source"])
                if alpha is not None:
                    region_alpha_cache[r1] = alpha
                    region_dim_cache[r1] = _effective_dim(sd["source"])

        if not comm_results:
            continue

        pair_results[pair_key] = {
            "source": r1,
            "target": r2,
            "n_sessions": len(comm_results),
            "n_mice": len(set(sd["mouse"] for sd in sessions_data)),
            "source_alpha": region_alpha_cache.get(r1),
            "source_eff_dim": region_dim_cache.get(r1),
            "mean_canon_corr": float(np.mean([r["mean_canon_corr"] for r in comm_results])),
            "mean_max_canon_corr": float(np.mean([r["max_canon_corr"] for r in comm_results])),
            "mean_n_significant": float(np.mean([r["n_significant_dims"] for r in comm_results])),
            "mean_comm_fraction": float(np.mean([r["comm_fraction"] for r in comm_results])),
        }

    prediction_tests = {}

    valid_pairs = [p for p in pair_results.values() if p["source_alpha"] is not None]

    if len(valid_pairs) >= 10:
        alphas = [p["source_alpha"] for p in valid_pairs]
        comm_fracs = [p["mean_comm_fraction"] for p in valid_pairs]
        canon_corrs = [p["mean_canon_corr"] for p in valid_pairs]
        sig_dims = [p["mean_n_significant"] for p in valid_pairs]

        rho, p = spearmanr(alphas, comm_fracs)
        prediction_tests["alpha_vs_comm_fraction"] = {
            "rho": float(rho), "p": float(p), "n": len(valid_pairs),
            "interpretation": (
                "Negative rho means high-alpha (CKA-type) sources have smaller "
                "communication subspace fraction — more selective communication."
            ),
        }

        rho, p = spearmanr(alphas, canon_corrs)
        prediction_tests["alpha_vs_canon_corr"] = {
            "rho": float(rho), "p": float(p), "n": len(valid_pairs),
            "interpretation": (
                "Positive rho means high-alpha regions have stronger canonical "
                "correlations — more linearly predictable communication."
            ),
        }

        rho, p = spearmanr(alphas, sig_dims)
        prediction_tests["alpha_vs_comm_dimensionality"] = {
            "rho": float(rho), "p": float(p), "n": len(valid_pairs),
            "interpretation": (
                "Negative rho means high-alpha regions communicate through fewer "
                "significant dimensions."
            ),
        }

    source_summaries = {}
    for p in pair_results.values():
        src = p["source"]
        if src not in source_summaries:
            source_summaries[src] = {
                "alpha": p["source_alpha"],
                "eff_dim": p["source_eff_dim"],
                "canon_corrs": [],
                "comm_fracs": [],
                "sig_dims": [],
            }
        source_summaries[src]["canon_corrs"].append(p["mean_canon_corr"])
        source_summaries[src]["comm_fracs"].append(p["mean_comm_fraction"])
        source_summaries[src]["sig_dims"].append(p["mean_n_significant"])

    for src in source_summaries:
        s = source_summaries[src]
        s["mean_canon_corr"] = float(np.mean(s["canon_corrs"]))
        s["mean_comm_frac"] = float(np.mean(s["comm_fracs"]))
        s["mean_sig_dims"] = float(np.mean(s["sig_dims"]))
        del s["canon_corrs"], s["comm_fracs"], s["sig_dims"]

    results = {
        "timestamp": datetime.now().isoformat(),
        "n_sessions": len(sessions),
        "n_pairs_analyzed": len(pair_results),
        "n_unique_sources": len(set(p["source"] for p in pair_results.values())),
        "pair_results": pair_results,
        "prediction_tests": prediction_tests,
        "source_summaries": source_summaries,
    }

    out_path = RESULTS_DIR / "communication_subspace_grassmannian.json"
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
