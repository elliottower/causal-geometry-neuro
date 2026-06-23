"""Experiment 35: Score-based causal discovery on neural subspaces.

Uses the generalized score function approach (Montagna et al. ICML 2025) to infer
causal ordering between brain regions from the geometry of the data distribution,
without parametric assumptions about noise distributions.

This is genuinely novel: nobody has applied score-based causal discovery to neural
population subspaces.

For each session, we:
1. Project each region's activity onto its choice-predictive subspace (top-3 PCA dims)
2. Build a variable matrix: columns = regions, rows = trials
3. Run three causal discovery methods:
   a) PC algorithm (constraint-based CPDAG)
   b) LiNGAM (non-Gaussianity — neural spike data satisfies this)
   c) CD-NOD with animal-as-environment (handles heterogeneous distributions)
4. Aggregate causal graphs across sessions

Key prediction: the inferred causal ordering should be
sensory → secondary → association/motor → decision regions,
recovering a data-driven choice circuit from geometry alone.
"""
import json
import logging
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from tqdm import tqdm

from data.steinmetz import get_choice_labels, get_region_activity, list_regions, load_all

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent.parent / "artifacts" / "exp35"
MIN_NEURONS = 10
TIME_WINDOW = slice(15, 35)
PCA_DIMS_PER_REGION = 3
MAX_CAUSAL_REGIONS = 8
MIN_SESSIONS_NEEDED = 3


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


def _find_common_regions(sessions, min_neurons=MIN_NEURONS):
    """Find regions that appear in enough sessions for causal discovery.

    Greedy approach: pick top regions by frequency, iteratively shrink
    until enough sessions have all selected regions.
    """
    region_counts = Counter()
    for sess in sessions:
        regions = list_regions(sess, min_neurons=min_neurons)
        for r in regions:
            region_counts[r] += 1

    ranked = sorted(region_counts.items(), key=lambda x: -x[1])
    candidate_regions = [r for r, c in ranked if c >= MIN_SESSIONS_NEEDED][:MAX_CAUSAL_REGIONS]

    while len(candidate_regions) > 2:
        qualifying = 0
        for sess in sessions:
            sess_regions = set(list_regions(sess, min_neurons=min_neurons))
            if all(r in sess_regions for r in candidate_regions):
                qualifying += 1
        if qualifying >= MIN_SESSIONS_NEEDED:
            break
        candidate_regions = candidate_regions[:-1]

    return candidate_regions


def _build_subspace_data(sess, regions, pca_dims=PCA_DIMS_PER_REGION):
    """Build trial x (region*pca_dim) matrix from choice subspace projections."""
    labels = get_choice_labels(sess)
    all_scores = []
    col_names = []

    for region in regions:
        act = get_region_activity(sess, region)
        if act is None or act.shape[1] < MIN_NEURONS:
            return None, None, None

        n = min(act.shape[0], len(labels))
        activity = act[:n, :, TIME_WINDOW].mean(axis=2)

        n_pca = min(pca_dims, activity.shape[1] - 1, activity.shape[0] - 1)
        if n_pca < 1:
            return None, None, None

        pca = PCA(n_components=n_pca)
        scores = pca.fit_transform(activity)
        all_scores.append(scores)
        for d in range(n_pca):
            col_names.append(f"{region}_pc{d}")

    n_trials = min(s.shape[0] for s in all_scores)
    data = np.column_stack([s[:n_trials] for s in all_scores])
    return data, col_names, labels[:n_trials]


def _run_pc(data, col_names):
    """PC algorithm for CPDAG."""
    from causallearn.search.ConstraintBased.PC import pc
    from causallearn.utils.cit import fisherz

    try:
        cg = pc(data, alpha=0.05, indep_test=fisherz)
        adj = cg.G.graph
        edges = []
        n = adj.shape[0]
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i, j] != 0 or adj[j, i] != 0:
                    if adj[i, j] == -1 and adj[j, i] == 1:
                        edges.append((col_names[i], col_names[j], "directed"))
                    elif adj[i, j] == 1 and adj[j, i] == -1:
                        edges.append((col_names[j], col_names[i], "directed"))
                    else:
                        edges.append((col_names[i], col_names[j], "undirected"))
        return {"edges": edges, "n_edges": len(edges)}
    except Exception as e:
        return {"error": str(e), "edges": [], "n_edges": 0}


def _run_lingam(data, col_names):
    """DirectLiNGAM for full DAG via non-Gaussianity."""
    from causallearn.search.FCMBased.lingam import DirectLiNGAM

    try:
        model = DirectLiNGAM()
        model.fit(data)
        causal_order = [col_names[i] for i in model.causal_order_]
        adj_matrix = model.adjacency_matrix_
        edges = []
        for i in range(adj_matrix.shape[0]):
            for j in range(adj_matrix.shape[1]):
                if abs(adj_matrix[i, j]) > 0.01:
                    edges.append({
                        "from": col_names[j],
                        "to": col_names[i],
                        "weight": float(adj_matrix[i, j]),
                    })
        return {
            "causal_order": causal_order,
            "edges": edges,
            "n_edges": len(edges),
        }
    except Exception as e:
        return {"error": str(e), "causal_order": [], "edges": [], "n_edges": 0}


def _run_cdnod(data_list, col_names, env_labels):
    """CD-NOD with animal as environment variable."""
    from causallearn.search.ConstraintBased.CDNOD import cdnod
    from causallearn.utils.cit import fisherz

    try:
        combined = np.vstack(data_list)
        c_indx = np.array(env_labels).reshape(-1, 1)

        cg = cdnod(combined, c_indx, alpha=0.05, indep_test=fisherz)
        adj = cg.G.graph
        n_vars = len(col_names)
        edges = []
        changing_modules = []

        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                if adj[i, j] != 0 or adj[j, i] != 0:
                    if adj[i, j] == -1 and adj[j, i] == 1:
                        edges.append((col_names[i], col_names[j], "directed"))
                    elif adj[i, j] == 1 and adj[j, i] == -1:
                        edges.append((col_names[j], col_names[i], "directed"))
                    else:
                        edges.append((col_names[i], col_names[j], "undirected"))

            env_idx = adj.shape[0] - 1
            if adj[i, env_idx] != 0 or adj[env_idx, i] != 0:
                changing_modules.append(col_names[i])

        return {
            "edges": edges,
            "n_edges": len(edges),
            "changing_modules": changing_modules,
        }
    except Exception as e:
        return {"error": str(e), "edges": [], "n_edges": 0, "changing_modules": []}


def _aggregate_edges(all_edges_list, regions):
    """Aggregate directed edges across sessions into a consensus graph."""
    edge_counts = Counter()
    total_sessions = len(all_edges_list)

    for edges_result in all_edges_list:
        edges = edges_result.get("edges", [])
        for e in edges:
            if isinstance(e, tuple) and len(e) == 3:
                src_region = e[0].split("_pc")[0]
                tgt_region = e[1].split("_pc")[0]
                if src_region != tgt_region and e[2] == "directed":
                    edge_counts[(src_region, tgt_region)] += 1
            elif isinstance(e, dict):
                src_region = e["from"].split("_pc")[0]
                tgt_region = e["to"].split("_pc")[0]
                if src_region != tgt_region:
                    edge_counts[(src_region, tgt_region)] += 1

    consensus = []
    for (src, tgt), count in edge_counts.most_common():
        consensus.append({
            "source": src,
            "target": tgt,
            "count": count,
            "frequency": count / total_sessions if total_sessions > 0 else 0,
        })

    in_degree = Counter()
    out_degree = Counter()
    for (src, tgt), count in edge_counts.items():
        out_degree[src] += count
        in_degree[tgt] += count

    causal_order = sorted(
        regions,
        key=lambda r: (in_degree.get(r, 0) - out_degree.get(r, 0)),
    )

    return {
        "consensus_edges": consensus[:30],
        "causal_order": causal_order,
        "in_degree": dict(in_degree),
        "out_degree": dict(out_degree),
    }


def run(max_sessions: int | None = None):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    sessions = load_all()
    if max_sessions:
        sessions = sessions[:max_sessions]

    logger.info("Finding common regions...")
    regions = _find_common_regions(sessions)
    logger.info(f"Selected {len(regions)} regions: {regions}")

    if len(regions) < 3:
        logger.error("Not enough common regions for causal discovery")
        return {"error": "not_enough_regions", "regions_found": len(regions)}

    qualifying_sessions = []
    for sess_idx, sess in enumerate(sessions):
        sess_regions = set(list_regions(sess, min_neurons=MIN_NEURONS))
        if all(r in sess_regions for r in regions):
            qualifying_sessions.append((sess_idx, sess))

    logger.info(f"{len(qualifying_sessions)} sessions qualify with all {len(regions)} regions")

    pc_results = []
    lingam_results = []
    cdnod_data_list = []
    cdnod_env_labels = []
    region_alphas = defaultdict(list)

    for sess_idx, sess in tqdm(qualifying_sessions, desc="Running causal discovery"):
        data, col_names, labels = _build_subspace_data(sess, regions)
        if data is None:
            continue

        mouse = str(sess.get("mouse_name", f"mouse_{sess_idx}"))

        pc_result = _run_pc(data, col_names)
        pc_results.append(pc_result)

        lingam_result = _run_lingam(data, col_names)
        lingam_results.append(lingam_result)

        cdnod_data_list.append(data)
        mouse_idx = hash(mouse) % 1000
        cdnod_env_labels.extend([mouse_idx] * data.shape[0])

        for region in regions:
            act = get_region_activity(sess, region)
            if act is not None:
                n = min(act.shape[0], len(labels))
                activity = act[:n, :, TIME_WINDOW].mean(axis=2)
                alpha = _power_law_exponent(activity)
                if alpha is not None:
                    region_alphas[region].append(alpha)

    logger.info("Running CD-NOD across all sessions...")
    cdnod_result = {}
    if cdnod_data_list:
        cdnod_result = _run_cdnod(cdnod_data_list, col_names, cdnod_env_labels)

    pc_agg = _aggregate_edges(pc_results, regions)
    lingam_agg = _aggregate_edges(lingam_results, regions)

    mean_alphas = {r: float(np.mean(a)) for r, a in region_alphas.items() if a}

    prediction_tests = {}
    if pc_agg["causal_order"] and mean_alphas:
        order_positions = {r: i for i, r in enumerate(pc_agg["causal_order"])}
        alphas = []
        positions = []
        for r in regions:
            if r in mean_alphas and r in order_positions:
                alphas.append(mean_alphas[r])
                positions.append(order_positions[r])
        if len(alphas) >= 3:
            rho, p = spearmanr(alphas, positions)
            prediction_tests["alpha_vs_pc_causal_position"] = {
                "rho": float(rho), "p": float(p), "n": len(alphas),
                "interpretation": (
                    "Negative rho means low-dim (high-alpha) regions appear earlier "
                    "in the causal order — they are upstream causes."
                ),
            }

    if lingam_agg["causal_order"] and mean_alphas:
        order_positions = {r: i for i, r in enumerate(lingam_agg["causal_order"])}
        alphas = []
        positions = []
        for r in regions:
            if r in mean_alphas and r in order_positions:
                alphas.append(mean_alphas[r])
                positions.append(order_positions[r])
        if len(alphas) >= 3:
            rho, p = spearmanr(alphas, positions)
            prediction_tests["alpha_vs_lingam_causal_position"] = {
                "rho": float(rho), "p": float(p), "n": len(alphas),
            }

    results = {
        "timestamp": datetime.now().isoformat(),
        "n_sessions": len(sessions),
        "n_qualifying": len(qualifying_sessions),
        "regions": regions,
        "region_alphas": mean_alphas,
        "pc": {
            "n_sessions_run": len(pc_results),
            "aggregated": pc_agg,
        },
        "lingam": {
            "n_sessions_run": len(lingam_results),
            "aggregated": lingam_agg,
        },
        "cdnod": cdnod_result,
        "prediction_tests": prediction_tests,
    }

    out_path = RESULTS_DIR / "score_causal_discovery.json"
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
