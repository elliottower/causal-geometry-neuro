"""Experiment 32: Multi-method causal discovery on inter-regional neural data.

Applies causal discovery algorithms (PC, LiNGAM, CD-NOD) from the
factorization-circuits causal discovery toolkit to neural population data.

Each variable is a brain region's top PCA component (or top-k components).
Each observation is a trial. Different mice provide the heterogeneous
environments that CD-NOD exploits for edge orientation.

Key questions:
1. What is the directed causal graph among brain regions during decision-making?
2. Does CD-NOD (using cross-mouse distribution shifts) orient more edges than PC?
3. Does the discovered causal structure align with known neuroanatomy
   (sensory → association → motor)?
4. Do high-alpha (flat spectrum) regions have more incoming vs outgoing edges?
"""
import json
import logging
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from tqdm import tqdm

from data.steinmetz import get_choice_labels, get_region_activity, list_regions, load_all

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent.parent / "artifacts" / "exp32"
MIN_NEURONS = 15
TIME_WINDOW = slice(15, 35)
N_PCA_COMPONENTS = 3
MIN_SESSIONS_WITH_REGION = 3


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


def _run_pc(data, var_names, alpha=0.05):
    """PC algorithm via causal-learn."""
    from causallearn.search.ConstraintBased.PC import pc
    from causallearn.utils.cit import fisherz

    t0 = time.time()
    cg = pc(data, alpha=alpha, indep_test=fisherz,
            stable=True, uc_rule=0, uc_priority=2, verbose=False, show_progress=False)
    elapsed = time.time() - t0

    adj = cg.G.graph
    edges = []
    n = adj.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j] == -1 and adj[j, i] == 1:
                edges.append((var_names[j], var_names[i], "directed"))
            elif adj[i, j] == 1 and adj[j, i] == -1:
                edges.append((var_names[i], var_names[j], "directed"))
            elif adj[i, j] == -1 and adj[j, i] == -1:
                edges.append((var_names[i], var_names[j], "undirected"))

    return {
        "edges": edges,
        "n_edges": len(edges),
        "n_directed": sum(1 for e in edges if e[2] == "directed"),
        "elapsed_s": elapsed,
    }


def _run_lingam(data, var_names):
    """DirectLiNGAM — gives full DAG orientation."""
    from causallearn.search.FCMBased.lingam import DirectLiNGAM

    t0 = time.time()
    model = DirectLiNGAM()
    model.fit(data)
    elapsed = time.time() - t0

    adj = model.adjacency_matrix_
    edges = []
    n = adj.shape[0]
    for i in range(n):
        for j in range(n):
            if i != j and abs(adj[i, j]) > 0.01:
                edges.append((var_names[j], var_names[i], "directed", float(adj[i, j])))

    return {
        "edges": [(e[0], e[1], e[2]) for e in edges],
        "weighted_edges": edges,
        "n_edges": len(edges),
        "causal_order": [var_names[i] for i in model.causal_order_],
        "elapsed_s": elapsed,
    }


def _run_cdnod(data, c_indx, var_names, alpha=0.05):
    """CD-NOD using cross-mouse domain shifts for orientation."""
    from causallearn.search.ConstraintBased.CDNOD import cdnod
    from causallearn.utils.cit import fisherz

    t0 = time.time()
    cg = cdnod(data, c_indx, alpha=alpha, indep_test=fisherz, stable=True,
               uc_rule=0, uc_priority=2, verbose=False, show_progress=False)
    elapsed = time.time() - t0

    adj = cg.G.graph
    n_vars = data.shape[1]
    edges = []
    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            if adj[i, j] == -1 and adj[j, i] == 1:
                edges.append((var_names[j], var_names[i], "directed"))
            elif adj[i, j] == 1 and adj[j, i] == -1:
                edges.append((var_names[i], var_names[j], "directed"))
            elif adj[i, j] == -1 and adj[j, i] == -1:
                edges.append((var_names[i], var_names[j], "undirected"))

    return {
        "edges": edges,
        "n_edges": len(edges),
        "n_directed": sum(1 for e in edges if e[2] == "directed"),
        "elapsed_s": elapsed,
    }


def run(max_sessions: int | None = None):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    sessions = load_all()
    if max_sessions:
        sessions = sessions[:max_sessions]

    region_pca_by_session = {}
    mouse_labels = {}
    region_alphas = {}

    for sess_idx, sess in enumerate(tqdm(sessions, desc="Extracting PCA")):
        labels = get_choice_labels(sess)
        if len(np.unique(labels)) < 2:
            continue

        mouse = str(sess.get("mouse_name", f"mouse_{sess_idx}"))
        mouse_labels[sess_idx] = mouse
        regions = list_regions(sess, min_neurons=MIN_NEURONS)

        for region in regions:
            act = get_region_activity(sess, region)
            if act is None or act.shape[1] < MIN_NEURONS:
                continue
            n = min(act.shape[0], len(labels))
            activity = act[:n, :, TIME_WINDOW].mean(axis=2)

            n_comp = min(N_PCA_COMPONENTS, activity.shape[1] - 1, activity.shape[0] - 1)
            if n_comp < 1:
                continue
            pca = PCA(n_components=n_comp)
            scores = pca.fit_transform(activity)

            if region not in region_alphas:
                alpha = _power_law_exponent(activity)
                if alpha is not None:
                    region_alphas[region] = alpha

            if sess_idx not in region_pca_by_session:
                region_pca_by_session[sess_idx] = {}
            region_pca_by_session[sess_idx][region] = scores

    all_regions = sorted(set(
        r for regions in region_pca_by_session.values() for r in regions
    ))
    region_session_counts = {r: 0 for r in all_regions}
    for sess_regions in region_pca_by_session.values():
        for r in sess_regions:
            region_session_counts[r] += 1

    ranked_regions = sorted(region_session_counts.items(), key=lambda x: -x[1])
    logger.info(f"Region session counts (top 15): {ranked_regions[:15]}")

    MAX_CAUSAL_REGIONS = 10
    min_sessions_needed = 5

    candidate_regions = [r for r, c in ranked_regions if c >= min_sessions_needed]
    if len(candidate_regions) > MAX_CAUSAL_REGIONS:
        candidate_regions = candidate_regions[:MAX_CAUSAL_REGIONS]

    sessions_with_all = []
    for sess_idx, sess_regions in region_pca_by_session.items():
        if all(r in sess_regions for r in candidate_regions):
            sessions_with_all.append(sess_idx)

    while len(sessions_with_all) < 5 and len(candidate_regions) > 3:
        candidate_regions = candidate_regions[:-1]
        sessions_with_all = [
            s for s, sr in region_pca_by_session.items()
            if all(r in sr for r in candidate_regions)
        ]

    common_regions = sorted(candidate_regions)
    logger.info(f"Selected {len(common_regions)} regions with {len(sessions_with_all)} sessions: {common_regions}")

    if len(common_regions) < 3 or len(sessions_with_all) < 3:
        logger.warning("Not enough common regions/sessions for causal discovery")
        results = {
            "timestamp": datetime.now().isoformat(),
            "error": f"Only {len(common_regions)} regions in {len(sessions_with_all)} sessions",
            "n_sessions": len(sessions),
            "region_session_counts": dict(ranked_regions[:20]),
        }
        out_path = RESULTS_DIR / "causal_discovery_interregion.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        return results

    var_names = []
    for r in common_regions:
        for c in range(N_PCA_COMPONENTS):
            var_names.append(f"{r}_pc{c}")

    per_session_results = []
    pooled_data_parts = []
    pooled_mouse_ids = []
    mouse_name_to_id = {}

    logger.info(f"sessions_with_all: {len(sessions_with_all)} sessions, mouse_labels: {len(mouse_labels)} entries")
    for sess_idx in tqdm(sessions_with_all, desc="Building data"):
        sess_regions = region_pca_by_session[sess_idx]
        n_trials = min(sess_regions[r].shape[0] for r in common_regions)
        if n_trials < 20:
            logger.warning(f"Session {sess_idx} skipped: only {n_trials} trials")
            continue

        data_matrix = np.hstack([sess_regions[r][:n_trials] for r in common_regions])

        mouse = mouse_labels.get(sess_idx, f"unknown_{sess_idx}")
        if mouse not in mouse_name_to_id:
            mouse_name_to_id[mouse] = len(mouse_name_to_id)

        pooled_data_parts.append(data_matrix)
        pooled_mouse_ids.extend([mouse_name_to_id[mouse]] * n_trials)

        try:
            pc_result = _run_pc(data_matrix, var_names)
            per_session_results.append({
                "sess_idx": sess_idx,
                "mouse": mouse,
                "n_trials": n_trials,
                "pc": pc_result,
            })
        except Exception as e:
            logger.warning(f"PC failed for session {sess_idx}: {e}")

    pooled_results = {}

    if pooled_data_parts:
        pooled_data = np.vstack(pooled_data_parts)
        c_indx = np.array(pooled_mouse_ids).reshape(-1, 1)
        logger.info(f"Pooled data: {pooled_data.shape}, {len(mouse_name_to_id)} mice")

        try:
            pooled_results["pc"] = _run_pc(pooled_data, var_names)
            logger.info(f"PC pooled: {pooled_results['pc']['n_edges']} edges, "
                        f"{pooled_results['pc']['n_directed']} directed")
        except Exception as e:
            logger.error(f"PC pooled failed: {e}")

        try:
            pooled_results["lingam"] = _run_lingam(pooled_data, var_names)
            logger.info(f"LiNGAM pooled: {pooled_results['lingam']['n_edges']} edges")
        except Exception as e:
            logger.error(f"LiNGAM pooled failed: {e}")

        try:
            pooled_results["cdnod"] = _run_cdnod(pooled_data, c_indx, var_names)
            logger.info(f"CD-NOD pooled: {pooled_results['cdnod']['n_edges']} edges, "
                        f"{pooled_results['cdnod']['n_directed']} directed")
        except Exception as e:
            logger.error(f"CD-NOD pooled failed: {e}")

    region_edge_profile = {r: {"in": 0, "out": 0, "undirected": 0} for r in common_regions}
    for method_name, method_result in pooled_results.items():
        for edge in method_result.get("edges", []):
            src_region = edge[0].rsplit("_pc", 1)[0]
            tgt_region = edge[1].rsplit("_pc", 1)[0]
            if src_region == tgt_region:
                continue
            if edge[2] == "directed":
                region_edge_profile[src_region]["out"] += 1
                region_edge_profile[tgt_region]["in"] += 1
            else:
                region_edge_profile[src_region]["undirected"] += 1
                region_edge_profile[tgt_region]["undirected"] += 1

    alpha_vs_edges = {}
    valid_regions = [(r, region_alphas.get(r), region_edge_profile.get(r))
                     for r in common_regions
                     if r in region_alphas and r in region_edge_profile]

    if len(valid_regions) >= 4:
        alphas = [v[1] for v in valid_regions]
        in_counts = [v[2]["in"] for v in valid_regions]
        out_counts = [v[2]["out"] for v in valid_regions]
        in_out_ratio = [v[2]["in"] / max(v[2]["out"], 1) for v in valid_regions]

        rho_in, p_in = spearmanr(alphas, in_counts)
        rho_out, p_out = spearmanr(alphas, out_counts)
        rho_ratio, p_ratio = spearmanr(alphas, in_out_ratio)

        alpha_vs_edges = {
            "alpha_vs_in_degree": {"rho": float(rho_in), "p": float(p_in)},
            "alpha_vs_out_degree": {"rho": float(rho_out), "p": float(p_out)},
            "alpha_vs_in_out_ratio": {
                "rho": float(rho_ratio), "p": float(p_ratio),
                "interpretation": (
                    "Positive rho means high-alpha regions receive more causal "
                    "input than they send — consistent with association cortex "
                    "integrating from multiple sources."
                ),
            },
        }

    consensus = {}
    if len(pooled_results) >= 2:
        edge_sets = {}
        for method_name, result in pooled_results.items():
            directed = set()
            for e in result.get("edges", []):
                if e[2] == "directed":
                    src_r = e[0].rsplit("_pc", 1)[0]
                    tgt_r = e[1].rsplit("_pc", 1)[0]
                    if src_r != tgt_r:
                        directed.add((src_r, tgt_r))
            edge_sets[method_name] = directed

        all_directed = set()
        for s in edge_sets.values():
            all_directed |= s

        consensus_edges = []
        for edge in all_directed:
            methods_agreeing = [m for m, s in edge_sets.items() if edge in s]
            if len(methods_agreeing) >= 2:
                consensus_edges.append({
                    "source": edge[0], "target": edge[1],
                    "methods": methods_agreeing, "n_methods": len(methods_agreeing),
                })
        consensus = {
            "n_consensus_edges": len(consensus_edges),
            "edges": sorted(consensus_edges, key=lambda x: -x["n_methods"]),
        }

    method_comparison = {}
    if "pc" in pooled_results and "cdnod" in pooled_results:
        method_comparison["pc_vs_cdnod"] = {
            "pc_total": pooled_results["pc"]["n_edges"],
            "pc_directed": pooled_results["pc"]["n_directed"],
            "cdnod_total": pooled_results["cdnod"]["n_edges"],
            "cdnod_directed": pooled_results["cdnod"]["n_directed"],
            "cdnod_orients_more": pooled_results["cdnod"]["n_directed"] > pooled_results["pc"]["n_directed"],
            "interpretation": (
                "CD-NOD exploits cross-mouse distribution shifts to orient edges "
                "that PC leaves undirected. More directed edges = more power from heterogeneity."
            ),
        }

    results = {
        "timestamp": datetime.now().isoformat(),
        "n_sessions": len(sessions),
        "n_mice": len(mouse_name_to_id),
        "common_regions": common_regions,
        "n_common_regions": len(common_regions),
        "n_variables": len(var_names),
        "pooled_results": {k: {kk: vv for kk, vv in v.items() if kk != "weighted_edges"}
                           for k, v in pooled_results.items()},
        "region_edge_profile": region_edge_profile,
        "region_alphas": {r: region_alphas.get(r) for r in common_regions},
        "alpha_vs_edges": alpha_vs_edges,
        "consensus": consensus,
        "method_comparison": method_comparison,
        "n_per_session_runs": len(per_session_results),
    }

    out_path = RESULTS_DIR / "causal_discovery_interregion.json"
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
