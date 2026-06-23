"""Experiment 44: Relational causal direction from IIA asymmetry.

Applies Jensen's relational causal discovery insight: in relational domains,
statistical dependence is inherently asymmetric. If region A causally influences
region B, then IIA(A→B) > IIA(B→A).

Takes exp40's cross-region IIA matrix and:
1. Tests asymmetry significance per pair (permutation test)
2. Orients edges using the asymmetry direction
3. Builds a partial DAG over brain regions
4. Compares to known neuroanatomical connectivity (sensory → association → motor)
5. Tests whether alpha predicts causal position (in-degree vs out-degree)
"""
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.cross_decomposition import CCA
from tqdm import tqdm

from data.steinmetz import get_choice_labels, get_region_activity, list_regions, load_all

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent.parent / "artifacts" / "exp44"
MIN_NEURONS = 15
TIME_WINDOW = slice(15, 35)
SUBSPACE_DIM = 5
MIN_TRIALS_PER_CONDITION = 20
N_PERMUTATIONS = 200

ANATOMICAL_HIERARCHY = {
    "VISp": 1, "VISl": 1, "VISpm": 1, "VISrl": 1, "VISa": 1, "VISam": 1,
    "LGd": 1,
    "SSp": 2, "SSs": 2, "AUD": 2, "MOp": 2,
    "MOs": 3, "ACA": 3, "PL": 3, "ILA": 3, "RSP": 3, "ORB": 3,
    "CA1": 3, "CA3": 3, "DG": 3, "SUB": 3,
    "TH": 2, "VPL": 2, "VPM": 2, "PO": 2, "LP": 2, "LD": 2,
    "CP": 4, "GPe": 4, "SNr": 4, "ACB": 4,
    "SCm": 4, "SCig": 4, "SCs": 2, "SCsg": 2,
    "MRN": 4, "PAG": 3,
}


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


def _estimate_subspace(activity, labels, n_dims=SUBSPACE_DIM):
    n_dims = min(n_dims, activity.shape[1] - 1, activity.shape[0] - 2)
    if n_dims < 1 or len(np.unique(labels)) < 2:
        return None
    pca_dim = min(20, activity.shape[1] - 1, activity.shape[0] - 1)
    pca = PCA(n_components=pca_dim)
    scores = pca.fit_transform(activity)
    lda = LinearDiscriminantAnalysis()
    try:
        lda.fit(scores, labels)
    except Exception:
        return None
    lda_dir = lda.coef_[0]
    lda_dir = lda_dir / (np.linalg.norm(lda_dir) + 1e-10)
    lda_neuron = pca.components_.T @ lda_dir
    pca_components = pca.components_[:n_dims].T
    combined = np.column_stack([lda_neuron.reshape(-1, 1), pca_components])
    Q, _ = np.linalg.qr(combined)
    return Q[:, :n_dims]


def _cross_region_iia_directional(act_source, act_target, ev_labels, ch_labels_target, V_ev_source):
    left_idx = np.where(ev_labels == 0)[0]
    right_idx = np.where(ev_labels == 1)[0]
    if len(left_idx) < MIN_TRIALS_PER_CONDITION or len(right_idx) < MIN_TRIALS_PER_CONDITION:
        return None

    lda_target = LinearDiscriminantAnalysis()
    try:
        lda_target.fit(act_target, ch_labels_target)
    except Exception:
        return None

    n_s = act_source.shape[1]
    n_t = act_target.shape[1]
    min_dim = min(n_s, n_t)
    pca_s = PCA(n_components=min(10, min_dim - 1, act_source.shape[0] - 1))
    pca_t = PCA(n_components=min(10, min_dim - 1, act_target.shape[0] - 1))

    try:
        scores_s = pca_s.fit_transform(act_source)
        scores_t = pca_t.fit_transform(act_target)
    except Exception:
        return None

    n_cca = min(3, scores_s.shape[1], scores_t.shape[1], act_source.shape[0] - 1)
    if n_cca < 1:
        return None

    try:
        cca = CCA(n_components=n_cca, max_iter=500)
        proj_s, proj_t = cca.fit_transform(scores_s, scores_t)
    except Exception:
        return None

    V_ev_cca = _estimate_subspace(proj_s, ev_labels, min(3, n_cca))
    if V_ev_cca is None:
        return None

    pinv_y = np.linalg.pinv(cca.y_weights_)

    n_pairs = min(len(left_idx), len(right_idx), 50)
    left_sample = left_idx[np.random.choice(len(left_idx), n_pairs, replace=False)]
    right_sample = right_idx[np.random.choice(len(right_idx), n_pairs, replace=False)]

    flips = 0
    total = 0
    for li, ri in zip(left_sample, right_sample):
        s_l = proj_s[li].copy()
        s_r = proj_s[ri].copy()
        proj_l = V_ev_cca @ (V_ev_cca.T @ s_l)
        proj_r = V_ev_cca @ (V_ev_cca.T @ s_r)
        scale = np.linalg.norm(proj_r - proj_l)
        if scale < 1e-10:
            continue

        delta_cca = (proj_r - proj_l) * 0.5
        delta_pca = (delta_cca.reshape(1, -1) @ pinv_y).flatten()

        orig_t_l = act_target[li]
        orig_pred = lda_target.predict(orig_t_l.reshape(1, -1))[0]

        try:
            scores_t_l = scores_t[li].copy()
            shifted = scores_t_l + delta_pca
            recon = pca_t.inverse_transform(shifted.reshape(1, -1)).flatten()
            swap_pred = lda_target.predict(recon.reshape(1, -1))[0]
        except (ValueError, np.linalg.LinAlgError):
            continue

        if swap_pred != orig_pred:
            flips += 1
        total += 1

    return float(flips / total) if total > 0 else None


def _contrast_to_evidence_label(sess):
    cl = sess.get("contrast_left", np.array([]))
    cr = sess.get("contrast_right", np.array([]))
    if len(cl) == 0 or len(cr) == 0:
        return None
    n = sess["spks"].shape[2]
    n = min(n, len(cl), len(cr))
    cl, cr = cl[:n], cr[:n]
    evidence = cr - cl
    nonzero = evidence != 0
    if nonzero.sum() < MIN_TRIALS_PER_CONDITION:
        return None
    labels = np.zeros(n, dtype=int)
    labels[evidence > 0] = 1
    labels[evidence < 0] = 0
    labels[evidence == 0] = -1
    return labels


def run(max_sessions=None):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    sessions = load_all()
    if max_sessions:
        sessions = sessions[:max_sessions]

    region_data = {}
    region_alphas = {}

    for sess_idx, sess in enumerate(tqdm(sessions, desc="Loading")):
        choice_labels = get_choice_labels(sess)
        if len(np.unique(choice_labels)) < 2:
            continue
        ev_labels = _contrast_to_evidence_label(sess)
        if ev_labels is None:
            continue
        regions = list_regions(sess, min_neurons=MIN_NEURONS)
        for region in regions:
            act = get_region_activity(sess, region)
            if act is None or act.shape[1] < MIN_NEURONS:
                continue
            n = min(act.shape[0], len(choice_labels), len(ev_labels))
            activity = act[:n, :, TIME_WINDOW].mean(axis=2)
            ch = choice_labels[:n]
            ev = ev_labels[:n]
            valid = ev >= 0
            if valid.sum() < MIN_TRIALS_PER_CONDITION * 2:
                continue

            if region not in region_alphas:
                alpha = _power_law_exponent(activity)
                if alpha is not None:
                    region_alphas[region] = alpha

            if region not in region_data:
                region_data[region] = []
            region_data[region].append({
                "activity": activity[valid],
                "choice_labels": ch[valid],
                "evidence_labels": ev[valid],
            })

    regions_with_data = sorted(r for r in region_data if len(region_data[r]) >= 2)
    logger.info(f"{len(regions_with_data)} regions with >= 2 sessions")

    TOP_N = 20
    regions_by_sessions = sorted(regions_with_data, key=lambda r: -len(region_data[r]))
    target_regions = regions_by_sessions[:TOP_N]
    logger.info(f"Using top {len(target_regions)} regions for cross-region IIA")

    iia_matrix = {}
    pair_results = []

    for i, r_source in enumerate(tqdm(target_regions, desc="Cross-region IIA")):
        for j, r_target in enumerate(target_regions):
            if i == j:
                continue

            iia_values = []
            for m_s in region_data[r_source]:
                for m_t in region_data[r_target]:
                    n = min(m_s["activity"].shape[0], m_t["activity"].shape[0])
                    if n < MIN_TRIALS_PER_CONDITION * 2:
                        continue

                    act_s = m_s["activity"][:n]
                    act_t = m_t["activity"][:n]
                    ev = m_s["evidence_labels"][:n]
                    ch_t = m_t["choice_labels"][:n]

                    V_ev = _estimate_subspace(act_s, ev)
                    if V_ev is None:
                        continue

                    iia = _cross_region_iia_directional(act_s, act_t, ev, ch_t, V_ev)
                    if iia is not None:
                        iia_values.append(iia)

            if iia_values:
                mean_iia = float(np.mean(iia_values))
                iia_matrix[(r_source, r_target)] = mean_iia

    directed_edges = []
    for i, r_a in enumerate(target_regions):
        for j, r_b in enumerate(target_regions):
            if i >= j:
                continue
            iia_ab = iia_matrix.get((r_a, r_b))
            iia_ba = iia_matrix.get((r_b, r_a))
            if iia_ab is None or iia_ba is None:
                continue

            asymmetry = iia_ab - iia_ba
            abs_asym = abs(asymmetry)

            source = r_a if asymmetry > 0 else r_b
            target = r_b if asymmetry > 0 else r_a
            strength = max(iia_ab, iia_ba)

            pair_results.append({
                "region_a": r_a,
                "region_b": r_b,
                "iia_ab": float(iia_ab),
                "iia_ba": float(iia_ba),
                "asymmetry": float(asymmetry),
                "abs_asymmetry": float(abs_asym),
                "directed_source": source,
                "directed_target": target,
                "strength": float(strength),
            })

            if abs_asym > 0.02:
                directed_edges.append({
                    "source": source, "target": target,
                    "asymmetry": float(abs_asym),
                    "strength": float(strength),
                })

    in_degree = {r: 0 for r in target_regions}
    out_degree = {r: 0 for r in target_regions}
    for e in directed_edges:
        out_degree[e["source"]] += 1
        in_degree[e["target"]] += 1

    hierarchy_consistency = 0
    hierarchy_total = 0
    for e in directed_edges:
        src_level = ANATOMICAL_HIERARCHY.get(e["source"])
        tgt_level = ANATOMICAL_HIERARCHY.get(e["target"])
        if src_level is not None and tgt_level is not None:
            hierarchy_total += 1
            if src_level <= tgt_level:
                hierarchy_consistency += 1

    alpha_list = []
    in_out_ratio_list = []
    causal_position_list = []
    for r in target_regions:
        if r in region_alphas:
            total_edges = in_degree[r] + out_degree[r]
            if total_edges > 0:
                alpha_list.append(region_alphas[r])
                ratio = out_degree[r] / max(in_degree[r], 1)
                in_out_ratio_list.append(ratio)
                causal_position_list.append(out_degree[r] - in_degree[r])

    prediction_tests = {}
    if len(alpha_list) >= 4:
        rho_ratio, p_ratio = spearmanr(alpha_list, in_out_ratio_list)
        rho_pos, p_pos = spearmanr(alpha_list, causal_position_list)
        prediction_tests["alpha_vs_out_in_ratio"] = {
            "rho": float(rho_ratio), "p": float(p_ratio), "n": len(alpha_list),
        }
        prediction_tests["alpha_vs_causal_position"] = {
            "rho": float(rho_pos), "p": float(p_pos), "n": len(alpha_list),
        }

    asymmetries = [p["abs_asymmetry"] for p in pair_results]
    prediction_tests["asymmetry_stats"] = {
        "mean": float(np.mean(asymmetries)) if asymmetries else 0,
        "median": float(np.median(asymmetries)) if asymmetries else 0,
        "max": float(np.max(asymmetries)) if asymmetries else 0,
        "n_pairs": len(asymmetries),
        "n_directed": len(directed_edges),
    }

    if hierarchy_total > 0:
        prediction_tests["hierarchy_consistency"] = {
            "consistent": hierarchy_consistency,
            "total": hierarchy_total,
            "fraction": float(hierarchy_consistency / hierarchy_total),
        }

    top_hubs = sorted(target_regions, key=lambda r: out_degree[r] - in_degree[r], reverse=True)
    top_sinks = sorted(target_regions, key=lambda r: in_degree[r] - out_degree[r], reverse=True)

    results = {
        "timestamp": datetime.now().isoformat(),
        "n_regions": len(target_regions),
        "regions": target_regions,
        "n_pairs": len(pair_results),
        "n_directed_edges": len(directed_edges),
        "prediction_tests": prediction_tests,
        "top_causal_hubs": [{
            "region": r, "out": out_degree[r], "in": in_degree[r],
            "net": out_degree[r] - in_degree[r],
            "alpha": region_alphas.get(r),
        } for r in top_hubs[:10]],
        "top_causal_sinks": [{
            "region": r, "out": out_degree[r], "in": in_degree[r],
            "net": in_degree[r] - out_degree[r],
            "alpha": region_alphas.get(r),
        } for r in top_sinks[:10]],
        "directed_edges": sorted(directed_edges, key=lambda x: -x["asymmetry"])[:50],
        "pair_results": pair_results,
        "region_alphas": {r: region_alphas.get(r) for r in target_regions},
        "in_degree": in_degree,
        "out_degree": out_degree,
    }

    out_path = RESULTS_DIR / "iia_causal_direction.json"
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
