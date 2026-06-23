"""Experiment 41: Cross-paper bridge — attribution-causal orthogonality in biology.

Bridges the transformer factorization paper (outline_B_v6) with the neuroscience paper
by computing two quantities that have direct analogs in both domains:

1. PCA-LDA angle per region: the angle between the max-variance subspace (PCA)
   and the choice-discriminative subspace (LDA). In transformers, this is ~88 deg
   (attribution subspace is nearly orthogonal to causal subspace). In neural data,
   high-dimensional regions should show larger PCA-LDA angles because their variance
   is dominated by non-choice dimensions.

2. Conceptor AND/NOT on V_ev and V_ch: AND(V_ev, V_ch) = dimensions shared between
   evidence and choice representations. NOT = evidence encoded but causally disconnected
   from choice. Regions where NOT is large encode evidence but don't use it for decisions.

3. Cross-condition principal angles: principal angles between easy-trial and hard-trial
   choice subspaces per region. High-dim regions should show larger angles (like the
   58-78 deg per-counterfactual result in transformers).
"""
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from tqdm import tqdm

from data.steinmetz import get_choice_labels, get_region_activity, list_regions, load_all
from geometry.distances import all_subspace_distances

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent.parent / "artifacts" / "exp41"
MIN_NEURONS = 10
TIME_WINDOW = slice(15, 35)
SUBSPACE_DIM = 5
MIN_TRIALS_PER_CONDITION = 20
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


def _pca_lda_distances(activity, labels, k=SUBSPACE_DIM):
    """All subspace distances between top-k PCA and top-k LDA subspaces."""
    k = min(k, activity.shape[1] - 1, activity.shape[0] - 2)
    if k < 1 or len(np.unique(labels)) < 2:
        return None

    pca = PCA(n_components=k)
    pca.fit(activity)
    V_pca = pca.components_.T  # (n_neurons, k)

    pca_full = PCA(n_components=min(20, activity.shape[1] - 1, activity.shape[0] - 1))
    scores = pca_full.fit_transform(activity)
    lda = LinearDiscriminantAnalysis()
    try:
        lda.fit(scores, labels)
    except Exception:
        return None
    lda_dir = lda.coef_[0]
    lda_dir = lda_dir / (np.linalg.norm(lda_dir) + 1e-10)
    lda_neuron = pca_full.components_.T @ lda_dir
    V_lda_col = lda_neuron.reshape(-1, 1)
    pca_top = pca.components_[:k].T
    V_lda = np.column_stack([V_lda_col, pca_top])
    V_lda, _ = np.linalg.qr(V_lda)
    V_lda = V_lda[:, :k]

    return all_subspace_distances(V_pca, V_lda)


def _subspace_and_not(V_a, V_b, threshold=0.5):
    """Compute AND (shared) and NOT (in A but not B) subspace dimensions.

    AND: dimensions of V_a that have large projection onto V_b (cosine > threshold).
    NOT: dimensions of V_a with small projection onto V_b.
    """
    if V_a is None or V_b is None:
        return None

    proj = V_b @ V_b.T @ V_a
    overlap = np.linalg.norm(proj, axis=0)
    overlap = np.clip(overlap, 0, 1)

    n_and = int(np.sum(overlap > threshold))
    n_not = int(np.sum(overlap <= threshold))
    mean_overlap = float(np.mean(overlap))

    return {
        "n_shared_dims": n_and,
        "n_private_dims": n_not,
        "mean_overlap": mean_overlap,
        "per_dim_overlap": [float(o) for o in overlap],
    }


def _estimate_subspace(activity, labels, n_dims=SUBSPACE_DIM):
    """LDA + PCA discriminative subspace."""
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


def _cross_condition_distances(activity, labels, condition_labels, k=SUBSPACE_DIM):
    """All subspace distances between choice subspaces on different conditions."""
    conds = np.unique(condition_labels)
    if len(conds) < 2:
        return None

    c0_mask = condition_labels == conds[0]
    c1_mask = condition_labels == conds[1]

    if c0_mask.sum() < MIN_TRIALS_PER_CONDITION or c1_mask.sum() < MIN_TRIALS_PER_CONDITION:
        return None

    V0 = _estimate_subspace(activity[c0_mask], labels[c0_mask], k)
    V1 = _estimate_subspace(activity[c1_mask], labels[c1_mask], k)

    if V0 is None or V1 is None:
        return None

    kk = min(V0.shape[1], V1.shape[1])
    return all_subspace_distances(V0[:, :kk], V1[:, :kk])


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
    labels = np.full(n, -1, dtype=int)
    labels[evidence > 0] = 1
    labels[evidence < 0] = 0
    return labels


def _easy_hard_labels(sess):
    """Split trials into easy (max contrast >= 0.5) and hard (<= 0.25)."""
    cl = sess.get("contrast_left", np.array([]))
    cr = sess.get("contrast_right", np.array([]))
    n = sess["spks"].shape[2]
    n = min(n, len(cl), len(cr))
    max_contrast = np.maximum(cl[:n], cr[:n])
    labels = np.full(n, -1, dtype=int)
    labels[max_contrast >= 0.5] = 0  # easy
    labels[max_contrast <= 0.25] = 1  # hard
    return labels


def _rebuild_region_results_from_jsonl(jsonl_path):
    """Load incremental results and rebuild region_results dict."""
    region_results = {}
    computed = set()
    if not jsonl_path.exists():
        return region_results, computed
    with open(jsonl_path) as jf:
        for line in jf:
            r = json.loads(line)
            region = r["region"]
            sess_idx = r["sess_idx"]
            computed.add((region, sess_idx))
            if region not in region_results:
                region_results[region] = {
                    "pca_lda_distances": [],
                    "conceptor_and_not": [],
                    "cross_condition_distances": [],
                    "alphas": [],
                }
            if r.get("alpha") is not None:
                region_results[region]["alphas"].append(r["alpha"])
            if r.get("pca_lda") is not None:
                region_results[region]["pca_lda_distances"].append(r["pca_lda"])
            if r.get("conceptor") is not None:
                region_results[region]["conceptor_and_not"].append(r["conceptor"])
            if r.get("cross_condition") is not None:
                region_results[region]["cross_condition_distances"].append(r["cross_condition"])
    return region_results, computed


def run(max_sessions: int | None = None):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    sessions = load_all()
    if max_sessions:
        sessions = sessions[:max_sessions]

    logger.info(f"{datetime.now().isoformat()} Starting cross-paper bridge with {len(sessions)} sessions")

    jsonl_path = RESULTS_DIR / "cross_paper_bridge_incremental.jsonl"
    region_results, computed = _rebuild_region_results_from_jsonl(jsonl_path)
    if computed:
        logger.info(f"Resuming: loaded {len(computed)} pre-computed region-session pairs")

    for sess_idx, sess in enumerate(tqdm(sessions, desc="Sessions")):
        choice_labels = get_choice_labels(sess)
        if len(np.unique(choice_labels)) < 2:
            continue

        evidence_labels = _contrast_to_evidence_label(sess)
        difficulty_labels = _easy_hard_labels(sess)
        regions = list_regions(sess, min_neurons=MIN_NEURONS)

        for region in regions:
            if (region, sess_idx) in computed:
                continue

            act = get_region_activity(sess, region)
            if act is None or act.shape[1] < MIN_NEURONS:
                continue

            n = min(act.shape[0], len(choice_labels))
            activity = act[:n, :, TIME_WINDOW].mean(axis=2)
            ch = choice_labels[:n]

            if region not in region_results:
                region_results[region] = {
                    "pca_lda_distances": [],
                    "conceptor_and_not": [],
                    "cross_condition_distances": [],
                    "alphas": [],
                }

            record = {"region": region, "sess_idx": sess_idx, "alpha": None,
                       "pca_lda": None, "conceptor": None, "cross_condition": None}

            alpha = _power_law_exponent(activity)
            if alpha is not None:
                region_results[region]["alphas"].append(alpha)
                record["alpha"] = alpha

            dist_result = _pca_lda_distances(activity, ch)
            if dist_result is not None:
                region_results[region]["pca_lda_distances"].append(dist_result)
                record["pca_lda"] = dist_result

            if evidence_labels is not None:
                ev = evidence_labels[:n]
                valid = ev >= 0
                if valid.sum() >= 2 * MIN_TRIALS_PER_CONDITION:
                    V_ev = _estimate_subspace(activity[valid], ev[valid])
                    V_ch = _estimate_subspace(activity[valid], ch[valid])
                    conceptor = _subspace_and_not(V_ev, V_ch)
                    if conceptor is not None:
                        region_results[region]["conceptor_and_not"].append(conceptor)
                        record["conceptor"] = conceptor

            diff = difficulty_labels[:n]
            valid_diff = diff >= 0
            if valid_diff.sum() >= 2 * MIN_TRIALS_PER_CONDITION:
                cross_cond = _cross_condition_distances(
                    activity[valid_diff], ch[valid_diff], diff[valid_diff])
                if cross_cond is not None:
                    region_results[region]["cross_condition_distances"].append(cross_cond)
                    record["cross_condition"] = cross_cond

            with open(jsonl_path, "a") as jf:
                jf.write(json.dumps(record, default=str) + "\n")

    # Aggregate per region
    DIST_METRICS = ["grassmannian", "chordal", "mean_principal_angle_deg", "subspace_overlap"]
    summary = {}
    alpha_pca_lda = {m: ([], []) for m in DIST_METRICS}
    alpha_conceptor = ([], [])
    alpha_cross_cond = {m: ([], []) for m in DIST_METRICS}

    for region, data in region_results.items():
        alpha = float(np.mean(data["alphas"])) if data["alphas"] else None

        pca_lda_agg = None
        if data["pca_lda_distances"]:
            pca_lda_agg = {
                m: float(np.mean([d[m] for d in data["pca_lda_distances"]]))
                for m in DIST_METRICS
            }

        mean_shared = None
        mean_private = None
        mean_overlap = None
        if data["conceptor_and_not"]:
            mean_shared = float(np.mean([c["n_shared_dims"] for c in data["conceptor_and_not"]]))
            mean_private = float(np.mean([c["n_private_dims"] for c in data["conceptor_and_not"]]))
            mean_overlap = float(np.mean([c["mean_overlap"] for c in data["conceptor_and_not"]]))

        cross_cond_agg = None
        if data["cross_condition_distances"]:
            cross_cond_agg = {
                m: float(np.mean([d[m] for d in data["cross_condition_distances"]]))
                for m in DIST_METRICS
            }

        summary[region] = {
            "power_law_alpha": alpha,
            "n_sessions": len(data["alphas"]),
            "pca_lda_distances": pca_lda_agg,
            "n_pca_lda": len(data["pca_lda_distances"]),
            "conceptor_mean_shared_dims": mean_shared,
            "conceptor_mean_private_dims": mean_private,
            "conceptor_mean_overlap": mean_overlap,
            "n_conceptor": len(data["conceptor_and_not"]),
            "cross_condition_distances": cross_cond_agg,
            "n_cross_condition": len(data["cross_condition_distances"]),
        }

        if alpha is not None and pca_lda_agg is not None:
            for m in DIST_METRICS:
                alpha_pca_lda[m][0].append(alpha)
                alpha_pca_lda[m][1].append(pca_lda_agg[m])
        if alpha is not None and mean_private is not None:
            alpha_conceptor[0].append(alpha)
            alpha_conceptor[1].append(mean_private)
        if alpha is not None and cross_cond_agg is not None:
            for m in DIST_METRICS:
                alpha_cross_cond[m][0].append(alpha)
                alpha_cross_cond[m][1].append(cross_cond_agg[m])

    prediction_tests = {}

    for metric in DIST_METRICS:
        alphas, vals = alpha_pca_lda[metric]
        if len(alphas) >= 5:
            boot = _bootstrap_spearman(alphas, vals)
            perm_p = _permutation_test_spearman(alphas, vals)
            prediction_tests[f"alpha_vs_pca_lda_{metric}"] = {
                **boot,
                "permutation_p": perm_p,
                "metric": metric,
                "interpretation": (
                    f"PCA-LDA {metric}: biological analog of ~88 deg attribution-causal "
                    "orthogonality in transformer paper."
                ),
            }

    alphas_c, vals_c = alpha_conceptor
    if len(alphas_c) >= 5:
        boot = _bootstrap_spearman(alphas_c, vals_c)
        perm_p = _permutation_test_spearman(alphas_c, vals_c)
        prediction_tests["alpha_vs_evidence_not_choice"] = {
            **boot,
            "permutation_p": perm_p,
            "interpretation": (
                "Positive rho = low-dim (high-alpha) regions have MORE evidence dims disconnected "
                "from choice. These regions encode evidence but don't route it to choice. "
                "Negative rho = high-dim regions encode more 'unused' evidence."
            ),
        }

    for metric in DIST_METRICS:
        alphas, vals = alpha_cross_cond[metric]
        if len(alphas) >= 5:
            boot = _bootstrap_spearman(alphas, vals)
            perm_p = _permutation_test_spearman(alphas, vals)
            prediction_tests[f"alpha_vs_cross_condition_{metric}"] = {
                **boot,
                "permutation_p": perm_p,
                "metric": metric,
                "interpretation": (
                    f"Cross-condition {metric}: analog of 58-78 deg per-counterfactual "
                    "result in transformer paper."
                ),
            }

    # Summary statistics across all metrics
    for metric in DIST_METRICS:
        _, vals = alpha_pca_lda[metric]
        if vals:
            prediction_tests[f"pca_lda_{metric}_distribution"] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
                "n_regions": len(vals),
            }
        _, vals = alpha_cross_cond[metric]
        if vals:
            prediction_tests[f"cross_condition_{metric}_distribution"] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
                "n_regions": len(vals),
            }

    results = {
        "timestamp": datetime.now().isoformat(),
        "n_sessions": len(sessions),
        "n_regions": len(summary),
        "region_results": summary,
        "prediction_tests": prediction_tests,
        "top_pca_lda_grassmannian": sorted(
            [(r, v["pca_lda_distances"]["grassmannian"]) for r, v in summary.items()
             if v["pca_lda_distances"] is not None],
            key=lambda x: x[1], reverse=True)[:10],
        "top_evidence_not_choice": sorted(
            [(r, v["conceptor_mean_private_dims"]) for r, v in summary.items()
             if v["conceptor_mean_private_dims"] is not None],
            key=lambda x: x[1], reverse=True)[:10],
        "top_cross_condition_grassmannian": sorted(
            [(r, v["cross_condition_distances"]["grassmannian"]) for r, v in summary.items()
             if v["cross_condition_distances"] is not None],
            key=lambda x: x[1], reverse=True)[:10],
    }

    out_path = RESULTS_DIR / "cross_paper_bridge.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"{datetime.now().isoformat()} Saved to {out_path}")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-sessions", type=int, default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    run(max_sessions=args.max_sessions)
