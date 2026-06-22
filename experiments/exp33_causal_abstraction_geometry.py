"""Experiment 33: Causal abstraction via geometric interchange interventions.

Tests whether brain regions implement a high-level causal model of
decision-making, using natural interventions (contrast conditions) as
the biological analog of DAS interchange interventions.

High-level causal model: evidence → choice
Alignment maps: LDA subspaces in neural activity
Interchange intervention: compare geometry across contrast conditions

For each region, we estimate:
- Evidence subspace V_ev (LDA: high-left-contrast vs high-right-contrast trials)
- Choice subspace V_ch (LDA: left-choice vs right-choice trials)
- Grassmannian distance between V_ev and V_ch (measures alignment)

Key predictions:
1. Regions where d_G(V_ev, V_ch) is small are "evidence-to-choice routing" regions
   — evidence and choice live in overlapping subspaces.
2. CKA-type (low-dim) regions should have higher geometric IIA for linear
   causal variables than Procrustes-type (high-dim) regions.
3. The contrast manipulation is a natural do(evidence) intervention — comparing
   geometry across contrast conditions tests the causal, not just correlational,
   structure.

This has never been done: causal abstraction analysis on biological neural
data using geometric (Grassmannian) alignment maps rather than activation patching.
"""
import json
import logging
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from tqdm import tqdm

from data.steinmetz import get_choice_labels, get_region_activity, list_regions, load_all
from geometry.distances import all_subspace_distances, grassmannian_distance, subspace_overlap

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent.parent / "artifacts" / "exp33"
MIN_NEURONS = 15
TIME_WINDOW = slice(15, 35)
SUBSPACE_DIM = 5
MIN_TRIALS_PER_CONDITION = 20


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
    """Estimate a discriminative subspace via LDA + PCA.

    Returns orthonormal basis (n_neurons, min(n_dims, n_neurons-1)).
    """
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

    lda_direction = lda.coef_[0]
    lda_direction = lda_direction / (np.linalg.norm(lda_direction) + 1e-10)
    lda_in_neuron_space = pca.components_.T @ lda_direction

    pca_components = pca.components_[:n_dims].T
    combined = np.column_stack([lda_in_neuron_space.reshape(-1, 1), pca_components])
    Q, _ = np.linalg.qr(combined)
    return Q[:, :n_dims]


def _geometric_iia(activity, evidence_labels, choice_labels):
    """Geometric interchange intervention accuracy.

    Measures how much the evidence subspace predicts the choice subspace
    geometry. High IIA = evidence and choice are geometrically aligned.
    """
    V_ev = _estimate_subspace(activity, evidence_labels)
    V_ch = _estimate_subspace(activity, choice_labels)

    if V_ev is None or V_ch is None:
        return None

    k = min(V_ev.shape[1], V_ch.shape[1])
    V_ev_k = V_ev[:, :k]
    V_ch_k = V_ch[:, :k]

    try:
        dists = all_subspace_distances(V_ev_k, V_ch_k)
    except Exception:
        return None

    evidence_projected = activity @ V_ev_k
    ev_pred_choice = np.corrcoef(evidence_projected[:, 0], choice_labels)[0, 1]

    return {
        **dists,
        "grassmannian_distance": dists["grassmannian"],
        "evidence_choice_corr": float(ev_pred_choice) if not np.isnan(ev_pred_choice) else 0.0,
        "subspace_dims": k,
    }


def _contrast_to_evidence_label(sess):
    """Create evidence labels from contrast conditions.

    Returns binary labels: 1 = evidence favors right, 0 = evidence favors left.
    Uses the sign of (contrast_right - contrast_left).
    """
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


def _cross_condition_geometry(activity, evidence_labels, choice_labels, n_dims=SUBSPACE_DIM):
    """Test if evidence subspace rotates between left-evidence and right-evidence trials.

    This is the geometric analog of the interchange intervention:
    how much does the choice subspace change when evidence changes?
    """
    left_ev = evidence_labels == 0
    right_ev = evidence_labels == 1

    if left_ev.sum() < MIN_TRIALS_PER_CONDITION or right_ev.sum() < MIN_TRIALS_PER_CONDITION:
        return None

    V_choice_left = _estimate_subspace(activity[left_ev], choice_labels[left_ev])
    V_choice_right = _estimate_subspace(activity[right_ev], choice_labels[right_ev])

    if V_choice_left is None or V_choice_right is None:
        return None

    k = min(V_choice_left.shape[1], V_choice_right.shape[1])
    try:
        dists = all_subspace_distances(V_choice_left[:, :k], V_choice_right[:, :k])
    except Exception:
        return None

    return {
        **{f"choice_shift_{m}": v for m, v in dists.items()},
        "choice_subspace_shift": dists["grassmannian"],
        "choice_subspace_overlap_across_evidence": dists["subspace_overlap"],
        "n_left_ev": int(left_ev.sum()),
        "n_right_ev": int(right_ev.sum()),
    }


def run(max_sessions: int | None = None):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    sessions = load_all()
    if max_sessions:
        sessions = sessions[:max_sessions]

    region_data = {}

    for sess_idx, sess in enumerate(tqdm(sessions, desc="Sessions")):
        choice_labels = get_choice_labels(sess)
        if len(np.unique(choice_labels)) < 2:
            continue

        evidence_labels = _contrast_to_evidence_label(sess)
        if evidence_labels is None:
            continue

        mouse = str(sess.get("mouse_name", f"mouse_{sess_idx}"))
        regions = list_regions(sess, min_neurons=MIN_NEURONS)

        for region in regions:
            act = get_region_activity(sess, region)
            if act is None or act.shape[1] < MIN_NEURONS:
                continue

            n = min(act.shape[0], len(choice_labels), len(evidence_labels))
            activity = act[:n, :, TIME_WINDOW].mean(axis=2)
            ch = choice_labels[:n]
            ev = evidence_labels[:n]

            valid = ev >= 0
            if valid.sum() < MIN_TRIALS_PER_CONDITION * 2:
                continue

            if region not in region_data:
                region_data[region] = []
            region_data[region].append({
                "session_idx": sess_idx,
                "mouse": mouse,
                "activity": activity[valid],
                "choice_labels": ch[valid],
                "evidence_labels": ev[valid],
                "n_neurons": activity.shape[1],
            })

    region_results = {}
    alpha_list = []
    iia_list = []
    shift_list = []
    overlap_list = []

    for region, measurements in tqdm(region_data.items(), desc="Causal abstraction"):
        iia_scores = []
        shifts = []
        overlaps = []

        for m in measurements:
            iia = _geometric_iia(m["activity"], m["evidence_labels"], m["choice_labels"])
            if iia is not None:
                iia_scores.append(iia)

            cross = _cross_condition_geometry(
                m["activity"], m["evidence_labels"], m["choice_labels"])
            if cross is not None:
                shifts.append(cross["choice_subspace_shift"])
                overlaps.append(cross["choice_subspace_overlap_across_evidence"])

        alphas = [_power_law_exponent(m["activity"]) for m in measurements]
        alphas = [a for a in alphas if a is not None]
        alpha = float(np.mean(alphas)) if alphas else None

        region_results[region] = {
            "n_sessions": len(measurements),
            "power_law_alpha": alpha,
            "mean_grassmannian_ev_ch": (
                float(np.mean([s["grassmannian_distance"] for s in iia_scores]))
                if iia_scores else None
            ),
            "mean_subspace_overlap": (
                float(np.mean([s["subspace_overlap"] for s in iia_scores]))
                if iia_scores else None
            ),
            "mean_ev_choice_corr": (
                float(np.mean([s["evidence_choice_corr"] for s in iia_scores]))
                if iia_scores else None
            ),
            "mean_choice_subspace_shift": float(np.mean(shifts)) if shifts else None,
            "mean_cross_evidence_overlap": float(np.mean(overlaps)) if overlaps else None,
        }

        if alpha is not None and iia_scores:
            alpha_list.append(alpha)
            iia_list.append(np.mean([s["subspace_overlap"] for s in iia_scores]))
            if shifts:
                shift_list.append(np.mean(shifts))
            if overlaps:
                overlap_list.append(np.mean(overlaps))

    prediction_tests = {}

    if len(alpha_list) >= 4 and len(iia_list) >= 4:
        rho, p = spearmanr(alpha_list, iia_list)
        prediction_tests["alpha_vs_geometric_iia"] = {
            "rho": float(rho), "p": float(p), "n": len(alpha_list),
            "interpretation": (
                "Positive rho means low-dim (high-alpha) regions have higher "
                "evidence-choice subspace overlap — CKA-type regions implement "
                "the linear evidence→choice causal variable more faithfully."
            ),
        }

    if len(alpha_list) >= 4 and len(shift_list) >= 4:
        n = min(len(alpha_list), len(shift_list))
        rho, p = spearmanr(alpha_list[:n], shift_list[:n])
        prediction_tests["alpha_vs_choice_shift"] = {
            "rho": float(rho), "p": float(p), "n": n,
            "interpretation": (
                "Negative rho means high-dim regions show larger choice subspace "
                "rotation when evidence changes — their geometry is more sensitive "
                "to the evidence intervention."
            ),
        }

    routing_regions = sorted(
        [(r, v) for r, v in region_results.items()
         if v["mean_grassmannian_ev_ch"] is not None],
        key=lambda x: x[1]["mean_grassmannian_ev_ch"]
    )
    top_routing = routing_regions[:5] if routing_regions else []
    bottom_routing = routing_regions[-5:] if routing_regions else []

    results = {
        "timestamp": datetime.now().isoformat(),
        "n_sessions": len(sessions),
        "n_regions": len(region_data),
        "n_regions_analyzed": len(region_results),
        "region_results": region_results,
        "prediction_tests": prediction_tests,
        "top_routing_regions": [
            {"region": r, "grassmannian_ev_ch": v["mean_grassmannian_ev_ch"],
             "alpha": v["power_law_alpha"]}
            for r, v in top_routing
        ],
        "bottom_routing_regions": [
            {"region": r, "grassmannian_ev_ch": v["mean_grassmannian_ev_ch"],
             "alpha": v["power_law_alpha"]}
            for r, v in bottom_routing
        ],
    }

    out_path = RESULTS_DIR / "causal_abstraction_geometry.json"
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
