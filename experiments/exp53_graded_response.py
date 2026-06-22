"""Experiment 53: Graded response — progressive subspace ablation.

MECHVAL E5 (Graded Response): "Partial ablation produces partial effects."

Tests dose-response: progressively remove dimensions from the evidence subspace
and measure choice decoding degradation. If the response is monotonic (more
dimensions removed → worse decoding), this establishes graded causal involvement
rather than all-or-nothing.

Also tests graded IIA: swap only k of SUBSPACE_DIM dimensions and measure
flip rate. If flip rate increases monotonically with k, the subspace has
graded causal influence.

From pharmacology: a causally relevant component produces systematic dose-
response, not just binary on/off.
"""
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from data.steinmetz import get_choice_labels, get_region_activity, list_regions, load_all

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent.parent / "artifacts" / "exp53"
MIN_NEURONS = 15
TIME_WINDOW = slice(15, 35)
SUBSPACE_DIM = 5
MIN_TRIALS_PER_CONDITION = 20


def _contrast_to_evidence_label(sess):
    cl = sess.get("contrast_left", np.array([]))
    cr = sess.get("contrast_right", np.array([]))
    if len(cl) == 0 or len(cr) == 0:
        return None
    n = min(len(cl), len(cr))
    evidence = cr[:n] - cl[:n]
    if (evidence != 0).sum() < MIN_TRIALS_PER_CONDITION:
        return None
    labels = np.zeros(n, dtype=int)
    labels[evidence > 0] = 1
    labels[evidence < 0] = 0
    labels[evidence == 0] = -1
    return labels


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


def _ablated_decoding(activity, ch_labels, V_ev, k_remove):
    if k_remove == 0:
        return _cv_decode(activity, ch_labels)
    V_remove = V_ev[:, :k_remove]
    proj = activity @ V_remove @ V_remove.T
    ablated = activity - proj
    return _cv_decode(ablated, ch_labels)


def _cv_decode(activity, labels, n_splits=5):
    if len(np.unique(labels)) < 2:
        return None
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []
    for train_idx, test_idx in skf.split(activity, labels):
        lda = LinearDiscriminantAnalysis()
        try:
            lda.fit(activity[train_idx], labels[train_idx])
            scores.append(lda.score(activity[test_idx], labels[test_idx]))
        except Exception:
            return None
    return float(np.mean(scores))


def _graded_iia(activity, ev_labels, ch_labels, V_ev, k_swap):
    left_idx = np.where(ev_labels == 0)[0]
    right_idx = np.where(ev_labels == 1)[0]
    if len(left_idx) < MIN_TRIALS_PER_CONDITION or len(right_idx) < MIN_TRIALS_PER_CONDITION:
        return None

    lda = LinearDiscriminantAnalysis()
    try:
        lda.fit(activity, ch_labels)
    except Exception:
        return None

    V_partial = V_ev[:, :k_swap]
    n_pairs = min(len(left_idx), len(right_idx), 50)
    rng = np.random.default_rng(42)
    sample_l = left_idx[rng.choice(len(left_idx), n_pairs, replace=False)]
    sample_r = right_idx[rng.choice(len(right_idx), n_pairs, replace=False)]

    flips, total = 0, 0
    for li, ri in zip(sample_l, sample_r):
        x = activity[li].copy()
        proj_l = V_partial @ (V_partial.T @ activity[li])
        proj_r = V_partial @ (V_partial.T @ activity[ri])
        if np.linalg.norm(proj_r - proj_l) < 1e-10:
            continue
        orig = lda.predict(x.reshape(1, -1))[0]
        swapped = x - proj_l + proj_r
        if lda.predict(swapped.reshape(1, -1))[0] != orig:
            flips += 1
        total += 1
    return float(flips / total) if total > 0 else None


def run(max_sessions=None):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    sessions = load_all()
    if max_sessions:
        sessions = sessions[:max_sessions]

    region_dose_response = {}

    for sess_idx, sess in enumerate(tqdm(sessions, desc="Graded response")):
        choice_labels = get_choice_labels(sess)
        if len(np.unique(choice_labels)) < 2:
            continue
        ev_labels = _contrast_to_evidence_label(sess)
        if ev_labels is None:
            continue

        for region in list_regions(sess, min_neurons=MIN_NEURONS):
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

            act_v = activity[valid]
            ev_v = ev[valid]
            ch_v = ch[valid]

            V_ev = _estimate_subspace(act_v, ev_v)
            if V_ev is None:
                continue

            n_dims = V_ev.shape[1]
            ablation_curve = []
            for k in range(n_dims + 1):
                acc = _ablated_decoding(act_v, ch_v, V_ev, k)
                ablation_curve.append({"k_removed": k, "accuracy": acc})

            iia_curve = []
            for k in range(1, n_dims + 1):
                iia = _graded_iia(act_v, ev_v, ch_v, V_ev, k)
                iia_curve.append({"k_swapped": k, "iia": iia})

            abl_accs = [e["accuracy"] for e in ablation_curve if e["accuracy"] is not None]
            iia_vals = [e["iia"] for e in iia_curve if e["iia"] is not None]

            abl_monotonic = all(abl_accs[i] >= abl_accs[i+1] - 0.02 for i in range(len(abl_accs)-1)) if len(abl_accs) >= 3 else None
            iia_monotonic = all(iia_vals[i] <= iia_vals[i+1] + 0.02 for i in range(len(iia_vals)-1)) if len(iia_vals) >= 3 else None

            entry = {
                "ablation_curve": ablation_curve,
                "iia_curve": iia_curve,
                "ablation_monotonic": abl_monotonic,
                "iia_monotonic": iia_monotonic,
                "n_dims": n_dims,
                "n_neurons": int(act_v.shape[1]),
            }

            region_dose_response.setdefault(region, []).append(entry)

    summary = {}
    for region, entries in region_dose_response.items():
        abl_mono = [e["ablation_monotonic"] for e in entries if e["ablation_monotonic"] is not None]
        iia_mono = [e["iia_monotonic"] for e in entries if e["iia_monotonic"] is not None]
        summary[region] = {
            "n_sessions": len(entries),
            "ablation_monotonic_rate": float(np.mean(abl_mono)) if abl_mono else None,
            "iia_monotonic_rate": float(np.mean(iia_mono)) if iia_mono else None,
        }

    abl_rates = [s["ablation_monotonic_rate"] for s in summary.values() if s["ablation_monotonic_rate"] is not None]
    iia_rates = [s["iia_monotonic_rate"] for s in summary.values() if s["iia_monotonic_rate"] is not None]

    results = {
        "timestamp": datetime.now().isoformat(),
        "mechval_criterion": "E5 Graded Response (dose-response)",
        "mechval_threshold": "monotonic dose-response in >= 70% of regions",
        "n_regions": len(summary),
        "overall_ablation_monotonic": float(np.mean(abl_rates)) if abl_rates else None,
        "overall_iia_monotonic": float(np.mean(iia_rates)) if iia_rates else None,
        "graded_response_pass": (float(np.mean(abl_rates)) >= 0.7 if abl_rates else False),
        "per_region": summary,
    }

    out_path = RESULTS_DIR / "graded_response.json"
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
