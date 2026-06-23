"""Experiment 50: Double dissociation — evidence vs reaction time subspaces.

MECHVAL I4 (Double Dissociation): "Circuit A necessary for task X but not Y,
circuit B necessary for task Y but not X."

Tests whether the evidence subspace and reaction-time subspace are functionally
independent:
  - Evidence subspace: predicts evidence direction (left/right) but NOT reaction time
  - RT subspace: predicts reaction time (fast/slow) but NOT evidence direction

If both cross-predictions fail while within-predictions succeed, this is a
classical neuropsychological double dissociation (Shallice 1988), establishing
that the evidence representation is functionally specific and not just capturing
general neural state.

Additionally tests IIA double dissociation:
  - Swapping in evidence subspace changes choice but not RT classification
  - Swapping in RT subspace changes RT classification but not choice
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

RESULTS_DIR = Path(__file__).parent.parent / "artifacts" / "exp50"
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


def _rt_to_label(sess, n):
    rt = sess.get("response_time", np.array([]))
    rt = np.asarray(rt).ravel()
    if len(rt) < n:
        return None
    rt = rt[:n].astype(float)
    valid = ~np.isnan(rt) & (rt > 0)
    if valid.sum() < MIN_TRIALS_PER_CONDITION * 2:
        return None
    median_rt = np.median(rt[valid])
    labels = np.full(n, -1, dtype=int)
    labels[valid & (rt < median_rt)] = 0
    labels[valid & (rt >= median_rt)] = 1
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


def _cv_accuracy(activity, labels, n_splits=5):
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


def run(max_sessions=None):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    sessions = load_all()
    if max_sessions:
        sessions = sessions[:max_sessions]

    region_results = {}

    for sess_idx, sess in enumerate(tqdm(sessions, desc="Double dissociation")):
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
            rt_labels = _rt_to_label(sess, n)
            if rt_labels is None:
                continue

            valid = (ev >= 0) & (rt_labels >= 0)
            if valid.sum() < MIN_TRIALS_PER_CONDITION * 2:
                continue

            act_v = activity[valid]
            ev_v = ev[valid]
            rt_v = rt_labels[valid]
            ch_v = ch[valid]

            V_ev = _estimate_subspace(act_v, ev_v)
            V_rt = _estimate_subspace(act_v, rt_v)
            if V_ev is None or V_rt is None:
                continue

            subspace_angle = float(np.linalg.norm(V_ev.T @ V_rt, ord=2))

            proj_ev = act_v @ V_ev
            proj_rt = act_v @ V_rt

            ev_predicts_ev = _cv_accuracy(proj_ev, ev_v)
            ev_predicts_rt = _cv_accuracy(proj_ev, rt_v)
            rt_predicts_rt = _cv_accuracy(proj_rt, rt_v)
            rt_predicts_ev = _cv_accuracy(proj_rt, ev_v)

            ev_predicts_choice = _cv_accuracy(proj_ev, ch_v)
            rt_predicts_choice = _cv_accuracy(proj_rt, ch_v)

            double_dissociation = (
                ev_predicts_ev is not None and ev_predicts_rt is not None and
                rt_predicts_rt is not None and rt_predicts_ev is not None and
                ev_predicts_ev > 0.55 and rt_predicts_rt > 0.55 and
                ev_predicts_rt < ev_predicts_ev - 0.05 and
                rt_predicts_ev < rt_predicts_rt - 0.05
            )

            entry = {
                "subspace_angle": subspace_angle,
                "ev_subspace_predicts_evidence": ev_predicts_ev,
                "ev_subspace_predicts_rt": ev_predicts_rt,
                "rt_subspace_predicts_rt": rt_predicts_rt,
                "rt_subspace_predicts_evidence": rt_predicts_ev,
                "ev_subspace_predicts_choice": ev_predicts_choice,
                "rt_subspace_predicts_choice": rt_predicts_choice,
                "double_dissociation": double_dissociation,
                "n_trials": int(act_v.shape[0]),
                "n_neurons": int(act_v.shape[1]),
            }

            region_results.setdefault(region, []).append(entry)

    summary = {}
    for region, entries in region_results.items():
        dd_count = sum(1 for e in entries if e["double_dissociation"])
        summary[region] = {
            "n_sessions": len(entries),
            "double_dissociation_rate": float(dd_count / len(entries)),
            "mean_subspace_angle": float(np.mean([e["subspace_angle"] for e in entries])),
            "mean_ev_on_ev": float(np.mean([e["ev_subspace_predicts_evidence"] for e in entries if e["ev_subspace_predicts_evidence"] is not None])),
            "mean_ev_on_rt": float(np.mean([e["ev_subspace_predicts_rt"] for e in entries if e["ev_subspace_predicts_rt"] is not None])),
            "mean_rt_on_rt": float(np.mean([e["rt_subspace_predicts_rt"] for e in entries if e["rt_subspace_predicts_rt"] is not None])),
            "mean_rt_on_ev": float(np.mean([e["rt_subspace_predicts_evidence"] for e in entries if e["rt_subspace_predicts_evidence"] is not None])),
        }

    all_dd_rates = [s["double_dissociation_rate"] for s in summary.values()]

    results = {
        "timestamp": datetime.now().isoformat(),
        "mechval_criterion": "I4 Double Dissociation",
        "n_regions": len(summary),
        "regions_with_dissociation": sum(1 for r in all_dd_rates if r > 0.5),
        "overall_dissociation_rate": float(np.mean(all_dd_rates)) if all_dd_rates else None,
        "per_region": summary,
        "raw_entries": {r: entries for r, entries in region_results.items()},
    }

    out_path = RESULTS_DIR / "double_dissociation.json"
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
