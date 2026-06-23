"""Experiment 48: Sufficiency test — does the evidence subspace alone predict choice?

MECHVAL I2 (Sufficiency): "Isolating/restoring the circuit alone reproduces behavior."

Three computational analogs of stimulation:
  1. Subspace-restricted decoding: project onto evidence subspace, decode choice from
     projection only. If accuracy >> chance, the subspace is informationally sufficient.
  2. Synthetic restoration: zero out the evidence subspace, replace with opposite-
     condition mean projection, measure choice flip rate. If flips >> chance, the
     subspace is causally sufficient for choice.
  3. Subspace-only reconstruction: decode choice from ONLY the evidence subspace
     projection vs from the full activity. Report recovery fraction (≥0.70 = pass).

These are the observational analogs of optogenetic stimulation — we can't activate
neurons, but we can computationally isolate the evidence representation and test
whether it alone determines choice.
"""
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from data.steinmetz import get_choice_labels, get_region_activity, list_regions, load_all

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent.parent / "artifacts" / "exp48"
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


def _subspace_restricted_decoding(activity, ch_labels, V_ev):
    proj = activity @ V_ev
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    for train_idx, test_idx in skf.split(proj, ch_labels):
        lda = LinearDiscriminantAnalysis()
        lda.fit(proj[train_idx], ch_labels[train_idx])
        scores.append(lda.score(proj[test_idx], ch_labels[test_idx]))
    return float(np.mean(scores))


def _full_decoding(activity, ch_labels):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    for train_idx, test_idx in skf.split(activity, ch_labels):
        lda = LinearDiscriminantAnalysis()
        lda.fit(activity[train_idx], ch_labels[train_idx])
        scores.append(lda.score(activity[test_idx], ch_labels[test_idx]))
    return float(np.mean(scores))


def _synthetic_restoration(activity, ev_labels, ch_labels, V_ev):
    left_idx = np.where(ev_labels == 0)[0]
    right_idx = np.where(ev_labels == 1)[0]
    if len(left_idx) < MIN_TRIALS_PER_CONDITION or len(right_idx) < MIN_TRIALS_PER_CONDITION:
        return None

    lda = LinearDiscriminantAnalysis()
    try:
        lda.fit(activity, ch_labels)
    except Exception:
        return None

    mean_left_proj = np.mean(activity[left_idx] @ V_ev, axis=0)
    mean_right_proj = np.mean(activity[right_idx] @ V_ev, axis=0)

    flips = 0
    total = 0
    n_test = min(len(left_idx), 100)
    rng = np.random.default_rng(42)
    test_trials = rng.choice(left_idx, n_test, replace=False)

    for ti in test_trials:
        x = activity[ti].copy()
        orig_pred = lda.predict(x.reshape(1, -1))[0]
        proj = V_ev @ (V_ev.T @ x)
        x_zeroed = x - proj
        x_restored = x_zeroed + V_ev @ mean_right_proj
        new_pred = lda.predict(x_restored.reshape(1, -1))[0]
        if new_pred != orig_pred:
            flips += 1
        total += 1

    for ti in rng.choice(right_idx, min(len(right_idx), 100), replace=False):
        x = activity[ti].copy()
        orig_pred = lda.predict(x.reshape(1, -1))[0]
        proj = V_ev @ (V_ev.T @ x)
        x_zeroed = x - proj
        x_restored = x_zeroed + V_ev @ mean_left_proj
        new_pred = lda.predict(x_restored.reshape(1, -1))[0]
        if new_pred != orig_pred:
            flips += 1
        total += 1

    return float(flips / total) if total > 0 else None


def run(max_sessions=None):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    sessions = load_all()
    if max_sessions:
        sessions = sessions[:max_sessions]

    region_results = {}

    for sess_idx, sess in enumerate(tqdm(sessions, desc="Sufficiency tests")):
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
            ch_v = ch[valid]
            ev_v = ev[valid]

            V_ev = _estimate_subspace(act_v, ev_v)
            if V_ev is None:
                continue

            full_acc = _full_decoding(act_v, ch_v)
            restricted_acc = _subspace_restricted_decoding(act_v, ch_v, V_ev)
            recovery = restricted_acc / full_acc if full_acc > 0.5 else None
            restoration = _synthetic_restoration(act_v, ev_v, ch_v, V_ev)

            entry = {
                "full_accuracy": full_acc,
                "restricted_accuracy": restricted_acc,
                "recovery_fraction": recovery,
                "restoration_flip_rate": restoration,
                "n_neurons": int(act_v.shape[1]),
                "n_trials": int(act_v.shape[0]),
            }

            if region not in region_results:
                region_results[region] = []
            region_results[region].append(entry)

    summary = {}
    for region, entries in region_results.items():
        recoveries = [e["recovery_fraction"] for e in entries if e["recovery_fraction"] is not None]
        restorations = [e["restoration_flip_rate"] for e in entries if e["restoration_flip_rate"] is not None]
        restricted = [e["restricted_accuracy"] for e in entries]
        full = [e["full_accuracy"] for e in entries]

        summary[region] = {
            "n_sessions": len(entries),
            "mean_full_accuracy": float(np.mean(full)),
            "mean_restricted_accuracy": float(np.mean(restricted)),
            "mean_recovery_fraction": float(np.mean(recoveries)) if recoveries else None,
            "mean_restoration_flip_rate": float(np.mean(restorations)) if restorations else None,
            "sufficiency_pass": float(np.mean(recoveries)) >= 0.70 if recoveries else False,
        }

    all_recoveries = [s["mean_recovery_fraction"] for s in summary.values() if s["mean_recovery_fraction"] is not None]
    all_restorations = [s["mean_restoration_flip_rate"] for s in summary.values() if s["mean_restoration_flip_rate"] is not None]

    results = {
        "timestamp": datetime.now().isoformat(),
        "mechval_criterion": "I2 Sufficiency",
        "mechval_threshold": "recovery_fraction >= 0.70",
        "n_regions": len(summary),
        "regions_passing_sufficiency": sum(1 for s in summary.values() if s["sufficiency_pass"]),
        "overall_mean_recovery": float(np.mean(all_recoveries)) if all_recoveries else None,
        "overall_mean_restoration": float(np.mean(all_restorations)) if all_restorations else None,
        "per_region": summary,
        "raw_entries": {r: entries for r, entries in region_results.items()},
    }

    out_path = RESULTS_DIR / "sufficiency.json"
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
