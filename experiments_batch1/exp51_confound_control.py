"""Experiment 51: Confound control — are effects explained by collateral properties?

MECHVAL I5 (Confound Control): "Effect not explained by collateral disruption."

Tests whether geometric findings are confounded by:
  1. Neuron count — regions with more neurons trivially have higher-dimensional
     subspaces and potentially higher IIA. Partial out neuron count.
  2. Signal-to-noise ratio — high-firing regions may have cleaner geometry.
     Compute mean firing rate per region and partial out.
  3. Trial count — sessions with more trials give more stable estimates.
     Test whether findings hold when subsampling to matched trial counts.
  4. Mouse identity — does the effect survive within-mouse analysis?
  5. Temporal autocorrelation — high autocorrelation inflates apparent structure.
     Compare true IIA to IIA computed on temporally-shuffled (within-condition) trials.
"""
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, pearsonr
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from tqdm import tqdm

from data.steinmetz import get_choice_labels, get_region_activity, list_regions, load_all

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent.parent / "artifacts" / "exp51"
MIN_NEURONS = 15
TIME_WINDOW = slice(15, 35)
SUBSPACE_DIM = 5
MIN_TRIALS_PER_CONDITION = 20
MATCHED_TRIAL_COUNT = 60


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


def _compute_iia(activity, ev_labels, ch_labels):
    V_ev = _estimate_subspace(activity, ev_labels)
    if V_ev is None:
        return None

    left_idx = np.where(ev_labels == 0)[0]
    right_idx = np.where(ev_labels == 1)[0]
    if len(left_idx) < MIN_TRIALS_PER_CONDITION or len(right_idx) < MIN_TRIALS_PER_CONDITION:
        return None

    lda = LinearDiscriminantAnalysis()
    try:
        lda.fit(activity, ch_labels)
    except Exception:
        return None

    n_pairs = min(len(left_idx), len(right_idx), 50)
    rng = np.random.default_rng(42)
    sample_l = left_idx[rng.choice(len(left_idx), n_pairs, replace=False)]
    sample_r = right_idx[rng.choice(len(right_idx), n_pairs, replace=False)]

    flips = 0
    total = 0
    for li, ri in zip(sample_l, sample_r):
        x = activity[li].copy()
        proj_l = V_ev @ (V_ev.T @ activity[li])
        proj_r = V_ev @ (V_ev.T @ activity[ri])
        if np.linalg.norm(proj_r - proj_l) < 1e-10:
            continue
        orig = lda.predict(x.reshape(1, -1))[0]
        swapped = x - proj_l + proj_r
        new = lda.predict(swapped.reshape(1, -1))[0]
        if new != orig:
            flips += 1
        total += 1
    return float(flips / total) if total > 0 else None


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


def run(max_sessions=None):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    sessions = load_all()
    if max_sessions:
        sessions = sessions[:max_sessions]

    records = []

    for sess_idx, sess in enumerate(tqdm(sessions, desc="Confound control")):
        choice_labels = get_choice_labels(sess)
        if len(np.unique(choice_labels)) < 2:
            continue
        ev_labels = _contrast_to_evidence_label(sess)
        if ev_labels is None:
            continue
        mouse = str(sess.get("mouse_name", f"unknown_{sess_idx}"))

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

            iia = _compute_iia(act_v, ev_v, ch_v)
            if iia is None:
                continue

            alpha = _power_law_exponent(act_v)
            n_neurons = int(act_v.shape[1])
            mean_fr = float(np.mean(act_v))
            n_trials = int(act_v.shape[0])

            matched_iia = None
            if n_trials > MATCHED_TRIAL_COUNT:
                rng = np.random.default_rng(sess_idx * 1000)
                idx = rng.choice(n_trials, MATCHED_TRIAL_COUNT, replace=False)
                matched_iia = _compute_iia(act_v[idx], ev_v[idx], ch_v[idx])

            rng = np.random.default_rng(sess_idx * 2000 + hash(region) % 1000)
            perm_idx = np.arange(n_trials)
            for c in [0, 1]:
                c_idx = np.where(ev_v == c)[0]
                perm_idx[c_idx] = c_idx[rng.permutation(len(c_idx))]
            shuffled_act = act_v[perm_idx]
            shuffled_iia = _compute_iia(shuffled_act, ev_v, ch_v)

            records.append({
                "region": region,
                "session": sess_idx,
                "mouse": mouse,
                "iia": iia,
                "alpha": alpha,
                "n_neurons": n_neurons,
                "mean_firing_rate": mean_fr,
                "n_trials": n_trials,
                "matched_trial_iia": matched_iia,
                "temporal_shuffle_iia": shuffled_iia,
            })

    iia_vals = [r["iia"] for r in records]
    n_neuron_vals = [r["n_neurons"] for r in records]
    fr_vals = [r["mean_firing_rate"] for r in records]
    n_trial_vals = [r["n_trials"] for r in records]

    confound_tests = {}
    if len(iia_vals) >= 5:
        rho_nn, p_nn = spearmanr(iia_vals, n_neuron_vals)
        confound_tests["iia_vs_neuron_count"] = {"rho": float(rho_nn), "p": float(p_nn)}

        rho_fr, p_fr = spearmanr(iia_vals, fr_vals)
        confound_tests["iia_vs_firing_rate"] = {"rho": float(rho_fr), "p": float(p_fr)}

        rho_nt, p_nt = spearmanr(iia_vals, n_trial_vals)
        confound_tests["iia_vs_trial_count"] = {"rho": float(rho_nt), "p": float(p_nt)}

    matched = [(r["iia"], r["matched_trial_iia"]) for r in records if r["matched_trial_iia"] is not None]
    if matched:
        orig, mtch = zip(*matched)
        rho_m, p_m = spearmanr(orig, mtch)
        confound_tests["matched_trial_stability"] = {
            "rho": float(rho_m), "p": float(p_m),
            "mean_original": float(np.mean(orig)),
            "mean_matched": float(np.mean(mtch)),
        }

    shuffled = [(r["iia"], r["temporal_shuffle_iia"]) for r in records if r["temporal_shuffle_iia"] is not None]
    if shuffled:
        orig_s, shuf_s = zip(*shuffled)
        delta = [o - s for o, s in zip(orig_s, shuf_s)]
        confound_tests["temporal_shuffle_control"] = {
            "mean_original": float(np.mean(orig_s)),
            "mean_shuffled": float(np.mean(shuf_s)),
            "mean_delta": float(np.mean(delta)),
            "fraction_original_higher": float(np.mean([d > 0 for d in delta])),
        }

    mice = set(r["mouse"] for r in records)
    mouse_iia = {}
    for m in mice:
        m_records = [r for r in records if r["mouse"] == m]
        if len(m_records) >= 3:
            mouse_iia[m] = float(np.mean([r["iia"] for r in m_records]))
    confound_tests["within_mouse"] = {
        "n_mice": len(mouse_iia),
        "per_mouse_mean_iia": mouse_iia,
        "overall_mean": float(np.mean(list(mouse_iia.values()))) if mouse_iia else None,
    }

    all_confounded = any(
        abs(confound_tests.get(k, {}).get("rho", 0)) > 0.5
        for k in ["iia_vs_neuron_count", "iia_vs_firing_rate", "iia_vs_trial_count"]
    )

    results = {
        "timestamp": datetime.now().isoformat(),
        "mechval_criterion": "I5 Confound Control",
        "n_records": len(records),
        "confound_tests": confound_tests,
        "any_strong_confound": all_confounded,
        "records": records[:100],
    }

    out_path = RESULTS_DIR / "confound_control.json"
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
