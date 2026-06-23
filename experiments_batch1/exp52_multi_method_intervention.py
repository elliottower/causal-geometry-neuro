"""Experiment 52: Multi-method intervention agreement.

MECHVAL E1 (Intervention Reach): "Different intervention methods agree."

Tests whether the IIA effect is robust across 4 intervention methods:
  1. Linear projection swap (current method) — swap evidence projection
  2. Gaussian noise injection — add noise proportional to evidence direction
  3. Mean-shift — shift activity toward opposite-condition mean in evidence subspace
  4. Subspace zeroing — zero out evidence subspace projection entirely

If all four methods produce similar choice-flip patterns (rank correlation across
regions ≥ 0.7), the effect is not an artifact of the specific intervention.
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

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent.parent / "artifacts" / "exp52"
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


def _iia_projection_swap(activity, ev_labels, ch_labels, V_ev, lda, rng):
    left_idx = np.where(ev_labels == 0)[0]
    right_idx = np.where(ev_labels == 1)[0]
    n_pairs = min(len(left_idx), len(right_idx), 50)
    sample_l = left_idx[rng.choice(len(left_idx), n_pairs, replace=False)]
    sample_r = right_idx[rng.choice(len(right_idx), n_pairs, replace=False)]
    flips, total = 0, 0
    for li, ri in zip(sample_l, sample_r):
        x = activity[li].copy()
        proj_l = V_ev @ (V_ev.T @ activity[li])
        proj_r = V_ev @ (V_ev.T @ activity[ri])
        if np.linalg.norm(proj_r - proj_l) < 1e-10:
            continue
        orig = lda.predict(x.reshape(1, -1))[0]
        swapped = x - proj_l + proj_r
        if lda.predict(swapped.reshape(1, -1))[0] != orig:
            flips += 1
        total += 1
    return float(flips / total) if total > 0 else None


def _iia_noise_injection(activity, ev_labels, ch_labels, V_ev, lda, rng):
    left_idx = np.where(ev_labels == 0)[0]
    right_idx = np.where(ev_labels == 1)[0]
    mean_left = np.mean(activity[left_idx] @ V_ev, axis=0)
    mean_right = np.mean(activity[right_idx] @ V_ev, axis=0)
    direction = mean_right - mean_left
    scale = np.linalg.norm(direction)
    if scale < 1e-10:
        return None
    direction = direction / scale

    n_test = min(len(left_idx), 100)
    test_idx = left_idx[rng.choice(len(left_idx), n_test, replace=False)]
    flips, total = 0, 0
    for ti in test_idx:
        x = activity[ti].copy()
        noise = V_ev @ (direction * scale * 2 + rng.normal(0, scale * 0.3, size=direction.shape))
        orig = lda.predict(x.reshape(1, -1))[0]
        if lda.predict((x + noise).reshape(1, -1))[0] != orig:
            flips += 1
        total += 1
    return float(flips / total) if total > 0 else None


def _iia_mean_shift(activity, ev_labels, ch_labels, V_ev, lda, rng):
    left_idx = np.where(ev_labels == 0)[0]
    right_idx = np.where(ev_labels == 1)[0]
    mean_left_proj = np.mean(activity[left_idx] @ V_ev, axis=0)
    mean_right_proj = np.mean(activity[right_idx] @ V_ev, axis=0)

    n_test = min(len(left_idx), 100)
    test_idx = left_idx[rng.choice(len(left_idx), n_test, replace=False)]
    flips, total = 0, 0
    for ti in test_idx:
        x = activity[ti].copy()
        current_proj = V_ev.T @ x
        shift = V_ev @ (mean_right_proj - current_proj)
        orig = lda.predict(x.reshape(1, -1))[0]
        if lda.predict((x + shift).reshape(1, -1))[0] != orig:
            flips += 1
        total += 1
    return float(flips / total) if total > 0 else None


def _iia_subspace_zeroing(activity, ev_labels, ch_labels, V_ev, lda, rng):
    n_test = min(len(activity), 200)
    test_idx = rng.choice(len(activity), n_test, replace=False)
    flips, total = 0, 0
    for ti in test_idx:
        x = activity[ti].copy()
        proj = V_ev @ (V_ev.T @ x)
        zeroed = x - proj
        orig = lda.predict(x.reshape(1, -1))[0]
        if lda.predict(zeroed.reshape(1, -1))[0] != orig:
            flips += 1
        total += 1
    return float(flips / total) if total > 0 else None


def run(max_sessions=None):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    sessions = load_all()
    if max_sessions:
        sessions = sessions[:max_sessions]

    methods = ["projection_swap", "noise_injection", "mean_shift", "subspace_zeroing"]
    method_fns = [_iia_projection_swap, _iia_noise_injection, _iia_mean_shift, _iia_subspace_zeroing]
    region_method_iia = {m: {} for m in methods}

    for sess_idx, sess in enumerate(tqdm(sessions, desc="Multi-method intervention")):
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

            lda = LinearDiscriminantAnalysis()
            try:
                lda.fit(act_v, ch_v)
            except Exception:
                continue

            rng = np.random.default_rng(42)
            for method_name, method_fn in zip(methods, method_fns):
                iia = method_fn(act_v, ev_v, ch_v, V_ev, lda, rng)
                if iia is not None:
                    region_method_iia[method_name].setdefault(region, []).append(iia)

    method_means = {}
    for m in methods:
        method_means[m] = {r: float(np.mean(v)) for r, v in region_method_iia[m].items()}

    matched_regions = sorted(
        set.intersection(*[set(method_means[m].keys()) for m in methods])
    )

    cross_method_correlations = {}
    for i, m1 in enumerate(methods):
        for m2 in methods[i+1:]:
            vals1 = [method_means[m1][r] for r in matched_regions]
            vals2 = [method_means[m2][r] for r in matched_regions]
            if len(vals1) >= 4:
                rho, p = spearmanr(vals1, vals2)
                cross_method_correlations[f"{m1}_vs_{m2}"] = {"rho": float(rho), "p": float(p)}

    all_rhos = [v["rho"] for v in cross_method_correlations.values()]
    mean_agreement = float(np.mean(all_rhos)) if all_rhos else None

    results = {
        "timestamp": datetime.now().isoformat(),
        "mechval_criterion": "E1 Intervention Reach / Multi-Method",
        "mechval_threshold": "cross-method rank correlation >= 0.7",
        "n_matched_regions": len(matched_regions),
        "matched_regions": matched_regions,
        "mean_cross_method_rho": mean_agreement,
        "intervention_reach_pass": mean_agreement >= 0.7 if mean_agreement else False,
        "cross_method_correlations": cross_method_correlations,
        "per_method": {m: {"n_regions": len(v), "mean_iia": float(np.mean(list(v.values()))) if v else None}
                       for m, v in method_means.items()},
    }

    out_path = RESULTS_DIR / "multi_method.json"
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
