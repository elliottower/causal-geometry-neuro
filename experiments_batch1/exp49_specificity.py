"""Experiment 49: Specificity — IIA on evidence axis vs control axes.

MECHVAL I3 (Specificity): "Effect selective; control-axis IIA ~ 0 while
causal-axis IIA high."

Tests whether the IIA effect is specific to the evidence axis or fires on
any axis. Computes IIA for:
  1. Evidence axis (contrast_right - contrast_left) — should be HIGH
  2. Reaction time axis (fast vs slow) — should be LOW
  3. Feedback axis (correct vs error) — should be LOW
  4. Random axis (random binary split) — should be ~0

If evidence IIA >> control IIA, the intervention is specific.
The on-task:off-task ratio ≥ 2:1 is the MECHVAL threshold for specificity.
"""
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from tqdm import tqdm

from data.steinmetz import get_choice_labels, get_region_activity, list_regions, load_all

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent.parent / "artifacts" / "exp49"
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


def _feedback_to_label(sess, n):
    fb = sess.get("feedback_type", np.array([]))
    fb = np.asarray(fb).ravel()
    if len(fb) < n:
        return None
    fb = fb[:n]
    labels = np.full(n, -1, dtype=int)
    labels[fb == 1] = 1
    labels[fb == -1] = 0
    if (labels == 0).sum() < MIN_TRIALS_PER_CONDITION or (labels == 1).sum() < MIN_TRIALS_PER_CONDITION:
        return None
    return labels


def _random_label(n, seed):
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 2, size=n)
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


def _compute_iia(activity, swap_labels, ch_labels, V_swap):
    idx_0 = np.where(swap_labels == 0)[0]
    idx_1 = np.where(swap_labels == 1)[0]
    if len(idx_0) < MIN_TRIALS_PER_CONDITION or len(idx_1) < MIN_TRIALS_PER_CONDITION:
        return None

    lda = LinearDiscriminantAnalysis()
    try:
        lda.fit(activity, ch_labels)
    except Exception:
        return None

    n_pairs = min(len(idx_0), len(idx_1), 50)
    rng = np.random.default_rng(42)
    sample_0 = idx_0[rng.choice(len(idx_0), n_pairs, replace=False)]
    sample_1 = idx_1[rng.choice(len(idx_1), n_pairs, replace=False)]

    flips = 0
    total = 0
    for i0, i1 in zip(sample_0, sample_1):
        x = activity[i0].copy()
        proj_0 = V_swap @ (V_swap.T @ activity[i0])
        proj_1 = V_swap @ (V_swap.T @ activity[i1])
        if np.linalg.norm(proj_1 - proj_0) < 1e-10:
            continue
        orig_pred = lda.predict(x.reshape(1, -1))[0]
        swapped = x - proj_0 + proj_1
        swap_pred = lda.predict(swapped.reshape(1, -1))[0]
        if swap_pred != orig_pred:
            flips += 1
        total += 1

    return float(flips / total) if total > 0 else None


def run(max_sessions=None):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    sessions = load_all()
    if max_sessions:
        sessions = sessions[:max_sessions]

    axes = ["evidence", "reaction_time", "feedback", "random"]
    region_iia = {ax: {} for ax in axes}

    for sess_idx, sess in enumerate(tqdm(sessions, desc="Specificity tests")):
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

            label_sets = {
                "evidence": ev,
                "reaction_time": _rt_to_label(sess, n),
                "feedback": _feedback_to_label(sess, n),
                "random": _random_label(n, seed=sess_idx * 1000 + hash(region) % 1000),
            }

            for ax_name, ax_labels in label_sets.items():
                if ax_labels is None:
                    continue
                valid = ax_labels >= 0
                if valid.sum() < MIN_TRIALS_PER_CONDITION * 2:
                    continue

                V = _estimate_subspace(activity[valid], ax_labels[valid])
                if V is None:
                    continue

                iia = _compute_iia(activity[valid], ax_labels[valid], ch[valid], V)
                if iia is not None:
                    region_iia[ax_name].setdefault(region, []).append(iia)

    summary = {}
    for ax_name in axes:
        ax_means = {r: float(np.mean(v)) for r, v in region_iia[ax_name].items()}
        summary[ax_name] = {
            "n_regions": len(ax_means),
            "mean_iia": float(np.mean(list(ax_means.values()))) if ax_means else None,
            "per_region": ax_means,
        }

    specificity_ratios = {}
    matched = sorted(set(region_iia["evidence"].keys()) & set(region_iia["random"].keys()))
    for region in matched:
        ev_iia = np.mean(region_iia["evidence"][region])
        rand_iia = np.mean(region_iia["random"][region])
        specificity_ratios[region] = float(ev_iia / rand_iia) if rand_iia > 0.01 else float("inf")

    ev_all = [np.mean(v) for v in region_iia["evidence"].values()]
    rand_all = [np.mean(v) for v in region_iia["random"].values() if region_iia["random"]]

    tests = {}
    if len(ev_all) >= 3 and len(rand_all) >= 3:
        u, p = mannwhitneyu(ev_all, rand_all, alternative="greater")
        tests["evidence_vs_random"] = {"U": float(u), "p": float(p)}

    rt_all = [np.mean(v) for v in region_iia["reaction_time"].values()] if region_iia["reaction_time"] else []
    if len(ev_all) >= 3 and len(rt_all) >= 3:
        u, p = mannwhitneyu(ev_all, rt_all, alternative="greater")
        tests["evidence_vs_rt"] = {"U": float(u), "p": float(p)}

    overall_ratio = float(np.mean(ev_all) / np.mean(rand_all)) if rand_all and np.mean(rand_all) > 0 else None

    results = {
        "timestamp": datetime.now().isoformat(),
        "mechval_criterion": "I3 Specificity",
        "mechval_threshold": "on-task:off-task ratio >= 2.0",
        "overall_specificity_ratio": overall_ratio,
        "specificity_pass": overall_ratio >= 2.0 if overall_ratio else False,
        "tests": tests,
        "per_axis": summary,
        "specificity_ratios": specificity_ratios,
    }

    out_path = RESULTS_DIR / "specificity.json"
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
