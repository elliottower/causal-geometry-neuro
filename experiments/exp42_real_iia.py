"""Experiment 42: Real swap-based IIA for all regions.

Computes actual interchange intervention accuracy:
- Swap evidence projections between opposite-evidence trial pairs
- Measure choice classifier flip rate
- Compare against random-subspace and label-shuffle null distributions
- Cross-validated (split-half) IIA to address train/test overlap concern

Uses exp40's _compute_iia machinery but skips the cross-region analysis that crashed.
"""
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from tqdm import tqdm

from data.steinmetz import get_choice_labels, get_region_activity, list_regions, load_all

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent.parent / "artifacts" / "exp42"
MIN_NEURONS = 15
TIME_WINDOW = slice(15, 35)
SUBSPACE_DIM = 5
MIN_TRIALS_PER_CONDITION = 20
N_NULL_REPEATS = 100
N_CROSSVAL_SPLITS = 10


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


def _compute_iia(activity, evidence_labels, choice_labels, V_ev):
    left_idx = np.where(evidence_labels == 0)[0]
    right_idx = np.where(evidence_labels == 1)[0]
    if len(left_idx) < MIN_TRIALS_PER_CONDITION or len(right_idx) < MIN_TRIALS_PER_CONDITION:
        return None

    lda = LinearDiscriminantAnalysis()
    try:
        lda.fit(activity, choice_labels)
    except Exception:
        return None

    n_pairs = min(len(left_idx), len(right_idx), 100)
    left_sample = left_idx[np.random.choice(len(left_idx), n_pairs, replace=False)]
    right_sample = right_idx[np.random.choice(len(right_idx), n_pairs, replace=False)]

    flips = 0
    total = 0
    for li, ri in zip(left_sample, right_sample):
        act_l = activity[li].copy()
        act_r = activity[ri].copy()
        proj_l = V_ev @ (V_ev.T @ act_l)
        proj_r = V_ev @ (V_ev.T @ act_r)
        act_l_swapped = act_l - proj_l + proj_r
        act_r_swapped = act_r - proj_r + proj_l

        orig_pred_l = lda.predict(act_l.reshape(1, -1))[0]
        orig_pred_r = lda.predict(act_r.reshape(1, -1))[0]
        swap_pred_l = lda.predict(act_l_swapped.reshape(1, -1))[0]
        swap_pred_r = lda.predict(act_r_swapped.reshape(1, -1))[0]

        if swap_pred_l != orig_pred_l:
            flips += 1
        if swap_pred_r != orig_pred_r:
            flips += 1
        total += 2

    return float(flips / total) if total > 0 else None


def _iia_null_random_subspace(activity, evidence_labels, choice_labels, n_dims=SUBSPACE_DIM):
    n_neurons = activity.shape[1]
    k = min(n_dims, n_neurons - 1)
    if k < 1:
        return None
    null_iias = []
    for _ in range(N_NULL_REPEATS):
        V_rand = np.linalg.qr(np.random.randn(n_neurons, k))[0]
        iia = _compute_iia(activity, evidence_labels, choice_labels, V_rand)
        if iia is not None:
            null_iias.append(iia)
    if not null_iias:
        return None
    return null_iias


def _iia_null_label_shuffle(activity, evidence_labels, choice_labels, V_ev):
    if V_ev is None:
        return None
    null_iias = []
    for _ in range(N_NULL_REPEATS):
        shuffled_ev = evidence_labels.copy()
        np.random.shuffle(shuffled_ev)
        iia = _compute_iia(activity, shuffled_ev, choice_labels, V_ev)
        if iia is not None:
            null_iias.append(iia)
    if not null_iias:
        return None
    return null_iias


def _iia_crossval(activity, evidence_labels, choice_labels, n_dims=SUBSPACE_DIM):
    iias = []
    n = len(evidence_labels)
    for _ in range(N_CROSSVAL_SPLITS):
        perm = np.random.permutation(n)
        half = n // 2
        train_idx, test_idx = perm[:half], perm[half:]

        V_ev_train = _estimate_subspace(activity[train_idx], evidence_labels[train_idx], n_dims)
        if V_ev_train is None:
            continue

        lda = LinearDiscriminantAnalysis()
        try:
            lda.fit(activity[train_idx], choice_labels[train_idx])
        except Exception:
            continue

        test_ev = evidence_labels[test_idx]
        test_ch = choice_labels[test_idx]
        test_act = activity[test_idx]

        left_idx = np.where(test_ev == 0)[0]
        right_idx = np.where(test_ev == 1)[0]
        if len(left_idx) < 5 or len(right_idx) < 5:
            continue

        n_pairs = min(len(left_idx), len(right_idx))
        flips = 0
        total = 0
        for li, ri in zip(left_idx[:n_pairs], right_idx[:n_pairs]):
            act_l = test_act[li].copy()
            act_r = test_act[ri].copy()
            proj_l = V_ev_train @ (V_ev_train.T @ act_l)
            proj_r = V_ev_train @ (V_ev_train.T @ act_r)
            act_l_swapped = act_l - proj_l + proj_r
            act_r_swapped = act_r - proj_r + proj_l

            orig_l = lda.predict(act_l.reshape(1, -1))[0]
            orig_r = lda.predict(act_r.reshape(1, -1))[0]
            swap_l = lda.predict(act_l_swapped.reshape(1, -1))[0]
            swap_r = lda.predict(act_r_swapped.reshape(1, -1))[0]

            if swap_l != orig_l:
                flips += 1
            if swap_r != orig_r:
                flips += 1
            total += 2

        if total > 0:
            iias.append(flips / total)

    if not iias:
        return None
    return {"mean": float(np.mean(iias)), "std": float(np.std(iias)), "n_splits": len(iias)}


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
    jsonl_path = RESULTS_DIR / "iia_incremental.jsonl"

    computed_regions = set()
    if jsonl_path.exists():
        with open(jsonl_path) as f:
            for line in f:
                rec = json.loads(line)
                if "region" in rec:
                    computed_regions.add(rec["region"])
        logger.info(f"Resuming: {len(computed_regions)} regions already computed")

    sessions = load_all()
    if max_sessions:
        sessions = sessions[:max_sessions]

    region_data = {}
    for sess_idx, sess in enumerate(tqdm(sessions, desc="Loading sessions")):
        choice_labels = get_choice_labels(sess)
        if len(np.unique(choice_labels)) < 2:
            continue
        evidence_labels = _contrast_to_evidence_label(sess)
        if evidence_labels is None:
            continue
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
                "activity": activity[valid],
                "choice_labels": ch[valid],
                "evidence_labels": ev[valid],
            })

    region_results = {}
    for region in tqdm(sorted(region_data.keys()), desc="Computing IIA"):
        if region in computed_regions:
            continue

        measurements = region_data[region]
        iia_values = []
        null_random_all = []
        null_shuffle_all = []
        crossval_values = []
        alphas = []

        for m in measurements:
            act = m["activity"]
            ev = m["evidence_labels"]
            ch = m["choice_labels"]

            alpha = _power_law_exponent(act)
            if alpha is not None:
                alphas.append(alpha)

            V_ev = _estimate_subspace(act, ev)
            if V_ev is None:
                continue

            iia = _compute_iia(act, ev, ch, V_ev)
            if iia is not None:
                iia_values.append(iia)

            null_rand = _iia_null_random_subspace(act, ev, ch)
            if null_rand is not None:
                null_random_all.extend(null_rand)

            null_shuf = _iia_null_label_shuffle(act, ev, ch, V_ev)
            if null_shuf is not None:
                null_shuffle_all.extend(null_shuf)

            cv = _iia_crossval(act, ev, ch)
            if cv is not None:
                crossval_values.append(cv["mean"])

        if not iia_values:
            continue

        mean_iia = float(np.mean(iia_values))
        result = {
            "region": region,
            "mean_iia": mean_iia,
            "iia_per_session": iia_values,
            "n_sessions": len(measurements),
            "power_law_alpha": float(np.mean(alphas)) if alphas else None,
        }

        if null_random_all:
            stat, p = mannwhitneyu(iia_values, null_random_all, alternative="greater")
            result["null_random_mean"] = float(np.mean(null_random_all))
            result["null_random_std"] = float(np.std(null_random_all))
            result["vs_random_p"] = float(p)
            result["vs_random_u"] = float(stat)

        if null_shuffle_all:
            stat, p = mannwhitneyu(iia_values, null_shuffle_all, alternative="greater")
            result["null_shuffle_mean"] = float(np.mean(null_shuffle_all))
            result["null_shuffle_std"] = float(np.std(null_shuffle_all))
            result["vs_shuffle_p"] = float(p)
            result["vs_shuffle_u"] = float(stat)

        if crossval_values:
            result["crossval_iia_mean"] = float(np.mean(crossval_values))
            result["crossval_iia_std"] = float(np.std(crossval_values))

        region_results[region] = result

        with open(jsonl_path, "a") as f:
            f.write(json.dumps(result, default=str) + "\n")

    all_results = {}
    if jsonl_path.exists():
        with open(jsonl_path) as f:
            for line in f:
                rec = json.loads(line)
                all_results[rec["region"]] = rec

    alphas_for_corr = []
    iias_for_corr = []
    for r, v in all_results.items():
        if v.get("power_law_alpha") is not None and v.get("mean_iia") is not None:
            alphas_for_corr.append(v["power_law_alpha"])
            iias_for_corr.append(v["mean_iia"])

    summary = {
        "timestamp": datetime.now().isoformat(),
        "n_regions": len(all_results),
        "region_results": all_results,
    }

    if len(alphas_for_corr) >= 4:
        rho, p = spearmanr(alphas_for_corr, iias_for_corr)
        summary["alpha_vs_iia"] = {"rho": float(rho), "p": float(p), "n": len(alphas_for_corr)}

    regions_exceeding_random = sum(
        1 for v in all_results.values()
        if v.get("vs_random_p") is not None and v["vs_random_p"] < 0.05
    )
    regions_exceeding_shuffle = sum(
        1 for v in all_results.values()
        if v.get("vs_shuffle_p") is not None and v["vs_shuffle_p"] < 0.05
    )
    summary["regions_exceeding_random_null"] = regions_exceeding_random
    summary["regions_exceeding_shuffle_null"] = regions_exceeding_shuffle
    summary["total_regions_with_nulls"] = sum(1 for v in all_results.values() if v.get("vs_random_p") is not None)

    out_path = RESULTS_DIR / "real_iia.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"Saved to {out_path}")

    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-sessions", type=int, default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    run(max_sessions=args.max_sessions)
