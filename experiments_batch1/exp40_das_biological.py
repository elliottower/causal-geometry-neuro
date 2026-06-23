"""Experiment 40: Full DAS-style causal abstraction on biological neural data.

Extends exp33 with:
1. Actual IIA number — swap evidence projections, measure classifier flip rate
2. Prior estimation — psychometric curve bias as second intervention variable
3. Cross-region causal graph — swap evidence in region A, measure choice shift in B
4. Geometric type stratification — IIA as step function across dimensionality regimes

This is the biological analog of Distributed Alignment Search (Geiger et al.):
- Natural do(evidence) intervention via contrast manipulation
- Natural do(prior) intervention via block-induced psychometric shift
- Grassmannian geometry as alignment constraint (prevents trivial abstraction)
"""
import json
import logging
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from tqdm import tqdm

from data.steinmetz import get_choice_labels, get_region_activity, list_regions, load_all
from geometry.distances import all_subspace_distances, grassmannian_distance

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent.parent / "artifacts" / "exp40"
MIN_NEURONS = 10
TIME_WINDOW = slice(15, 35)
SUBSPACE_DIM = 5
MIN_TRIALS_PER_CONDITION = 15
N_BOOTSTRAP = 1000
N_PERMUTATIONS = 500


def _bootstrap_spearman(x, y, n_boot=N_BOOTSTRAP, ci=0.95):
    """Bootstrap CI for Spearman rho."""
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
    """Permutation test for Spearman rho. Returns empirical p-value."""
    x, y = np.array(x), np.array(y)
    rho_obs = spearmanr(x, y)[0]
    count = 0
    for _ in range(n_perm):
        perm = np.random.permutation(len(y))
        if abs(spearmanr(x, y[perm])[0]) >= abs(rho_obs):
            count += 1
    return float((count + 1) / (n_perm + 1))


def _iia_null_random_subspace(activity, evidence_labels, choice_labels, n_dims=SUBSPACE_DIM, n_repeats=50):
    """Null distribution: IIA using random subspaces instead of V_ev."""
    left_idx = np.where(evidence_labels == 0)[0]
    right_idx = np.where(evidence_labels == 1)[0]
    if len(left_idx) < MIN_TRIALS_PER_CONDITION or len(right_idx) < MIN_TRIALS_PER_CONDITION:
        return None

    lda = LinearDiscriminantAnalysis()
    try:
        lda.fit(activity, choice_labels)
    except Exception:
        return None

    null_iias = []
    n_neurons = activity.shape[1]
    k = min(n_dims, n_neurons - 1)
    if k < 1:
        return None

    for _ in range(n_repeats):
        V_rand = np.linalg.qr(np.random.randn(n_neurons, k))[0]
        iia = _compute_iia(activity, evidence_labels, choice_labels, V_rand)
        if iia is not None:
            null_iias.append(iia)

    if not null_iias:
        return None
    return {
        "mean": float(np.mean(null_iias)),
        "std": float(np.std(null_iias)),
        "p95": float(np.percentile(null_iias, 95)),
        "p99": float(np.percentile(null_iias, 99)),
    }


def _iia_null_label_shuffle(activity, evidence_labels, choice_labels, V_ev, n_repeats=50):
    """Null distribution: IIA with shuffled evidence labels (breaks causal link)."""
    if V_ev is None:
        return None
    null_iias = []
    for _ in range(n_repeats):
        shuffled_ev = evidence_labels.copy()
        np.random.shuffle(shuffled_ev)
        iia = _compute_iia(activity, shuffled_ev, choice_labels, V_ev)
        if iia is not None:
            null_iias.append(iia)
    if not null_iias:
        return None
    return {
        "mean": float(np.mean(null_iias)),
        "std": float(np.std(null_iias)),
        "p95": float(np.percentile(null_iias, 95)),
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
    """LDA + PCA discriminative subspace. Returns (n_neurons, k) orthonormal basis."""
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
    """Actual interchange intervention accuracy.

    For each pair of trials with opposite evidence, swap their projections
    onto V_ev and check if an LDA classifier predicts the opposite choice.
    Returns IIA in [0, 1].
    """
    if V_ev is None:
        return None

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


def _compute_iia_crossval(activity, evidence_labels, choice_labels, n_dims=SUBSPACE_DIM, n_splits=10):
    """Split-half cross-validated IIA.

    Addresses the concern that V_ev and the choice classifier are estimated
    and evaluated on the same trials. For each split:
      1. Estimate V_ev on the train half via _estimate_subspace
      2. Train the choice LDA classifier on the train half
      3. Compute IIA on the held-out test half
    Returns the mean IIA across splits, or None if no split succeeded.
    """
    iias = []
    n = len(evidence_labels)
    for _ in range(n_splits):
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

        # IIA on test set
        test_ev = evidence_labels[test_idx]
        test_act = activity[test_idx]

        left = np.where(test_ev == 0)[0]
        right = np.where(test_ev == 1)[0]
        if len(left) < 5 or len(right) < 5:
            continue

        n_pairs = min(len(left), len(right), 50)
        left_sample = np.random.choice(left, n_pairs, replace=False)
        right_sample = np.random.choice(right, n_pairs, replace=False)

        flips = 0
        total = 0
        for li, ri in zip(left_sample, right_sample):
            orig_pred = lda.predict(test_act[li].reshape(1, -1))[0]

            proj_l = V_ev_train @ (V_ev_train.T @ test_act[li])
            proj_r = V_ev_train @ (V_ev_train.T @ test_act[ri])
            swapped = test_act[li] - proj_l + proj_r
            swap_pred = lda.predict(swapped.reshape(1, -1))[0]

            if swap_pred != orig_pred:
                flips += 1
            total += 1

        if total > 0:
            iias.append(flips / total)

    if not iias:
        return None
    return float(np.mean(iias))


def _estimate_prior_bias(sess):
    """Estimate session-level prior bias from psychometric curve.

    Fits logistic: P(right) = sigmoid(beta_contrast * evidence + beta_bias).
    Returns beta_bias as proxy for prior.
    """
    cl = sess.get("contrast_left", np.array([]))
    cr = sess.get("contrast_right", np.array([]))
    choices = get_choice_labels(sess)

    n = min(len(cl), len(cr), len(choices))
    if n < 30:
        return None

    evidence = cr[:n] - cl[:n]
    y = choices[:n]

    if len(np.unique(y)) < 2:
        return None

    def neg_log_lik(params):
        beta_ev, beta_bias = params
        logit = beta_ev * evidence + beta_bias
        logit = np.clip(logit, -20, 20)
        p = 1 / (1 + np.exp(-logit))
        ll = y * np.log(p + 1e-10) + (1 - y) * np.log(1 - p + 1e-10)
        return -np.sum(ll)

    try:
        result = minimize(neg_log_lik, [1.0, 0.0], method="Nelder-Mead")
        return float(result.x[1])
    except Exception:
        return None


def _estimate_prior_subspace(activity, prior_bias, trial_evidence, n_dims=SUBSPACE_DIM):
    """Estimate prior subspace from trials grouped by inferred prior.

    Split trials into prior-left (negative bias portion of session) and
    prior-right (positive bias portion) using a running estimate.
    """
    n = activity.shape[0]
    if n < 2 * MIN_TRIALS_PER_CONDITION:
        return None

    window = min(30, n // 3)
    running_bias = np.convolve(trial_evidence, np.ones(window) / window, mode="same")

    prior_labels = np.zeros(n, dtype=int)
    median_bias = np.median(running_bias)
    prior_labels[running_bias > median_bias] = 1
    prior_labels[running_bias <= median_bias] = 0

    if np.sum(prior_labels == 0) < MIN_TRIALS_PER_CONDITION or np.sum(prior_labels == 1) < MIN_TRIALS_PER_CONDITION:
        return None

    return _estimate_subspace(activity, prior_labels, n_dims)


def _cross_region_iia(activity_source, activity_target, evidence_labels, choice_labels_target):
    """Cross-region IIA: swap evidence in source, measure choice flip in target.

    Returns the fraction of trials where swapping the source region's evidence
    projection causes the target region's choice classifier to flip.
    """
    V_ev_source = _estimate_subspace(activity_source, evidence_labels)
    if V_ev_source is None:
        return None

    left_idx = np.where(evidence_labels == 0)[0]
    right_idx = np.where(evidence_labels == 1)[0]
    if len(left_idx) < MIN_TRIALS_PER_CONDITION or len(right_idx) < MIN_TRIALS_PER_CONDITION:
        return None

    lda_target = LinearDiscriminantAnalysis()
    try:
        lda_target.fit(activity_target, choice_labels_target)
    except Exception:
        return None

    n_source = activity_source.shape[1]
    n_target = activity_target.shape[1]
    min_dim = min(n_source, n_target)

    pca_s = PCA(n_components=min(10, min_dim - 1, activity_source.shape[0] - 1))
    pca_t = PCA(n_components=min(10, min_dim - 1, activity_target.shape[0] - 1))

    try:
        scores_s = pca_s.fit_transform(activity_source)
        scores_t = pca_t.fit_transform(activity_target)
    except Exception:
        return None

    n_cca = min(3, scores_s.shape[1], scores_t.shape[1], activity_source.shape[0] - 1)
    if n_cca < 1:
        return None

    try:
        cca = CCA(n_components=n_cca, max_iter=500)
        proj_s, proj_t = cca.fit_transform(scores_s, scores_t)
    except Exception:
        return None

    V_ev_cca = _estimate_subspace(proj_s, evidence_labels, min(3, n_cca))
    if V_ev_cca is None:
        return None

    n_pairs = min(len(left_idx), len(right_idx), 50)
    left_sample = left_idx[np.random.choice(len(left_idx), n_pairs, replace=False)]
    right_sample = right_idx[np.random.choice(len(right_idx), n_pairs, replace=False)]

    pinv_y = np.linalg.pinv(cca.y_weights_)

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

        orig_t_l_full = activity_target[li]
        orig_pred_l = lda_target.predict(orig_t_l_full.reshape(1, -1))[0]

        try:
            scores_t_l = scores_t[li].copy()
            scores_t_l_shifted = scores_t_l + delta_pca
            t_l_recon = pca_t.inverse_transform(scores_t_l_shifted.reshape(1, -1)).flatten()
            swap_pred_l = lda_target.predict(t_l_recon.reshape(1, -1))[0]
        except (ValueError, np.linalg.LinAlgError):
            continue

        if swap_pred_l != orig_pred_l:
            flips += 1
        total += 1

    return float(flips / total) if total > 0 else None


def _contrast_to_evidence_label(sess):
    cl = sess.get("contrast_left", np.array([]))
    cr = sess.get("contrast_right", np.array([]))
    if len(cl) == 0 or len(cr) == 0:
        return None, None
    n = sess["spks"].shape[2]
    n = min(n, len(cl), len(cr))
    cl, cr = cl[:n], cr[:n]
    evidence = cr - cl
    nonzero = evidence != 0
    if nonzero.sum() < MIN_TRIALS_PER_CONDITION:
        return None, None
    labels = np.full(n, -1, dtype=int)
    labels[evidence > 0] = 1
    labels[evidence < 0] = 0
    return labels, evidence


def run(max_sessions: int | None = None):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    sessions = load_all()
    if max_sessions:
        sessions = sessions[:max_sessions]

    logger.info(f"{datetime.now().isoformat()} Starting DAS-biological with {len(sessions)} sessions")

    region_data = {}
    session_meta = {}

    for sess_idx, sess in enumerate(tqdm(sessions, desc="Loading sessions")):
        choice_labels = get_choice_labels(sess)
        if len(np.unique(choice_labels)) < 2:
            continue
        evidence_labels, evidence_values = _contrast_to_evidence_label(sess)
        if evidence_labels is None:
            continue
        prior_bias = _estimate_prior_bias(sess)
        mouse = str(sess.get("mouse_name", f"mouse_{sess_idx}"))
        regions = list_regions(sess, min_neurons=MIN_NEURONS)

        session_meta[sess_idx] = {
            "mouse": mouse,
            "prior_bias": prior_bias,
            "n_trials": len(choice_labels),
        }

        for region in regions:
            act = get_region_activity(sess, region)
            if act is None or act.shape[1] < MIN_NEURONS:
                continue
            n = min(act.shape[0], len(choice_labels), len(evidence_labels))
            activity = act[:n, :, TIME_WINDOW].mean(axis=2)
            ch = choice_labels[:n]
            ev = evidence_labels[:n]
            ev_vals = evidence_values[:n] if evidence_values is not None else None

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
                "evidence_values": ev_vals[valid] if ev_vals is not None else None,
                "prior_bias": prior_bias,
                "n_neurons": activity.shape[1],
            })

    logger.info(f"{datetime.now().isoformat()} {len(region_data)} regions loaded")

    # --- Part 1: Per-region IIA, null distributions, and prior analysis ---
    region_results = {}
    alpha_list, iia_list = [], []

    jsonl_path = RESULTS_DIR / "das_biological_incremental.jsonl"
    computed_regions = set()
    if jsonl_path.exists():
        with open(jsonl_path) as jf:
            for line in jf:
                r = json.loads(line)
                computed_regions.add(r["region"])
                region_results[r["region"]] = r
        logger.info(f"Resuming: loaded {len(computed_regions)} pre-computed regions")

    for region, measurements in tqdm(region_data.items(), desc="Per-region DAS"):
        if region in computed_regions:
            rr = region_results[region]
            if rr.get("power_law_alpha") is not None and rr.get("mean_iia") is not None:
                alpha_list.append(rr["power_law_alpha"])
                iia_list.append(rr["mean_iia"])
            continue
        iia_scores = []
        iia_crossval_scores = []
        null_random_sub = []
        null_label_shuf = []
        grass_ev_ch = []
        grass_ev_prior = []

        for m in measurements:
            V_ev = _estimate_subspace(m["activity"], m["evidence_labels"])
            V_ch = _estimate_subspace(m["activity"], m["choice_labels"])

            iia = _compute_iia(m["activity"], m["evidence_labels"], m["choice_labels"], V_ev)
            if iia is not None:
                iia_scores.append(iia)

            iia_cv = _compute_iia_crossval(m["activity"], m["evidence_labels"], m["choice_labels"])
            if iia_cv is not None:
                iia_crossval_scores.append(iia_cv)

            null_rand = _iia_null_random_subspace(
                m["activity"], m["evidence_labels"], m["choice_labels"])
            if null_rand is not None:
                null_random_sub.append(null_rand["mean"])

            null_shuf = _iia_null_label_shuffle(
                m["activity"], m["evidence_labels"], m["choice_labels"], V_ev)
            if null_shuf is not None:
                null_label_shuf.append(null_shuf["mean"])

            if V_ev is not None and V_ch is not None:
                k = min(V_ev.shape[1], V_ch.shape[1])
                try:
                    dists = all_subspace_distances(V_ev[:, :k], V_ch[:, :k])
                    grass_ev_ch.append(dists)
                except Exception:
                    pass

            if m["evidence_values"] is not None and m["prior_bias"] is not None:
                V_prior = _estimate_prior_subspace(
                    m["activity"], m["prior_bias"], m["evidence_values"])
                if V_prior is not None and V_ev is not None:
                    k = min(V_ev.shape[1], V_prior.shape[1])
                    try:
                        dists = all_subspace_distances(V_ev[:, :k], V_prior[:, :k])
                        grass_ev_prior.append(dists)
                    except Exception:
                        pass

        alphas = [_power_law_exponent(m["activity"]) for m in measurements]
        alphas = [a for a in alphas if a is not None]
        alpha = float(np.mean(alphas)) if alphas else None

        mean_iia = float(np.mean(iia_scores)) if iia_scores else None
        mean_iia_crossval = float(np.mean(iia_crossval_scores)) if iia_crossval_scores else None
        mean_null_rand = float(np.mean(null_random_sub)) if null_random_sub else None
        mean_null_shuf = float(np.mean(null_label_shuf)) if null_label_shuf else None
        iia_above_null = None
        if mean_iia is not None and mean_null_rand is not None:
            iia_above_null = float(mean_iia - mean_null_rand)

        region_results[region] = {
            "n_sessions": len(measurements),
            "power_law_alpha": alpha,
            "mean_iia": mean_iia,
            "iia_crossval": mean_iia_crossval,
            "std_iia": float(np.std(iia_scores)) if len(iia_scores) > 1 else None,
            "n_iia_sessions": len(iia_scores),
            "null_random_subspace_iia": mean_null_rand,
            "null_label_shuffle_iia": mean_null_shuf,
            "iia_above_null": iia_above_null,
            "ev_ch_distances": {
                metric: float(np.mean([d[metric] for d in grass_ev_ch]))
                for metric in ["grassmannian", "chordal", "mean_principal_angle_deg", "subspace_overlap"]
            } if grass_ev_ch else None,
            "ev_prior_distances": {
                metric: float(np.mean([d[metric] for d in grass_ev_prior]))
                for metric in ["grassmannian", "chordal", "mean_principal_angle_deg", "subspace_overlap"]
            } if grass_ev_prior else None,
        }

        if alpha is not None and iia_scores:
            alpha_list.append(alpha)
            iia_list.append(np.mean(iia_scores))

        with open(jsonl_path, "a") as jf:
            jf.write(json.dumps({"region": region, **region_results[region]}, default=str) + "\n")

    # --- Part 2: Prediction tests with bootstrap CIs and permutation nulls ---
    prediction_tests = {}

    # IIA aggregate null: across all regions, is real IIA > null?
    all_real_iia = [v["mean_iia"] for v in region_results.values() if v["mean_iia"] is not None]
    all_null_iia = [v["null_random_subspace_iia"] for v in region_results.values() if v["null_random_subspace_iia"] is not None]
    if all_real_iia and all_null_iia:
        from scipy.stats import mannwhitneyu
        try:
            u_stat, u_p = mannwhitneyu(all_real_iia, all_null_iia, alternative="greater")
        except Exception:
            u_stat, u_p = None, None
        prediction_tests["iia_vs_random_subspace_null"] = {
            "real_iia_mean": float(np.mean(all_real_iia)),
            "null_iia_mean": float(np.mean(all_null_iia)),
            "effect_size": float(np.mean(all_real_iia) - np.mean(all_null_iia)),
            "mann_whitney_U": float(u_stat) if u_stat is not None else None,
            "p_one_sided": float(u_p) if u_p is not None else None,
            "n_regions_real": len(all_real_iia),
            "n_regions_null": len(all_null_iia),
            "interpretation": "Real IIA should be significantly above random-subspace null (p < 0.05).",
        }

    if len(alpha_list) >= 5:
        boot = _bootstrap_spearman(alpha_list, iia_list)
        perm_p = _permutation_test_spearman(alpha_list, iia_list)
        prediction_tests["alpha_vs_iia"] = {
            **boot,
            "permutation_p": perm_p,
            "interpretation": (
                "Positive rho = low-dim (high-alpha) regions have higher IIA. "
                "Bootstrap CI and permutation p provide non-parametric guarantees."
            ),
        }

    # Stratify by geometric type with Kruskal-Wallis test
    if len(alpha_list) >= 9:
        sorted_pairs = sorted(zip(alpha_list, iia_list))
        n = len(sorted_pairs)
        tercile = n // 3
        low_alpha = [iia for _, iia in sorted_pairs[:tercile]]
        mid_alpha = [iia for _, iia in sorted_pairs[tercile:2*tercile]]
        high_alpha = [iia for _, iia in sorted_pairs[2*tercile:]]

        from scipy.stats import kruskal
        try:
            kw_stat, kw_p = kruskal(low_alpha, mid_alpha, high_alpha)
        except Exception:
            kw_stat, kw_p = None, None

        prediction_tests["iia_by_geometric_type"] = {
            "low_alpha_mean_iia": float(np.mean(low_alpha)),
            "mid_alpha_mean_iia": float(np.mean(mid_alpha)),
            "high_alpha_mean_iia": float(np.mean(high_alpha)),
            "low_alpha_std": float(np.std(low_alpha)),
            "mid_alpha_std": float(np.std(mid_alpha)),
            "high_alpha_std": float(np.std(high_alpha)),
            "low_alpha_n": len(low_alpha),
            "mid_alpha_n": len(mid_alpha),
            "high_alpha_n": len(high_alpha),
            "low_alpha_range": [float(sorted_pairs[0][0]), float(sorted_pairs[tercile-1][0])],
            "high_alpha_range": [float(sorted_pairs[2*tercile][0]), float(sorted_pairs[-1][0])],
            "kruskal_wallis_H": float(kw_stat) if kw_stat is not None else None,
            "kruskal_wallis_p": float(kw_p) if kw_p is not None else None,
            "interpretation": "Step function: high-alpha regions should have IIA > 0.7, low-alpha < 0.4. Kruskal-Wallis tests group differences.",
        }

    # Prior analysis with bootstrap CI
    regions_with_prior = [(r, v) for r, v in region_results.items()
                          if v["ev_prior_distances"] is not None and v["power_law_alpha"] is not None]
    if len(regions_with_prior) >= 5:
        alphas_p = [v["power_law_alpha"] for _, v in regions_with_prior]
        grass_p = [v["ev_prior_distances"]["grassmannian"] for _, v in regions_with_prior]
        boot = _bootstrap_spearman(alphas_p, grass_p)
        perm_p = _permutation_test_spearman(alphas_p, grass_p)
        prediction_tests["alpha_vs_ev_prior_separation"] = {
            **boot,
            "permutation_p": perm_p,
            "interpretation": (
                "Negative rho = high-dim regions keep evidence and prior in separate subspaces. "
                "Positive rho = low-dim regions merge evidence and prior."
            ),
        }

    # IIA-above-null vs alpha: does the EXCESS IIA (above random subspace) still correlate?
    alpha_excess, iia_excess = [], []
    for r, v in region_results.items():
        if v["power_law_alpha"] is not None and v["iia_above_null"] is not None:
            alpha_excess.append(v["power_law_alpha"])
            iia_excess.append(v["iia_above_null"])
    if len(alpha_excess) >= 5:
        boot = _bootstrap_spearman(alpha_excess, iia_excess)
        prediction_tests["alpha_vs_excess_iia"] = {
            **boot,
            "interpretation": "Correlation with IIA minus random-subspace null. Ensures result isn't a dimensionality artifact.",
        }

    # --- Part 3: Cross-region causal graph (top 15 regions by session count) ---
    logger.info(f"{datetime.now().isoformat()} Computing cross-region causal graph")
    top_regions = sorted(region_data.keys(),
                         key=lambda r: len(region_data[r]), reverse=True)[:15]

    causal_graph = {}
    for source_region in tqdm(top_regions, desc="Cross-region IIA"):
        for target_region in top_regions:
            if source_region == target_region:
                continue

            cross_iias = []
            for m_src in region_data[source_region]:
                for m_tgt in region_data[target_region]:
                    if m_src["session_idx"] != m_tgt["session_idx"]:
                        continue
                    iia = _cross_region_iia(
                        m_src["activity"], m_tgt["activity"],
                        m_src["evidence_labels"], m_tgt["choice_labels"])
                    if iia is not None:
                        cross_iias.append(iia)

            if cross_iias:
                key = f"{source_region}_to_{target_region}"
                causal_graph[key] = {
                    "source": source_region,
                    "target": target_region,
                    "mean_cross_iia": float(np.mean(cross_iias)),
                    "n_sessions": len(cross_iias),
                    "source_alpha": region_results[source_region]["power_law_alpha"],
                    "target_alpha": region_results[target_region]["power_law_alpha"],
                }

    # Causal graph statistics
    top_edges = sorted(causal_graph.values(), key=lambda x: x["mean_cross_iia"], reverse=True)[:20]

    # Asymmetry test: is A→B different from B→A? (directional causal graph)
    asymmetry_scores = []
    for source_region in top_regions:
        for target_region in top_regions:
            if source_region >= target_region:
                continue
            fwd_key = f"{source_region}_to_{target_region}"
            rev_key = f"{target_region}_to_{source_region}"
            if fwd_key in causal_graph and rev_key in causal_graph:
                fwd = causal_graph[fwd_key]["mean_cross_iia"]
                rev = causal_graph[rev_key]["mean_cross_iia"]
                asymmetry_scores.append(abs(fwd - rev))

    causal_graph_stats = {}
    if asymmetry_scores:
        causal_graph_stats["mean_asymmetry"] = float(np.mean(asymmetry_scores))
        causal_graph_stats["max_asymmetry"] = float(np.max(asymmetry_scores))
        causal_graph_stats["n_pairs"] = len(asymmetry_scores)
        causal_graph_stats["interpretation"] = (
            "Mean asymmetry > 0 means the causal graph is directional, not symmetric. "
            "High asymmetry implies genuine causal flow, not just shared information."
        )

    # Bootstrap CI on mean IIA of top edges vs bottom edges
    if len(top_edges) >= 5:
        top5_iias = [e["mean_cross_iia"] for e in top_edges[:5]]
        all_edge_iias = [e["mean_cross_iia"] for e in causal_graph.values()]
        boot_means = []
        for _ in range(N_BOOTSTRAP):
            sample = np.random.choice(all_edge_iias, len(top5_iias), replace=True)
            boot_means.append(float(np.mean(sample)))
        causal_graph_stats["top5_mean_cross_iia"] = float(np.mean(top5_iias))
        causal_graph_stats["population_mean_cross_iia"] = float(np.mean(all_edge_iias))
        causal_graph_stats["bootstrap_p_top5_above_mean"] = float(
            np.mean([b >= np.mean(top5_iias) for b in boot_means]))

    results = {
        "timestamp": datetime.now().isoformat(),
        "n_sessions": len(sessions),
        "n_regions": len(region_data),
        "n_regions_analyzed": len(region_results),
        "region_results": region_results,
        "prediction_tests": prediction_tests,
        "causal_graph_n_edges": len(causal_graph),
        "causal_graph": causal_graph,
        "causal_graph_stats": causal_graph_stats,
        "top_causal_edges": top_edges,
        "top_iia_regions": sorted(
            [(r, v["mean_iia"]) for r, v in region_results.items() if v["mean_iia"] is not None],
            key=lambda x: x[1], reverse=True)[:10],
        "bottom_iia_regions": sorted(
            [(r, v["mean_iia"]) for r, v in region_results.items() if v["mean_iia"] is not None],
            key=lambda x: x[1])[:10],
    }

    out_path = RESULTS_DIR / "das_biological.json"
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
