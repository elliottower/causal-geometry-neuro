"""Experiment 61: Engagement subspace from pre-stimulus neural activity.

Derives the engagement index from pre-stimulus population activity (Steinmetz
et al. 2019, PMC6913580) and tests whether there is a low-dimensional engagement
subspace that is orthogonal to or overlapping with the choice subspace.

Engagement labels:
- Disengaged: NoGo responses on high-contrast (easy) trials where the mouse
  should respond but does not, OR very long reaction times (>2s).
- Engaged: correct responses on medium/high contrast trials with reasonable
  reaction times.

For each brain region:
1. Fit an engagement subspace from PRE-stimulus activity (0-150ms) using LDA.
2. Fit a choice subspace from POST-stimulus activity (250-450ms) using LDA.
3. Fit both using a structured VAE.
4. Compute Grassmannian distance between engagement and choice subspaces.
5. Compute IIA for the engagement subspace: swap engagement projections between
   engaged/disengaged pairs, measure whether the choice classifier prediction
   changes.

Key questions:
- Is the engagement subspace orthogonal to choice? (independent coding dimensions)
- Does the engagement subspace predict trial outcome? (engaged -> better decoding)
- Does the VAE find engagement subspaces that LDA misses?
"""
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from scipy.stats import mannwhitneyu, spearmanr, wilcoxon
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from tqdm import tqdm

from data.steinmetz import get_choice_labels, get_region_activity, list_regions, load_all
from experiments.exp57_structured_vae import (
    _iia_null_random_subspace,
    _power_law_exponent,
    train_vae,
)
from geometry.distances import grassmannian_distance

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent.parent / "artifacts" / "exp61"
MIN_NEURONS = 15
MIN_TRIALS_PER_CONDITION = 20

# Time windows (10ms bins; stimulus onset ~ bin 25)
TIME_WINDOW_PRE = slice(0, 15)    # 0-150ms: pre-stimulus
TIME_WINDOW_POST = slice(25, 45)  # 250-450ms: post-stimulus (0-200ms after stim)

# VAE hyperparameters (same as exp57)
Z_ENGAGE_DIM = 3
Z_CHOICE_DIM = 3
Z_OTHER_DIM = 15
HIDDEN_DIM = 128
N_EPOCHS = 300
BATCH_SIZE = 64
LR = 1e-3
BETA_KL = 1.0
ALPHA_SUPERVISED = 10.0

# Reaction time threshold for disengagement (seconds)
RT_DISENGAGE_THRESHOLD = 2.0


# ---------------------------------------------------------------------------
# Engagement label derivation
# ---------------------------------------------------------------------------

def _get_engagement_labels(sess: dict) -> np.ndarray:
    """Derive engagement labels from behavioral signals.

    Disengaged (0): NoGo on easy trials (high contrast, should respond but
    does not), or very long reaction times (>2s) on easy trials.
    Engaged (1): correct responses on medium/high contrast trials with
    reasonable reaction times.
    Excluded (-1): everything else.

    Returns (n_trials,) array with values in {-1, 0, 1}.
    """
    choice = sess.get("response", np.array([]))
    cl = sess.get("contrast_left", np.array([]))
    cr = sess.get("contrast_right", np.array([]))
    fb = sess.get("feedback_type", np.array([]))
    rt = sess.get("response_time", np.array([]))

    if len(choice) == 0 or len(cl) == 0 or len(cr) == 0:
        return np.array([])

    n = sess["spks"].shape[2]
    n = min(n, len(choice), len(cl), len(cr))
    choice = np.asarray(choice).ravel()[:n]
    cl = np.asarray(cl).ravel()[:n].astype(float)
    cr = np.asarray(cr).ravel()[:n].astype(float)
    fb = np.asarray(fb).ravel()[:n] if len(fb) >= n else np.zeros(n)
    rt = np.asarray(rt).ravel()[:n] if len(rt) >= n else np.full(n, np.nan)

    max_contrast = np.maximum(cl, cr)
    easy = max_contrast >= 0.5
    nogo = choice == 0

    labels = np.full(n, -1, dtype=int)

    # Disengaged: NoGo on easy trials
    labels[easy & nogo] = 0

    # Disengaged: very long RT on easy trials (if RT available)
    if not np.all(np.isnan(rt)):
        labels[easy & (rt > RT_DISENGAGE_THRESHOLD)] = 0

    # Engaged: correct responses on medium/high contrast trials
    medium_or_high = max_contrast >= 0.25
    correct = fb == 1
    responded = choice != 0
    reasonable_rt = np.isnan(rt) | (rt <= RT_DISENGAGE_THRESHOLD)
    labels[medium_or_high & correct & responded & reasonable_rt] = 1

    return labels


# ---------------------------------------------------------------------------
# Subspace estimation
# ---------------------------------------------------------------------------

def _estimate_lda_subspace(activity: np.ndarray, labels: np.ndarray,
                           n_dims: int = 3) -> np.ndarray | None:
    """LDA+PCA subspace estimation (same protocol as exp57/exp58)."""
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


# ---------------------------------------------------------------------------
# IIA: swap engagement projections, measure choice prediction change
# ---------------------------------------------------------------------------

def _compute_engagement_iia(
    pre_activity: np.ndarray,
    post_activity: np.ndarray,
    engagement_labels: np.ndarray,
    choice_labels: np.ndarray,
    V_engage: np.ndarray,
) -> float | None:
    """IIA for engagement subspace.

    Swap the engagement-subspace projection of PRE-stimulus activity between
    engaged/disengaged trial pairs, concatenate with (unswapped) POST-stimulus
    activity, and measure whether a choice classifier trained on the combined
    representation changes its prediction.

    Since engagement is a pre-stimulus state, the causal question is whether
    swapping the engagement projection changes downstream choice decoding.
    """
    engaged_idx = np.where(engagement_labels == 1)[0]
    disengaged_idx = np.where(engagement_labels == 0)[0]
    if len(engaged_idx) < MIN_TRIALS_PER_CONDITION or len(disengaged_idx) < MIN_TRIALS_PER_CONDITION:
        return None

    # Train choice classifier on post-stimulus activity
    lda = LinearDiscriminantAnalysis()
    try:
        lda.fit(post_activity, choice_labels)
    except Exception:
        return None

    n_pairs = min(len(engaged_idx), len(disengaged_idx), 100)
    sample_eng = engaged_idx[np.random.choice(len(engaged_idx), n_pairs, replace=False)]
    sample_dis = disengaged_idx[np.random.choice(len(disengaged_idx), n_pairs, replace=False)]

    flips = 0
    total = 0
    for ie, id_ in zip(sample_eng, sample_dis):
        act_eng = post_activity[ie].copy()
        act_dis = post_activity[id_].copy()

        # Compute engagement-subspace projections from PRE-stimulus
        pre_eng = pre_activity[ie]
        pre_dis = pre_activity[id_]
        proj_eng = V_engage @ (V_engage.T @ pre_eng)
        proj_dis = V_engage @ (V_engage.T @ pre_dis)

        # Swap engagement projections onto POST-stimulus activity
        # The idea: the pre-stimulus engagement state biases post-stimulus
        # representations. We transfer that bias.
        act_eng_swapped = act_eng - proj_eng + proj_dis
        act_dis_swapped = act_dis - proj_dis + proj_eng

        orig_pred_eng = lda.predict(act_eng.reshape(1, -1))[0]
        orig_pred_dis = lda.predict(act_dis.reshape(1, -1))[0]
        swap_pred_eng = lda.predict(act_eng_swapped.reshape(1, -1))[0]
        swap_pred_dis = lda.predict(act_dis_swapped.reshape(1, -1))[0]

        if swap_pred_eng != orig_pred_eng:
            flips += 1
        if swap_pred_dis != orig_pred_dis:
            flips += 1
        total += 2

    return float(flips / total) if total > 0 else None


def _compute_self_iia(activity: np.ndarray, labels: np.ndarray,
                      V: np.ndarray) -> float | None:
    """Standard self-IIA: swap subspace projections between opposite-label pairs,
    measure prediction flip rate (same as exp57/exp58 protocol)."""
    group0 = np.where(labels == 0)[0]
    group1 = np.where(labels == 1)[0]
    if len(group0) < MIN_TRIALS_PER_CONDITION or len(group1) < MIN_TRIALS_PER_CONDITION:
        return None

    lda = LinearDiscriminantAnalysis()
    try:
        lda.fit(activity, labels)
    except Exception:
        return None

    n_pairs = min(len(group0), len(group1), 100)
    sample0 = group0[np.random.choice(len(group0), n_pairs, replace=False)]
    sample1 = group1[np.random.choice(len(group1), n_pairs, replace=False)]

    flips = 0
    total = 0
    for i0, i1 in zip(sample0, sample1):
        act0 = activity[i0].copy()
        act1 = activity[i1].copy()
        proj0 = V @ (V.T @ act0)
        proj1 = V @ (V.T @ act1)
        act0_swapped = act0 - proj0 + proj1
        act1_swapped = act1 - proj1 + proj0

        orig_pred0 = lda.predict(act0.reshape(1, -1))[0]
        orig_pred1 = lda.predict(act1.reshape(1, -1))[0]
        swap_pred0 = lda.predict(act0_swapped.reshape(1, -1))[0]
        swap_pred1 = lda.predict(act1_swapped.reshape(1, -1))[0]

        if swap_pred0 != orig_pred0:
            flips += 1
        if swap_pred1 != orig_pred1:
            flips += 1
        total += 2

    return float(flips / total) if total > 0 else None


# ---------------------------------------------------------------------------
# Choice decoding quality by engagement state
# ---------------------------------------------------------------------------

def _choice_accuracy_by_engagement(
    post_activity: np.ndarray,
    choice_labels: np.ndarray,
    engagement_labels: np.ndarray,
) -> dict | None:
    """Compare choice decoding accuracy on engaged vs disengaged trials."""
    engaged = engagement_labels == 1
    disengaged = engagement_labels == 0

    if engaged.sum() < MIN_TRIALS_PER_CONDITION * 2 or disengaged.sum() < MIN_TRIALS_PER_CONDITION * 2:
        return None

    # Overall LDA
    lda = LinearDiscriminantAnalysis()
    try:
        lda.fit(post_activity, choice_labels)
    except Exception:
        return None

    preds = lda.predict(post_activity)
    acc_engaged = float(np.mean(preds[engaged] == choice_labels[engaged]))
    acc_disengaged = float(np.mean(preds[disengaged] == choice_labels[disengaged]))

    return {
        "acc_engaged": acc_engaged,
        "acc_disengaged": acc_disengaged,
        "acc_diff": acc_engaged - acc_disengaged,
        "n_engaged": int(engaged.sum()),
        "n_disengaged": int(disengaged.sum()),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(max_sessions: int | None = None) -> dict:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    sessions = load_all()
    if max_sessions:
        sessions = sessions[:max_sessions]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"{datetime.now().isoformat()} Starting engagement subspace experiment "
                f"with {len(sessions)} sessions on {device}")

    # --- Load data per region ---
    region_data: dict[str, list[dict]] = {}

    for sess_idx, sess in enumerate(tqdm(sessions, desc="Loading sessions")):
        choice_labels_full = get_choice_labels(sess)
        if len(np.unique(choice_labels_full)) < 2:
            continue

        engagement_labels_full = _get_engagement_labels(sess)
        if len(engagement_labels_full) == 0:
            continue
        n_engage_valid = np.sum(engagement_labels_full >= 0)
        if n_engage_valid < MIN_TRIALS_PER_CONDITION * 2:
            continue

        mouse = str(sess.get("mouse_name", f"mouse_{sess_idx}"))
        regions = list_regions(sess, min_neurons=MIN_NEURONS)

        for region in regions:
            act_raw = get_region_activity(sess, region)
            if act_raw is None or act_raw.shape[1] < MIN_NEURONS:
                continue

            n = min(act_raw.shape[0], len(choice_labels_full), len(engagement_labels_full))
            pre_activity = act_raw[:n, :, TIME_WINDOW_PRE].mean(axis=2)
            post_activity = act_raw[:n, :, TIME_WINDOW_POST].mean(axis=2)
            ch = choice_labels_full[:n]
            eng = engagement_labels_full[:n]

            # Need both engagement labels and choice labels to be valid
            valid = eng >= 0
            if valid.sum() < MIN_TRIALS_PER_CONDITION * 2:
                continue
            n_engaged = np.sum(eng[valid] == 1)
            n_disengaged = np.sum(eng[valid] == 0)
            if n_engaged < MIN_TRIALS_PER_CONDITION or n_disengaged < MIN_TRIALS_PER_CONDITION:
                continue

            if region not in region_data:
                region_data[region] = []
            region_data[region].append({
                "session_idx": sess_idx,
                "mouse": mouse,
                "pre_activity": pre_activity[valid],
                "post_activity": post_activity[valid],
                "choice_labels": ch[valid],
                "engagement_labels": eng[valid],
                "n_neurons": int(pre_activity.shape[1]),
            })

    logger.info(f"{datetime.now().isoformat()} {len(region_data)} regions loaded with "
                f"sufficient engagement/choice data")

    # --- Per-region analysis ---
    region_results: dict[str, dict] = {}
    jsonl_path = RESULTS_DIR / "engagement_incremental.jsonl"

    # Resume support
    computed_regions: set[str] = set()
    if jsonl_path.exists():
        with open(jsonl_path) as jf:
            for line in jf:
                r = json.loads(line)
                computed_regions.add(r["region"])
                region_results[r["region"]] = r
        logger.info(f"Resuming: loaded {len(computed_regions)} pre-computed regions")

    for region, measurements in tqdm(region_data.items(), desc="Engagement analysis"):
        if region in computed_regions:
            continue

        lda_engage_iias = []
        lda_choice_iias = []
        vae_engage_iias = []
        vae_choice_iias = []
        cross_iias = []  # engagement subspace swap -> choice prediction
        grassmann_lda = []
        grassmann_vae = []
        choice_by_engage = []
        null_engage_iias = []

        for m in measurements:
            pre_act = m["pre_activity"]
            post_act = m["post_activity"]
            ch = m["choice_labels"]
            eng = m["engagement_labels"]
            n_neurons = pre_act.shape[1]

            z_eng = min(Z_ENGAGE_DIM, n_neurons // 5, n_neurons - 1)
            z_ch = min(Z_CHOICE_DIM, n_neurons // 5, n_neurons - 1)
            z_other = min(Z_OTHER_DIM, n_neurons // 3, n_neurons - max(z_eng, z_ch) - 1)
            if z_eng < 1 or z_ch < 1 or z_other < 1:
                continue
            hidden = min(HIDDEN_DIM, n_neurons * 2)

            # --- LDA subspaces ---
            V_engage_lda = _estimate_lda_subspace(pre_act, eng, n_dims=z_eng)
            V_choice_lda = _estimate_lda_subspace(post_act, ch, n_dims=z_ch)

            if V_engage_lda is not None:
                # Self-IIA on engagement (pre-stimulus)
                iia_eng = _compute_self_iia(pre_act, eng, V_engage_lda)
                if iia_eng is not None:
                    lda_engage_iias.append(iia_eng)

                # Cross-IIA: swap engagement projection, measure choice flip
                if V_engage_lda.shape[0] == post_act.shape[1]:
                    cross = _compute_engagement_iia(pre_act, post_act, eng, ch, V_engage_lda)
                    if cross is not None:
                        cross_iias.append(cross)

                # Null: random engagement subspace
                null_iias = _iia_null_random_subspace(pre_act, eng, eng,
                                                      n_dims=z_eng, n_repeats=50)
                if null_iias is not None:
                    null_engage_iias.extend(null_iias)

            if V_choice_lda is not None:
                # Self-IIA on choice (post-stimulus)
                iia_ch = _compute_self_iia(post_act, ch, V_choice_lda)
                if iia_ch is not None:
                    lda_choice_iias.append(iia_ch)

            # Grassmannian distance between LDA subspaces
            if V_engage_lda is not None and V_choice_lda is not None:
                k = min(V_engage_lda.shape[1], V_choice_lda.shape[1])
                try:
                    d = grassmannian_distance(V_engage_lda[:, :k], V_choice_lda[:, :k])
                    grassmann_lda.append(d)
                except Exception:
                    pass

            # --- VAE subspaces ---
            try:
                vae_eng_result = train_vae(
                    pre_act, eng,
                    z_choice_dim=z_eng, z_other_dim=z_other,
                    hidden_dim=hidden, device=device,
                )
                V_engage_vae = vae_eng_result["subspace_directions"]
                iia_eng_vae = _compute_self_iia(pre_act, eng, V_engage_vae)
                if iia_eng_vae is not None:
                    vae_engage_iias.append(iia_eng_vae)
            except Exception as e:
                logger.warning(f"VAE engagement failed for {region} sess "
                               f"{m['session_idx']}: {e}")
                V_engage_vae = None

            try:
                vae_ch_result = train_vae(
                    post_act, ch,
                    z_choice_dim=z_ch, z_other_dim=z_other,
                    hidden_dim=hidden, device=device,
                )
                V_choice_vae = vae_ch_result["subspace_directions"]
                iia_ch_vae = _compute_self_iia(post_act, ch, V_choice_vae)
                if iia_ch_vae is not None:
                    vae_choice_iias.append(iia_ch_vae)
            except Exception as e:
                logger.warning(f"VAE choice failed for {region} sess "
                               f"{m['session_idx']}: {e}")
                V_choice_vae = None

            # Grassmannian distance between VAE subspaces
            if V_engage_vae is not None and V_choice_vae is not None:
                k = min(V_engage_vae.shape[1], V_choice_vae.shape[1])
                try:
                    d = grassmannian_distance(V_engage_vae[:, :k], V_choice_vae[:, :k])
                    grassmann_vae.append(d)
                except Exception:
                    pass

            # Choice accuracy by engagement state
            acc_result = _choice_accuracy_by_engagement(post_act, ch, eng)
            if acc_result is not None:
                choice_by_engage.append(acc_result)

        # --- Aggregate region results ---
        result = {
            "region": region,
            "n_sessions": len(measurements),
            # LDA engagement subspace
            "lda_engage_iia_mean": float(np.mean(lda_engage_iias)) if lda_engage_iias else None,
            "lda_engage_iia_std": float(np.std(lda_engage_iias)) if len(lda_engage_iias) > 1 else None,
            "lda_engage_iia_n": len(lda_engage_iias),
            # LDA choice subspace
            "lda_choice_iia_mean": float(np.mean(lda_choice_iias)) if lda_choice_iias else None,
            "lda_choice_iia_std": float(np.std(lda_choice_iias)) if len(lda_choice_iias) > 1 else None,
            "lda_choice_iia_n": len(lda_choice_iias),
            # VAE engagement subspace
            "vae_engage_iia_mean": float(np.mean(vae_engage_iias)) if vae_engage_iias else None,
            "vae_engage_iia_std": float(np.std(vae_engage_iias)) if len(vae_engage_iias) > 1 else None,
            "vae_engage_iia_n": len(vae_engage_iias),
            # VAE choice subspace
            "vae_choice_iia_mean": float(np.mean(vae_choice_iias)) if vae_choice_iias else None,
            "vae_choice_iia_std": float(np.std(vae_choice_iias)) if len(vae_choice_iias) > 1 else None,
            "vae_choice_iia_n": len(vae_choice_iias),
            # Cross-IIA: engagement swap -> choice prediction
            "cross_iia_mean": float(np.mean(cross_iias)) if cross_iias else None,
            "cross_iia_std": float(np.std(cross_iias)) if len(cross_iias) > 1 else None,
            "cross_iia_n": len(cross_iias),
            # Null engagement IIA
            "null_engage_iia_mean": float(np.mean(null_engage_iias)) if null_engage_iias else None,
            "null_engage_iia_std": float(np.std(null_engage_iias)) if null_engage_iias else None,
            # Grassmannian distances (engagement vs choice)
            "grassmann_lda_mean": float(np.mean(grassmann_lda)) if grassmann_lda else None,
            "grassmann_lda_std": float(np.std(grassmann_lda)) if len(grassmann_lda) > 1 else None,
            "grassmann_vae_mean": float(np.mean(grassmann_vae)) if grassmann_vae else None,
            "grassmann_vae_std": float(np.std(grassmann_vae)) if len(grassmann_vae) > 1 else None,
            # Choice accuracy by engagement state
            "choice_acc_engaged": (
                float(np.mean([c["acc_engaged"] for c in choice_by_engage]))
                if choice_by_engage else None
            ),
            "choice_acc_disengaged": (
                float(np.mean([c["acc_disengaged"] for c in choice_by_engage]))
                if choice_by_engage else None
            ),
            "choice_acc_diff": (
                float(np.mean([c["acc_diff"] for c in choice_by_engage]))
                if choice_by_engage else None
            ),
        }

        region_results[region] = result
        with open(jsonl_path, "a") as jf:
            jf.write(json.dumps(result, default=str) + "\n")

    logger.info(f"{datetime.now().isoformat()} All regions processed")

    # --- Aggregate statistical tests ---
    prediction_tests = {}

    # Q1: Is engagement subspace orthogonal to choice?
    lda_dists = [v["grassmann_lda_mean"] for v in region_results.values()
                 if v.get("grassmann_lda_mean") is not None]
    vae_dists = [v["grassmann_vae_mean"] for v in region_results.values()
                 if v.get("grassmann_vae_mean") is not None]
    if lda_dists:
        prediction_tests["engagement_choice_orthogonality_lda"] = {
            "mean_grassmannian_distance": float(np.mean(lda_dists)),
            "std_grassmannian_distance": float(np.std(lda_dists)),
            "n_regions": len(lda_dists),
            "interpretation": (
                "Distance near pi/2 (1.57) = orthogonal (independent coding). "
                "Distance near 0 = overlapping (shared coding dimensions)."
            ),
        }
    if vae_dists:
        prediction_tests["engagement_choice_orthogonality_vae"] = {
            "mean_grassmannian_distance": float(np.mean(vae_dists)),
            "std_grassmannian_distance": float(np.std(vae_dists)),
            "n_regions": len(vae_dists),
        }

    # Q2: Does engagement predict trial outcome?
    acc_diffs = [v["choice_acc_diff"] for v in region_results.values()
                 if v.get("choice_acc_diff") is not None]
    if len(acc_diffs) >= 5:
        try:
            w_stat, w_p = wilcoxon(acc_diffs, alternative="greater")
        except Exception:
            w_stat, w_p = None, None
        prediction_tests["engagement_predicts_choice_accuracy"] = {
            "mean_acc_diff": float(np.mean(acc_diffs)),
            "median_acc_diff": float(np.median(acc_diffs)),
            "n_regions": len(acc_diffs),
            "n_positive": int(np.sum(np.array(acc_diffs) > 0)),
            "wilcoxon_W": float(w_stat) if w_stat is not None else None,
            "wilcoxon_p": float(w_p) if w_p is not None else None,
            "interpretation": (
                "Positive acc_diff = engaged trials have better choice decoding. "
                "Wilcoxon tests whether this is significant across regions."
            ),
        }

    # Q3: VAE vs LDA for engagement subspace
    paired_vae_eng = []
    paired_lda_eng = []
    for v in region_results.values():
        if v.get("vae_engage_iia_mean") is not None and v.get("lda_engage_iia_mean") is not None:
            paired_vae_eng.append(v["vae_engage_iia_mean"])
            paired_lda_eng.append(v["lda_engage_iia_mean"])

    if len(paired_vae_eng) >= 5:
        diffs = np.array(paired_vae_eng) - np.array(paired_lda_eng)
        try:
            w_stat, w_p = wilcoxon(diffs, alternative="greater")
        except Exception:
            w_stat, w_p = None, None
        prediction_tests["vae_vs_lda_engagement_iia"] = {
            "vae_mean": float(np.mean(paired_vae_eng)),
            "lda_mean": float(np.mean(paired_lda_eng)),
            "mean_diff": float(np.mean(diffs)),
            "n_regions": len(paired_vae_eng),
            "n_vae_wins": int(np.sum(diffs > 0)),
            "wilcoxon_W": float(w_stat) if w_stat is not None else None,
            "wilcoxon_p": float(w_p) if w_p is not None else None,
        }

    # Engagement IIA vs null
    all_engage_iia = [v["lda_engage_iia_mean"] for v in region_results.values()
                      if v.get("lda_engage_iia_mean") is not None]
    all_null_iia = [v["null_engage_iia_mean"] for v in region_results.values()
                    if v.get("null_engage_iia_mean") is not None]
    if all_engage_iia and all_null_iia:
        try:
            u_stat, u_p = mannwhitneyu(all_engage_iia, all_null_iia, alternative="greater")
        except Exception:
            u_stat, u_p = None, None
        prediction_tests["engagement_iia_vs_null"] = {
            "engage_mean": float(np.mean(all_engage_iia)),
            "null_mean": float(np.mean(all_null_iia)),
            "effect_size": float(np.mean(all_engage_iia) - np.mean(all_null_iia)),
            "mann_whitney_U": float(u_stat) if u_stat is not None else None,
            "p_one_sided": float(u_p) if u_p is not None else None,
        }

    # Cross-IIA: does engagement subspace causally influence choice?
    cross_iia_values = [v["cross_iia_mean"] for v in region_results.values()
                        if v.get("cross_iia_mean") is not None]
    if cross_iia_values:
        prediction_tests["cross_engagement_choice_iia"] = {
            "mean_cross_iia": float(np.mean(cross_iia_values)),
            "std_cross_iia": float(np.std(cross_iia_values)),
            "n_regions": len(cross_iia_values),
            "interpretation": (
                "Cross-IIA > null = pre-stimulus engagement subspace causally "
                "influences post-stimulus choice representation."
            ),
        }

    # Grassmannian distance vs cross-IIA correlation
    grass_list = []
    cross_list = []
    for v in region_results.values():
        if v.get("grassmann_lda_mean") is not None and v.get("cross_iia_mean") is not None:
            grass_list.append(v["grassmann_lda_mean"])
            cross_list.append(v["cross_iia_mean"])
    if len(grass_list) >= 5:
        rho, p = spearmanr(grass_list, cross_list)
        prediction_tests["orthogonality_vs_cross_iia"] = {
            "rho": float(rho),
            "p": float(p),
            "n": len(grass_list),
            "interpretation": (
                "Negative rho = more orthogonal engagement/choice subspaces have "
                "LOWER cross-IIA (independent coding). Positive rho = overlapping "
                "subspaces have higher cross-influence."
            ),
        }

    # --- Top/bottom regions ---
    ranked_engage = sorted(
        [(r, v["lda_engage_iia_mean"]) for r, v in region_results.items()
         if v.get("lda_engage_iia_mean") is not None],
        key=lambda x: x[1], reverse=True,
    )
    ranked_cross = sorted(
        [(r, v["cross_iia_mean"]) for r, v in region_results.items()
         if v.get("cross_iia_mean") is not None],
        key=lambda x: x[1], reverse=True,
    )

    results = {
        "timestamp": datetime.now().isoformat(),
        "n_sessions": len(sessions),
        "n_regions": len(region_data),
        "n_regions_analyzed": len(region_results),
        "device": device,
        "hyperparameters": {
            "z_engage_dim": Z_ENGAGE_DIM,
            "z_choice_dim": Z_CHOICE_DIM,
            "z_other_dim": Z_OTHER_DIM,
            "hidden_dim": HIDDEN_DIM,
            "n_epochs": N_EPOCHS,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "beta_kl": BETA_KL,
            "alpha_supervised": ALPHA_SUPERVISED,
            "time_window_pre": "slice(0, 15)",
            "time_window_post": "slice(25, 45)",
            "rt_disengage_threshold": RT_DISENGAGE_THRESHOLD,
        },
        "region_results": {r: v for r, v in region_results.items()},
        "prediction_tests": prediction_tests,
        "top_engagement_iia_regions": ranked_engage[:10],
        "top_cross_iia_regions": ranked_cross[:10],
        "bottom_engagement_iia_regions": (
            ranked_engage[-10:] if len(ranked_engage) >= 10 else ranked_engage
        ),
    }

    # Save
    out_path = RESULTS_DIR / "engagement_subspace.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"{datetime.now().isoformat()} Saved results to {out_path}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-sessions", type=int, default=None)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    run(max_sessions=args.max_sessions)
