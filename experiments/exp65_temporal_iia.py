"""Experiment 65: Temporal IIA -- when does the causal choice subspace emerge?

Shows when the choice subspace IIA rises during the trial relative to the
behavioral response.  If IIA increases BEFORE the mouse moves the wheel, the
subspace carries a decision signal, not a post-hoc motor artifact.

Steinmetz 2019 spike counts are in 10ms bins:
    Bins 0-24  (0-250ms):   pre-stimulus baseline
    Bin  25    (~250ms):     approximate stimulus onset
    Bins 25-50 (250-500ms): post-stimulus / decision period
    Response typically at bin 30-40 (300-400ms from trial start)

Protocol:
    1. Sliding-window LDA IIA (fast) -- window 5 bins / slide 2 bins, full range.
    2. VAE IIA at 5 key timepoints (bins 10, 20, 25, 30, 40) for comparison.
    3. Define onset = first window where IIA > (pre-stimulus mean + 2 SD).
    4. Compare onset time to behavioral response time per region.

Per-region outputs:
    - IIA timecourse (window_center, iia_lda) for every window
    - IIA at 5 VAE timepoints (window_center, iia_vae)
    - Onset time (LDA and VAE)
    - Whether onset precedes behavioral response

Aggregate outputs:
    - Mean IIA timecourse across regions with SEM bands
    - Distribution of onset times
    - Fraction of regions where subspace emerges pre-response
"""
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from scipy.stats import wilcoxon
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from tqdm import tqdm

from data.steinmetz import get_choice_labels, get_region_activity, list_regions, load_all
from experiments.exp57_structured_vae import train_vae

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent.parent / "artifacts" / "exp65"
MIN_NEURONS = 15
MIN_TRIALS_PER_CONDITION = 20

# Sliding window parameters (in 10ms bins)
WINDOW_SIZE = 5    # 50ms
WINDOW_SLIDE = 2   # 20ms
WINDOW_START = 0
WINDOW_END = 45    # last window starts here (centers up to ~47)

# VAE is only trained at these center bins to keep runtime practical
VAE_CENTER_BINS = [10, 20, 25, 30, 40]

# Pre-stimulus baseline range for onset detection
BASELINE_END_BIN = 25  # everything before stimulus onset

# VAE hyperparameters (same as exp57)
Z_CHOICE_DIM = 3
Z_OTHER_DIM = 15
HIDDEN_DIM = 128
N_EPOCHS = 300
BATCH_SIZE = 64
LR = 1e-3
BETA_KL = 1.0
ALPHA_CHOICE = 10.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sliding_window_centers() -> list[float]:
    """Return the center bin for each sliding window position."""
    centers = []
    start = WINDOW_START
    while start + WINDOW_SIZE <= WINDOW_END + WINDOW_SIZE:
        center = start + WINDOW_SIZE / 2.0
        centers.append(center)
        start += WINDOW_SLIDE
    return centers


def _window_slices() -> list[slice]:
    """Return the bin slices for each sliding window position."""
    slices = []
    start = WINDOW_START
    while start + WINDOW_SIZE <= WINDOW_END + WINDOW_SIZE:
        slices.append(slice(start, start + WINDOW_SIZE))
        start += WINDOW_SLIDE
    return slices


def _contrast_to_evidence_label(sess: dict) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Binary evidence label from contrast difference."""
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


def _estimate_lda_subspace(
    activity: np.ndarray, labels: np.ndarray, n_dims: int = 3
) -> np.ndarray | None:
    """LDA+PCA subspace (same protocol as exp57/exp61)."""
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


def _compute_iia(
    activity: np.ndarray,
    evidence_labels: np.ndarray,
    choice_labels: np.ndarray,
    V: np.ndarray,
) -> float | None:
    """Interchange intervention accuracy using subspace V.

    Swap evidence projections between opposite-evidence trial pairs,
    measure choice classifier flip rate.
    """
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
        proj_l = V @ (V.T @ act_l)
        proj_r = V @ (V.T @ act_r)
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


def _get_window_activity(
    act_raw: np.ndarray, win_slice: slice
) -> np.ndarray:
    """Average activity across time bins in a window.

    Args:
        act_raw: (n_trials, n_neurons, n_bins) from get_region_activity
        win_slice: slice over the bin axis

    Returns:
        (n_trials, n_neurons)
    """
    return act_raw[:, :, win_slice].mean(axis=2)


def _detect_onset(
    iia_values: list[float | None],
    centers: list[float],
    baseline_end: float,
) -> float | None:
    """Find the first window center where IIA exceeds baseline mean + 2 SD.

    Only uses windows whose center < baseline_end for the baseline stats.
    Returns the center bin of the onset window, or None if never exceeded.
    """
    baseline_iias = [
        v for v, c in zip(iia_values, centers)
        if v is not None and c < baseline_end
    ]
    if len(baseline_iias) < 3:
        return None
    bl_mean = np.mean(baseline_iias)
    bl_std = np.std(baseline_iias)
    threshold = bl_mean + 2.0 * bl_std

    for v, c in zip(iia_values, centers):
        if v is not None and c >= baseline_end and v > threshold:
            return float(c)
    return None


def _median_response_bin(sess: dict) -> float | None:
    """Median response time in bin units (10ms bins from trial start).

    response_time is in seconds; stimulus onset is at bin 25 (~250ms).
    We convert to bins from trial start.
    """
    rt = sess.get("response_time", np.array([]))
    if len(rt) == 0:
        return None
    rt = np.asarray(rt).ravel()
    # Filter out NaN and unreasonable values
    valid = np.isfinite(rt) & (rt > 0) & (rt < 5.0)
    if valid.sum() < 10:
        return None
    median_rt_sec = float(np.median(rt[valid]))
    # response_time is relative to stimulus onset (~bin 25)
    # Convert to bin from trial start: stimulus_onset_bin + rt_in_bins
    return 25.0 + median_rt_sec / 0.01


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(max_sessions: int | None = None) -> dict:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    sessions = load_all()
    if max_sessions:
        sessions = sessions[:max_sessions]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"{datetime.now().isoformat()} Starting temporal IIA experiment "
                f"with {len(sessions)} sessions on {device}")

    win_slices = _window_slices()
    win_centers = _sliding_window_centers()
    n_windows = len(win_centers)

    # Map VAE center bins to the nearest window indices
    vae_window_indices = []
    for vae_bin in VAE_CENTER_BINS:
        best_idx = int(np.argmin([abs(c - vae_bin) for c in win_centers]))
        vae_window_indices.append(best_idx)

    # --- Load data per region ---
    region_data: dict[str, list[dict]] = {}

    for sess_idx, sess in enumerate(tqdm(sessions, desc="Loading sessions")):
        choice_labels = get_choice_labels(sess)
        if len(np.unique(choice_labels)) < 2:
            continue
        evidence_labels, _ = _contrast_to_evidence_label(sess)
        if evidence_labels is None:
            continue

        median_resp_bin = _median_response_bin(sess)
        mouse = str(sess.get("mouse_name", f"mouse_{sess_idx}"))
        regions = list_regions(sess, min_neurons=MIN_NEURONS)

        for region in regions:
            act_raw = get_region_activity(sess, region)
            if act_raw is None or act_raw.shape[1] < MIN_NEURONS:
                continue
            # Ensure trial counts are consistent
            n = min(act_raw.shape[0], len(choice_labels), len(evidence_labels))
            act_raw = act_raw[:n]
            ch = choice_labels[:n]
            ev = evidence_labels[:n]

            # Filter to trials with valid evidence labels
            valid = ev >= 0
            if valid.sum() < MIN_TRIALS_PER_CONDITION * 2:
                continue

            if region not in region_data:
                region_data[region] = []
            region_data[region].append({
                "session_idx": sess_idx,
                "mouse": mouse,
                "act_raw": act_raw[valid],   # (n_valid_trials, n_neurons, n_bins)
                "choice_labels": ch[valid],
                "evidence_labels": ev[valid],
                "median_response_bin": median_resp_bin,
                "n_neurons": int(act_raw.shape[1]),
            })

    logger.info(f"{datetime.now().isoformat()} {len(region_data)} regions loaded")

    # --- Per-region temporal IIA ---
    region_results: dict[str, dict] = {}
    jsonl_path = RESULTS_DIR / "temporal_iia_incremental.jsonl"

    # Resume support
    computed_regions: set[str] = set()
    if jsonl_path.exists():
        with open(jsonl_path) as jf:
            for line in jf:
                r = json.loads(line)
                computed_regions.add(r["region"])
                region_results[r["region"]] = r
        logger.info(f"Resuming: loaded {len(computed_regions)} pre-computed regions")

    for region, measurements in tqdm(region_data.items(), desc="Temporal IIA"):
        if region in computed_regions:
            continue

        # Collect per-session timecourses then average
        session_lda_timecourses: list[list[float | None]] = []
        session_vae_points: list[dict[int, float | None]] = []
        session_response_bins: list[float | None] = []

        for m in measurements:
            act_raw = m["act_raw"]
            ch = m["choice_labels"]
            ev = m["evidence_labels"]
            n_neurons = m["n_neurons"]

            z_choice = min(Z_CHOICE_DIM, n_neurons // 5, n_neurons - 1)
            z_other = min(Z_OTHER_DIM, n_neurons // 3, n_neurons - z_choice - 1)
            if z_choice < 1 or z_other < 1:
                continue
            hidden = min(HIDDEN_DIM, n_neurons * 2)

            # --- LDA sliding window ---
            lda_iia_timecourse: list[float | None] = []
            for ws in win_slices:
                # Check that the window fits within the available bins
                if ws.stop > act_raw.shape[2]:
                    lda_iia_timecourse.append(None)
                    continue
                activity = _get_window_activity(act_raw, ws)
                V_lda = _estimate_lda_subspace(activity, ev, n_dims=z_choice)
                if V_lda is None:
                    lda_iia_timecourse.append(None)
                    continue
                iia = _compute_iia(activity, ev, ch, V_lda)
                lda_iia_timecourse.append(iia)

            session_lda_timecourses.append(lda_iia_timecourse)
            session_response_bins.append(m["median_response_bin"])

            # --- VAE at key timepoints ---
            vae_points: dict[int, float | None] = {}
            for wi in vae_window_indices:
                ws = win_slices[wi]
                if ws.stop > act_raw.shape[2]:
                    vae_points[wi] = None
                    continue
                activity = _get_window_activity(act_raw, ws)
                try:
                    vae_result = train_vae(
                        activity, ch,
                        z_choice_dim=z_choice,
                        z_other_dim=z_other,
                        hidden_dim=hidden,
                        device=device,
                    )
                    V_vae = vae_result["subspace_directions"]
                    iia = _compute_iia(activity, ev, ch, V_vae)
                    vae_points[wi] = iia
                except Exception as e:
                    logger.warning(f"VAE failed for {region} sess {m['session_idx']} "
                                   f"win_idx {wi}: {e}")
                    vae_points[wi] = None

            session_vae_points.append(vae_points)

        # --- Average across sessions for this region ---
        if not session_lda_timecourses:
            continue

        # LDA timecourse: mean across sessions at each window
        mean_lda_iia: list[float | None] = []
        for wi in range(n_windows):
            vals = [tc[wi] for tc in session_lda_timecourses if tc[wi] is not None]
            mean_lda_iia.append(float(np.mean(vals)) if vals else None)

        # VAE at key points: mean across sessions
        mean_vae_iia: dict[int, float | None] = {}
        for wi in vae_window_indices:
            vals = [sp[wi] for sp in session_vae_points if sp.get(wi) is not None]
            mean_vae_iia[wi] = float(np.mean(vals)) if vals else None

        # Median response bin across sessions
        resp_bins = [rb for rb in session_response_bins if rb is not None]
        region_median_response_bin = float(np.median(resp_bins)) if resp_bins else None

        # Onset detection for LDA
        lda_onset = _detect_onset(mean_lda_iia, win_centers, BASELINE_END_BIN)

        # Onset detection for VAE (sparse: only at VAE timepoints)
        vae_iia_at_vae_centers = [mean_vae_iia.get(wi) for wi in vae_window_indices]
        vae_centers_for_onset = [win_centers[wi] for wi in vae_window_indices]
        vae_onset = _detect_onset(vae_iia_at_vae_centers, vae_centers_for_onset, BASELINE_END_BIN)

        # Does onset precede response?
        lda_pre_response = (
            lda_onset is not None
            and region_median_response_bin is not None
            and lda_onset < region_median_response_bin
        )
        vae_pre_response = (
            vae_onset is not None
            and region_median_response_bin is not None
            and vae_onset < region_median_response_bin
        )

        # Build timecourse arrays for JSON (center, iia pairs)
        lda_timecourse_out = [
            {"center_bin": float(c), "center_ms": float(c * 10), "iia": v}
            for c, v in zip(win_centers, mean_lda_iia)
        ]
        vae_timecourse_out = [
            {
                "center_bin": float(win_centers[wi]),
                "center_ms": float(win_centers[wi] * 10),
                "iia": mean_vae_iia[wi],
            }
            for wi in vae_window_indices
        ]

        result = {
            "region": region,
            "n_sessions": len(measurements),
            "n_sessions_with_data": len(session_lda_timecourses),
            "lda_timecourse": lda_timecourse_out,
            "vae_timecourse": vae_timecourse_out,
            "lda_onset_bin": lda_onset,
            "lda_onset_ms": lda_onset * 10 if lda_onset is not None else None,
            "vae_onset_bin": vae_onset,
            "vae_onset_ms": vae_onset * 10 if vae_onset is not None else None,
            "median_response_bin": region_median_response_bin,
            "median_response_ms": (
                region_median_response_bin * 10
                if region_median_response_bin is not None else None
            ),
            "lda_pre_response": lda_pre_response,
            "vae_pre_response": vae_pre_response,
        }

        region_results[region] = result
        with open(jsonl_path, "a") as jf:
            jf.write(json.dumps(result, default=str) + "\n")

    logger.info(f"{datetime.now().isoformat()} All regions processed")

    # --- Aggregate statistics ---
    prediction_tests: dict[str, dict] = {}

    # 1. Mean IIA timecourse across regions (with SEM)
    all_lda_timecourses: list[list[float | None]] = []
    for v in region_results.values():
        tc = v.get("lda_timecourse")
        if tc is not None:
            all_lda_timecourses.append([pt["iia"] for pt in tc])

    mean_timecourse = []
    sem_timecourse = []
    for wi in range(n_windows):
        vals = [
            tc[wi] for tc in all_lda_timecourses
            if wi < len(tc) and tc[wi] is not None
        ]
        if vals:
            mean_timecourse.append(float(np.mean(vals)))
            sem_timecourse.append(float(np.std(vals) / np.sqrt(len(vals))))
        else:
            mean_timecourse.append(None)
            sem_timecourse.append(None)

    aggregate_timecourse = [
        {
            "center_bin": float(c),
            "center_ms": float(c * 10),
            "mean_iia": m,
            "sem_iia": s,
        }
        for c, m, s in zip(win_centers, mean_timecourse, sem_timecourse)
    ]

    # 2. Onset time distribution
    lda_onsets = [
        v["lda_onset_bin"] for v in region_results.values()
        if v.get("lda_onset_bin") is not None
    ]
    vae_onsets = [
        v["vae_onset_bin"] for v in region_results.values()
        if v.get("vae_onset_bin") is not None
    ]

    # 3. Fraction of regions where subspace emerges pre-response
    n_lda_pre = sum(
        1 for v in region_results.values() if v.get("lda_pre_response") is True
    )
    n_lda_with_onset = sum(
        1 for v in region_results.values() if v.get("lda_onset_bin") is not None
    )
    n_vae_pre = sum(
        1 for v in region_results.values() if v.get("vae_pre_response") is True
    )
    n_vae_with_onset = sum(
        1 for v in region_results.values() if v.get("vae_onset_bin") is not None
    )

    prediction_tests["pre_response_emergence"] = {
        "lda_n_pre_response": n_lda_pre,
        "lda_n_with_onset": n_lda_with_onset,
        "lda_frac_pre_response": (
            float(n_lda_pre / n_lda_with_onset) if n_lda_with_onset > 0 else None
        ),
        "vae_n_pre_response": n_vae_pre,
        "vae_n_with_onset": n_vae_with_onset,
        "vae_frac_pre_response": (
            float(n_vae_pre / n_vae_with_onset) if n_vae_with_onset > 0 else None
        ),
        "interpretation": (
            "Fraction > 0.5 = most regions show causal subspace emergence before "
            "the behavioral response, supporting a decision signal interpretation."
        ),
    }

    # 4. Onset time vs response time (paired)
    onset_bins = []
    response_bins = []
    for v in region_results.values():
        if v.get("lda_onset_bin") is not None and v.get("median_response_bin") is not None:
            onset_bins.append(v["lda_onset_bin"])
            response_bins.append(v["median_response_bin"])

    if len(onset_bins) >= 5:
        diffs = np.array(response_bins) - np.array(onset_bins)  # positive = onset before response
        try:
            w_stat, w_p = wilcoxon(diffs, alternative="greater")
        except Exception:
            w_stat, w_p = None, None
        prediction_tests["onset_vs_response_timing"] = {
            "mean_onset_bin": float(np.mean(onset_bins)),
            "mean_onset_ms": float(np.mean(onset_bins) * 10),
            "mean_response_bin": float(np.mean(response_bins)),
            "mean_response_ms": float(np.mean(response_bins) * 10),
            "mean_lead_bins": float(np.mean(diffs)),
            "mean_lead_ms": float(np.mean(diffs) * 10),
            "median_lead_ms": float(np.median(diffs) * 10),
            "n_regions": len(onset_bins),
            "n_onset_before_response": int(np.sum(diffs > 0)),
            "wilcoxon_W": float(w_stat) if w_stat is not None else None,
            "wilcoxon_p": float(w_p) if w_p is not None else None,
            "interpretation": (
                "Positive mean_lead = subspace onset precedes behavioral response. "
                "Wilcoxon tests whether the lead is significant across regions."
            ),
        }

    # 5. LDA vs VAE agreement at shared timepoints
    lda_at_vae = []
    vae_at_vae = []
    for v in region_results.values():
        lda_tc = v.get("lda_timecourse")
        vae_tc = v.get("vae_timecourse")
        if lda_tc is None or vae_tc is None:
            continue
        for vae_pt in vae_tc:
            vae_iia = vae_pt.get("iia")
            if vae_iia is None:
                continue
            # Find matching LDA window by center_bin
            target_bin = vae_pt["center_bin"]
            matching = [pt["iia"] for pt in lda_tc if abs(pt["center_bin"] - target_bin) < 1.0]
            if matching and matching[0] is not None:
                lda_at_vae.append(matching[0])
                vae_at_vae.append(vae_iia)

    if len(lda_at_vae) >= 5:
        from scipy.stats import spearmanr
        rho, p = spearmanr(lda_at_vae, vae_at_vae)
        prediction_tests["lda_vae_agreement"] = {
            "spearman_rho": float(rho),
            "spearman_p": float(p),
            "n_pairs": len(lda_at_vae),
            "mean_lda": float(np.mean(lda_at_vae)),
            "mean_vae": float(np.mean(vae_at_vae)),
            "interpretation": (
                "High rho = LDA and VAE IIA agree across timepoints. "
                "VAE may still find better subspaces (higher absolute IIA)."
            ),
        }

    # 6. Onset time distribution summary
    if lda_onsets:
        prediction_tests["onset_distribution_lda"] = {
            "mean_onset_bin": float(np.mean(lda_onsets)),
            "mean_onset_ms": float(np.mean(lda_onsets) * 10),
            "std_onset_ms": float(np.std(lda_onsets) * 10),
            "median_onset_ms": float(np.median(lda_onsets) * 10),
            "n_regions": len(lda_onsets),
            "quartiles_ms": [float(np.percentile(lda_onsets, q) * 10) for q in [25, 50, 75]],
        }
    if vae_onsets:
        prediction_tests["onset_distribution_vae"] = {
            "mean_onset_bin": float(np.mean(vae_onsets)),
            "mean_onset_ms": float(np.mean(vae_onsets) * 10),
            "std_onset_ms": float(np.std(vae_onsets) * 10),
            "median_onset_ms": float(np.median(vae_onsets) * 10),
            "n_regions": len(vae_onsets),
        }

    # --- Final results ---
    results = {
        "timestamp": datetime.now().isoformat(),
        "n_sessions": len(sessions),
        "n_regions": len(region_data),
        "n_regions_analyzed": len(region_results),
        "device": device,
        "parameters": {
            "window_size_bins": WINDOW_SIZE,
            "window_slide_bins": WINDOW_SLIDE,
            "window_size_ms": WINDOW_SIZE * 10,
            "window_slide_ms": WINDOW_SLIDE * 10,
            "n_windows": n_windows,
            "vae_center_bins": VAE_CENTER_BINS,
            "baseline_end_bin": BASELINE_END_BIN,
            "z_choice_dim": Z_CHOICE_DIM,
            "z_other_dim": Z_OTHER_DIM,
            "hidden_dim": HIDDEN_DIM,
            "n_epochs": N_EPOCHS,
        },
        "aggregate_timecourse": aggregate_timecourse,
        "prediction_tests": prediction_tests,
        "region_results": {r: v for r, v in region_results.items()},
    }

    out_path = RESULTS_DIR / "temporal_iia.json"
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
