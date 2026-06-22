"""Experiment 58: Multi-task generalization of structured VAE causal subspaces.

Tests whether the structured VAE advantage (exp57: 3.5x better causal subspaces
than LDA for left/right choice) generalizes beyond binary choice to other task
variables available in Steinmetz 2019:

1. Evidence strength: |contrast_left - contrast_right|, median-split into
   easy (high |diff|) vs hard (low |diff|) trials.
2. Feedback/reward: correct (feedback_type=1) vs incorrect (feedback_type=-1).
3. Stimulus side: which side has higher contrast (distinct from the animal's
   choice — stimulus side is the world state, choice is the action).

For each task variable, trains a structured VAE with that variable as the
supervised label, computes IIA for the learned subspace, and compares against
an LDA baseline — same protocol as exp57.

Key question: is the VAE advantage specific to choice decoding, or does the
Bayesian subspace estimation generalize to any binary neural coding variable?
"""
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr, wilcoxon
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from tqdm import tqdm

from data.steinmetz import get_region_activity, list_regions, load_all
from experiments.exp57_structured_vae import (
    StructuredVAE,
    _iia_null_random_subspace,
    _power_law_exponent,
    train_vae,
    vae_loss,
)

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent.parent / "artifacts" / "exp58"
MIN_NEURONS = 15
TIME_WINDOW = slice(15, 35)
MIN_TRIALS_PER_CONDITION = 20

# VAE hyperparameters (same as exp57)
Z_CHOICE_DIM = 3
Z_OTHER_DIM = 15
HIDDEN_DIM = 128
N_EPOCHS = 300
BATCH_SIZE = 64
LR = 1e-3
BETA_KL = 1.0
ALPHA_CHOICE = 10.0

# Task variable names
TASK_EVIDENCE_STRENGTH = "evidence_strength"
TASK_FEEDBACK = "feedback"
TASK_STIMULUS_SIDE = "stimulus_side"
ALL_TASKS = [TASK_EVIDENCE_STRENGTH, TASK_FEEDBACK, TASK_STIMULUS_SIDE]


# ---------------------------------------------------------------------------
# Label extraction per task variable
# ---------------------------------------------------------------------------

def _get_evidence_strength_labels(sess: dict) -> np.ndarray | None:
    """Median-split |contrast_left - contrast_right| into easy (1) vs hard (0).

    Excludes zero-evidence trials (both contrasts equal).
    Returns (n_trials,) with values in {0, 1} or -1 for excluded trials.
    """
    cl = sess.get("contrast_left", np.array([]))
    cr = sess.get("contrast_right", np.array([]))
    if len(cl) == 0 or len(cr) == 0:
        return None
    n = sess["spks"].shape[2]
    n = min(n, len(cl), len(cr))
    cl, cr = cl[:n].astype(float), cr[:n].astype(float)
    diff = np.abs(cl - cr)
    # Exclude zero-evidence trials
    nonzero = diff > 0
    if nonzero.sum() < MIN_TRIALS_PER_CONDITION * 2:
        return None
    median_diff = np.median(diff[nonzero])
    labels = np.full(n, -1, dtype=int)
    # Hard trials: below or equal to median evidence
    labels[(diff > 0) & (diff <= median_diff)] = 0
    # Easy trials: above median evidence
    labels[diff > median_diff] = 1
    if np.sum(labels == 0) < MIN_TRIALS_PER_CONDITION or np.sum(labels == 1) < MIN_TRIALS_PER_CONDITION:
        return None
    return labels


def _get_feedback_labels(sess: dict) -> np.ndarray | None:
    """Correct (1) vs incorrect (0) feedback labels.

    feedback_type is -1 (incorrect) or 1 (correct) in Steinmetz data.
    """
    fb = sess.get("feedback_type", np.array([]))
    if len(fb) == 0:
        return None
    n = sess["spks"].shape[2]
    n = min(n, len(fb))
    fb = fb[:n]
    labels = np.full(n, -1, dtype=int)
    labels[fb == 1] = 1
    labels[fb == -1] = 0
    if np.sum(labels == 0) < MIN_TRIALS_PER_CONDITION or np.sum(labels == 1) < MIN_TRIALS_PER_CONDITION:
        return None
    return labels


def _get_stimulus_side_labels(sess: dict) -> np.ndarray | None:
    """Which side has higher contrast: left (0) vs right (1).

    Excludes equal-contrast trials. Distinct from choice (the animal's response).
    """
    cl = sess.get("contrast_left", np.array([]))
    cr = sess.get("contrast_right", np.array([]))
    if len(cl) == 0 or len(cr) == 0:
        return None
    n = sess["spks"].shape[2]
    n = min(n, len(cl), len(cr))
    cl, cr = cl[:n].astype(float), cr[:n].astype(float)
    labels = np.full(n, -1, dtype=int)
    labels[cl > cr] = 0  # left-dominant stimulus
    labels[cr > cl] = 1  # right-dominant stimulus
    if np.sum(labels == 0) < MIN_TRIALS_PER_CONDITION or np.sum(labels == 1) < MIN_TRIALS_PER_CONDITION:
        return None
    return labels


LABEL_EXTRACTORS = {
    TASK_EVIDENCE_STRENGTH: _get_evidence_strength_labels,
    TASK_FEEDBACK: _get_feedback_labels,
    TASK_STIMULUS_SIDE: _get_stimulus_side_labels,
}


# ---------------------------------------------------------------------------
# IIA computation (generalized from exp57)
# ---------------------------------------------------------------------------

def _compute_iia(activity: np.ndarray, intervention_labels: np.ndarray,
                 task_labels: np.ndarray, V: np.ndarray) -> float | None:
    """Interchange intervention accuracy using subspace V.

    Swap subspace projections between opposite-intervention-label trial pairs,
    measure whether a task-label classifier's prediction flips.

    Args:
        activity: (n_trials, n_neurons)
        intervention_labels: (n_trials,) binary labels defining which groups
            to swap between (the "evidence" analog)
        task_labels: (n_trials,) binary labels for the classifier to predict
            (the "choice" analog — what should flip after intervention)
        V: (n_neurons, k) orthonormal subspace directions
    """
    group0 = np.where(intervention_labels == 0)[0]
    group1 = np.where(intervention_labels == 1)[0]
    if len(group0) < MIN_TRIALS_PER_CONDITION or len(group1) < MIN_TRIALS_PER_CONDITION:
        return None

    lda = LinearDiscriminantAnalysis()
    try:
        lda.fit(activity, task_labels)
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


def _estimate_lda_subspace(activity: np.ndarray, labels: np.ndarray,
                           n_dims: int = 5) -> np.ndarray | None:
    """LDA+PCA baseline subspace (same as exp57)."""
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
# Per-task analysis
# ---------------------------------------------------------------------------

def _analyze_task(
    task_name: str,
    region_data: dict[str, list[dict]],
    device: str,
    results_dir: Path,
) -> dict[str, dict]:
    """Run VAE + LDA IIA comparison for one task variable across all regions.

    For each task variable, the IIA protocol is:
    - Train VAE supervised on the task label
    - The task label also serves as the intervention variable (swap between
      opposite-label groups) AND the prediction target (does the classifier flip?)
    - This tests whether the subspace captures the causal structure for that variable
    """
    jsonl_path = results_dir / f"{task_name}_incremental.jsonl"

    # Resume support
    computed_regions: set[str] = set()
    region_results: dict[str, dict] = {}
    if jsonl_path.exists():
        with open(jsonl_path) as jf:
            for line in jf:
                r = json.loads(line)
                computed_regions.add(r["region"])
                region_results[r["region"]] = r
        logger.info(f"  Resuming {task_name}: loaded {len(computed_regions)} pre-computed regions")

    for region, measurements in tqdm(region_data.items(), desc=f"  {task_name}"):
        if region in computed_regions:
            continue

        vae_iias = []
        lda_iias = []
        null_iias_all = []
        choice_accs = []
        posterior_uncertainties = []
        recon_losses = []
        alphas = []

        for m in measurements:
            activity = m["activity"]
            task_labels = m["task_labels"]
            n_neurons = activity.shape[1]

            # Adapt latent dims to neuron count
            z_task = min(Z_CHOICE_DIM, n_neurons // 5, n_neurons - 1)
            z_other = min(Z_OTHER_DIM, n_neurons // 3, n_neurons - z_task - 1)
            if z_task < 1 or z_other < 1:
                continue
            hidden = min(HIDDEN_DIM, n_neurons * 2)

            # Train VAE supervised on task labels
            try:
                vae_result = train_vae(
                    activity, task_labels,
                    z_choice_dim=z_task,
                    z_other_dim=z_other,
                    hidden_dim=hidden,
                    device=device,
                )
            except Exception as e:
                logger.warning(f"VAE training failed for {task_name}/{region} "
                               f"sess {m['session_idx']}: {e}")
                continue

            V_vae = vae_result["subspace_directions"]
            choice_accs.append(vae_result["choice_accuracy"])
            posterior_uncertainties.append(vae_result["posterior_uncertainty"])
            recon_losses.append(vae_result["final_loss"])

            # IIA: swap between opposite task-label groups, predict task label
            iia_vae = _compute_iia(activity, task_labels, task_labels, V_vae)
            if iia_vae is not None:
                vae_iias.append(iia_vae)

            # LDA baseline (matched dimensionality)
            V_lda = _estimate_lda_subspace(activity, task_labels, n_dims=z_task)
            if V_lda is not None:
                iia_lda = _compute_iia(activity, task_labels, task_labels, V_lda)
                if iia_lda is not None:
                    lda_iias.append(iia_lda)

            # Random subspace null
            null_iias = _iia_null_random_subspace(
                activity, task_labels, task_labels,
                n_dims=z_task, n_repeats=50,
            )
            if null_iias is not None:
                null_iias_all.extend(null_iias)

            # Power law exponent
            alpha = _power_law_exponent(activity)
            if alpha is not None:
                alphas.append(alpha)

        # Aggregate
        result = {
            "region": region,
            "task": task_name,
            "n_sessions": len(measurements),
            "vae_iia_mean": float(np.mean(vae_iias)) if vae_iias else None,
            "vae_iia_std": float(np.std(vae_iias)) if len(vae_iias) > 1 else None,
            "vae_iia_n": len(vae_iias),
            "lda_iia_mean": float(np.mean(lda_iias)) if lda_iias else None,
            "lda_iia_std": float(np.std(lda_iias)) if len(lda_iias) > 1 else None,
            "lda_iia_n": len(lda_iias),
            "null_iia_mean": float(np.mean(null_iias_all)) if null_iias_all else None,
            "null_iia_std": float(np.std(null_iias_all)) if null_iias_all else None,
            "iia_vae_above_null": (
                float(np.mean(vae_iias) - np.mean(null_iias_all))
                if vae_iias and null_iias_all else None
            ),
            "iia_vae_above_lda": (
                float(np.mean(vae_iias) - np.mean(lda_iias))
                if vae_iias and lda_iias else None
            ),
            "classifier_accuracy_mean": float(np.mean(choice_accs)) if choice_accs else None,
            "posterior_uncertainty_mean": float(np.mean(posterior_uncertainties)) if posterior_uncertainties else None,
            "recon_loss_mean": float(np.mean(recon_losses)) if recon_losses else None,
            "power_law_alpha": float(np.mean(alphas)) if alphas else None,
        }

        region_results[region] = result
        with open(jsonl_path, "a") as jf:
            jf.write(json.dumps(result, default=str) + "\n")

    return region_results


# ---------------------------------------------------------------------------
# Statistical tests across tasks
# ---------------------------------------------------------------------------

def _paired_comparison(region_results: dict[str, dict]) -> dict | None:
    """Paired Wilcoxon test: VAE IIA vs LDA IIA across regions for one task."""
    paired_vae = []
    paired_lda = []
    for v in region_results.values():
        if v.get("vae_iia_mean") is not None and v.get("lda_iia_mean") is not None:
            paired_vae.append(v["vae_iia_mean"])
            paired_lda.append(v["lda_iia_mean"])

    if len(paired_vae) < 5:
        return None

    diffs = np.array(paired_vae) - np.array(paired_lda)
    try:
        w_stat, w_p = wilcoxon(diffs, alternative="greater")
    except Exception:
        w_stat, w_p = None, None

    return {
        "vae_mean": float(np.mean(paired_vae)),
        "lda_mean": float(np.mean(paired_lda)),
        "mean_diff": float(np.mean(diffs)),
        "median_diff": float(np.median(diffs)),
        "n_regions": len(paired_vae),
        "n_vae_wins": int(np.sum(diffs > 0)),
        "vae_win_rate": float(np.mean(diffs > 0)),
        "wilcoxon_W": float(w_stat) if w_stat is not None else None,
        "wilcoxon_p": float(w_p) if w_p is not None else None,
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
    logger.info(f"{datetime.now().isoformat()} Starting multi-task VAE experiment "
                f"with {len(sessions)} sessions on {device}")

    # --- Build per-task, per-region data ---
    # For each task, we need: activity, task_labels (binary), plus session metadata
    task_region_data: dict[str, dict[str, list[dict]]] = {t: {} for t in ALL_TASKS}

    for sess_idx, sess in enumerate(tqdm(sessions, desc="Loading sessions")):
        mouse = str(sess.get("mouse_name", f"mouse_{sess_idx}"))
        regions = list_regions(sess, min_neurons=MIN_NEURONS)

        # Extract labels for each task
        task_labels_map: dict[str, np.ndarray | None] = {}
        for task_name in ALL_TASKS:
            task_labels_map[task_name] = LABEL_EXTRACTORS[task_name](sess)

        for region in regions:
            act = get_region_activity(sess, region)
            if act is None or act.shape[1] < MIN_NEURONS:
                continue

            for task_name in ALL_TASKS:
                labels = task_labels_map[task_name]
                if labels is None:
                    continue

                n = min(act.shape[0], len(labels))
                activity = act[:n, :, TIME_WINDOW].mean(axis=2)
                tl = labels[:n]

                # Keep only valid (non -1) trials
                valid = tl >= 0
                if valid.sum() < MIN_TRIALS_PER_CONDITION * 2:
                    continue

                if region not in task_region_data[task_name]:
                    task_region_data[task_name][region] = []
                task_region_data[task_name][region].append({
                    "session_idx": sess_idx,
                    "mouse": mouse,
                    "activity": activity[valid],
                    "task_labels": tl[valid],
                    "n_neurons": int(activity.shape[1]),
                })

    for task_name in ALL_TASKS:
        n_regions = len(task_region_data[task_name])
        n_measurements = sum(len(v) for v in task_region_data[task_name].values())
        logger.info(f"{datetime.now().isoformat()} {task_name}: "
                    f"{n_regions} regions, {n_measurements} measurements")

    # --- Per-task VAE training and IIA ---
    all_task_results: dict[str, dict[str, dict]] = {}
    for task_name in ALL_TASKS:
        logger.info(f"{datetime.now().isoformat()} === Processing task: {task_name} ===")
        all_task_results[task_name] = _analyze_task(
            task_name,
            task_region_data[task_name],
            device,
            RESULTS_DIR,
        )

    logger.info(f"{datetime.now().isoformat()} All tasks processed")

    # --- Cross-task comparison ---
    cross_task = {}
    for task_name in ALL_TASKS:
        comparison = _paired_comparison(all_task_results[task_name])
        if comparison is not None:
            cross_task[task_name] = comparison

    # Does VAE advantage generalize? Count tasks where VAE significantly beats LDA
    generalization_summary = {
        "tasks_tested": len(ALL_TASKS),
        "tasks_with_data": len(cross_task),
        "tasks_vae_wins_majority": sum(
            1 for v in cross_task.values()
            if v.get("vae_win_rate", 0) > 0.5
        ),
        "tasks_vae_significant": sum(
            1 for v in cross_task.values()
            if v.get("wilcoxon_p") is not None and v["wilcoxon_p"] < 0.05
        ),
        "per_task": {
            task_name: {
                "vae_mean_iia": v.get("vae_mean"),
                "lda_mean_iia": v.get("lda_mean"),
                "mean_diff": v.get("mean_diff"),
                "vae_win_rate": v.get("vae_win_rate"),
                "wilcoxon_p": v.get("wilcoxon_p"),
                "n_regions": v.get("n_regions"),
            }
            for task_name, v in cross_task.items()
        },
    }

    # --- Alpha vs IIA correlation per task ---
    alpha_correlations = {}
    for task_name, region_results in all_task_results.items():
        alpha_list = []
        iia_list = []
        for v in region_results.values():
            if v.get("power_law_alpha") is not None and v.get("vae_iia_mean") is not None:
                alpha_list.append(v["power_law_alpha"])
                iia_list.append(v["vae_iia_mean"])
        if len(alpha_list) >= 5:
            rho, p = spearmanr(alpha_list, iia_list)
            alpha_correlations[task_name] = {
                "rho": float(rho),
                "p": float(p),
                "n": len(alpha_list),
            }

    # --- Assemble final results ---
    results = {
        "timestamp": datetime.now().isoformat(),
        "n_sessions": len(sessions),
        "device": device,
        "hyperparameters": {
            "z_choice_dim": Z_CHOICE_DIM,
            "z_other_dim": Z_OTHER_DIM,
            "hidden_dim": HIDDEN_DIM,
            "n_epochs": N_EPOCHS,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "beta_kl": BETA_KL,
            "alpha_choice": ALPHA_CHOICE,
        },
        "tasks": ALL_TASKS,
        "per_task_n_regions": {
            task_name: len(task_region_data[task_name]) for task_name in ALL_TASKS
        },
        "per_task_region_results": {
            task_name: {r: v for r, v in rr.items()}
            for task_name, rr in all_task_results.items()
        },
        "vae_vs_lda_per_task": cross_task,
        "generalization_summary": generalization_summary,
        "alpha_vs_iia_per_task": alpha_correlations,
    }

    # Save
    out_path = RESULTS_DIR / "multi_task_vae.json"
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
