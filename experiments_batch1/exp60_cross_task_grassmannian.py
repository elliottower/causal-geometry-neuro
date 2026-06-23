"""Experiment 60: Cross-task Grassmannian distances for functional multiplexing.

For each brain region, fit subspaces for 4 task variables (choice, evidence
strength, feedback, stimulus side) using both LDA and structured VAE, then
compute all 6 pairwise Grassmannian (geodesic) distances between the 4
task-variable subspaces. This reveals which regions multiplex overlapping
representations vs maintain orthogonal coding axes.

Novel contribution: no prior paper has measured Grassmannian geometry of
task-variable subspaces within brain regions. Existing work measures
cross-region distances (Sorscher et al., Bernstein et al.) or uses CKA
for representational similarity. This experiment applies the Grassmannian
geodesic to task-variable pairs within a region, directly quantifying
functional multiplexing geometry.

Key analyses:
1. Which task-variable pairs have the most orthogonal subspaces? (separation)
2. Which regions multiplex the most variables in overlapping subspaces?
3. Does the VAE find more separated subspaces than LDA?
4. Correlation between region dimensionality and degree of multiplexing
"""
import json
import logging
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from tqdm import tqdm

from data.steinmetz import get_choice_labels, get_region_activity, list_regions, load_all
from experiments.exp57_structured_vae import StructuredVAE, train_vae
from experiments.exp58_multi_task import (
    LABEL_EXTRACTORS,
    _get_evidence_strength_labels,
    _get_feedback_labels,
    _get_stimulus_side_labels,
)
from geometry.distances import all_subspace_distances, grassmannian_distance, principal_angles

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent.parent / "artifacts" / "exp60"
MIN_NEURONS = 15
TIME_WINDOW = slice(15, 35)
MIN_TRIALS_PER_CONDITION = 20

# Task variables
TASK_CHOICE = "choice"
TASK_EVIDENCE = "evidence_strength"
TASK_FEEDBACK = "feedback"
TASK_STIMULUS = "stimulus_side"
ALL_TASKS = [TASK_CHOICE, TASK_EVIDENCE, TASK_FEEDBACK, TASK_STIMULUS]

# VAE hyperparameters (same as exp57/58)
Z_DIM = 3
Z_OTHER_DIM = 15
HIDDEN_DIM = 128
N_EPOCHS = 300
BATCH_SIZE = 64
LR = 1e-3
BETA_KL = 1.0
ALPHA_TASK = 10.0

# Subspace dimensionality for LDA
LDA_K = 3


# ---------------------------------------------------------------------------
# Label extraction
# ---------------------------------------------------------------------------

def _get_choice_labels_from_session(sess: dict) -> np.ndarray | None:
    """Binary choice labels (left=0 / right=1), matching the exp58 convention."""
    response = sess.get("response", np.array([]))
    if len(response) == 0:
        return None
    n = sess["spks"].shape[2]
    n = min(n, len(response))
    response = response[:n]
    labels = np.full(n, -1, dtype=int)
    labels[response > 0] = 1
    labels[response < 0] = 0
    if np.sum(labels == 0) < MIN_TRIALS_PER_CONDITION or np.sum(labels == 1) < MIN_TRIALS_PER_CONDITION:
        return None
    return labels


TASK_LABEL_EXTRACTORS = {
    TASK_CHOICE: _get_choice_labels_from_session,
    TASK_EVIDENCE: _get_evidence_strength_labels,
    TASK_FEEDBACK: _get_feedback_labels,
    TASK_STIMULUS: _get_stimulus_side_labels,
}


# ---------------------------------------------------------------------------
# Subspace fitting: LDA
# ---------------------------------------------------------------------------

def _fit_lda_subspace(activity: np.ndarray, labels: np.ndarray, k: int = LDA_K) -> np.ndarray | None:
    """PCA(20) -> LDA -> extract top-k discriminant directions -> QR orthogonalize.

    Returns (n_neurons, k) orthonormal basis, or None on failure.
    """
    n_trials, n_neurons = activity.shape
    k = min(k, n_neurons - 1, n_trials - 2)
    if k < 1 or len(np.unique(labels)) < 2:
        return None
    pca_dim = min(20, n_neurons - 1, n_trials - 1)
    if pca_dim < 1:
        return None
    pca = PCA(n_components=pca_dim)
    scores = pca.fit_transform(activity)
    lda = LinearDiscriminantAnalysis()
    try:
        lda.fit(scores, labels)
    except Exception:
        return None
    # LDA discriminant direction in PCA space
    lda_dir = lda.coef_[0]
    lda_dir = lda_dir / (np.linalg.norm(lda_dir) + 1e-10)
    # Map back to neuron space
    lda_neuron = pca.components_.T @ lda_dir
    # Fill remaining dimensions with top PCA directions
    pca_components = pca.components_[:k].T  # (n_neurons, k)
    combined = np.column_stack([lda_neuron.reshape(-1, 1), pca_components])
    Q, _ = np.linalg.qr(combined)
    return Q[:, :k]


# ---------------------------------------------------------------------------
# Subspace fitting: VAE
# ---------------------------------------------------------------------------

def _fit_vae_subspace(
    activity: np.ndarray,
    labels: np.ndarray,
    z_dim: int = Z_DIM,
    device: str = "cpu",
) -> np.ndarray | None:
    """Train structured VAE and extract the supervised-factor subspace.

    Returns (n_neurons, z_dim) orthonormal basis, or None on failure.
    """
    n_neurons = activity.shape[1]
    z_task = min(z_dim, n_neurons // 5, n_neurons - 1)
    z_other = min(Z_OTHER_DIM, n_neurons // 3, n_neurons - z_task - 1)
    if z_task < 1 or z_other < 1:
        return None
    hidden = min(HIDDEN_DIM, n_neurons * 2)
    try:
        vae_result = train_vae(
            activity, labels,
            z_choice_dim=z_task,
            z_other_dim=z_other,
            hidden_dim=hidden,
            n_epochs=N_EPOCHS,
            batch_size=BATCH_SIZE,
            lr=LR,
            beta_kl=BETA_KL,
            alpha_choice=ALPHA_TASK,
            device=device,
        )
    except Exception as e:
        logger.warning(f"VAE training failed: {e}")
        return None
    return vae_result["subspace_directions"]


# ---------------------------------------------------------------------------
# Grassmannian distance between subspaces of different dimensionality
# ---------------------------------------------------------------------------

def _grassmannian_distance_mixed_k(V1: np.ndarray, V2: np.ndarray) -> dict:
    """Compute Grassmannian distance between subspaces that may have different k.

    Truncates both to min(k1, k2) columns for a fair comparison on the
    same Grassmannian Gr(k, n).

    Returns dict with geodesic distance, principal angles, and overlap.
    """
    k = min(V1.shape[1], V2.shape[1])
    U1 = V1[:, :k]
    U2 = V2[:, :k]
    # Re-orthogonalize after truncation
    U1, _ = np.linalg.qr(U1)
    U2, _ = np.linalg.qr(U2)
    U1 = U1[:, :k]
    U2 = U2[:, :k]
    return all_subspace_distances(U1, U2)


# ---------------------------------------------------------------------------
# Main run
# ---------------------------------------------------------------------------

def run(max_sessions: int | None = None) -> dict:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    sessions = load_all()
    if max_sessions:
        sessions = sessions[:max_sessions]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"{datetime.now().isoformat()} Starting cross-task Grassmannian experiment "
                f"with {len(sessions)} sessions on {device}")

    # --- Load data: for each region, collect activity + labels for all 4 tasks ---
    # A region-session is usable only if ALL 4 task variables have valid labels
    region_data: dict[str, list[dict]] = {}

    for sess_idx, sess in enumerate(tqdm(sessions, desc="Loading sessions")):
        mouse = str(sess.get("mouse_name", f"mouse_{sess_idx}"))

        # Extract labels for all 4 tasks
        task_labels_map: dict[str, np.ndarray | None] = {}
        for task_name in ALL_TASKS:
            task_labels_map[task_name] = TASK_LABEL_EXTRACTORS[task_name](sess)

        # Skip session if any task has no valid labels
        if any(v is None for v in task_labels_map.values()):
            continue

        regions = list_regions(sess, min_neurons=MIN_NEURONS)
        for region in regions:
            act = get_region_activity(sess, region)
            if act is None or act.shape[1] < MIN_NEURONS:
                continue

            # Find the common valid trial mask across all tasks
            n = act.shape[0]
            for task_name in ALL_TASKS:
                n = min(n, len(task_labels_map[task_name]))
            activity = act[:n, :, TIME_WINDOW].mean(axis=2)

            # Intersect valid trials (non -1) across all tasks
            valid = np.ones(n, dtype=bool)
            for task_name in ALL_TASKS:
                valid &= task_labels_map[task_name][:n] >= 0
            if valid.sum() < MIN_TRIALS_PER_CONDITION * 2:
                continue

            entry = {
                "session_idx": sess_idx,
                "mouse": mouse,
                "activity": activity[valid],
                "n_neurons": int(activity.shape[1]),
                "n_trials": int(valid.sum()),
            }
            for task_name in ALL_TASKS:
                entry[f"labels_{task_name}"] = task_labels_map[task_name][:n][valid]

            if region not in region_data:
                region_data[region] = []
            region_data[region].append(entry)

    logger.info(f"{datetime.now().isoformat()} {len(region_data)} regions with all 4 task variables")

    # --- Per-region: fit subspaces and compute pairwise distances ---
    task_pairs = list(combinations(ALL_TASKS, 2))
    pair_names = [f"{a}-{b}" for a, b in task_pairs]

    region_results = {}
    jsonl_path = RESULTS_DIR / "cross_task_incremental.jsonl"

    # Resume support
    computed_regions: set[str] = set()
    if jsonl_path.exists():
        with open(jsonl_path) as jf:
            for line in jf:
                r = json.loads(line)
                computed_regions.add(r["region"])
                region_results[r["region"]] = r
        logger.info(f"Resuming: loaded {len(computed_regions)} pre-computed regions")

    for region, measurements in tqdm(region_data.items(), desc="Cross-task Grassmannian"):
        if region in computed_regions:
            continue

        # Accumulate per-measurement distances, then average
        lda_pair_dists: dict[str, list[float]] = {pn: [] for pn in pair_names}
        vae_pair_dists: dict[str, list[float]] = {pn: [] for pn in pair_names}
        method_dists: dict[str, list[float]] = {t: [] for t in ALL_TASKS}
        lda_overlaps: dict[str, list[float]] = {pn: [] for pn in pair_names}
        vae_overlaps: dict[str, list[float]] = {pn: [] for pn in pair_names}

        for m in measurements:
            activity = m["activity"]
            n_neurons = m["n_neurons"]

            # Fit subspaces for each task variable using both methods
            lda_subspaces: dict[str, np.ndarray | None] = {}
            vae_subspaces: dict[str, np.ndarray | None] = {}

            for task_name in ALL_TASKS:
                labels = m[f"labels_{task_name}"]
                lda_subspaces[task_name] = _fit_lda_subspace(activity, labels, k=LDA_K)
                vae_subspaces[task_name] = _fit_vae_subspace(activity, labels, z_dim=Z_DIM, device=device)

            # Pairwise Grassmannian distances between task-variable subspaces
            for (t1, t2), pn in zip(task_pairs, pair_names):
                # LDA
                V1_lda, V2_lda = lda_subspaces[t1], lda_subspaces[t2]
                if V1_lda is not None and V2_lda is not None:
                    dists = _grassmannian_distance_mixed_k(V1_lda, V2_lda)
                    lda_pair_dists[pn].append(dists["grassmannian"])
                    lda_overlaps[pn].append(dists["subspace_overlap"])

                # VAE
                V1_vae, V2_vae = vae_subspaces[t1], vae_subspaces[t2]
                if V1_vae is not None and V2_vae is not None:
                    dists = _grassmannian_distance_mixed_k(V1_vae, V2_vae)
                    vae_pair_dists[pn].append(dists["grassmannian"])
                    vae_overlaps[pn].append(dists["subspace_overlap"])

            # Method disagreement: LDA vs VAE for same task variable
            for task_name in ALL_TASKS:
                V_lda = lda_subspaces[task_name]
                V_vae = vae_subspaces[task_name]
                if V_lda is not None and V_vae is not None:
                    dists = _grassmannian_distance_mixed_k(V_lda, V_vae)
                    method_dists[task_name].append(dists["grassmannian"])

        # Aggregate across measurements for this region
        result: dict = {
            "region": region,
            "n_sessions": len(measurements),
            "n_neurons_range": [min(m["n_neurons"] for m in measurements),
                                max(m["n_neurons"] for m in measurements)],
            "n_trials_range": [min(m["n_trials"] for m in measurements),
                               max(m["n_trials"] for m in measurements)],
        }

        # LDA pairwise distances
        for pn in pair_names:
            vals = lda_pair_dists[pn]
            result[f"lda_dist_{pn}"] = float(np.mean(vals)) if vals else None
            result[f"lda_dist_{pn}_std"] = float(np.std(vals)) if len(vals) > 1 else None
            result[f"lda_dist_{pn}_n"] = len(vals)
            ovals = lda_overlaps[pn]
            result[f"lda_overlap_{pn}"] = float(np.mean(ovals)) if ovals else None

        # VAE pairwise distances
        for pn in pair_names:
            vals = vae_pair_dists[pn]
            result[f"vae_dist_{pn}"] = float(np.mean(vals)) if vals else None
            result[f"vae_dist_{pn}_std"] = float(np.std(vals)) if len(vals) > 1 else None
            result[f"vae_dist_{pn}_n"] = len(vals)
            ovals = vae_overlaps[pn]
            result[f"vae_overlap_{pn}"] = float(np.mean(ovals)) if ovals else None

        # Method disagreement per task
        for task_name in ALL_TASKS:
            vals = method_dists[task_name]
            result[f"method_dist_{task_name}"] = float(np.mean(vals)) if vals else None
            result[f"method_dist_{task_name}_n"] = len(vals)

        # Summary: mean pairwise distance (multiplexing index)
        all_lda_dists = [v for vals in lda_pair_dists.values() for v in vals]
        all_vae_dists = [v for vals in vae_pair_dists.values() for v in vals]
        result["lda_mean_pairwise_dist"] = float(np.mean(all_lda_dists)) if all_lda_dists else None
        result["vae_mean_pairwise_dist"] = float(np.mean(all_vae_dists)) if all_vae_dists else None

        region_results[region] = result
        with open(jsonl_path, "a") as jf:
            jf.write(json.dumps(result, default=str) + "\n")

    logger.info(f"{datetime.now().isoformat()} All regions processed")

    # --- Aggregate analyses ---
    analyses = {}

    # 1. Which task pairs have the most orthogonal subspaces (highest distance)?
    pair_avg_lda = {}
    pair_avg_vae = {}
    for pn in pair_names:
        lda_vals = [r[f"lda_dist_{pn}"] for r in region_results.values() if r.get(f"lda_dist_{pn}") is not None]
        vae_vals = [r[f"vae_dist_{pn}"] for r in region_results.values() if r.get(f"vae_dist_{pn}") is not None]
        if lda_vals:
            pair_avg_lda[pn] = {"mean": float(np.mean(lda_vals)), "std": float(np.std(lda_vals)), "n": len(lda_vals)}
        if vae_vals:
            pair_avg_vae[pn] = {"mean": float(np.mean(vae_vals)), "std": float(np.std(vae_vals)), "n": len(vae_vals)}

    analyses["task_pair_separation_lda"] = dict(sorted(pair_avg_lda.items(), key=lambda x: x[1]["mean"], reverse=True))
    analyses["task_pair_separation_vae"] = dict(sorted(pair_avg_vae.items(), key=lambda x: x[1]["mean"], reverse=True))

    # 2. Which regions multiplex the most? (lowest mean pairwise distance = most overlap)
    multiplex_lda = sorted(
        [(r, v["lda_mean_pairwise_dist"]) for r, v in region_results.items()
         if v.get("lda_mean_pairwise_dist") is not None],
        key=lambda x: x[1],
    )
    multiplex_vae = sorted(
        [(r, v["vae_mean_pairwise_dist"]) for r, v in region_results.items()
         if v.get("vae_mean_pairwise_dist") is not None],
        key=lambda x: x[1],
    )
    analyses["most_multiplexed_lda"] = multiplex_lda[:10]
    analyses["least_multiplexed_lda"] = multiplex_lda[-10:] if len(multiplex_lda) >= 10 else multiplex_lda
    analyses["most_multiplexed_vae"] = multiplex_vae[:10]
    analyses["least_multiplexed_vae"] = multiplex_vae[-10:] if len(multiplex_vae) >= 10 else multiplex_vae

    # 3. Does VAE find more separated subspaces than LDA?
    paired_lda_means = []
    paired_vae_means = []
    for r, v in region_results.items():
        lda_m = v.get("lda_mean_pairwise_dist")
        vae_m = v.get("vae_mean_pairwise_dist")
        if lda_m is not None and vae_m is not None:
            paired_lda_means.append(lda_m)
            paired_vae_means.append(vae_m)

    if len(paired_lda_means) >= 5:
        diffs = np.array(paired_vae_means) - np.array(paired_lda_means)
        from scipy.stats import wilcoxon
        try:
            w_stat, w_p = wilcoxon(diffs, alternative="greater")
        except Exception:
            w_stat, w_p = None, None
        analyses["vae_vs_lda_separation"] = {
            "vae_mean_dist": float(np.mean(paired_vae_means)),
            "lda_mean_dist": float(np.mean(paired_lda_means)),
            "mean_diff": float(np.mean(diffs)),
            "n_regions": len(paired_lda_means),
            "n_vae_more_separated": int(np.sum(diffs > 0)),
            "wilcoxon_W": float(w_stat) if w_stat is not None else None,
            "wilcoxon_p": float(w_p) if w_p is not None else None,
            "interpretation": (
                "Positive mean_diff = VAE finds more orthogonal (separated) task subspaces than LDA. "
                "This would suggest VAE disentangles task variables better."
            ),
        }

    # 4. Method disagreement: how different are LDA and VAE subspaces for the same task?
    method_agreement = {}
    for task_name in ALL_TASKS:
        vals = [r[f"method_dist_{task_name}"] for r in region_results.values()
                if r.get(f"method_dist_{task_name}") is not None]
        if vals:
            method_agreement[task_name] = {
                "mean_distance": float(np.mean(vals)),
                "std_distance": float(np.std(vals)),
                "n_regions": len(vals),
            }
    analyses["method_agreement_per_task"] = method_agreement

    # 5. Correlation: region dimensionality (n_neurons) vs multiplexing degree
    neuron_counts = []
    multiplex_scores = []
    for r, v in region_results.items():
        n_range = v.get("n_neurons_range")
        m_score = v.get("lda_mean_pairwise_dist")
        if n_range is not None and m_score is not None:
            neuron_counts.append(np.mean(n_range))
            multiplex_scores.append(m_score)

    if len(neuron_counts) >= 5:
        rho, p = spearmanr(neuron_counts, multiplex_scores)
        analyses["dimensionality_vs_multiplexing"] = {
            "rho": float(rho),
            "p": float(p),
            "n": len(neuron_counts),
            "interpretation": (
                "Negative rho = larger regions have more overlapping (multiplexed) subspaces. "
                "Positive rho = larger regions maintain more orthogonal task representations."
            ),
        }

    # --- Assemble final results ---
    results = {
        "timestamp": datetime.now().isoformat(),
        "n_sessions": len(sessions),
        "n_regions": len(region_data),
        "n_regions_analyzed": len(region_results),
        "device": device,
        "task_variables": ALL_TASKS,
        "task_pairs": pair_names,
        "hyperparameters": {
            "lda_k": LDA_K,
            "vae_z_dim": Z_DIM,
            "vae_z_other_dim": Z_OTHER_DIM,
            "vae_hidden_dim": HIDDEN_DIM,
            "vae_n_epochs": N_EPOCHS,
            "vae_batch_size": BATCH_SIZE,
            "vae_lr": LR,
            "vae_beta_kl": BETA_KL,
            "vae_alpha_task": ALPHA_TASK,
        },
        "region_results": {r: v for r, v in region_results.items()},
        "analyses": analyses,
    }

    out_path = RESULTS_DIR / "cross_task_grassmannian.json"
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
