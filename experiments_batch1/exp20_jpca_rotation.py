"""Experiment 20: jPCA rotation frequency comparison (Steinmetz).

Test whether rotational dynamics frequencies are conserved across animals.
jPCA finds the 2D plane where population activity rotates most strongly.
The rotation frequency is an invariant of the dynamical mechanism.

Two regions with the same rotation frequency implement the same dynamical
motif, even if the rotation plane is oriented differently (Grassmannian
would say "different"). This separates dynamics from geometry.

Also tests null: shuffle temporal bins, rotation should disappear.
"""
import json
import logging
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.linalg import schur
from tqdm import tqdm

from data.steinmetz import get_choice_labels, get_region_activity, list_regions, load_all

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent.parent / "artifacts" / "exp20"
MIN_NEURONS = 15
N_PCS = 6


def _jpca(activity_3d, labels, n_pcs=6):
    """Simplified jPCA: find the plane of maximal rotation.

    Args:
        activity_3d: (trials, neurons, time_bins)
        labels: (trials,) choice labels
        n_pcs: number of PCs to use

    Returns:
        rotation_freq: dominant rotation frequency (rad/bin)
        rotation_strength: R² of skew-symmetric fit
        jpca_plane: (n_neurons, 2) — the rotation plane
    """
    n_trials, n_neurons, n_time = activity_3d.shape

    mean_traj = activity_3d.mean(axis=0)

    from sklearn.decomposition import PCA
    pca = PCA(n_components=min(n_pcs, n_neurons, n_time))
    traj_pc = pca.fit_transform(mean_traj.T)

    if traj_pc.shape[0] < 3:
        return None, None, None

    dx = np.diff(traj_pc, axis=0)
    x = traj_pc[:-1]

    M = np.linalg.lstsq(x, dx, rcond=None)[0]

    M_skew = 0.5 * (M - M.T)
    M_sym = 0.5 * (M + M.T)

    dx_pred_skew = x @ M_skew.T
    ss_skew = np.sum(dx_pred_skew ** 2)
    ss_total = np.sum(dx ** 2)
    rotation_strength = float(ss_skew / ss_total) if ss_total > 0 else 0.0

    eigenvalues = np.linalg.eigvals(M_skew)
    imag_parts = np.abs(eigenvalues.imag)
    if imag_parts.max() > 0:
        rotation_freq = float(imag_parts.max())
    else:
        rotation_freq = 0.0

    idx = np.argsort(-imag_parts)[:2]
    eigvecs = np.linalg.eig(M_skew)[1][:, idx]
    jpca_plane = np.real(pca.components_.T @ eigvecs)

    return rotation_freq, rotation_strength, jpca_plane


def run(max_sessions: int | None = None):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    sessions = load_all()
    if max_sessions:
        sessions = sessions[:max_sessions]

    region_data = {}

    for sess_idx, sess in enumerate(tqdm(sessions, desc="Sessions")):
        labels = get_choice_labels(sess)
        if len(np.unique(labels)) < 2:
            continue

        mouse = str(sess.get("mouse_name", f"mouse_{sess_idx}"))
        regions = list_regions(sess, min_neurons=MIN_NEURONS)

        for region in regions:
            act = get_region_activity(sess, region)
            if act is None or act.shape[1] < MIN_NEURONS:
                continue

            n = min(act.shape[0], len(labels))
            activity_3d = act[:n]

            try:
                freq, strength, plane = _jpca(activity_3d, labels[:n], n_pcs=N_PCS)
                if freq is None:
                    continue

                shuffled_3d = activity_3d.copy()
                for trial in range(n):
                    perm = np.random.permutation(shuffled_3d.shape[2])
                    shuffled_3d[trial] = shuffled_3d[trial][:, perm]
                freq_shuf, strength_shuf, _ = _jpca(shuffled_3d, labels[:n], n_pcs=N_PCS)

                if region not in region_data:
                    region_data[region] = []
                region_data[region].append({
                    "session_idx": sess_idx,
                    "mouse": mouse,
                    "rotation_freq": freq,
                    "rotation_strength": strength,
                    "null_freq": freq_shuf,
                    "null_strength": strength_shuf,
                    "n_neurons": act.shape[1],
                })
            except Exception as e:
                logger.warning(f"Failed {mouse}/{region}: {e}")

    pairs = []
    for region, measurements in tqdm(region_data.items(), desc="Regions"):
        if len(measurements) < 2:
            continue

        for (i, j) in combinations(range(len(measurements)), 2):
            m1, m2 = measurements[i], measurements[j]
            freq_diff = abs(m1["rotation_freq"] - m2["rotation_freq"])
            strength_diff = abs(m1["rotation_strength"] - m2["rotation_strength"])

            pairs.append({
                "region": region,
                "mouse_1": m1["mouse"],
                "mouse_2": m2["mouse"],
                "same_mouse": m1["mouse"] == m2["mouse"],
                "freq_1": m1["rotation_freq"],
                "freq_2": m2["rotation_freq"],
                "freq_diff": freq_diff,
                "strength_1": m1["rotation_strength"],
                "strength_2": m2["rotation_strength"],
                "strength_diff": strength_diff,
            })

    region_summaries = {}
    for region, measurements in region_data.items():
        freqs = [m["rotation_freq"] for m in measurements]
        strengths = [m["rotation_strength"] for m in measurements]
        null_strengths = [m["null_strength"] for m in measurements if m["null_strength"] is not None]

        region_summaries[region] = {
            "n_sessions": len(measurements),
            "mean_freq": float(np.mean(freqs)),
            "std_freq": float(np.std(freqs)),
            "cv_freq": float(np.std(freqs) / np.mean(freqs)) if np.mean(freqs) > 0 else None,
            "mean_strength": float(np.mean(strengths)),
            "mean_null_strength": float(np.mean(null_strengths)) if null_strengths else None,
            "rotation_above_null": float(np.mean(strengths)) > float(np.mean(null_strengths)) * 1.5 if null_strengths else None,
        }

    results = {
        "timestamp": datetime.now().isoformat(),
        "n_regions": len(region_data),
        "n_pairs": len(pairs),
        "pairs": pairs,
        "region_summaries": region_summaries,
    }

    out_path = RESULTS_DIR / "jpca_rotation.json"
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
