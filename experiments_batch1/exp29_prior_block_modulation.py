"""Experiment 29: Prior-block modulation of geometric type.

The Steinmetz dataset has contrast-varying trials. We approximate "prior blocks"
by binning trials into high-contrast (easy, strong prior) vs low-contrast
(hard, weak prior) conditions, then ask: does geometric type change between
conditions?

If geometric type is stable across conditions → regions have intrinsic
computational identities (circuit identity).
If geometric type shifts → the same circuit operates in different geometric
regimes depending on cognitive context (circuit mode).

Either result is meaningful and publishable.
"""
import json
import logging
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.spatial import procrustes
from scipy.stats import spearmanr, wilcoxon
from sklearn.decomposition import PCA
from tqdm import tqdm

from data.steinmetz import get_choice_labels, get_region_activity, list_regions, load_all
from geometry.distances import cka

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent.parent / "artifacts" / "exp29"
MIN_NEURONS = 15
UMAP_DIM = 5
TIME_WINDOW = slice(15, 35)
MIN_TRIALS_PER_CONDITION = 30


def _umap_embed(activity, n_components=5):
    from umap import UMAP
    reducer = UMAP(n_components=n_components, n_neighbors=15, min_dist=0.1, random_state=42)
    return reducer.fit_transform(activity)


def _procrustes_distance(X, Y):
    n = min(X.shape[0], Y.shape[0])
    if n < 5:
        return None
    _, _, disparity = procrustes(X[:n], Y[:n])
    return float(disparity)


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


def _effective_dim(activity):
    pca = PCA(n_components=min(50, activity.shape[1], activity.shape[0]))
    pca.fit(activity)
    ev = pca.explained_variance_
    ev = ev[ev > 0]
    return float((ev.sum() ** 2) / (ev ** 2).sum())


def _split_by_difficulty(sess):
    """Split trials into easy (high contrast) and hard (low contrast) conditions.

    Easy: max(contrast_left, contrast_right) >= 0.5
    Hard: max(contrast_left, contrast_right) <= 0.25 (excluding zero-contrast)
    """
    cl = sess.get("contrast_left", np.array([]))
    cr = sess.get("contrast_right", np.array([]))
    if len(cl) == 0 or len(cr) == 0:
        return None, None

    n_trials = sess["spks"].shape[2]
    n = min(n_trials, len(cl), len(cr))
    cl, cr = cl[:n], cr[:n]

    max_contrast = np.maximum(np.abs(cl), np.abs(cr))
    easy_mask = max_contrast >= 0.5
    hard_mask = (max_contrast > 0) & (max_contrast <= 0.25)

    return easy_mask, hard_mask


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

        easy_mask, hard_mask = _split_by_difficulty(sess)
        if easy_mask is None:
            continue

        n = min(len(labels), len(easy_mask))
        easy_mask, hard_mask = easy_mask[:n], hard_mask[:n]

        if easy_mask.sum() < MIN_TRIALS_PER_CONDITION or hard_mask.sum() < MIN_TRIALS_PER_CONDITION:
            continue

        mouse = str(sess.get("mouse_name", f"mouse_{sess_idx}"))
        regions = list_regions(sess, min_neurons=MIN_NEURONS)

        for region in regions:
            act = get_region_activity(sess, region)
            if act is None or act.shape[1] < MIN_NEURONS:
                continue

            act_2d = act[:n, :, TIME_WINDOW].mean(axis=2)
            easy_act = act_2d[easy_mask]
            hard_act = act_2d[hard_mask]

            if region not in region_data:
                region_data[region] = []
            region_data[region].append({
                "session_idx": sess_idx,
                "mouse": mouse,
                "easy_activity": easy_act,
                "hard_activity": hard_act,
                "full_activity": act_2d,
                "n_easy": int(easy_mask.sum()),
                "n_hard": int(hard_mask.sum()),
                "n_neurons": act_2d.shape[1],
            })

    region_results = {}
    alpha_shifts = []
    dim_shifts = []
    cka_shifts = []
    proc_shifts = []
    region_alphas = []

    for region, measurements in tqdm(region_data.items(), desc="Regions"):
        if len(measurements) < 1:
            continue

        easy_alphas = []
        hard_alphas = []
        easy_dims = []
        hard_dims = []
        within_ckas = []
        within_procs = []

        for m in measurements:
            a_easy = _power_law_exponent(m["easy_activity"])
            a_hard = _power_law_exponent(m["hard_activity"])
            d_easy = _effective_dim(m["easy_activity"])
            d_hard = _effective_dim(m["hard_activity"])

            if a_easy is not None:
                easy_alphas.append(a_easy)
            if a_hard is not None:
                hard_alphas.append(a_hard)
            easy_dims.append(d_easy)
            hard_dims.append(d_hard)

            n_shared = min(m["n_easy"], m["n_hard"])
            within_cka = cka(m["easy_activity"][:n_shared], m["hard_activity"][:n_shared])
            within_ckas.append(within_cka)

            try:
                e_easy = _umap_embed(m["easy_activity"], n_components=min(UMAP_DIM, m["n_neurons"] - 1))
                e_hard = _umap_embed(m["hard_activity"], n_components=min(UMAP_DIM, m["n_neurons"] - 1))
                within_proc = _procrustes_distance(e_easy, e_hard)
                if within_proc is not None:
                    within_procs.append(within_proc)
            except Exception:
                pass

        full_alphas = [_power_law_exponent(m["full_activity"]) for m in measurements]
        full_alphas = [a for a in full_alphas if a is not None]
        full_alpha = float(np.mean(full_alphas)) if full_alphas else None

        alpha_shift = None
        if easy_alphas and hard_alphas:
            alpha_shift = float(np.mean(hard_alphas) - np.mean(easy_alphas))

        dim_shift = float(np.mean(hard_dims) - np.mean(easy_dims)) if hard_dims and easy_dims else None

        region_results[region] = {
            "n_sessions": len(measurements),
            "full_alpha": full_alpha,
            "easy_alpha_mean": float(np.mean(easy_alphas)) if easy_alphas else None,
            "hard_alpha_mean": float(np.mean(hard_alphas)) if hard_alphas else None,
            "alpha_shift": alpha_shift,
            "easy_dim_mean": float(np.mean(easy_dims)),
            "hard_dim_mean": float(np.mean(hard_dims)),
            "dim_shift": dim_shift,
            "within_cka_mean": float(np.mean(within_ckas)) if within_ckas else None,
            "within_procrustes_mean": float(np.mean(within_procs)) if within_procs else None,
        }

        if full_alpha is not None:
            region_alphas.append(full_alpha)
            if alpha_shift is not None:
                alpha_shifts.append(alpha_shift)
            if dim_shift is not None:
                dim_shifts.append(dim_shift)
            if within_ckas:
                cka_shifts.append(np.mean(within_ckas))
            if within_procs:
                proc_shifts.append(np.mean(within_procs))

    stability_tests = {}

    if len(alpha_shifts) >= 4:
        rho, p = spearmanr(region_alphas[:len(alpha_shifts)], [abs(s) for s in alpha_shifts])
        stability_tests["alpha_vs_alpha_shift_magnitude"] = {
            "rho": float(rho), "p": float(p), "n": len(alpha_shifts),
            "interpretation": "Does spectral structure predict how much geometric type shifts with task difficulty?"
        }

    if len(alpha_shifts) >= 4:
        try:
            stat, p = wilcoxon(alpha_shifts)
            stability_tests["alpha_shift_wilcoxon"] = {
                "statistic": float(stat), "p": float(p), "n": len(alpha_shifts),
                "mean_shift": float(np.mean(alpha_shifts)),
                "interpretation": "Is there a systematic direction to alpha shifts (easy→hard)?"
            }
        except Exception:
            pass

    if len(cka_shifts) >= 4 and len(region_alphas) >= 4:
        n = min(len(cka_shifts), len(region_alphas))
        rho, p = spearmanr(region_alphas[:n], cka_shifts[:n])
        stability_tests["alpha_vs_within_cka"] = {
            "rho": float(rho), "p": float(p), "n": n,
            "interpretation": "Do high-alpha regions have lower within-condition CKA (less stable linear structure)?"
        }

    summary = {
        "n_regions": len(region_results),
        "mean_alpha_shift": float(np.mean(alpha_shifts)) if alpha_shifts else None,
        "mean_dim_shift": float(np.mean(dim_shifts)) if dim_shifts else None,
        "mean_within_cka": float(np.mean(cka_shifts)) if cka_shifts else None,
        "mean_within_procrustes": float(np.mean(proc_shifts)) if proc_shifts else None,
        "verdict": (
            "stable" if alpha_shifts and abs(np.mean(alpha_shifts)) < 0.1
            else "shifting" if alpha_shifts and abs(np.mean(alpha_shifts)) >= 0.1
            else "insufficient_data"
        ),
    }

    results = {
        "timestamp": datetime.now().isoformat(),
        "n_sessions": len(sessions),
        "n_regions": len(region_data),
        "region_results": region_results,
        "stability_tests": stability_tests,
        "summary": summary,
    }

    out_path = RESULTS_DIR / "prior_block_modulation.json"
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
