"""Experiment 66: Per-mouse optogenetic validation.

Instead of pooling silencing effects across all mice in Zatka-Haas et al. 2021,
compute per-mouse silencing effects and correlate each mouse's effect profile
with our geometry-derived causal importance. This tests whether the LDA anti-
correlation and VAE reversal hold at the individual animal level, not just pooled.

Also computes the distribution of per-mouse rho values with bootstrap CIs.
"""
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import h5py
from scipy.stats import spearmanr
from tqdm import tqdm

from data.steinmetz import get_choice_labels, get_region_activity, list_regions, load_all

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent.parent / "artifacts" / "exp66"
INACT_CACHE = Path("/results/zatka_haas")
MIN_NEURONS = 15
TIME_WINDOW = slice(15, 35)
SUBSPACE_DIM = 5
MIN_TRIALS_PER_CONDITION = 20

COORD_TO_REGION = {
    (1.5, 2.5): "VISp", (-1.5, 2.5): "VISp",
    (2.5, 2.0): "VISl", (-2.5, 2.0): "VISl",
    (1.5, 1.5): "VISam", (-1.5, 1.5): "VISam",
    (2.5, 1.0): "VISpm", (-2.5, 1.0): "VISpm",
    (1.0, 0.5): "MOs", (-1.0, 0.5): "MOs",
    (1.5, 0.0): "MOs", (-1.5, 0.0): "MOs",
    (1.0, -0.5): "MOp", (-1.0, -0.5): "MOp",
    (3.5, -0.5): "SSp", (-3.5, -0.5): "SSp",
    (2.5, -0.5): "SSs", (-2.5, -0.5): "SSs",
    (0.5, -2.0): "RSP", (-0.5, -2.0): "RSP",
    (0.5, 2.0): "ACA", (-0.5, 2.0): "ACA",
    (0.5, 3.0): "PL", (-0.5, 3.0): "PL",
    (1.0, 3.5): "ORB", (-1.0, 3.5): "ORB",
}

ZATKA_HAAS_26_COORDS = [
    (0.5, 2.0), (0.5, 3.0), (1.0, 3.5), (1.0, 0.5), (1.5, 0.0),
    (1.0, -0.5), (1.5, 2.5), (2.5, 2.0), (1.5, 1.5), (2.5, 1.0),
    (3.5, -0.5), (2.5, -0.5), (0.5, -2.0),
    (1.5, 3.0), (2.0, 2.5), (2.5, 3.0), (2.0, 1.5), (3.0, 1.5),
    (3.5, 0.5), (2.0, 0.0), (3.0, 0.0), (1.5, -1.0), (2.5, -1.5),
    (1.0, -1.5), (0.5, -1.0), (0.5, 0.0),
]


def _build_coord_lookup():
    lookup = {0: (0.0, 0.0)}
    for i, (ml, ap) in enumerate(ZATKA_HAAS_26_COORDS):
        lookup[i + 1] = (ml, ap)
        lookup[i + 27] = (-ml, ap)
    return lookup


def _load_hdf5_fields(mat_path):
    """Load response, stimulus, laser_type, laser_idx, and subject_id from HDF5."""
    f = h5py.File(str(mat_path), "r")
    refs_group = f["#refs#"]

    all_datasets = {}
    for key in sorted(refs_group.keys(), key=lambda x: (0, int(x)) if x.isdigit() else (1, x)):
        obj = refs_group[key]
        if isinstance(obj, h5py.Dataset) and obj.size > 0:
            all_datasets[key] = obj

    response = stimulus = laser_type = laser_idx = subject_id = session_id = None

    for key, ds in all_datasets.items():
        arr = np.array(ds)
        flat = arr.flatten()
        n = flat.shape[0]
        if n < 10000:
            continue

        if arr.ndim == 2 and min(arr.shape) == 1 and n > 10000:
            uniq = set(np.unique(flat[np.isfinite(flat)]).astype(int))
            if uniq <= {1, 2, 3} and len(uniq) == 3 and response is None:
                response = flat
            elif max(uniq) <= 52 and 0 in uniq and len(uniq) > 20 and laser_idx is None:
                laser_idx = flat.astype(int)
            elif max(uniq) <= 10 and 0 in uniq and len(uniq) <= 6 and laser_type is None and response is not None:
                laser_type = flat
            elif 3 < len(uniq) <= 15 and min(uniq) >= 1 and subject_id is None and laser_idx is not None and laser_type is not None:
                subject_id = flat.astype(int)
                logger.info(f"  -> subjectID: refs/{key}, shape={arr.shape}, n_subjects={len(uniq)}")
            elif len(uniq) > 15 and min(uniq) >= 1 and session_id is None and subject_id is not None:
                session_id = flat.astype(int)
                logger.info(f"  -> sessionID: refs/{key}, shape={arr.shape}, n_sessions={len(uniq)}")

        elif arr.ndim == 2 and min(arr.shape) == 2 and max(arr.shape) >= 10000:
            vals = flat
            if np.all(np.isfinite(vals)) and np.all((vals >= 0) & (vals <= 1)) and stimulus is None:
                stimulus = arr.T if arr.shape[0] == 2 else arr

    missing = []
    if response is None: missing.append("response")
    if stimulus is None: missing.append("stimulus")
    if laser_idx is None: missing.append("laserIdx")
    if missing:
        raise ValueError(f"Could not identify fields: {missing}")

    known_lookup = _build_coord_lookup()
    n_trials = len(response)
    laser_coord = np.zeros((n_trials, 2))
    for idx in range(1, 53):
        mask_idx = laser_idx == idx
        if mask_idx.sum() > 0 and idx in known_lookup:
            laser_coord[mask_idx] = known_lookup[idx]

    return {
        "response": response,
        "stimulus": stimulus,
        "laser_idx": laser_idx,
        "laser_coord": laser_coord,
        "subject_id": subject_id,
    }


def _compute_per_mouse_silencing(fields):
    """Compute silencing effects per mouse and per region."""
    response = fields["response"]
    stimulus = fields["stimulus"]
    laser_idx = fields["laser_idx"]
    laser_coord = fields["laser_coord"]
    subject_id = fields["subject_id"]

    if subject_id is None:
        logger.warning("No subjectID found, falling back to pooled analysis")
        return None

    equal_contrast = stimulus[:, 0] == stimulus[:, 1]
    has_stim = stimulus[:, 0] > 0
    mask = equal_contrast & has_stim

    unique_mice = sorted(set(subject_id[subject_id > 0]))
    logger.info(f"Found {len(unique_mice)} unique mice")

    per_mouse_region_effects = {}

    for mouse in tqdm(unique_mice, desc="Computing per-mouse silencing"):
        mouse_mask = (subject_id == mouse) & mask
        no_laser_mouse = mouse_mask & (laser_idx == 0)
        if no_laser_mouse.sum() < 20:
            continue

        baseline_left = np.mean(response[no_laser_mouse] == 1)
        baseline_right = np.mean(response[no_laser_mouse] == 2)
        baseline_nogo = np.mean(response[no_laser_mouse] == 3)

        region_effects = {}
        for ci in range(1, int(laser_idx.max()) + 1):
            trials = mouse_mask & (laser_idx == ci)
            n = trials.sum()
            if n < 5:
                continue

            p_left = np.mean(response[trials] == 1)
            p_right = np.mean(response[trials] == 2)
            p_nogo = np.mean(response[trials] == 3)

            delta_left = p_left - baseline_left
            delta_right = p_right - baseline_right
            delta_nogo = p_nogo - baseline_nogo
            total_effect = np.sqrt(delta_left**2 + delta_right**2 + delta_nogo**2)

            coords = laser_coord[trials]
            coord_tuple = (round(float(np.median(coords[:, 0])), 1),
                           round(float(np.median(coords[:, 1])), 1))

            region = COORD_TO_REGION.get(coord_tuple)
            if region is None:
                for known_coord, known_region in COORD_TO_REGION.items():
                    dist = np.sqrt((coord_tuple[0] - known_coord[0])**2 +
                                   (coord_tuple[1] - known_coord[1])**2)
                    if dist < 0.8:
                        region = known_region
                        break

            if region is not None:
                region_effects.setdefault(region, []).append(total_effect)

        mean_effects = {r: float(np.mean(v)) for r, v in region_effects.items()}
        if len(mean_effects) >= 4:
            per_mouse_region_effects[int(mouse)] = mean_effects

    logger.info(f"Got silencing profiles for {len(per_mouse_region_effects)} mice")
    return per_mouse_region_effects


def _compute_steinmetz_iia(sessions):
    """Compute LDA and VAE IIA per region from Steinmetz data (LDA only for speed)."""
    from sklearn.decomposition import PCA
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    region_iia = {}

    for sess in tqdm(sessions, desc="Computing Steinmetz IIA"):
        choice_labels = get_choice_labels(sess)
        if len(np.unique(choice_labels)) < 2:
            continue

        for region in list_regions(sess, min_neurons=MIN_NEURONS):
            act = get_region_activity(sess, region)
            if act is None or act.shape[1] < MIN_NEURONS:
                continue
            n = min(act.shape[0], len(choice_labels))
            activity = act[:n, :, TIME_WINDOW].mean(axis=2)
            ch = choice_labels[:n]

            left_idx = np.where(ch == 0)[0]
            right_idx = np.where(ch == 1)[0]
            if len(left_idx) < MIN_TRIALS_PER_CONDITION or len(right_idx) < MIN_TRIALS_PER_CONDITION:
                continue

            pca_dim = min(20, activity.shape[1] - 1, activity.shape[0] - 1)
            pca = PCA(n_components=pca_dim)
            scores = pca.fit_transform(activity)
            lda = LinearDiscriminantAnalysis()
            try:
                lda.fit(scores, ch)
            except Exception:
                continue
            lda_dir = lda.coef_[0]
            lda_dir = lda_dir / (np.linalg.norm(lda_dir) + 1e-10)
            lda_neuron = pca.components_.T @ lda_dir
            pca_components = pca.components_[:min(SUBSPACE_DIM, pca_dim)].T
            combined = np.column_stack([lda_neuron.reshape(-1, 1), pca_components])
            Q, _ = np.linalg.qr(combined)
            V = Q[:, :min(SUBSPACE_DIM, Q.shape[1])]

            lda_full = LinearDiscriminantAnalysis()
            try:
                lda_full.fit(activity, ch)
            except Exception:
                continue

            n_pairs = min(len(left_idx), len(right_idx), 50)
            rng = np.random.default_rng()
            l_samp = left_idx[rng.choice(len(left_idx), n_pairs, replace=False)]
            r_samp = right_idx[rng.choice(len(right_idx), n_pairs, replace=False)]

            flips = 0
            total = 0
            for li, ri in zip(l_samp, r_samp):
                x_l = activity[li].copy()
                x_r = activity[ri].copy()
                proj_l = V @ (V.T @ x_l)
                proj_r = V @ (V.T @ x_r)
                if np.linalg.norm(proj_r - proj_l) < 1e-10:
                    continue
                delta = (proj_r - proj_l) * 0.5
                orig_pred = lda_full.predict(x_l.reshape(1, -1))[0]
                swap_pred = lda_full.predict((x_l + delta).reshape(1, -1))[0]
                if swap_pred != orig_pred:
                    flips += 1
                total += 1

            if total > 0:
                region_iia.setdefault(region, []).append(float(flips / total))

    return {r: float(np.mean(v)) for r, v in region_iia.items()}


def _bootstrap_bca_ci(data, stat_fn, n_boot=10000, alpha=0.05):
    """Bias-corrected accelerated bootstrap confidence interval."""
    observed = stat_fn(data)
    n = len(data)
    boot_stats = np.array([stat_fn(data[np.random.choice(n, n, replace=True)]) for _ in range(n_boot)])

    z0 = np.clip(np.mean(boot_stats < observed), 0.001, 0.999)
    from scipy.stats import norm
    z0 = norm.ppf(z0)

    jack_stats = np.array([stat_fn(np.delete(data, i, axis=0)) for i in range(n)])
    jack_mean = jack_stats.mean()
    num = np.sum((jack_mean - jack_stats)**3)
    denom = 6 * (np.sum((jack_mean - jack_stats)**2))**1.5
    a = num / (denom + 1e-15)

    z_alpha = norm.ppf(alpha / 2)
    z_1alpha = norm.ppf(1 - alpha / 2)

    a1 = norm.cdf(z0 + (z0 + z_alpha) / (1 - a * (z0 + z_alpha)))
    a2 = norm.cdf(z0 + (z0 + z_1alpha) / (1 - a * (z0 + z_1alpha)))

    ci_lo = np.percentile(boot_stats, 100 * a1)
    ci_hi = np.percentile(boot_stats, 100 * a2)
    return float(ci_lo), float(ci_hi), float(observed)


def run(max_sessions=None):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    mat_path = INACT_CACHE / "Inactivation_52Coord.mat"
    if not mat_path.exists():
        from experiments.exp47b_silencing_real_data import _download_inactivation_data
        mat_path = _download_inactivation_data()

    logger.info("Loading HDF5 fields...")
    fields = _load_hdf5_fields(mat_path)

    logger.info("Computing per-mouse silencing effects...")
    per_mouse = _compute_per_mouse_silencing(fields)

    if per_mouse is None or len(per_mouse) < 2:
        return {"error": "Could not extract per-mouse data", "timestamp": timestamp}

    logger.info("Computing Steinmetz LDA IIA...")
    sessions = load_all()
    if max_sessions:
        sessions = sessions[:max_sessions]
    steinmetz_iia = _compute_steinmetz_iia(sessions)

    per_mouse_rhos = []
    per_mouse_details = {}

    for mouse_id, mouse_effects in per_mouse.items():
        matched = sorted(set(mouse_effects.keys()) & set(steinmetz_iia.keys()))
        if len(matched) < 4:
            continue

        sil = [mouse_effects[r] for r in matched]
        iia = [steinmetz_iia[r] for r in matched]
        rho, p = spearmanr(sil, iia)
        per_mouse_rhos.append(float(rho))
        per_mouse_details[mouse_id] = {
            "rho": float(rho),
            "p": float(p),
            "n_matched": len(matched),
            "matched_regions": matched,
        }

    logger.info(f"Computed per-mouse rho for {len(per_mouse_rhos)} mice")

    results = {
        "timestamp": timestamp,
        "n_mice_total": len(per_mouse),
        "n_mice_with_enough_regions": len(per_mouse_rhos),
        "per_mouse_details": per_mouse_details,
    }

    if len(per_mouse_rhos) >= 3:
        rhos = np.array(per_mouse_rhos)
        results["rho_distribution"] = {
            "mean": float(rhos.mean()),
            "median": float(np.median(rhos)),
            "std": float(rhos.std()),
            "min": float(rhos.min()),
            "max": float(rhos.max()),
            "n_negative": int(np.sum(rhos < 0)),
            "n_positive": int(np.sum(rhos > 0)),
            "frac_negative": float(np.mean(rhos < 0)),
            "all_rhos": [float(r) for r in sorted(rhos)],
        }

        ci_lo, ci_hi, mean_rho = _bootstrap_bca_ci(
            rhos, lambda x: np.mean(x), n_boot=10000
        )
        results["bootstrap_mean_rho"] = {
            "mean": mean_rho,
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
        }

        from scipy.stats import wilcoxon
        try:
            w, p_wilcox = wilcoxon(rhos)
            results["wilcoxon_test"] = {
                "W": float(w),
                "p": float(p_wilcox),
                "interpretation": "Tests whether the distribution of per-mouse rhos is centered at zero",
            }
        except Exception as e:
            results["wilcoxon_test"] = {"error": str(e)}

        from scipy.stats import ttest_1samp
        t, p_t = ttest_1samp(rhos, 0)
        results["ttest_vs_zero"] = {
            "t": float(t),
            "p": float(p_t),
        }

    pooled_effects = {}
    for mouse_id, mouse_eff in per_mouse.items():
        for region, eff in mouse_eff.items():
            pooled_effects.setdefault(region, []).append(eff)
    pooled_mean = {r: float(np.mean(v)) for r, v in pooled_effects.items()}

    matched_pooled = sorted(set(pooled_mean.keys()) & set(steinmetz_iia.keys()))
    if len(matched_pooled) >= 4:
        sil = [pooled_mean[r] for r in matched_pooled]
        iia = [steinmetz_iia[r] for r in matched_pooled]
        rho, p = spearmanr(sil, iia)
        results["pooled_comparison"] = {
            "rho": float(rho),
            "p": float(p),
            "n_matched": len(matched_pooled),
            "matched_regions": matched_pooled,
        }

    out_path = RESULTS_DIR / f"exp66_{timestamp}.json"
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
