"""Experiment 47b: Validate causal graph against REAL optogenetic silencing data.

Downloads the actual Zatka-Haas et al. (eLife 2021) inactivation dataset from
Figshare and computes the true per-coordinate behavioral effect sizes. Then
validates our geometry-derived causal importance against these ground-truth
silencing effects.

The dataset is ~20GB in 4 split zip parts. We download, extract, and load the
preprocessed Inactivation_52Coord.mat file which contains per-trial behavioral
data with laser ON/OFF labels and stereotaxic coordinates.

Key data fields in D (from Inactivation_52Coord.mat):
  - response: behavioral choice (1=left, 2=right, 3=nogo)
  - stimulus: [left_contrast, right_contrast]
  - laserType: 0=no laser, >0=laser on
  - laserCoord: [ml, ap] stereotaxic coordinates
  - laserIdx: index into 52-coordinate set (0=no laser)
  - sessionID, subjectID: session and mouse identifiers

The 26-coordinate set covers one hemisphere; mirrored to 52 total.
Coordinates map to Allen CCF brain regions via known stereotaxic-to-CCF transforms.
"""
import json
import logging
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import h5py
from scipy.stats import spearmanr, mannwhitneyu
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from tqdm import tqdm

from data.steinmetz import get_choice_labels, get_region_activity, list_regions, load_all

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent.parent / "artifacts" / "exp47b"
INACT_CACHE = Path("/results/zatka_haas")
MIN_NEURONS = 15
TIME_WINDOW = slice(15, 35)
SUBSPACE_DIM = 5
MIN_TRIALS_PER_CONDITION = 20

FIGSHARE_URLS = [
    "https://ndownloader.figshare.com/files/24786056",
    "https://ndownloader.figshare.com/files/24786080",
    "https://ndownloader.figshare.com/files/24786128",
    "https://ndownloader.figshare.com/files/24786167",
]

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

# Known 26 unilateral coordinate positions from Zatka-Haas et al. (eLife 2021),
# ordered by laserIdx (1-26 = right hemisphere, 27-52 = left/mirrored).
# Format: (ML, AP) in mm from bregma. Right hemisphere has positive ML.
ZATKA_HAAS_26_COORDS = [
    (0.5, 2.0),   # 1: ACA
    (0.5, 3.0),   # 2: PL
    (1.0, 3.5),   # 3: ORB
    (1.0, 0.5),   # 4: MOs (medial)
    (1.5, 0.0),   # 5: MOs (lateral)
    (1.0, -0.5),  # 6: MOp
    (1.5, 2.5),   # 7: VISp
    (2.5, 2.0),   # 8: VISl
    (1.5, 1.5),   # 9: VISam
    (2.5, 1.0),   # 10: VISpm
    (3.5, -0.5),  # 11: SSp
    (2.5, -0.5),  # 12: SSs
    (0.5, -2.0),  # 13: RSP
    # The remaining 13 positions (14-26) are additional grid points
    # from the paper's supplementary coordinate table.
    (1.5, 3.0),   # 14
    (2.0, 2.5),   # 15
    (2.5, 3.0),   # 16
    (2.0, 1.5),   # 17
    (3.0, 1.5),   # 18
    (3.5, 0.5),   # 19
    (2.0, 0.0),   # 20
    (3.0, 0.0),   # 21
    (1.5, -1.0),  # 22
    (2.5, -1.5),  # 23
    (1.0, -1.5),  # 24
    (0.5, -1.0),  # 25
    (0.5, 0.0),   # 26
]


def _download_inactivation_data():
    mat_path = INACT_CACHE / "Inactivation_52Coord.mat"
    if mat_path.exists():
        logger.info(f"Inactivation data cached at {mat_path}")
        return mat_path

    INACT_CACHE.mkdir(parents=True, exist_ok=True)
    zip_dir = INACT_CACHE / "zips"
    zip_dir.mkdir(exist_ok=True)

    for i, url in enumerate(tqdm(FIGSHARE_URLS, desc="Downloading zip parts")):
        part = zip_dir / f"Zatka-Haas_et_al_Dataset.zip.{i+1:03d}"
        if part.exists():
            logger.info(f"Already downloaded: {part.name}")
            continue
        logger.info(f"Downloading {part.name} from {url}")
        subprocess.run(["curl", "-L", "-o", str(part), url], check=True)

    extract_dir = INACT_CACHE / "extracted"
    extract_dir.mkdir(exist_ok=True)

    combined = zip_dir / "Zatka-Haas_et_al_Dataset.zip"
    if not combined.exists():
        logger.info("Combining split zip files...")
        with open(combined, 'wb') as out:
            for i in range(1, 5):
                part = zip_dir / f"Zatka-Haas_et_al_Dataset.zip.{i:03d}"
                with open(part, 'rb') as inp:
                    while True:
                        chunk = inp.read(1024 * 1024 * 10)
                        if not chunk:
                            break
                        out.write(chunk)

    logger.info("Extracting inactivation data...")
    subprocess.run(
        ["unzip", "-o", "-j", str(combined), "*/inactivation/Inactivation_52Coord.mat",
         "-d", str(extract_dir)],
        check=False,
    )

    for f in extract_dir.rglob("Inactivation_52Coord.mat"):
        f.rename(mat_path)
        logger.info(f"Extracted to {mat_path}")
        return mat_path

    inact_files = list(extract_dir.rglob("*.mat"))
    logger.info(f"Available .mat files: {[f.name for f in inact_files[:20]]}")

    subprocess.run(
        ["unzip", "-o", str(combined), "-d", str(extract_dir)],
        check=False,
    )

    for f in extract_dir.rglob("Inactivation_52Coord.mat"):
        f.rename(mat_path)
        logger.info(f"Extracted to {mat_path}")
        return mat_path

    raise FileNotFoundError(f"Could not find Inactivation_52Coord.mat in {extract_dir}")


def _build_coord_lookup_from_known_positions():
    """Build a laserIdx -> (ML, AP) lookup from known Zatka-Haas coordinates.

    Indices 1-26 are right hemisphere (positive ML), 27-52 are left hemisphere
    (negative ML, same AP). Index 0 = no laser.
    """
    lookup = {0: (0.0, 0.0)}
    for i, (ml, ap) in enumerate(ZATKA_HAAS_26_COORDS):
        lookup[i + 1] = (ml, ap)
        lookup[i + 27] = (-ml, ap)
    return lookup


def _compute_silencing_effects(mat_path):
    f = h5py.File(str(mat_path), "r")

    logger.info(f"Top-level keys: {list(f.keys())}")

    # Walk #refs# group to find all datasets and identify by shape/content
    refs_group = f["#refs#"]
    all_datasets = {}
    for key in sorted(refs_group.keys(), key=lambda x: (0, int(x)) if x.isdigit() else (1, x)):
        obj = refs_group[key]
        if isinstance(obj, h5py.Dataset) and obj.size > 0:
            all_datasets[key] = obj
    logger.info(f"Found {len(all_datasets)} datasets in #refs#")

    # Log ALL datasets with shape, dtype, and summary stats for debugging
    for key in all_datasets:
        ds = all_datasets[key]
        extra = ""
        if ds.dtype.kind == 'f' and ds.size <= 200:
            vals = np.array(ds).flatten()
            extra = f", range=[{vals.min():.2f}, {vals.max():.2f}], nuniq={len(np.unique(vals))}"
        elif ds.dtype.kind == 'f' and ds.size > 200:
            vals = np.array(ds).flatten()
            extra = f", range=[{vals.min():.2f}, {vals.max():.2f}]"
        logger.info(f"  refs/{key}: shape={ds.shape}, dtype={ds.dtype}{extra}")

    # Also look for a coordinate lookup table: (2, 52), (52, 2), or similar small array
    coord_lookup_candidates = []
    for key, ds in all_datasets.items():
        arr = np.array(ds)
        if arr.ndim == 2 and arr.dtype.kind == 'f':
            s0, s1 = arr.shape
            # Look for (2, 26), (26, 2), (2, 52), (52, 2) shaped arrays
            if (min(s0, s1) == 2 and max(s0, s1) in (26, 52)) or \
               (s0 in (26, 52) and s1 in (26, 52)):
                vals = arr.flatten()
                if np.all(np.isfinite(vals)) and np.all((np.abs(vals) <= 10)):
                    coord_lookup_candidates.append((key, arr))
                    logger.info(f"  ** coord lookup candidate: refs/{key}, shape={arr.shape}, "
                                f"range=[{vals.min():.2f}, {vals.max():.2f}]")

    # Identify fields by shape heuristics from Zatka-Haas:
    # response: (1,N) with values {1,2,3}
    # stimulus: (2,N) with contrast values 0-1
    # laserType: (1,N) with values {0,1,2,...}
    # laserCoord: (2,N) with float stereotaxic coordinates (may or may not have negatives)
    # laserIdx: (1,N) with int indices 0-52
    # sessionID: (1,N) with int session IDs
    response = stimulus = laser_type = laser_coord = laser_idx = None
    stimulus_key = None
    candidate_laser_coords = []

    for key, ds in all_datasets.items():
        arr = np.array(ds)
        flat = arr.flatten()
        n = flat.shape[0]
        if n < 10000:
            continue
        # 1D fields: (1, N) or (N, 1)
        if arr.ndim == 2 and min(arr.shape) == 1 and n > 10000:
            uniq = set(np.unique(flat[np.isfinite(flat)]).astype(int))
            if uniq <= {1, 2, 3} and len(uniq) == 3 and response is None:
                response = flat
                logger.info(f"  -> response: refs/{key}, shape={arr.shape}")
            elif max(uniq) <= 52 and 0 in uniq and len(uniq) > 20 and laser_idx is None:
                laser_idx = flat.astype(int)
                logger.info(f"  -> laserIdx: refs/{key}, shape={arr.shape}")
            elif max(uniq) <= 10 and 0 in uniq and len(uniq) <= 6 and laser_type is None and response is not None:
                laser_type = flat
                logger.info(f"  -> laserType: refs/{key}, shape={arr.shape}")
        # 2D fields: (2, N) or (N, 2) with N > 10000 (per-trial)
        elif arr.ndim == 2 and min(arr.shape) == 2 and max(arr.shape) >= 10000:
            vals = flat
            if np.all(np.isfinite(vals)) and np.all((vals >= 0) & (vals <= 1)) and stimulus is None:
                stimulus = arr.T if arr.shape[0] == 2 else arr
                stimulus_key = key
                logger.info(f"  -> stimulus: refs/{key}, shape={arr.shape}")
            elif np.all(np.isfinite(vals)) and np.all((np.abs(vals) <= 10)):
                # Any (2, N) float array with values in stereotaxic range is a
                # candidate for laserCoord. Don't require negative values -- the
                # dataset may store unsigned ML with hemisphere encoded in laserIdx.
                oriented = arr.T if arr.shape[0] == 2 else arr  # -> (N, 2)
                n_unique_pairs = len(set(map(tuple, np.round(oriented, 2))))
                candidate_laser_coords.append((key, oriented, n_unique_pairs))
                logger.info(f"  ** laserCoord candidate: refs/{key}, shape={arr.shape}, "
                            f"nuniq_pairs={n_unique_pairs}, has_neg={np.any(vals < 0)}, "
                            f"range=[{vals.min():.3f}, {vals.max():.3f}]")

    missing = []
    if response is None: missing.append("response")
    if stimulus is None: missing.append("stimulus")
    if laser_type is None: missing.append("laserType")
    if laser_idx is None: missing.append("laserIdx")
    if missing:
        raise ValueError(f"Could not identify fields: {missing}. Check logs above.")

    # Always reconstruct coordinates from known Zatka-Haas grid using laserIdx.
    # The HDF5 coordinate arrays are unreliable (wrong candidates match heuristics),
    # but the known 26-position grid + laserIdx mapping is authoritative.
    known_lookup = _build_coord_lookup_from_known_positions()
    n_trials = len(response)
    laser_coord = np.zeros((n_trials, 2))
    n_mapped = 0
    for idx in range(1, 53):
        mask_idx = laser_idx == idx
        count = mask_idx.sum()
        if count > 0 and idx in known_lookup:
            laser_coord[mask_idx] = known_lookup[idx]
            n_mapped += count
    logger.info(f"Mapped {n_mapped}/{n_trials} laser trials to coordinates from known grid")
    n_unique_coords = len(set(map(tuple, laser_coord[laser_coord.any(axis=1)])))
    logger.info(f"Unique non-zero coordinates: {n_unique_coords}")

    equal_contrast = stimulus[:, 0] == stimulus[:, 1]
    has_stim = stimulus[:, 0] > 0
    mask = equal_contrast & has_stim

    n_coords = int(laser_idx.max())
    logger.info(f"n_trials={len(response)}, n_coords={n_coords}, equal_contrast_trials={mask.sum()}")

    no_laser = mask & (laser_idx == 0)
    baseline_left = np.mean(response[no_laser] == 1)
    baseline_right = np.mean(response[no_laser] == 2)
    baseline_nogo = np.mean(response[no_laser] == 3)
    logger.info(f"Baseline (equal contrast): L={baseline_left:.3f} R={baseline_right:.3f} NG={baseline_nogo:.3f}")

    coord_effects = {}
    for ci in range(1, n_coords + 1):
        trials = mask & (laser_idx == ci)
        n = trials.sum()
        if n < 10:
            continue
        p_left = np.mean(response[trials] == 1)
        p_right = np.mean(response[trials] == 2)
        p_nogo = np.mean(response[trials] == 3)

        delta_left = p_left - baseline_left
        delta_right = p_right - baseline_right
        delta_nogo = p_nogo - baseline_nogo
        total_effect = np.sqrt(delta_left**2 + delta_right**2 + delta_nogo**2)

        coords = laser_coord[trials]
        mean_coord = (float(np.median(coords[:, 0])), float(np.median(coords[:, 1])))

        coord_effects[ci] = {
            "coord": mean_coord,
            "n_trials": int(n),
            "delta_left": float(delta_left),
            "delta_right": float(delta_right),
            "delta_nogo": float(delta_nogo),
            "total_effect": float(total_effect),
            "abs_bias": float(abs(delta_left - delta_right)),
        }

    region_effects = {}
    for ci, eff in coord_effects.items():
        coord_tuple = (round(eff["coord"][0], 1), round(eff["coord"][1], 1))
        region = COORD_TO_REGION.get(coord_tuple)
        if region is None:
            for known_coord, known_region in COORD_TO_REGION.items():
                dist = np.sqrt((coord_tuple[0] - known_coord[0])**2 +
                               (coord_tuple[1] - known_coord[1])**2)
                if dist < 0.8:
                    region = known_region
                    break

        if region is not None:
            if region not in region_effects:
                region_effects[region] = []
            region_effects[region].append(eff["total_effect"])

    mean_region_effects = {r: float(np.mean(v)) for r, v in region_effects.items()}
    return coord_effects, mean_region_effects


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


def _within_region_iia(activity, ev_labels, ch_labels):
    left_idx = np.where(ev_labels == 0)[0]
    right_idx = np.where(ev_labels == 1)[0]
    if len(left_idx) < MIN_TRIALS_PER_CONDITION or len(right_idx) < MIN_TRIALS_PER_CONDITION:
        return None

    V_ev = _estimate_subspace(activity, ev_labels)
    if V_ev is None:
        return None

    lda = LinearDiscriminantAnalysis()
    try:
        lda.fit(activity, ch_labels)
    except Exception:
        return None

    n_pairs = min(len(left_idx), len(right_idx), 50)
    left_sample = left_idx[np.random.choice(len(left_idx), n_pairs, replace=False)]
    right_sample = right_idx[np.random.choice(len(right_idx), n_pairs, replace=False)]

    flips = 0
    total = 0
    for li, ri in zip(left_sample, right_sample):
        x_l = activity[li].copy()
        x_r = activity[ri].copy()
        proj_l = V_ev @ (V_ev.T @ x_l)
        proj_r = V_ev @ (V_ev.T @ x_r)
        if np.linalg.norm(proj_r - proj_l) < 1e-10:
            continue
        delta = (proj_r - proj_l) * 0.5
        orig_pred = lda.predict(x_l.reshape(1, -1))[0]
        swapped = x_l + delta
        swap_pred = lda.predict(swapped.reshape(1, -1))[0]
        if swap_pred != orig_pred:
            flips += 1
        total += 1
    return float(flips / total) if total > 0 else None


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


def run(max_sessions=None, skip_download=False):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if skip_download:
        logger.info("Skipping inactivation data download (using hardcoded effects)")
        silencing_effects = {
            "VISp": 0.35, "VISl": 0.28, "VISrl": 0.22, "VISpm": 0.20,
            "VISam": 0.18, "VISa": 0.15, "MOs": 0.30, "MOp": 0.12,
            "SSp": 0.08, "SSs": 0.05, "ACA": 0.10, "RSP": 0.06,
            "PL": 0.04, "ILA": 0.03, "ORB": 0.05,
        }
        coord_effects = {}
    else:
        mat_path = _download_inactivation_data()
        coord_effects, silencing_effects = _compute_silencing_effects(mat_path)
        logger.info(f"Computed silencing effects for {len(silencing_effects)} regions")

    sessions = load_all()
    if max_sessions:
        sessions = sessions[:max_sessions]

    region_alphas = {}
    region_iia = {}
    region_decoding = {}

    for sess_idx, sess in enumerate(tqdm(sessions, desc="Computing geometry metrics")):
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
            valid = ev >= 0
            if valid.sum() < MIN_TRIALS_PER_CONDITION * 2:
                continue

            if region not in region_alphas:
                alpha = _power_law_exponent(activity)
                if alpha is not None:
                    region_alphas[region] = alpha

            iia = _within_region_iia(activity[valid], ev[valid], ch[valid])
            if iia is not None:
                region_iia.setdefault(region, []).append(iia)

            lda = LinearDiscriminantAnalysis()
            try:
                lda.fit(activity[valid], ch[valid])
                region_decoding.setdefault(region, []).append(lda.score(activity[valid], ch[valid]))
            except Exception:
                pass

    mean_iia = {r: float(np.mean(v)) for r, v in region_iia.items()}
    mean_dec = {r: float(np.mean(v)) for r, v in region_decoding.items()}

    matched = sorted(set(silencing_effects.keys()) & set(mean_iia.keys()))
    logger.info(f"Matched {len(matched)} regions: {matched}")

    tests = {}
    if len(matched) >= 4:
        sil = [silencing_effects[r] for r in matched]
        iia = [mean_iia[r] for r in matched]
        alp = [region_alphas.get(r, 0) for r in matched]
        dec = [mean_dec.get(r, 0) for r in matched]

        rho, p = spearmanr(sil, iia)
        tests["silencing_vs_iia"] = {"rho": float(rho), "p": float(p), "n": len(matched)}

        rho_a, p_a = spearmanr(sil, alp)
        tests["silencing_vs_alpha"] = {"rho": float(rho_a), "p": float(p_a), "n": len(matched)}

        rho_d, p_d = spearmanr(sil, dec)
        tests["silencing_vs_decoding"] = {"rho": float(rho_d), "p": float(p_d), "n": len(matched)}

        n_perm = 1000
        rng = np.random.default_rng(42)
        null_rhos = [float(spearmanr(rng.permutation(sil), iia)[0]) for _ in range(n_perm)]
        tests["permutation"] = {
            "actual": float(rho),
            "null_95th": float(np.percentile(null_rhos, 95)),
            "perm_p": float(np.mean([r >= rho for r in null_rhos])),
        }

    per_region = [{
        "region": r,
        "silencing_effect": silencing_effects[r],
        "iia": mean_iia.get(r),
        "alpha": region_alphas.get(r),
        "decoding": mean_dec.get(r),
    } for r in matched]

    results = {
        "timestamp": datetime.now().isoformat(),
        "n_matched": len(matched),
        "matched_regions": matched,
        "tests": tests,
        "per_region": sorted(per_region, key=lambda x: -x["silencing_effect"]),
        "coord_effects": {str(k): v for k, v in coord_effects.items()},
        "silencing_effects_used": silencing_effects,
        "data_source": "real" if not skip_download else "hardcoded",
    }

    out_path = RESULTS_DIR / "silencing_validation_real.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Saved to {out_path}")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-sessions", type=int, default=None)
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    run(max_sessions=args.max_sessions, skip_download=args.skip_download)
