"""Experiment 75: CCF coordinate matching for optogenetic validation.

The current optogenetic validation matches Steinmetz 2019 regions to Zatka-Haas
2021 silencing sites by STRING MATCHING on region names, yielding only n=12.
This experiment uses SPATIAL NEAREST-NEIGHBOR MATCHING in stereotaxic (ML, AP)
coordinate space to increase the matched set to n=25+.

All coordinates are in mm from bregma: ML positive = right, AP positive = anterior.

Region centroids for Steinmetz come from the Allen CCF reference atlas.
Zatka-Haas coordinates come from the known 26-position grid (already in exp47b).

CPU only. <30min.
"""
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent / "results" / "exp75"

# Zatka-Haas 26 unilateral silencing positions (ML, AP) in mm from bregma.
# Right hemisphere (positive ML). Mirrored for left hemisphere.
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

# Approximate (ML, AP) centroids for Steinmetz brain regions from Allen CCF.
# Dorsal cortical regions that could plausibly overlap with photoinhibition sites.
# Subcortical regions (thalamus, hippocampus, etc.) are excluded — laser can't reach them.
STEINMETZ_REGION_COORDS = {
    "ACA": (0.3, 1.8),
    "MOs": (1.0, 0.5),
    "MOp": (1.5, -0.5),
    "SSp": (3.0, -0.5),
    "SSs": (3.5, -1.0),
    "VISp": (2.5, -3.0),
    "VISl": (3.0, -2.5),
    "VISam": (1.5, -2.5),
    "VISpm": (2.0, -3.0),
    "VISrl": (2.5, -2.0),
    "VISal": (3.0, -2.0),
    "RSP": (0.5, -2.5),
    "RSPagl": (0.5, -2.0),
    "RSPd": (0.3, -2.5),
    "RSPv": (0.3, -2.8),
    "PL": (0.3, 2.5),
    "ILA": (0.3, 2.0),
    "ORB": (1.5, 3.0),
    "ORBl": (2.0, 3.0),
    "ORBm": (0.5, 3.0),
    "PTLp": (2.5, -1.5),
    "AUDp": (4.0, -1.5),
    "AUDd": (3.5, -1.0),
    "TEa": (4.5, -1.5),
}


def nearest_neighbor_matching(steinmetz_coords, zatka_coords, max_distance_mm):
    """Match Steinmetz regions to nearest Zatka-Haas silencing sites.

    Returns list of (steinmetz_region, zatka_idx, distance) tuples.
    Each Steinmetz region is matched to at most one Zatka-Haas site.
    """
    s_names = list(steinmetz_coords.keys())
    s_coords = np.array([steinmetz_coords[n] for n in s_names])
    z_coords = np.array(zatka_coords)

    dists = cdist(s_coords, z_coords)
    matches = []
    for i, name in enumerate(s_names):
        j = np.argmin(dists[i])
        d = dists[i, j]
        if d <= max_distance_mm:
            matches.append((name, int(j), float(d)))
    return matches


def run(max_sessions=None):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Import silencing data from exp47b
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "experiments"))
    from exp47b_silencing_real_data import _compute_silencing_effects, _download_inactivation_data

    mat_path = _download_inactivation_data()
    coord_effects, _ = _compute_silencing_effects(mat_path)

    # Build per-Zatka-Haas-index effect sizes
    zh_effects = {}
    for ci, eff in coord_effects.items():
        if ci >= 1 and ci <= 26:
            zh_effects[ci - 1] = eff["total_effect"]

    logger.info(f"Zatka-Haas effect sizes for {len(zh_effects)} / 26 positions")

    # Load Steinmetz IIA values (from existing experiment results)
    from data.steinmetz import get_choice_labels, get_region_activity, list_regions, load_all
    from geometry.distances import cka

    sessions = load_all()
    if max_sessions:
        sessions = sessions[:max_sessions]

    # Compute mean decoding accuracy per region as a proxy for IIA
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from tqdm import tqdm

    region_decoding = {}
    for sess_idx, sess in enumerate(tqdm(sessions, desc="Sessions")):
        labels = get_choice_labels(sess)
        if len(np.unique(labels)) < 2:
            continue
        regions = list_regions(sess, min_neurons=15)
        for region in regions:
            act = get_region_activity(sess, region)
            if act is None or act.shape[1] < 15:
                continue
            n = min(act.shape[0], len(labels))
            activity = act[:n, :, 15:35].mean(axis=2)
            try:
                clf = LogisticRegression(max_iter=1000, solver="lbfgs")
                scores = cross_val_score(clf, activity, labels[:n], cv=5, scoring="accuracy")
                acc = float(scores.mean())
                if region not in region_decoding:
                    region_decoding[region] = []
                region_decoding[region].append(acc)
            except Exception:
                pass

    mean_decoding = {r: float(np.mean(v)) for r, v in region_decoding.items()}
    logger.info(f"Computed decoding accuracy for {len(mean_decoding)} Steinmetz regions")

    # Run matching at multiple distance cutoffs
    cutoffs = [0.5, 0.75, 1.0, 1.5, 2.0]
    results_by_cutoff = {}

    for cutoff in cutoffs:
        matches = nearest_neighbor_matching(STEINMETZ_REGION_COORDS, ZATKA_HAAS_26_COORDS, cutoff)

        matched_pairs = []
        for region, zh_idx, dist in matches:
            if zh_idx in zh_effects and region in mean_decoding:
                matched_pairs.append({
                    "steinmetz_region": region,
                    "zatka_haas_idx": zh_idx + 1,
                    "distance_mm": dist,
                    "silencing_effect": zh_effects[zh_idx],
                    "decoding_accuracy": mean_decoding[region],
                })

        n_matched = len(matched_pairs)
        correlation = None
        if n_matched >= 4:
            sil = [p["silencing_effect"] for p in matched_pairs]
            dec = [p["decoding_accuracy"] for p in matched_pairs]
            rho, p = stats.spearmanr(sil, dec)
            correlation = {
                "spearman_rho": float(rho),
                "p_value": float(p),
                "n_matched": n_matched,
            }

        results_by_cutoff[str(cutoff)] = {
            "cutoff_mm": cutoff,
            "n_matched": n_matched,
            "correlation": correlation,
            "matched_pairs": matched_pairs,
        }

        if correlation:
            print(f"  Cutoff {cutoff}mm: n={n_matched}, rho={correlation['spearman_rho']:.3f} (p={correlation['p_value']:.3e})")
        else:
            print(f"  Cutoff {cutoff}mm: n={n_matched} (too few for correlation)")

    # Also report the old string-matching result for comparison
    string_matched = sorted(set(STEINMETZ_REGION_COORDS.keys()) & set(
        ["VISp", "VISl", "VISam", "VISpm", "MOs", "MOp", "SSp", "SSs", "RSP", "ACA", "PL", "ORB"]))

    summary = {
        "timestamp": datetime.now().isoformat(),
        "n_steinmetz_regions": len(STEINMETZ_REGION_COORDS),
        "n_zatka_haas_positions": 26,
        "n_with_effect_data": len(zh_effects),
        "n_with_decoding": len(mean_decoding),
        "string_matching_n": len(string_matched),
        "string_matching_regions": string_matched,
        "results_by_cutoff": results_by_cutoff,
    }

    with open(RESULTS_DIR / "ccf_coordinate_matching.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Saved to {RESULTS_DIR}")
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
