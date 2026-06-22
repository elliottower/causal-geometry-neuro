"""Experiment 9: Multiple realization test (Steinmetz).

Steinmetz records 42 regions, many encoding choice with similar decoding accuracy.
For region pairs with matched decoding accuracy but different anatomical connectivity,
test whether Grassmannian distance and holonomy similarity are smaller or larger
than predicted by CKA.

Prediction: CKA is uniform (behaviorally equivalent), but Grassmannian distance
and holonomy discriminate mechanism identity — demonstrating that CKA and
mechanism identity are genuinely dissociable.

Falsification: CKA and d_G are equally discriminative, suggesting no empirical
difference between representational similarity and mechanism identity.
"""
import json
import logging
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from tqdm import tqdm

from data.steinmetz import get_choice_labels, get_region_activity, list_regions, load_all
from geometry.distances import cka, grassmannian_distance
from geometry.holonomy import estimate_holonomy, holonomy_angle, holonomy_distance
from geometry.subspace import fit_lda_subspace

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent.parent / "artifacts" / "exp9"
SUBSPACE_K = 5
MIN_NEURONS = 15
ACCURACY_MATCH_TOL = 0.10
TIME_WINDOW = slice(15, 35)  # deliberation period (10ms bins)
HOLONOMY_TIME_POINTS = [slice(0, 10), slice(15, 25), slice(25, 35), slice(35, 45), slice(45, 50)]


def _decode_accuracy(activity: np.ndarray, labels: np.ndarray) -> float:
    clf = LogisticRegression(max_iter=1000, solver="lbfgs")
    scores = cross_val_score(clf, activity, labels, cv=min(5, len(np.unique(labels))), scoring="accuracy")
    return float(scores.mean())


def run(max_sessions: int | None = None):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    sessions = load_all()
    if max_sessions:
        sessions = sessions[:max_sessions]

    all_region_data = []

    for sess_idx, sess in enumerate(tqdm(sessions, desc="Sessions")):
        mouse = str(sess.get("mouse_name", f"mouse_{sess_idx}"))
        labels = get_choice_labels(sess)
        if len(np.unique(labels)) < 2:
            continue

        regions = list_regions(sess, min_neurons=MIN_NEURONS)

        for region in regions:
            act = get_region_activity(sess, region)
            if act is None or act.shape[1] < MIN_NEURONS:
                continue

            n = min(act.shape[0], len(labels))
            act = act[:n]
            activity_delib = act[:, :, TIME_WINDOW].mean(axis=2)

            try:
                acc = _decode_accuracy(activity_delib, labels[:n])
                k = min(SUBSPACE_K, act.shape[1] - 1)
                U = fit_lda_subspace(activity_delib, labels[:n], k=k)

                time_acts = [act[:, :, ts].mean(axis=2) for ts in HOLONOMY_TIME_POINTS]
                H = estimate_holonomy(time_acts, labels[:n], k=k)
                h_angle = holonomy_angle(H)

                all_region_data.append({
                    "session_idx": sess_idx,
                    "mouse": mouse,
                    "region": region,
                    "accuracy": acc,
                    "subspace": U,
                    "activity": activity_delib,
                    "holonomy": H,
                    "holonomy_angle": h_angle,
                })
            except Exception as e:
                logger.warning(f"Failed {mouse}/{region}: {e}")

    logger.info(f"Total region measurements: {len(all_region_data)}")

    pairs = []
    for (i, j) in tqdm(list(combinations(range(len(all_region_data)), 2)), desc="Pairs"):
        d1, d2 = all_region_data[i], all_region_data[j]

        acc_diff = abs(d1["accuracy"] - d2["accuracy"])
        if acc_diff > ACCURACY_MATCH_TOL:
            continue

        same_region = d1["region"] == d2["region"]
        same_mouse = d1["mouse"] == d2["mouse"]

        if d1["subspace"].shape[0] == d2["subspace"].shape[0]:
            d_g = grassmannian_distance(d1["subspace"], d2["subspace"])
        else:
            d_g = None

        n_shared = min(d1["activity"].shape[0], d2["activity"].shape[0])
        n_feat = min(d1["activity"].shape[1], d2["activity"].shape[1])
        cka_val = cka(d1["activity"][:n_shared, :n_feat], d2["activity"][:n_shared, :n_feat])

        if d1["holonomy"].shape == d2["holonomy"].shape:
            h_dist = holonomy_distance(d1["holonomy"], d2["holonomy"])
        else:
            h_dist = None

        pairs.append({
            "region_1": d1["region"],
            "region_2": d2["region"],
            "mouse_1": d1["mouse"],
            "mouse_2": d2["mouse"],
            "accuracy_1": d1["accuracy"],
            "accuracy_2": d2["accuracy"],
            "same_region": same_region,
            "same_mouse": same_mouse,
            "grassmannian_distance": d_g,
            "cka": cka_val,
            "holonomy_distance": h_dist,
        })

    logger.info(f"Accuracy-matched pairs: {len(pairs)}")

    pairs_with_dg = [p for p in pairs if p["grassmannian_distance"] is not None]
    cka_vals = [p["cka"] for p in pairs_with_dg]
    dg_vals = [p["grassmannian_distance"] for p in pairs_with_dg]

    if len(cka_vals) >= 5:
        pearson_r, pearson_p = pearsonr(cka_vals, dg_vals)
        spearman_r, spearman_p = spearmanr(cka_vals, dg_vals)
    else:
        pearson_r = pearson_p = spearman_r = spearman_p = None

    same_region_pairs = [p for p in pairs if p["same_region"]]
    cross_region_pairs = [p for p in pairs if not p["same_region"]]

    def _pair_stats(pair_list):
        if not pair_list:
            return {}
        dg_vals_sub = [p["grassmannian_distance"] for p in pair_list if p["grassmannian_distance"] is not None]
        h_vals = [p["holonomy_distance"] for p in pair_list if p["holonomy_distance"] is not None]
        return {
            "n": len(pair_list),
            "cka_mean": float(np.mean([p["cka"] for p in pair_list])),
            "dg_mean": float(np.mean(dg_vals_sub)) if dg_vals_sub else None,
            "dg_n": len(dg_vals_sub),
            "holonomy_mean": float(np.mean(h_vals)) if h_vals else None,
        }

    results = {
        "timestamp": datetime.now().isoformat(),
        "k": SUBSPACE_K,
        "accuracy_match_tolerance": ACCURACY_MATCH_TOL,
        "n_regions_measured": len(all_region_data),
        "n_matched_pairs": len(pairs),
        "cka_dg_correlation": {
            "pearson_r": pearson_r,
            "pearson_p": pearson_p,
            "spearman_r": spearman_r,
            "spearman_p": spearman_p,
            "prediction": "correlation < 1.0, significantly so for cross-region pairs",
        },
        "same_region_pairs": _pair_stats(same_region_pairs),
        "cross_region_pairs": _pair_stats(cross_region_pairs),
        "pairs": [{k: v for k, v in p.items()} for p in pairs],
    }

    logger.info(f"\nCKA-dG Pearson r = {pearson_r}, p = {pearson_p}")
    logger.info(f"Same-region: {_pair_stats(same_region_pairs)}")
    logger.info(f"Cross-region: {_pair_stats(cross_region_pairs)}")

    out_path = RESULTS_DIR / "multiple_realization.json"
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
