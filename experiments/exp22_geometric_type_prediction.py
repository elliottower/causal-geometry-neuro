"""Experiment 22: Geometric type predicts optimal decoder.

The prediction test: if geometric type (linear vs nonlinear) is a real
property of each brain region, then:
  - "Linear-type" regions (high CKA, low UMAP Procrustes) should be
    better decoded by a linear decoder (LDA).
  - "Nonlinear-type" regions (low CKA, high UMAP Procrustes) should be
    better decoded by a nonlinear decoder (kNN on UMAP embedding).

This turns the dissociation finding into a testable prediction, making the
paper a diagnostic tool paper rather than just a descriptive finding.
"""
import json
import logging
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.spatial import procrustes
from scipy.stats import spearmanr
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm

from data.steinmetz import get_choice_labels, get_region_activity, list_regions, load_all
from geometry.distances import cka
from geometry.subspace import fit_lda_subspace

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent.parent / "artifacts" / "exp22"
UMAP_DIM = 5
MIN_NEURONS = 15
TIME_WINDOW = slice(15, 35)
N_FOLDS = 5
N_NEIGHBORS = 5


def _umap_embed(activity, n_components=5):
    from umap import UMAP
    reducer = UMAP(n_components=n_components, n_neighbors=15, min_dist=0.1, random_state=42)
    return reducer.fit_transform(activity)


def _procrustes_distance(X, Y):
    n = min(X.shape[0], Y.shape[0])
    _, _, disparity = procrustes(X[:n], Y[:n])
    return float(disparity)


def _cross_val_accuracy(X, y, clf, n_folds=N_FOLDS):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    accs = []
    for train_idx, test_idx in skf.split(X, y):
        clf_copy = type(clf)(**clf.get_params())
        clf_copy.fit(X[train_idx], y[train_idx])
        accs.append(float(np.mean(clf_copy.predict(X[test_idx]) == y[test_idx])))
    return float(np.mean(accs))


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
            activity = act[:n, :, TIME_WINDOW].mean(axis=2)
            y = labels[:n]

            try:
                lda_acc = _cross_val_accuracy(
                    activity, y, LinearDiscriminantAnalysis()
                )

                emb = _umap_embed(activity, n_components=min(UMAP_DIM, activity.shape[1] - 1))
                knn_acc = _cross_val_accuracy(
                    emb, y, KNeighborsClassifier(n_neighbors=N_NEIGHBORS)
                )

                if region not in region_data:
                    region_data[region] = []
                region_data[region].append({
                    "session_idx": sess_idx,
                    "mouse": mouse,
                    "activity": activity,
                    "embedding": emb,
                    "lda_accuracy": lda_acc,
                    "knn_accuracy": knn_acc,
                    "n_neurons": activity.shape[1],
                    "n_trials": n,
                })
            except Exception as e:
                logger.warning(f"Failed {mouse}/{region}: {e}")

    region_profiles = {}
    pairs = []

    for region, measurements in tqdm(region_data.items(), desc="Regions"):
        lda_accs = [m["lda_accuracy"] for m in measurements]
        knn_accs = [m["knn_accuracy"] for m in measurements]

        region_profiles[region] = {
            "n_sessions": len(measurements),
            "lda_accuracy_mean": float(np.mean(lda_accs)),
            "knn_accuracy_mean": float(np.mean(knn_accs)),
            "lda_advantage": float(np.mean(lda_accs) - np.mean(knn_accs)),
        }

        if len(measurements) < 2:
            continue
        for (i, j) in combinations(range(len(measurements)), 2):
            m1, m2 = measurements[i], measurements[j]
            n_shared = min(m1["n_trials"], m2["n_trials"])
            cka_val = cka(m1["activity"][:n_shared], m2["activity"][:n_shared])
            d_proc = _procrustes_distance(m1["embedding"], m2["embedding"])

            pairs.append({
                "region": region,
                "cka_linear": float(cka_val),
                "procrustes_distance": float(d_proc),
            })

        if pairs:
            rp = [p for p in pairs if p["region"] == region]
            if rp:
                region_profiles[region]["cka_mean"] = float(np.mean([p["cka_linear"] for p in rp]))
                region_profiles[region]["proc_mean"] = float(np.mean([p["procrustes_distance"] for p in rp]))

    prediction_test = {}
    regions_with_both = [
        r for r, p in region_profiles.items()
        if "cka_mean" in p and p["n_sessions"] >= 2
    ]

    if len(regions_with_both) >= 4:
        cka_vals = [region_profiles[r]["cka_mean"] for r in regions_with_both]
        lda_advantages = [region_profiles[r]["lda_advantage"] for r in regions_with_both]

        rho, p_val = spearmanr(cka_vals, lda_advantages)
        prediction_test = {
            "n_regions": len(regions_with_both),
            "spearman_rho": float(rho),
            "p_value": float(p_val),
            "interpretation": (
                "Positive rho means high-CKA (linear-type) regions have larger "
                "LDA advantage over kNN, confirming geometric type predicts "
                "optimal decoder choice."
            ),
            "regions": {
                r: {
                    "cka_mean": region_profiles[r].get("cka_mean"),
                    "lda_accuracy": region_profiles[r]["lda_accuracy_mean"],
                    "knn_accuracy": region_profiles[r]["knn_accuracy_mean"],
                    "lda_advantage": region_profiles[r]["lda_advantage"],
                }
                for r in regions_with_both
            },
        }

    double_dissociation = {}
    if regions_with_both:
        sorted_by_cka = sorted(regions_with_both, key=lambda r: region_profiles[r].get("cka_mean", 0))
        low_cka_region = sorted_by_cka[0]
        high_cka_region = sorted_by_cka[-1]

        lp = region_profiles[low_cka_region]
        hp = region_profiles[high_cka_region]

        double_dissociation = {
            "nonlinear_type_region": low_cka_region,
            "nonlinear_type_cka": lp.get("cka_mean"),
            "nonlinear_type_lda_acc": lp["lda_accuracy_mean"],
            "nonlinear_type_knn_acc": lp["knn_accuracy_mean"],
            "linear_type_region": high_cka_region,
            "linear_type_cka": hp.get("cka_mean"),
            "linear_type_lda_acc": hp["lda_accuracy_mean"],
            "linear_type_knn_acc": hp["knn_accuracy_mean"],
            "is_double_dissociation": (
                lp["knn_accuracy_mean"] > lp["lda_accuracy_mean"]
                and hp["lda_accuracy_mean"] > hp["knn_accuracy_mean"]
            ),
        }

    results = {
        "timestamp": datetime.now().isoformat(),
        "n_regions": len(region_data),
        "n_pairs": len(pairs),
        "region_profiles": region_profiles,
        "prediction_test": prediction_test,
        "double_dissociation": double_dissociation,
    }

    out_path = RESULTS_DIR / "geometric_type_prediction.json"
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
