"""Experiment 43: Quantitative clustering validation.

Computes CKA-based hierarchical clustering and validates against
ground-truth anatomical labels using ARI and NMI.
"""
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from tqdm import tqdm

from data.steinmetz import get_choice_labels, get_region_activity, list_regions, load_all

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent.parent / "artifacts" / "exp43"
MIN_NEURONS = 15
TIME_WINDOW = slice(15, 35)

ANATOMY_LABELS = {
    "VISp": "visual", "VISl": "visual", "VISpm": "visual", "VISrl": "visual",
    "VISa": "visual", "VISam": "visual",
    "ACA": "cortical", "MOs": "cortical", "PL": "cortical", "ILA": "cortical",
    "RSP": "cortical", "ORB": "cortical", "SSp": "cortical", "SSs": "cortical",
    "MOp": "cortical", "AUD": "cortical",
    "CA1": "hippocampal", "CA3": "hippocampal", "DG": "hippocampal",
    "SUB": "hippocampal", "POST": "hippocampal",
    "TH": "thalamic", "VPL": "thalamic", "VPM": "thalamic", "PO": "thalamic",
    "LP": "thalamic", "LD": "thalamic", "RT": "thalamic", "VAL": "thalamic",
    "CL": "thalamic", "SPF": "thalamic", "APN": "thalamic", "LGd": "thalamic",
    "MG": "thalamic",
    "CP": "basal_ganglia", "GPe": "basal_ganglia", "SNr": "basal_ganglia",
    "ACB": "basal_ganglia", "LS": "basal_ganglia", "LSr": "basal_ganglia",
    "LSc": "basal_ganglia",
    "SCm": "midbrain", "SCig": "midbrain", "SCs": "midbrain", "SCsg": "midbrain",
    "IC": "midbrain", "MRN": "midbrain", "PAG": "midbrain",
    "BLA": "amygdala", "MEA": "amygdala", "COA": "amygdala", "EPd": "amygdala",
    "ZI": "subthalamic", "LH": "hypothalamic",
    "OT": "olfactory", "TT": "olfactory", "PIR": "olfactory",
}


def _linear_cka(X, Y):
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)
    hsic_xy = np.linalg.norm(X.T @ Y, 'fro') ** 2
    hsic_xx = np.linalg.norm(X.T @ X, 'fro') ** 2
    hsic_yy = np.linalg.norm(Y.T @ Y, 'fro') ** 2
    return float(hsic_xy / (np.sqrt(hsic_xx * hsic_yy) + 1e-10))


def run(max_sessions=None):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    sessions = load_all()
    if max_sessions:
        sessions = sessions[:max_sessions]

    region_activities = {}
    for sess_idx, sess in enumerate(tqdm(sessions, desc="Loading")):
        choice_labels = get_choice_labels(sess)
        if len(np.unique(choice_labels)) < 2:
            continue
        regions = list_regions(sess, min_neurons=MIN_NEURONS)
        for region in regions:
            act = get_region_activity(sess, region)
            if act is None or act.shape[1] < MIN_NEURONS:
                continue
            n = min(act.shape[0], len(choice_labels))
            activity = act[:n, :, TIME_WINDOW].mean(axis=2)
            if region not in region_activities:
                region_activities[region] = []
            region_activities[region].append(activity)

    multi_session_regions = [r for r, acts in region_activities.items() if len(acts) >= 2]
    multi_session_regions.sort()
    n_regions = len(multi_session_regions)
    logger.info(f"{n_regions} regions with >= 2 sessions")

    cka_matrix = np.zeros((n_regions, n_regions))
    for i in tqdm(range(n_regions), desc="CKA matrix"):
        for j in range(i, n_regions):
            cka_values = []
            for act_i in region_activities[multi_session_regions[i]]:
                for act_j in region_activities[multi_session_regions[j]]:
                    n_trials = min(act_i.shape[0], act_j.shape[0])
                    cka = _linear_cka(act_i[:n_trials], act_j[:n_trials])
                    cka_values.append(cka)
            mean_cka = float(np.mean(cka_values)) if cka_values else 0.0
            cka_matrix[i, j] = mean_cka
            cka_matrix[j, i] = mean_cka

    dissimilarity = 1.0 - cka_matrix
    np.fill_diagonal(dissimilarity, 0)
    condensed = squareform(dissimilarity, checks=False)
    Z = linkage(condensed, method='ward')

    ground_truth = []
    labeled_indices = []
    for i, region in enumerate(multi_session_regions):
        if region in ANATOMY_LABELS:
            ground_truth.append(ANATOMY_LABELS[region])
            labeled_indices.append(i)

    results = {
        "timestamp": datetime.now().isoformat(),
        "n_regions": n_regions,
        "regions": multi_session_regions,
        "n_labeled_regions": len(labeled_indices),
    }

    for n_clusters in [3, 5, 7, 10]:
        cluster_labels = fcluster(Z, t=n_clusters, criterion='maxclust')

        if labeled_indices:
            pred_labels = [int(cluster_labels[i]) for i in labeled_indices]
            ari = adjusted_rand_score(ground_truth, pred_labels)
            nmi = normalized_mutual_info_score(ground_truth, pred_labels)
        else:
            ari, nmi = 0.0, 0.0

        cluster_contents = {}
        for c in range(1, n_clusters + 1):
            members = [multi_session_regions[i] for i in range(n_regions) if cluster_labels[i] == c]
            cluster_contents[f"cluster_{c}"] = members

        results[f"k{n_clusters}"] = {
            "ari": float(ari),
            "nmi": float(nmi),
            "n_clusters": n_clusters,
            "clusters": cluster_contents,
        }

    out_path = RESULTS_DIR / "clustering_validation.json"
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
