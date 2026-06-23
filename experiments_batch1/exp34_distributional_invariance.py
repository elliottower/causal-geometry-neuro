"""Experiment 34: Distributional invariance test for geometric types.

Tests whether population geometry metrics are mechanistic or spurious by checking
regression invariance across environments (different mice).

For each region:
- Fit regression: subspace coordinates → choice, separately per mouse
- Test if regression coefficients are stable across mice (Kruskal-Wallis on residuals)
- Prediction: CKA-type regions (low-dim, linearly stable) should be MORE invariant
  (stable regression coefficients across environments)
- Procrustes-type regions (high-dim, geometry-stable but not linearly stable) should
  FAIL the invariance test when environments shift

This connects geometric type taxonomy to a formal causal invariance criterion:
"geometric type predicts the degree to which a region's representation is
causally invariant to environmental shifts"
"""
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.stats import kruskal, spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

from data.steinmetz import get_choice_labels, get_region_activity, list_regions, load_all

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent.parent / "artifacts" / "exp34"
MIN_NEURONS = 15
TIME_WINDOW = slice(15, 35)
N_PCA = 10
MIN_MICE = 3
MIN_TRIALS = 30


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


def _regression_invariance(mouse_data, n_pca=N_PCA):
    """Test regression invariance across mice.

    Returns:
        invariance_score: 1 - (between-mouse variance / total variance) of coefficients
        kruskal_p: p-value for residual exchangeability across mice
        coef_cv: coefficient of variation of regression weights across mice
        mouse_accuracies: per-mouse accuracy
    """
    coefs_per_mouse = []
    residuals_per_mouse = []
    accuracies = []

    for mouse, session_list in mouse_data.items():
        best_session = max(session_list, key=lambda s: s["activity"].shape[0])
        activity = best_session["activity"]
        labels = best_session["labels"]

        if len(labels) < MIN_TRIALS or len(np.unique(labels)) < 2:
            continue

        pca_dim = min(n_pca, activity.shape[1] - 1, activity.shape[0] - 1)
        if pca_dim < 1:
            continue

        pca = PCA(n_components=pca_dim)
        scores = pca.fit_transform(activity)

        clf = LogisticRegression(max_iter=500, solver="lbfgs")
        try:
            clf.fit(scores, labels)
        except Exception:
            continue

        coefs_per_mouse.append(clf.coef_[0])
        pred_proba = clf.predict_proba(scores)[:, 1]
        residuals = labels.astype(float) - pred_proba
        residuals_per_mouse.append(residuals)
        accuracies.append(float(clf.score(scores, labels)))

    if len(coefs_per_mouse) < MIN_MICE:
        return None

    coefs_arr = np.array(coefs_per_mouse)
    mean_coefs = np.mean(coefs_arr, axis=0)
    between_var = np.var(coefs_arr, axis=0)
    total_var = np.var(coefs_arr)

    coef_cv = float(np.mean(np.std(coefs_arr, axis=0) / (np.abs(mean_coefs) + 1e-10)))
    invariance_score = float(1.0 - np.mean(between_var) / (total_var + 1e-10))

    kruskal_p = None
    if all(len(r) >= 2 for r in residuals_per_mouse):
        try:
            _, kruskal_p = kruskal(*residuals_per_mouse)
            kruskal_p = float(kruskal_p)
        except Exception:
            pass

    coef_cosine_sims = []
    for i in range(len(coefs_per_mouse)):
        for j in range(i + 1, len(coefs_per_mouse)):
            c1, c2 = coefs_per_mouse[i], coefs_per_mouse[j]
            cos = np.dot(c1, c2) / (np.linalg.norm(c1) * np.linalg.norm(c2) + 1e-10)
            coef_cosine_sims.append(float(cos))

    return {
        "invariance_score": invariance_score,
        "kruskal_p": kruskal_p,
        "coef_cv": coef_cv,
        "mean_coef_cosine": float(np.mean(coef_cosine_sims)) if coef_cosine_sims else None,
        "n_mice": len(coefs_per_mouse),
        "mean_accuracy": float(np.mean(accuracies)),
        "mouse_accuracies": accuracies,
    }


def run(max_sessions: int | None = None):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    sessions = load_all()
    if max_sessions:
        sessions = sessions[:max_sessions]

    region_mouse_data = {}

    for sess_idx, sess in enumerate(tqdm(sessions, desc="Loading sessions")):
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

            if region not in region_mouse_data:
                region_mouse_data[region] = {}

            if mouse not in region_mouse_data[region]:
                region_mouse_data[region][mouse] = []
            region_mouse_data[region][mouse].append({
                "activity": activity,
                "labels": labels[:n],
            })

    region_results = {}
    alpha_list = []
    invariance_list = []
    kruskal_list = []
    cosine_list = []

    for region, mouse_data in tqdm(region_mouse_data.items(), desc="Invariance test"):
        if len(mouse_data) < MIN_MICE:
            continue

        all_activities = []
        for session_list in mouse_data.values():
            for s in session_list:
                all_activities.append(s["activity"])

        alphas = [_power_law_exponent(a) for a in all_activities]
        alphas = [a for a in alphas if a is not None]
        alpha = float(np.mean(alphas)) if alphas else None

        eff_dims = [_effective_dim(a) for a in all_activities]
        eff_dim = float(np.mean(eff_dims))

        inv = _regression_invariance(mouse_data)
        if inv is None:
            continue

        region_results[region] = {
            "power_law_alpha": alpha,
            "effective_dim": eff_dim,
            **inv,
        }

        if alpha is not None:
            alpha_list.append(alpha)
            invariance_list.append(inv["invariance_score"])
            if inv["kruskal_p"] is not None:
                kruskal_list.append(inv["kruskal_p"])
            if inv["mean_coef_cosine"] is not None:
                cosine_list.append(inv["mean_coef_cosine"])

    prediction_tests = {}

    if len(alpha_list) >= 4 and len(invariance_list) >= 4:
        rho, p = spearmanr(alpha_list, invariance_list)
        prediction_tests["alpha_vs_invariance"] = {
            "rho": float(rho), "p": float(p), "n": len(alpha_list),
            "interpretation": (
                "Negative rho means high-alpha (CKA-type, steep spectrum) regions "
                "have LESS invariant regression coefficients. Positive rho would "
                "mean CKA-type is more invariant — the predicted direction."
            ),
        }

    if len(alpha_list) >= 4 and len(cosine_list) >= 4:
        n = min(len(alpha_list), len(cosine_list))
        rho, p = spearmanr(alpha_list[:n], cosine_list[:n])
        prediction_tests["alpha_vs_coef_cosine"] = {
            "rho": float(rho), "p": float(p), "n": n,
            "interpretation": (
                "Positive rho means high-alpha regions have more similar regression "
                "weight directions across mice (the invariance prediction)."
            ),
        }

    invariant_regions = sorted(
        [(r, v) for r, v in region_results.items()],
        key=lambda x: x[1]["invariance_score"],
        reverse=True,
    )

    results = {
        "timestamp": datetime.now().isoformat(),
        "n_sessions": len(sessions),
        "n_regions_analyzed": len(region_results),
        "region_results": region_results,
        "prediction_tests": prediction_tests,
        "most_invariant": [
            {"region": r, "invariance": v["invariance_score"],
             "alpha": v["power_law_alpha"], "kruskal_p": v["kruskal_p"]}
            for r, v in invariant_regions[:5]
        ],
        "least_invariant": [
            {"region": r, "invariance": v["invariance_score"],
             "alpha": v["power_law_alpha"], "kruskal_p": v["kruskal_p"]}
            for r, v in invariant_regions[-5:]
        ],
    }

    out_path = RESULTS_DIR / "distributional_invariance.json"
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
