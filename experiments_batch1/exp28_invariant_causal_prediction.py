"""Experiment 28: Invariant Causal Prediction across animals.

ICP (Peters et al. 2016) finds feature sets whose predictive relationship to
an outcome is *invariant* across environments. Here each mouse is an environment.
The question: which neural subspace dimensions have an invariant relationship
to choice, versus which are animal-specific?

We predict:
1. The invariant subspace is lower-dimensional than the full choice subspace.
2. The gap (full dim - invariant dim) is largest in flat-spectrum (high-alpha)
   regions, because high-dimensional representations have more room for
   animal-specific variation.
3. This directly extends the dimensionality-mediation result from exp24.
"""
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

from data.steinmetz import get_choice_labels, get_region_activity, list_regions, load_all

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent.parent / "artifacts" / "exp28"
MIN_NEURONS = 15
MIN_SESSIONS_PER_REGION = 3
TIME_WINDOW = slice(15, 35)
N_PCA_DIMS = 20
ICP_ALPHA = 0.05


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


def _choice_subspace_dim(activity, labels, n_pca=N_PCA_DIMS):
    """Dimension of the full choice-predictive subspace via cross-validated logistic regression."""
    n_pca = min(n_pca, activity.shape[1] - 1, activity.shape[0] - 1)
    if n_pca < 1:
        return None, None
    pca = PCA(n_components=n_pca)
    scores = pca.fit_transform(activity)

    best_dim = 1
    best_acc = 0
    for d in range(1, n_pca + 1):
        X = scores[:, :d]
        clf = LogisticRegression(max_iter=500, solver="lbfgs")
        from sklearn.model_selection import cross_val_score
        cv_acc = cross_val_score(clf, X, labels, cv=min(5, len(labels) // 4), scoring="accuracy")
        mean_acc = cv_acc.mean()
        if mean_acc > best_acc + 0.01:
            best_acc = mean_acc
            best_dim = d
    return best_dim, best_acc


def _icp_invariant_set(region_sessions, n_pca=N_PCA_DIMS, alpha=ICP_ALPHA):
    """Find the invariant causal predictive set across environments (mice).

    Simplified ICP: for each subset of PCA dimensions, test whether the
    regression coefficients are statistically invariant across environments.
    We use a residual-based test: fit on each environment, check if residuals
    are exchangeable across environments.

    Returns the largest invariant set (set of PC indices) and its dimensionality.
    """
    all_envs = []
    shared_pca_dim = min(n_pca, min(s["activity"].shape[1] - 1 for s in region_sessions),
                         min(s["activity"].shape[0] - 1 for s in region_sessions))
    if shared_pca_dim < 1:
        return None, None

    for s in region_sessions:
        pca = PCA(n_components=shared_pca_dim)
        scores = pca.fit_transform(s["activity"])
        all_envs.append({"scores": scores, "labels": s["labels"]})

    invariant_dims = []

    for d in range(shared_pca_dim):
        coefs = []
        residuals_by_env = []
        for env in all_envs:
            X = env["scores"][:, d:d+1]
            y = env["labels"].astype(float)
            clf = LogisticRegression(max_iter=500, solver="lbfgs")
            try:
                clf.fit(X, y)
                coefs.append(clf.coef_[0, 0])
                pred_proba = clf.predict_proba(X)[:, 1]
                residuals_by_env.append(y - pred_proba)
            except Exception:
                continue

        if len(coefs) < MIN_SESSIONS_PER_REGION:
            continue

        coefs = np.array(coefs)
        coef_cv = np.std(coefs) / (np.abs(np.mean(coefs)) + 1e-10)

        from scipy.stats import kruskal
        if all(len(r) >= 2 for r in residuals_by_env) and len(residuals_by_env) >= 2:
            try:
                stat, p_val = kruskal(*residuals_by_env)
                if p_val > alpha and coef_cv < 1.0:
                    invariant_dims.append(d)
            except Exception:
                pass

    multi_dim_invariant = _test_joint_invariance(all_envs, invariant_dims, alpha)

    return multi_dim_invariant, shared_pca_dim


def _test_joint_invariance(all_envs, candidate_dims, alpha):
    """Test joint invariance of candidate dimension set."""
    if not candidate_dims:
        return []

    for size in range(len(candidate_dims), 0, -1):
        dims = candidate_dims[:size]
        coefs_all = []
        residuals_all = []

        for env in all_envs:
            X = env["scores"][:, dims]
            y = env["labels"].astype(float)
            clf = LogisticRegression(max_iter=500, solver="lbfgs")
            try:
                clf.fit(X, y)
                coefs_all.append(clf.coef_[0])
                pred = clf.predict_proba(X)[:, 1]
                residuals_all.append(y - pred)
            except Exception:
                continue

        if len(coefs_all) < MIN_SESSIONS_PER_REGION:
            continue

        coefs_arr = np.array(coefs_all)
        coef_cvs = np.std(coefs_arr, axis=0) / (np.abs(np.mean(coefs_arr, axis=0)) + 1e-10)

        from scipy.stats import kruskal
        try:
            stat, p_val = kruskal(*residuals_all)
            if p_val > alpha and np.all(coef_cvs < 1.5):
                return list(dims)
        except Exception:
            continue

    return []


def run(max_sessions: int | None = None):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    sessions = load_all()
    if max_sessions:
        sessions = sessions[:max_sessions]

    mouse_sessions = {}
    for sess_idx, sess in enumerate(sessions):
        mouse = str(sess.get("mouse_name", f"mouse_{sess_idx}"))
        if mouse not in mouse_sessions:
            mouse_sessions[mouse] = []
        mouse_sessions[mouse].append((sess_idx, sess))

    logger.info(f"Loaded {len(sessions)} sessions from {len(mouse_sessions)} mice")

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

            if region not in region_data:
                region_data[region] = []
            region_data[region].append({
                "session_idx": sess_idx,
                "mouse": mouse,
                "activity": activity,
                "labels": labels[:n],
                "n_neurons": activity.shape[1],
            })

    results_by_region = {}
    icp_dims_list = []
    choice_dims_list = []
    alpha_list = []
    gap_list = []

    for region, measurements in tqdm(region_data.items(), desc="ICP by region"):
        mice_with_data = set(m["mouse"] for m in measurements)
        if len(mice_with_data) < MIN_SESSIONS_PER_REGION:
            continue

        alphas_per = [_power_law_exponent(m["activity"]) for m in measurements]
        alphas_valid = [a for a in alphas_per if a is not None]
        alpha = float(np.mean(alphas_valid)) if alphas_valid else None
        eff_dim = float(np.mean([_effective_dim(m["activity"]) for m in measurements]))

        largest = max(measurements, key=lambda m: m["activity"].shape[0])
        choice_dim, choice_acc = _choice_subspace_dim(largest["activity"], largest["labels"])

        invariant_set, total_pca_dim = _icp_invariant_set(measurements)

        icp_dim = len(invariant_set) if invariant_set else 0
        gap = (choice_dim - icp_dim) if choice_dim is not None else None

        results_by_region[region] = {
            "n_sessions": len(measurements),
            "n_mice": len(mice_with_data),
            "power_law_alpha": alpha,
            "effective_dim": eff_dim,
            "choice_subspace_dim": choice_dim,
            "choice_accuracy": choice_acc,
            "invariant_set_dims": invariant_set if invariant_set else [],
            "invariant_dim": icp_dim,
            "dim_gap": gap,
            "total_pca_dim": total_pca_dim,
        }

        if alpha is not None and gap is not None:
            alpha_list.append(alpha)
            gap_list.append(gap)
            icp_dims_list.append(icp_dim)
            if choice_dim is not None:
                choice_dims_list.append(choice_dim)

    prediction_tests = {}
    if len(alpha_list) >= 4:
        rho_gap, p_gap = spearmanr(alpha_list, gap_list)
        rho_icp, p_icp = spearmanr(alpha_list, icp_dims_list)
        prediction_tests["alpha_vs_gap"] = {
            "rho": float(rho_gap), "p": float(p_gap), "n": len(alpha_list),
            "interpretation": "Positive rho means flatter spectrum → larger gap between full and invariant subspace"
        }
        prediction_tests["alpha_vs_invariant_dim"] = {
            "rho": float(rho_icp), "p": float(p_icp), "n": len(alpha_list),
        }

    if len(icp_dims_list) >= 4 and len(choice_dims_list) >= 4:
        rho_dims, p_dims = spearmanr(choice_dims_list[:len(icp_dims_list)], icp_dims_list)
        prediction_tests["choice_dim_vs_invariant_dim"] = {
            "rho": float(rho_dims), "p": float(p_dims),
        }

    summary = {
        "n_regions_analyzed": len(results_by_region),
        "mean_invariant_dim": float(np.mean(icp_dims_list)) if icp_dims_list else None,
        "mean_choice_dim": float(np.mean(choice_dims_list)) if choice_dims_list else None,
        "mean_gap": float(np.mean(gap_list)) if gap_list else None,
        "fraction_with_invariant_set": (
            sum(1 for r in results_by_region.values() if r["invariant_dim"] > 0) / len(results_by_region)
            if results_by_region else None
        ),
    }

    results = {
        "timestamp": datetime.now().isoformat(),
        "n_sessions": len(sessions),
        "n_mice": len(mouse_sessions),
        "n_regions": len(region_data),
        "n_regions_analyzed": len(results_by_region),
        "region_results": results_by_region,
        "prediction_tests": prediction_tests,
        "summary": summary,
    }

    out_path = RESULTS_DIR / "icp_across_animals.json"
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
