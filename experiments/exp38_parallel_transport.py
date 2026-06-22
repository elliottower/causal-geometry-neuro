"""Experiment 38: Parallel transport on the Grassmannian (Steinmetz).

Instead of only measuring the Grassmannian distance between choice subspaces,
we parallel-transport tangent vectors along geodesics and measure the holonomy
(transport defect) around triangles of sessions.

Key idea: for each region with 3+ sessions, form all triangles of sessions
(A, B, C). Compute the top-k PCA subspace of each, then compose the
Procrustes transport maps A->B->C->A. The deviation of this round-trip
from the identity matrix measures the curvature (holonomy) of the
representational manifold. If subspaces across sessions are truly "parallel"
(zero holonomy), the representation is invariant. Otherwise, the transport
defect quantifies how much the representation rotates.

Predictions:
- alpha_vs_holonomy: high-dimensional (low-alpha) regions have larger
  transport defects because the subspace is less constrained.
- holonomy_vs_grassmannian: holonomy captures structure beyond pairwise
  distance (non-redundant information).
- within_mouse_vs_across_mouse_holonomy: within-mouse triangles have
  smaller holonomy than across-mouse triangles.
"""
import json
import logging
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.linalg import svd
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.decomposition import PCA
from tqdm import tqdm

from data.steinmetz import get_choice_labels, get_region_activity, list_regions, load_all

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent.parent / "artifacts" / "exp38"
MIN_NEURONS = 10
TIME_WINDOW = slice(15, 35)
MIN_SESSIONS_PER_REGION = 3
SUBSPACE_K = 4
N_BOOTSTRAP = 1000
N_PERMUTATIONS = 500


def _bootstrap_spearman(x, y, n_boot=N_BOOTSTRAP, ci=0.95):
    x, y = np.array(x), np.array(y)
    n = len(x)
    rho_obs, p_obs = spearmanr(x, y)
    rhos = np.empty(n_boot)
    for i in range(n_boot):
        idx = np.random.choice(n, n, replace=True)
        rhos[i] = spearmanr(x[idx], y[idx])[0]
    alpha = (1 - ci) / 2
    lo, hi = np.nanpercentile(rhos, [100 * alpha, 100 * (1 - alpha)])
    return {
        "rho": float(rho_obs),
        "p": float(p_obs),
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "n": n,
        "bootstrap_se": float(np.nanstd(rhos)),
    }


def _permutation_test_spearman(x, y, n_perm=N_PERMUTATIONS):
    x, y = np.array(x), np.array(y)
    rho_obs = spearmanr(x, y)[0]
    count = 0
    for _ in range(n_perm):
        perm = np.random.permutation(len(y))
        if abs(spearmanr(x, y[perm])[0]) >= abs(rho_obs):
            count += 1
    return float((count + 1) / (n_perm + 1))


def _fit_pca_subspace(activity: np.ndarray, k: int) -> np.ndarray:
    """Top-k PCA subspace of activity matrix.

    Args:
        activity: (n_trials, n_neurons)
        k: subspace dimensionality

    Returns:
        (n_neurons, k) orthonormal basis
    """
    k = min(k, activity.shape[0] - 1, activity.shape[1] - 1)
    pca = PCA(n_components=k)
    pca.fit(activity)
    return pca.components_.T  # (n_neurons, k)


def _procrustes_transport(U_from: np.ndarray, U_to: np.ndarray) -> np.ndarray:
    """Procrustes-optimal rotation mapping U_from basis to U_to basis.

    This is the discrete parallel transport map on the Grassmannian:
    the orthogonal matrix Q minimizing ||U_to - U_from @ Q||_F.

    Args:
        U_from: (n, k) orthonormal basis at source
        U_to: (n, k) orthonormal basis at target

    Returns:
        (k, k) orthogonal transport matrix
    """
    M = U_from.T @ U_to
    U, _, Vt = svd(M)
    return U @ Vt


def _holonomy_frobenius(U_a: np.ndarray, U_b: np.ndarray, U_c: np.ndarray) -> float:
    """Holonomy of the triangle A->B->C->A measured as Frobenius deviation from identity.

    Composes Procrustes transport maps around the triangle and returns
    ||H - I||_F where H = Q_{CA} @ Q_{BC} @ Q_{AB}.
    """
    k = U_a.shape[1]
    Q_ab = _procrustes_transport(U_a, U_b)
    Q_bc = _procrustes_transport(U_b, U_c)
    Q_ca = _procrustes_transport(U_c, U_a)
    H = Q_ca @ Q_bc @ Q_ab
    return float(np.linalg.norm(H - np.eye(k), "fro"))


def _grassmannian_distance(U: np.ndarray, V: np.ndarray) -> float:
    """Geodesic distance on Gr(k, n) via principal angles."""
    _, s, _ = svd(U.T @ V, full_matrices=False)
    s = np.clip(s, -1.0, 1.0)
    angles = np.arccos(s)
    return float(np.sqrt(np.sum(angles**2)))


def _power_law_exponent(activity: np.ndarray) -> float | None:
    """Spectral power-law exponent alpha from PCA eigenvalues."""
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


def run(max_sessions: int | None = None):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    sessions = load_all()
    if max_sessions:
        sessions = sessions[:max_sessions]

    # Collect per-region, per-session subspaces and metadata
    region_sessions: dict[str, list[dict]] = {}

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
            activity = act[:n, :, TIME_WINDOW].mean(axis=2)  # (n_trials, n_neurons)

            k = min(SUBSPACE_K, activity.shape[1] - 1, activity.shape[0] - 1)
            if k < 1:
                continue

            try:
                U = _fit_pca_subspace(activity, k)
            except Exception as e:
                logger.warning(f"PCA failed for {mouse}/{region} session {sess_idx}: {e}")
                continue

            alpha = _power_law_exponent(activity)

            entry = {
                "session_idx": sess_idx,
                "mouse": mouse,
                "region": region,
                "subspace": U,
                "activity": activity,
                "alpha": alpha,
                "n_neurons": int(activity.shape[1]),
                "n_trials": int(activity.shape[0]),
                "k": int(k),
            }

            if region not in region_sessions:
                region_sessions[region] = []
            region_sessions[region].append(entry)

    logger.info(f"Collected {sum(len(v) for v in region_sessions.values())} "
                f"region-session pairs across {len(region_sessions)} regions")

    # Filter to regions with enough sessions
    eligible_regions = {
        r: entries for r, entries in region_sessions.items()
        if len(entries) >= MIN_SESSIONS_PER_REGION
    }
    logger.info(f"{len(eligible_regions)} regions with >= {MIN_SESSIONS_PER_REGION} sessions")

    # Compute holonomy and Grassmannian distance for all triangles
    all_triangles = []
    region_summaries = {}

    for region, entries in tqdm(eligible_regions.items(), desc="Regions"):
        # All entries must share subspace dimension k for comparison
        # Use the minimum k across sessions in this region
        k_shared = min(e["k"] for e in entries)
        if k_shared < 1:
            continue

        # Recompute subspaces at shared k if needed, and align neuron dimensions
        # (sessions may have different neuron counts for the same region,
        # so we work in the subspace of shared trial structure)
        triangle_holonomies = []
        triangle_distances = []
        triangle_meta = []

        for i, j, m in combinations(range(len(entries)), 3):
            ea, eb, ec = entries[i], entries[j], entries[m]

            # All subspaces must have the same ambient dimension (n_neurons)
            # and subspace dimension (k). Sessions from the same region in
            # different animals may have different neuron counts, so skip
            # mismatched triples.
            if not (ea["n_neurons"] == eb["n_neurons"] == ec["n_neurons"]):
                continue

            Ua = ea["subspace"][:, :k_shared]
            Ub = eb["subspace"][:, :k_shared]
            Uc = ec["subspace"][:, :k_shared]

            try:
                holonomy = _holonomy_frobenius(Ua, Ub, Uc)
            except Exception as e:
                logger.warning(f"Holonomy failed for {region} "
                               f"({ea['session_idx']}, {eb['session_idx']}, "
                               f"{ec['session_idx']}): {e}")
                continue

            # Mean pairwise Grassmannian distance in the triangle
            d_ab = _grassmannian_distance(Ua, Ub)
            d_bc = _grassmannian_distance(Ub, Uc)
            d_ca = _grassmannian_distance(Uc, Ua)
            mean_dist = (d_ab + d_bc + d_ca) / 3.0

            mice = {ea["mouse"], eb["mouse"], ec["mouse"]}
            is_within_mouse = len(mice) == 1

            alphas = [e["alpha"] for e in (ea, eb, ec) if e["alpha"] is not None]
            mean_alpha = float(np.mean(alphas)) if alphas else None

            rec = {
                "region": region,
                "sessions": [ea["session_idx"], eb["session_idx"], ec["session_idx"]],
                "mice": sorted(mice),
                "within_mouse": is_within_mouse,
                "holonomy": holonomy,
                "mean_grassmannian_distance": mean_dist,
                "distances": {"ab": d_ab, "bc": d_bc, "ca": d_ca},
                "mean_alpha": mean_alpha,
                "k": k_shared,
            }
            all_triangles.append(rec)
            triangle_holonomies.append(holonomy)
            triangle_distances.append(mean_dist)
            triangle_meta.append(rec)

        if triangle_holonomies:
            region_summaries[region] = {
                "n_sessions": len(entries),
                "n_triangles": len(triangle_holonomies),
                "mean_holonomy": float(np.mean(triangle_holonomies)),
                "std_holonomy": float(np.std(triangle_holonomies)),
                "mean_grassmannian_distance": float(np.mean(triangle_distances)),
                "std_grassmannian_distance": float(np.std(triangle_distances)),
                "k": k_shared,
            }

    logger.info(f"Computed {len(all_triangles)} triangles across "
                f"{len(region_summaries)} regions")

    # Prediction tests
    prediction_tests = {}

    # 1. Alpha vs holonomy
    alphas_for_test = [t["mean_alpha"] for t in all_triangles if t["mean_alpha"] is not None]
    holonomies_for_test = [t["holonomy"] for t in all_triangles if t["mean_alpha"] is not None]
    if len(alphas_for_test) >= 4:
        boot = _bootstrap_spearman(alphas_for_test, holonomies_for_test)
        perm_p = _permutation_test_spearman(alphas_for_test, holonomies_for_test)
        prediction_tests["alpha_vs_holonomy"] = {
            **boot,
            "permutation_p": perm_p,
            "interpretation": (
                "Negative rho means high-alpha (steep spectral decay, low-dimensional) "
                "regions have smaller transport defects -- the subspace is more constrained "
                "and parallel transport is closer to identity."
            ),
        }
        logger.info(f"alpha vs holonomy: rho={boot['rho']:.3f}, p={boot['p']:.4f}, n={len(alphas_for_test)}")

    # 2. Holonomy vs Grassmannian distance
    all_hol = [t["holonomy"] for t in all_triangles]
    all_dist = [t["mean_grassmannian_distance"] for t in all_triangles]
    if len(all_hol) >= 4:
        boot = _bootstrap_spearman(all_hol, all_dist)
        perm_p = _permutation_test_spearman(all_hol, all_dist)
        prediction_tests["holonomy_vs_grassmannian"] = {
            **boot,
            "permutation_p": perm_p,
            "interpretation": (
                "If rho < 1, holonomy captures structure beyond pairwise distance. "
                "Triangles with equal mean distance but different holonomy indicate "
                "curvature variation on the Grassmannian."
            ),
        }
        logger.info(f"holonomy vs distance: rho={boot['rho']:.3f}, p={boot['p']:.4f}, n={len(all_hol)}")

    # 3. Within-mouse vs across-mouse holonomy
    within_hol = [t["holonomy"] for t in all_triangles if t["within_mouse"]]
    across_hol = [t["holonomy"] for t in all_triangles if not t["within_mouse"]]
    if len(within_hol) >= 2 and len(across_hol) >= 2:
        stat, p = mannwhitneyu(within_hol, across_hol, alternative="less")
        prediction_tests["within_mouse_vs_across_mouse_holonomy"] = {
            "within_mouse_mean": float(np.mean(within_hol)),
            "within_mouse_std": float(np.std(within_hol)),
            "within_mouse_n": len(within_hol),
            "across_mouse_mean": float(np.mean(across_hol)),
            "across_mouse_std": float(np.std(across_hol)),
            "across_mouse_n": len(across_hol),
            "mann_whitney_U": float(stat),
            "p_one_sided": float(p),
            "interpretation": (
                "One-sided test: within-mouse holonomy < across-mouse holonomy. "
                "Significant p means the representation is more parallel within an "
                "animal than across animals."
            ),
        }
        logger.info(f"within-mouse holonomy: {np.mean(within_hol):.4f} "
                     f"vs across-mouse: {np.mean(across_hol):.4f}, p={p:.4f}")

    # Serialize triangles (drop numpy arrays)
    triangles_serializable = []
    for t in all_triangles:
        rec = dict(t)
        rec["mice"] = list(rec["mice"])
        triangles_serializable.append(rec)

    results = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "subspace_k": SUBSPACE_K,
            "min_neurons": MIN_NEURONS,
            "time_window": [TIME_WINDOW.start, TIME_WINDOW.stop],
            "min_sessions_per_region": MIN_SESSIONS_PER_REGION,
        },
        "n_total_triangles": len(all_triangles),
        "n_regions": len(region_summaries),
        "region_summaries": region_summaries,
        "prediction_tests": prediction_tests,
        "triangles": triangles_serializable,
    }

    out_path = RESULTS_DIR / "parallel_transport.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Saved to {out_path}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Exp38: Parallel transport on the Grassmannian")
    parser.add_argument("--max-sessions", type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    run(max_sessions=args.max_sessions)
