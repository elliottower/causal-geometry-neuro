"""Holonomy estimation on causal subspaces.

When a causal subspace is parallel-transported around a closed loop in
condition space (e.g., trial onset → mid-deliberation → post-choice → baseline),
the result may differ from the original subspace. This discrepancy is holonomy.

Holonomy is gauge-invariant and provides a fingerprint of the circuit's
global transport structure. Same-mechanism populations should have isomorphic
holonomy groups.
"""
import numpy as np
from scipy.linalg import svd, polar


def estimate_subspace_at_time(
    activity: np.ndarray,
    labels: np.ndarray,
    k: int = 5,
) -> np.ndarray:
    """Estimate causal subspace from activity at a single time point.

    Simple PCA-based estimate. For full DAS, use geometry.subspace.fit_das_subspace.

    Args:
        activity: (n_trials, n_neurons)
        labels: (n_trials,) binary
        k: subspace dimension

    Returns:
        (n_neurons, k) orthonormal basis
    """
    from geometry.subspace import fit_pca_subspace
    return fit_pca_subspace(activity, labels, k=k)


def parallel_transport_map(U_from: np.ndarray, U_to: np.ndarray) -> np.ndarray:
    """Estimate the parallel transport map between two subspace bases.

    Uses the Procrustes-optimal rotation: the orthogonal matrix Q that minimizes
    ||U_to - U_from @ Q||_F. This is the discrete approximation to parallel
    transport on the Grassmannian.

    Args:
        U_from: (n, k) orthonormal basis at source
        U_to: (n, k) orthonormal basis at target

    Returns:
        (k, k) orthogonal transport matrix Q
    """
    M = U_from.T @ U_to
    U, _, Vt = svd(M)
    return U @ Vt


def estimate_holonomy(
    activity_sequence: list[np.ndarray],
    labels: np.ndarray,
    k: int = 5,
) -> np.ndarray:
    """Estimate holonomy around a trial loop.

    Given activity at T time points forming a loop (first ≈ last condition),
    compute the parallel transport around the loop. The holonomy matrix H
    is the composition of transport maps around the loop.

    Args:
        activity_sequence: list of T arrays, each (n_trials, n_neurons),
            representing activity at successive time points in a trial.
            The first and last should be at similar conditions (e.g., ITI).
        labels: (n_trials,) binary labels (stable across time points)
        k: subspace dimension

    Returns:
        (k, k) holonomy matrix H. For zero curvature, H = I.
    """
    bases = [estimate_subspace_at_time(act, labels, k) for act in activity_sequence]

    H = np.eye(k)
    for i in range(len(bases) - 1):
        Q = parallel_transport_map(bases[i], bases[i + 1])
        H = H @ Q

    Q_close = parallel_transport_map(bases[-1], bases[0])
    H = H @ Q_close

    return H


def holonomy_angle(H: np.ndarray) -> float:
    """Total rotation angle of the holonomy matrix.

    This is the geodesic distance from H to I on SO(k).
    """
    _, R = polar(H)
    eigenvalues = np.linalg.eigvals(R)
    angles = np.abs(np.angle(eigenvalues))
    return float(np.sqrt(np.sum(angles**2)))


def holonomy_distance(H1: np.ndarray, H2: np.ndarray) -> float:
    """Distance between two holonomy matrices on SO(k).

    Measures whether two populations have similar circuit transport structure.
    """
    diff = H1 @ np.linalg.inv(H2)
    return holonomy_angle(diff)
