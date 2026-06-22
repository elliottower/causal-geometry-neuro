"""Sheaf cohomology for circuit localizability.

A circuit sheaf F over the brain's region graph assigns:
- To each region r: the causal subspace F(r) ∈ Gr(k, n_r)
- To each connection (r, s): a restriction map from effective connectivity

H⁰(F) ≠ 0 iff a globally consistent single-region localization exists.
H¹(F) ≠ 0 iff the mechanism is provably distributed.

Implementation uses Čech cohomology over the region graph with
boundary matrices and rank computation via scipy.sparse.
"""
import numpy as np
from scipy.linalg import svd
from scipy.sparse import lil_matrix


class CircuitSheaf:
    """Sheaf over a brain region graph for computing cohomological localizability.

    Usage:
        sheaf = CircuitSheaf()
        sheaf.add_region('MOs', subspace_MOs)
        sheaf.add_region('ACA', subspace_ACA)
        sheaf.add_connection('MOs', 'ACA', connectivity_matrix)
        h0, h1 = sheaf.compute_cohomology()
    """

    def __init__(self):
        self.regions: dict[str, np.ndarray] = {}
        self.connections: dict[tuple[str, str], np.ndarray] = {}

    def add_region(self, name: str, subspace: np.ndarray):
        """Add a region with its causal subspace.

        Args:
            name: region identifier (e.g., 'MOs')
            subspace: (n_neurons, k) orthonormal basis for the causal subspace
        """
        self.regions[name] = subspace

    def add_connection(self, source: str, target: str, connectivity: np.ndarray):
        """Add a directed connection with its effective connectivity matrix.

        Args:
            source, target: region names
            connectivity: (n_target, n_source) effective connectivity matrix
                estimated from spike cross-correlations at short lags
        """
        if source not in self.regions or target not in self.regions:
            raise ValueError(f"Both regions must be added first: {source}, {target}")
        self.connections[(source, target)] = connectivity

    def restriction_map(self, source: str, target: str) -> np.ndarray:
        """Compute the restriction map: project source subspace through connectivity.

        φ(S) = span(W @ S) where W is the connectivity matrix.

        Returns:
            (k_target, k_source) matrix representing the restriction
        """
        W = self.connections[(source, target)]
        S_source = self.regions[source]

        transported = W @ S_source
        U, _, _ = svd(transported, full_matrices=False)
        k_target = self.regions[target].shape[1]
        U_trunc = U[:, :k_target]

        return self.regions[target].T @ U_trunc

    def _boundary_0(self) -> np.ndarray:
        """Construct the 0-th boundary operator δ⁰: C⁰ → C¹.

        C⁰ = ⊕_v F(v) (sections over vertices)
        C¹ = ⊕_e F(e) (sections over edges)

        δ⁰ maps a global section to its consistency defect on each edge.
        """
        region_list = sorted(self.regions.keys())
        edge_list = sorted(self.connections.keys())

        region_idx = {r: i for i, r in enumerate(region_list)}

        ks = [self.regions[r].shape[1] for r in region_list]
        total_vertex_dim = sum(ks)
        total_edge_dim = sum(min(ks[region_idx[s]], ks[region_idx[t]]) for s, t in edge_list)

        if total_edge_dim == 0 or total_vertex_dim == 0:
            return np.zeros((total_edge_dim, total_vertex_dim))

        delta = np.zeros((total_edge_dim, total_vertex_dim))

        edge_offset = 0
        for s, t in edge_list:
            k_s = ks[region_idx[s]]
            k_t = ks[region_idx[t]]
            k_e = min(k_s, k_t)

            rmap = self.restriction_map(s, t)  # (k_t, k_s)
            rmap_trunc = rmap[:k_e, :k_s]

            s_offset = sum(ks[:region_idx[s]])
            t_offset = sum(ks[:region_idx[t]])

            delta[edge_offset : edge_offset + k_e, s_offset : s_offset + k_s] = rmap_trunc
            delta[edge_offset : edge_offset + k_e, t_offset : t_offset + k_t] = -np.eye(k_e, k_t)

            edge_offset += k_e

        return delta

    def compute_cohomology(self, tol: float = 1e-6) -> tuple[int, int]:
        """Compute H⁰ and H¹ of the circuit sheaf.

        H⁰ = ker(δ⁰) — global sections. Nonzero iff mechanism is localizable.
        H¹ = coker(δ⁰) = C¹ / im(δ⁰). Nonzero iff mechanism is provably distributed.

        Returns:
            (dim_H0, dim_H1)
        """
        delta = self._boundary_0()

        if delta.size == 0:
            n_vertex_dims = sum(s.shape[1] for s in self.regions.values())
            return (n_vertex_dims, 0)

        _, s, _ = svd(delta, full_matrices=False)
        rank = int(np.sum(s > tol))

        n_vertices_dim = delta.shape[1]
        n_edges_dim = delta.shape[0]

        h0 = n_vertices_dim - rank  # dim ker(δ⁰)
        h1 = n_edges_dim - rank  # dim coker(δ⁰)

        return (h0, h1)

    def is_localizable(self) -> bool:
        """Quick check: does a global section exist?"""
        h0, _ = self.compute_cohomology()
        return h0 > 0

    def is_distributed(self) -> bool:
        """Quick check: is the mechanism provably distributed?"""
        _, h1 = self.compute_cohomology()
        return h1 > 0
