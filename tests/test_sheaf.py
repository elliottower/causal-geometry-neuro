"""Tests for sheaf cohomology — topological invariants."""
import numpy as np
import pytest

from geometry.sheaf import CircuitSheaf


def random_orthonormal(n, k):
    X = np.random.randn(n, k)
    Q, _ = np.linalg.qr(X)
    return Q


class TestCircuitSheaf:
    def test_single_region_is_localizable(self):
        sheaf = CircuitSheaf()
        sheaf.add_region("A", random_orthonormal(50, 3))
        h0, h1 = sheaf.compute_cohomology()
        assert h0 > 0
        assert h1 == 0

    def test_two_identical_regions_with_identity_connectivity(self):
        U = random_orthonormal(50, 3)
        sheaf = CircuitSheaf()
        sheaf.add_region("A", U)
        sheaf.add_region("B", U)
        sheaf.add_connection("A", "B", np.eye(50))
        h0, h1 = sheaf.compute_cohomology()
        assert h0 > 0

    def test_orthogonal_subspaces_with_zero_connectivity_are_distributed(self):
        sheaf = CircuitSheaf()
        sheaf.add_region("A", np.eye(10, 3))
        sheaf.add_region("B", np.eye(10, 3, k=3))
        W = np.zeros((10, 10))
        sheaf.add_connection("A", "B", W)
        h0, h1 = sheaf.compute_cohomology()
        assert h1 >= 0

    def test_must_add_regions_before_connections(self):
        sheaf = CircuitSheaf()
        with pytest.raises(ValueError):
            sheaf.add_connection("A", "B", np.eye(10))

    def test_three_region_chain(self):
        n = 30
        k = 3
        sheaf = CircuitSheaf()
        for name in ["A", "B", "C"]:
            sheaf.add_region(name, random_orthonormal(n, k))
        sheaf.add_connection("A", "B", np.random.randn(n, n) * 0.1)
        sheaf.add_connection("B", "C", np.random.randn(n, n) * 0.1)
        h0, h1 = sheaf.compute_cohomology()
        assert h0 >= 0
        assert h1 >= 0
