"""Tests for holonomy estimation — geometric invariants."""
import numpy as np
import pytest

from geometry.holonomy import (
    estimate_holonomy,
    holonomy_angle,
    holonomy_distance,
    parallel_transport_map,
)


def random_orthonormal(n, k):
    X = np.random.randn(n, k)
    Q, _ = np.linalg.qr(X)
    return Q


class TestParallelTransport:
    def test_transport_is_orthogonal(self):
        U = random_orthonormal(50, 5)
        V = random_orthonormal(50, 5)
        Q = parallel_transport_map(U, V)
        np.testing.assert_allclose(Q @ Q.T, np.eye(5), atol=1e-6)

    def test_transport_to_self_is_identity(self):
        U = random_orthonormal(50, 5)
        Q = parallel_transport_map(U, U)
        np.testing.assert_allclose(np.abs(Q), np.eye(5), atol=1e-6)


class TestHolonomyAngle:
    def test_identity_has_zero_angle(self):
        H = np.eye(5)
        assert holonomy_angle(H) == pytest.approx(0.0, abs=1e-10)

    def test_nonnegative(self):
        for _ in range(20):
            H = random_orthonormal(5, 5)
            assert holonomy_angle(H) >= -1e-10


class TestHolonomyDistance:
    def test_same_holonomy_gives_zero(self):
        H = random_orthonormal(5, 5)
        assert holonomy_distance(H, H) == pytest.approx(0.0, abs=1e-10)

    def test_symmetric(self):
        H1 = random_orthonormal(5, 5)
        H2 = random_orthonormal(5, 5)
        assert holonomy_distance(H1, H2) == pytest.approx(
            holonomy_distance(H2, H1), abs=1e-10
        )
