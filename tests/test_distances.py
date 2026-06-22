"""Tests for Grassmannian distances — mathematical invariants."""
import numpy as np
import pytest

from geometry.distances import (
    cka,
    grassmannian_distance,
    principal_angles,
    subspace_overlap,
)


def random_orthonormal(n, k):
    X = np.random.randn(n, k)
    Q, _ = np.linalg.qr(X)
    return Q


class TestPrincipalAngles:
    def test_identical_subspaces_give_zero_angles(self):
        U = random_orthonormal(50, 5)
        angles = principal_angles(U, U)
        assert all(a == pytest.approx(0.0, abs=1e-6) for a in angles)

    def test_orthogonal_subspaces_give_pi_over_2(self):
        U = np.eye(10, 5)
        V = np.eye(10, 5, k=5)
        angles = principal_angles(U, V)
        assert all(a == pytest.approx(np.pi / 2, abs=1e-6) for a in angles)

    def test_rotation_invariance(self):
        U = random_orthonormal(50, 5)
        V = random_orthonormal(50, 5)
        R = random_orthonormal(5, 5)
        angles_orig = principal_angles(U, V)
        angles_rotated = principal_angles(U @ R, V)
        np.testing.assert_allclose(sorted(angles_orig), sorted(angles_rotated), atol=1e-10)

    def test_angles_are_nonnegative(self):
        for _ in range(20):
            U = random_orthonormal(30, 4)
            V = random_orthonormal(30, 4)
            angles = principal_angles(U, V)
            assert all(a >= -1e-10 for a in angles)


class TestGrassmannianDistance:
    def test_is_a_metric_identity(self):
        U = random_orthonormal(50, 5)
        assert grassmannian_distance(U, U) == pytest.approx(0.0, abs=1e-6)

    def test_is_a_metric_symmetry(self):
        U = random_orthonormal(50, 5)
        V = random_orthonormal(50, 5)
        assert grassmannian_distance(U, V) == pytest.approx(
            grassmannian_distance(V, U), abs=1e-6
        )

    def test_is_a_metric_triangle_inequality(self):
        for _ in range(20):
            U = random_orthonormal(50, 5)
            V = random_orthonormal(50, 5)
            W = random_orthonormal(50, 5)
            d_uv = grassmannian_distance(U, V)
            d_vw = grassmannian_distance(V, W)
            d_uw = grassmannian_distance(U, W)
            assert d_uw <= d_uv + d_vw + 1e-10

    def test_orthogonal_subspaces_have_max_distance(self):
        U = np.eye(10, 5)
        V = np.eye(10, 5, k=5)
        d = grassmannian_distance(U, V)
        assert d == pytest.approx(np.pi / 2 * np.sqrt(5), abs=1e-6)

    def test_rotation_within_subspace_doesnt_change_distance(self):
        U = random_orthonormal(50, 5)
        V = random_orthonormal(50, 5)
        R = random_orthonormal(5, 5)
        d_orig = grassmannian_distance(U, V)
        d_rotated = grassmannian_distance(U @ R, V)
        assert d_orig == pytest.approx(d_rotated, abs=1e-6)


class TestCKA:
    def test_identical_gives_one(self):
        X = np.random.randn(100, 20)
        assert cka(X, X) == pytest.approx(1.0, abs=1e-6)

    def test_orthogonal_rotation_invariance(self):
        X = np.random.randn(100, 20)
        R = random_orthonormal(20, 20)
        assert cka(X, X @ R) == pytest.approx(1.0, abs=1e-4)

    def test_range_zero_to_one(self):
        for _ in range(10):
            X = np.random.randn(50, 15)
            Y = np.random.randn(50, 15)
            val = cka(X, Y)
            assert -0.01 <= val <= 1.01
