"""Tests for gradient-based F-update."""

import numpy as np
import pytest

from alsgls import als_gls, nll_per_row, simulate_sur
from alsgls.ops import grad_F_nll, woodbury_chol


def _finite_diff_grad_F(R, F, D, lam_F, eps=1e-6):
    """Compute gradient via finite differences for verification."""
    K, k = F.shape
    grad = np.zeros_like(F)

    for i in range(K):
        for j in range(k):
            F_plus = F.copy()
            F_minus = F.copy()
            F_plus[i, j] += eps
            F_minus[i, j] -= eps

            nll_plus = nll_per_row(R, F_plus, D) + 0.5 * lam_F * np.sum(F_plus**2)
            nll_minus = nll_per_row(R, F_minus, D) + 0.5 * lam_F * np.sum(F_minus**2)

            grad[i, j] = (nll_plus - nll_minus) / (2 * eps)

    return grad


class TestGradFNll:
    def test_gradient_matches_finite_diff(self):
        """Verify analytical gradient matches finite differences."""
        np.random.seed(42)
        N, K, k = 50, 10, 3
        R = np.random.randn(N, K)
        F = np.random.randn(K, k) * 0.5
        D = 0.1 + 0.5 * np.random.rand(K)
        lam_F = 0.01

        Dinv, C_chol = woodbury_chol(F, D)
        grad_analytical = grad_F_nll(R, F, D, Dinv, C_chol, lam_F)
        grad_numerical = _finite_diff_grad_F(R, F, D, lam_F)

        np.testing.assert_allclose(grad_analytical, grad_numerical, rtol=1e-4, atol=1e-6)

    def test_gradient_zero_at_optimum(self):
        """Gradient should be near zero when F is optimal."""
        np.random.seed(123)
        N, K, k = 100, 8, 2
        F_true = np.random.randn(K, k) * 0.3
        D_true = 0.2 + 0.3 * np.random.rand(K)
        Sigma = F_true @ F_true.T + np.diag(D_true)

        L = np.linalg.cholesky(Sigma)
        R = np.random.randn(N, K) @ L.T

        lam_F = 0.0
        Dinv, C_chol = woodbury_chol(F_true, D_true)
        grad = grad_F_nll(R, F_true, D_true, Dinv, C_chol, lam_F)

        assert np.linalg.norm(grad) < 1.0

    def test_gradient_shape(self):
        """Gradient should have same shape as F."""
        N, K, k = 30, 12, 4
        R = np.random.randn(N, K)
        F = np.random.randn(K, k)
        D = np.abs(np.random.randn(K)) + 0.1

        Dinv, C_chol = woodbury_chol(F, D)
        grad = grad_F_nll(R, F, D, Dinv, C_chol, lam_F=0.01)

        assert grad.shape == F.shape


class TestALSWithGradient:
    def test_als_converges(self):
        """ALS with gradient-based F-update should converge."""
        Xs_tr, Y_tr, _, _ = simulate_sur(N_tr=100, N_te=50, K=20, p=3, k=3, seed=42)
        B, F, D, _, info = als_gls(Xs_tr, Y_tr, k=4, sweeps=10)

        nll_trace = info["nll_trace"]
        assert len(nll_trace) > 1
        assert nll_trace[-1] < nll_trace[0]

    def test_als_nll_monotonic(self):
        """NLL should be non-increasing across sweeps."""
        Xs_tr, Y_tr, _, _ = simulate_sur(N_tr=150, N_te=50, K=15, p=2, k=2, seed=123)
        _, _, _, _, info = als_gls(Xs_tr, Y_tr, k=3, sweeps=8)

        nll_trace = info["nll_trace"]
        for i in range(1, len(nll_trace)):
            assert nll_trace[i] <= nll_trace[i - 1] + 1e-10

    def test_als_improves_over_baseline(self):
        """ALS should improve NLL over initial estimate."""
        Xs_tr, Y_tr, _, _ = simulate_sur(N_tr=200, N_te=50, K=20, p=3, k=3, seed=456)
        _, _, _, _, info = als_gls(Xs_tr, Y_tr, k=4, sweeps=10)

        nll_trace = info["nll_trace"]
        initial_nll = nll_trace[0]
        final_nll = nll_trace[-1]
        assert final_nll < initial_nll

    def test_als_produces_valid_outputs(self):
        """Check output shapes and values."""
        Xs_tr, Y_tr, _, _ = simulate_sur(N_tr=80, N_te=40, K=12, p=2, k=2, seed=789)
        B, F, D, mem_mb, info = als_gls(Xs_tr, Y_tr, k=3, sweeps=6)

        K = 12
        k = 3
        assert len(B) == K
        assert F.shape == (K, k)
        assert D.shape == (K,)
        assert np.all(D > 0)
        assert mem_mb > 0
        assert "nll_trace" in info
