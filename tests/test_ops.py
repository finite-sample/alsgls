import numpy as np
import sys
from pathlib import Path

# Ensure package root is on sys.path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from alsgls.ops import woodbury_pieces, apply_siginv_to_matrix, cg_solve


def test_woodbury_pieces_and_apply_siginv_to_matrix():
    rng = np.random.default_rng(0)
    K, k = 5, 2
    F = rng.standard_normal((K, k))
    D = rng.uniform(0.5, 2.0, size=K)

    Sigma = F @ F.T + np.diag(D)
    Sigma_inv = np.linalg.inv(Sigma)

    # Validate woodbury_pieces
    Dinv, Cf = woodbury_pieces(F, D)
    Sigma_inv_wb = np.diag(Dinv) - (F * Dinv[:, None]) @ Cf @ (F.T * Dinv)
    assert np.allclose(Sigma_inv_wb, Sigma_inv, atol=1e-12, rtol=1e-12)

    # Validate apply_siginv_to_matrix against explicit inverse
    M = rng.standard_normal((3, K))
    expected = M @ Sigma_inv
    got = apply_siginv_to_matrix(M, F, D)
    assert np.allclose(got, expected, atol=1e-12, rtol=1e-12)


def test_cg_solve_small_system():
    """cg_solve should match np.linalg.solve on a tiny SPD system."""
    # Small symmetric positive definite matrix and right-hand side
    A = np.array([[4.0, 1.0],
                  [1.0, 3.0]])
    b = np.array([1.0, 2.0])

    def operator_mv(x):
        return A @ x

    # Jacobi (diagonal) preconditioner M^{-1}
    diag_inv = 1.0 / np.diag(A)

    def M_pre(x):
        return x * diag_inv

    x, info = cg_solve(operator_mv, b, maxit=20, tol=1e-12, M_pre=M_pre)

    expected = np.linalg.solve(A, b)

    assert np.allclose(x, expected, atol=1e-10, rtol=1e-10)
    # Convergence should yield a tiny residual and finish before maxit
    assert info["residual"] < 1e-10
    assert info["iterations"] < 20
