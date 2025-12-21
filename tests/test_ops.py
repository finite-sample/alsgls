import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure package root is on sys.path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from alsgls.ops import (
    apply_siginv_to_matrix,
    cg_solve,
    stack_B_list,
    unstack_B_vec,
    woodbury_chol,
)


def test_apply_siginv_to_matrix():
    rng = np.random.default_rng(0)
    K, k = 5, 2
    F = rng.standard_normal((K, k))
    D = rng.uniform(0.5, 2.0, size=K)

    Sigma = F @ F.T + np.diag(D)
    Sigma_inv = np.linalg.inv(Sigma)

    # Get Cholesky factor
    Dinv, C_chol = woodbury_chol(F, D)
    
    # Validate apply_siginv_to_matrix against explicit inverse
    M = rng.standard_normal((3, K))
    expected = M @ Sigma_inv
    # Test with required C_chol parameter
    got = apply_siginv_to_matrix(M, F, D, C_chol=C_chol)
    assert np.allclose(got, expected, atol=1e-12, rtol=1e-12)
    # with cached Dinv
    got_cached = apply_siginv_to_matrix(M, F, D, Dinv=Dinv, C_chol=C_chol)
    assert np.allclose(got_cached, expected, atol=1e-12, rtol=1e-12)


def test_stack_and_unstack_B_list():
    """Stack heterogeneous B_j blocks and recover them."""
    rng = np.random.default_rng(0)
    # heterogeneous shapes for each B_j
    p_list = [1, 3, 2]
    B_list = [rng.standard_normal((p, 1)) for p in p_list]

    # Stack into vector then unstack back to list
    b_vec = stack_B_list(B_list)
    recovered = unstack_B_vec(b_vec, p_list)

    # Each recovered block should match the original exactly
    for orig, rec in zip(B_list, recovered, strict=False):
        assert np.allclose(orig, rec)


def test_cg_solve_raises_on_non_spd_operator():
    """cg_solve should fail when the operator is not SPD."""
    A = np.array([[1.0, 0.0], [0.0, -1.0]])

    def mv(x):
        return A @ x

    b = np.array([1.0, 1.0])
    with pytest.raises(ValueError, match=r"p\^T A p"):
        cg_solve(mv, b)


def test_cg_solve_raises_on_non_spd_preconditioner():
    """cg_solve should fail when the preconditioner is not SPD."""
    A = np.array([[2.0, 0.0], [0.0, 1.0]])

    def mv(x):
        return A @ x

    def M_pre(x):
        return -x

    b = np.array([1.0, 1.0])
    with pytest.raises(ValueError, match=r"r\^T z"):
        cg_solve(mv, b, M_pre=M_pre)
