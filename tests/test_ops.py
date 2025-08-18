import numpy as np
import sys
from pathlib import Path

# Ensure package root is on sys.path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from alsgls.ops import (
    woodbury_pieces,
    apply_siginv_to_matrix,
    stack_B_list,
    unstack_B_vec,
)


def test_woodbury_pieces_and_apply_siginv_to_matrix():
    rng = np.random.default_rng(0)
    K, k = 5, 2
    F = rng.standard_normal((K, k))
    D = rng.uniform(0.5, 2.0, size=K)

    Sigma = F @ F.T + np.diag(D)
    Sigma_inv = np.linalg.inv(Sigma)

    # Validate woodbury_pieces
    Dinv, Cf, M = woodbury_pieces(F, D)
    Sigma_inv_wb = np.diag(Dinv) - (F * Dinv[:, None]) @ Cf @ (F.T * Dinv)
    assert np.allclose(Sigma_inv_wb, Sigma_inv, atol=1e-12, rtol=1e-12)
    assert np.allclose(M, F.T @ (F * Dinv[:, None]))

    # Validate apply_siginv_to_matrix against explicit inverse
    X = rng.standard_normal((3, K))
    expected = X @ Sigma_inv
    # without cached pieces
    got = apply_siginv_to_matrix(X, F, D)
    assert np.allclose(got, expected, atol=1e-12, rtol=1e-12)
    # with cached pieces
    got_cached = apply_siginv_to_matrix(X, F, D, Dinv=Dinv, Cf=Cf)
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
    for orig, rec in zip(B_list, recovered):
        assert np.allclose(orig, rec)
