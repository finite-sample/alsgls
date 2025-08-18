import numpy as np
import sys
from pathlib import Path

# Ensure package root is on sys.path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from alsgls.ops import woodbury_pieces, apply_siginv_to_matrix


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
