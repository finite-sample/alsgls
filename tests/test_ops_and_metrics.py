import numpy as np
import numpy.linalg as npl

from alsgls.metrics import nll_per_row
from alsgls.ops import apply_siginv_to_matrix, siginv_diag, woodbury_chol


def rand_spd_diag(K, rng):
    # moderately conditioned diagonal
    d = 0.3 + rng.random(K)
    return d


def test_woodbury_matches_dense_small():
    rng = np.random.default_rng(0)
    K, k = 8, 3
    N = 5
    F = rng.standard_normal((K, k))
    D = rand_spd_diag(K, rng)
    M = rng.standard_normal((N, K))

    # Dense Σ^{-1}
    S = F @ F.T + np.diag(D)
    S_inv = npl.inv(S)
    MD = M @ S_inv

    # Woodbury
    MD_w = apply_siginv_to_matrix(M, F, D)

    assert np.allclose(MD, MD_w, rtol=5e-7, atol=5e-8)


def test_diag_of_siginv_matches_dense():
    rng = np.random.default_rng(1)
    K, k = 10, 2
    F = rng.standard_normal((K, k))
    D = rand_spd_diag(K, rng)

    S = F @ F.T + np.diag(D)
    S_inv = npl.inv(S)

    Dinv, C_chol = woodbury_chol(F, D)
    d_w = siginv_diag(F, Dinv, C_chol)
    d_d = np.diag(S_inv)

    assert np.allclose(d_w, d_d, rtol=5e-7, atol=5e-8)


def test_nll_is_finite_and_sane():
    rng = np.random.default_rng(2)
    N, K, k = 20, 12, 3
    F = rng.standard_normal((K, k))
    D = 0.2 + rng.random(K)
    R = rng.standard_normal((N, K))
    val = nll_per_row(R, F, D)
    assert np.isfinite(val)
    # crude bound: should be at least the logdet term / 2N-ish; just check non-negativity
    assert val > 0.0
