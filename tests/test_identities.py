import numpy as np
import numpy.linalg as npl

from alsgls.metrics import nll_per_row
from alsgls.ops import apply_siginv_to_matrix, siginv_diag, woodbury_chol

RTOL = 5e-8
ATOL = 5e-9

def rand_spd_diag(K, rng):
    # moderately conditioned diagonal
    d = 0.25 + rng.random(K)
    return d

def test_woodbury_matches_dense_right_multiply():
    rng = np.random.default_rng(42)
    N, K, k = 7, 9, 3
    F = rng.standard_normal((K, k))
    D = rand_spd_diag(K, rng)
    M = rng.standard_normal((N, K))

    S = F @ F.T + np.diag(D)
    S_inv = npl.inv(S)

    dense = M @ S_inv
    # Woodbury path (default uses explicit small inverse internally)
    wb = apply_siginv_to_matrix(M, F, D)

    assert np.allclose(dense, wb, rtol=RTOL, atol=ATOL)

def test_diag_of_siginv_matches_dense():
    rng = np.random.default_rng(123)
    K, k = 12, 4
    F = rng.standard_normal((K, k))
    D = rand_spd_diag(K, rng)

    S = F @ F.T + np.diag(D)
    S_inv = npl.inv(S)

    Dinv, C_chol = woodbury_chol(F, D)
    d_w = siginv_diag(F, Dinv, C_chol)
    d_d = np.diag(S_inv)

    assert np.allclose(d_w, d_d, rtol=RTOL, atol=ATOL)

def test_nll_logdet_via_cholesky_consistency():
    rng = np.random.default_rng(7)
    N, K, k = 10, 8, 3
    R = rng.standard_normal((N, K))
    F = rng.standard_normal((K, k))
    D = 0.3 + rng.random(K)

    # NLL via library
    val = nll_per_row(R, F, D)
    assert np.isfinite(val)

    # Dense cross-check for the two analytic pieces:
    S = F @ F.T + np.diag(D)
    # tr(R S^{-1} R^T) = sum_ij (R S^{-1})_{ij} R_{ij}
    quad_dense = float(np.sum(R @ npl.inv(S) * R))
    logdet_dense = float(np.linalg.slogdet(S)[1])
    val_dense = 0.5 * (quad_dense / N + logdet_dense + K * np.log(2.0 * np.pi))

    assert np.allclose(val, val_dense, rtol=RTOL, atol=ATOL)


def test_determinant_lemma_matches_dense():
    rng = np.random.default_rng(19)
    K, k = 7, 3
    F = rng.standard_normal((K, k))
    D = rand_spd_diag(K, rng)

    S = F @ F.T + np.diag(D)
    logdet_dense = float(np.linalg.slogdet(S)[1])

    Dinv, C_chol = woodbury_chol(F, D)
    logdet_diag = np.sum(np.log(np.clip(D, 1e-30, None)))
    logdet_small = 2.0 * np.sum(np.log(np.diag(C_chol)))
    logdet_lemma = logdet_diag + logdet_small

    assert np.allclose(logdet_dense, logdet_lemma, rtol=RTOL, atol=ATOL)
