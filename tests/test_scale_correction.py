import numpy as np

from alsgls.metrics import nll_per_row
from alsgls.ops import apply_siginv_to_matrix, woodbury_chol

RTOL = 1e-9
ATOL = 1e-10

def _nll(R, F, D):
    return float(nll_per_row(R, F, D))

def test_mle_scalar_c_is_optimal_and_non_worsening():
    rng = np.random.default_rng(99)
    N, K, k = 80, 20, 4
    # Random residuals and a plausible Sigma shape
    R = rng.standard_normal((N, K))
    F = rng.standard_normal((K, k)) / np.sqrt(K)
    D = 0.4 + rng.random(K)

    # Compute c* = (1/NK) tr(R Σ^{-1} R^T)
    Dinv, C_chol = woodbury_chol(F, D)
    RSinv = apply_siginv_to_matrix(R, F, D, Dinv=Dinv, C_chol=C_chol)
    quad_over_N = float(np.sum(RSinv * R)) / N
    c_star = quad_over_N / K
    c_star = max(c_star, 1e-12)

    nll_before = _nll(R, F, D)
    nll_at_cstar = _nll(R, F * np.sqrt(c_star), D * c_star)

    # Scalar optimality: c* should be <= any nearby c
    for eps in [-1e-3, -5e-4, -1e-4, 1e-4, 5e-4, 1e-3]:
        cc = max(c_star * (1.0 + eps), 1e-12)
        nll_eps = _nll(R, F * np.sqrt(cc), D * cc)
        assert nll_at_cstar <= nll_eps + 5e-12

    # And it shouldn't be worse than doing nothing (barring fp dust)
    assert nll_at_cstar <= nll_before + 1e-12
