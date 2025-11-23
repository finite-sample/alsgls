import numpy as np


def test_scale_derivative_zero_at_cstar():
    rng = np.random.default_rng(1)
    N,K,k = 100, 15, 3
    R = rng.standard_normal((N,K))
    F = rng.standard_normal((K,k)) / np.sqrt(K)
    D = 0.3 + rng.random(K)
    from alsgls.metrics import nll_per_row
    from alsgls.ops import apply_siginv_to_matrix, woodbury_chol
    Dinv, C_chol = woodbury_chol(F,D)
    RSinv = apply_siginv_to_matrix(R,F,D,Dinv=Dinv,C_chol=C_chol)
    quad_over_N = float(np.sum(RSinv*R))/N
    cstar = quad_over_N / K
    # finite-diff derivative
    eps = 1e-6
    nll_plus  = nll_per_row(R, F*np.sqrt(cstar+eps), D*(cstar+eps))
    nll_minus = nll_per_row(R, F*np.sqrt(cstar-eps), D*(cstar-eps))
    deriv = (nll_plus - nll_minus) / (2*eps)
    assert abs(deriv) < 1e-6
