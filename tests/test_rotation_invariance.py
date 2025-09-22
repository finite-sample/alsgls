import numpy as np

from alsgls import als_gls
from alsgls.metrics import nll_per_row
from alsgls.ops import XB_from_Blist


def test_nll_invariant_under_factor_rotation():
    rng = np.random.default_rng(5)
    N, K, p, k = 150, 6, 4, 3
    Xs = [rng.standard_normal((N, p)) for _ in range(K)]
    B_true = [rng.standard_normal((p, 1)) for _ in range(K)]
    F_true = rng.standard_normal((K, k)) / np.sqrt(K)
    D_true = 0.2 + 0.3 * rng.random(K)
    Z = rng.standard_normal((N, k))
    Y = XB_from_Blist(Xs, B_true) + Z @ F_true.T + rng.standard_normal((N, K)) * np.sqrt(D_true)

    B_hat, F_hat, D_hat, _, _ = als_gls(Xs, Y, k=k, sweeps=10, rel_tol=1e-8)
    residuals = Y - XB_from_Blist(Xs, B_hat)
    base = float(nll_per_row(residuals, F_hat, D_hat))

    Q, _ = np.linalg.qr(rng.standard_normal((k, k)))
    F_rot = F_hat @ Q
    cov_diag = np.diag(F_hat @ F_hat.T + np.diag(D_hat))
    D_rot = np.maximum(cov_diag - np.sum(F_rot * F_rot, axis=1), 1e-12)

    rotated = float(nll_per_row(residuals, F_rot, D_rot))
    assert np.isclose(base, rotated, rtol=5e-8, atol=5e-9)
