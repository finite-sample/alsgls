from __future__ import annotations

import numpy as np

from .ops import apply_siginv_to_matrix, woodbury_chol


def mse(Y: np.ndarray, Yhat: np.ndarray) -> float:
    """Mean squared error between two matrices."""
    return float(np.mean((Y - Yhat) ** 2))


def nll_per_row(R: np.ndarray, F: np.ndarray, D: np.ndarray) -> float:
    """
    Negative log-likelihood per row for residual matrix ``R`` under
    Σ = F F^T + diag(D) with Gaussian errors.

    Returns
    -------
    float
        0.5 * [ tr(R Σ^{-1} R^T)/N + log det(Σ) + K log(2π) ]
        where N is the number of rows in R.
    """
    N, K = R.shape
    # Woodbury factors
    Dinv, C_chol = woodbury_chol(F, D)

    # Quadratic term: tr(R Σ^{-1} R^T) via one right-multiply by Σ^{-1}
    RSinv = apply_siginv_to_matrix(R, F, D, Dinv=Dinv, C_chol=C_chol)
    quad = float(np.sum(RSinv * R))

    # log det(Σ) = log det(D) + log det(I + F^T D^{-1} F)
    # Use Cholesky to get log det of the small core stably.
    logdet_D = float(np.sum(np.log(np.clip(D, 1e-12, None))))
    logdet_core = 2.0 * float(np.sum(np.log(np.diag(C_chol))))
    logdet = logdet_D + logdet_core

    return float(0.5 * (quad / N + logdet + K * np.log(2.0 * np.pi)))
