import numpy as np
from .ops import woodbury_pieces

def mse(Y, Yhat):
    return float(np.mean((Y - Yhat) ** 2))

def nll_per_row(R, F, D):
    """Negative log-likelihood per row for residual matrix ``R``.

    Parameters
    ----------
    R : ndarray (N × K)
        Residual matrix.
    F : ndarray (K × k)
        Factor loadings.
    D : ndarray (K,)
        Diagonal noise variances.

    Returns
    -------
    float
        ``0.5 * [tr(R Σ^{-1} R^T) + logdet(Σ) + K log(2π)]``
        averaged over rows.
    """
    K = R.shape[1]
    Dinv, C_inv = woodbury_pieces(F, D)  # C_inv = (I + F^T D^{-1} F)^{-1}
    # tr(R Σ^{-1} R^T) = sum over rows of r Σ^{-1} r^T
    # Efficiently: R Σ^{-1} = apply_siginv_to_matrix(R, F, D), but avoid circular import.
    # Inline Woodbury:
    RDinv = R * Dinv[None, :]
    T1 = RDinv @ F          # N x k
    T2 = T1 @ C_inv         # N x k
    RSinv = RDinv - T2 @ (F.T * Dinv)  # N x K
    quad = float(np.sum(RSinv * R))
    # logdet via matrix determinant lemma:
    # det(FF^T + D) = det(D) det(I + F^T D^{-1} F)
    logdet = float(np.sum(np.log(np.clip(D, 1e-12, None)))) \
             + float(np.linalg.slogdet(np.eye(F.shape[1]) + F.T @ (F * Dinv[:, None]))[1])
    return 0.5 * (quad / R.shape[0] + logdet + K * np.log(2 * np.pi))
