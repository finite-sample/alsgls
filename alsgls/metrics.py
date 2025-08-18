import numpy as np
from .ops import woodbury_pieces

def mse(Y, Yhat):
    return float(np.mean((Y - Yhat) ** 2))

def nll_per_row(R, F, D, *, Dinv=None, Cf=None):
    """Negative log-likelihood per row for residual matrix ``R``.

    Parameters
    ----------
    R : ndarray (N × K)
        Residual matrix.
    F : ndarray (K × k)
        Factor loadings.
    D : ndarray (K,)
        Diagonal noise variances.
    Dinv : ndarray (K,), optional
        Precomputed ``1/D`` values.  If ``None`` (default) they are computed
        internally via :func:`woodbury_pieces`.
    Cf : ndarray (k × k), optional
        Precomputed ``(I + F^T D^{-1} F)^{-1}``.  If ``None`` (default) it is
        computed internally via :func:`woodbury_pieces`.

    Returns
    -------
    float
        ``0.5 * [tr(R Σ^{-1} R^T) + logdet(Σ) + K log(2π)]``
        averaged over rows.
    """
    K = R.shape[1]
    if Dinv is None or Cf is None:
        Dinv, Cf, _ = woodbury_pieces(F, D)
    # tr(R Σ^{-1} R^T) = sum over rows of r Σ^{-1} r^T
    # Efficiently: R Σ^{-1} = apply_siginv_to_matrix(R, F, D), but avoid circular import.
    # Inline Woodbury:
    RDinv = R * Dinv[None, :]
    T1 = RDinv @ F          # N x k
    T2 = T1 @ Cf            # N x k
    RSinv = RDinv - T2 @ (F.T * Dinv)  # N x K
    quad = float(np.sum(RSinv * R))
    # logdet via matrix determinant lemma:
    # det(FF^T + D) = det(D) det(I + F^T D^{-1} F)
    # Use Cf = (I + F^T D^{-1} F)^{-1} to avoid recomputing F^T D^{-1} F
    logdet = float(
        np.sum(np.log(np.clip(D, 1e-12, None))) - np.linalg.slogdet(Cf)[1]
    )
    return 0.5 * (quad / R.shape[0] + logdet + K * np.log(2 * np.pi))
