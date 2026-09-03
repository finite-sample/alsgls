"""Woodbury-based linear algebra kernels for the ALS-GLS solver."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


def woodbury_chol(F: np.ndarray, D: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (Dinv, C_chol) with the Cholesky factor of C = I + F^T D^{-1} F.

    Intended for numerically stable downstream solves that avoid forming
    C^{-1} explicitly.
    """
    D = np.asarray(D)
    Dinv = 1.0 / np.clip(D, 1e-12, None)
    FtDinv = F.T * Dinv  # k x K
    M = FtDinv @ F  # k x k
    C = np.eye(F.shape[1]) + M
    C_chol = np.linalg.cholesky(C)  # upper or lower (NumPy returns lower-triangular)
    return Dinv, C_chol


def _right_solve_with_C(T: np.ndarray, C_chol: np.ndarray) -> np.ndarray:
    """Solve (I + F^T D^{-1} F)^{-1} T for multiple RHS given the Cholesky factor.

    Args:
        T: Right-hand sides (as columns).
        C_chol: Cholesky factor of C = I + F^T D^{-1} F.

    Returns:
        X
    """
    # Solve C X = T using two triangular solves with the Cholesky factor
    # NumPy's solve works fine for triangular systems as well.
    Y = np.linalg.solve(C_chol, T)  # C_chol Y = T
    return np.linalg.solve(C_chol.T, Y)  # C_chol^T X = Y


def apply_siginv_to_matrix(
    M: np.ndarray,
    F: np.ndarray,
    D: np.ndarray,
    *,
    Dinv: np.ndarray | None = None,
    C_chol: np.ndarray,
) -> np.ndarray:
    """Right-multiply an (NxK) matrix M by Σ^{-1} using Woodbury.

    Σ = F F^T + diag(D), which is never formed densely.

    Uses numerically stable Cholesky factorization approach.

    Args:
        M: (NxK) matrix to right-multiply by Σ^{-1}
        F: (Kxk) factor loadings matrix
        D: (K,) diagonal noise variances
        Dinv: Pre-computed 1/D. If None, computed from D.
        C_chol: Cholesky factor of C = I + F^T D^{-1} F

    Returns:
        np.ndarray: M @ Σ^{-1}
    """
    if Dinv is None:
        Dinv = 1.0 / np.clip(np.asarray(D), 1e-12, None)

    MDinv = M * Dinv[None, :]
    T1 = MDinv @ F  # (N x k)
    # Compute T2 = T1 @ C^{-1} without forming C^{-1}
    # Solve C Z^T = T1^T  ->  Z^T = C^{-1} T1^T  ->  T2 = Z^T
    ZT = _right_solve_with_C(T1.T, C_chol)  # (k x N)
    T2 = ZT.T
    T3 = T2 @ (F.T * Dinv)  # (N x K)
    return np.asarray(MDinv - T3)


def stack_B_list(B_list: list[np.ndarray]) -> np.ndarray:
    """Stack list of (p_jx1) blocks into a flat vector."""
    return np.concatenate([b.ravel() for b in B_list], axis=0)


def unstack_B_vec(bvec: np.ndarray, p_list: list[int]) -> list[np.ndarray]:
    """Inverse of stack: vector -> list of (p_jx1)."""
    out, i = [], 0
    for p in p_list:
        out.append(bvec[i : i + p].reshape(p, 1))
        i += p
    return out


def XB_from_Blist(Xs: list[np.ndarray], B_list: list[np.ndarray]) -> np.ndarray:
    """Return N x K matrix of predictions."""
    return np.column_stack([Xs[j] @ B_list[j] for j in range(len(Xs))])


def cg_solve(
    operator_mv: Callable[[np.ndarray], np.ndarray],
    b: np.ndarray,
    x0: np.ndarray | None = None,
    maxit: int = 500,
    tol: float = 1e-7,
    M_pre: Callable[[np.ndarray], np.ndarray] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Conjugate gradient for SPD operator A (matrix-free).

    Args:
        operator_mv: Function that returns A @ x for a given x.
        b: Right-hand side.
        x0: Initial guess.
        maxit: Maximum CG iterations.
        tol: Relative residual tolerance.
        M_pre: Preconditioner application: returns M^{-1} @ r.

    Returns:
        x: Approximate solution.
        info: Iterations and final residual norm.

    Raises:
        ValueError: If the operator or the preconditioner turns out not to be
            positive definite, which conjugate gradients requires.
    """
    x = np.zeros_like(b) if x0 is None else x0.copy()
    r = b - operator_mv(x)
    z = M_pre(r) if M_pre is not None else r
    p = z.copy()
    rz_old = float(r @ z)
    iterations = 0

    for _ in range(maxit):
        iterations += 1

        Ap = operator_mv(p)
        pAp = float(p @ Ap)
        if pAp <= 0:
            raise ValueError(
                "Operator is not positive definite: p^T A p ≤ 0. "
                "This may indicate numerical issues or incorrectly specified problem. "
                "Try increasing regularization (lam_F, lam_B) or check "
                "input data for singularities."
            )

        alpha = rz_old / pAp
        x += alpha * p
        r -= alpha * Ap

        res_norm = np.linalg.norm(r)
        if res_norm <= tol * (np.linalg.norm(b) + 1e-30):
            break

        z = M_pre(r) if M_pre is not None else r
        rz_new = float(r @ z)
        # Guard the quantity actually divided by, and report the one that is
        # wrong. The check used to test rz_old while the message quoted r^T z,
        # so a non-positive rz_new was only caught one iteration later, after
        # it had already been used as a numerator.
        if rz_new <= 0 or rz_old <= 0:
            raise ValueError(
                f"Preconditioner is not positive definite: r^T z = {rz_new:.3e} "
                f"(previous {rz_old:.3e}), and conjugate gradients requires it "
                "to be positive. Try disabling preconditioning (M_pre=None) or "
                "using simpler preconditioning."
            )
        beta = rz_new / rz_old
        p = z + beta * p
        rz_old = rz_new

    info = {"iterations": iterations, "residual": float(np.linalg.norm(r))}
    return x, info


def siginv_diag(F: np.ndarray, Dinv: np.ndarray, C_chol: np.ndarray) -> np.ndarray:
    """Compute the diagonal of Σ^{-1} without forming the inverse.

    Σ^{-1} = D^{-1} - D^{-1} F C^{-1} F^T D^{-1}, evaluated from ``Dinv`` and the
    Cholesky factor of ``C``.

    Args:
        F: Factor loadings, ``(K, k)``.
        Dinv: Reciprocal of the diagonal variances, length ``K``.
        C_chol: Cholesky factor of the ``k x k`` Woodbury core.

    Returns:
        diag_Sinv: The diagonal entries of Σ^{-1}.
    """
    # Compute C^{-1} F^T via two triangular solves
    Cinv_Ft = _right_solve_with_C(F.T, C_chol)  # (k x K)
    # Row-wise quadratic forms f_j^T C^{-1} f_j  =  sum over k of (F * (C^{-1} F^T)^T)
    row_q = np.sum(F * Cinv_Ft.T, axis=1)  # (K,)
    diag_Sinv = Dinv - (Dinv**2) * row_q
    return np.asarray(diag_Sinv)


def apply_siginv_F(F: np.ndarray, Dinv: np.ndarray, C_chol: np.ndarray) -> np.ndarray:
    """Compute Σ^{-1} @ F efficiently using Woodbury.

    Σ^{-1} @ F = D^{-1} F - D^{-1} F C^{-1} F^T D^{-1} F

    Args:
        F: Factor loadings
        Dinv: Inverse diagonal
        C_chol: Cholesky factor of C = I + F^T D^{-1} F

    Returns:
        SinvF: Σ^{-1} @ F
    """
    # D^{-1} F
    DinvF = Dinv[:, None] * F  # K x k

    # F^T D^{-1} F
    FtDinvF = F.T @ DinvF  # k x k

    # C^{-1} (F^T D^{-1} F)
    Cinv_FtDinvF = _right_solve_with_C(FtDinvF, C_chol)  # k x k

    # D^{-1} F C^{-1} F^T D^{-1} F
    correction = DinvF @ Cinv_FtDinvF  # K x k

    return np.asarray(DinvF - correction)


def df_rescaled(
    F: np.ndarray, D: np.ndarray, n: int, p_list: Sequence[int]
) -> tuple[np.ndarray, np.ndarray]:
    """Rescale a fitted low-rank covariance for the residual degrees of freedom.

    ``Sigma_hat`` is estimated from residuals of a fitted model, so its entries
    are biased toward zero by the fitting; the correction every SUR package
    applies is ``Sigma_ij * n / sqrt((n - p_i)(n - p_j))`` (linearmodels
    ``debiased=True``, R systemfit ``methodResidCov="geomean"``, Stata
    ``sureg, dfk``). That elementwise scaling is ``diag(sqrt(c)) Sigma
    diag(sqrt(c))`` with ``c_i = n / (n - p_i)``, and since

        diag(sqrt(c)) (F F' + diag(D)) diag(sqrt(c))
            = (diag(sqrt(c)) F)(diag(sqrt(c)) F)' + diag(c * D),

    it preserves the low-rank-plus-diagonal structure exactly. Scale row ``i``
    of ``F`` by ``sqrt(c_i)`` and ``D_i`` by ``c_i``.

    This corrects the bias in ``Sigma_hat`` itself. It does nothing about the
    variance of ``Sigma_hat``, which is the larger part of the finite-sample
    shortfall in the plug-in standard errors; see ``BootstrapResults``.

    Args:
        F: Factor loadings, ``(K, k)``.
        D: Diagonal variances, length ``K``.
        n: Number of rows each equation was fitted on.
        p_list: Number of regressors in each equation, length ``K``.

    Returns:
        The rescaled ``(F, D)``.

    Raises:
        ValueError: If any equation has no residual degrees of freedom.
    """
    p_arr = np.asarray(p_list, dtype=float)
    if (n - p_arr <= 0).any():
        raise ValueError(
            "df rescale needs n > p_j in every equation; got "
            f"n={n} and p_list={list(p_list)}"
        )
    c = n / (n - p_arr)
    return np.sqrt(c)[:, None] * F, c * np.asarray(D, dtype=float)


def compute_XtSigmaInvX(
    Xs: list[np.ndarray],
    F: np.ndarray,
    D: np.ndarray,
    lam_B: float = 0.0,
) -> np.ndarray:
    """Compute (X'Σ⁻¹X + λI) using the Woodbury identity.

    For GLS with Σ = FF' + diag(D), we need the precision-weighted design
    matrix cross-product for computing coefficient standard errors:

        Var(β̂) = (X'Σ⁻¹X + λI)⁻¹

    This is exact only when ``lam_B`` is 0. For a ridge estimator the variance
    is the sandwich ``A⁻¹ (X'Σ⁻¹X) A⁻¹`` with ``A = X'Σ⁻¹X + λI``, and the form
    above understates it; at the default ``lam_B = 1e-3`` the difference is
    negligible, but it grows with the penalty. Σ is also treated as known rather
    than estimated, which is the usual feasible-GLS convention.

    Using Woodbury: Σ⁻¹ = D⁻¹ - D⁻¹F C⁻¹ F'D⁻¹ where C = I + F'D⁻¹F

    The block structure gives:
        [X'Σ⁻¹X]_{jl} = X_j' [Σ⁻¹]_{jl} X_l

    Args:
        Xs: List of design matrices [X_0, ..., X_{K-1}] where X_j is (N, p_j)
        F: (K, k) factor loadings matrix
        D: (K,) diagonal noise variances
        lam_B: Ridge penalty to add to diagonal (for regularization)

    Returns:
        XtSinvX: (p_total, p_total) matrix where p_total = sum(p_j)
    """
    K = len(Xs)
    p_list = [X.shape[1] for X in Xs]
    p_total = sum(p_list)

    Dinv, C_chol = woodbury_chol(F, D)

    Cinv_Ft = _right_solve_with_C(F.T, C_chol)

    XtSinvX = np.zeros((p_total, p_total))

    row_start = 0
    for j in range(K):
        p_j = p_list[j]
        X_j = Xs[j]
        d_j_inv = Dinv[j]
        f_j = F[j, :]

        col_start = 0
        for ell in range(K):
            p_ell = p_list[ell]
            X_ell = Xs[ell]
            d_ell_inv = Dinv[ell]

            if j == ell:
                XtX_jj = X_j.T @ X_j
                block = d_j_inv * XtX_jj
            else:
                block = np.zeros((p_j, p_ell))

            Cinv_f_ell = Cinv_Ft[:, ell]
            coef = d_j_inv * d_ell_inv * (f_j @ Cinv_f_ell)
            XtX_jl = X_j.T @ X_ell
            block -= coef * XtX_jl

            XtSinvX[row_start : row_start + p_j, col_start : col_start + p_ell] = block
            col_start += p_ell

        row_start += p_j

    if lam_B > 0:
        XtSinvX += lam_B * np.eye(p_total)

    return XtSinvX


def compute_prediction_variance(
    Xs: Sequence[np.ndarray],
    F: np.ndarray,
    D: np.ndarray,
    cov_params: np.ndarray,
    include_residual: bool = True,
) -> np.ndarray:
    """Compute prediction variances for new observations.

    For each observation i and equation j:
    - Var(E[y_j|X]) = X_j[i,:] @ Cov(β̂_j) @ X_j[i,:]
    - Var(y_j|X) = Var(E[y_j|X]) + Σ_jj where Σ_jj = ||F[j,:]||² + D[j]

    Args:
        Xs: List of design matrices [X_0, ..., X_{K-1}] where X_j is (N_new, p_j)
        F: (K, k) factor loadings matrix
        D: (K,) diagonal noise variances
        cov_params: (p_total, p_total) covariance matrix of parameter estimates
        include_residual: If True, add Σ_jj (residual variance) for
            prediction intervals. If False, return only the variance of the
            mean prediction (confidence intervals).

    Returns:
        var_pred: (N_new, K) array of prediction variances

    Raises:
        ValueError: If ``Xs`` is empty.
    """
    K = len(Xs)
    if K == 0:
        raise ValueError("Xs must contain at least one design matrix")

    N_new = Xs[0].shape[0]
    p_list = [X.shape[1] for X in Xs]
    offsets = np.cumsum([0, *p_list])

    var_pred = np.zeros((N_new, K))

    for j in range(K):
        X_j = Xs[j]
        cov_j = cov_params[offsets[j] : offsets[j + 1], offsets[j] : offsets[j + 1]]
        var_pred[:, j] = np.einsum("np,pq,nq->n", X_j, cov_j, X_j)

        if include_residual:
            sigma_jj = np.sum(F[j, :] ** 2) + D[j]
            var_pred[:, j] += sigma_jj

    return var_pred
