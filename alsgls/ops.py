from __future__ import annotations

from typing import Any, Callable

import numpy as np



def woodbury_chol(F: np.ndarray, D: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (Dinv, C_chol) where C_chol is the Cholesky factor of C = I + F^T D^{-1} F.

    Intended for numerically stable downstream solves that avoid forming C^{-1} explicitly.
    """
    D = np.asarray(D)
    Dinv = 1.0 / np.clip(D, 1e-12, None)
    FtDinv = F.T * Dinv  # k × K
    M = FtDinv @ F  # k × k
    C = np.eye(F.shape[1]) + M
    C_chol = np.linalg.cholesky(C)  # upper or lower (NumPy returns lower-triangular)
    return Dinv, C_chol


def _right_solve_with_C(T: np.ndarray, C_chol: np.ndarray) -> np.ndarray:
    """
    Solve (I + F^T D^{-1} F)^{-1} T for multiple RHS given the Cholesky factor.

    Parameters
    ----------
    T : (k × m) array
        Right-hand sides (as columns).
    C_chol : (k × k) array
        Cholesky factor of C = I + F^T D^{-1} F.

    Returns
    -------
    X : (k × m) array solving C X = T.
    """
    # Solve C X = T using two triangular solves with the Cholesky factor
    # NumPy's solve works fine for triangular systems as well.
    Y = np.linalg.solve(C_chol, T)  # C_chol Y = T
    X = np.linalg.solve(C_chol.T, Y)  # C_chol^T X = Y
    return X


def apply_siginv_to_matrix(
    M: np.ndarray,
    F: np.ndarray,
    D: np.ndarray,
    *,
    Dinv: np.ndarray | None = None,
    C_chol: np.ndarray,
) -> np.ndarray:
    """
    Right-multiply an (N×K) matrix M by Σ^{-1} using Woodbury, where
    Σ = F F^T + diag(D).

    Uses numerically stable Cholesky factorization approach.
    
    Parameters
    ----------
    M : np.ndarray
        (N×K) matrix to right-multiply by Σ^{-1}
    F : np.ndarray
        (K×k) factor loadings matrix
    D : np.ndarray
        (K,) diagonal noise variances
    Dinv : np.ndarray, optional
        Pre-computed 1/D. If None, computed from D.
    C_chol : np.ndarray
        Cholesky factor of C = I + F^T D^{-1} F
        
    Returns
    -------
    np.ndarray
        M @ Σ^{-1}
    """
    if Dinv is None:
        Dinv = 1.0 / np.clip(np.asarray(D), 1e-12, None)

    MDinv = M * Dinv[None, :]
    T1 = MDinv @ F  # (N × k)
    # Compute T2 = T1 @ C^{-1} without forming C^{-1}
    # Solve C Z^T = T1^T  ->  Z^T = C^{-1} T1^T  ->  T2 = Z^T
    ZT = _right_solve_with_C(T1.T, C_chol)  # (k × N)
    T2 = ZT.T
    T3 = T2 @ (F.T * Dinv)  # (N × K)
    return np.asarray(MDinv - T3)


def stack_B_list(B_list: list[np.ndarray]) -> np.ndarray:
    """Stack list of (p_j×1) blocks into a flat vector."""
    return np.concatenate([b.ravel() for b in B_list], axis=0)


def unstack_B_vec(bvec: np.ndarray, p_list: list[int]) -> list[np.ndarray]:
    """Inverse of stack: vector -> list of (p_j×1)."""
    out, i = [], 0
    for p in p_list:
        out.append(bvec[i : i + p].reshape(p, 1))
        i += p
    return out


def XB_from_Blist(Xs: list[np.ndarray], B_list: list[np.ndarray]) -> np.ndarray:
    """Return N × K matrix of predictions."""
    return np.column_stack([Xs[j] @ B_list[j] for j in range(len(Xs))])


def cg_solve(
    operator_mv: Callable[[np.ndarray], np.ndarray], 
    b: np.ndarray, 
    x0: np.ndarray | None = None, 
    maxit: int = 500, 
    tol: float = 1e-7, 
    M_pre: Callable[[np.ndarray], np.ndarray] | None = None
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Conjugate gradient for SPD operator A (matrix-free).

    Parameters
    ----------
    operator_mv : callable
        Function that returns A @ x for a given x.
    b : ndarray
        Right-hand side.
    x0 : ndarray, optional
        Initial guess.
    maxit : int
        Maximum CG iterations.
    tol : float
        Relative residual tolerance.
    M_pre : callable, optional
        Preconditioner application: returns M^{-1} @ r.

    Returns
    -------
    x : ndarray
        Approximate solution.
    info : dict
        Iterations and final residual norm.
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
                "Try increasing regularization (lam_F, lam_B) or check input data for singularities."
            )

        alpha = rz_old / pAp
        x += alpha * p
        r -= alpha * Ap

        res_norm = np.linalg.norm(r)
        if res_norm <= tol * (np.linalg.norm(b) + 1e-30):
            break

        z = M_pre(r) if M_pre is not None else r
        rz_new = float(r @ z)
        if rz_old <= 0:
            raise ValueError(
                "Preconditioner is not positive definite: r^T z ≤ 0. "
                "This indicates a problem with the preconditioner. "
                "Try disabling preconditioning (M_pre=None) or using simpler preconditioning."
            )
        beta = rz_new / rz_old
        p = z + beta * p
        rz_old = rz_new

    info = {"iterations": iterations, "residual": float(np.linalg.norm(r))}
    return x, info


def siginv_diag(F: np.ndarray, Dinv: np.ndarray, C_chol: np.ndarray) -> np.ndarray:
    """
    Compute the diagonal of Σ^{-1} = D^{-1} - D^{-1} F C^{-1} F^T D^{-1}
    efficiently given Dinv and the Cholesky factor of C.

    Returns
    -------
    diag_Sinv : (K,) array
        The diagonal entries of Σ^{-1}.
    """
    # Compute C^{-1} F^T via two triangular solves
    Cinv_Ft = _right_solve_with_C(F.T, C_chol)  # (k × K)
    # Row-wise quadratic forms f_j^T C^{-1} f_j  =  sum over k of (F * (C^{-1} F^T)^T)
    row_q = np.sum(F * Cinv_Ft.T, axis=1)  # (K,)
    diag_Sinv = Dinv - (Dinv**2) * row_q
    return np.asarray(diag_Sinv)
