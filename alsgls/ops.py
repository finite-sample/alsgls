import numpy as np


def woodbury_pieces(F: np.ndarray, D: np.ndarray):
    """
    Return (Dinv, C_inv) used in Woodbury for Σ = F F^T + diag(D).

    Σ^{-1} = D^{-1} - D^{-1} F C^{-1} F^T D^{-1},  where C = (I + F^T D^{-1} F).
    For backward compatibility, this returns the explicit small inverse C_inv.
    """
    D = np.asarray(D)
    Dinv = 1.0 / np.clip(D, 1e-12, None)
    FtDinv = (F.T * Dinv)            # k × K (row-scale F^T by Dinv)
    M = FtDinv @ F                   # k × k  == F^T D^{-1} F
    # Solve small k×k system to get explicit C_inv for callers that expect it
    C = np.eye(F.shape[1]) + M
    C_inv = np.linalg.solve(C, np.eye(F.shape[1]))
    return Dinv, C_inv


def woodbury_chol(F: np.ndarray, D: np.ndarray):
    """
    Return (Dinv, C_chol) where C_chol is the Cholesky factor of C = I + F^T D^{-1} F.

    Intended for numerically stable downstream solves that avoid forming C^{-1} explicitly.
    """
    D = np.asarray(D)
    Dinv = 1.0 / np.clip(D, 1e-12, None)
    FtDinv = (F.T * Dinv)            # k × K
    M = FtDinv @ F                   # k × k
    C = np.eye(F.shape[1]) + M
    C_chol = np.linalg.cholesky(C)   # upper or lower (NumPy returns lower-triangular)
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
    Y = np.linalg.solve(C_chol, T)       # C_chol Y = T
    X = np.linalg.solve(C_chol.T, Y)     # C_chol^T X = Y
    return X


def apply_siginv_to_matrix(
    M: np.ndarray,
    F: np.ndarray,
    D: np.ndarray,
    *,
    Dinv: np.ndarray | None = None,
    C_inv: np.ndarray | None = None,
    C_chol: np.ndarray | None = None,
) -> np.ndarray:
    """
    Right-multiply an (N×K) matrix M by Σ^{-1} using Woodbury, where
    Σ = F F^T + diag(D).

    You may pass either C_inv (explicit inverse of I + F^T D^{-1} F) or
    C_chol (its Cholesky factor). If neither is provided, a small explicit
    inverse will be computed internally for convenience.
    """
    if Dinv is None:
        Dinv = 1.0 / np.clip(np.asarray(D), 1e-12, None)

    if C_chol is not None:
        # Numerically stable path with Cholesky solves (preferred)
        MDinv = M * Dinv[None, :]
        T1 = MDinv @ F                   # (N × k)
        # Compute T2 = T1 @ C^{-1} without forming C^{-1}
        # Solve C Z^T = T1^T  ->  Z^T = C^{-1} T1^T  ->  T2 = Z^T
        ZT = _right_solve_with_C(T1.T, C_chol)  # (k × N)
        T2 = ZT.T
        T3 = T2 @ (F.T * Dinv)           # (N × K)
        return MDinv - T3

    # Fallback to explicit inverse if provided (or compute it)
    if C_inv is None:
        Dinv_tmp, C_inv = woodbury_pieces(F, D)
        # If caller provided Dinv, prefer it (they may have a cached copy)
        if Dinv is None:
            Dinv = Dinv_tmp

    MDinv = M * Dinv[None, :]
    T1 = MDinv @ F                       # (N × k)
    T2 = T1 @ C_inv                      # (N × k)
    T3 = T2 @ (F.T * Dinv)               # (N × K)
    return MDinv - T3


def stack_B_list(B_list):
    """Stack list of (p_j×1) blocks into a flat vector."""
    return np.concatenate([b.ravel() for b in B_list], axis=0)


def unstack_B_vec(bvec, p_list):
    """Inverse of stack: vector -> list of (p_j×1)."""
    out, i = [], 0
    for p in p_list:
        out.append(bvec[i : i + p].reshape(p, 1))
        i += p
    return out


def XB_from_Blist(Xs, B_list):
    """Return N × K matrix of predictions."""
    return np.column_stack([Xs[j] @ B_list[j] for j in range(len(Xs))])


def cg_solve(operator_mv, b, x0=None, maxit=500, tol=1e-7, M_pre=None):
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
            raise ValueError("Operator is not SPD: p^T A p ≤ 0")

        alpha = rz_old / pAp
        x += alpha * p
        r -= alpha * Ap

        res_norm = np.linalg.norm(r)
        if res_norm <= tol * (np.linalg.norm(b) + 1e-30):
            break

        z = M_pre(r) if M_pre is not None else r
        rz_new = float(r @ z)
        if rz_old <= 0:
            raise ValueError("Preconditioner is not SPD: r^T z ≤ 0")
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
    row_q = np.sum(F * Cinv_Ft.T, axis=1)       # (K,)
    diag_Sinv = Dinv - (Dinv ** 2) * row_q
    return diag_Sinv

