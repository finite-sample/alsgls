import numpy as np

def woodbury_pieces(F: np.ndarray, D: np.ndarray):
    """
    Return Dinv, Cf used in Woodbury:
    Σ = F F^T + diag(D)
    Σ^{-1} = D^{-1} - D^{-1} F (I + F^T D^{-1} F)^{-1} F^T D^{-1}
    """
    D = np.asarray(D)
    Dinv = 1.0 / np.clip(D, 1e-12, None)
    FtDinv = (F.T * Dinv)           # k x K (row-scale F.T by Dinv)
    M = F.T @ (F * Dinv[:, None])   # k x k == F^T D^{-1} F
    # solve small kxk: (I + M)^{-1}
    Cf = np.linalg.inv(np.eye(F.shape[1]) + M)
    return Dinv, Cf

def apply_siginv_to_matrix(M: np.ndarray, F: np.ndarray, D: np.ndarray):
    """
    Right-multiply an N x K matrix M by Σ^{-1} using Woodbury.
    """
    Dinv, Cf = woodbury_pieces(F, D)
    # M * Dinv - M*(Dinv F) Cf (F^T Dinv)
    MDinv = M * Dinv[None, :]
    T1 = MDinv @ F              # N x k
    T2 = T1 @ Cf                # N x k
    T3 = T2 @ (F.T * Dinv)      # N x K
    return MDinv - T3

def stack_B_list(B_list):
    """Stack list of (p_j,1) into flat vector."""
    return np.concatenate([b.ravel() for b in B_list], axis=0)

def unstack_B_vec(bvec, p_list):
    """Inverse of stack: vector -> list of (p_j,1)."""
    out, i = [], 0
    for p in p_list:
        out.append(bvec[i:i+p].reshape(p, 1))
        i += p
    return out

def XB_from_Blist(Xs, B_list):
    """Return N x K matrix of predictions."""
    return np.column_stack([Xs[j] @ B_list[j] for j in range(len(Xs))])

def cg_solve(operator_mv, b, x0=None, maxit=500, tol=1e-7, M_pre=None):
    """
    Conjugate gradient for SPD operator A (matrix-free).
    operator_mv(x) -> A x
    M_pre(x) -> apply preconditioner M^{-1} x (optional)
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
        alpha = rz_old / max(1e-30, float(p @ Ap))
        x += alpha * p
        r -= alpha * Ap
        res_norm = np.linalg.norm(r)
        if res_norm <= tol * (np.linalg.norm(b) + 1e-30):
            break
        z = M_pre(r) if M_pre is not None else r
        rz_new = float(r @ z)
        beta = rz_new / max(1e-30, rz_old)
        p = z + beta * p
        rz_old = rz_new
    info = {"iterations": iterations, "residual": float(np.linalg.norm(r))}
    return x, info
