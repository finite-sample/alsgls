import numpy as np
from .ops import XB_from_Blist

def em_gls(Xs, Y, k, lam_F=1e-3, lam_B=1e-3, iters=30, d_floor=1e-8):
    """
    Dense-ish EM baseline for low-rank+diag GLS.
    Builds Σ^{-1} explicitly (KxK) in the β-step to mimic O(K^2) memory.
    Returns (B_list, F, D, mem_MB_est, info)
    """
    # Input validation
    if not isinstance(Xs, list) or len(Xs) == 0:
        raise ValueError("Xs must be a non-empty list of arrays")
    if Y.ndim != 2:
        raise ValueError("Y must be a 2D array")
    N, K = Y.shape
    if len(Xs) != K:
        raise ValueError(f"Number of X matrices ({len(Xs)}) must match Y columns ({K})")
    for j, X in enumerate(Xs):
        if X.ndim != 2 or X.shape[0] != N:
            raise ValueError(f"X[{j}] must be 2D with {N} rows")
    if not (1 <= k <= min(K, N)):
        raise ValueError(f"k must be between 1 and min(K={K}, N={N})")
    if lam_F < 0 or lam_B < 0:
        raise ValueError("Regularization parameters must be non-negative")
    
    p_list = [X.shape[1] for X in Xs]

    # init B (OLS per equation)
    B = []
    for j, X in enumerate(Xs):
        XtX = X.T @ X + lam_B * np.eye(X.shape[1])
        Xty = X.T @ Y[:, [j]]
        B.append(np.linalg.solve(XtX, Xty))

    R = Y - XB_from_Blist(Xs, B)
    U, s, Vt = np.linalg.svd(R, full_matrices=False)
    r = min(k, (s > 1e-8).sum() or 1)
    F = Vt.T[:, :r] * np.sqrt(np.maximum(s[:r], 1e-12))
    if r < k:
        F = np.pad(F, ((0, 0), (0, k - r)))
    D = np.maximum(np.var(R, axis=0), d_floor)

    # Precompute Gram blocks
    G = [[Xs[j].T @ Xs[l] for l in range(K)] for j in range(K)]

    for _ in range(iters):
        # E-step-like: nothing explicit (we directly update F,D after β)

        # β-step (dense normal equations using Σ^{-1})
        Dinv = 1.0 / np.clip(D, 1e-12, None)
        M = F.T @ (F * Dinv[:, None])             # k x k
        Cf = np.linalg.inv(np.eye(k) + M)
        # Build Σ^{-1} explicitly (KxK)
        Sigma_inv = np.diag(Dinv) - (F * Dinv[:, None]) @ Cf @ (F.T * Dinv[None, :])

        A = np.zeros((sum(p_list), sum(p_list)))
        rhs = np.zeros((sum(p_list), 1))
        # Blocks A_{j,l} = Σ^{-1}_{l,j} * X_j^T X_l
        row = 0
        for j in range(K):
            col = 0
            Sj = Sigma_inv[:, j]
            for l in range(K):
                A[row:row+G[j][l].shape[0], col:col+G[j][l].shape[1]] = Sj[l] * G[j][l]
                col += G[j][l].shape[1]
            rhs[row:row+G[j][j].shape[0], :] = Xs[j].T @ (Y @ Sj.reshape(-1, 1))
            row += G[j][j].shape[0]
        A += lam_B * np.eye(A.shape[0])
        bvec = np.linalg.solve(A, rhs).ravel()
        B = []
        i = 0
        for p in p_list:
            B.append(bvec[i:i+p].reshape(p, 1))
            i += p

        # Update residuals and then F, D
        R = Y - XB_from_Blist(Xs, B)
        # Update scores/loadings by two ridge solves
        FtF = F.T @ F + lam_F * np.eye(F.shape[1])
        Uhat = R @ F @ np.linalg.inv(FtF)
        UtU = Uhat.T @ Uhat + lam_F * np.eye(F.shape[1])
        F = R.T @ Uhat @ np.linalg.inv(UtU)
        D = np.maximum(np.mean((R - Uhat @ F.T) ** 2, axis=0), d_floor)

    mem_mb_est = (K * K) * 8 / 1e6  # explicit Σ^{-1}
    info = {"p_list": p_list}
    return B, F, D, mem_mb_est, info
