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
    s_thresh = max(s[0] * 1e-10, 1e-8) if len(s) > 0 else 1e-8
    r = min(k, (s > s_thresh).sum() or 1)
    F = Vt.T[:, :r] * np.sqrt(np.maximum(s[:r], 1e-12))
    if r < k:
        F = np.pad(F, ((0, 0), (0, k - r)))
    D = np.maximum(np.var(R, axis=0), d_floor)

    for _ in range(iters):
        # E-step-like: nothing explicit (we directly update F,D after β)

        # β-step (dense normal equations using Σ^{-1})
        Dinv = 1.0 / np.clip(D, 1e-12, None)
        M = F.T @ (F * Dinv[:, None])             # k x k
        C_inv = np.linalg.solve(np.eye(k) + M, np.eye(k))  # (I + F^T D^{-1} F)^{-1}
        # Build Σ^{-1} explicitly (KxK)
        Sigma_inv = np.diag(Dinv) - (F * Dinv[:, None]) @ C_inv @ (F.T * Dinv[None, :])

        A = np.zeros((sum(p_list), sum(p_list)))
        rhs = np.zeros((sum(p_list), 1))
        p_offsets = np.cumsum([0] + p_list)
        # Blocks A_{j,l} = Σ^{-1}_{l,j} * X_j^T X_l (symmetric in j,l)
        for j in range(K):
            Sj = Sigma_inv[:, j]
            Xj = Xs[j]
            r0, r1 = p_offsets[j], p_offsets[j + 1]
            rhs[r0:r1, :] = Xj.T @ (Y @ Sj.reshape(-1, 1))
            for l in range(j, K):
                c0, c1 = p_offsets[l], p_offsets[l + 1]
                block = Xj.T @ Xs[l]
                scalar = Sj[l]
                A[r0:r1, c0:c1] = scalar * block
                if l != j:
                    # Mirror to the symmetric block to maintain A symmetric
                    A[c0:c1, r0:r1] = scalar * block.T
        A += lam_B * np.eye(A.shape[0])
        A = (A + A.T) * 0.5  # enforce symmetry
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
        Uhat = np.linalg.solve(FtF, (R @ F).T).T
        UtU = Uhat.T @ Uhat + lam_F * np.eye(F.shape[1])
        F = np.linalg.solve(UtU, Uhat.T @ R).T
        D = np.maximum(np.mean((R - Uhat @ F.T) ** 2, axis=0), d_floor)

    mem_mb_est = (K * K) * 8 / 1e6  # explicit Σ^{-1}
    info = {"p_list": p_list}
    return B, F, D, mem_mb_est, info
