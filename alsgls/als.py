import numpy as np
from .ops import (
    apply_siginv_to_matrix,
    stack_B_list, unstack_B_vec, XB_from_Blist, cg_solve
)

def als_gls(
    Xs, Y, k,
    lam_F=1e-3, lam_B=1e-3, sweeps=8, d_floor=1e-8,
    cg_maxit=800, cg_tol=3e-7
):
    """
    ALS for low-rank+diag GLS.
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
    # PCA-like init for F with relative threshold
    U, s, Vt = np.linalg.svd(R, full_matrices=False)
    s_thresh = max(s[0] * 1e-10, 1e-8) if len(s) > 0 else 1e-8
    r = min(k, (s > s_thresh).sum() or 1)
    F = Vt.T[:, :r] * np.sqrt(np.maximum(s[:r], 1e-12))
    if r < k:
        F = np.pad(F, ((0, 0), (0, k - r)))
    D = np.maximum(np.var(R, axis=0), d_floor)

    # main ALS loop
    prev = None
    cg_info = None
    for _ in range(sweeps):
        # Precompute D^{-1} once per sweep for the preconditioner
        Dinv = 1.0 / np.clip(D, 1e-12, None)

        def A_mv(bvec):
            """Matrix-free normal operator H(B) = X^T Σ^{-1} X · b + lam_B b"""
            B_dir = unstack_B_vec(bvec, p_list)
            M = XB_from_Blist(Xs, B_dir)                 # N x K
            S = apply_siginv_to_matrix(M, F, D)  # N x K
            out_blocks = []
            for j, X in enumerate(Xs):
                out_blocks.append(X.T @ S[:, [j]])
            out = np.concatenate(out_blocks, axis=0).ravel()
            return out + lam_B * bvec

        # simple diagonal preconditioner: approx diag of H
        def M_pre(v):
            diag_entries = []
            # Rough diag: X_j^T (Σ^{-1} e_j e_j^T) X_j ≈ X_j^T (Dinv_j) X_j
            for j, X in enumerate(Xs):
                w = float(Dinv[j])
                diag_entries.extend([w] * X.shape[1])
            d = np.array(diag_entries) + lam_B
            return v / np.maximum(d, 1e-8)

        # β-step via CG
        rhs_blocks = []
        S_y = apply_siginv_to_matrix(Y, F, D)
        for j, X in enumerate(Xs):
            rhs_blocks.append(X.T @ S_y[:, [j]])
        b = np.concatenate(rhs_blocks, axis=0).ravel()
        bvec0 = stack_B_list(B)
        bvec, cg_info = cg_solve(A_mv, b, x0=bvec0, maxit=cg_maxit, tol=cg_tol, M_pre=M_pre)
        B = unstack_B_vec(bvec, p_list)

        # factor step
        R = Y - XB_from_Blist(Xs, B)
        # update U (scores) and F (loadings) by two ridge solves
        FtF = F.T @ F + lam_F * np.eye(F.shape[1])
        U = np.linalg.solve(FtF, (R @ F).T).T
        UtU = U.T @ U + lam_F * np.eye(F.shape[1])
        F = np.linalg.solve(UtU, U.T @ R).T

        # diagonal noise
        D = np.maximum(np.mean((R - U @ F.T) ** 2, axis=0), d_floor)

        # cheap objective proxy: per-row NLL (using current F,D and R)
        obj = 0.5 * np.mean((R * (1.0 / np.sqrt(np.maximum(D, 1e-12)))) ** 2)  # weighted residuals
        if np.isfinite(obj):
            if prev is not None:
                rel = (prev - obj) / max(1.0, abs(prev))
                if rel < 1e-6:
                    break
            prev = obj

    # Memory estimate: F (K×k) + D (K) + U (N×k) + intermediate matrices
    mem_mb_est = (K * F.shape[1] + K + N * F.shape[1] + K * F.shape[1]) * 8 / 1e6
    info = {"p_list": p_list, "cg": cg_info}
    return B, F, D, mem_mb_est, info
