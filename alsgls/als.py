import numpy as np
from .ops import (
    apply_siginv_to_matrix,
    stack_B_list,
    unstack_B_vec,
    XB_from_Blist,
    cg_solve,
    woodbury_chol,
    siginv_diag,
)
from .metrics import nll_per_row


def als_gls(
    Xs,
    Y,
    k,
    lam_F: float = 1e-3,
    lam_B: float = 1e-3,
    sweeps: int = 8,
    d_floor: float = 1e-8,
    cg_maxit: int = 800,
    cg_tol: float = 3e-7,
    *,
    scale_correct: bool = True,
    scale_floor: float = 1e-8,
):
    """
    Alternating-least-squares GLS with a low-rank-plus-diagonal covariance model.
    Uses Woodbury throughout to avoid materializing K×K dense matrices.

    Additions:
      - Cached Woodbury pieces per sweep.
      - Stronger block-Jacobi preconditioner that incorporates diag(Σ^{-1}).
      - PCA init of F scaled so F F^T ≈ R^T R / N.
      - Optional MLE scale-correction of Σ each sweep (scale_correct=True).

    Parameters
    ----------
    Xs : list of (N×p_j) arrays
        Per-equation design matrices (length K).
    Y : (N×K) array
        Response matrix (columns are equations).
    k : int
        Target rank for the low-rank component (F has shape K×k).
    lam_F : float
        Ridge regularization for ALS updates of U and F.
    lam_B : float
        Ridge regularization for the β-step (per-equation coefficients).
    sweeps : int
        Maximum number of ALS sweeps.
    d_floor : float
        Floor for diagonal noise variances D to ensure SPD.
    cg_maxit : int
        Maximum iterations for CG in the β-step.
    cg_tol : float
        Relative tolerance for CG in the β-step.
    scale_correct : bool
        If True, apply an MLE scalar correction to Σ each sweep.
    scale_floor : float
        Minimum scalar used in the scale correction (guards degeneracy).

    Returns
    -------
    B_list : list of (p_j×1) arrays
    F : (K×k) array
    D : (K,) array
    mem_MB_est : float
        Rough memory footprint estimate in MB for major state (F, D, U).
    info : dict
        Includes p_list, cg info, and an NLL trace per sweep.
    """
    # ----------------------------
    # Input validation
    # ----------------------------
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

    # ----------------------------
    # Initialization
    # ----------------------------
    # Start with per-equation ridge/OLS for B
    B = []
    for j, X in enumerate(Xs):
        p = X.shape[1]
        XtX = X.T @ X + lam_B * np.eye(p)
        Xty = X.T @ Y[:, [j]]
        B.append(np.linalg.solve(XtX, Xty))

    # Residuals
    R = Y - XB_from_Blist(Xs, B)

    # PCA-like init for F with scale matched to column covariance: R^T R / N
    # SVD: R = U diag(s) V^T, so R^T R / N = V diag(s^2 / N) V^T.
    # Set F = V diag(s / sqrt(N)) to get F F^T ≈ R^T R / N.
    if N > 0:
        _, s, Vt = np.linalg.svd(R, full_matrices=False)
        if s.size == 0:
            F = np.zeros((K, k))
        else:
            s_thresh = max(float(s[0]) * 1e-10, 1e-8)
            r = int(min(k, max(1, (s > s_thresh).sum())))
            F = Vt.T[:, :r] * (s[:r] / np.sqrt(max(N, 1.0)))
            if r < k:
                F = np.pad(F, ((0, 0), (0, k - r)))
    else:
        F = np.zeros((K, k))

    # Diagonal noise (start from residual variances, floored)
    D = np.maximum(np.var(R, axis=0), d_floor)

    # ----------------------------
    # Main ALS loop
    # ----------------------------
    prev_nll = None
    nll_trace = []
    cg_info = None

    for _ in range(sweeps):
        # Cache Woodbury pieces once per sweep
        Dinv, C_chol = woodbury_chol(F, D)

        # Build a better diagonal preconditioner that reflects Σ^{-1} diagonal
        # Σ^{-1} = D^{-1} - D^{-1} F (I + F^T D^{-1} F)^{-1} F^T D^{-1}
        # diag(Σ^{-1})_j = Dinv_j - Dinv_j^2 * f_j^T C^{-1} f_j
        diag_sinv = siginv_diag(F, Dinv, C_chol)  # (K,)

        # Precompute the diagonal of the CG operator for a block-Jacobi preconditioner
        # Each block j contributes diag(X_j^T Σ^{-1} X_j) ≈ diag_sinv[j] * diag(X_j^T X_j)
        block_diags = [diag_sinv[j] * np.sum(X * X, axis=0) for j, X in enumerate(Xs)]
        Mpre_diag = np.concatenate(block_diags, axis=0) + lam_B

        def M_pre(v):
            return v / np.maximum(Mpre_diag, 1e-8)

        # Matrix-free normal operator H(B) = X^T Σ^{-1} X · b + lam_B b
        def A_mv(bvec):
            B_dir = unstack_B_vec(bvec, p_list)
            M = XB_from_Blist(Xs, B_dir)  # N × K
            S = apply_siginv_to_matrix(M, F, D, Dinv=Dinv, C_chol=C_chol)  # N × K
            out_blocks = [Xs[j].T @ S[:, [j]] for j in range(K)]
            out = np.concatenate(out_blocks, axis=0).ravel()
            return out + lam_B * bvec

        # β-step via CG
        S_y = apply_siginv_to_matrix(Y, F, D, Dinv=Dinv, C_chol=C_chol)
        rhs_blocks = [Xs[j].T @ S_y[:, [j]] for j in range(K)]
        b = np.concatenate(rhs_blocks, axis=0).ravel()
        bvec0 = stack_B_list(B)
        bvec, cg_info = cg_solve(
            A_mv, b, x0=bvec0, maxit=cg_maxit, tol=cg_tol, M_pre=M_pre
        )
        B = unstack_B_vec(bvec, p_list)

        # Residuals with new B
        R = Y - XB_from_Blist(Xs, B)

        # Two ridge LS updates for U (scores) and F (loadings)
        FtF = F.T @ F + lam_F * np.eye(F.shape[1])
        U = np.linalg.solve(FtF, (R @ F).T).T  # N × k
        UtU = U.T @ U + lam_F * np.eye(F.shape[1])
        F = np.linalg.solve(UtU, U.T @ R).T     # K × k

        # Update diagonal noise with floor
        D = np.maximum(np.mean((R - U @ F.T) ** 2, axis=0), d_floor)

        # --- MLE scale correction for Σ = c * (F F^T + diag D)
        # NLL(c) = 0.5 * [ (1/c) * (1/N) tr(R Σ^{-1} R^T) + K log c ] + const
        # c* = (1/NK) tr(R Σ^{-1} R^T) with Σ built from current (F, D).
        if scale_correct:
            Dinv_sc, C_chol_sc = woodbury_chol(F, D)
            RSinv_sc = apply_siginv_to_matrix(R, F, D, Dinv=Dinv_sc, C_chol=C_chol_sc)
            quad_over_N = float(np.sum(RSinv_sc * R)) / N
            c_star = max(quad_over_N / K, scale_floor)
            # Rescale F and D to apply Σ ← c* Σ
            F *= np.sqrt(c_star)
            D *= c_star

        # Track true NLL (cheap via Woodbury + det-lemma)
        cur_nll = float(nll_per_row(R, F, D))
        nll_trace.append(cur_nll)

        if np.isfinite(cur_nll):
            if prev_nll is not None:
                rel = (prev_nll - cur_nll) / max(1.0, abs(prev_nll))
                if rel < 1e-6:
                    break
            prev_nll = cur_nll

    # Memory estimate: F (K×k) + D (K) + U (N×k) doubles
    mem_mb_est = (K * F.shape[1] + K + N * F.shape[1]) * 8 / 1e6

    info = {"p_list": p_list, "cg": cg_info, "nll_trace": nll_trace}

    return B, F, D, mem_mb_est, info
