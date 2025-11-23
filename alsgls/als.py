import numpy as np

from .metrics import nll_per_row
from .ops import (
    XB_from_Blist,
    apply_siginv_to_matrix,
    cg_solve,
    siginv_diag,
    stack_B_list,
    unstack_B_vec,
    woodbury_chol,
)


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
    rel_tol: float = 1e-6,
):
    """
    Alternating-least-squares GLS with low-rank-plus-diagonal covariance.
    Uses Woodbury throughout; never materializes K×K dense Σ.

    Enhancements (correctness-first):
      - Cached Woodbury pieces per sweep.
      - Block-Jacobi preconditioner using diag(Σ^{-1}).
      - PCA init of F with F F^T ≈ R^T R / N.
      - Guarded MLE scale-correction of Σ each sweep.
      - β-step REVERT if it worsens NLL (keeps trace non-increasing).
      - Backtracking/damped acceptance on (F, D) to accept only NLL-improving updates.
      - Dual traces in `info`: nll_beta_trace (post-β), nll_trace/nll_sigma_trace (post-Σ).

    Parameters
    ----------
    Xs : list of (N×p_j) arrays   (length K)
    Y  : (N×K) array
    k  : int                      rank of low-rank component (F: K×k)
    lam_F : float                 ridge for U/F ALS updates
    lam_B : float                 ridge for β-step (per-equation)
    sweeps : int                  max ALS sweeps
    d_floor : float               min variance for D to ensure SPD
    cg_maxit : int                max CG iterations in β-step
    cg_tol : float                CG relative tolerance
    scale_correct : bool          if True, try guarded MLE scale fix on Σ each sweep
    scale_floor : float           min scalar for scale correction
    rel_tol : float               relative NLL improvement threshold for early stopping

    Returns
    -------
    B_list, F, D, mem_MB_est, info
      info includes:
        - p_list
        - cg (last sweep)
        - nll_trace          (post-Σ, non-increasing)
        - nll_sigma_trace    (alias of nll_trace)
        - nll_beta_trace     (post-β baseline per sweep)
        - accept_t           (list of accepted backtracking t)
        - scale_used         (list of accepted scale factors, 1.0 if not applied)
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
    if rel_tol < 0:
        raise ValueError("rel_tol must be non-negative")

    p_list = [X.shape[1] for X in Xs]

    # ----------------------------
    # Initialization
    # ----------------------------
    # Per-equation ridge/OLS for B
    B = []
    for j, X in enumerate(Xs):
        p = X.shape[1]
        XtX = X.T @ X + lam_B * np.eye(p)
        Xty = X.T @ Y[:, [j]]
        B.append(np.linalg.solve(XtX, Xty))

    # Residuals
    R = Y - XB_from_Blist(Xs, B)

    # PCA-like init for F with scale matched to column covariance: R^T R / N
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
    # Traces & baseline
    # ----------------------------
    nll_trace = []
    nll_beta_trace = []
    accept_t_trace = []
    scale_used_trace = []

    # Starting NLL (explicit baseline before any sweep)
    nll_prev = float(nll_per_row(R, F, D))
    nll_trace.append(nll_prev)

    # ----------------------------
    # Main ALS loop
    # ----------------------------
    cg_info = None

    for _ in range(sweeps):
        # Cache Woodbury pieces once per sweep
        Dinv, C_chol = woodbury_chol(F, D)

        # diag(Σ^{-1}) for preconditioning: Σ^{-1} = D^{-1} - D^{-1}F C^{-1} F^T D^{-1}
        diag_sinv = siginv_diag(F, Dinv, C_chol)  # (K,)
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

        # --- β-step via CG (keep a copy to allow revert if NLL worsens)
        B_prev = [b.copy() for b in B]

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

        # Baseline NLL for this sweep *after* β-step (the per-sweep baseline)
        base_nll = float(nll_per_row(R, F, D))
        nll_beta_trace.append(base_nll)

        # If β worsened NLL, revert to previous B (ensures non-increase vs prior Σ)
        if base_nll > nll_prev + 1e-12:
            B = B_prev
            R = Y - XB_from_Blist(Xs, B)
            base_nll = nll_prev  # true baseline for this sweep

        # --- Propose U,F,D updates (unconstrained proposal)
        FtF = F.T @ F + lam_F * np.eye(F.shape[1])
        U_prop = np.linalg.solve(FtF, (R @ F).T).T  # N × k
        UtU = U_prop.T @ U_prop + lam_F * np.eye(F.shape[1])
        F_prop = np.linalg.solve(UtU, U_prop.T @ R).T  # K × k
        D_prop = np.maximum(np.mean((R - U_prop @ F_prop.T) ** 2, axis=0), d_floor)

        # Guarded scale correction helper (applied to a candidate F,D)
        def try_with_scale(F_try, D_try):
            """Return (F_out, D_out, nll_out, scale_used)."""
            nll0 = float(nll_per_row(R, F_try, D_try))
            if not scale_correct:
                return F_try, D_try, nll0, 1.0

            # MLE scalar c* for Σ = c * (F_try F_try^T + diag D_try)
            Dinv_s, C_chol_s = woodbury_chol(F_try, D_try)
            RSinv_s = apply_siginv_to_matrix(R, F_try, D_try, Dinv=Dinv_s, C_chol=C_chol_s)
            quad_over_N = float(np.sum(RSinv_s * R)) / N
            c_star = max(quad_over_N / K, scale_floor)

            sqrt_c = np.sqrt(c_star)
            F_sc = F_try * sqrt_c
            D_sc = D_try * c_star
            nll_sc = float(nll_per_row(R, F_sc, D_sc))

            if nll_sc <= nll0 + 1e-12:
                return F_sc, D_sc, nll_sc, c_star
            else:
                return F_try, D_try, nll0, 1.0

        # --- Backtracking/damped acceptance on (F, D)
        F_old, D_old = F, D
        dF = F_prop - F_old
        dD = D_prop - D_old

        best_nll = base_nll
        best_F, best_D = F_old, D_old
        accepted_t = 0.0
        used_scale = 1.0

        for t in (1.0, 0.5, 0.25, 0.125, 0.0625):
            F_try = F_old + t * dF
            D_try = np.maximum(D_old + t * dD, d_floor)
            F_acc, D_acc, nll_acc, sc_used = try_with_scale(F_try, D_try)
            # Accept only if we beat the per-sweep baseline
            if nll_acc < best_nll - 1e-12:
                best_nll = nll_acc
                best_F, best_D = F_acc, D_acc
                accepted_t = t
                used_scale = sc_used
                break  # first improving step is fine (monotone)

        # Accept (or keep old F,D if no improvement)
        F, D = best_F, best_D
        nll_curr = best_nll
        accept_t_trace.append(accepted_t)
        scale_used_trace.append(float(used_scale))

        # Append post-Σ NLL (non-increasing by construction)
        nll_trace.append(nll_curr)

        # Convergence: stop if relative improvement w.r.t previous post-Σ NLL is tiny
        rel_impr = (nll_prev - nll_curr) / max(1.0, abs(nll_prev))
        nll_prev = nll_curr
        if rel_impr < rel_tol:
            break

    # Memory estimate: F (K×k) + D (K) + U (N×k) doubles
    mem_mb_est = (K * F.shape[1] + K + N * F.shape[1]) * 8 / 1e6

    info = {
        "p_list": p_list,
        "cg": cg_info,
        "nll_trace": nll_trace,           # post-Σ
        "nll_sigma_trace": nll_trace,     # alias for clarity
        "nll_beta_trace": nll_beta_trace, # post-β (per-sweep baseline)
        "accept_t": accept_t_trace,       # accepted t (0.0 means kept previous F,D)
        "scale_used": scale_used_trace,   # accepted c* (1.0 means no scale applied)
    }
    return B, F, D, mem_mb_est, info
