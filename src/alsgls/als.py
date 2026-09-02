"""Alternating-least-squares solver for low-rank-plus-diagonal GLS."""

from __future__ import annotations

from typing import Any

import numpy as np

from ._validation import (
    _validate_convergence_params,
    _validate_gls_inputs,
    _validate_positive_float,
)
from .metrics import nll_per_row
from .ops import (
    XB_from_Blist,
    apply_siginv_to_matrix,
    cg_solve,
    grad_F_nll,
    siginv_diag,
    stack_B_list,
    unstack_B_vec,
    woodbury_chol,
)

# Maximum number of step halvings in the F/D backtracking line search.
# 40 halvings reach t ~ 9e-13, enough to find an improving step at any data
# scale encountered in double precision.
_MAX_BACKTRACK = 40


def als_gls(
    Xs: list[np.ndarray],
    Y: np.ndarray,
    k: int,
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
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, float, dict[str, Any]]:
    """Alternating-least-squares GLS with low-rank-plus-diagonal covariance.

    Uses Woodbury throughout; never materializes KxK dense Σ.

    Enhancements (correctness-first):
      - Cached Woodbury pieces per sweep.
      - Block-Jacobi preconditioner using diag(Σ^{-1}).
      - PCA init of F with F F^T ≈ R^T R / N.
      - Guarded MLE scale-correction of Σ each sweep.
      - β-step REVERT if it worsens NLL (keeps trace non-increasing).
      - Backtracking/damped acceptance on (F, D) to accept only NLL-improving updates.
      - Dual traces in `info`: nll_beta_trace (post-β) and
        nll_trace/nll_sigma_trace (post-Σ).

    Args:
        Xs: One design matrix per equation, each ``(N, p_j)``. The equations may
            have different numbers of regressors.
        Y: Responses, ``(N, K)``, one column per equation.
        k: Rank of the latent factor block. The covariance is modelled as
            ``F F^T + D`` with ``F`` of shape ``(K, k)``, so ``k`` controls how
            much cross-equation dependence is shared; ``k << K`` is the point.
        lam_F: Ridge penalty on the factor loadings ``F``.
        lam_B: Ridge penalty on the coefficients ``beta``.
        sweeps: Maximum number of alternating passes over ``beta`` and
            ``(F, D)``.
        d_floor: Floor on each diagonal variance, expressed as a fraction of the
            mean initial residual variance rather than as an absolute variance.
            It keeps ``D`` positive definite and the Woodbury inverse well
            conditioned. The floor is relative so that it transforms correctly
            under ``Y -> sY``, where the true ``D`` scales as ``s^2``; an
            absolute floor would bind on every entry once ``Y`` is small enough.
            With the default ``1e-8`` and residuals of variance ~2, the
            effective floor is ~2e-8.
        cg_maxit: Iteration cap for the conjugate-gradient solve in the beta step.
        cg_tol: Relative residual tolerance for that solve.
        scale_correct: Apply the guarded MLE scale correction to ``Sigma`` each
            sweep. The correction is reverted when it does not improve the
            negative log-likelihood.
        scale_floor: Smallest permitted scale factor, so the correction cannot
            collapse ``Sigma`` toward zero.
        rel_tol: Relative decrease in the negative log-likelihood below which the
            sweeps stop early.

    Returns:
        B_list, F, D, mem_MB_est, info: ``info`` includes ``p_list``, ``cg``
        (last sweep), ``nll_trace`` (post-Σ; equals ``nll_per_row`` at the
        returned parameters, and non-increasing when ``lam_F`` is 0),
        ``obj_trace`` (the penalised objective ``NLL + lam_F/2 ||F||^2`` that the
        line search descends, non-increasing by construction),
        ``nll_sigma_trace`` (alias of ``nll_trace``), ``nll_beta_trace``
        (post-β baseline per sweep), ``accept_t`` (accepted backtracking
        step sizes), and ``scale_used`` (accepted scale factors, 1.0 when
        not applied).

    Raises:
        np.linalg.LinAlgError: If a Cholesky factorisation of the Woodbury
            core fails, which means the current Sigma is not positive definite.
    """
    # ----------------------------
    # Input validation
    # ----------------------------
    Xs, Y, k, lam_F, lam_B = _validate_gls_inputs(Xs, Y, k, lam_F=lam_F, lam_B=lam_B)
    _validate_convergence_params(
        sweeps=sweeps, rel_tol=rel_tol, cg_maxit=cg_maxit, cg_tol=cg_tol
    )
    # d_floor <= 0 is not a weaker floor, it is a broken one: D can then reach
    # zero or go negative, while woodbury_chol and nll_per_row clip D at 1e-12
    # internally. The returned (F, D) would describe a different -- and not
    # positive definite -- Sigma from the one every reported number was computed
    # under, and Sigma_jj = ||F_j||^2 + D_j could come out negative in
    # compute_prediction_variance.
    d_floor = _validate_positive_float(
        d_floor,
        "d_floor",
        hint="Try d_floor=1e-8 (a fraction of the residual variance).",
    )
    scale_floor = _validate_positive_float(
        scale_floor, "scale_floor", hint="Try scale_floor=1e-8."
    )
    if not isinstance(scale_correct, bool):
        raise ValueError(f"scale_correct must be a bool, got {scale_correct!r}")
    N, K = Y.shape

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
        try:
            B.append(np.linalg.solve(XtX, Xty))
        except np.linalg.LinAlgError as exc:
            # lam_B used to be floored at 1e-3 by a falsy-zero bug, so a
            # rank-deficient design was quietly ridged into solvability. Now
            # that lam_B = 0 means zero, say what happened rather than letting
            # a bare "Singular matrix" out -- and do not silently substitute a
            # pseudo-inverse, which would be the same kind of quiet
            # substitution that hid this in the first place.
            msg = f"Equation {j} has a singular X'X" + (
                " and lam_B is 0, so there is nothing to regularize it. "
                "Drop the collinear columns or pass lam_B > 0."
                if lam_B == 0
                else f" even with lam_B={lam_B}. Drop the collinear columns."
            )
            raise np.linalg.LinAlgError(msg) from exc

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

    # D is a variance, so its floor has to be a variance too. A fixed absolute
    # floor is not scale-equivariant: under Y -> sY the true D scales as s^2,
    # so for small enough s every entry lands on the floor and the fit stops
    # tracking the data. Measured on B, which must satisfy B(sY) == B(Y):
    # the error saturated at 2.6e-2 for s <= 1e-4 with an absolute 1e-8.
    # Taken relative to the residual variance scale, d_floor keeps its meaning
    # ("no variance below this fraction of a typical one") and transforms
    # correctly. The reference is fixed once, so it does not drift between
    # sweeps.
    var_ref = float(np.mean(np.var(R, axis=0)))
    if not np.isfinite(var_ref) or var_ref <= 0.0:
        var_ref = 1.0
    d_floor_eff = d_floor * var_ref

    # Diagonal noise (start from residual variances, floored)
    D = np.maximum(np.var(R, axis=0), d_floor_eff)

    # ----------------------------
    # Traces & baseline
    # ----------------------------
    nll_trace = []
    nll_beta_trace = []
    accept_t_trace = []
    scale_used_trace = []

    # Starting NLL (explicit baseline before any sweep)
    nll_prev = float(nll_per_row(R, F, D))
    obj_prev = nll_prev + 0.5 * lam_F * float(np.sum(F**2))
    nll_trace.append(nll_prev)
    obj_trace = [obj_prev]

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

        # Per-sweep quantities are bound as defaults so each closure captures
        # this sweep's values rather than the loop variables (B023).
        def M_pre(v, Mpre_diag=Mpre_diag):
            return v / np.maximum(Mpre_diag, 1e-8)

        # Matrix-free normal operator H(B) = X^T Σ^{-1} X · b + lam_B b
        def A_mv(bvec, F=F, D=D, Dinv=Dinv, C_chol=C_chol):
            B_dir = unstack_B_vec(bvec, p_list)
            M = XB_from_Blist(Xs, B_dir)  # N x K
            S = apply_siginv_to_matrix(M, F, D, Dinv=Dinv, C_chol=C_chol)  # N x K
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
        base_nll = float(nll_per_row(R, F, D)) + 0.5 * lam_F * float(np.sum(F**2))
        nll_beta_trace.append(base_nll - 0.5 * lam_F * float(np.sum(F**2)))

        # If β worsened NLL, revert to previous B (ensures non-increase vs prior Σ)
        if base_nll > obj_prev + 1e-12:
            B = B_prev
            R = Y - XB_from_Blist(Xs, B)
            base_nll = obj_prev  # true baseline for this sweep

        # --- Gradient-based F update
        # Compute gradient of NLL w.r.t. F
        grad_F = grad_F_nll(R, F, D, Dinv, C_chol, lam_F)

        # Steepest descent direction.
        dF = -grad_F

        # Scale-calibrated initial step length.
        #
        # The NLL is equivariant under Y -> sY, (F, D) -> (sF, s^2 D) up to an
        # additive K log s, so grad_F scales like 1/s while F scales like s.
        # A hard-coded unit step is therefore off by a factor of s^2: on
        # small-magnitude data (e.g. returns expressed as decimals) it
        # overshoots by orders of magnitude, every backtracked candidate is
        # rejected, and F stays frozen at its initialization -- not a
        # stationary point (docs/source/formal_methods.md, Theorems 1-2).
        # Taking t0 = ||F|| / ||grad_F|| makes the trial step transform the
        # same way F does, so the whole iteration is scale-equivariant.
        gnorm = float(np.linalg.norm(dF))
        fnorm = float(np.linalg.norm(F))
        t0 = (fnorm / gnorm if fnorm > 0.0 else 1.0 / gnorm) if gnorm > 0.0 else 0.0

        # D update: closed-form MLE given the trial F,
        # d_j = max(S_jj - (F F^T)_jj, d_floor)  with  S = R^T R / N.
        diag_S = np.sum(R**2, axis=0) / N

        def D_mle(F_try, diag_S=diag_S):
            return np.maximum(diag_S - np.sum(F_try**2, axis=1), d_floor_eff)

        # The objective the F-step actually descends. grad_F_nll adds lam_F * F
        # to the gradient, i.e. the gradient of (lam_F/2)||F||^2, so the search
        # direction belongs to this penalised objective and not to the bare NLL.
        # Testing acceptance on the bare NLL instead meant the iteration
        # descended one function while being judged on another, and could stop
        # at a point stationary for neither.
        def penalty(F_try):
            return 0.5 * lam_F * float(np.sum(F_try**2))

        # Guarded scale correction helper (applied to a candidate F,D)
        def try_with_scale(F_try, D_try, R=R):
            """Return (F_out, D_out, obj_out, scale_used)."""
            nll0 = float(nll_per_row(R, F_try, D_try)) + penalty(F_try)
            if not scale_correct:
                return F_try, D_try, nll0, 1.0

            # MLE scalar c* for Σ = c * (F_try F_try^T + diag D_try)
            Dinv_s, C_chol_s = woodbury_chol(F_try, D_try)
            RSinv_s = apply_siginv_to_matrix(
                R, F_try, D_try, Dinv=Dinv_s, C_chol=C_chol_s
            )
            quad_over_N = float(np.sum(RSinv_s * R)) / N
            c_star = max(quad_over_N / K, scale_floor)

            sqrt_c = np.sqrt(c_star)
            F_sc = F_try * sqrt_c
            D_sc = D_try * c_star
            nll_sc = float(nll_per_row(R, F_sc, D_sc)) + penalty(F_sc)

            if nll_sc <= nll0 + 1e-12:
                return F_sc, D_sc, nll_sc, c_star
            return F_try, D_try, nll0, 1.0

        # --- Backtracking line search on (F, D)
        F_old, D_old = F, D

        best_nll = base_nll
        best_F, best_D = F_old, D_old
        accepted_t = 0.0
        used_scale = 1.0

        # Halve the step until it improves the NLL. Backtracking must be
        # allowed to run to convergence: a ladder truncated at a fixed t_min
        # rejects every candidate whenever the initial step overshoots, which
        # silently freezes the F-step instead of finding a descent step.
        #
        # Two ladders, tried in order. The first is the closed-form MLE D at
        # each trial F, which is the good step whenever it is admissible.
        #
        # It is not, on its own, a continuation of the incumbent: the guarded
        # scale correction leaves D off the D_mle(F) manifold, so this ladder's
        # t -> 0 limit is (F, D_mle(F)) rather than (F, D). Once the correction
        # has moved D far enough that this limit is worse than the incumbent,
        # every candidate is rejected however small the step, and F is frozen
        # for the rest of the run -- the NLL then sits nats/row above what the
        # same objective reaches from the same starting point, while the trace
        # looks converged and further sweeps are exact no-ops.
        #
        # The second ladder holds D at the incumbent, so (F + t*dF, D_old) does
        # tend to the incumbent as t -> 0 and a small enough step along a
        # descent direction must improve. It runs only when the first ladder
        # accepted nothing, which keeps the good case bit-for-bit unchanged
        # rather than letting a weak large-t candidate preempt the halving
        # that would have found a better one.
        def D_keep(_F_try, D_old=D_old):
            return D_old

        for D_of in (D_mle, D_keep):
            t = t0
            for _ in range(_MAX_BACKTRACK):
                if t == 0.0:
                    break
                F_try = F_old + t * dF
                F_acc, D_acc, nll_acc, sc_used = try_with_scale(F_try, D_of(F_try))
                # Accept only if we beat the per-sweep baseline
                if nll_acc < best_nll - 1e-12:
                    best_nll = nll_acc
                    best_F, best_D = F_acc, D_acc
                    accepted_t = t
                    used_scale = sc_used
                    break  # first improving step is fine (monotone)
                t *= 0.5
            if accepted_t > 0.0:
                break

        # Accept (or keep old F,D if no improvement)
        F, D = best_F, best_D
        obj_curr = best_nll
        # The trace reports the likelihood itself, so nll_trace[-1] always equals
        # nll_per_row at the returned parameters; obj_trace reports the penalised
        # objective that the line search actually descends. They coincide when
        # lam_F is 0, and only the latter is guaranteed non-increasing.
        nll_curr = float(nll_per_row(R, F, D))
        accept_t_trace.append(accepted_t)
        scale_used_trace.append(float(used_scale))

        # Append post-Σ NLL, and the objective that is non-increasing by construction
        nll_trace.append(nll_curr)
        obj_trace.append(obj_curr)

        # Convergence: stop if the relative improvement in the objective is tiny
        rel_impr = (obj_prev - obj_curr) / max(1.0, abs(obj_prev))
        nll_prev = nll_curr
        obj_prev = obj_curr
        if rel_impr < rel_tol:
            break

    # Final beta refresh at the final Sigma.
    #
    # The sweep order is beta-then-Sigma, so on exit beta is the GLS solution at
    # the Sigma of the *previous* sweep, not the Sigma being returned. Callers
    # are entitled to assume the two agree: compute_XtSigmaInvX gives
    # (X' Sigma^-1 X)^-1 as the variance of beta, which is the right variance
    # only when beta is the GLS estimator at that Sigma. One more CG solve at
    # the final (F, D) makes the returned pair mutually consistent. Minimising
    # over beta at fixed Sigma cannot raise the NLL, and the same revert guard
    # as in the sweep keeps the trace non-increasing if CG lands short.
    if sweeps > 0:
        Dinv, C_chol = woodbury_chol(F, D)
        diag_sinv = siginv_diag(F, Dinv, C_chol)
        Mpre_diag = (
            np.concatenate(
                [diag_sinv[j] * np.sum(X * X, axis=0) for j, X in enumerate(Xs)],
                axis=0,
            )
            + lam_B
        )

        def M_pre_final(v, Mpre_diag=Mpre_diag):
            return v / np.maximum(Mpre_diag, 1e-8)

        def A_mv_final(bvec, F=F, D=D, Dinv=Dinv, C_chol=C_chol):
            M = XB_from_Blist(Xs, unstack_B_vec(bvec, p_list))
            S = apply_siginv_to_matrix(M, F, D, Dinv=Dinv, C_chol=C_chol)
            out = np.concatenate(
                [Xs[j].T @ S[:, [j]] for j in range(K)], axis=0
            ).ravel()
            return out + lam_B * bvec

        S_y = apply_siginv_to_matrix(Y, F, D, Dinv=Dinv, C_chol=C_chol)
        rhs = np.concatenate([Xs[j].T @ S_y[:, [j]] for j in range(K)], axis=0).ravel()
        bvec, cg_info = cg_solve(
            A_mv_final,
            rhs,
            x0=stack_B_list(B),
            maxit=cg_maxit,
            tol=cg_tol,
            M_pre=M_pre_final,
        )
        B_ref = unstack_B_vec(bvec, p_list)
        nll_ref = float(nll_per_row(Y - XB_from_Blist(Xs, B_ref), F, D))
        if nll_ref <= nll_trace[-1] + 1e-12:
            B = B_ref
            nll_trace[-1] = nll_ref
            obj_trace[-1] = nll_ref + 0.5 * lam_F * float(np.sum(F**2))

    # Memory estimate: F (Kxk) + D (K) + U (Nxk) doubles
    mem_mb_est = (K * F.shape[1] + K + N * F.shape[1]) * 8 / 1e6

    info = {
        "p_list": p_list,
        "cg": cg_info,
        "nll_trace": nll_trace,  # post-Σ
        "nll_sigma_trace": nll_trace,  # alias for clarity
        "obj_trace": obj_trace,  # penalised objective; the monotone one
        "nll_beta_trace": nll_beta_trace,  # post-β (per-sweep baseline)
        "accept_t": accept_t_trace,  # accepted t (0.0 means kept previous F,D)
        "scale_used": scale_used_trace,  # accepted c* (1.0 means no scale applied)
    }
    return B, F, D, mem_mb_est, info
