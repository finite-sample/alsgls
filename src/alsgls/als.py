"""Alternating-least-squares solver for low-rank-plus-diagonal GLS."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import minimize

from ._validation import _validate_convergence_params, _validate_gls_inputs
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

#: Iteration cap for the quasi-Newton Sigma step. It converges in about ten
#: iterations on every case measured; the cap exists so a pathological fit
#: cannot hang.
_SIGMA_MAX_ITER = 200

#: Relative decrease of the profile likelihood below which the Sigma step
#: stops. With a quasi-Newton step this is reached in a handful of iterations,
#: so it can be tight at no real cost.
_SIGMA_TOL = 1e-12


def _lawley_F(R: np.ndarray, D: np.ndarray, k: int) -> np.ndarray:
    """The maximising ``F`` at fixed ``D``, in closed form.

    For ``S = R^T R / N`` and the top ``k`` eigenpairs ``(Theta, P)`` of
    ``D^{-1/2} S D^{-1/2}``, ``F = D^{1/2} P (Theta - I)^{1/2}`` (Lawley; see
    Lawley and Maxwell 1971, and eq. 8 of Fukasaku et al., arXiv:2402.08181).
    Those eigenvectors are the top ``k`` right singular vectors of
    ``R D^{-1/2} / sqrt(N)`` and the eigenvalues its squared singular values, so
    the step is taken from the ``(N, K)`` residual matrix and no ``K x K``
    matrix is formed. This is the route ``sklearn.decomposition.FactorAnalysis``
    and R's ``factanal`` both take.

    Args:
        R: Residual matrix, ``(N, K)``.
        D: Diagonal variances, length ``K``.
        k: Factor rank.

    Returns:
        The loadings, ``(K, k)``. Eigenvalues below 1 give zero columns.
    """
    N, K = R.shape
    root_D = np.sqrt(D)
    Z = (R / root_D[None, :]) / np.sqrt(N)
    _, s, Vt = np.linalg.svd(Z, full_matrices=False)
    m = min(k, s.size)
    F = np.zeros((K, k))
    if m > 0:
        gain = np.sqrt(np.maximum(s[:m] ** 2 - 1.0, 0.0))
        F[:, :m] = (root_D[:, None] * Vt[:m].T) * gain[None, :]
    return F


def _sigma_step(
    R: np.ndarray,
    D: np.ndarray,
    k: int,
    d_floor_eff: float,
    *,
    max_iter: int = _SIGMA_MAX_ITER,
    tol: float = _SIGMA_TOL,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    """Fit ``Sigma = F F^T + diag(D)`` to residuals ``R`` by maximum likelihood.

    ``F`` has a closed form at fixed ``D`` (:func:`_lawley_F`), so the problem
    is concentrated to ``D`` alone and solved by L-BFGS-B on ``log D``, boxed
    between the floor and each equation's total residual variance. This is
    what R's ``factanal`` does, and for the reason it gives: alternating the
    two closed forms is a fixed-point iteration that crawls when a diagonal
    variance is small. Measured on bootstrap replicates at ``n = 20`` the
    alternation's per-fit cost had a median of 24 ms but a mean of 142 ms and
    a maximum of 598 ms (11,619 inner iterations); the quasi-Newton step's
    mean is 1.7 ms and its maximum 7 ms, and on the replicate where the
    alternation hit its cap it also found a likelihood 4.6e-4 nats/row better.

    The gradient is the envelope theorem: at the profile, ``F`` is optimal, so
    ``d nll / d D_j`` is the partial at fixed ``F``,
    ``0.5 [(Sigma^{-1})_jj - (Sigma^{-1} S Sigma^{-1})_jj]``, both of which the
    Woodbury kernels give without forming ``K x K``. Verified against finite
    differences to 3.7e-9.

    Args:
        R: Residual matrix, ``(N, K)``.
        D: Starting diagonal variances, length ``K``.
        k: Factor rank.
        d_floor_eff: Smallest permitted variance, already scaled to the data.
        max_iter: Iteration cap for the optimiser.
        tol: Relative decrease of the objective below which it stops.

    Returns:
        ``(F, D, nll, iters)``: the fit, its negative log-likelihood per row,
        and the number of optimiser iterations.
    """
    N, K = R.shape
    diag_S = np.sum(R**2, axis=0) / N

    # Solve on residuals standardised to unit variance, as factanal does on a
    # correlation matrix, and map back. diag(1/sd) Sigma diag(1/sd) is
    # (F/sd)(F/sd)^T + diag(D/sd^2), so the structure survives, and the
    # standardised problem is the same problem at every scale of Y. That is
    # what makes the fit scale-equivariant: the optimiser's stopping rules are
    # not scale-free (L-BFGS-B's ftol is relative to max(|f|, 1), and the NLL
    # shifts by K log s under Y -> sY), and on a flat likelihood a different
    # stopping point is a different local optimum.
    sd = np.sqrt(np.maximum(diag_S, d_floor_eff))
    Rs = R / sd[None, :]
    floor_s = d_floor_eff / float(np.min(sd) ** 2)
    log_lo = np.log(floor_s)
    # A unique variance cannot exceed the equation's total variance (1 after
    # standardising); the tiny slack keeps the closed form off an exact
    # zero-loading corner.
    log_hi = np.full(K, np.log1p(1e-9))
    z0 = np.clip(np.log(np.asarray(D, dtype=float) / sd**2), log_lo, log_hi)

    def objective(z: np.ndarray) -> tuple[float, np.ndarray]:
        D_z = np.exp(z)
        F_z = _lawley_F(Rs, D_z, k)
        Dinv, C_chol = woodbury_chol(F_z, D_z)
        RSinv = apply_siginv_to_matrix(Rs, F_z, D_z, Dinv=Dinv, C_chol=C_chol)
        nll = 0.5 * (
            float(np.sum(RSinv * Rs)) / N
            + float(np.sum(np.log(D_z)))
            + 2.0 * float(np.sum(np.log(np.diag(C_chol))))
        )
        grad_D = 0.5 * (siginv_diag(F_z, Dinv, C_chol) - np.sum(RSinv**2, axis=0) / N)
        return nll, grad_D * D_z  # chain rule for log D

    res = minimize(
        objective,
        z0,
        jac=True,
        method="L-BFGS-B",
        bounds=list(zip(np.full(K, log_lo), log_hi, strict=True)),
        options={"maxiter": max_iter, "ftol": tol, "gtol": 1e-10},
    )
    D_out = np.exp(res.x) * sd**2
    F_out = _lawley_F(R, D_out, k)
    return F_out, D_out, float(nll_per_row(R, F_out, D_out)), int(res.nit)


def als_gls(
    Xs: list[np.ndarray],
    Y: np.ndarray,
    k: int,
    lam_B: float = 1e-3,
    sweeps: int = 8,
    d_floor: float = 1e-8,
    cg_maxit: int = 800,
    cg_tol: float = 3e-7,
    *,
    rel_tol: float = 1e-6,
    init_D: np.ndarray | None = None,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, float, dict[str, Any]]:
    """Alternating GLS with a low-rank-plus-diagonal covariance.

    Alternates two exact steps until the likelihood stops falling: ``beta`` by
    matrix-free conjugate gradients at the current ``Sigma``, then ``Sigma`` by
    the closed-form factor-analysis update in :func:`_sigma_step`. Woodbury is
    used throughout and no ``K x K`` matrix is formed.

    Args:
        Xs: One design matrix per equation, each ``(N, p_j)``. The equations may
            have different numbers of regressors.
        Y: Responses, ``(N, K)``, one column per equation.
        k: Rank of the latent factor block. The covariance is modelled as
            ``F F^T + D`` with ``F`` of shape ``(K, k)``, so ``k`` controls how
            much cross-equation dependence is shared; ``k << K`` is the point.
        lam_B: Ridge penalty on the coefficients ``beta``, expressed relative to
            the mean initial residual variance rather than in absolute units. An
            absolute penalty would make the fit depend on the units of ``Y``:
            ``X' Sigma^-1 X`` scales as ``1/s^2`` under ``Y -> sY`` while a fixed
            ``lam_B`` does not, so at ``lam_B = 1e-3`` scaling ``Y`` by ``1e4``
            moved every coefficient by 100%. Dividing by the variance scale makes
            the penalty transform correctly and the whole fit equivariant.
        sweeps: Maximum number of alternating passes over ``beta`` and ``Sigma``.
        d_floor: Floor on each diagonal variance, as a fraction of the mean
            initial residual variance rather than an absolute variance, for the
            same reason ``lam_B`` is relative: the true ``D`` scales as ``s^2``
            under ``Y -> sY``, so an absolute floor binds on every entry once
            ``Y`` is small enough.
        cg_maxit: Iteration cap for the conjugate-gradient solve in the beta step.
        cg_tol: Relative residual tolerance for that solve.
        rel_tol: Relative decrease in the negative log-likelihood below which the
            sweeps stop early.
        init_D: Starting diagonal variances for the first Sigma step, length
            ``K``. Defaults to the residual variances of the initial fit. A
            bootstrap replicate passes the parent fit's ``D`` here: the
            replicate's answer is close to the parent's, so starting from it
            skips most of the inner alternation. Only ``D`` is needed, because
            the Sigma step recomputes ``F`` from ``D`` in closed form.

    Returns:
        B_list, F, D, mem_MB_est, info: ``info`` includes ``p_list``, ``cg``
        (the final beta solve), ``nll_trace`` (per sweep, non-increasing, and
        equal to ``nll_per_row`` at the returned parameters), ``nll_beta_trace``
        (post-beta, per sweep), ``sigma_iters`` (inner alternation counts), and
        ``var_ref``/``lam_B_eff`` (the variance scale the relative penalties were
        resolved against, and the resulting absolute ridge).

    Raises:
        ValueError: If an argument is outside its domain, if ``Xs`` or ``Y``
            holds a non-finite entry, or if their shapes disagree.
        np.linalg.LinAlgError: If a Cholesky factorisation of the Woodbury
            core fails, which means the current Sigma is not positive definite.
    """
    # ----------------------------
    # Input validation
    # ----------------------------
    Xs, Y, k, lam_B = _validate_gls_inputs(Xs, Y, k, lam_B=lam_B)
    _validate_convergence_params(
        sweeps=sweeps, rel_tol=rel_tol, cg_maxit=cg_maxit, cg_tol=cg_tol
    )
    # d_floor <= 0 is not a weaker floor, it is a broken one: D can then reach
    # zero or go negative, while woodbury_chol and nll_per_row clip D at 1e-12
    # internally. The returned (F, D) would describe a different -- and not
    # positive definite -- Sigma from the one every reported number was computed
    # under, and Sigma_jj = ||F_j||^2 + D_j could come out negative in
    # compute_prediction_variance.
    if (
        not isinstance(d_floor, (int, float))
        or isinstance(d_floor, bool)
        or not np.isfinite(d_floor)
        or d_floor <= 0
    ):
        raise ValueError(
            f"d_floor must be a positive finite number, got {d_floor}. "
            "Try d_floor=1e-8 (a fraction of the residual variance)."
        )
    d_floor = float(d_floor)
    N, K = Y.shape

    p_list = [X.shape[1] for X in Xs]

    # ----------------------------
    # Initialization
    # ----------------------------
    # The variance scale that d_floor and lam_B are expressed relative to. Fixed
    # once from the OLS residuals so it cannot drift between sweeps.
    ols_R = Y - XB_from_Blist(
        Xs,
        [np.linalg.lstsq(X, Y[:, [j]], rcond=None)[0] for j, X in enumerate(Xs)],
    )
    var_ref = float(np.mean(np.var(ols_R, axis=0)))
    if not np.isfinite(var_ref) or var_ref <= 0.0:
        var_ref = 1.0
    d_floor_eff = d_floor * var_ref
    lam_B_eff = lam_B / var_ref

    # Per-equation ridge/OLS for B.
    #
    # This solve uses the nominal lam_B, not lam_B_eff. An OLS ridge objective
    # ||Y - XB||^2 + lam ||B||^2 scales uniformly under Y -> sY, so B -> sB for
    # any fixed lam; a GLS ridge (Y - XB)' Sigma^-1 (Y - XB) + lam ||B||^2 has a
    # dimensionless first term and needs lam -> lam / s^2. The two penalties
    # have the same relative strength when lam_ols = lam_gls * var_ref, and
    # lam_B_eff * var_ref is lam_B. Using lam_B_eff here made the starting
    # point scale-dependent, which the previous fixed-point Sigma step was too
    # forgiving to expose and the quasi-Newton one is not.
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

    R = Y - XB_from_Blist(Xs, B)

    # ----------------------------
    # Traces & baseline
    # ----------------------------
    if init_D is None:
        D0 = np.maximum(np.var(R, axis=0), d_floor_eff)
    else:
        D0 = np.asarray(init_D, dtype=float)
        if D0.shape != (K,):
            raise ValueError(f"init_D must have shape ({K},), got {D0.shape}")
        if not np.isfinite(D0).all() or (D0 <= 0).any():
            raise ValueError("init_D must be finite and strictly positive")
        D0 = np.maximum(D0, d_floor_eff)
    F, D, nll_prev, n_inner = _sigma_step(R, D0, k, d_floor_eff)
    nll_trace = [nll_prev]
    nll_beta_trace: list[float] = []
    sigma_iters: list[int] = [n_inner]

    # ----------------------------
    # Main loop
    # ----------------------------
    cg_info = None

    for _ in range(sweeps):
        # Cache Woodbury pieces once per sweep
        Dinv, C_chol = woodbury_chol(F, D)

        # diag(Σ^{-1}) for preconditioning: Σ^{-1} = D^{-1} - D^{-1}F C^{-1} F^T D^{-1}
        diag_sinv = siginv_diag(F, Dinv, C_chol)  # (K,)
        block_diags = [diag_sinv[j] * np.sum(X * X, axis=0) for j, X in enumerate(Xs)]
        Mpre_diag = np.concatenate(block_diags, axis=0) + lam_B_eff

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
            return out + lam_B_eff * bvec

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

        R = Y - XB_from_Blist(Xs, B)
        base_nll = float(nll_per_row(R, F, D))
        nll_beta_trace.append(base_nll)

        # If β worsened NLL, revert to previous B (ensures non-increase vs prior Σ)
        if base_nll > nll_prev + 1e-12:
            B = B_prev
            R = Y - XB_from_Blist(Xs, B)

        # --- Σ-step: exact, warm-started from the current D
        F_try, D_try, nll_try, n_inner = _sigma_step(R, D, k, d_floor_eff)
        if nll_try <= nll_prev + 1e-12:
            F, D = F_try, D_try
            nll_curr = nll_try
        else:
            # _sigma_step already returns its own best, so this can only fire
            # when the beta step moved the residuals somewhere the alternation
            # cannot improve on. Keeping the incumbent keeps the trace monotone.
            nll_curr = float(nll_per_row(R, F, D))
        sigma_iters.append(n_inner)

        nll_trace.append(nll_curr)

        rel_impr = (nll_prev - nll_curr) / max(1.0, abs(nll_prev))
        nll_prev = nll_curr
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
            + lam_B_eff
        )

        def M_pre_final(v, Mpre_diag=Mpre_diag):
            return v / np.maximum(Mpre_diag, 1e-8)

        def A_mv_final(bvec, F=F, D=D, Dinv=Dinv, C_chol=C_chol):
            M = XB_from_Blist(Xs, unstack_B_vec(bvec, p_list))
            S = apply_siginv_to_matrix(M, F, D, Dinv=Dinv, C_chol=C_chol)
            out = np.concatenate(
                [Xs[j].T @ S[:, [j]] for j in range(K)], axis=0
            ).ravel()
            return out + lam_B_eff * bvec

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

    # Memory estimate: F (Kxk) + D (K) + U (Nxk) doubles
    mem_mb_est = (K * F.shape[1] + K + N * F.shape[1]) * 8 / 1e6

    info = {
        "p_list": p_list,
        "cg": cg_info,
        "nll_trace": nll_trace,
        "nll_beta_trace": nll_beta_trace,
        "sigma_iters": sigma_iters,
        # The variance scale d_floor and lam_B are expressed relative to.
        # compute_XtSigmaInvX needs it to charge the same effective penalty the
        # fit used, or the reported variance would not match the estimator.
        "var_ref": var_ref,
        "lam_B_eff": lam_B_eff,
    }
    return B, F, D, mem_mb_est, info
