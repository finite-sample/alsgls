import numpy as np

from .diagnostics import coverage_and_mahalanobis, whitening_matrix, top_eigvecs_sym


def apply_scale(F, D, s):
    s = float(s)
    return np.sqrt(s) * F, s * D


def calibrate_s_bisection(Xs, Y, B_list, F, D, target=0.03, tol=0.005,
                          s_lo=0.7, s_hi=2.0, maxit=25):
    def tail95_at(s):
        F_s, D_s = apply_scale(F, D, s)
        diag = coverage_and_mahalanobis(Y, Xs, B_list, F_s, D_s)
        return diag["m2_frac_gt_95"], diag
    f_lo, _ = tail95_at(s_lo)
    f_hi, _ = tail95_at(s_hi)
    if (f_lo - target) * (f_hi - target) > 0:
        grid = np.linspace(min(0.5, s_lo), max(2.2, s_hi), 21)
        vals = [tail95_at(s)[0] for s in grid]
        return float(grid[int(np.argmin([abs(v - target) for v in vals]))])
    lo, hi = s_lo, s_hi
    for _ in range(maxit):
        mid = 0.5 * (lo + hi)
        f_mid, _ = tail95_at(mid)
        if abs(f_mid - target) <= tol:
            return float(mid)
        if (f_mid - target) * (f_lo - target) <= 0:
            hi = mid
        else:
            lo = mid
            f_lo = f_mid
    return float(0.5 * (lo + hi))


def inflate_corr_fixed_diag(F, D, gamma, d_floor=1e-6):
    FF_diag = np.sum(F ** 2, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        gamma_max = np.min(np.where(FF_diag > 0, (D - d_floor) / FF_diag, np.inf))
    gamma_eff = float(max(0.0, min(gamma, gamma_max if np.isfinite(gamma_max) else gamma)))
    F_g = np.sqrt(1.0 + gamma_eff) * F
    D_g = np.maximum(D - gamma_eff * FF_diag, d_floor)
    return F_g, D_g, gamma_eff


def apply_alpha(D, alpha):
    return float(alpha) * D


def _score(diag, cov_target=0.975, tail_target=0.03):
    cov_err = abs(diag["z_coverage_overall"]["cov@95%"]["overall"] - cov_target)
    tail_err = abs(diag["m2_frac_gt_95"] - tail_target) + 0.5 * abs(
        diag["m2_frac_gt_99"] - 0.01
    )
    whiten = diag["whiten_frob_dev"]
    w_white = 0.6 + 0.9 * float((cov_err < 0.01) and (tail_err < 0.02))
    return 2.0 * cov_err + 2.0 * tail_err + w_white * whiten


def _gamma_line_search(Xs, Y, B_list, F, D_a, tail_target, cov_target,
                       d_floor=1e-6, coarse_pts=11, refine_steps=9):
    FF_diag = np.sum(F ** 2, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        gamma_max = np.min(np.where(FF_diag > 0, (D_a - d_floor) / FF_diag, np.inf))
    if (not np.isfinite(gamma_max)) or (gamma_max <= 0):
        return 0.0, None

    g_hi = 0.95 * float(gamma_max)
    grid = np.linspace(0.0, g_hi, coarse_pts)
    best = None
    for g in grid:
        F_g, D_g, _ = inflate_corr_fixed_diag(F, D_a, g, d_floor=d_floor)
        s_hat = calibrate_s_bisection(Xs, Y, B_list, F_g, D_g, target=tail_target)
        F_s, D_s = apply_scale(F_g, D_g, s_hat)
        diag = coverage_and_mahalanobis(Y, Xs, B_list, F_s, D_s)
        sc = _score(diag, cov_target, tail_target)
        if (best is None) or (sc < best[0]):
            best = (sc, g, s_hat, diag)
    _, g0, s0, _ = best

    phi = 0.5 * (3 - np.sqrt(5))
    a = max(0.0, g0 - 0.25 * g_hi)
    b = min(g0 + 0.25 * g_hi, g_hi)
    x1 = a + (1 - phi) * (b - a)
    x2 = a + phi * (b - a)

    def eval_g(g):
        F_g, D_g, _ = inflate_corr_fixed_diag(F, D_a, g, d_floor=d_floor)
        s_hat = calibrate_s_bisection(Xs, Y, B_list, F_g, D_g, target=tail_target)
        F_s, D_s = apply_scale(F_g, D_g, s_hat)
        diag = coverage_and_mahalanobis(Y, Xs, B_list, F_s, D_s)
        return _score(diag, cov_target, tail_target), s_hat, diag

    f1, s1, d1 = eval_g(x1)
    f2, s2, d2 = eval_g(x2)
    for _ in range(refine_steps):
        if f1 < f2:
            b, f2, s2, d2 = x2, f1, s1, d1
            x2 = x1
            x1 = a + (1 - phi) * (b - a)
            f1, s1, d1 = eval_g(x1)
        else:
            a, f1, s1, d1 = x1, f2, s2, d2
            x1 = x2
            x2 = a + phi * (b - a)
            f2, s2, d2 = eval_g(x2)
    return (float(x1), (f1, s1, d1)) if f1 < f2 else (float(x2), (f2, s2, d2))


def _eigen_boost_line_search(Xs, Y, B_list, F_g, D_g, D_alpha,
                             tail_target, cov_target, r=2, d_floor=1e-6,
                             coarse_pts=11, refine_steps=9):
    W = whitening_matrix(Y, Xs, B_list, F_g, D_g)
    vals, U = top_eigvecs_sym(W - np.eye(W.shape[0]), r=r)
    if vals[-1] <= 0:
        return 0.0, None

    base = np.sqrt(D_alpha)[:, None] * U
    rowpow = np.sum(base ** 2, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        tau_max = np.min(np.where(rowpow > 0, (D_g - d_floor) / rowpow, np.inf))
    if (not np.isfinite(tau_max)) or (tau_max <= 0):
        return 0.0, None

    t_hi = 0.95 * float(tau_max)
    grid = np.linspace(0.0, t_hi, coarse_pts)
    best = None
    for t in grid:
        F_aug = np.column_stack([F_g, np.sqrt(t) * base])
        D_new = np.maximum(D_g - t * rowpow, d_floor)
        s_hat = calibrate_s_bisection(Xs, Y, B_list, F_aug, D_new, target=tail_target)
        F_s, D_s = apply_scale(F_aug, D_new, s_hat)
        diag = coverage_and_mahalanobis(Y, Xs, B_list, F_s, D_s)
        sc = _score(diag, cov_target, tail_target)
        if (best is None) or (sc < best[0]):
            best = (sc, t, s_hat, diag)
    _, t0, s0, _ = best

    phi = 0.5 * (3 - np.sqrt(5))
    a = max(0.0, t0 - 0.25 * t_hi)
    b = min(t0 + 0.25 * t_hi, t_hi)
    x1 = a + (1 - phi) * (b - a)
    x2 = a + phi * (b - a)

    def eval_t(t):
        F_aug = np.column_stack([F_g, np.sqrt(t) * base])
        D_new = np.maximum(D_g - t * rowpow, d_floor)
        s_hat = calibrate_s_bisection(Xs, Y, B_list, F_aug, D_new, target=tail_target)
        F_s, D_s = apply_scale(F_aug, D_new, s_hat)
        diag = coverage_and_mahalanobis(Y, Xs, B_list, F_s, D_s)
        return _score(diag, cov_target, tail_target), s_hat, diag

    f1, s1, d1 = eval_t(x1)
    f2, s2, d2 = eval_t(x2)
    for _ in range(refine_steps):
        if f1 < f2:
            b, f2, s2, d2 = x2, f1, s1, d1
            x2 = x1
            x1 = a + (1 - phi) * (b - a)
            f1, s1, d1 = eval_t(x1)
        else:
            a, f1, s1, d1 = x1, f2, s2, d2
            x1 = x2
            x2 = a + phi * (b - a)
            f2, s2, d2 = eval_t(x2)
    return (float(x1), (f1, s1, d1)) if f1 < f2 else (float(x2), (f2, s2, d2))


def calibrate_alpha_gamma_s_cv3_conservative(
    Xs,
    Y,
    B_list,
    F,
    D,
    alpha_grid=None,
    Kfolds=3,
    val_frac=0.5,
    tail_target=0.03,
    cov_target=0.975,
    r_boost=2,
    rng_seed=123,
    d_floor=1e-6,
):
    rng = np.random.default_rng(rng_seed)
    N = Y.shape[0]
    if alpha_grid is None:
        alpha_grid = np.linspace(0.78, 1.10, 17)

    folds = []
    for _ in range(Kfolds):
        val_idx = rng.choice(N, size=int(val_frac * N), replace=False)
        trn_mask = np.ones(N, dtype=bool)
        trn_mask[val_idx] = False
        folds.append(
            (
                [X[trn_mask] for X in Xs],
                Y[trn_mask],
                [X[val_idx] for X in Xs],
                Y[val_idx],
            )
        )

    best = None
    for alpha in alpha_grid:
        fold_picks = []
        for (Xs_tr, Y_tr, Xs_va, Y_va) in folds:
            D_a = apply_alpha(D, alpha)
            g_star, gpack = _gamma_line_search(
                Xs_va,
                Y_va,
                B_list,
                F,
                D_a,
                tail_target=tail_target,
                cov_target=cov_target,
                d_floor=d_floor,
            )
            if gpack is None:
                F_g, D_g, _ = inflate_corr_fixed_diag(F, D_a, 0.0, d_floor=d_floor)
                s_g = calibrate_s_bisection(Xs_va, Y_va, B_list, F_g, D_g, target=tail_target)
                F_sg, D_sg = apply_scale(F_g, D_g, s_g)
                diag_g = coverage_and_mahalanobis(Y_va, Xs_va, B_list, F_sg, D_sg)
                score_g = _score(diag_g, cov_target, tail_target)
            else:
                score_g, s_g, diag_g = gpack
                F_g, D_g, _ = inflate_corr_fixed_diag(F, D_a, g_star, d_floor=d_floor)
                F_sg, D_sg = apply_scale(F_g, D_g, s_g)

            t_star, tpack = _eigen_boost_line_search(
                Xs_va,
                Y_va,
                B_list,
                F_g,
                D_g,
                D_a,
                tail_target=tail_target,
                cov_target=cov_target,
                r=r_boost,
                d_floor=d_floor,
            )
            if tpack is None:
                score_t = np.inf
            else:
                score_t, s_t, _ = tpack

            if score_t < score_g:
                fold_picks.append(("eig", g_star, t_star, s_t))
            else:
                fold_picks.append(("gam", g_star, 0.0, s_g))

        modes = [m for (m, _, _, _) in fold_picks]
        use_eig = modes.count("eig") >= modes.count("gam")
        gammas = np.array([g for (_, g, _, _) in fold_picks])
        taus = np.array([t for (_, _, t, _) in fold_picks])
        ss = np.array([s for (_, _, _, s) in fold_picks])
        q = 0.75
        gamma_q = float(np.quantile(gammas, q))
        tau_q = float(np.quantile(taus, q)) if use_eig else 0.0
        s_q = float(np.quantile(ss, q))

        D_a = apply_alpha(D, alpha)
        F_g, D_g, _ = inflate_corr_fixed_diag(F, D_a, gamma_q, d_floor=d_floor)
        if use_eig and tau_q > 0:
            W_tr = whitening_matrix(Y, Xs, B_list, F_g, D_g)
            _, U = top_eigvecs_sym(W_tr - np.eye(F_g.shape[0]), r=r_boost)
            base = np.sqrt(D_a)[:, None] * U
            rowpw = np.sum(base ** 2, axis=1)
            F_aug = np.column_stack([F_g, np.sqrt(tau_q) * base])
            D_new = np.maximum(D_g - tau_q * rowpw, d_floor)
        else:
            F_aug, D_new = F_g, D_g
        F_s, D_s = apply_scale(F_aug, D_new, s_q)
        diag_try = coverage_and_mahalanobis(Y, Xs, B_list, F_s, D_s)
        summary_score = _score(diag_try, cov_target, tail_target)
        if (best is None) or (summary_score < best[0]):
            best = (summary_score, alpha, gamma_q, tau_q, s_q, use_eig)

    _, alpha_star, gamma_star, tau_star, s_star, use_eig_star = best
    return (
        float(alpha_star),
        float(gamma_star),
        float(tau_star),
        float(s_star),
        bool(use_eig_star),
    )


def finalize_on_dataset(Xs, Y, B_list, F, D, alpha, gamma, tau, s,
                         use_eig=True, r_boost=2, d_floor=1e-6):
    D_a = apply_alpha(D, alpha)
    F_g, D_g, _ = inflate_corr_fixed_diag(F, D_a, gamma, d_floor=d_floor)
    if use_eig and (tau > 0):
        W = whitening_matrix(Y, Xs, B_list, F_g, D_g)
        _, U = top_eigvecs_sym(W - np.eye(F_g.shape[0]), r=r_boost)
        base = np.sqrt(D_a)[:, None] * U
        rowpw = np.sum(base ** 2, axis=1)
        F_g = np.column_stack([F_g, np.sqrt(tau) * base])
        D_g = np.maximum(D_g - tau * rowpw, d_floor)
    F_fin, D_fin = apply_scale(F_g, D_g, s)
    return F_fin, D_fin
