# lowrank_gls/core.py
# Core implementation: low-rank+diag GLS/SUR with ALS (matrix-free) and EM (dense),
# diagnostics (coverage, Mahalanobis, whitening), and a conservative CV calibrator
# (alpha–gamma–tau–s with optional r-direction eigen-boost). Includes a diagonal
# preconditioner for CG to tighten ALS–EM likelihood gaps and speed convergence.

import time
import numpy as np
from numpy.linalg import solve, LinAlgError, eigh
from scipy.sparse.linalg import LinearOperator, cg
from scipy.stats import norm, chi2

# =========================
# Numerics & Woodbury utils
# =========================

def safe_inv(A, lam=1e-6):
    I = np.eye(A.shape[0])
    ridge = max(lam * np.linalg.norm(A, 'fro') / A.shape[0], 1e-8)
    try:
        return solve(A + ridge * I, I)
    except LinAlgError:
        return np.linalg.pinv(A + ridge * I)

def safe_solve(A, b, lam=1e-6):
    I = np.eye(A.shape[0])
    ridge = max(lam * np.linalg.norm(A, 'fro') / A.shape[0], 1e-8)
    try:
        return solve(A + ridge * I, b)
    except LinAlgError:
        return np.linalg.lstsq(A + ridge * I, b, rcond=None)[0]

def wpca_init(R, k, D):
    # Weighted PCA on residuals with weights 1/sqrt(D)
    W = R / np.sqrt(D)[None, :]
    _, s, Vt = np.linalg.svd(W, full_matrices=False)
    r = max(1, min(k, (s > 1e-8).sum()))
    Fw = Vt.T[:, :r] * np.sqrt(s[:r])
    F  = Fw * np.sqrt(D)[:, None]
    return np.pad(F, ((0, 0), (0, k - r)))

def cf_matrix(F, Dinv):
    return np.eye(F.shape[1]) + F.T @ (Dinv[:, None] * F)

def siginv_apply_to_matrix(M, F, D, Cf_inv=None):
    # Compute M @ Σ^{-1} using Woodbury; M is N×K.
    Dinv = 1.0 / D
    if Cf_inv is None:
        Cf_inv = safe_inv(cf_matrix(F, Dinv))
    Vd  = M * Dinv[None, :]
    FD  = Dinv[:, None] * F
    return Vd - ((Vd @ F) @ Cf_inv) @ FD.T

# =========================
# Pack/unpack for β
# =========================

def pack_list_to_vec(B_list):
    return np.concatenate([b.ravel() for b in B_list], axis=0)

def unpack_vec_to_list(v, p_list):
    out, i = [], 0
    for p in p_list:
        out.append(v[i:i+p].reshape(p, 1))
        i += p
    return out

def predict_Y(Xs, B_list):
    return np.column_stack([Xs[j] @ B_list[j] for j in range(len(Xs))])

def mse(Y, Yhat):
    return float(np.mean((Y - Yhat)**2))

# =========================
# Likelihood (reporting)
# =========================

def _safe_logdet_psd(A):
    sign, logdet = np.linalg.slogdet(A)
    if sign > 0 and np.isfinite(logdet):
        return logdet
    # Fallback: perturb slightly to ensure PD
    eps = 1e-10
    return np.linalg.slogdet(A + eps*np.eye(A.shape[0]))[1]

def penalized_nll(Y, Xs, B_list, F, D, lam_F=0.0, lam_B=0.0):
    N, K = Y.shape
    XB = predict_Y(Xs, B_list)
    R  = Y - XB
    Dinv = 1.0 / D
    Cf = cf_matrix(F, Dinv); Cf_i = safe_inv(Cf)
    term1 = np.sum(R * (R * Dinv[None, :]))
    M = (R * Dinv[None, :]) @ F
    term2 = np.sum(M * (M @ Cf_i))
    quad = term1 - term2
    logdet = np.sum(np.log(D)) + _safe_logdet_psd(Cf)
    pen = lam_F*np.sum(F**2) + lam_B*sum(np.sum(b**2) for b in B_list)
    return 0.5*(N*logdet + quad) + pen

def test_nll(Y, Xs, B_list, F, D):
    N, K = Y.shape
    XB = predict_Y(Xs, B_list)
    R  = Y - XB
    Dinv = 1.0 / D
    Cf = cf_matrix(F, Dinv); Cf_i = safe_inv(Cf)
    term1 = np.sum(R * (R * Dinv[None, :]))
    M = (R * Dinv[None, :]) @ F
    term2 = np.sum(M * (M @ Cf_i))
    quad = term1 - term2
    logdet = np.sum(np.log(D)) + _safe_logdet_psd(Cf)
    return 0.5*(N*logdet + quad)/N

# =========================
# Preconditioner for CG (diag/Jacobi)
# =========================

def build_diag_precond(Xs, F, D, Cf_i, lam_B):
    K = len(Xs)
    p_list = [X.shape[1] for X in Xs]
    Dinv = 1.0 / D
    # diag(F Cf_i F^T)
    diag_term = np.sum(F * (F @ Cf_i), axis=1)
    sigma_inv_diag = Dinv - (Dinv**2) * diag_term
    diag_blocks = []
    for j in range(K):
        Hjj = Xs[j].T @ Xs[j]
        d = sigma_inv_diag[j] * np.diag(Hjj) + lam_B
        diag_blocks.append(d)
    d_all = np.concatenate(diag_blocks)
    d_all = np.maximum(d_all, 1e-12)
    return LinearOperator((d_all.size, d_all.size), matvec=lambda v: v / d_all)

# =========================
# ALS solver (matrix-free β with optional preconditioning)
# =========================

def als_gls(Xs, Y, k, lam_F=1e-3, lam_B=1e-3, sweeps=12, tol=1e-5, eps=1e-6,
            d_floor=1e-6, use_cg_beta=True, cg_maxit=800, cg_tol=3e-7,
            use_diag_precond=True):
    t0 = time.time()
    N, K = Y.shape
    p_list = [X.shape[1] for X in Xs]
    P = sum(p_list)

    # init
    B_list = [safe_solve(Xs[j].T @ Xs[j], Xs[j].T @ Y[:, [j]]) for j in range(K)]
    def XB(Bs): return predict_Y(Xs, Bs)
    R = Y - XB(B_list)
    D = np.var(R, axis=0) + eps
    F = wpca_init(R, k, D)

    prev = None  # <-- guard to avoid inf/inf on first iteration

    for _ in range(sweeps):
        Dinv = 1.0 / D
        Cf   = cf_matrix(F, Dinv)
        Cf_i = safe_inv(Cf)

        # RHS
        YSig = siginv_apply_to_matrix(Y, F, D, Cf_i)
        rhs_blocks = [Xs[j].T @ YSig[:, [j]] for j in range(K)]
        rhs = pack_list_to_vec(rhs_blocks)

        # matrix-free A·vec
        def mv(vec):
            Bs = unpack_vec_to_list(vec, p_list)
            V  = predict_Y(Xs, Bs)
            T  = siginv_apply_to_matrix(V, F, D, Cf_i)
            out_blocks = [Xs[j].T @ T[:, [j]] for j in range(K)]
            out = pack_list_to_vec(out_blocks)
            return out + lam_B * vec

        if use_cg_beta:
            Aop = LinearOperator((P, P), matvec=mv)
            M = build_diag_precond(Xs, F, D, Cf_i, lam_B) if use_diag_precond else None
            sol, info = cg(Aop, rhs, tol=cg_tol, maxiter=cg_maxit, M=M)
            if info != 0:
                # Fallback: assemble and solve
                Sigma_inv = np.diag(Dinv) - (Dinv[:, None] * (F @ Cf_i @ F.T)) * Dinv[None, :]
                A = np.zeros((P, P))
                row_off = 0
                for j in range(K):
                    Sj = Sigma_inv[:, j]
                    col_off = 0
                    Xtj = Xs[j].T
                    for l in range(K):
                        Gjl = Xtj @ Xs[l]
                        A[row_off:row_off+p_list[j], col_off:col_off+p_list[l]] = Sj[l] * Gjl
                        col_off += p_list[l]
                    row_off += p_list[j]
                sol = safe_solve(A + lam_B*np.eye(P), rhs)
            B_list = unpack_vec_to_list(sol, p_list)
        else:
            Sigma_inv = np.diag(Dinv) - (Dinv[:, None] * (F @ Cf_i @ F.T)) * Dinv[None, :]
            A = np.zeros((P, P))
            row_off = 0
            for j in range(K):
                Sj = Sigma_inv[:, j]
                col_off = 0
                Xtj = Xs[j].T
                for l in range(K):
                    Gjl = Xtj @ Xs[l]
                    A[row_off:row_off+p_list[j], col_off:col_off+p_list[l]] = Sj[l] * Gjl
                    col_off += p_list[l]
                row_off += p_list[j]
            sol = safe_solve(A + lam_B*np.eye(P), rhs)
            B_list = unpack_vec_to_list(sol, p_list)

        # update F, D
        R = Y - XB(B_list)
        FtF = F.T @ F
        U   = R @ F @ safe_inv(FtF + lam_F*np.eye(k))
        UtU = U.T @ U
        F   = R.T @ U @ safe_inv(UtU + lam_F*np.eye(k))
        D   = np.maximum(np.mean((R - U @ F.T)**2, axis=0) + eps, d_floor)

        # objective + guarded convergence
        obj = penalized_nll(Y, Xs, B_list, F, D, lam_F=lam_F, lam_B=lam_B)
        if not np.isfinite(obj):
            break
        if prev is not None:
            denom = max(1.0, abs(prev))
            rel_impr = (prev - obj) / denom
            # tolerate tiny non-monotone wiggles from inexact CG
            if rel_impr < 0 and abs(prev - obj) / denom < tol:
                break
            if rel_impr < tol:
                break
        prev = obj

    mem_mb = ((K*k) + K) * 8 / 1e6  # ~F and D
    return B_list, F, D, mem_mb, time.time() - t0

# =========================
# Dense EM solver (β via full A), with optional warm-start
# =========================

def em_gls(Xs, Y, k, lam_F=1e-3, lam_B=1e-3, iters=45, tol=1e-5, eps=1e-6, d_floor=1e-6,
           B_init=None, F_init=None, D_init=None):
    t0 = time.time()
    N, K = Y.shape
    p_list = [X.shape[1] for X in Xs]
    P = sum(p_list)

    # init (allow warm-start)
    if B_init is None:
        B_list = [safe_solve(Xs[j].T @ Xs[j], Xs[j].T @ Y[:, [j]]) for j in range(K)]
    else:
        B_list = [b.copy() for b in B_init]
    def XB(Bs): return predict_Y(Xs, Bs)
    R = Y - XB(B_list)
    if D_init is None:
        D = np.var(R, axis=0) + eps
    else:
        D = D_init.copy()
    if F_init is None:
        F = wpca_init(R, k, D)
    else:
        F = F_init.copy()

    prev = None  # <-- guard to avoid inf/inf on first iteration

    for _ in range(iters):
        # E-step
        Dinv = 1.0 / D
        Cf   = cf_matrix(F, Dinv)
        Cf_i = safe_inv(Cf)
        EZ   = (R * Dinv[None, :]) @ F @ Cf_i
        EZZ  = EZ.T @ EZ + N * Cf_i

        # M-step for F, D
        F    = R.T @ EZ @ safe_inv(EZZ + lam_F*np.eye(k))
        Rmd  = R - EZ @ F.T
        D    = np.maximum(np.mean(Rmd**2, axis=0) + eps, d_floor)

        # Σ^{-1}
        Dinv = 1.0 / D
        Cf   = cf_matrix(F, Dinv)
        Cf_i = safe_inv(Cf)
        Sigma_inv = np.diag(Dinv) - (Dinv[:, None] * (F @ Cf_i @ F.T)) * Dinv[None, :]

        # β step: assemble A and rhs
        A   = np.zeros((P, P))
        rhs = np.zeros(P)
        row_off = 0
        for j in range(K):
            Sj = Sigma_inv[:, j]
            YSj = Y @ Sj
            rhs[row_off:row_off+p_list[j]] = Xs[j].T @ YSj
            col_off = 0
            Xtj = Xs[j].T
            for l in range(K):
                Gjl = Xtj @ Xs[l]
                A[row_off:row_off+p_list[j], col_off:col_off+p_list[l]] = Sj[l] * Gjl
                col_off += p_list[l]
            row_off += p_list[j]
        sol = safe_solve(A + lam_B*np.eye(P), rhs)
        B_list = unpack_vec_to_list(sol, p_list)

        # residual for next E-step
        R = Y - XB(B_list)

        # objective + guarded convergence
        obj = penalized_nll(Y, Xs, B_list, F, D, lam_F=lam_F, lam_B=lam_B)
        if not np.isfinite(obj):
            break
        if prev is not None:
            denom = max(1.0, abs(prev))
            rel_impr = (prev - obj) / denom
            if rel_impr < 0 and abs(prev - obj) / denom < tol:
                break
            if rel_impr < tol:
                break
        prev = obj

    mem_mb = (Sigma_inv.nbytes + A.nbytes) / 1e6  # peak-ish
    return B_list, F, D, mem_mb, time.time() - t0

# =========================
# Diagnostics
# =========================

def _diag_sigma_diag(F, D):
    return (F**2).sum(axis=1) + D

def coverage_and_mahalanobis(Y, Xs, B_list, F, D, alphas=(0.10, 0.05, 0.01)):
    N, K = Y.shape
    XB = predict_Y(Xs, B_list)
    R  = Y - XB

    s  = np.sqrt(_diag_sigma_diag(F, D))
    Z  = R / s[None, :]
    cov_overall = {}
    for a in alphas:
        thr = norm.ppf(1 - a/2.0)
        per = (np.abs(Z) <= thr).mean(axis=0)
        cov_overall[f"cov@{int((1-a)*100)}%"] = {
            "overall": float(per.mean()),
            "min_eq": float(per.min()),
            "max_eq": float(per.max())
        }

    Dinv = 1.0 / D
    Cf_i = safe_inv(cf_matrix(F, Dinv))
    RSig = siginv_apply_to_matrix(R, F, D, Cf_i)
    m2   = np.sum(R * RSig, axis=1)
    q95  = float(chi2.ppf(0.95, df=K))
    q99  = float(chi2.ppf(0.99, df=K))
    W    = (R.T @ RSig) / N
    frob = float(np.linalg.norm(W - np.eye(K), 'fro') / np.sqrt(K))
    offm = float(np.max(np.abs(W - np.diag(np.diag(W)))))
    return {
        "z_coverage_overall": cov_overall,
        "m2_mean": float(m2.mean()), "m2_var": float(m2.var()),
        "m2_frac_gt_95": float((m2 > q95).mean()), "m2_frac_gt_99": float((m2 > q99).mean()),
        "m2_q95": q95, "m2_q99": q99,
        "whiten_frob_dev": frob, "whiten_offdiag_max": offm,
        "K": K, "N": N
    }

def whitening_matrix(Y, Xs, B_list, F, D):
    N, K = Y.shape
    XB = predict_Y(Xs, B_list)
    R  = Y - XB
    Dinv = 1.0 / D
    Cf_i = safe_inv(cf_matrix(F, Dinv))
    RSig = siginv_apply_to_matrix(R, F, D, Cf_i)
    W    = (R.T @ RSig) / N
    return W

def top_eigvecs_sym(M, r=2):
    S = (M + M.T)/2.0
    vals, vecs = eigh(S)
    return vals[-r:], vecs[:, -r:]

def print_diag(label, d):
    print(f"\n[{label}]  K={d['K']}  N={d['N']}")
    for k in ["cov@90%","cov@95%","cov@99%"]:
        if k in d["z_coverage_overall"]:
            v = d["z_coverage_overall"][k]
            print(f"  z {k}: overall={v['overall']:.3f}, min_eq={v['min_eq']:.3f}, max_eq={v['max_eq']:.3f}")
    print(f"  m2 mean={d['m2_mean']:.2f} (≈K), var={d['m2_var']:.2f} (≈2K)")
    print(f"  frac m2>χ2_95={d['m2_frac_gt_95']:.3f} (≈0.05), >χ2_99={d['m2_frac_gt_99']:.3f} (≈0.01)")
    print(f"  whiten frob_dev={d['whiten_frob_dev']:.3f}, offdiag_max={d['whiten_offdiag_max']:.3f}")

# =========================
# Calibration: α–γ–τ–s (CV v3.1, conservative)
# =========================

def apply_scale(F, D, s):
    s = float(s); return np.sqrt(s) * F, s * D

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
        mid = 0.5*(lo+hi)
        f_mid, _ = tail95_at(mid)
        if abs(f_mid - target) <= tol: return float(mid)
        if (f_mid - target) * (f_lo - target) <= 0:
            hi = mid
        else:
            lo = mid; f_lo = f_mid
    return float(0.5*(lo+hi))

def inflate_corr_fixed_diag(F, D, gamma, d_floor=1e-6):
    FF_diag = np.sum(F**2, axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        gamma_max = np.min(np.where(FF_diag > 0, (D - d_floor) / FF_diag, np.inf))
    gamma_eff = float(max(0.0, min(gamma, gamma_max if np.isfinite(gamma_max) else gamma)))
    F_g = np.sqrt(1.0 + gamma_eff) * F
    D_g = np.maximum(D - gamma_eff * FF_diag, d_floor)
    return F_g, D_g, gamma_eff

def apply_alpha(D, alpha):
    return float(alpha) * D

def _score(diag, cov_target=0.975, tail_target=0.03):
    cov_err  = abs(diag["z_coverage_overall"]["cov@95%"]["overall"] - cov_target)
    tail_err = abs(diag["m2_frac_gt_95"] - tail_target) + 0.5*abs(diag["m2_frac_gt_99"] - 0.01)
    whiten   = diag["whiten_frob_dev"]
    w_white  = 0.6 + 0.9 * float((cov_err < 0.01) and (tail_err < 0.02))
    return 2.0*cov_err + 2.0*tail_err + w_white*whiten

def _gamma_line_search(Xs, Y, B_list, F, D_a, tail_target, cov_target,
                       d_floor=1e-6, coarse_pts=11, refine_steps=9):
    FF_diag = np.sum(F**2, axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
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
        if (best is None) or (sc < best[0]): best = (sc, g, s_hat, diag)
    _, g0, s0, _ = best

    phi = 0.5*(3 - np.sqrt(5))
    a = max(0.0, g0 - 0.25*g_hi); b = min(g0 + 0.25*g_hi, g_hi)
    x1 = a + (1 - phi)*(b - a); x2 = a + phi*(b - a)

    def eval_g(g):
        F_g, D_g, _ = inflate_corr_fixed_diag(F, D_a, g, d_floor=d_floor)
        s_hat = calibrate_s_bisection(Xs, Y, B_list, F_g, D_g, target=tail_target)
        F_s, D_s = apply_scale(F_g, D_g, s_hat)
        diag = coverage_and_mahalanobis(Y, Xs, B_list, F_s, D_s)
        return _score(diag, cov_target, tail_target), s_hat, diag

    f1, s1, d1 = eval_g(x1); f2, s2, d2 = eval_g(x2)
    for _ in range(refine_steps):
        if f1 < f2:
            b, f2, s2, d2 = x2, f1, s1, d1
            x2 = x1; x1 = a + (1 - phi)*(b - a)
            f1, s1, d1 = eval_g(x1)
        else:
            a, f1, s1, d1 = x1, f2, s2, d2
            x1 = x2; x2 = a + phi*(b - a)
            f2, s2, d2 = eval_g(x2)
    return (float(x1), (f1, s1, d1)) if f1 < f2 else (float(x2), (f2, s2, d2))

def top_eigvecs_sym(M, r=2):
    S = (M + M.T)/2.0
    vals, vecs = eigh(S)
    return vals[-r:], vecs[:, -r:]

def whitening_matrix(Y, Xs, B_list, F, D):
    N, K = Y.shape
    XB = predict_Y(Xs, B_list)
    R  = Y - XB
    Dinv = 1.0 / D
    Cf_i = safe_inv(cf_matrix(F, Dinv))
    RSig = siginv_apply_to_matrix(R, F, D, Cf_i)
    W    = (R.T @ RSig) / N
    return W

def _eigen_boost_line_search(Xs, Y, B_list, F_g, D_g, D_alpha,
                             tail_target, cov_target, r=2, d_floor=1e-6,
                             coarse_pts=11, refine_steps=9):
    W = whitening_matrix(Y, Xs, B_list, F_g, D_g)
    vals, U = top_eigvecs_sym(W - np.eye(W.shape[0]), r=r)
    if vals[-1] <= 0:
        return 0.0, None

    base = np.sqrt(D_alpha)[:, None] * U
    rowpow = np.sum(base**2, axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
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
        if (best is None) or (sc < best[0]): best = (sc, t, s_hat, diag)
    _, t0, s0, _ = best

    phi = 0.5*(3 - np.sqrt(5))
    a = max(0.0, t0 - 0.25*t_hi); b = min(t0 + 0.25*t_hi, t_hi)
    x1 = a + (1 - phi)*(b - a); x2 = a + phi*(b - a)

    def eval_t(t):
        F_aug = np.column_stack([F_g, np.sqrt(t) * base])
        D_new = np.maximum(D_g - t * rowpow, d_floor)
        s_hat = calibrate_s_bisection(Xs, Y, B_list, F_aug, D_new, target=tail_target)
        F_s, D_s = apply_scale(F_aug, D_new, s_hat)
        diag = coverage_and_mahalanobis(Y, Xs, B_list, F_s, D_s)
        return _score(diag, cov_target, tail_target), s_hat, diag

    f1, s1, d1 = eval_t(x1); f2, s2, d2 = eval_t(x2)
    for _ in range(refine_steps):
        if f1 < f2:
            b, f2, s2, d2 = x2, f1, s1, d1
            x2 = x1; x1 = a + (1 - phi)*(b - a)
            f1, s1, d1 = eval_t(x1)
        else:
            a, f1, s1, d1 = x1, f2, s2, d2
            x1 = x2; x2 = a + phi*(b - a)
            f2, s2, d2 = eval_t(x2)
    return (float(x1), (f1, s1, d1)) if f1 < f2 else (float(x2), (f2, s2, d2))

def calibrate_alpha_gamma_s_cv3_conservative(
    Xs, Y, B_list, F, D,
    alpha_grid=None, Kfolds=3, val_frac=0.5,
    tail_target=0.03, cov_target=0.975, r_boost=2,
    rng_seed=123, d_floor=1e-6
):
    rng = np.random.default_rng(rng_seed)
    N = Y.shape[0]
    if alpha_grid is None:
        alpha_grid = np.linspace(0.78, 1.10, 17)

    # folds
    folds = []
    for _ in range(Kfolds):
        val_idx = rng.choice(N, size=int(val_frac*N), replace=False)
        trn_mask = np.ones(N, dtype=bool); trn_mask[val_idx] = False
        folds.append((
            [X[trn_mask] for X in Xs], Y[trn_mask],
            [X[val_idx] for X in Xs],  Y[val_idx]
        ))

    best = None
    for alpha in alpha_grid:
        fold_picks = []
        for (Xs_tr, Y_tr, Xs_va, Y_va) in folds:
            D_a = apply_alpha(D, alpha)
            g_star, gpack = _gamma_line_search(
                Xs_va, Y_va, B_list, F, D_a,
                tail_target=tail_target, cov_target=cov_target, d_floor=d_floor
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
                Xs_va, Y_va, B_list, F_g, D_g, D_a,
                tail_target=tail_target, cov_target=cov_target, r=r_boost, d_floor=d_floor
            )
            if tpack is None:
                score_t = np.inf
            else:
                score_t, s_t, _ = tpack

            if score_t < score_g:
                fold_picks.append(("eig", g_star, t_star, s_t))
            else:
                fold_picks.append(("gam", g_star, 0.0, s_g))

        modes  = [m for (m,_,_,_) in fold_picks]
        use_eig = (modes.count("eig") >= modes.count("gam"))
        gammas = np.array([g for (_,g,_,_) in fold_picks])
        taus   = np.array([t for (_,_,t,_) in fold_picks])
        ss     = np.array([s for (_,_,_,s) in fold_picks])
        q = 0.75  # conservative summary
        gamma_q = float(np.quantile(gammas, q))
        tau_q   = float(np.quantile(taus, q)) if use_eig else 0.0
        s_q     = float(np.quantile(ss, q))

        D_a = apply_alpha(D, alpha)
        F_g, D_g, _ = inflate_corr_fixed_diag(F, D_a, gamma_q, d_floor=d_floor)
        if use_eig and tau_q > 0:
            W_tr = whitening_matrix(Y, Xs, B_list, F_g, D_g)
            _, U = top_eigvecs_sym(W_tr - np.eye(F_g.shape[0]), r=r_boost)
            base  = np.sqrt(D_a)[:, None] * U
            rowpw = np.sum(base**2, axis=1)
            F_aug = np.column_stack([F_g, np.sqrt(tau_q)*base])
            D_new = np.maximum(D_g - tau_q * rowpw, d_floor)
        else:
            F_aug, D_new = F_g, D_g
        F_s, D_s = apply_scale(F_aug, D_new, s_q)
        diag_try = coverage_and_mahalanobis(Y, Xs, B_list, F_s, D_s)
        summary_score = _score(diag_try, cov_target, tail_target)
        if (best is None) or (summary_score < best[0]):
            best = (summary_score, alpha, gamma_q, tau_q, s_q, use_eig)

    _, alpha_star, gamma_star, tau_star, s_star, use_eig_star = best
    return float(alpha_star), float(gamma_star), float(tau_star), float(s_star), bool(use_eig_star)

def finalize_on_dataset(Xs, Y, B_list, F, D, alpha, gamma, tau, s, use_eig=True, r_boost=2, d_floor=1e-6):
    D_a = apply_alpha(D, alpha)
    F_g, D_g, _ = inflate_corr_fixed_diag(F, D_a, gamma, d_floor=d_floor)
    if use_eig and (tau > 0):
        W = whitening_matrix(Y, Xs, B_list, F_g, D_g)
        _, U = top_eigvecs_sym(W - np.eye(F_g.shape[0]), r=r_boost)
        base  = np.sqrt(D_a)[:, None] * U
        rowpw = np.sum(base**2, axis=1)
        F_g   = np.column_stack([F_g, np.sqrt(tau) * base])
        D_g   = np.maximum(D_g - tau * rowpw, d_floor)
    F_fin, D_fin = apply_scale(F_g, D_g, s)
    return F_fin, D_fin
# lowrank_gls/core.py
# Core implementation: low-rank+diag GLS/SUR with ALS (matrix-free) and EM (dense),
# diagnostics (coverage, Mahalanobis, whitening), and a conservative CV calibrator
# (alpha–gamma–tau–s with optional r-direction eigen-boost). Includes a diagonal
# preconditioner for CG to tighten ALS–EM likelihood gaps and speed convergence.

import time
import numpy as np
from numpy.linalg import solve, LinAlgError, eigh
from scipy.sparse.linalg import LinearOperator, cg
from scipy.stats import norm, chi2

# =========================
# Numerics & Woodbury utils
# =========================

def safe_inv(A, lam=1e-6):
    I = np.eye(A.shape[0])
    ridge = max(lam * np.linalg.norm(A, 'fro') / A.shape[0], 1e-8)
    try:
        return solve(A + ridge * I, I)
    except LinAlgError:
        return np.linalg.pinv(A + ridge * I)

def safe_solve(A, b, lam=1e-6):
    I = np.eye(A.shape[0])
    ridge = max(lam * np.linalg.norm(A, 'fro') / A.shape[0], 1e-8)
    try:
        return solve(A + ridge * I, b)
    except LinAlgError:
        return np.linalg.lstsq(A + ridge * I, b, rcond=None)[0]

def wpca_init(R, k, D):
    # Weighted PCA on residuals with weights 1/sqrt(D)
    W = R / np.sqrt(D)[None, :]
    _, s, Vt = np.linalg.svd(W, full_matrices=False)
    r = max(1, min(k, (s > 1e-8).sum()))
    Fw = Vt.T[:, :r] * np.sqrt(s[:r])
    F  = Fw * np.sqrt(D)[:, None]
    return np.pad(F, ((0, 0), (0, k - r)))

def cf_matrix(F, Dinv):
    return np.eye(F.shape[1]) + F.T @ (Dinv[:, None] * F)

def siginv_apply_to_matrix(M, F, D, Cf_inv=None):
    # Compute M @ Σ^{-1} using Woodbury; M is N×K.
    Dinv = 1.0 / D
    if Cf_inv is None:
        Cf_inv = safe_inv(cf_matrix(F, Dinv))
    Vd  = M * Dinv[None, :]
    FD  = Dinv[:, None] * F
    return Vd - ((Vd @ F) @ Cf_inv) @ FD.T

# =========================
# Pack/unpack for β
# =========================

def pack_list_to_vec(B_list):
    return np.concatenate([b.ravel() for b in B_list], axis=0)

def unpack_vec_to_list(v, p_list):
    out, i = [], 0
    for p in p_list:
        out.append(v[i:i+p].reshape(p, 1))
        i += p
    return out

def predict_Y(Xs, B_list):
    return np.column_stack([Xs[j] @ B_list[j] for j in range(len(Xs))])

def mse(Y, Yhat):
    return float(np.mean((Y - Yhat)**2))

# =========================
# Likelihood (reporting)
# =========================

def _safe_logdet_psd(A):
    sign, logdet = np.linalg.slogdet(A)
    if sign > 0 and np.isfinite(logdet):
        return logdet
    # Fallback: perturb slightly to ensure PD
    eps = 1e-10
    return np.linalg.slogdet(A + eps*np.eye(A.shape[0]))[1]

def penalized_nll(Y, Xs, B_list, F, D, lam_F=0.0, lam_B=0.0):
    N, K = Y.shape
    XB = predict_Y(Xs, B_list)
    R  = Y - XB
    Dinv = 1.0 / D
    Cf = cf_matrix(F, Dinv); Cf_i = safe_inv(Cf)
    term1 = np.sum(R * (R * Dinv[None, :]))
    M = (R * Dinv[None, :]) @ F
    term2 = np.sum(M * (M @ Cf_i))
    quad = term1 - term2
    logdet = np.sum(np.log(D)) + _safe_logdet_psd(Cf)
    pen = lam_F*np.sum(F**2) + lam_B*sum(np.sum(b**2) for b in B_list)
    return 0.5*(N*logdet + quad) + pen

def test_nll(Y, Xs, B_list, F, D):
    N, K = Y.shape
    XB = predict_Y(Xs, B_list)
    R  = Y - XB
    Dinv = 1.0 / D
    Cf = cf_matrix(F, Dinv); Cf_i = safe_inv(Cf)
    term1 = np.sum(R * (R * Dinv[None, :]))
    M = (R * Dinv[None, :]) @ F
    term2 = np.sum(M * (M @ Cf_i))
    quad = term1 - term2
    logdet = np.sum(np.log(D)) + _safe_logdet_psd(Cf)
    return 0.5*(N*logdet + quad)/N

# =========================
# Preconditioner for CG (diag/Jacobi)
# =========================

def build_diag_precond(Xs, F, D, Cf_i, lam_B):
    K = len(Xs)
    p_list = [X.shape[1] for X in Xs]
    Dinv = 1.0 / D
    # diag(F Cf_i F^T)
    diag_term = np.sum(F * (F @ Cf_i), axis=1)
    sigma_inv_diag = Dinv - (Dinv**2) * diag_term
    diag_blocks = []
    for j in range(K):
        Hjj = Xs[j].T @ Xs[j]
        d = sigma_inv_diag[j] * np.diag(Hjj) + lam_B
        diag_blocks.append(d)
    d_all = np.concatenate(diag_blocks)
    d_all = np.maximum(d_all, 1e-12)
    return LinearOperator((d_all.size, d_all.size), matvec=lambda v: v / d_all)

# =========================
# ALS solver (matrix-free β with optional preconditioning)
# =========================

def als_gls(Xs, Y, k, lam_F=1e-3, lam_B=1e-3, sweeps=12, tol=1e-5, eps=1e-6,
            d_floor=1e-6, use_cg_beta=True, cg_maxit=800, cg_tol=3e-7,
            use_diag_precond=True):
    t0 = time.time()
    N, K = Y.shape
    p_list = [X.shape[1] for X in Xs]
    P = sum(p_list)

    # init
    B_list = [safe_solve(Xs[j].T @ Xs[j], Xs[j].T @ Y[:, [j]]) for j in range(K)]
    def XB(Bs): return predict_Y(Xs, Bs)
    R = Y - XB(B_list)
    D = np.var(R, axis=0) + eps
    F = wpca_init(R, k, D)

    prev = None  # <-- guard to avoid inf/inf on first iteration

    for _ in range(sweeps):
        Dinv = 1.0 / D
        Cf   = cf_matrix(F, Dinv)
        Cf_i = safe_inv(Cf)

        # RHS
        YSig = siginv_apply_to_matrix(Y, F, D, Cf_i)
        rhs_blocks = [Xs[j].T @ YSig[:, [j]] for j in range(K)]
        rhs = pack_list_to_vec(rhs_blocks)

        # matrix-free A·vec
        def mv(vec):
            Bs = unpack_vec_to_list(vec, p_list)
            V  = predict_Y(Xs, Bs)
            T  = siginv_apply_to_matrix(V, F, D, Cf_i)
            out_blocks = [Xs[j].T @ T[:, [j]] for j in range(K)]
            out = pack_list_to_vec(out_blocks)
            return out + lam_B * vec

        if use_cg_beta:
            Aop = LinearOperator((P, P), matvec=mv)
            M = build_diag_precond(Xs, F, D, Cf_i, lam_B) if use_diag_precond else None
            sol, info = cg(Aop, rhs, tol=cg_tol, maxiter=cg_maxit, M=M)
            if info != 0:
                # Fallback: assemble and solve
                Sigma_inv = np.diag(Dinv) - (Dinv[:, None] * (F @ Cf_i @ F.T)) * Dinv[None, :]
                A = np.zeros((P, P))
                row_off = 0
                for j in range(K):
                    Sj = Sigma_inv[:, j]
                    col_off = 0
                    Xtj = Xs[j].T
                    for l in range(K):
                        Gjl = Xtj @ Xs[l]
                        A[row_off:row_off+p_list[j], col_off:col_off+p_list[l]] = Sj[l] * Gjl
                        col_off += p_list[l]
                    row_off += p_list[j]
                sol = safe_solve(A + lam_B*np.eye(P), rhs)
            B_list = unpack_vec_to_list(sol, p_list)
        else:
            Sigma_inv = np.diag(Dinv) - (Dinv[:, None] * (F @ Cf_i @ F.T)) * Dinv[None, :]
            A = np.zeros((P, P))
            row_off = 0
            for j in range(K):
                Sj = Sigma_inv[:, j]
                col_off = 0
                Xtj = Xs[j].T
                for l in range(K):
                    Gjl = Xtj @ Xs[l]
                    A[row_off:row_off+p_list[j], col_off:col_off+p_list[l]] = Sj[l] * Gjl
                    col_off += p_list[l]
                row_off += p_list[j]
            sol = safe_solve(A + lam_B*np.eye(P), rhs)
            B_list = unpack_vec_to_list(sol, p_list)

        # update F, D
        R = Y - XB(B_list)
        FtF = F.T @ F
        U   = R @ F @ safe_inv(FtF + lam_F*np.eye(k))
        UtU = U.T @ U
        F   = R.T @ U @ safe_inv(UtU + lam_F*np.eye(k))
        D   = np.maximum(np.mean((R - U @ F.T)**2, axis=0) + eps, d_floor)

        # objective + guarded convergence
        obj = penalized_nll(Y, Xs, B_list, F, D, lam_F=lam_F, lam_B=lam_B)
        if not np.isfinite(obj):
            break
        if prev is not None:
            denom = max(1.0, abs(prev))
            rel_impr = (prev - obj) / denom
            # tolerate tiny non-monotone wiggles from inexact CG
            if rel_impr < 0 and abs(prev - obj) / denom < tol:
                break
            if rel_impr < tol:
                break
        prev = obj

    mem_mb = ((K*k) + K) * 8 / 1e6  # ~F and D
    return B_list, F, D, mem_mb, time.time() - t0

# =========================
# Dense EM solver (β via full A), with optional warm-start
# =========================

def em_gls(Xs, Y, k, lam_F=1e-3, lam_B=1e-3, iters=45, tol=1e-5, eps=1e-6, d_floor=1e-6,
           B_init=None, F_init=None, D_init=None):
    t0 = time.time()
    N, K = Y.shape
    p_list = [X.shape[1] for X in Xs]
    P = sum(p_list)

    # init (allow warm-start)
    if B_init is None:
        B_list = [safe_solve(Xs[j].T @ Xs[j], Xs[j].T @ Y[:, [j]]) for j in range(K)]
    else:
        B_list = [b.copy() for b in B_init]
    def XB(Bs): return predict_Y(Xs, Bs)
    R = Y - XB(B_list)
    if D_init is None:
        D = np.var(R, axis=0) + eps
    else:
        D = D_init.copy()
    if F_init is None:
        F = wpca_init(R, k, D)
    else:
        F = F_init.copy()

    prev = None  # <-- guard to avoid inf/inf on first iteration

    for _ in range(iters):
        # E-step
        Dinv = 1.0 / D
        Cf   = cf_matrix(F, Dinv)
        Cf_i = safe_inv(Cf)
        EZ   = (R * Dinv[None, :]) @ F @ Cf_i
        EZZ  = EZ.T @ EZ + N * Cf_i

        # M-step for F, D
        F    = R.T @ EZ @ safe_inv(EZZ + lam_F*np.eye(k))
        Rmd  = R - EZ @ F.T
        D    = np.maximum(np.mean(Rmd**2, axis=0) + eps, d_floor)

        # Σ^{-1}
        Dinv = 1.0 / D
        Cf   = cf_matrix(F, Dinv)
        Cf_i = safe_inv(Cf)
        Sigma_inv = np.diag(Dinv) - (Dinv[:, None] * (F @ Cf_i @ F.T)) * Dinv[None, :]

        # β step: assemble A and rhs
        A   = np.zeros((P, P))
        rhs = np.zeros(P)
        row_off = 0
        for j in range(K):
            Sj = Sigma_inv[:, j]
            YSj = Y @ Sj
            rhs[row_off:row_off+p_list[j]] = Xs[j].T @ YSj
            col_off = 0
            Xtj = Xs[j].T
            for l in range(K):
                Gjl = Xtj @ Xs[l]
                A[row_off:row_off+p_list[j], col_off:col_off+p_list[l]] = Sj[l] * Gjl
                col_off += p_list[l]
            row_off += p_list[j]
        sol = safe_solve(A + lam_B*np.eye(P), rhs)
        B_list = unpack_vec_to_list(sol, p_list)

        # residual for next E-step
        R = Y - XB(B_list)

        # objective + guarded convergence
        obj = penalized_nll(Y, Xs, B_list, F, D, lam_F=lam_F, lam_B=lam_B)
        if not np.isfinite(obj):
            break
        if prev is not None:
            denom = max(1.0, abs(prev))
            rel_impr = (prev - obj) / denom
            if rel_impr < 0 and abs(prev - obj) / denom < tol:
                break
            if rel_impr < tol:
                break
        prev = obj

    mem_mb = (Sigma_inv.nbytes + A.nbytes) / 1e6  # peak-ish
    return B_list, F, D, mem_mb, time.time() - t0

# =========================
# Diagnostics
# =========================

def _diag_sigma_diag(F, D):
    return (F**2).sum(axis=1) + D

def coverage_and_mahalanobis(Y, Xs, B_list, F, D, alphas=(0.10, 0.05, 0.01)):
    N, K = Y.shape
    XB = predict_Y(Xs, B_list)
    R  = Y - XB

    s  = np.sqrt(_diag_sigma_diag(F, D))
    Z  = R / s[None, :]
    cov_overall = {}
    for a in alphas:
        thr = norm.ppf(1 - a/2.0)
        per = (np.abs(Z) <= thr).mean(axis=0)
        cov_overall[f"cov@{int((1-a)*100)}%"] = {
            "overall": float(per.mean()),
            "min_eq": float(per.min()),
            "max_eq": float(per.max())
        }

    Dinv = 1.0 / D
    Cf_i = safe_inv(cf_matrix(F, Dinv))
    RSig = siginv_apply_to_matrix(R, F, D, Cf_i)
    m2   = np.sum(R * RSig, axis=1)
    q95  = float(chi2.ppf(0.95, df=K))
    q99  = float(chi2.ppf(0.99, df=K))
    W    = (R.T @ RSig) / N
    frob = float(np.linalg.norm(W - np.eye(K), 'fro') / np.sqrt(K))
    offm = float(np.max(np.abs(W - np.diag(np.diag(W)))))
    return {
        "z_coverage_overall": cov_overall,
        "m2_mean": float(m2.mean()), "m2_var": float(m2.var()),
        "m2_frac_gt_95": float((m2 > q95).mean()), "m2_frac_gt_99": float((m2 > q99).mean()),
        "m2_q95": q95, "m2_q99": q99,
        "whiten_frob_dev": frob, "whiten_offdiag_max": offm,
        "K": K, "N": N
    }

def whitening_matrix(Y, Xs, B_list, F, D):
    N, K = Y.shape
    XB = predict_Y(Xs, B_list)
    R  = Y - XB
    Dinv = 1.0 / D
    Cf_i = safe_inv(cf_matrix(F, Dinv))
    RSig = siginv_apply_to_matrix(R, F, D, Cf_i)
    W    = (R.T @ RSig) / N
    return W

def top_eigvecs_sym(M, r=2):
    S = (M + M.T)/2.0
    vals, vecs = eigh(S)
    return vals[-r:], vecs[:, -r:]

def print_diag(label, d):
    print(f"\n[{label}]  K={d['K']}  N={d['N']}")
    for k in ["cov@90%","cov@95%","cov@99%"]:
        if k in d["z_coverage_overall"]:
            v = d["z_coverage_overall"][k]
            print(f"  z {k}: overall={v['overall']:.3f}, min_eq={v['min_eq']:.3f}, max_eq={v['max_eq']:.3f}")
    print(f"  m2 mean={d['m2_mean']:.2f} (≈K), var={d['m2_var']:.2f} (≈2K)")
    print(f"  frac m2>χ2_95={d['m2_frac_gt_95']:.3f} (≈0.05), >χ2_99={d['m2_frac_gt_99']:.3f} (≈0.01)")
    print(f"  whiten frob_dev={d['whiten_frob_dev']:.3f}, offdiag_max={d['whiten_offdiag_max']:.3f}")

# =========================
# Calibration: α–γ–τ–s (CV v3.1, conservative)
# =========================

def apply_scale(F, D, s):
    s = float(s); return np.sqrt(s) * F, s * D

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
        mid = 0.5*(lo+hi)
        f_mid, _ = tail95_at(mid)
        if abs(f_mid - target) <= tol: return float(mid)
        if (f_mid - target) * (f_lo - target) <= 0:
            hi = mid
        else:
            lo = mid; f_lo = f_mid
    return float(0.5*(lo+hi))

def inflate_corr_fixed_diag(F, D, gamma, d_floor=1e-6):
    FF_diag = np.sum(F**2, axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        gamma_max = np.min(np.where(FF_diag > 0, (D - d_floor) / FF_diag, np.inf))
    gamma_eff = float(max(0.0, min(gamma, gamma_max if np.isfinite(gamma_max) else gamma)))
    F_g = np.sqrt(1.0 + gamma_eff) * F
    D_g = np.maximum(D - gamma_eff * FF_diag, d_floor)
    return F_g, D_g, gamma_eff

def apply_alpha(D, alpha):
    return float(alpha) * D

def _score(diag, cov_target=0.975, tail_target=0.03):
    cov_err  = abs(diag["z_coverage_overall"]["cov@95%"]["overall"] - cov_target)
    tail_err = abs(diag["m2_frac_gt_95"] - tail_target) + 0.5*abs(diag["m2_frac_gt_99"] - 0.01)
    whiten   = diag["whiten_frob_dev"]
    w_white  = 0.6 + 0.9 * float((cov_err < 0.01) and (tail_err < 0.02))
    return 2.0*cov_err + 2.0*tail_err + w_white*whiten

def _gamma_line_search(Xs, Y, B_list, F, D_a, tail_target, cov_target,
                       d_floor=1e-6, coarse_pts=11, refine_steps=9):
    FF_diag = np.sum(F**2, axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
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
        if (best is None) or (sc < best[0]): best = (sc, g, s_hat, diag)
    _, g0, s0, _ = best

    phi = 0.5*(3 - np.sqrt(5))
    a = max(0.0, g0 - 0.25*g_hi); b = min(g0 + 0.25*g_hi, g_hi)
    x1 = a + (1 - phi)*(b - a); x2 = a + phi*(b - a)

    def eval_g(g):
        F_g, D_g, _ = inflate_corr_fixed_diag(F, D_a, g, d_floor=d_floor)
        s_hat = calibrate_s_bisection(Xs, Y, B_list, F_g, D_g, target=tail_target)
        F_s, D_s = apply_scale(F_g, D_g, s_hat)
        diag = coverage_and_mahalanobis(Y, Xs, B_list, F_s, D_s)
        return _score(diag, cov_target, tail_target), s_hat, diag

    f1, s1, d1 = eval_g(x1); f2, s2, d2 = eval_g(x2)
    for _ in range(refine_steps):
        if f1 < f2:
            b, f2, s2, d2 = x2, f1, s1, d1
            x2 = x1; x1 = a + (1 - phi)*(b - a)
            f1, s1, d1 = eval_g(x1)
        else:
            a, f1, s1, d1 = x1, f2, s2, d2
            x1 = x2; x2 = a + phi*(b - a)
            f2, s2, d2 = eval_g(x2)
    return (float(x1), (f1, s1, d1)) if f1 < f2 else (float(x2), (f2, s2, d2))

def top_eigvecs_sym(M, r=2):
    S = (M + M.T)/2.0
    vals, vecs = eigh(S)
    return vals[-r:], vecs[:, -r:]

def whitening_matrix(Y, Xs, B_list, F, D):
    N, K = Y.shape
    XB = predict_Y(Xs, B_list)
    R  = Y - XB
    Dinv = 1.0 / D
    Cf_i = safe_inv(cf_matrix(F, Dinv))
    RSig = siginv_apply_to_matrix(R, F, D, Cf_i)
    W    = (R.T @ RSig) / N
    return W

def _eigen_boost_line_search(Xs, Y, B_list, F_g, D_g, D_alpha,
                             tail_target, cov_target, r=2, d_floor=1e-6,
                             coarse_pts=11, refine_steps=9):
    W = whitening_matrix(Y, Xs, B_list, F_g, D_g)
    vals, U = top_eigvecs_sym(W - np.eye(W.shape[0]), r=r)
    if vals[-1] <= 0:
        return 0.0, None

    base = np.sqrt(D_alpha)[:, None] * U
    rowpow = np.sum(base**2, axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
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
        if (best is None) or (sc < best[0]): best = (sc, t, s_hat, diag)
    _, t0, s0, _ = best

    phi = 0.5*(3 - np.sqrt(5))
    a = max(0.0, t0 - 0.25*t_hi); b = min(t0 + 0.25*t_hi, t_hi)
    x1 = a + (1 - phi)*(b - a); x2 = a + phi*(b - a)

    def eval_t(t):
        F_aug = np.column_stack([F_g, np.sqrt(t) * base])
        D_new = np.maximum(D_g - t * rowpow, d_floor)
        s_hat = calibrate_s_bisection(Xs, Y, B_list, F_aug, D_new, target=tail_target)
        F_s, D_s = apply_scale(F_aug, D_new, s_hat)
        diag = coverage_and_mahalanobis(Y, Xs, B_list, F_s, D_s)
        return _score(diag, cov_target, tail_target), s_hat, diag

    f1, s1, d1 = eval_t(x1); f2, s2, d2 = eval_t(x2)
    for _ in range(refine_steps):
        if f1 < f2:
            b, f2, s2, d2 = x2, f1, s1, d1
            x2 = x1; x1 = a + (1 - phi)*(b - a)
            f1, s1, d1 = eval_t(x1)
        else:
            a, f1, s1, d1 = x1, f2, s2, d2
            x1 = x2; x2 = a + phi*(b - a)
            f2, s2, d2 = eval_t(x2)
    return (float(x1), (f1, s1, d1)) if f1 < f2 else (float(x2), (f2, s2, d2))

def calibrate_alpha_gamma_s_cv3_conservative(
    Xs, Y, B_list, F, D,
    alpha_grid=None, Kfolds=3, val_frac=0.5,
    tail_target=0.03, cov_target=0.975, r_boost=2,
    rng_seed=123, d_floor=1e-6
):
    rng = np.random.default_rng(rng_seed)
    N = Y.shape[0]
    if alpha_grid is None:
        alpha_grid = np.linspace(0.78, 1.10, 17)

    # folds
    folds = []
    for _ in range(Kfolds):
        val_idx = rng.choice(N, size=int(val_frac*N), replace=False)
        trn_mask = np.ones(N, dtype=bool); trn_mask[val_idx] = False
        folds.append((
            [X[trn_mask] for X in Xs], Y[trn_mask],
            [X[val_idx] for X in Xs],  Y[val_idx]
        ))

    best = None
    for alpha in alpha_grid:
        fold_picks = []
        for (Xs_tr, Y_tr, Xs_va, Y_va) in folds:
            D_a = apply_alpha(D, alpha)
            g_star, gpack = _gamma_line_search(
                Xs_va, Y_va, B_list, F, D_a,
                tail_target=tail_target, cov_target=cov_target, d_floor=d_floor
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
                Xs_va, Y_va, B_list, F_g, D_g, D_a,
                tail_target=tail_target, cov_target=cov_target, r=r_boost, d_floor=d_floor
            )
            if tpack is None:
                score_t = np.inf
            else:
                score_t, s_t, _ = tpack

            if score_t < score_g:
                fold_picks.append(("eig", g_star, t_star, s_t))
            else:
                fold_picks.append(("gam", g_star, 0.0, s_g))

        modes  = [m for (m,_,_,_) in fold_picks]
        use_eig = (modes.count("eig") >= modes.count("gam"))
        gammas = np.array([g for (_,g,_,_) in fold_picks])
        taus   = np.array([t for (_,_,t,_) in fold_picks])
        ss     = np.array([s for (_,_,_,s) in fold_picks])
        q = 0.75  # conservative summary
        gamma_q = float(np.quantile(gammas, q))
        tau_q   = float(np.quantile(taus, q)) if use_eig else 0.0
        s_q     = float(np.quantile(ss, q))

        D_a = apply_alpha(D, alpha)
        F_g, D_g, _ = inflate_corr_fixed_diag(F, D_a, gamma_q, d_floor=d_floor)
        if use_eig and tau_q > 0:
            W_tr = whitening_matrix(Y, Xs, B_list, F_g, D_g)
            _, U = top_eigvecs_sym(W_tr - np.eye(F_g.shape[0]), r=r_boost)
            base  = np.sqrt(D_a)[:, None] * U
            rowpw = np.sum(base**2, axis=1)
            F_aug = np.column_stack([F_g, np.sqrt(tau_q)*base])
            D_new = np.maximum(D_g - tau_q * rowpw, d_floor)
        else:
            F_aug, D_new = F_g, D_g
        F_s, D_s = apply_scale(F_aug, D_new, s_q)
        diag_try = coverage_and_mahalanobis(Y, Xs, B_list, F_s, D_s)
        summary_score = _score(diag_try, cov_target, tail_target)
        if (best is None) or (summary_score < best[0]):
            best = (summary_score, alpha, gamma_q, tau_q, s_q, use_eig)

    _, alpha_star, gamma_star, tau_star, s_star, use_eig_star = best
    return float(alpha_star), float(gamma_star), float(tau_star), float(s_star), bool(use_eig_star)

def finalize_on_dataset(Xs, Y, B_list, F, D, alpha, gamma, tau, s, use_eig=True, r_boost=2, d_floor=1e-6):
    D_a = apply_alpha(D, alpha)
    F_g, D_g, _ = inflate_corr_fixed_diag(F, D_a, gamma, d_floor=d_floor)
    if use_eig and (tau > 0):
        W = whitening_matrix(Y, Xs, B_list, F_g, D_g)
        _, U = top_eigvecs_sym(W - np.eye(F_g.shape[0]), r=r_boost)
        base  = np.sqrt(D_a)[:, None] * U
        rowpw = np.sum(base**2, axis=1)
        F_g   = np.column_stack([F_g, np.sqrt(tau) * base])
        D_g   = np.maximum(D_g - tau * rowpw, d_floor)
    F_fin, D_fin = apply_scale(F_g, D_g, s)
    return F_fin, D_fin
