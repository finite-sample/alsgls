import time
import numpy as np
from scipy.sparse.linalg import LinearOperator, cg

from .numerics import (
    safe_solve,
    wpca_init,
    cf_matrix,
    safe_inv,
    siginv_apply_to_matrix,
    pack_list_to_vec,
    unpack_vec_to_list,
    predict_Y,
    penalized_nll,
)


def build_diag_precond(Xs, F, D, Cf_i, lam_B):
    K = len(Xs)
    p_list = [X.shape[1] for X in Xs]
    Dinv = 1.0 / D
    diag_term = np.sum(F * (F @ Cf_i), axis=1)
    sigma_inv_diag = Dinv - (Dinv ** 2) * diag_term
    diag_blocks = []
    for j in range(K):
        Hjj = Xs[j].T @ Xs[j]
        d = sigma_inv_diag[j] * np.diag(Hjj) + lam_B
        diag_blocks.append(d)
    d_all = np.concatenate(diag_blocks)
    d_all = np.maximum(d_all, 1e-12)
    return LinearOperator((d_all.size, d_all.size), matvec=lambda v: v / d_all)


def als_gls(
    Xs,
    Y,
    k,
    lam_F=1e-3,
    lam_B=1e-3,
    sweeps=12,
    tol=1e-5,
    eps=1e-6,
    d_floor=1e-6,
    use_cg_beta=True,
    cg_maxit=800,
    cg_tol=3e-7,
    use_diag_precond=True,
):
    t0 = time.time()
    N, K = Y.shape
    p_list = [X.shape[1] for X in Xs]
    P = sum(p_list)

    B_list = [safe_solve(Xs[j].T @ Xs[j], Xs[j].T @ Y[:, [j]]) for j in range(K)]
    def XB(Bs):
        return predict_Y(Xs, Bs)
    R = Y - XB(B_list)
    D = np.var(R, axis=0) + eps
    F = wpca_init(R, k, D)

    prev = None
    for _ in range(sweeps):
        Dinv = 1.0 / D
        Cf = cf_matrix(F, Dinv)
        Cf_i = safe_inv(Cf)

        YSig = siginv_apply_to_matrix(Y, F, D, Cf_i)
        rhs_blocks = [Xs[j].T @ YSig[:, [j]] for j in range(K)]
        rhs = pack_list_to_vec(rhs_blocks)

        def mv(vec):
            Bs = unpack_vec_to_list(vec, p_list)
            V = predict_Y(Xs, Bs)
            T = siginv_apply_to_matrix(V, F, D, Cf_i)
            out_blocks = [Xs[j].T @ T[:, [j]] for j in range(K)]
            out = pack_list_to_vec(out_blocks)
            return out + lam_B * vec

        if use_cg_beta:
            Aop = LinearOperator((P, P), matvec=mv)
            M = build_diag_precond(Xs, F, D, Cf_i, lam_B) if use_diag_precond else None
            sol, info = cg(Aop, rhs, tol=cg_tol, maxiter=cg_maxit, M=M)
            if info != 0:
                Sigma_inv = np.diag(Dinv) - (Dinv[:, None] * (F @ Cf_i @ F.T)) * Dinv[None, :]
                A = np.zeros((P, P))
                row_off = 0
                for j in range(K):
                    Sj = Sigma_inv[:, j]
                    col_off = 0
                    Xtj = Xs[j].T
                    for l in range(K):
                        Gjl = Xtj @ Xs[l]
                        A[row_off:row_off + p_list[j], col_off:col_off + p_list[l]] = Sj[l] * Gjl
                        col_off += p_list[l]
                    row_off += p_list[j]
                sol = safe_solve(A + lam_B * np.eye(P), rhs)
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
                    A[row_off:row_off + p_list[j], col_off:col_off + p_list[l]] = Sj[l] * Gjl
                    col_off += p_list[l]
                row_off += p_list[j]
            sol = safe_solve(A + lam_B * np.eye(P), rhs)
            B_list = unpack_vec_to_list(sol, p_list)

        R = Y - XB(B_list)
        FtF = F.T @ F
        U = R @ F @ safe_inv(FtF + lam_F * np.eye(k))
        UtU = U.T @ U
        F = R.T @ U @ safe_inv(UtU + lam_F * np.eye(k))
        D = np.maximum(np.mean((R - U @ F.T) ** 2, axis=0) + eps, d_floor)

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

    mem_mb = ((K * k) + K) * 8 / 1e6
    return B_list, F, D, mem_mb, time.time() - t0


def em_gls(
    Xs,
    Y,
    k,
    lam_F=1e-3,
    lam_B=1e-3,
    iters=45,
    tol=1e-5,
    eps=1e-6,
    d_floor=1e-6,
    B_init=None,
    F_init=None,
    D_init=None,
):
    t0 = time.time()
    N, K = Y.shape
    p_list = [X.shape[1] for X in Xs]
    P = sum(p_list)

    if B_init is None:
        B_list = [safe_solve(Xs[j].T @ Xs[j], Xs[j].T @ Y[:, [j]]) for j in range(K)]
    else:
        B_list = [b.copy() for b in B_init]
    def XB(Bs):
        return predict_Y(Xs, Bs)
    R = Y - XB(B_list)
    if D_init is None:
        D = np.var(R, axis=0) + eps
    else:
        D = D_init.copy()
    if F_init is None:
        F = wpca_init(R, k, D)
    else:
        F = F_init.copy()

    prev = None
    for _ in range(iters):
        Dinv = 1.0 / D
        Cf = cf_matrix(F, Dinv)
        Cf_i = safe_inv(Cf)
        EZ = (R * Dinv[None, :]) @ F @ Cf_i
        EZZ = EZ.T @ EZ + N * Cf_i

        F = R.T @ EZ @ safe_inv(EZZ + lam_F * np.eye(k))
        Rmd = R - EZ @ F.T
        D = np.maximum(np.mean(Rmd ** 2, axis=0) + eps, d_floor)

        Dinv = 1.0 / D
        Cf = cf_matrix(F, Dinv)
        Cf_i = safe_inv(Cf)
        Sigma_inv = np.diag(Dinv) - (Dinv[:, None] * (F @ Cf_i @ F.T)) * Dinv[None, :]

        A = np.zeros((P, P))
        rhs = np.zeros(P)
        row_off = 0
        for j in range(K):
            Sj = Sigma_inv[:, j]
            YSj = Y @ Sj
            rhs[row_off:row_off + p_list[j]] = Xs[j].T @ YSj
            col_off = 0
            Xtj = Xs[j].T
            for l in range(K):
                Gjl = Xtj @ Xs[l]
                A[row_off:row_off + p_list[j], col_off:col_off + p_list[l]] = Sj[l] * Gjl
                col_off += p_list[l]
            row_off += p_list[j]
        sol = safe_solve(A + lam_B * np.eye(P), rhs)
        B_list = unpack_vec_to_list(sol, p_list)

        R = Y - XB(B_list)

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

    mem_mb = (Sigma_inv.nbytes + A.nbytes) / 1e6
    return B_list, F, D, mem_mb, time.time() - t0
