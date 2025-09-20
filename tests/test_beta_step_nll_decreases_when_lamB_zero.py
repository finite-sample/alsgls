import numpy as np

def test_beta_step_nll_decreases_when_lamB_zero():
    rng = np.random.default_rng(0)
    N,K,p,k = 200, 10, 3, 2
    Xs = [rng.standard_normal((N,p)) for _ in range(K)]
    # Construct well-conditioned X stack
    Xstack = np.hstack(Xs); assert np.linalg.matrix_rank(Xstack) == p*K
    B0 = [np.zeros((p,1)) for _ in range(K)]
    F = rng.standard_normal((K,k)) / np.sqrt(K)
    D = 0.5 + rng.random(K)
    # synth data
    B_true = [rng.standard_normal((p,1)) for _ in range(K)]
    Y = np.column_stack([Xs[j] @ B_true[j] for j in range(K)]) + rng.standard_normal((N,K))*np.sqrt(D)

    # NLL before β-step with lam_B=0
    from alsgls.metrics import nll_per_row
    from alsgls.ops import woodbury_chol, apply_siginv_to_matrix, XB_from_Blist
    R0 = Y - XB_from_Blist(Xs, B0)
    nll0 = float(nll_per_row(R0, F, D))

    # one β-step (lam_B=0)
    from alsgls.ops import stack_B_list, unstack_B_vec, cg_solve
    Dinv, C_chol = woodbury_chol(F, D)
    def A_mv(bvec):
        Bdir = unstack_B_vec(bvec, [p]*K)
        M = XB_from_Blist(Xs, Bdir)
        S = apply_siginv_to_matrix(M, F, D, Dinv=Dinv, C_chol=C_chol)
        outs = [Xs[j].T @ S[:,[j]] for j in range(K)]
        return np.concatenate(outs, axis=0).ravel()  # no lam_B
    Sy = apply_siginv_to_matrix(Y, F, D, Dinv=Dinv, C_chol=C_chol)
    b = np.concatenate([Xs[j].T @ Sy[:,[j]] for j in range(K)], axis=0).ravel()

    b0 = stack_B_list(B0)
    bsol, _ = cg_solve(A_mv, b, x0=b0, maxit=5000, tol=1e-10)
    B1 = unstack_B_vec(bsol, [p]*K)
    R1 = Y - XB_from_Blist(Xs, B1)
    nll1 = float(nll_per_row(R1, F, D))
    assert nll1 < nll0 - 1e-8
