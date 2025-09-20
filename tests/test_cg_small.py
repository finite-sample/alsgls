import numpy as np
import numpy.linalg as npl

from alsgls.ops import (
    apply_siginv_to_matrix,
    stack_B_list,
    unstack_B_vec,
    XB_from_Blist,
    cg_solve,
)

RTOL = 5e-8
ATOL = 5e-9

def test_cg_matches_dense_beta_solve():
    rng = np.random.default_rng(2025)
    N, K = 20, 5
    p_list = [3, 2, 4, 1, 3]
    Xs = [rng.standard_normal((N, pj)) for pj in p_list]
    B_true = [rng.standard_normal((pj, 1)) for pj in p_list]
    Y = np.column_stack([Xs[j] @ B_true[j] for j in range(K)]) + 0.01 * rng.standard_normal((N, K))

    # A small Sigma (use dense path here just for the test)
    k = 2
    F = rng.standard_normal((K, k)) / np.sqrt(K)
    D = 0.5 + rng.random(K)
    S = F @ F.T + np.diag(D)
    S_inv = npl.inv(S)

    lam_B = 1e-3

    # ---- Build dense A = X^T S^{-1} X + lam_B I (block-by-block, correct shapes)
    blocks = [[None for _ in range(K)] for _ in range(K)]
    for j in range(K):
        Xj = Xs[j]  # N x p_j
        for ell in range(K):
            Xl = Xs[ell]  # N x p_ell
            # A_{j,ell} = X_j^T (X_ell * S_inv[ell, j])
            blocks[j][ell] = Xj.T @ (Xl * float(S_inv[ell, j]))
    A_dense = np.block(blocks)
    # add ridge
    A_dense += lam_B * np.eye(sum(p_list))

    # ---- Dense rhs: b = X^T S^{-1} y (stacked)
    b_blocks = []
    for j in range(K):
        term = np.zeros((p_list[j], 1))
        for ell in range(K):
            term += Xs[j].T @ (Y[:, [ell]] * float(S_inv[ell, j]))
        b_blocks.append(term)
    b_dense = np.concatenate(b_blocks, axis=0).ravel()

    beta_dense = npl.solve(A_dense, b_dense)

    # ---- CG operator path (matrix-free)
    def A_mv(bvec):
        B_dir = unstack_B_vec(bvec, p_list)
        M = XB_from_Blist(Xs, B_dir)          # N × K
        S_M = M @ S_inv                        # N × K
        out_blocks = [Xs[j].T @ S_M[:, [j]] for j in range(K)]
        out = np.concatenate(out_blocks, axis=0).ravel()
        return out + lam_B * bvec

    def M_pre(v):
        # Jacobi preconditioner on the dense A for this tiny test
        return v / np.maximum(np.diag(A_dense), 1e-12)

    b0 = stack_B_list([np.zeros_like(b) for b in B_true])
    beta_cg, _ = cg_solve(A_mv, b_dense, x0=b0, maxit=5000, tol=1e-10, M_pre=M_pre)

    assert np.allclose(beta_cg, beta_dense, rtol=RTOL, atol=ATOL)
