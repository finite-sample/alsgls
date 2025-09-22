import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from alsgls.lsqr_gls import (
    GLSLinearOperator,
    WoodburyWeight,
    make_block_design_ops,
    solve_gls_weighted,
)


@pytest.mark.parametrize("k", [0, 3])
def test_woodbury_weight_matches_dense_inverse(k):
    rng = np.random.default_rng(0)
    N, K = 4, 6
    T = rng.standard_normal((N, K))
    d = 0.2 + rng.random(K)
    F = None if k == 0 else rng.standard_normal((K, k))

    weight = WoodburyWeight(d=d, F=F)
    applied = weight.WT_apply(weight.W_apply(T))

    Sigma = np.diag(d)
    if F is not None:
        Sigma = Sigma + F @ F.T
    Sigma_inv = np.linalg.inv(Sigma)
    expected = T @ Sigma_inv

    np.testing.assert_allclose(applied, expected, rtol=1e-10, atol=1e-10)


def _build_dense_design(X_blocks, Sigma_inv):
    N = X_blocks[0].shape[0]
    K = len(X_blocks)
    P = sum(X.shape[1] for X in X_blocks)
    W = np.linalg.cholesky(Sigma_inv)

    A = np.zeros((N * K, P))
    for n in range(N):
        offset = 0
        for j, X in enumerate(X_blocks):
            width = X.shape[1]
            block_rows = np.zeros((K, width))
            block_rows[j, :] = X[n, :]
            A[n * K : (n + 1) * K, offset : offset + width] = W @ block_rows
            offset += width
    return A, W


def _stack_rhs(y, W):
    N, K = y.shape
    rhs = np.empty((N * K,), dtype=float)
    for n in range(N):
        rhs[n * K : (n + 1) * K] = W @ y[n, :]
    return rhs


@pytest.mark.parametrize("method", ["lsmr", "lsqr"])
def test_solve_gls_weighted_matches_dense(method):
    rng = np.random.default_rng(1)
    N, K = 20, 5
    p_blocks = [3, 2, 4, 1, 3]
    X_blocks = [rng.standard_normal((N, p)) for p in p_blocks]
    X_dot, X_Tdot = make_block_design_ops(X_blocks)
    beta_true = np.concatenate([rng.standard_normal(p) for p in p_blocks])
    y = X_dot(beta_true)

    d = 0.5 + rng.random(K)
    F = rng.standard_normal((K, 3)) * 0.2
    Sigma = np.diag(d) + F @ F.T
    Sigma_inv = np.linalg.inv(Sigma)

    beta_hat, info = solve_gls_weighted(X_dot, X_Tdot, y, d, F, method=method)

    assert info["method"] == method
    A_dense, W = _build_dense_design(X_blocks, Sigma_inv)
    rhs = _stack_rhs(y, W)
    beta_dense, *_ = np.linalg.lstsq(A_dense, rhs, rcond=None)

    np.testing.assert_allclose(beta_hat, beta_dense, rtol=1e-6, atol=1e-8)


def test_linear_operator_shapes():
    rng = np.random.default_rng(2)
    N, K = 5, 4
    X_blocks = [rng.standard_normal((N, 3)) for _ in range(K)]
    X_dot, X_Tdot = make_block_design_ops(X_blocks)
    W = WoodburyWeight(d=np.ones(K), F=rng.standard_normal((K, 2)))
    P = sum(X.shape[1] for X in X_blocks)

    op = GLSLinearOperator(X_dot, X_Tdot, W, N=N, K=K, P=P)
    vec = rng.standard_normal(P)
    res = op.matvec(vec)
    assert res.shape == (N * K,)

    dual = rng.standard_normal(N * K)
    res_t = op.rmatvec(dual)
    assert res_t.shape == (P,)
