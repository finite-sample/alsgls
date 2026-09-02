import sys
from pathlib import Path

import numpy as np

# Ensure the package root is on the import path when tests are executed
sys.path.append(str(Path(__file__).resolve().parents[1]))
from alsgls import simulate_gls, simulate_sur


def test_simulate_sur_shapes_and_reproducibility():
    N_tr, N_te, K, p, k, seed = 20, 10, 3, 5, 2, 42
    X_tr, Y_tr, X_te, Y_te = simulate_sur(N_tr, N_te, K, p, k, seed=seed)

    assert len(X_tr) == K == len(X_te)
    for X in X_tr:
        assert X.shape == (N_tr, p)
    for X in X_te:
        assert X.shape == (N_te, p)
    assert Y_tr.shape == (N_tr, K)
    assert Y_te.shape == (N_te, K)

    X_tr2, Y_tr2, X_te2, Y_te2 = simulate_sur(N_tr, N_te, K, p, k, seed=seed)
    for X1, X2 in zip(X_tr, X_tr2, strict=False):
        assert np.array_equal(X1, X2)
    for X1, X2 in zip(X_te, X_te2, strict=False):
        assert np.array_equal(X1, X2)
    assert np.array_equal(Y_tr, Y_tr2)
    assert np.array_equal(Y_te, Y_te2)


def test_simulate_gls_shapes_and_reproducibility():
    N_tr, N_te, p_list, k, seed = 15, 7, [4, 3, 5], 2, 123
    X_tr, Y_tr, X_te, Y_te = simulate_gls(N_tr, N_te, p_list, k, seed=seed)
    K = len(p_list)

    assert len(X_tr) == K == len(X_te)
    for X, p in zip(X_tr, p_list, strict=False):
        assert X.shape == (N_tr, p)
    for X, p in zip(X_te, p_list, strict=False):
        assert X.shape == (N_te, p)
    assert Y_tr.shape == (N_tr, K)
    assert Y_te.shape == (N_te, K)

    X_tr2, Y_tr2, X_te2, Y_te2 = simulate_gls(N_tr, N_te, p_list, k, seed=seed)
    for X1, X2 in zip(X_tr, X_tr2, strict=False):
        assert np.array_equal(X1, X2)
    for X1, X2 in zip(X_te, X_te2, strict=False):
        assert np.array_equal(X1, X2)
    assert np.array_equal(Y_tr, Y_tr2)
    assert np.array_equal(Y_te, Y_te2)
