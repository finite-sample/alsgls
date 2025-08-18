import numpy as np
import sys
from pathlib import Path

# Ensure package root is on sys.path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from alsgls import als_gls, em_gls, simulate_sur, mse, nll_per_row


def test_als_vs_em_basic():
    """Test that ALS and EM produce similar results on a small problem."""
    rng = np.random.default_rng(1)
    N_tr, N_te, K, p, k = 30, 10, 5, 3, 2

    # Generate test data
    Xs_tr, Y_tr, Xs_te, Y_te = simulate_sur(N_tr, N_te, K, p, k, seed=1)

    # Run both algorithms
    B_als, F_als, D_als, mem_als, info_als = als_gls(
        Xs_tr, Y_tr, k=k, lam_F=1e-3, lam_B=1e-3, sweeps=6
    )
    B_em, F_em, D_em, mem_em, info_em = em_gls(
        Xs_tr, Y_tr, k=k, lam_F=1e-3, lam_B=1e-3, iters=15
    )

    # Check output shapes and finiteness
    assert F_als.shape == F_em.shape == (K, k)
    assert D_als.shape == D_em.shape == (K,)
    assert len(B_als) == len(B_em) == K
    
    for arr in [F_als, D_als, F_em, D_em]:
        assert np.isfinite(arr).all()
    for b_list in [B_als, B_em]:
        for b in b_list:
            assert np.isfinite(b).all()
    for v in [mem_als, mem_em]:
        assert np.isfinite(v) and v > 0

    # Check that both achieve similar test MSE
    from alsgls import XB_from_Blist
    Y_pred_als = XB_from_Blist(Xs_te, B_als)
    Y_pred_em = XB_from_Blist(Xs_te, B_em)
    
    mse_als = mse(Y_te, Y_pred_als)
    mse_em = mse(Y_te, Y_pred_em)
    
    assert np.isfinite(mse_als) and np.isfinite(mse_em)
    # They should be reasonably close (within 10% relative difference)
    rel_diff = abs(mse_als - mse_em) / max(mse_als, mse_em)
    assert rel_diff < 0.1

    # Check that NLL computation works
    R_als = Y_tr - XB_from_Blist(Xs_tr, B_als)
    R_em = Y_tr - XB_from_Blist(Xs_tr, B_em)

    nll_als = nll_per_row(R_als, F_als, D_als)
    nll_em = nll_per_row(R_em, F_em, D_em)
    
    assert np.isfinite(nll_als) and np.isfinite(nll_em)


def test_input_validation():
    """Test that input validation works correctly."""
    rng = np.random.default_rng(42)
    N, K, p = 10, 3, 2
    
    # Valid inputs
    Xs = [rng.normal(size=(N, p)) for _ in range(K)]
    Y = rng.normal(size=(N, K))
    
    # Should work fine
    B, F, D, mem, info = als_gls(Xs, Y, k=2)
    assert len(B) == K
    
    # Test various invalid inputs
    try:
        als_gls([], Y, k=2)  # Empty Xs
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    
    try:
        als_gls(Xs, Y.ravel(), k=2)  # 1D Y
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    
    try:
        als_gls(Xs, Y, k=0)  # Invalid k
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    
    try:
        als_gls(Xs, Y, k=2, lam_F=-1)  # Negative regularization
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_memory_estimate_scaling():
    """Test that memory estimates scale appropriately with problem size."""
    rng = np.random.default_rng(123)
    k = 3
    
    # Small problem
    N1, K1, p1 = 20, 10, 2
    Xs1 = [rng.normal(size=(N1, p1)) for _ in range(K1)]
    Y1 = rng.normal(size=(N1, K1))
    _, _, _, mem1, _ = als_gls(Xs1, Y1, k=k, sweeps=2)
    
    # Larger problem (double dimensions)
    N2, K2, p2 = 40, 20, 4
    Xs2 = [rng.normal(size=(N2, p2)) for _ in range(K2)]
    Y2 = rng.normal(size=(N2, K2))
    _, _, _, mem2, _ = als_gls(Xs2, Y2, k=k, sweeps=2)
    
    # Memory should scale up (roughly 4x since K doubled and N doubled)
    assert mem2 > mem1
    assert mem2 < 10 * mem1  # Reasonable upper bound