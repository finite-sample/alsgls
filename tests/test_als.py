import sys
from pathlib import Path

import numpy as np

# Ensure the package root is on the path when tests run from within the
# ``tests`` directory.
sys.path.append(str(Path(__file__).resolve().parents[1]))

from alsgls.als import als_gls
from alsgls.ops import XB_from_Blist
from alsgls.sim import simulate_gls


def test_als_shapes_and_mse_improvement():
    # Synthesize a small GLS problem
    p_list = [3, 5, 4]
    X_tr, Y_tr, X_te, Y_te = simulate_gls(40, 40, p_list, k=2, seed=0)

    # Baseline per-equation ridge OLS
    lam_B = 1e-3
    B_ols = []
    for j, X in enumerate(X_tr):
        XtX = X.T @ X + lam_B * np.eye(X.shape[1])
        Xty = X.T @ Y_tr[:, [j]]
        B_ols.append(np.linalg.solve(XtX, Xty))
    baseline_mse = np.mean((Y_te - XB_from_Blist(X_te, B_ols)) ** 2)

    # Run ALS GLS solver
    B_list, F, D, _, _ = als_gls(X_tr, Y_tr, k=2, lam_B=lam_B, sweeps=8)

    # Assert shapes
    assert len(B_list) == len(p_list)
    for j, Bj in enumerate(B_list):
        assert Bj.shape == (p_list[j], 1)
    assert F.shape == (len(p_list), 2)
    assert D.shape == (len(p_list),)

    # Compare test MSE to baseline
    final_mse = np.mean((Y_te - XB_from_Blist(X_te, B_list)) ** 2)
    assert final_mse < baseline_mse
