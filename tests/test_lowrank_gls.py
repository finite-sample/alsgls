import numpy as np
import sys
from pathlib import Path

# Ensure repository root is on the path for importing als_sim
sys.path.append(str(Path(__file__).resolve().parents[1]))

from als_sim.lowrank_gls.als_solver import als_gls, em_gls
from als_sim.lowrank_gls.numerics import penalized_nll


def test_als_vs_em_penalized_nll():
    rng = np.random.default_rng(1)
    N = 30
    K = 5
    p_list = [2] * K

    Xs = [rng.normal(size=(N, p)) for p in p_list]
    B_true = [rng.normal(size=(p, 1)) for p in p_list]
    U = rng.normal(size=(N, 2))
    F_true = rng.normal(size=(K, 2))
    D_true = rng.uniform(0.5, 1.5, size=K)
    Y = (
        np.hstack([Xs[j] @ B_true[j] for j in range(K)])
        + U @ F_true.T
        + rng.normal(scale=np.sqrt(D_true), size=(N, K))
    )

    B_als, F_als, D_als, mem_als, time_als = als_gls(
        Xs, Y, k=2, use_cg_beta=False
    )
    B_em, F_em, D_em, mem_em, time_em = em_gls(Xs, Y, k=2, iters=20)

    for arr in [F_als, D_als, F_em, D_em]:
        assert np.isfinite(arr).all()
    for b_list in [B_als, B_em]:
        for b in b_list:
            assert np.isfinite(b).all()
    for v in [mem_als, time_als, mem_em, time_em]:
        assert np.isfinite(v)

    assert F_als.shape == F_em.shape == (K, 2)
    assert D_als.shape == D_em.shape == (K,)
    assert len(B_als) == len(B_em) == K
    for b_a, b_e in zip(B_als, B_em):
        assert b_a.shape == b_e.shape

    als_nll = penalized_nll(
        Y, Xs, B_als, F_als, D_als, lam_F=1e-3, lam_B=1e-3
    )
    em_nll = penalized_nll(
        Y, Xs, B_em, F_em, D_em, lam_F=1e-3, lam_B=1e-3
    )
    assert np.isfinite(als_nll) and np.isfinite(em_nll)
    assert als_nll <= em_nll + 1e-6
