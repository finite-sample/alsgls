"""The (F, D) line search must keep making progress, not freeze after two sweeps.

The trial points the backtracking ladder proposes are ``(F + t*dF, D_mle(F + t*dF))``.
Their ``t -> 0`` limit is ``(F, D_mle(F))``, which is only the incumbent when ``D``
already sits on the ``D_mle`` manifold. The guarded scale correction moves it off,
and once ``nll(F, D_mle(F))`` exceeds ``nll(F, D)`` no step of any size can be
accepted, so ``F`` stops moving while the sweeps keep running.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import minimize

from alsgls import XB_from_Blist, als_gls, nll_per_row, simulate_sur


def _fit(K: int, k: int, seed: int, **kw):
    Xs, Y, _, _ = simulate_sur(N_tr=200, N_te=5, K=K, p=3, k=k, seed=seed)
    return Xs, Y, als_gls(Xs, Y, k=k, lam_B=0.0, **kw)


@pytest.mark.parametrize(("K", "k", "seed"), [(20, 4, 7), (12, 2, 1), (30, 5, 3)])
def test_f_step_keeps_accepting_steps(K: int, k: int, seed: int) -> None:
    """Most sweeps should move (F, D); freezing showed up as 1-3 accepts in 60."""
    _, _, (_, _, _, _, info) = _fit(K, k, seed, sweeps=60, rel_tol=0.0)
    accepted = sum(t > 0.0 for t in info["accept_t"])
    assert accepted > 0.5 * len(info["accept_t"]), (
        f"only {accepted}/{len(info['accept_t'])} sweeps moved (F, D); "
        "the line search is rejecting every candidate"
    )


@pytest.mark.parametrize(("K", "k", "seed"), [(20, 4, 7), (12, 2, 1), (30, 5, 3)])
def test_extra_sweeps_are_not_a_no_op(K: int, k: int, seed: int) -> None:
    """Running 60 sweeps must beat running 4. Freezing made them bit-identical."""
    _, _, (_, _, _, _, short) = _fit(K, k, seed, sweeps=4, rel_tol=0.0)
    _, _, (_, _, _, _, long_) = _fit(K, k, seed, sweeps=60, rel_tol=0.0)
    assert long_["nll_trace"][-1] < short["nll_trace"][-1] - 1e-6


@pytest.mark.parametrize(("K", "k", "seed"), [(20, 4, 7), (12, 2, 1), (30, 5, 3)])
def test_close_to_what_an_independent_optimiser_reaches(K: int, k: int, seed: int):
    """L-BFGS-B on the same objective, started from the ALS answer, with B held
    fixed, should have little left to take. Freezing left 5-15 nats/row."""
    Xs, Y, (B, F, D, _, info) = _fit(K, k, seed, sweeps=200, rel_tol=0.0)
    R = Y - XB_from_Blist(Xs, B)

    def obj(z: np.ndarray) -> float:
        return nll_per_row(R, z[: K * k].reshape(K, k), np.exp(z[K * k :]))

    z0 = np.concatenate([F.ravel(), np.log(np.maximum(D, 1e-12))])
    ref = minimize(
        obj, z0, method="L-BFGS-B", options={"maxiter": 20000, "maxfun": 40000}
    ).fun
    assert info["nll_trace"][-1] - ref < 3.0


@pytest.mark.parametrize(("K", "k", "seed"), [(20, 4, 7), (12, 2, 1), (30, 5, 3)])
def test_trace_stays_non_increasing(K: int, k: int, seed: int) -> None:
    """Theorem 1 still has to hold: the extra candidate must not break monotonicity."""
    _, _, (_, _, _, _, info) = _fit(K, k, seed, sweeps=60, rel_tol=0.0)
    assert np.diff(info["nll_trace"]).max() <= 1e-9


@pytest.mark.parametrize(("K", "k", "seed"), [(20, 4, 7), (12, 2, 1), (30, 5, 3)])
def test_returned_beta_is_the_gls_solution_at_the_returned_sigma(K, k, seed) -> None:
    """The sweep ends on a Sigma step, so beta would otherwise be the GLS
    solution at the *previous* sweep's Sigma. cov_params reports
    (X' Sigma^-1 X)^-1 as beta's variance, which is only its variance when the
    two agree, so als_gls refreshes beta at the final Sigma before returning."""
    Xs, Y, (B, F, D, _, _) = _fit(K, k, seed, sweeps=30, rel_tol=0.0)
    Sigma_inv = np.linalg.inv(F @ F.T + np.diag(D))
    p_list = [X.shape[1] for X in Xs]
    off = np.cumsum([0, *p_list])
    A = np.zeros((off[-1], off[-1]))
    rhs = np.zeros(off[-1])
    for j in range(K):
        for m in range(K):
            A[off[j] : off[j + 1], off[m] : off[m + 1]] = Sigma_inv[j, m] * (
                Xs[j].T @ Xs[m]
            )
        rhs[off[j] : off[j + 1]] = sum(
            Sigma_inv[j, m] * (Xs[j].T @ Y[:, m]) for m in range(K)
        )
    beta = np.concatenate([b.ravel() for b in B])
    rel = np.abs(A @ beta - rhs).max() / np.abs(rhs).max()
    assert rel < 1e-5, f"beta does not solve the GLS normal equations: {rel:.3e}"
