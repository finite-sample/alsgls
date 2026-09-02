"""The Sigma step must actually reach the maximum likelihood fit.

The gradient F-step this replaced stopped improving after about two sweeps and
then ran the rest as exact no-ops, leaving the fit 2 to 20 nats/row short of
what the same objective reaches from the same starting point. Nothing about the
result looked wrong: the trace was monotone and rel_tol reported convergence.

What follows are the properties that would have caught that, stated so they do
not depend on how the step is implemented.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import minimize

from alsgls import XB_from_Blist, als_gls, nll_per_row, select_rank_bic, simulate_sur


def _fit(K: int, k: int, seed: int, **kw):
    Xs, Y, _, _ = simulate_sur(N_tr=200, N_te=5, K=K, p=3, k=k, seed=seed)
    return Xs, Y, als_gls(Xs, Y, k=k, lam_B=0.0, **kw)


CASES = [(20, 4, 7), (12, 2, 1), (30, 5, 3)]


@pytest.mark.parametrize(("K", "k", "seed"), CASES)
def test_nothing_is_left_for_an_independent_optimiser(K: int, k: int, seed: int):
    """L-BFGS-B on the same objective, started from the returned answer with
    beta held fixed, should find essentially nothing. This is the check that
    exposed the original defect, where it found 5 to 15 nats/row."""
    Xs, Y, (B, F, D, _, info) = _fit(K, k, seed, sweeps=30, rel_tol=0.0)
    R = Y - XB_from_Blist(Xs, B)

    def objective(z: np.ndarray) -> float:
        return nll_per_row(R, z[: K * k].reshape(K, k), np.exp(z[K * k :]))

    start = np.concatenate([F.ravel(), np.log(np.maximum(D, 1e-12))])
    best = minimize(objective, start, method="L-BFGS-B", options={"maxiter": 20000}).fun
    assert info["nll_trace"][-1] - best < 0.05


@pytest.mark.parametrize(("K", "k", "seed"), CASES)
def test_the_default_sweep_budget_is_enough(K: int, k: int, seed: int):
    """The shipped default must reach what a large budget reaches. It used to be
    12 sweeps against a need for about 1000."""
    _, _, (_, _, _, _, default) = _fit(K, k, seed)
    _, _, (_, _, _, _, generous) = _fit(K, k, seed, sweeps=300, rel_tol=0.0)
    assert default["nll_trace"][-1] - generous["nll_trace"][-1] < 1e-3


@pytest.mark.parametrize(("K", "k", "seed"), CASES)
def test_trace_stays_non_increasing(K: int, k: int, seed: int) -> None:
    """The inner alternation is a fixed-point iteration on the joint stationarity
    conditions, not coordinate-wise maximisation, so it carries no descent
    guarantee of its own. _sigma_step returning its best iterate is what makes
    the sweep-level trace monotone, and this is the test of that."""
    _, _, (_, _, _, _, info) = _fit(K, k, seed, sweeps=40, rel_tol=0.0)
    assert np.diff(info["nll_trace"]).max() <= 1e-9


@pytest.mark.parametrize(("K", "k", "seed"), CASES)
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


# --------------------------------------------------------------------------
# The consequence the user actually sees.
# --------------------------------------------------------------------------

RANK_CASES = [(12, 2, 1), (20, 4, 7), (30, 5, 3), (16, 3, 9), (24, 6, 13)]


@pytest.mark.parametrize(("K", "true_k", "seed"), RANK_CASES)
def test_the_likelihood_is_non_increasing_in_the_rank(K, true_k, seed) -> None:
    """A rank-k model nests rank-(k-1), so its maximised likelihood cannot be
    worse. A fit that stops short breaks this, because how far short it stops
    varies with k. Measured rises of up to 2.0 nats/row before the change."""
    Xs, Y, _, _ = simulate_sur(N_tr=200, N_te=5, K=K, p=3, k=true_k, seed=seed)
    nll = np.array(
        [als_gls(Xs, Y, k=k, sweeps=30)[4]["nll_trace"][-1] for k in range(1, 9)]
    )
    assert np.diff(nll).max() <= 1e-6, f"rises by {np.diff(nll).max():.4f}"


@pytest.mark.parametrize(("K", "true_k", "seed"), RANK_CASES)
def test_bic_recovers_the_true_rank(K: int, true_k: int, seed: int) -> None:
    """The point of fitting the likelihood properly. When the fit stopped short,
    BIC chose 4, 8, 6, 7 and 10 on these five fixtures where the truth is
    2, 4, 5, 3 and 6: the shortfall shrinks as k grows, so the likelihood kept
    improving for a reason that had nothing to do with the data."""
    Xs, Y, _, _ = simulate_sur(N_tr=200, N_te=5, K=K, p=3, k=true_k, seed=seed)
    chosen, _ = select_rank_bic(Xs, Y, k_candidates=list(range(1, 11)), sweeps=30)
    assert chosen == true_k


def test_a_heywood_case_still_reaches_the_optimum() -> None:
    """One equation with no idiosyncratic variance is the documented weak spot
    of this alternation: it is known to crawl when a diagonal entry approaches
    zero, and here the inner loop does run to its 1000-iteration cap.

    Hitting the cap costs time, not accuracy. The outer sweep calls the step
    again, so the alternation resumes where it stopped, and the fit still lands
    within 1e-9 of what L-BFGS-B reaches on the same objective. This is recorded
    because the cap being reached looks alarming in ``sigma_iters`` and is not.
    """
    rng = np.random.default_rng(0)
    N, K, k = 200, 8, 2
    Y = rng.standard_normal((N, k)) @ rng.standard_normal((K, k)).T
    Y[:, 1:] += rng.standard_normal((N, K - 1)) * 0.5  # column 0 keeps zero unique var
    Xs = [np.ones((N, 1)) for _ in range(K)]

    B, F, D, _, info = als_gls(Xs, Y, k=k)
    R = Y - XB_from_Blist(Xs, B)

    def objective(z: np.ndarray) -> float:
        return nll_per_row(R, z[: K * k].reshape(K, k), np.exp(z[K * k :]))

    start = np.concatenate([F.ravel(), np.log(np.maximum(D, 1e-14))])
    best = minimize(objective, start, method="L-BFGS-B", options={"maxiter": 50000}).fun
    assert info["nll_trace"][-1] - best < 1e-9
    assert (D > 0).all()
