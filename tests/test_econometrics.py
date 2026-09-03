"""Does the estimator estimate the right thing, and do its intervals cover?

Two properties were untested before this file existed, both of them the reason
the package exists.

**Unbiasedness.** ``B_true`` is constructed in six test files to *generate* the
data and is never compared to what comes back out. So nothing checked that
``als_gls`` recovers the coefficients it claims to estimate. Every one of those
tests would still pass against an implementation that returned zeros of the right
shape.

**Coverage.** ``test_asymptotic_coverage`` asserted ``coverage > 0.80`` for a
nominal 95% interval, and computed that number from a *single* fitted model by
pooling ``N_te * K`` prediction points. Those points come from one draw of one
dataset, so they are heavily dependent: pooling them multiplies the apparent
sample size without adding information, and makes a badly calibrated interval
look precisely measured. It is also one-sided -- an interval covering 100% of the
time, which means it is far too wide, passed it.

Both are now Monte Carlo studies with gates derived from the replicate count, via
`simcheck <https://github.com/finite-sample/simcheck>`_.

**The truth is fixed and only the noise is redrawn.** This is not incidental. An
earlier draft of this file redrew ``B`` from ``N(0, 1)`` on every replicate and
tracked the estimation error. That silently tests the wrong thing: an estimator
shrunk entirely to zero returns an error of ``-B``, whose mean over a mean-zero
prior is zero, so it looks unbiased. Measured, ``lam_B=1e9`` -- total shrinkage --
produced a bias t statistic of -0.77 and sailed through. Holding the truth fixed
is what makes ``E[B_hat] = B`` the thing being tested.
"""

from __future__ import annotations

import numpy as np
import pytest
from simcheck import (
    Estimate,
    assert_proportion,
    assert_unbiased,
    monte_carlo,
)

from alsgls import ALSGLSSystem, als_gls
from alsgls.ops import XB_from_Blist

REPS = 300
# Rank 1: K = 4 equations identify only a single factor (Ledermann's bound).
N, K, P, RANK = 200, 4, 3, 1


def _fixed_truth(seed: int = 0):
    """The parameters every replicate below shares.

    Drawn once, then held fixed: replicates differ only in their noise, which is
    what makes ``E[B_hat] = B`` testable at all.

    Args:
        seed: Seed for the one draw.

    Returns:
        tuple: ``(Xs, B, F, D)``.
    """
    rng = np.random.default_rng(seed)
    Xs = [rng.standard_normal((N, P)) for _ in range(K)]
    B = [rng.standard_normal((P, 1)) for _ in range(K)]
    F = rng.standard_normal((K, RANK)) / np.sqrt(K)
    D = 0.4 + 0.2 * rng.random(K)
    return Xs, B, F, D


TRUTH_XS, TRUTH_B, TRUTH_F, TRUTH_D = _fixed_truth()


def _draw_y(rng: np.random.Generator, Xs, B, F, D):
    """Draw one response matrix from the fixed parameters.

    Args:
        rng: Source of randomness.
        Xs: Design matrices.
        B: True coefficients.
        F: True factor loadings.
        D: True diagonal variances.

    Returns:
        ndarray: One draw of ``Y``.
    """
    n = Xs[0].shape[0]
    Z = rng.standard_normal((n, F.shape[1]))
    return (
        XB_from_Blist(Xs, B)
        + Z @ F.T
        + rng.standard_normal((n, len(D))) * np.sqrt(D)[None, :]
    )


# --------------------------------------------------------------------------
# Unbiasedness: the property no test checked.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("equation", range(K))
def test_the_coefficients_are_unbiased(equation):
    """E[B_hat] must equal B_true, equation by equation.

    Parametrised by equation rather than averaged over all of them, because a
    bias in one equation cancelling a bias in another would pass a pooled test.

    Args:
        equation: Which equation's first coefficient to track.
    """
    target = float(TRUTH_B[equation][0, 0])

    def replicate(rng: np.random.Generator) -> Estimate:
        Y = _draw_y(rng, TRUTH_XS, TRUTH_B, TRUTH_F, TRUTH_D)
        # lam_B=0: a ridge penalty biases coefficients toward zero on purpose,
        # so leaving it on would make this test assert the wrong thing.
        B_hat, _, _, _, _ = als_gls(TRUTH_XS, Y, k=RANK, lam_B=0.0, sweeps=12)
        return Estimate(value=float(B_hat[equation][0, 0]))

    result = monte_carlo(replicate, truth=target, reps=REPS, seed=equation)
    assert_unbiased(result, f"equation {equation} coefficient 0")


def test_the_ridge_penalty_is_caught_biasing_toward_zero():
    """The guard on the guard: a biased estimator must be *caught*.

    If ``assert_unbiased`` passed on everything, the test above would be
    worthless. A large ridge penalty shrinks coefficients toward zero by
    construction, so the same machinery must flag it -- and the sign of the bias
    must point at zero, not merely be non-zero.
    """
    target = float(TRUTH_B[0][0, 0])

    def replicate(rng: np.random.Generator) -> Estimate:
        Y = _draw_y(rng, TRUTH_XS, TRUTH_B, TRUTH_F, TRUTH_D)
        B_hat, _, _, _, _ = als_gls(TRUTH_XS, Y, k=RANK, lam_B=1e6, sweeps=12)
        return Estimate(value=float(B_hat[0][0, 0]))

    result = monte_carlo(replicate, truth=target, reps=REPS, seed=99)
    with pytest.raises(AssertionError, match="Monte Carlo"):
        assert_unbiased(result, "heavily penalised coefficient")

    # Shrinkage pulls toward zero, so the estimate must sit between zero and the
    # truth -- a bias in the other direction would be a different bug.
    assert abs(result.bias) > 0.0
    assert abs(np.mean(result.estimates)) < abs(target)


def test_the_estimator_is_consistent():
    """The estimation error must shrink as the sample grows.

    Unbiasedness at one sample size does not imply the estimator converges; one
    that is unbiased but whose variance does not vanish is useless.
    """
    spreads = []
    for n in (100, 400, 1600):
        rng = np.random.default_rng(11)
        Xs = [rng.standard_normal((n, P)) for _ in range(K)]

        def replicate(rng: np.random.Generator, Xs=Xs) -> Estimate:
            Y = _draw_y(rng, Xs, TRUTH_B, TRUTH_F, TRUTH_D)
            B_hat, _, _, _, _ = als_gls(Xs, Y, k=RANK, lam_B=0.0, sweeps=12)
            return Estimate(value=float(B_hat[0][0, 0]))

        result = monte_carlo(replicate, truth=float(TRUTH_B[0][0, 0]), reps=120, seed=7)
        spreads.append(result.sampling_sd)

    assert spreads[0] > spreads[1] > spreads[2], spreads
    # Quadrupling n should roughly halve the spread. Asserting the rate, not only
    # the direction, so an estimator converging far too slowly is caught.
    assert 1.5 < spreads[0] / spreads[1] < 3.0, spreads
    assert 1.5 < spreads[1] / spreads[2] < 3.0, spreads


# --------------------------------------------------------------------------
# Coverage, one binomial per replicate.
# --------------------------------------------------------------------------


def test_prediction_intervals_cover_at_the_nominal_rate():
    """A 95% prediction interval must contain a fresh observation 95% of the time.

    One held-out point per replicate, not every point of one fit pooled together.
    The pooled version cannot distinguish a well-calibrated interval from a badly
    calibrated one, because the points it averages are all downstream of the same
    single draw.

    The target of a *prediction* interval is a fresh observation, which is
    redrawn every replicate, so the replicate records its own hit rather than
    being compared against a fixed truth.
    """
    n_train = 120

    rng_design = np.random.default_rng(5)
    Xs_tr = [rng_design.standard_normal((n_train, P)) for _ in range(K)]
    x_new = [rng_design.standard_normal((1, P)) for _ in range(K)]

    def replicate(rng: np.random.Generator) -> Estimate:
        Y_tr = _draw_y(rng, Xs_tr, TRUTH_B, TRUTH_F, TRUTH_D)
        system = {f"eq{j}": (Y_tr[:, j], Xs_tr[j]) for j in range(K)}
        results = ALSGLSSystem(system, rank=RANK, max_sweeps=12, lam_B=1e-6).fit()

        y_new = _draw_y(rng, x_new, TRUTH_B, TRUTH_F, TRUTH_D)
        pred = results.get_prediction({f"eq{j}": x_new[j] for j in range(K)})
        interval = pred.conf_int_obs(alpha=0.05)

        # Equation 0 only: the K equations of one fit share its estimated
        # covariance, so counting all of them would count one event K times.
        low = float(interval[0, 0, 0])
        high = float(interval[0, 0, 1])
        hit = low <= float(y_new[0, 0]) <= high
        return Estimate(value=float(hit))

    result = monte_carlo(replicate, truth=0.95, reps=400, seed=3)
    coverage = float(np.mean(result.estimates))
    assert_proportion(coverage, result.reps, 0.95, "prediction interval")


def test_a_deliberately_narrow_interval_is_caught():
    """The guard on the guard, for coverage.

    Halving the interval must fail. Without this, the coverage test above could
    be passing because the gate never fires.
    """
    n_train = 120
    rng_design = np.random.default_rng(5)
    Xs_tr = [rng_design.standard_normal((n_train, P)) for _ in range(K)]
    x_new = [rng_design.standard_normal((1, P)) for _ in range(K)]

    def replicate(rng: np.random.Generator) -> Estimate:
        Y_tr = _draw_y(rng, Xs_tr, TRUTH_B, TRUTH_F, TRUTH_D)
        system = {f"eq{j}": (Y_tr[:, j], Xs_tr[j]) for j in range(K)}
        results = ALSGLSSystem(system, rank=RANK, max_sweeps=12, lam_B=1e-6).fit()

        y_new = _draw_y(rng, x_new, TRUTH_B, TRUTH_F, TRUTH_D)
        pred = results.get_prediction({f"eq{j}": x_new[j] for j in range(K)})
        interval = pred.conf_int_obs(alpha=0.05)

        centre = float(pred.predicted_mean[0, 0])
        half = 0.5 * (float(interval[0, 0, 1]) - float(interval[0, 0, 0])) / 2.0
        hit = centre - half <= float(y_new[0, 0]) <= centre + half
        return Estimate(value=float(hit))

    result = monte_carlo(replicate, truth=0.95, reps=400, seed=3)
    coverage = float(np.mean(result.estimates))
    with pytest.raises(AssertionError, match="outside the 3-sigma band"):
        assert_proportion(coverage, result.reps, 0.95, "half-width interval")
