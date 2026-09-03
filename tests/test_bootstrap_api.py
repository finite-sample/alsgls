"""Contract tests for ``bootstrap()`` and the pieces it is built on.

The calibration of the bootstrap is gated by the Monte Carlo studies in
``test_standard_errors.py``. These are the cheap structural checks.
"""

from __future__ import annotations

import numpy as np
import pytest

from alsgls import ALSGLS, ALSGLSSystem, BootstrapResults, als_gls, simulate_sur
from alsgls._validation import max_identified_rank
from alsgls.bootstrap import SCHEMES
from alsgls.ops import df_rescaled


@pytest.fixture(scope="module")
def fitted():
    Xs, Y, _, _ = simulate_sur(N_tr=60, N_te=5, K=6, p=3, k=2, seed=3)
    system = {f"eq{j}": (Y[:, j], Xs[j]) for j in range(6)}
    return ALSGLSSystem(system, rank=2, lam_B=0.0).fit()


def test_same_seed_same_replicates(fitted) -> None:
    a = fitted.bootstrap(B=30, seed=7)
    b = fitted.bootstrap(B=30, seed=7)
    assert np.array_equal(a.estimates, b.estimates)
    assert np.array_equal(a.tstats, b.tstats)


def test_raising_B_extends_rather_than_reshuffles(fitted) -> None:
    """Replicate b is a function of (seed, b) alone, as in simcheck.monte_carlo."""
    short = fitted.bootstrap(B=20, seed=7)
    long_ = fitted.bootstrap(B=40, seed=7)
    assert np.array_equal(short.estimates, long_.estimates[:20])


@pytest.mark.parametrize("method", SCHEMES)
def test_every_scheme_runs_and_has_the_right_shapes(fitted, method: str) -> None:
    boot = fitted.bootstrap(B=25, method=method, seed=0)
    p = fitted.params.size
    assert isinstance(boot, BootstrapResults)
    assert boot.B == 25
    assert boot.estimates.shape == (25, p)
    assert boot.tstats.shape == (25, p)
    assert boot.bse.shape == (p,)
    assert (boot.bse > 0).all()
    ci = boot.conf_int()
    assert ci.shape == (p, 2)
    assert (ci[:, 0] <= ci[:, 1]).all()
    assert ((boot.pvalues > 0) & (boot.pvalues <= 1)).all()
    assert "Bootstrap-t" in boot.summary()


def test_interval_is_studentised_not_symmetric(fitted) -> None:
    """Percentile-t uses separate lower and upper quantiles of t*, so the
    interval need not be centred on the estimate. Check the construction
    against a direct computation."""
    boot = fitted.bootstrap(B=60, seed=1)
    lo_q = np.quantile(boot.tstats, 0.025, axis=0)
    hi_q = np.quantile(boot.tstats, 0.975, axis=0)
    ci = boot.conf_int(0.05)
    assert np.allclose(ci[:, 0], boot.params - hi_q * boot.se_plugin)
    assert np.allclose(ci[:, 1], boot.params - lo_q * boot.se_plugin)


def test_pvalue_cannot_be_zero(fitted) -> None:
    """The +1 continuity adjustment: B replicates cannot certify p = 0."""
    boot = fitted.bootstrap(B=19, seed=0)
    assert boot.pvalues.min() >= 1 / 20


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"B": 0}, "positive integer"),
        ({"B": True}, "positive integer"),
        ({"B": 2.5}, "positive integer"),
        ({"method": "pairs"}, "must be one of"),
    ],
)
def test_bad_arguments_are_rejected(fitted, kwargs, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        fitted.bootstrap(seed=0, **{"B": 10, **kwargs})


@pytest.mark.parametrize("bad", [0.0, 1.0, 1.5, -0.1])
def test_conf_int_rejects_alpha_outside_unit_interval(fitted, bad) -> None:
    boot = fitted.bootstrap(B=10, seed=0)
    with pytest.raises(ValueError, match="alpha"):
        boot.conf_int(bad)


def test_sklearn_estimator_bootstrap_matches_system_bootstrap() -> None:
    Xs, Y, _, _ = simulate_sur(N_tr=60, N_te=5, K=6, p=3, k=2, seed=3)
    est = ALSGLS(rank=2, lam_B=0.0).fit(Xs, Y)
    sys_ = ALSGLSSystem(
        {f"eq{j}": (Y[:, j], Xs[j]) for j in range(6)}, rank=2, lam_B=0.0
    ).fit()
    a = est.bootstrap(B=15, seed=4)
    b = sys_.bootstrap(B=15, seed=4)
    assert np.allclose(a.estimates, b.estimates)
    assert np.allclose(a.se_plugin, b.se_plugin)


# --------------------------------------------------------------------------
# The pieces underneath.
# --------------------------------------------------------------------------


def test_df_rescale_is_the_geomean_correction_and_keeps_the_structure() -> None:
    """diag(sqrt c) (FF' + D) diag(sqrt c) == (diag(sqrt c) F)(...)' + diag(c D),
    which is what makes the correction free to apply inside Woodbury."""
    rng = np.random.default_rng(0)
    K, k, n = 5, 2, 20
    F = rng.standard_normal((K, k))
    D = 0.5 + rng.random(K)
    p_list = [3, 2, 4, 3, 1]
    Fc, Dc = df_rescaled(F, D, n, p_list)
    c = n / (n - np.asarray(p_list, float))
    expected = np.sqrt(c)[:, None] * (F @ F.T + np.diag(D)) * np.sqrt(c)[None, :]
    assert np.abs((Fc @ Fc.T + np.diag(Dc)) - expected).max() < 1e-12


def test_df_rescale_refuses_when_an_equation_has_no_residual_df() -> None:
    with pytest.raises(ValueError, match="n > p_j"):
        df_rescaled(np.zeros((2, 1)), np.ones(2), n=3, p_list=[3, 1])


def test_plugin_standard_errors_grew_by_the_df_factor() -> None:
    """The rescale multiplies each equation's SE by about sqrt(n / (n - p))."""
    Xs, Y, _, _ = simulate_sur(N_tr=30, N_te=5, K=6, p=3, k=2, seed=9)
    res = ALSGLSSystem(
        {f"eq{j}": (Y[:, j], Xs[j]) for j in range(6)}, rank=2, lam_B=0.0
    ).fit()
    from alsgls.ops import compute_XtSigmaInvX

    raw = np.sqrt(np.diag(np.linalg.inv(compute_XtSigmaInvX(Xs, res.F, res.D))))
    ratio = res.bse / raw
    # Exact only when every equation has the same p; here they do.
    assert np.allclose(ratio, np.sqrt(30 / 27), rtol=1e-6)


def test_init_D_warm_start_reaches_the_same_answer() -> None:
    Xs, Y, _, _ = simulate_sur(N_tr=60, N_te=5, K=6, p=3, k=2, seed=5)
    _, _, D0, _, i0 = als_gls(Xs, Y, k=2, lam_B=0.0)
    _, _, D1, _, i1 = als_gls(Xs, Y, k=2, lam_B=0.0, init_D=D0)
    assert abs(i0["nll_trace"][-1] - i1["nll_trace"][-1]) < 1e-6
    assert np.abs(D1 - D0).max() / D0.max() < 1e-4


@pytest.mark.parametrize(
    "bad", [np.ones(3), np.array([1.0, -1.0] * 3), np.full(6, np.nan)]
)
def test_init_D_is_validated(bad) -> None:
    Xs, Y, _, _ = simulate_sur(N_tr=40, N_te=5, K=6, p=3, k=2, seed=5)
    with pytest.raises(ValueError, match="init_D"):
        als_gls(Xs, Y, k=2, init_D=bad)


@pytest.mark.parametrize(
    ("K", "expected"), [(2, 0), (3, 1), (4, 1), (5, 2), (6, 3), (8, 4), (20, 14)]
)
def test_max_identified_rank_matches_ledermann(K: int, expected: int) -> None:
    """(K - k)^2 >= K + k, equivalently factanal df = ((K-k)^2 - K - k)/2 >= 0."""
    assert max_identified_rank(K) == expected


def test_unidentified_rank_is_refused_with_the_largest_identified_rank_named() -> None:
    Xs, Y, _, _ = simulate_sur(N_tr=40, N_te=5, K=4, p=3, k=1, seed=5)
    with pytest.raises(ValueError, match="largest identified rank at K=4 is 1"):
        als_gls(Xs, Y, k=2)


def test_auto_rank_never_exceeds_the_identified_rank() -> None:
    from alsgls.api import _auto_rank

    for K in range(3, 40):
        assert 1 <= _auto_rank(K) <= max_identified_rank(K)
