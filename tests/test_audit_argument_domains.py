"""Public arguments with a defined domain must fail at the boundary, not silently.

Each of these returned plausible-looking output before: a fit that never moved,
or an interval whose lower bound exceeded its upper.
"""

from __future__ import annotations

import numpy as np
import pytest

from alsgls import ALSGLS, ALSGLSSystem, als_gls, select_rank_bic, simulate_sur


@pytest.fixture(scope="module")
def data():
    Xs, Y, _, _ = simulate_sur(N_tr=120, N_te=5, K=6, p=3, k=2, seed=2)
    return Xs, Y


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("name", ["lam_F", "lam_B"])
def test_non_finite_penalties_are_rejected(data, name: str, bad: float) -> None:
    """``nan < 0`` is False, so NaN used to pass the non-negativity guard and
    make every backtracked candidate NaN: F was returned at its initialization
    with no error raised."""
    Xs, Y = data
    with pytest.raises(ValueError, match="finite non-negative"):
        als_gls(Xs, Y, k=2, **{name: bad})


def test_finite_penalties_still_accepted(data) -> None:
    Xs, Y = data
    for value in (0.0, 1e-3, 5):
        als_gls(Xs, Y, k=2, sweeps=1, lam_F=value, lam_B=value)


@pytest.mark.parametrize("bad", [-0.5, 0.0, 1.0, 1.5, np.nan, np.inf])
def test_predict_interval_rejects_alpha_outside_unit_interval(data, bad) -> None:
    """alpha > 1 makes the t quantile negative, which inverted the interval."""
    Xs, Y = data
    est = ALSGLS(rank=2).fit(Xs, Y)
    with pytest.raises(ValueError, match="alpha"):
        est.predict_interval(Xs, alpha=bad)


@pytest.mark.parametrize("bad", [-0.5, 0.0, 1.0, 1.5, np.nan])
def test_system_interval_methods_reject_bad_alpha(data, bad) -> None:
    Xs, Y = data
    res = ALSGLSSystem(
        {f"e{j}": (Y[:, j], Xs[j]) for j in range(Y.shape[1])}, rank=2
    ).fit()
    for call in (
        lambda: res.conf_int(bad),
        lambda: res.summary(bad),
        lambda: res.get_prediction(alpha=bad),
        lambda: res.get_prediction().conf_int_mean(bad),
        lambda: res.get_prediction().conf_int_obs(bad),
    ):
        with pytest.raises(ValueError, match="alpha"):
            call()


def test_valid_alpha_gives_an_ordered_interval(data) -> None:
    Xs, Y = data
    est = ALSGLS(rank=2).fit(Xs, Y)
    pi = est.predict_interval(Xs, alpha=0.05)
    assert (pi["lower"] <= pi["upper"]).all()


def test_bic_matches_the_textbook_definition(data) -> None:
    """Reported BIC must be -2*loglik + p*log(N), not half of it."""
    Xs, Y = data
    N = Y.shape[0]
    _, results = select_rank_bic(Xs, Y, k_candidates=[1, 2])
    for r in results:
        assert r["bic"] == pytest.approx(2 * N * r["nll"] + r["n_params"] * np.log(N))


def test_score_is_the_negative_nll_not_the_negative_mse(data) -> None:
    """The docstring used to promise negative MSE while returning negative NLL."""
    from alsgls import mse, nll_per_row

    Xs, Y = data
    est = ALSGLS(rank=2).fit(Xs, Y)
    resid = Y - est.predict(Xs)
    assert est.score(Xs, Y) == pytest.approx(-nll_per_row(resid, est.F_, est.D_))
    assert est.score(Xs, Y) != pytest.approx(-mse(Y, est.predict(Xs)))
