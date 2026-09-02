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


@pytest.mark.parametrize("bad", [-1.0, 0.0, np.nan, np.inf])
@pytest.mark.parametrize("name", ["d_floor", "scale_floor"])
def test_non_positive_floors_are_rejected(data, name: str, bad: float) -> None:
    """d_floor <= 0 let D reach zero or go negative while woodbury_chol clipped
    it at 1e-12 internally, so the returned (F, D) described a different and
    not positive definite Sigma from the one every reported number used."""
    Xs, Y = data
    with pytest.raises(ValueError, match="positive finite"):
        als_gls(Xs, Y, k=2, **{name: bad})


def test_returned_D_is_always_positive(data) -> None:
    Xs, Y = data
    _, _, D, _, _ = als_gls(Xs, Y, k=2, d_floor=1e-8)
    assert (D > 0).all()


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"sweeps": True}, "positive integer"),
        ({"cg_maxit": True}, "positive integer"),
        ({"scale_correct": "yes"}, "must be a bool"),
        ({"rel_tol": np.inf}, "finite"),
        ({"cg_tol": np.nan}, "finite"),
    ],
)
def test_wrong_typed_solver_options_are_rejected(data, kwargs, match: str) -> None:
    Xs, Y = data
    with pytest.raises(ValueError, match=match):
        als_gls(Xs, Y, k=2, **kwargs)


@pytest.mark.parametrize("bad", [np.nan, np.inf])
@pytest.mark.parametrize("where", ["Y", "X"])
def test_non_finite_data_reports_which_input_is_bad(data, where: str, bad) -> None:
    """Previously surfaced as a bare 'SVD did not converge' from deep inside."""
    Xs, Y = data
    if where == "Y":
        Y = Y.copy()
        Y[0, 0] = bad
    else:
        Xs = [x.copy() for x in Xs]
        Xs[0][0, 0] = bad
    with pytest.raises(ValueError, match=r"NaN and .* infinite entries"):
        als_gls(Xs, Y, k=2)


def test_bic_counts_identified_parameters(data) -> None:
    """K*k loadings less the k(k-1)/2 rotations that leave F F^T fixed, plus K
    variances, plus the regression coefficients. R's factanal reports the
    complementary df = ((K-k)^2 - K - k)/2."""
    from alsgls.rank_selection import _n_params

    Xs, Y = data
    K = Y.shape[1]
    p_total = sum(X.shape[1] for X in Xs)
    for k in (1, 2, 3):
        assert _n_params(K, k, p_total) == K * k + K - k * (k - 1) // 2 + p_total
        factanal_df = ((K - k) ** 2 - K - k) / 2
        cov_params = _n_params(K, k, 0)
        assert K * (K + 1) / 2 - cov_params == factanal_df

    _, results = select_rank_bic(Xs, Y, k_candidates=[1, 2])
    for r in results:
        assert r["n_params"] == _n_params(K, r["k"], p_total)


def test_objective_trace_is_the_monotone_one(data) -> None:
    """The line search descends NLL + lam_F/2 ||F||^2, so that is what is
    guaranteed non-increasing; nll_trace stays equal to nll_per_row at the
    returned parameters so the reported likelihood is self-consistent."""
    from alsgls import XB_from_Blist, nll_per_row

    Xs, Y = data
    B, F, D, _, info = als_gls(Xs, Y, k=2, sweeps=25, lam_F=1e-2, rel_tol=0.0)
    assert np.diff(info["obj_trace"]).max() <= 1e-9
    assert info["nll_trace"][-1] == pytest.approx(
        nll_per_row(Y - XB_from_Blist(Xs, B), F, D)
    )
    assert info["obj_trace"][-1] == pytest.approx(
        info["nll_trace"][-1] + 0.5 * 1e-2 * float(np.sum(F**2))
    )
