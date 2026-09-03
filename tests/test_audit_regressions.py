"""Regression tests for defects found by the correctness audit.

Each test is pinned to a contract the package states about itself, and each
one fails on the code as it stood before the accompanying fix.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats
from scipy.optimize import minimize

from alsgls import ALSGLS, ALSGLSSystem, simulate_sur
from alsgls._validation import _sanitize_regularization_params
from alsgls.als import als_gls
from alsgls.metrics import nll_per_row
from alsgls.ops import (
    XB_from_Blist,
    compute_prediction_variance,
    woodbury_chol,
)


class TestRegularizationIsNotOverridden:
    """``_sanitize_regularization_params`` validates that lam >= 0 is legal.

    An explicit request for lam = 0 must therefore be honoured, not replaced
    by the default.
    """

    @pytest.mark.parametrize("lam", [0.0, 0])
    def test_zero_is_preserved_by_the_validator(self, lam):
        lam_F, lam_B = _sanitize_regularization_params(lam, lam)
        assert lam_F == 0.0
        assert lam_B == 0.0

    def test_none_still_yields_the_documented_default(self):
        assert _sanitize_regularization_params(None, None) == (1e-3, 1e-3)

    def test_zero_ridge_changes_the_fit(self):
        """lam=0 must not produce the same numbers as the default lam=1e-3."""
        Xs, Y, _, _ = simulate_sur(N_tr=80, N_te=5, K=5, p=3, k=2, seed=3)
        B0, _, _, _, _ = als_gls(Xs, Y, k=2, sweeps=1, lam_B=0.0)
        Bd, _, _, _, _ = als_gls(Xs, Y, k=2, sweeps=1, lam_B=1e-3)
        delta = max(np.abs(a - b).max() for a, b in zip(B0, Bd, strict=True))
        assert delta > 0.0, "lam=0 was silently replaced by the default"

    def test_zero_ridge_solves_the_unregularised_gls_normal_equations(self):
        """With lam_B=0 the beta-step target is X'S^-1 X b = X'S^-1 y exactly."""
        from alsgls.ops import apply_siginv_to_matrix, compute_XtSigmaInvX

        Xs, Y, _, _ = simulate_sur(N_tr=80, N_te=5, K=5, p=3, k=2, seed=3)
        B, F, D, _, _ = als_gls(Xs, Y, k=2, sweeps=20, lam_B=0.0)

        A = compute_XtSigmaInvX(Xs, F, D, lam_B=0.0)
        S_y = apply_siginv_to_matrix(Y, F, D, C_chol=woodbury_chol(F, D)[1])
        rhs = np.concatenate([Xs[j].T @ S_y[:, [j]] for j in range(len(Xs))]).ravel()
        beta = np.concatenate([b.ravel() for b in B])

        resid = A @ beta - rhs
        assert np.abs(resid).max() / np.abs(rhs).max() < 1e-5, (
            "returned beta does not solve the unregularised GLS normal "
            "equations; relative residual "
            f"{np.abs(resid).max() / np.abs(rhs).max():.3e}"
        )


class TestSigmaStepReachesTheOptimum:
    """The Sigma step is the closed-form factor-analysis update.

    It replaced a steepest-descent step with a backtracking line search that
    stopped improving F after about two sweeps and left the fit nats/row short
    of the likelihood the same objective reaches from the same starting point.
    There is no step size to test any more, so what is tested is the thing the
    step size existed to achieve: that the answer is the optimum.
    """

    @staticmethod
    def _small_scale_problem():
        Xs, Y, _, _ = simulate_sur(N_tr=200, N_te=5, K=12, p=3, k=3, seed=4)
        return Xs, Y * 1e-4

    def test_sigma_step_moves_on_small_scale_data(self):
        """A fixed absolute step or floor would freeze the fit at this scale."""
        Xs, Y = self._small_scale_problem()
        _, F, _, _, info = als_gls(Xs, Y, k=3, sweeps=8)
        assert np.any(F != 0.0)
        assert info["nll_trace"][-1] < info["nll_trace"][0] - 1e-9

    def test_nothing_is_left_on_the_table_for_an_independent_optimiser(self):
        """L-BFGS-B on the same objective, from the returned answer, with beta
        held fixed. The gradient step used to leave 5 to 15 nats/row here."""
        Xs, Y, _, _ = simulate_sur(N_tr=200, N_te=5, K=20, p=3, k=4, seed=7)
        K, k = 20, 4
        B, F, D, _, info = als_gls(Xs, Y, k=k, sweeps=30, lam_B=0.0)
        R = Y - XB_from_Blist(Xs, B)

        def objective(z):
            return nll_per_row(R, z[: K * k].reshape(K, k), np.exp(z[K * k :]))

        start = np.concatenate([F.ravel(), np.log(np.maximum(D, 1e-12))])
        best = minimize(
            objective, start, method="L-BFGS-B", options={"maxiter": 20000}
        ).fun
        assert info["nll_trace"][-1] - best < 0.05

    @pytest.mark.parametrize("s", [1e-2, 1e2])
    def test_fit_is_scale_equivariant_at_the_default_penalty(self, s):
        """Under Y -> sY the fit must satisfy B -> sB, F -> sF, D -> s^2 D.

        This used to hold only at lam_B = 0, because an absolute ridge does not
        transform with the data. lam_B is now relative to the residual variance
        scale, so equivariance holds at the shipped default.

        The tolerance is 1e-3 rather than machine precision because the inner
        alternation stops on a relative likelihood decrease and takes a slightly
        different number of iterations at different scales (60/45/13/3 against
        55/41/12/2 at s = 1e-2), so the two fits stop at marginally different
        points on the same trajectory. The scaling itself is exact: var_ref and
        lam_B_eff both transform to 1.000000 of their predicted ratios.
        """
        Xs, Y, _, _ = simulate_sur(N_tr=200, N_te=5, K=12, p=3, k=3, seed=6)
        B1, F1, D1, _, _ = als_gls(Xs, Y, k=3, sweeps=20)
        B2, F2, D2, _, _ = als_gls(Xs, s * Y, k=3, sweeps=20)

        b1 = np.concatenate([b.ravel() for b in B1])
        b2 = np.concatenate([b.ravel() for b in B2]) / s
        assert np.abs(b2 - b1).max() / max(np.abs(b1).max(), 1e-30) < 1e-3
        assert np.abs(D2 / s**2 - D1).max() / np.abs(D1).max() < 1e-3
        # F is identified only up to rotation, so compare F F^T.
        g1, g2 = F1 @ F1.T, (F2 @ F2.T) / s**2
        assert np.abs(g2 - g1).max() / np.abs(g1).max() < 1e-3


class TestResidualDegreesOfFreedom:
    """Inference must use the stacked system's sample size.

    A SUR system of ``K`` equations with ``N`` rows each has ``N * K``
    observations. Subtracting a system-wide parameter count from ``N`` alone
    mixes two different sample sizes and goes negative once ``p_total > N``,
    silently clamping the t reference distribution to 1 d.o.f.
    """

    @staticmethod
    def _wide_system(K=120, N=300, p=3):
        # The README's own benchmark grid: N=300, p=3, K up to 120.
        Xs, Y, _, _ = simulate_sur(N_tr=N, N_te=5, K=K, p=p, k=3, seed=11)
        system = {f"eq{j}": (Y[:, j], Xs[j]) for j in range(K)}
        return Xs, Y, ALSGLSSystem(system, rank=3, max_sweeps=6).fit()

    def test_system_df_uses_total_observations(self):
        _, _, res = self._wide_system()
        expected = res.model.nobs * res.model.keqs - len(res.params)
        assert expected > 0
        assert res.df_resid == expected

    def test_wide_system_df_does_not_collapse(self):
        _, _, res = self._wide_system()
        assert len(res.params) > res.model.nobs, "test setup must have p_total > N"
        assert res.df_resid > 1, (
            "d.o.f. collapsed to 1; the t reference distribution is wrong"
        )

    def test_conf_int_width_matches_correct_df(self):
        _, _, res = self._wide_system()
        ci = res.conf_int(0.05)
        half = (ci[:, 1] - ci[:, 0]) / 2.0
        q = stats.t.ppf(0.975, res.model.nobs * res.model.keqs - len(res.params))
        np.testing.assert_allclose(half, q * res.bse, rtol=1e-10)

    def test_pvalues_use_correct_df(self):
        _, _, res = self._wide_system()
        q = stats.t.ppf(0.975, res.df_resid)
        # t(0.975, df=1) = 12.71 vs t(0.975, df=35640) = 1.96.
        assert q < 2.0
        np.testing.assert_allclose(
            res.pvalues, 2.0 * stats.t.sf(np.abs(res.tvalues), res.df_resid)
        )


@pytest.mark.parametrize("s", [1e-3, 1.0, 1e3])
def test_nll_trace_is_self_consistent_at_any_scale(s):
    """info['nll_trace'][-1] must equal the NLL recomputed from the output."""
    Xs, Y, _, _ = simulate_sur(N_tr=100, N_te=5, K=6, p=2, k=2, seed=4)
    B, F, D, _, info = als_gls(Xs, s * Y, k=2, sweeps=20)
    R = s * Y - XB_from_Blist(Xs, B)
    assert info["nll_trace"][-1] == pytest.approx(
        float(nll_per_row(R, F, D)), rel=1e-12, abs=1e-12
    )


def test_predict_interval_uses_stacked_df():
    """ALSGLS.predict_interval must use the same corrected d.o.f."""
    N, p, K = 300, 3, 120
    Xs, Y, _, _ = simulate_sur(N_tr=N, N_te=5, K=K, p=p, k=3, seed=11)
    est = ALSGLS(rank=3, max_sweeps=6).fit(Xs, Y)
    p_total = sum(est.n_features_in_)
    assert p_total > est.n_obs_, "test setup must have p_total > N"
    expected_df = est.n_obs_ * est.n_targets_ - p_total

    band = est.predict_interval(Xs, alpha=0.05, return_type="confidence")
    half = (band["upper"] - band["lower"]) / 2.0
    se = np.sqrt(
        np.maximum(
            compute_prediction_variance(
                Xs, est.F_, est.D_, est.cov_params_, include_residual=False
            ),
            0.0,
        )
    )
    np.testing.assert_allclose(half / se, stats.t.ppf(0.975, expected_df), rtol=1e-8)


def test_zero_ridge_on_a_singular_design_explains_itself():
    """lam_B = 0 on a rank-deficient X must say why, not emit "Singular matrix".

    lam_B was floored at 1e-3 by a falsy-zero bug, so a rank-deficient design
    was quietly ridged into solvability. Honouring lam_B = 0 exposes the
    singularity, which is correct -- but the message has to name the cause,
    and it must not silently substitute a pseudo-inverse, which would be the
    same class of quiet substitution the falsy-zero fix removed.
    """
    import numpy as np
    import pytest

    from alsgls import als_gls

    n = 10
    x = np.c_[np.ones(n), np.ones(n)]  # duplicated column: rank 1 of 2
    y = np.c_[np.linspace(0, 1, n), np.linspace(1, 0, n), np.linspace(0, 2, n)]

    with pytest.raises(np.linalg.LinAlgError, match="lam_B is 0"):
        als_gls([x, x, x], y, k=1, lam_B=0.0, sweeps=1)

    # A positive ridge still solves it.
    als_gls([x, x, x], y, k=1, lam_B=1e-3, sweeps=1)


class TestVarianceFloorIsRelative:
    """``d_floor`` is a variance, so an absolute value is not scale-equivariant.

    Under ``Y -> sY`` the true ``D`` scales as ``s**2``, so a fixed floor of
    1e-8 swallows the whole covariance once ``s`` is small: every entry lands
    on the floor and the fit stops tracking the data.  Measured on ``B``, which
    must satisfy ``B(sY) == B(Y)`` exactly, the error saturated at 2.6e-2 for
    every ``s <= 1e-4``.

    Taken relative to the residual variance scale it transforms correctly, and
    ``d_floor`` keeps a meaning the caller can reason about: "no variance below
    this fraction of a typical one".

    Known bound: below roughly ``s = 1e-6`` other absolute constants take over
    -- ``clip(D, 1e-12)`` in ops.py, the ``1e-8`` floors on the SVD threshold
    and the preconditioner in als.py -- and equivariance degrades again.
    Making those relative means threading a scale reference through the
    numerical core, which is not attempted here.
    """

    def test_fit_is_scale_invariant_over_the_practical_range(self):
        import numpy as np

        from alsgls import ALSGLS

        rng = np.random.RandomState(7)
        n, n_eq, p = 120, 5, 3
        xs = [rng.randn(n, p) for _ in range(n_eq)]
        beta = rng.randn(p, n_eq)
        y = np.column_stack([xs[k] @ beta[:, k] for k in range(n_eq)])
        y = y + 0.4 * rng.randn(n, n_eq)

        def fit_at(scale):
            m = ALSGLS(rank=2, cv_random_state=0)
            m.fit(xs, y * scale)
            return np.concatenate([np.ravel(b) for b in m.B_list_]) / scale

        base = fit_at(1.0)
        for scale in (1e-2, 1e-4):
            assert np.max(np.abs(fit_at(scale) - base)) < 1e-3, (
                f"B is not invariant at s={scale}"
            )
