"""Regression tests for defects found by the correctness audit.

Each test is pinned to a contract the package states about itself, and each
one fails on the code as it stood before the accompanying fix.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from alsgls import ALSGLS, ALSGLSSystem, simulate_sur
from alsgls._validation import _sanitize_regularization_params
from alsgls.als import als_gls
from alsgls.metrics import nll_per_row
from alsgls.ops import (
    XB_from_Blist,
    compute_prediction_variance,
    grad_F_nll,
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
        B0, _, _, _, _ = als_gls(Xs, Y, k=2, sweeps=1, lam_F=0.0, lam_B=0.0)
        Bd, _, _, _, _ = als_gls(Xs, Y, k=2, sweeps=1, lam_F=1e-3, lam_B=1e-3)
        delta = max(np.abs(a - b).max() for a, b in zip(B0, Bd, strict=True))
        assert delta > 0.0, "lam=0 was silently replaced by the default"

    def test_zero_ridge_solves_the_unregularised_gls_normal_equations(self):
        """With lam_B=0 the beta-step target is X'S^-1 X b = X'S^-1 y exactly."""
        from alsgls.ops import apply_siginv_to_matrix, compute_XtSigmaInvX

        Xs, Y, _, _ = simulate_sur(N_tr=80, N_te=5, K=5, p=3, k=2, seed=3)
        B, F, D, _, _ = als_gls(Xs, Y, k=2, sweeps=20, lam_F=0.0, lam_B=0.0)

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


class TestFStepLineSearch:
    """docs/source/formal_methods.md Sec. 5-6.

    "The F-step uses gradient descent with backtracking line search";
    "Theorem 2 (Convergence to Stationary Point)".

    A backtracking line search shrinks the step until it finds an improving
    one. It may not abandon the search while a descent step is still
    available, and its first trial step must be calibrated to the problem
    rather than hard-coded to 1.
    """

    @staticmethod
    def _small_scale_problem():
        # Magnitude used by examples/real_data_fama_french.py, which converts
        # percentage returns to decimals (Y ~ 1e-2).
        Xs, Y, _, _ = simulate_sur(N_tr=200, N_te=10, K=10, p=3, k=3, seed=5)
        return Xs, Y / 100.0

    def test_f_step_is_updated_on_small_scale_data(self):
        """F must actually move; accept_t must not be identically zero."""
        Xs, Y = self._small_scale_problem()
        _, _, _, _, info = als_gls(Xs, Y, k=3, sweeps=30)
        assert max(info["accept_t"]) > 0.0, (
            f"line search accepted no step at any sweep: {info['accept_t']}"
        )

    def test_no_improving_step_remains_along_the_descent_direction(self):
        """The line search must not stop while a descent step still helps.

        Step sizes are probed relative to the scale-calibrated reference step
        ``t0 = ||F|| / ||grad_F||`` so that the check means the same thing at
        every data scale.
        """
        Xs, Y = self._small_scale_problem()
        B, F, D, _, _ = als_gls(Xs, Y, k=3, sweeps=30)
        R = Y - XB_from_Blist(Xs, B)
        Dinv, C_chol = woodbury_chol(F, D)
        dF = -grad_F_nll(R, F, D, Dinv, C_chol, 1e-3)

        t0 = np.linalg.norm(F) / np.linalg.norm(dF)
        diag_S = np.sum(R**2, axis=0) / R.shape[0]
        nll_returned = float(nll_per_row(R, F, D))

        best = nll_returned
        t = t0
        for _ in range(40):
            F_try = F + t * dF
            D_try = np.maximum(diag_S - np.sum(F_try**2, axis=1), 1e-8)
            best = min(best, float(nll_per_row(R, F_try, D_try)))
            t *= 0.5

        assert nll_returned - best < 1e-3, (
            f"a step along the same descent direction improves the NLL by "
            f"{nll_returned - best:.6f} per row, so the search stopped early"
        )

    @pytest.mark.parametrize("s", [1e-2, 1e2])
    def test_unpenalised_fit_is_scale_equivariant(self, s):
        """With no ridge, rescaling the data must rescale the fit exactly.

        The Gaussian NLL satisfies l(sY; sB, sF, s^2 D) = l(Y; B, F, D)
        + K log s, so an unpenalised solver run on ``sY`` must trace the
        rescaled iterates of the run on ``Y``.  (The ridge terms are absolute
        and therefore scale-dependent by construction, which is why this is
        asserted at lam=0.)

        ``d_floor`` used to be an absolute variance and had to be scaled by
        s**2 here to compensate.  It is now taken relative to the residual
        variance, so the caller no longer scales it -- passing a scaled value
        would floor the fit twice over.  ``scale_floor`` bounds a ratio of
        quadratic forms, which is already scale free, so it is left alone.
        """
        Xs, Y, _, _ = simulate_sur(N_tr=120, N_te=10, K=8, p=3, k=2, seed=1)
        kw = {"k": 2, "sweeps": 30, "lam_F": 0.0, "lam_B": 0.0}
        B1, F1, D1, _, _ = als_gls(Xs, Y, **kw)
        _, _, _, _, infos = als_gls(Xs, s * Y, **kw)

        R = s * Y - XB_from_Blist(Xs, [s * b for b in B1])
        ref = float(nll_per_row(R, s * F1, s * s * D1))
        got = infos["nll_trace"][-1]
        assert got == pytest.approx(ref, abs=1e-5), (
            f"at scale s={s} the solver reaches NLL {got:.8f} but the exactly "
            f"rescaled unit-scale fit gives {ref:.8f}"
        )


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
    y = np.c_[np.linspace(0, 1, n), np.linspace(1, 0, n)]

    with pytest.raises(np.linalg.LinAlgError, match="lam_B is 0"):
        als_gls([x, x], y, k=1, lam_B=0.0, lam_F=0.0, sweeps=1)

    # A positive ridge still solves it.
    als_gls([x, x], y, k=1, lam_B=1e-3, lam_F=1e-3, sweeps=1)


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
