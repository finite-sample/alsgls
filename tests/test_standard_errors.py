"""Tests for standard errors and statistical inference.

Most of this file checks shapes, signs, symmetry and positive semi-definiteness.
None of that can tell whether the inference is *right*: a covariance matrix half
its correct size is still symmetric, still positive definite, still the right
shape, and still produces intervals that contain the point estimate. The three
claims ``bse``, ``conf_int()`` and ``pvalues`` actually make -- that the reported
standard error matches the sampling spread, that the interval covers 95% of the
time, and that a 5% test rejects a true null 5% of the time -- are gated at the
bottom of this file as Monte Carlo studies, using
`simcheck <https://github.com/finite-sample/simcheck>`_ in the style of
``tests/test_econometrics.py``.

**What the plug-in is.** ``cov_params`` is ``(X' S_c^-1 X + lam I)^-1`` with
``S_c`` the estimated ``FF' + diag(D)`` after the residual degrees-of-freedom
rescale every SUR package applies. It is the variance of a GLS estimator whose
covariance is *known*. Ours is estimated from the same data, and that costs in
two ways, which Freedman and Peters (1984, JASA 79, Theorem 1) order as
``var(beta_FGLS) > var(beta_GLS) > E[reported var]``. An oracle experiment on
this file's fixture splits the measured shortfall exactly: at n = 20 the se
ratio of 0.69 is 0.77 (the plug-in's bias at S_hat; Jensen, since the formula
is concave in S) times 0.88 (the extra spread feasible GLS has over GLS at the
true S, which no formula at a fixed S_hat can see). The oracle at the true S is
calibrated at 1.03, so the linear algebra is right and its inputs are not.

**The fixture.** Four equations, three regressors each, rank 1. It used to be
rank 2, which K = 4 does not identify: the factor model would spend 11
parameters on a covariance with 10 free entries (Ledermann's bound, factanal
df = -1), the likelihood has a ridge of maxima, and every number this module
recorded before was measured on a S the data could not pin down. ``als_gls``
now refuses that geometry.

**Measured, at rank 1, 100 replicates, B = 199** (``se ratio`` is mean reported
SE over the actual spread; ``width`` is mean interval width over what a
correctly calibrated normal interval would have):

    n = 20                se ratio  coverage  size   width
    plug-in (df-rescaled)   0.83      0.875    0.105   0.83
    bootstrap-t, parametric 0.81      0.930    0.050   1.00
    bootstrap-t, residual   0.86      0.928    0.045   1.00
    bootstrap-t, wild       0.85      0.898    0.068   0.93
    3-sigma band                   [0.885, 1]  [0, 0.115]

The plug-in fails both gates. The parametric bootstrap-t interval passes both,
with size exactly at nominal and width at 1.00 -- it is calibrated and it is
not vacuous. Note the bootstrap's own ``se ratio`` is still 0.81: the bootstrap
*standard error* is biased down (Freedman and Peters measured 20-30%), because
each replicate is drawn from a S_hat that is too small. The studentised
interval does not care, because the plug-in is biased the same way inside each
replicate as in the sample, and the quantiles absorb it. That is why the
interval, not the SE, is the calibrated object (Rilstone and Veall 1996;
Fiebig and Kim 2000; Horowitz 2019).

**Why that is a property of the estimator and not of the study.** The same
replicates are also fitted by equation-by-equation OLS. Marginally, equation j's
error is ``N(0, (FF')_jj + D_j)`` and independent across rows, so textbook OLS
inference is *exact* for this data generating process at any sample size. It is
run through the identical gates, and it passes them at n = 20 where the plug-in
fails them. Whatever is wrong is not the sample size, the fixture or the gate.

Note that OLS's own ``se_ratio`` sits at about 0.985 rather than 1.000, and that
is correct: ``E[s]`` is below ``sigma`` by ``1/(4 df)`` for a chi distribution,
which is 1.5% at 17 residual degrees of freedom. It is well inside the gate, and
it is worth knowing that the gate is measuring something fine enough to see it.
"""

import functools

import numpy as np
import pytest
from scipy import stats
from simcheck import (
    MonteCarloResult,
    assert_coverage,
    assert_intervals_informative,
    assert_proportion,
    assert_se_calibrated,
    binomial_band,
)

from alsgls import ALSGLS, ALSGLSSystem
from alsgls.ops import XB_from_Blist, compute_prediction_variance, compute_XtSigmaInvX


def _random_sur(rng, N=100, K=4, p=3, k=1):
    """Generate random SUR data with known factor structure."""
    Xs = [rng.standard_normal((N, p)) for _ in range(K)]
    B = [rng.standard_normal((p, 1)) for _ in range(K)]
    F = rng.standard_normal((K, k)) / np.sqrt(K)
    D = 0.4 + 0.2 * rng.random(K)
    Z = rng.standard_normal((N, k))
    Y = (
        XB_from_Blist(Xs, B)
        + Z @ F.T
        + rng.standard_normal((N, K)) * np.sqrt(D)[None, :]
    )
    return Xs, Y, B, F, D


def _dense_XtSigmaInvX(Xs, F, D, lam_B=0.0):
    """Compute X'Σ⁻¹X by forming dense matrices (for verification)."""
    K = len(Xs)
    N = Xs[0].shape[0]
    p_list = [X.shape[1] for X in Xs]
    p_total = sum(p_list)

    Sigma = F @ F.T + np.diag(D)
    Sigma_inv = np.linalg.inv(Sigma)

    X_block = np.zeros((N * K, p_total))
    row = 0
    col = 0
    for j in range(K):
        p_j = p_list[j]
        X_block[row : row + N, col : col + p_j] = Xs[j]
        row += N
        col += p_j

    Sigma_inv_kron = np.kron(Sigma_inv, np.eye(N))
    XtSinvX = X_block.T @ Sigma_inv_kron @ X_block

    if lam_B > 0:
        XtSinvX += lam_B * np.eye(p_total)

    return XtSinvX


class TestComputeXtSigmaInvX:
    def test_matches_dense_computation(self):
        """Woodbury-based computation should match dense matrix computation."""
        rng = np.random.default_rng(42)
        Xs, _Y, _, F, D = _random_sur(rng, N=50, K=4, p=3, k=1)

        woodbury_result = compute_XtSigmaInvX(Xs, F, D, lam_B=0.0)
        dense_result = _dense_XtSigmaInvX(Xs, F, D, lam_B=0.0)

        assert np.allclose(woodbury_result, dense_result, rtol=1e-8, atol=1e-10)

    def test_matches_dense_with_regularization(self):
        """Regularization should be added correctly."""
        rng = np.random.default_rng(43)
        Xs, _Y, _, F, D = _random_sur(rng, N=50, K=4, p=3, k=1)

        lam_B = 0.1
        woodbury_result = compute_XtSigmaInvX(Xs, F, D, lam_B=lam_B)
        dense_result = _dense_XtSigmaInvX(Xs, F, D, lam_B=lam_B)

        assert np.allclose(woodbury_result, dense_result, rtol=1e-8, atol=1e-10)

    def test_is_symmetric(self):
        """X'Σ⁻¹X should be symmetric."""
        rng = np.random.default_rng(44)
        Xs, _Y, _, F, D = _random_sur(rng, N=60, K=5, p=4, k=3)

        result = compute_XtSigmaInvX(Xs, F, D, lam_B=0.0)
        assert np.allclose(result, result.T, rtol=1e-10)

    def test_is_positive_definite(self):
        """X'Σ⁻¹X should be positive definite."""
        rng = np.random.default_rng(45)
        Xs, _Y, _, F, D = _random_sur(rng, N=100, K=4, p=3, k=1)

        result = compute_XtSigmaInvX(Xs, F, D, lam_B=0.01)
        eigvals = np.linalg.eigvalsh(result)
        assert np.all(eigvals > 0)


class TestInferenceProperties:
    @pytest.fixture
    def fitted_results(self):
        """Fit a model and return results."""
        rng = np.random.default_rng(100)
        Xs, Y, _, _, _ = _random_sur(rng, N=100, K=4, p=3, k=1)
        system = {f"eq{j}": (Y[:, j], Xs[j]) for j in range(4)}
        model = ALSGLSSystem(system, rank=1, max_sweeps=12, lam_B=1e-3)
        return model.fit()

    def test_cov_params_shape(self, fitted_results):
        """Covariance matrix should be (p_total, p_total)."""
        n_params = len(fitted_results.params)
        assert fitted_results.cov_params.shape == (n_params, n_params)

    def test_cov_params_symmetric(self, fitted_results):
        """Covariance matrix should be symmetric."""
        cov = fitted_results.cov_params
        assert np.allclose(cov, cov.T, rtol=1e-10)

    def test_cov_params_psd(self, fitted_results):
        """Covariance matrix should be positive semi-definite."""
        eigvals = np.linalg.eigvalsh(fitted_results.cov_params)
        assert np.all(eigvals >= -1e-10)

    def test_bse_shape(self, fitted_results):
        """Standard errors should have same length as params."""
        assert fitted_results.bse.shape == fitted_results.params.shape

    def test_bse_positive(self, fitted_results):
        """Standard errors should be non-negative."""
        assert np.all(fitted_results.bse >= 0)

    def test_tvalues_shape(self, fitted_results):
        """t-values should have same length as params."""
        assert fitted_results.tvalues.shape == fitted_results.params.shape

    def test_pvalues_shape(self, fitted_results):
        """p-values should have same length as params."""
        assert fitted_results.pvalues.shape == fitted_results.params.shape

    def test_pvalues_range(self, fitted_results):
        """p-values should be in [0, 1]."""
        assert np.all(fitted_results.pvalues >= 0)
        assert np.all(fitted_results.pvalues <= 1)

    def test_conf_int_shape(self, fitted_results):
        """Confidence intervals should be (n_params, 2)."""
        ci = fitted_results.conf_int()
        n_params = len(fitted_results.params)
        assert ci.shape == (n_params, 2)

    def test_conf_int_bounds(self, fitted_results):
        """Lower bound should be less than upper bound."""
        ci = fitted_results.conf_int()
        assert np.all(ci[:, 0] <= ci[:, 1])

    def test_conf_int_contains_estimate(self, fitted_results):
        """Point estimate should be within confidence interval."""
        ci = fitted_results.conf_int()
        params = fitted_results.params
        assert np.all(params >= ci[:, 0])
        assert np.all(params <= ci[:, 1])

    def test_tvalue_pvalue_consistency(self, fitted_results):
        """Large |t| should correspond to small p-values."""
        large_t_mask = np.abs(fitted_results.tvalues) > 3
        if np.any(large_t_mask):
            assert np.all(fitted_results.pvalues[large_t_mask] < 0.05)

    def test_summary_returns_string(self, fitted_results):
        """summary() should return a non-empty string."""
        s = fitted_results.summary()
        assert isinstance(s, str)
        assert len(s) > 100

    def test_summary_contains_key_info(self, fitted_results):
        """summary() should contain key information."""
        s = fitted_results.summary()
        assert "Observations" in s
        assert "Equations" in s
        assert "Factor rank" in s
        assert "Log-Likelihood" in s
        assert "Coef" in s
        assert "Std Err" in s


class TestInferenceCaching:
    def test_cov_params_cached(self):
        """cov_params should be computed once and cached."""
        rng = np.random.default_rng(200)
        Xs, Y, _, _, _ = _random_sur(rng, N=80, K=4, p=2, k=1)
        system = {f"eq{j}": (Y[:, j], Xs[j]) for j in range(4)}
        model = ALSGLSSystem(system, rank=1)
        results = model.fit()

        cov1 = results.cov_params
        cov2 = results.cov_params
        assert cov1 is cov2


class TestEdgeCases:
    def test_single_equation_is_refused(self):
        """One equation cannot identify a factor covariance.

        With K = 1 the model F F' + D has two parameters for one variance, and
        the likelihood is flat along F^2 + D = const. Any F, D returned would
        be arbitrary, and a package that reports them is reporting noise.
        factanal refuses for the same reason; so does this, with a pointer to
        the estimator that is right for the job."""
        rng = np.random.default_rng(300)
        N, p = 100, 3
        X = rng.standard_normal((N, p))
        Y = X @ rng.standard_normal((p, 1)) + rng.standard_normal((N, 1))

        with pytest.raises(ValueError, match="fit each by OLS"):
            ALSGLSSystem({"eq0": (Y, X)}, rank=1, lam_B=1e-3).fit()

    def test_many_parameters(self):
        """Inference should work with many parameters."""
        rng = np.random.default_rng(301)
        N, K, p, k = 200, 10, 5, 3
        Xs, Y, _, _, _ = _random_sur(rng, N=N, K=K, p=p, k=k)

        system = {f"eq{j}": (Y[:, j], Xs[j]) for j in range(K)}
        model = ALSGLSSystem(system, rank=k, lam_B=1e-3)
        results = model.fit()

        n_params = K * p
        assert results.bse.shape == (n_params,)
        assert np.all(results.bse > 0)


class TestComputePredictionVariance:
    def test_shape(self):
        """Prediction variance should have shape (N_new, K)."""
        rng = np.random.default_rng(400)
        N, K, p, k = 50, 4, 3, 2
        Xs, _Y, _, F, D = _random_sur(rng, N=N, K=K, p=p, k=k)
        cov_params = np.eye(K * p) * 0.01

        var_pred = compute_prediction_variance(
            Xs, F, D, cov_params, include_residual=True
        )
        assert var_pred.shape == (N, K)

    def test_include_residual_increases_variance(self):
        """Including residual variance should increase prediction variance."""
        rng = np.random.default_rng(401)
        N, K, p, k = 50, 4, 3, 2
        Xs, _Y, _, F, D = _random_sur(rng, N=N, K=K, p=p, k=k)
        cov_params = np.eye(K * p) * 0.01

        var_mean = compute_prediction_variance(
            Xs, F, D, cov_params, include_residual=False
        )
        var_obs = compute_prediction_variance(
            Xs, F, D, cov_params, include_residual=True
        )

        assert np.all(var_obs > var_mean)

    def test_variance_nonnegative(self):
        """Prediction variance should always be non-negative."""
        rng = np.random.default_rng(402)
        N, K, p, k = 50, 4, 3, 2
        Xs, _Y, _, F, D = _random_sur(rng, N=N, K=K, p=p, k=k)
        cov_params = np.eye(K * p) * 0.01

        var_pred = compute_prediction_variance(
            Xs, F, D, cov_params, include_residual=True
        )
        assert np.all(var_pred >= 0)


class TestPredictionIntervalsSystemResults:
    @pytest.fixture
    def model_and_data(self):
        """Fit model and generate test data."""
        rng = np.random.default_rng(500)
        N_tr, N_te, K, p, k = 100, 30, 4, 3, 1

        Xs_tr, Y_tr, B, F, D = _random_sur(rng, N=N_tr, K=K, p=p, k=k)
        Xs_te = [rng.standard_normal((N_te, p)) for _ in range(K)]
        Z_te = rng.standard_normal((N_te, k))
        Y_te = (
            XB_from_Blist(Xs_te, B)
            + Z_te @ F.T
            + rng.standard_normal((N_te, K)) * np.sqrt(D)[None, :]
        )

        system = {f"eq{j}": (Y_tr[:, j], Xs_tr[j]) for j in range(K)}
        model = ALSGLSSystem(system, rank=k, max_sweeps=12, lam_B=1e-3)
        results = model.fit()

        return results, Xs_te, Y_te, K, N_te

    def test_get_prediction_shape(self, model_and_data):
        """get_prediction should return correct shapes."""
        results, Xs_te, _Y_te, K, N_te = model_and_data
        exog = {f"eq{j}": Xs_te[j] for j in range(K)}

        pred = results.get_prediction(exog)

        assert pred.predicted_mean.shape == (N_te, K)
        assert pred.se_mean.shape == (N_te, K)
        assert pred.se_obs.shape == (N_te, K)

    def test_prediction_intervals_wider_than_confidence(self, model_and_data):
        """Prediction intervals should be wider than confidence intervals."""
        results, Xs_te, _Y_te, K, _N_te = model_and_data
        exog = {f"eq{j}": Xs_te[j] for j in range(K)}

        pred = results.get_prediction(exog)
        ci_mean = pred.conf_int_mean(alpha=0.05)
        ci_obs = pred.conf_int_obs(alpha=0.05)

        width_mean = ci_mean[:, :, 1] - ci_mean[:, :, 0]
        width_obs = ci_obs[:, :, 1] - ci_obs[:, :, 0]

        assert np.all(width_obs > width_mean)

    def test_conf_int_shape(self, model_and_data):
        """Confidence intervals should have shape (N, K, 2)."""
        results, Xs_te, _Y_te, K, N_te = model_and_data
        exog = {f"eq{j}": Xs_te[j] for j in range(K)}

        pred = results.get_prediction(exog)
        ci_mean = pred.conf_int_mean(alpha=0.05)
        ci_obs = pred.conf_int_obs(alpha=0.05)

        assert ci_mean.shape == (N_te, K, 2)
        assert ci_obs.shape == (N_te, K, 2)

    def test_conf_int_contains_prediction(self, model_and_data):
        """Point prediction should be within confidence interval."""
        results, Xs_te, _Y_te, K, _N_te = model_and_data
        exog = {f"eq{j}": Xs_te[j] for j in range(K)}

        pred = results.get_prediction(exog)
        ci_mean = pred.conf_int_mean(alpha=0.05)

        lower = ci_mean[:, :, 0]
        upper = ci_mean[:, :, 1]
        assert np.all(pred.predicted_mean >= lower)
        assert np.all(pred.predicted_mean <= upper)

    def test_se_obs_greater_than_se_mean(self, model_and_data):
        """SE for observations should be larger than SE for mean."""
        results, Xs_te, _Y_te, K, _N_te = model_and_data
        exog = {f"eq{j}": Xs_te[j] for j in range(K)}

        pred = results.get_prediction(exog)
        assert np.all(pred.se_obs > pred.se_mean)

    def test_get_prediction_default_uses_training_data(self, model_and_data):
        """get_prediction with no args should use training data."""
        results, _Xs_te, _Y_te, K, _N_te = model_and_data

        pred = results.get_prediction()

        assert pred.predicted_mean.shape == (results.model.nobs, K)

    def test_most_observations_land_inside_the_interval(self, model_and_data):
        """A smoke check that the intervals are not absurdly narrow.

        This used to be called ``test_asymptotic_coverage`` and asserted
        ``coverage > 0.80`` for a nominal 95% interval, which it could not
        support. The rate was computed from a *single* fitted model by pooling
        ``N_te * K`` prediction points; those points are all downstream of one
        draw of one dataset, so pooling them multiplies the apparent sample size
        without adding information and makes a badly calibrated interval look
        precisely measured. It was also one-sided: an interval covering 100% of
        the time -- far too wide -- passed it.

        The real coverage study is
        ``tests/test_econometrics.py::test_prediction_intervals_cover_at_the_nominal_rate``:
        400 refits, one held-out point each, gated on a binomial band derived
        from the replicate count. It measures 0.9425 against a nominal 0.95.

        What survives here is only what a single fit can honestly support -- that
        the intervals are in the right ballpark -- and it is named for that.
        """
        results, Xs_te, Y_te, K, _N_te = model_and_data
        exog = {f"eq{j}": Xs_te[j] for j in range(K)}

        pred = results.get_prediction(exog)
        ci = pred.conf_int_obs(alpha=0.05)

        inside = ((Y_te >= ci[:, :, 0]) & (Y_te <= ci[:, :, 1])).mean()
        assert 0.80 < inside < 1.0, f"{inside:.2%} of points inside the interval"

    def test_smaller_alpha_gives_wider_intervals(self, model_and_data):
        """Smaller alpha (e.g., 0.01) should give wider intervals."""
        results, Xs_te, _Y_te, K, _N_te = model_and_data
        exog = {f"eq{j}": Xs_te[j] for j in range(K)}

        pred = results.get_prediction(exog)
        ci_95 = pred.conf_int_obs(alpha=0.05)
        ci_99 = pred.conf_int_obs(alpha=0.01)

        width_95 = ci_95[:, :, 1] - ci_95[:, :, 0]
        width_99 = ci_99[:, :, 1] - ci_99[:, :, 0]

        assert np.all(width_99 > width_95)


class TestPredictionIntervalsALSGLS:
    @pytest.fixture
    def fitted_alsgls(self):
        """Fit ALSGLS model and return with test data."""
        rng = np.random.default_rng(600)
        N_tr, N_te, K, p, k = 100, 30, 4, 3, 1

        Xs_tr, Y_tr, B, F, D = _random_sur(rng, N=N_tr, K=K, p=p, k=k)
        Xs_te = [rng.standard_normal((N_te, p)) for _ in range(K)]
        Z_te = rng.standard_normal((N_te, k))
        Y_te = (
            XB_from_Blist(Xs_te, B)
            + Z_te @ F.T
            + rng.standard_normal((N_te, K)) * np.sqrt(D)[None, :]
        )

        model = ALSGLS(rank=k, max_sweeps=12, lam_B=1e-3)
        model.fit(Xs_tr, Y_tr)

        return model, Xs_te, Y_te, K, N_te

    def test_predict_interval_shape(self, fitted_alsgls):
        """predict_interval should return correct shapes."""
        model, Xs_te, _Y_te, K, N_te = fitted_alsgls

        result = model.predict_interval(Xs_te, alpha=0.05, return_type="prediction")

        assert result["mean"].shape == (N_te, K)
        assert result["lower"].shape == (N_te, K)
        assert result["upper"].shape == (N_te, K)

    def test_predict_interval_ordering(self, fitted_alsgls):
        """Lower < mean < upper should hold."""
        model, Xs_te, _Y_te, _K, _N_te = fitted_alsgls

        result = model.predict_interval(Xs_te)

        assert np.all(result["lower"] < result["mean"])
        assert np.all(result["mean"] < result["upper"])

    def test_prediction_wider_than_confidence(self, fitted_alsgls):
        """Prediction intervals should be wider than confidence intervals."""
        model, Xs_te, _Y_te, _K, _N_te = fitted_alsgls

        pred = model.predict_interval(Xs_te, return_type="prediction")
        conf = model.predict_interval(Xs_te, return_type="confidence")

        width_pred = pred["upper"] - pred["lower"]
        width_conf = conf["upper"] - conf["lower"]

        assert np.all(width_pred > width_conf)

    def test_invalid_return_type_raises(self, fitted_alsgls):
        """Invalid return_type should raise ValueError."""
        model, Xs_te, _Y_te, _K, _N_te = fitted_alsgls

        with pytest.raises(ValueError, match="return_type"):
            model.predict_interval(Xs_te, return_type="invalid")


class TestPredictionIntervalsEdgeCases:
    def test_single_equation_is_refused(self):
        """See ``TestEdgeCases.test_single_equation_is_refused``."""
        rng = np.random.default_rng(700)
        N_tr, p = 100, 3
        X_tr = rng.standard_normal((N_tr, p))
        Y_tr = X_tr @ rng.standard_normal((p, 1)) + rng.standard_normal((N_tr, 1))
        with pytest.raises(ValueError, match="fit each by OLS"):
            ALSGLSSystem({"eq0": (Y_tr, X_tr)}, rank=1, lam_B=1e-3).fit()

    def test_single_observation(self):
        """Prediction intervals should work with single new observation."""
        rng = np.random.default_rng(701)
        N_tr, K, p, k = 100, 4, 3, 1

        Xs_tr, Y_tr, _, _, _ = _random_sur(rng, N=N_tr, K=K, p=p, k=k)
        Xs_te = [rng.standard_normal((1, p)) for _ in range(K)]

        system = {f"eq{j}": (Y_tr[:, j], Xs_tr[j]) for j in range(K)}
        model = ALSGLSSystem(system, rank=k, lam_B=1e-3)
        results = model.fit()

        exog = {f"eq{j}": Xs_te[j] for j in range(K)}
        pred = results.get_prediction(exog)

        assert pred.predicted_mean.shape == (1, K)
        ci = pred.conf_int_obs(alpha=0.05)
        assert ci.shape == (1, K, 2)


# ==========================================================================
# Monte Carlo studies of the inference the rest of this file only
# shape-checks. See the module docstring for what they measured.
# ==========================================================================

# K = 4 identifies only a single factor: Ledermann's bound (K - k)^2 >= K + k
# fails at k = 2, where the factor model would spend 11 parameters on a 4 x 4
# covariance with 10 free entries. Every number this module used to record was
# measured at k = 2, on a Sigma the data could not pin down.
MC_K, MC_P, MC_RANK = 4, 3, 1

# The sample size every other fixture in this repository uses, and the one the
# README's examples are the size of.
MC_N = 200

# Small enough that the anti-conservatism is unmistakable rather than a two-point
# shortfall a study of this size could not resolve. At n = 200 the interval
# covers about 0.942 against a nominal 0.95, which sits inside a 3-sigma band
# until roughly 2500 replicates; at n = 20 it covers 0.838, which is outside the
# band at 100. Gating the deficit where it is only just visible would buy a test
# that fails on the seed rather than on the estimator.
MC_SMALL_N = 20

# Fixed rather than taken from the simcheck tier, for the reason given at
# ``test_econometrics.py::test_a_deliberately_narrow_interval_is_caught``:
# the fast tier's 100 replicates put the coverage floor at 0.885 and the
# se-ratio's own Monte Carlo spread at 7%, which is too coarse to say anything
# about a correctly calibrated estimator, let alone to separate one from a
# 3%-off one. At 400 the floor is 0.917 and the ratio's spread is 3.5%.
MC_REPS = 400

# The first coefficient of each equation. Chosen before running anything, rather
# than by looking at which coefficients behave: a gate applied to the parameter
# that happened to pass is not a gate.
MC_TRACKED = (0, 3, 6, 9)

# The last coefficient of each equation, whose true value is exactly zero. Size
# is only meaningful under a true null, so the null has to be built into the
# truth rather than hoped for.
MC_NULLS = (2, 5, 8, 11)


def _mc_truth(n, truth_seed=0):
    """Parameters every replicate shares, drawn once and then held fixed.

    Redrawing the truth per replicate would make the study measure something
    else; see the note at the top of ``tests/test_econometrics.py``.

    Args:
        n: Rows per equation.
        truth_seed: Seed for the single draw. Named apart from the replicate
            seed on purpose -- they are different quantities, and sharing the
            name invites threading one into the other.

    Returns:
        tuple: ``(Xs, B, F, D, params)``, where ``params`` is the flattened
        coefficient vector in the order ``ALSGLSSystemResults.params`` uses.
    """
    rng = np.random.default_rng(truth_seed)
    Xs = [rng.standard_normal((n, MC_P)) for _ in range(MC_K)]
    B = [rng.standard_normal((MC_P, 1)) for _ in range(MC_K)]
    for coefficients in B:
        coefficients[-1, 0] = 0.0
    F = rng.standard_normal((MC_K, MC_RANK)) / np.sqrt(MC_K)
    D = 0.4 + 0.2 * rng.random(MC_K)
    return Xs, B, F, D, np.concatenate([b.ravel() for b in B])


def _mc_draw(rng, Xs, B, F, D):
    """One draw of the response matrix from the fixed parameters.

    Args:
        rng: Source of randomness.
        Xs: Design matrices.
        B: True coefficients.
        F: True factor loadings.
        D: True idiosyncratic variances.

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


def _ols_by_equation(Xs, Y):
    """Equation-by-equation OLS with textbook standard errors.

    The positive control. Marginally, equation j's error is normal with variance
    ``(FF')_jj + D_j`` and independent across rows, so this is exact inference
    for the data generating process above -- at any sample size, with no
    asymptotics involved. It ignores the cross-equation correlation and so throws
    away the efficiency the package exists to recover, which is not what is being
    tested here: ``se_ratio`` compares each estimator's reported standard error
    to *its own* spread, so a less efficient estimator is not penalised.

    Args:
        Xs: Design matrices.
        Y: Response matrix.

    Returns:
        tuple: ``(params, bse, pvalues)``, flattened in the same order as
        ``ALSGLSSystemResults.params``.
    """
    n = Y.shape[0]
    coefficients, errors = [], []
    for j, X in enumerate(Xs):
        gram_inverse = np.linalg.inv(X.T @ X)
        beta = gram_inverse @ (X.T @ Y[:, j])
        residual = Y[:, j] - X @ beta
        variance = float(residual @ residual) / (n - MC_P)
        coefficients.append(beta)
        errors.append(np.sqrt(variance * np.diag(gram_inverse)))
    beta = np.concatenate(coefficients)
    error = np.concatenate(errors)
    pvalues = 2.0 * stats.t.sf(np.abs(beta / error), n - MC_P)
    return beta, error, pvalues


@functools.cache
def _mc_studies(n, reps=MC_REPS, seed=0):
    """Fit both estimators on the same replicates and package the results.

    One pass, two estimators, twelve coefficients: fitting separately per gate
    would multiply the cost by twenty-eight and, worse, would stop the alsgls and
    OLS numbers from being paired on the same draws.

    Replicate ``i`` is a function of ``(seed, i)`` alone, spawned from a seed
    sequence exactly as ``simcheck.monte_carlo`` does, so raising ``reps`` adds
    replicates instead of changing the existing ones.

    Args:
        n: Rows per equation.
        reps: Number of replicates.
        seed: Seed for the replicate stream.

    Returns:
        dict: ``{"alsgls": {index: MonteCarloResult}, "ols": {...}}``.
    """
    Xs, B, F, D, truth = _mc_truth(n)
    n_params = MC_K * MC_P
    ols_quantile = stats.t.ppf(0.975, n - MC_P)

    recorded = {
        tag: {
            field: np.empty((reps, n_params))
            for field in ("estimate", "error", "lower", "upper")
        }
        for tag in ("alsgls", "ols")
    }
    for tag in recorded:
        recorded[tag]["rejected"] = np.empty((reps, n_params), dtype=bool)

    for i, child in enumerate(np.random.SeedSequence(seed).spawn(reps)):
        rng = np.random.default_rng(child)
        Y = _mc_draw(rng, Xs, B, F, D)

        system = {f"eq{j}": (Y[:, j], Xs[j]) for j in range(MC_K)}
        # lam_B=0: a ridge penalty biases the coefficients on purpose and is not
        # reflected in cov_params either, which would confound the two defects.
        fitted = ALSGLSSystem(system, rank=MC_RANK, lam_B=0.0, max_sweeps=12).fit()
        interval = fitted.conf_int(alpha=0.05)
        store = recorded["alsgls"]
        store["estimate"][i] = fitted.params
        store["error"][i] = fitted.bse
        store["lower"][i] = interval[:, 0]
        store["upper"][i] = interval[:, 1]
        store["rejected"][i] = fitted.pvalues < 0.05

        beta, error, pvalues = _ols_by_equation(Xs, Y)
        store = recorded["ols"]
        store["estimate"][i] = beta
        store["error"][i] = error
        store["lower"][i] = beta - ols_quantile * error
        store["upper"][i] = beta + ols_quantile * error
        store["rejected"][i] = pvalues < 0.05

    studies = {}
    for tag, store in recorded.items():
        studies[tag] = {
            index: MonteCarloResult(
                estimates=store["estimate"][:, index],
                standard_errors=store["error"][:, index],
                covered=(store["lower"][:, index] <= truth[index])
                & (truth[index] <= store["upper"][:, index]),
                rejected=store["rejected"][:, index],
                truth=float(truth[index]),
            )
            for index in range(n_params)
        }
    return studies


# --------------------------------------------------------------------------
# The positive control: gates that pass on inference known to be exact.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("index", MC_TRACKED)
def test_exact_ols_standard_errors_pass_the_standard_error_gate(index):
    """Textbook OLS is exact for this DGP, so the gate must accept it.

    Run at the *small* sample size, which is the whole point: this is what says
    that the failures below are the estimator rather than n = 20 being too small
    for anyone to do inference at.

    Args:
        index: Which coefficient to track.
    """
    study = _mc_studies(MC_SMALL_N)["ols"][index]
    assert_se_calibrated(study, f"OLS coefficient {index}, n={MC_SMALL_N}")


@pytest.mark.parametrize("index", MC_TRACKED)
def test_exact_ols_intervals_pass_the_coverage_gate(index):
    """The same, for coverage. Measured 0.9486 to 0.9524 over 20000 replicates.

    Args:
        index: Which coefficient to track.
    """
    study = _mc_studies(MC_SMALL_N)["ols"][index]
    assert_coverage(study, 0.95, f"OLS coefficient {index}, n={MC_SMALL_N}")


@pytest.mark.parametrize("index", MC_NULLS)
def test_exact_ols_tests_pass_the_size_gate(index):
    """And for size, at coefficients whose true value is exactly zero.

    Args:
        index: Which null coefficient to track.
    """
    study = _mc_studies(MC_SMALL_N)["ols"][index]
    assert_proportion(
        study.rejection_rate,
        study.reps,
        0.05,
        f"OLS size at coefficient {index}, n={MC_SMALL_N}",
    )


# --------------------------------------------------------------------------
# The finding: the same gates, the same replicates, the package's own numbers.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("index", MC_TRACKED)
def test_the_standard_errors_are_caught_understating_the_spread(index):
    """``bse`` is smaller than the spread the estimator actually has.

    ``cov_params`` is the covariance of a GLS estimator whose weighting matrix is
    *known*. The weighting matrix here is estimated, and none of that
    variability reaches the standard error. Measured ratio of reported to actual
    spread at n = 20, after the df rescale: 0.79 to 0.88 across the tracked
    coefficients, against 0.94 to 1.00 for exact OLS on the identical draws.
    The ceiling for *any* formula evaluated at a fixed ``S_hat`` -- the SE at
    the true ``S`` over the actual FGLS spread -- is 0.88 at this n, so the
    plug-in is now close to as good as a plug-in can be, and still short.

    Written as a required failure rather than a tolerance, because that is the
    honest encoding: the claim is false at this sample size, and if a future
    change makes ``assert_se_calibrated`` pass here, this test says so loudly
    instead of quietly starting to certify something it never checked. The
    calibrated object is ``bootstrap()``; see the tests below.

    Args:
        index: Which coefficient to track.
    """
    study = _mc_studies(MC_SMALL_N)["alsgls"][index]
    with pytest.raises(AssertionError, match="observed spread"):
        assert_se_calibrated(study, f"alsgls coefficient {index}")

    # Measured 0.79-0.88; the fixed-S_hat ceiling is 0.88. Anything above 0.92
    # would mean the plug-in is exceeding what the oracle says is possible.
    assert study.se_ratio < 0.92, (
        f"coefficient {index}: reported SE is {study.se_ratio:.3f} times the "
        "observed spread"
    )


def _mean_over(studies, indices, attr):
    return float(np.mean([getattr(studies[i], attr) for i in indices]))


def test_the_confidence_intervals_are_caught_under_covering():
    """A nominal 95% plug-in interval covers about 0.90 at n = 20.

    The consequence of the previous test, and the one a user would feel. Exact
    OLS on the same replicates covers at 0.95, so this is not a statement about
    how hard inference is at n = 20.

    Gated on the mean over the tracked coefficients rather than per
    coefficient. After the df rescale the per-coefficient coverage runs 0.875
    to 0.915 against a 3-sigma floor of 0.917 at 400 replicates, so one
    coefficient sits a hair inside the band and a per-coefficient gate would
    fail on the seed rather than on the estimator. The mean, 0.899, is not
    marginal.
    """
    studies = _mc_studies(MC_SMALL_N)["alsgls"]
    coverage = _mean_over(studies, MC_TRACKED, "coverage")
    low, _ = binomial_band(0.95, studies[MC_TRACKED[0]].reps)
    assert coverage < low, f"mean coverage {coverage:.3f} against a floor of {low:.3f}"
    assert coverage > 0.80, f"mean coverage {coverage:.3f}: worse than measured"


def test_the_p_values_are_caught_over_rejecting_a_true_null():
    """A nominal 5% plug-in test rejects a true null about 11% of the time at n = 20.

    The same defect expressed as size. Measured per null coefficient 0.09 to
    0.14 against a 3-sigma ceiling of 0.083; gated on the mean, 0.111, for the
    reason given in the coverage test.
    """
    studies = _mc_studies(MC_SMALL_N)["alsgls"]
    size = _mean_over(studies, MC_NULLS, "rejection_rate")
    _, high = binomial_band(0.05, studies[MC_NULLS[0]].reps)
    assert size > high, f"mean size {size:.3f} against a ceiling of {high:.3f}"
    assert size < 0.20, f"mean size {size:.3f}: worse than measured"


@pytest.mark.parametrize("index", MC_TRACKED)
def test_the_deficit_shrinks_as_the_sample_grows(index):
    """It is a finite-sample effect, not a wrong formula, and this pins that.

    Nuisance-parameter cost is O(1/n), so the gap must close. Measured ratio of
    reported standard error to observed spread, per tracked coefficient: 0.79 to
    0.88 at n = 20 against 0.98 to 1.01 at n = 200, and coverage 0.875 to 0.915
    against 0.93 to 0.96.

    This is also the test that would fail if somebody "fixed" the small-sample
    behaviour by inflating every standard error by a constant: the deficit would
    stop shrinking with n, because a constant does not.

    Args:
        index: Which coefficient to track.
    """
    small = _mc_studies(MC_SMALL_N)["alsgls"][index]
    large = _mc_studies(MC_N)["alsgls"][index]

    assert small.se_ratio < large.se_ratio, (small.se_ratio, large.se_ratio)
    assert small.coverage < large.coverage, (small.coverage, large.coverage)
    # The measured gap is 0.09 to 0.19 in the ratio and 0.03 to 0.06 in
    # coverage. Half the smallest is a margin against Monte Carlo noise that
    # still catches the deficit failing to close.
    assert large.se_ratio - small.se_ratio > 0.05, (small.se_ratio, large.se_ratio)
    assert large.coverage - small.coverage > 0.015, (small.coverage, large.coverage)


def test_the_deficit_survives_at_the_sample_size_this_repository_fixtures_at():
    """At n = 200 the shortfall stops being obvious and starts being silent.

    Every other fixture in this repository uses n = 200 or less. At that size the
    reported standard error is about 2% short and the interval covers 0.945
    against a nominal 0.95 -- inside a 3-sigma band at 400 replicates, and it
    takes something like 2500 replicates to resolve. Coverage is therefore
    deliberately *not* gated here in either direction: a study this size cannot
    honestly call it either way, and asserting either would be asserting the
    seed.

    What 400 replicates can support is the paired comparison. Both estimators see
    the same twelve coefficients on the same draws, so the difference between
    their se ratios is not draw noise: alsgls reports 0.979 of its own spread
    where exact OLS reports 1.007 of its own. The gap is small, one-directional,
    and exactly the size the diagnosis predicts once n is large enough for
    ``F_hat`` and ``D_hat`` to be nearly pinned down.
    """
    alsgls = _mc_studies(MC_N)["alsgls"]
    ols = _mc_studies(MC_N)["ols"]

    alsgls_ratio = float(np.mean([study.se_ratio for study in alsgls.values()]))
    ols_ratio = float(np.mean([study.se_ratio for study in ols.values()]))
    assert alsgls_ratio < ols_ratio, (alsgls_ratio, ols_ratio)

    # Still short by more than a percent, and the exact estimator is not. A gap
    # of 0.027 over twelve paired coefficients is far outside what re-drawing a
    # handful of replicates could move.
    assert ols_ratio - alsgls_ratio > 0.01, (alsgls_ratio, ols_ratio)


# --------------------------------------------------------------------------
# The fix: bootstrap-t inference passes the gates the plug-in fails.
# --------------------------------------------------------------------------

# Fewer replicates than the plug-in study, because each carries a bootstrap:
# 100 replicates x B = 99 is ~10,000 refits, about three minutes. The 3-sigma
# coverage band at 100 replicates is [0.885, 1.0], loose but not vacuous, and
# the same trade the prediction-interval study in test_econometrics.py makes.
# B = 99 rather than 999 because these gates are on coverage and size at 5%,
# where 99 is adequate (Davidson and MacKinnon's rule wants alpha (B + 1)
# integer, and 0.05 x 100 = 5), and the measured difference between B = 99 and
# B = 199 on this fixture is inside the band.
BOOT_REPS = 100
BOOT_B = 99
BOOT_METHOD = "parametric"


@functools.cache
def _mc_bootstrap_study(n, reps=BOOT_REPS, B=BOOT_B, method=BOOT_METHOD, seed=0):
    """Fit, bootstrap, and record both the plug-in and the bootstrap-t inference.

    Both are recorded on the same draws so the gates below can be *paired*:
    the difference between two estimators on identical replicates has far less
    Monte Carlo noise than either level, which is what makes 100 replicates
    enough. An earlier version gated the bootstrap on its level alone and
    passed with the bootstrap sabotaged to return the plug-in interval, because
    on these draws the plug-in happens to sit inside the loose 100-replicate
    band. Endpoints are recorded so ``assert_intervals_informative`` can rule
    out a vacuous interval.

    Args:
        n: Rows per equation.
        reps: Number of replicates.
        B: Bootstrap replicates per fit.
        method: Resampling scheme.
        seed: Seed for the replicate stream.

    Returns:
        dict: ``{"boot": {index: MonteCarloResult}, "plugin": {...}}``.
    """
    Xs, B_true, F, D, truth = _mc_truth(n)
    n_params = MC_K * MC_P
    quantile = stats.t.ppf(0.975, n * MC_K - n_params)
    rec = {
        tag: {f: np.empty((reps, n_params)) for f in ("est", "se", "lo", "hi")}
        for tag in ("boot", "plugin")
    }
    for tag in rec:
        rec[tag]["rej"] = np.empty((reps, n_params), dtype=bool)

    for i, child in enumerate(np.random.SeedSequence(seed).spawn(reps)):
        rng = np.random.default_rng(child)
        Y = _mc_draw(rng, Xs, B_true, F, D)
        system = {f"eq{j}": (Y[:, j], Xs[j]) for j in range(MC_K)}
        fitted = ALSGLSSystem(system, rank=MC_RANK, lam_B=0.0, max_sweeps=12).fit()

        r = rec["plugin"]
        r["est"][i] = fitted.params
        r["se"][i] = fitted.bse
        r["lo"][i] = fitted.params - quantile * fitted.bse
        r["hi"][i] = fitted.params + quantile * fitted.bse
        r["rej"][i] = fitted.pvalues < 0.05

        boot = fitted.bootstrap(
            B=B, method=method, seed=int(child.generate_state(1)[0])
        )
        ci = boot.conf_int(alpha=0.05)
        r = rec["boot"]
        r["est"][i] = fitted.params
        r["se"][i] = boot.bse
        r["lo"][i] = ci[:, 0]
        r["hi"][i] = ci[:, 1]
        r["rej"][i] = boot.pvalues < 0.05

    return {
        tag: {
            index: MonteCarloResult(
                estimates=r["est"][:, index],
                standard_errors=r["se"][:, index],
                covered=None,
                lowers=r["lo"][:, index],
                uppers=r["hi"][:, index],
                rejected=r["rej"][:, index],
                truth=float(truth[index]),
            )
            for index in range(n_params)
        }
        for tag, r in rec.items()
    }


def test_bootstrap_t_intervals_pass_the_coverage_gate():
    """The percentile-t interval covers at the nominal rate where the plug-in does not.

    Same draws, same n = 20. Two assertions, and the second is the one that
    discriminates. First, the bootstrap's mean coverage over the tracked
    coefficients sits inside the 3-sigma band, measured 0.93 against a floor of
    0.885 at 100 replicates. Second, and paired on identical draws, it exceeds
    the plug-in's by a margin: measured +0.055, gated at +0.02. Sabotaging the
    bootstrap to return the plug-in interval passes the first assertion on these
    draws and fails the second by construction, which is why the second exists.

    The parametric scheme was chosen because it had the best size of the three
    (0.050 against 0.045 residual and 0.068 wild) and a width ratio of 1.00.
    """
    study = _mc_bootstrap_study(MC_SMALL_N)
    boot = _mean_over(study["boot"], MC_TRACKED, "coverage")
    plugin = _mean_over(study["plugin"], MC_TRACKED, "coverage")
    low, _ = binomial_band(0.95, study["boot"][MC_TRACKED[0]].reps)
    assert boot >= low, (
        f"bootstrap-t mean coverage {boot:.3f} below the floor {low:.3f}"
    )
    assert boot - plugin > 0.02, (
        f"bootstrap-t coverage {boot:.3f} is not above the plug-in's {plugin:.3f} "
        "on the same draws"
    )


def test_bootstrap_t_tests_pass_the_size_gate():
    """A 5% bootstrap-t test rejects a true null 5% of the time at n = 20.

    Measured mean size over the null coefficients 0.05 against a 3-sigma
    ceiling of 0.115 at 100 replicates. The paired assertion is the one that
    discriminates: the plug-in rejects 0.11 on the same draws, a gap of 0.06,
    gated at 0.02. See the coverage test for why the level alone is not enough.
    """
    study = _mc_bootstrap_study(MC_SMALL_N)
    boot = _mean_over(study["boot"], MC_NULLS, "rejection_rate")
    plugin = _mean_over(study["plugin"], MC_NULLS, "rejection_rate")
    _, high = binomial_band(0.05, study["boot"][MC_NULLS[0]].reps)
    assert boot <= high, (
        f"bootstrap-t mean size {boot:.3f} above the ceiling {high:.3f}"
    )
    assert plugin - boot > 0.02, (
        f"bootstrap-t size {boot:.3f} is not below the plug-in's {plugin:.3f} on the "
        "same draws"
    )


@pytest.mark.parametrize("index", MC_TRACKED)
def test_bootstrap_t_intervals_are_not_vacuous(index):
    """Covering by being wide is not calibration.

    simcheck's ``assert_intervals_informative`` exists because a previous
    package shipped an inflation heuristic that drove the reported standard
    error to 3e7 times the estimation error while coverage stayed high. The
    percentile-t interval's mean width is 1.00 times what a correctly
    calibrated normal interval would have, so this passes with room; the gate
    is here so that it keeps passing.

    Args:
        index: Which coefficient to track.
    """
    study = _mc_bootstrap_study(MC_SMALL_N)["boot"][index]
    assert_intervals_informative(study, 0.95, f"bootstrap-t coefficient {index}")


def test_the_bootstrap_standard_error_is_not_the_calibrated_object():
    """``bse`` from the bootstrap is still short; the interval is what is calibrated.

    Freedman and Peters (1984) measured the bootstrap standard error 20-30%
    short in their SUR design, because each replicate is drawn from a
    ``Sigma_hat`` that is itself too small. Measured here 0.81 to 0.86 of the
    actual spread. This records that fact so that nobody reads ``boot.bse`` as
    a calibrated number: the studentised quantiles absorb the bias, the raw
    spread of the replicates does not.
    """
    studies = _mc_bootstrap_study(MC_SMALL_N)["boot"]
    ratio = _mean_over(studies, MC_TRACKED, "se_ratio")
    assert 0.7 < ratio < 0.95, f"bootstrap se ratio {ratio:.3f}"
