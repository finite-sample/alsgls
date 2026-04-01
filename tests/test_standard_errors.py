"""Tests for standard errors and statistical inference."""

import numpy as np
import pytest

from alsgls import ALSGLS, ALSGLSSystem
from alsgls.ops import XB_from_Blist, compute_prediction_variance, compute_XtSigmaInvX


def _random_sur(rng, N=100, K=4, p=3, k=2):
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
        Xs, _Y, _, F, D = _random_sur(rng, N=50, K=4, p=3, k=2)

        woodbury_result = compute_XtSigmaInvX(Xs, F, D, lam_B=0.0)
        dense_result = _dense_XtSigmaInvX(Xs, F, D, lam_B=0.0)

        assert np.allclose(woodbury_result, dense_result, rtol=1e-8, atol=1e-10)

    def test_matches_dense_with_regularization(self):
        """Regularization should be added correctly."""
        rng = np.random.default_rng(43)
        Xs, _Y, _, F, D = _random_sur(rng, N=50, K=4, p=3, k=2)

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
        Xs, _Y, _, F, D = _random_sur(rng, N=100, K=4, p=3, k=2)

        result = compute_XtSigmaInvX(Xs, F, D, lam_B=0.01)
        eigvals = np.linalg.eigvalsh(result)
        assert np.all(eigvals > 0)


class TestInferenceProperties:
    @pytest.fixture
    def fitted_results(self):
        """Fit a model and return results."""
        rng = np.random.default_rng(100)
        Xs, Y, _, _, _ = _random_sur(rng, N=100, K=4, p=3, k=2)
        system = {f"eq{j}": (Y[:, j], Xs[j]) for j in range(4)}
        model = ALSGLSSystem(system, rank=2, max_sweeps=12, lam_B=1e-3)
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
        Xs, Y, _, _, _ = _random_sur(rng, N=80, K=3, p=2, k=2)
        system = {f"eq{j}": (Y[:, j], Xs[j]) for j in range(3)}
        model = ALSGLSSystem(system, rank=2)
        results = model.fit()

        cov1 = results.cov_params
        cov2 = results.cov_params
        assert cov1 is cov2


class TestEdgeCases:
    def test_single_equation(self):
        """Inference should work with single equation."""
        rng = np.random.default_rng(300)
        N, p = 100, 3
        X = rng.standard_normal((N, p))
        beta = rng.standard_normal((p, 1))
        Y = X @ beta + rng.standard_normal((N, 1))

        system = {"eq0": (Y, X)}
        model = ALSGLSSystem(system, rank=1, lam_B=1e-3)
        results = model.fit()

        assert results.bse.shape == (p,)
        assert np.all(results.bse > 0)
        assert results.pvalues.shape == (p,)

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
        N_tr, N_te, K, p, k = 100, 30, 4, 3, 2

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

    def test_asymptotic_coverage(self, model_and_data):
        """95% prediction intervals should have ~95% coverage."""
        results, Xs_te, Y_te, K, _N_te = model_and_data
        exog = {f"eq{j}": Xs_te[j] for j in range(K)}

        pred = results.get_prediction(exog)
        ci = pred.conf_int_obs(alpha=0.05)

        lower = ci[:, :, 0]
        upper = ci[:, :, 1]
        in_interval = (Y_te >= lower) & (Y_te <= upper)
        coverage = in_interval.mean()

        assert coverage > 0.80, f"Coverage {coverage:.2%} is too low"

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
        N_tr, N_te, K, p, k = 100, 30, 4, 3, 2

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
        """lower < mean < upper should hold."""
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
    def test_single_equation(self):
        """Prediction intervals should work with single equation."""
        rng = np.random.default_rng(700)
        N_tr, N_te, p = 100, 20, 3
        X_tr = rng.standard_normal((N_tr, p))
        beta = rng.standard_normal((p, 1))
        Y_tr = X_tr @ beta + rng.standard_normal((N_tr, 1))
        X_te = rng.standard_normal((N_te, p))

        system = {"eq0": (Y_tr, X_tr)}
        model = ALSGLSSystem(system, rank=1, lam_B=1e-3)
        results = model.fit()

        pred = results.get_prediction({"eq0": X_te})
        assert pred.predicted_mean.shape == (N_te, 1)
        assert pred.se_obs.shape == (N_te, 1)

        ci = pred.conf_int_obs(alpha=0.05)
        assert ci.shape == (N_te, 1, 2)

    def test_single_observation(self):
        """Prediction intervals should work with single new observation."""
        rng = np.random.default_rng(701)
        N_tr, K, p, k = 100, 4, 3, 2

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
