"""High-level estimator APIs for ALS-based GLS fitting."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import stats

from ._validation import max_identified_rank
from .als import als_gls
from .bootstrap import BootstrapResults, bootstrap_system
from .kackar_harville import kh_correction
from .metrics import nll_per_row
from .ops import (
    XB_from_Blist,
    compute_prediction_variance,
    compute_XtSigmaInvX,
    df_rescaled,
)
from .rank_selection import select_rank_bic, select_rank_cv


def _check_alpha(alpha: float) -> float:
    """Validate a significance level.

    ``stats.t.ppf(1 - alpha/2, df)`` is negative for ``alpha > 1``, which
    silently returns intervals whose lower bound exceeds their upper bound, and
    is NaN for ``alpha < 0``. Neither is reported as an error by the interval
    machinery, so reject them here.

    Args:
        alpha: Significance level, in ``(0, 1)``.

    Returns:
        The validated level.

    Raises:
        ValueError: If ``alpha`` is not a finite number in ``(0, 1)``.
    """
    if not isinstance(alpha, (int, float)) or isinstance(alpha, bool):
        raise ValueError(f"alpha must be a number in (0, 1), got {alpha!r}")
    if not np.isfinite(alpha) or not (0.0 < float(alpha) < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    return float(alpha)


#: Covariance estimators ``covariance()`` accepts.
COV_METHODS = ("kh", "plugin")


def _coefficient_covariance(
    Xs: list[np.ndarray],
    F: np.ndarray,
    D: np.ndarray,
    n: int,
    lam_B_eff: float,
    method: str,
) -> np.ndarray:
    """The covariance of ``beta_hat`` by either method, shared by both estimators.

    ``"plugin"`` is ``(X' Sigma_c^-1 X + lam I)^-1`` with ``Sigma_c`` the fitted
    covariance after the residual degrees-of-freedom rescale every SUR package
    applies (linearmodels ``debiased=True``, systemfit ``"geomean"``, Stata
    ``dfk``). It is the variance of a GLS estimator whose covariance is known.

    ``"kh"`` adds the Kackar-Harville term for the covariance being estimated,
    ``Phi + Lambda`` with ``Lambda = sum_ij W_ij Phi (Q_ij - P_i Phi P_j) Phi``,
    evaluated at the same rescaled ``Sigma_c``. See :mod:`alsgls.kackar_harville`
    for what it does and does not fix.

    Args:
        Xs: Design matrices.
        F: Fitted factor loadings.
        D: Fitted diagonal variances.
        n: Rows per equation.
        lam_B_eff: The effective ridge the fit charged.
        method: ``"kh"`` or ``"plugin"``.

    Returns:
        The ``(p_total, p_total)`` covariance.

    Raises:
        ValueError: If ``method`` is not one of :data:`COV_METHODS`.
    """
    if method not in COV_METHODS:
        raise ValueError(f"method must be one of {COV_METHODS}, got {method!r}")
    # The fit charged lam_B / var_ref, so the variance has to charge the same
    # thing; using the nominal lam_B here would report the variance of an
    # estimator that was never computed.
    Fc, Dc = df_rescaled(F, D, n, [X.shape[1] for X in Xs])
    if method == "plugin":
        return np.linalg.inv(compute_XtSigmaInvX(Xs, Fc, Dc, lam_B=lam_B_eff))
    Phi, Lam = kh_correction(Xs, Fc, Dc, lam_B=lam_B_eff)
    return Phi + Lam


def _auto_rank(num_equations: int) -> int:
    """Heuristic rank used when the user does not provide one."""
    if num_equations <= 0:
        raise ValueError("num_equations must be positive")
    # Cap the rank to avoid chasing noise; allow moderate growth with K, and
    # never exceed what K equations can identify.
    heuristic = max(1, min(8, int(np.ceil(num_equations / 10))))
    return max(1, min(heuristic, max_identified_rank(num_equations)))


def _asarray_2d(x: Any, *, dtype: Any = np.float64) -> np.ndarray:
    """Convert array-like input to a 2D ``numpy.ndarray``."""
    arr = x.to_numpy() if hasattr(x, "to_numpy") else np.asarray(x)
    arr = np.asarray(arr, dtype=dtype)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError("Input must be convertible to a 2D array")
    return arr


def _column_names(obj: Any, size: int) -> list[str]:
    if hasattr(obj, "columns"):
        return list(obj.columns)
    if hasattr(obj, "dtype") and getattr(obj.dtype, "names", None):
        return list(obj.dtype.names)
    return [f"x{i}" for i in range(size)]


def _eq_name(name: Any, index: int) -> str:
    return str(name) if name is not None else f"eq{index}"


class ALSGLS:
    """Scikit-learn style estimator for low-rank GLS via ALS."""

    def __init__(
        self,
        *,
        rank: int | str | None = "auto",
        rank_candidates: list[int] | None = None,
        cv_folds: int = 5,
        cv_random_state: int | None = None,
        lam_B: float = 1e-3,
        max_sweeps: int = 12,
        rel_tol: float = 1e-6,
        d_floor: float = 1e-8,
        cg_maxit: int = 800,
        cg_tol: float = 3e-7,
    ) -> None:
        """Store the estimator hyperparameters.

        Args:
            rank: Rank of the low-rank factor structure. An int fixes the
                rank; "auto" (the default, also used for None) picks a
                heuristic based on K; "bic" selects via BIC minimization;
                "cv" selects via cross-validation.
            rank_candidates: Candidate ranks for "bic" or "cv" selection.
            cv_folds: Number of folds for "cv" rank selection.
            cv_random_state: Random state for CV fold splits.
            lam_B: Ridge penalty on the coefficients, relative to the mean
                initial residual variance so the fit does not depend on the
                units of Y.
            max_sweeps: Maximum number of alternating sweeps.
            rel_tol: Relative decrease in the negative log-likelihood below
                which the sweeps stop early.
            d_floor: Floor on each diagonal variance, as a fraction of the mean
                initial residual variance.
            cg_maxit: Iteration cap for the conjugate-gradient solve.
            cg_tol: Relative residual tolerance for that solve.
        """
        self.rank = rank
        self.rank_candidates = rank_candidates
        self.cv_folds = cv_folds
        self.cv_random_state = cv_random_state
        self.lam_B = lam_B
        self.max_sweeps = max_sweeps
        self.rel_tol = rel_tol
        self.d_floor = d_floor
        self.cg_maxit = cg_maxit
        self.cg_tol = cg_tol

    # ------------------------------------------------------------------
    # Scikit-learn estimator protocol
    # ------------------------------------------------------------------
    def get_params(self, deep: bool = True) -> dict[str, Any]:  # noqa: ARG002
        """Return the estimator hyperparameters (scikit-learn protocol)."""
        return {
            "rank": self.rank,
            "rank_candidates": self.rank_candidates,
            "cv_folds": self.cv_folds,
            "cv_random_state": self.cv_random_state,
            "lam_B": self.lam_B,
            "max_sweeps": self.max_sweeps,
            "rel_tol": self.rel_tol,
            "d_floor": self.d_floor,
            "cg_maxit": self.cg_maxit,
            "cg_tol": self.cg_tol,
        }

    def set_params(self, **params: Any) -> ALSGLS:
        """Set estimator hyperparameters in place (scikit-learn protocol)."""
        for key, value in params.items():
            if not hasattr(self, key):
                raise ValueError(f"Unknown parameter {key!r}")
            setattr(self, key, value)
        return self

    # ------------------------------------------------------------------
    # Fitting / inference
    # ------------------------------------------------------------------
    def als_kwargs(self) -> dict[str, Any]:
        """The keyword arguments this estimator passes to :func:`als_gls`."""
        return {
            "lam_B": self.lam_B,
            "sweeps": self.max_sweeps,
            "d_floor": self.d_floor,
            "cg_maxit": self.cg_maxit,
            "cg_tol": self.cg_tol,
            "rel_tol": self.rel_tol,
        }

    def fit(self, Xs: Sequence[Any], Y: Any) -> ALSGLS:
        """Fit the GLS system, selecting the rank first when requested."""
        X_list = [_asarray_2d(X) for X in Xs]
        Y_arr = _asarray_2d(Y)

        N, K = Y_arr.shape
        if len(X_list) != K:
            raise ValueError(
                f"Received {len(X_list)} design matrices for {K} equations"
            )

        for j, X in enumerate(X_list):
            if X.shape[0] != N:
                raise ValueError(f"X[{j}] has {X.shape[0]} rows but Y has {N}")

        rank_selection_results: list[dict[str, Any]] | None = None

        if self.rank == "bic":
            k, rank_selection_results = select_rank_bic(
                X_list, Y_arr, k_candidates=self.rank_candidates, **self.als_kwargs()
            )
        elif self.rank == "cv":
            k, rank_selection_results = select_rank_cv(
                X_list,
                Y_arr,
                k_candidates=self.rank_candidates,
                n_folds=self.cv_folds,
                random_state=self.cv_random_state,
                **self.als_kwargs(),
            )
        elif self.rank == "auto" or self.rank is None:
            k = _auto_rank(K)
        else:
            k = int(self.rank)

        if not (1 <= k <= min(K, N)):
            raise ValueError(f"rank must be in [1, min(K={K}, N={N})]")

        B_list, F, D, mem_mb, info = als_gls(X_list, Y_arr, k=k, **self.als_kwargs())

        self.B_list_ = B_list
        self.F_ = F
        self.D_ = D
        self.mem_mb_est_ = mem_mb
        self.info_ = info
        self.n_features_in_ = tuple(X.shape[1] for X in X_list)
        self.n_targets_ = K
        self.n_obs_ = N
        self.rank_ = k
        self.rank_selection_results_ = rank_selection_results
        self.training_residuals_ = Y_arr - XB_from_Blist(X_list, B_list)
        self._training_Xs = X_list
        self.is_fitted_ = True
        return self

    def _ensure_fitted(self) -> None:
        if not getattr(self, "is_fitted_", False):
            raise RuntimeError("The estimator has not been fitted yet")

    def _resid_df(self) -> int:
        """Residual degrees of freedom of the stacked system.

        The SUR system stacks ``K`` equations of ``N`` rows each, so it has
        ``N * K`` observations, not ``N``.  Subtracting the system-wide
        parameter count from the per-equation row count instead mixes two
        different sample sizes and goes negative as soon as ``p_total > N``.
        """
        self._ensure_fitted()
        n_total = self.n_obs_ * self.n_targets_
        return max(n_total - sum(self.n_features_in_), 1)

    def predict(self, Xs: Sequence[Any]) -> np.ndarray:
        """Predict responses for new design matrices, one per equation."""
        self._ensure_fitted()
        X_list = [_asarray_2d(X) for X in Xs]
        if len(X_list) != len(self.B_list_):
            raise ValueError("Number of design matrices does not match fitted model")
        for j, (X, B) in enumerate(zip(X_list, self.B_list_, strict=False)):
            if X.shape[1] != B.shape[0]:
                raise ValueError(
                    f"X[{j}] has {X.shape[1]} columns but expected {B.shape[0]}"
                )
        return XB_from_Blist(X_list, self.B_list_)

    def score(self, Xs: Sequence[Any], Y: Any) -> float:
        """Return the negative Gaussian NLL per row of ``Y`` under the fitted Sigma.

        Higher is better, as scikit-learn's ``score`` protocol requires. This
        scores the whole fitted covariance, not just the conditional mean, so
        it is not the negative MSE; use ``alsgls.mse`` for that.
        """
        self._ensure_fitted()
        Y_arr = _asarray_2d(Y)
        if Y_arr.shape[1] != self.n_targets_:
            raise ValueError("Y has incompatible number of targets")
        preds = self.predict(Xs)
        if preds.shape != Y_arr.shape:
            raise ValueError("Predictions and Y have incompatible shapes")
        residual = Y_arr - preds
        return -float(nll_per_row(residual, self.F_, self.D_))

    def covariance(self, method: str = "kh") -> np.ndarray:
        """Covariance of the coefficients by the named method.

        ``"kh"`` (default) is the Kackar-Harville corrected covariance, which
        accounts for ``Sigma`` being estimated; ``"plugin"`` is the df-rescaled
        plug-in that linearmodels, systemfit and Stata report. Measured on the
        test suite's 4-equation, rank-1 fixture, the reported standard error is
        0.85 (plug-in) and 0.89 (kh) of the actual spread at ``n = 20``, and
        0.94 / 0.96 at ``n = 40``. Neither is calibrated at ``n = 20``; for
        that use :meth:`bootstrap`.

        Args:
            method: ``"kh"`` or ``"plugin"``.

        Returns:
            The ``(p_total, p_total)`` covariance.

        Raises:
            RuntimeError: If the training designs were not stored.
        """
        self._ensure_fitted()
        cache = self.__dict__.setdefault("_covariance_cache", {})
        if method not in cache:
            Xs = getattr(self, "_training_Xs", None)
            if Xs is None:
                raise RuntimeError(
                    "Training design matrices not stored. "
                    "covariance() requires re-fitting or use ALSGLSSystem."
                )
            cache[method] = _coefficient_covariance(
                Xs, self.F_, self.D_, self.n_obs_, self.info_["lam_B_eff"], method
            )
        return cache[method]

    @property
    def cov_params_(self) -> np.ndarray:
        """The default coefficient covariance, ``covariance("kh")``."""
        return self.covariance()

    def bootstrap(
        self, B: int = 999, method: str = "parametric", seed: int | None = None
    ) -> BootstrapResults:
        """Bootstrap the whole fit and return studentised (bootstrap-t) inference.

        Each replicate refits ``F``, ``D`` and ``beta`` on resampled data, which
        is what captures the part of the sampling variance the plug-in
        :attr:`cov_params_` cannot: that ``Sigma`` is estimated. The
        calibrated object is the percentile-t interval, not the bootstrap
        standard error; see :class:`~alsgls.bootstrap.BootstrapResults`.

        Args:
            B: Number of replicates. ``999`` makes ``0.05 * (B + 1)`` an integer,
                which is what a 95% percentile interval wants; ``199`` is enough
                if only the standard errors are of interest.
            method: ``"parametric"`` draws errors from the fitted
                ``F F' + diag(D)``; ``"wild"`` multiplies each residual row by
                a Rademacher sign; ``"residual"`` resamples whole residual rows.
                The last two make no distributional assumption.
            seed: Seed for the replicate stream; the same seed gives the same
                replicates.

        Returns:
            The bootstrap distribution and the inference built on it.
        """
        self._ensure_fitted()
        Xs = self._training_Xs
        Y = XB_from_Blist(Xs, self.B_list_) + self.training_residuals_
        estimates, tstats = bootstrap_system(
            Xs,
            Y,
            B_list=self.B_list_,
            F=self.F_,
            D=self.D_,
            k=self.rank_,
            als_kwargs=self.als_kwargs(),
            B=B,
            method=method,
            seed=seed,
        )
        params = np.concatenate([b.ravel() for b in self.B_list_])
        se = np.sqrt(np.maximum(np.diag(self.cov_params_), 0.0))
        return BootstrapResults(
            params=params,
            se_plugin=se,
            estimates=estimates,
            tstats=tstats,
            method=method,
            seed=seed,
            df=self._resid_df(),
        )

    def predict_interval(
        self,
        Xs: Sequence[Any],
        alpha: float = 0.05,
        return_type: str = "prediction",
    ) -> dict[str, np.ndarray]:
        """Compute prediction or confidence intervals for new observations.

        Args:
            Xs: List of design matrices [X_0, ..., X_{K-1}] for new observations.
            alpha: Significance level (default 0.05 for 95% intervals).
            return_type: "prediction" for prediction intervals (includes
                residual variance), "confidence" for confidence intervals
                (variance of E[y|X] only).

        Returns:
            dict: {"mean": (N, K), "lower": (N, K), "upper": (N, K)}

        Raises:
            ValueError: If ``return_type`` is neither ``"prediction"`` nor
                ``"confidence"``, or the design matrices do not match the fitted
                model in count or in number of columns.
        """
        self._ensure_fitted()
        alpha = _check_alpha(alpha)

        if return_type not in ("prediction", "confidence"):
            raise ValueError("return_type must be 'prediction' or 'confidence'")

        X_list = [_asarray_2d(X) for X in Xs]
        if len(X_list) != len(self.B_list_):
            raise ValueError("Number of design matrices does not match fitted model")
        for j, (X, B) in enumerate(zip(X_list, self.B_list_, strict=False)):
            if X.shape[1] != B.shape[0]:
                raise ValueError(
                    f"X[{j}] has {X.shape[1]} columns but expected {B.shape[0]}"
                )

        mean = XB_from_Blist(X_list, self.B_list_)
        include_residual = return_type == "prediction"
        variance = compute_prediction_variance(
            X_list,
            self.F_,
            self.D_,
            self.cov_params_,
            include_residual=include_residual,
        )

        df = self._resid_df()
        q = stats.t.ppf(1 - alpha / 2, df)
        se = np.sqrt(np.maximum(variance, 0.0))

        return {
            "mean": mean,
            "lower": mean - q * se,
            "upper": mean + q * se,
        }


@dataclass
class PredictionResults:
    """Container for prediction results with intervals."""

    predicted_mean: np.ndarray
    se_mean: np.ndarray
    se_obs: np.ndarray
    _df: int
    _alpha_default: float = 0.05

    def conf_int_mean(self, alpha: float | None = None) -> np.ndarray:
        """Confidence intervals for E[y|X].

        Args:
            alpha: Significance level (default 0.05 for 95% CI)

        Returns:
            ci: (N, K, 2) array with lower and upper bounds
        """
        alpha = self._alpha_default if alpha is None else _check_alpha(alpha)
        q = stats.t.ppf(1 - alpha / 2, self._df)
        lower = self.predicted_mean - q * self.se_mean
        upper = self.predicted_mean + q * self.se_mean
        return np.stack([lower, upper], axis=-1)

    def conf_int_obs(self, alpha: float | None = None) -> np.ndarray:
        """Prediction intervals for y|X (includes residual variance).

        Args:
            alpha: Significance level (default 0.05 for 95% CI)

        Returns:
            ci: (N, K, 2) array with lower and upper bounds
        """
        alpha = self._alpha_default if alpha is None else _check_alpha(alpha)
        q = stats.t.ppf(1 - alpha / 2, self._df)
        lower = self.predicted_mean - q * self.se_obs
        upper = self.predicted_mean + q * self.se_obs
        return np.stack([lower, upper], axis=-1)


@dataclass
class _SystemEquation:
    name: str
    y: np.ndarray
    X: np.ndarray
    column_names: list[str]


class ALSGLSSystem:
    """Statsmodels-style system container for ALS GLS fitting."""

    def __init__(
        self,
        system: Mapping[Any, tuple[Any, Any]] | Sequence[tuple[Any, tuple[Any, Any]]],
        *,
        rank: int | str | None = "auto",
        lam_B: float = 1e-3,
        max_sweeps: int = 12,
        rel_tol: float = 1e-6,
        d_floor: float = 1e-8,
        cg_maxit: int = 800,
        cg_tol: float = 3e-7,
    ) -> None:
        """Parse and validate the equation system; see the class docstring."""
        items = list(system.items()) if isinstance(system, Mapping) else list(system)
        if len(items) == 0:
            raise ValueError("system must contain at least one equation")

        equations: list[_SystemEquation] = []
        n_obs: int | None = None

        for idx, (name, (y, X)) in enumerate(items):
            y_arr = _asarray_2d(y)
            X_arr = _asarray_2d(X)
            if y_arr.shape[1] != 1:
                raise ValueError("Each equation's response must be 1D")
            y_arr = y_arr.reshape(-1, 1)
            if n_obs is None:
                n_obs = y_arr.shape[0]
            elif y_arr.shape[0] != n_obs:
                raise ValueError("All equations must share the same number of rows")
            if X_arr.shape[0] != n_obs:
                raise ValueError("Design matrix rows must match the response length")
            equations.append(
                _SystemEquation(
                    name=_eq_name(name, idx),
                    y=y_arr,
                    X=X_arr,
                    column_names=_column_names(X, X_arr.shape[1]),
                )
            )

        self._equations = equations
        self.rank = rank
        self.lam_B = lam_B
        self.max_sweeps = max_sweeps
        self.rel_tol = rel_tol
        self.d_floor = d_floor
        self.cg_maxit = cg_maxit
        self.cg_tol = cg_tol

    @property
    def equations(self) -> list[_SystemEquation]:
        """The parsed equations, in system order."""
        return self._equations

    @property
    def nobs(self) -> int:
        """Number of observations shared by every equation."""
        return self._equations[0].y.shape[0]

    @property
    def keqs(self) -> int:
        """Number of equations in the system."""
        return len(self._equations)

    def as_arrays(self) -> tuple[list[np.ndarray], np.ndarray]:
        """Return the system as (list of X_j, stacked Y) arrays."""
        Xs = [eq.X for eq in self._equations]
        Y = np.column_stack([eq.y for eq in self._equations])
        return Xs, Y

    def fit(self) -> ALSGLSSystemResults:
        """Fit the system with ALS-GLS and return a results container."""
        estimator = ALSGLS(
            rank=self.rank,
            lam_B=self.lam_B,
            max_sweeps=self.max_sweeps,
            rel_tol=self.rel_tol,
            d_floor=self.d_floor,
            cg_maxit=self.cg_maxit,
            cg_tol=self.cg_tol,
        )
        Xs, Y = self.as_arrays()
        estimator.fit(Xs, Y)
        self.estimator_ = estimator
        result = ALSGLSSystemResults(self, estimator)
        self.result_ = result
        return result


class ALSGLSSystemResults:
    """Lightweight results container mimicking ``statsmodels`` outputs."""

    def __init__(self, model: ALSGLSSystem, estimator: ALSGLS) -> None:
        """Collect fitted quantities from the model and estimator."""
        self.model = model
        self.estimator = estimator
        Xs, Y = model.as_arrays()

        flattened = [b.ravel() for b in estimator.B_list_ if b.size]
        self.params = np.concatenate(flattened) if flattened else np.empty(0)
        self.param_labels = [
            (eq.name, col) for eq in model.equations for col in eq.column_names
        ]
        self.B_list = estimator.B_list_
        self.F = estimator.F_
        self.D = estimator.D_
        self.mem_mb_est = estimator.mem_mb_est_
        self.info = estimator.info_
        self.rank = estimator.rank_

        self.fittedvalues = XB_from_Blist(Xs, self.B_list)
        self.resids = Y - self.fittedvalues
        self.nll_per_row = float(nll_per_row(self.resids, self.F, self.D))
        self.loglike = -self.nll_per_row * self.model.nobs

    def params_as_series(self):
        """Return coefficients as a pandas Series with (equation, variable) index."""
        try:
            import pandas as pd  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("pandas is required for params_as_series()") from exc

        mi = pd.MultiIndex.from_tuples(
            self.param_labels, names=["equation", "variable"]
        )
        return pd.Series(self.params, index=mi)

    def predict(
        self, exog: Mapping[Any, Any] | Sequence[Any] | None = None
    ) -> np.ndarray:
        """Predict fitted values, for the training design or new ``exog``."""
        if exog is None:
            Xs = [eq.X for eq in self.model.equations]
        else:
            if isinstance(exog, Mapping):
                items = [exog[eq.name] for eq in self.model.equations]
            else:
                items = list(exog)
            if len(items) != len(self.model.equations):
                raise ValueError("Expected design matrices for all equations")
            Xs = []
            for item, eq in zip(items, self.model.equations, strict=False):
                arr = _asarray_2d(item)
                if arr.shape[1] != eq.X.shape[1]:
                    raise ValueError("Design matrix has incompatible number of columns")
                Xs.append(arr)
        return XB_from_Blist(Xs, self.B_list)

    def summary_dict(self) -> dict[str, Any]:
        """Return the headline fit statistics as a plain dict."""
        return {
            "rank": self.rank,
            "nobs": self.model.nobs,
            "keqs": self.model.keqs,
            "mem_mb_est": self.mem_mb_est,
            "nll_per_row": self.nll_per_row,
            "loglike": self.loglike,
        }

    def covariance(self, method: str = "kh") -> np.ndarray:
        """Covariance of the coefficients by the named method.

        ``"kh"`` (default) is the Kackar-Harville corrected covariance, which
        accounts for ``Sigma`` being estimated; ``"plugin"`` is the df-rescaled
        plug-in that linearmodels, systemfit and Stata report. Measured on the
        test suite's 4-equation, rank-1 fixture, the reported standard error is
        0.85 (plug-in) and 0.89 (kh) of the actual spread at ``n = 20``, and
        0.94 / 0.96 at ``n = 40``. Neither is calibrated at ``n = 20``; for
        that use :meth:`bootstrap`.

        Args:
            method: ``"kh"`` or ``"plugin"``.

        Returns:
            The ``(p_total, p_total)`` covariance.
        """
        cache = self.__dict__.setdefault("_covariance_cache", {})
        if method not in cache:
            Xs = [eq.X for eq in self.model.equations]
            cache[method] = _coefficient_covariance(
                Xs, self.F, self.D, self.model.nobs, self.info["lam_B_eff"], method
            )
        return cache[method]

    @property
    def cov_params(self) -> np.ndarray:
        """The default coefficient covariance, ``covariance("kh")``."""
        return self.covariance()

    @property
    def bse(self) -> np.ndarray:
        """Standard errors of parameter estimates (sqrt of diagonal of cov_params)."""
        return np.sqrt(np.maximum(np.diag(self.cov_params), 0.0))

    @property
    def tvalues(self) -> np.ndarray:
        """t-statistics for H₀: β = 0."""
        se = self.bse
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(se > 0, self.params / se, 0.0)

    def _resid_df(self) -> int:
        """Residual degrees of freedom of the stacked system.

        The system stacks ``keqs`` equations of ``nobs`` rows each, so the
        total number of observations is ``nobs * keqs``.
        """
        n_total = self.model.nobs * self.model.keqs
        return max(n_total - len(self.params), 1)

    @property
    def df_resid(self) -> int:
        """Residual degrees of freedom: ``nobs * keqs - n_params``."""
        return self._resid_df()

    @property
    def pvalues(self) -> np.ndarray:
        """Two-sided p-values for H₀: β = 0."""
        return 2.0 * stats.t.sf(np.abs(self.tvalues), self._resid_df())

    def conf_int(self, alpha: float = 0.05) -> np.ndarray:
        """Confidence intervals for parameters.

        Args:
            alpha: Significance level (default 0.05 for 95% CI)

        Returns:
            ci: (n_params, 2) array with lower and upper bounds
        """
        alpha = _check_alpha(alpha)
        q = stats.t.ppf(1 - alpha / 2, self._resid_df())
        se = self.bse
        lower = self.params - q * se
        upper = self.params + q * se
        return np.column_stack([lower, upper])

    def get_prediction(
        self,
        exog: Mapping[Any, Any] | Sequence[Any] | None = None,
        alpha: float = 0.05,
    ) -> PredictionResults:
        """Get prediction results with standard errors and intervals.

        Args:
            exog: Design matrices for new observations. If None, uses the
                training data. Can be a dict {eq_name: X_j} or a list
                [X_0, ..., X_{K-1}].
            alpha: Default significance level for intervals (default 0.05).

        Returns:
            PredictionResults: Object with predicted_mean, se_mean, se_obs,
            and interval methods.

        Raises:
            ValueError: If design matrices are not supplied for every equation, or
                one has the wrong number of columns.
        """
        if exog is None:
            Xs = [eq.X for eq in self.model.equations]
        else:
            if isinstance(exog, Mapping):
                items = [exog[eq.name] for eq in self.model.equations]
            else:
                items = list(exog)
            if len(items) != len(self.model.equations):
                raise ValueError("Expected design matrices for all equations")
            Xs = []
            for item, eq in zip(items, self.model.equations, strict=False):
                arr = _asarray_2d(item)
                if arr.shape[1] != eq.X.shape[1]:
                    raise ValueError("Design matrix has incompatible number of columns")
                Xs.append(arr)

        alpha = _check_alpha(alpha)
        predicted_mean = XB_from_Blist(Xs, self.B_list)

        var_mean = compute_prediction_variance(
            Xs, self.F, self.D, self.cov_params, include_residual=False
        )
        var_obs = compute_prediction_variance(
            Xs, self.F, self.D, self.cov_params, include_residual=True
        )

        se_mean = np.sqrt(np.maximum(var_mean, 0.0))
        se_obs = np.sqrt(np.maximum(var_obs, 0.0))

        df = self._resid_df()

        return PredictionResults(
            predicted_mean=predicted_mean,
            se_mean=se_mean,
            se_obs=se_obs,
            _df=df,
            _alpha_default=alpha,
        )

    def bootstrap(
        self, B: int = 999, method: str = "parametric", seed: int | None = None
    ) -> BootstrapResults:
        """Bootstrap the whole fit and return studentised (bootstrap-t) inference.

        Each replicate refits ``F``, ``D`` and ``beta`` on resampled data, which
        is what captures the part of the sampling variance the plug-in
        :attr:`cov_params` cannot: that ``Sigma`` is estimated. The
        calibrated object is the percentile-t interval, not the bootstrap
        standard error; see :class:`~alsgls.bootstrap.BootstrapResults`.

        Args:
            B: Number of replicates. ``999`` makes ``0.05 * (B + 1)`` an integer,
                which is what a 95% percentile interval wants; ``199`` is enough
                if only the standard errors are of interest.
            method: ``"parametric"`` draws errors from the fitted
                ``F F' + diag(D)``; ``"wild"`` multiplies each residual row by
                a Rademacher sign; ``"residual"`` resamples whole residual rows.
                The last two make no distributional assumption.
            seed: Seed for the replicate stream; the same seed gives the same
                replicates.

        Returns:
            The bootstrap distribution and the inference built on it.
        """
        Xs, Y = self.model.as_arrays()
        est = self.estimator
        estimates, tstats = bootstrap_system(
            Xs,
            Y,
            B_list=self.B_list,
            F=self.F,
            D=self.D,
            k=self.rank,
            als_kwargs=est.als_kwargs(),
            B=B,
            method=method,
            seed=seed,
        )
        return BootstrapResults(
            params=self.params.copy(),
            se_plugin=self.bse.copy(),
            estimates=estimates,
            tstats=tstats,
            method=method,
            seed=seed,
            df=self._resid_df(),
        )

    def summary(self, alpha: float = 0.05) -> str:
        """Text summary of estimation results (statsmodels-style).

        Args:
            alpha: Significance level for confidence intervals

        Returns:
            str: Formatted summary table
        """
        alpha = _check_alpha(alpha)
        ci = self.conf_int(alpha)
        lines = []
        sep = "=" * 78
        lines.append(sep)
        lines.append("              ALS-GLS System Estimation Results")
        lines.append(sep)
        lines.append(f"Observations: {self.model.nobs}")
        lines.append(f"Equations: {self.model.keqs}")
        lines.append(f"Factor rank: {self.rank}")
        lines.append(f"Log-Likelihood: {self.loglike:.2f}")
        lines.append("-" * 78)
        header = f"{'Parameter':<40} {'Coef':>10} {'Std Err':>10} {'t':>8} {'P>|t|':>8}"
        lines.append(header)
        lines.append("-" * 78)

        for i, (eq_name, var_name) in enumerate(self.param_labels):
            label = f"{eq_name}:{var_name}"
            if len(label) > 38:
                label = label[:35] + "..."
            coef = self.params[i]
            se = self.bse[i]
            t = self.tvalues[i]
            p = self.pvalues[i]
            line = f"{label:<40} {coef:>10.4f} {se:>10.4f} {t:>8.2f} {p:>8.4f}"
            lines.append(line)

        lines.append(sep)
        lines.append(f"Confidence intervals ({100 * (1 - alpha):.0f}%):")
        lines.append(
            f"{'Parameter':<40} {'[' + str(alpha / 2):>10} "
            f"{str(1 - alpha / 2) + ']':>10}"
        )
        lines.append("-" * 78)
        for i, (eq_name, var_name) in enumerate(self.param_labels):
            label = f"{eq_name}:{var_name}"
            if len(label) > 38:
                label = label[:35] + "..."
            line = f"{label:<40} {ci[i, 0]:>10.4f} {ci[i, 1]:>10.4f}"
            lines.append(line)
        lines.append(sep)

        return "\n".join(lines)
