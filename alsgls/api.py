"""High-level estimator APIs for ALS-based GLS fitting."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .als import als_gls
from .metrics import nll_per_row
from .ops import XB_from_Blist


def _auto_rank(num_equations: int) -> int:
    """Heuristic rank used when the user does not provide one."""

    if num_equations <= 0:
        raise ValueError("num_equations must be positive")
    # Cap the rank to avoid chasing noise; allow moderate growth with K.
    return max(1, min(8, int(np.ceil(num_equations / 10))))


def _asarray_2d(x: Any, *, dtype: np.dtype = np.float64) -> np.ndarray:
    """Convert array-like input to a 2D ``numpy.ndarray``."""

    if hasattr(x, "to_numpy"):
        arr = x.to_numpy()
    else:
        arr = np.asarray(x)
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
        lam_F: float = 1e-3,
        lam_B: float = 1e-3,
        max_sweeps: int = 12,
        rel_tol: float = 1e-6,
        d_floor: float = 1e-8,
        cg_maxit: int = 800,
        cg_tol: float = 3e-7,
        scale_correct: bool = True,
        scale_floor: float = 1e-8,
    ) -> None:
        self.rank = rank
        self.lam_F = lam_F
        self.lam_B = lam_B
        self.max_sweeps = max_sweeps
        self.rel_tol = rel_tol
        self.d_floor = d_floor
        self.cg_maxit = cg_maxit
        self.cg_tol = cg_tol
        self.scale_correct = scale_correct
        self.scale_floor = scale_floor

    # ------------------------------------------------------------------
    # Scikit-learn estimator protocol
    # ------------------------------------------------------------------
    def get_params(self, deep: bool = True) -> dict[str, Any]:  # noqa: D401 - sklearn API
        return {
            "rank": self.rank,
            "lam_F": self.lam_F,
            "lam_B": self.lam_B,
            "max_sweeps": self.max_sweeps,
            "rel_tol": self.rel_tol,
            "d_floor": self.d_floor,
            "cg_maxit": self.cg_maxit,
            "cg_tol": self.cg_tol,
            "scale_correct": self.scale_correct,
            "scale_floor": self.scale_floor,
        }

    def set_params(self, **params: Any) -> ALSGLS:  # noqa: D401 - sklearn API
        for key, value in params.items():
            if not hasattr(self, key):
                raise ValueError(f"Unknown parameter {key!r}")
            setattr(self, key, value)
        return self

    # ------------------------------------------------------------------
    # Fitting / inference
    # ------------------------------------------------------------------
    def fit(self, Xs: Sequence[Any], Y: Any) -> ALSGLS:
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

        if self.rank == "auto" or self.rank is None:
            k = _auto_rank(K)
        else:
            k = int(self.rank)

        if not (1 <= k <= min(K, N)):
            raise ValueError(f"rank must be in [1, min(K={K}, N={N})]")

        B_list, F, D, mem_mb, info = als_gls(
            X_list,
            Y_arr,
            k=k,
            lam_F=self.lam_F,
            lam_B=self.lam_B,
            sweeps=self.max_sweeps,
            d_floor=self.d_floor,
            cg_maxit=self.cg_maxit,
            cg_tol=self.cg_tol,
            scale_correct=self.scale_correct,
            scale_floor=self.scale_floor,
            rel_tol=self.rel_tol,
        )

        self.B_list_ = B_list
        self.F_ = F
        self.D_ = D
        self.mem_mb_est_ = mem_mb
        self.info_ = info
        self.n_features_in_ = tuple(X.shape[1] for X in X_list)
        self.n_targets_ = K
        self.n_obs_ = N
        self.rank_ = k
        self.training_residuals_ = Y_arr - XB_from_Blist(X_list, B_list)
        self.is_fitted_ = True
        return self

    def _ensure_fitted(self) -> None:
        if not getattr(self, "is_fitted_", False):
            raise RuntimeError("The estimator has not been fitted yet")

    def predict(self, Xs: Sequence[Any]) -> np.ndarray:
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
        self._ensure_fitted()
        Y_arr = _asarray_2d(Y)
        if Y_arr.shape[1] != self.n_targets_:
            raise ValueError("Y has incompatible number of targets")
        preds = self.predict(Xs)
        if preds.shape != Y_arr.shape:
            raise ValueError("Predictions and Y have incompatible shapes")
        residual = Y_arr - preds
        return -float(nll_per_row(residual, self.F_, self.D_))


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
        lam_F: float = 1e-3,
        lam_B: float = 1e-3,
        max_sweeps: int = 12,
        rel_tol: float = 1e-6,
        d_floor: float = 1e-8,
        cg_maxit: int = 800,
        cg_tol: float = 3e-7,
        scale_correct: bool = True,
        scale_floor: float = 1e-8,
    ) -> None:
        if isinstance(system, Mapping):
            items = list(system.items())
        else:
            items = list(system)
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
        self.lam_F = lam_F
        self.lam_B = lam_B
        self.max_sweeps = max_sweeps
        self.rel_tol = rel_tol
        self.d_floor = d_floor
        self.cg_maxit = cg_maxit
        self.cg_tol = cg_tol
        self.scale_correct = scale_correct
        self.scale_floor = scale_floor

    @property
    def nobs(self) -> int:
        return self._equations[0].y.shape[0]

    @property
    def keqs(self) -> int:
        return len(self._equations)

    def as_arrays(self) -> tuple[list[np.ndarray], np.ndarray]:
        Xs = [eq.X for eq in self._equations]
        Y = np.column_stack([eq.y for eq in self._equations])
        return Xs, Y

    def fit(self) -> ALSGLSSystemResults:
        estimator = ALSGLS(
            rank=self.rank,
            lam_F=self.lam_F,
            lam_B=self.lam_B,
            max_sweeps=self.max_sweeps,
            rel_tol=self.rel_tol,
            d_floor=self.d_floor,
            cg_maxit=self.cg_maxit,
            cg_tol=self.cg_tol,
            scale_correct=self.scale_correct,
            scale_floor=self.scale_floor,
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
        self.model = model
        self.estimator = estimator
        Xs, Y = model.as_arrays()

        flattened = [b.ravel() for b in estimator.B_list_ if b.size]
        self.params = np.concatenate(flattened) if flattened else np.empty(0)
        self.param_labels = [
            (eq.name, col) for eq in model._equations for col in eq.column_names
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
        try:
            import pandas as pd  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("pandas is required for params_as_series()") from exc

        mi = pd.MultiIndex.from_tuples(
            self.param_labels, names=["equation", "variable"]
        )
        return pd.Series(self.params, index=mi)

    def predict(
        self, exog: Mapping[Any, Any] | Sequence[Any] | None = None
    ) -> np.ndarray:
        if exog is None:
            Xs = [eq.X for eq in self.model._equations]
        else:
            if isinstance(exog, Mapping):
                items = [exog[eq.name] for eq in self.model._equations]
            else:
                items = list(exog)
            if len(items) != len(self.model._equations):
                raise ValueError("Expected design matrices for all equations")
            Xs = []
            for item, eq in zip(items, self.model._equations, strict=False):
                arr = _asarray_2d(item)
                if arr.shape[1] != eq.X.shape[1]:
                    raise ValueError("Design matrix has incompatible number of columns")
                Xs.append(arr)
        return XB_from_Blist(Xs, self.B_list)

    def summary_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "nobs": self.model.nobs,
            "keqs": self.model.keqs,
            "mem_mb_est": self.mem_mb_est,
            "nll_per_row": self.nll_per_row,
            "loglike": self.loglike,
        }
