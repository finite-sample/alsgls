"""Input validation and sanitization helpers for alsgls.

This module provides standardized validation functions to ensure consistent
input handling across all solvers and improve error message quality.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _validate_design_matrices(Xs: Any, *, name: str = "Xs") -> list[np.ndarray]:
    """Validate and convert design matrices to standardized format.
    
    Parameters
    ----------
    Xs : list-like
        List of design matrices, each should be array-like.
    name : str, optional
        Variable name for error messages.
        
    Returns
    -------
    list[np.ndarray]
        Validated list of 2D numpy arrays.
        
    Raises
    ------
    ValueError
        If Xs is empty, not list-like, or contains invalid matrices.
    """
    if not isinstance(Xs, (list, tuple)) or len(Xs) == 0:
        raise ValueError(
            f"{name} must be a non-empty list or tuple of arrays. "
            f"Got {type(Xs).__name__} with length {len(Xs) if hasattr(Xs, '__len__') else '?'}"
        )
    
    validated_Xs = []
    for j, X in enumerate(Xs):
        try:
            X_arr = np.asarray(X, dtype=np.float64)
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"{name}[{j}] cannot be converted to numeric array: {e}"
            ) from e
            
        if X_arr.ndim != 2:
            raise ValueError(
                f"{name}[{j}] must be 2D, got {X_arr.ndim}D with shape {X_arr.shape}. "
                f"Try {name}[{j}].reshape(-1, {X_arr.size}) for 1D arrays."
            )
            
        validated_Xs.append(X_arr)
    
    return validated_Xs


def _validate_response_matrix(Y: Any, *, name: str = "Y") -> np.ndarray:
    """Validate and convert response matrix to standardized format.
    
    Parameters
    ----------
    Y : array-like
        Response matrix, should be 2D or convertible to 2D.
    name : str, optional
        Variable name for error messages.
        
    Returns
    -------
    np.ndarray
        Validated 2D numpy array.
        
    Raises
    ------
    ValueError
        If Y cannot be converted to a 2D array.
    """
    try:
        Y_arr = np.asarray(Y, dtype=np.float64)
    except (TypeError, ValueError) as e:
        raise ValueError(f"{name} cannot be converted to numeric array: {e}") from e
    
    if Y_arr.ndim == 1:
        Y_arr = Y_arr[:, None]
    elif Y_arr.ndim != 2:
        raise ValueError(
            f"{name} must be 1D or 2D, got {Y_arr.ndim}D with shape {Y_arr.shape}. "
            f"Try {name}.reshape({Y_arr.shape[0]}, -1) to flatten to 2D."
        )
    
    return Y_arr


def _check_array_compatibility(
    Xs: list[np.ndarray], 
    Y: np.ndarray, 
    *, 
    X_name: str = "Xs", 
    Y_name: str = "Y"
) -> None:
    """Check that design matrices and response have compatible dimensions.
    
    Parameters
    ----------
    Xs : list[np.ndarray]
        Validated design matrices.
    Y : np.ndarray 
        Validated response matrix.
    X_name, Y_name : str, optional
        Variable names for error messages.
        
    Raises
    ------
    ValueError
        If dimensions are incompatible.
    """
    N, K = Y.shape
    
    if len(Xs) != K:
        raise ValueError(
            f"Number of design matrices ({len(Xs)}) must match {Y_name} columns ({K}). "
            f"Either provide {K} design matrices or reshape {Y_name} to have {len(Xs)} columns."
        )
    
    for j, X in enumerate(Xs):
        if X.shape[0] != N:
            raise ValueError(
                f"{X_name}[{j}] has {X.shape[0]} rows but {Y_name} has {N}. "
                f"All inputs must have the same number of samples."
            )


def _validate_rank_parameter(k: Any, N: int, K: int, *, name: str = "k") -> int:
    """Validate and convert rank parameter.
    
    Parameters
    ----------
    k : int-like
        Rank parameter for low-rank component.
    N : int
        Number of samples.
    K : int
        Number of equations.
    name : str, optional
        Parameter name for error messages.
        
    Returns
    -------
    int
        Validated rank parameter.
        
    Raises
    ------
    ValueError
        If k is not a valid rank.
    """
    try:
        k_int = int(k)
    except (TypeError, ValueError) as e:
        raise ValueError(f"{name} must be an integer, got {type(k).__name__}") from e
    
    if k_int != k:
        raise ValueError(f"{name} must be an integer, got {k}")
    
    max_rank = min(K, N)
    if not (1 <= k_int <= max_rank):
        raise ValueError(
            f"{name}={k_int} must be between 1 and min(K={K}, N={N})={max_rank}. "
            f"Try reducing {name} or increasing the number of samples/equations."
        )
    
    return k_int


def _sanitize_regularization_params(
    lam_F: float | None = None, 
    lam_B: float | None = None
) -> tuple[float, float]:
    """Validate and sanitize regularization parameters.
    
    Parameters
    ----------
    lam_F : float, optional
        Regularization for factor loadings.
    lam_B : float, optional 
        Regularization for regression coefficients.
        
    Returns
    -------
    tuple[float, float]
        Validated (lam_F, lam_B) parameters.
        
    Raises
    ------
    ValueError
        If regularization parameters are negative.
    """
    if lam_F is not None:
        if not isinstance(lam_F, (int, float)) or lam_F < 0:
            raise ValueError(
                f"lam_F must be non-negative, got {lam_F}. "
                f"Try lam_F=1e-3 for light regularization."
            )
    
    if lam_B is not None:
        if not isinstance(lam_B, (int, float)) or lam_B < 0:
            raise ValueError(
                f"lam_B must be non-negative, got {lam_B}. "
                f"Try lam_B=1e-3 for light regularization."
            )
    
    return float(lam_F or 1e-3), float(lam_B or 1e-3)


def _validate_gls_inputs(
    Xs: Any, 
    Y: Any, 
    k: Any, 
    *,
    lam_F: float | None = None,
    lam_B: float | None = None
) -> tuple[list[np.ndarray], np.ndarray, int, float, float]:
    """Comprehensive validation for GLS solver inputs.
    
    This function combines all individual validation steps for consistency.
    
    Parameters
    ----------
    Xs : list-like
        Design matrices.
    Y : array-like
        Response matrix.
    k : int-like
        Rank parameter.
    lam_F, lam_B : float, optional
        Regularization parameters.
        
    Returns
    -------
    tuple
        Validated (Xs, Y, k, lam_F, lam_B).
    """
    # Step 1: Validate individual components
    Xs_valid = _validate_design_matrices(Xs)
    Y_valid = _validate_response_matrix(Y) 
    N, K = Y_valid.shape
    k_valid = _validate_rank_parameter(k, N, K)
    lam_F_valid, lam_B_valid = _sanitize_regularization_params(lam_F, lam_B)
    
    # Step 2: Check compatibility
    _check_array_compatibility(Xs_valid, Y_valid)
    
    return Xs_valid, Y_valid, k_valid, lam_F_valid, lam_B_valid


def _validate_convergence_params(
    sweeps: Any = None,
    rel_tol: Any = None, 
    cg_maxit: Any = None,
    cg_tol: Any = None
) -> dict[str, int | float]:
    """Validate convergence and solver parameters.
    
    Parameters
    ----------
    sweeps : int, optional
        Maximum number of ALS sweeps.
    rel_tol : float, optional
        Relative tolerance for convergence.
    cg_maxit : int, optional
        Maximum CG iterations.
    cg_tol : float, optional
        CG tolerance.
        
    Returns
    -------
    dict
        Validated parameters.
    """
    params: dict[str, int | float] = {}
    
    if sweeps is not None:
        if not isinstance(sweeps, int) or sweeps < 1:
            raise ValueError(
                f"sweeps must be a positive integer, got {sweeps}. "
                f"Try sweeps=8 for typical problems."
            )
        params['sweeps'] = sweeps
    
    if rel_tol is not None:
        if not isinstance(rel_tol, (int, float)) or rel_tol < 0:
            raise ValueError(
                f"rel_tol must be non-negative, got {rel_tol}. "
                f"Try rel_tol=1e-6 for standard convergence."
            )
        params['rel_tol'] = float(rel_tol)
    
    if cg_maxit is not None:
        if not isinstance(cg_maxit, int) or cg_maxit < 1:
            raise ValueError(
                f"cg_maxit must be a positive integer, got {cg_maxit}. "
                f"Try cg_maxit=800 for typical problems."
            )
        params['cg_maxit'] = cg_maxit
    
    if cg_tol is not None:
        if not isinstance(cg_tol, (int, float)) or cg_tol <= 0:
            raise ValueError(
                f"cg_tol must be positive, got {cg_tol}. "
                f"Try cg_tol=3e-7 for standard accuracy."
            )
        params['cg_tol'] = float(cg_tol)
        
    return params