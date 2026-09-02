"""Synthetic data generators for GLS/SUR experiments and tests."""

from __future__ import annotations

import numpy as np

from .ops import XB_from_Blist


def simulate_sur(
    N_tr: int, N_te: int, K: int, p: int, k: int, seed: int = 0
) -> tuple[list[np.ndarray], np.ndarray, list[np.ndarray], np.ndarray]:
    """Simulate a Seemingly Unrelated Regression (SUR) dataset.

    Args:
        N_tr: Number of training samples.
        N_te: Number of test samples.
        K: Number of response equations.
        p: Number of features per equation.
        k: Latent factor dimension controlling correlated noise.
        seed: Seed for the NumPy random number generator. Defaults to ``0``.

    Returns:
        A tuple ``(X_tr, Y_tr, X_te, Y_te)``: lists of per-equation feature
        matrices of shape ``(N_tr, p)`` and ``(N_te, p)``, and response
        matrices of shape ``(N_tr, K)`` and ``(N_te, K)``.

    Notes:
        Randomness is controlled via ``numpy.random.default_rng(seed)``; pass a
        different ``seed`` for different simulations.

    Examples:
        >>> from alsgls import simulate_sur
        >>> Xtr, Ytr, Xte, Yte = simulate_sur(100, 20, K=3, p=5, k=2, seed=42)
    """
    rng = np.random.default_rng(seed)
    N = N_tr + N_te
    base = rng.standard_normal((N, p))
    Xs = [base + 0.5 * rng.standard_normal((N, p)) for _ in range(K)]
    B = [rng.standard_normal((p, 1)) for _ in range(K)]
    F0 = 1.0 * rng.standard_normal((K, k))
    D0 = 0.05 + 0.20 * rng.random(K)
    U = rng.standard_normal((N, k))
    Y = (
        XB_from_Blist(Xs, B)
        + U @ F0.T
        + rng.standard_normal((N, K)) * np.sqrt(D0)[None, :]
    )
    return [X[:N_tr] for X in Xs], Y[:N_tr], [X[N_tr:] for X in Xs], Y[N_tr:]


def simulate_gls(
    N_tr: int, N_te: int, p_list: list[int], k: int, seed: int = 0
) -> tuple[list[np.ndarray], np.ndarray, list[np.ndarray], np.ndarray]:
    """Simulate a generalized least squares (GLS) dataset.

    This variant allows each response equation to have its own number of
    features as specified by ``p_list``.

    Args:
        N_tr: Number of training samples.
        N_te: Number of test samples.
        p_list: Number of features for each equation.
        k: Latent factor dimension controlling correlated noise.
        seed: Seed for the NumPy random number generator. Defaults to ``0``.

    Returns:
        A tuple ``(X_tr, Y_tr, X_te, Y_te)``: per-equation feature matrices
        with ``X_tr[j]`` of shape ``(N_tr, p_list[j])`` and ``X_te[j]`` of
        shape ``(N_te, p_list[j])``, and responses of shape ``(N_tr, K)``
        and ``(N_te, K)`` where ``K = len(p_list)``.

    Notes:
        Randomness is controlled via ``numpy.random.default_rng(seed)``; pass a
        different ``seed`` for different simulations.

    Examples:
        >>> from alsgls import simulate_gls
        >>> p_list = [3, 5, 2]
        >>> Xtr, Ytr, Xte, Yte = simulate_gls(100, 20, p_list, k=2, seed=0)
    """
    rng = np.random.default_rng(seed)
    K = len(p_list)
    N = N_tr + N_te
    Xs = []
    for p in p_list:
        base = rng.standard_normal((N, p))
        Xs.append(base + 0.5 * rng.standard_normal((N, p)))
    B = [rng.standard_normal((p, 1)) for p in p_list]
    F0 = 1.0 * rng.standard_normal((K, k))
    D0 = 0.05 + 0.20 * rng.random(K)
    U = rng.standard_normal((N, k))
    Y = (
        XB_from_Blist(Xs, B)
        + U @ F0.T
        + rng.standard_normal((N, K)) * np.sqrt(D0)[None, :]
    )
    return [X[:N_tr] for X in Xs], Y[:N_tr], [X[N_tr:] for X in Xs], Y[N_tr:]
