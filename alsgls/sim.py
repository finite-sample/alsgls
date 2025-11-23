import numpy as np

from .ops import XB_from_Blist


def simulate_sur(N_tr, N_te, K, p, k, seed=0):
    """Simulate a Seemingly Unrelated Regression (SUR) dataset.

    Parameters
    ----------
    N_tr : int
        Number of training samples.
    N_te : int
        Number of test samples.
    K : int
        Number of response equations.
    p : int
        Number of features per equation.
    k : int
        Latent factor dimension controlling correlated noise.
    seed : int, optional
        Seed for the NumPy random number generator. Defaults to ``0``.

    Returns
    -------
    X_tr : list of ndarray
        Feature matrices for the training set. Each element has shape ``(N_tr, p)``.
    Y_tr : ndarray
        Training responses of shape ``(N_tr, K)``.
    X_te : list of ndarray
        Feature matrices for the test set. Each element has shape ``(N_te, p)``.
    Y_te : ndarray
        Test responses of shape ``(N_te, K)``.

    Notes
    -----
    Randomness is controlled via ``numpy.random.default_rng(seed)``; pass a
    different ``seed`` for different simulations.

    Examples
    --------
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


def simulate_gls(N_tr, N_te, p_list, k, seed=0):
    """Simulate a generalized least squares (GLS) dataset.

    This variant allows each response equation to have its own number of
    features as specified by ``p_list``.

    Parameters
    ----------
    N_tr : int
        Number of training samples.
    N_te : int
        Number of test samples.
    p_list : sequence of int
        Number of features for each equation.
    k : int
        Latent factor dimension controlling correlated noise.
    seed : int, optional
        Seed for the NumPy random number generator. Defaults to ``0``.

    Returns
    -------
    X_tr : list of ndarray
        Feature matrices for the training set. ``X_tr[j]`` has shape
        ``(N_tr, p_list[j])``.
    Y_tr : ndarray
        Training responses of shape ``(N_tr, K)`` where ``K = len(p_list)``.
    X_te : list of ndarray
        Feature matrices for the test set. ``X_te[j]`` has shape
        ``(N_te, p_list[j])``.
    Y_te : ndarray
        Test responses of shape ``(N_te, K)``.

    Notes
    -----
    Randomness is controlled via ``numpy.random.default_rng(seed)``; pass a
    different ``seed`` for different simulations.

    Examples
    --------
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
