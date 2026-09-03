"""Rank selection methods for ALS-GLS: BIC and cross-validation."""

from __future__ import annotations

from typing import Any

import numpy as np

from ._validation import max_identified_rank
from .als import als_gls
from .metrics import nll_per_row
from .ops import XB_from_Blist


def _n_params(K: int, k: int, p_total: int) -> int:
    """Free parameters of the low-rank-plus-diagonal SUR model.

    ``F`` contributes ``K*k`` loadings less the ``k*(k-1)/2`` dimensions of the
    orthogonal group that leave ``F F^T`` fixed, ``D`` contributes ``K``, and
    the regression contributes ``p_total``.

    Args:
        K: Number of equations.
        k: Factor rank.
        p_total: Total number of regression coefficients.

    Returns:
        The number of free parameters.
    """
    return K * k + K - k * (k - 1) // 2 + p_total


def _default_k_candidates(K: int) -> list[int]:
    """Generate default candidate ranks based on number of equations."""
    max_k = min(K // 2, 12, max_identified_rank(K))
    if max_k < 1:
        return [1]
    return list(range(1, max_k + 1))


def select_rank_bic(
    Xs: list[np.ndarray],
    Y: np.ndarray,
    k_candidates: list[int] | None = None,
    **als_kwargs: Any,
) -> tuple[int, list[dict[str, Any]]]:
    """Select rank by minimizing BIC (Bayesian Information Criterion).

    BIC(k) = 2 * N * nll_per_row + n_params * log(N)

    which is the usual ``-2 * loglik + n_params * log(N)``, since
    ``N * nll_per_row`` is the negative log-likelihood of the sample. The
    value used to be reported at half this, so it was not comparable with the
    ``bic`` attribute of statsmodels or any other package; the rank chosen is
    unchanged, because halving is monotone.

    ``n_params`` counts the free parameters of the whole fitted model: the
    ``K*k`` factor loadings in ``F`` less the ``k*(k-1)/2`` orthogonal rotations
    ``F -> F Q`` that leave ``F F^T`` unchanged and so are not identified, the
    ``K`` diagonal variances in ``D``, and the ``sum(p_j)`` regression
    coefficients, since ``nll_per_row`` is evaluated at the fitted ``beta`` and
    a BIC has to charge for it. This is the standard factor-analysis count;
    R's ``factanal`` reports the complementary ``df = ((K-k)^2 - K - k) / 2``.

    A rank whose fit raised carries the message in its ``error`` key.

    The count used to be ``K*(k+1) + k``, which neither subtracted the rotational
    redundancy nor charged for ``beta``. Only the first term varies with ``k``,
    so on the fixtures tested the selected rank is unchanged; the reported value
    was wrong either way.

    Args:
        Xs: Design matrices for each equation.
        Y: Response matrix.
        k_candidates: Candidate ranks to evaluate. Defaults to
            range(1, min(K//2, 12)+1).
        **als_kwargs: Additional arguments passed to als_gls().

    Returns:
        best_k: The rank with minimum BIC.
        results: Per-rank dicts of 'k', 'nll', 'bic', 'n_params', 'converged'.

    Raises:
        RuntimeError: If no candidate rank produced a usable fit.
    """
    N, K = Y.shape
    if k_candidates is None:
        k_candidates = _default_k_candidates(K)

    results: list[dict[str, Any]] = []
    for k in k_candidates:
        try:
            _, _, _, _, info = als_gls(Xs, Y, k=k, **als_kwargs)
            nll = info["nll_trace"][-1]
            n_params = _n_params(K, k, sum(X.shape[1] for X in Xs))
            bic = 2 * N * nll + n_params * np.log(N)
            results.append(
                {
                    "k": k,
                    "nll": nll,
                    "bic": bic,
                    "n_params": n_params,
                    "converged": True,
                }
            )
        except (np.linalg.LinAlgError, ValueError, RuntimeError) as exc:
            # Narrow: a bare ``except Exception`` here also swallowed
            # KeyboardInterrupt-adjacent bugs and typos in ``als_kwargs``,
            # reporting them as a rank that merely failed to converge.
            results.append(
                {
                    "k": k,
                    "nll": np.inf,
                    "bic": np.inf,
                    "converged": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    valid_results = [r for r in results if r.get("converged", False)]
    if not valid_results:
        raise RuntimeError("BIC rank selection failed: no valid fits")

    best = min(valid_results, key=lambda x: x["bic"])
    return best["k"], results


def select_rank_cv(
    Xs: list[np.ndarray],
    Y: np.ndarray,
    k_candidates: list[int] | None = None,
    n_folds: int = 5,
    random_state: int | np.random.Generator | None = None,
    **als_kwargs: Any,
) -> tuple[int, list[dict[str, Any]]]:
    """Select rank by k-fold cross-validation on validation NLL.

    Args:
        Xs: Design matrices for each equation.
        Y: Response matrix.
        k_candidates: Candidate ranks to evaluate. Defaults to
            range(1, min(K//2, 12)+1).
        n_folds: Number of cross-validation folds.
        random_state: Random state for reproducible fold splits.
        **als_kwargs: Additional arguments passed to als_gls().

    Returns:
        best_k: The rank with minimum mean CV NLL.
        results: Per-rank results containing 'k', 'cv_nll', 'cv_std', 'fold_nlls'.

    Raises:
        ValueError: If ``n_folds`` is below 2 or exceeds the number of rows.
        RuntimeError: If no candidate rank produced a usable fit.
    """
    N, K = Y.shape
    if k_candidates is None:
        k_candidates = _default_k_candidates(K)

    if n_folds < 2:
        raise ValueError("n_folds must be at least 2")
    if n_folds > N:
        raise ValueError(f"n_folds={n_folds} cannot exceed N={N}")

    rng = np.random.default_rng(random_state)
    indices = np.arange(N)
    rng.shuffle(indices)
    folds = np.array_split(indices, n_folds)

    results: list[dict[str, Any]] = []
    for k in k_candidates:
        fold_nlls = []
        for i in range(n_folds):
            val_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != i])

            Xs_tr = [X[train_idx] for X in Xs]
            Y_tr = Y[train_idx]
            Xs_val = [X[val_idx] for X in Xs]
            Y_val = Y[val_idx]

            try:
                B, F, D, _, _ = als_gls(Xs_tr, Y_tr, k=k, **als_kwargs)
                R_val = Y_val - XB_from_Blist(Xs_val, B)
                val_nll = nll_per_row(R_val, F, D)
                fold_nlls.append(val_nll)
            except (np.linalg.LinAlgError, ValueError, RuntimeError):
                fold_nlls.append(np.inf)

        cv_nll = float(np.mean(fold_nlls))
        cv_std = float(np.std(fold_nlls))
        results.append(
            {"k": k, "cv_nll": cv_nll, "cv_std": cv_std, "fold_nlls": fold_nlls}
        )

    valid_results = [r for r in results if np.isfinite(r["cv_nll"])]
    if not valid_results:
        raise RuntimeError("CV rank selection failed: no valid fits")

    best = min(valid_results, key=lambda x: x["cv_nll"])
    return best["k"], results
