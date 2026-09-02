"""Check the Sigma step against implementations that were written independently.

Everything else in the suite tests alsgls against alsgls. These two tests are
the only ones whose expected answer comes from somewhere else, which is the
point: a self-consistent implementation of the wrong estimator passes every
internal check.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.decomposition import FactorAnalysis

from alsgls import XB_from_Blist, als_gls, nll_per_row, simulate_sur
from alsgls.als import _sigma_step


@pytest.mark.parametrize(("K", "k", "seed"), [(20, 4, 7), (12, 3, 1), (30, 5, 3)])
def test_sigma_step_agrees_with_sklearn_factor_analysis(K, k, seed) -> None:
    """With no regressors the Sigma step is exactly maximum-likelihood factor
    analysis, which scikit-learn implements by the same Lawley update. The two
    were written from the same result but not from each other, so agreement on
    the implied covariance is real evidence.

    F itself is identified only up to a k x k rotation, so the comparison is on
    Sigma = F F^T + diag(D), which is rotation invariant.
    """
    rng = np.random.default_rng(seed)
    F_true = rng.standard_normal((K, k))
    D_true = 0.2 + rng.random(K)
    N = 400
    X = rng.standard_normal((N, k)) @ F_true.T + rng.standard_normal((N, K)) * np.sqrt(
        D_true
    )
    X = X - X.mean(axis=0)

    fa = FactorAnalysis(n_components=k, svd_method="lapack", tol=1e-12, max_iter=5000)
    fa.fit(X)
    sk_sigma = fa.components_.T @ fa.components_ + np.diag(fa.noise_variance_)

    # Both are run to a tolerance far below the agreement being asserted, so
    # what is measured is whether they reach the same optimum rather than how
    # early each stopped. At the shipped tolerance they agree to about 1e-6.
    var_ref = float(np.mean(np.var(X, axis=0)))
    F, D, nll, _ = _sigma_step(
        X, np.var(X, axis=0), k, 1e-12 * var_ref, tol=1e-16, max_iter=20000
    )
    our_sigma = F @ F.T + np.diag(D)

    rel = np.abs(our_sigma - sk_sigma).max() / np.abs(sk_sigma).max()
    assert rel < 1e-7, f"implied covariances differ by {rel:.3e}"

    # The sharper statement: the two reach the same likelihood, not merely a
    # similar covariance.
    assert nll == pytest.approx(
        nll_per_row(X, fa.components_.T, fa.noise_variance_), rel=1e-9
    )


@pytest.mark.parametrize("K", [6, 12])
def test_identical_regressors_reduce_to_equation_by_equation_ols(K: int) -> None:
    """Zellner (1962): when every equation has the same design matrix, the SUR
    GLS estimator collapses to OLS run separately on each equation, whatever the
    cross-equation covariance is. An exactly right answer from outside the
    package, and one no internal consistency check would notice being wrong.
    """
    Xs, Y, _, _ = simulate_sur(N_tr=300, N_te=5, K=K, p=3, k=2, seed=7)
    shared = [Xs[0]] * K
    B, _, _, _, _ = als_gls(shared, Y, k=2, sweeps=40, lam_B=0.0, rel_tol=1e-12)

    ols = np.linalg.lstsq(Xs[0], Y, rcond=None)[0]
    err = max(np.abs(B[j].ravel() - ols[:, j]).max() for j in range(K))
    assert err < 1e-8, f"GLS and OLS differ by {err:.3e}"
    # and the fitted values agree, which is what a user would notice
    assert np.abs(XB_from_Blist(shared, B) - Xs[0] @ ols).max() < 1e-8
