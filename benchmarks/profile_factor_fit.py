"""Profile factor decompositions inside complete ALS fits.

Run `python benchmarks/profile_factor_fit.py` with BLAS limited to one thread.
Set OPENBLAS_NUM_THREADS=1 and VECLIB_MAXIMUM_THREADS=1 before starting Python.
CSV rows go to stdout, fixture diagnostics to stderr. Block zero warms each method.
The experimental Gram route forms an additional equations-by-equations matrix.
The decomposition timer covers only SVD or eigh. Fit time includes forming the Gram
matrix for eigh, all loading updates, and the coefficient fits.
"""

import csv
import sys
from dataclasses import dataclass
from time import perf_counter
from unittest.mock import patch

import numpy as np
from scipy import linalg

from alsgls import XB_from_Blist, als, als_gls, simulate_sur
from alsgls.als import _lawley_F


@dataclass
class _Fit:
    fit_ms: float
    decomposition_ms: float
    calls: int
    loss: float
    beta: np.ndarray
    covariance: np.ndarray


def _gram_loadings(R, D, k):
    root = np.sqrt(D)
    z = (R / root) / np.sqrt(len(R))
    equations = len(D)
    eigenvalues, vectors = linalg.eigh(
        z.T @ z, subset_by_index=(equations - k, equations - 1)
    )
    gains = np.sqrt(np.maximum(eigenvalues[::-1] - 1, 0))
    return (root[:, None] * vectors[:, ::-1]) * gains


def _dense_loss(residual, covariance):
    chol = linalg.cholesky(covariance, lower=True)
    logdet = 2 * np.log(np.diag(chol)).sum()
    whitened = linalg.solve_triangular(chol, residual.T, lower=True)
    quad = np.sum(whitened**2) / len(residual)
    return 0.5 * (quad + logdet + covariance.shape[0] * np.log(2 * np.pi))


def _check_close_factors():
    # Prescribe the sample spectrum so that sampling noise cannot erase the tie.
    rng = np.random.default_rng(617)
    U = np.linalg.qr(rng.normal(size=(80, 6)))[0]
    V = np.linalg.qr(rng.normal(size=(6, 6)))[0]
    spectrum = np.array([3, 2.000001, 2, 1.2, 0.9, 0.5])
    D = np.linspace(0.5, 2, 6)
    residual = np.sqrt(80) * (U * spectrum) @ V.T * np.sqrt(D)
    for k in (2, 3):  # Near tie at the cutoff, then both factors retained.
        scaled = np.sqrt(D)[:, None] * V[:, :k]
        expected = np.diag(D) + (scaled * (spectrum[:k] ** 2 - 1)) @ scaled.T
        for name, method in (("svd", _lawley_F), ("gram", _gram_loadings)):
            factors = method(residual, D, k)
            covariance = np.diag(D) + factors @ factors.T
            np.testing.assert_allclose(covariance, expected, atol=1e-7, rtol=0)
            loss = _dense_loss(residual, covariance)
            np.testing.assert_allclose(
                loss, _dense_loss(residual, expected), atol=1e-10, rtol=0
            )
            error = np.max(np.abs(covariance - expected))
            print(
                f"near tie: k={k} {name} max_cov_error={error:.3g} loss={loss:.12g}",
                file=sys.stderr,
            )


def _fit(method, Xs, Y, rank):
    if method == "svd":
        loadings, module, routine = _lawley_F, np.linalg, "svd"
    else:
        loadings, module, routine = _gram_loadings, linalg, "eigh"
    original = getattr(module, routine)
    seconds = 0.0
    calls = 0

    def timed(*args, **kwargs):
        nonlocal seconds, calls
        start = perf_counter()
        result = original(*args, **kwargs)
        seconds += perf_counter() - start
        calls += 1
        return result

    with patch.object(als, "_lawley_F", loadings), patch.object(module, routine, timed):
        start = perf_counter()
        B, F, D, _, info = als_gls(Xs, Y, k=rank)
        fit_seconds = perf_counter() - start

    covariance = F @ F.T + np.diag(D)
    loss = _dense_loss(Y - XB_from_Blist(Xs, B), covariance)
    np.testing.assert_allclose(loss, info["nll_trace"][-1], atol=1e-8, rtol=0)
    return _Fit(
        fit_seconds * 1000, seconds * 1000, calls, loss, np.concatenate(B), covariance
    )


def main():
    """Compare the shipping SVD with Gram eigenpairs at several shapes and ranks."""
    _check_close_factors()
    writer = csv.writer(sys.stdout)
    writer.writerow(
        (
            "rows",
            "equations",
            "rank",
            "block",
            "method",
            "fit_ms",
            "decomposition_ms",
            "calls",
            "loss",
            "max_beta_gap",
            "max_cov_gap",
        )
    )
    for rows, equations, rank in (
        (400, 8, 2),
        (400, 32, 2),
        (400, 32, 6),
        (1200, 32, 2),
        (25, 192, 2),
    ):
        Xs, Y, _, _ = simulate_sur(rows, 1, equations, p=2, k=rank, seed=713)
        for block in range(6):
            order = ("svd", "gram") if block % 2 == 0 else ("gram", "svd")
            results = {method: _fit(method, Xs, Y, rank) for method in order}
            svd, gram = results["svd"], results["gram"]
            beta_gap = np.max(np.abs(svd.beta - gram.beta))
            cov_gap = np.max(np.abs(svd.covariance - gram.covariance))
            np.testing.assert_allclose(svd.beta, gram.beta, rtol=1e-6, atol=1e-7)
            np.testing.assert_allclose(
                svd.covariance, gram.covariance, rtol=1e-6, atol=1e-6
            )
            np.testing.assert_allclose(svd.loss, gram.loss, rtol=0, atol=1e-8)
            for method, result in results.items():
                writer.writerow(
                    (
                        rows,
                        equations,
                        rank,
                        block,
                        method,
                        result.fit_ms,
                        result.decomposition_ms,
                        result.calls,
                        result.loss,
                        beta_gap,
                        cov_gap,
                    )
                )


if __name__ == "__main__":
    main()
