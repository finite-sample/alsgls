"""Kackar-Harville correction: the checks that establish it computes the right thing.

Ordered from cheapest to most expensive, and from most to least mechanical.
The first three are about the code; the last two are about whether the
statistics come out where the theory says they should.
"""

from __future__ import annotations

import numpy as np
import pytest

from alsgls import ALSGLSSystem, nll_per_row, simulate_sur
from alsgls.kackar_harville import (
    covariance_information,
    kh_correction,
    lambda_from_derivatives,
    sigma_derivatives,
)
from alsgls.ops import assemble_blocks, gram_blocks


def _brute_force(Xs, F, D):
    """The r^2 double loop, straight from Kackar and Harville's expression."""
    n = Xs[0].shape[0]
    gram = gram_blocks(Xs)
    Sigma = F @ F.T + np.diag(D)
    Sinv = np.linalg.inv(Sigma)
    Phi = np.linalg.inv(assemble_blocks(Sinv, gram))
    dS = sigma_derivatives(F)
    r = dS.shape[0]
    W = np.linalg.pinv(covariance_information(Sigma, dS, n), rcond=1e-9)
    P = [-assemble_blocks(Sinv @ dS[i] @ Sinv, gram) for i in range(r)]
    Lam = np.zeros_like(Phi)
    for i in range(r):
        for j in range(r):
            if abs(W[i, j]) < 1e-14:
                continue
            Q = assemble_blocks(Sinv @ dS[i] @ Sinv @ dS[j] @ Sinv, gram)
            Lam += W[i, j] * (Phi @ (Q - P[i] @ Phi @ P[j]) @ Phi)
    return Phi, Lam


# --------------------------------------------------------------------------
# 1. The derivatives and the information are what they claim to be.
# --------------------------------------------------------------------------


def test_sigma_derivatives_match_finite_differences() -> None:
    rng = np.random.default_rng(0)
    K, k = 6, 2
    F = rng.standard_normal((K, k))
    D = 0.5 + rng.random(K)
    dS = sigma_derivatives(F)

    def sigma(theta):
        Fz = theta[: K * k].reshape(K, k, order="F")
        return Fz @ Fz.T + np.diag(theta[K * k :])

    theta = np.concatenate([F.ravel(order="F"), D])
    h = 1e-6
    for i in range(theta.size):
        e = np.zeros_like(theta)
        e[i] = h
        fd = (sigma(theta + e) - sigma(theta - e)) / (2 * h)
        assert np.abs(fd - dS[i]).max() < 1e-8


def test_information_matches_the_nll_hessian_on_the_identified_subspace() -> None:
    """Expected information against the observed Hessian at a large N, where
    they coincide. Compared on the subspace orthogonal to the k(k-1)/2 rotation
    directions, which carry no information by construction."""
    rng = np.random.default_rng(1)
    K, k, N = 5, 2, 100_000
    F = rng.standard_normal((K, k))
    D = 0.5 + rng.random(K)
    R = rng.standard_normal((N, k)) @ F.T + rng.standard_normal((N, K)) * np.sqrt(D)
    dS = sigma_derivatives(F)
    info = covariance_information(F @ F.T + np.diag(D), dS, N)

    theta = np.concatenate([F.ravel(order="F"), D])

    def nll(t):
        return N * nll_per_row(R, t[: K * k].reshape(K, k, order="F"), t[K * k :])

    h = 1e-3
    hess = np.zeros((theta.size, theta.size))
    for i in range(theta.size):
        for j in range(i, theta.size):
            ei = np.zeros_like(theta)
            ej = np.zeros_like(theta)
            ei[i] = h
            ej[j] = h
            hess[i, j] = hess[j, i] = (
                nll(theta + ei + ej)
                - nll(theta + ei - ej)
                - nll(theta - ei + ej)
                + nll(theta - ei - ej)
            ) / (4 * h * h)

    vals, vecs = np.linalg.eigh(info)
    order = np.argsort(vals)[::-1]
    assert vals[order[-1]] / vals[order[0]] < 1e-10, "expected one null direction"
    keep = vecs[:, order[: theta.size - k * (k - 1) // 2]]
    A = keep.T @ info @ keep
    B = keep.T @ hess @ keep
    # Sampling noise in the observed Hessian is O(1/sqrt(N)); 1.2% at N = 1e5.
    assert np.abs(A - B).max() / np.abs(A).max() < 3e-2


# --------------------------------------------------------------------------
# 2. The structured implementation is the brute-force one.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(("K", "k"), [(4, 1), (8, 2), (12, 3)])
def test_structured_equals_brute_force(K: int, k: int) -> None:
    rng = np.random.default_rng(K)
    Xs, _, _, _ = simulate_sur(N_tr=60, N_te=5, K=K, p=3, k=k, seed=K)
    F = rng.standard_normal((K, k))
    D = 0.5 + rng.random(K)
    Phi_a, Lam_a = _brute_force(Xs, F, D)
    Phi_b, Lam_b = kh_correction(Xs, F, D)
    assert np.abs(Phi_a - Phi_b).max() < 1e-12
    assert np.abs(Lam_a - Lam_b).max() < 1e-12 * max(1.0, np.abs(Lam_a).max() / 1e-3)


# --------------------------------------------------------------------------
# 3. Lambda does not depend on the rotation of F, which validates the pinv.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(("K", "k"), [(8, 2), (12, 3)])
def test_lambda_is_invariant_to_rotating_F(K: int, k: int) -> None:
    """F -> FQ leaves Sigma fixed. The information is singular along those
    directions and W is a pseudo-inverse; if the choice of pseudo-inverse
    leaked into Lambda, rotating F would change it."""
    rng = np.random.default_rng(10 + K)
    Xs, _, _, _ = simulate_sur(N_tr=60, N_te=5, K=K, p=3, k=k, seed=K)
    F = rng.standard_normal((K, k))
    D = 0.5 + rng.random(K)
    Q, _ = np.linalg.qr(rng.standard_normal((k, k)))
    _, L1 = kh_correction(Xs, F, D)
    _, L2 = kh_correction(Xs, F @ Q, D)
    assert np.abs(L1 - L2).max() < 1e-12


# --------------------------------------------------------------------------
# 4. Against a closed form the machinery is not built from.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(("T", "k1", "k2"), [(20, 2, 2), (40, 3, 3), (80, 2, 3)])
def test_two_equation_orthogonal_sur_gives_lambda_over_phi_of_one_over_T(T, k1, k2):
    """Two equations, orthogonal regressors, unstructured 2x2 Sigma: the
    Kackar-Harville excess variance is exactly 1/T for every coefficient, for
    every correlation. The factor-model code never sees this parameterisation,
    so agreement here is evidence about the Lambda machinery itself. Monte
    Carlo on the same design (4000 replicates, run when this was written)
    gives 1.076, 1.037 and 1.010 for var/Phi against 1.050, 1.025 and 1.0125:
    right to first order, with a rho-dependent higher-order remainder at
    small T."""
    rng = np.random.default_rng(T)
    Q, _ = np.linalg.qr(rng.standard_normal((T, k1 + k2)))
    X1 = Q[:, :k1] * np.sqrt(T)
    X2 = Q[:, k1:] * np.sqrt(T)
    dS = np.zeros((3, 2, 2))
    dS[0, 0, 0] = 1.0
    dS[1, 0, 1] = dS[1, 1, 0] = 1.0
    dS[2, 1, 1] = 1.0
    for rho in (0.0, 0.5, 0.9):
        Sigma = np.array([[1.0, rho], [rho, 1.0]])
        Phi, Lam = lambda_from_derivatives([X1, X2], Sigma, dS)
        assert np.allclose(np.diag(Lam) / np.diag(Phi), 1.0 / T, rtol=1e-8)


# --------------------------------------------------------------------------
# 5. At the truth, Phi + Lambda is the actual spread of the feasible estimator.
# --------------------------------------------------------------------------


def test_phi_plus_lambda_at_the_truth_matches_the_fgls_spread() -> None:
    """The statement Kackar and Harville make, checked on the test suite's
    Monte Carlo fixture. Measured 0.979 when this was written, against 0.940
    for Phi alone at the truth -- the ceiling for any formula that holds Sigma
    fixed. This is what justifies shipping the correction: it is the right
    first-order term. What it does not fix is the bias of Phi_hat at
    Sigma_hat, which is why the estimate at (F_hat, D_hat) lands at 0.89
    rather than 0.98; see ``test_standard_errors``."""
    import sys

    sys.path.insert(0, "tests")
    import test_standard_errors as mc

    Xs, B0, F0, D0, _ = mc._mc_truth(20)
    tracked = list(mc.MC_TRACKED)
    est = []
    for child in np.random.SeedSequence(0).spawn(300):
        rng = np.random.default_rng(child)
        Y = mc._mc_draw(rng, Xs, B0, F0, D0)
        system = {f"eq{j}": (Y[:, j], Xs[j]) for j in range(mc.MC_K)}
        est.append(
            ALSGLSSystem(system, rank=mc.MC_RANK, lam_B=0.0, max_sweeps=12).fit().params
        )
    sd = np.array(est)[:, tracked].std(0, ddof=1)
    Phi, Lam = kh_correction(Xs, F0, D0)
    ratio = (np.sqrt(np.diag(Phi + Lam))[tracked] / sd).mean()
    assert 0.95 < ratio < 1.02, f"Phi + Lambda at the truth: se ratio {ratio:.3f}"
    plugin = (np.sqrt(np.diag(Phi))[tracked] / sd).mean()
    assert plugin < ratio - 0.02, (
        f"Lambda should lift the plug-in: {plugin:.3f} -> {ratio:.3f}"
    )
