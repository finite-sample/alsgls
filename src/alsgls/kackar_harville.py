"""Kackar-Harville correction to the coefficient covariance.

The plug-in ``Phi = (X' Sigma^-1 X)^-1`` is the variance of a GLS estimator
whose covariance is known. When ``Sigma = Sigma(theta)`` is estimated from the
same data, Kackar and Harville (1984, JASA 79, 853-862) show the variance of
the feasible estimator is, to first order,

    Var(beta_hat) ~ Phi + Lambda,
    Lambda = sum_ij W_ij * Phi (Q_ij - P_i Phi P_j) Phi,

with ``W = Cov(theta_hat)``, ``P_i = -X' V^-1 (dV/dtheta_i) V^-1 X`` and
``Q_ij = X' V^-1 (dV/dtheta_i) V^-1 (dV/dtheta_j) V^-1 X``. Here
``V = Sigma (x) I_n`` and ``Sigma = F F' + diag(D)``, so every one of those is
``X' (M (x) I) X`` for a ``K x K`` matrix ``M`` -- the block assembly the GLS
normal matrix already uses.

What this is and is not. Evaluated at the true ``(F, D)`` on the test suite's
Monte Carlo fixture, ``Phi + Lambda`` reproduces the actual spread of the
feasible estimator to 0.98 (with ``W`` from the expected information) and
0.995 (with the true ``W``): the correction is right. Evaluated at the
estimated ``(F_hat, D_hat)`` it lands at 0.89 at ``n = 20`` against 0.85 for
the df-rescaled plug-in, because ``Phi_hat`` itself is biased low by Jensen's
inequality on a noisy ``Sigma_hat``, and that bias is not what ``Lambda``
corrects. Kenward and Roger's (1997) second-order expansion is meant to
correct it and does not here: it predicts the bias with the wrong sign at
this nuisance-parameter ratio and made the estimate worse (0.77). That is the
documented failure mode -- Kenward and Roger (2009, CSDA 53) report the 1997
formula "does not perform as well" for covariance structures nonlinear in
their parameters, and the ``mmrm`` package notes the adjusted estimate is not
invariant to reparameterising ``theta``. So this module implements the
first-order term only, which is what SAS ships as ``DDFM=KR(FIRSTORDER)`` and
applies to its factor-analytic ``TYPE=FA0()`` structure. The calibrated object
at small ``n`` remains the bootstrap.

The information is the ML expected information. Kenward and Roger derive for
REML, whose information carries two extra O(1) terms against this one's O(n);
the fit here is ML, so the ML information is the consistent choice.

Rotation. ``F -> F Q`` leaves ``Sigma`` fixed for orthogonal ``Q``, so the
information for ``theta`` is singular in ``k(k-1)/2`` directions. Along them
``dSigma/dtheta`` vanishes, so ``P`` and ``Q`` vanish and ``Lambda`` does not
depend on how ``W`` is completed there: two generalised inverses differ by a
term supported on the null space, which the vanishing derivatives annihilate.
``W`` is taken as the pseudo-inverse of rank ``r - k(k-1)/2``, and the tests
check ``Lambda`` is invariant to rotating ``F_hat`` to machine precision. (The
same argument does not cover the second-derivative term, which is one more
reason it is not used.)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .ops import GramBlocks, assemble_blocks, gram_blocks

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "covariance_information",
    "kh_correction",
    "lambda_from_derivatives",
    "sigma_derivatives",
]


def sigma_derivatives(F: np.ndarray) -> np.ndarray:
    """``dSigma/dtheta`` for ``theta = (vec F, D)``, stacked as ``(r, K, K)``.

    ``dSigma/dF_ab = e_a f_b' + f_b e_a'`` and ``dSigma/dD_a = e_a e_a'``.
    ``vec F`` is column-major, so index ``b*K + a`` is ``F[a, b]`` and the
    last ``K`` entries are ``D``. Only ``F`` enters; ``D`` is not needed
    because its derivative does not depend on its value.

    Args:
        F: Factor loadings, ``(K, k)``.

    Returns:
        The stacked derivatives, ``(K*k + K, K, K)``.
    """
    K, k = F.shape
    out = np.zeros((K * k + K, K, K))
    for b in range(k):
        for a in range(K):
            i = b * K + a
            out[i, a, :] += F[:, b]
            out[i, :, a] += F[:, b]
    for a in range(K):
        out[K * k + a, a, a] = 1.0
    return out


def covariance_information(Sigma: np.ndarray, dS: np.ndarray, n: int) -> np.ndarray:
    """Expected Fisher information for ``theta`` under ``y ~ N(., Sigma (x) I_n)``.

    ``I_ij = (n / 2) tr(Sigma^-1 dSigma_i Sigma^-1 dSigma_j)``, the standard
    result for Gaussian covariance parameters, with the ``n`` because each of
    the ``n`` rows contributes independently. Takes ``Sigma`` and its
    derivatives rather than ``(F, D)`` so the same code serves any
    parameterisation; the tests use it with an unstructured ``Sigma``.

    Args:
        Sigma: The ``K x K`` covariance.
        dS: Stacked derivatives ``(r, K, K)``.
        n: Number of rows.

    Returns:
        The ``(r, r)`` information matrix.
    """
    Sinv = np.linalg.inv(Sigma)
    r, K, _ = dS.shape
    # A_i = Sigma^-1 dSigma_i, batched. tr(A_i A_j) = <vec A_i, vec A_j'>, so the
    # whole r x r matrix is one (r, K^2) @ (K^2, r) product. Written as matmuls
    # rather than einsum: an unoptimised einsum here cost eleven seconds at K=60.
    A = Sinv @ dS
    A_flat = A.reshape(r, K * K)
    At_flat = A.transpose(0, 2, 1).reshape(r, K * K)
    return 0.5 * n * (A_flat @ At_flat.T)


def _pinv_rank(info: np.ndarray, rank: int) -> tuple[np.ndarray, np.ndarray]:
    """Eigen-decompose a symmetric PSD matrix and invert its top ``rank`` eigenvalues.

    Returns the inverse eigenvalues and eigenvectors, so callers can use the
    factored form ``W = V diag(w) V'`` directly.

    Args:
        info: Symmetric positive semi-definite matrix.
        rank: Number of eigenvalues to keep.

    Returns:
        ``(w, V)`` with ``w`` of length ``rank`` and ``V`` of shape ``(r, rank)``.
    """
    vals, vecs = np.linalg.eigh(info)
    order = np.argsort(vals)[::-1][:rank]
    return 1.0 / vals[order], vecs[:, order]


def lambda_from_derivatives(
    Xs: Sequence[np.ndarray],
    Sigma: np.ndarray,
    dS: np.ndarray,
    *,
    rank: int | None = None,
    lam_B: float = 0.0,
    gram: GramBlocks | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """``(Phi, Lambda)`` for any covariance parameterisation given its derivatives.

    Uses the structure that keeps this affordable at large ``K``. With
    ``W = sum_s w_s v_s v_s'`` and ``Delta_s = sum_i v_si dSigma_i``,

        sum_ij W_ij Q_ij        = X'(Sigma^-1 T Sigma^-1 (x) I) X,
                                  T = sum_s w_s Delta_s Sigma^-1 Delta_s
        sum_ij W_ij P_i Phi P_j = sum_s w_s Ptilde_s Phi Ptilde_s,
                                  Ptilde_s = -X'(Sigma^-1 Delta_s Sigma^-1 (x) I) X

    so the ``r^2`` double sum becomes ``r`` block assemblies. The brute-force
    double loop is kept in the tests as the reference.

    Args:
        Xs: One design matrix per equation.
        Sigma: The ``K x K`` covariance.
        dS: Stacked derivatives ``(r, K, K)``.
        rank: Rank of the information matrix, if it is singular by
            construction. Defaults to full rank.
        lam_B: Ridge added to the GLS normal matrix, as the fit used it.
        gram: Output of :func:`alsgls.ops.gram_blocks`, if already computed.

    Returns:
        ``Phi`` and ``Lambda``, both ``(p_total, p_total)``.
    """
    n = Xs[0].shape[0]
    if gram is None:
        gram = gram_blocks(Xs)

    Sinv = np.linalg.inv(Sigma)
    normal = assemble_blocks(Sinv, gram)
    if lam_B > 0:
        normal = normal + lam_B * np.eye(normal.shape[0])
    Phi = np.linalg.inv(normal)

    info = covariance_information(Sigma, dS, n)
    w, V = _pinv_rank(info, dS.shape[0] if rank is None else rank)

    r, K, _ = dS.shape
    kept = w.size

    # Delta_s = sum_i V[i, s] dSigma_i, one K x K matrix per kept eigenvector.
    Delta = (V.T @ dS.reshape(r, K * K)).reshape(kept, K, K)

    # Term 1 through a single K x K matrix: T = sum_s w_s Delta_s Sigma^-1 Delta_s,
    # as one (K, kept*K) @ (kept*K, K) product.
    B = (w[:, None, None] * Delta) @ Sinv
    T = B.transpose(1, 0, 2).reshape(K, kept * K) @ Delta.reshape(kept * K, K)
    term1 = assemble_blocks(Sinv @ T @ Sinv, gram)

    # Term 2 needs Phi between the two P's, so it is a sum over s.
    term2 = np.zeros_like(Phi)
    for s_idx in range(w.size):
        P_s = -assemble_blocks(Sinv @ Delta[s_idx] @ Sinv, gram)
        term2 += w[s_idx] * (P_s @ Phi @ P_s)

    return Phi, Phi @ (term1 - term2) @ Phi


def kh_correction(
    Xs: Sequence[np.ndarray],
    F: np.ndarray,
    D: np.ndarray,
    *,
    lam_B: float = 0.0,
    gram: GramBlocks | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """``(Phi, Lambda)`` for ``Sigma = F F' + diag(D)``.

    The information is singular in ``k(k-1)/2`` rotation directions, along
    which the derivatives vanish, so ``W`` is taken at that reduced rank.

    Args:
        Xs: One design matrix per equation.
        F: Factor loadings, ``(K, k)``.
        D: Diagonal variances, length ``K``.
        lam_B: Ridge added to the GLS normal matrix, as the fit used it.
        gram: Output of :func:`alsgls.ops.gram_blocks`, if already computed.

    Returns:
        ``Phi`` and ``Lambda``, both ``(p_total, p_total)``. The corrected
        covariance is their sum.
    """
    k = F.shape[1]
    dS = sigma_derivatives(F)
    return lambda_from_derivatives(
        Xs,
        F @ F.T + np.diag(D),
        dS,
        rank=dS.shape[0] - k * (k - 1) // 2,
        lam_B=lam_B,
        gram=gram,
    )
