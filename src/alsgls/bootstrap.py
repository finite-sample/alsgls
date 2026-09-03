"""Bootstrap inference for the low-rank GLS fit.

The plug-in variance ``(X' Sigma_hat^-1 X)^-1`` is the variance of a GLS
estimator whose covariance is *known*. Here ``Sigma_hat`` is estimated from the
same data, and that costs in two ways that Freedman and Peters (1984, JASA 79,
97-106, Theorem 1) order as

    var(beta_FGLS)  >  var(beta_GLS)  >  E[reported var].

The right inequality is Jensen: ``(X' Sigma^-1 X)^-1`` is concave in ``Sigma``,
so plugging in a noisy ``Sigma_hat`` biases the reported variance down even when
``Sigma_hat`` is unbiased. The left inequality is the extra spread feasible GLS
has over infeasible GLS. No formula evaluated at any single ``Sigma_hat`` can
see the left inequality; only refitting ``Sigma`` on resampled data can.

So every replicate here refits everything -- ``F``, ``D`` and ``beta`` -- and
the object that is calibrated is not the bootstrap standard error but the
studentised interval: ``t*(b) = (beta(b) - beta_hat) / se(b)`` with ``se(b)``
the replicate's own plug-in standard error. The plug-in is biased the same way
in the bootstrap world as in the real one, so the quantiles of ``t*`` absorb the
bias (Rilstone and Veall 1996; Fiebig and Kim 2000; Horowitz 2019). And
``se(b)`` is a free byproduct of the replicate's fit, so studentising costs
nothing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

from .als import als_gls
from .ops import XB_from_Blist, compute_XtSigmaInvX, df_rescaled

__all__ = ["SCHEMES", "BootstrapResults", "bootstrap_system"]

#: Resampling schemes. Pairs resampling is deliberately absent: at small ``n`` a
#: with-replacement draw has about 63% distinct rows, and a factor covariance
#: fitted to a dozen distinct K-vectors is not a replicate of anything.
SCHEMES = ("parametric", "wild", "residual")


@dataclass(frozen=True)
class BootstrapResults:
    """Bootstrap distribution of the coefficients and the inference built on it."""

    #: The parent fit's coefficients, length ``p_total``.
    params: np.ndarray
    #: The parent fit's plug-in standard errors.
    se_plugin: np.ndarray
    #: Coefficients from each replicate, ``(B, p_total)``.
    estimates: np.ndarray
    #: Studentised deviations ``(beta(b) - beta_hat) / se(b)``, ``(B, p_total)``.
    tstats: np.ndarray
    #: Which resampling scheme produced them.
    method: str
    #: The seed the replicate stream was spawned from.
    seed: int | None
    #: Residual degrees of freedom of the parent fit.
    df: int

    @property
    def B(self) -> int:
        """Number of replicates."""
        return int(self.estimates.shape[0])

    @property
    def bse(self) -> np.ndarray:
        """Bootstrap standard errors: the spread of ``beta`` across replicates.

        Better than the plug-in, but still biased down (Freedman and Peters
        measured 20-30% in their design), because each replicate is generated
        from a ``Sigma_hat`` that is itself too small. The interval, not this
        number, is the calibrated object.
        """
        return self.estimates.std(axis=0, ddof=1)

    @property
    def tvalues(self) -> np.ndarray:
        """``params / se_plugin``, the statistic the bootstrap-t p-value refers to."""
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(self.se_plugin > 0, self.params / self.se_plugin, 0.0)

    def conf_int(self, alpha: float = 0.05) -> np.ndarray:
        """Percentile-t (bootstrap-t) confidence intervals.

        ``[beta_hat - t*_{1-alpha/2} se, beta_hat - t*_{alpha/2} se]`` with the
        quantiles taken from the studentised replicates and ``se`` the parent
        plug-in. Asymmetric when the studentised distribution is.

        Args:
            alpha: Significance level.

        Returns:
            ``(p_total, 2)`` array of lower and upper bounds.
        """
        alpha = _check_alpha(alpha)
        lo_q = np.quantile(self.tstats, alpha / 2, axis=0)
        hi_q = np.quantile(self.tstats, 1 - alpha / 2, axis=0)
        return np.column_stack(
            [self.params - hi_q * self.se_plugin, self.params - lo_q * self.se_plugin]
        )

    @property
    def pvalues(self) -> np.ndarray:
        """Two-sided bootstrap-t p-values for ``H0: beta = 0``.

        The share of replicates whose studentised deviation is at least as
        large in absolute value as the observed ``t``, with the ``+1``
        continuity adjustment so ``B`` replicates cannot report zero.
        """
        t_obs = np.abs(self.tvalues)
        exceed = (np.abs(self.tstats) >= t_obs[None, :]).sum(axis=0)
        return (exceed + 1) / (self.B + 1)

    def summary(self, alpha: float = 0.05) -> str:
        """Text summary in the same layout as ``ALSGLSSystemResults.summary``.

        Args:
            alpha: Significance level for the intervals.

        Returns:
            The formatted table.
        """
        ci = self.conf_int(alpha)
        sep = "=" * 78
        lines = [
            sep,
            f"     Bootstrap-t inference ({self.method}, B={self.B}, seed={self.seed})",
            sep,
            f"{'Parameter':<12} {'Coef':>10} {'Boot SE':>10} {'t':>8} {'P>|t|':>8} "
            f"{'[' + str(alpha / 2):>10} {str(1 - alpha / 2) + ']':>10}",
            "-" * 78,
        ]
        bse, t, p = self.bse, self.tvalues, self.pvalues
        lines.extend(
            f"{i:<12} {self.params[i]:>10.4f} {bse[i]:>10.4f} {t[i]:>8.2f} "
            f"{p[i]:>8.4f} {ci[i, 0]:>10.4f} {ci[i, 1]:>10.4f}"
            for i in range(self.params.size)
        )
        lines.append(sep)
        return "\n".join(lines)


def _check_alpha(alpha: float) -> float:
    if not isinstance(alpha, (int, float)) or isinstance(alpha, bool):
        raise ValueError(f"alpha must be a number in (0, 1), got {alpha!r}")
    if not np.isfinite(alpha) or not (0.0 < float(alpha) < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    return float(alpha)


def _plugin_se(Xs: list[np.ndarray], F: np.ndarray, D: np.ndarray, lam_B_eff: float):
    n = Xs[0].shape[0]
    Fc, Dc = df_rescaled(F, D, n, [X.shape[1] for X in Xs])
    cov = np.linalg.inv(compute_XtSigmaInvX(Xs, Fc, Dc, lam_B=lam_B_eff))
    return np.sqrt(np.maximum(np.diag(cov), 0.0))


def _draw_factory(
    method: str,
    fitted: np.ndarray,
    resid: np.ndarray,
    F: np.ndarray,
    D: np.ndarray,
    p_list: list[int],
) -> Callable[[np.random.Generator], np.ndarray]:
    """Return a function that generates one replicate response matrix."""
    n, K = fitted.shape
    k = F.shape[1]
    if method == "parametric":

        def draw(rng: np.random.Generator) -> np.ndarray:
            noise = rng.standard_normal((n, k)) @ F.T
            noise += rng.standard_normal((n, K)) * np.sqrt(D)[None, :]
            return fitted + noise

        return draw

    # The residuals are shrunk by fitting; inflate each equation's by
    # sqrt(n / (n - p_j)) so their scale matches the disturbances they stand in
    # for. Centring keeps the replicate mean where the parent's is.
    scale = np.sqrt(n / (n - np.asarray(p_list, dtype=float)))
    R = (resid - resid.mean(axis=0)) * scale[None, :]

    if method == "wild":
        # One scalar per row, applied to the whole K-vector: preserves each
        # row's outer product in expectation and so the cross-equation
        # covariance, while imposing no distribution on the errors.
        def draw(rng: np.random.Generator) -> np.ndarray:
            v = rng.choice([-1.0, 1.0], size=(n, 1))
            return fitted + R * v

        return draw

    if method == "residual":
        # Resample whole rows, so the K residuals of one observation travel
        # together and the cross-equation structure survives.
        def draw(rng: np.random.Generator) -> np.ndarray:
            return fitted + R[rng.integers(0, n, n)]

        return draw

    raise ValueError(f"method must be one of {SCHEMES}, got {method!r}")


def bootstrap_system(
    Xs: list[np.ndarray],
    Y: np.ndarray,
    *,
    B_list: list[np.ndarray],
    F: np.ndarray,
    D: np.ndarray,
    k: int,
    als_kwargs: dict[str, Any],
    B: int = 999,
    method: str = "parametric",
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run ``B`` full refits on resampled data and collect the coefficients.

    Args:
        Xs: The parent fit's design matrices.
        Y: The parent fit's responses.
        B_list: The parent fit's coefficients, one block per equation.
        F: The parent fit's factor loadings.
        D: The parent fit's diagonal variances.
        k: Factor rank. Each replicate's plug-in standard error charges the
            effective ridge that replicate's own fit resolved, exactly as the
            parent's does.
        als_kwargs: Keyword arguments forwarded to :func:`als_gls`, so the
            replicate is fitted exactly as the parent was.
        B: Number of replicates.
        method: One of :data:`SCHEMES`.
        seed: Seed for the replicate stream. Replicate ``b`` is a function of
            ``(seed, b)`` alone, so raising ``B`` adds replicates without
            changing the existing ones.

    Returns:
        ``(estimates, tstats)``, each ``(B, p_total)``.

    Raises:
        ValueError: If ``B`` is not a positive integer or ``method`` is unknown.
    """
    if not isinstance(B, int) or isinstance(B, bool) or B < 1:
        raise ValueError(f"B must be a positive integer, got {B!r}")
    if method not in SCHEMES:
        raise ValueError(f"method must be one of {SCHEMES}, got {method!r}")

    p_list = [X.shape[1] for X in Xs]
    beta_hat = np.concatenate([b.ravel() for b in B_list])
    fitted = XB_from_Blist(Xs, B_list)
    draw = _draw_factory(method, fitted, Y - fitted, F, D, p_list)

    # Warm-start every replicate from the parent's D. The replicate's answer is
    # near the parent's, so this skips most of the inner alternation: measured
    # 180 ms -> 3 ms per replicate at n=20, where a cold start crawls.
    kwargs = {**als_kwargs, "init_D": D}

    estimates = np.empty((B, beta_hat.size))
    tstats = np.empty((B, beta_hat.size))
    for b, child in enumerate(np.random.SeedSequence(seed).spawn(B)):
        rng = np.random.default_rng(child)
        Yb = draw(rng)
        Bb, Fb, Db, _, info_b = als_gls(Xs, Yb, k=k, **kwargs)
        beta_b = np.concatenate([bb.ravel() for bb in Bb])
        se_b = _plugin_se(Xs, Fb, Db, info_b["lam_B_eff"])
        estimates[b] = beta_b
        with np.errstate(divide="ignore", invalid="ignore"):
            tstats[b] = np.where(se_b > 0, (beta_b - beta_hat) / se_b, 0.0)
    return estimates, tstats
