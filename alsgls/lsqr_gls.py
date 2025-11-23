"""High-accuracy GLS solves via LSQR/LSMR without squaring the condition number.

This module exposes utilities for solving the weighted least-squares problem

    min_beta (y - X beta)^T Σ^{-1} (y - X beta)

in the common "low-rank plus diagonal" setting used across the package.  The
implementation avoids the explicit normal equations that the in-package CG
routine currently relies on and instead wraps the design matrix inside a
``scipy.sparse.linalg.LinearOperator`` so that the LSQR/LSMR Krylov solvers can
be used directly.  In ill-conditioned designs this provides noticeably better
convergence and is less sensitive to round-off.

The implementation follows the write-up in the project documentation and is
careful to avoid forming dense K×K matrices except for a skinny SVD of the
Woodbury core.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.linalg import svd

try:  # pragma: no cover - import guard exercised in tests
    from scipy.sparse.linalg import LinearOperator, lsmr, lsqr
except Exception as exc:  # pragma: no cover - exercised when SciPy missing
    raise ImportError("The lsqr_gls module requires scipy to be installed.") from exc


ArrayLike = np.ndarray | Sequence[float]


@dataclass
class WoodburyWeight:
    """Row-wise operator ``W`` satisfying ``W.T @ W = Σ^{-1}``.

    Parameters
    ----------
    d:
        Diagonal of ``D`` in ``Σ = D + F F^T``.
    F:
        Optional factor loadings.  If ``None`` or empty then ``Σ`` is purely
        diagonal and the action reduces to simple scaling by ``D^{-1/2}``.
    d_floor:
        Lower bound applied element-wise to ``d`` to avoid singularities.
    sv_tol:
        Relative tolerance used to trim tiny singular values when computing the
        skinny SVD of ``U = D^{-1/2} F``.
    """

    d: ArrayLike
    F: ArrayLike | None
    d_floor: float = 1e-12
    sv_tol: float = 1e-12

    def __post_init__(self) -> None:
        d = np.asarray(self.d, dtype=float).copy()
        d[d < self.d_floor] = self.d_floor
        self.d = d
        self._D_isqrt = 1.0 / np.sqrt(d)

        if self.F is None:
            self._Q = None
            self._S_diag = None
            return

        F = np.asarray(self.F, dtype=float)
        if F.size == 0:
            self._Q = None
            self._S_diag = None
            return

        # Compute U = D^{-1/2} F and take an economy SVD.
        U = self._D_isqrt[:, None] * F
        Q, s, _ = svd(U, full_matrices=False)
        if s.size == 0:
            self._Q = None
            self._S_diag = None
            return

        s_max = s.max(initial=0.0)
        keep = s > (self.sv_tol * max(1.0, s_max))
        if not np.any(keep):
            self._Q = None
            self._S_diag = None
            return

        self._Q = Q[:, keep]
        s = s[keep]
        self._S_diag = 1.0 / np.sqrt(1.0 + s * s)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def _apply_m_inv_half(self, cols: np.ndarray) -> np.ndarray:
        if self._Q is None:
            return cols

        proj = self._Q.T @ cols
        p_perp = cols - (self._Q @ proj)
        q_term = self._Q @ (self._S_diag[:, None] * proj)
        return p_perp + q_term

    def W_apply(self, T: np.ndarray) -> np.ndarray:
        """Apply ``W`` to an ``(N, K)`` array row-by-row."""

        T = np.asarray(T, dtype=float)
        cols = (self._D_isqrt[:, None]) * T.T
        out = self._apply_m_inv_half(cols)
        return out.T

    def WT_apply(self, T: np.ndarray) -> np.ndarray:
        """Apply the adjoint ``W.T`` to an ``(N, K)`` array row-by-row."""

        T = np.asarray(T, dtype=float)
        cols = self._apply_m_inv_half(T.T)
        cols = (self._D_isqrt[:, None]) * cols
        return cols.T


class GLSLinearOperator(LinearOperator):
    """Linear operator representing ``A = W X`` for LSQR/LSMR."""

    def __init__(
        self,
        X_dot: Callable[[np.ndarray], np.ndarray],
        X_Tdot: Callable[[np.ndarray], np.ndarray],
        W: WoodburyWeight,
        *,
        N: int,
        K: int,
        P: int,
    ) -> None:
        self._X_dot = X_dot
        self._X_Tdot = X_Tdot
        self._W = W
        self._N = int(N)
        self._K = int(K)
        self._P = int(P)
        super().__init__(dtype=float, shape=(self._N * self._K, self._P))

    def _matvec(self, v: np.ndarray) -> np.ndarray:
        Y = self._X_dot(v)
        WY = self._W.W_apply(Y)
        return WY.reshape(self._N * self._K)

    def _rmatvec(self, u: np.ndarray) -> np.ndarray:
        U = np.asarray(u, dtype=float).reshape(self._N, self._K)
        WT_U = self._W.WT_apply(U)
        return self._X_Tdot(WT_U)


def solve_gls_weighted(
    X_dot: Callable[[np.ndarray], np.ndarray],
    X_Tdot: Callable[[np.ndarray], np.ndarray],
    y: np.ndarray,
    d: ArrayLike,
    F: ArrayLike | None,
    *,
    method: str = "lsmr",
    atol: float = 1e-10,
    btol: float = 1e-10,
    conlim: float = 1e8,
    maxiter: int | None = None,
    verbose: bool = False,
):
    """Solve ``argmin_beta || W (X beta - y) ||_2`` via LSQR or LSMR.

    The design is provided through matrix-free callbacks ``X_dot`` and
    ``X_Tdot`` matching the interfaces used throughout the rest of the
    ``alsgls`` package.  The solver works directly with the GLS geometry and
    therefore avoids squaring the condition number of ``X``.

    Returns
    -------
    beta:
        The concatenated coefficient vector.
    info:
        Diagnostics returned by the underlying Krylov solver.
    """

    y = np.asarray(y, dtype=float)
    N, K = y.shape

    probe = X_Tdot(np.zeros_like(y))
    P = int(np.asarray(probe).shape[0])

    W = WoodburyWeight(
        d=np.asarray(d, dtype=float),
        F=None if F is None else np.asarray(F, dtype=float),
    )
    A = GLSLinearOperator(X_dot, X_Tdot, W, N=N, K=K, P=P)
    b = W.W_apply(y).reshape(N * K)

    if method == "lsmr":
        sol = lsmr(
            A, b, atol=atol, btol=btol, conlim=conlim, maxiter=maxiter, show=verbose
        )
        beta = sol[0]
        info = {
            "method": "lsmr",
            "istop": sol[1],
            "iters": sol[2],
            "normr": sol[3],
            "normAres": sol[4],
            "normA": sol[5],
            "condA": sol[6],
            "normx": sol[7],
        }
    elif method == "lsqr":
        sol = lsqr(
            A, b, atol=atol, btol=btol, conlim=conlim, iter_lim=maxiter, show=verbose
        )
        beta = sol[0]
        info = {
            "method": "lsqr",
            "istop": sol[1],
            "iters": sol[2],
            "r1norm": sol[3],
            "r2norm": sol[4],
            "anorm": sol[5],
            "acond": sol[6],
            "arnorm": sol[7],
            "xnorm": sol[8],
        }
    else:
        raise ValueError("method must be either 'lsmr' or 'lsqr'")

    return beta, info


def make_block_design_ops(X_blocks: Sequence[np.ndarray]):
    """Build ``X_dot``/``X_Tdot`` callbacks for SUR-style block designs."""

    X_blocks = [np.asarray(X, dtype=float) for X in X_blocks]
    K = len(X_blocks)
    if K == 0:
        raise ValueError("X_blocks must contain at least one block")

    N = X_blocks[0].shape[0]
    if any(X.shape[0] != N for X in X_blocks):
        raise ValueError("All blocks in X_blocks must have the same number of rows")

    p_sizes = [X.shape[1] for X in X_blocks]
    cuts = np.cumsum([0, *p_sizes])

    def X_dot(beta: np.ndarray) -> np.ndarray:
        beta = np.asarray(beta, dtype=float)
        if beta.shape[0] != cuts[-1]:
            raise ValueError("beta has incorrect length for the provided blocks")
        Y = np.empty((N, K), dtype=float)
        for j, Xj in enumerate(X_blocks):
            bj = beta[cuts[j] : cuts[j + 1]]
            Y[:, j] = (Xj @ bj).ravel()
        return Y

    def X_Tdot(U: np.ndarray) -> np.ndarray:
        U = np.asarray(U, dtype=float)
        if U.shape != (N, K):
            raise ValueError("U must have shape (N, K)")
        out = np.empty(cuts[-1], dtype=float)
        for j, Xj in enumerate(X_blocks):
            uj = U[:, j]
            out[cuts[j] : cuts[j + 1]] = (Xj.T @ uj).ravel()
        return out

    return X_dot, X_Tdot


__all__ = [
    "WoodburyWeight",
    "GLSLinearOperator",
    "solve_gls_weighted",
    "make_block_design_ops",
]
