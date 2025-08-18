"""
lowrank_gls: lightweight low-rank+diagonal GLS/SUR

Public API
----------
Solvers
    - als_gls:  Matrix-free Alternating Least Squares for low-rank GLS/SUR.
    - em_gls:   Dense EM baseline (optionally warm-started).

Calibration
    - calibrate_alpha_gamma_s_cv3_conservative: Cross-validated α–γ–τ–s calibration.
    - finalize_on_dataset: Apply (α, γ, τ, s) on a target split to get (F*, D*).
    - apply_scale: Scale (F, D) by s.
    - inflate_corr_fixed_diag: Inflate factor part while keeping per-equation variances.
    - apply_alpha: Rescale diagonal D by α.

Diagnostics
    - coverage_and_mahalanobis: z-coverage, χ² tails, and whitening diagnostics.
    - whitening_matrix: Empirical whitening quality matrix.
    - top_eigvecs_sym: Top eigenpairs for symmetric matrices (used in τ-boost).

Utilities
    - penalized_nll, test_nll: Penalized/train and test NLL computations.
    - predict_Y: Build fitted Y from list-of-designs and coefficient list.
    - mse: Mean squared error helper.

Version
    - __version__: semantic version of the library.

Typical usage
-------------
>>> from lowrank_gls import als_gls, calibrate_alpha_gamma_s_cv3_conservative, finalize_on_dataset
>>> B, F, D, mem_mb, sec = als_gls(Xs_tr, Y_tr, k=6, lam_F=1e-3, lam_B=1e-3)
>>> alpha, gamma, tau, s, use_eig = calibrate_alpha_gamma_s_cv3_conservative(Xs_tr, Y_tr, B, F, D)
>>> F_fin, D_fin = finalize_on_dataset(Xs_te, Y_te, B, F, D, alpha, gamma, tau, s, use_eig=use_eig)

Note: Data simulation and experiment scripts live in `experiments/` and are not
imported here to keep the package lean and dependency-free.
"""

from .als_solver import (
    als_gls,
    em_gls,
)
from .calibration import (
    calibrate_alpha_gamma_s_cv3_conservative,
    finalize_on_dataset,
    apply_scale,
    inflate_corr_fixed_diag,
    apply_alpha,
)
from .diagnostics import (
    coverage_and_mahalanobis,
    whitening_matrix,
    top_eigvecs_sym,
)
from .numerics import (
    penalized_nll,
    test_nll,
    predict_Y,
    mse,
)

__all__ = [
    # Solvers
    "als_gls", "em_gls",
    # Calibration
    "calibrate_alpha_gamma_s_cv3_conservative", "finalize_on_dataset",
    "apply_scale", "inflate_corr_fixed_diag", "apply_alpha",
    # Diagnostics
    "coverage_and_mahalanobis", "whitening_matrix", "top_eigvecs_sym",
    # Likelihood & metrics
    "penalized_nll", "test_nll", "predict_Y", "mse",
]

__version__ = "0.1.0"
