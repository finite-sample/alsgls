# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is `alsgls`, a Python package implementing Alternating Least Squares (ALS) for low-rank+diagonal GLS estimation. The package provides memory-efficient solutions for Seemingly Unrelated Regressions (SUR) and other GLS problems by using low-rank factor models instead of dense covariance matrices.

## Development Commands

### Testing
```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_als.py

# Run with verbose output
python -m pytest tests/ -v
```

### Installation
```bash
# Development installation (editable)
pip install -e .

# Standard installation
pip install .
```

### Examples
```bash
# See examples/ for runnable scripts
ls examples/
```

## Architecture

### Core Modules

- **`alsgls/als.py`**: Main solver (`als_gls()`). Alternates a matrix-free conjugate-gradient β-step with the closed-form factor-analysis Σ-step (`_sigma_step`), using the Woodbury identity throughout for O(Kk) memory
- **`alsgls/api.py`**: `ALSGLS` (scikit-learn style) and `ALSGLSSystem` (statsmodels style)
- **`alsgls/rank_selection.py`**: `select_rank_bic()` and `select_rank_cv()`
- **`alsgls/ops.py`**: Core linear algebra operations including Woodbury matrix utilities, matrix-free operators, and conjugate gradient solver
- **`alsgls/sim.py`**: Data simulation functions (`simulate_sur()`, `simulate_gls()`) for testing and benchmarking
- **`alsgls/metrics.py`**: Evaluation metrics (MSE, negative log-likelihood per row)

### Key Algorithms

`als_gls` alternates two exact steps until the likelihood stops falling:

1. **β-step**: matrix-free conjugate gradient at the current Σ, using the
   Woodbury identity Σ⁻¹ = D⁻¹ - D⁻¹F(I + F^T D⁻¹F)⁻¹F^T D⁻¹ to avoid forming
   dense normal equations. Memory O(Kk) with k << K.
2. **Σ-step**: L-BFGS-B on the profile likelihood over log D, with F
   concentrated out in closed form (Lawley: top-k right singular vectors of the
   D-standardised residuals), on residuals standardised to unit variance. This
   is what R's `factanal` does. The gradient is the envelope theorem, computed
   with the Woodbury kernels. A pure fixed-point alternation of the two closed
   forms (what sklearn does) was tried first and crawls when a diagonal
   variance is small.

**Inference.** `cov_params` is the plug-in `(X'Σ̂⁻¹X)⁻¹` after the df rescale
every SUR package applies; it treats Σ̂ as known and understates in small
samples (Freedman & Peters 1984, Theorem 1). `results.bootstrap()` refits
everything per replicate and returns percentile-t intervals, which are the
calibrated object. The Monte Carlo fixtures in `tests/test_standard_errors.py`
and `tests/test_econometrics.py` must stay at an *identified* rank
(Ledermann: `(K-k)² ≥ K+k`); they were at K=4, k=2 for a long time, which is
not.

`em_gls` was removed in 1.0; the exploratory EM code lives in `als_sim/` and is
not installed.

### Data Structure Conventions

- **X matrices**: List of feature matrices `[X₀, X₁, ..., X_{K-1}]` where `X_j` has shape `(N, p_j)`
- **Y matrix**: Response matrix of shape `(N, K)` 
- **B coefficients**: List of coefficient vectors `[B₀, B₁, ..., B_{K-1}]` where `B_j` has shape `(p_j, 1)`
- **Factor structure**: `F` (K×k loadings), `D` (K,) diagonal noise, covariance Σ = FF^T + diag(D)

### Testing Strategy

Tests focus on:
- Shape consistency of returned parameters
- MSE improvement over baseline ridge regression
- Numerical stability with different problem sizes
- Convergence: the fit must match what an independent optimiser (L-BFGS-B, and
  `sklearn.decomposition.FactorAnalysis`) reaches on the same objective
- Rank recovery: BIC must select the true rank on known-truth fixtures
- Invariances: scale equivariance under Y -> sY, Zellner's identical-regressors
  result, self-consistency of the reported likelihood

The `als_sim/` directory contains Jupyter notebooks with detailed experiments and mathematical background.