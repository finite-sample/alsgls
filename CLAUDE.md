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
# Run ALS vs EM comparison
python examples/compare_als_vs_em.py
```

## Architecture

### Core Modules

- **`alsgls/als.py`**: Main ALS solver (`als_gls()`) using matrix-free conjugate gradient and Woodbury matrix identity for O(Kk) memory complexity
- **`alsgls/em.py`**: Baseline EM solver (`em_gls()`) that builds full K×K covariance matrices for comparison
- **`alsgls/ops.py`**: Core linear algebra operations including Woodbury matrix utilities, matrix-free operators, and conjugate gradient solver
- **`alsgls/sim.py`**: Data simulation functions (`simulate_sur()`, `simulate_gls()`) for testing and benchmarking
- **`alsgls/metrics.py`**: Evaluation metrics (MSE, negative log-likelihood per row)

### Key Algorithms

The package implements two approaches to low-rank+diagonal GLS:

1. **ALS Solver** (`als_gls`):
   - Uses Woodbury matrix identity: Σ⁻¹ = D⁻¹ - D⁻¹F(I + F^T D⁻¹F)⁻¹F^T D⁻¹
   - Matrix-free conjugate gradient for β-updates avoiding dense normal equations
   - Memory complexity: O(Kk) where k << K
   - Alternates between updating regression coefficients (β) and factor loadings (F, D)

2. **EM Solver** (`em_gls`):
   - Builds explicit K×K covariance inverse for comparison
   - Memory complexity: O(K²)
   - Provided as baseline to demonstrate memory savings of ALS approach

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
- Comparison between ALS and EM solutions

The `als_sim/` directory contains Jupyter notebooks with detailed experiments and mathematical background.