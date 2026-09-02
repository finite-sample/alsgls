# Changelog

All notable changes to alsgls will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

[Unreleased]: https://github.com/finite-sample/alsgls/compare/v2.0.0...HEAD

## [2.0.0] - 2026-09-02

Note on version history: PyPI has only ever carried 0.1.0. The 1.0.0 and 1.1.0
entries below describe work that landed on the default branch but was never
tagged or published, so for anyone installing from PyPI this release also
carries everything in those two sections.

### Changed

- **The Σ-step is now the closed-form factor-analysis update, and fitted values
  change for every user.** The factor loadings were previously fitted by
  steepest descent with a backtracking line search, which stopped improving
  after about two sweeps and left the fit 2 to 20 nats/row short of the
  likelihood the same objective reaches from the same starting point. Running
  more sweeps did not help: 500 sweeps produced bit-identical output to 4.

  Given `D`, the maximising `F` has a closed form going back to Lawley, obtained
  from the top `k` right singular vectors of the `D`-standardised residuals.
  This is what `sklearn.decomposition.FactorAnalysis` computes and what R's
  `stats::factanal` optimises over; the new implementation agrees with sklearn's
  to 1e-7 on the implied covariance and to 1e-9 on the likelihood.

  The visible consequence is rank selection. `select_rank_bic` chose 4, 8, 6, 7
  and 10 on five fixtures whose true ranks are 2, 4, 5, 3 and 6, because the
  optimiser's shortfall shrank as `k` grew and so the likelihood kept improving
  for a reason unrelated to the data. It now recovers the true rank on all five.
  The fit is also faster in wall clock, 1.4x at K=20 rising to 2.1x at K=200,
  because it converges and stops instead of running its sweep budget.

- **`lam_B` is now relative to the residual variance scale.** An absolute ridge
  made the fit depend on the units of `Y`: at the default `1e-3`, scaling `Y` by
  `1e4` moved every coefficient by 100% and the estimated correlation matrix by
  1.16. The fit is now equivariant under `Y -> sY`.

### Removed

- **`lam_F`**, from `als_gls`, `ALSGLS` and `ALSGLSSystem`. The Σ-step is the
  exact conditional solution, so there is no search direction for a penalty on
  `F` to bias. It never described a coherent estimator: the direction was
  penalised while acceptance was tested on the unpenalised likelihood, so the
  iteration could stop at a point stationary for neither, and it was the sole
  cause of the scale-dependence above.
- **`scale_correct` and `scale_floor`.** The guarded rescaling of Σ existed to
  patch the gradient step. Measured after a closed-form step, the optimal scale
  factor is 0.9999995 — a no-op.
- **`grad_F_nll`** from `alsgls.ops`, now unused.
- `info` no longer carries `accept_t`, `scale_used`, `obj_trace` or the
  `nll_sigma_trace` alias, none of which have meaning without a line search. It
  gains `sigma_iters`, `var_ref` and `lam_B_eff`.

### Fixed

- **The `(F, D)` line search froze after two sweeps.** The backtracking ladder
  proposed `(F + t*dF, D_mle(F + t*dF))`, whose `t -> 0` limit is
  `(F, D_mle(F))` rather than the incumbent `(F, D)`. The guarded scale
  correction moved `D` off that manifold, so every candidate started nats
  behind the incumbent, all 40 halvings were rejected, and `F` never moved
  again. Superseded by the closed-form step above, but fixed first so the two
  changes could be reviewed apart.
- **`beta` was stale relative to the returned `Sigma`.** The sweep ends on a
  `Sigma` step, so `beta` solved the GLS normal equations at the *previous*
  sweep's `Sigma`. This matters because `cov_params` reports
  `(X' Sigma^-1 X)^-1` as `beta`'s variance, which is only its variance when
  the two agree. `als_gls` now refreshes `beta` at the final `Sigma`.
- **`select_rank_bic` reported half the textbook BIC** (`N*nll + p/2*log N`
  where `-2*loglik + p*log N` is `2*N*nll + p*log N`), and counted parameters
  that are not free while skipping ones that are. `n_params` is now
  `K*k + K - k(k-1)/2 + sum(p_j)`: the loadings less the `k(k-1)/2` rotations
  `F -> FQ` that leave `F F^T` fixed, the diagonal variances, and the
  regression coefficients. Checked against R's `factanal`, which reports the
  complementary `df = ((K-k)^2 - K - k)/2`.
- **Out-of-domain arguments were accepted silently.** `lam_B = nan` passed the
  non-negativity guard, since `nan < 0` is False, and returned `F` at its
  initialisation with no error. `alpha` outside `(0, 1)` returned intervals
  whose lower bound exceeded their upper for every parameter. `d_floor <= 0`
  let `D` reach zero or go negative while the internals clipped at `1e-12`, so
  the returned `(F, D)` described a different and not positive definite `Sigma`
  from the one every reported number used. `sweeps=True` passed the
  positive-integer check and ran one sweep. Non-finite data surfaced as
  `SVD did not converge`. All are now rejected at the public boundary.
- **`ALSGLS.score` documented the negative mean squared error** and returned
  the negative log-likelihood per row.
- **`d_floor` documented as an absolute variance** while being applied as a
  fraction of the mean residual variance. The relative form is correct and
  deliberate; only the docstring was wrong.
- Documentation taught `em_gls()`, removed in 1.0, and used four argument names
  `als_gls` has never accepted (`max_iter`, `tol`, `verbose`, and
  `simulate_sur` kwargs that do not exist), so every example on three pages
  failed on the import.
- `cg_solve`'s positive-definiteness guard tested the previous iteration's
  value while its message quoted the current one.

### Changed (infrastructure)

- Adopted the py-canon fleet standard: src/ layout, shared CI/docs/release
  workflows, ruff + pyright + pydoclint linting (mypy retired), and
  tag-driven trusted publishing.
- `scikit-learn` added as a test-only dependency, used as an independent
  implementation to check the `Sigma` step against.

[2.0.0]: https://github.com/finite-sample/alsgls/releases/tag/v2.0.0

## [1.1.0] - 2025-03-31

### New Features
- **Rank selection methods**: `rank="bic"` and `rank="cv"` for automatic rank selection
- **Real data example**: Fama-French 49 industry portfolios demonstration
- **Formal methods documentation**: Rigorous mathematical foundations

### Improvements
- Replaced heuristic ALS F-update with gradient-based descent
- Added `select_rank_bic()` and `select_rank_cv()` functions
- New parameters: `rank_candidates`, `cv_folds`, `cv_random_state`

### Documentation
- New `formal_methods.md` with convergence proofs and complexity analysis
- New `real_world_applications.md` with finance example

## [1.0.0] - 2024-12-21

### 🚨 BREAKING CHANGES
This is a major release with significant API changes that improve type safety, performance, and maintainability.

#### Removed Functions
- **`em_gls()`** - Dense EM baseline algorithm removed entirely
- **`woodbury_pieces()`** - Deprecated function that computed explicit inverse removed

#### API Changes  
- **`apply_siginv_to_matrix()`** - `C_inv` parameter removed, `C_chol` now required
  - Before: `apply_siginv_to_matrix(M, F, D)` or `apply_siginv_to_matrix(M, F, D, C_inv=C_inv)`
  - After: `apply_siginv_to_matrix(M, F, D, C_chol=C_chol)` (Cholesky factor required)

#### Migration Guide
- Replace `em_gls()` calls with `als_gls()` - they provide equivalent statistical results
- Update `apply_siginv_to_matrix()` calls to use `woodbury_chol()` for the Cholesky factor:
  ```python
  # Old approach
  Dinv, C_inv = woodbury_pieces(F, D)
  result = apply_siginv_to_matrix(M, F, D, C_inv=C_inv)

  # New approach
  Dinv, C_chol = woodbury_chol(F, D)
  result = apply_siginv_to_matrix(M, F, D, C_chol=C_chol)
  ```

### Added
- **Full type safety** - Comprehensive type hints throughout with mypy compliance
- **Enhanced error messages** - More informative validation with actionable suggestions
- **Input validation helpers** - Centralized validation with better error reporting

### Changed
- **Mandatory numerical stability** - All operations now use Cholesky factorization
- **Cleaner API** - Single computational path eliminates confusion
- **Improved documentation** - Focus on ALS benefits without legacy comparisons

### Fixed
- **Type consistency** - All return types properly specified and validated
- **Error message quality** - Include context and suggestions for common issues

## [0.3.0] - 2024-01-XX

### Added
- High-level `ALSGLS` estimator with scikit-learn API
- `ALSGLSSystem` for statsmodels-style system estimation
- Automatic rank selection with `rank="auto"`
- Comprehensive documentation with Sphinx

### Changed
- Improved conjugate gradient solver stability
- Better memory usage tracking
- Enhanced convergence diagnostics in info dict

### Fixed
- Numerical stability for near-singular matrices
- Edge cases in diagonal floor handling

## [0.2.0] - 2024-01-XX

### Added
- EM baseline implementation (`em_gls`) for comparison
- Matrix-free conjugate gradient solver
- Woodbury matrix identity optimization
- Performance benchmarking scripts

### Changed
- Refactored core operations into `ops.py`
- Improved simulation functions
- Better default parameters

## [0.1.0] - 2024-01-XX

### Added
- Initial release
- Core `als_gls` function
- Basic simulation utilities
- MSE and NLL metrics
- Example scripts
