# Changelog

All notable changes to alsgls will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **`results.bootstrap(B, method, seed)`** on both `ALSGLSSystemResults` and
  `ALSGLS`, returning a `BootstrapResults` with percentile-*t* `conf_int()`,
  bootstrap-*t* `pvalues`, bootstrap `bse`, and the raw replicate arrays. Each
  replicate refits `F`, `D` and `beta`, which is what captures the part of the
  sampling variance the plug-in cannot: that `Sigma` is estimated. The
  studentised interval is the calibrated object, per Rilstone and Veall (1996),
  Fiebig and Kim (2000) and Horowitz (2019); the plug-in is biased the same way
  inside each replicate as in the sample, so the quantiles absorb the bias.
  Schemes: `"parametric"`, `"wild"`, `"residual"`. Pairs resampling is
  deliberately absent — at `n = 20` it leaves ~63% distinct rows.

  Validated on the Monte Carlo fixture at `n = 20` (100 replicates, B = 199):
  the plug-in covers 0.875 and rejects a true null 0.105 of the time; the
  parametric bootstrap-t covers 0.930 and rejects 0.050, with an interval width
  1.00 times what a correctly calibrated normal interval would have. Cost:
  `B = 999` takes about 20 s at K = 20, 40 s at K = 60 and 1.7 min at K = 100.
- **`init_D`** keyword on `als_gls` to warm-start the Sigma step.
- **`max_identified_rank(K)`** in `alsgls._validation`.

### Changed

- **The Sigma step is now L-BFGS-B on the profile likelihood over `log D`**,
  with `F` concentrated out in closed form, on residuals standardised to unit
  variance — exactly what R's `factanal` does, and for the reason it gives:
  alternating the two closed forms is a fixed-point iteration that crawls when
  a diagonal variance is small. On bootstrap replicates at `n = 20` the
  alternation's per-fit cost had a median of 24 ms but a mean of 142 ms and a
  maximum of 598 ms (11,619 inner iterations); the quasi-Newton step's mean is
  1.7 ms and its maximum 7 ms, and where the alternation hit its cap it also
  found a likelihood 4.6e-4 nats/row better. The gradient is the envelope
  theorem, computed with the existing Woodbury kernels and verified against
  finite differences to 3.7e-9. Every guard from 2.0.0 still holds: agreement
  with sklearn's `FactorAnalysis`, BIC rank recovery, Zellner, monotonicity
  in `k`.
- **Standard errors now apply the residual degrees-of-freedom rescale**
  `Sigma_ij * n / sqrt((n - p_i)(n - p_j))` that `linearmodels`
  (`debiased=True`), R `systemfit` (`"geomean"`) and Stata `sureg` (`dfk`) all
  apply. Applied as `F -> diag(sqrt c) F`, `D -> c * D`, which preserves the
  low-rank structure exactly. Standard errors grow by `sqrt(n / (n - p))`.
- **The OLS ridge initialisation uses the nominal `lam_B`**, not the
  GLS-scaled `lam_B_eff`. An OLS ridge objective scales uniformly under
  `Y -> sY`, so a fixed penalty is what gives `B -> sB`; the GLS objective has a
  dimensionless first term and needs `lam / s^2`. Using the GLS scaling in the
  OLS init made the starting point scale-dependent, which the forgiving
  fixed-point Sigma step hid and the quasi-Newton one exposed.

### Fixed

- **Unidentified factor ranks were accepted.** The `k`-factor model spends
  `K*k + K - k(k-1)/2` parameters on a covariance with `K(K+1)/2` free entries;
  past Ledermann's bound `(K - k)^2 >= K + k` the loadings are not identified
  and the likelihood has a ridge of maxima. R's `factanal` refuses with
  "degrees of freedom < 0"; `als_gls` now does too, with the largest identified
  rank in the message. `_auto_rank` and `_default_k_candidates` respect it.
  Both Monte Carlo test suites had been running at `K = 4, k = 2` — 11
  parameters for 10 free entries, df = -1 — and one helper at `K = 3, k = 2`.
  Every calibration number they recorded was measured on a `Sigma` the data
  could not pin down. They now run at `k = 1`.

### Root cause of the standard-error shortfall, measured

The plug-in `(X' Sigma_hat^-1 X)^-1` understated the sampling spread — se
ratio 0.69 at `n = 20`. An oracle experiment splits it exactly:
`0.69 = 0.77 x 0.88`. The first factor is the plug-in's bias at `Sigma_hat`
(Jensen: the formula is concave in `Sigma`), the second is the extra spread
feasible GLS carries over GLS at the true `Sigma`, which no formula at a fixed
`Sigma_hat` can see. Both are Freedman and Peters (1984, JASA, Theorem 1). The
oracle at the true `Sigma` is calibrated (1.03), so the linear algebra was
never wrong; its inputs were. The df rescale closes about a fifth of the gap.
The bootstrap closes the rest.

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

[Unreleased]: https://github.com/finite-sample/alsgls/commits/main

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
