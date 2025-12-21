# Changelog

All notable changes to alsgls will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[0.3.0]: https://github.com/finite-sample/alsgls/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/finite-sample/alsgls/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/finite-sample/alsgls/releases/tag/v0.1.0