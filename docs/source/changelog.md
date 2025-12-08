# Changelog

All notable changes to alsgls will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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