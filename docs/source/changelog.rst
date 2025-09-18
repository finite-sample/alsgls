Changelog
=========

All notable changes to this project will be documented in this file.

The format is based on `Keep a Changelog <https://keepachangelog.com/en/1.0.0/>`_,
and this project adheres to `Semantic Versioning <https://semver.org/spec/v2.0.0.html>`_.

[Unreleased]
------------

Added
~~~~~
- Comprehensive Sphinx documentation with GitHub Pages deployment
- Mathematical background documentation
- Detailed API reference with autodoc
- Example gallery and tutorials
- Contributing guidelines

[0.1.0] - 2024-01-01
--------------------

Added
~~~~~
- Initial release of alsgls package
- Core ALS solver (``als_gls``) with memory-efficient implementation
- Baseline EM solver (``em_gls``) for comparison
- Data simulation functions (``simulate_sur``, ``simulate_gls``)
- Evaluation metrics (``mse``, ``nll_per_row``)
- Linear algebra operations with Woodbury matrix utilities
- Matrix-free conjugate gradient solver
- Comprehensive test suite
- Example scripts demonstrating ALS vs EM comparison
- Jupyter notebooks with mathematical background and experiments

Features
~~~~~~~~
- **Memory efficiency**: O(Kk) memory complexity vs O(K²) for traditional methods
- **Fast convergence**: Typically 5-6 ALS iterations vs 20-50 EM iterations  
- **Matrix-free operations**: No dense K×K matrix formations required
- **Woodbury identity**: Efficient precision matrix computations
- **Flexible data structures**: Support for varying numbers of regressors per equation
- **Comprehensive metrics**: MSE and negative log-likelihood evaluation
- **Statistical equivalence**: Same MLE estimates as EM with better computational properties

Documentation
~~~~~~~~~~~~~
- README with quick start guide and performance comparisons
- Detailed docstrings following NumPy conventions
- Mathematical background in Jupyter notebooks
- Memory usage benchmarks and timing comparisons

Testing
~~~~~~~
- Unit tests for all core functions
- Integration tests for end-to-end workflows  
- Performance regression tests
- Shape consistency validation
- Numerical accuracy verification