Contributing
============

We welcome contributions to alsgls! This document provides guidelines for contributing to the project.

Development Setup
-----------------

1. **Fork and clone the repository**:

   .. code-block:: bash

      git clone https://github.com/yourusername/alsgls.git
      cd alsgls

2. **Create a virtual environment**:

   .. code-block:: bash

      python -m venv venv
      source venv/bin/activate  # On Windows: venv\\Scripts\\activate

3. **Install in development mode**:

   .. code-block:: bash

      pip install -e .
      pip install -e .[dev]

4. **Install pre-commit hooks** (optional but recommended):

   .. code-block:: bash

      pip install pre-commit
      pre-commit install

Running Tests
-------------

The test suite uses pytest:

.. code-block:: bash

   # Run all tests
   python -m pytest tests/

   # Run specific test file
   python -m pytest tests/test_als.py

   # Run with verbose output
   python -m pytest tests/ -v

   # Run with coverage
   pip install pytest-cov
   python -m pytest tests/ --cov=alsgls --cov-report=html

Code Style
----------

We follow Python best practices:

- **PEP 8**: Use standard Python style guidelines
- **Type hints**: Add type annotations for function signatures
- **Docstrings**: Use NumPy-style docstrings for all public functions
- **Line length**: Limit lines to 88 characters (Black default)

Example function with proper documentation:

.. code-block:: python

   def als_gls(
       Xs: List[np.ndarray], 
       Y: np.ndarray, 
       k: int,
       max_iter: int = 10,
       tol: float = 1e-6
   ) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray, float, Dict]:
       """
       Fit GLS model using Alternating Least Squares.

       Parameters
       ----------
       Xs : List[np.ndarray]
           List of design matrices, one per equation.
       Y : np.ndarray
           Response matrix of shape (N, K).
       k : int
           Number of latent factors.
       max_iter : int, optional
           Maximum number of ALS iterations, by default 10.
       tol : float, optional
           Convergence tolerance, by default 1e-6.

       Returns
       -------
       B : List[np.ndarray]
           Regression coefficients for each equation.
       F : np.ndarray
           Factor loadings matrix of shape (K, k).
       D : np.ndarray
           Diagonal noise variances of shape (K,).
       memory : float
           Peak memory usage in MB.
       info : Dict
           Convergence information.

       Examples
       --------
       >>> from alsgls import simulate_sur, als_gls
       >>> Xs, Y, _, _ = simulate_sur(N_tr=100, N_te=50, K=20, p=2, k=3)
       >>> B, F, D, mem, info = als_gls(Xs, Y, k=3)
       >>> print(f"Converged in {info['iterations']} iterations")
       """

Testing Guidelines
------------------

When adding new features or fixing bugs:

1. **Write tests first**: Follow test-driven development when possible
2. **Test edge cases**: Include tests for boundary conditions and error cases  
3. **Maintain coverage**: Aim for >90% test coverage
4. **Use fixtures**: Create reusable test data with pytest fixtures

Example test structure:

.. code-block:: python

   import pytest
   import numpy as np
   from alsgls import als_gls, simulate_sur

   class TestALSGLS:
       @pytest.fixture
       def sample_data(self):
           """Create sample data for testing."""
           return simulate_sur(N_tr=100, N_te=50, K=10, p=2, k=3)
       
       def test_basic_functionality(self, sample_data):
           """Test basic ALS functionality."""
           Xs_tr, Y_tr, _, _ = sample_data
           B, F, D, memory, info = als_gls(Xs_tr, Y_tr, k=3)
           
           # Test return types and shapes
           assert isinstance(B, list)
           assert len(B) == len(Xs_tr)
           assert F.shape == (Y_tr.shape[1], 3)
           assert D.shape == (Y_tr.shape[1],)
           assert isinstance(memory, float)
           assert isinstance(info, dict)
       
       def test_convergence(self, sample_data):
           """Test convergence behavior.""" 
           Xs_tr, Y_tr, _, _ = sample_data
           _, _, _, _, info = als_gls(Xs_tr, Y_tr, k=3, max_iter=20, tol=1e-8)
           
           assert info['converged'] in [True, False]
           assert info['iterations'] <= 20

Documentation
-------------

Documentation is built with Sphinx and hosted on GitHub Pages.

**Building documentation locally**:

.. code-block:: bash

   cd docs
   make html
   open build/html/index.html  # View in browser

**Documentation types**:

- **Docstrings**: In-code documentation for all public functions
- **User guides**: High-level tutorials and explanations  
- **Examples**: Working code examples with explanations
- **API reference**: Auto-generated from docstrings

**Notebook integration**: Jupyter notebooks in ``als_sim/`` are automatically 
included in the documentation via nbsphinx.

Submitting Changes
------------------

1. **Create a feature branch**:

   .. code-block:: bash

      git checkout -b feature/your-feature-name

2. **Make your changes**: Implement your feature or fix
3. **Add tests**: Ensure your changes are well-tested
4. **Update documentation**: Add or update relevant documentation
5. **Run the test suite**: Verify all tests pass

   .. code-block:: bash

      python -m pytest tests/

6. **Commit your changes**:

   .. code-block:: bash

      git add .
      git commit -m "Add feature: brief description"

7. **Push and create a pull request**:

   .. code-block:: bash

      git push origin feature/your-feature-name

**Pull request guidelines**:

- Provide a clear description of the changes
- Reference any related issues
- Include examples if adding new features
- Ensure CI passes (tests, linting, documentation)

Types of Contributions
----------------------

We welcome various types of contributions:

**Bug reports**
   Open an issue with a minimal reproducible example

**Feature requests**  
   Discuss the feature in an issue before implementing

**Performance improvements**
   Profile your changes and include benchmarks

**Documentation improvements**
   Fix typos, add examples, improve clarity

**New algorithms**
   Alternative solvers or extensions to existing methods

Release Process
---------------

For maintainers:

1. **Update version** in ``pyproject.toml``
2. **Update CHANGELOG** with new features and fixes  
3. **Create release tag**: ``git tag v0.x.y``
4. **Build and upload**: ``python -m build && twine upload dist/*``
5. **Update documentation**: Ensure docs are current

Community Guidelines
--------------------

- **Be respectful**: Follow the code of conduct
- **Be patient**: Maintainers volunteer their time
- **Be clear**: Provide detailed bug reports and feature requests
- **Be collaborative**: Work together to improve the package

Getting Help
------------

- **GitHub Issues**: For bug reports and feature requests
- **Discussions**: For questions and general discussion
- **Email**: Contact maintainers for sensitive issues

Thank you for contributing to alsgls!