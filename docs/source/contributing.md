# Contributing

We welcome contributions to alsgls! This document provides guidelines for contributing to the project.

## Development Setup

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/your-username/alsgls.git
   cd alsgls
   ```

3. Install in development mode with dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

## Code Standards

### Style Guide

We use `ruff` for code formatting and linting:

```bash
# Format code
ruff format .

# Check for issues  
ruff check .

# Auto-fix issues
ruff check --fix .
```

### Type Hints

Add type hints to new functions:

```python
def als_gls(
    Xs: List[np.ndarray],
    Y: np.ndarray,
    k: int = 4,
    **kwargs
) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray, float, Dict]:
    ...
```

## Testing

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=alsgls --cov-report=html

# Run specific test file
pytest tests/test_als.py -v
```

### Writing Tests

Add tests for new functionality:

```python
def test_new_feature():
    """Test description."""
    # Arrange
    data = setup_test_data()
    
    # Act
    result = new_feature(data)
    
    # Assert
    assert result.shape == expected_shape
    assert np.allclose(result, expected, rtol=1e-5)
```

## Documentation

### Docstrings

Use NumPy-style docstrings:

```python
def function(param1, param2):
    """Brief description.
    
    Longer description if needed.
    
    Parameters
    ----------
    param1 : type
        Description of param1.
    param2 : type
        Description of param2.
        
    Returns
    -------
    type
        Description of return value.
        
    Examples
    --------
    >>> function(1, 2)
    3
    """
```

### Building Documentation

```bash
cd docs
make clean
make html
```

View at `docs/build/html/index.html`.

## Submitting Changes

### Workflow

1. Create a feature branch:
   ```bash
   git checkout -b feature-name
   ```

2. Make changes and commit:
   ```bash
   git add .
   git commit -m "Add feature: description"
   ```

3. Run tests and linting:
   ```bash
   pytest tests/
   ruff check .
   ```

4. Push to your fork:
   ```bash
   git push origin feature-name
   ```

5. Open a Pull Request on GitHub

### Pull Request Guidelines

- **Title**: Clear, concise description
- **Description**: What, why, and how
- **Tests**: Include tests for new features
- **Documentation**: Update docs if needed
- **Changelog**: Add entry if user-facing

### Commit Messages

Follow conventional commits:

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation only
- `style:` Code style (formatting, etc.)
- `refactor:` Code restructuring
- `test:` Adding tests
- `chore:` Maintenance tasks

Example:
```bash
git commit -m "feat: add sparse factor support to als_gls"
```

## Issue Reports

### Bug Reports

Include:
- Python version
- alsgls version
- Minimal reproducible example
- Error messages/traceback
- Expected vs actual behavior

### Feature Requests

Describe:
- Use case
- Proposed API
- Alternatives considered
- Implementation ideas

## Performance Contributions

When optimizing:

1. Profile first:
   ```python
   import cProfile
   cProfile.run('als_gls(Xs, Y, k=5)')
   ```

2. Benchmark changes:
   ```python
   import timeit
   before = timeit.timeit(old_code, number=100)
   after = timeit.timeit(new_code, number=100)
   ```

3. Document improvements in PR

## Questions?

- Open a [GitHub issue](https://github.com/finite-sample/alsgls/issues)
- Check existing issues first
- Use clear, descriptive titles

Thank you for contributing to alsgls!