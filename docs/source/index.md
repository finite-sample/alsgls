# alsgls: Lightweight ALS Solver for Iterative GLS

[![PyPI version](https://img.shields.io/pypi/v/alsgls.svg)](https://pypi.org/project/alsgls/)
[![PyPI Downloads](https://static.pepy.tech/badge/alsgls)](https://pepy.tech/projects/alsgls)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Python package implementing Alternating Least Squares (ALS) for low-rank+diagonal GLS estimation. 
The package provides memory-efficient solutions for Seemingly Unrelated Regressions (SUR) and other 
GLS problems by using low-rank factor models instead of dense covariance matrices.

```{include} _snippets/synopsis.md
```

## Quick Start

```{include} _snippets/installation.md
```

Basic usage:

```python
from alsgls import als_gls, simulate_sur, nll_per_row, XB_from_Blist

# Simulate data
Xs_tr, Y_tr, Xs_te, Y_te = simulate_sur(N_tr=240, N_te=120, K=60, p=3, k=4)

# Fit ALS model
B, F, D, mem, _ = als_gls(Xs_tr, Y_tr, k=4)

# Make predictions
Yhat_te = XB_from_Blist(Xs_te, B)
nll = nll_per_row(Y_te - Yhat_te, F, D)
```

## Table of Contents

```{toctree}
:maxdepth: 2
:caption: User Guide

quickstart
mathematical_background
als_vs_em
examples
```

```{toctree}
:maxdepth: 2
:caption: API Reference

api/alsgls
```

```{toctree}
:maxdepth: 1
:caption: Development

contributing
changelog
```

## Indices and tables

* {ref}`genindex`
* {ref}`modindex`
* {ref}`search`