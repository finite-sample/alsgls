alsgls: Lightweight ALS Solver for Iterative GLS
===============================================

.. image:: https://img.shields.io/pypi/v/alsgls.svg
   :target: https://pypi.org/project/alsgls/
   :alt: PyPI version

.. image:: https://static.pepy.tech/badge/alsgls
   :target: https://pepy.tech/projects/alsgls
   :alt: PyPI Downloads

.. image:: https://img.shields.io/badge/python-3.8+-blue.svg
   :target: https://www.python.org/downloads/
   :alt: Python 3.8+

.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :target: https://opensource.org/licenses/MIT
   :alt: License: MIT

A Python package implementing Alternating Least Squares (ALS) for low-rank+diagonal GLS estimation. 
The package provides memory-efficient solutions for Seemingly Unrelated Regressions (SUR) and other 
GLS problems by using low-rank factor models instead of dense covariance matrices.

When a GLS problem involves hundreds of equations, the K × K covariance matrix becomes the computational 
bottleneck. A simple statistical remedy is to assume that most of the cross‑equation dependence can be 
captured by a *handful of latent factors* plus equation‑specific noise. This "low‑rank + diagonal" 
assumption slashes the number of unknowns from roughly K² to about K×k parameters, where **k** (the 
latent factor rank) is much smaller than K.

Quick Start
-----------

Install the package:

.. code-block:: bash

   pip install alsgls

Basic usage:

.. code-block:: python

   from alsgls import als_gls, simulate_sur, nll_per_row, XB_from_Blist

   # Simulate data
   Xs_tr, Y_tr, Xs_te, Y_te = simulate_sur(N_tr=240, N_te=120, K=60, p=3, k=4)
   
   # Fit ALS model
   B, F, D, mem, _ = als_gls(Xs_tr, Y_tr, k=4)
   
   # Make predictions
   Yhat_te = XB_from_Blist(Xs_te, B)
   nll = nll_per_row(Y_te - Yhat_te, F, D)

Table of Contents
-----------------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   quickstart
   mathematical_background
   als_vs_em
   examples

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/alsgls

.. toctree::
   :maxdepth: 1
   :caption: Development

   contributing
   changelog

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`