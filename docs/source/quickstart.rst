Quick Start Guide
=================

Installation
------------

Install alsgls from PyPI:

.. code-block:: bash

   pip install alsgls

For development, clone the repository and install in editable mode:

.. code-block:: bash

   git clone https://github.com/finite-sample/alsgls.git
   cd alsgls
   pip install -e .

Basic Usage
-----------

The core functionality of alsgls revolves around the :func:`alsgls.als_gls` function, which implements 
the Alternating Least Squares algorithm for low-rank+diagonal GLS estimation.

Simulating Data
~~~~~~~~~~~~~~~

First, let's generate some synthetic data using the built-in simulation functions:

.. code-block:: python

   from alsgls import simulate_sur

   # Generate Seemingly Unrelated Regressions data
   Xs_train, Y_train, Xs_test, Y_test = simulate_sur(
       N_tr=240,    # Training observations
       N_te=120,    # Test observations
       K=60,        # Number of equations
       p=3,         # Regressors per equation
       k=4          # Latent factor rank
   )

Fitting the Model
~~~~~~~~~~~~~~~~~

Now we can fit the ALS model to estimate the regression coefficients and factor structure:

.. code-block:: python

   from alsgls import als_gls

   # Fit ALS model
   B, F, D, memory_usage, convergence_info = als_gls(
       Xs_train, 
       Y_train, 
       k=4,         # Factor rank
       max_iter=10  # Maximum ALS iterations
   )

The function returns:

- ``B``: List of regression coefficient vectors for each equation
- ``F``: Factor loadings matrix (K × k)
- ``D``: Diagonal noise variances (K,)
- ``memory_usage``: Peak memory usage during computation
- ``convergence_info``: Dictionary with convergence statistics

Making Predictions
~~~~~~~~~~~~~~~~~~~

Use the fitted coefficients to make predictions on new data:

.. code-block:: python

   from alsgls import XB_from_Blist, nll_per_row

   # Generate predictions
   Y_pred = XB_from_Blist(Xs_test, B)

   # Evaluate using negative log-likelihood per row
   residuals = Y_test - Y_pred
   nll = nll_per_row(residuals, F, D)
   print(f"Negative log-likelihood per row: {nll:.4f}")

Comparing with EM
~~~~~~~~~~~~~~~~~

You can also fit the same model using the EM algorithm for comparison:

.. code-block:: python

   from alsgls import em_gls

   # Fit EM model (higher memory usage)
   B_em, F_em, D_em, memory_em, _ = em_gls(
       Xs_train, 
       Y_train, 
       k=4,
       max_iter=50
   )

   print(f"ALS memory usage: {memory_usage:.3f} MB")
   print(f"EM memory usage: {memory_em:.3f} MB")
   print(f"Memory ratio (EM/ALS): {memory_em/memory_usage:.1f}x")

Complete Example
~~~~~~~~~~~~~~~~

Here's a complete working example:

.. code-block:: python

   from alsgls import (
       simulate_sur, als_gls, em_gls, 
       XB_from_Blist, nll_per_row, mse
   )

   # Simulate data
   Xs_tr, Y_tr, Xs_te, Y_te = simulate_sur(N_tr=240, N_te=120, K=60, p=3, k=4)

   # Fit both models
   B_als, F_als, D_als, mem_als, _ = als_gls(Xs_tr, Y_tr, k=4)
   B_em, F_em, D_em, mem_em, _ = em_gls(Xs_tr, Y_tr, k=4)

   # Compare predictions
   Y_pred_als = XB_from_Blist(Xs_te, B_als)
   Y_pred_em = XB_from_Blist(Xs_te, B_em)

   mse_als = mse(Y_te, Y_pred_als)
   mse_em = mse(Y_te, Y_pred_em)

   print(f"ALS MSE: {mse_als:.6f}, Memory: {mem_als:.3f} MB")
   print(f"EM MSE: {mse_em:.6f}, Memory: {mem_em:.3f} MB")
   print(f"Memory savings: {mem_em/mem_als:.1f}x")

Next Steps
----------

- Read about the :doc:`mathematical_background` behind the algorithms
- Learn about the differences in :doc:`als_vs_em` approaches  
- Explore more detailed :doc:`examples` and use cases