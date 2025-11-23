ALS vs EM Comparison
=====================

This section provides a detailed comparison between the Alternating Least Squares (ALS) and 
Expectation-Maximization (EM) approaches for low-rank + diagonal GLS estimation.

Algorithmic Differences
-----------------------

Expectation-Maximization (EM)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The EM algorithm alternates between:

1. **E-step**: Compute posterior expectations of latent factors given current parameters
2. **M-step**: Update parameters to maximize expected log-likelihood

.. code-block:: python

   # Pseudocode for EM iteration
   for iteration in range(max_iter):
       # E-step: compute factor posteriors
       E_z, E_zz = compute_posterior_moments(Y, F, D, X, B)
       
       # M-step: update all parameters
       F = update_factor_loadings(Y, E_z, E_zz)
       D = update_noise_variances(Y, F, E_z)
       B = update_regression_coeffs(X, Y, F, D)  # Forms full K×K inverse!

**Memory bottleneck**: The M-step for β typically reconstructs the full K × K precision matrix.

Alternating Least Squares (ALS)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ALS algorithm directly alternates between parameter blocks:

1. **β-step**: Update regression coefficients using Woodbury identity
2. **Factor-step**: Update factor loadings and noise variances

.. code-block:: python

   # Pseudocode for ALS iteration  
   for iteration in range(max_iter):
       # β-step: solve GLS via matrix-free CG
       B = solve_gls_woodbury(X, Y, F, D)  # Only k×k inverse needed
       
       # Factor-step: update factor structure
       residuals = Y - XB_from_Blist(X, B)
       F, D = update_factors(residuals)

**Memory advantage**: Never forms matrices larger than K × k.

Performance Comparison
----------------------

Memory Usage
~~~~~~~~~~~~

The memory requirements scale differently:

- **EM**: O(K²) due to full covariance matrix operations
- **ALS**: O(Kk) with k ≪ K

Empirical memory usage for SUR problems:

.. list-table:: Memory Usage Comparison
   :header-rows: 1
   :widths: 10 15 15 15 15 20

   * - K
     - β-RMSE EM  
     - β-RMSE ALS
     - Peak MB EM
     - Peak MB ALS
     - Memory Ratio
   * - 50
     - 0.021
     - 0.021  
     - 0.020
     - 0.002
     - 10×
   * - 80
     - 0.020
     - 0.020
     - 0.051
     - 0.003  
     - 17×
   * - 120
     - 0.020
     - 0.020
     - 0.115
     - 0.004
     - 29×

Convergence Speed
~~~~~~~~~~~~~~~~~

- **ALS**: Typically converges in 5-6 iterations
- **EM**: Often requires 20-50 iterations for comparable precision

The faster convergence of ALS stems from its direct optimization of the objective function 
rather than iterative expectation computation.

Statistical Equivalence
-----------------------

Both algorithms converge to the same maximum likelihood estimates under standard regularity 
conditions. The key differences are computational, not statistical.

Numerical Stability
~~~~~~~~~~~~~~~~~~~

- **EM**: More robust to poor initialization due to its probabilistic foundation
- **ALS**: Can be sensitive to initialization but typically more stable once converged

Practical Guidelines
--------------------

When to Use EM
~~~~~~~~~~~~~~

- **Small problems** (K < 50): Dense methods are often faster
- **Research/prototyping**: EM's probabilistic interpretation aids model development  
- **Uncertainty quantification**: Natural framework for computing standard errors

When to Use ALS
~~~~~~~~~~~~~~~

- **Large problems** (K > 100): Memory constraints make ALS essential
- **Production systems**: Faster convergence and lower memory usage
- **Real-time applications**: Predictable memory footprint and runtime

Hybrid Approaches
~~~~~~~~~~~~~~~~~

For some applications, a hybrid strategy works well:

1. Use EM for the first few iterations to establish good parameter values
2. Switch to ALS for final convergence and production deployment

Example Comparison
------------------

Here's a practical comparison using the package:

.. code-block:: python

   import time
   from alsgls import simulate_sur, als_gls, em_gls, mse, XB_from_Blist

   # Simulate moderately large problem
   Xs_tr, Y_tr, Xs_te, Y_te = simulate_sur(N_tr=500, N_te=200, K=100, p=4, k=5)

   # Time ALS
   start = time.time()
   B_als, F_als, D_als, mem_als, info_als = als_gls(Xs_tr, Y_tr, k=5, max_iter=10)
   time_als = time.time() - start

   # Time EM  
   start = time.time()
   B_em, F_em, D_em, mem_em, info_em = em_gls(Xs_tr, Y_tr, k=5, max_iter=50)
   time_em = time.time() - start

   # Compare results
   Y_pred_als = XB_from_Blist(Xs_te, B_als)
   Y_pred_em = XB_from_Blist(Xs_te, B_em)

   print(f"ALS: {time_als:.2f}s, {mem_als:.1f}MB, MSE: {mse(Y_te, Y_pred_als):.6f}")
   print(f"EM:  {time_em:.2f}s, {mem_em:.1f}MB, MSE: {mse(Y_te, Y_pred_em):.6f}")
   print(f"Speedup: {time_em/time_als:.1f}x, Memory savings: {mem_em/mem_als:.1f}x")

This typically shows ALS achieving 5-10x speedup and 10-30x memory reduction while 
maintaining equivalent statistical accuracy.