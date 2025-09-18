Examples
========

This section provides detailed examples of using alsgls for various applications.

Basic SUR Example
-----------------

Seemingly Unrelated Regressions (SUR) is a classic application of GLS where we have multiple 
regression equations with correlated error terms.

.. code-block:: python

   from alsgls import simulate_sur, als_gls, XB_from_Blist, mse
   import numpy as np

   # Simulate SUR data
   # N_tr=300 observations, K=50 equations, p=3 regressors per equation, k=4 factors
   Xs_train, Y_train, Xs_test, Y_test = simulate_sur(
       N_tr=300, N_te=100, K=50, p=3, k=4, 
       noise_scale=0.1, factor_scale=1.0
   )

   print(f"Training data: {len(Xs_train)} equations")
   print(f"Equation shapes: {[X.shape for X in Xs_train[:3]]}...")
   print(f"Response matrix: {Y_train.shape}")

   # Fit ALS model
   B, F, D, memory_usage, info = als_gls(
       Xs_train, Y_train, k=4, 
       max_iter=10, tol=1e-6, verbose=True
   )

   print(f"Converged in {info['iterations']} iterations")
   print(f"Final objective: {info['objective']:.6f}")
   print(f"Memory usage: {memory_usage:.3f} MB")

   # Make predictions and evaluate
   Y_pred = XB_from_Blist(Xs_test, B)
   test_mse = mse(Y_test, Y_pred)
   print(f"Test MSE: {test_mse:.6f}")

Large-Scale Example
-------------------

This example demonstrates the memory advantages for larger problems:

.. code-block:: python

   from alsgls import simulate_sur, als_gls, em_gls
   import psutil
   import os

   def get_memory_usage():
       """Get current memory usage in MB"""
       process = psutil.Process(os.getpid())
       return process.memory_info().rss / 1024 / 1024

   # Large problem: 200 equations, 500 observations
   print("Generating large-scale data...")
   Xs_tr, Y_tr, Xs_te, Y_te = simulate_sur(N_tr=500, N_te=200, K=200, p=4, k=6)

   print(f"Problem size: {Y_tr.shape[0]} obs × {Y_tr.shape[1]} equations")
   print(f"Total parameters in dense Σ: {Y_tr.shape[1]**2:,}")
   print(f"Parameters in low-rank model: {Y_tr.shape[1] * 6 + Y_tr.shape[1]:,}")

   # Compare ALS vs EM
   mem_before = get_memory_usage()

   print("\\nFitting ALS model...")
   B_als, F_als, D_als, mem_als, info_als = als_gls(Xs_tr, Y_tr, k=6, max_iter=8)
   print(f"ALS converged in {info_als['iterations']} iterations")

   print("\\nFitting EM model...")  
   B_em, F_em, D_em, mem_em, info_em = em_gls(Xs_tr, Y_tr, k=6, max_iter=30)
   print(f"EM converged in {info_em['iterations']} iterations")

   # Compare predictions
   Y_pred_als = XB_from_Blist(Xs_te, B_als)
   Y_pred_em = XB_from_Blist(Xs_te, B_em)

   mse_als = mse(Y_te, Y_pred_als)
   mse_em = mse(Y_te, Y_pred_em)

   print(f"\\nResults:")
   print(f"ALS - Memory: {mem_als:.1f}MB, MSE: {mse_als:.6f}")
   print(f"EM  - Memory: {mem_em:.1f}MB, MSE: {mse_em:.6f}")
   print(f"Memory ratio (EM/ALS): {mem_em/mem_als:.1f}x")
   print(f"MSE difference: {abs(mse_als - mse_em):.2e}")

Custom Data Example
-------------------

Working with your own data instead of simulated data:

.. code-block:: python

   import numpy as np
   from alsgls import als_gls, XB_from_Blist

   # Prepare your data
   # Xs should be a list of design matrices, one per equation
   # Y should be an (N, K) matrix of responses

   # Example: 3 equations with different numbers of regressors
   N = 200  # observations
   K = 3    # equations

   # Generate some example data
   np.random.seed(42)
   
   # Equation 1: 2 regressors
   X1 = np.random.randn(N, 2)
   
   # Equation 2: 3 regressors  
   X2 = np.random.randn(N, 3)
   
   # Equation 3: 1 regressor
   X3 = np.random.randn(N, 1)
   
   Xs = [X1, X2, X3]
   
   # Correlated responses (you would load your actual data here)
   true_B = [np.array([[1.5], [-0.8]]),           # coeffs for eq 1
             np.array([[0.5], [1.2], [-0.3]]),     # coeffs for eq 2  
             np.array([[2.0]])]                     # coeffs for eq 3
   
   # Generate responses with correlation structure
   Y = np.column_stack([
       X1 @ true_B[0].flatten() + 0.1 * np.random.randn(N),
       X2 @ true_B[1].flatten() + 0.1 * np.random.randn(N), 
       X3 @ true_B[2].flatten() + 0.1 * np.random.randn(N)
   ])
   
   # Add some cross-equation correlation
   factor = np.random.randn(N, 1)
   Y += 0.3 * factor @ np.random.randn(1, K)

   print("Data shapes:")
   for i, X in enumerate(Xs):
       print(f"  Equation {i+1}: X{i+1} = {X.shape}")
   print(f"  Responses: Y = {Y.shape}")

   # Fit the model
   B_hat, F, D, memory, info = als_gls(Xs, Y, k=2, max_iter=10, verbose=True)

   print(f"\\nEstimated coefficients:")
   for i, b in enumerate(B_hat):
       print(f"  Equation {i+1}: {b.flatten()}")
       print(f"  True values: {true_B[i].flatten()}")
       print(f"  Error: {np.linalg.norm(b.flatten() - true_B[i].flatten()):.4f}")

Factor Structure Analysis
-------------------------

Examining the estimated factor structure:

.. code-block:: python

   from alsgls import simulate_sur, als_gls
   import matplotlib.pyplot as plt

   # Simulate data with known factor structure
   Xs_tr, Y_tr, _, _ = simulate_sur(N_tr=400, N_te=100, K=30, p=3, k=3)

   # Fit model
   B, F, D, _, _ = als_gls(Xs_tr, Y_tr, k=3, max_iter=15)

   print(f"Factor loadings shape: {F.shape}")
   print(f"Diagonal variances shape: {D.shape}")

   # Examine factor loadings
   print("\\nFactor loadings (first 5 equations):")
   print(F[:5, :])

   # Compute explained variance by factors
   factor_var = np.var(F @ F.T, axis=1)  
   total_var = factor_var + D
   explained_ratio = factor_var / total_var

   print(f"\\nVariance explained by factors:")
   print(f"  Mean: {explained_ratio.mean():.3f}")
   print(f"  Min:  {explained_ratio.min():.3f}")
   print(f"  Max:  {explained_ratio.max():.3f}")

   # Plot factor loadings heatmap
   plt.figure(figsize=(8, 6))
   plt.imshow(F, aspect='auto', cmap='RdBu_r')
   plt.colorbar(label='Loading')
   plt.xlabel('Factor')
   plt.ylabel('Equation')
   plt.title('Factor Loadings Matrix')
   plt.show()

Cross-Validation Example
------------------------

Selecting the optimal number of factors using cross-validation:

.. code-block:: python

   from alsgls import simulate_sur, als_gls, XB_from_Blist, mse
   import numpy as np

   # Generate data
   Xs_tr, Y_tr, Xs_val, Y_val = simulate_sur(N_tr=300, N_te=150, K=40, p=3, k=4)

   # Test different numbers of factors
   k_values = range(1, 11)
   val_mses = []

   for k in k_values:
       print(f"Testing k={k}...")
       
       # Fit model
       B, F, D, _, info = als_gls(Xs_tr, Y_tr, k=k, max_iter=10)
       
       # Validate
       Y_pred = XB_from_Blist(Xs_val, B)
       val_mse = mse(Y_val, Y_pred)
       val_mses.append(val_mse)
       
       print(f"  Validation MSE: {val_mse:.6f}")

   # Find optimal k
   best_k = k_values[np.argmin(val_mses)]
   print(f"\\nOptimal number of factors: k={best_k}")

   # Plot validation curve
   plt.figure(figsize=(8, 5))
   plt.plot(k_values, val_mses, 'o-')
   plt.axvline(best_k, color='red', linestyle='--', label=f'Optimal k={best_k}')
   plt.xlabel('Number of factors (k)')
   plt.ylabel('Validation MSE')
   plt.title('Factor Selection via Cross-Validation')
   plt.legend()
   plt.grid(True, alpha=0.3)
   plt.show()

Performance Profiling
---------------------

Detailed timing and memory profiling:

.. code-block:: python

   import time
   from alsgls import simulate_sur, als_gls, em_gls

   def profile_solver(solver_func, Xs, Y, k, **kwargs):
       """Profile a solver function"""
       start_time = time.time()
       result = solver_func(Xs, Y, k=k, **kwargs)
       end_time = time.time()
       
       B, F, D, memory, info = result
       runtime = end_time - start_time
       
       return {
           'runtime': runtime,
           'memory': memory, 
           'iterations': info['iterations'],
           'objective': info['objective'],
           'B': B, 'F': F, 'D': D
       }

   # Test different problem sizes
   problem_sizes = [(100, 30), (200, 60), (300, 90)]

   for N, K in problem_sizes:
       print(f"\\nProblem size: N={N}, K={K}")
       
       # Generate data
       Xs, Y, _, _ = simulate_sur(N_tr=N, N_te=50, K=K, p=3, k=5)
       
       # Profile ALS
       als_result = profile_solver(als_gls, Xs, Y, k=5, max_iter=8)
       
       # Profile EM  
       em_result = profile_solver(em_gls, Xs, Y, k=5, max_iter=30)
       
       print(f"ALS: {als_result['runtime']:.2f}s, {als_result['memory']:.1f}MB, "
             f"{als_result['iterations']} iter")
       print(f"EM:  {em_result['runtime']:.2f}s, {em_result['memory']:.1f}MB, "
             f"{em_result['iterations']} iter")
       print(f"Speedup: {em_result['runtime']/als_result['runtime']:.1f}x")
       print(f"Memory reduction: {em_result['memory']/als_result['memory']:.1f}x")