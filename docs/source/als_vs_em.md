# ALS vs EM Comparison

This section compares the Alternating Least Squares (ALS) and Expectation-Maximization (EM) 
approaches for low-rank+diagonal GLS estimation.

## Mathematical Background

Both algorithms solve the same statistical problem: estimating regression coefficients β 
when the error covariance has a low-rank+diagonal structure:

Σ = FF^T + diag(D)

where F is K×k factor loadings and D is K×1 diagonal noise.

## EM Algorithm

The EM algorithm alternates between:

**E-step**: Compute expected sufficient statistics given current parameters
**M-step**: Update parameters by solving weighted least squares with full Σ^(-1)

The critical issue: EM's M-step explicitly forms the K×K inverse covariance matrix, 
requiring O(K²) memory even though the model has only O(Kk) parameters.

```python
# EM's expensive step (pseudocode)
Sigma = F @ F.T + np.diag(D)  # K×K dense matrix
Sigma_inv = np.linalg.inv(Sigma)  # K×K inversion
# Use Sigma_inv for β updates...
```

## ALS Algorithm  

ALS also alternates between updating β and updating (F, D), but uses the Woodbury 
matrix identity to avoid forming dense matrices:

Σ^(-1) = D^(-1) - D^(-1)F(I + F^T D^(-1)F)^(-1)F^T D^(-1)

The key insight: we only need to invert a k×k matrix (I + F^T D^(-1)F), not K×K.

```python
# ALS's efficient step (pseudocode)  
D_inv = 1.0 / D  # K×1 vector
small_inv = np.linalg.inv(np.eye(k) + F.T @ (D_inv[:, None] * F))  # k×k
# Apply Woodbury formula without forming K×K matrices
```

## Memory Comparison

| Algorithm | Largest Array | Memory Complexity |
|-----------|---------------|------------------|
| EM | Σ^(-1) (K×K dense) | O(K²) |
| ALS | F (K×k skinny) | O(Kk) |

For K=100 equations and k=5 factors:
- EM: 100×100 = 10,000 floats
- ALS: 100×5 = 500 floats  
- **Ratio: 20×**

## Computational Comparison

```python
from alsgls import simulate_sur, als_gls, em_gls
import time

# Generate test problem
K = 100  # equations
N = 300  # observations  
k = 5    # factors
Xs_tr, Y_tr, _, _ = simulate_sur(N_tr=N, N_te=50, K=K, p=3, k=k)

# Time ALS
t0 = time.time()
B_als, F_als, D_als, mem_als, info_als = als_gls(
    Xs_tr, Y_tr, k=k, sweeps=8
)
time_als = time.time() - t0

# Time EM
t0 = time.time()
B_em, F_em, D_em, mem_em, info_em = em_gls(
    Xs_tr, Y_tr, k=k, iters=30
)
time_em = time.time() - t0

print(f"ALS: {time_als:.2f}s, {mem_als:.1f}MB")
print(f"EM:  {time_em:.2f}s, {mem_em:.1f}MB")
print(f"Memory ratio: {mem_em/mem_als:.1f}×")
```

## Statistical Equivalence

Despite different computational approaches, both algorithms optimize the same 
likelihood and produce statistically indistinguishable estimates:

```python
from alsgls import XB_from_Blist, mse
import numpy as np

# Compare predictions
Y_pred_als = XB_from_Blist(Xs_te, B_als)
Y_pred_em = XB_from_Blist(Xs_te, B_em)

# MSE should be nearly identical
mse_als = mse(Y_te, Y_pred_als)
mse_em = mse(Y_te, Y_pred_em)

print(f"ALS MSE: {mse_als:.6f}")
print(f"EM MSE:  {mse_em:.6f}")
print(f"Difference: {abs(mse_als - mse_em):.2e}")

# Factor structures should be equivalent (up to rotation)
cov_als = F_als @ F_als.T + np.diag(D_als)
cov_em = F_em @ F_em.T + np.diag(D_em)
print(f"Covariance difference: {np.linalg.norm(cov_als - cov_em):.2e}")
```

## When to Use Each

**Use ALS when:**
- K is large (>50 equations)
- Memory is constrained
- You need the Woodbury form for other computations
- k << K (low-rank assumption holds)

**Use EM when:**
- K is small (<30 equations)  
- You need the full covariance matrix anyway
- Implementing a standard EM framework
- Debugging (EM is conceptually simpler)

## Implementation Details

The package provides both for comparison:

- `als_gls()`: Memory-efficient ALS implementation
- `em_gls()`: Baseline EM implementation  

Both use the same convergence criteria and regularization options for fair comparison.