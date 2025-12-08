# Mathematical Background

## The GLS Problem

Consider a system of K regression equations with N observations:

y_j = X_j β_j + ε_j,  j = 1, ..., K

where:
- y_j is N×1 response vector for equation j
- X_j is N×p_j design matrix for equation j  
- β_j is p_j×1 coefficient vector
- ε_j is N×1 error vector

Stacking all equations:

Y = [y_1, ..., y_K] (N×K matrix)
ε = [ε_1, ..., ε_K] (N×K matrix)

## Error Structure

The key assumption is that errors are correlated across equations but independent across observations:

E[ε_i ε_i^T] = Σ (K×K covariance matrix)

where ε_i is the i-th row of ε (errors for observation i across all K equations).

## Low-Rank + Diagonal Structure

Instead of estimating the full K×K covariance matrix (K²/2 parameters), we assume:

Σ = FF^T + diag(D)

where:
- F is K×k factor loadings matrix (Kk parameters)
- D is K×1 diagonal noise vector (K parameters)
- k << K is the latent factor rank

Total parameters: K(k+1) << K²

## The Woodbury Identity

The key to efficient computation is the Woodbury matrix identity:

(A + UCV)^(-1) = A^(-1) - A^(-1)U(C^(-1) + VA^(-1)U)^(-1)VA^(-1)

Applied to our case with A = diag(D), U = F, C = I, V = F^T:

Σ^(-1) = D^(-1) - D^(-1)F(I + F^T D^(-1)F)^(-1)F^T D^(-1)

This reduces the inversion from K×K to k×k.

## ALS Algorithm

The algorithm alternates between two steps:

### Step 1: Update β given (F, D)

For each equation j, solve the weighted least squares problem:

β_j = argmin Σ_i w_ij ||y_ij - x_ij^T β_j||²

where the weights come from Σ^(-1). We use conjugate gradient to avoid forming normal equations.

### Step 2: Update (F, D) given β

Given residuals R = Y - Ŷ, estimate the factor model:

1. Compute sample covariance: S = R^T R / N
2. Extract k leading eigenvectors of S → F
3. Set D = diag(S - FF^T)

## Objective Function

Both ALS and EM minimize the negative log-likelihood:

NLL = (N/2)[K log(2π) + log|Σ| + tr(Σ^(-1)S)]

where S is the sample covariance of residuals.

## Regularization

To ensure stability, we add ridge penalties:

- λ_F||F||² for factor loadings
- λ_B||β||² for regression coefficients
- d_floor as minimum diagonal variance

## Convergence Criteria

The algorithm stops when:

1. Relative NLL improvement < rel_tol (default 1e-6)
2. Number of sweeps exceeds max_sweeps
3. Line search fails (accept_t = 0)

## Computational Complexity

| Operation | EM | ALS |
|-----------|-----|-----|
| Memory | O(K²) | O(Kk) |
| Per iteration | O(K³ + NKp) | O(k³ + NKp) |
| Iterations | 20-50 | 5-10 |

## Statistical Properties

Under standard regularity conditions:

1. **Consistency**: β̂ → β₀ as N → ∞
2. **Efficiency**: Achieves GLS efficiency when (F, D) are known
3. **Robustness**: Ridge regularization prevents singularities

## Extensions

The low-rank framework extends to:

- Time-varying factors: F_t
- Sparse factors: L1 penalty on F
- Hierarchical models: Multi-level factor structures
- Missing data: EM naturally handles missing Y values