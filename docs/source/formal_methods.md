# Formal Methods

This document provides rigorous mathematical foundations for the ALS-GLS algorithm.

## 1. Problem Setup

### Model Specification

Consider a system of K regression equations:

$$
y_j = X_j \beta_j + \varepsilon_j, \quad j = 1, \ldots, K
$$

Stacking row-wise across equations for each observation $i$:

$$
y_i = (y_{i1}, \ldots, y_{iK})^T \in \mathbb{R}^K
$$

The full model in matrix form:

$$
Y = X B + E
$$

where:
- $Y \in \mathbb{R}^{N \times K}$: response matrix
- $X_j \in \mathbb{R}^{N \times p_j}$: design matrix for equation $j$
- $B_j \in \mathbb{R}^{p_j}$: coefficient vector for equation $j$
- $E \in \mathbb{R}^{N \times K}$: error matrix

### Error Distribution

Errors follow a matrix-normal distribution:

$$
\text{vec}(E^T) \sim \mathcal{N}(0, I_N \otimes \Sigma)
$$

Equivalently, rows of $E$ are i.i.d.:

$$
\varepsilon_i \sim \mathcal{N}(0, \Sigma), \quad i = 1, \ldots, N
$$

### Low-Rank Covariance Structure

**Assumption (Factor Model):** The covariance admits a low-rank plus diagonal decomposition:

$$
\Sigma = F F^T + D
$$

where:
- $F \in \mathbb{R}^{K \times k}$: factor loadings matrix with $k \ll K$
- $D = \text{diag}(d_1, \ldots, d_K)$: diagonal idiosyncratic variances with $d_j > 0$

**Parameter Count:**
- Full covariance: $K(K+1)/2$ parameters
- Factor model: $Kk + K$ parameters
- Reduction ratio: $\frac{K+1}{2(k+1)} \approx \frac{K}{2k}$ when $K \gg k$

---

## 2. Optimization Problem

### Negative Log-Likelihood

The NLL for the model is:

$$
\mathcal{L}(\beta, F, D) = \frac{N}{2} \left[ K \log(2\pi) + \log |\Sigma| + \text{tr}(\Sigma^{-1} S) \right]
$$

where $S = \frac{1}{N} \sum_{i=1}^N r_i r_i^T$ is the sample covariance of residuals $r_i = y_i - X_i \beta$.

**Per-observation NLL** (used in the code):

$$
\ell(\beta, F, D) = \frac{1}{2} \left[ K \log(2\pi) + \log |\Sigma| + \text{tr}(\Sigma^{-1} S) \right]
$$

### Equivalent Formulations

Using $R = Y - \hat{Y}$ as the $N \times K$ residual matrix:

$$
\text{tr}(\Sigma^{-1} S) = \frac{1}{N} \text{tr}(R^T R \Sigma^{-1}) = \frac{1}{N} \sum_{i,j} (R \Sigma^{-1})_{ij} R_{ij}
$$

---

## 3. Woodbury Matrix Identity

### Statement

For invertible $A$ and appropriately sized $U, C, V$:

$$
(A + UCV)^{-1} = A^{-1} - A^{-1} U (C^{-1} + V A^{-1} U)^{-1} V A^{-1}
$$

### Application to Factor Covariance

With $A = D$, $U = F$, $C = I_k$, $V = F^T$:

$$
\Sigma^{-1} = (FF^T + D)^{-1} = D^{-1} - D^{-1} F (I_k + F^T D^{-1} F)^{-1} F^T D^{-1}
$$

**Proof sketch:** Verify by direct multiplication $\Sigma \Sigma^{-1} = I_K$. □

### Computational Benefit

| Operation | Dense | Woodbury |
|-----------|-------|----------|
| Inversion | $O(K^3)$ | $O(k^3 + K k^2)$ |
| Storage | $O(K^2)$ | $O(Kk)$ |

---

## 4. Determinant Lemma

### Statement

$$
|\Sigma| = |FF^T + D| = |D| \cdot |I_k + F^T D^{-1} F|
$$

### Proof

Using the matrix determinant lemma for $|A + UCV| = |C^{-1}| |A| |C + VA^{-1}U|$:

$$
|FF^T + D| = |I_k|^{-1} |D| |I_k + F^T D^{-1} F| = |D| |I_k + F^T D^{-1} F|
$$

### Log-Determinant Computation

For numerical stability, compute via Cholesky factorization:

$$
\log |\Sigma| = \sum_{j=1}^K \log d_j + 2 \sum_{i=1}^k \log L_{ii}
$$

where $L$ is the Cholesky factor of $C = I_k + F^T D^{-1} F$.

---

## 5. ALS Algorithm

### Block Coordinate Descent Structure

The algorithm alternates between three blocks:

1. **β-step:** Update regression coefficients given $(F, D)$
2. **F-step:** Update factor loadings given $(\beta, D)$
3. **D-step:** Update diagonal variances given $(\beta, F)$

### β-Step: Weighted Least Squares

Given $\Sigma$, the optimal $\beta$ solves:

$$
\hat{\beta} = \arg\min_\beta \sum_{i=1}^N (y_i - X_i \beta)^T \Sigma^{-1} (y_i - X_i \beta)
$$

This is equivalent to the normal equations:

$$
\left( \sum_j X_j^T \Sigma^{-1}_{jj} X_j + \lambda_B I \right) \beta_j = \sum_j X_j^T (\Sigma^{-1} Y)_j
$$

Solved via matrix-free conjugate gradient using the Woodbury identity.

### F-Step: Gradient Descent

The gradient of NLL with respect to $F$ is:

$$
\nabla_F \ell = \Sigma^{-1} F - \Sigma^{-1} S \Sigma^{-1} F + \lambda_F F
$$

where $S = R^T R / N$ is the sample covariance.

**Derivation:**
1. $\frac{\partial}{\partial F} \log |\Sigma| = 2 \Sigma^{-1} F$ (standard matrix derivative)
2. $\frac{\partial}{\partial F} \text{tr}(\Sigma^{-1} S) = -2 \Sigma^{-1} S \Sigma^{-1} F$ (chain rule)

### D-Step: Closed-Form MLE

Given $F$, the optimal diagonal is:

$$
d_j = \max\left( S_{jj} - (FF^T)_{jj}, d_{\min} \right)
$$

where $d_{\min} > 0$ ensures positive definiteness.

---

## 6. Convergence Properties

### Theorem 1 (Monotonicity)

**Statement:** The NLL sequence $\{\ell^{(t)}\}$ is non-increasing: $\ell^{(t+1)} \leq \ell^{(t)}$.

**Proof:**
- The β-step minimizes NLL exactly (or with reversion safeguard)
- The F-step uses gradient descent with backtracking line search
- The D-step is closed-form MLE given F
- Backtracking ensures each step is NLL-nonincreasing

Combined, each sweep satisfies $\ell^{(t+1)} \leq \ell^{(t)}$. □

### Theorem 2 (Convergence to Stationary Point)

**Statement:** Under regularity conditions, the sequence $(\beta^{(t)}, F^{(t)}, D^{(t)})$ converges to a stationary point of $\ell$.

**Proof sketch:**
1. NLL is bounded below (by $-\infty$ from log-det, regularized to prevent this)
2. The sequence is monotonically non-increasing (Theorem 1)
3. Monotone bounded sequences converge
4. Limit point satisfies first-order optimality conditions

□

### Regularity Conditions

1. $E[X_j^T X_j]$ is full rank for each $j$
2. $d_{\min} > 0$ enforces bounded eigenvalues
3. Ridge regularization $\lambda_B, \lambda_F > 0$ ensures strict convexity in blocks

---

## 7. Statistical Properties

### Consistency

**Proposition:** Under standard regularity conditions, as $N \to \infty$:

$$
\hat{\beta}_j \xrightarrow{p} \beta_j^0, \quad j = 1, \ldots, K
$$

**Proof sketch:** With correctly specified $\Sigma$, GLS is consistent. With estimated $\hat{\Sigma}$, feasible GLS retains consistency under mild conditions on rate of convergence of $\hat{\Sigma}$.

### Efficiency

When $\Sigma$ is known, GLS achieves the Gauss-Markov lower bound for linear unbiased estimators:

$$
\text{Var}(\hat{\beta}_{GLS}) = (X^T (\Sigma^{-1} \otimes I_N) X)^{-1}
$$

With estimated $\hat{\Sigma}$, asymptotic efficiency is preserved under regularity conditions.

### Asymptotic Distribution

Under standard conditions:

$$
\sqrt{N}(\hat{\beta} - \beta^0) \xrightarrow{d} \mathcal{N}(0, V)
$$

where $V$ is the asymptotic variance matrix.

---

## 8. Complexity Analysis

### Memory Complexity

| Component | Size | Total |
|-----------|------|-------|
| Factor matrix $F$ | $K \times k$ | $Kk$ |
| Diagonal $D$ | $K$ | $K$ |
| Latent scores (implicit) | $N \times k$ | $Nk$ |
| Woodbury core $C$ | $k \times k$ | $k^2$ |

**Total:** $O(Kk + Nk + k^2) = O((K+N)k)$ when $k \ll K$.

**Comparison:** Dense GLS requires $O(K^2)$ for $\Sigma^{-1}$.

### Time Complexity Per Sweep

| Operation | Complexity |
|-----------|------------|
| Woodbury factors | $O(Kk^2 + k^3)$ |
| β-step (CG) | $O(T_{CG} \cdot Nkp)$ |
| Gradient of F | $O(NK k + K k^2)$ |
| D update | $O(NK)$ |

**Total per sweep:** $O(T_{CG} \cdot NKp + NK k)$

---

## 9. Rank Misspecification

### Over-Specification ($k > k_0$)

When the true rank is $k_0 < k$:
- Extra factors absorb noise
- Slight efficiency loss
- No asymptotic bias

### Under-Specification ($k < k_0$)

When the true rank is $k_0 > k$:
- Omitted factor structure remains in residuals
- Potential bias in $\hat{\beta}$
- Model misspecification

### Practical Guidance

Use information criteria for rank selection:

$$
\text{BIC}(k) = N \cdot \ell(k) + \frac{1}{2} p(k) \log(N)
$$

where $p(k) = K(k+1) + k$ is the number of parameters.

---

## References

- Woodbury, M.A. (1950). Inverting modified matrices. Memorandum Report 42.
- Bai, J., & Ng, S. (2002). Determining the number of factors in approximate factor models. Econometrica.
- Zellner, A. (1962). An efficient method of estimating seemingly unrelated regressions. JASA.
