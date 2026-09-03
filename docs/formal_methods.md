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

### Sigma-Step: Closed Form

Given $D$, the maximising $F$ is available in closed form. Write $S = R^T R / N$
and let $(\theta_1, p_1), \dots, (\theta_K, p_K)$ be the eigenpairs of
$D^{-1/2} S D^{-1/2}$ in decreasing order, $P = (p_1, \dots, p_k)$ and
$\Theta = \mathrm{diag}(\theta_1, \dots, \theta_k)$. Then

$$
F = D^{1/2} P (\Theta - I_k)^{1/2},
$$

with eigenvalues below 1 truncated to zero. This is the classical result of
Lawley (see Lawley and Maxwell 1971, and eq. 8 of Fukasaku et al.,
arXiv:2402.08181), and it is what `stats::factanal` in R and
`sklearn.decomposition.FactorAnalysis` both compute.

In implementation the eigendecomposition is never formed. Since
$D^{-1/2} S D^{-1/2} = Z^T Z$ for $Z = R D^{-1/2} / \sqrt{N}$, the required
eigenvectors are the top $k$ right singular vectors of the $N \times K$ matrix
$Z$ and the eigenvalues are its squared singular values. No $K \times K$ matrix
is materialised.

Given $F$, the other stationarity condition is

$$
d_j = \max\left( S_{jj} - (FF^T)_{jj}, d_{\min} \right),
$$

with $d_{\min} > 0$ for positive definiteness.

**What this is not.** $\mathrm{diag}(S - FF^T)$ is *not* the conditional
maximiser of $D$ at fixed $F$. That condition is
$\mathrm{diag}\!\left(\Sigma^{-1}(\Sigma - S)\Sigma^{-1}\right) = 0$, which has
no closed-form solution. The pair above is a fixed point of the *joint*
stationarity conditions, so alternating them is a fixed-point iteration rather
than coordinate-wise maximisation and carries no descent guarantee of its own.
The implementation therefore evaluates the likelihood at every inner iteration
and returns the best iterate seen, which is what makes Theorem 1 hold.

---

## 6. Convergence Properties

### Theorem 1 (Monotonicity)

**Statement:** The NLL sequence $\{\ell^{(t)}\}$ is non-increasing: $\ell^{(t+1)} \leq \ell^{(t)}$.

**Proof:**
- The β-step minimises NLL exactly at fixed $\Sigma$, and reverts if the solve
  lands short
- The Σ-step returns the best iterate it evaluated, and is only accepted if it
  improves on the incumbent
- Neither step is accepted unless it lowers $\ell$

Combined, each sweep satisfies $\ell^{(t+1)} \leq \ell^{(t)}$. □

Note that the guard in the Σ-step is doing real work: the inner alternation is a
fixed-point iteration and can in principle step past the optimum, so
monotonicity comes from measuring the likelihood rather than from the form of
the update.

### Theorem 2 (Convergence to Stationary Point)

**Statement:** Under regularity conditions, the sequence $(\beta^{(t)}, F^{(t)}, D^{(t)})$ converges to a stationary point of $\ell$.

Both blocks are now exact, so there is no separate penalised objective to
converge to instead: with $\lambda_B = 0$ the fixed point is the maximum
likelihood estimate. The ridge $\lambda_B$ is expressed relative to the residual
variance scale, so the fit is equivariant under $Y \to sY$ rather than depending
on the units of $Y$.

On the number of sweeps: the shipped defaults (8 for `als_gls`, 12 for
`ALSGLS`) reach the fixed point on the problems tested, which the previous
gradient-based F-step did not — it needed on the order of 1000. The test
`test_the_default_sweep_budget_is_enough` gates this.

**Proof sketch:**
1. NLL is bounded below (by $-\infty$ from log-det, regularized to prevent this)
2. The sequence is monotonically non-increasing (Theorem 1)
3. Monotone bounded sequences converge
4. Limit point satisfies first-order optimality conditions

□

### Regularity Conditions

1. $E[X_j^T X_j]$ is full rank for each $j$
2. $d_{\min} > 0$ enforces bounded eigenvalues
3. Ridge regularization $\lambda_B > 0$ ensures strict convexity in the β block

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

where $V$ is the asymptotic variance matrix, and feasible GLS with a
consistently estimated $\hat\Sigma$ has the same first-order limit as GLS at
the true $\Sigma$.

### What the reported standard errors are, and are not

`cov_params` reports

$$
\widehat{\operatorname{Var}}(\hat\beta) = \left(X^T \hat\Sigma_c^{-1} X + \lambda I\right)^{-1},
\qquad
\hat\Sigma_c = \operatorname{diag}(\sqrt c)\,\hat\Sigma\,\operatorname{diag}(\sqrt c),
\quad c_j = \frac{N}{N - p_j},
$$

the variance of a GLS estimator whose covariance is *known*, evaluated at the
estimated one after the degrees-of-freedom rescale every SUR package applies
(`linearmodels` `debiased=True`, R `systemfit` `"geomean"`, Stata `sureg, dfk`).
Since $\hat\Sigma = \hat F\hat F^T + \operatorname{diag}(\hat D)$, the rescale
is applied as $\hat F \to \operatorname{diag}(\sqrt c)\hat F$,
$\hat D \to c \odot \hat D$, which preserves the structure exactly.

This understates the finite-sample variance, and the shortfall has two parts.
Freedman and Peters (1984, *JASA* 79, 97–106, Theorem 1) order them as

$$
\operatorname{Var}(\hat\beta_{\text{FGLS}}) \;>\; \operatorname{Var}(\hat\beta_{\text{GLS}}) \;>\; E\!\left[\widehat{\operatorname{Var}}(\hat\beta)\right].
$$

The right inequality is Jensen's: $(X^T\Sigma^{-1}X)^{-1}$ is concave in
$\Sigma$, so a noisy $\hat\Sigma$ biases the plug-in down even when unbiased.
The left is the extra spread feasible GLS carries over infeasible GLS. No
formula evaluated at a single $\hat\Sigma$ can see the left inequality.

**Measured**, on a 4-equation, 3-regressor, rank-1 system, 300 replicates,
$\text{se ratio} = $ mean reported SE / actual spread of $\hat\beta$:

| | mean SE | sd$(\hat\beta)$ | se ratio |
|---|---|---|---|
| plug-in at $\hat\Sigma$ | 0.144 | 0.211 | 0.69 |
| oracle GLS at the true $\Sigma$ | 0.186 | 0.182 | 1.03 |
| $\hat\beta_{\text{FGLS}}$, SE at the true $\Sigma$ | 0.186 | 0.211 | 0.88 |

The oracle is calibrated, so the formula is right and its inputs are not.
The third row is the ceiling on any correction that keeps $\hat\Sigma$ fixed:
0.88 at $n = 20$, 0.92 at $n = 30$, above 0.98 from $n = 50$.

The shortfall tracks the number of covariance parameters over the sample. With
$r = Kk + K - k(k-1)/2$ and $n$ fixed at 40:

| $r/(nK)$ | 0.050 | 0.069 | 0.072 | 0.094 | 0.106 | 0.144 |
|---|---|---|---|---|---|---|
| se ratio | 0.92 | 0.87 | 0.85 | 0.83 | 0.75 | 0.67 |

which is the $O(r/n)$ nuisance-parameter cost the theory predicts, and is why
the low-rank structure is what makes small $n$ feasible at all: an unstructured
$\hat\Sigma$ at $K = 60$, $n = 20$ is singular.

### Calibrated inference: `bootstrap()`

The literature on SUR inference is unanimous that the fix is not a better
standard error but a better *statistic* (Rilstone and Veall 1996; Fiebig and
Kim 2000; Horowitz 2019). `bootstrap(B, method, seed)` refits $F$, $D$ and
$\beta$ on each of $B$ resampled datasets and records the studentised
deviation

$$
t^{(b)} = \frac{\hat\beta^{(b)} - \hat\beta}{\widehat{\text{se}}^{(b)}},
$$

with $\widehat{\text{se}}^{(b)}$ the replicate's *own* plug-in. The plug-in is
biased the same way inside each replicate as in the sample, so the quantiles
of $t^{(b)}$ absorb the bias, and the percentile-$t$ interval

$$
\left[\hat\beta - t^*_{1-\alpha/2}\,\widehat{\text{se}},\;
      \hat\beta - t^*_{\alpha/2}\,\widehat{\text{se}}\right]
$$

is the calibrated object. The bootstrap standard error `bse` is reported too,
but is itself biased down (Freedman and Peters measured 20–30%), because each
replicate is drawn from a $\hat\Sigma$ that is too small.

Three schemes, all refitting $\Sigma$: `"parametric"` draws from the fitted
$\hat F\hat F^T + \operatorname{diag}(\hat D)$; `"wild"` multiplies each
residual row by a Rademacher sign, which preserves the cross-equation
structure without a distributional assumption; `"residual"` resamples whole
residual rows. Pairs resampling is deliberately absent: at $n = 20$ a
with-replacement draw has about 63% distinct rows, and a factor covariance
fitted to a dozen distinct $K$-vectors is not a replicate of anything.

### The default standard error: Kackar–Harville

Kackar and Harville (1984, *JASA* 79, 853–862) write the variance of the
feasible estimator, to first order, as $\Phi + \Lambda$ with

$$
\Lambda = \sum_{i,j} W_{ij}\,\Phi\left(Q_{ij} - P_i \Phi P_j\right)\Phi,
\qquad
P_i = -X^T V^{-1}\frac{\partial V}{\partial\theta_i}V^{-1}X,
\quad
Q_{ij} = X^T V^{-1}\frac{\partial V}{\partial\theta_i}V^{-1}\frac{\partial V}{\partial\theta_j}V^{-1}X,
$$

$W = \operatorname{Cov}(\hat\theta)$ the inverse expected information
$I_{ij} = \tfrac{n}{2}\operatorname{tr}(\Sigma^{-1}\partial_i\Sigma\,\Sigma^{-1}\partial_j\Sigma)$.
Here $V = \Sigma\otimes I_n$ and $\Sigma = FF^T + \operatorname{diag}(D)$, so
every piece is $X^T(M\otimes I)X$ for a $K\times K$ matrix $M$ — the same block
assembly as the GLS normal matrix — with
$\partial\Sigma/\partial F_{ab} = e_a f_b^T + f_b e_a^T$ and
$\partial\Sigma/\partial D_a = e_a e_a^T$. `cov_params` reports
$\Phi_c + \Lambda$ at the df-rescaled $\hat\Sigma_c$; `covariance("plugin")`
reports $\Phi_c$ alone, which is what linearmodels, systemfit and Stata report.
SAS ships the same first-order term as `DDFM=KR(FIRSTORDER)` and applies it to
its factor-analytic `TYPE=FA0()` structure.

**Rotation.** $F \to FQ$ leaves $\Sigma$ fixed, so $I$ is singular in
$k(k-1)/2$ directions. Along them $\partial\Sigma/\partial\theta$ vanishes, so
$P$ and $Q$ vanish and $\Lambda$ does not depend on how $W$ is completed there
(two generalised inverses differ by a term supported on the null space, which
the vanishing derivatives annihilate). $W$ is the pseudo-inverse of rank
$r - k(k-1)/2$; the test suite checks $\Lambda(FQ) = \Lambda(F)$ to $10^{-16}$.

**What it corrects, measured.** Evaluated at the *true* $(F, D)$ on the
4-equation, rank-1 fixture at $n=20$, $\Phi+\Lambda$ gives an se ratio of 0.98
(0.995 with the Monte Carlo $\operatorname{Cov}(\hat\theta)$ in place of
$I^{-1}$), against 0.94 for $\Phi$ alone: the term is the FGLS-over-GLS excess,
and it is essentially exact. An independent check: for two equations with
orthogonal regressors and an unstructured $\Sigma$, the machinery gives
$\Lambda/\Phi = 1/T$ exactly, for every coefficient and every correlation, and
Monte Carlo on that design gives $1.076, 1.037, 1.010$ for
$\operatorname{Var}/\Phi$ at $T = 20, 40, 80$ against $1.05, 1.025, 1.0125$.

**What it does not correct.** Evaluated at $(\hat F, \hat D)$:

| $n$ | plug-in | + df rescale | + df + Kackar–Harville (default) |
|---|---|---|---|
| 20 | 0.784 | 0.851 | **0.885** |
| 40 | 0.907 | 0.943 | **0.963** |
| 100 | 0.948 | 0.963 | **0.972** |
| 200 | — | 0.994 | **0.996** |

At $n=20$ the remaining gap is the bias of $\hat\Phi$ itself: $\hat\Sigma$ from
an ML factor fit at $r/(nK) = 0.1$ is far noisier and more shrunk than a sample
covariance, and $(X^T\Sigma^{-1}X)^{-1}$ is concave in $\Sigma$. With an
*unstructured* $\hat\Sigma$ on the two-equation design the plug-in is nearly
unbiased ($0.92$–$1.15$), so this is a factor-model effect, not a generic SUR
one.

**Why not Kenward–Roger.** Kenward and Roger (1997) add a second-order term to
correct that bias: $\Phi_A = \hat\Phi + 2\Lambda - R^*$ with
$R^* = \tfrac12\sum W_{ij}\Phi R_{ij}\Phi$ and $R_{ij}$ built from
$\partial^2\Sigma$, which is nonzero here because $\Sigma$ is quadratic in $F$.
Measured on the same fixture it makes things worse — 0.774 against the
plug-in's 0.784 — because the expansion predicts $\hat\Phi$'s bias as $+2\%$
when it is $-31\%$; the $O(n^{-1})$ expansion is not accurate at this
nuisance ratio. That is the documented failure mode: Kenward and Roger (2009,
*CSDA* 53) report the 1997 form "does not perform as well" for covariance
structures nonlinear in their parameters, and the second-derivative term is
also not invariant to the choice of generalised inverse under rotation. So the
second-order term is not used, and the calibrated object at small $n$ remains
`bootstrap()`.

### Identification

The $k$-factor model spends $r = Kk + K - k(k-1)/2$ parameters on a covariance
with $K(K+1)/2$ free entries. When $r$ exceeds that — Ledermann's bound,
$(K-k)^2 < K + k$ — the loadings are not identified, the likelihood has a ridge
of maxima, and any inference built on $\hat\Sigma$ is inference on a quantity
the data cannot pin down. R's `factanal` refuses with "degrees of freedom < 0";
so does `als_gls`. At $K = 4$ only $k = 1$ is identified; $K = 6$ admits
$k \le 3$; $K = 20$ admits $k \le 14$.

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

where $p(k) = Kk + K - k(k-1)/2 + \sum_j p_j$ is the number of free parameters:
the $Kk$ loadings less the $k(k-1)/2$ rotations $F \to FQ$ that leave $FF^T$
fixed and so are not identified, the $K$ diagonal variances, and the regression
coefficients. R's `factanal` reports the complementary
$\mathrm{df} = ((K-k)^2 - K - k)/2$.

---

## References

- Woodbury, M.A. (1950). Inverting modified matrices. Memorandum Report 42.
- Bai, J., & Ng, S. (2002). Determining the number of factors in approximate factor models. Econometrica.
- Zellner, A. (1962). An efficient method of estimating seemingly unrelated regressions. JASA.
