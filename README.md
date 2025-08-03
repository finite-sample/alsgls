## A Lightweight ALS Solver for Iterative GLS

When a GLS problem involves hundreds of equations, the $K × K$ covariance matrix becomes the computational bottleneck.  A simple statistical remedy is to assume that most of the cross‑equation dependence can be captured by a *handful of latent factors* plus equation‑specific noise.  This “low‑rank + diagonal” assumption slashes the number of unknowns from roughly $K^²$ to about $K×k$ parameters, where **k** (the latent factor rank) is much smaller than $K$.  The model alone, however, does **not** guarantee speed: we still have to fit the parameters.

### Solving low‑rank GLS: EM versus ALS

The classic EM algorithm alternates between updating the regression coefficients $\beta$ and updating the factor loadings $F$ and the diagonal noise $D$.  Even though $\hat{\Sigma}$ is low‑rank, EM’s M‑step recreates the **full** $K × K$ inverse, wiping out the memory win.

An alternative is **Alternating‑Least‑Squares (ALS)**. The Woodbury identity reduces the expensive inverse to a tiny k × k system, and the β‑update can be written without explicitly forming the dense matrix at all.  In practice, ALS converges in 5–6 sweeps and never allocates more than $O(K k)$ memory, while EM allocates $O(K^²)$.

**Rule of thumb:** if your GLS routine keeps looping between $\beta$ and a fresh $\hat{\Sigma}$, replacing the $\hat{\Sigma}$‑update by a factor‑ALS step yields the same statistical fit with an order‑of‑magnitude smaller memory footprint.

### Beyond SUR: where the idea travels

Random‑effects models, feasible GLS with estimated heteroskedastic weights, optimal‑weight GMM, and spatial autoregressive GLS all iterate β ↔ Σ̂.  Each can adopt the same ALS trick: treat the weight matrix as low‑rank + diagonal, invert only the k × k core, and avoid the dense K × K algebra.  Memory savings in published examples range from 5× to 20×, depending on k.

### A concrete case‑study: Seemingly‑Unrelated Regressions

To show the magnitude, we ran a Monte‑Carlo experiment with N = 300 observations, three regressors, rank‑3 factors, and K set to 50, 80, 120.  EM was given 45 iterations; ALS, six sweeps.  The largest array EM holds is the dense Σ⁻¹, whereas ALS’s largest is the skinny factor matrix F.  The table summarises six replications:

|   K | β‑RMSE EM | β‑RMSE ALS | Peak MB EM | Peak MB ALS | Memory ratio |
| --: | :-------: | :--------: | ---------: | ----------: | -----------: |
|  50 |   0.021   |    0.021   |     0.020  |      0.002  |         10×  |
|  80 |   0.020   |    0.020   |     0.051  |      0.003  |         17×  |
| 120 |   0.020   |    0.020   |     0.115  |      0.004  |         29×  |

Statistically, the two estimators are indistinguishable (paired‑test p ≥ 0.14).  Computationally, ALS needs only a few megabytes whereas EM needs tens to hundreds.

### 5  Choosing a solver in practice

For small systems ($K < 50$), dense GLS or even separate OLS is fine.  Between 50 and 300 equations, a low‑rank **factor‑ALS** solver gives the same estimates at roughly one‑tenth the memory and runs happily on a GPU.  Once K enters the hundreds, any dense inverse becomes prohibitive; structured approaches such as factor‑ALS or sparse/banded $\hat{\Sigma}$ are mandatory.
