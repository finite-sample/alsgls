## A Lightweight ALS Solver for Iterative GLS

[![PyPI version](https://img.shields.io/pypi/v/alsgls.svg)](https://pypi.org/project/alsgls/)
[![PyPI Downloads](https://static.pepy.tech/badge/alsgls)](https://pepy.tech/projects/alsgls)
[![Python](https://img.shields.io/badge/dynamic/toml?url=https://raw.githubusercontent.com/finite-sample/alsgls/main/pyproject.toml&query=$.project.requires-python&label=Python)](https://github.com/finite-sample/alsgls)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


When a GLS problem involves hundreds of equations, the $K × K$ covariance matrix becomes the computational bottleneck.  A simple statistical remedy is to assume that most of the cross‑equation dependence can be captured by a *handful of latent factors* plus equation‑specific noise.  This “low‑rank + diagonal” assumption slashes the number of unknowns from roughly $K^²$ to about $K×k$ parameters, where **k** (the latent factor rank) is much smaller than $K$.  The model alone, however, does **not** guarantee speed: we still have to fit the parameters.

### Installation

Install the library from PyPI:

```bash
pip install alsgls
```

For local development, clone the repo and use an editable install:

```bash
pip install -e .
```

### Usage

```python
from alsgls import als_gls, simulate_sur, nll_per_row, XB_from_Blist

Xs_tr, Y_tr, Xs_te, Y_te = simulate_sur(N_tr=240, N_te=120, K=60, p=3, k=4)
B, F, D, mem, _ = als_gls(Xs_tr, Y_tr, k=4)
Yhat_te = XB_from_Blist(Xs_te, B)
nll = nll_per_row(Y_te - Yhat_te, F, D)
```

See `examples/compare_als_vs_em.py` for a complete ALS versus EM comparison.

### Documentation and notebooks

Background material and reproducible experiments are available in the notebooks under [`als_sim/`](als_sim/), such as [`als_sim/als_comparison.ipynb`](als_sim/als_comparison.ipynb) and [`als_sim/als_sur.ipynb`](als_sim/als_sur.ipynb).

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

