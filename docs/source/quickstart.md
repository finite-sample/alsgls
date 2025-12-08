# Quick Start Guide

```{include} _snippets/installation.md
```

For development, clone the repository and install in editable mode:

```bash
git clone https://github.com/finite-sample/alsgls.git
cd alsgls
pip install -e .
```

## Basic Usage

The high-level `alsgls.ALSGLS` estimator exposes a familiar scikit-learn
API. Under the hood it calls `alsgls.als_gls`, the alternating
least-squares solver for low-rank-plus-diagonal GLS.

### Simulating Data

First, let's generate some synthetic data using the built-in simulation functions:

```python
from alsgls import simulate_sur

# Generate Seemingly Unrelated Regressions data
Xs_train, Y_train, Xs_test, Y_test = simulate_sur(
    N_tr=240,    # Training observations
    N_te=120,    # Test observations
    K=60,        # Number of equations
    p=3,         # Regressors per equation
    k=4          # Latent factor rank
)
```

### Fitting the Model

Fit the ALS model and inspect the diagnostic trace:

```python
from alsgls import ALSGLS

estimator = ALSGLS(rank="auto", max_sweeps=12)
estimator.fit(Xs_train, Y_train)
print(f"Chosen rank: {estimator.rank_}")
print(f"NLL trace: {estimator.info_['nll_trace']}")
```

The fitted estimator exposes

- `B_list_`: list of regression coefficient vectors for each equation,
- `F_`: factor loadings matrix `(K × rank_)`,
- `D_`: diagonal noise variances,
- `info_`: convergence diagnostics including the NLL trace and CG stats.

### Making Predictions

Use `alsgls.ALSGLS.predict` and `alsgls.ALSGLS.score` to evaluate on
held-out data. `score` returns the negative Gaussian log-likelihood per
observation (larger is better).

```python
Y_pred = estimator.predict(Xs_test)
test_score = estimator.score(Xs_test, Y_test)
print(f"Test NLL per observation: {-test_score:.4f}")
```

### Statsmodels-style System API

To mirror `statsmodels` and `linearmodels` SUR interfaces, use
`alsgls.ALSGLSSystem` with a dictionary mapping equation names to
`(y, X)` pairs:

```python
from alsgls import ALSGLSSystem

system = {f"eq{j}": (Y_train[:, j], Xs_train[j]) for j in range(Y_train.shape[1])}
sys_model = ALSGLSSystem(system, rank="auto")
sys_results = sys_model.fit()
print(sys_results.summary_dict())
```

The returned `alsgls.ALSGLSSystemResults` object stores the fitted
coefficients, residuals, and NLL trace, and provides `predict` and
`params_as_series` (optional pandas dependency) for easy comparisons with
classical SUR packages.

### Complete Example

Here's a complete working example comparing ALS with the EM baseline:

```python
from alsgls import simulate_sur, ALSGLS, em_gls, XB_from_Blist, mse

Xs_tr, Y_tr, Xs_te, Y_te = simulate_sur(N_tr=240, N_te=120, K=60, p=3, k=4)

als = ALSGLS(rank=4, max_sweeps=10)
als.fit(Xs_tr, Y_tr)
Y_pred_als = als.predict(Xs_te)

B_em, F_em, D_em, mem_em, _ = em_gls(Xs_tr, Y_tr, k=4)
Y_pred_em = XB_from_Blist(Xs_te, B_em)

mse_als = mse(Y_te, Y_pred_als)
mse_em = mse(Y_te, Y_pred_em)

print(f"ALS MSE: {mse_als:.6f}")
print(f"EM MSE:  {mse_em:.6f}")
print(f"ALS sweeps used: {len(als.info_['nll_trace']) - 1}")
```

## Defaults and Troubleshooting

- **Rank heuristic** – The estimator uses `min(8, ceil(K / 10))` when
  `rank="auto"`; raise the rank if residual correlations persist, or lower it
  to avoid overfitting tiny samples.
- **Ridge parameters** – `lam_F` and `lam_B` default to `1e-3`. Increase
  them if the CG solver reports many iterations or the NLL trace stagnates.
- **Diagonal floor** – `d_floor` keeps the diagonal noise positive. Tighten it
  (e.g. `1e-6`) in ill-conditioned problems to prevent breakdowns.
- **Stopping criteria** – ALS stops when the relative improvement in the NLL is
  below `rel_tol` (default `1e-6`) or when `max_sweeps` is reached. Inspect
  `info_["nll_trace"]` and `info_["accept_t"]` to diagnose plateaus.

## Next Steps

- Read about the [mathematical background](mathematical_background.md) behind the algorithms
- Learn about the differences in [ALS vs EM](als_vs_em.md) approaches  
- Explore more detailed [examples](examples.md) and use cases