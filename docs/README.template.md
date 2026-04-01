## A Lightweight ALS Solver for Iterative GLS

[![PyPI version](https://img.shields.io/pypi/v/alsgls.svg)](https://pypi.org/project/alsgls/)
[![PyPI Downloads](https://static.pepy.tech/badge/alsgls)](https://pepy.tech/projects/alsgls)
[![Python](https://img.shields.io/badge/dynamic/toml?url=https://raw.githubusercontent.com/finite-sample/alsgls/main/pyproject.toml&query=$.project.requires-python&label=Python)](https://github.com/finite-sample/alsgls)
[![License](https://img.shields.io/badge/dynamic/toml?url=https://raw.githubusercontent.com/finite-sample/alsgls/main/pyproject.toml&query=$.project.license.text&label=License)](https://opensource.org/licenses/MIT)


```{include} docs/source/_snippets/synopsis.md
```

```{include} docs/source/_snippets/installation.md
```

```{include} docs/source/_snippets/basic_usage.md
```

### Rank Selection

The package supports automatic rank selection via BIC or cross-validation:

```python
from alsgls import ALSGLS

# BIC-based rank selection
est = ALSGLS(rank="bic", max_sweeps=15)
est.fit(Xs, Y)
print(f"Selected rank: {est.rank_}")

# Cross-validation rank selection
est = ALSGLS(rank="cv", cv_folds=5, cv_random_state=42)
est.fit(Xs, Y)
print(f"Selected rank: {est.rank_}")
```

### Real Data Example

See `examples/real_data_fama_french.py` for a demonstration using Fama-French 49 industry portfolios.

The `benchmarks/compare_sur.py` script contrasts ALS-GLS with `statsmodels` and
`linearmodels` SUR implementations on matched simulation grids while recording
peak memory (via Memray, Fil, or the POSIX RSS high-water mark).

### Documentation and notebooks

Background material and reproducible experiments are available in the notebooks under [`als_sim/`](als_sim/), such as [`als_sim/als_comparison.ipynb`](als_sim/als_comparison.ipynb) and [`als_sim/als_sur.ipynb`](als_sim/als_sur.ipynb).

### Type-Safe ALS Solver

This package provides a modern, type-safe implementation of **Alternating-Least-Squares (ALS)** for low-rank GLS problems. The Woodbury identity reduces the expensive inverse to a tiny k × k system, and the β-update can be written without explicitly forming dense matrices. 

**New in v1.2.0:**
- **Standard errors** (`bse`) for all regression coefficients
- **t-statistics** (`tvalues`) and **p-values** (`pvalues`) for hypothesis testing
- **Confidence intervals** via `conf_int()` method
- **Summary tables** via `summary()` for statsmodels-style output

**New in v1.1.0:**
- **Rank selection**: BIC and cross-validation for automatic rank selection
- **Gradient-based factor update**: Cleaner theory, same convergence guarantees
- **Real-world example**: Fama-French 49 industry portfolios demonstration
- **Formal methods documentation**: Rigorous mathematical foundations

**Core features:**
- **Full type safety** with mypy compliance and comprehensive type hints
- **Numerically stable** implementation using Cholesky factorization throughout
- **Clean API** with single computational path and enhanced error messages
- **Memory efficient** with O(K k) complexity, converging in 5–6 sweeps

**Rule of thumb:** if your GLS routine keeps looping between $\beta$ and a fresh $\hat{\Sigma}$, the ALS approach yields the same statistical fit with an order‑of‑magnitude smaller memory footprint and better numerical stability.

### Beyond SUR: where the idea travels

Random‑effects models, feasible GLS with estimated heteroskedastic weights, optimal‑weight GMM, and spatial autoregressive GLS all iterate β ↔ Σ̂.  Each can adopt the same ALS trick: treat the weight matrix as low‑rank + diagonal, invert only the k × k core, and avoid the dense K × K algebra.  Memory savings in published examples range from 5× to 20×, depending on k.

### A concrete case‑study: Seemingly‑Unrelated Regressions

To demonstrate performance, we benchmark ALS against traditional methods with N = 300 observations, three regressors, rank‑3 factors, and K ranging from 50 to 120 equations. The largest array that traditional methods need is the dense Σ⁻¹ (K×K), whereas ALS's largest is the skinny factor matrix F (K×k).

```{include} docs/source/_snippets/performance_table.md
```

The ALS implementation achieves the same statistical performance while using only a few megabytes of memory, providing substantial computational advantages for large systems.

### Defaults, tuning knobs, and failure modes

- **Rank (`k`)** – By default the high-level APIs pick `min(8, ceil(K / 10))`, a
  conservative fraction of the number of equations. Increase `rank` if the
  cross-equation correlation matrix is slow to decay; decrease it when the
  diagonal dominates.
- **ALS ridge terms (`lam_F`, `lam_B`)** – Defaults to `1e-3` for both the
  latent-factor and regression updates; raise them slightly (e.g. `1e-2`) if CG
  struggles to converge or the NLL trace plateaus early.
- **Noise floor (`d_floor`)** – Keeps the diagonal component positive; the
  default `1e-8` protects against breakdowns when an equation is nearly
  deterministic. Increase it in highly ill-conditioned settings.
- **Stopping criteria** – ALS stops when the relative drop in NLL per sweep is
  below `1e-6` (configurable via `rel_tol`) or after `max_sweeps`. Inspect
  `info["nll_trace"]` to diagnose stagnation.
- **Possible failures** – Large condition numbers or nearly-collinear regressors
  can make the β-step CG solve slow; adjust `cg_tol`/`cg_maxit`, add stronger
  ridge, or re-scale predictors. If `info["accept_t"]` stays at zero and the
  NLL does not improve, the factor rank may be too large relative to the sample
  size.