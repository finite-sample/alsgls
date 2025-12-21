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

See `examples/compare_als_vs_em.py` for a complete ALS versus EM comparison. The
`benchmarks/compare_sur.py` script contrasts ALS-GLS with `statsmodels` and
`linearmodels` SUR implementations on matched simulation grids while recording
peak memory (via Memray, Fil, or the POSIX RSS high-water mark).

### Documentation and notebooks

Background material and reproducible experiments are available in the notebooks under [`als_sim/`](als_sim/), such as [`als_sim/als_comparison.ipynb`](als_sim/als_comparison.ipynb) and [`als_sim/als_sur.ipynb`](als_sim/als_sur.ipynb).

### Solving low‑rank GLS: EM versus ALS

The classic EM algorithm alternates between updating the regression coefficients $\beta$ and updating the factor loadings $F$ and the diagonal noise $D$.  Even though $\hat{\Sigma}$ is low‑rank, EM's M‑step recreates the **full** $K × K$ inverse, wiping out the memory win.

An alternative is **Alternating‑Least‑Squares (ALS)**. The Woodbury identity reduces the expensive inverse to a tiny k × k system, and the β‑update can be written without explicitly forming the dense matrix at all.  In practice, ALS converges in 5–6 sweeps and never allocates more than $O(K k)$ memory, while EM allocates $O(K^²)$.

**Rule of thumb:** if your GLS routine keeps looping between $\beta$ and a fresh $\hat{\Sigma}$, replacing the $\hat{\Sigma}$‑update by a factor‑ALS step yields the same statistical fit with an order‑of‑magnitude smaller memory footprint.

### Beyond SUR: where the idea travels

Random‑effects models, feasible GLS with estimated heteroskedastic weights, optimal‑weight GMM, and spatial autoregressive GLS all iterate β ↔ Σ̂.  Each can adopt the same ALS trick: treat the weight matrix as low‑rank + diagonal, invert only the k × k core, and avoid the dense K × K algebra.  Memory savings in published examples range from 5× to 20×, depending on k.

### A concrete case‑study: Seemingly‑Unrelated Regressions

To show the magnitude, we ran a Monte‑Carlo experiment with N = 300 observations, three regressors, rank‑3 factors, and K set to 50, 80, 120.  EM was given 45 iterations; ALS, six sweeps.  The largest array EM holds is the dense Σ⁻¹, whereas ALS's largest is the skinny factor matrix F.  The table summarizes six replications:

```{include} docs/source/_snippets/performance_table.md
```

Statistically, the two estimators are indistinguishable (paired‑test p ≥ 0.14).  Computationally, ALS needs only a few megabytes whereas EM needs tens to hundreds.

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