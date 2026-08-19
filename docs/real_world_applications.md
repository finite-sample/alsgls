# Real-World Applications

This page demonstrates `alsgls` on real datasets, showing how low-rank GLS handles many-equation systems efficiently.

## Fama-French 49 Industry Portfolios

A classic finance application: modeling cross-sectional asset returns.

### The Problem

Monthly returns of 49 US industry portfolios from Kenneth French's Data Library. Each industry's excess return is regressed on Fama-French factors:

```
r_{j,t} - r_{f,t} = α_j + β_{j,1}(MKT_t - r_{f,t}) + β_{j,2}SMB_t + β_{j,3}HML_t + ε_{j,t}
```

With K=49 equations and correlated residuals (industries with similar exposures move together), traditional SUR becomes expensive.

### Running the Example

```bash
python examples/real_data_fama_french.py
```

### Key Results

```
Problem size: N=434 observations, K=49 equations, p=4 features
Selected rank: k=4
Final NLL per row: -89.46

BIC by rank:
  k=1: BIC=-37630, NLL=-87.40
  k=2: BIC=-38007, NLL=-88.62
  k=4: BIC=-38072, NLL=-89.46  <- optimal
```

The BIC criterion selects k=4 latent factors, capturing the dominant sources of cross-industry correlation beyond the Fama-French factors.

### Memory Efficiency

Traditional SUR would require a 49×49 = 2,401 element covariance matrix.
ALS-GLS with k=4 uses only 49×4 + 49 = 245 parameters for the covariance structure—a 10× reduction.

### Interpretation

The estimated factor loadings F reveal which industries co-move:

- **Factor 1**: Captures broad market sensitivity beyond beta
- **Factor 2-4**: Sector-specific risk exposures (tech vs. utilities, cyclicals vs. defensives)

### Code Walkthrough

```python
from alsgls import ALSGLS
import numpy as np

# Y: excess returns (N × K matrix)
# Xs: list of K design matrices [1, Mkt-RF, SMB, HML] for each equation

est = ALSGLS(rank="bic", max_sweeps=15)
est.fit(Xs, Y)

# Selected rank
print(f"Rank: {est.rank_}")

# Factor loadings reveal industry clustering
F = est.F_  # K × k matrix
```

## When to Use ALS-GLS

ALS-GLS is most beneficial when:

1. **Many equations (K > 20)**: The memory savings become substantial
2. **Correlated residuals**: There's meaningful cross-equation structure to capture
3. **Iterative estimation**: You're alternating between β and Σ estimation
4. **Factor structure expected**: The true covariance is approximately low-rank

### Example Applications

- **Finance**: Portfolio returns, factor models, cross-sectional pricing
- **Macroeconomics**: Multi-country VAR, regional panel data
- **Marketing**: Multi-product demand systems
- **Environmental**: Spatial correlation in air quality monitoring
