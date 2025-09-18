Mathematical Background
=======================

Low-Rank + Diagonal GLS
------------------------

The Generalized Least Squares (GLS) problem with K equations involves minimizing:

.. math::

   \min_{\beta} (Y - X\beta)^T \Sigma^{-1} (Y - X\beta)

where :math:`\Sigma` is a K × K covariance matrix. For large K, storing and inverting :math:`\Sigma` 
becomes computationally prohibitive, requiring O(K²) memory and O(K³) operations.

Factor Model Structure
~~~~~~~~~~~~~~~~~~~~~~

The key insight is to assume a low-rank plus diagonal structure for the covariance matrix:

.. math::

   \Sigma = FF^T + \text{diag}(D)

where:

- :math:`F` is a K × k factor loadings matrix with k ≪ K
- :math:`D` is a K-dimensional vector of equation-specific noise variances

This reduces the parameter count from roughly K² to approximately K×k + K parameters.

Woodbury Matrix Identity
~~~~~~~~~~~~~~~~~~~~~~~~

The Woodbury identity allows us to compute :math:`\Sigma^{-1}` efficiently:

.. math::

   \Sigma^{-1} = D^{-1} - D^{-1}F(I + F^T D^{-1}F)^{-1}F^T D^{-1}

The key advantage is that we only need to invert a k × k matrix :math:`(I + F^T D^{-1}F)` 
instead of the full K × K matrix :math:`\Sigma`.

Alternating Least Squares Algorithm
------------------------------------

The ALS algorithm alternates between updating the regression coefficients and the factor structure:

**β-step**: Update regression coefficients using the Woodbury identity to avoid forming dense matrices

**Factor-step**: Update F and D using standard factor analysis techniques

Convergence typically occurs in 5-6 iterations, making it very efficient in practice.

Matrix-Free Conjugate Gradient
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The β-update can be written as solving:

.. math::

   (X^T \Sigma^{-1} X) \beta = X^T \Sigma^{-1} Y

Rather than forming the dense normal equations, we use conjugate gradient with matrix-vector 
products computed via the Woodbury identity. This maintains the O(Kk) memory complexity.

Expectation-Maximization Comparison
------------------------------------

The classical EM algorithm for factor models:

1. **E-step**: Compute posterior expectations of latent factors
2. **M-step**: Update parameters using MLE formulas

However, the M-step typically reconstructs the full K × K precision matrix, negating memory 
savings from the low-rank structure.

Memory Complexity Comparison
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

+----------+----------------+-------------------+
| Method   | Memory         | Inverse Size      |
+==========+================+===================+
| Dense    | O(K²)          | K × K             |
+----------+----------------+-------------------+
| EM       | O(K²)          | K × K (M-step)    |
+----------+----------------+-------------------+
| ALS      | O(Kk)          | k × k             |
+----------+----------------+-------------------+

Statistical Properties
----------------------

Under standard regularity conditions, both ALS and EM converge to the same maximum likelihood 
estimates. The algorithms differ only in their computational approach, not their statistical 
properties.

Convergence Guarantees
~~~~~~~~~~~~~~~~~~~~~~

- Both algorithms converge to local maxima of the likelihood
- Multiple random initializations recommended for global optimization
- ALS typically requires fewer iterations than EM due to its direct optimization approach

Applications Beyond SUR
------------------------

The low-rank + diagonal assumption and ALS approach apply to many econometric and statistical 
models:

- **Random Effects Models**: Panel data with individual-specific effects
- **Spatial Econometrics**: Approximate spatial weight matrices
- **Heteroskedastic GLS**: Estimated variance structures
- **GMM**: Optimal weighting matrices with many moment conditions

The memory savings become more pronounced as K increases, making ALS particularly valuable 
for large-scale applications.