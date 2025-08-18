import numpy as np
import os
import sys

# Ensure package root on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from alsgls.metrics import nll_per_row
from alsgls.ops import woodbury_pieces


def test_nll_matches_explicit():
    rng = np.random.default_rng(0)
    N, K, k = 5, 4, 2
    R = rng.normal(size=(N, K))
    F = rng.normal(size=(K, k))
    D = rng.uniform(0.5, 1.5, size=K)

    nll_uncached = nll_per_row(R, F, D)
    Dinv, Cf, _ = woodbury_pieces(F, D)
    nll_cached = nll_per_row(R, F, D, Dinv=Dinv, Cf=Cf)

    Sigma = F @ F.T + np.diag(D)
    Sigma_inv = np.linalg.inv(Sigma)
    logdet = np.linalg.slogdet(Sigma)[1]
    quad = np.sum(R @ Sigma_inv * R)
    nll_explicit = 0.5 * (quad / N + logdet + K * np.log(2 * np.pi))

    assert np.allclose(nll_uncached, nll_explicit)
    assert np.allclose(nll_cached, nll_explicit)
